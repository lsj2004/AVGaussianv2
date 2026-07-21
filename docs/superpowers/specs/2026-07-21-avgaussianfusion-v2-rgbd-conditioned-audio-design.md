# AVGaussianFusionv2 RGBD-Conditioned Audio Design

## 1. Goal

Build a new standalone `AVGaussianFusionv2` project that combines two independently modeled Gaussian fields:

- FreeTimeGS++ provides the dynamic visual Gaussian field.
- AudioGS Replay provides the acoustic Gaussian field and dual-branch Audio U-Net renderer.
- AVGaussianFusionv2 aligns the two fields at render time and conditions audio rendering on RGB/depth features rendered for the same time and target viewpoint.

The first version implements one-way visual-to-audio conditioning only. It does not share Gaussian parameters, assume point correspondence, or feed audio information into visual rendering.

The initial deliverable includes project code, configurations, automated tests, and short smoke-training runs for `scene1_opera` and `Scene7playing`. Full training and quantitative evaluation are a later milestone.

## 2. Repository and Dependency Boundary

Create `/mnt/sda/lisujing/Dataset/AVGaussianFusionv2` as an independent Git repository. Do not copy either upstream source tree into it.

The project loads the local upstream implementations through configured paths:

- `/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus`
- `/mnt/sda/lisujing/Dataset/audioGS-replay`

All upstream-specific calls are isolated behind backend adapters. Core fusion, data alignment, training, and evaluation code must not import upstream training-script internals directly.

## 3. Architecture

### 3.1 Independent Gaussian fields

The visual and acoustic fields remain independent modules with independent parameters, point counts, coordinate attributes, optimizers, and checkpoints. No visual Gaussian index is assumed to correspond to an acoustic Gaussian index.

The only coupling in v2 is the differentiable render-time path:

```text
time + target camera
        |
        v
FreeTimeGS++ visual Gaussians
        |
        +-- rendered RGB
        +-- rendered expected depth + alpha
                    |
                    v
              RGBD encoder
                    |
              view embedding
                    |
       zero-initialized FiLM adapters
                    |
source audio -> AudioGS acoustic Gaussians
                    |
                    v
         conditioned dual-branch U-Net
                    |
                    v
           target binaural audio
```

During end-to-end fine-tuning, audio loss is allowed to backpropagate through the condition encoder and differentiable RGB/depth renderer into FreeTimeGS++. A simultaneous visual reconstruction objective constrains visual quality.

### 3.2 Visual Gaussian backend

`VisualGaussianBackend` loads a FreeTimeGS++ checkpoint and exposes:

```python
render_rgbd(time, w2c, intrinsic, image_size) -> RGBDRender
```

`RGBDRender` contains:

- `rgb`: rendered target-view RGB in `[0, 1]`.
- `depth`: differentiable expected camera-space depth.
- `alpha`: accumulated opacity used as the valid-depth mask.

The adapter uses the original FreeTimeGS++ dynamic Gaussian state and `gsplat` rasterization. It adds an RGB+expected-depth render path without changing the semantic meaning of upstream Gaussian parameters. A single call must use the same time, camera extrinsic, intrinsic, image size, and dynamic Gaussian state for RGB and depth.

### 3.3 Audio Gaussian backend

`AudioGaussianBackend` loads the AudioGS Replay acoustic parameters and preserves its STFT settings, mono/diff features, viewpoint-dependent spherical harmonics, distance cues, and dual-branch U-Net behavior.

It supports two rendering modes:

- `condition=None`: reproduce the unconditioned AudioGS output.
- `condition=view_embedding`: render through the FiLM-conditioned Audio U-Net.

This explicit switch provides a baseline and makes conditioning ablations possible without constructing a different model.

### 3.4 RGBD condition encoder

`RGBDConditionEncoder` receives target-view rendered RGB, normalized depth, and a valid-depth mask derived from alpha. Alpha is used for masking and may be included as an explicit encoder channel; it is not treated as a third target modality.

Depth preprocessing is differentiable:

1. Mark pixels with alpha below the configured threshold as invalid.
2. Clamp valid positive depth with a small epsilon.
3. Apply `log1p`.
4. Normalize valid pixels using per-frame robust center and scale.
5. Fill invalid pixels with zero and retain the mask.

The encoder is a lightweight convolutional network with global pooling that emits a fixed-size view embedding. The embedding represents target-view appearance, visibility, and geometry without pretending that image pixels align with audio time-frequency bins.

### 3.5 FiLM-conditioned Audio U-Net

The selected conditioning approach is multi-scale FiLM/AdaGN modulation.

For each selected Audio U-Net encoder, bottleneck, and decoder block, a learned adapter maps the view embedding to channel-wise scale and shift values:

```text
y = normalized_audio_feature * (1 + scale) + shift
```

The final linear layers that produce `scale` and `shift` are initialized to zero. Therefore, immediately after loading an AudioGS checkpoint, conditioned execution is numerically equivalent to the original AudioGS U-Net up to the configured test tolerance. Conditioning effects are learned gradually.

The first version uses a single compact view embedding rather than resized RGBD concatenation or cross-attention. This avoids artificial image/TF-grid correspondence and limits memory and overfitting risk on the two initial scenes.

### 3.6 Fusion model

`AVGaussianFusionV2` owns the visual backend, audio backend, condition encoder, and FiLM adapters. Its forward interface accepts an aligned sample and returns a structured result containing:

- rendered RGB, depth, and alpha;
- visual condition embedding;
- predicted binaural audio;
- optional diagnostics needed by losses and smoke tests.

It also exposes named parameter groups and explicit freeze/unfreeze methods for staged training.

## 4. Data Alignment

### 4.1 Canonical sample identity

Every training or evaluation sample is anchored by:

```text
scene_id, target_camera, frame_index
```

`AlignedAVDataset` resolves this identity into:

- physical `time_seconds` for audio cropping;
- the exact FreeTimeGS++ model time stored by the visual dataset or memmap;
- target-camera `w2c` and intrinsic matrices;
- target RGB frame;
- synchronized source and target binaural audio crops;
- the camera identifiers required by AudioGS Replay.

Physical seconds and visual model time are separate fields and are never substituted for one another.

### 4.2 Camera mapping

Each scene configuration declares an explicit mapping among manifest camera names, FreeTimeGS++ camera indices, and AudioGS viewpoint identifiers. Construction fails if a requested camera is missing on either side or if two names map ambiguously.

The default training/evaluation split follows the existing `scene1_opera` and `Scene7playing` manifests, including their existing held-out camera choice.

### 4.3 Temporal sampling

The visual frame is rendered at its exact visual model time. Source and target audio use a crop centered at the same physical timestamp. The dataset validates that the audio crop center differs from the frame timestamp by no more than half a video-frame interval.

Samples that require out-of-range audio padding are excluded from training and reported in dataset statistics. Evaluation must use an explicit padding policy recorded in its output metadata.

## 5. Training

### 5.1 Initialization

Training requires separately pretrained FreeTimeGS++ and AudioGS checkpoints. The v2 project does not reimplement their independent pretraining commands, but scene documentation records the expected upstream commands and artifact paths.

### 5.2 Condition warmup

The warmup stage freezes both pretrained Gaussian backends and trains only:

- `RGBDConditionEncoder`;
- FiLM scale/shift adapters.

This stage verifies that the new path learns a nontrivial condition without immediately perturbing either pretrained field.

### 5.3 End-to-end fine-tuning

The joint stage unfreezes FreeTimeGS++, AudioGS, the RGBD encoder, and FiLM adapters. Audio loss may update the visual field through the differentiable condition path.

The total objective is:

```text
L_total = lambda_audio * L_AudioGS
        + lambda_rgb * (L1_RGB + lambda_dssim * DSSIM)
        + lambda_visual_anchor * L_visual_anchor
```

`L_AudioGS` uses the AudioGS Replay loss configuration for the selected scene. `L_visual_anchor` is an L2 penalty against the pretrained visual parameters selected for fine-tuning. Its purpose is to prevent audio gradients from improving the condition by destroying visual reconstruction.

All loss weights, trainable parameter groups, crop duration, condition image resolution, and stage lengths are configuration values saved with the checkpoint.

### 5.4 Gradient policy

The implementation must not detach RGB, depth, the view embedding, or the condition path during joint fine-tuning. A diagnostic records gradient norms for:

- visual Gaussian parameters;
- acoustic Gaussian parameters;
- RGBD encoder;
- FiLM adapters;
- the original Audio U-Net.

If the visual gradient attributable to audio loss remains zero for a configurable consecutive-step threshold, training fails with a diagnostic message.

## 6. Checkpoints

The joint checkpoint stores state dictionaries and provenance instead of pickling the complete upstream model objects:

- visual checkpoint source path and identifier;
- audio checkpoint source path and identifier;
- visual backend state dictionary;
- audio backend state dictionary;
- RGBD encoder state dictionary;
- FiLM adapter state dictionary;
- scene, camera, and time mappings;
- STFT and model-shape metadata;
- full resolved training configuration;
- current stage, step, optimizer state, and loss history;
- upstream commit identifiers when available.

Loading validates checkpoint schema version, tensor shapes, Audio U-Net channel layout, STFT parameters, scene mapping, and upstream adapter compatibility. Missing or incompatible required weights are fatal errors; they are not silently ignored.

## 7. Error Handling and Diagnostics

The system fails early with contextual errors for:

- missing or unsupported checkpoints;
- missing scene files or camera mappings;
- inconsistent sample rates or channel counts;
- audio crops shorter than the configured STFT requirement;
- timestamp mismatch beyond half a frame;
- non-finite RGB, depth, condition, audio, loss, or gradients;
- incompatible checkpoint network/STFT metadata;
- a disconnected audio-to-visual gradient path during joint training.

CUDA out-of-memory errors are not handled by silently changing the experiment. The error message suggests explicit configuration changes, such as lowering condition resolution or audio crop length, so reruns remain reproducible.

Smoke-run artifacts include resolved configuration, loss history, per-module gradient norms, rendered RGB/depth previews, predicted audio, and sample identity.

## 8. Tests and Acceptance Criteria

### 8.1 Unit tests

- Depth normalization produces finite values, respects alpha masking, and retains gradients.
- RGBD encoder output has the documented shape for supported input sizes.
- FiLM parameters match every selected U-Net block shape.
- Zero-initialized FiLM reproduces the unconditioned AudioGS output within numerical tolerance.
- Nonzero conditioning changes the audio output.
- Audio loss backpropagates through RGBD input and into a differentiable visual-backend parameter.

### 8.2 Data tests

- Camera, frame, physical time, model time, and audio crops align for both initial scenes.
- Boundary windows, held-out cameras, missing files, and invalid mappings follow the documented policy.
- The time conversion test uses known frame timestamps and exact expected model times.

### 8.3 Checkpoint tests

- Supported local FreeTimeGS++ and AudioGS checkpoints load successfully.
- Saving and restoring a v2 checkpoint reproduces model outputs within tolerance.
- Shape, STFT, schema, and mapping incompatibilities are rejected with specific errors.

### 8.4 Integration and smoke tests

- A lightweight fake visual/audio backend completes a CPU forward/backward integration test.
- `scene1_opera` completes configured short condition-warmup and joint-training runs with real checkpoints.
- `Scene7playing` completes the same smoke sequence.
- Each smoke run produces finite losses, finite gradients in all intended trainable modules, and the required artifacts.
- Condition-off and condition-on inference both run, and condition-on produces a measurable output difference after warmup.

The initial milestone does not require condition-on audio metrics to beat the unconditioned baseline. Full convergence, held-out quantitative comparison, and bidirectional audio-visual conditioning belong to the later full-training milestone.

## 9. Explicit Non-Goals for the First Version

- No shared visual/acoustic Gaussian parameters.
- No point-level correspondence or cross-field nearest-neighbor coupling.
- No audio-to-visual rendering condition.
- No image-to-spectrogram direct resize and concatenation.
- No RGBD/audio cross-attention.
- No requirement to fully train either initial scene during the implementation milestone.

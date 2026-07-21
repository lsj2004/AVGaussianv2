# RGBD-Conditioned AVGaussianFusionv2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, tested AVGaussianFusionv2 package that loads independent FreeTimeGS++ and AudioGS Replay models, renders target-view RGB/depth, and uses those features to FiLM-condition the AudioGS dual-branch U-Net.

**Architecture:** Upstream-specific code is isolated in visual and audio backend adapters. The visual backend performs differentiable RGB+expected-depth rasterization; an RGBD CNN produces a view embedding; a zero-initialized FiLM wrapper modulates the existing AudioGS renderer while preserving unconditioned checkpoint behavior. An aligned dataset and staged trainer provide condition warmup followed by end-to-end joint fine-tuning.

**Tech Stack:** Python 3.10+, PyTorch, gsplat through FreeTimeGS++, AudioGS Replay, NumPy, SoundFile, PyYAML, pytest, uv.

## Global Constraints

- Visual and acoustic Gaussian parameters, point counts, and checkpoints remain independent.
- The only v2 coupling is one-way differentiable visual-to-audio conditioning at render time.
- Joint training must not detach RGB, depth, the view embedding, or the condition path.
- FiLM output projections are zero-initialized so an imported AudioGS checkpoint initially preserves its output.
- Initial real-scene coverage is `scene1_opera` and `Scene7playing` with their existing held-out-camera split.
- The initial milestone runs short smoke training only; full quantitative training is a later milestone.
- Do not copy either upstream repository into this repository.

## Planned File Structure

```text
AVGaussianFusionv2/
├── pyproject.toml                         # package metadata and test configuration
├── README.md                              # setup, checkpoint, smoke-run commands
├── avgaussianv2/
│   ├── __init__.py                        # public API
│   ├── contracts.py                       # shared dataclasses and protocols
│   ├── config.py                          # strict YAML configuration loading
│   ├── backends/
│   │   ├── visual_ftgspp.py               # FreeTimeGS++ loading and RGBD rasterization
│   │   └── audio_audiogs.py               # AudioGS loading and conditioned invocation
│   ├── models/
│   │   ├── rgbd.py                        # depth preprocessing and RGBD encoder
│   │   ├── film_unet.py                   # zero-init FiLM wrapper for AudioGS renderer
│   │   └── fusion.py                      # top-level fusion forward and parameter groups
│   ├── data/aligned.py                    # synchronized AV sample construction
│   ├── losses.py                          # joint audio, RGB, DSSIM, and anchor losses
│   ├── checkpoint.py                      # schema-validated state-dict persistence
│   ├── train.py                           # warmup/joint steps and gradient diagnostics
│   └── cli/train.py                       # command-line entrypoint
├── configs/
│   ├── scene1_opera.yaml                  # first scene smoke configuration
│   └── Scene7playing.yaml                 # second scene smoke configuration
└── tests/                                 # unit, data, checkpoint, integration tests
```

---

### Task 1: Package Skeleton, Contracts, and Strict Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `avgaussianv2/__init__.py`
- Create: `avgaussianv2/contracts.py`
- Create: `avgaussianv2/config.py`
- Create: `tests/test_config_and_contracts.py`

**Interfaces:**
- Produces: `RGBDRender`, `AlignedAVSample`, `FusionOutput`, and `ProjectConfig` used by every later task.
- Produces: `load_project_config(path: str | Path) -> ProjectConfig` with missing-key and value validation.

- [ ] **Step 1: Write failing contract/config tests**

```python
def test_load_project_config_rejects_missing_audio_checkpoint(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("scene:\n  id: scene1_opera\n")
    with pytest.raises(ValueError, match="paths.audio_checkpoint"):
        load_project_config(path)


def test_rgbd_render_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shared batch and image shape"):
        RGBDRender(
            rgb=torch.zeros(1, 8, 8, 3),
            depth=torch.zeros(1, 4, 4, 1),
            alpha=torch.zeros(1, 8, 8, 1),
        )
```

- [ ] **Step 2: Run tests and confirm missing imports fail**

Run: `uv run --with pytest --with torch --with pyyaml pytest tests/test_config_and_contracts.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'avgaussianv2'`.

- [ ] **Step 3: Implement immutable contracts and validated config dataclasses**

```python
@dataclass(frozen=True)
class RGBDRender:
    rgb: Tensor       # (B,H,W,3)
    depth: Tensor     # (B,H,W,1)
    alpha: Tensor     # (B,H,W,1)

    def __post_init__(self) -> None:
        spatial = self.rgb.shape[:3]
        if self.rgb.ndim != 4 or self.rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape (B,H,W,3)")
        if self.depth.shape != (*spatial, 1) or self.alpha.shape != (*spatial, 1):
            raise ValueError("rgb, depth, and alpha must share batch and image shape")


@dataclass(frozen=True)
class AlignedAVSample:
    scene_id: str
    camera: str
    frame_index: int
    time_seconds: float
    visual_time: Tensor
    w2c: Tensor
    intrinsic: Tensor
    audio_cam_pose: Tensor
    source_audio: Tensor
    target_audio: Tensor
    target_rgb: Tensor
    image_size: tuple[int, int]


@dataclass(frozen=True)
class FusionOutput:
    rgbd: RGBDRender
    condition: Tensor
    predicted_audio: Tensor


@dataclass(frozen=True)
class ProjectConfig:
    scene: SceneConfig
    paths: PathConfig
    model: ModelConfig
    train: TrainConfig


def load_project_config(path: str | Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text())
    require_path(raw, "paths.visual_checkpoint")
    require_path(raw, "paths.audio_checkpoint")
    cfg = ProjectConfig.from_dict(raw)
    cfg.validate()
    return cfg
```

Use `pyproject.toml` to declare the package, Python `>=3.10`, and pytest `testpaths = ["tests"]`.

- [ ] **Step 4: Run tests and package import check**

Run: `uv run --with pytest --with torch --with pyyaml pytest tests/test_config_and_contracts.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml avgaussianv2 tests/test_config_and_contracts.py
git commit -m "feat: add project contracts and configuration"
```

---

### Task 2: Differentiable FreeTimeGS++ RGBD Backend

**Files:**
- Create: `avgaussianv2/backends/__init__.py`
- Create: `avgaussianv2/backends/visual_ftgspp.py`
- Create: `tests/test_visual_backend.py`

**Interfaces:**
- Consumes: `RGBDRender` from Task 1.
- Produces: `FTGSVisualBackend.load(checkpoint, upstream_root)` and `render_rgbd(time, w2c, intrinsic, image_size) -> RGBDRender`.

- [ ] **Step 1: Write fake-rasterizer tests for RGB, expected depth, and gradient flow**

```python
def test_render_rgbd_splits_rgb_depth_and_preserves_gradient(fake_gaussians):
    backend = FTGSVisualBackend(fake_gaussians, rasterize=fake_rgb_ed_rasterize)
    result = backend.render_rgbd(
        time=torch.tensor([[0.25]]),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        image_size=(8, 12),
    )
    assert result.rgb.shape == (1, 8, 12, 3)
    assert result.depth.shape == (1, 8, 12, 1)
    result.depth.sum().backward()
    assert fake_gaussians.means.grad is not None
```

Also test multiple times in one render batch are rejected and a checkpoint without `means_t`, `opacities_t`, or SH tensors raises `UnsupportedFTGSCheckpoint`.

- [ ] **Step 2: Run tests and confirm backend is missing**

Run: `uv run --with pytest --with torch pytest tests/test_visual_backend.py -v`

Expected: FAIL importing `FTGSVisualBackend`.

- [ ] **Step 3: Implement lazy upstream loading and RGB+ED rasterization**

```python
class FTGSVisualBackend(nn.Module):
    def render_rgbd(self, time, w2c, intrinsic, image_size):
        t = require_single_time(time)
        rendered, alpha, _ = self._rasterize(
            means=self.gaussians.means_t(t),
            quats=self.gaussians.quats,
            scales=self.gaussians.scales.exp(),
            opacities=self.gaussians.opacities_t(t).squeeze(-1),
            colors=torch.cat([self.gaussians.sh_0, self.gaussians.sh_n], dim=1),
            viewmats=w2c,
            Ks=intrinsic,
            width=image_size[1],
            height=image_size[0],
            sh_degree=self.gaussians.sh_degree,
            render_mode="RGB+ED",
        )
        return RGBDRender(
            rgb=rendered[..., :3].clamp(0, 1),
            depth=rendered[..., 3:4],
            alpha=alpha,
        )
```

`load()` temporarily adds the configured upstream root to `sys.path`, imports `ftgspp.models.gaussians` and `gsplat`, loads with `weights_only=False`, validates the payload, and removes the temporary path even when loading fails.

- [ ] **Step 4: Run visual backend tests**

Run: `uv run --with pytest --with torch pytest tests/test_visual_backend.py -v`

Expected: all tests PASS without requiring CUDA or real gsplat.

- [ ] **Step 5: Commit**

```bash
git add avgaussianv2/backends tests/test_visual_backend.py
git commit -m "feat: add differentiable FTGS RGBD backend"
```

---

### Task 3: RGBD Preprocessing and View Encoder

**Files:**
- Create: `avgaussianv2/models/__init__.py`
- Create: `avgaussianv2/models/rgbd.py`
- Create: `tests/test_rgbd_encoder.py`

**Interfaces:**
- Consumes: `RGBDRender`.
- Produces: `normalize_depth(depth, alpha, alpha_threshold, eps) -> (normalized_depth, valid_mask)`.
- Produces: `RGBDConditionEncoder(embedding_dim).forward(render: RGBDRender) -> Tensor` shaped `(B, embedding_dim)`.

- [ ] **Step 1: Write tests for masked robust normalization and gradients**

```python
def test_normalize_depth_masks_background_and_keeps_gradient():
    depth = torch.tensor([[[[1.0], [3.0]], [[9.0], [100.0]]]], requires_grad=True)
    alpha = torch.tensor([[[[1.0], [1.0]], [[1.0], [0.0]]]])
    normalized, mask = normalize_depth(depth, alpha, alpha_threshold=0.01)
    assert normalized[0, 1, 1, 0] == 0
    assert not mask[0, 1, 1, 0]
    assert torch.isfinite(normalized).all()
    normalized.sum().backward()
    assert torch.isfinite(depth.grad).all()


def test_encoder_returns_fixed_embedding(rgbd_render):
    encoder = RGBDConditionEncoder(embedding_dim=128)
    assert encoder(rgbd_render).shape == (2, 128)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run --with pytest --with torch pytest tests/test_rgbd_encoder.py -v`

Expected: FAIL importing `normalize_depth`.

- [ ] **Step 3: Implement preprocessing and a five-channel CNN**

```python
def normalize_depth(depth, alpha, alpha_threshold=1e-3, eps=1e-6):
    valid = (alpha >= alpha_threshold) & torch.isfinite(depth) & (depth > 0)
    log_depth = torch.log1p(depth.clamp_min(eps))
    flat = log_depth.flatten(1)
    flat_valid = valid.flatten(1)
    center = masked_median(flat, flat_valid).view(-1, 1, 1, 1)
    deviation = (log_depth - center).abs()
    scale = masked_median(deviation.flatten(1), flat_valid).view(-1, 1, 1, 1)
    normalized = torch.where(valid, (log_depth - center) / scale.clamp_min(eps), 0.0)
    return torch.nan_to_num(normalized), valid


class RGBDConditionEncoder(nn.Module):
    def forward(self, render: RGBDRender) -> Tensor:
        depth, mask = normalize_depth(render.depth, render.alpha, self.alpha_threshold)
        x = torch.cat([render.rgb, depth, mask.to(depth)], dim=-1).permute(0, 3, 1, 2)
        return self.projection(self.pool(self.features(x)).flatten(1))
```

Use three stride-2 Conv/GroupNorm/SiLU blocks, adaptive average pooling, and a linear projection.

- [ ] **Step 4: Run encoder tests**

Run: `uv run --with pytest --with torch pytest tests/test_rgbd_encoder.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add avgaussianv2/models tests/test_rgbd_encoder.py
git commit -m "feat: encode rendered RGBD conditions"
```

---

### Task 4: Zero-Initialized FiLM Wrapper for AudioGS U-Net

**Files:**
- Create: `avgaussianv2/models/film_unet.py`
- Create: `tests/test_film_unet.py`

**Interfaces:**
- Consumes: an upstream AudioGS `DualBranchAudioUNet` instance and `(B, embedding_dim)` condition.
- Produces: `FiLMConditionedAudioUNet(base, embedding_dim)` with `use_condition(condition)` context manager and unchanged `forward(mono_features, diff_features)` signature.

- [ ] **Step 1: Write identity, conditioning, cleanup, and gradient tests**

```python
def test_zero_init_is_identical_to_base(audio_unet, mono, diff, condition):
    expected = audio_unet(mono, diff)
    wrapped = FiLMConditionedAudioUNet(audio_unet, embedding_dim=condition.shape[-1])
    with wrapped.use_condition(condition):
        actual = wrapped(mono, diff)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_context_is_cleared_after_exception(wrapped, condition):
    with pytest.raises(RuntimeError):
        with wrapped.use_condition(condition):
            raise RuntimeError("boom")
    assert wrapped.active_condition is None
```

Set one adapter weight nonzero and assert output changes and `condition.grad` is finite.

- [ ] **Step 2: Run tests and confirm wrapper is missing**

Run: `uv run --with pytest --with torch pytest tests/test_film_unet.py -v`

Expected: FAIL importing `FiLMConditionedAudioUNet`.

- [ ] **Step 3: Implement explicit AudioGS forward orchestration with FiLM calls**

```python
class FiLM(nn.Module):
    def __init__(self, embedding_dim, channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(embedding_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, condition):
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


@contextmanager
def use_condition(self, condition):
    if self.active_condition is not None:
        raise RuntimeError("nested AudioGS condition contexts are not supported")
    self.active_condition = condition
    try:
        yield
    finally:
        self.active_condition = None
```

Mirror the upstream encoder/decoder sequence by calling `base.enc1`, `base.diff_enc1`, `base.enc2` through `base.dec1`; apply named FiLM modules after each block and retain the upstream resize, output activation, and mono/diff semantics. When condition is `None`, bypass every FiLM call.

- [ ] **Step 4: Run FiLM tests**

Run: `uv run --with pytest --with torch pytest tests/test_film_unet.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add avgaussianv2/models/film_unet.py tests/test_film_unet.py
git commit -m "feat: condition AudioGS U-Net with zero-init FiLM"
```

---

### Task 5: AudioGS Backend and Top-Level Fusion Model

**Files:**
- Create: `avgaussianv2/backends/audio_audiogs.py`
- Create: `avgaussianv2/models/fusion.py`
- Create: `tests/test_audio_backend_and_fusion.py`

**Interfaces:**
- Consumes: configured AudioGS upstream root, AudioGS checkpoint, `RGBDConditionEncoder`, and `FTGSVisualBackend`.
- Produces: `AudioGSBackend.render(cam_pose, source_audio, condition=None) -> Tensor`.
- Produces: `AVGaussianFusionV2.forward(sample: AlignedAVSample) -> FusionOutput`.

- [ ] **Step 1: Write tests for checkpoint loading, condition propagation, and audio-to-visual gradients**

```python
def test_fusion_audio_loss_reaches_visual_parameter(fake_visual, fake_audio, sample):
    model = AVGaussianFusionV2(fake_visual, RGBDConditionEncoder(32), fake_audio)
    output = model(sample)
    output.predicted_audio.square().mean().backward()
    assert fake_visual.gaussian_parameter.grad is not None
    assert fake_visual.gaussian_parameter.grad.abs().sum() > 0


def test_audio_backend_rejects_checkpoint_without_model_state(tmp_path, model_factory):
    path = tmp_path / "bad.pth"
    torch.save({"optimizer_state_dict": {}}, path)
    with pytest.raises(AudioCheckpointError, match="model_state_dict"):
        AudioGSBackend.load(path, model_factory)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run --with pytest --with torch pytest tests/test_audio_backend_and_fusion.py -v`

Expected: FAIL importing `AudioGSBackend`.

- [ ] **Step 3: Implement AudioGS loader and scoped condition invocation**

```python
class AudioGSBackend(nn.Module):
    def render(self, cam_pose, source_audio, condition=None):
        if condition is None:
            return self.model(cam_pose, source_audio)
        with self.conditioned_renderer.use_condition(condition):
            return self.model(cam_pose, source_audio)

    @classmethod
    def load(cls, checkpoint, model_factory, embedding_dim):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" not in payload:
            raise AudioCheckpointError("AudioGS checkpoint is missing model_state_dict")
        model = model_factory(payload.get("cfg"))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        wrapped = FiLMConditionedAudioUNet(model.renderer, embedding_dim)
        model.renderer = wrapped
        return cls(model, wrapped, source_path=Path(checkpoint))
```

Use a temporary upstream import context and select the AudioGS model class from an explicit config value; never infer a model class from tensor shapes.

- [ ] **Step 4: Implement the fusion forward**

```python
def forward(self, sample: AlignedAVSample) -> FusionOutput:
    rgbd = self.visual.render_rgbd(
        sample.visual_time, sample.w2c, sample.intrinsic, sample.image_size
    )
    condition = self.condition_encoder(rgbd)
    predicted = self.audio.render(sample.audio_cam_pose, sample.source_audio, condition)
    return FusionOutput(rgbd=rgbd, condition=condition, predicted_audio=predicted)
```

Add `freeze_pretrained()`, `unfreeze_all()`, and named parameter groups for visual, acoustic, condition encoder, FiLM, and base Audio U-Net parameters.

- [ ] **Step 5: Run backend/fusion tests and commit**

Run: `uv run --with pytest --with torch pytest tests/test_audio_backend_and_fusion.py -v`

Expected: all tests PASS, including nonzero visual gradient.

```bash
git add avgaussianv2/backends/audio_audiogs.py avgaussianv2/models/fusion.py tests/test_audio_backend_and_fusion.py
git commit -m "feat: combine independent visual and audio backends"
```

---

### Task 6: Aligned Scene Dataset for Both Initial Scenes

**Files:**
- Create: `avgaussianv2/data/__init__.py`
- Create: `avgaussianv2/data/aligned.py`
- Create: `configs/scene1_opera.yaml`
- Create: `configs/Scene7playing.yaml`
- Create: `tests/test_aligned_dataset.py`

**Interfaces:**
- Consumes: manifest, FTGS memmap/camera metadata, source/target WAV paths, and explicit camera mapping.
- Produces: `AlignedAVDataset[index] -> AlignedAVSample` with separate `time_seconds` and `visual_time`.

- [ ] **Step 1: Write synthetic alignment and boundary tests**

```python
def test_dataset_keeps_physical_and_visual_time_separate(scene_fixture):
    dataset = AlignedAVDataset(scene_fixture.config, split="train")
    sample = dataset[3]
    assert sample.time_seconds == pytest.approx(0.1)
    assert sample.visual_time.item() == pytest.approx(-0.8)
    assert sample.source_audio.shape == (1, 2, 8000)
    assert sample.target_audio.shape == (1, 2, 8000)


def test_missing_camera_mapping_is_fatal(scene_fixture):
    del scene_fixture.config.camera_mapping["cam10"]
    with pytest.raises(ValueError, match="cam10.*camera mapping"):
        AlignedAVDataset(scene_fixture.config, split="eval")
```

Also assert unpadded training windows exclude boundary frames and a timestamp mismatch greater than `0.5 / fps` raises an error.

- [ ] **Step 2: Run tests and confirm dataset is missing**

Run: `uv run --with pytest --with torch --with numpy --with soundfile pytest tests/test_aligned_dataset.py -v`

Expected: FAIL importing `AlignedAVDataset`.

- [ ] **Step 3: Implement deterministic sample indexing and centered crops**

```python
class AlignedAVDataset(Dataset):
    def __getitem__(self, index):
        record = self.records[index]
        center = round(record.time_seconds * self.sample_rate)
        start = center - self.crop_samples // 2
        source = read_exact_crop(self.source_path, start, self.crop_samples)
        target = read_exact_crop(record.target_audio_path, start, self.crop_samples)
        return AlignedAVSample(
            scene_id=self.scene_id,
            camera=record.camera,
            frame_index=record.frame_index,
            time_seconds=record.time_seconds,
            visual_time=record.visual_time,
            w2c=record.w2c,
            intrinsic=record.intrinsic,
            audio_cam_pose=record.audio_cam_pose,
            source_audio=source.unsqueeze(0),
            target_audio=target.unsqueeze(0),
            target_rgb=record.target_rgb,
            image_size=record.image_size,
        )
```

Both YAML files must explicitly define upstream roots, checkpoint paths, manifest/memmap paths, sample rate, crop seconds, condition resolution, train/eval cameras, and all camera-name/viewpoint mappings.

- [ ] **Step 4: Run data tests and validate YAML parsing**

Run: `uv run --with pytest --with torch --with numpy --with soundfile --with pyyaml pytest tests/test_aligned_dataset.py tests/test_config_and_contracts.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add avgaussianv2/data configs tests/test_aligned_dataset.py
git commit -m "feat: align visual frames and audio windows"
```

---

### Task 7: Joint Losses, Staged Training, and Gradient Diagnostics

**Files:**
- Create: `avgaussianv2/losses.py`
- Create: `avgaussianv2/train.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Consumes: `FusionOutput`, `AlignedAVSample`, pretrained visual snapshots, and `TrainConfig`.
- Produces: `compute_joint_loss(...) -> (Tensor, dict[str, Tensor])`.
- Produces: `run_condition_warmup(...)` and `run_joint_finetune(...)` with finite-value and gradient-path checks.

- [ ] **Step 1: Write tests for loss composition, freeze policy, and diagnostics**

```python
def test_joint_loss_matches_weighted_components(output, sample, visual_anchor):
    total, parts = compute_joint_loss(output, sample, visual_anchor, weights)
    expected = (
        weights.audio * parts["audio"]
        + weights.rgb * parts["rgb"]
        + weights.visual_anchor * parts["visual_anchor"]
    )
    torch.testing.assert_close(total, expected)


def test_joint_step_reports_all_gradient_groups(fake_fusion, sample):
    stats = joint_train_step(fake_fusion, sample, optimizer, config)
    assert stats.gradient_norms.keys() >= {
        "visual", "acoustic", "condition_encoder", "film", "audio_unet"
    }
    assert stats.gradient_norms["visual"] > 0
```

Add tests that warmup changes only condition/FiLM parameters and that NaN output raises `NonFiniteTrainingError` with scene/camera/frame context.

- [ ] **Step 2: Run tests and confirm training module is missing**

Run: `uv run --with pytest --with torch pytest tests/test_training.py -v`

Expected: FAIL importing `compute_joint_loss`.

- [ ] **Step 3: Implement loss composition and visual anchor selection**

```python
def compute_joint_loss(output, sample, anchor, weights):
    audio = audiogs_mono_diff_loss(output.predicted_audio, sample.target_audio)
    l1 = F.l1_loss(output.rgbd.rgb, sample.target_rgb)
    rgb = l1 + weights.dssim * dssim(output.rgbd.rgb, sample.target_rgb)
    anchor_loss = sum((p - anchor[name]).square().mean() for name, p in anchor.items())
    total = weights.audio * audio + weights.rgb * rgb + weights.visual_anchor * anchor_loss
    return total, {"audio": audio, "rgb": rgb, "visual_anchor": anchor_loss}
```

Import the exact AudioGS loss through the audio adapter rather than maintaining a diverging local copy.

- [ ] **Step 4: Implement warmup/joint loops and an audio-only gradient probe**

Warmup calls `freeze_pretrained()` and optimizes only condition encoder and FiLM parameters. Joint training calls `unfreeze_all()` and periodically uses `torch.autograd.grad(audio_loss, visual_parameters, retain_graph=True, allow_unused=True)` to prove that audio loss—not merely RGB loss—reaches the visual field. Fail after the configured number of consecutive zero probes.

- [ ] **Step 5: Run training tests and commit**

Run: `uv run --with pytest --with torch pytest tests/test_training.py -v`

Expected: all tests PASS.

```bash
git add avgaussianv2/losses.py avgaussianv2/train.py tests/test_training.py
git commit -m "feat: add staged joint AV training"
```

---

### Task 8: Versioned Checkpoint Save and Restore

**Files:**
- Create: `avgaussianv2/checkpoint.py`
- Create: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: fusion model, optimizer, resolved config, provenance, stage, step, and history.
- Produces: `save_checkpoint(path, state)` and `load_checkpoint(path, model, expected_config) -> ResumeState`.

- [ ] **Step 1: Write round-trip and incompatibility tests**

```python
def test_checkpoint_roundtrip_reproduces_output(tmp_path, fusion, sample, config):
    before = fusion(sample).predicted_audio.detach().clone()
    path = tmp_path / "joint.pt"
    save_checkpoint(path, build_checkpoint_state(fusion, None, config, stage="joint", step=4))
    restored = fresh_fusion()
    resume = load_checkpoint(path, restored, config)
    torch.testing.assert_close(restored(sample).predicted_audio, before)
    assert resume.step == 4


def test_checkpoint_rejects_stft_mismatch(tmp_path, checkpoint, config):
    mismatched = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, n_fft=1024),
    )
    with pytest.raises(CheckpointCompatibilityError, match="n_fft"):
        load_checkpoint(checkpoint, fresh_fusion(), mismatched)
```

- [ ] **Step 2: Run tests and confirm checkpoint module is missing**

Run: `uv run --with pytest --with torch pytest tests/test_checkpoint.py -v`

Expected: FAIL importing `save_checkpoint`.

- [ ] **Step 3: Implement schema version 1 and strict metadata validation**

```python
SCHEMA_VERSION = 1

def save_checkpoint(path, state):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "visual_state_dict": state.model.visual.state_dict(),
        "audio_state_dict": state.model.audio.state_dict(),
        "condition_state_dict": state.model.condition_encoder.state_dict(),
        "config": state.resolved_config,
        "provenance": state.provenance,
        "stage": state.stage,
        "step": state.step,
        "optimizer_state_dict": None if state.optimizer is None else state.optimizer.state_dict(),
        "loss_history": state.loss_history,
    }
    atomic_torch_save(payload, Path(path))
```

Validate schema, STFT values, embedding dimension, U-Net channel layout, scene ID, camera mapping hash, and all required keys before loading any weights. Load required state dictionaries with `strict=True`.

- [ ] **Step 4: Run checkpoint tests and commit**

Run: `uv run --with pytest --with torch pytest tests/test_checkpoint.py -v`

Expected: all tests PASS.

```bash
git add avgaussianv2/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: save versioned fusion checkpoints"
```

---

### Task 9: CLI, CPU Integration Test, Real-Checkpoint Smoke Scripts, and Documentation

**Files:**
- Create: `avgaussianv2/cli/__init__.py`
- Create: `avgaussianv2/cli/train.py`
- Create: `tests/test_integration.py`
- Create: `scripts/smoke_scene1_opera.sh`
- Create: `scripts/smoke_Scene7playing.sh`
- Create: `README.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: `uv run python -m avgaussianv2.cli.train --config CONFIG --stage {warmup,joint,all}`.
- Produces: two reproducible real-checkpoint smoke commands and documented artifacts.

- [ ] **Step 1: Write a CPU fake-backend end-to-end test**

```python
def test_cpu_warmup_then_joint_smoke(tmp_path, fake_project_config):
    result = run_training(
        fake_project_config,
        output_dir=tmp_path,
        warmup_steps=2,
        joint_steps=2,
        backend_factory=fake_backend_factory,
    )
    assert result.completed_stage == "joint"
    assert all(math.isfinite(row["total"]) for row in result.history)
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert (tmp_path / "artifacts" / "sample_pred.wav").exists()
```

- [ ] **Step 2: Run the integration test and confirm CLI is missing**

Run: `uv run --with pytest --with torch --with numpy --with soundfile --with pyyaml pytest tests/test_integration.py -v`

Expected: FAIL importing `run_training`.

- [ ] **Step 3: Implement CLI orchestration and artifact writing**

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_project_config(args.config)
    seed_everything(config.train.seed)
    result = run_training(config, stage=args.stage, output_dir=args.output_dir)
    write_json(Path(args.output_dir) / "run_summary.json", result.to_dict())
```

Every run writes resolved YAML/JSON config, latest checkpoint, loss history, gradient norms, RGB/depth preview, predicted WAV, and the selected sample identity. Shell scripts set explicit config/output paths and call the CLI without hidden defaults.

- [ ] **Step 4: Document setup, pretraining expectations, and smoke commands**

README sections must cover repository purpose, independent Gaussian fields, local upstream paths, environment setup, expected checkpoint formats, scene configuration, condition warmup, joint fine-tuning, smoke commands, output artifacts, and the condition-off ablation command.

- [ ] **Step 5: Run the complete CPU suite**

Run: `uv run --with pytest --with torch --with numpy --with soundfile --with pyyaml pytest -v`

Expected: all tests PASS.

- [ ] **Step 6: Run real-checkpoint smoke training for each scene**

Run:

```bash
scripts/smoke_scene1_opera.sh
scripts/smoke_Scene7playing.sh
```

Expected for each: warmup and joint stages exit 0; finite loss history and all required artifacts are present; condition-on output differs from condition-off after warmup. If a configured upstream checkpoint is absent, the script must exit nonzero and print the exact missing path rather than substituting a checkpoint.

- [ ] **Step 7: Commit**

```bash
git add avgaussianv2/cli scripts tests/test_integration.py README.md
git commit -m "feat: add reproducible AV fusion smoke pipeline"
```

---

### Task 10: Final Verification and GitHub Synchronization

**Files:**
- Modify only files required by verification failures; do not perform unrelated refactors.

**Interfaces:**
- Consumes: completed Tasks 1-9.
- Produces: a clean, verified `main` branch synchronized with `origin/main`.

- [ ] **Step 1: Run format and syntax checks**

Run: `uv run --with ruff ruff check .`

Expected: exit 0 with no diagnostics.

- [ ] **Step 2: Run the complete test suite from a clean process**

Run: `uv run --with pytest --with torch --with numpy --with soundfile --with pyyaml pytest -v`

Expected: all tests PASS with zero skips except explicitly marked real-CUDA tests.

- [ ] **Step 3: Re-run both real smoke scripts**

Run:

```bash
scripts/smoke_scene1_opera.sh
scripts/smoke_Scene7playing.sh
```

Expected: both exit 0 and produce finite warmup/joint results.

- [ ] **Step 4: Inspect repository state and commit any verification-only fixes**

```bash
git status --short
git diff --check
git add avgaussianv2 configs scripts tests README.md pyproject.toml
git commit -m "fix: address final verification findings"
```

Skip the commit if there are no verification fixes; `git diff --check` must emit no output.

- [ ] **Step 5: Push verified commits**

```bash
git push origin main
```

Expected: `main -> main` or `Everything up-to-date`.

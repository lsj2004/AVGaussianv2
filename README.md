# AVGaussianFusionV2

AVGaussianFusionV2 combines two **independent Gaussian** fields: FreeTimeGS++ models the
time-varying visual scene, while AudioGS models the acoustic field. The first v2 milestone
keeps both pretrained representations separate and conditions the AudioGS U-Net on an RGBD
render from the same timestamp and camera. RGB, robustly normalized depth, and alpha validity
are encoded by a small CNN; zero-initialized multi-scale FiLM adapters inject the embedding at
the AudioGS U-Net encoder and decoder stages.

For a map of the core model, experiment infrastructure, benchmark protocol, and the purpose
of each active branch, see
[`docs/codebase-and-branch-guide.zh-CN.md`](docs/codebase-and-branch-guide.zh-CN.md).
For a protocol-aware comparison of the results committed across those branches, see
[`docs/experiment-results-across-branches.zh-CN.md`](docs/experiment-results-across-branches.zh-CN.md).

## Upstream models and environment

The checked-in scene files expect these local upstream projects:

- `/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus`
- `/mnt/sda/lisujing/Dataset/audioGS-replay`

The visual checkpoint must deserialize to the FreeTimeGS++ Gaussian module used by
`gaussians.pt`. The audio checkpoint must contain `model_state_dict` and its original YACS
`cfg`; loading is strict and the loss is selected by the same `cfg.model.file` rule as
`Audio3DGSTrainer`. The provided checkpoints are GS-only variants, so v2 uses their inherited
`Audio3DGSMonoDiff.forward` only as a conditional residual:
`native GS-only + conditioned U-Net - plain U-Net`. This puts FiLM on the U-Net path while
preserving the native checkpoint output exactly when FiLM is zero-initialized or disabled.
Missing or incompatible checkpoints fail with the exact path or tensor shape instead of
silently substituting weights.

For CPU development and tests:

```bash
UV_OFFLINE=1 uv run --with pytest pytest -v
```

Real rasterization needs the CUDA build of `gsplat==1.5.3`. The smoke scripts default to the
existing FreeTimeGS++ environment. The v2 loader avoids importing AudioGS dataset readers, so
that environment only needs YACS for checkpoint metadata and SoundFile for WAV output:

```bash
uv pip install \
  --python /mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python \
  yacs soundfile
```

Set `AVGAUSSIANV2_PYTHON=/path/to/python` to use another environment containing compatible
PyTorch, gsplat, FreeTimeGS++, YACS, NumPy, PyYAML, and SoundFile installations.

## Scene configuration

[`configs/scene1_opera.yaml`](configs/scene1_opera.yaml) and
[`configs/Scene7playing.yaml`](configs/Scene7playing.yaml) contain explicit upstream roots,
pretrained checkpoint paths, aligned frame/audio manifests, camera mappings, and memmaps. The
same camera transform and physical timestamp drive the visual RGBD render and AudioGS pose;
model-normalized visual time is kept separate from physical audio time.

## Training stages

The **condition warmup** freezes both pretrained Gaussian models and the base AudioGS U-Net,
updating only the RGBD encoder and FiLM adapters. **joint fine-tuning** then unfreezes the visual
Gaussians, acoustic parameters, base U-Net, encoder, and FiLM adapters. The audio loss is allowed
to backpropagate through RGBD conditioning into the visual Gaussians; training records that
gradient norm and fails if it remains disconnected for the configured number of probes.

```bash
uv run python -m avgaussianv2.cli.train \
  --config configs/scene1_opera.yaml \
  --output-dir runs/scene1_opera \
  --stage all
```

Use `--stage warmup` or `--stage joint` to run one stage. A joint-only **condition-off** ablation
is available as:

```bash
uv run python -m avgaussianv2.cli.train \
  --config configs/scene1_opera.yaml \
  --output-dir runs/scene1_opera_condition_off \
  --stage joint --joint-steps 2 --condition-off
```

## Real-checkpoint smoke tests

Run both checked scenes explicitly:

```bash
scripts/smoke_scene1_opera.sh
scripts/smoke_Scene7playing.sh
```

Each command performs one condition warmup step and one joint fine-tuning step, then verifies
finite losses, required files, and that condition-on audio differs from condition-off audio.
Override only the output root with `AVGAUSSIANV2_OUTPUT=/absolute/path`.

## Scene1 diagnostic pilot

The bounded Scene1 diagnostic is launched with:

```bash
scripts/pilot_scene1_opera.sh
```

It uses GPUs 0, 1, and 2 and runs three training variants: joint conditioned, frozen visual,
and condition off. The defaults are 200 condition-warmup steps and at most 500 joint steps,
with quick validation every 50 joint steps. Early stopping starts only after 200 joint steps
and requires four validations without at least 0.5% relative improvement. Quick validation
uses 32 evenly spaced held-out samples; the final comparison evaluates the full held-out
`cam10` split.

The report contains five systems: imported baseline, joint conditioned on, the same joint
checkpoint evaluated with conditioning off, frozen visual with conditioning on, and the
separately trained condition-off system. It reports ten metrics and their directions:
`audio_total`, `audio_mono`, `audio_diff`, `waveform_l1`, `mono_lsd`, `diff_lsd`,
`lre_error_db`, and `rgb_l1` are lower-is-better; `rgb_psnr` and `rgb_ssim` are
higher-is-better. The long-training recommendation is `READY` only when the joint worker's
quick-best visual result stays within 0.5 dB PSNR and 0.01 SSIM of its baseline, the
conditioned full-split `audio_total` mean beats the paired condition-off evaluation, the
paired on-minus-off median is negative, and the audio-to-visual gradient is strictly
positive. Mean, median, and gradient gates are distinct; full-split visual feasibility and
the other system comparisons are descriptive.

Runtime state is written under `runs/pilot_scene1_opera/`: `logs/` contains baseline, worker,
and evaluation logs; `workers/<variant>/latest.pt` is the exact-resume checkpoint and
`best.pt` is the selected quick-validation checkpoint. The authoritative report is the
generation referenced by `report/current`; its `comparison.json`, `comparison.csv`, and
`comparison.md` must agree with `status.json`. Resume a compatible partial run with:

```bash
AVGAUSSIANV2_PYTHON=/path/to/python python -m avgaussianv2.cli.pilot \
  --config configs/scene1_opera.yaml \
  --output-dir runs/pilot_scene1_opera \
  --gpus 0,1,2 \
  --resume \
  --trust-upstream-artifacts
```

Verify an already complete run without launching runtime or GPU processes with:

```bash
AVGAUSSIANV2_PYTHON=/path/to/python python -m avgaussianv2.cli.pilot \
  --config configs/scene1_opera.yaml \
  --output-dir runs/pilot_scene1_opera \
  --gpus 0,1,2 \
  --verify-only
```

The production worker requires `--trust-upstream-artifacts`. FreeTimeGS++ and AudioGS are
loaded through the configured Python interpreter, and their legacy checkpoints may execute
code while being deserialized. SHA-256 hashes establish artifact identity, not safety: use
this flag only for known local upstream roots and checkpoints. The legacy `train` command
preserves its old implicit trusted-loading behavior for compatibility.

Configuration paths that are relative now resolve against the configuration file's directory,
not the caller's current working directory. Existing configs that relied on the old working
directory behavior must migrate their relative paths.

Workers checkpoint after every completed step so resume restores the exact model, optimizer,
selector, stopper, RNG, and progress state. This improves failure recovery but can create
substantial write amplification for large Gaussian checkpoints. Provision disk space and
inspect the report's checkpoint I/O metrics (`save_count`, bytes, duration, backup-copy
counts/bytes/duration, and failures) and any durability warnings.

The pilot never starts a long training run automatically. `READY` is only a recommendation;
review the metrics, logs, I/O cost, and durability warnings before separately authorizing a
long run.

## Dual-dataset cam38 benchmark

The production comparison uses the same strict camera split for `scene1_opera`
and `Scene7playing`: `cam00`–`cam37` train and `cam38` test. The immutable
project configs are
[`configs/benchmark_cam38/scene1_opera.yaml`](configs/benchmark_cam38/scene1_opera.yaml)
and
[`configs/benchmark_cam38/Scene7playing.yaml`](configs/benchmark_cam38/Scene7playing.yaml).
They use seed 42 and independent `cam38_strict` output, extraction, memmap,
COLMAP, point, and temporal-flow namespaces; historical cam10 artifacts are
not valid initialization.

The native AudioGS budget is batch size 1 for 61 epochs. This resolves to
2,318 updates for scene1 (38 training examples) and 6,954 updates for Scene7
(114 training examples). Scene7 is deliberately a single shared viewpoint-39 model
over all three clips: `A3DGS_FRAME_ID` is unset. The native FreeTimeGS++ budget
is batch size 1 for 30,000 updates. Optimizer updates, epochs, dataset length,
batch size, and sample exposures are all recorded so later comparisons do not
silently equate unlike epoch definitions.

Both native launch paths bind seed 42 in the process that runs the upstream
code. AudioGS receives `seed 42` in its real configuration override argv. The
FTGS++ entrypoint sets Python, NumPy, Torch, CUDA, cuDNN, and deterministic
algorithm state before loading upstream modules and writes audited seed records.
Its 38-slot train-only dataset uses camera 37 only as an internal training
monitor; this monitor is not the benchmark evaluation. The upstream pipeline
stops after preparation, precomputes bidirectional UFM flow explicitly for
cameras 0–37, verifies every expected pair/camera cache, and only then runs
point initialization and training. It never runs the upstream final-eval stage;
the repository's later common evaluator is the first consumer of cam38 targets.

The upstream preparation entrypoints are:

```bash
scripts/train_audiogs_cam38_baselines.sh
scripts/prepare_ftgspp_cam38_baselines.sh
```

Each command audits the strict split and initialization provenance, prints its
fully resolved upstream commands, and does not launch training unless `--execute` is supplied.
In particular, the FTGS++ train-only source hard-links only cam00
through cam37 and audits each link against the matching sampled-file inode;
the committed provenance permits cam38 pose/intrinsics but
fails closed if cam38 RGB/depth is declared as an image-driven initialization,
point, SfM/COLMAP, or temporal-flow input. These scripts prepare native assets
only; the fixed 30,000-update fusion/audio-only/visual-only continuations and
final common cam38 evaluator are separate benchmark stages.

### Three-GPU benchmark orchestration

After reviewing and trusting the configured upstream Python/checkpoints, run
the complete two-scene benchmark with:

```bash
AVGAUSSIANV2_PYTHON=/absolute/path/to/python \
  scripts/run_cam38_benchmark_suite.sh --gpus 0,1,2
```

The suite invokes both native baseline scripts with their explicit
`--execute` gate, then creates/validates the immutable AudioGS and FreeTimeGS++
native contracts, then runs `joint_conditioned`, `audio_only`, and
`visual_only` on GPUs 0/1/2. Evaluation is not started until all three
30,000-update workers finish. Every continuation is evaluated at 5k, 10k, and
30k on `cam38`; native AudioGS emits audio only and native FreeTimeGS++ emits
RGB only in the common evaluator. Per-scene reports are written below
`runs/cam38_benchmark/<scene>/report`, and macro/micro aggregation is written
to `runs/cam38_benchmark/report`.

Before native training, both scene configurations complete dependency, disk,
GPU, and source-hash preflight. With `--gpus A,B,C`, FreeTimeGS++ scene 1 runs
on physical GPU `A`, FreeTimeGS++ Scene7 on `B`, and the two AudioGS scenes run
serially on `C`; each isolated child addresses its assigned card as `cuda:0`.
The fail-closed GPU preflight maps each of `A`, `B`, and `C` into an isolated
child of the configured `AVGAUSSIANV2_PYTHON`, the FreeTimeGS++ virtualenv, and
the AudioGS `avcloud` conda environment. Every child must import its production
dependencies (including gsplat/tiny-cuda-nn or the audio stack), observe exactly
one CUDA device, allocate a `cuda:0` tensor, execute a kernel, and synchronize.
No native baseline process is launched unless all nine runtime probes succeed.
Preflight, status, protocol, and attempt-log records use immutable,
hash-chained generations. A partial resume validates these records and every
committed worker contract before writing anything; only workers with a durable
checkpoint receive `--resume`.

Use `--resume` after interruption. To reuse native training, pass
`--skip-native-training`; this succeeds only when both strict immutable native
contracts already exist and verify. It never treats an arbitrary checkpoint
as a native baseline. Checkpoint cadence is 500 updates with bounded rolling
retention and immutable 5k/10k/30k milestones; preflight records free disk,
GPU memory, dependency availability, source hashes, and this write policy.

Independent verification constructs no model/dataset runtime, launches no
process, and writes no files:

```bash
scripts/run_cam38_benchmark_suite.sh --verify-only
```

### Published benchmark result

The completed 2026-07-26 two-scene result, including the human-readable
analysis, machine-readable aggregate JSON, and long-form metrics CSV, is
versioned at
[`results/cam38_benchmark/2026-07-26`](results/cam38_benchmark/2026-07-26/README.md).
The result covers 423 held-out cam38 samples and is bound to suite content
SHA-256
`cc57caeaa1ab7e170c27b57be9eae80d5b11d4965d978b4eeee21a835e483c4a`.

For one scene, use
`scripts/run_cam38_benchmark_scene.sh scene1_opera --gpus 0,1,2` or replace the
scene with `Scene7playing`. A failed worker terminates its live siblings while
preserving checkpoints, atomic status events, and attempt logs for exact
resume.

## Cross-attention AudioGS postprocessor comparison

The `cross_attention_tokens` backend exposes the audited
`Audio3DGSMonoDiffGSOnly` attributes (`xyz`, quaternion, mono/diff SH fields,
native TF coordinate and pose-relative responses) through a versioned schema.
Their 89,436-point TF field is structurally pooled to a 16x16 acoustic-token
grid. Source-audio TF tokens cross-attend RGBD, pose and acoustic-Gaussian
tokens, then decode a bounded complex-spectrogram residual. The residual is
added to the native AudioGS render so the comparison with FiLM+U-Net has the
same anchor. The upstream AudioGS U-Net is removed and cannot be called.

The comparison against `FiLM+AudioGS U-Net` is therefore a postprocessor
ablation: both systems share the AudioGS Gaussian checkpoint, AudioGS
criterion, FTGS++ checkpoint, split, sample order, seed, warmup, and update
budget. They differ in the RGBD-conditioned postprocessor and its parameter
count.

The strict derived configs differ from the published FiLM configs only at
`model.audio_backend`. Preparation reuses the exact cam00–37/cam38 split,
ordered 30,000-sample sequence, seed 42, 2,000-step conditioner warmup,
30,000-step main budget, FTGS++ visual initialization, and AudioGS
acoustic-Gaussian initialization. Preparation verifies both immutable native
contracts and rejects a changed checkpoint or a non-GS-only AudioGS model.

Run one independently schedulable stage with:

```bash
scripts/run_cross_attention_cam38.sh scene1_opera prepare 0
scripts/run_cross_attention_cam38.sh scene1_opera train 0
scripts/run_cross_attention_cam38.sh scene1_opera eval 0 cross_attention 5000
scripts/run_cross_attention_cam38.sh scene1_opera eval 1 cross_attention_no_rgbd 5000
scripts/run_cross_attention_cam38.sh scene1_opera eval 2 cross_attention_shuffled_rgbd 5000
scripts/run_cross_attention_cam38.sh scene1_opera eval 0 cross_attention_no_gaussians 5000
scripts/run_cross_attention_cam38.sh scene1_opera eval 1 cross_attention_no_pose 5000
```

Repeat evaluation for steps 10,000 and 30,000, then run:

```bash
scripts/run_cross_attention_cam38.sh scene1_opera report
```

Replace `scene1_opera` with `Scene7playing` for the second dataset. Evaluation
jobs are read-only with respect to the shared checkpoint and may run concurrently
or share a GPU when memory permits. The five causal evaluations reuse the same
trained checkpoint. They measure inference-time reliance; they are not
retrained capacity ablations.

## Outputs

Every run writes `resolved_config.json`, `loss_history.json`, `gradient_norms.json`,
`selected_sample.json`, `run_summary.json`, and the versioned `checkpoint_latest.pt`. The
`artifacts/` directory contains predicted and condition-off WAV files, RGB/depth previews, and
the numerical condition difference. Checkpoints include separate visual, AudioGS, RGBD encoder,
and FiLM states plus compatibility hashes and upstream provenance.

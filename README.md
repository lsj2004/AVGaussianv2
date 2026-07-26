# AVGaussianFusionV2

AVGaussianFusionV2 combines two **independent Gaussian** fields: FreeTimeGS++ models the
time-varying visual scene, while AudioGS models the acoustic field. The first v2 milestone
keeps both pretrained representations separate and conditions the AudioGS U-Net on an RGBD
render from the same timestamp and camera. RGB, robustly normalized depth, and alpha validity
are encoded by a small CNN; zero-initialized multi-scale FiLM adapters inject the embedding at
the AudioGS U-Net encoder and decoder stages.

For a module-by-module explanation with architecture, data-flow, training, and gradient
diagrams, see the [Chinese architecture and code walkthrough](docs/architecture-and-code-walkthrough.zh-CN.md).

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

## Outputs

Every run writes `resolved_config.json`, `loss_history.json`, `gradient_norms.json`,
`selected_sample.json`, `run_summary.json`, and the versioned `checkpoint_latest.pt`. The
`artifacts/` directory contains predicted and condition-off WAV files, RGB/depth previews, and
the numerical condition difference. Checkpoints include separate visual, AudioGS, RGBD encoder,
and FiLM states plus compatibility hashes and upstream provenance.

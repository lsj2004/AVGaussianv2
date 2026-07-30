# Archived Implementation Plan

This plan describes the retired Scene1 pilot workflow. Its implementation
remains available in Git history on `origin/agent/scene1-pilot`; the current
clean branch uses the strict cam38 benchmark instead.

# Scene1 Pilot Training and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible three-GPU `scene1_opera` diagnostic that trains three controlled variants, selects visually feasible best checkpoints, evaluates the complete held-out `cam10` split, and reports whether RGBD conditioning is ready for long training.

**Architecture:** Add a focused `avgaussianv2.experiment` package for contracts, deterministic sampling, metrics, evaluation, selection, training, resume metadata, and reports. Keep upstream adapters unchanged; expose the existing model/dataset factory as a public runtime function, then drive it from one worker CLI per GPU and a parent pilot orchestrator.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SoundFile, PyYAML, FreeTimeGS++, AudioGS, gsplat, tiny-cuda-nn, pytest, Ruff.

---

### Task 0: Create an Isolated Implementation Worktree

**Files:**
- Create worktree: `.worktrees/scene1-pilot`

- [ ] **Step 1: Create and enter the feature worktree**

```bash
git worktree add .worktrees/scene1-pilot -b agent/scene1-pilot
cd .worktrees/scene1-pilot
git status --short --branch
```
Expected: the branch is `agent/scene1-pilot` and the worktree is clean. Perform every remaining
task and commit from this worktree; do not modify or commit from `main`.

---

### Task 1: Pilot Contracts and Deterministic Shared Sampling

**Files:**
- Create: `avgaussianv2/experiment/__init__.py`
- Create: `avgaussianv2/experiment/contracts.py`
- Create: `avgaussianv2/experiment/sampling.py`
- Create: `tests/test_experiment_sampling.py`

- [ ] **Step 1: Write failing contract and sampling tests**

```python
from avgaussianv2.experiment.contracts import PilotConfig, Variant
from avgaussianv2.experiment.sampling import build_shared_indices, evenly_spaced_indices


def test_pilot_defaults_match_approved_design():
    config = PilotConfig()
    assert config.warmup_steps == 200
    assert config.joint_steps == 500
    assert config.validation_interval == 50
    assert config.minimum_joint_steps == 200
    assert config.patience == 4
    assert config.minimum_relative_improvement == 0.005
    assert config.quick_validation_samples == 32


def test_shared_joint_sequence_is_identical_for_all_variants():
    indices = build_shared_indices(dataset_size=101, warmup_steps=200, joint_steps=500, seed=7)
    assert len(indices.warmup) == 200
    assert len(indices.joint) == 500
    assert indices.for_variant(Variant.JOINT_CONDITIONED).joint == indices.joint
    assert indices.for_variant(Variant.FROZEN_VISUAL).joint == indices.joint
    assert indices.for_variant(Variant.CONDITION_OFF).warmup == ()
    assert indices.for_variant(Variant.CONDITION_OFF).joint == indices.joint


def test_evenly_spaced_validation_indices_cover_full_split():
    assert evenly_spaced_indices(dataset_size=101, count=5) == (0, 25, 50, 75, 100)
```

- [ ] **Step 2: Run the tests and verify the experiment package is missing**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianv2-uv-cache uv run --with pytest pytest tests/test_experiment_sampling.py -v
```

Expected: collection fails with `ModuleNotFoundError: avgaussianv2.experiment`.

- [ ] **Step 3: Implement immutable pilot contracts**

```python
# avgaussianv2/experiment/contracts.py
from dataclasses import dataclass
from enum import StrEnum


class Variant(StrEnum):
    JOINT_CONDITIONED = "joint_conditioned"
    FROZEN_VISUAL = "frozen_visual"
    CONDITION_OFF = "condition_off"


@dataclass(frozen=True)
class PilotConfig:
    warmup_steps: int = 200
    joint_steps: int = 500
    validation_interval: int = 50
    minimum_joint_steps: int = 200
    patience: int = 4
    minimum_relative_improvement: float = 0.005
    quick_validation_samples: int = 32
    psnr_tolerance_db: float = 0.5
    ssim_tolerance: float = 0.01

    def validate(self) -> None:
        if self.warmup_steps < 0 or self.joint_steps <= 0:
            raise ValueError("pilot step counts are invalid")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if not 0 < self.minimum_relative_improvement < 1:
            raise ValueError("minimum_relative_improvement must be in (0,1)")


@dataclass(frozen=True)
class VariantIndices:
    warmup: tuple[int, ...]
    joint: tuple[int, ...]


@dataclass(frozen=True)
class SharedIndices:
    warmup: tuple[int, ...]
    joint: tuple[int, ...]

    def for_variant(self, variant: Variant) -> VariantIndices:
        warmup = () if variant is Variant.CONDITION_OFF else self.warmup
        return VariantIndices(warmup=warmup, joint=self.joint)
```

- [ ] **Step 4: Implement seeded sampling without prefix bias**

```python
# avgaussianv2/experiment/sampling.py
import numpy as np

from avgaussianv2.experiment.contracts import SharedIndices


def _draw_epochs(size: int, count: int, rng: np.random.Generator) -> tuple[int, ...]:
    if size <= 0:
        raise ValueError("dataset_size must be positive")
    values: list[int] = []
    while len(values) < count:
        values.extend(int(index) for index in rng.permutation(size))
    return tuple(values[:count])


def build_shared_indices(dataset_size: int, warmup_steps: int, joint_steps: int, seed: int) -> SharedIndices:
    rng = np.random.default_rng(seed)
    return SharedIndices(
        warmup=_draw_epochs(dataset_size, warmup_steps, rng),
        joint=_draw_epochs(dataset_size, joint_steps, rng),
    )


def evenly_spaced_indices(dataset_size: int, count: int) -> tuple[int, ...]:
    if dataset_size <= 0 or count <= 0:
        raise ValueError("dataset_size and count must be positive")
    actual = min(dataset_size, count)
    return tuple(int(round(value)) for value in np.linspace(0, dataset_size - 1, actual))
```

- [ ] **Step 5: Run tests and commit**

```bash
UV_CACHE_DIR=/tmp/avgaussianv2-uv-cache uv run --with pytest pytest tests/test_experiment_sampling.py -v
git add avgaussianv2/experiment tests/test_experiment_sampling.py
git commit -m "feat: add deterministic pilot sampling"
```

Expected: all sampling tests pass.

---

### Task 2: Dependency-Light Audio and Visual Metrics

**Files:**
- Create: `avgaussianv2/experiment/metrics.py`
- Create: `tests/test_experiment_metrics.py`

- [ ] **Step 1: Write failing metric tests with analytically known cases**

```python
import math

import pytest
import torch

from avgaussianv2.experiment.metrics import (
    aggregate_metrics,
    lre_error_db,
    log_spectral_distance,
    psnr,
    rgb_l1,
    ssim,
    waveform_l1,
)


def test_identical_inputs_have_ideal_metrics():
    audio = torch.randn(1, 2, 8000)
    image = torch.rand(1, 8, 8, 3)
    assert waveform_l1(audio, audio) == 0
    assert log_spectral_distance(audio, audio, component="mono") == 0
    assert lre_error_db(audio, audio) == 0
    assert math.isinf(psnr(image, image))
    assert ssim(image, image) == pytest.approx(1.0)
    assert rgb_l1(image, image) == 0


def test_aggregate_reports_mean_std_and_median():
    summary = aggregate_metrics([{"audio_total": 1.0}, {"audio_total": 3.0}])
    assert summary["audio_total"] == {"mean": 2.0, "std": 1.0, "median": 2.0}


def test_nonfinite_metric_input_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        waveform_l1(torch.tensor([float("nan")]), torch.zeros(1))
```

- [ ] **Step 2: Run tests and verify the metrics module is missing**

Run: `uv run --with pytest pytest tests/test_experiment_metrics.py -v`

Expected: import failure for `avgaussianv2.experiment.metrics`.

- [ ] **Step 3: Implement finite checks and waveform/spatial metrics**

```python
# avgaussianv2/experiment/metrics.py
import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from avgaussianv2.losses import dssim


def _finite(name: str, *values: Tensor) -> None:
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError(f"non-finite {name} input")


def waveform_l1(predicted: Tensor, target: Tensor) -> float:
    _finite("waveform_l1", predicted, target)
    return float(F.l1_loss(predicted, target).cpu())


def lre_error_db(predicted: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    _finite("lre_error_db", predicted, target)
    def ratio(value: Tensor) -> Tensor:
        left = value[:, 0].square().sum(-1)
        right = value[:, 1].square().sum(-1)
        return 10.0 * torch.log10((left + eps) / (right + eps))
    return float((ratio(predicted) - ratio(target)).abs().mean().cpu())


def rgb_l1(predicted: Tensor, target: Tensor) -> float:
    _finite("rgb_l1", predicted, target)
    return float(F.l1_loss(predicted, target).cpu())


def psnr(predicted: Tensor, target: Tensor) -> float:
    _finite("psnr", predicted, target)
    mse = F.mse_loss(predicted, target)
    return math.inf if mse == 0 else float((-10.0 * torch.log10(mse)).cpu())


def ssim(predicted: Tensor, target: Tensor) -> float:
    _finite("ssim", predicted, target)
    return 1.0 - 2.0 * float(dssim(predicted, target).cpu())
```

- [ ] **Step 4: Implement configured STFT log-spectral distance and aggregation**

```python
def log_spectral_distance(
    predicted: Tensor,
    target: Tensor,
    component: str,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    eps: float = 1e-7,
) -> float:
    _finite("log_spectral_distance", predicted, target)
    if component not in {"mono", "diff"}:
        raise ValueError("component must be mono or diff")
    sign = 1.0 if component == "mono" else -1.0
    pred = predicted[:, 0] + sign * predicted[:, 1]
    truth = target[:, 0] + sign * target[:, 1]
    window = torch.hamming_window(win_length, device=pred.device, dtype=pred.dtype)
    def log_mag(value: Tensor) -> Tensor:
        spectrum = torch.stft(
            value, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True,
        )
        return torch.log(spectrum.abs().clamp_min(eps))
    distance = (log_mag(pred) - log_mag(truth)).square().mean(dim=1).sqrt().mean()
    return float(distance.cpu())


def aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("metric rows must not be empty")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("metric rows have inconsistent keys")
    result = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite aggregate for {key}")
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
        }
    return result
```

- [ ] **Step 5: Run metric tests and commit**

```bash
uv run --with pytest pytest tests/test_experiment_metrics.py -v
git add avgaussianv2/experiment/metrics.py tests/test_experiment_metrics.py
git commit -m "feat: add pilot audio visual metrics"
```

---

### Task 3: Paired Held-Out Evaluator and Atomic Metric Outputs

**Files:**
- Create: `avgaussianv2/experiment/evaluation.py`
- Create: `tests/test_experiment_evaluation.py`
- Modify: `avgaussianv2/cli/train.py` (reuse a public sample-to-device helper)

- [ ] **Step 1: Write a CPU fake-model evaluator test**

```python
def test_evaluator_writes_per_sample_and_summary(tmp_path, tiny_fusion, eval_samples, audio_loss):
    evaluator = Evaluator(tiny_fusion, audio_loss, device=torch.device("cpu"))
    result = evaluator.evaluate(
        eval_samples,
        indices=(0, 2),
        system_name="joint_conditioned_on",
        condition_enabled=True,
        output_dir=tmp_path,
    )
    assert result.count == 2
    assert result.system_name == "joint_conditioned_on"
    assert result.summary.keys() >= {"audio_total", "rgb_psnr", "rgb_ssim"}
    assert (tmp_path / "metrics_per_sample.jsonl").is_file()
    assert (tmp_path / "metrics_summary.json").is_file()
    assert tiny_fusion.condition_enabled is True
    assert tiny_fusion.training is True
```

- [ ] **Step 2: Run the test and confirm `Evaluator` is missing**

Run: `uv run --with pytest pytest tests/test_experiment_evaluation.py -v`

Expected: import failure for `Evaluator`.

- [ ] **Step 3: Add evaluation result contracts and atomic JSON helpers**

```python
# append to contracts.py
@dataclass(frozen=True)
class EvaluationResult:
    system_name: str
    count: int
    rows: tuple[dict[str, object], ...]
    summary: dict[str, dict[str, float]]
```

```python
# evaluation.py
def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)
```

- [ ] **Step 4: Implement paired evaluation with model state restoration**

```python
class Evaluator:
    def __init__(self, model: nn.Module, audio_loss_fn: AudioLoss, device: torch.device) -> None:
        self.model = model
        self.audio_loss_fn = audio_loss_fn
        self.device = device

    def evaluate(self, samples, indices, system_name, condition_enabled, output_dir) -> EvaluationResult:
        was_training = self.model.training
        previous_condition = self.model.condition_enabled
        self.model.eval()
        self.model.condition_enabled = condition_enabled
        rows = []
        try:
            with torch.no_grad():
                for index in indices:
                    sample = move_sample(samples[index], self.device)
                    output = self.model(sample)
                    upstream = self.audio_loss_fn(output.predicted_audio, sample.target_audio)
                    row = {
                        "scene_id": sample.scene_id,
                        "camera": sample.camera,
                        "frame_index": sample.frame_index,
                        "time_seconds": sample.time_seconds,
                        "audio_total": float(upstream["total_loss"].cpu()),
                        "audio_mono": float(upstream["mono_loss"].cpu()),
                        "audio_diff": float(upstream["diff_loss"].cpu()),
                        "waveform_l1": waveform_l1(output.predicted_audio, sample.target_audio),
                        "mono_lsd": log_spectral_distance(output.predicted_audio, sample.target_audio, "mono"),
                        "diff_lsd": log_spectral_distance(output.predicted_audio, sample.target_audio, "diff"),
                        "lre_error_db": lre_error_db(output.predicted_audio, sample.target_audio),
                        "rgb_psnr": psnr(output.rgbd.rgb, sample.target_rgb.to(output.rgbd.rgb)),
                        "rgb_ssim": ssim(output.rgbd.rgb, sample.target_rgb.to(output.rgbd.rgb)),
                        "rgb_l1": rgb_l1(output.rgbd.rgb, sample.target_rgb.to(output.rgbd.rgb)),
                    }
                    rows.append(row)
        finally:
            self.model.condition_enabled = previous_condition
            self.model.train(was_training)
        numeric = [{key: value for key, value in row.items() if isinstance(value, float) and key != "time_seconds"} for row in rows]
        result = EvaluationResult(system_name, len(rows), tuple(rows), aggregate_metrics(numeric))
        write_evaluation(output_dir, result)
        return result
```

Move `_move_sample` from `avgaussianv2/cli/train.py` to public `move_sample` in
`avgaussianv2/experiment/evaluation.py`, then import it in the existing CLI so old behavior stays
covered by `tests/test_integration.py`.

- [ ] **Step 5: Run evaluator and existing integration tests, then commit**

```bash
uv run --with pytest pytest tests/test_experiment_evaluation.py tests/test_integration.py -v
git add avgaussianv2/experiment/evaluation.py avgaussianv2/experiment/contracts.py avgaussianv2/cli/train.py tests
git commit -m "feat: evaluate heldout audio visual metrics"
```

---

### Task 4: Visual Feasibility, Best Selection, and Early Stopping

**Files:**
- Create: `avgaussianv2/experiment/selection.py`
- Create: `tests/test_experiment_selection.py`

- [ ] **Step 1: Write failing selection tests**

```python
def test_visual_feasibility_uses_approved_tolerances():
    baseline = {"rgb_psnr": {"mean": 30.0}, "rgb_ssim": {"mean": 0.95}}
    assert visual_feasible({"rgb_psnr": {"mean": 29.5}, "rgb_ssim": {"mean": 0.94}}, baseline, 0.5, 0.01)
    assert not visual_feasible({"rgb_psnr": {"mean": 29.49}, "rgb_ssim": {"mean": 0.95}}, baseline, 0.5, 0.01)


def test_early_stopper_respects_minimum_steps_relative_delta_and_patience():
    stopper = EarlyStopper(minimum_steps=200, patience=4, relative_delta=0.005)
    assert not stopper.update(50, 1.0)
    assert not stopper.update(200, 0.99)
    assert not stopper.update(250, 0.989)
    assert not stopper.update(300, 0.988)
    assert not stopper.update(350, 0.987)
    assert stopper.update(400, 0.986)


def test_best_selector_rejects_audio_gain_with_visual_degradation():
    selector = BestSelector(visual_baseline=BASELINE, psnr_tolerance_db=0.5, ssim_tolerance=0.01)
    assert not selector.consider(step=50, summary=DEGRADED_BUT_GOOD_AUDIO)
    assert selector.best_step is None
```

- [ ] **Step 2: Run tests and verify selection APIs are missing**

Run: `uv run --with pytest pytest tests/test_experiment_selection.py -v`

Expected: import failure for `avgaussianv2.experiment.selection`.

- [ ] **Step 3: Implement selection and early-stopping state**

```python
@dataclass
class EarlyStopper:
    minimum_steps: int
    patience: int
    relative_delta: float
    best: float = math.inf
    stale: int = 0

    def update(self, step: int, value: float) -> bool:
        improved = math.isinf(self.best) or value <= self.best * (1.0 - self.relative_delta)
        if improved:
            self.best = value
            self.stale = 0
        elif step >= self.minimum_steps:
            self.stale += 1
        return step >= self.minimum_steps and self.stale >= self.patience


def visual_feasible(candidate, baseline, psnr_tolerance_db, ssim_tolerance) -> bool:
    return (
        candidate["rgb_psnr"]["mean"] >= baseline["rgb_psnr"]["mean"] - psnr_tolerance_db
        and candidate["rgb_ssim"]["mean"] >= baseline["rgb_ssim"]["mean"] - ssim_tolerance
    )


@dataclass
class BestSelector:
    visual_baseline: dict
    psnr_tolerance_db: float
    ssim_tolerance: float
    best_step: int | None = None
    best_audio_total: float = math.inf

    def consider(self, step: int, summary: dict) -> bool:
        feasible = visual_feasible(summary, self.visual_baseline, self.psnr_tolerance_db, self.ssim_tolerance)
        audio = summary["audio_total"]["mean"]
        if feasible and audio < self.best_audio_total:
            self.best_step = step
            self.best_audio_total = audio
            return True
        return False
```

- [ ] **Step 4: Run tests and commit**

```bash
uv run --with pytest pytest tests/test_experiment_selection.py -v
git add avgaussianv2/experiment/selection.py tests/test_experiment_selection.py
git commit -m "feat: select visually feasible pilot checkpoints"
```

---

### Task 5: Variant-Aware Step APIs and Periodic Pilot Trainer

**Files:**
- Modify: `avgaussianv2/train.py`
- Modify: `avgaussianv2/models/fusion.py`
- Create: `avgaussianv2/experiment/training.py`
- Create: `tests/test_experiment_training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write failing tests for parameter policies and callbacks**

```python
@pytest.mark.parametrize(
    ("variant", "visual_trainable", "condition_enabled"),
    [
        (Variant.JOINT_CONDITIONED, True, True),
        (Variant.FROZEN_VISUAL, False, True),
        (Variant.CONDITION_OFF, False, False),
    ],
)
def test_configure_variant_sets_exact_policy(tiny_fusion, variant, visual_trainable, condition_enabled):
    configure_variant(tiny_fusion, variant, stage="joint")
    assert any(p.requires_grad for p in tiny_fusion.visual.parameters()) is visual_trainable
    assert tiny_fusion.condition_enabled is condition_enabled


def test_pilot_trainer_validates_every_50_steps_and_stops(tmp_path, fake_runtime):
    result = PilotTrainer(PilotConfig(), fake_runtime.evaluator).run(
        model=fake_runtime.model,
        samples=fake_runtime.samples,
        indices=fake_runtime.indices,
        variant=Variant.JOINT_CONDITIONED,
        output_dir=tmp_path,
    )
    assert [row["step"] for row in result.validation_history] == [50, 100, 150, 200, 250, 300, 350, 400]
    assert result.stop_reason == "early_stopping"


def test_frozen_visual_skips_audio_to_visual_gradient_probe(fake_runtime):
    result = PilotTrainer(SHORT_PILOT, fake_runtime.evaluator).run(
        model=fake_runtime.model,
        samples=fake_runtime.samples,
        indices=fake_runtime.indices,
        variant=Variant.FROZEN_VISUAL,
        output_dir=fake_runtime.output_dir,
    )
    assert all(row["audio_to_visual_grad_norm"] == 0.0 for row in result.training_history)
```

- [ ] **Step 2: Run tests and confirm pilot training APIs are missing**

Run: `uv run --with pytest pytest tests/test_experiment_training.py -v`

Expected: import failure for `avgaussianv2.experiment.training`.

- [ ] **Step 3: Refactor reusable optimizer and single-step functions**

In `avgaussianv2/train.py`, add:

```python
def build_warmup_optimizer(model: nn.Module, learning_rate: float):
    groups = model.named_parameter_groups()
    return torch.optim.Adam([*groups["condition_encoder"], *groups["film"]], lr=learning_rate)


def condition_warmup_step(model, sample, optimizer, audio_loss_fn) -> TrainStepStats:
    optimizer.zero_grad(set_to_none=True)
    output = model(sample)
    _require_finite_tensor("predicted audio", output.predicted_audio, sample)
    audio = audio_loss_fn(output.predicted_audio, sample.target_audio)
    audio = audio["total_loss"] if isinstance(audio, Mapping) else audio
    _require_finite_tensor("warmup audio loss", audio, sample)
    audio.backward()
    groups = model.named_parameter_groups()
    norms = {name: _gradient_norm(parameters) for name, parameters in groups.items()}
    optimizer.step()
    return TrainStepStats(float(audio.detach().cpu()), {"audio": float(audio.detach().cpu())}, norms, 0.0)


def build_joint_optimizer(model: nn.Module, config: TrainConfig):
    groups = model.named_parameter_groups()
    specs = (
        ("visual", config.visual_lr),
        ("acoustic", config.audio_lr),
        ("audio_unet", config.audio_lr),
        ("condition_encoder", config.condition_lr),
        ("film", config.condition_lr),
    )
    return torch.optim.Adam([
        {"params": [p for p in groups[name] if p.requires_grad], "lr": lr}
        for name, lr in specs if any(p.requires_grad for p in groups[name])
    ])
```

Remove the unconditional `model.unfreeze_all()` from `joint_train_step`; callers must configure
the policy before building the optimizer. Add a `probe_audio_visual_gradient: bool = True`
argument: when false, do not call `torch.autograd.grad` and report `0.0`. The pilot trainer passes
true only for `joint_conditioned`, preventing an empty/frozen visual parameter probe. Rewrite
existing `run_condition_warmup` and `run_joint_finetune` in terms of these functions so all
existing tests remain valid.

- [ ] **Step 4: Implement exact variant policy**

```python
# experiment/training.py
def configure_variant(model: nn.Module, variant: Variant, stage: str) -> None:
    model.unfreeze_all()
    model.condition_enabled = variant is not Variant.CONDITION_OFF
    if stage == "warmup":
        model.freeze_pretrained()
        return
    if variant in {Variant.FROZEN_VISUAL, Variant.CONDITION_OFF}:
        for parameter in model.visual.parameters():
            parameter.requires_grad_(False)
    if variant is Variant.CONDITION_OFF:
        for parameter in model.condition_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.audio.film_parameters():
            parameter.requires_grad_(False)
```

- [ ] **Step 5: Implement periodic validation and stop decisions**

`PilotTrainer.run` must:

1. configure warmup and execute the shared warmup indices for conditioned variants;
2. configure joint policy and build one persistent joint optimizer;
3. call `joint_train_step` for the shared joint indices;
4. validate at steps divisible by 50;
5. emit a validation event at every validation and a best-candidate event when
   `BestSelector.consider` returns true; Task 6 connects these events to atomic checkpoints;
6. stop only after 200 joint steps and four stale validations;
7. require nonzero probed audio-to-visual gradients only for `joint_conditioned`;
8. write `training_curve.csv` and `worker_summary.json`.

Use these explicit result contracts:

```python
@dataclass(frozen=True)
class PilotTrainingResult:
    variant: str
    completed_warmup_steps: int
    completed_joint_steps: int
    best_step: int | None
    stop_reason: str
    validation_history: tuple[dict[str, object], ...]
```

- [ ] **Step 6: Run focused and regression tests, then commit**

```bash
uv run --with pytest pytest tests/test_training.py tests/test_experiment_training.py tests/test_integration.py -v
git add avgaussianv2/train.py avgaussianv2/models/fusion.py avgaussianv2/experiment/training.py tests
git commit -m "feat: add variant aware pilot training"
```

---

### Task 6: Strict Pilot Checkpoints and Resume Metadata

**Files:**
- Create: `avgaussianv2/experiment/checkpoint.py`
- Create: `tests/test_experiment_checkpoint.py`
- Modify: `avgaussianv2/experiment/training.py`

- [ ] **Step 1: Write failing compatibility tests**

```python
def test_resume_rejects_variant_mismatch(tmp_path, pilot_checkpoint):
    with pytest.raises(PilotResumeError, match="variant"):
        load_pilot_checkpoint(pilot_checkpoint, expected=replace(EXPECTED, variant="condition_off"))


def test_resume_rejects_changed_indices(tmp_path, pilot_checkpoint):
    with pytest.raises(PilotResumeError, match="index_hash"):
        load_pilot_checkpoint(pilot_checkpoint, expected=replace(EXPECTED, index_hash="changed"))


def test_upstream_hashes_are_content_hashes(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"weights")
    assert sha256_file(path) == hashlib.sha256(b"weights").hexdigest()
```

- [ ] **Step 2: Run tests and confirm pilot resume APIs are missing**

Run: `uv run --with pytest pytest tests/test_experiment_checkpoint.py -v`

Expected: import failure for `avgaussianv2.experiment.checkpoint`.

- [ ] **Step 3: Implement immutable compatibility metadata**

```python
@dataclass(frozen=True)
class PilotCompatibility:
    scene_id: str
    variant: str
    seed: int
    index_hash: str
    visual_checkpoint_sha256: str
    audio_checkpoint_sha256: str
    camera_mapping_sha256: str
    n_fft: int
    hop_length: int
    win_length: int
    sample_rate: int


def validate_compatibility(actual: PilotCompatibility, expected: PilotCompatibility) -> None:
    for field in dataclasses.fields(PilotCompatibility):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if left != right:
            raise PilotResumeError(f"pilot checkpoint {field.name} mismatch: {left!r} != {right!r}")
```

Store this mapping plus optimizer, early-stopper, best-selector, stage, warmup position, joint
position, histories, and stop reason in the existing checkpoint `provenance`. Load model tensors
through `load_checkpoint`, then validate pilot provenance before restoring optimizer state.

- [ ] **Step 4: Integrate atomic `best.pt` and `latest.pt` saves into trainer**

`latest.pt` must carry the active optimizer and exact next index position. `best.pt` must carry
the candidate model plus evaluation summary and may omit optimizer state. Resume accepts only a
nonempty output directory containing a compatible `latest.pt` and continues without replaying a
completed step.

- [ ] **Step 5: Run checkpoint and training tests, then commit**

```bash
uv run --with pytest pytest tests/test_checkpoint.py tests/test_experiment_checkpoint.py tests/test_experiment_training.py -v
git add avgaussianv2/experiment/checkpoint.py avgaussianv2/experiment/training.py tests
git commit -m "feat: resume strict pilot checkpoints"
```

---

### Task 7: Public Runtime Factory and Single-GPU Worker CLI

**Files:**
- Create: `avgaussianv2/runtime.py`
- Modify: `avgaussianv2/cli/train.py`
- Create: `avgaussianv2/cli/pilot_worker.py`
- Create: `tests/test_pilot_worker.py`

- [ ] **Step 1: Write failing worker CLI tests**

```python
def test_condition_off_skips_configured_warmup(tmp_path, project_config, fake_factory):
    result = run_worker(
        project_config,
        SHORT_PILOT,
        Variant.CONDITION_OFF,
        tmp_path,
        "cpu",
        fake_factory,
    )
    assert result.completed_warmup_steps == 0


def test_worker_writes_required_files(tmp_path, fake_factory, project_config):
    result = run_worker(project_config, SHORT_PILOT, Variant.FROZEN_VISUAL, tmp_path, "cpu", fake_factory)
    assert result.variant == "frozen_visual"
    for name in ("best.pt", "latest.pt", "training_curve.csv", "worker_summary.json"):
        assert (tmp_path / name).is_file()
```

- [ ] **Step 2: Run tests and verify worker APIs are missing**

Run: `uv run --with pytest pytest tests/test_pilot_worker.py -v`

Expected: import failure for `avgaussianv2.cli.pilot_worker`.

- [ ] **Step 3: Extract the shared runtime factory**

Move `TrainingBundle` and `_default_backend_factory` from `avgaussianv2/cli/train.py` into:

```python
# avgaussianv2/runtime.py
@dataclass(frozen=True)
class TrainingBundle:
    model: nn.Module
    train_samples: Sequence[AlignedAVSample]
    eval_samples: Sequence[AlignedAVSample]
    audio_loss_fn: AudioLoss


def build_runtime(config: ProjectConfig, device: torch.device) -> TrainingBundle:
    visual = FTGSVisualBackend.load(config.paths.visual_checkpoint, config.paths.visual_upstream_root)
    audio = AudioGSBackend.load(
        config.paths.audio_checkpoint,
        embedding_dim=config.model.embedding_dim,
        upstream_root=config.paths.audio_upstream_root,
        model_class=config.model.audio_model_class,
    )
    model = AVGaussianFusionV2(visual, RGBDConditionEncoder(config.model.embedding_dim, config.model.alpha_threshold), audio).to(device)
    return TrainingBundle(
        model=model,
        train_samples=AlignedAVDataset(config, "train"),
        eval_samples=AlignedAVDataset(config, "eval"),
        audio_loss_fn=audio.build_criterion().to(device),
    )
```

Update the existing training CLI to consume `bundle.train_samples`; keep its public behavior and
tests unchanged.

- [ ] **Step 4: Implement worker parser and run function**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=tuple(Variant))
    parser.add_argument("--shared-indices", required=True, type=Path)
    parser.add_argument("--visual-baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser
```

`run_worker` loads the shared manifest, creates one runtime, creates `Evaluator` and
`PilotTrainer`, validates empty-output/resume rules, executes the variant, and exits nonzero on
any missing artifact or non-finite result.

- [ ] **Step 5: Run worker and old CLI tests, then commit**

```bash
uv run --with pytest pytest tests/test_pilot_worker.py tests/test_integration.py -v
git add avgaussianv2/runtime.py avgaussianv2/cli/train.py avgaussianv2/cli/pilot_worker.py tests
git commit -m "feat: add single gpu pilot worker"
```

---

### Task 8: Cross-System Comparison and Acceptance Report

**Files:**
- Create: `avgaussianv2/experiment/report.py`
- Create: `tests/test_experiment_report.py`

- [ ] **Step 1: Write failing report tests**

```python
def test_report_distinguishes_all_five_systems(tmp_path):
    report = build_comparison(SUMMARIES, output_dir=tmp_path)
    assert set(report.systems) == {
        "baseline_imported", "joint_conditioned_on", "joint_conditioned_off",
        "frozen_visual_on", "condition_off",
    }
    assert (tmp_path / "comparison.csv").is_file()
    assert (tmp_path / "comparison.md").is_file()


def test_acceptance_requires_mean_and_median_condition_gain():
    decision = decide_long_training(READY_SUMMARIES)
    assert decision.ready
    changed = copy.deepcopy(READY_SUMMARIES)
    changed["joint_conditioned_on"]["paired_audio_delta_median"] = 0.01
    assert not decide_long_training(changed).ready


def test_paired_condition_delta_joins_identical_sample_ids():
    paired = paired_audio_deltas(CONDITION_ON_RECORDS, CONDITION_OFF_RECORDS)
    assert paired["sample_count"] == len(CONDITION_ON_RECORDS)
    assert paired["median"] == pytest.approx(EXPECTED_MEDIAN)
```

- [ ] **Step 2: Run tests and verify report APIs are missing**

Run: `uv run --with pytest pytest tests/test_experiment_report.py -v`

Expected: import failure for `avgaussianv2.experiment.report`.

- [ ] **Step 3: Implement comparison rows and decision contract**

```python
@dataclass(frozen=True)
class PilotDecision:
    ready: bool
    reasons: tuple[str, ...]


def decide_long_training(systems: Mapping[str, dict]) -> PilotDecision:
    conditioned = systems["joint_conditioned_on"]
    unconditioned = systems["joint_conditioned_off"]
    reasons = []
    if not conditioned["visual_feasible"]:
        reasons.append("joint_conditioned best checkpoint violates visual constraints")
    if conditioned["audio_total"]["mean"] >= unconditioned["audio_total"]["mean"]:
        reasons.append("condition-on does not improve mean held-out audio_total")
    if conditioned["paired_audio_delta_median"] >= 0:
        reasons.append("condition-on does not improve median per-sample audio_total")
    if conditioned["max_audio_to_visual_grad_norm"] <= 0:
        reasons.append("audio loss never reached visual Gaussians")
    return PilotDecision(ready=not reasons, reasons=tuple(reasons))
```

- [ ] **Step 4: Implement deterministic JSON/CSV/Markdown writers**

`comparison.csv` contains one row per system and flattened mean/std/median metrics.
`paired_audio_deltas` must join condition-on/off per-sample records by the stable evaluator
sample ID, reject missing/duplicate IDs, and compute `on.audio_total - off.audio_total` before
aggregation. `comparison.md` includes baseline deltas, visual feasibility, stop step/reason,
paired condition effect, frozen-visual comparison, condition-off continuation comparison, and
the exact `PilotDecision.reasons` list. Sort system and metric keys to make repeated runs
diffable.

- [ ] **Step 5: Run report tests and commit**

```bash
uv run --with pytest pytest tests/test_experiment_report.py -v
git add avgaussianv2/experiment/report.py tests/test_experiment_report.py
git commit -m "feat: report pilot comparisons"
```

---

### Task 9: Three-GPU Pilot Orchestrator

**Files:**
- Create: `avgaussianv2/cli/pilot.py`
- Create: `avgaussianv2/cli/pilot_eval.py`
- Create: `tests/test_pilot_orchestrator.py`
- Create: `tests/test_pilot_eval.py`
- Create: `scripts/pilot_scene1_opera.sh`

- [ ] **Step 1: Write failing orchestration tests with a fake process runner**

```python
def test_pilot_assigns_one_variant_per_gpu(tmp_path, fake_runner):
    result = run_pilot(CONFIG, tmp_path, gpu_ids=(0, 1, 2), process_runner=fake_runner)
    assert fake_runner.assignments == {
        "joint_conditioned": "0",
        "frozen_visual": "1",
        "condition_off": "2",
    }
    assert result.decision_file == tmp_path / "comparison.md"


def test_nonempty_output_requires_resume(tmp_path):
    (tmp_path / "existing.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="--resume"):
        run_pilot(CONFIG, tmp_path, gpu_ids=(0, 1, 2), process_runner=FakeRunner())


def test_worker_failure_preserves_logs_and_fails_pilot(tmp_path, failing_runner):
    with pytest.raises(PilotProcessError, match="frozen_visual"):
        run_pilot(CONFIG, tmp_path, gpu_ids=(0, 1, 2), process_runner=failing_runner)
    assert (tmp_path / "workers/frozen_visual/stderr.log").is_file()


def test_verify_only_checks_outputs_without_starting_processes(tmp_path, complete_pilot, fake_runner):
    verify_pilot_outputs(complete_pilot)
    assert fake_runner.assignments == {}
```

- [ ] **Step 2: Run tests and verify pilot CLI is missing**

Run: `uv run --with pytest pytest tests/test_pilot_orchestrator.py -v`

Expected: import failure for `avgaussianv2.cli.pilot`.

- [ ] **Step 3: Implement preflight, baseline, and shared manifest**

The parent process must validate exactly three distinct GPU IDs, all configured checkpoint and
manifest paths, output directory policy, and `scene_id == "scene1_opera"`. It builds shared
indices, hashes upstream checkpoints, writes `experiment_manifest.json`, and runs the immutable
baseline evaluation on the first GPU before launching workers.

- [ ] **Step 4: Implement concurrent worker launch with isolated CUDA visibility**

```python
VARIANT_GPU = zip(
    (Variant.JOINT_CONDITIONED, Variant.FROZEN_VISUAL, Variant.CONDITION_OFF),
    gpu_ids,
    strict=True,
)
for variant, gpu in VARIANT_GPU:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    command = [
        sys.executable, "-m", "avgaussianv2.cli.pilot_worker",
        "--config", str(config_path),
        "--variant", variant.value,
        "--shared-indices", str(manifest_path),
        "--visual-baseline", str(baseline_summary_path),
        "--output-dir", str(output_dir / "workers" / variant.value),
        "--device", "cuda:0",
    ]
    processes.append(process_runner.start(command, env, stdout_path, stderr_path))
```

Wait for every process, collect every exit status, and raise one `PilotProcessError` listing all
failed variants. Do not discard successful worker outputs.

- [ ] **Step 5: Implement full-split final evaluation and reporting**

Implement `avgaussianv2.cli.pilot_eval` with `--config`, optional `--checkpoint`, repeatable
`--condition {on,off}`, `--system-name`, `--output-dir`, and `--device`. It constructs the public
runtime, loads the checkpoint when supplied, evaluates the complete held-out split, and writes
one summary plus per-sample JSONL per requested condition. The baseline job omits `--checkpoint`
and evaluates the imported model with condition off.

After successful training, launch evaluation jobs on the same three GPUs to evaluate each
`best.pt`. The joint-conditioned evaluator emits both on and off systems from identical weights;
the other two emit `frozen_visual_on` and `condition_off`. Validate all five system summaries,
join paired records by sample ID, then call `build_comparison`.

- [ ] **Step 6: Add CLI and explicit scene script**

The parent parser accepts `--config`, `--output-dir`, `--gpus`, `--resume`, and `--verify-only`.
`--verify-only` must call `verify_pilot_outputs`, never construct a runtime or start a subprocess,
and exit nonzero for an unreadable checkpoint, missing/non-finite system metric, inconsistent
sample count, absent visual-feasibility field, or missing decision.

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AVGAUSSIANV2_PYTHON:-/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python}"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m avgaussianv2.cli.pilot \
  --config "$ROOT/configs/scene1_opera.yaml" \
  --output-dir "$ROOT/runs/pilot_scene1_opera" \
  --gpus 0,1,2
```

- [ ] **Step 7: Run orchestration tests and commit**

```bash
bash -n scripts/pilot_scene1_opera.sh
uv run --with pytest pytest tests/test_pilot_orchestrator.py tests/test_pilot_eval.py -v
git add avgaussianv2/cli/pilot.py avgaussianv2/cli/pilot_eval.py tests/test_pilot_orchestrator.py tests/test_pilot_eval.py scripts/pilot_scene1_opera.sh
git commit -m "feat: orchestrate three gpu scene1 pilot"
```

---

### Task 10: Documentation, Full Verification, and Real Scene1 Diagnostic

**Files:**
- Modify: `README.md`
- Modify only files required by verification failures
- Runtime output: `runs/pilot_scene1_opera/`

- [ ] **Step 1: Document pilot command, variants, metrics, outputs, and resume**

Add a `Scene1 diagnostic pilot` section containing:

```bash
scripts/pilot_scene1_opera.sh

# Resume only an interrupted compatible experiment
AVGAUSSIANV2_PYTHON=/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python \
python -m avgaussianv2.cli.pilot \
  --config configs/scene1_opera.yaml \
  --output-dir runs/pilot_scene1_opera \
  --gpus 0,1,2 \
  --resume
```

Document the five evaluated systems, lower/higher metric direction, PSNR/SSIM tolerances,
early-stopping rule, and long-training acceptance gate.

- [ ] **Step 2: Run Ruff and the complete CPU suite**

```bash
UV_CACHE_DIR=/tmp/avgaussianv2-uv-cache uv run --with ruff ruff check .
UV_CACHE_DIR=/tmp/avgaussianv2-uv-cache uv run --with pytest --with torch --with numpy --with soundfile --with pyyaml pytest -v
```

Expected: Ruff emits `All checks passed!`; every test passes with zero failures.

- [ ] **Step 3: Run the real three-GPU pilot**

```bash
scripts/pilot_scene1_opera.sh
```

Expected:

- baseline evaluation exits 0;
- all three worker processes exit 0;
- each worker contains finite `best.pt`, `latest.pt`, `training_curve.csv`, and
  `worker_summary.json`;
- complete `cam10` evaluations exist for all five systems;
- `comparison.csv` and `comparison.md` exist;
- the report contains an explicit ready/not-ready decision with reasons.

- [ ] **Step 4: Verify runtime outputs programmatically**

```bash
/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python -m avgaussianv2.cli.pilot \
  --config configs/scene1_opera.yaml \
  --output-dir runs/pilot_scene1_opera \
  --gpus 0,1,2 \
  --verify-only
```

Expected: exits 0 after checking checkpoint readability, finite metrics, required systems,
sample counts, visual feasibility fields, and comparison decision.

- [ ] **Step 5: Commit implementation/docs and push the verified branch**

```bash
git status --short
git diff --check
git add README.md avgaussianv2 scripts tests pyproject.toml uv.lock
git commit -m "feat: add scene1 diagnostic training evaluation"
git push -u origin agent/scene1-pilot
```

Skip the commit if no files changed after Task 9. Do not merge into `main` until the real report
has been reviewed and all Critical/Important code-review findings are resolved.

---

## Plan Self-Review Checklist

- Execution starts from an isolated feature worktree (Task 0).
- Every approved spec section maps to a task: sampling (Task 1), metrics (Task 2), evaluation
  (Task 3), selection (Task 4), training (Task 5), resume (Task 6), worker (Task 7), reporting
  (Task 8), orchestration (Task 9), and real acceptance (Task 10).
- Variant names, step counts, validation interval, patience, relative delta, and visual
  tolerances are identical to the approved design.
- The condition-off joint sequence is exactly the conditioned variants' joint sequence; it does
  not receive condition warmup.
- Final comparison includes the imported baseline and paired on/off evaluation from identical
  joint-conditioned weights.
- No long `Scene7playing` training or hyperparameter sweep is included.

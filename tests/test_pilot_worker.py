from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from avgaussianv2.config import (
    ModelConfig,
    PathConfig,
    ProjectConfig,
    SceneConfig,
    TrainConfig,
)
from avgaussianv2.experiment.checkpoint import hash_index_manifest, sha256_file
from avgaussianv2.experiment.contracts import PilotConfig, Variant
from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write_config(tmp_path: Path) -> tuple[Path, ProjectConfig]:
    visual = tmp_path / "visual.pt"
    audio = tmp_path / "audio.pt"
    dataset_manifest = tmp_path / "dataset.json"
    visual.write_bytes(b"visual")
    audio.write_bytes(b"audio")
    dataset_manifest.write_text('{"scene_id":"scene1_opera"}\n')
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
scene:
  id: scene1_opera
  fps: 20
  train_cameras: [cam00]
  eval_cameras: [cam01]
  camera_mapping: {{cam00: 0, cam01: 1}}
paths:
  visual_upstream_root: {tmp_path}
  audio_upstream_root: {tmp_path}
  visual_checkpoint: {visual}
  audio_checkpoint: {audio}
  manifest: {dataset_manifest}
model:
  embedding_dim: 8
  n_fft: 16
  hop_length: 4
  win_length: 8
  sample_rate: 16000
train:
  warmup_steps: 1
  joint_steps: 1
  seed: 7
"""
    )
    config = ProjectConfig(
        scene=SceneConfig(
            "scene1_opera", 20.0, ("cam00",), ("cam01",), {"cam00": 0, "cam01": 1}
        ),
        paths=PathConfig(
            tmp_path, tmp_path, visual, audio, dataset_manifest
        ),
        model=ModelConfig(
            embedding_dim=8,
            n_fft=16,
            hop_length=4,
            win_length=8,
            sample_rate=16000,
        ),
        train=TrainConfig(warmup_steps=1, joint_steps=1, seed=7),
    )
    return config_path, config


def _manifest_files(
    tmp_path: Path,
    *,
    pilot: PilotConfig | None = None,
    train_size: int = 2,
    eval_size: int = 2,
) -> tuple[Path, Path, Path, ProjectConfig]:
    from avgaussianv2.cli.pilot_worker import MANIFEST_SCHEMA, MANIFEST_VERSION

    config_path, config = _write_config(tmp_path)
    pilot = pilot or PilotConfig(
        warmup_steps=1,
        joint_steps=1,
        validation_interval=1,
        minimum_joint_steps=1,
        patience=2,
        quick_validation_samples=1,
    )
    baseline = {
        name: {"mean": 0.5, "std": 0.1, "median": 0.5}
        for name in (
            "audio_total",
            "audio_mono",
            "audio_diff",
            "waveform_l1",
            "mono_lsd",
            "diff_lsd",
            "lre_error_db",
            "rgb_l1",
        )
    }
    baseline["rgb_psnr"] = {"mean": 30.0, "std": 0.2, "median": 30.0}
    baseline["rgb_ssim"] = {"mean": 0.95, "std": 0.01, "median": 0.95}
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline, sort_keys=True) + "\n")
    shared = {
        "warmup": [index % train_size for index in range(pilot.warmup_steps)],
        "joint": [(index + 1) % train_size for index in range(pilot.joint_steps)],
        "quick_heldout": list(
            range(min(eval_size, pilot.quick_validation_samples))
        ),
    }
    pilot_seed = 7

    compatibilities = {}
    for variant in Variant:
        indices = {
            "warmup": [] if variant == Variant.CONDITION_OFF else shared["warmup"],
            "joint": shared["joint"],
            "quick_heldout": shared["quick_heldout"],
        }
        compatibilities[variant.value] = {
            "scene_id": config.scene.scene_id,
            "variant": variant.value,
            "seed": pilot_seed,
            "index_hash": hash_index_manifest(indices),
            "visual_checkpoint_sha256": sha256_file(config.paths.visual_checkpoint),
            "audio_checkpoint_sha256": sha256_file(config.paths.audio_checkpoint),
            "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
            "n_fft": config.model.n_fft,
            "hop_length": config.model.hop_length,
            "win_length": config.model.win_length,
            "sample_rate": config.model.sample_rate,
        }
    payload = {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "scene_id": config.scene.scene_id,
        "seed": pilot_seed,
        "pilot_config": asdict(pilot),
        "shared_indices": {
            "warmup": shared["warmup"],
            "joint": shared["joint"],
        },
        "quick_heldout_indices": shared["quick_heldout"],
        "source_hashes": {
            "project_config_sha256": sha256_file(config_path),
            "dataset_manifest_sha256": sha256_file(config.paths.manifest),
            "visual_checkpoint_sha256": sha256_file(config.paths.visual_checkpoint),
            "audio_checkpoint_sha256": sha256_file(config.paths.audio_checkpoint),
            "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
        },
        "compatibility": compatibilities,
        "component_identities": {
            "model_class": "tests.worker.FakeModel-v1",
            "warmup_optimizer_factory": "avgaussianv2.train.build_warmup_optimizer-v1",
            "joint_optimizer_factory": "avgaussianv2.train.build_joint_optimizer-v1",
            "warmup_optimizer_class": "torch.optim.adam.Adam",
            "joint_optimizer_class": "torch.optim.adam.Adam",
            "warmup_step_fn": "avgaussianv2.train.condition_warmup_step-v1",
            "joint_step_fn": "avgaussianv2.train.joint_train_step-v1",
            "audio_loss_fn": "tests.worker.audio_loss-v1",
        },
        "runtime_identity": {
            "model_class": f"{_WorkerModel.__module__}.{_WorkerModel.__qualname__}",
            "model_format_version": "state-dict-v1",
        },
        "dataset_lengths": {"train": train_size, "eval": eval_size},
        "visual_baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
            "summary": baseline,
        },
    }
    manifest_path = tmp_path / "shared.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return config_path, manifest_path, baseline_path, config


class _Part(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))


class _WorkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Part(0.2)
        self.acoustic = _Part(0.1)
        self.condition_encoder = _Part(0.3)
        self.film = _Part(0.4)
        self.audio_unet = _Part(0.5)
        self.condition_enabled = True

    def forward(self, sample):
        visual = self.visual.value
        condition = visual * self.condition_encoder.value
        if not self.condition_enabled:
            condition = condition * 0
        gain = self.acoustic.value + self.audio_unet.value + condition * self.film.value
        rgb = visual.sigmoid().expand(1, 8, 8, 3)
        depth = (visual + 2).expand(1, 8, 8, 1)
        return FusionOutput(
            RGBDRender(rgb, depth, torch.ones_like(depth)),
            condition.reshape(1, 1),
            sample.source_audio * gain,
        )

    def named_parameter_groups(self):
        return {
            "visual": list(self.visual.parameters()),
            "acoustic": list(self.acoustic.parameters()),
            "condition_encoder": list(self.condition_encoder.parameters()),
            "film": list(self.film.parameters()),
            "audio_unet": list(self.audio_unet.parameters()),
        }

    def freeze_pretrained(self):
        self.visual.requires_grad_(False)
        self.acoustic.requires_grad_(False)
        self.audio_unet.requires_grad_(False)
        self.condition_encoder.requires_grad_(True)
        self.film.requires_grad_(True)

    def unfreeze_all(self):
        self.requires_grad_(True)


def _sample(frame: int) -> AlignedAVSample:
    return AlignedAVSample(
        scene_id="scene1_opera",
        camera="cam00",
        frame_index=frame,
        time_seconds=frame / 20,
        visual_time=torch.tensor([[0.1]]),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 12),
        source_audio=torch.full((1, 2, 32), 0.25),
        target_audio=torch.full((1, 2, 32), 0.5),
        target_rgb=torch.full((1, 8, 8, 3), 0.4),
        image_size=(8, 8),
    )


def _audio_loss(predicted, target):
    value = torch.nn.functional.mse_loss(predicted, target)
    return {"total_loss": value, "mono_loss": value, "diff_loss": value}


class _FeasibleEvaluator:
    instances = []

    def __init__(self, model, audio_loss_fn, device):
        self.model = model
        self.audio_loss_fn = audio_loss_fn
        self.device = device
        self.calls = []
        self.instances.append(self)

    def evaluate(self, samples, indices, system_name, condition_enabled, output_dir):
        from avgaussianv2.experiment.contracts import EvaluationResult

        self.calls.append((samples, tuple(indices), condition_enabled))
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            name: {"mean": 0.5, "std": 0.1, "median": 0.5}
            for name in (
                "audio_total",
                "audio_mono",
                "audio_diff",
                "waveform_l1",
                "mono_lsd",
                "diff_lsd",
                "lre_error_db",
                "rgb_l1",
            )
        }
        summary["rgb_psnr"] = {"mean": 30.0, "std": 0.2, "median": 30.0}
        summary["rgb_ssim"] = {"mean": 0.95, "std": 0.01, "median": 0.95}
        return EvaluationResult(system_name, len(indices), (), summary)


def _fake_runtime(config, device):
    del config
    from avgaussianv2.runtime import TrainingBundle

    return TrainingBundle(
        _WorkerModel().to(device),
        [_sample(0), _sample(1)],
        [_sample(2), _sample(3)],
        _audio_loss,
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(extra=True), "fields mismatch"),
        (lambda value: value.pop("scene_id"), "fields mismatch"),
        (lambda value: value.__setitem__("scene_id", "wrong"), "scene"),
        (
            lambda value: value["source_hashes"].__setitem__(
                "visual_checkpoint_sha256", "0" * 64
            ),
            "visual checkpoint",
        ),
        (
            lambda value: value["shared_indices"].__setitem__("joint", [99]),
            "compatibility",
        ),
        (
            lambda value: value["dataset_lengths"].__setitem__("train", 0),
            "positive",
        ),
    ],
)
def test_manifest_errors_are_rejected_before_runtime_factory(
    tmp_path, mutation, match
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    payload = json.loads(manifest.read_text())
    mutation(payload)
    manifest.write_text(json.dumps(payload))
    calls = []

    with pytest.raises((TypeError, ValueError), match=match):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=lambda *_: calls.append(True),
        )
    assert calls == []


def test_baseline_mismatch_is_rejected_before_runtime_factory(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    changed = json.loads(baseline.read_text())
    changed["rgb_psnr"]["mean"] = 1
    baseline.write_text(json.dumps(changed) + "\n")
    calls = []
    with pytest.raises(ValueError, match="visual baseline hash"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=lambda *_: calls.append(True),
        )
    assert calls == []


@pytest.mark.parametrize("invalid", [{}, {"bool": True}, {"nan": float("nan")}])
def test_invalid_baseline_file_schema_is_rejected_before_runtime_factory(
    tmp_path, invalid
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    baseline.write_text(json.dumps(invalid))
    calls = []
    with pytest.raises((TypeError, ValueError)):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=lambda *_: calls.append(True),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda summary: summary.clear(), "metric fields"),
        (lambda summary: summary.pop("audio_mono"), "metric fields"),
        (lambda summary: summary.update(extra={}), "metric fields"),
        (
            lambda summary: summary["audio_total"].pop("median"),
            "aggregate fields",
        ),
        (
            lambda summary: summary["audio_total"].update(extra=1),
            "aggregate fields",
        ),
        (
            lambda summary: summary["audio_total"].__setitem__("mean", "bad"),
            "numeric",
        ),
        (
            lambda summary: summary["audio_total"].__setitem__("mean", True),
            "numeric",
        ),
        (
            lambda summary: summary["audio_total"].__setitem__("std", -0.1),
            "std.*nonnegative",
        ),
        (
            lambda summary: summary["audio_total"].__setitem__("median", -0.1),
            "nonnegative",
        ),
        (
            lambda summary: summary["rgb_l1"].__setitem__("mean", 1.1),
            "at most 1",
        ),
        (
            lambda summary: summary["rgb_ssim"].__setitem__("median", 1.1),
            r"\[-1, 1\]",
        ),
    ],
)
def test_invalid_baseline_schema_is_rejected_before_runtime_factory(
    tmp_path, mutation, match
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    payload = json.loads(manifest.read_text())
    summary = payload["visual_baseline"]["summary"]
    mutation(summary)
    baseline.write_text(json.dumps(summary, sort_keys=True) + "\n")
    payload["visual_baseline"]["sha256"] = sha256_file(baseline)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    calls = []
    with pytest.raises((TypeError, ValueError), match=match):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=lambda *_: calls.append(True),
        )
    assert calls == []


def test_nonfinite_baseline_is_rejected_before_runtime_factory(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["visual_baseline"]["summary"]["rgb_psnr"]["mean"] = float("nan")
    manifest.write_text(json.dumps(payload))
    calls = []
    with pytest.raises(ValueError, match="non-finite"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=lambda *_: calls.append(True),
        )
    assert calls == []


def test_runtime_dataset_mismatch_is_rejected_after_one_factory_call(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.runtime import TrainingBundle

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    calls = []

    def factory(*_):
        calls.append(True)
        return TrainingBundle(
            _WorkerModel(), [object()], [object(), object()], _audio_loss
        )

    with pytest.raises(ValueError, match="training dataset length"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=factory,
        )
    assert calls == [True]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_class", "tests.worker.WrongModel", "model class"),
        ("model_format_version", "state-dict-v2", "model format"),
    ],
)
def test_runtime_must_match_manifest_declared_identity(
    tmp_path, field, value, match
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["runtime_identity"][field] = value
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    calls = []
    with pytest.raises(ValueError, match=match):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=lambda *args: calls.append(args) or _fake_runtime(*args),
        )
    assert len(calls) == 1


def test_worker_rejects_checkpoint_changed_during_runtime_load(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, project = _manifest_files(tmp_path)
    calls = []

    def factory(*args):
        calls.append(True)
        project.paths.visual_checkpoint.write_bytes(b"changed")
        return _fake_runtime(*args)

    with pytest.raises(ValueError, match="changed after runtime load"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            tmp_path / "run",
            device="cpu",
            runtime_factory=factory,
        )
    assert calls == [True]


@pytest.mark.parametrize(
    ("variant", "expected_warmup", "condition_enabled"),
    [
        (Variant.FROZEN_VISUAL, 1, True),
        (Variant.CONDITION_OFF, 0, False),
    ],
)
def test_worker_runs_exact_variant_and_writes_inspectable_outputs(
    tmp_path, variant, expected_warmup, condition_enabled
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    _FeasibleEvaluator.instances.clear()
    result = run_worker(
        config,
        variant,
        manifest,
        baseline,
        tmp_path / "run",
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )

    assert result.completed_warmup_steps == expected_warmup
    assert result.completed_joint_steps == 1
    assert result.best_step == 1
    assert _FeasibleEvaluator.instances[0].calls[0][2] is condition_enabled
    output = tmp_path / "run"
    for name in ("best.pt", "latest.pt", "training_curve.csv", "worker_summary.json"):
        assert (output / name).is_file()
    assert torch.load(output / "latest.pt", weights_only=True)["stage"] == "complete"
    assert torch.load(output / "best.pt", weights_only=True)["checkpoint_kind"] == "best"
    summary = json.loads((output / "worker_summary.json").read_text())
    assert summary["worker"]["variant"] == variant.value
    assert summary["worker"]["device"] == "cpu"


def test_worker_resumes_interrupted_run_exactly_and_complete_is_finalize_only(
    tmp_path,
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.experiment.training import PilotTrainer
    from avgaussianv2.train import joint_train_step

    pilot = PilotConfig(
        warmup_steps=0,
        joint_steps=2,
        validation_interval=2,
        minimum_joint_steps=2,
        patience=2,
        quick_validation_samples=1,
    )
    config, manifest, baseline, _ = _manifest_files(tmp_path, pilot=pilot)
    output = tmp_path / "run"
    calls = 0

    def interrupted_factory(config, evaluator, *, train_config):
        def interrupted_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return joint_train_step(*args, **kwargs)

        return PilotTrainer(
            config,
            evaluator,
            train_config=train_config,
            joint_step_fn=interrupted_step,
        )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            runtime_factory=_fake_runtime,
            evaluator_factory=_FeasibleEvaluator,
            trainer_factory=interrupted_factory,
        )
    assert torch.load(output / "latest.pt", weights_only=True)[
        "next_joint_position"
    ] == 1

    resumed = run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        resume=True,
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    assert resumed.completed_joint_steps == 2
    before = (output / "latest.pt").read_bytes()

    finalized = run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        resume=True,
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    assert finalized == resumed
    assert (output / "latest.pt").read_bytes() == before
    summary = json.loads((output / "worker_summary.json").read_text())
    assert summary["checkpoint_io"]["save_count"] == 0


def test_resume_rejects_corrupt_checkpoint_before_runtime_factory(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "latest.pt").write_bytes(b"not a safe checkpoint")
    calls = []
    with pytest.raises(Exception, match="checkpoint|load|invalid|read"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            resume=True,
            runtime_factory=lambda *_: calls.append(True),
        )
    assert calls == []


def test_resume_uses_canonical_manifest_semantics_not_whitespace(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.experiment.checkpoint import PilotResumeError

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    # Formatting-only changes preserve canonical validated semantics.
    manifest.write_text(json.dumps(json.loads(manifest.read_text())))
    calls = []

    def factory(*args):
        calls.append(args)
        return _fake_runtime(*args)

    result = run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        resume=True,
        runtime_factory=factory,
        evaluator_factory=_FeasibleEvaluator,
    )
    assert result.best_step == 1
    assert len(calls) == 1


def test_resume_preserves_and_validates_explicit_model_class_identity(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.experiment.checkpoint import PilotResumeError

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    payload = json.loads(manifest.read_text())
    payload["component_identities"]["model_class"] = "tests.worker.OtherModel-v2"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    calls = []
    with pytest.raises(PilotResumeError, match="fingerprint"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            resume=True,
            runtime_factory=lambda *args: calls.append(args) or _fake_runtime(*args),
            evaluator_factory=_FeasibleEvaluator,
        )
    assert calls == []


def test_resume_rejects_changed_baseline_values_before_runtime_factory(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.experiment.checkpoint import PilotResumeError

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    payload = json.loads(manifest.read_text())
    payload["visual_baseline"]["summary"]["rgb_psnr"]["mean"] = 29.0
    baseline.write_text(
        json.dumps(payload["visual_baseline"]["summary"], sort_keys=True) + "\n"
    )
    payload["visual_baseline"]["sha256"] = sha256_file(baseline)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    calls = []
    with pytest.raises(PilotResumeError, match="fingerprint"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            resume=True,
            runtime_factory=lambda *args: calls.append(args) or _fake_runtime(*args),
            evaluator_factory=_FeasibleEvaluator,
        )
    assert calls == []


def test_resume_rejects_changed_project_config_before_runtime_factory(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.experiment.checkpoint import PilotResumeError

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    config.write_text(config.read_text().replace("embedding_dim: 8", "embedding_dim: 9"))
    payload = json.loads(manifest.read_text())
    payload["source_hashes"]["project_config_sha256"] = sha256_file(config)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    calls = []
    with pytest.raises(PilotResumeError, match="fingerprint"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            resume=True,
            runtime_factory=lambda *args: calls.append(args) or _fake_runtime(*args),
            evaluator_factory=_FeasibleEvaluator,
        )
    assert calls == []


def test_worker_acquires_lock_before_runtime_and_loser_factory_is_not_called(
    tmp_path,
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.experiment.checkpoint import PilotResumeError

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    loser_calls = []

    def winner_factory(*args):
        with pytest.raises(PilotResumeError, match="owned by another process"):
            run_worker(
                config,
                Variant.FROZEN_VISUAL,
                manifest,
                baseline,
                output,
                device="cpu",
                runtime_factory=lambda *_: loser_calls.append(True),
            )
        raise RuntimeError("stop winner")

    with pytest.raises(RuntimeError, match="stop winner"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            runtime_factory=winner_factory,
        )
    assert loser_calls == []


def test_worker_releases_lock_when_runtime_factory_fails(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="factory failed"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            runtime_factory=lambda *_: (_ for _ in ()).throw(
                RuntimeError("factory failed")
            ),
        )
    result = run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    assert result.best_step == 1


def test_complete_resume_loads_latest_checkpoint_exactly_once(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    original_load = checkpoint.torch.load
    latest_loads = 0

    def counting_load(path, *args, **kwargs):
        nonlocal latest_loads
        loaded_path = (
            Path(os.readlink(f"/proc/self/fd/{path.fileno()}"))
            if hasattr(path, "fileno")
            else Path(path)
        )
        if loaded_path == output / "latest.pt":
            latest_loads += 1
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", counting_load)
    run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        resume=True,
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    assert latest_loads == 1


def test_resume_rejects_changed_trust_mode_before_runtime_factory(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker
    from avgaussianv2.experiment.checkpoint import PilotResumeError

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    run_worker(
        config,
        Variant.FROZEN_VISUAL,
        manifest,
        baseline,
        output,
        device="cpu",
        runtime_factory=_fake_runtime,
        evaluator_factory=_FeasibleEvaluator,
    )
    calls = []
    with pytest.raises(PilotResumeError, match="fingerprint"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            resume=True,
            trust_upstream_artifacts=True,
            runtime_factory=lambda *args: calls.append(args) or _fake_runtime(*args),
            evaluator_factory=_FeasibleEvaluator,
        )
    assert calls == []


@pytest.mark.parametrize("name", [".pilot.lock", ".pilot-best-backup.pt"])
def test_worker_rejects_checkpoint_control_symlinks_without_touching_target(
    tmp_path, name
) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    target = tmp_path / "target"
    target.write_text("unchanged")
    (output / name).symlink_to(target)
    calls = []
    with pytest.raises((OSError, ValueError), match="symlink|regular|unsafe"):
        run_worker(
            config,
            Variant.FROZEN_VISUAL,
            manifest,
            baseline,
            output,
            device="cpu",
            runtime_factory=lambda *_: calls.append(True),
        )
    assert target.read_text() == "unchanged"
    assert calls == []


def test_parser_has_exact_worker_surface_and_validates_devices() -> None:
    from avgaussianv2.cli.pilot_worker import build_parser, pilot_index_hash
    from avgaussianv2.experiment.contracts import SharedIndices

    shared = SharedIndices((1,), (2,))
    assert pilot_index_hash(shared, (3,), Variant.CONDITION_OFF) == hash_index_manifest(
        {"warmup": [], "joint": [2], "quick_heldout": [3]}
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "--config", "c.yaml",
            "--variant", "condition_off",
            "--shared-indices", "shared.json",
            "--visual-baseline", "baseline.json",
            "--output-dir", "out",
        ]
    )
    assert args.device == "cuda:0"
    assert args.variant == Variant.CONDITION_OFF
    assert args.trust_upstream_artifacts is False
    assert not hasattr(args, "overwrite")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config", "c",
                "--variant", "nope",
                "--shared-indices", "m",
                "--visual-baseline", "b",
                "--output-dir", "o",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config", "c",
                "--variant", "condition_off",
                "--shared-indices", "m",
                "--visual-baseline", "b",
                "--output-dir", "o",
                "--device", "cuda",
            ]
        )


def test_worker_help_does_not_import_runtime_or_upstream_modules() -> None:
    script = """
import runpy
import sys
sys.argv = ["pilot_worker", "--help"]
try:
    runpy.run_module("avgaussianv2.cli.pilot_worker", run_name="__main__")
except SystemExit as error:
    assert error.code == 0
assert "avgaussianv2.runtime" not in sys.modules
assert not any(name.startswith(("scene", "arguments", "gaussian_renderer")) for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_fresh_nonempty_and_resume_missing_refuse_before_factory(tmp_path) -> None:
    from avgaussianv2.cli.pilot_worker import run_worker

    config, manifest, baseline, _ = _manifest_files(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "foreign").write_text("keep")
    calls = []
    with pytest.raises(FileExistsError, match="nonempty"):
        run_worker(
            config, Variant.FROZEN_VISUAL, manifest, baseline, output,
            device="cpu", runtime_factory=lambda *_: calls.append(True),
        )
    assert (output / "foreign").read_text() == "keep"
    assert calls == []

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="latest"):
        run_worker(
            config, Variant.FROZEN_VISUAL, manifest, baseline, missing,
            device="cpu", resume=True, runtime_factory=lambda *_: calls.append(True),
        )
    assert calls == []


def test_public_runtime_factory_loads_configured_backends_and_both_splits(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.runtime as runtime

    _, config = _write_config(tmp_path)
    calls = []
    visual = nn.Linear(1, 1)
    audio = nn.Linear(1, 1)
    criterion = nn.Linear(1, 1)
    audio.build_criterion = lambda: criterion

    monkeypatch.setattr(
        runtime,
        "FTGSVisualBackend",
        SimpleNamespace(
            load=lambda checkpoint, root: calls.append(
                ("visual", checkpoint, root)
            )
            or visual
        ),
    )
    monkeypatch.setattr(
        runtime,
        "AudioGSBackend",
        SimpleNamespace(
            load=lambda checkpoint, **kwargs: calls.append(
                ("audio", checkpoint, kwargs)
            )
            or audio
        )
    )
    encoders = []
    monkeypatch.setattr(
        runtime,
        "RGBDConditionEncoder",
        lambda **kwargs: encoders.append(kwargs) or nn.Linear(1, 1),
    )

    class FakeFusion(nn.Module):
        def __init__(self, **parts):
            super().__init__()
            self.parts = parts

    monkeypatch.setattr(runtime, "AVGaussianFusionV2", FakeFusion)
    monkeypatch.setattr(
        runtime,
        "AlignedAVDataset",
        lambda cfg, split: calls.append(("dataset", cfg, split)) or [split],
    )

    bundle = runtime.build_runtime(
        config, torch.device("cpu"), trusted_upstream_artifacts=True
    )

    assert bundle.train_samples == ["train"]
    assert bundle.eval_samples == ["eval"]
    assert bundle.audio_loss_fn is criterion
    assert bundle.model.parts["visual"] is visual
    assert bundle.model.parts["audio"] is audio
    assert encoders == [
        {
            "embedding_dim": config.model.embedding_dim,
            "alpha_threshold": config.model.alpha_threshold,
        }
    ]
    assert calls[:2] == [
        ("visual", config.paths.visual_checkpoint, config.paths.visual_upstream_root),
        (
            "audio",
            config.paths.audio_checkpoint,
            {
                "embedding_dim": config.model.embedding_dim,
                "upstream_root": config.paths.audio_upstream_root,
                "model_class": config.model.audio_model_class,
            },
        ),
    ]
    assert [call[-1] for call in calls if call[0] == "dataset"] == ["train", "eval"]


def test_runtime_refuses_untrusted_upstream_before_any_loader(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.runtime as runtime

    _, config = _write_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        runtime,
        "FTGSVisualBackend",
        SimpleNamespace(load=lambda *_args, **_kwargs: calls.append("visual")),
    )
    monkeypatch.setattr(
        runtime,
        "AudioGSBackend",
        SimpleNamespace(load=lambda *_args, **_kwargs: calls.append("audio")),
    )
    with pytest.raises(PermissionError, match="unsafe legacy pickle|trusted"):
        runtime.build_runtime(config, torch.device("cpu"))
    assert calls == []


def test_runtime_can_build_train_only_and_validates_datasets_before_device_move(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.runtime as runtime

    _, config = _write_config(tmp_path)
    events = []
    visual = nn.Linear(1, 1)
    criterion = nn.Linear(1, 1)
    audio = nn.Linear(1, 1)
    audio.build_criterion = lambda: criterion
    monkeypatch.setattr(
        runtime, "FTGSVisualBackend", SimpleNamespace(load=lambda *_: visual)
    )
    monkeypatch.setattr(
        runtime,
        "AudioGSBackend",
        SimpleNamespace(load=lambda *_args, **_kwargs: audio),
    )
    monkeypatch.setattr(runtime, "RGBDConditionEncoder", lambda **_: nn.Linear(1, 1))

    class OrderedFusion(nn.Module):
        def __init__(self, **_parts):
            super().__init__()

        def to(self, device):
            events.append(("model.to", str(device)))
            return self

    class OrderedCriterion(nn.Module):
        def to(self, device):
            events.append(("criterion.to", str(device)))
            return self

    criterion = OrderedCriterion()
    audio.build_criterion = lambda: criterion
    monkeypatch.setattr(runtime, "AVGaussianFusionV2", OrderedFusion)
    monkeypatch.setattr(
        runtime,
        "AlignedAVDataset",
        lambda _cfg, split: events.append(("dataset", split)) or [split],
    )

    bundle = runtime.build_runtime(
        config,
        torch.device("cpu"),
        trusted_upstream_artifacts=True,
        include_eval=False,
    )

    assert bundle.eval_samples is None
    assert events == [
        ("dataset", "train"),
        ("model.to", "cpu"),
        ("criterion.to", "cpu"),
    ]

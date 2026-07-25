from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from avgaussianv2.cli.pilot_eval import (
    EvaluationSpec,
    parse_evaluation_spec,
    run_evaluation,
)
from avgaussianv2.experiment.evaluation import METRIC_NAMES, _write_metric_pair
from avgaussianv2.experiment.metrics import aggregate_metrics


def test_evaluation_spec_matrix_is_exact() -> None:
    assert parse_evaluation_spec("joint_conditioned_on:on") == EvaluationSpec(
        "joint_conditioned_on", True
    )
    assert parse_evaluation_spec("joint_conditioned_off:off") == EvaluationSpec(
        "joint_conditioned_off", False
    )
    for invalid in (
        "baseline_imported:on",
        "frozen_visual_on:off",
        "condition_off:on",
        "unknown:off",
        "joint_conditioned_on",
    ):
        with pytest.raises(ValueError):
            parse_evaluation_spec(invalid)


def test_eval_help_is_cuda_lazy() -> None:
    script = (
        "import sys, runpy; "
        "sys.argv=['pilot_eval','--help']; "
        "runpy.run_module('avgaussianv2.cli.pilot_eval',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--trust-upstream-artifacts" in result.stdout


def _config(tmp_path: Path) -> tuple[Path, Path]:
    for name in ("visual.pt", "audio.pt", "dataset.json"):
        (tmp_path / name).write_bytes(name.encode())
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
scene:
  id: scene1_opera
  fps: 20
  train_cameras: [cam00]
  eval_cameras: [cam10]
  camera_mapping: {{cam00: 0, cam10: 10}}
paths:
  visual_upstream_root: {tmp_path}
  audio_upstream_root: {tmp_path}
  visual_checkpoint: {tmp_path / "visual.pt"}
  audio_checkpoint: {tmp_path / "audio.pt"}
  manifest: {tmp_path / "dataset.json"}
model: {{}}
train:
  seed: 7
"""
    )
    return config, tmp_path / "audio.pt"


class _Model:
    checkpoint_format_version = "state-dict-v1"
    condition_enabled = True


class _Evaluator:
    calls = []
    mutate: Path | None = None

    def __init__(self, model, loss, device):
        self.model = model

    def evaluate(self, samples, indices, system_name, condition_enabled, output_dir):
        from avgaussianv2.experiment.contracts import EvaluationResult

        selected = tuple(indices)
        self.calls.append((selected, system_name, condition_enabled))
        rows = []
        for index in selected:
            row = {
                "sample_id": f"sample-{index}",
                "scene_id": "scene1_opera",
                "camera": "cam10",
                "frame_index": index,
                "time_seconds": index / 20,
            }
            row.update({metric: 0.5 for metric in METRIC_NAMES})
            row["rgb_psnr"] = 30.0
            row["rgb_ssim"] = 0.95
            rows.append(row)
        summary = aggregate_metrics(
            [{metric: row[metric] for metric in METRIC_NAMES} for row in rows]
        )
        _write_metric_pair(Path(output_dir), rows, summary)
        if self.mutate is not None:
            self.mutate.write_bytes(b"changed")
        return EvaluationResult(system_name, len(rows), tuple(rows), summary)


def _runtime(config, device):
    del config, device
    return SimpleNamespace(
        model=_Model(),
        train_samples=(object(), object()),
        eval_samples=(object(), object(), object()),
        audio_loss_fn=lambda *_: {},
    )


def test_baseline_cpu_uses_full_indices_and_resume_is_zero_runtime(tmp_path) -> None:
    config, _ = _config(tmp_path)
    _Evaluator.calls.clear()
    output = tmp_path / "evaluation"
    calls = []

    def runtime(config, device):
        calls.append((config, device))
        return _runtime(config, device)

    result = run_evaluation(
        config,
        output,
        ("baseline_imported:off",),
        device="cpu",
        runtime_factory=runtime,
        evaluator_factory=_Evaluator,
    )
    assert len(calls) == 1
    assert _Evaluator.calls == [((0, 1, 2), "baseline_imported", False)]
    assert result.artifacts[0].evaluation_indices == (0, 1, 2)

    resumed = run_evaluation(
        config,
        output,
        ("baseline_imported:off",),
        device="cpu",
        resume=True,
        runtime_factory=lambda *_: pytest.fail("resume built a runtime"),
        evaluator_factory=_Evaluator,
    )
    assert resumed.manifest_path == result.manifest_path


def test_owned_partial_resume_restarts_but_source_mutation_is_rejected(tmp_path) -> None:
    config, audio = _config(tmp_path)
    output = tmp_path / "evaluation"
    partial = output / "baseline_imported"
    partial.mkdir(parents=True)
    (partial / "metrics_per_sample.jsonl").write_text("partial")
    run_evaluation(
        config,
        output,
        ("baseline_imported:off",),
        device="cpu",
        resume=True,
        runtime_factory=_runtime,
        evaluator_factory=_Evaluator,
    )
    assert (output / "evaluation_manifest.json").is_file()

    _Evaluator.mutate = audio
    try:
        with pytest.raises(ValueError, match="source changed"):
            run_evaluation(
                config,
                tmp_path / "mutated",
                ("baseline_imported:off",),
                device="cpu",
                runtime_factory=_runtime,
                evaluator_factory=_Evaluator,
            )
    finally:
        _Evaluator.mutate = None

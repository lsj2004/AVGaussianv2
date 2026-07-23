from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

import avgaussianv2.experiment.report as report_module
from avgaussianv2.experiment.contracts import EvaluationResult
from avgaussianv2.experiment.evaluation import METRIC_NAMES
from avgaussianv2.experiment.metrics import aggregate_metrics
from avgaussianv2.experiment.report import (
    REQUIRED_SYSTEMS,
    SystemReportInput,
    build_comparison,
    decide_long_training,
    paired_audio_deltas,
)
from avgaussianv2.experiment import PilotDecision as ExportedPilotDecision


def _row(index: int, audio: float, *, psnr: float = 30.0, ssim: float = 0.95):
    values = {
        name: audio + index * 0.01
        for name in METRIC_NAMES
    }
    values.update(rgb_psnr=psnr, rgb_ssim=ssim, rgb_l1=0.1)
    return {
        "sample_id": f"sample-{index}",
        "scene_id": "scene1_opera",
        "camera": f"cam{index:02d}",
        "frame_index": index,
        "time_seconds": index / 20.0,
        **values,
    }


def _evaluation(name: str, audio: tuple[float, ...], *, psnr=30.0, ssim=0.95):
    rows = tuple(_row(i, value, psnr=psnr, ssim=ssim) for i, value in enumerate(audio))
    summary = aggregate_metrics(
        [{metric: row[metric] for metric in METRIC_NAMES} for row in rows]
    )
    return EvaluationResult(name, len(rows), rows, summary)


def _worker(variant: str, grad: float = 0.2):
    validation_summary = {
        name: {"mean": 0.5, "std": 0.1, "median": 0.5}
        for name in METRIC_NAMES
    }
    validation_summary["rgb_psnr"] = {"mean": 30.0, "std": 0.1, "median": 30.0}
    validation_summary["rgb_ssim"] = {"mean": 0.95, "std": 0.01, "median": 0.95}
    return {
        "variant": variant,
        "completed_warmup_steps": 0,
        "completed_joint_steps": 1,
        "best_step": 1,
        "stop_reason": "max_steps",
        "training_history": [
            {
                "stage": "joint",
                "step": 1,
                "sample_index": 0,
                "total": 1.0,
                "audio_to_visual_grad_norm": grad,
                "losses": {"audio": 1.0},
                "gradient_norms": {"visual": grad},
            }
        ],
        "validation_history": [{"step": 1, "summary": validation_summary}],
        "selector_state": {
            "visual_baseline": {
                "rgb_psnr": {"mean": 30.0},
                "rgb_ssim": {"mean": 0.95},
            },
            "psnr_tolerance_db": 0.5,
            "ssim_tolerance": 0.01,
            "best_step": 1,
            "best_audio_total": 0.5,
            "last_step": 1,
        },
        "stopper_state": {
            "minimum_steps": 1,
            "patience": 2,
            "relative_delta": 0.01,
            "best": 0.5,
            "stale": 0,
            "last_step": 1,
        },
        "checkpoint_io": {
            "save_count": 1,
            "save_attempt_count": 1,
            "failed_save_count": 0,
            "save_bytes": 100,
            "save_duration_seconds": 0.1,
            "scope": "current_invocation",
            "backup_copy_count": 0,
            "backup_copy_attempt_count": 0,
            "backup_copy_failure_count": 0,
            "backup_copy_bytes": 0,
            "backup_copy_duration_seconds": 0.0,
            "cadence": "every_completed_optimizer_step",
        },
        "worker": {
            "variant": variant,
            "device": "cuda:0",
            "scene_id": "scene1_opera",
            "config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "visual_baseline_sha256": "c" * 64,
            "trusted_upstream_artifacts": False,
        },
    }


def _system(name: str, audio, *, psnr=30.0, ssim=0.95, grad=0.2):
    variants = {
        "joint_conditioned_on": "joint_conditioned",
        "joint_conditioned_off": "joint_conditioned",
        "frozen_visual_on": "frozen_visual",
        "condition_off": "condition_off",
    }
    condition = name.endswith("_on") if name != "baseline_imported" else False
    return SystemReportInput(
        name=name,
        evaluation=_evaluation(name, tuple(audio), psnr=psnr, ssim=ssim),
        worker_summary=None if name == "baseline_imported" else _worker(variants[name], grad),
        provenance={
            "scene_id": "scene1_opera",
            "checkpoint_sha256": (
                "d" * 64
                if name in {"joint_conditioned_on", "joint_conditioned_off"}
                else (name[0] * 64)
            ),
            "checkpoint_generation": (
                7 if name in {"joint_conditioned_on", "joint_conditioned_off"} else 1
            ),
            "condition_enabled": condition,
        },
    )


def _ready_systems():
    return [
        _system("baseline_imported", [1.0, 1.0, 1.0]),
        _system("joint_conditioned_on", [0.6, 0.7, 0.8]),
        _system("joint_conditioned_off", [0.9, 0.9, 0.9]),
        _system("frozen_visual_on", [0.75, 0.8, 0.85]),
        _system("condition_off", [0.95, 0.95, 0.95]),
    ]


def test_paired_audio_deltas_is_order_independent_and_sorted():
    on = [_row(2, 0.7), _row(0, 0.5), _row(1, 0.8)]
    off = [_row(1, 0.9), _row(2, 1.0), _row(0, 0.6)]
    result = paired_audio_deltas(on, off)
    assert [row["sample_id"] for row in result["records"]] == [
        "sample-0", "sample-1", "sample-2"
    ]
    assert [row["audio_total_delta"] for row in result["records"]] == pytest.approx(
        [-0.1, -0.1, -0.3]
    )
    assert result["sample_count"] == 3
    assert result["median"] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda on, off: on.append(dict(on[0])), "duplicate"),
        (lambda on, off: off.pop(), "sample IDs"),
        (lambda on, off: off[0].update(camera="other"), "metadata"),
        (lambda on, off: on[0].update(audio_total=float("nan")), "finite"),
    ],
)
def test_paired_audio_deltas_rejects_invalid_pairs(mutation, match):
    on = [_row(0, 0.5), _row(1, 0.7)]
    off = [_row(0, 0.8), _row(1, 0.9)]
    mutation(on, off)
    with pytest.raises((TypeError, ValueError), match=match):
        paired_audio_deltas(on, off)


def test_build_comparison_ready_and_writes_exact_deterministic_files(tmp_path):
    assert ExportedPilotDecision is report_module.PilotDecision
    result = build_comparison(_ready_systems(), tmp_path)
    assert result.decision.ready
    assert result.decision.reasons == ()
    assert set(result.systems) == set(REQUIRED_SYSTEMS)
    expected = {
        "comparison.json",
        "comparison.csv",
        "comparison.md",
        "paired_condition_deltas.jsonl",
    }
    assert {path.name for path in tmp_path.iterdir() if not path.name.startswith(".")} == expected
    json.loads((tmp_path / "comparison.json").read_text())
    rows = list(csv.DictReader((tmp_path / "comparison.csv").read_text().splitlines()))
    assert [row["system"] for row in rows] == list(REQUIRED_SYSTEMS)
    assert "# Pilot comparison: READY" in (tmp_path / "comparison.md").read_text()
    assert "Lower is better" in (tmp_path / "comparison.md").read_text()
    payload = json.loads((tmp_path / "comparison.json").read_text())
    comparison = payload["descriptive_comparisons"][
        "condition_off_vs_joint_conditioned_off"
    ]
    assert comparison == {
        "condition_off_audio_total_mean": pytest.approx(0.96),
        "joint_conditioned_off_audio_total_mean": pytest.approx(0.91),
        "audio_total_mean_delta": pytest.approx(0.05),
        "delta_definition": "condition_off - joint_conditioned_off",
        "interpretation": "negative favors separately trained condition_off",
        "decision_gate": False,
    }
    markdown = (tmp_path / "comparison.md").read_text()
    assert (
        "Separately trained condition_off vs joint_conditioned_off: audio_total "
        "mean 0.96 vs 0.91; signed delta (condition_off - "
        "joint_conditioned_off) 0.05. Negative favors separately trained "
        "condition_off. This is descriptive and is not a decision gate."
    ) in markdown
    before = {name: (tmp_path / name).read_bytes() for name in expected}
    build_comparison(list(reversed(_ready_systems())), tmp_path)
    assert before == {name: (tmp_path / name).read_bytes() for name in expected}


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (
            lambda systems: systems.__setitem__(
                1, _system("joint_conditioned_on", [1.0, 1.0, 1.0])
            ),
            "audio_total mean is not strictly lower",
        ),
        (
            lambda systems: systems.__setitem__(
                1, _system("joint_conditioned_on", [0.4, 0.9, 1.0])
            ),
            "paired audio_total median delta is not strictly negative",
        ),
        (
            lambda systems: systems.__setitem__(
                1, _system("joint_conditioned_on", [0.6, 0.7, 0.8], psnr=29.49)
            ),
            "PSNR drop exceeds 0.5 dB",
        ),
        (
            lambda systems: systems.__setitem__(
                1, _system("joint_conditioned_on", [0.6, 0.7, 0.8], ssim=0.939)
            ),
            "SSIM drop exceeds 0.01",
        ),
        (
            lambda systems: systems.__setitem__(
                1, _system("joint_conditioned_on", [0.6, 0.7, 0.8], grad=0.0)
            ),
            "max_audio_to_visual_grad_norm is not strictly positive",
        ),
    ],
)
def test_each_gate_fails_independently(tmp_path, mutator, reason):
    systems = _ready_systems()
    mutator(systems)
    result = build_comparison(systems, tmp_path)
    assert not result.decision.ready
    assert any(reason in item for item in result.decision.reasons)


def test_multiple_failure_reasons_are_complete_and_boundary_is_inclusive_for_visual(tmp_path):
    systems = _ready_systems()
    systems[1] = _system(
        "joint_conditioned_on", [1.0, 1.0, 1.0], psnr=29.5, ssim=0.94, grad=0.0
    )
    result = build_comparison(systems, tmp_path)
    assert len(result.decision.reasons) == 3
    assert not any("PSNR" in reason or "SSIM" in reason for reason in result.decision.reasons)
    markdown = (tmp_path / "comparison.md").read_text()
    assert all(reason in markdown for reason in result.decision.reasons)


def test_exact_system_names_and_same_checkpoint_pair_are_required(tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        build_comparison(_ready_systems()[:-1], tmp_path)
    duplicated = _ready_systems()
    duplicated[-1] = duplicated[0]
    with pytest.raises(ValueError, match="duplicate"):
        build_comparison(duplicated, tmp_path)
    systems = _ready_systems()
    systems[2] = replace(
        systems[2],
        provenance={**systems[2].provenance, "checkpoint_generation": 8},
    )
    with pytest.raises(ValueError, match="same checkpoint"):
        build_comparison(systems, tmp_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda systems: systems[1].evaluation.summary.pop("rgb_l1"), "metric fields"),
        (
            lambda systems: object.__setattr__(systems[1].evaluation, "count", 99),
            "sample count",
        ),
        (
            lambda systems: systems[1].worker_summary["training_history"].clear(),
            "training_history",
        ),
        (
            lambda systems: systems[1].worker_summary["checkpoint_io"].__setitem__(
                "save_count", True
            ),
            "save_count",
        ),
        (
            lambda systems: systems[1].worker_summary["validation_history"][0][
                "summary"
            ]["audio_total"].__setitem__("mean", float("inf")),
            "finite",
        ),
        (
            lambda systems: systems[1].worker_summary.update(extra=True),
            "worker summary fields",
        ),
        (
            lambda systems: systems[1].provenance.update(extra=True),
            "provenance fields",
        ),
    ],
)
def test_malformed_contracts_are_rejected(tmp_path, mutation, match):
    systems = _ready_systems()
    mutation(systems)
    with pytest.raises((TypeError, ValueError), match=match):
        build_comparison(systems, tmp_path)


def test_zero_minimum_steps_worker_summary_is_valid(tmp_path):
    systems = _ready_systems()
    for system in systems[1:]:
        system.worker_summary["stopper_state"]["minimum_steps"] = 0
    assert build_comparison(systems, tmp_path).decision.ready


def test_rows_must_match_each_system_provenance_scene_even_when_all_agree(tmp_path):
    systems = _ready_systems()
    for system in systems:
        for row in system.evaluation.rows:
            row["scene_id"] = "consistently_wrong"
    decision = decide_long_training(systems)
    assert not decision.ready
    assert decision.reasons == (
        "comparison data is missing, nonfinite, or inconsistent",
    )
    with pytest.raises(ValueError, match="row scene.*provenance"):
        build_comparison(systems, tmp_path)
    assert not (tmp_path / "comparison.json").exists()


def test_metric_deltas_follow_documented_direction(tmp_path):
    result = build_comparison(_ready_systems(), tmp_path)
    joint = result.systems["joint_conditioned_on"]
    assert joint["deltas_vs_baseline"]["audio_total"]["mean"] == pytest.approx(-0.3)
    assert joint["deltas_vs_baseline"]["rgb_psnr"]["mean"] == pytest.approx(0.0)
    assert joint["metric_directions"]["audio_total"] == "lower_is_better"
    assert joint["metric_directions"]["rgb_psnr"] == "higher_is_better"


def test_failed_atomic_publication_does_not_publish_final_summary(tmp_path, monkeypatch):
    original = report_module.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced publication failure")
        return original(source, destination)

    monkeypatch.setattr(report_module.os, "replace", fail_second)
    with pytest.raises(OSError, match="forced"):
        build_comparison(_ready_systems(), tmp_path)
    assert not (tmp_path / "comparison.json").exists()
    assert not (tmp_path / "comparison.md").exists()

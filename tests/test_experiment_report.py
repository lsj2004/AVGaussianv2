from __future__ import annotations

import csv
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import avgaussianv2.experiment.report as report_module
from avgaussianv2.experiment.contracts import EvaluationResult
from avgaussianv2.experiment.evaluation import METRIC_NAMES
from avgaussianv2.experiment.metrics import aggregate_metrics
from avgaussianv2.experiment.report import (
    EvaluationProvenance,
    REQUIRED_SYSTEMS,
    SystemReportInput,
    build_comparison,
    build_evaluation_run_id,
    decide_long_training,
    paired_audio_deltas,
    resolve_current_report,
)
from avgaussianv2.experiment.checkpoint import hash_index_manifest, sha256_file
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
        "camera": "cam10",
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


def _worker(
    variant: str,
    grad: float = 0.2,
    *,
    quick_psnr: float = 30.0,
    quick_ssim: float = 0.95,
):
    def validation_summary(audio: float, psnr: float, ssim: float):
        summary = {
            name: {"mean": audio, "std": 0.1, "median": audio}
            for name in METRIC_NAMES
        }
        summary["rgb_psnr"] = {"mean": psnr, "std": 0.1, "median": psnr}
        summary["rgb_ssim"] = {"mean": ssim, "std": 0.01, "median": ssim}
        return summary

    warmup = 0 if variant == "condition_off" else 1
    training_history = []
    if warmup:
        training_history.append(
            {
                "stage": "warmup",
                "step": 1,
                "sample_index": 11,
                "total": 1.2,
                "audio_to_visual_grad_norm": 0.0,
                "losses": {"audio": 1.2},
                "gradient_norms": {"visual": 0.0},
            }
        )
    for step in range(1, 4):
        training_history.append(
            {
                "stage": "joint",
                "step": step,
                "sample_index": 20 + step,
                "total": 1.0 - step * 0.1,
                "audio_to_visual_grad_norm": grad,
                "losses": {"audio": 1.0 - step * 0.1},
                "gradient_norms": {"visual": grad},
            }
        )
    return {
        "variant": variant,
        "completed_warmup_steps": warmup,
        "completed_joint_steps": 3,
        "best_step": 3,
        "stop_reason": "max_steps",
        "training_history": training_history,
        "validation_history": [
            {"step": 1, "summary": validation_summary(0.7, 30.0, 0.95)},
            {"step": 2, "summary": validation_summary(0.6, 30.0, 0.95)},
            {
                "step": 3,
                "summary": validation_summary(0.5, quick_psnr, quick_ssim),
            },
        ],
        "selector_state": {
            "visual_baseline": {
                "rgb_psnr": {"mean": 30.0},
                "rgb_ssim": {"mean": 0.95},
            },
            "psnr_tolerance_db": 0.5,
            "ssim_tolerance": 0.01,
            "best_step": 3,
            "best_audio_total": 0.5,
            "last_step": 3,
        },
        "stopper_state": {
            "minimum_steps": 1,
            "patience": 2,
            "relative_delta": 0.01,
            "best": 0.5,
            "stale": 0,
            "last_step": 3,
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


def _system(
    name: str,
    audio,
    provenance: EvaluationProvenance,
    *,
    psnr=30.0,
    ssim=0.95,
    grad=0.2,
    quick_psnr=30.0,
    quick_ssim=0.95,
):
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
        worker_summary=(
            None
            if name == "baseline_imported"
            else _worker(
                variants[name],
                grad,
                quick_psnr=quick_psnr,
                quick_ssim=quick_ssim,
            )
        ),
        provenance=provenance,
    )


_RUN_FINGERPRINT_INPUTS = {"fixture": "producer-shaped-v1"}
_RUN_FINGERPRINT = {
    "algorithm": "avgaussianv2-pilot-fixture-v1",
    "sha256": hash_index_manifest(_RUN_FINGERPRINT_INPUTS),
    "inputs": _RUN_FINGERPRINT_INPUTS,
}
_EVALUATION_INDICES_HASH = hash_index_manifest([0, 1, 2])


def _artifact(path: Path, *, generation: int, pilot: bool) -> str:
    if pilot:
        torch.save(
            {
                "checkpoint_kind": "best",
                "generation": generation,
                "run_fingerprint": _RUN_FINGERPRINT,
            },
            path,
        )
    else:
        path.write_bytes(b"imported-baseline-artifact-v1")
    return sha256_file(path)


def _provenance(
    path: Path,
    *,
    generation: int,
    condition_enabled: bool,
    pilot: bool,
) -> EvaluationProvenance:
    sha = _artifact(path, generation=generation, pilot=pilot)
    run_id = build_evaluation_run_id(
        checkpoint_path=path,
        checkpoint_sha256=sha,
        checkpoint_generation=generation,
        run_fingerprint=_RUN_FINGERPRINT,
        evaluation_indices_hash=_EVALUATION_INDICES_HASH,
    )
    return EvaluationProvenance(
        scene_id="scene1_opera",
        checkpoint_path=path,
        checkpoint_sha256=sha,
        checkpoint_generation=generation,
        run_fingerprint=_RUN_FINGERPRINT,
        evaluation_indices_hash=_EVALUATION_INDICES_HASH,
        condition_enabled=condition_enabled,
        evaluation_run_id=run_id,
    )


def _ready_systems(tmp_path: Path):
    baseline = _provenance(
        tmp_path / "baseline.bin",
        generation=0,
        condition_enabled=False,
        pilot=False,
    )
    joint = _provenance(
        tmp_path / "joint-best.pt",
        generation=7,
        condition_enabled=True,
        pilot=True,
    )
    joint_off = replace(joint, condition_enabled=False)
    frozen = _provenance(
        tmp_path / "frozen-best.pt",
        generation=3,
        condition_enabled=True,
        pilot=True,
    )
    condition_off = _provenance(
        tmp_path / "condition-off-best.pt",
        generation=5,
        condition_enabled=False,
        pilot=True,
    )
    return [
        _system("baseline_imported", [1.0, 1.0, 1.0], baseline),
        _system("joint_conditioned_on", [0.6, 0.7, 0.8], joint),
        _system("joint_conditioned_off", [0.9, 0.9, 0.9], joint_off),
        _system("frozen_visual_on", [0.75, 0.8, 0.85], frozen),
        _system("condition_off", [0.95, 0.95, 0.95], condition_off),
    ]


def _replace_system(
    system: SystemReportInput,
    audio,
    *,
    psnr=30.0,
    ssim=0.95,
    grad: float | None = None,
    quick_psnr: float | None = None,
    quick_ssim: float | None = None,
) -> SystemReportInput:
    worker = copy.deepcopy(system.worker_summary)
    if grad is not None:
        for row in worker["training_history"]:
            if row["stage"] == "joint":
                row["audio_to_visual_grad_norm"] = grad
                row["gradient_norms"]["visual"] = grad
    if quick_psnr is not None:
        worker["validation_history"][-1]["summary"]["rgb_psnr"].update(
            mean=quick_psnr, median=quick_psnr
        )
    if quick_ssim is not None:
        worker["validation_history"][-1]["summary"]["rgb_ssim"].update(
            mean=quick_ssim, median=quick_ssim
        )
    return replace(
        system,
        evaluation=_evaluation(
            system.name, tuple(audio), psnr=psnr, ssim=ssim
        ),
        worker_summary=worker,
    )


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
    systems = _ready_systems(tmp_path)
    result = build_comparison(systems, tmp_path / "report")
    assert result.decision.ready
    assert result.decision.reasons == ()
    assert set(result.systems) == set(REQUIRED_SYSTEMS)
    expected = {
        "comparison.json",
        "comparison.csv",
        "comparison.md",
        "paired_condition_deltas.jsonl",
    }
    report_dir = tmp_path / "report"
    assert {
        path.name for path in report_dir.iterdir() if not path.name.startswith(".")
    } == expected | {"current"}
    json.loads((report_dir / "comparison.json").read_text())
    rows = list(csv.DictReader((report_dir / "comparison.csv").read_text().splitlines()))
    assert [row["system"] for row in rows] == list(REQUIRED_SYSTEMS)
    assert "# Pilot comparison: READY" in (report_dir / "comparison.md").read_text()
    assert "Lower is better" in (report_dir / "comparison.md").read_text()
    payload = json.loads((report_dir / "comparison.json").read_text())
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
    markdown = (report_dir / "comparison.md").read_text()
    assert (
        "Separately trained condition_off vs joint_conditioned_off: audio_total "
        "mean 0.96 vs 0.91; signed delta (condition_off - "
        "joint_conditioned_off) 0.05. Negative favors separately trained "
        "condition_off. This is descriptive and is not a decision gate."
    ) in markdown
    before = {name: (report_dir / name).read_bytes() for name in expected}
    assert all(
        str(tmp_path).encode("utf-8") not in content
        for content in before.values()
    )
    first_generation = result.generation_path
    manifest_before = (
        first_generation / "generation_manifest.json"
    ).read_bytes()
    rerun = build_comparison(list(reversed(systems)), report_dir)
    assert before == {name: (report_dir / name).read_bytes() for name in expected}
    assert rerun.generation_path == first_generation
    assert (
        rerun.generation_path / "generation_manifest.json"
    ).read_bytes() == manifest_before
    assert resolve_current_report(report_dir) == first_generation


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (
            lambda systems: systems.__setitem__(
                1, _replace_system(systems[1], [1.0, 1.0, 1.0])
            ),
            "audio_total mean is not strictly lower",
        ),
        (
            lambda systems: systems.__setitem__(
                1, _replace_system(systems[1], [0.4, 0.9, 1.0])
            ),
            "paired audio_total median delta is not strictly negative",
        ),
        (
            lambda systems: systems.__setitem__(
                1, _replace_system(
                    systems[1], [0.6, 0.7, 0.8], grad=0.0
                )
            ),
            "max_audio_to_visual_grad_norm is not strictly positive",
        ),
    ],
)
def test_each_gate_fails_independently(tmp_path, mutator, reason):
    systems = _ready_systems(tmp_path)
    mutator(systems)
    result = build_comparison(systems, tmp_path)
    assert not result.decision.ready
    assert any(reason in item for item in result.decision.reasons)


def test_multiple_failure_reasons_are_complete_and_boundary_is_inclusive_for_visual(tmp_path):
    systems = _ready_systems(tmp_path)
    systems[1] = _replace_system(
        systems[1],
        [1.0, 1.0, 1.0],
        quick_psnr=29.5,
        quick_ssim=0.94,
        grad=0.0,
    )
    result = build_comparison(systems, tmp_path)
    assert len(result.decision.reasons) == 3
    assert not any("PSNR" in reason or "SSIM" in reason for reason in result.decision.reasons)
    markdown = (tmp_path / "comparison.md").read_text()
    assert all(reason in markdown for reason in result.decision.reasons)


def test_quick_best_feasible_can_gate_ready_when_full_split_is_infeasible(tmp_path):
    systems = _ready_systems(tmp_path)
    systems[1] = _replace_system(
        systems[1], [0.6, 0.7, 0.8], psnr=20.0, ssim=0.8
    )
    result = build_comparison(systems, tmp_path / "report")
    joint = result.systems["joint_conditioned_on"]
    assert joint["acceptance_visual_feasible"] is True
    assert joint["full_split_visual_feasible"] is False
    assert result.decision.ready


def test_quick_infeasible_worker_is_corruption_even_when_full_split_is_feasible(
    tmp_path,
):
    systems = _ready_systems(tmp_path)
    systems[1] = _replace_system(
        systems[1], [0.6, 0.7, 0.8], quick_psnr=29.49
    )
    assert not decide_long_training(systems).ready
    with pytest.raises(ValueError, match="selector_state does not replay"):
        build_comparison(systems, tmp_path / "report")


def test_exact_system_names_and_same_checkpoint_pair_are_required(tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        build_comparison(_ready_systems(tmp_path)[:-1], tmp_path / "report")
    duplicated = _ready_systems(tmp_path)
    duplicated[-1] = duplicated[0]
    with pytest.raises(ValueError, match="duplicate"):
        build_comparison(duplicated, tmp_path)
    systems = _ready_systems(tmp_path)
    changed = replace(systems[2].provenance, checkpoint_generation=8)
    changed = replace(
        changed,
        evaluation_run_id=build_evaluation_run_id(
            checkpoint_path=changed.checkpoint_path,
            checkpoint_sha256=changed.checkpoint_sha256,
            checkpoint_generation=changed.checkpoint_generation,
            run_fingerprint=changed.run_fingerprint,
            evaluation_indices_hash=changed.evaluation_indices_hash,
        ),
    )
    systems[2] = replace(
        systems[2],
        provenance=changed,
    )
    with pytest.raises(ValueError, match="checkpoint generation mismatch"):
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
            lambda systems: systems.__setitem__(
                1, replace(systems[1], provenance={"bad": True})
            ),
            "EvaluationProvenance",
        ),
    ],
)
def test_malformed_contracts_are_rejected(tmp_path, mutation, match):
    systems = _ready_systems(tmp_path)
    mutation(systems)
    with pytest.raises((TypeError, ValueError), match=match):
        build_comparison(systems, tmp_path)


def test_zero_minimum_steps_worker_summary_is_valid(tmp_path):
    systems = _ready_systems(tmp_path)
    for system in systems[1:]:
        system.worker_summary["stopper_state"]["minimum_steps"] = 0
    assert build_comparison(systems, tmp_path).decision.ready


def test_rows_must_match_each_system_provenance_scene_even_when_all_agree(tmp_path):
    systems = _ready_systems(tmp_path)
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


def test_artifact_path_is_verified_instead_of_trusting_stamped_identity(tmp_path):
    systems = _ready_systems(tmp_path)
    impostor = tmp_path / "impostor.pt"
    impostor.write_bytes(b"different-real-artifact")
    systems[2] = replace(
        systems[2],
        provenance=replace(
            systems[2].provenance,
            checkpoint_path=impostor,
        ),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_comparison(systems, tmp_path / "report")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda worker: worker["training_history"].__setitem__(
                0, worker["training_history"][1]
            ),
            "contiguous warmup then joint",
        ),
        (
            lambda worker: worker["selector_state"].__setitem__(
                "best_audio_total", 0.4
            ),
            "selector_state does not replay",
        ),
        (
            lambda worker: worker["stopper_state"].__setitem__("stale", 1),
            "stopper_state does not replay",
        ),
        (
            lambda worker: worker["validation_history"].reverse(),
            "validation steps",
        ),
    ],
)
def test_worker_history_and_selection_state_are_replayed(
    tmp_path, mutation, match
):
    systems = _ready_systems(tmp_path)
    mutation(systems[1].worker_summary)
    with pytest.raises(ValueError, match=match):
        build_comparison(systems, tmp_path / "report")


def test_producer_shaped_worker_and_evaluation_survive_json_roundtrip(tmp_path):
    systems = _ready_systems(tmp_path)
    roundtripped = []
    for system in systems:
        payload = json.loads(
            json.dumps(
                {
                    "worker": system.worker_summary,
                    "rows": list(system.evaluation.rows),
                    "summary": system.evaluation.summary,
                },
                allow_nan=False,
            )
        )
        evaluation = EvaluationResult(
            system.name,
            len(payload["rows"]),
            tuple(payload["rows"]),
            payload["summary"],
        )
        roundtripped.append(
            replace(
                system,
                evaluation=evaluation,
                worker_summary=payload["worker"],
            )
        )
    result = build_comparison(roundtripped, tmp_path / "report")
    assert result.decision.ready
    assert all(
        row["camera"] == "cam10"
        for row in result.rows["joint_conditioned_on"]
    )


def test_metric_deltas_follow_documented_direction(tmp_path):
    result = build_comparison(_ready_systems(tmp_path), tmp_path / "report")
    joint = result.systems["joint_conditioned_on"]
    assert joint["deltas_vs_baseline"]["audio_total"]["mean"] == pytest.approx(-0.3)
    assert joint["deltas_vs_baseline"]["rgb_psnr"]["mean"] == pytest.approx(0.0)
    assert joint["metric_directions"]["audio_total"] == "lower_is_better"
    assert joint["metric_directions"]["rgb_psnr"] == "higher_is_better"


def test_failed_atomic_publication_does_not_publish_final_summary(tmp_path, monkeypatch):
    original = report_module._write_generation_file
    calls = 0

    def fail_second(directory_fd, name, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced publication failure")
        return original(directory_fd, name, content)

    monkeypatch.setattr(report_module, "_write_generation_file", fail_second)
    with pytest.raises(OSError, match="forced"):
        build_comparison(_ready_systems(tmp_path), tmp_path / "report")
    assert not (tmp_path / "report" / "current").exists()
    with pytest.raises(ValueError, match="no authoritative"):
        resolve_current_report(tmp_path / "report")


def test_failed_overwrite_keeps_old_complete_generation_authoritative(
    tmp_path, monkeypatch
):
    report_dir = tmp_path / "report"
    systems = _ready_systems(tmp_path)
    ready = build_comparison(systems, report_dir)
    systems[1] = _replace_system(systems[1], [1.0, 1.0, 1.0])
    original = report_module.os.rename

    def fail_current(source, destination, **kwargs):
        if destination == "current":
            raise OSError("forced pointer failure")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(report_module.os, "rename", fail_current)
    with pytest.raises(OSError, match="pointer"):
        build_comparison(systems, report_dir)
    assert ready.generation_path.is_dir()
    assert resolve_current_report(report_dir) == ready.generation_path
    assert "# Pilot comparison: READY" in (
        report_dir / "comparison.md"
    ).read_text()


def test_distinct_artifacts_are_hashed_and_inspected_once(tmp_path, monkeypatch):
    hash_calls = []
    identity_calls = []
    original_hash = report_module._secure_sha256_file
    original_identity = report_module.resolve_pilot_checkpoint_identity

    def counting_hash(path):
        hash_calls.append(Path(path).resolve())
        return original_hash(path)

    def counting_identity(path):
        identity_calls.append(Path(path).resolve())
        return original_identity(path)

    monkeypatch.setattr(report_module, "_secure_sha256_file", counting_hash)
    build_comparison(
        _ready_systems(tmp_path),
        tmp_path / "report",
        checkpoint_identity_resolver=counting_identity,
    )
    assert len(hash_calls) == len(set(hash_calls)) == 4
    assert len(identity_calls) == len(set(identity_calls)) == 3


def test_crash_left_partial_generation_is_cleaned_before_publish(tmp_path):
    report_dir = tmp_path / "report"
    stale = report_dir / ".report-generations" / ".crash.tmp"
    stale.mkdir(parents=True)
    (stale / "comparison.json").write_text("partial")
    result = build_comparison(_ready_systems(tmp_path), report_dir)
    assert resolve_current_report(report_dir) == result.generation_path
    assert not stale.exists()


def test_generation_hash_verification_detects_post_publish_corruption(tmp_path):
    report_dir = tmp_path / "report"
    result = build_comparison(_ready_systems(tmp_path), report_dir)
    result.generation_path.chmod(0o700)
    corrupted = result.generation_path / "comparison.csv"
    corrupted.chmod(0o600)
    corrupted.write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_current_report(report_dir)


def test_output_and_parent_symlinks_are_rejected(tmp_path):
    systems = _ready_systems(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        build_comparison(systems, output_link)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        build_comparison(systems, parent_link / "report")


def test_unsafe_existing_current_pointer_is_rejected(tmp_path):
    report_dir = tmp_path / "report"
    build_comparison(_ready_systems(tmp_path), report_dir)
    (report_dir / "current").unlink()
    (report_dir / "current").symlink_to("/tmp")
    with pytest.raises(ValueError, match="current pointer is unsafe"):
        build_comparison(_ready_systems(tmp_path), report_dir)

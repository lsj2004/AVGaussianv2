from __future__ import annotations

import csv
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import avgaussianv2.experiment.report as report_module
from avgaussianv2.experiment.contracts import EvaluationResult
from avgaussianv2.experiment.evaluation import METRIC_NAMES
from avgaussianv2.experiment.metrics import aggregate_metrics
from avgaussianv2.experiment.report import (
    EvaluationArtifactProvenance,
    EvaluationProvenance,
    REQUIRED_SYSTEMS,
    SystemReportInput,
    WorkerArtifactProvenance,
    build_comparison,
    build_evaluation_manifest_sha256,
    build_evaluation_run_id,
    decide_long_training,
    paired_audio_deltas,
    resolve_current_report,
)


def test_worker_artifact_provenance_contract_is_exported():
    assert hasattr(report_module, "WorkerArtifactProvenance")
from avgaussianv2.config import TrainConfig
from avgaussianv2.experiment.checkpoint import (
    PilotCompatibility,
    PilotResumeError,
    build_pilot_payload,
    build_run_fingerprint,
    hash_index_manifest,
    sha256_file,
)
from avgaussianv2.experiment.contracts import PilotConfig, VariantIndices
from avgaussianv2.experiment.selection import BestSelector, EarlyStopper
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
            "minimum_steps": 0,
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
    worker_summary=None,
    worker_provenance=None,
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
            worker_summary
            if worker_summary is not None
            else (
                None
                if name == "baseline_imported"
                else _worker(
                variants[name],
                grad,
                quick_psnr=quick_psnr,
                quick_ssim=quick_ssim,
            )
            )
        ),
        provenance=provenance,
        worker_provenance=worker_provenance,
    )


_RUN_FINGERPRINT_INPUTS = {"fixture": "producer-shaped-v1"}
_RUN_FINGERPRINT = {
    "algorithm": "avgaussianv2-pilot-fixture-v1",
    "sha256": hash_index_manifest(_RUN_FINGERPRINT_INPUTS),
    "inputs": _RUN_FINGERPRINT_INPUTS,
}
_EVALUATION_INDICES_HASH = hash_index_manifest([0, 1, 2])


def _artifact(
    path: Path, *, generation: int, pilot: bool, variant: str
):
    if pilot:
        model = nn.Linear(2, 1)
        pilot_config = PilotConfig(
            warmup_steps=1,
            joint_steps=3,
            validation_interval=1,
            minimum_joint_steps=0,
            patience=2,
            minimum_relative_improvement=0.01,
            psnr_tolerance_db=0.5,
            ssim_tolerance=0.01,
        )
        baseline = {
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        }
        run_fingerprint = build_run_fingerprint(
            pilot_config=pilot_config,
            train_config=TrainConfig(),
            visual_baseline=baseline,
            model=model,
            warmup_optimizer_factory=hash_index_manifest,
            joint_optimizer_factory=hash_index_manifest,
            warmup_step_fn=hash_index_manifest,
            joint_step_fn=hash_index_manifest,
            audio_loss_fn=hash_index_manifest,
        )
        compatibility = PilotCompatibility(
            scene_id="scene1_opera",
            variant=variant,
            seed=7,
            index_hash="1" * 64,
            visual_checkpoint_sha256="2" * 64,
            audio_checkpoint_sha256="3" * 64,
            camera_mapping_sha256="4" * 64,
            n_fft=512,
            hop_length=128,
            win_length=512,
            sample_rate=48_000,
        )
        worker = _worker(variant)
        indices = VariantIndices(
            () if variant == "condition_off" else (11,),
            (21, 22, 23),
        )
        selector = BestSelector(baseline, 0.5, 0.01)
        stopper = EarlyStopper(0, 2, 0.01)
        for validation in worker["validation_history"]:
            selector.consider(validation["step"], validation["summary"])
            stopper.update(
                validation["step"],
                validation["summary"]["audio_total"]["mean"],
            )
        payload = build_pilot_payload(
            model=model,
            compatibility=compatibility,
            run_fingerprint=run_fingerprint,
            stage="joint",
            next_warmup_position=len(indices.warmup),
            next_joint_position=3,
            optimizer=None,
            optimizer_stage=None,
            selector=selector,
            stopper=stopper,
            training_history=worker["training_history"],
            validation_history=worker["validation_history"],
            maximum_positive_audio_visual_gradient=0.2,
            checkpoint_kind="best",
            generation=generation,
            best_generation=generation,
            validation_summary=worker["validation_history"][-1]["summary"],
            best_evaluation_summary=worker["validation_history"][-1]["summary"],
        )
        torch.save(payload, path)
    else:
        path.write_bytes(b"imported-baseline-artifact-v1")
        compatibility = None
        indices = None
        pilot_config = None
        run_fingerprint = _RUN_FINGERPRINT
    return (
        sha256_file(path),
        compatibility,
        indices,
        pilot_config,
        run_fingerprint,
    )


def _provenance(
    path: Path,
    *,
    generation: int,
    condition_enabled: bool,
    pilot: bool,
    variant: str,
) -> EvaluationProvenance:
    sha, compatibility, indices, pilot_config, run_fingerprint = _artifact(
        path, generation=generation, pilot=pilot, variant=variant
    )
    run_id = build_evaluation_run_id(
        checkpoint_path=path,
        checkpoint_sha256=sha,
        checkpoint_generation=generation,
        run_fingerprint=run_fingerprint,
        evaluation_indices_hash=_EVALUATION_INDICES_HASH,
    )
    return EvaluationProvenance(
        scene_id="scene1_opera",
        checkpoint_path=path,
        checkpoint_sha256=sha,
        checkpoint_generation=generation,
        run_fingerprint=run_fingerprint,
        evaluation_indices_hash=_EVALUATION_INDICES_HASH,
        condition_enabled=condition_enabled,
        evaluation_run_id=run_id,
        compatibility=compatibility,
        variant_indices=indices,
        pilot_config=pilot_config,
    )


def _worker_artifacts(
    root: Path,
    checkpoint: EvaluationProvenance,
    worker: dict[str, object],
    *,
    label: str,
) -> WorkerArtifactProvenance:
    artifact_dir = root / "worker-artifacts" / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    worker_path = artifact_dir / "worker_summary.json"
    worker_path.write_text(
        json.dumps(worker, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    selector = BestSelector(
        worker["selector_state"]["visual_baseline"],
        worker["selector_state"]["psnr_tolerance_db"],
        worker["selector_state"]["ssim_tolerance"],
    )
    stopper = EarlyStopper(
        worker["stopper_state"]["minimum_steps"],
        worker["stopper_state"]["patience"],
        worker["stopper_state"]["relative_delta"],
    )
    for validation in worker["validation_history"]:
        selector.consider(validation["step"], validation["summary"])
        stopper.update(
            validation["step"],
            validation["summary"]["audio_total"]["mean"],
        )
    latest_path = artifact_dir / "latest.pt"
    latest_generation = checkpoint.checkpoint_generation + 1
    model = nn.Linear(2, 1)
    torch.save(
        build_pilot_payload(
            model=model,
            compatibility=checkpoint.compatibility,
            run_fingerprint=checkpoint.run_fingerprint,
            stage="complete",
            next_warmup_position=worker["completed_warmup_steps"],
            next_joint_position=worker["completed_joint_steps"],
            optimizer=None,
            optimizer_stage=None,
            selector=selector,
            stopper=stopper,
            training_history=worker["training_history"],
            validation_history=worker["validation_history"],
            maximum_positive_audio_visual_gradient=max(
                row["audio_to_visual_grad_norm"]
                for row in worker["training_history"]
                if row["stage"] == "joint"
            ),
            checkpoint_kind="latest",
            generation=latest_generation,
            best_generation=checkpoint.checkpoint_generation,
            stop_requested=worker["stop_reason"] == "early_stop",
            stop_reason=worker["stop_reason"],
            validation_summary=worker["validation_history"][-1]["summary"],
            best_evaluation_summary=worker["validation_history"][-1]["summary"],
        ),
        latest_path,
    )
    return WorkerArtifactProvenance(
        worker_summary_path=worker_path,
        worker_summary_sha256=sha256_file(worker_path),
        latest_checkpoint_path=latest_path,
        latest_checkpoint_sha256=sha256_file(latest_path),
        latest_checkpoint_generation=latest_generation,
        run_fingerprint=checkpoint.run_fingerprint,
    )


def _ready_systems(tmp_path: Path):
    baseline = _provenance(
        tmp_path / "baseline.bin",
        generation=0,
        condition_enabled=False,
        pilot=False,
        variant="joint_conditioned",
    )
    joint = _provenance(
        tmp_path / "joint-best.pt",
        generation=7,
        condition_enabled=True,
        pilot=True,
        variant="joint_conditioned",
    )
    joint_off = replace(joint, condition_enabled=False)
    frozen = _provenance(
        tmp_path / "frozen-best.pt",
        generation=3,
        condition_enabled=True,
        pilot=True,
        variant="frozen_visual",
    )
    condition_off = _provenance(
        tmp_path / "condition-off-best.pt",
        generation=5,
        condition_enabled=False,
        pilot=True,
        variant="condition_off",
    )
    joint_worker = _worker("joint_conditioned")
    frozen_worker = _worker("frozen_visual")
    condition_off_worker = _worker("condition_off")
    joint_artifacts = _worker_artifacts(
        tmp_path, joint, joint_worker, label="joint"
    )
    frozen_artifacts = _worker_artifacts(
        tmp_path, frozen, frozen_worker, label="frozen"
    )
    condition_off_artifacts = _worker_artifacts(
        tmp_path, condition_off, condition_off_worker, label="condition-off"
    )
    systems = [
        _system(
            "baseline_imported",
            [1.0, 1.0, 1.0],
            baseline,
            worker_provenance=None,
        ),
        _system(
            "joint_conditioned_on",
            [0.6, 0.7, 0.8],
            joint,
            worker_summary=joint_worker,
            worker_provenance=joint_artifacts,
        ),
        _system(
            "joint_conditioned_off",
            [0.9, 0.9, 0.9],
            joint_off,
            worker_summary=copy.deepcopy(joint_worker),
            worker_provenance=joint_artifacts,
        ),
        _system(
            "frozen_visual_on",
            [0.75, 0.8, 0.85],
            frozen,
            worker_summary=frozen_worker,
            worker_provenance=frozen_artifacts,
        ),
        _system(
            "condition_off",
            [0.95, 0.95, 0.95],
            condition_off,
            worker_summary=condition_off_worker,
            worker_provenance=condition_off_artifacts,
        ),
    ]
    return [_bind_evaluation_artifact(system, tmp_path) for system in systems]


def _bind_evaluation_artifact(
    system: SystemReportInput, root: Path
) -> SystemReportInput:
    checkpoint = (
        system.provenance.checkpoint
        if isinstance(system.provenance, EvaluationArtifactProvenance)
        else system.provenance
    )
    artifact_dir = root / "evaluation-artifacts" / system.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows_path = artifact_dir / "metrics_per_sample.jsonl"
    summary_path = artifact_dir / "metrics_summary.json"
    rows_path.write_text(
        "".join(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
            for row in system.evaluation.rows
        )
    )
    summary_path.write_text(
        json.dumps(
            system.evaluation.summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    rows_sha = sha256_file(rows_path)
    summary_sha = sha256_file(summary_path)
    indices = tuple(range(system.evaluation.count))
    indices_hash = hash_index_manifest(list(indices))
    assert indices_hash == checkpoint.evaluation_indices_hash
    manifest_sha = build_evaluation_manifest_sha256(
        metrics_per_sample_sha256=rows_sha,
        metrics_summary_sha256=summary_sha,
        system_name=system.name,
        condition_enabled=checkpoint.condition_enabled,
        count=system.evaluation.count,
        evaluation_indices=indices,
        evaluation_indices_hash=indices_hash,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        checkpoint_generation=checkpoint.checkpoint_generation,
        evaluation_run_id=checkpoint.evaluation_run_id,
    )
    provenance = EvaluationArtifactProvenance(
        metrics_per_sample_path=rows_path,
        metrics_per_sample_sha256=rows_sha,
        metrics_summary_path=summary_path,
        metrics_summary_sha256=summary_sha,
        manifest_sha256=manifest_sha,
        system_name=system.name,
        condition_enabled=checkpoint.condition_enabled,
        count=system.evaluation.count,
        evaluation_indices=indices,
        evaluation_indices_hash=indices_hash,
        checkpoint=checkpoint,
        evaluation_run_id=checkpoint.evaluation_run_id,
    )
    return replace(system, provenance=provenance)


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
    updated = replace(
        system,
        evaluation=_evaluation(
            system.name, tuple(audio), psnr=psnr, ssim=ssim
        ),
        worker_summary=worker,
    )
    return _bind_evaluation_artifact(
        updated,
        system.provenance.metrics_per_sample_path.parents[2],
    )


def _rebind_worker_run(
    systems: list[SystemReportInput],
    index: int,
) -> None:
    system = systems[index]
    worker = copy.deepcopy(system.worker_summary)
    checkpoint = system.provenance.checkpoint
    root = system.worker_provenance.worker_summary_path.parents[2]
    label = system.worker_provenance.worker_summary_path.parent.name
    provenance = _worker_artifacts(
        root,
        checkpoint,
        worker,
        label=label,
    )
    if system.name.startswith("joint_conditioned_"):
        for joint_index in (1, 2):
            systems[joint_index] = replace(
                systems[joint_index],
                worker_summary=copy.deepcopy(worker),
                worker_provenance=provenance,
            )
    else:
        systems[index] = replace(
            system,
            worker_summary=worker,
            worker_provenance=provenance,
        )


def _set_verified_joint_gradient(
    systems: list[SystemReportInput], gradient: float
) -> None:
    systems[1] = _replace_system(
        systems[1], [0.6, 0.7, 0.8], grad=gradient
    )
    _rebind_worker_run(systems, 1)


def _rewrite_joint_worker_artifact(
    systems: list[SystemReportInput], mutate
) -> None:
    worker = copy.deepcopy(systems[1].worker_summary)
    mutate(worker)
    provenance = systems[1].worker_provenance
    provenance.worker_summary_path.write_text(
        json.dumps(worker, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    provenance = replace(
        provenance,
        worker_summary_sha256=sha256_file(provenance.worker_summary_path),
    )
    for index in (1, 2):
        systems[index] = replace(
            systems[index],
            worker_summary=copy.deepcopy(worker),
            worker_provenance=provenance,
        )


def _rewrite_joint_latest_artifact(
    systems: list[SystemReportInput], mutate
) -> None:
    provenance = systems[1].worker_provenance
    payload = torch.load(provenance.latest_checkpoint_path, weights_only=True)
    mutate(payload)
    torch.save(payload, provenance.latest_checkpoint_path)
    provenance = replace(
        provenance,
        latest_checkpoint_sha256=sha256_file(provenance.latest_checkpoint_path),
    )
    for index in (1, 2):
        systems[index] = replace(
            systems[index],
            worker_provenance=provenance,
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
    "field,value",
    [
        ("warmup_steps", 2),
        ("joint_steps", 4),
        ("validation_interval", 2),
        ("minimum_joint_steps", 1),
        ("patience", 3),
        ("minimum_relative_improvement", 0.02),
        ("quick_validation_samples", 33),
        ("psnr_tolerance_db", 0.4),
        ("ssim_tolerance", 0.02),
    ],
)
def test_report_rejects_every_pilot_behavior_mismatch(tmp_path, field, value):
    systems = _ready_systems(tmp_path)
    checkpoint = systems[1].provenance.checkpoint
    mismatched = replace(
        checkpoint,
        pilot_config=replace(checkpoint.pilot_config, **{field: value}),
    )
    systems[1] = replace(
        systems[1],
        provenance=replace(systems[1].provenance, checkpoint=mismatched),
    )

    assert not decide_long_training(systems).ready
    with pytest.raises((TypeError, ValueError), match="pilot|tolerance"):
        build_comparison(systems, tmp_path / "report")
    assert not (tmp_path / "report" / "current").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("warmup_steps", True),
        lambda value: value.__setitem__("warmup_steps", 1.0),
        lambda value: value.__setitem__("minimum_relative_improvement", 1),
        lambda value: value.__setitem__("extra", 1),
        lambda value: value.pop("warmup_steps"),
    ],
)
def test_pilot_config_binding_rejects_type_coercion_and_key_drift(mutate):
    canonical = report_module._canonical_pilot_config(
        PilotConfig(
            warmup_steps=1,
            joint_steps=3,
            validation_interval=1,
            minimum_joint_steps=0,
            patience=2,
            minimum_relative_improvement=0.01,
            psnr_tolerance_db=0.5,
            ssim_tolerance=0.01,
        ),
        "pilot",
    )
    fingerprint_config = copy.deepcopy(canonical)
    mutate(fingerprint_config)
    fingerprint = {"inputs": {"pilot_config": fingerprint_config}}

    with pytest.raises((TypeError, ValueError), match="pilot_config"):
        report_module._require_matching_pilot_config(
            canonical, fingerprint, "fingerprint"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_steps", True),
        ("warmup_steps", 1.0),
        ("minimum_relative_improvement", 1),
    ],
)
def test_provenance_pilot_config_rejects_bool_and_numeric_coercion(field, value):
    with pytest.raises(TypeError, match="exact type"):
        report_module._canonical_pilot_config(
            replace(PilotConfig(), **{field: value}),
            "pilot",
        )


def test_warmup_config_drift_is_not_ready_then_canonical_run_is_green(tmp_path):
    systems = _ready_systems(tmp_path)
    checkpoint = systems[1].provenance.checkpoint
    drifted = replace(
        checkpoint,
        pilot_config=replace(
            checkpoint.pilot_config,
            warmup_steps=checkpoint.pilot_config.warmup_steps + 999,
        ),
    )
    systems[1] = replace(
        systems[1],
        provenance=replace(systems[1].provenance, checkpoint=drifted),
    )
    assert not decide_long_training(systems).ready

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean = _ready_systems(clean_root)
    result = build_comparison(clean, tmp_path / "green-report")
    assert result.decision.ready
    assert "# Pilot comparison: READY" in (
        tmp_path / "green-report" / "comparison.md"
    ).read_text()


def test_durability_warning_sidecar_must_have_one_link(tmp_path):
    report_dir = tmp_path / "report"
    published = build_comparison(_ready_systems(tmp_path), report_dir)
    warnings_dir = report_dir / ".report-warnings"
    warnings_dir.mkdir()
    sidecar = warnings_dir / f"{published.content_digest}.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "avgaussianv2.report-durability-warnings",
                "version": 1,
                "content_digest": published.content_digest,
                "warnings": ["test warning"],
            }
        )
    )
    (warnings_dir / "second-link.json").hardlink_to(sidecar)

    with pytest.raises(ValueError, match="sidecar is unsafe"):
        resolve_current_report(report_dir, include_warnings=True)


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
                lambda systems: _set_verified_joint_gradient(systems, 0.0),
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
        grad=0.0,
    )
    _rebind_worker_run(systems, 1)
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
    with pytest.raises(ValueError, match="in-memory worker summary disagrees"):
        build_comparison(systems, tmp_path / "report")


def test_widened_worker_visual_tolerance_can_never_create_false_ready(tmp_path):
    systems = _ready_systems(tmp_path)
    worker = systems[1].worker_summary
    worker["selector_state"]["psnr_tolerance_db"] = 1.0
    worker["validation_history"][-1]["summary"]["rgb_psnr"].update(
        mean=29.2, median=29.2
    )
    assert not decide_long_training(systems).ready
    with pytest.raises(ValueError, match="in-memory worker summary disagrees"):
        build_comparison(systems, tmp_path / "report")


def test_negative_loss_and_gradient_norm_evidence_is_rejected(tmp_path):
    systems = _ready_systems(tmp_path)
    worker = systems[1].worker_summary
    worker["training_history"][0]["losses"]["audio"] = -0.1
    _rebind_worker_run(systems, 1)
    with pytest.raises(ValueError, match="must be nonnegative"):
        build_comparison(systems, tmp_path / "negative-loss")

    gradient_root = tmp_path / "gradient"
    gradient_root.mkdir()
    systems = _ready_systems(gradient_root)
    worker = systems[1].worker_summary
    worker["training_history"][0]["gradient_norms"]["visual"] = -0.1
    _rebind_worker_run(systems, 1)
    with pytest.raises(ValueError, match="must be nonnegative"):
        build_comparison(systems, tmp_path / "negative-gradient")


def test_checkpoint_io_success_failure_attempt_counters_must_balance(tmp_path):
    systems = _ready_systems(tmp_path)
    worker = systems[1].worker_summary
    worker["checkpoint_io"]["save_attempt_count"] += 1
    _rebind_worker_run(systems, 1)
    with pytest.raises(ValueError, match="save counters are incoherent"):
        build_comparison(systems, tmp_path / "report")


def test_worker_selector_baseline_is_bound_to_canonical_checkpoint(tmp_path):
    systems = _ready_systems(tmp_path)
    worker = systems[1].worker_summary
    worker["selector_state"]["visual_baseline"]["rgb_ssim"]["mean"] = 0.94
    _rebind_worker_run(systems, 1)
    with pytest.raises(
        (ValueError, PilotResumeError), match="visual_baseline.*disagrees"
    ):
        build_comparison(systems, tmp_path / "report")


def test_in_memory_worker_summary_cannot_override_hashed_worker_artifact(tmp_path):
    systems = _ready_systems(tmp_path)
    systems[1].worker_summary["worker"]["device"] = "forged-device"
    with pytest.raises(ValueError, match="in-memory worker summary disagrees"):
        build_comparison(systems, tmp_path / "report")


def test_worker_summary_file_tampering_with_stale_hash_is_rejected(tmp_path):
    systems = _ready_systems(tmp_path)
    path = systems[1].worker_provenance.worker_summary_path
    path.write_text(path.read_text().replace("cuda:0", "forged"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_comparison(systems, tmp_path / "report")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda worker: worker["training_history"][-1].__setitem__(
                "audio_to_visual_grad_norm", 0.1
            ),
            "worker/latest training_history mismatch",
        ),
        (
            lambda worker: worker["training_history"][-1].__setitem__(
                "total", 0.1
            ),
            "worker/latest training_history mismatch",
        ),
        (
            lambda worker: worker.__setitem__("stop_reason", "early_stop"),
            "stop_reason",
        ),
        (
            lambda worker: worker.__setitem__("completed_joint_steps", 2),
            "completed",
        ),
    ],
)
def test_resigned_worker_gradient_history_stop_and_counts_cannot_override_latest(
    tmp_path, mutate, match
):
    systems = _ready_systems(tmp_path)
    _rewrite_joint_worker_artifact(systems, mutate)
    with pytest.raises(ValueError, match=match):
        build_comparison(systems, tmp_path / "report")


def test_zero_gradient_latest_cannot_be_overridden_by_positive_worker(tmp_path):
    systems = _ready_systems(tmp_path)

    def zero_latest(payload):
        for row in payload["training_history"]:
            if row["stage"] == "joint":
                row["audio_to_visual_grad_norm"] = 0.0
                row["gradient_norms"]["visual"] = 0.0
        payload["maximum_positive_audio_visual_gradient"] = 0.0

    _rewrite_joint_latest_artifact(systems, zero_latest)
    with pytest.raises(ValueError, match="worker/latest training_history mismatch"):
        build_comparison(systems, tmp_path / "report")


def test_latest_gradient_summary_must_equal_verified_joint_history(tmp_path):
    systems = _ready_systems(tmp_path)
    _rewrite_joint_latest_artifact(
        systems,
        lambda payload: payload.__setitem__(
            "maximum_positive_audio_visual_gradient", 0.0
        ),
    )
    with pytest.raises(ValueError, match="gradient summary mismatch"):
        build_comparison(systems, tmp_path / "report")


def test_joint_conditions_must_share_exact_worker_and_latest_identity(tmp_path):
    systems = _ready_systems(tmp_path)
    separate = _worker_artifacts(
        tmp_path,
        systems[2].provenance.checkpoint,
        copy.deepcopy(systems[2].worker_summary),
        label="joint-separate",
    )
    systems[2] = replace(systems[2], worker_provenance=separate)
    with pytest.raises(ValueError, match="share identical worker/latest"):
        build_comparison(systems, tmp_path / "report")


def test_training_row_total_must_be_nonnegative(tmp_path):
    systems = _ready_systems(tmp_path)
    _rewrite_joint_worker_artifact(
        systems,
        lambda worker: worker["training_history"][-1].__setitem__(
            "total", -0.1
        ),
    )
    with pytest.raises(ValueError, match="total must be nonnegative"):
        build_comparison(systems, tmp_path / "report")


def test_exact_system_names_and_same_checkpoint_pair_are_required(tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        build_comparison(_ready_systems(tmp_path)[:-1], tmp_path / "report")
    duplicated = _ready_systems(tmp_path)
    duplicated[-1] = duplicated[0]
    with pytest.raises(ValueError, match="duplicate"):
        build_comparison(duplicated, tmp_path)
    systems = _ready_systems(tmp_path)
    changed = replace(
        systems[2].provenance,
        checkpoint=replace(
            systems[2].provenance.checkpoint,
            checkpoint_generation=8,
        ),
    )
    systems[2] = replace(
        systems[2],
        provenance=changed,
    )
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        build_comparison(systems, tmp_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda systems: systems[1].evaluation.summary.pop("rgb_l1"),
            "in-memory evaluation disagrees",
        ),
        (
            lambda systems: object.__setattr__(systems[1].evaluation, "count", 99),
            "in-memory evaluation disagrees",
        ),
        (
            lambda systems: systems[1].worker_summary["training_history"].clear(),
            "in-memory worker summary disagrees",
        ),
        (
            lambda systems: systems[1].worker_summary["checkpoint_io"].__setitem__(
                "save_count", True
            ),
            "in-memory worker summary disagrees",
        ),
        (
            lambda systems: systems[1].worker_summary["validation_history"][0][
                "summary"
            ]["audio_total"].__setitem__("mean", float("inf")),
            "in-memory worker summary disagrees",
        ),
        (
            lambda systems: systems[1].worker_summary.update(extra=True),
            "in-memory worker summary disagrees",
        ),
        (
            lambda systems: systems.__setitem__(
                1, replace(systems[1], provenance={"bad": True})
            ),
            "EvaluationArtifactProvenance",
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
    with pytest.raises(ValueError, match="in-memory evaluation disagrees"):
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
            checkpoint=replace(
                systems[2].provenance.checkpoint,
                checkpoint_path=impostor,
            ),
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
            "in-memory worker summary disagrees",
        ),
        (
            lambda worker: worker["selector_state"].__setitem__(
                "best_audio_total", 0.4
            ),
            "in-memory worker summary disagrees",
        ),
        (
            lambda worker: worker["stopper_state"].__setitem__("stale", 1),
            "in-memory worker summary disagrees",
        ),
        (
            lambda worker: worker["validation_history"].reverse(),
            "in-memory worker summary disagrees",
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


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("fault", ["fsync", "identity"])
def test_post_current_fault_reports_committed_generation(
    tmp_path, monkeypatch, persistent, fault
):
    report_dir = tmp_path / "report"
    previous = build_comparison(_ready_systems(tmp_path), report_dir)
    next_dir = tmp_path / "next"
    next_dir.mkdir()
    systems = _ready_systems(next_dir)
    systems[1] = _replace_system(systems[1], [1.0, 1.0, 1.0])
    state = {"committed": False, "raised": False}
    original_rename = report_module.os.rename
    original_fsync = report_module.os.fsync
    original_identity = report_module._verify_output_identity

    def tracking_rename(source, destination, **kwargs):
        result = original_rename(source, destination, **kwargs)
        if destination == "current":
            state["committed"] = True
        return result

    def failing_fsync(descriptor):
        if state["committed"] and (persistent or not state["raised"]):
            state["raised"] = True
            raise OSError("forced post-current fsync failure")
        return original_fsync(descriptor)

    def failing_identity(descriptor, output):
        if state["committed"] and (persistent or not state["raised"]):
            state["raised"] = True
            raise ValueError("forced post-current identity failure")
        return original_identity(descriptor, output)

    monkeypatch.setattr(report_module.os, "rename", tracking_rename)
    if fault == "fsync":
        monkeypatch.setattr(report_module.os, "fsync", failing_fsync)
    else:
        monkeypatch.setattr(
            report_module, "_verify_output_identity", failing_identity
        )

    published = build_comparison(systems, report_dir)
    assert published.committed is True
    assert published.generation_path != previous.generation_path
    expected_warning_count = 2 if persistent and fault == "fsync" else 1
    assert len(published.durability_warnings) == expected_warning_count
    assert fault in published.durability_warnings[0]
    if expected_warning_count == 2:
        assert "warning persistence failed" in published.durability_warnings[1]

    monkeypatch.setattr(report_module.os, "fsync", original_fsync)
    monkeypatch.setattr(
        report_module, "_verify_output_identity", original_identity
    )
    assert resolve_current_report(report_dir) == published.generation_path
    resolved = resolve_current_report(report_dir, include_warnings=True)
    assert resolved.generation_path == published.generation_path
    if persistent and fault == "fsync":
        assert resolved.durability_warnings == ()
    else:
        assert resolved.durability_warnings == published.durability_warnings


@pytest.mark.parametrize(
    "fault,close_number,warning_text",
    [
        ("unlock", None, "lock release failed"),
        ("close", 1, "generations fd close failed"),
        ("close", 2, "lock fd close failed"),
        ("close", 3, "output fd close failed"),
    ],
)
def test_all_post_current_cleanup_failures_are_nonraising_and_persisted(
    tmp_path, monkeypatch, fault, close_number, warning_text
):
    report_dir = tmp_path / "report"
    build_comparison(_ready_systems(tmp_path), report_dir)
    next_root = tmp_path / "next-cleanup"
    next_root.mkdir()
    systems = _ready_systems(next_root)
    systems[1] = _replace_system(systems[1], [1.0, 1.0, 1.0])
    state = {"committed": False, "close_count": 0, "raised": False}
    original_rename = report_module.os.rename
    original_close = report_module.os.close
    original_flock = report_module.fcntl.flock

    def tracking_rename(source, destination, **kwargs):
        result = original_rename(source, destination, **kwargs)
        if destination == "current":
            state["committed"] = True
        return result

    def failing_close(descriptor):
        if state["committed"]:
            state["close_count"] += 1
            if (
                fault == "close"
                and state["close_count"] == close_number
                and not state["raised"]
            ):
                state["raised"] = True
                raise OSError("forced post-current close failure")
        return original_close(descriptor)

    def failing_flock(descriptor, operation):
        if (
            fault == "unlock"
            and state["committed"]
            and operation == report_module.fcntl.LOCK_UN
            and not state["raised"]
        ):
            state["raised"] = True
            raise OSError("forced post-current unlock failure")
        return original_flock(descriptor, operation)

    monkeypatch.setattr(report_module.os, "rename", tracking_rename)
    monkeypatch.setattr(report_module.os, "close", failing_close)
    monkeypatch.setattr(report_module.fcntl, "flock", failing_flock)
    published = build_comparison(systems, report_dir)
    assert published.committed is True
    assert any(warning_text in item for item in published.durability_warnings)

    monkeypatch.setattr(report_module.os, "close", original_close)
    monkeypatch.setattr(report_module.fcntl, "flock", original_flock)
    resolved = resolve_current_report(report_dir, include_warnings=True)
    assert resolved.generation_path == published.generation_path
    assert resolved.durability_warnings == published.durability_warnings


def test_distinct_artifacts_are_hashed_and_inspected_once(tmp_path, monkeypatch):
    hash_calls = []
    worker_read_calls = []
    identity_calls = []
    original_hash = report_module._hash_fd
    original_read = report_module._read_verified_artifact
    original_identity = report_module.resolve_pilot_checkpoint_identity

    def counting_hash(descriptor):
        stat_result = report_module.os.fstat(descriptor)
        hash_calls.append((stat_result.st_dev, stat_result.st_ino))
        return original_hash(descriptor)

    def counting_identity(path, provenance):
        identity_calls.append(Path(path).resolve())
        return original_identity(path, provenance)

    def counting_read(path, expected_sha256, *, limit, name):
        if Path(path).name == "worker_summary.json":
            worker_read_calls.append(Path(path).resolve())
        return original_read(
            path, expected_sha256, limit=limit, name=name
        )

    monkeypatch.setattr(report_module, "_hash_fd", counting_hash)
    monkeypatch.setattr(
        report_module, "_read_verified_artifact", counting_read
    )
    build_comparison(
        _ready_systems(tmp_path),
        tmp_path / "report",
        checkpoint_identity_resolver=counting_identity,
    )
    assert len(hash_calls) == len(set(hash_calls)) == 7
    assert len(worker_read_calls) == len(set(worker_read_calls)) == 3
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

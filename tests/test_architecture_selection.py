from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avgaussianv2.benchmark.architecture_selection import (
    OBJECTIVES,
    select_architecture_winners,
)
from avgaussianv2.benchmark.evaluation import SCENE_SAMPLE_COUNTS


SCENES = ("scene1_opera", "Scene7playing")
SYSTEMS = ("audio_only", "query_dependent_p1")


def _manifest(tmp_path: Path) -> Path:
    configs = []
    runs = []
    for scene in SCENES:
        for system in SYSTEMS:
            config = tmp_path / f"{scene}-{system}.yaml"
            config.write_text("{}\n")
            config_id = f"{scene}-{system}"
            mode = "audio_only" if system == "audio_only" else "joint_conditioned"
            configs.append(
                {
                    "config_id": config_id,
                    "config": str(config),
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "scene": scene,
                    "system": system,
                    "training_mode": mode,
                    "seed": 42,
                    "lambda_lre": 0.0,
                }
            )
            evaluation_systems = [system]
            if system == "query_dependent_p1":
                evaluation_systems += [
                    "query_dependent_p1_no_rgbd",
                    "query_dependent_p1_wrong_camera",
                ]
            runs.append(
                {
                    "run_id": config_id,
                    "continuation_id": config_id,
                    "config_id": config_id,
                    "stage": "architecture",
                    "scene": scene,
                    "system": system,
                    "training_mode": mode,
                    "seed": 42,
                    "lambda_lre": 0.0,
                    "evaluation_systems": evaluation_systems,
                    "report_steps": [5_000],
                    "max_steps": 5_000,
                    "stop_after_step": 5_000,
                }
            )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "avgaussianv2.lre-loss-run-manifest",
                "version": 1,
                "stage": "architecture",
                "repository": {
                    "root": str(tmp_path.resolve()),
                    "commit": "1" * 40,
                    "clean": True,
                },
                "configs": configs,
                "runs": runs,
            }
        )
    )
    return path


def _result(scene: str, system: str, value: float, *, checkpoint: str):
    count = SCENE_SAMPLE_COUNTS[scene]
    sample_ids = tuple(f"{scene}/cam38/{index:06d}" for index in range(count))
    summary = {
        metric: {"mean": value + index * 0.001}
        for index, metric in enumerate(OBJECTIVES)
    }
    directions = {metric: "lower_is_better" for metric in OBJECTIVES}
    provenance = {
        "seed": 42,
        "index_sha256": "a" * 64,
        "planned_updates": 30_000,
        "completed_updates": 5_000,
        "checkpoint_step": 5_000,
        "batch_size": 1,
        "audio_initialization_sha256": "b" * 64,
        "visual_initialization_sha256": "c" * 64,
        "model_initialization_sha256": checkpoint,
        "checkpoint_sha256": checkpoint,
    }
    return SimpleNamespace(
        identity=SimpleNamespace(
            scene_id=scene,
            system_name=system,
            reporting_step=5_000,
            expected_sample_ids=sample_ids,
            expected_sample_count=count,
        ),
        count=count,
        summary=summary,
        metric_directions=directions,
        metric_protocol={
            "extra_metric_registry": {
                "paper_dpam": {
                    "direction": "lower_is_better",
                    "modality": "audio",
                    "protocol": {"model_state_sha256": "d" * 64},
                }
            }
        },
        provenance=provenance,
        content_sha256=hashlib.sha256(f"{scene}/{system}".encode()).hexdigest(),
    )


def _evaluations(tmp_path: Path):
    values = {}
    for scene in SCENES:
        for system, value in (("audio_only", 1.2), ("query_dependent_p1", 1.0)):
            root = tmp_path / system.replace("query_dependent_p1", scene + "-query_dependent_p1")
            if system == "audio_only":
                root = tmp_path / f"{scene}-audio_only"
            checkpoint = ("e" if system == "audio_only" else "f") * 64
            values[(root / "evaluations/step_005000").resolve()] = _result(
                scene, system, value, checkpoint=checkpoint
            )
            if system == "query_dependent_p1":
                for branch, branch_value in (
                    ("query_dependent_p1_no_rgbd", 1.1),
                    ("query_dependent_p1_wrong_camera", 1.15),
                ):
                    values[
                        (root / f"evaluations/{branch}/step_005000").resolve()
                    ] = _result(scene, branch, branch_value, checkpoint=checkpoint)
    return values


def test_architecture_selector_applies_fairness_causal_and_pareto_gates(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    evaluations = _evaluations(tmp_path)

    result = select_architecture_winners(
        manifest,
        tmp_path,
        tmp_path / "selection.json",
        evaluation_loader=lambda path: evaluations[path.resolve()],
        repository_identity_getter=lambda: {
            "root": str(tmp_path.resolve()),
            "commit": "1" * 40,
            "clean": True,
        },
    )

    assert result["selected_systems"] == ["query_dependent_p1"]
    assert result["pareto_systems"] == ["query_dependent_p1"]
    assert json.loads((tmp_path / "selection.json").read_text()) == result
    candidate = next(
        item for item in result["candidates"] if item["system"] == "query_dependent_p1"
    )
    assert candidate["causal_gate_passed"] is True
    assert len(candidate["causal_evidence"]) == 2


def test_architecture_selector_rejects_incomplete_dpam_protocol(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    evaluations = _evaluations(tmp_path)
    first = next(iter(evaluations.values()))
    first.metric_protocol.clear()

    with pytest.raises(ValueError, match="DPAM protocol is missing"):
        select_architecture_winners(
            manifest,
            tmp_path,
            tmp_path / "selection.json",
            evaluation_loader=lambda path: evaluations[path.resolve()],
            repository_identity_getter=lambda: {
                "root": str(tmp_path.resolve()),
                "commit": "1" * 40,
                "clean": True,
            },
        )


def test_architecture_selector_enforces_spatial_guardrail(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    evaluations = _evaluations(tmp_path)
    for result in evaluations.values():
        if result.identity.system_name == "query_dependent_p1":
            result.summary["paper_lre_db"]["mean"] = 2.0

    selected = select_architecture_winners(
        manifest,
        tmp_path,
        tmp_path / "selection.json",
        evaluation_loader=lambda path: evaluations[path.resolve()],
        repository_identity_getter=lambda: {
            "root": str(tmp_path.resolve()),
            "commit": "1" * 40,
            "clean": True,
        },
    )

    assert selected["selected_systems"] == ["audio_only"]
    candidate = next(
        item
        for item in selected["candidates"]
        if item["system"] == "query_dependent_p1"
    )
    assert candidate["spatial_guardrail_passed"] is False

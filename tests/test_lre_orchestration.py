from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from avgaussianv2.benchmark.lre_orchestration import (
    build_lre_pipelines,
    execute_lre_pipelines,
    load_lre_run_manifest,
    query_idle_gpus,
)
from avgaussianv2.benchmark.orchestration import OrchestrationError


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    configs = []
    runs = []
    for index, (system, mode) in enumerate(
        (("audio_only", "audio_only"), ("query_dependent_p1", "joint_conditioned"))
    ):
        config = tmp_path / f"config-{index}.yaml"
        config.write_text(f"system: {system}\n")
        config_id = f"config-{index}"
        configs.append(
            {
                "config_id": config_id,
                "system": system,
                "training_mode": mode,
                "scene": "scene1_opera",
                "seed": 42,
                "lambda_lre": 0.0,
                "base_config": str(config),
                "config": str(config),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
        )
        runs.append(
            {
                "run_id": f"run-{index}",
                "continuation_id": f"continuation-{index}",
                "stage": "screening",
                "config_id": config_id,
                "scene": "scene1_opera",
                "system": system,
                "training_mode": mode,
                "seed": 42,
                "lambda_lre": 0.0,
                "control_run_id": None,
                "report_steps": [5_000],
                "max_steps": 5_000,
                "stop_after_step": 5_000,
            }
        )
    value = {
        "schema": "avgaussianv2.lre-loss-run-manifest",
        "version": 1,
        "stage": "screening",
        "source_manifest": str(tmp_path / "source.yaml"),
        "source_manifest_sha256": "0" * 64,
        "winners": None,
        "configs": configs,
        "runs": runs,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path, value


def _native_root(tmp_path: Path) -> Path:
    root = tmp_path / "native"
    for kind in ("audiogs", "ftgspp"):
        (root / "scene1_opera" / kind / "native_contract").mkdir(parents=True)
    return root


class _Handle:
    def poll(self):
        return 0

    def wait(self, timeout=None):
        del timeout
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


class _Runner:
    def __init__(self):
        self.assignments = []

    def start(self, command, *, env, log_path):
        self.assignments.append((tuple(command), dict(env), log_path))
        return _Handle()


def test_lre_runner_consumes_stop_step_and_keeps_one_pipeline_per_gpu(tmp_path):
    path, _ = _manifest(tmp_path)
    manifest = load_lre_run_manifest(path)
    pipelines = build_lre_pipelines(
        manifest,
        output_root=tmp_path / "runs",
        native_root=_native_root(tmp_path),
        python_executable="python",
        compute_dpam=True,
        trust_upstream_artifacts=True,
        resume=False,
    )
    assert len(pipelines) == 2
    for pipeline in pipelines:
        assert (pipeline.run_dir / "evaluations").is_dir()
        assert [stage.name for stage in pipeline.stages] == [
            "prepare",
            "train",
            "eval_005000",
        ]
        train = pipeline.stages[1].command
        assert train[train.index("--stop-after-step") + 1] == "5000"
        assert "--compute-dpam" in pipeline.stages[2].command

    runner = _Runner()
    result = execute_lre_pipelines(
        pipelines, gpus=(1, 2), runner=runner, poll_seconds=0
    )
    assert len(result["runs"]) == 2
    for pipeline in pipelines:
        history = pipeline.run_dir / f"result_history/run_result.{pipeline.stage}"
        assert len(tuple(history.glob("attempt-*.json"))) == 1
    assert [assignment[1]["CUDA_VISIBLE_DEVICES"] for assignment in runner.assignments] == [
        "1",
        "2",
        "1",
        "2",
        "1",
        "2",
    ]


def test_lre_runner_resume_verifies_existing_evaluation_without_retraining(tmp_path):
    path, _ = _manifest(tmp_path)
    manifest = load_lre_run_manifest(path)
    subset = {
        **manifest,
        "configs": manifest["configs"][:1],
        "runs": manifest["runs"][:1],
    }
    native_root = _native_root(tmp_path)
    build_lre_pipelines(
        subset,
        output_root=tmp_path / "runs",
        native_root=native_root,
        python_executable="python",
        compute_dpam=True,
        trust_upstream_artifacts=True,
        resume=False,
    )
    run = tmp_path / "runs/continuation-0"
    (run / "protocol").mkdir(parents=True)
    (run / "protocol/preparation.json").write_text("{}")
    (run / "worker").mkdir()
    (run / "worker/progress.json").write_text(json.dumps({"exact_main_step": 5_000}))
    (run / "evaluations/step_005000").mkdir(parents=True)
    (run / "evaluations/step_005000/current.json").write_text("{}")

    pipelines = build_lre_pipelines(
        subset,
        output_root=tmp_path / "runs",
        native_root=native_root,
        python_executable="python",
        compute_dpam=True,
        trust_upstream_artifacts=True,
        resume=True,
    )
    assert [stage.name for stage in pipelines[0].stages] == ["eval_005000"]
    assert "--verify-only" in pipelines[0].stages[0].command


def test_lre_runner_refuses_unbound_or_mismatched_continuation(tmp_path):
    path, _ = _manifest(tmp_path)
    manifest = load_lre_run_manifest(path)
    subset = {
        **manifest,
        "configs": manifest["configs"][:1],
        "runs": manifest["runs"][:1],
    }
    run = tmp_path / "runs/continuation-0"
    run.mkdir(parents=True)
    (run / "foreign.txt").write_text("foreign\n")
    with pytest.raises(OrchestrationError, match="lacks identity"):
        build_lre_pipelines(
            subset,
            output_root=tmp_path / "runs",
            native_root=_native_root(tmp_path),
            python_executable="python",
            compute_dpam=True,
            trust_upstream_artifacts=True,
            resume=True,
        )


def test_architecture_pipeline_schedules_main_and_causal_evaluations(tmp_path):
    path, value = _manifest(tmp_path)
    value["stage"] = "architecture"
    run = value["runs"][1]
    run["stage"] = "architecture"
    run["evaluation_systems"] = [
        "query_dependent_p1",
        "query_dependent_p1_no_rgbd",
        "query_dependent_p1_wrong_camera",
    ]
    value["configs"] = value["configs"][1:]
    value["runs"] = value["runs"][1:]
    path.write_text(json.dumps(value))

    pipelines = build_lre_pipelines(
        load_lre_run_manifest(path),
        output_root=tmp_path / "runs",
        native_root=_native_root(tmp_path),
        python_executable="python",
        compute_dpam=True,
        trust_upstream_artifacts=True,
        resume=False,
    )

    stages = pipelines[0].stages
    assert [stage.name for stage in stages] == [
        "prepare",
        "train",
        "eval_005000",
        "eval_query_dependent_p1_no_rgbd_005000",
        "eval_query_dependent_p1_wrong_camera_005000",
    ]
    assert stages[2].command[stages[2].command.index("--output-dir") + 1].endswith(
        "evaluations/step_005000"
    )
    assert stages[3].command[stages[3].command.index("--output-dir") + 1].endswith(
        "evaluations/query_dependent_p1_no_rgbd/step_005000"
    )
    assert (
        pipelines[0].run_dir / "evaluations/query_dependent_p1_no_rgbd"
    ).is_dir()
    assert (
        pipelines[0].run_dir / "evaluations/query_dependent_p1_wrong_camera"
    ).is_dir()


def test_lre_manifest_and_gpu_preflight_fail_closed(tmp_path):
    path, value = _manifest(tmp_path)
    Path(value["configs"][0]["config"]).write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_lre_run_manifest(path)

    def busy(*args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess([], 0, "1, 2048, 100\n2, 48000, 0\n", "")

    with pytest.raises(OrchestrationError, match="busy"):
        query_idle_gpus((1, 2), command_runner=busy)

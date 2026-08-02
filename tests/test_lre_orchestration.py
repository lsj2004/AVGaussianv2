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
    run_lre_manifest,
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
        "repository": {
            "root": str(tmp_path.resolve()),
            "commit": "1" * 40,
            "clean": True,
        },
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


class _PendingHandle(_Handle):
    def poll(self):
        return None


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
        dpam_python="/opt/dpam/bin/python",
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
        assert pipeline.stages[2].command[
            pipeline.stages[2].command.index("--dpam-python") + 1
        ] == "/opt/dpam/bin/python"

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
    assert all(record["status"] == "succeeded" for record in result["runs"])


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
        dpam_python="/opt/dpam/bin/python",
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
    native_root = _native_root(tmp_path)
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
            native_root=native_root,
            python_executable="python",
            compute_dpam=True,
            trust_upstream_artifacts=True,
            resume=True,
        )

    path, value = _manifest(tmp_path)
    first = {**value, "configs": value["configs"][:1], "runs": value["runs"][:1]}
    run = tmp_path / "bound-runs/continuation-0"
    build_lre_pipelines(
        first,
        output_root=tmp_path / "bound-runs",
        native_root=native_root,
        python_executable="python",
        compute_dpam=False,
        trust_upstream_artifacts=True,
        resume=False,
    )
    changed = {
        **first,
        "repository": {**first["repository"], "commit": "2" * 40},
    }
    with pytest.raises(OrchestrationError, match="identity mismatch"):
        build_lre_pipelines(
            changed,
            output_root=tmp_path / "bound-runs",
            native_root=native_root,
            python_executable="python",
            compute_dpam=False,
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
        dpam_python="/opt/dpam/bin/python",
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
    assert "--compute-dpam" in stages[2].command
    assert "--dpam-python" in stages[2].command
    assert "--compute-dpam" not in stages[3].command
    assert "--dpam-python" not in stages[3].command
    assert "--compute-dpam" not in stages[4].command
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
        del kwargs
        command = args[0]
        if "--query-compute-apps=pid,gpu_uuid" in command:
            return subprocess.CompletedProcess([], 0, "", "")
        return subprocess.CompletedProcess(
            [], 0, "1, GPU-1, 2048, 100\n2, GPU-2, 48000, 0\n", ""
        )

    with pytest.raises(OrchestrationError, match="busy"):
        query_idle_gpus((1, 2), command_runner=busy)


def test_lre_gpu_preflight_rejects_live_compute_pid_but_ignores_stale_nvml():
    def query(*args, **kwargs):
        del kwargs
        command = args[0]
        if "--query-compute-apps=pid,gpu_uuid" in command:
            return subprocess.CompletedProcess(
                [], 0, "123, GPU-1\n999, GPU-2\n", ""
            )
        return subprocess.CompletedProcess(
            [], 0, "1, GPU-1, 48000, 0\n2, GPU-2, 48000, 0\n", ""
        )

    with pytest.raises(OrchestrationError, match="busy") as captured:
        query_idle_gpus(
            (1, 2),
            command_runner=query,
            process_state_getter=lambda pid: "S" if pid == 123 else None,
        )
    assert "123" in str(captured.value)
    assert "999" not in str(captured.value)

    accepted = query_idle_gpus(
        (2,),
        command_runner=query,
        process_state_getter=lambda _pid: None,
    )
    assert accepted[2]["uuid"] == "GPU-2"
    assert accepted[2]["compute_pids"] == []


def test_lre_runner_rejects_manifest_from_another_repository_revision(tmp_path):
    path, value = _manifest(tmp_path)
    current = {**value["repository"], "commit": "2" * 40}

    with pytest.raises(OrchestrationError, match="repository identity differs"):
        run_lre_manifest(
            path,
            output_root=tmp_path / "runs",
            native_root=_native_root(tmp_path),
            gpus=(1,),
            python_executable="python",
            repository_identity_getter=lambda: current,
            gpu_query=lambda devices: {devices[0]: {}},
        )


def test_lre_runner_allows_only_explicit_same_commit_clean_relocation(tmp_path):
    path, value = _manifest(tmp_path)
    relocated = {
        **value["repository"],
        "root": str((tmp_path / "relocated-clean-worktree").resolve()),
    }
    native_root = _native_root(tmp_path)
    runner = _Runner()

    with pytest.raises(OrchestrationError, match="repository identity differs"):
        run_lre_manifest(
            path,
            output_root=tmp_path / "strict-runs",
            native_root=native_root,
            gpus=(1,),
            python_executable="python",
            repository_identity_getter=lambda: relocated,
            gpu_query=lambda devices: {devices[0]: {}},
            runner=runner,
        )

    result = run_lre_manifest(
        path,
        output_root=tmp_path / "relocated-runs",
        native_root=native_root,
        gpus=(1,),
        python_executable="python",
        allow_repository_relocation=True,
        repository_identity_getter=lambda: relocated,
        gpu_query=lambda devices: {devices[0]: {}},
        runner=_Runner(),
    )

    assert result["repository"] == relocated
    assert result["manifest_repository"] == value["repository"]
    assert result["repository_relocation"] == {
        "manifest_root": value["repository"]["root"],
        "execution_root": relocated["root"],
        "same_clean_commit": True,
    }
    continuation = json.loads(
        (tmp_path / "relocated-runs/continuation-0/continuation_identity.json").read_text()
    )
    assert continuation["repository"] == value["repository"]
    manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_result_path = (
        tmp_path
        / "relocated-runs"
        / "runner_results"
        / "screening"
        / manifest_sha256
        / "runner_result.json"
    )
    manifest_result = json.loads(manifest_result_path.read_text())
    stage_pointer = json.loads(
        (tmp_path / "relocated-runs/runner_result.screening.json").read_text()
    )
    assert manifest_result == stage_pointer
    assert manifest_result["source_manifest_sha256"] == manifest_sha256
    assert manifest_result["manifest_result"] == str(manifest_result_path)
    manifest_history = manifest_result_path.parent / "result_history/runner_result"
    assert len(tuple(manifest_history.glob("attempt-*.json"))) == 1

    wrong_revision = {**relocated, "commit": "2" * 40}
    with pytest.raises(OrchestrationError, match="repository identity differs"):
        run_lre_manifest(
            path,
            output_root=tmp_path / "wrong-revision-runs",
            native_root=native_root,
            gpus=(1,),
            python_executable="python",
            allow_repository_relocation=True,
            repository_identity_getter=lambda: wrong_revision,
            gpu_query=lambda devices: {devices[0]: {}},
            runner=_Runner(),
        )


def test_failed_manifest_run_publishes_manifest_specific_failure(tmp_path):
    path, value = _manifest(tmp_path)

    class FailedHandle:
        def poll(self):
            return 7

        def wait(self, timeout=None):
            del timeout
            return 7

        def terminate(self):
            pass

        def kill(self):
            pass

    class FailedRunner:
        assignments = []

        def start(self, command, *, env, log_path):
            del command, env, log_path
            return FailedHandle()

    with pytest.raises(OrchestrationError, match="stage failed"):
        run_lre_manifest(
            path,
            output_root=tmp_path / "failed-runs",
            native_root=_native_root(tmp_path),
            gpus=(1,),
            python_executable="python",
            repository_identity_getter=lambda: value["repository"],
            gpu_query=lambda devices: {devices[0]: {}},
            runner=FailedRunner(),
        )

    manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    specific = json.loads(
        (
            tmp_path
            / "failed-runs"
            / "runner_results"
            / "screening"
            / manifest_sha256
            / "runner_result.json"
        ).read_text()
    )
    stage_pointer = json.loads(
        (tmp_path / "failed-runs/runner_result.screening.json").read_text()
    )
    assert specific == stage_pointer
    assert specific["status"] == "failed"
    assert specific["error"]["type"] == "OrchestrationError"


def test_failed_pipeline_publishes_failure_and_peer_abort_evidence(tmp_path):
    _, manifest = _manifest(tmp_path)
    pipelines = build_lre_pipelines(
        manifest,
        output_root=tmp_path / "runs",
        native_root=_native_root(tmp_path),
        python_executable="python",
        compute_dpam=False,
        trust_upstream_artifacts=True,
        resume=False,
    )

    class _TrackedHandle(_Handle):
        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

    class _FailureHandle(_TrackedHandle):
        def poll(self):
            return 9

    class _TrackedPendingHandle(_TrackedHandle):
        def poll(self):
            return None

    class _MixedRunner:
        def __init__(self):
            self.calls = 0
            self.handles = []

        def start(self, command, *, env, log_path):
            del command, env, log_path
            self.calls += 1
            handle = (
                _FailureHandle() if self.calls == 1 else _TrackedPendingHandle()
            )
            self.handles.append(handle)
            return handle

    runner = _MixedRunner()
    with pytest.raises(OrchestrationError, match="stage failed"):
        execute_lre_pipelines(
            pipelines, gpus=(1, 2), runner=runner, poll_seconds=0
        )

    failed = json.loads(
        (pipelines[0].run_dir / "run_result.screening.json").read_text()
    )
    aborted = json.loads(
        (pipelines[1].run_dir / "run_result.screening.json").read_text()
    )
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "prepare"
    assert failed["exit_code"] == 9
    assert aborted["status"] == "aborted_due_to_peer_failure"
    assert aborted["peer_failed_run_id"] == pipelines[0].run_id
    assert [handle.terminate_calls for handle in runner.handles] == [1, 1]
    assert [handle.kill_calls for handle in runner.handles] == [1, 1]
    peer_history = pipelines[1].run_dir / "result_history/run_result.screening"
    assert len(tuple(peer_history.glob("attempt-*.json"))) == 1

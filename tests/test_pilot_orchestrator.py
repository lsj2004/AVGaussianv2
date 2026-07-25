from __future__ import annotations

import subprocess
import sys
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from avgaussianv2.cli.pilot import (
    LaunchSpec,
    PilotProcessError,
    _evenly_spaced,
    _launch_group,
    _shared_indices,
    _validate_complete_status,
    _validate_experiment_types,
    _wait_group,
    parse_gpus,
    run_pilot,
)
from avgaussianv2.cli.pilot_worker import build_worker_component_identities
from avgaussianv2.experiment.contracts import PilotConfig, Variant


def test_three_gpu_parser_is_exact() -> None:
    assert parse_gpus("0,1,2") == (0, 1, 2)
    for invalid in ("0,1", "0,1,1", "0,-1,2", "a,1,2", "0,1,2,3"):
        with pytest.raises(ValueError):
            parse_gpus(invalid)


def test_pilot_help_is_cuda_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "avgaussianv2.cli.pilot", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--verify-only" in result.stdout


def test_parent_and_worker_share_one_component_identity_builder() -> None:
    identities = build_worker_component_identities("example.Model")
    assert identities["model_class"] == "example.Model"
    assert set(identities) == {
        "model_class",
        "warmup_optimizer_factory",
        "joint_optimizer_factory",
        "warmup_optimizer_class",
        "joint_optimizer_class",
        "warmup_step_fn",
        "joint_step_fn",
        "audio_loss_fn",
    }


def test_process_failures_are_aggregated_after_every_wait(tmp_path) -> None:
    class Handle:
        def __init__(self, code):
            self.code = code
            self.waited = False

        def wait(self):
            self.waited = True
            return self.code

        def terminate(self):
            pass

    handles = [Handle(3), Handle(0), Handle(7)]
    jobs = tuple(
        (f"job-{index}", handle, tmp_path / f"{index}.log")
        for index, handle in enumerate(handles)
    )
    with pytest.raises(PilotProcessError) as raised:
        _wait_group(jobs)
    assert all(handle.waited for handle in handles)
    assert [failure[:2] for failure in raised.value.failures] == [
        ("job-0", 3),
        ("job-2", 7),
    ]


def test_group_start_exception_terminates_and_reaps_prior_children(tmp_path) -> None:
    class Handle:
        def __init__(self):
            self.terminated = False
            self.waited = False
            self.stopped = threading.Event()

        def terminate(self):
            self.terminated = True
            self.stopped.set()

        def wait(self):
            self.waited = True
            self.stopped.wait()
            return -15

    class Runner:
        assignments = []

        def __init__(self):
            self.handles = [Handle(), Handle()]
            self.position = 0

        def start(self, command, *, env, log_path):
            self.assignments.append((tuple(command), dict(env), log_path))
            if self.position == 2:
                raise OSError("cannot start")
            handle = self.handles[self.position]
            self.position += 1
            return handle

    runner = Runner()
    specs = tuple(
        LaunchSpec(
            f"job-{index}",
            ("python", "-m", "worker", str(index)),
            index,
            tmp_path / f"job-{index}.log",
        )
        for index in range(3)
    )
    with pytest.raises(PilotProcessError) as raised:
        _launch_group(runner, specs)
    assert all(handle.terminated and handle.waited for handle in runner.handles)
    assert [failure[0] for failure in raised.value.failures] == [
        "job-0",
        "job-1",
        "job-2",
    ]
    assert [assignment[1]["CUDA_VISIBLE_DEVICES"] for assignment in runner.assignments] == [
        "0",
        "1",
        "2",
    ]
    assert [assignment[2] for assignment in runner.assignments] == [
        tmp_path / "job-0.log",
        tmp_path / "job-1.log",
        tmp_path / "job-2.log",
    ]


def test_fail_fast_supervision_kills_unresponsive_sibling(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.cli.pilot as pilot_module

    monkeypatch.setattr(pilot_module, "PROCESS_TERMINATION_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(pilot_module, "PROCESS_REAP_GRACE_SECONDS", 0.1)

    class Failed:
        def wait(self):
            return 5

        def terminate(self):
            pass

    class Stuck:
        def __init__(self):
            self.done = threading.Event()
            self.terminated = False
            self.killed = False

        def wait(self):
            self.done.wait()
            return -9

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.done.set()

    stuck = Stuck()
    jobs = (
        ("failed", Failed(), tmp_path / "failed.log"),
        ("stuck", stuck, tmp_path / "stuck.log"),
    )
    with pytest.raises(PilotProcessError) as raised:
        _wait_group(jobs)
    assert stuck.terminated and stuck.killed
    assert raised.value.outcomes == (
        ("failed", 5, tmp_path / "failed.log"),
        ("stuck", -9, tmp_path / "stuck.log"),
    )


def test_shared_sequences_are_deterministic_and_condition_off_has_no_warmup() -> None:
    pilot = PilotConfig(warmup_steps=7, joint_steps=11)
    first = _shared_indices(123, 5, pilot)
    second = _shared_indices(123, 5, pilot)
    assert first == second
    assert len(first.warmup) == 7
    assert len(first.joint) == 11
    assert first.for_variant(Variant.CONDITION_OFF).warmup == ()
    assert first.for_variant(Variant.CONDITION_OFF).joint == first.joint
    assert _evenly_spaced(100, 32) == _evenly_spaced(100, 32)
    assert _evenly_spaced(3, 32) == (0, 1, 2)


def test_expired_proc_checkpoint_provenance_is_a_targeted_migration(
    tmp_path,
) -> None:
    from avgaussianv2.cli.pilot import _has_expired_proc_checkpoint_provenance

    manifest = tmp_path / "evaluation_manifest.json"
    manifest.write_text(json.dumps({
        "systems": [{
            "checkpoint": {
                "checkpoint_path": "/proc/999999999/fd/47/best.pt"
            }
        }]
    }))
    assert _has_expired_proc_checkpoint_provenance(manifest)

    payload = json.loads(manifest.read_text())
    payload["systems"][0]["checkpoint"]["checkpoint_path"] = str(
        tmp_path / "missing.pt"
    )
    manifest.write_text(json.dumps(payload))
    assert not _has_expired_proc_checkpoint_provenance(manifest)


def _verify_config(tmp_path):
    for name in ("visual.pt", "audio.pt", "dataset.json"):
        (tmp_path / name).write_text(name)
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
train: {{seed: 7}}
"""
    )
    return config


def test_verify_only_derives_trusted_run_identity_without_cli_permission(tmp_path) -> None:
    config = _verify_config(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    experiment = {
        "schema": "avgaussianv2.scene1-three-gpu-pilot",
        "version": 1,
        "scene_id": "scene1_opera",
        "gpus": [0, 1, 2],
        "config_sha256": "a" * 64,
        "source_config_sha256": "a" * 64,
        "runtime_config_sha256": "a" * 64,
        "shared_manifest_path": str(output / "shared_manifest.json"),
        "shared_manifest_sha256": "b" * 64,
        "baseline_manifest_path": str(output / "baseline_manifest.json"),
        "baseline_manifest_sha256": "c" * 64,
        "source_hashes": {
            "project_config_sha256": "a" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "visual_checkpoint_sha256": "f" * 64,
            "audio_checkpoint_sha256": "0" * 64,
            "camera_mapping_sha256": "1" * 64,
        },
        "trusted_upstream_artifacts": True,
    }
    (output / "experiment_manifest.json").write_text(json.dumps(experiment))
    (output / ".pilot-orchestrator.lock").write_text("")

    class Runner:
        assignments = []

        def start(self, *args, **kwargs):
            pytest.fail("verify-only started a process")

    before = set(output.iterdir())
    with pytest.raises(FileNotFoundError):
        run_pilot(
            config,
            output,
            verify_only=True,
            runner=Runner(),
        )
    assert set(output.iterdir()) == before
    assert Runner.assignments == []

    experiment["trusted_upstream_artifacts"] = "true"
    (output / "experiment_manifest.json").write_text(json.dumps(experiment))
    with pytest.raises(TypeError, match="must be boolean"):
        run_pilot(
            config,
            output,
            verify_only=True,
            runner=Runner(),
        )


def test_run_pilot_sequences_baseline_workers_and_evaluations(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.cli.pilot as pilot_module
    from avgaussianv2.experiment.evaluation import METRIC_NAMES

    config = _verify_config(tmp_path)
    output = tmp_path / "pilot"
    events = []
    assignments = []
    baseline_summary = {
        metric: {"mean": 0.5, "std": 0.0, "median": 0.5}
        for metric in METRIC_NAMES
    }
    baseline_summary["rgb_psnr"]["mean"] = 30.0
    baseline_summary["rgb_psnr"]["median"] = 30.0
    baseline_summary["rgb_ssim"]["mean"] = 0.95
    baseline_summary["rgb_ssim"]["median"] = 0.95

    class Handle:
        def __init__(self, command):
            self.command = command

        def terminate(self):
            events.append(("terminate", tuple(self.command)))

        def wait(self):
            module = self.command[2]
            events.append(("wait", module, tuple(self.command)))
            destination = Path(
                self.command[self.command.index("--output-dir") + 1]
            )
            destination.mkdir(parents=True, exist_ok=True)
            if module == "avgaussianv2.cli.pilot_eval":
                (destination / "evaluation_manifest.json").write_text("{}")
                if "--checkpoint" not in self.command:
                    system = destination / "baseline_imported"
                    system.mkdir(exist_ok=True)
                    (system / "metrics_summary.json").write_text(
                        json.dumps(baseline_summary)
                    )
            else:
                (destination / "complete.marker").write_text("ok")
            return 0

    class Runner:
        def start(self, command, *, env, log_path):
            command = tuple(command)
            module = command[2]
            if module == "avgaussianv2.cli.pilot_worker":
                assert any(
                    event[:2] == ("wait", "avgaussianv2.cli.pilot_eval")
                    for event in events
                ), "worker started before baseline completed"
            if "--resume" in command or "--overwrite" in command:
                status = json.loads((output / "status.json").read_text())
                assert status["ready"] is False
                assert "mutating" in status["stages"].values()
            assignments.append((command, dict(env), log_path))
            events.append(("start", module, command))
            return Handle(command)

    def verified_eval(destination, specs, **kwargs):
        assert (destination / "evaluation_manifest.json").is_file()
        return SimpleNamespace(
            manifest_path=destination / "evaluation_manifest.json",
            artifacts=(),
            evaluations=(),
        )

    def verified_worker(*args, **kwargs):
        worker_dir = Path(args[3])
        assert (worker_dir / "complete.marker").is_file()
        return SimpleNamespace(summary={})

    monkeypatch.setattr(pilot_module, "_verify_existing", verified_eval)
    monkeypatch.setattr(pilot_module, "verify_worker_output", verified_worker)
    monkeypatch.setattr(
        pilot_module,
        "load_worker_manifest",
        lambda *args, **kwargs: SimpleNamespace(sha256="a" * 64),
    )
    comparison = SimpleNamespace(
            content_digest="b" * 64,
            generation_path=output / "report-generation",
            decision=SimpleNamespace(ready=True),
            durability_warnings=(),
        )

    def build_report(*args, **kwargs):
        report = output / "report"
        generation = report / ".report-generations" / ("b" * 64)
        generation.mkdir(parents=True, exist_ok=True)
        current = report / "current"
        if not current.exists():
            current.symlink_to(f".report-generations/{'b' * 64}")
        comparison.generation_path = generation
        return comparison

    monkeypatch.setattr(pilot_module, "_report_inputs", lambda *args: [])
    monkeypatch.setattr(pilot_module, "_build_report", build_report)
    monkeypatch.setattr(
        pilot_module,
        "verify_current_comparison",
        lambda *args, **kwargs: comparison,
    )

    # The baseline manifest drives deterministic shared-manifest construction.
    original_wait = Handle.wait

    def wait_with_baseline_manifest(self):
        code = original_wait(self)
        if (
            self.command[2] == "avgaussianv2.cli.pilot_eval"
            and "--checkpoint" not in self.command
        ):
            destination = Path(
                self.command[self.command.index("--output-dir") + 1]
            )
            (destination / "evaluation_manifest.json").write_text(
                json.dumps(
                    {
                        "train_length": 5,
                        "eval_length": 40,
                        "runtime_identity": {
                            "model_class": "tests.Model",
                            "model_format_version": "state-dict-v1",
                        },
                    }
                )
            )
        return code

    monkeypatch.setattr(Handle, "wait", wait_with_baseline_manifest)
    result = run_pilot(
        config,
        output,
        runner=Runner(),
        gpu_validator=lambda ids: None,
        trust_upstream_artifacts=True,
    )

    worker = [
        item for item in assignments
        if item[0][2] == "avgaussianv2.cli.pilot_worker"
    ]
    evaluations = [
        item for item in assignments
        if item[0][2] == "avgaussianv2.cli.pilot_eval"
        and "--checkpoint" in item[0]
    ]
    assert len(worker) == len(evaluations) == 3
    assert [item[1]["CUDA_VISIBLE_DEVICES"] for item in worker] == ["0", "1", "2"]
    assert [item[1]["CUDA_VISIBLE_DEVICES"] for item in evaluations] == [
        "0", "1", "2"
    ]
    for command, env, _ in assignments:
        assert env["AVGAUSSIANV2_ORCHESTRATOR_PID"] == str(os.getpid())
        assert command[command.index("--config") + 1].startswith(
            f"/proc/{os.getpid()}/fd/"
        )
        assert command[command.index("--output-dir") + 1].startswith(
            f"/proc/{os.getpid()}/fd/"
        )
    joint = evaluations[0][0]
    assert joint.count("--system") == 2
    assert "joint_conditioned_on:on" in joint
    assert "joint_conditioned_off:off" in joint
    shared = json.loads((output / "shared_manifest.json").read_text())
    assert shared["compatibility"]["condition_off"]["variant"] == "condition_off"
    assert (
        shared["config_identity"]["source_config_sha256"]
        != shared["config_identity"]["runtime_config_sha256"]
    )
    assert result.ready
    assert json.loads((output / "status.json").read_text())["ready"] is True

    starts_before = len(assignments)
    for variant in Variant:
        worker_dir = output / "workers" / variant.value
        (worker_dir / "complete.marker").unlink()
        (worker_dir / ".pilot.lock").write_text("retained failed attempt")
    run_pilot(
        config,
        output,
        resume=True,
        runner=Runner(),
        gpu_validator=lambda ids: None,
        trust_upstream_artifacts=True,
    )
    relaunched = [
        item
        for item in assignments[starts_before:]
        if item[0][2] == "avgaussianv2.cli.pilot_worker"
    ]
    assert len(relaunched) == 3
    assert all("--resume" not in item[0] for item in relaunched)

    starts_before = len(assignments)
    tree_before = {
        path.relative_to(output): (path.lstat().st_ino, path.lstat().st_mtime_ns)
        for path in output.rglob("*")
    }
    resumed = run_pilot(
        config,
        output,
        resume=True,
        runner=Runner(),
        gpu_validator=lambda ids: None,
        trust_upstream_artifacts=True,
    )
    tree_after = {
        path.relative_to(output): (path.lstat().st_ino, path.lstat().st_mtime_ns)
        for path in output.rglob("*")
    }
    assert len(assignments) == starts_before
    assert tree_after == tree_before
    assert resumed.ready is True

    (output / "status.json").unlink()
    starts_before = len(assignments)
    repaired = run_pilot(
        config,
        output,
        resume=True,
        runner=Runner(),
        gpu_validator=lambda ids: None,
        trust_upstream_artifacts=True,
    )
    repaired_status = json.loads((output / "status.json").read_text())
    assert len(assignments) == starts_before
    assert repaired.ready is True
    assert repaired_status["stages"]["report"] == "complete"
    assert repaired_status["ready"] is True

    monkeypatch.setattr(
        pilot_module,
        "verify_worker_resume_state",
        lambda *args, **kwargs: SimpleNamespace(stage="joint"),
    )
    joint_worker = output / "workers" / "joint_conditioned"
    (joint_worker / "complete.marker").unlink()
    (joint_worker / "latest.pt").write_text("compatible-active")
    starts_before = len(assignments)
    run_pilot(
        config,
        output,
        resume=True,
        runner=Runner(),
        gpu_validator=lambda ids: None,
        trust_upstream_artifacts=True,
    )
    new = assignments[starts_before:]
    assert len(new) == 2
    assert new[0][0][2] == "avgaussianv2.cli.pilot_worker"
    assert "--resume" in new[0][0]
    assert new[1][0][2] == "avgaussianv2.cli.pilot_eval"
    assert "--overwrite" in new[1][0]
    assert "--system" in new[1][0]

    frozen_eval = output / "evaluations" / "frozen_visual"
    (frozen_eval / "evaluation_manifest.json").unlink()
    frozen_system = frozen_eval / "frozen_visual_on"
    frozen_system.mkdir(exist_ok=True)
    (frozen_system / "metrics_summary.json").write_text("partial")
    starts_before = len(assignments)
    run_pilot(
        config,
        output,
        resume=True,
        runner=Runner(),
        gpu_validator=lambda ids: None,
        trust_upstream_artifacts=True,
    )
    new = assignments[starts_before:]
    assert len(new) == 1
    assert new[0][0][2] == "avgaussianv2.cli.pilot_eval"
    assert "--variant" in new[0][0]
    assert "frozen_visual" in new[0][0]
    assert "--resume" in new[0][0]

    starts_before = len(assignments)
    tree_before = {
        path.relative_to(output): (path.lstat().st_ino, path.lstat().st_mtime_ns)
        for path in output.rglob("*")
    }
    verified = run_pilot(
        config,
        output,
        verify_only=True,
        runner=Runner(),
    )
    tree_after = {
        path.relative_to(output): (path.lstat().st_ino, path.lstat().st_mtime_ns)
        for path in output.rglob("*")
    }
    assert len(assignments) == starts_before
    assert tree_after == tree_before
    assert verified.ready is True


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_experiment_and_status_versions_require_exact_integer(version) -> None:
    experiment = {
        "schema": "avgaussianv2.scene1-three-gpu-pilot",
        "version": version,
        "scene_id": "scene1_opera",
        "gpus": [0, 1, 2],
        "config_sha256": "a" * 64,
        "source_config_sha256": "a" * 64,
        "runtime_config_sha256": "a" * 64,
        "shared_manifest_path": "/tmp/shared.json",
        "shared_manifest_sha256": "b" * 64,
        "baseline_manifest_path": "/tmp/baseline.json",
        "baseline_manifest_sha256": "c" * 64,
        "source_hashes": {
            "project_config_sha256": "a" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "visual_checkpoint_sha256": "f" * 64,
            "audio_checkpoint_sha256": "0" * 64,
            "camera_mapping_sha256": "1" * 64,
        },
        "trusted_upstream_artifacts": True,
    }
    with pytest.raises(ValueError, match="version"):
        _validate_experiment_types(experiment)
    status = {
        "schema": "avgaussianv2.scene1-three-gpu-pilot",
        "version": version,
        "stages": {
            "baseline": "complete",
            "workers": "complete",
            "evaluations": "complete",
            "report": "complete",
        },
        "report_digest": "d" * 64,
        "ready": True,
        "durability_warnings": [],
    }
    with pytest.raises(ValueError, match="version"):
        _validate_complete_status(status)


@pytest.mark.parametrize("ready", [0, 1, "true"])
def test_status_ready_requires_exact_boolean(ready) -> None:
    status = {
        "schema": "avgaussianv2.scene1-three-gpu-pilot",
        "version": 1,
        "stages": {
            "baseline": "complete",
            "workers": "complete",
            "evaluations": "complete",
            "report": "complete",
        },
        "report_digest": "d" * 64,
        "ready": ready,
        "durability_warnings": [],
    }
    with pytest.raises(TypeError, match="boolean"):
        _validate_complete_status(status)


def test_resume_rejects_symlinked_run_root_without_touching_target(tmp_path) -> None:
    config = _verify_config(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / ".pilot-orchestrator.lock").write_text("")
    target = tmp_path / "logs-target"
    target.mkdir()
    marker = target / "keep"
    marker.write_text("safe")
    (output / "logs").symlink_to(target, target_is_directory=True)
    (output / "workers").mkdir()
    (output / "evaluations").mkdir()

    class Runner:
        assignments = []

        def start(self, *args, **kwargs):
            pytest.fail("unsafe root launched a process")

    with pytest.raises((OSError, ValueError), match="directory|root"):
        run_pilot(
            config,
            output,
            resume=True,
            runner=Runner(),
            gpu_validator=lambda ids: None,
            trust_upstream_artifacts=True,
        )
    assert marker.read_text() == "safe"

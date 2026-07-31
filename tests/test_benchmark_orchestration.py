from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

import avgaussianv2.benchmark.orchestration as orchestration
from avgaussianv2.benchmark.output import BenchmarkOutputError
from avgaussianv2.benchmark.orchestration import (
    OrchestrationError,
    SceneBenchmarkResult,
    run_benchmark_suite,
    run_scene_benchmark,
)


def _module(command: Sequence[str]) -> str:
    if "-m" not in command:
        return Path(command[1]).stem
    return command[command.index("-m") + 1]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


class _Handle:
    def __init__(
        self,
        command: tuple[str, ...],
        events: list[tuple[object, ...]],
        result: int | None,
    ) -> None:
        self.command = command
        self.events = events
        self.result = result
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        self.events.append(("poll", _module(self.command), self.result))
        return self.result

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(("wait", _module(self.command), timeout))
        if self.result is None:
            raise AssertionError("a live fake process was waited without termination")
        return self.result

    def terminate(self) -> None:
        self.events.append(("terminate", _module(self.command)))
        self.terminated = True
        self.result = -15

    def kill(self) -> None:
        self.events.append(("kill", _module(self.command)))
        self.killed = True
        self.result = -9


class _Runner:
    def __init__(
        self,
        result_for: Callable[[tuple[str, ...]], int | None] | None = None,
        events: list[tuple[object, ...]] | None = None,
    ) -> None:
        self.result_for = result_for or (lambda _command: 0)
        self.events = events if events is not None else []
        self.assignments: list[tuple[tuple[str, ...], Mapping[str, str], Path]] = []
        self.handles: list[_Handle] = []

    def start(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        log_path: Path,
    ) -> _Handle:
        argv = tuple(str(item) for item in command)
        self.assignments.append((argv, dict(env), log_path))
        self.events.append(("start", _module(argv)))
        handle = _Handle(argv, self.events, self.result_for(argv))
        self.handles.append(handle)
        return handle


def _config(repository: Path, scene: str = "scene1_opera") -> Path:
    path = repository / "configs" / "benchmark_cam38" / f"{scene}.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("strict benchmark placeholder\n")
    return path


def _verified(output: Path, *, repository: Path, scene: str) -> SceneBenchmarkResult:
    del repository
    return SceneBenchmarkResult(scene, output, output / "report", True)


def _preflight(*_args, **_kwargs) -> dict[str, object]:
    return {"verified": True}


def _crash_attempt(output: str) -> None:
    with orchestration.BenchmarkOutputLock(output) as pinned:
        attempt = orchestration._AttemptLogs(pinned, "scene1_opera").__enter__()
        (attempt / "worker_audio_only.log").write_text("checkpoint committed\n")
        os.kill(os.getpid(), signal.SIGKILL)


def test_scene_assigns_three_workers_and_evaluations_to_three_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    output = tmp_path / "result"
    runner = _Runner()
    monkeypatch.setattr(orchestration, "_require_native_contracts", lambda *_: None)

    result = run_scene_benchmark(
        config_path=config,
        output_dir=output,
        gpus=(4, 6, 8),
        skip_native_training=True,
        runner=runner,
        _verifier=_verified,
        _preflight_fn=_preflight,
    )

    workers = [
        assignment
        for assignment in runner.assignments
        if _module(assignment[0]) == "avgaussianv2.cli.benchmark_worker"
    ]
    evaluations = [
        assignment
        for assignment in runner.assignments
        if _module(assignment[0]) == "avgaussianv2.cli.benchmark_eval"
    ]
    assert result.verified
    assert len(workers) == 3
    assert [item[1]["CUDA_VISIBLE_DEVICES"] for item in workers] == ["4", "6", "8"]
    assert [item[1]["CUDA_VISIBLE_DEVICES"] for item in evaluations] == [
        str((4, 6, 8)[index % 3]) for index in range(len(evaluations))
    ]
    assert [
        item[0][item[0].index("--manifest") + 1].rsplit("/", 1)[-1] for item in workers
    ] == [
        "joint_conditioned.json",
        "audio_only.json",
        "visual_only.json",
    ]
    assert all(
        item[0][item[0].index("--output-dir") + 1].startswith(
            f"/proc/{os.getpid()}/fd/"
        )
        for item in workers + evaluations
    )
    continuation_evaluations = [
        item
        for item in evaluations
        if item[0][item[0].index("--system") + 1]
        in {"joint_conditioned", "audio_only", "visual_only"}
    ]
    assert all(
        item[0][item[0].index("--source") + 1]
        == str(
            output.absolute()
            / "workers"
            / item[0][item[0].index("--system") + 1]
        )
        for item in continuation_evaluations
    )
    assert all(
        not item[0][item[0].index("--source") + 1].startswith("/proc/")
        for item in continuation_evaluations
    )
    first_evaluation = runner.events.index(("start", "avgaussianv2.cli.benchmark_eval"))
    worker_completions = [
        index
        for index, event in enumerate(runner.events)
        if event == ("poll", "avgaussianv2.cli.benchmark_worker", 0)
    ]
    assert len(worker_completions) == 3
    assert max(worker_completions) < first_evaluation


def test_scene_starts_evaluation_only_after_training_and_supervises_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    output = tmp_path / "result"
    monkeypatch.setattr(orchestration, "_require_native_contracts", lambda *_: None)

    def result_for(command: tuple[str, ...]) -> int | None:
        if _module(command) != "avgaussianv2.cli.benchmark_worker":
            return 0
        manifest = command[command.index("--manifest") + 1]
        return 7 if manifest.endswith("joint_conditioned.json") else None

    runner = _Runner(result_for)
    with pytest.raises(OrchestrationError, match="joint_conditioned=7"):
        run_scene_benchmark(
            config_path=config,
            output_dir=output,
            skip_native_training=True,
            runner=runner,
            _verifier=_verified,
            _preflight_fn=_preflight,
        )

    modules = [_module(item[0]) for item in runner.assignments]
    # With two GPUs only two workers may be active.  A first-wave failure must
    # prevent the queued third worker and every evaluation from starting.
    assert modules.count("avgaussianv2.cli.benchmark_worker") == 2
    assert "avgaussianv2.cli.benchmark_eval" not in modules
    sibling_handles = runner.handles[-2:]
    assert all(handle.terminated for handle in sibling_handles)
    assert all(handle.killed for handle in sibling_handles)
    assert all(handle.result == -9 for handle in sibling_handles)


@pytest.mark.parametrize("mode", ["resume", "verify_only"])
def test_complete_resume_and_verify_only_launch_nothing_and_write_nothing(
    tmp_path: Path, mode: str
) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    output = tmp_path / "result"
    output.mkdir()
    (output / "immutable.json").write_text('{"complete":true}\n')
    runner = _Runner()
    calls: list[tuple[Path, Path, str]] = []

    def verifier(target: Path, *, repository: Path, scene: str) -> SceneBenchmarkResult:
        calls.append((target, repository, scene))
        return SceneBenchmarkResult(scene, target, target / "report", True)

    before = _tree_digest(output)
    result = run_scene_benchmark(
        config_path=config,
        output_dir=output,
        resume=mode == "resume",
        verify_only=mode == "verify_only",
        runner=runner,
        _verifier=verifier,
        _preflight_fn=lambda *_args: pytest.fail("completed run entered preflight"),
    )

    assert result.verified
    assert calls == [(output.absolute(), repository.absolute(), "scene1_opera")]
    assert runner.assignments == []
    assert _tree_digest(output) == before


def test_skip_native_training_requires_strict_native_contract_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    output = tmp_path / "result"
    runner = _Runner()
    checked: list[tuple[Path, str]] = []

    def reject(repo: Path, scene: str) -> None:
        checked.append((repo, scene))
        raise OrchestrationError("native contract is incomplete")

    monkeypatch.setattr(orchestration, "_require_native_contracts", reject)
    with pytest.raises(OrchestrationError, match="native contract is incomplete"):
        run_scene_benchmark(
            config_path=config,
            output_dir=output,
            skip_native_training=True,
            runner=runner,
            _verifier=_verified,
            _preflight_fn=_preflight,
        )

    assert checked == [(repository.absolute(), "scene1_opera")]
    assert runner.assignments == []


def test_partial_resume_rejects_symlink_before_preflight_or_mutation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    output = tmp_path / "result"
    with orchestration.BenchmarkOutputLock(output):
        pass
    target = tmp_path / "target"
    target.write_text("unchanged")
    (output / "unexpected").symlink_to(target)
    before = _tree_digest(output)

    with pytest.raises(OrchestrationError, match="unexpected or unsafe"):
        run_scene_benchmark(
            config_path=config,
            output_dir=output,
            resume=True,
            runner=_Runner(),
            _verifier=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("incomplete")
            ),
            _preflight_fn=lambda *_args: pytest.fail("entered preflight"),
        )

    assert _tree_digest(output) == before


def test_partial_resume_refuses_concurrent_writer_without_launch(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    output = tmp_path / "result"
    runner = _Runner()
    with orchestration.BenchmarkOutputLock(output):
        with pytest.raises(BenchmarkOutputError, match="locked by another process"):
            run_scene_benchmark(
                config_path=config,
                output_dir=output,
                resume=True,
                runner=runner,
                _verifier=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    ValueError("incomplete")
                ),
                _preflight_fn=lambda *_args: pytest.fail("entered preflight"),
            )
    assert runner.assignments == []


def test_partial_snapshot_is_rechecked_before_attempt_log_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    with orchestration.BenchmarkOutputLock(output) as pinned:
        snapshot = orchestration._tree_snapshot_sha256(pinned)
    (output / "changed").write_text("after shared inspection")

    with orchestration.BenchmarkOutputLock(output) as pinned:
        with pytest.raises(OrchestrationError, match="changed before exclusive"):
            with orchestration._AttemptLogs(
                pinned, "scene1_opera", expected_snapshot=snapshot
            ):
                pass

    assert not (output / "logs").exists()


def test_dead_tail_attempt_is_sealed_before_new_resume_attempt(tmp_path: Path) -> None:
    output = tmp_path / "result"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crash_attempt, args=(str(output),))
    process.start()
    process.join(10)
    assert process.exitcode == -signal.SIGKILL

    orchestration._verify_logs(output, "scene1_opera", allow_interrupted_tail=True)
    with orchestration.BenchmarkOutputLock(output) as pinned:
        snapshot = orchestration._tree_snapshot_sha256(pinned)
        with orchestration._AttemptLogs(
            pinned, "scene1_opera", expected_snapshot=snapshot
        ) as attempt:
            assert attempt.name == "attempt-000001"
            interrupted = json.loads(
                (pinned / "logs" / "attempt-000000" / "manifest.json").read_text()
            )
            assert interrupted["state"] == "interrupted"
            assert (attempt / "in_progress.json").is_file()

    orchestration._verify_logs(output, "scene1_opera")


def test_live_tail_attempt_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "result"
    with orchestration.BenchmarkOutputLock(output) as pinned:
        orchestration._AttemptLogs(pinned, "scene1_opera").__enter__()
        with pytest.raises(OrchestrationError, match="owner is still alive"):
            orchestration._verify_logs(
                pinned, "scene1_opera", allow_interrupted_tail=True
            )


def test_multiple_or_tampered_tail_attempts_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "result"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crash_attempt, args=(str(output),))
    process.start()
    process.join(10)
    first = output / "logs" / "attempt-000000"
    second = output / "logs" / "attempt-000001"
    shutil.copytree(first, second)
    with pytest.raises(OrchestrationError, match="uncommitted"):
        orchestration._verify_logs(output, "scene1_opera", allow_interrupted_tail=True)
    shutil.rmtree(second)
    tampered = first / "worker_audio_only.log"
    tampered.unlink()
    tampered.symlink_to(tmp_path / "outside")
    with pytest.raises(OrchestrationError, match="inventory is unsafe"):
        orchestration._verify_logs(output, "scene1_opera", allow_interrupted_tail=True)


@pytest.mark.parametrize("partial_atomic_file", [False, True])
def test_empty_tail_from_mkdir_crash_is_reused_safely(
    tmp_path: Path, partial_atomic_file: bool
) -> None:
    output = tmp_path / "result"
    with orchestration.BenchmarkOutputLock(output) as pinned:
        tail = pinned / "logs" / "attempt-000000"
        tail.mkdir(parents=True)
        if partial_atomic_file:
            (tail / ".in_progress.json.crash").write_text("partial")
    orchestration._verify_logs(output, "scene1_opera", allow_interrupted_tail=True)

    with orchestration.BenchmarkOutputLock(output) as pinned:
        snapshot = orchestration._tree_snapshot_sha256(pinned)
        with orchestration._AttemptLogs(
            pinned, "scene1_opera", expected_snapshot=snapshot
        ) as attempt:
            assert attempt.name == "attempt-000000"
            assert (attempt / "in_progress.json").is_file()

    orchestration._verify_logs(output, "scene1_opera")


def test_stale_current_after_manifest_publish_is_repaired(tmp_path: Path) -> None:
    output = tmp_path / "result"
    with orchestration.BenchmarkOutputLock(output) as pinned:
        with orchestration._AttemptLogs(pinned, "scene1_opera"):
            pass
        old_current = (pinned / "logs" / "current.json").read_bytes()
        with orchestration._AttemptLogs(pinned, "scene1_opera"):
            pass
        (pinned / "logs" / "current.json").write_bytes(old_current)
        (pinned / "logs" / ".current.json.crash").write_text("partial")

    with pytest.raises(OrchestrationError, match="pointer mismatch"):
        orchestration._verify_logs(output, "scene1_opera")
    orchestration._verify_logs(output, "scene1_opera", allow_interrupted_tail=True)
    with orchestration.BenchmarkOutputLock(output) as pinned:
        snapshot = orchestration._tree_snapshot_sha256(pinned)
        with orchestration._AttemptLogs(
            pinned, "scene1_opera", expected_snapshot=snapshot
        ) as attempt:
            assert attempt.name == "attempt-000002"

    orchestration._verify_logs(output, "scene1_opera")


def test_manifest_atomic_temp_is_removed_before_resealing_tail(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crash_attempt, args=(str(output),))
    process.start()
    process.join(10)
    tail = output / "logs" / "attempt-000000"
    (tail / ".manifest.json.crash").write_text("partial")
    orchestration._verify_logs(output, "scene1_opera", allow_interrupted_tail=True)

    with orchestration.BenchmarkOutputLock(output) as pinned:
        snapshot = orchestration._tree_snapshot_sha256(pinned)
        with orchestration._AttemptLogs(
            pinned, "scene1_opera", expected_snapshot=snapshot
        ) as attempt:
            assert attempt.name == "attempt-000001"
            assert not (
                pinned / "logs" / "attempt-000000" / ".manifest.json.crash"
            ).exists()

    orchestration._verify_logs(output, "scene1_opera")


@pytest.mark.parametrize("target", ["manifest", "current"])
@pytest.mark.parametrize("attack", ["symlink", "hardlink", "multiple"])
def test_malicious_atomic_temp_files_fail_closed(
    tmp_path: Path, target: str, attack: str
) -> None:
    output = tmp_path / "result"
    if target == "manifest":
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=_crash_attempt, args=(str(output),))
        process.start()
        process.join(10)
        directory = output / "logs" / "attempt-000000"
        prefix = ".manifest.json."
    else:
        with orchestration.BenchmarkOutputLock(output) as pinned:
            with orchestration._AttemptLogs(pinned, "scene1_opera"):
                pass
            old_current = (pinned / "logs" / "current.json").read_bytes()
            with orchestration._AttemptLogs(pinned, "scene1_opera"):
                pass
            (pinned / "logs" / "current.json").write_bytes(old_current)
        directory = output / "logs"
        prefix = ".current.json."
    outside = tmp_path / "outside"
    outside.write_text("outside")
    temporary = directory / f"{prefix}one"
    if attack == "symlink":
        temporary.symlink_to(outside)
        match = "unsafe"
    elif attack == "hardlink":
        os.link(outside, temporary)
        match = "unsafe"
    else:
        temporary.write_text("one")
        (directory / f"{prefix}two").write_text("two")
        match = "multiple"

    with pytest.raises(OrchestrationError, match=match):
        orchestration._verify_logs(output, "scene1_opera", allow_interrupted_tail=True)


def test_completed_resume_recovers_tail_before_strict_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    output = tmp_path / "result"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crash_attempt, args=(str(output),))
    process.start()
    process.join(10)
    (output / "report").mkdir()
    (output / "report" / "current.json").write_text("{}")

    def snapshot_verify(
        pinned: Path,
        *,
        repository: Path,
        scene: str,
        allow_interrupted_logs: bool = False,
    ) -> None:
        del repository
        orchestration._verify_logs(
            pinned, scene, allow_interrupted_tail=allow_interrupted_logs
        )

    def strict_verify(
        target: Path, *, repository: Path, scene: str
    ) -> SceneBenchmarkResult:
        del repository
        orchestration._verify_logs(target, scene)
        return SceneBenchmarkResult(scene, target, target / "report", True)

    monkeypatch.setattr(orchestration, "_verify_scene_snapshot", snapshot_verify)
    monkeypatch.setattr(orchestration, "verify_scene_outputs", strict_verify)
    result = run_scene_benchmark(
        config_path=config,
        output_dir=output,
        resume=True,
        _verifier=strict_verify,
        _preflight_fn=lambda *_args: pytest.fail("entered preflight"),
    )

    assert result.verified
    assert (
        json.loads((output / "logs" / "attempt-000000" / "manifest.json").read_text())[
            "state"
        ]
        == "interrupted"
    )


def test_completed_verify_only_rejects_tail_without_writes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crash_attempt, args=(str(output),))
    process.start()
    process.join(10)
    before = _tree_digest(output)
    with pytest.raises(OrchestrationError, match="uncommitted"):
        orchestration._verify_logs(output, "scene1_opera")
    assert _tree_digest(output) == before


def test_worker_resume_flag_is_only_emitted_for_committed_modes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    config = _config(repository)
    _, workers, _ = orchestration._scene_commands(
        repository=repository,
        output=tmp_path / "result",
        stable_output=tmp_path / "result",
        log_root=tmp_path / "logs",
        config=config,
        python="python",
        gpus=(0, 1, 2),
        resume_modes=frozenset({"audio_only"}),
    )
    by_mode = {mode: command for mode, command, _, _ in workers}

    assert "--resume" not in by_mode["joint_conditioned"]
    assert "--resume" in by_mode["audio_only"]
    assert "--resume" not in by_mode["visual_only"]


def test_completed_worker_resume_uses_final_evidence_not_rolling_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_worker = tmp_path / "pinned"
    stable_worker = tmp_path / "stable"
    pinned_worker.mkdir()
    stable_worker.mkdir()
    (pinned_worker / "artifact_hashes.json").write_text("{}")
    evidence_calls: list[tuple[Path, str, str, int]] = []
    audit_calls: list[tuple[object, object]] = []

    def evidence(worker: Path, *, scene_id: str, system: str, step: int) -> object:
        evidence_calls.append((worker, scene_id, system, step))
        return ("evidence", step)

    monkeypatch.setattr(orchestration, "continuation_training_evidence", evidence)
    monkeypatch.setattr(
        orchestration,
        "audit_training_evidence",
        lambda item, identity: audit_calls.append((item, identity)),
    )
    monkeypatch.setattr(
        orchestration,
        "verify_resume_artifacts",
        lambda *_args, **_kwargs: pytest.fail(
            "completed output used rolling resume inventory"
        ),
    )

    orchestration._verify_worker_for_resume(
        pinned_worker,
        stable_worker=stable_worker,
        worker_manifest={},
        scene="scene1_opera",
        mode="audio_only",
    )

    assert evidence_calls == [
        (stable_worker, "scene1_opera", "audio_only", step)
        for step in orchestration.REPORTING_STEPS
    ]
    assert [item[0] for item in audit_calls] == [
        ("evidence", step) for step in orchestration.REPORTING_STEPS
    ]


def test_suite_runs_scenes_in_declared_order_then_builds_and_verifies_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    output = tmp_path / "suite"
    events: list[tuple[object, ...]] = []
    runner = _Runner(events=events)
    monkeypatch.setattr(orchestration, "_native_contracts_valid", lambda *_: True)
    monkeypatch.setattr(orchestration, "_require_native_contracts", lambda *_: None)

    def scene_runner(**kwargs) -> SceneBenchmarkResult:
        scene = Path(kwargs["config_path"]).stem
        events.append(("scene", scene))
        assert kwargs["skip_native_training"] is True
        assert kwargs["runner"] is runner
        root = Path(kwargs["output_dir"])
        assert str(root).startswith(f"/proc/{os.getpid()}/fd/")
        assert kwargs["_stable_output_dir"] == output.absolute() / scene
        return SceneBenchmarkResult(scene, root, root / "report", True)

    def suite_verifier(path: Path) -> Mapping[str, object]:
        events.append(("verify-suite", path))
        return {"content_sha256": "a" * 64}

    result = run_benchmark_suite(
        repository=repository,
        output_dir=output,
        skip_native_training=True,
        runner=runner,
        _scene_runner=scene_runner,
        _suite_verifier=suite_verifier,
        _preflight_fn=_preflight,
    )

    assert result["content_sha256"] == "a" * 64
    assert [event for event in events if event[0] == "scene"] == [
        ("scene", "scene1_opera"),
        ("scene", "Scene7playing"),
    ]
    report_start = events.index(("start", "avgaussianv2.cli.benchmark_report"))
    second_scene = events.index(("scene", "Scene7playing"))
    suite_verify = next(
        index for index, event in enumerate(events) if event[0] == "verify-suite"
    )
    assert second_scene < report_start < suite_verify


def test_suite_runs_gpu_runtime_probe_once_and_reuses_nine_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    output = tmp_path / "suite"
    probe_calls = 0
    scene_payloads: list[Mapping[str, object]] = []

    def real_preflight_shape(
        *_args: object,
        _runtime_probes: Mapping[str, Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]:
        nonlocal probe_calls
        if _runtime_probes is None:
            probe_calls += 1
            probes: Mapping[str, Mapping[str, object]] = {
                str(gpu): {
                    runtime: {"verified": True}
                    for runtime in ("avgaussianv2", "ftgspp", "audiogs")
                }
                for gpu in (0, 1, 2)
            }
        else:
            probes = _runtime_probes
        return {"gpu_runtime_probes": probes}

    monkeypatch.setattr(orchestration, "_preflight", real_preflight_shape)
    monkeypatch.setattr(orchestration, "_native_contracts_valid", lambda *_: True)
    monkeypatch.setattr(orchestration, "_require_native_contracts", lambda *_: None)

    def scene_runner(**kwargs: object) -> SceneBenchmarkResult:
        scene_payloads.append(kwargs["_preflight_payload"])  # type: ignore[arg-type]
        assert kwargs["_skip_preflight"] is True
        root = Path(kwargs["output_dir"])  # type: ignore[arg-type]
        return SceneBenchmarkResult(
            Path(kwargs["config_path"]).stem,  # type: ignore[arg-type]
            root,
            root / "report",
            True,
        )

    run_benchmark_suite(
        repository=repository,
        output_dir=output,
        skip_native_training=True,
        runner=_Runner(),
        _scene_runner=scene_runner,
        _suite_verifier=lambda _path: {"content_sha256": "a" * 64},
        _preflight_fn=real_preflight_shape,
    )

    assert probe_calls == 1
    assert len(scene_payloads) == 2
    assert scene_payloads[0]["gpu_runtime_probes"] is scene_payloads[1][
        "gpu_runtime_probes"
    ]


def test_suite_opens_the_explicit_native_execute_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    output = tmp_path / "suite"
    runner = _Runner()
    monkeypatch.setattr(orchestration, "_native_contracts_valid", lambda *_: False)
    run_benchmark_suite(
        repository=repository,
        output_dir=output,
        skip_native_training=False,
        runner=runner,
        _scene_runner=lambda **kwargs: SceneBenchmarkResult(
            Path(kwargs["config_path"]).stem,
            Path(kwargs["output_dir"]),
            Path(kwargs["output_dir"]) / "report",
            True,
        ),
        _suite_verifier=lambda _path: {"content_sha256": "a" * 64},
        _preflight_fn=_preflight,
    )

    native = [command for command, _, _ in runner.assignments if command[0] == "bash"]
    assert len(native) == 6
    assert all("--preflight-only" in command for command in native[:3])
    assert all("--execute" in command for command in native[3:])


def test_suite_native_partial_resume_skips_audio_and_marks_ftgspp_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    output = tmp_path / "suite"
    runner = _Runner()
    checked: list[Path] = []
    monkeypatch.setattr(orchestration, "_native_contracts_valid", lambda *_: False)
    monkeypatch.setattr(
        orchestration,
        "_require_native_partial_resume_state",
        lambda path: checked.append(path),
    )
    monkeypatch.setattr(
        orchestration,
        "_native_contract_valid",
        lambda _repo, _scene, system: system == "native_audiogs",
    )
    run_benchmark_suite(
        repository=repository,
        output_dir=output,
        resume=True,
        runner=runner,
        _scene_runner=lambda **kwargs: SceneBenchmarkResult(
            Path(kwargs["config_path"]).stem,
            Path(kwargs["output_dir"]),
            Path(kwargs["output_dir"]) / "report",
            True,
        ),
        _suite_verifier=lambda _path: {"content_sha256": "a" * 64},
        _preflight_fn=_preflight,
    )

    native = [command for command, _, _ in runner.assignments if command[0] == "bash"]
    assert checked == [repository.absolute()]
    assert len(native) == 4
    assert all("train_audiogs_cam38_baselines.sh" not in command[1] for command in native)
    assert all("--resume" not in command for command in native[:2])
    assert all("--execute" in command and "--resume" in command for command in native[2:])


def test_suite_partial_resume_rejects_extra_before_preflight_or_mutation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    output = tmp_path / "suite"
    with orchestration.BenchmarkOutputLock(output):
        pass
    target = tmp_path / "target"
    target.write_text("unchanged")
    (output / "unexpected").symlink_to(target)
    before = _tree_digest(output)

    with pytest.raises(OrchestrationError, match="unexpected or unsafe"):
        run_benchmark_suite(
            repository=repository,
            output_dir=output,
            resume=True,
            runner=_Runner(),
            _preflight_fn=lambda *_args: pytest.fail("entered preflight"),
        )

    assert _tree_digest(output) == before


def test_suite_partial_resume_passes_stable_scene_path_to_worker_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    output = tmp_path / "suite"
    with orchestration.BenchmarkOutputLock(output) as pinned:
        (pinned / "scene1_opera").mkdir()
    observed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        orchestration,
        "verify_scene_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("partial")),
    )

    def inspect(
        path: Path,
        *,
        stable_output: Path,
        scene: str,
        config_path: Path,
    ) -> tuple[frozenset[str], str]:
        del scene, config_path
        observed.append((path, stable_output))
        return frozenset(), "snapshot"

    monkeypatch.setattr(orchestration, "_inspect_partial_resume", inspect)

    orchestration._inspect_partial_suite(output, repository=repository)

    assert str(observed[0][0]).startswith("/proc/self/fd/")
    assert observed[0][1] == output / "scene1_opera"


def test_failed_prepare_protocol_is_safe_to_retry_only_with_exact_evidence(
    tmp_path: Path,
) -> None:
    config = tmp_path / "scene1_opera.yaml"
    config.write_bytes(b"source config\n")
    output = tmp_path / "scene"
    with orchestration.BenchmarkOutputLock(output) as pinned:
        logs = pinned / "logs"
        with pytest.raises(OrchestrationError, match="prepare failed"):
            with orchestration._AttemptLogs(pinned, "scene1_opera") as attempt:
                (attempt / "prepare.log").write_text(
                    "Traceback\nbenchmark_prepare.py\nprepare_worker_manifests\n"
                )
                raise OrchestrationError("prepare failed")
        protocol = pinned / "protocol"
        protocol.mkdir()
        resolved = protocol / "resolved_project.yaml"
        resolved.write_bytes(b"resolved config\n")
        origin = {
            "schema": "avgaussianv2.cam38-resolved-config-origin",
            "version": 1,
            "source_path": str(config.resolve()),
            "source_sha256": orchestration.sha256_file(config),
            "resolved_sha256": orchestration.sha256_file(resolved),
        }
        (protocol / "resolved_project.origin.json").write_text(json.dumps(origin))
        orchestration._verify_failed_prepare_protocol(
            protocol,
            logs=logs,
            scene="scene1_opera",
            source_config=config,
        )

        origin["resolved_sha256"] = "0" * 64
        (protocol / "resolved_project.origin.json").write_text(json.dumps(origin))
        with pytest.raises(OrchestrationError, match="origin"):
            orchestration._verify_failed_prepare_protocol(
                protocol,
                logs=logs,
                scene="scene1_opera",
                source_config=config,
            )


@pytest.mark.parametrize(
    ("system", "artifact"),
    [
        ("native_audiogs", "step_005000"),
        ("joint_conditioned", "native"),
        ("audio_only", "step_007500"),
    ],
)
def test_partial_evaluation_matrix_rejects_cross_system_or_unreported_artifact(
    tmp_path: Path, system: str, artifact: str
) -> None:
    system_dir = tmp_path / system
    (system_dir / artifact).mkdir(parents=True)

    with pytest.raises(OrchestrationError, match="artifact matrix"):
        orchestration._partial_evaluation_artifacts(system_dir)


def test_partial_evaluation_matrix_rejects_empty_system_directory(
    tmp_path: Path,
) -> None:
    system_dir = tmp_path / "visual_only"
    system_dir.mkdir()

    with pytest.raises(OrchestrationError, match="artifact matrix"):
        orchestration._partial_evaluation_artifacts(system_dir)


def test_suite_verify_only_checks_both_scenes_in_order_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    output = tmp_path / "suite"
    with orchestration.BenchmarkOutputLock(output) as pinned:
        with orchestration._AttemptLogs(pinned, "suite"):
            pass
    for name in (*orchestration.SCENES, "report"):
        (output / name).mkdir()
    runner = _Runner()
    events: list[tuple[object, ...]] = []

    def scene_verifier(
        target: Path, *, repository: Path, scene: str
    ) -> SceneBenchmarkResult:
        events.append(("scene", scene, target, repository))
        return SceneBenchmarkResult(scene, target, target / "report", True)

    def suite_verifier(path: Path) -> Mapping[str, object]:
        events.append(("suite", path))
        return {"content_sha256": "b" * 64}

    monkeypatch.setattr(orchestration, "verify_scene_outputs", scene_verifier)
    before = _tree_digest(output)
    result = run_benchmark_suite(
        repository=repository,
        output_dir=output,
        verify_only=True,
        runner=runner,
        _suite_verifier=suite_verifier,
        _preflight_fn=_preflight,
    )

    assert result["content_sha256"] == "b" * 64
    assert [event[1] for event in events if event[0] == "scene"] == [
        "scene1_opera",
        "Scene7playing",
    ]
    assert events[-1] == ("suite", output.absolute() / "report")
    assert runner.assignments == []
    assert _tree_digest(output) == before

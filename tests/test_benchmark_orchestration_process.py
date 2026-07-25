from __future__ import annotations

import errno
import io
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from avgaussianv2.benchmark import orchestration
from avgaussianv2.benchmark.orchestration import (
    OrchestrationError,
    SubprocessRunner,
    _Handle,
    _run_one,
    _run_parallel,
)


class RecordingHandle:
    def __init__(
        self,
        events: list[str],
        *,
        name: str,
        wait_error: BaseException | None = None,
        poll_error: BaseException | None = None,
        wait_code: int = 0,
        poll_code: int | None = None,
    ) -> None:
        self.events = events
        self.name = name
        self.wait_error = wait_error
        self.poll_error = poll_error
        self.wait_code = wait_code
        self.poll_code = poll_code

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(f"{self.name}.wait({timeout})")
        if self.wait_error is not None:
            error, self.wait_error = self.wait_error, None
            raise error
        if timeout is not None:
            raise subprocess.TimeoutExpired(self.name, timeout)
        return self.wait_code

    def poll(self) -> int | None:
        self.events.append(f"{self.name}.poll")
        if self.poll_error is not None:
            error, self.poll_error = self.poll_error, None
            raise error
        return self.poll_code

    def terminate(self) -> None:
        self.events.append(f"{self.name}.terminate")

    def kill(self) -> None:
        self.events.append(f"{self.name}.kill")


class RecordingRunner:
    def __init__(
        self,
        handles: list[RecordingHandle],
        *,
        start_error_at: int | None = None,
        start_error: BaseException | None = None,
    ) -> None:
        self.handles = handles
        self.start_error_at = start_error_at
        self.start_error = start_error or RuntimeError("start failed")
        self.assignments = []
        self.starts = 0

    def start(self, command, *, env, log_path):
        self.starts += 1
        if self.start_error_at == self.starts:
            raise self.start_error
        return self.handles[self.starts - 1]


def _assert_forced_cleanup(events: list[str], name: str) -> None:
    assert events == [
        f"{name}.terminate",
        f"{name}.wait(10)",
        f"{name}.kill",
        f"{name}.wait(None)",
    ]


@pytest.mark.parametrize("error", [RuntimeError("start failed"), KeyboardInterrupt()])
def test_parallel_second_start_base_exception_cleans_up_first_process(
    tmp_path: Path, error: BaseException
) -> None:
    events: list[str] = []
    runner = RecordingRunner(
        [RecordingHandle(events, name="first")],
        start_error_at=2,
        start_error=error,
    )
    jobs = [
        ("first", ("first",), 0, tmp_path / "first.log"),
        ("second", ("second",), 1, tmp_path / "second.log"),
    ]

    with pytest.raises(type(error)):
        _run_parallel(runner, jobs)

    _assert_forced_cleanup(events, "first")


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(9)])
def test_run_one_wait_base_exception_forces_cleanup(
    tmp_path: Path, error: BaseException
) -> None:
    events: list[str] = []
    handle = RecordingHandle(events, name="one", wait_error=error)
    runner = RecordingRunner([handle])

    with pytest.raises(type(error)):
        _run_one(runner, ("command",), gpu=0, log=tmp_path / "one.log")

    assert events[0] == "one.wait(None)"
    _assert_forced_cleanup(events[1:], "one")


def test_run_one_nonzero_exit_forces_cleanup_before_raise(tmp_path: Path) -> None:
    events: list[str] = []
    handle = RecordingHandle(events, name="one", wait_code=7)
    runner = RecordingRunner([handle])

    with pytest.raises(OrchestrationError, match=r"subprocess failed \(7\)"):
        _run_one(runner, ("command",), gpu=0, log=tmp_path / "one.log")

    assert events[0] == "one.wait(None)"
    _assert_forced_cleanup(events[1:], "one")


@pytest.mark.parametrize(
    ("method", "expected_signal"),
    [("terminate", signal.SIGTERM), ("kill", signal.SIGKILL)],
)
def test_handle_signals_process_group_even_after_leader_exit_and_ignores_esrch(
    monkeypatch: pytest.MonkeyPatch, method: str, expected_signal: signal.Signals
) -> None:
    class ExitedProcess:
        pid = 12345

        @staticmethod
        def poll() -> int:
            return 7

    calls: list[tuple[int, signal.Signals]] = []

    def missing_group(pid: int, selected_signal: signal.Signals) -> None:
        calls.append((pid, selected_signal))
        raise ProcessLookupError(errno.ESRCH, "missing process group")

    monkeypatch.setattr(orchestration.os, "killpg", missing_group)
    handle = _Handle(ExitedProcess(), io.BytesIO())  # type: ignore[arg-type]

    getattr(handle, method)()

    assert calls == [(12345, expected_signal)]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(9)])
def test_parallel_poll_base_exception_forces_cleanup(
    tmp_path: Path, error: BaseException
) -> None:
    events: list[str] = []
    handle = RecordingHandle(events, name="one", poll_error=error)
    runner = RecordingRunner([handle])
    jobs = [("one", ("command",), 0, tmp_path / "one.log")]

    with pytest.raises(type(error)):
        _run_parallel(runner, jobs)

    assert events[0] == "one.poll"
    _assert_forced_cleanup(events[1:], "one")


def test_parallel_failure_cleans_failed_handle_and_sibling(tmp_path: Path) -> None:
    events: list[str] = []
    failed = RecordingHandle(events, name="failed", poll_code=7)
    sibling = RecordingHandle(events, name="sibling")
    runner = RecordingRunner([failed, sibling])
    jobs = [
        ("failed", ("failed",), 0, tmp_path / "failed.log"),
        ("sibling", ("sibling",), 1, tmp_path / "sibling.log"),
    ]

    with pytest.raises(OrchestrationError, match="failed=7"):
        _run_parallel(runner, jobs)

    for name in ("failed", "sibling"):
        assert f"{name}.terminate" in events
        assert f"{name}.wait(10)" in events
        assert f"{name}.kill" in events
        assert f"{name}.wait(None)" in events


def _pid_is_running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def test_run_one_kills_background_child_after_leader_exits_nonzero(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    leader_code = (
        "import pathlib,subprocess,sys;"
        f"child=subprocess.Popen([{sys.executable!r},'-c',{child_code!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        "sys.exit(7)"
    )

    with pytest.raises(OrchestrationError, match=r"subprocess failed \(7\)"):
        _run_one(
            SubprocessRunner(),
            (sys.executable, "-c", leader_code),
            gpu=-1,
            log=tmp_path / "leader.log",
        )

    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 5
    while _pid_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _pid_is_running(child_pid)

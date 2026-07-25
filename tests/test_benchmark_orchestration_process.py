from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from avgaussianv2.benchmark.orchestration import _run_one, _run_parallel


class RecordingHandle:
    def __init__(
        self,
        events: list[str],
        *,
        name: str,
        wait_error: BaseException | None = None,
        poll_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.name = name
        self.wait_error = wait_error
        self.poll_error = poll_error

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(f"{self.name}.wait({timeout})")
        if self.wait_error is not None:
            error, self.wait_error = self.wait_error, None
            raise error
        if timeout is not None:
            raise subprocess.TimeoutExpired(self.name, timeout)
        return 0

    def poll(self) -> int | None:
        self.events.append(f"{self.name}.poll")
        if self.poll_error is not None:
            error, self.poll_error = self.poll_error, None
            raise error
        return None

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

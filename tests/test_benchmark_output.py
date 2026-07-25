from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from avgaussianv2.benchmark.output import (
    BenchmarkOutputError,
    BenchmarkOutputLock,
    BenchmarkOutputReadLock,
    validate_output_children,
)


def _hold_lock(path: str, ready, release) -> None:
    with BenchmarkOutputLock(path):
        ready.set()
        release.wait(10)


def _write_nested(path: str) -> None:
    with BenchmarkOutputLock(Path(path) / "worker") as pinned:
        (pinned / "contract.json").write_text("{}")


def test_output_lock_is_exclusive_across_processes(tmp_path: Path) -> None:
    output = tmp_path / "run"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_lock, args=(str(output), ready, release))
    process.start()
    assert ready.wait(10)
    try:
        with pytest.raises(BenchmarkOutputError, match="locked"):
            with BenchmarkOutputLock(output):
                pass
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0


def test_output_lock_rejects_symlink_root_and_component(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(BenchmarkOutputError, match="symlink"):
        with BenchmarkOutputLock(root_link):
            pass

    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(BenchmarkOutputError, match="symlink"):
        with BenchmarkOutputLock(parent_link / "run"):
            pass


def test_output_lock_detects_rename_while_retaining_original_inode(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    lock = BenchmarkOutputLock(output)
    pinned = lock.__enter__()
    assert pinned.stat().st_ino == output.stat().st_ino
    moved = tmp_path / "moved"
    output.rename(moved)
    output.mkdir()
    with pytest.raises(BenchmarkOutputError, match="changed"):
        lock.__exit__(None, None, None)


def test_lock_records_process_owner_and_retained_identity(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with BenchmarkOutputLock(output) as pinned:
        record = (pinned / ".benchmark.lock").read_text()
        assert f'"pid": {os.getpid()}' in record
        assert f'"uid": {os.getuid()}' in record


def test_output_children_reject_nested_checkpoint_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    with BenchmarkOutputLock(tmp_path / "run") as pinned:
        checkpoints = pinned / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "main_step_000500.pt").symlink_to(outside)
        with pytest.raises(BenchmarkOutputError, match="unsafe"):
            validate_output_children(pinned)


def test_output_lock_rejects_hardlinked_lock_file(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_text("foreign")
    os.link(outside, output / ".benchmark.lock")
    with pytest.raises(BenchmarkOutputError, match="single-link"):
        with BenchmarkOutputLock(output):
            pass


def test_retained_fd_descendant_supports_nested_owned_locks(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with BenchmarkOutputLock(output) as pinned:
        with BenchmarkOutputLock(pinned / "workers" / "audio_only") as worker:
            (worker / "contract.json").write_text("{}")
        with BenchmarkOutputReadLock(pinned / "workers" / "audio_only") as worker:
            assert (worker / "contract.json").read_text() == "{}"


def test_parent_retained_fd_descendant_is_writable_by_child(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    with BenchmarkOutputLock(tmp_path / "run") as pinned:
        parent_pinned = str(pinned).replace("/proc/self/", f"/proc/{os.getpid()}/")
        process = context.Process(target=_write_nested, args=(parent_pinned,))
        process.start()
        process.join(10)
        assert process.exitcode == 0
        assert (pinned / "worker" / "contract.json").read_text() == "{}"

"""Exclusive retained-fd output boundary for benchmark workers."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path


class BenchmarkOutputError(RuntimeError):
    """Raised when the benchmark output boundary is unsafe or changes."""


def _is_proc_fd(path: Path) -> bool:
    parts = path.parts
    return (
        len(parts) == 5
        and parts[0] == "/"
        and parts[1] == "proc"
        and parts[3] == "fd"
        and parts[2] in {"self", str(os.getpid())}
        and parts[4].isdigit()
    )


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise BenchmarkOutputError(
                f"benchmark output contains a symlink component: {current}"
            )


class BenchmarkOutputLock:
    """Pin, own, and exclusively lock one output directory."""

    def __init__(self, path: str | Path) -> None:
        self.original = Path(os.path.abspath(path))
        self.directory_fd: int | None = None
        self.lock_fd: int | None = None
        self.identity: tuple[int, int] | None = None

    def __enter__(self) -> Path:
        if not _is_proc_fd(self.original):
            _reject_symlink_components(self.original.parent)
            try:
                self.original.mkdir(mode=0o700)
            except FileExistsError:
                pass
            _reject_symlink_components(self.original)
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        )
        if not _is_proc_fd(self.original):
            flags |= getattr(os, "O_NOFOLLOW", 0)
        self.directory_fd = os.open(self.original, flags)
        metadata = os.fstat(self.directory_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            self.close()
            raise BenchmarkOutputError("benchmark output must be a directory")
        if metadata.st_uid != os.getuid():
            self.close()
            raise BenchmarkOutputError("benchmark output must be owned by this user")
        self.identity = (metadata.st_dev, metadata.st_ino)

        lock_flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        self.lock_fd = os.open(
            ".benchmark.lock", lock_flags, 0o600, dir_fd=self.directory_fd
        )
        lock_metadata = os.fstat(self.lock_fd)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.getuid()
            or lock_metadata.st_nlink != 1
        ):
            self.close()
            raise BenchmarkOutputError(
                "benchmark lock must be an owned single-link regular file"
            )
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.close()
            raise BenchmarkOutputError(
                "benchmark output is locked by another process"
            ) from error
        record = (
            json.dumps(
                {
                    "pid": os.getpid(),
                    "uid": os.getuid(),
                    "output_dev": metadata.st_dev,
                    "output_ino": metadata.st_ino,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        os.ftruncate(self.lock_fd, 0)
        os.write(self.lock_fd, record)
        os.fsync(self.lock_fd)
        return Path(f"/proc/self/fd/{self.directory_fd}")

    def verify_identity(self) -> None:
        if self.directory_fd is None or self.identity is None:
            raise BenchmarkOutputError("benchmark output lock is not active")
        retained = os.fstat(self.directory_fd)
        if (retained.st_dev, retained.st_ino) != self.identity:
            raise BenchmarkOutputError("retained benchmark output identity changed")
        if not _is_proc_fd(self.original):
            try:
                current = self.original.lstat()
            except FileNotFoundError as error:
                raise BenchmarkOutputError(
                    "benchmark output directory disappeared while locked"
                ) from error
            if (
                stat.S_ISLNK(current.st_mode)
                or (
                    current.st_dev,
                    current.st_ino,
                )
                != self.identity
            ):
                raise BenchmarkOutputError(
                    "benchmark output directory changed while locked"
                )

    def close(self) -> None:
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        if self.directory_fd is not None:
            os.close(self.directory_fd)
            self.directory_fd = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        identity_error: BaseException | None = None
        try:
            self.verify_identity()
        except BaseException as error:
            identity_error = error
        finally:
            self.close()
        if identity_error is not None:
            raise identity_error
        return False


class BenchmarkOutputReadLock:
    """Retain an existing output inode under a non-mutating shared lock."""

    def __init__(self, path: str | Path) -> None:
        self.original = Path(os.path.abspath(path))
        self.directory_fd: int | None = None
        self.lock_fd: int | None = None
        self.identity: tuple[int, int] | None = None

    def __enter__(self) -> Path:
        if not _is_proc_fd(self.original):
            _reject_symlink_components(self.original)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if not _is_proc_fd(self.original):
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self.directory_fd = os.open(self.original, flags)
        except OSError as error:
            raise BenchmarkOutputError(
                "benchmark output does not exist or is unsafe"
            ) from error
        metadata = os.fstat(self.directory_fd)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            self.close()
            raise BenchmarkOutputError("benchmark output must be an owned directory")
        self.identity = (metadata.st_dev, metadata.st_ino)
        try:
            self.lock_fd = os.open(
                ".benchmark.lock",
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self.directory_fd,
            )
        except OSError as error:
            self.close()
            raise BenchmarkOutputError("benchmark output lock is missing") from error
        lock_metadata = os.fstat(self.lock_fd)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.getuid()
            or lock_metadata.st_nlink != 1
        ):
            self.close()
            raise BenchmarkOutputError("benchmark output lock is unsafe")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.close()
            raise BenchmarkOutputError(
                "benchmark output is locked by another process"
            ) from error
        return Path(f"/proc/self/fd/{self.directory_fd}")

    def verify_identity(self) -> None:
        if self.directory_fd is None or self.identity is None:
            raise BenchmarkOutputError("benchmark output read lock is not active")
        retained = os.fstat(self.directory_fd)
        if (retained.st_dev, retained.st_ino) != self.identity:
            raise BenchmarkOutputError("retained benchmark output identity changed")
        if not _is_proc_fd(self.original):
            current = self.original.lstat()
            if stat.S_ISLNK(current.st_mode) or (
                current.st_dev,
                current.st_ino,
            ) != self.identity:
                raise BenchmarkOutputError(
                    "benchmark output directory changed while locked"
                )

    def close(self) -> None:
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        if self.directory_fd is not None:
            os.close(self.directory_fd)
            self.directory_fd = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        identity_error: BaseException | None = None
        try:
            self.verify_identity()
        except BaseException as error:
            identity_error = error
        finally:
            self.close()
        if identity_error is not None:
            raise identity_error
        return False


def validate_output_children(root: Path) -> None:
    """Reject links, foreign owners, and unexpected first-level artifacts."""
    allowed_directories = {"checkpoints", "milestones"}
    allowed_files = {
        ".benchmark.lock",
        "artifact_hashes.json",
        "artifact_journal.json",
        "checkpoint_io.json",
        "contract.json",
        "final.pt",
        "progress.json",
        "runtime_contract.json",
    }
    for child in root.iterdir():
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise BenchmarkOutputError(
                f"benchmark output child must not be a symlink: {child.name}"
            )
        if metadata.st_uid != os.getuid():
            raise BenchmarkOutputError(
                f"benchmark output child has a foreign owner: {child.name}"
            )
        if child.name in allowed_directories:
            if not stat.S_ISDIR(metadata.st_mode):
                raise BenchmarkOutputError(
                    f"benchmark output child must be a directory: {child.name}"
                )
            for artifact in child.iterdir():
                artifact_metadata = artifact.lstat()
                if (
                    stat.S_ISLNK(artifact_metadata.st_mode)
                    or not stat.S_ISREG(artifact_metadata.st_mode)
                    or artifact_metadata.st_uid != os.getuid()
                ):
                    raise BenchmarkOutputError(
                        f"unsafe benchmark artifact: {child.name}/{artifact.name}"
                    )
        elif child.name in allowed_files:
            if not stat.S_ISREG(metadata.st_mode):
                raise BenchmarkOutputError(
                    f"benchmark output child must be a regular file: {child.name}"
                )
        else:
            raise BenchmarkOutputError(
                f"unexpected benchmark output child: {child.name}"
            )


__all__ = [
    "BenchmarkOutputError",
    "BenchmarkOutputLock",
    "BenchmarkOutputReadLock",
    "validate_output_children",
]

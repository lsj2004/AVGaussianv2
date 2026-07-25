"""Three-GPU Scene 1 pilot orchestration.

This parent never constructs a model runtime.  Runtime work is isolated in the
baseline, worker, and final-evaluation subprocesses.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import queue
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from avgaussianv2.cli.pilot_eval import (
    EvaluationJobResult,
    EvaluationSpec,
    _verify_existing,
)
from avgaussianv2.cli.pilot_worker import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    build_worker_component_identities,
    load_worker_manifest,
    pilot_index_hash,
    verify_worker_output,
    verify_worker_resume_state,
)
from avgaussianv2.config import load_project_config_bytes
from avgaussianv2.experiment.checkpoint import (
    PilotCompatibility,
    hash_index_manifest,
    inspect_pilot_checkpoint,
)
from avgaussianv2.experiment.contracts import PilotConfig, SharedIndices, Variant
from avgaussianv2.experiment.report import (
    REQUIRED_SYSTEMS,
    SystemReportInput,
    WorkerArtifactProvenance,
    build_comparison,
    verify_current_comparison,
)


EXPERIMENT_SCHEMA = "avgaussianv2.scene1-three-gpu-pilot"
EXPERIMENT_VERSION = 1
VARIANT_GPU_ORDER = (
    Variant.JOINT_CONDITIONED,
    Variant.FROZEN_VISUAL,
    Variant.CONDITION_OFF,
)
EVAL_SPECS = {
    Variant.JOINT_CONDITIONED: (
        EvaluationSpec("joint_conditioned_on", True),
        EvaluationSpec("joint_conditioned_off", False),
    ),
    Variant.FROZEN_VISUAL: (EvaluationSpec("frozen_visual_on", True),),
    Variant.CONDITION_OFF: (EvaluationSpec("condition_off", False),),
}
_EXPERIMENT_FIELDS = {
    "schema", "version", "scene_id", "gpus", "config_sha256",
    "source_config_sha256", "runtime_config_sha256",
    "shared_manifest_path", "shared_manifest_sha256",
    "baseline_manifest_path", "baseline_manifest_sha256", "source_hashes",
    "trusted_upstream_artifacts",
}
PROCESS_TERMINATION_GRACE_SECONDS = 10.0
PROCESS_REAP_GRACE_SECONDS = 2.0


class PilotProcessError(RuntimeError):
    def __init__(
        self,
        failures: Sequence[tuple[str, int, Path]],
        *,
        outcomes: Sequence[tuple[str, int, Path]] | None = None,
    ) -> None:
        self.failures = tuple(failures)
        self.outcomes = tuple(failures if outcomes is None else outcomes)
        super().__init__(
            "pilot subprocess failures: "
            + "; ".join(
                f"{name}=exit {code} log={log}" for name, code, log in failures
            )
        )


class ProcessHandle(Protocol):
    def wait(self) -> int: ...
    def terminate(self) -> None: ...


class ProcessRunner(Protocol):
    assignments: list[tuple[tuple[str, ...], Mapping[str, str]]]

    def start(
        self, command: Sequence[str], *, env: Mapping[str, str], log_path: Path
    ) -> ProcessHandle: ...


class _PopenHandle:
    def __init__(self, process: subprocess.Popen[bytes], stream: object) -> None:
        self._process = process
        self._stream = stream

        self._closed = False

    def _close_stream(self) -> None:
        if not self._closed:
            self._closed = True
            self._stream.close()

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self._process.wait(timeout=timeout)
        finally:
            if self._process.poll() is not None:
                self._close_stream()

    def poll(self) -> int | None:
        code = self._process.poll()
        if code is not None:
            self._close_stream()
        return code

    def terminate(self) -> None:
        if self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGTERM)

    def kill(self) -> None:
        if self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGKILL)


class SubprocessRunner:
    """Small injectable process boundary; commands are always argv with shell=False."""

    def __init__(self) -> None:
        self.assignments: list[tuple[tuple[str, ...], Mapping[str, str]]] = []

    def start(
        self, command: Sequence[str], *, env: Mapping[str, str], log_path: Path
    ) -> ProcessHandle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Each new attempt keeps the previous bounded log instead of discarding it.
        if log_path.exists():
            previous = log_path.with_suffix(log_path.suffix + ".previous")
            if previous.exists():
                previous.unlink()
            os.replace(log_path, previous)
        stream = open(log_path, "wb")  # noqa: SIM115 - owned by _PopenHandle
        argv = tuple(str(item) for item in command)
        self.assignments.append((argv, dict(env)))
        try:
            process = subprocess.Popen(
                argv,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(env),
                shell=False,
                start_new_session=True,
            )
        except BaseException:
            stream.close()
            raise
        return _PopenHandle(process, stream)


@dataclass(frozen=True)
class PilotOrchestrationResult:
    experiment_manifest: Path
    report_directory: Path
    ready: bool
    durability_warnings: tuple[str, ...]


@dataclass(frozen=True)
class LaunchSpec:
    name: str
    command: tuple[str, ...]
    gpu: int
    log_path: Path


def parse_gpus(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError("--gpus requires exactly three comma-separated nonnegative IDs")
    result = tuple(int(part) for part in parts)
    if len(set(result)) != 3:
        raise ValueError("--gpus IDs must be distinct")
    return result  # type: ignore[return-value]


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _validate_experiment_types(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _EXPERIMENT_FIELDS:
        raise ValueError("experiment manifest fields mismatch")
    if type(value["schema"]) is not str or value["schema"] != EXPERIMENT_SCHEMA:
        raise ValueError("experiment schema mismatch")
    if type(value["version"]) is not int or value["version"] != EXPERIMENT_VERSION:
        raise ValueError("experiment version mismatch")
    if type(value["scene_id"]) is not str or not value["scene_id"]:
        raise TypeError("experiment scene_id must be a string")
    if (
        not isinstance(value["gpus"], list)
        or len(value["gpus"]) != 3
        or any(type(item) is not int or item < 0 for item in value["gpus"])
        or len(set(value["gpus"])) != 3
    ):
        raise TypeError("experiment gpus must be three distinct nonnegative integers")
    if type(value["trusted_upstream_artifacts"]) is not bool:
        raise TypeError("experiment trusted_upstream_artifacts must be boolean")
    for name in (
        "config_sha256", "source_config_sha256", "runtime_config_sha256",
        "shared_manifest_sha256", "baseline_manifest_sha256"
    ):
        digest = value[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"experiment {name} must be a SHA-256 digest")
    for name in ("shared_manifest_path", "baseline_manifest_path"):
        raw_path = value[name]
        if type(raw_path) is not str or not raw_path:
            raise TypeError(f"experiment {name} must be a string")
        path = Path(raw_path)
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise ValueError(f"experiment {name} must be normalized absolute")
    source_fields = {
        "project_config_sha256", "dataset_manifest_sha256",
        "visual_checkpoint_sha256", "audio_checkpoint_sha256",
        "camera_mapping_sha256",
    }
    sources = value["source_hashes"]
    if not isinstance(sources, Mapping) or set(sources) != source_fields:
        raise ValueError("experiment source_hashes fields mismatch")
    for name in source_fields:
        digest = sources[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"experiment source_hashes.{name} is invalid")
    if (
        value["config_sha256"] != value["runtime_config_sha256"]
        or sources["project_config_sha256"] != value["runtime_config_sha256"]
    ):
        raise ValueError("experiment runtime config hashes are inconsistent")
    return value


def _validate_complete_status(value: object) -> Mapping[str, object]:
    expected_fields = {
        "schema", "version", "stages", "report_digest", "ready",
        "durability_warnings",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("status fields mismatch")
    if (
        type(value["schema"]) is not str
        or value["schema"] != EXPERIMENT_SCHEMA
        or type(value["version"]) is not int
        or value["version"] != EXPERIMENT_VERSION
    ):
        raise ValueError("status schema/version mismatch")
    if value["stages"] != {
        "baseline": "complete",
        "workers": "complete",
        "evaluations": "complete",
        "report": "complete",
    }:
        raise ValueError("status stages are incomplete")
    digest = value["report_digest"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("status report_digest is invalid")
    if type(value["ready"]) is not bool:
        raise TypeError("status ready must be boolean")
    if not isinstance(value["durability_warnings"], list) or any(
        type(item) is not str for item in value["durability_warnings"]
    ):
        raise TypeError("status durability_warnings must be strings")
    return value


def _read_regular(path: Path, limit: int = 16 * 1024 * 1024) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"input must be a non-symlink regular file: {path}")
    if metadata.st_size > limit:
        raise ValueError(f"input is too large: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks: list[bytes] = []
        total = 0
        while block := os.read(descriptor, min(1024 * 1024, limit + 1 - total)):
            chunks.append(block)
            total += len(block)
            if total > limit:
                break
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(data) > limit or (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise ValueError(f"input changed while reading: {path}")
    return data


def _sha(path: Path) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source must be a non-symlink regular file: {path}")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def _verify_output_ancestors(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"pilot output parent is unsafe: {current}")


def _open_secure_child_directory(
    output_fd: int,
    output: Path,
    name: str,
    *,
    create: bool,
) -> tuple[int, Path, tuple[int, int]]:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=output_fd)
        except FileExistsError:
            pass
    before = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"pilot run root is not a directory: {name}")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output_fd,
    )
    opened = os.fstat(descriptor)
    identity = (before.st_dev, before.st_ino)
    if (opened.st_dev, opened.st_ino) != identity:
        os.close(descriptor)
        raise ValueError(f"pilot run root changed while opening: {name}")
    return descriptor, output / name, identity


def _verify_directory_identity(
    descriptor: int, path: Path, identity: tuple[int, int]
) -> None:
    opened = os.fstat(descriptor)
    current = path.lstat()
    if (
        not stat.S_ISDIR(current.st_mode)
        or (opened.st_dev, opened.st_ino) != identity
        or (current.st_dev, current.st_ino) != identity
    ):
        raise ValueError(f"pilot run root identity changed: {path}")


def _atomic_json(path: Path, value: object) -> None:
    data = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _canonical_config_bytes(
    original: bytes, config: object
) -> bytes:
    import yaml

    value = yaml.safe_load(original)
    if not isinstance(value, dict) or not isinstance(value.get("paths"), dict):
        raise ValueError("project config must contain a paths mapping")
    for name in (
        "visual_upstream_root",
        "audio_upstream_root",
        "visual_checkpoint",
        "audio_checkpoint",
        "manifest",
    ):
        value["paths"][name] = str(getattr(config.paths, name).resolve(strict=True))
    if config.paths.visual_memmap is not None:
        value["paths"]["visual_memmap"] = str(
            config.paths.visual_memmap.resolve(strict=True)
        )
    return yaml.safe_dump(value, sort_keys=True).encode("utf-8")


def _open_config_snapshot(
    output_fd: int,
    output: Path,
    content: bytes,
    *,
    create: bool,
) -> tuple[int, int, Path, tuple[int, int, int, int]]:
    inputs_fd, inputs_path, _ = _open_secure_child_directory(
        output_fd, output, ".orchestrator-inputs", create=create
    )
    name = "project_config.yaml"
    if not create:
        existing = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=inputs_fd,
        )
        metadata = os.fstat(existing)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or os.read(existing, len(content) + 1) != content
        ):
            os.close(existing)
            os.close(inputs_fd)
            raise ValueError("pinned project config snapshot mismatch")
        os.lseek(existing, 0, os.SEEK_SET)
        config_fd = existing
    else:
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
                dir_fd=inputs_fd,
            )
        except FileExistsError:
            existing = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=inputs_fd,
            )
            metadata = os.fstat(existing)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or os.read(existing, len(content) + 1) != content
            ):
                os.close(existing)
                os.close(inputs_fd)
                raise ValueError("pinned project config snapshot mismatch")
            os.lseek(existing, 0, os.SEEK_SET)
            config_fd = existing
        else:
            try:
                remaining = memoryview(content)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("short write while pinning project config")
                    remaining = remaining[written:]
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
            finally:
                os.close(descriptor)
            os.fsync(inputs_fd)
            config_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=inputs_fd,
            )
    metadata = os.fstat(config_fd)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
    ):
        os.close(config_fd)
        os.close(inputs_fd)
        raise ValueError("pinned project config must be single-link regular")
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    child_path = Path(
        f"/proc/{os.getpid()}/fd/{inputs_fd}/{name}"
    )
    return inputs_fd, config_fd, child_path, identity


def _stage_status(
    output: Path,
    *,
    baseline: str,
    workers: str,
    evaluations: str,
    report: str,
    report_digest: str | None = None,
    ready: bool | None = None,
    durability_warnings: Sequence[str] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": EXPERIMENT_SCHEMA,
        "version": EXPERIMENT_VERSION,
        "stages": {
            "baseline": baseline,
            "workers": workers,
            "evaluations": evaluations,
            "report": report,
        },
    }
    if report_digest is not None:
        payload["report_digest"] = report_digest
    if ready is not None:
        payload["ready"] = ready
    if durability_warnings is not None:
        payload["durability_warnings"] = list(durability_warnings)
    _atomic_json(output / "status.json", payload)


def _evenly_spaced(length: int, count: int) -> tuple[int, ...]:
    count = min(length, count)
    if count <= 0:
        raise ValueError("evaluation split must be nonempty")
    if count == 1:
        return (0,)
    return tuple((position * (length - 1)) // (count - 1) for position in range(count))


def _shared_indices(seed: int, train_length: int, pilot: PilotConfig) -> SharedIndices:
    if train_length <= 0:
        raise ValueError("training split must be nonempty")
    rng = random.Random(seed)
    return SharedIndices(
        warmup=tuple(rng.randrange(train_length) for _ in range(pilot.warmup_steps)),
        joint=tuple(rng.randrange(train_length) for _ in range(pilot.joint_steps)),
    )


def _wait_group(
    jobs: Sequence[tuple[str, ProcessHandle, Path]],
    *,
    stop_immediately: bool = False,
) -> None:
    if not jobs:
        return
    completed: queue.Queue[tuple[int, int]] = queue.Queue()
    running = set(range(len(jobs)))
    outcomes: dict[int, int] = {}

    def waiter(position: int, handle: ProcessHandle) -> None:
        try:
            code = handle.wait()
        except BaseException:
            code = -1
        completed.put((position, code))

    for position, (_, handle, _) in enumerate(jobs):
        threading.Thread(
            target=waiter,
            args=(position, handle),
            name=f"pilot-wait-{position}",
            daemon=True,
        ).start()

    def drain_until(deadline: float) -> None:
        while running and time.monotonic() < deadline:
            timeout = max(0.0, min(0.05, deadline - time.monotonic()))
            try:
                position, code = completed.get(timeout=timeout)
            except queue.Empty:
                continue
            if position in running:
                running.remove(position)
                outcomes[position] = code

    def stop_all() -> None:
        for position in tuple(running):
            try:
                jobs[position][1].terminate()
            except BaseException:
                pass
        drain_until(time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS)
        for position in tuple(running):
            handle = jobs[position][1]
            try:
                kill = getattr(handle, "kill", None)
                if callable(kill):
                    kill()
                else:
                    handle.terminate()
            except BaseException:
                pass
        drain_until(time.monotonic() + PROCESS_REAP_GRACE_SECONDS)
        for position in tuple(running):
            running.remove(position)
            outcomes[position] = -9

    try:
        first_failure = False
        if stop_immediately:
            stop_all()
        else:
            while running:
                position, code = completed.get()
                if position not in running:
                    continue
                running.remove(position)
                outcomes[position] = code
                if code != 0:
                    first_failure = True
                    break
            if first_failure:
                stop_all()
    except (KeyboardInterrupt, SystemExit):
        stop_all()
        raise
    all_outcomes = [
        (name, outcomes.get(position, -9), log)
        for position, (name, _, log) in enumerate(jobs)
    ]
    failures = [item for item in all_outcomes if item[1] != 0]
    if failures:
        raise PilotProcessError(failures, outcomes=all_outcomes)


def _launch_group(
    runner: ProcessRunner, specs: Sequence[LaunchSpec]
) -> None:
    jobs: list[tuple[str, ProcessHandle, Path]] = []
    try:
        for spec in specs:
            handle = runner.start(
                spec.command,
                env=_command_env(spec.gpu),
                log_path=spec.log_path,
            )
            jobs.append((spec.name, handle, spec.log_path))
    except (KeyboardInterrupt, SystemExit):
        for _, handle, _ in jobs:
            try:
                handle.terminate()
            except BaseException:
                pass
        try:
            _wait_group(jobs, stop_immediately=True)
        except PilotProcessError:
            pass
        raise
    except BaseException:
        failed_spec = specs[len(jobs)]
        try:
            _wait_group(
                [
                    *jobs,
                    (
                        failed_spec.name,
                        _FailedStartHandle(),
                        failed_spec.log_path,
                    ),
                ],
                stop_immediately=True,
            )
        except PilotProcessError as error:
            raise error
    _wait_group(jobs)


class _FailedStartHandle:
    def wait(self) -> int:
        return -1

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


def _command_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["AVGAUSSIANV2_ORCHESTRATOR_PID"] = str(os.getpid())
    return env


def _source_hashes(config_bytes: bytes, config: object) -> dict[str, str]:
    return {
        "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_manifest_sha256": _sha(config.paths.manifest),
        "visual_checkpoint_sha256": _sha(config.paths.visual_checkpoint),
        "audio_checkpoint_sha256": _sha(config.paths.audio_checkpoint),
        "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
    }


def _worker_manifest(
    *,
    config: object,
    config_bytes: bytes,
    source_config_sha256: str,
    pilot: PilotConfig,
    shared: SharedIndices,
    heldout: tuple[int, ...],
    baseline_path: Path,
    baseline_summary: Mapping[str, object],
    baseline_job: Mapping[str, object],
    train_length: int,
    eval_length: int,
) -> dict[str, object]:
    hashes = _source_hashes(config_bytes, config)
    compatibilities = {}
    for variant in Variant:
        compatibilities[variant.value] = PilotCompatibility(
            scene_id=config.scene.scene_id,
            variant=variant.value,
            seed=config.train.seed,
            index_hash=pilot_index_hash(shared, heldout, variant),
            visual_checkpoint_sha256=hashes["visual_checkpoint_sha256"],
            audio_checkpoint_sha256=hashes["audio_checkpoint_sha256"],
            camera_mapping_sha256=hashes["camera_mapping_sha256"],
            n_fft=config.model.n_fft,
            hop_length=config.model.hop_length,
            win_length=config.model.win_length,
            sample_rate=config.model.sample_rate,
        ).to_mapping()
    runtime = baseline_job["runtime_identity"]
    return {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "scene_id": config.scene.scene_id,
        "seed": config.train.seed,
        "pilot_config": asdict(pilot),
        "shared_indices": asdict(shared),
        "quick_heldout_indices": list(heldout),
        "config_identity": {
            "source_config_sha256": source_config_sha256,
            "runtime_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "source_hashes": hashes,
        "compatibility": compatibilities,
        "component_identities": build_worker_component_identities(
            runtime["model_class"]
        ),
        "runtime_identity": runtime,
        "dataset_lengths": {"train": train_length, "eval": eval_length},
        "visual_baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": _sha(baseline_path),
            "summary": baseline_summary,
        },
    }


def _run_one(
    runner: ProcessRunner,
    name: str,
    command: Sequence[str],
    gpu: int,
    log: Path,
) -> None:
    handle = runner.start(command, env=_command_env(gpu), log_path=log)
    _wait_group(((name, handle, log),))


def _worker_provenance(
    worker_dir: Path, evaluation: EvaluationJobResult
) -> WorkerArtifactProvenance:
    checkpoint = evaluation.artifacts[0].checkpoint
    latest = inspect_pilot_checkpoint(
        worker_dir / "latest.pt",
        expected_compatibility=checkpoint.compatibility,
        indices=checkpoint.variant_indices,
        expected_run_fingerprint=checkpoint.run_fingerprint,
        active_resume=False,
        allow_complete=True,
    )
    return WorkerArtifactProvenance(
        worker_summary_path=(worker_dir / "worker_summary.json").resolve(),
        worker_summary_sha256=_sha(worker_dir / "worker_summary.json"),
        latest_checkpoint_path=(worker_dir / "latest.pt").resolve(),
        latest_checkpoint_sha256=_sha(worker_dir / "latest.pt"),
        latest_checkpoint_generation=latest.generation,
        run_fingerprint=latest.run_fingerprint,
    )


def _report_inputs(
    output: Path,
    baseline: EvaluationJobResult,
    eval_jobs: Mapping[Variant, EvaluationJobResult],
    verified_workers: Mapping[Variant, object],
) -> list[SystemReportInput]:
    inputs: list[SystemReportInput] = [
        SystemReportInput(
            "baseline_imported",
            baseline.evaluations[0],
            None,
            baseline.artifacts[0],
            None,
        )
    ]
    for variant in VARIANT_GPU_ORDER:
        job = eval_jobs[variant]
        worker_dir = output / "workers" / variant.value
        summary = verified_workers[variant].summary
        worker_provenance = _worker_provenance(worker_dir, job)
        for evaluation, artifact in zip(job.evaluations, job.artifacts, strict=True):
            inputs.append(
                SystemReportInput(
                    evaluation.system_name,
                    evaluation,
                    summary,
                    artifact,
                    worker_provenance,
                )
            )
    if {item.name for item in inputs} != set(REQUIRED_SYSTEMS):
        raise ValueError("final report requires exactly five systems")
    return inputs


def _build_report(
    output: Path,
    baseline: EvaluationJobResult,
    eval_jobs: Mapping[Variant, EvaluationJobResult],
    verified_workers: Mapping[Variant, object],
) -> object:
    return build_comparison(
        _report_inputs(output, baseline, eval_jobs, verified_workers),
        output / "report",
    )


def run_pilot(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    gpus: tuple[int, int, int] = (0, 1, 2),
    resume: bool = False,
    verify_only: bool = False,
    trust_upstream_artifacts: bool = False,
    runner: ProcessRunner | None = None,
    gpu_validator: Callable[[tuple[int, int, int]], None] | None = None,
    python: str = sys.executable,
) -> PilotOrchestrationResult:
    """Orchestrate baseline -> three workers -> three final evaluation jobs."""
    if parse_gpus(",".join(map(str, gpus))) != gpus:
        raise ValueError("invalid GPU assignment")
    production_runner = runner is None
    if production_runner and not verify_only and not trust_upstream_artifacts:
        raise PermissionError(
            "production pilot requires --trust-upstream-artifacts"
        )
    requested_config = Path(config_path)
    config_bytes = _read_regular(requested_config)
    source_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config_source = requested_config.resolve(strict=True)
    config = load_project_config_bytes(config_bytes, base_dir=config_source.parent)
    if config.scene.scene_id != "scene1_opera":
        raise ValueError("three-GPU pilot requires exact scene_id scene1_opera")
    output = Path(output_dir).absolute()
    _verify_output_ancestors(output)
    if output.exists():
        metadata = output.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("pilot output must be a non-symlink directory")
        entries = [
            item
            for item in output.iterdir()
            if item.name != ".pilot-orchestrator.lock"
        ]
        if entries and not (resume or verify_only):
            raise FileExistsError("fresh pilot refuses nonempty output")
    else:
        if verify_only:
            raise FileNotFoundError("verify-only requires an existing output")
        output.mkdir(parents=True)
    effective_trust = trust_upstream_artifacts
    existing_experiment_path = output / "experiment_manifest.json"
    if verify_only and existing_experiment_path.is_file():
        existing_experiment = _validate_experiment_types(json.loads(
            _read_regular(existing_experiment_path).decode()
        ))
        recorded_trust = existing_experiment["trusted_upstream_artifacts"]
        effective_trust = recorded_trust
    if runner is None:
        runner = SubprocessRunner()
    if gpu_validator is None and not verify_only:
        import torch

        def gpu_validator(ids: tuple[int, int, int]) -> None:
            if not torch.cuda.is_available() or max(ids) >= torch.cuda.device_count():
                raise ValueError("requested CUDA GPU is unavailable")
    if not verify_only:
        gpu_validator(gpus)

    output_fd = os.open(
        output,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    canonical_config = _canonical_config_bytes(config_bytes, config)
    try:
        (
            inputs_fd,
            pinned_config_fd,
            config_source,
            pinned_config_identity,
        ) = _open_config_snapshot(
            output_fd,
            output,
            canonical_config,
            create=not verify_only,
        )
    except BaseException:
        os.close(output_fd)
        raise
    config_bytes = canonical_config
    try:
        lock_fd = os.open(
            ".pilot-orchestrator.lock",
            os.O_RDWR
            | (0 if verify_only else os.O_CREAT)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            0o600,
            dir_fd=output_fd,
        )
    except BaseException:
        os.close(pinned_config_fd)
        os.close(inputs_fd)
        os.close(output_fd)
        raise
    child_directory_fds: list[tuple[int, Path, tuple[int, int]]] = []
    try:
        lock_metadata = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_nlink != 1
        ):
            raise ValueError(
                "pilot orchestrator lock must be a single-link regular file"
            )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("pilot output already has an active parent") from error
        experiment_path = output / "experiment_manifest.json"
        shared_path = output / "shared_manifest.json"
        for name in ("logs", "workers", "evaluations"):
            try:
                child_directory_fds.append(
                    _open_secure_child_directory(
                        output_fd,
                        output,
                        name,
                        create=not verify_only,
                    )
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"verify-only requires existing directory: {output / name}"
                ) from None
        logs = child_directory_fds[0][1]
        workers_root = child_directory_fds[1][1]
        eval_root = child_directory_fds[2][1]
        logs = Path(f"/proc/self/fd/{child_directory_fds[0][0]}")
        baseline_entry = _open_secure_child_directory(
            output_fd,
            output,
            "baseline",
            create=not verify_only,
        )
        child_directory_fds.append(baseline_entry)
        baseline_dir = Path(f"/proc/self/fd/{baseline_entry[0]}")
        baseline_child_dir = Path(
            f"/proc/{os.getpid()}/fd/{baseline_entry[0]}"
        )
        worker_dirs: dict[Variant, tuple[Path, Path]] = {}
        evaluation_dirs: dict[Variant, tuple[Path, Path]] = {}
        workers_fd = child_directory_fds[1][0]
        evaluations_fd = child_directory_fds[2][0]
        for variant in VARIANT_GPU_ORDER:
            worker_entry = _open_secure_child_directory(
                workers_fd,
                workers_root,
                variant.value,
                create=not verify_only,
            )
            child_directory_fds.append(worker_entry)
            worker_dirs[variant] = (
                Path(f"/proc/self/fd/{worker_entry[0]}"),
                Path(f"/proc/{os.getpid()}/fd/{worker_entry[0]}"),
            )
            evaluation_entry = _open_secure_child_directory(
                evaluations_fd,
                eval_root,
                variant.value,
                create=not verify_only,
            )
            child_directory_fds.append(evaluation_entry)
            evaluation_dirs[variant] = (
                Path(f"/proc/self/fd/{evaluation_entry[0]}"),
                Path(f"/proc/{os.getpid()}/fd/{evaluation_entry[0]}"),
            )

        baseline_specs = (EvaluationSpec("baseline_imported", False),)
        baseline_manifest = baseline_dir / "evaluation_manifest.json"
        baseline_launched = False
        if not baseline_manifest.exists():
            if verify_only:
                raise FileNotFoundError("baseline evaluation is incomplete")
            command = [
                python, "-m", "avgaussianv2.cli.pilot_eval",
                "--config", str(config_source),
                "--system", "baseline_imported:off",
                "--output-dir", str(baseline_child_dir),
                "--device", "cuda:0",
            ]
            if trust_upstream_artifacts:
                command.append("--trust-upstream-artifacts")
            if resume and baseline_dir.exists():
                command.append("--resume")
            if resume:
                _stage_status(
                    output,
                    baseline="mutating",
                    workers="mutating",
                    evaluations="mutating",
                    report="mutating",
                    ready=False,
                )
            _run_one(runner, "baseline", command, gpus[0], logs / "baseline.log")
            baseline_launched = True
            for descriptor, path, identity in child_directory_fds:
                _verify_directory_identity(descriptor, path, identity)
        baseline_job = _verify_existing(
            baseline_dir,
            baseline_specs,
            config_path=config_source,
        )
        if not verify_only and not resume:
            _stage_status(
                output,
                baseline="complete",
                workers="pending",
                evaluations="pending",
                report="pending",
            )
        baseline_raw = json.loads(_read_regular(baseline_manifest).decode())
        baseline_summary_path = (
            baseline_dir / "baseline_imported" / "metrics_summary.json"
        )
        visual_baseline = output / "visual_baseline.json"
        if not visual_baseline.exists():
            if verify_only:
                raise FileNotFoundError("visual baseline binding is missing")
            _atomic_bytes(visual_baseline, _read_regular(baseline_summary_path))
        elif _read_regular(visual_baseline) != _read_regular(baseline_summary_path):
            raise ValueError("visual baseline binding is corrupt")

        pilot = PilotConfig()
        if (
            type(baseline_raw.get("train_length")) is not int
            or baseline_raw["train_length"] <= 0
            or type(baseline_raw.get("eval_length")) is not int
            or baseline_raw["eval_length"] <= 0
        ):
            raise TypeError("baseline dataset lengths must be positive integers")
        train_length = baseline_raw["train_length"]
        eval_length = baseline_raw["eval_length"]
        shared = _shared_indices(config.train.seed, train_length, pilot)
        heldout = _evenly_spaced(eval_length, pilot.quick_validation_samples)
        desired_shared = json.loads(
            json.dumps(
                _worker_manifest(
                    config=config,
                    config_bytes=config_bytes,
                    source_config_sha256=source_config_sha256,
                    pilot=pilot,
                    shared=shared,
                    heldout=heldout,
                    baseline_path=visual_baseline,
                    baseline_summary=json.loads(
                        _read_regular(visual_baseline).decode()
                    ),
                    baseline_job=baseline_raw,
                    train_length=train_length,
                    eval_length=eval_length,
                ),
                sort_keys=True,
                allow_nan=False,
            )
        )
        if shared_path.exists():
            existing = json.loads(_read_regular(shared_path).decode())
            if not _strict_json_equal(existing, desired_shared):
                raise ValueError("shared manifest/config/source identity mismatch")
        elif verify_only:
            raise FileNotFoundError("shared worker manifest is missing")
        else:
            _atomic_json(shared_path, desired_shared)
        # Exercise the exact worker verifier before dispatch.
        loaded = load_worker_manifest(
            shared_path, config_path=config_source, config=config
        )

        experiment = {
            "schema": EXPERIMENT_SCHEMA,
            "version": EXPERIMENT_VERSION,
            "scene_id": config.scene.scene_id,
            "gpus": list(gpus),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_config_sha256": source_config_sha256,
            "runtime_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "shared_manifest_path": str(shared_path.resolve()),
            "shared_manifest_sha256": loaded.sha256,
            "baseline_manifest_path": str(baseline_manifest.resolve()),
            "baseline_manifest_sha256": _sha(baseline_manifest),
            "source_hashes": desired_shared["source_hashes"],
            "trusted_upstream_artifacts": effective_trust,
        }
        if experiment_path.exists():
            existing = _validate_experiment_types(
                json.loads(_read_regular(experiment_path).decode())
            )
            if not _strict_json_equal(existing, experiment):
                raise ValueError("experiment manifest identity mismatch")
        elif verify_only:
            raise FileNotFoundError("experiment manifest is missing")
        else:
            _atomic_json(experiment_path, experiment)

        worker_specs: list[LaunchSpec] = []
        verified_workers: dict[Variant, object] = {}
        resumed_variants: set[Variant] = set()
        for variant, gpu in zip(VARIANT_GPU_ORDER, gpus, strict=True):
            worker_dir, worker_child_dir = worker_dirs[variant]
            has_entries = worker_dir.is_dir() and any(
                entry.name != ".pilot.lock" for entry in worker_dir.iterdir()
            )
            if has_entries:
                try:
                    verified_workers[variant] = verify_worker_output(
                        config_source,
                        shared_path,
                        visual_baseline,
                        worker_dir,
                        variant,
                        trust_upstream_artifacts=effective_trust,
                    )
                    continue
                except Exception:
                    if verify_only or not resume:
                        raise
                    if not (worker_dir / "latest.pt").is_file():
                        raise
                    verify_worker_resume_state(
                        config_source,
                        shared_path,
                        visual_baseline,
                        worker_dir,
                        variant,
                        trust_upstream_artifacts=effective_trust,
                    )
            elif verify_only:
                raise FileNotFoundError(f"{variant.value} worker is incomplete")
            command = [
                python, "-m", "avgaussianv2.cli.pilot_worker",
                "--config", str(config_source),
                "--variant", variant.value,
                "--shared-indices", str(shared_path),
                "--visual-baseline", str(visual_baseline),
                "--output-dir", str(worker_child_dir),
                "--device", "cuda:0",
            ]
            if resume and has_entries:
                command.append("--resume")
                resumed_variants.add(variant)
            if trust_upstream_artifacts:
                command.append("--trust-upstream-artifacts")
            log = logs / f"worker-{variant.value}.log"
            worker_specs.append(
                LaunchSpec(variant.value, tuple(command), gpu, log)
            )
        if worker_specs:
            if resume:
                _stage_status(
                    output,
                    baseline="complete",
                    workers="mutating",
                    evaluations="mutating",
                    report="mutating",
                    ready=False,
                )
            _launch_group(runner, worker_specs)
            for descriptor, path, identity in child_directory_fds:
                _verify_directory_identity(descriptor, path, identity)
        for variant in VARIANT_GPU_ORDER:
            verified_workers[variant] = verify_worker_output(
                config_source,
                shared_path,
                visual_baseline,
                worker_dirs[variant][0],
                variant,
                trust_upstream_artifacts=effective_trust,
            )
        if not verify_only and not resume:
            _stage_status(
                output,
                baseline="complete",
                workers="complete",
                evaluations="pending",
                report="pending",
            )

        eval_jobs: dict[Variant, EvaluationJobResult] = {}
        pending: list[LaunchSpec] = []
        for variant, gpu in zip(VARIANT_GPU_ORDER, gpus, strict=True):
            destination, child_destination = evaluation_dirs[variant]
            specs = EVAL_SPECS[variant]
            replace_stale = variant in resumed_variants
            if (
                (destination / "evaluation_manifest.json").is_file()
                and not replace_stale
            ):
                eval_jobs[variant] = _verify_existing(
                    destination,
                    specs,
                    config_path=config_source,
                    shared_manifest=shared_path,
                    variant=variant,
                )
                continue
            if verify_only:
                raise FileNotFoundError(f"{variant.value} final evaluation is incomplete")
            command = [
                python, "-m", "avgaussianv2.cli.pilot_eval",
                "--config", str(config_source),
                "--checkpoint", str(worker_dirs[variant][1] / "best.pt"),
                "--manifest", str(shared_path),
                "--variant", variant.value,
                "--output-dir", str(child_destination),
                "--device", "cuda:0",
            ]
            for spec in specs:
                command.extend(
                    ["--system", f"{spec.system_name}:{'on' if spec.condition_enabled else 'off'}"]
                )
            if trust_upstream_artifacts:
                command.append("--trust-upstream-artifacts")
            if replace_stale:
                command.append("--overwrite")
            elif resume and destination.exists():
                command.append("--resume")
            log = logs / f"eval-{variant.value}.log"
            pending.append(
                LaunchSpec(variant.value, tuple(command), gpu, log)
            )
        if pending:
            if resume:
                _stage_status(
                    output,
                    baseline="complete",
                    workers="complete",
                    evaluations="mutating",
                    report="mutating",
                    ready=False,
                )
            _launch_group(runner, pending)
            for descriptor, path, identity in child_directory_fds:
                _verify_directory_identity(descriptor, path, identity)
            for variant in VARIANT_GPU_ORDER:
                eval_jobs[variant] = _verify_existing(
                    evaluation_dirs[variant][0],
                    EVAL_SPECS[variant],
                    config_path=config_source,
                    shared_manifest=shared_path,
                    variant=variant,
                )
        if not verify_only and not resume:
            _stage_status(
                output,
                baseline="complete",
                workers="complete",
                evaluations="complete",
                report="pending",
            )

        if _read_regular(config_source) != config_bytes:
            raise ValueError("project config changed during orchestration")
        current_config = os.fstat(pinned_config_fd)
        if (
            current_config.st_dev,
            current_config.st_ino,
            current_config.st_size,
            current_config.st_mtime_ns,
        ) != pinned_config_identity:
            raise ValueError("pinned project config identity changed")
        if _source_hashes(config_bytes, config) != desired_shared["source_hashes"]:
            raise ValueError("upstream source changed during orchestration")
        if load_worker_manifest(
            shared_path, config_path=config_source, config=config
        ).sha256 != loaded.sha256:
            raise ValueError("shared manifest changed during orchestration")
        for descriptor, path, identity in child_directory_fds:
            _verify_directory_identity(descriptor, path, identity)

        report_inputs = _report_inputs(
            output, baseline_job, eval_jobs, verified_workers
        )
        if verify_only:
            comparison = verify_current_comparison(
                report_inputs,
                output / "report",
            )
            status_path = output / "status.json"
            status = _validate_complete_status(
                json.loads(_read_regular(status_path).decode())
            )
            if status.get("report_digest") != comparison.content_digest:
                raise ValueError("status/report digest mismatch")
            ready = comparison.decision.ready
            if status["ready"] is not ready:
                raise ValueError("status/report decision mismatch")
            if status["durability_warnings"] != list(
                comparison.durability_warnings
            ):
                raise ValueError("status/report durability warnings mismatch")
            return PilotOrchestrationResult(
                experiment_path.resolve(), comparison.generation_path, ready,
                comparison.durability_warnings,
            )
        if (
            resume
            and not baseline_launched
            and not worker_specs
            and not pending
        ):
            complete_status = False
            status: Mapping[str, object] | None = None
            try:
                status = _validate_complete_status(
                    json.loads(_read_regular(output / "status.json").decode())
                )
                complete_status = True
            except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
                complete_status = False
            try:
                comparison = verify_current_comparison(
                    report_inputs, output / "report"
                )
            except (FileNotFoundError, TypeError, ValueError):
                if complete_status:
                    raise
                comparison = _build_report(
                    output, baseline_job, eval_jobs, verified_workers
                )
            status_valid = False
            if complete_status and status is not None:
                status_valid = (
                    status["report_digest"] == comparison.content_digest
                    and status["ready"] is comparison.decision.ready
                    and status["durability_warnings"]
                    == list(comparison.durability_warnings)
                )
                if not status_valid:
                    raise ValueError("completed resume status/report mismatch")
            if not status_valid:
                _stage_status(
                    output,
                    baseline="complete",
                    workers="complete",
                    evaluations="complete",
                    report="complete",
                    report_digest=comparison.content_digest,
                    ready=comparison.decision.ready,
                    durability_warnings=comparison.durability_warnings,
                )
            return PilotOrchestrationResult(
                experiment_path.resolve(),
                comparison.generation_path,
                comparison.decision.ready,
                comparison.durability_warnings,
            )
        comparison = _build_report(
            output, baseline_job, eval_jobs, verified_workers
        )
        _stage_status(
            output,
            baseline="complete",
            workers="complete",
            evaluations="complete",
            report="complete",
            report_digest=comparison.content_digest,
            ready=comparison.decision.ready,
            durability_warnings=comparison.durability_warnings,
        )
        return PilotOrchestrationResult(
            experiment_path.resolve(),
            comparison.generation_path,
            comparison.decision.ready,
            comparison.durability_warnings,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        for descriptor, _, _ in child_directory_fds:
            os.close(descriptor)
        os.close(pinned_config_fd)
        os.close(inputs_fd)
        os.close(output_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the three-GPU Scene 1 pilot")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpus", default=(0, 1, 2), type=parse_gpus)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_pilot(
        args.config,
        args.output_dir,
        gpus=args.gpus,
        resume=args.resume,
        verify_only=args.verify_only,
        trust_upstream_artifacts=args.trust_upstream_artifacts,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

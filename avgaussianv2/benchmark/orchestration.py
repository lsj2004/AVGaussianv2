"""Resumable process orchestration for the strict cam38 benchmark."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from avgaussianv2.benchmark.assets import (
    EXPECTED,
    audit_ftgspp_resume_state,
    audit_ftgspp_upstream_config,
)
from avgaussianv2.benchmark.evaluation import (
    CONTINUATION_SYSTEMS,
    NATIVE_SYSTEMS,
    REPORTING_STEPS,
    audit_training_evidence,
    verify_evaluation,
)
from avgaussianv2.benchmark.artifacts import (
    _load_pinned,
    _publish_pinned,
    canonical_json,
)
from avgaussianv2.benchmark.native import verify_native_contract
from avgaussianv2.benchmark.output import (
    BenchmarkOutputReadLock,
    BenchmarkOutputLock,
    validate_output_children,
)
from avgaussianv2.benchmark.production import (
    continuation_training_evidence,
    expected_identity,
    sha256_file,
)
from avgaussianv2.benchmark.report import verify_scene_report, verify_suite_report
from avgaussianv2.benchmark.training import (
    hash_shared_indices,
    make_shared_indices,
    verify_resume_artifacts,
)


SCENES = ("scene1_opera", "Scene7playing")
GPU_ORDER = ("joint_conditioned", "audio_only", "visual_only")
SCHEMA = "avgaussianv2.cam38-scene-orchestration"
MIN_FREE_BYTES = 300 * 1024**3
FTGSPP_PYTHON = Path("/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python")
GPU_PROBE_TIMEOUT_SECONDS = 120.0
GPU_PROBE_TERMINATE_SECONDS = 5.0


def _cuda_probe_source(modules: Sequence[str]) -> str:
    """Return a fail-closed CUDA/import probe for an isolated child runtime."""

    return (
        "import importlib,json,sys,torch;"
        f"names={tuple(modules)!r};"
        "loaded={};"
        'exec("for name in names:\\n'
        " module=importlib.import_module(name)\\n"
        " loaded[name]=str(getattr(module,'__version__','unknown'))\");"
        "assert torch.cuda.is_available(),'torch.cuda.is_available() is false';"
        "assert torch.cuda.device_count()==1,"
        "f'expected exactly one visible GPU, got {torch.cuda.device_count()}';"
        "x=torch.arange(4096,device='cuda:0',dtype=torch.float32);"
        "value=((x+1.0)*(x+2.0)).sum();"
        "torch.cuda.synchronize();"
        "assert bool(torch.isfinite(value).item()),'CUDA kernel returned non-finite';"
        "print(json.dumps({'python':sys.executable,'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'device_count':torch.cuda.device_count(),"
        "'device':torch.cuda.get_device_name(0),'modules':loaded},sort_keys=True))"
    )


def _runtime_probe_commands(
    python_executable: str,
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    return (
        (
            "avgaussianv2",
            (python_executable, "-c"),
            ("numpy", "yaml", "soundfile", "gsplat", "tinycudann"),
        ),
        (
            "ftgspp",
            (str(FTGSPP_PYTHON), "-c"),
            ("numpy", "gsplat", "tinycudann"),
        ),
        (
            "audiogs",
            ("conda", "run", "-n", "avcloud", "python", "-c"),
            ("numpy", "yaml", "soundfile", "scipy", "librosa", "torchaudio"),
        ),
    )


def _run_probe_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float = GPU_PROBE_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run one probe in a private process group and reap it fail-closed."""
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(environment),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=GPU_PROBE_TERMINATE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        raise subprocess.TimeoutExpired(
            list(command), timeout_seconds, output=error.output, stderr=error.stderr
        ) from error
    completed = subprocess.CompletedProcess(
        list(command), process.returncode, stdout=stdout, stderr=stderr
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            list(command),
            output=stdout,
            stderr=stderr,
        )
    return completed


def _probe_gpu_runtimes(
    gpus: Sequence[int], python_executable: str
) -> dict[str, dict[str, object]]:
    probes: dict[str, dict[str, object]] = {}
    for gpu in gpus:
        gpu_probes: dict[str, object] = {}
        for name, prefix, modules in _runtime_probe_commands(python_executable):
            command = [*prefix, _cuda_probe_source(modules)]
            environment = dict(os.environ)
            environment.update(
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONHASHSEED="42",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
            )
            try:
                completed = _run_probe_command(
                    command,
                    environment=environment,
                )
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                payload = json.loads(lines[-1])
                if (
                    not isinstance(payload, dict)
                    or payload.get("device_count") != 1
                    or not isinstance(payload.get("modules"), dict)
                    or set(payload["modules"]) != set(modules)
                ):
                    raise ValueError("probe returned an invalid dependency payload")
            except (
                IndexError,
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                ValueError,
            ) as error:
                detail = (
                    error.stderr.strip()
                    if isinstance(error, subprocess.CalledProcessError)
                    and isinstance(error.stderr, str)
                    else str(error)
                )
                raise OrchestrationError(
                    f"{name} CUDA preflight failed on physical GPU {gpu}: {detail}"
                ) from error
            gpu_probes[name] = payload
        probes[str(gpu)] = gpu_probes
    return probes


class OrchestrationError(RuntimeError):
    pass


def _require_safe_inode(path: Path, *, directory: bool) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise OrchestrationError(
            f"missing or unsafe benchmark artifact: {path}"
        ) from error
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or (not directory and metadata.st_nlink != 1)
    ):
        raise OrchestrationError(f"unsafe benchmark artifact inode: {path}")
    return metadata


def _safe_bytes(path: Path) -> bytes:
    _require_safe_inode(path, directory=False)
    return path.read_bytes()


def _safe_json(path: Path) -> object:
    return json.loads(_safe_bytes(path))


def _valid_generation_name(name: str) -> bool:
    suffix = name.removeprefix("generation-")
    return (
        name.startswith("generation-")
        and len(suffix) == 32
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _cross_process_path(pinned: Path) -> Path:
    """Make this process's retained descriptor addressable by child processes."""
    prefix = Path("/proc/self/fd")
    relative = pinned.relative_to(prefix)
    return Path("/proc") / str(os.getpid()) / "fd" / relative


def _tree_snapshot_sha256(root: Path) -> str:
    """Hash a validated partial tree, excluding mutable lock records."""
    digest = hashlib.sha256()

    def visit(directory: Path, relative: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            child_relative = relative / child.name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise OrchestrationError(
                    f"unsafe partial snapshot entry: {child_relative}"
                )
            digest.update(child_relative.as_posix().encode())
            digest.update(str(stat.S_IFMT(metadata.st_mode)).encode())
            if stat.S_ISDIR(metadata.st_mode):
                visit(child, child_relative)
            elif stat.S_ISREG(metadata.st_mode):
                if child.name != ".benchmark.lock":
                    _require_safe_inode(child, directory=False)
                    with child.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
            else:
                raise OrchestrationError(
                    f"unsupported partial snapshot entry: {child_relative}"
                )

    visit(root, Path())
    return digest.hexdigest()


class ProcessHandle(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


class ProcessRunner(Protocol):
    assignments: list[tuple[tuple[str, ...], Mapping[str, str], Path]]

    def start(
        self, command: Sequence[str], *, env: Mapping[str, str], log_path: Path
    ) -> ProcessHandle: ...


class _Handle:
    def __init__(self, process: subprocess.Popen[bytes], stream) -> None:
        self.process = process
        self.stream = stream

    def poll(self) -> int | None:
        value = self.process.poll()
        if value is not None and not self.stream.closed:
            self.stream.close()
        return value

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self.process.wait(timeout=timeout)
        finally:
            if self.process.poll() is not None and not self.stream.closed:
                self.stream.close()

    def terminate(self) -> None:
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except OSError as error:
            if error.errno != errno.ESRCH:
                raise

    def kill(self) -> None:
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except OSError as error:
            if error.errno != errno.ESRCH:
                raise


class SubprocessRunner:
    def __init__(self) -> None:
        self.assignments: list[tuple[tuple[str, ...], Mapping[str, str], Path]] = []

    def start(
        self, command: Sequence[str], *, env: Mapping[str, str], log_path: Path
    ) -> ProcessHandle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            raise FileExistsError(f"refusing to overwrite immutable log: {log_path}")
        stream = log_path.open("wb")
        argv = tuple(str(value) for value in command)
        self.assignments.append((argv, dict(env), log_path))
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
        return _Handle(process, stream)


@dataclass(frozen=True)
class SceneBenchmarkResult:
    scene_id: str
    output_dir: Path
    report_dir: Path
    verified: bool


class _AttemptLogs:
    def __init__(
        self, root: Path, identity: str, expected_snapshot: str | None = None
    ) -> None:
        self.logs = root / "logs"
        self.root = root
        self.identity = identity
        self.expected_snapshot = expected_snapshot
        self.sequence = 0
        self.previous: str | None = None
        self.attempt: Path | None = None

    def __enter__(self) -> Path:
        source_snapshot = _tree_snapshot_sha256(self.root)
        if (
            self.expected_snapshot is not None
            and source_snapshot != self.expected_snapshot
        ):
            raise OrchestrationError(
                "partial benchmark output changed before exclusive resume"
            )
        self.logs.mkdir(exist_ok=True)
        _recover_interrupted_attempt(self.logs, self.identity)
        current = self.logs / "current.json"
        if current.is_file():
            value = json.loads(current.read_text())
            self.sequence = int(value["sequence"]) + 1
            self.previous = value["manifest_sha256"]
        self.attempt = self.logs / f"attempt-{self.sequence:06d}"
        self.attempt.mkdir()
        _atomic_json(
            self.attempt / "in_progress.json",
            {
                "schema": f"{SCHEMA}.logs.in-progress",
                "version": 1,
                "identity": self.identity,
                "sequence": self.sequence,
                "previous_manifest_sha256": self.previous,
                "owner_pid": os.getpid(),
                "owner_start_ticks": _process_start_ticks(os.getpid()),
                "started_time_ns": time.time_ns(),
                "source_snapshot_sha256": source_snapshot,
            },
        )
        return self.attempt

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self._finalize(exc_type)
        except BaseException:
            if exc_type is None:
                raise
        return False

    def _finalize(self, exc_type) -> None:
        assert self.attempt is not None
        hashes = {}
        for path in self.attempt.iterdir():
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not path.is_file()
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
            ):
                raise OrchestrationError("attempt log is unsafe")
            hashes[path.name] = sha256_file(path)
        manifest = {
            "schema": f"{SCHEMA}.logs.attempt",
            "version": 1,
            "identity": self.identity,
            "sequence": self.sequence,
            "previous_manifest_sha256": self.previous,
            "state": "failed" if exc_type is not None else "complete",
            "exception_type": None if exc_type is None else exc_type.__name__,
            "sha256": hashes,
        }
        data = canonical_json(manifest)
        _atomic_json(self.attempt / "manifest.json", manifest)
        digest = hashlib.sha256(data).hexdigest()
        _atomic_json(
            self.logs / "current.json",
            {
                "schema": f"{SCHEMA}.logs.current",
                "version": 1,
                "identity": self.identity,
                "sequence": self.sequence,
                "attempt": self.attempt.name,
                "manifest_sha256": digest,
            },
        )


def _process_start_ticks(pid: int) -> int:
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
        return int(value.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError) as error:
        raise OrchestrationError(f"cannot identify attempt owner pid {pid}") from error


def _attempt_owner_live(contract: Mapping[str, object]) -> bool:
    pid = contract.get("owner_pid")
    start = contract.get("owner_start_ticks")
    if not isinstance(pid, int) or isinstance(pid, bool) or not isinstance(start, int):
        raise OrchestrationError("in-progress attempt owner contract is invalid")
    try:
        return _process_start_ticks(pid) == start
    except OrchestrationError:
        return False


def _read_in_progress_attempt(
    attempt: Path,
    *,
    identity: str,
    sequence: int,
    previous: str | None,
) -> dict[str, object]:
    _require_safe_inode(attempt, directory=True)
    contract = _safe_json(attempt / "in_progress.json")
    if (
        not isinstance(contract, dict)
        or set(contract)
        != {
            "schema",
            "version",
            "identity",
            "sequence",
            "previous_manifest_sha256",
            "owner_pid",
            "owner_start_ticks",
            "started_time_ns",
            "source_snapshot_sha256",
        }
        or contract.get("schema") != f"{SCHEMA}.logs.in-progress"
        or contract.get("version") != 1
        or contract.get("identity") != identity
        or contract.get("sequence") != sequence
        or contract.get("previous_manifest_sha256") != previous
        or not isinstance(contract.get("started_time_ns"), int)
        or not isinstance(contract.get("source_snapshot_sha256"), str)
    ):
        raise OrchestrationError("in-progress attempt contract mismatch")
    manifest_temps = set(_atomic_temp_files(attempt, "manifest.json"))
    for child in attempt.iterdir():
        metadata = child.lstat()
        if (
            (
                child.name != "in_progress.json"
                and child.suffix != ".log"
                and child not in manifest_temps
            )
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise OrchestrationError("in-progress attempt inventory is unsafe")
    return contract


def _atomic_temp_files(directory: Path, target: str) -> tuple[Path, ...]:
    prefix = f".{target}."
    matches = tuple(
        child
        for child in directory.iterdir()
        if child.name.startswith(prefix) and len(child.name) > len(prefix)
    )
    if len(matches) > 1:
        raise OrchestrationError(f"multiple atomic temp files for {target}")
    for child in matches:
        metadata = child.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise OrchestrationError(f"unsafe atomic temp file for {target}")
    return matches


def _is_incomplete_attempt_creation(attempt: Path) -> bool:
    entries = list(attempt.iterdir())
    if not entries:
        return True
    for child in entries:
        metadata = child.lstat()
        if (
            not child.name.startswith(".in_progress.json.")
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            return False
    return True


def _seal_interrupted_attempt(
    logs: Path,
    attempt: Path,
    *,
    identity: str,
    sequence: int,
    previous: str | None,
) -> None:
    hashes = {
        child.name: sha256_file(child)
        for child in attempt.iterdir()
        if child.name != "manifest.json"
    }
    manifest = {
        "schema": f"{SCHEMA}.logs.attempt",
        "version": 1,
        "identity": identity,
        "sequence": sequence,
        "previous_manifest_sha256": previous,
        "state": "interrupted",
        "exception_type": "ProcessExit",
        "sha256": hashes,
    }
    data = canonical_json(manifest)
    _atomic_json(attempt / "manifest.json", manifest)
    digest = hashlib.sha256(data).hexdigest()
    _atomic_json(
        logs / "current.json",
        {
            "schema": f"{SCHEMA}.logs.current",
            "version": 1,
            "identity": identity,
            "sequence": sequence,
            "attempt": attempt.name,
            "manifest_sha256": digest,
        },
    )


def _recover_interrupted_attempt(logs: Path, identity: str) -> None:
    tail, pointer_repair, current_temps = _verify_logs_path(
        logs, identity, allow_interrupted_tail=True
    )
    for temporary in current_temps:
        temporary.unlink()
    if pointer_repair is not None:
        _atomic_json(logs / "current.json", pointer_repair)
        tail, second_repair, second_temps = _verify_logs_path(
            logs, identity, allow_interrupted_tail=True
        )
        if second_repair is not None or second_temps:
            raise OrchestrationError("attempt log pointer repair did not converge")
    if tail is None:
        return
    if not (tail / "in_progress.json").exists():
        if not _is_incomplete_attempt_creation(tail):
            raise OrchestrationError("incomplete attempt creation is unsafe")
        for child in tail.iterdir():
            child.unlink()
        tail.rmdir()
        return
    for temporary in _atomic_temp_files(tail, "manifest.json"):
        temporary.unlink()
    current = logs / "current.json"
    previous = None
    sequence = 0
    if current.is_file():
        pointer = _safe_json(current)
        assert isinstance(pointer, dict)
        previous = str(pointer["manifest_sha256"])
        sequence = int(pointer["sequence"]) + 1
    _seal_interrupted_attempt(
        logs,
        tail,
        identity=identity,
        sequence=sequence,
        previous=previous,
    )


def _verify_logs(
    root: Path, identity: str, *, allow_interrupted_tail: bool = False
) -> None:
    tail, _, _ = _verify_logs_path(
        root / "logs", identity, allow_interrupted_tail=allow_interrupted_tail
    )
    if tail is not None and not allow_interrupted_tail:
        raise OrchestrationError("attempt log has an uncommitted tail")


def _verify_logs_path(
    logs: Path, identity: str, *, allow_interrupted_tail: bool
) -> tuple[Path | None, dict[str, object] | None, tuple[Path, ...]]:
    _require_safe_inode(logs, directory=True)
    current_path = logs / "current.json"
    current = _safe_json(current_path) if current_path.exists() else None
    if current is not None and not isinstance(current, dict):
        raise OrchestrationError("attempt log current pointer is invalid")
    current_temps = _atomic_temp_files(logs, "current.json")
    children = tuple(logs.iterdir())
    attempt_candidates = tuple(
        path for path in children if path.name.startswith("attempt-")
    )
    if {path.name for path in children} != {
        *(path.name for path in attempt_candidates),
        *(path.name for path in current_temps),
        *(("current.json",) if current_path.exists() else ()),
    }:
        raise OrchestrationError("attempt log root inventory mismatch")
    try:
        attempts = sorted(
            attempt_candidates,
            key=lambda path: int(path.name.removeprefix("attempt-")),
        )
    except ValueError as error:
        raise OrchestrationError("attempt log sequence is unsafe") from error
    previous: str | None = None
    committed = 0
    committed_pointers: list[dict[str, object]] = []
    tail: Path | None = None
    for sequence, attempt in enumerate(attempts):
        if attempt.name != f"attempt-{sequence:06d}":
            raise OrchestrationError("attempt log sequence is unsafe")
        _require_safe_inode(attempt, directory=True)
        manifest_path = attempt / "manifest.json"
        if not manifest_path.exists():
            if (
                not allow_interrupted_tail
                or sequence != len(attempts) - 1
                or tail is not None
            ):
                raise OrchestrationError("attempt log has an uncommitted entry")
            if not _is_incomplete_attempt_creation(attempt):
                contract = _read_in_progress_attempt(
                    attempt,
                    identity=identity,
                    sequence=sequence,
                    previous=previous,
                )
                if _attempt_owner_live(contract):
                    raise OrchestrationError("in-progress attempt owner is still alive")
            tail = attempt
            continue
        manifest_data = _safe_bytes(manifest_path)
        manifest = json.loads(manifest_data)
        if (
            set(manifest)
            != {
                "schema",
                "version",
                "identity",
                "sequence",
                "previous_manifest_sha256",
                "state",
                "exception_type",
                "sha256",
            }
            or manifest.get("schema") != f"{SCHEMA}.logs.attempt"
            or manifest.get("version") != 1
            or manifest.get("identity") != identity
            or manifest.get("sequence") != sequence
            or manifest.get("previous_manifest_sha256") != previous
            or manifest.get("state") not in {"complete", "failed", "interrupted"}
        ):
            raise OrchestrationError("attempt log manifest chain mismatch")
        expected = {*manifest["sha256"], "manifest.json"}
        if {path.name for path in attempt.iterdir()} != expected:
            raise OrchestrationError("attempt log inventory mismatch")
        for name, digest in manifest["sha256"].items():
            _require_safe_inode(attempt / name, directory=False)
            if sha256_file(attempt / name) != digest:
                raise OrchestrationError("attempt log hash mismatch")
        previous = hashlib.sha256(manifest_data).hexdigest()
        committed_pointers.append(
            {
                "schema": f"{SCHEMA}.logs.current",
                "version": 1,
                "identity": identity,
                "sequence": sequence,
                "attempt": attempt.name,
                "manifest_sha256": previous,
            }
        )
        committed += 1
    expected_current = committed_pointers[-1] if committed_pointers else None
    pointer_repair = None
    if current != expected_current:
        previous_current = (
            committed_pointers[-2] if len(committed_pointers) >= 2 else None
        )
        if (
            allow_interrupted_tail
            and tail is None
            and current == previous_current
            and expected_current is not None
        ):
            pointer_repair = expected_current
        else:
            raise OrchestrationError("attempt log current pointer mismatch")
    if current_temps and pointer_repair is None:
        raise OrchestrationError(
            "atomic current temp has no recoverable pointer transition"
        )
    return tail, pointer_repair, current_temps


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _record_status(root: Path, *, scene: str, phase: str, detail: str) -> None:
    status = root / "status"
    status.mkdir(exist_ok=True)
    sequence = 0
    previous: str | None = None
    if (status / "current.json").is_file():
        _, _, manifest, previous = _load_pinned(status, schema=SCHEMA)
        sequence = int(manifest["identity"]["sequence"]) + 1
    event = {
        "schema": SCHEMA,
        "version": 1,
        "sequence": sequence,
        "scene_id": scene,
        "phase": phase,
        "detail": detail,
        "time_ns": time.time_ns(),
    }
    _publish_pinned(
        status,
        schema=SCHEMA,
        identity={
            "scene_id": scene,
            "sequence": sequence,
            "previous_manifest_sha256": previous,
        },
        files={"event.json": canonical_json(event)},
    )


def _verify_status(root: Path, scene: str) -> None:
    status = root / "status"
    _require_safe_inode(status, directory=True)
    _require_safe_inode(status / "current.json", directory=False)
    _, _, current_manifest, current_digest = _load_pinned(status, schema=SCHEMA)
    generations = status / "generations"
    _require_safe_inode(generations, directory=True)
    if any(path.name.startswith(".") for path in generations.iterdir()):
        raise OrchestrationError("status generations contain an extra entry")
    records = []
    for path in generations.iterdir():
        if not _valid_generation_name(path.name):
            raise OrchestrationError("status generations contain an extra entry")
        _require_safe_inode(path, directory=True)
        manifest_data = _safe_bytes(path / "manifest.json")
        records.append(
            (
                int(json.loads(manifest_data)["identity"]["sequence"]),
                path,
                manifest_data,
            )
        )
    records.sort(key=lambda item: item[0])
    previous: str | None = None
    for sequence, (_, path, manifest_data) in enumerate(records):
        manifest = json.loads(manifest_data)
        event_data = _safe_bytes(path / "event.json")
        if (
            set(manifest) != {"schema", "version", "identity", "sha256"}
            or manifest.get("schema") != f"{SCHEMA}.generation"
            or manifest.get("version") != 1
            or manifest.get("identity")
            != {
                "scene_id": scene,
                "sequence": sequence,
                "previous_manifest_sha256": previous,
            }
            or set(manifest.get("sha256", {})) != {"event.json"}
            or hashlib.sha256(event_data).hexdigest()
            != manifest["sha256"]["event.json"]
            or {child.name for child in path.iterdir()}
            != {"manifest.json", "event.json"}
        ):
            raise OrchestrationError("status generation chain mismatch")
        previous = hashlib.sha256(manifest_data).hexdigest()
    if (
        not records
        or previous != current_digest
        or current_manifest["identity"]["sequence"] != len(records) - 1
        or {path.name for path in status.iterdir()} != {"current.json", "generations"}
    ):
        raise OrchestrationError("status current generation mismatch")


def _record_preflight(root: Path, *, scene: str, payload: Mapping[str, object]) -> None:
    target = root / "preflight"
    target.mkdir(exist_ok=True)
    sequence = 0
    previous: str | None = None
    if (target / "current.json").is_file():
        _, _, manifest, previous = _load_pinned(target, schema=f"{SCHEMA}.preflight")
        sequence = int(manifest["identity"]["sequence"]) + 1
    _publish_pinned(
        target,
        schema=f"{SCHEMA}.preflight",
        identity={
            "scene_id": scene,
            "sequence": sequence,
            "previous_manifest_sha256": previous,
        },
        files={"preflight.json": canonical_json(dict(payload))},
    )


def _verify_preflight_history(root: Path, scene: str) -> None:
    target = root / "preflight"
    _require_safe_inode(target, directory=True)
    if {path.name for path in target.iterdir()} != {"current.json", "generations"}:
        raise OrchestrationError("preflight root inventory mismatch")
    _require_safe_inode(target / "current.json", directory=False)
    _, _, current, digest = _load_pinned(target, schema=f"{SCHEMA}.preflight")
    _require_safe_inode(target / "generations", directory=True)
    paths = list(target.joinpath("generations").iterdir())
    if any(not _valid_generation_name(path.name) for path in paths):
        raise OrchestrationError("preflight generations contain an extra entry")
    records = []
    for path in paths:
        _require_safe_inode(path, directory=True)
        manifest_data = _safe_bytes(path / "manifest.json")
        records.append(
            (
                int(json.loads(manifest_data)["identity"]["sequence"]),
                path,
                manifest_data,
            )
        )
    records.sort(key=lambda item: item[0])
    previous: str | None = None
    for sequence, (_, path, manifest_data) in enumerate(records):
        manifest = json.loads(manifest_data)
        payload = _safe_bytes(path / "preflight.json")
        if (
            set(manifest) != {"schema", "version", "identity", "sha256"}
            or manifest.get("schema") != f"{SCHEMA}.preflight.generation"
            or manifest.get("version") != 1
            or manifest["identity"]
            != {
                "scene_id": scene,
                "sequence": sequence,
                "previous_manifest_sha256": previous,
            }
            or set(manifest["sha256"]) != {"preflight.json"}
            or hashlib.sha256(payload).hexdigest()
            != manifest["sha256"]["preflight.json"]
            or {child.name for child in path.iterdir()}
            != {"manifest.json", "preflight.json"}
        ):
            raise OrchestrationError("preflight generation chain mismatch")
        previous = hashlib.sha256(manifest_data).hexdigest()
    if previous != digest or current["identity"]["sequence"] != len(records) - 1:
        raise OrchestrationError("preflight current generation mismatch")


def parse_gpus(value: str | Sequence[int]) -> tuple[int, ...]:
    parts = value.split(",") if isinstance(value, str) else tuple(value)
    try:
        result = tuple(int(item) for item in parts)
    except (TypeError, ValueError) as error:
        raise ValueError("GPUs must be comma-separated nonnegative IDs") from error
    if not result or len(set(result)) != len(result) or any(item < 0 for item in result):
        raise ValueError("GPUs must be one or more distinct nonnegative IDs")
    return result


def _environment(gpu: int) -> dict[str, str]:
    value = dict(os.environ)
    value.update(PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if gpu >= 0:
        value["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        value.pop("CUDA_VISIBLE_DEVICES", None)
    return value


def _run_one(
    runner: ProcessRunner,
    command: Sequence[str],
    *,
    gpu: int,
    log: Path,
) -> None:
    handle = runner.start(command, env=_environment(gpu), log_path=log)
    try:
        code = handle.wait()
    except BaseException:
        _terminate_handles((handle,))
        raise
    if code:
        _terminate_handles((handle,))
        raise OrchestrationError(f"subprocess failed ({code}); log={log}")


def _terminate_handles(handles: Sequence[ProcessHandle]) -> None:
    """Best-effort shutdown that does not mask an orchestration exception."""

    for handle in handles:
        try:
            handle.terminate()
        except BaseException:
            pass
    for handle in handles:
        try:
            handle.wait(timeout=10)
        except BaseException:
            pass
    for handle in handles:
        try:
            handle.kill()
        except BaseException:
            pass
    for handle in handles:
        try:
            handle.wait()
        except BaseException:
            pass


def _run_parallel(
    runner: ProcessRunner,
    jobs: Sequence[tuple[str, Sequence[str], int, Path]],
) -> None:
    queued = list(jobs)
    active: dict[int, tuple[str, ProcessHandle, Path]] = {}
    failures: list[tuple[str, int, Path]] = []
    failed_handles: list[ProcessHandle] = []
    try:
        while queued or active:
            busy = set(active)
            deferred = []
            for name, command, gpu, log in queued:
                if gpu in busy:
                    deferred.append((name, command, gpu, log))
                    continue
                handle = runner.start(command, env=_environment(gpu), log_path=log)
                active[gpu] = (name, handle, log)
                busy.add(gpu)
            queued = deferred
            for gpu, (name, handle, log) in tuple(active.items()):
                code = handle.poll()
                if code is None:
                    continue
                del active[gpu]
                if code:
                    failures.append((name, code, log))
                    failed_handles.append(handle)
            if failures:
                _terminate_handles(
                    (*failed_handles, *(handle for _, handle, _ in active.values()))
                )
                break
            if queued or active:
                time.sleep(0.05)
    except BaseException:
        _terminate_handles(tuple(handle for _, handle, _ in active.values()))
        raise
    if failures:
        raise OrchestrationError(
            "benchmark subprocess failure: "
            + "; ".join(f"{name}={code} log={log}" for name, code, log in failures)
        )


def _preflight(
    repository: Path,
    config: Path,
    output_parent: Path,
    gpus: tuple[int, ...],
    python_executable: str,
    *,
    _runtime_probes: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if not config.is_file():
        raise OrchestrationError(f"protocol config is missing: {config}")
    if shutil.which(python_executable) is None:
        raise OrchestrationError("selected Python executable is unavailable")
    if shutil.which("conda") is None:
        raise OrchestrationError("conda is unavailable for the AudioGS preflight")
    try:
        query = _run_probe_command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            environment=os.environ,
        ).stdout
        available = {
            int(line.split(",", 1)[0].strip()): int(line.split(",", 1)[1].strip())
            for line in query.splitlines()
            if line.strip()
        }
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        raise OrchestrationError(f"cannot verify assigned GPUs: {error}") from error
    if any(gpu not in available or available[gpu] <= 0 for gpu in gpus):
        raise OrchestrationError("assigned GPU is unavailable or has no free memory")
    runtime_probes = (
        dict(_runtime_probes)
        if _runtime_probes is not None
        else _probe_gpu_runtimes(gpus, python_executable)
    )
    free = shutil.disk_usage(output_parent).free
    if free < MIN_FREE_BYTES:
        raise OrchestrationError(
            f"insufficient checkpoint write budget: {free} < {MIN_FREE_BYTES} bytes"
        )
    source_files = (
        repository / "avgaussianv2" / "benchmark" / "training.py",
        repository / "avgaussianv2" / "benchmark" / "evaluation.py",
        repository / "avgaussianv2" / "benchmark" / "orchestration.py",
        repository / "avgaussianv2" / "benchmark" / "production.py",
        repository / "avgaussianv2" / "benchmark" / "runtime.py",
        repository / "avgaussianv2" / "cli" / "benchmark_worker.py",
        repository / "avgaussianv2" / "cli" / "benchmark_eval.py",
        repository / "scripts" / "train_audiogs_cam38_baselines.sh",
        repository / "scripts" / "prepare_ftgspp_cam38_baselines.sh",
        config,
    )
    return {
        "gpus": list(gpus),
        "free_bytes": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "gpu_runtime_probes": runtime_probes,
        "gpu_free_mib": {str(gpu): available[gpu] for gpu in gpus},
        "source_sha256": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in source_files
        },
        "checkpoint_write_budget": {
            "workers": 3,
            "periodic_interval": 500,
            "milestones": list(REPORTING_STEPS),
            "rolling_retention": 2,
        },
    }


def _native_dirs(repository: Path, scene: str) -> dict[str, Path]:
    root = repository / "runs" / "cam38_strict" / scene
    return {
        "native_audiogs": root / "audiogs" / "native_contract",
        "native_ftgspp": root / "ftgspp" / "native_contract",
    }


def _require_native_contracts(repository: Path, scene: str) -> None:
    for system, path in _native_dirs(repository, scene).items():
        verify_native_contract(
            path,
            expected_scene=scene,
            expected_model_kind="audiogs" if system.endswith("audiogs") else "ftgspp",
        )


def _native_contracts_valid(repository: Path, scenes: Sequence[str]) -> bool:
    try:
        for scene in scenes:
            _require_native_contracts(repository, scene)
    except Exception:
        return False
    return True


def _native_contract_valid(repository: Path, scene: str, system: str) -> bool:
    try:
        verify_native_contract(
            _native_dirs(repository, scene)[system],
            expected_scene=scene,
            expected_model_kind="audiogs" if system == "native_audiogs" else "ftgspp",
        )
    except Exception:
        return False
    return True


def _require_native_partial_resume_state(repository: Path) -> None:
    """Verify every external native asset before a failed-suite resume mutates logs."""
    ftgspp_root = Path("/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus")
    sampled_root = Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf")
    for scene in SCENES:
        if not _native_contract_valid(repository, scene, "native_audiogs"):
            raise OrchestrationError(
                f"{scene} AudioGS native contract is incomplete; unsafe to resume"
            )
        if _native_contract_valid(repository, scene, "native_ftgspp"):
            continue
        strict_root = repository / "runs" / "cam38_strict" / scene
        protocol = strict_root / "protocol"
        source = sampled_root / scene
        config = repository / "configs" / "benchmark_cam38" / f"{scene}.yaml"
        rendered = protocol / "ftgspp_config" / f"{scene}.toml"
        details = audit_ftgspp_upstream_config(
            rendered,
            protocol_config=config,
            repo_root=repository,
            ftgspp_root=ftgspp_root,
            sampled_scene_root=source,
        )
        audit_ftgspp_resume_state(
            scene_id=scene,
            source_root=source,
            train_source=strict_root / "ftgspp" / "train_only_source",
            namespaces=details["namespaces"],
            run_root=strict_root / "ftgspp" / "native",
            marker_root=protocol / "namespace_markers",
            prep_seed_record=protocol / "seed_records" / "prep.json",
            train_seed_record=protocol / "seed_records" / "train.json",
            frame_count=EXPECTED[scene]["test_samples"],
            keyframe_stride=10,
        )


def _native_jobs(
    repository: Path,
    log_root: Path,
    gpus: tuple[int, ...],
    *,
    preflight_only: bool,
    scene: str | None = None,
    resume_native: bool = False,
) -> list[tuple[str, tuple[str, ...], int, Path]]:
    mode = "--preflight-only" if preflight_only else "--execute"
    scenes = (scene,) if scene is not None else SCENES
    if resume_native:
        scenes = tuple(
            item
            for item in scenes
            if not _native_contract_valid(repository, item, "native_ftgspp")
        )
    jobs = [
        (
            f"ftgspp_{item}",
            (
                "bash",
                str(repository / "scripts" / "prepare_ftgspp_cam38_baselines.sh"),
                mode,
                *(("--resume",) if resume_native and not preflight_only else ()),
                "--scene",
                item,
            ),
            gpus[SCENES.index(item) % len(gpus)],
            log_root
            / f"native_{'preflight_' if preflight_only else ''}ftgspp_{item}.log",
        )
        for item in scenes
    ]
    if not resume_native:
        jobs.append(
            (
                "audiogs",
                (
                    "bash",
                    str(repository / "scripts" / "train_audiogs_cam38_baselines.sh"),
                    mode,
                    *(("--scene", scene) if scene is not None else ()),
                ),
                gpus[len(scenes) % len(gpus)],
                log_root
                / f"native_{'preflight_' if preflight_only else ''}audiogs.log",
            )
        )
    return jobs


def verify_scene_outputs(
    output_dir: Path, *, repository: Path, scene: str
) -> SceneBenchmarkResult:
    """Recursively verify a scene without construction, process launch, or writes."""
    original = Path(output_dir)
    with BenchmarkOutputReadLock(original) as pinned:
        _verify_scene_snapshot(pinned, repository=repository, scene=scene)
    return SceneBenchmarkResult(scene, original, original / "report", True)


def _verify_scene_snapshot(
    output_dir: Path,
    *,
    repository: Path,
    scene: str,
    allow_interrupted_logs: bool = False,
) -> None:
    expected_root_entries = {
        ".benchmark.lock",
        "preflight",
        "status",
        "protocol",
        "workers",
        "logs",
        "evaluations",
        "report",
    }
    actual_root_entries = {path.name for path in output_dir.iterdir()}
    if actual_root_entries != expected_root_entries or any(
        path.is_symlink() for path in output_dir.iterdir()
    ):
        raise OrchestrationError("scene output root inventory mismatch")
    _verify_status(output_dir, scene)
    _verify_preflight_history(output_dir, scene)
    _verify_logs(output_dir, scene, allow_interrupted_tail=allow_interrupted_logs)
    preparation_path = output_dir / "protocol" / "preparation.json"
    protocol_root = preparation_path.parent
    if {path.name for path in protocol_root.iterdir()} != {
        "preparation.json",
        "resolved_project.yaml",
        "resolved_project.origin.json",
        "worker_manifests",
        "immutable",
    } or {path.name for path in (protocol_root / "worker_manifests").iterdir()} != {
        f"{mode}.json" for mode in GPU_ORDER
    }:
        raise OrchestrationError("protocol output inventory mismatch")
    preparation = json.loads(preparation_path.read_text())
    if (
        preparation.get("schema") != "avgaussianv2.cam38-production-preparation"
        or preparation.get("scene_id") != scene
        or preparation.get("include_eval") is not False
        or sha256_file(output_dir / "protocol" / "resolved_project.yaml")
        != preparation["resolved_config_sha256"]
        or set(preparation.get("worker_manifests", {})) != set(GPU_ORDER)
        or set(preparation.get("native_contracts", {})) != {"audiogs", "ftgspp"}
    ):
        raise OrchestrationError("production preparation contract mismatch")
    _verify_preparation(
        preparation_path,
        scene,
        repository / "configs" / "benchmark_cam38" / f"{scene}.yaml",
    )
    for kind, record in preparation["native_contracts"].items():
        contract = verify_native_contract(
            Path(record["path"]),
            expected_scene=scene,
            expected_model_kind=kind,
        )
        if (
            contract["_manifest_sha256"] != record["manifest_sha256"]
            or contract["checkpoint"]["sha256"] != record["checkpoint_sha256"]
        ):
            raise OrchestrationError("prepared native contract identity mismatch")
    evaluations = output_dir / "evaluations"
    if {path.name for path in evaluations.iterdir()} != {
        *NATIVE_SYSTEMS,
        *CONTINUATION_SYSTEMS,
    }:
        raise OrchestrationError("evaluation system inventory mismatch")
    for system in NATIVE_SYSTEMS:
        if {path.name for path in (evaluations / system).iterdir()} != {"native"}:
            raise OrchestrationError("native evaluation inventory mismatch")
        verify_evaluation(
            evaluations / system / "native",
            identity=expected_identity(scene, system, None),
        )
    for system in CONTINUATION_SYSTEMS:
        if {path.name for path in (evaluations / system).iterdir()} != {
            f"step_{step:06d}" for step in REPORTING_STEPS
        }:
            raise OrchestrationError("continuation evaluation inventory mismatch")
        for step in REPORTING_STEPS:
            verify_evaluation(
                evaluations / system / f"step_{step:06d}",
                identity=expected_identity(scene, system, step),
            )
    verify_scene_report(output_dir / "report")


def _recover_completed_scene(
    output: Path, *, repository: Path, scene: str
) -> SceneBenchmarkResult:
    with BenchmarkOutputReadLock(output) as pinned:
        _verify_scene_snapshot(
            pinned,
            repository=repository,
            scene=scene,
            allow_interrupted_logs=True,
        )
        snapshot = _tree_snapshot_sha256(pinned)
    with BenchmarkOutputLock(output) as pinned:
        if _tree_snapshot_sha256(pinned) != snapshot:
            raise OrchestrationError(
                "completed scene changed before exclusive log recovery"
            )
        _recover_interrupted_attempt(pinned / "logs", scene)
    return verify_scene_outputs(output, repository=repository, scene=scene)


def _verify_preparation(path: Path, scene: str, source_config: Path) -> None:
    _require_safe_inode(path.parent, directory=True)
    _require_safe_inode(path, directory=False)
    _require_safe_inode(path.parent / "resolved_project.yaml", directory=False)
    _require_safe_inode(path.parent / "resolved_project.origin.json", directory=False)
    _require_safe_inode(path.parent / "worker_manifests", directory=True)
    for mode in GPU_ORDER:
        _require_safe_inode(
            path.parent / "worker_manifests" / f"{mode}.json", directory=False
        )
    raw = json.loads(path.read_text())
    if (
        raw.get("schema") != "avgaussianv2.cam38-production-preparation"
        or raw.get("version") != 1
        or raw.get("scene_id") != scene
        or raw.get("include_eval") is not False
        or sha256_file(path.parent / "resolved_project.yaml")
        != raw.get("resolved_config_sha256")
        or set(raw.get("worker_manifests", {})) != set(GPU_ORDER)
        or set(raw.get("native_contracts", {})) != {"audiogs", "ftgspp"}
    ):
        raise OrchestrationError("existing preparation is incompatible")
    immutable = path.parent / "immutable"
    _require_safe_inode(immutable, directory=True)
    _require_safe_inode(immutable / ".benchmark.lock", directory=False)
    _require_safe_inode(immutable / "current.json", directory=False)
    _require_safe_inode(immutable / "generations", directory=True)
    if {entry.name for entry in immutable.iterdir()} != {
        ".benchmark.lock",
        "current.json",
        "generations",
    }:
        raise OrchestrationError("immutable protocol root inventory mismatch")
    _, files, _, _ = _load_pinned(
        immutable,
        schema="avgaussianv2.cam38-production-preparation",
        expected_identity={"scene_id": scene},
    )
    expected_files = {
        "resolved_project.yaml": (path.parent / "resolved_project.yaml").read_bytes(),
        "resolved_project.origin.json": (
            path.parent / "resolved_project.origin.json"
        ).read_bytes(),
        "joint_conditioned.json": (
            path.parent / "worker_manifests" / "joint_conditioned.json"
        ).read_bytes(),
        "audio_only.json": (
            path.parent / "worker_manifests" / "audio_only.json"
        ).read_bytes(),
        "visual_only.json": (
            path.parent / "worker_manifests" / "visual_only.json"
        ).read_bytes(),
        "preparation.json": path.read_bytes(),
    }
    if files != expected_files:
        raise OrchestrationError(
            "mutable protocol files differ from immutable generation"
        )
    origin = json.loads((path.parent / "resolved_project.origin.json").read_text())
    if (
        origin.get("schema") != "avgaussianv2.cam38-resolved-config-origin"
        or origin.get("source_path") != str(source_config.resolve())
        or origin.get("source_sha256") != sha256_file(source_config)
        or origin.get("resolved_sha256") != raw["resolved_config_sha256"]
    ):
        raise OrchestrationError("resolved config origin is incompatible")
    expected_indices = make_shared_indices(
        len(raw["runtime"]["dataset_sample_ids"]), 30_000, 42
    )
    expected_index_sha256 = hash_shared_indices(expected_indices)
    for mode in raw["worker_manifests"]:
        payload = json.loads(
            (path.parent / "worker_manifests" / f"{mode}.json").read_text()
        )
        compatibility = payload.get("compatibility", {})
        if (
            payload.get("schema") != "avgaussianv2.cam38-benchmark-worker"
            or payload.get("version") != 1
            or payload.get("mode") != mode
            or payload.get("shared_indices") != list(expected_indices)
            or compatibility.get("scene_id") != scene
            or compatibility.get("mode") != mode
            or compatibility.get("index_sha256") != expected_index_sha256
            or any(
                compatibility.get(name) != raw["runtime"].get(name)
                for name in (
                    "config_sha256",
                    "source_sha256",
                    "visual_initialization_sha256",
                    "audio_initialization_sha256",
                    "model_initialization_sha256",
                )
            )
        ):
            raise OrchestrationError("existing worker manifest is incompatible")


def _partial_evaluation_artifacts(
    system_dir: Path,
) -> tuple[tuple[Path, int | None], ...]:
    system = system_dir.name
    if system not in {*NATIVE_SYSTEMS, *CONTINUATION_SYSTEMS}:
        raise OrchestrationError("partial evaluation set has an extra system")
    artifacts = tuple(system_dir.iterdir())
    actual = {artifact.name for artifact in artifacts}
    allowed = (
        {"native"}
        if system in NATIVE_SYSTEMS
        else {f"step_{step:06d}" for step in REPORTING_STEPS}
    )
    if not actual or not actual.issubset(allowed):
        raise OrchestrationError("partial evaluation artifact matrix mismatch")
    return tuple(
        (
            artifact,
            None
            if system in NATIVE_SYSTEMS
            else int(artifact.name.removeprefix("step_")),
        )
        for artifact in artifacts
    )


def _inspect_partial_resume(
    output: Path,
    *,
    stable_output: Path,
    scene: str,
    config_path: Path,
) -> tuple[frozenset[str], str]:
    """Validate every existing partial artifact before any mutation/runtime."""
    allowed = {
        ".benchmark.lock",
        "preflight",
        "status",
        "protocol",
        "workers",
        "logs",
        "evaluations",
        "report",
    }
    with BenchmarkOutputReadLock(output) as pinned:
        for child in pinned.iterdir():
            metadata = child.lstat()
            if child.name not in allowed or child.is_symlink():
                raise OrchestrationError(
                    f"unexpected or unsafe partial output: {child.name}"
                )
            if metadata.st_uid != os.getuid():
                raise OrchestrationError("partial output has foreign owner")
        if (pinned / "status").exists():
            _verify_status(pinned, scene)
        if (pinned / "preflight").exists():
            _verify_preflight_history(pinned, scene)
        if (pinned / "logs").exists():
            _verify_logs(pinned, scene, allow_interrupted_tail=True)
        protocol = pinned / "protocol"
        preparation = protocol / "preparation.json"
        if protocol.exists():
            if preparation.is_file() and not preparation.is_symlink():
                _verify_preparation(preparation, scene, config_path)
            else:
                _verify_failed_prepare_protocol(
                    protocol,
                    logs=pinned / "logs",
                    scene=scene,
                    source_config=config_path,
                )
        workers_root = pinned / "workers"
        resume_modes: set[str] = set()
        if workers_root.exists():
            actual = {path.name for path in workers_root.iterdir()}
            if not actual.issubset(set(GPU_ORDER)):
                raise OrchestrationError("partial worker set contains an extra entry")
            if not preparation.exists():
                raise OrchestrationError("worker output exists before preparation")
            for mode in actual:
                worker = workers_root / mode
                with BenchmarkOutputReadLock(worker) as pinned_worker:
                    validate_output_children(pinned_worker)
                    entries = {
                        path.name
                        for path in pinned_worker.iterdir()
                        if path.name != ".benchmark.lock"
                    }
                    if not entries:
                        continue
                    if "contract.json" not in entries:
                        raise OrchestrationError(
                            f"{mode} partial output has no committed contract"
                        )
                    contract = json.loads((pinned_worker / "contract.json").read_text())
                    manifest = json.loads(
                        (protocol / "worker_manifests" / f"{mode}.json").read_text()
                    )
                    if contract.get("compatibility") != manifest.get(
                        "compatibility"
                    ) or contract.get("shared_indices") != manifest.get(
                        "shared_indices"
                    ):
                        raise OrchestrationError(
                            f"{mode} partial output is incompatible"
                        )
                    _verify_worker_for_resume(
                        pinned_worker,
                        stable_worker=stable_output / "workers" / mode,
                        worker_manifest=manifest,
                        scene=scene,
                        mode=mode,
                    )
                    resume_modes.add(mode)
        evaluations = pinned / "evaluations"
        if evaluations.exists():
            for system_dir in evaluations.iterdir():
                for artifact, step in _partial_evaluation_artifacts(system_dir):
                    if not (artifact / "current.json").is_file():
                        raise OrchestrationError(
                            "partial evaluation is not an immutable generation"
                        )
                    verify_evaluation(
                        artifact,
                        identity=expected_identity(scene, system_dir.name, step),
                    )
        report = pinned / "report"
        if report.exists() and any(
            child.name != ".benchmark.lock" for child in report.iterdir()
        ):
            raise OrchestrationError("partial scene report cannot be resumed")
        return frozenset(resume_modes), _tree_snapshot_sha256(pinned)


def _verify_worker_for_resume(
    pinned_worker: Path,
    *,
    stable_worker: Path,
    worker_manifest: Mapping[str, object],
    scene: str,
    mode: str,
) -> None:
    """Audit finalized outputs by immutable milestones; otherwise audit resume state."""
    if (pinned_worker / "artifact_hashes.json").is_file():
        for step in REPORTING_STEPS:
            evidence = continuation_training_evidence(
                stable_worker,
                scene_id=scene,
                system=mode,
                step=step,
            )
            audit_training_evidence(
                evidence,
                expected_identity(scene, mode, step),
            )
        return
    verify_resume_artifacts(pinned_worker, worker_manifest=worker_manifest)


def _verify_failed_prepare_protocol(
    protocol: Path,
    *,
    logs: Path,
    scene: str,
    source_config: Path,
) -> None:
    """Allow retry only for the exact files left by a failed prepare subprocess."""
    if protocol.is_symlink() or not protocol.is_dir():
        raise OrchestrationError("failed prepare protocol root is unsafe")
    expected_protocol = {
        "resolved_project.yaml",
        "resolved_project.origin.json",
    }
    if {entry.name for entry in protocol.iterdir()} != expected_protocol:
        raise OrchestrationError("partial protocol is not committed")
    resolved = protocol / "resolved_project.yaml"
    origin_path = protocol / "resolved_project.origin.json"
    _require_safe_inode(resolved, directory=False)
    _require_safe_inode(origin_path, directory=False)

    _verify_logs(protocol.parent, scene, allow_interrupted_tail=True)
    current = _safe_json(logs / "current.json")
    attempt_name = current.get("attempt") if isinstance(current, Mapping) else None
    if (
        not isinstance(attempt_name, str)
        or not attempt_name.startswith("attempt-")
        or not attempt_name.removeprefix("attempt-").isdigit()
    ):
        raise OrchestrationError("failed prepare log pointer is invalid")
    attempt = logs / attempt_name
    manifest = _safe_json(attempt / "manifest.json")
    hashes = manifest.get("sha256") if isinstance(manifest, Mapping) else None
    if (
        manifest.get("state") != "failed"
        or manifest.get("exception_type") != "OrchestrationError"
        or manifest.get("identity") != scene
        or not isinstance(hashes, Mapping)
        or set(hashes) != {"in_progress.json", "prepare.log"}
    ):
        raise OrchestrationError("failed prepare log evidence is invalid")
    prepare_log = _safe_bytes(attempt / "prepare.log").decode(
        "utf-8", errors="strict"
    )
    if not all(
        marker in prepare_log
        for marker in ("Traceback", "benchmark_prepare.py", "prepare_worker_manifests")
    ):
        raise OrchestrationError("failed prepare traceback evidence is invalid")

    origin = _safe_json(origin_path)
    expected_origin_fields = {
        "schema",
        "version",
        "source_path",
        "source_sha256",
        "resolved_sha256",
    }
    if (
        set(origin) != expected_origin_fields
        or origin.get("schema") != "avgaussianv2.cam38-resolved-config-origin"
        or origin.get("version") != 1
        or origin.get("source_path") != str(source_config.resolve())
        or origin.get("source_sha256") != sha256_file(source_config)
        or origin.get("resolved_sha256") != sha256_file(resolved)
    ):
        raise OrchestrationError("failed prepare resolved config origin mismatches")


def _inspect_partial_suite(
    output: Path,
    *,
    repository: Path,
) -> str:
    """Reject every unsafe/incompatible suite artifact before any mutation."""
    allowed = {".benchmark.lock", *SCENES, "logs", "report"}
    with BenchmarkOutputReadLock(output) as pinned:
        for child in pinned.iterdir():
            metadata = child.lstat()
            if (
                child.name not in allowed
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.getuid()
            ):
                raise OrchestrationError(
                    f"unexpected or unsafe partial suite output: {child.name}"
                )
        if (pinned / "logs").exists():
            _verify_logs(pinned, "suite", allow_interrupted_tail=True)
        for scene in SCENES:
            scene_root = pinned / scene
            if not scene_root.exists():
                continue
            try:
                verify_scene_outputs(scene_root, repository=repository, scene=scene)
            except Exception:
                _inspect_partial_resume(
                    scene_root,
                    stable_output=output / scene,
                    scene=scene,
                    config_path=repository
                    / "configs"
                    / "benchmark_cam38"
                    / f"{scene}.yaml",
                )
        report = pinned / "report"
        if report.exists() and any(
            child.name != ".benchmark.lock" for child in report.iterdir()
        ):
            raise OrchestrationError("partial suite report cannot be resumed")
        return _tree_snapshot_sha256(pinned)


def _scene_commands(
    *,
    repository: Path,
    output: Path,
    stable_output: Path,
    log_root: Path,
    config: Path,
    python: str,
    gpus: tuple[int, ...],
    resume_modes: frozenset[str],
) -> tuple[
    list[str],
    list[tuple[str, list[str], int, Path]],
    list[tuple[str, list[str], int, Path]],
]:
    protocol = output / "protocol"
    prepare = [
        python,
        "-m",
        "avgaussianv2.cli.benchmark_prepare",
        "--config",
        str(config),
        "--output-dir",
        str(protocol),
        "--devices",
        ",".join(f"cuda:{gpu}" for gpu in gpus),
        "--native-audiogs-contract",
        str(_native_dirs(repository, config.stem)["native_audiogs"]),
        "--native-ftgspp-contract",
        str(_native_dirs(repository, config.stem)["native_ftgspp"]),
        "--trust-upstream-artifacts",
    ]
    workers = []
    for index, mode in enumerate(GPU_ORDER):
        gpu = gpus[index % len(gpus)]
        command = [
            python,
            "-m",
            "avgaussianv2.cli.benchmark_worker",
            "--manifest",
            str(protocol / "worker_manifests" / f"{mode}.json"),
            "--config",
            str(protocol / "resolved_project.yaml"),
            "--output-dir",
            str(output / "workers" / mode),
            "--device",
            "cuda:0",
            "--trust-upstream-artifacts",
        ]
        if mode in resume_modes:
            command.append("--resume")
        workers.append((mode, command, gpu, log_root / f"worker_{mode}.log"))
    evals = []
    sources = _native_dirs(repository, config.stem)
    specs = [(name, None, sources[name]) for name in sorted(NATIVE_SYSTEMS)]
    specs += [
        (name, step, stable_output / "workers" / name)
        for name in sorted(CONTINUATION_SYSTEMS)
        for step in REPORTING_STEPS
    ]
    specs += [
        (name, step, stable_output / "workers" / "joint_conditioned")
        for name in ("joint_conditioned_no_rgbd", "joint_conditioned_wrong_camera")
        for step in REPORTING_STEPS
    ]
    for index, (system, step, source) in enumerate(specs):
        label = f"{system}_{'native' if step is None else step}"
        destination = (
            output / "evaluations" / system / "native"
            if step is None
            else output / "evaluations" / system / f"step_{step:06d}"
        )
        command = [
            python,
            "-m",
            "avgaussianv2.cli.benchmark_eval",
            "--scene",
            config.stem,
            "--system",
            system,
            "--resolved-config",
            str(protocol / "resolved_project.yaml"),
            "--source",
            str(source),
            "--output-dir",
            str(destination),
            "--device",
            "cuda:0",
            "--trust-upstream-artifacts",
            "--compute-dpam",
        ]
        if step is not None:
            command += ["--step", str(step)]
        if (destination / "current.json").is_file():
            command.append("--resume")
        gpu = gpus[index % len(gpus)]
        evals.append((label, command, gpu, log_root / f"eval_{label}.log"))
    return prepare, workers, evals


def run_scene_benchmark(
    *,
    config_path: Path,
    output_dir: Path,
    gpus: Sequence[int] = (0, 1),
    python_executable: str = sys.executable,
    resume: bool = False,
    verify_only: bool = False,
    skip_native_training: bool = False,
    runner: ProcessRunner | None = None,
    _verifier: Callable[..., SceneBenchmarkResult] = verify_scene_outputs,
    _preflight_fn: Callable[..., Mapping[str, object]] = _preflight,
    _skip_preflight: bool = False,
    _preflight_payload: Mapping[str, object] | None = None,
    _stable_output_dir: Path | None = None,
) -> SceneBenchmarkResult:
    config_path = Path(config_path).absolute()
    repository = config_path.parent.parent.parent
    scene = config_path.stem
    if scene not in SCENES:
        raise ValueError("strict suite supports only scene1_opera and Scene7playing")
    output = Path(output_dir).absolute()
    stable_output = (
        output
        if _stable_output_dir is None
        else Path(_stable_output_dir).absolute()
    )
    resume_modes: frozenset[str] = frozenset()
    resume_snapshot: str | None = None
    if verify_only:
        return _verifier(output, repository=repository, scene=scene)
    if resume and (output / "report" / "current.json").is_file():
        try:
            return _verifier(output, repository=repository, scene=scene)
        except Exception:
            if _verifier is not verify_scene_outputs:
                raise
            return _recover_completed_scene(output, repository=repository, scene=scene)
    if resume and output.exists():
        try:
            return _verifier(output, repository=repository, scene=scene)
        except Exception:
            resume_modes, resume_snapshot = _inspect_partial_resume(
                output,
                stable_output=stable_output,
                scene=scene,
                config_path=config_path,
            )
    devices = parse_gpus(gpus)
    if output.exists() and not resume and any(output.iterdir()):
        raise OrchestrationError("scene output exists; pass --resume")
    if skip_native_training:
        _require_native_contracts(repository, scene)
    elif resume:
        if not _native_contracts_valid(repository, (scene,)):
            raise OrchestrationError(
                "native baseline stage is incomplete; its upstream trainers have no "
                "exact-resume contract, so repair/complete Task11 native training "
                "before resuming continuations"
            )
        _require_native_contracts(repository, scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    runner = runner or SubprocessRunner()
    with (
        BenchmarkOutputLock(output) as pinned_output,
        _AttemptLogs(
            pinned_output, scene, expected_snapshot=resume_snapshot
        ) as attempt_logs,
    ):
        child_output = _cross_process_path(pinned_output)
        if _skip_preflight:
            if _preflight_payload is None:
                raise OrchestrationError("suite scene is missing shared preflight evidence")
            preflight = _preflight_payload
        else:
            preflight = _preflight_fn(
                repository,
                config_path,
                output.parent,
                devices,
                python_executable,
            )
        _record_preflight(pinned_output, scene=scene, payload=dict(preflight))
        _record_status(pinned_output, scene=scene, phase="preflight", detail="complete")
        if not (skip_native_training or resume):
            _run_parallel(
                runner,
                _native_jobs(
                    repository,
                    attempt_logs,
                    devices,
                    preflight_only=True,
                    scene=scene,
                ),
            )
            _run_parallel(
                runner,
                _native_jobs(
                    repository,
                    attempt_logs,
                    devices,
                    preflight_only=False,
                    scene=scene,
                ),
            )
            _require_native_contracts(repository, scene)
        _record_status(pinned_output, scene=scene, phase="native", detail="verified")
        prepare, workers, evaluations = _scene_commands(
            repository=repository,
            output=child_output,
            stable_output=stable_output,
            log_root=attempt_logs,
            config=config_path,
            python=python_executable,
            gpus=devices,
            resume_modes=resume_modes,
        )
        preparation = pinned_output / "protocol" / "preparation.json"
        if not (resume and preparation.is_file()):
            _run_one(
                runner,
                prepare,
                gpu=-1,
                log=attempt_logs / "prepare.log",
            )
        else:
            _verify_preparation(preparation, scene, config_path)
        _record_status(pinned_output, scene=scene, phase="prepare", detail="complete")
        _run_parallel(runner, workers)
        _record_status(pinned_output, scene=scene, phase="training", detail="complete")
        _run_parallel(runner, evaluations)
        _record_status(
            pinned_output, scene=scene, phase="evaluation", detail="complete"
        )
        report = [
            python_executable,
            "-m",
            "avgaussianv2.cli.benchmark_report",
            "--kind",
            "scene",
            "--scene",
            scene,
            "--evaluations-root",
            str(stable_output / "evaluations"),
            "--output-dir",
            str(stable_output / "report"),
        ]
        if resume and (output / "report" / "current.json").is_file():
            report.append("--resume")
        _run_one(
            runner,
            report,
            gpu=devices[0],
            log=attempt_logs / "report.log",
        )
        _record_status(pinned_output, scene=scene, phase="report", detail="complete")
    result = _verifier(output, repository=repository, scene=scene)
    return result


def _verify_completed_suite_snapshot(
    pinned: Path,
    *,
    repository: Path,
    allow_interrupted_logs: bool = False,
    report_verifier: Callable[[Path], Mapping[str, object]] = verify_suite_report,
    report_path: Path | None = None,
) -> dict[str, object]:
    if {path.name for path in pinned.iterdir()} != {
        ".benchmark.lock",
        *SCENES,
        "logs",
        "report",
    }:
        raise OrchestrationError("suite output root inventory mismatch")
    _verify_logs(pinned, "suite", allow_interrupted_tail=allow_interrupted_logs)
    for scene in SCENES:
        verify_scene_outputs(pinned / scene, repository=repository, scene=scene)
    return dict(
        report_verifier(pinned / "report" if report_path is None else report_path)
    )


def run_benchmark_suite(
    *,
    repository: Path,
    output_dir: Path,
    gpus: Sequence[int] = (0, 1),
    python_executable: str = sys.executable,
    resume: bool = False,
    verify_only: bool = False,
    skip_native_training: bool = False,
    runner: ProcessRunner | None = None,
    _scene_runner: Callable[..., SceneBenchmarkResult] = run_scene_benchmark,
    _suite_verifier: Callable[[Path], Mapping[str, object]] = verify_suite_report,
    _preflight_fn: Callable[..., Mapping[str, object]] = _preflight,
) -> dict[str, object]:
    repository = Path(repository).absolute()
    output = Path(output_dir).absolute()
    resume_snapshot: str | None = None
    if verify_only:
        with BenchmarkOutputReadLock(output) as pinned:
            return _verify_completed_suite_snapshot(
                pinned,
                repository=repository,
                report_verifier=_suite_verifier,
                report_path=output / "report",
            )
    if resume and (output / "report" / "current.json").is_file():
        try:
            with BenchmarkOutputReadLock(output) as pinned:
                return _verify_completed_suite_snapshot(
                    pinned,
                    repository=repository,
                    report_verifier=_suite_verifier,
                    report_path=output / "report",
                )
        except Exception:
            if _suite_verifier is not verify_suite_report:
                raise
        with BenchmarkOutputReadLock(output) as pinned:
            _verify_completed_suite_snapshot(
                pinned, repository=repository, allow_interrupted_logs=True
            )
            snapshot = _tree_snapshot_sha256(pinned)
        with BenchmarkOutputLock(output) as pinned:
            if _tree_snapshot_sha256(pinned) != snapshot:
                raise OrchestrationError(
                    "completed suite changed before exclusive log recovery"
                )
            _recover_interrupted_attempt(pinned / "logs", "suite")
        with BenchmarkOutputReadLock(output) as pinned:
            return _verify_completed_suite_snapshot(
                pinned,
                repository=repository,
                report_verifier=_suite_verifier,
                report_path=output / "report",
            )
    if resume and output.exists():
        resume_snapshot = _inspect_partial_suite(output, repository=repository)
    if output.exists() and not resume and any(output.iterdir()):
        raise OrchestrationError("suite output exists; pass --resume")
    devices = parse_gpus(gpus)
    runner = runner or SubprocessRunner()
    native_complete = _native_contracts_valid(repository, SCENES)
    if skip_native_training:
        if not native_complete:
            raise OrchestrationError(
                "native baseline stage is incomplete; complete both immutable "
                "native contracts before suite resume/reuse"
            )
        for scene in SCENES:
            _require_native_contracts(repository, scene)
    elif resume and not native_complete:
        _require_native_partial_resume_state(repository)
    elif resume:
        for scene in SCENES:
            _require_native_contracts(repository, scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Suite-wide Python/GPU/disk/source validation completes for both scenes
    # before any native trainer is allowed to start.
    preflight_by_scene: dict[str, Mapping[str, object]] = {}
    shared_runtime_probes: Mapping[str, Mapping[str, object]] | None = None
    for scene in SCENES:
        arguments = (
            repository,
            repository / "configs" / "benchmark_cam38" / f"{scene}.yaml",
            output.parent,
            devices,
            python_executable,
        )
        if _preflight_fn is _preflight:
            payload = _preflight_fn(
                *arguments, _runtime_probes=shared_runtime_probes
            )
            shared_runtime_probes = payload["gpu_runtime_probes"]  # type: ignore[assignment]
        else:
            payload = _preflight_fn(*arguments)
        preflight_by_scene[scene] = payload
    with (
        BenchmarkOutputLock(output) as pinned_output,
        _AttemptLogs(
            pinned_output, "suite", expected_snapshot=resume_snapshot
        ) as attempt_logs,
    ):
        child_output = _cross_process_path(pinned_output)
        # Native scripts cover both scenes. Run them once, then every scene
        # requires the resulting contracts and cannot silently retrain.
        if not skip_native_training and not native_complete:
            _run_parallel(
                runner,
                _native_jobs(
                    repository,
                    attempt_logs,
                    devices,
                    preflight_only=True,
                    resume_native=resume,
                ),
            )
            _run_parallel(
                runner,
                _native_jobs(
                    repository,
                    attempt_logs,
                    devices,
                    preflight_only=False,
                    resume_native=resume,
                ),
            )
        results = [
            _scene_runner(
                config_path=repository
                / "configs"
                / "benchmark_cam38"
                / f"{scene}.yaml",
                output_dir=child_output / scene,
                _stable_output_dir=output / scene,
                gpus=devices,
                python_executable=python_executable,
                resume=resume,
                verify_only=False,
                skip_native_training=True,
                runner=runner,
                _skip_preflight=True,
                _preflight_payload=preflight_by_scene[scene],
            )
            for scene in SCENES
        ]
        command = [
            python_executable,
            "-m",
            "avgaussianv2.cli.benchmark_report",
            "--kind",
            "suite",
            "--output-dir",
            str(child_output / "report"),
        ]
        for result in results:
            command += ["--scene-report", str(result.report_dir)]
        if resume and (output / "report" / "current.json").is_file():
            command.append("--resume")
        _run_one(
            runner,
            command,
            gpu=devices[0],
            log=attempt_logs / "suite_report.log",
        )
    return dict(_suite_verifier(output / "report"))


__all__ = [
    "OrchestrationError",
    "SceneBenchmarkResult",
    "SubprocessRunner",
    "parse_gpus",
    "run_benchmark_suite",
    "run_scene_benchmark",
    "verify_scene_outputs",
]

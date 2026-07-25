"""Resumable process orchestration for the strict cam38 benchmark."""

from __future__ import annotations

import json
import hashlib
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

from avgaussianv2.benchmark.evaluation import (
    CONTINUATION_SYSTEMS,
    NATIVE_SYSTEMS,
    REPORTING_STEPS,
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
from avgaussianv2.benchmark.production import expected_identity, sha256_file
from avgaussianv2.benchmark.report import verify_scene_report, verify_suite_report
from avgaussianv2.benchmark.training import (
    hash_shared_indices,
    make_shared_indices,
)


SCENES = ("scene1_opera", "Scene7playing")
GPU_ORDER = ("joint_conditioned", "audio_only", "visual_only")
SCHEMA = "avgaussianv2.cam38-scene-orchestration"
MIN_FREE_BYTES = 300 * 1024**3
FTGSPP_PYTHON = Path("/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python")


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
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=environment,
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
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)

    def kill(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)


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
        if (
            self.expected_snapshot is not None
            and _tree_snapshot_sha256(self.root) != self.expected_snapshot
        ):
            raise OrchestrationError(
                "partial benchmark output changed before exclusive resume"
            )
        self.logs.mkdir(exist_ok=True)
        current = self.logs / "current.json"
        if current.is_file():
            value = json.loads(current.read_text())
            self.sequence = int(value["sequence"]) + 1
            self.previous = value["manifest_sha256"]
        self.attempt = self.logs / f"attempt-{self.sequence:06d}"
        self.attempt.mkdir()
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
            "outcome": "failed" if exc_type is not None else "complete",
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


def _verify_logs(root: Path, identity: str) -> None:
    logs = root / "logs"
    _require_safe_inode(logs, directory=True)
    current = _safe_json(logs / "current.json")
    if not isinstance(current, dict):
        raise OrchestrationError("attempt log current pointer is invalid")
    attempts = sorted(
        (path for path in logs.iterdir() if path.name != "current.json"),
        key=lambda path: int(path.name.removeprefix("attempt-")),
    )
    previous: str | None = None
    for sequence, attempt in enumerate(attempts):
        if attempt.name != f"attempt-{sequence:06d}":
            raise OrchestrationError("attempt log sequence is unsafe")
        _require_safe_inode(attempt, directory=True)
        manifest_data = _safe_bytes(attempt / "manifest.json")
        manifest = json.loads(manifest_data)
        if (
            manifest.get("schema") != f"{SCHEMA}.logs.attempt"
            or manifest.get("identity") != identity
            or manifest.get("sequence") != sequence
            or manifest.get("previous_manifest_sha256") != previous
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
    if not attempts or current != {
        "schema": f"{SCHEMA}.logs.current",
        "version": 1,
        "identity": identity,
        "sequence": len(attempts) - 1,
        "attempt": attempts[-1].name,
        "manifest_sha256": previous,
    }:
        raise OrchestrationError("attempt log current pointer mismatch")


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


def parse_gpus(value: str | Sequence[int]) -> tuple[int, int, int]:
    parts = value.split(",") if isinstance(value, str) else tuple(value)
    try:
        result = tuple(int(item) for item in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "GPUs must be three comma-separated nonnegative IDs"
        ) from error
    if len(result) != 3 or len(set(result)) != 3 or any(item < 0 for item in result):
        raise ValueError("GPUs must be three distinct nonnegative IDs")
    return result  # type: ignore[return-value]


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
            try:
                handle.kill()
            except BaseException:
                pass
            try:
                handle.wait()
            except BaseException:
                pass


def _run_parallel(
    runner: ProcessRunner,
    jobs: Sequence[tuple[str, Sequence[str], int, Path]],
) -> None:
    active: list[tuple[str, ProcessHandle, Path]] = []
    try:
        for name, command, gpu, log in jobs:
            handle = runner.start(command, env=_environment(gpu), log_path=log)
            active.append((name, handle, log))
    except BaseException:
        _terminate_handles(tuple(handle for _, handle, _ in active))
        raise
    failures: list[tuple[str, int, Path]] = []
    pending = list(active)
    try:
        while pending:
            next_pending = []
            for name, handle, log in pending:
                code = handle.poll()
                if code is None:
                    next_pending.append((name, handle, log))
                elif code:
                    failures.append((name, code, log))
            if failures:
                _terminate_handles(tuple(handle for _, handle, _ in next_pending))
                break
            pending = next_pending
            if pending:
                time.sleep(0.05)
    except BaseException:
        _terminate_handles(tuple(handle for _, handle, _ in pending))
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
    gpus: tuple[int, int, int],
    python_executable: str,
) -> dict[str, object]:
    if not config.is_file():
        raise OrchestrationError(f"protocol config is missing: {config}")
    if shutil.which(python_executable) is None:
        raise OrchestrationError("selected Python executable is unavailable")
    if shutil.which("conda") is None:
        raise OrchestrationError("conda is unavailable for the AudioGS preflight")
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        available = {
            int(line.split(",", 1)[0].strip()): int(line.split(",", 1)[1].strip())
            for line in query.splitlines()
            if line.strip()
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise OrchestrationError(f"cannot verify assigned GPUs: {error}") from error
    if any(gpu not in available or available[gpu] <= 0 for gpu in gpus):
        raise OrchestrationError("assigned GPU is unavailable or has no free memory")
    runtime_probes = _probe_gpu_runtimes(gpus, python_executable)
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


def _native_jobs(
    repository: Path,
    log_root: Path,
    gpus: tuple[int, int, int],
    *,
    preflight_only: bool,
    scene: str | None = None,
) -> list[tuple[str, tuple[str, ...], int, Path]]:
    mode = "--preflight-only" if preflight_only else "--execute"
    scenes = (scene,) if scene is not None else SCENES
    jobs = [
        (
            f"ftgspp_{item}",
            (
                "bash",
                str(repository / "scripts" / "prepare_ftgspp_cam38_baselines.sh"),
                mode,
                "--scene",
                item,
            ),
            gpus[index],
            log_root
            / f"native_{'preflight_' if preflight_only else ''}ftgspp_{item}.log",
        )
        for index, item in enumerate(scenes)
    ]
    jobs.append(
        (
            "audiogs",
            (
                "bash",
                str(repository / "scripts" / "train_audiogs_cam38_baselines.sh"),
                mode,
                *(("--scene", scene) if scene is not None else ()),
            ),
            gpus[2],
            log_root / f"native_{'preflight_' if preflight_only else ''}audiogs.log",
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


def _verify_scene_snapshot(output_dir: Path, *, repository: Path, scene: str) -> None:
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
    _verify_logs(output_dir, scene)
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


def _inspect_partial_resume(
    output: Path,
    *,
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
            _verify_logs(pinned, scene)
        protocol = pinned / "protocol"
        preparation = protocol / "preparation.json"
        if protocol.exists():
            if not preparation.is_file() or preparation.is_symlink():
                raise OrchestrationError("partial protocol is not committed")
            _verify_preparation(preparation, scene, config_path)
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
                    progress = pinned_worker / "progress.json"
                    if not progress.is_file() and "final.pt" not in entries:
                        raise OrchestrationError(
                            f"{mode} has no committed resume checkpoint"
                        )
                    resume_modes.add(mode)
        evaluations = pinned / "evaluations"
        if evaluations.exists():
            for system_dir in evaluations.iterdir():
                if system_dir.name not in {*NATIVE_SYSTEMS, *CONTINUATION_SYSTEMS}:
                    raise OrchestrationError(
                        "partial evaluation set has an extra system"
                    )
                for artifact in system_dir.iterdir():
                    if not (artifact / "current.json").is_file():
                        raise OrchestrationError(
                            "partial evaluation is not an immutable generation"
                        )
                    step = (
                        None
                        if artifact.name == "native"
                        else int(artifact.name.removeprefix("step_"))
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
            _verify_logs(pinned, "suite")
        for scene in SCENES:
            scene_root = pinned / scene
            if not scene_root.exists():
                continue
            try:
                verify_scene_outputs(scene_root, repository=repository, scene=scene)
            except Exception:
                _inspect_partial_resume(
                    scene_root,
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
    log_root: Path,
    config: Path,
    python: str,
    gpus: tuple[int, int, int],
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
    for mode, gpu in zip(GPU_ORDER, gpus, strict=True):
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
        (name, step, output / "workers" / name)
        for name in sorted(CONTINUATION_SYSTEMS)
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
        ]
        if step is not None:
            command += ["--step", str(step)]
        if (destination / "current.json").is_file():
            command.append("--resume")
        gpu = gpus[index % 3]
        evals.append((label, command, gpu, log_root / f"eval_{label}.log"))
    return prepare, workers, evals


def run_scene_benchmark(
    *,
    config_path: Path,
    output_dir: Path,
    gpus: Sequence[int] = (0, 1, 2),
    python_executable: str = sys.executable,
    resume: bool = False,
    verify_only: bool = False,
    skip_native_training: bool = False,
    runner: ProcessRunner | None = None,
    _verifier: Callable[..., SceneBenchmarkResult] = verify_scene_outputs,
    _preflight_fn: Callable[..., Mapping[str, object]] = _preflight,
) -> SceneBenchmarkResult:
    config_path = Path(config_path).absolute()
    repository = config_path.parent.parent.parent
    scene = config_path.stem
    if scene not in SCENES:
        raise ValueError("strict suite supports only scene1_opera and Scene7playing")
    output = Path(output_dir).absolute()
    resume_modes: frozenset[str] = frozenset()
    resume_snapshot: str | None = None
    if verify_only:
        return _verifier(output, repository=repository, scene=scene)
    if resume and (output / "report" / "current.json").is_file():
        # A published final report declares a complete tree. Any verification
        # failure is corruption/incompatibility and must fail closed.
        return _verifier(output, repository=repository, scene=scene)
    if resume and output.exists():
        try:
            return _verifier(output, repository=repository, scene=scene)
        except Exception:
            resume_modes, resume_snapshot = _inspect_partial_resume(
                output, scene=scene, config_path=config_path
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
        for offset in range(0, len(evaluations), 3):
            _run_parallel(runner, evaluations[offset : offset + 3])
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
            str(child_output / "evaluations"),
            "--output-dir",
            str(child_output / "report"),
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


def run_benchmark_suite(
    *,
    repository: Path,
    output_dir: Path,
    gpus: Sequence[int] = (0, 1, 2),
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
    if verify_only or (resume and (output / "report" / "current.json").is_file()):
        with BenchmarkOutputReadLock(output) as pinned:
            if {path.name for path in pinned.iterdir()} != {
                ".benchmark.lock",
                *SCENES,
                "logs",
                "report",
            }:
                raise OrchestrationError("suite output root inventory mismatch")
            _verify_logs(pinned, "suite")
            for scene in SCENES:
                verify_scene_outputs(output / scene, repository=repository, scene=scene)
            return dict(_suite_verifier(output / "report"))
    if resume and output.exists():
        resume_snapshot = _inspect_partial_suite(output, repository=repository)
    if output.exists() and not resume and any(output.iterdir()):
        raise OrchestrationError("suite output exists; pass --resume")
    devices = parse_gpus(gpus)
    runner = runner or SubprocessRunner()
    native_complete = _native_contracts_valid(repository, SCENES)
    if skip_native_training or resume:
        if not native_complete:
            raise OrchestrationError(
                "native baseline stage is incomplete; complete both immutable "
                "native contracts before suite resume/reuse"
            )
        for scene in SCENES:
            _require_native_contracts(repository, scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Suite-wide Python/GPU/disk/source validation completes for both scenes
    # before any native trainer is allowed to start.
    for scene in SCENES:
        _preflight_fn(
            repository,
            repository / "configs" / "benchmark_cam38" / f"{scene}.yaml",
            output.parent,
            devices,
            python_executable,
        )
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
                _native_jobs(repository, attempt_logs, devices, preflight_only=True),
            )
            _run_parallel(
                runner,
                _native_jobs(repository, attempt_logs, devices, preflight_only=False),
            )
        results = [
            _scene_runner(
                config_path=repository
                / "configs"
                / "benchmark_cam38"
                / f"{scene}.yaml",
                output_dir=child_output / scene,
                gpus=devices,
                python_executable=python_executable,
                resume=resume,
                verify_only=False,
                skip_native_training=True,
                runner=runner,
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

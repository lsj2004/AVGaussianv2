"""Resumable process orchestration for the strict cam38 benchmark."""

from __future__ import annotations

import json
import os
import shutil
import signal
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
from avgaussianv2.benchmark.native import verify_native_contract
from avgaussianv2.benchmark.output import BenchmarkOutputLock
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


class OrchestrationError(RuntimeError):
    pass


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
            prior = log_path.with_suffix(log_path.suffix + ".previous")
            prior.unlink(missing_ok=True)
            os.replace(log_path, prior)
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
    events = root / "status_events"
    events.mkdir(exist_ok=True)
    sequence = len(tuple(events.glob("*.json")))
    event = {
        "schema": SCHEMA,
        "version": 1,
        "sequence": sequence,
        "scene_id": scene,
        "phase": phase,
        "detail": detail,
        "time_ns": time.time_ns(),
    }
    _atomic_json(events / f"{sequence:06d}.json", event)
    _atomic_json(root / "status.json", event)


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
    try:
        dependency_probe = subprocess.run(
            [
                python_executable,
                "-c",
                "import json,numpy,torch,yaml;"
                "print(json.dumps({'python':__import__('sys').executable,"
                "'torch':torch.__version__,'cuda':torch.version.cuda}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dependencies = json.loads(dependency_probe)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise OrchestrationError(
            f"selected Python lacks production dependencies: {error}"
        ) from error
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
    free = shutil.disk_usage(output_parent).free
    if free < MIN_FREE_BYTES:
        raise OrchestrationError(
            f"insufficient checkpoint write budget: {free} < {MIN_FREE_BYTES} bytes"
        )
    source_files = (
        repository / "avgaussianv2" / "benchmark" / "training.py",
        repository / "avgaussianv2" / "benchmark" / "evaluation.py",
        repository / "avgaussianv2" / "benchmark" / "orchestration.py",
        config,
    )
    return {
        "gpus": list(gpus),
        "free_bytes": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "dependencies": {
            **dependencies,
            "modules": ["torch", "yaml", "numpy"],
        },
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


def verify_scene_outputs(
    output_dir: Path, *, repository: Path, scene: str
) -> SceneBenchmarkResult:
    """Recursively verify a scene without construction, process launch, or writes."""
    preparation_path = output_dir / "protocol" / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    if (
        preparation.get("schema") != "avgaussianv2.cam38-production-preparation"
        or preparation.get("scene_id") != scene
        or preparation.get("include_eval") is not False
        or sha256_file(Path(preparation["resolved_config"]))
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
    for system in NATIVE_SYSTEMS:
        verify_evaluation(
            evaluations / system / "native",
            identity=expected_identity(scene, system, None),
        )
    for system in CONTINUATION_SYSTEMS:
        for step in REPORTING_STEPS:
            verify_evaluation(
                evaluations / system / f"step_{step:06d}",
                identity=expected_identity(scene, system, step),
            )
    verify_scene_report(output_dir / "report")
    return SceneBenchmarkResult(scene, output_dir, output_dir / "report", True)


def _verify_preparation(path: Path, scene: str, source_config: Path) -> None:
    raw = json.loads(path.read_text())
    if (
        raw.get("schema") != "avgaussianv2.cam38-production-preparation"
        or raw.get("version") != 1
        or raw.get("scene_id") != scene
        or raw.get("include_eval") is not False
        or sha256_file(Path(raw["resolved_config"]))
        != raw.get("resolved_config_sha256")
        or set(raw.get("worker_manifests", {})) != set(GPU_ORDER)
        or set(raw.get("native_contracts", {})) != {"audiogs", "ftgspp"}
    ):
        raise OrchestrationError("existing preparation is incompatible")
    origin = json.loads(
        Path(raw["resolved_config"])
        .with_name("resolved_project.origin.json")
        .read_text()
    )
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
    for mode, manifest in raw["worker_manifests"].items():
        payload = json.loads(Path(manifest).read_text())
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


def _scene_commands(
    *,
    repository: Path,
    output: Path,
    config: Path,
    python: str,
    gpus: tuple[int, int, int],
    resume: bool,
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
        if resume:
            command.append("--resume")
        workers.append((mode, command, gpu, output / "logs" / f"worker_{mode}.log"))
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
        if resume and (destination / "current.json").is_file():
            command.append("--resume")
        gpu = gpus[index % 3]
        evals.append((label, command, gpu, output / "logs" / f"eval_{label}.log"))
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
            # An incomplete tree is expected after interruption. The strict
            # phase-specific checks below still fail closed on incompatibility.
            pass
    devices = parse_gpus(gpus)
    if output.exists() and not resume and any(output.iterdir()):
        raise OrchestrationError("scene output exists; pass --resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    runner = runner or SubprocessRunner()
    with BenchmarkOutputLock(output):
        preflight = _preflight_fn(
            repository,
            config_path,
            output.parent,
            devices,
            python_executable,
        )
        _atomic_json(output / "preflight.json", dict(preflight))
        _record_status(output, scene=scene, phase="preflight", detail="complete")
        if skip_native_training or (
            resume and _native_contracts_valid(repository, (scene,))
        ):
            _require_native_contracts(repository, scene)
        elif resume:
            raise OrchestrationError(
                "native baseline stage is incomplete; its upstream trainers have no "
                "exact-resume contract, so repair/complete Task11 native training "
                "before resuming continuations"
            )
        else:
            for script in (
                repository / "scripts" / "train_audiogs_cam38_baselines.sh",
                repository / "scripts" / "prepare_ftgspp_cam38_baselines.sh",
            ):
                _run_one(
                    runner,
                    ("bash", str(script), "--execute", "--scene", scene),
                    gpu=-1,
                    log=output / "logs" / f"native_{script.stem}.log",
                )
            _require_native_contracts(repository, scene)
        _record_status(output, scene=scene, phase="native", detail="verified")
        prepare, workers, evaluations = _scene_commands(
            repository=repository,
            output=output,
            config=config_path,
            python=python_executable,
            gpus=devices,
            resume=resume,
        )
        preparation = output / "protocol" / "preparation.json"
        if not (resume and preparation.is_file()):
            _run_one(
                runner,
                prepare,
                gpu=-1,
                log=output / "logs" / "prepare.log",
            )
        else:
            _verify_preparation(preparation, scene, config_path)
        _record_status(output, scene=scene, phase="prepare", detail="complete")
        _run_parallel(runner, workers)
        _record_status(output, scene=scene, phase="training", detail="complete")
        for offset in range(0, len(evaluations), 3):
            _run_parallel(runner, evaluations[offset : offset + 3])
        _record_status(output, scene=scene, phase="evaluation", detail="complete")
        report = [
            python_executable,
            "-m",
            "avgaussianv2.cli.benchmark_report",
            "--kind",
            "scene",
            "--scene",
            scene,
            "--evaluations-root",
            str(output / "evaluations"),
            "--output-dir",
            str(output / "report"),
        ]
        if resume and (output / "report" / "current.json").is_file():
            report.append("--resume")
        _run_one(runner, report, gpu=devices[0], log=output / "logs" / "report.log")
        _record_status(output, scene=scene, phase="report", detail="complete")
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
) -> dict[str, object]:
    repository = Path(repository).absolute()
    output = Path(output_dir).absolute()
    if verify_only:
        for scene in SCENES:
            verify_scene_outputs(output / scene, repository=repository, scene=scene)
        return dict(_suite_verifier(output / "report"))
    if resume and (output / "report" / "current.json").is_file():
        for scene in SCENES:
            verify_scene_outputs(output / scene, repository=repository, scene=scene)
        return dict(_suite_verifier(output / "report"))
    devices = parse_gpus(gpus)
    runner = runner or SubprocessRunner()
    output.parent.mkdir(parents=True, exist_ok=True)
    with BenchmarkOutputLock(output):
        # Native scripts cover both scenes. Run them once, then every scene
        # requires the resulting contracts and cannot silently retrain.
        native_complete = _native_contracts_valid(repository, SCENES)
        if resume and not skip_native_training and not native_complete:
            raise OrchestrationError(
                "native baseline stage is incomplete; complete both immutable "
                "native contracts before suite resume"
            )
        if not skip_native_training and not native_complete:
            for script in (
                repository / "scripts" / "train_audiogs_cam38_baselines.sh",
                repository / "scripts" / "prepare_ftgspp_cam38_baselines.sh",
            ):
                _run_one(
                    runner,
                    ("bash", str(script), "--execute"),
                    gpu=-1,
                    log=output / "logs" / f"native_{script.stem}.log",
                )
        results = [
            _scene_runner(
                config_path=repository
                / "configs"
                / "benchmark_cam38"
                / f"{scene}.yaml",
                output_dir=output / scene,
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
            str(output / "report"),
        ]
        for result in results:
            command += ["--scene-report", str(result.report_dir)]
        if resume and (output / "report" / "current.json").is_file():
            command.append("--resume")
        _run_one(
            runner,
            command,
            gpu=devices[0],
            log=output / "logs" / "suite_report.log",
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

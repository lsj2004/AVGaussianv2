"""Two-GPU execution for generated LRE screening/confirmation manifests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from avgaussianv2.benchmark.orchestration import (
    MAX_IDLE_GPU_UTILIZATION_PERCENT,
    MIN_GPU_FREE_MIB,
    OrchestrationError,
    ProcessHandle,
    ProcessRunner,
    SubprocessRunner,
    _environment,
    _terminate_handles,
    parse_gpus,
)
from avgaussianv2.benchmark.evaluation import EVALUATION_CONTINUATION_SYSTEMS


SCHEMA = "avgaussianv2.lre-loss-runner-result"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def _publish_result(path: Path, value: object) -> Path:
    history = path.parent / "result_history" / path.stem
    sequence = 0
    while (history / f"attempt-{sequence:06d}.json").exists():
        sequence += 1
    immutable = history / f"attempt-{sequence:06d}.json"
    _atomic_json(immutable, value)
    _atomic_json(path, value)
    return immutable


def load_lre_run_manifest(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read LRE run manifest: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != "avgaussianv2.lre-loss-run-manifest"
        or value.get("version") != 1
        or value.get("stage")
        not in {"smoke", "architecture", "screening", "confirmation", "robustness"}
        or not isinstance(value.get("configs"), list)
        or not isinstance(value.get("runs"), list)
    ):
        raise ValueError("unsupported LRE run manifest")
    configs: dict[str, Mapping[str, object]] = {}
    for record in value["configs"]:
        if not isinstance(record, Mapping):
            raise ValueError("LRE config record must be a mapping")
        config_id = record.get("config_id")
        config_path = record.get("config")
        digest = record.get("config_sha256")
        if (
            not isinstance(config_id, str)
            or config_id in configs
            or not isinstance(config_path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("invalid or duplicate LRE config record")
        resolved = Path(config_path).resolve()
        if not resolved.is_file() or _sha256(resolved) != digest:
            raise ValueError(f"LRE config hash mismatch: {config_id}")
        configs[config_id] = record
    run_ids: set[str] = set()
    continuation_ids: set[str] = set()
    for run in value["runs"]:
        if not isinstance(run, Mapping):
            raise ValueError("LRE run record must be a mapping")
        run_id = run.get("run_id")
        continuation_id = run.get("continuation_id")
        config_id = run.get("config_id")
        report_steps = run.get("report_steps")
        evaluation_systems = run.get("evaluation_systems", [run.get("system")])
        max_steps = run.get("max_steps")
        stop_after = run.get("stop_after_step")
        if (
            not isinstance(run_id, str)
            or run_id in run_ids
            or not isinstance(continuation_id, str)
            or not continuation_id
            or continuation_id in continuation_ids
            or run.get("stage") != value["stage"]
            or config_id not in configs
            or run.get("scene") not in {"scene1_opera", "Scene7playing"}
            or run.get("training_mode") not in {"audio_only", "joint_conditioned"}
            or not isinstance(evaluation_systems, list)
            or not evaluation_systems
            or evaluation_systems[0] != run.get("system")
            or len(set(evaluation_systems)) != len(evaluation_systems)
            or any(not isinstance(system, str) or not system for system in evaluation_systems)
            or any(system not in EVALUATION_CONTINUATION_SYSTEMS for system in evaluation_systems)
            or not isinstance(report_steps, list)
            or not report_steps
            or any(
                not isinstance(step, int) or isinstance(step, bool) or step <= 0
                for step in report_steps
            )
            or report_steps != sorted(set(report_steps))
            or max_steps != report_steps[-1]
            or (
                value["stage"] == "smoke"
                and max_steps != 5_000
            )
            or (
                value["stage"] != "smoke"
                and max_steps not in {5_000, 10_000, 30_000}
            )
            or stop_after
            != (max_steps if max_steps < 30_000 else None)
        ):
            raise ValueError(f"invalid LRE run record: {run_id!r}")
        config = configs[str(config_id)]
        for field in ("scene", "system", "training_mode", "seed", "lambda_lre"):
            if run.get(field) != config.get(field):
                raise ValueError(f"LRE run/config mismatch for {run_id}: {field}")
        run_ids.add(run_id)
        continuation_ids.add(continuation_id)
    return value


@dataclass(frozen=True)
class LREStage:
    name: str
    command: tuple[str, ...]
    log: Path


@dataclass(frozen=True)
class LREPipeline:
    run_id: str
    stage: str
    run_dir: Path
    stages: tuple[LREStage, ...]


def _next_attempt_log_dir(run_dir: Path) -> Path:
    logs = run_dir / "logs"
    sequence = 0
    while (logs / f"attempt-{sequence:06d}").exists():
        sequence += 1
    return logs / f"attempt-{sequence:06d}"


def _worker_progress(worker: Path) -> int | None:
    try:
        value = json.loads((worker / "progress.json").read_text())
        step = value["exact_main_step"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def _bind_continuation_identity(
    run_dir: Path,
    run: Mapping[str, object],
    config_record: Mapping[str, object],
) -> None:
    identity = {
        "schema": "avgaussianv2.lre-loss-continuation-identity",
        "version": 1,
        "continuation_id": run["continuation_id"],
        "config_sha256": config_record["config_sha256"],
        "scene": run["scene"],
        "system": run["system"],
        "training_mode": run["training_mode"],
        "seed": run["seed"],
        "lambda_lre": run["lambda_lre"],
    }
    path = run_dir / "continuation_identity.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise OrchestrationError(
                f"cannot read LRE continuation identity: {path}: {error}"
            ) from error
        if existing != identity:
            raise OrchestrationError(
                f"LRE continuation identity mismatch; refusing directory reuse: {run_dir}"
            )
        return
    existing_entries = tuple(run_dir.iterdir()) if run_dir.is_dir() else ()
    if existing_entries:
        raise OrchestrationError(
            f"nonempty LRE continuation lacks identity; refusing directory reuse: {run_dir}"
        )
    _atomic_json(path, identity)


def build_lre_pipelines(
    manifest: Mapping[str, object],
    *,
    output_root: Path,
    native_root: Path,
    python_executable: str,
    compute_dpam: bool,
    trust_upstream_artifacts: bool,
    resume: bool,
) -> tuple[LREPipeline, ...]:
    configs = {
        str(record["config_id"]): record
        for record in manifest["configs"]  # type: ignore[index]
    }
    pipelines = []
    for run in manifest["runs"]:  # type: ignore[index]
        run_id = str(run["run_id"])
        run_dir = Path(output_root).resolve() / str(run["continuation_id"])
        if run_dir.exists() and not resume and any(run_dir.iterdir()):
            raise FileExistsError(f"LRE run already exists; pass --resume: {run_dir}")
        config_record = configs[str(run["config_id"])]
        protocol = run_dir / "protocol"
        worker = run_dir / "worker"
        evaluation_root = run_dir / "evaluations"
        config = Path(str(config_record["config"])).resolve()
        scene = str(run["scene"])
        system = str(run["system"])
        training_mode = str(run["training_mode"])
        native_scene = Path(native_root).resolve() / scene
        audiogs = native_scene / "audiogs" / "native_contract"
        ftgspp = native_scene / "ftgspp" / "native_contract"
        if not audiogs.is_dir() or not ftgspp.is_dir():
            raise FileNotFoundError(f"native contracts are incomplete for {scene}")
        _bind_continuation_identity(run_dir, run, config_record)
        evaluation_root.mkdir(parents=True, exist_ok=True)
        log_dir = _next_attempt_log_dir(run_dir)
        stages: list[LREStage] = []
        common_trust = ("--trust-upstream-artifacts",) if trust_upstream_artifacts else ()
        if not (protocol / "preparation.json").is_file():
            stages.append(
                LREStage(
                    "prepare",
                    (
                        python_executable,
                        "-m",
                        "avgaussianv2.cli.benchmark_prepare",
                        "--config",
                        str(config),
                        "--output-dir",
                        str(protocol),
                        "--devices",
                        "cuda:0",
                        "--native-audiogs-contract",
                        str(audiogs),
                        "--native-ftgspp-contract",
                        str(ftgspp),
                        *common_trust,
                    ),
                    log_dir / "prepare.log",
                )
            )
        maximum = int(run["max_steps"])
        current = _worker_progress(worker)
        worker_complete = (
            (maximum == 30_000 and (worker / "artifact_hashes.json").is_file())
            or (maximum < 30_000 and current is not None and current >= maximum)
        )
        if not worker_complete:
            worker_command = [
                python_executable,
                "-m",
                "avgaussianv2.cli.benchmark_worker",
                "--manifest",
                str(protocol / "worker_manifests" / f"{training_mode}.json"),
                "--config",
                str(protocol / "resolved_project.yaml"),
                "--output-dir",
                str(worker),
                "--device",
                "cuda:0",
                *common_trust,
            ]
            if worker.exists() and any(worker.iterdir()):
                worker_command.append("--resume")
            if run["stop_after_step"] is not None:
                worker_command.extend(
                    ("--stop-after-step", str(run["stop_after_step"]))
                )
            stages.append(
                LREStage("train", tuple(worker_command), log_dir / "train.log")
            )
        evaluation_systems = run.get("evaluation_systems", [system])
        for evaluation_system in evaluation_systems:
            for step in run["report_steps"]:
                evaluation = (
                    evaluation_root / f"step_{int(step):06d}"
                    if evaluation_system == system
                    else evaluation_root
                    / str(evaluation_system)
                    / f"step_{int(step):06d}"
                )
                command = [
                    python_executable,
                    "-m",
                    "avgaussianv2.cli.benchmark_eval",
                    "--scene",
                    scene,
                    "--system",
                    str(evaluation_system),
                    "--step",
                    str(step),
                    "--resolved-config",
                    str(protocol / "resolved_project.yaml"),
                    "--source",
                    str(worker),
                    "--output-dir",
                    str(evaluation),
                    "--device",
                    "cuda:0",
                    *common_trust,
                ]
                if compute_dpam:
                    command.append("--compute-dpam")
                if (evaluation / "current.json").is_file():
                    command.append("--verify-only")
                stage_name = (
                    f"eval_{int(step):06d}"
                    if evaluation_system == system
                    else f"eval_{evaluation_system}_{int(step):06d}"
                )
                stages.append(
                    LREStage(
                        stage_name,
                        tuple(command),
                        log_dir / f"{stage_name}.log",
                    )
                )
        pipelines.append(
            LREPipeline(run_id, str(manifest["stage"]), run_dir, tuple(stages))
        )
    return tuple(pipelines)


def query_idle_gpus(
    gpus: Sequence[int],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[int, dict[str, int]]:
    completed = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    status = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, free_mib, utilization = (
            int(value.strip()) for value in line.split(",")
        )
        status[index] = {"free_mib": free_mib, "utilization_percent": utilization}
    rejected = {
        gpu: status.get(gpu)
        for gpu in gpus
        if gpu not in status
        or status[gpu]["free_mib"] < MIN_GPU_FREE_MIB
        or status[gpu]["utilization_percent"]
        > MAX_IDLE_GPU_UTILIZATION_PERCENT
    }
    if rejected:
        raise OrchestrationError(
            f"LRE GPUs are busy or below the free-memory threshold: {rejected}"
        )
    return {gpu: status[gpu] for gpu in gpus}


def sample_gpu_usage(
    gpus: Sequence[int],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[int, dict[str, int]]:
    completed = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wanted = set(gpus)
    result = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, used_mib, utilization = (
            int(value.strip()) for value in line.split(",")
        )
        if index in wanted:
            result[index] = {
                "memory_used_mib": used_mib,
                "utilization_percent": utilization,
            }
    if set(result) != wanted:
        raise OrchestrationError("cannot sample every active LRE GPU")
    return result


def execute_lre_pipelines(
    pipelines: Sequence[LREPipeline],
    *,
    gpus: Sequence[int],
    runner: ProcessRunner | None = None,
    poll_seconds: float = 0.1,
    usage_sampler: Callable[[Sequence[int]], Mapping[int, Mapping[str, int]]] | None = None,
) -> dict[str, object]:
    runner = runner or SubprocessRunner()
    pending = list(pipelines)
    active: dict[int, tuple[LREPipeline, int, ProcessHandle, int]] = {}
    completed = []
    usage: dict[str, dict[str, int]] = {}
    last_sample = 0.0
    try:
        while pending or active:
            free = [gpu for gpu in gpus if gpu not in active]
            while free and pending:
                gpu = free.pop(0)
                pipeline = pending.pop(0)
                if not pipeline.stages:
                    completed.append(
                        {"run_id": pipeline.run_id, "gpu": None, "stages": []}
                    )
                    continue
                stage = pipeline.stages[0]
                handle = runner.start(
                    stage.command, env=_environment(gpu), log_path=stage.log
                )
                active[gpu] = (pipeline, 0, handle, time.time_ns())
                usage[pipeline.run_id] = {
                    "peak_memory_used_mib": 0,
                    "maximum_gpu_utilization_percent": 0,
                }
            now = time.monotonic()
            if usage_sampler is not None and active and now - last_sample >= 1.0:
                snapshot = usage_sampler(tuple(active))
                for gpu, (pipeline, _, _, _) in active.items():
                    observed = snapshot[gpu]
                    metrics = usage[pipeline.run_id]
                    metrics["peak_memory_used_mib"] = max(
                        metrics["peak_memory_used_mib"],
                        int(observed["memory_used_mib"]),
                    )
                    metrics["maximum_gpu_utilization_percent"] = max(
                        metrics["maximum_gpu_utilization_percent"],
                        int(observed["utilization_percent"]),
                    )
                last_sample = now
            for gpu, (pipeline, index, handle, started) in tuple(active.items()):
                code = handle.poll()
                if code is None:
                    continue
                if code:
                    _terminate_handles(
                        (handle, *(item[2] for key, item in active.items() if key != gpu))
                    )
                    raise OrchestrationError(
                        f"LRE stage failed: run={pipeline.run_id} "
                        f"stage={pipeline.stages[index].name} code={code} "
                        f"log={pipeline.stages[index].log}"
                    )
                next_index = index + 1
                if next_index < len(pipeline.stages):
                    stage = pipeline.stages[next_index]
                    next_handle = runner.start(
                        stage.command, env=_environment(gpu), log_path=stage.log
                    )
                    active[gpu] = (pipeline, next_index, next_handle, started)
                else:
                    del active[gpu]
                    record = {
                        "run_id": pipeline.run_id,
                        "gpu": gpu,
                        "elapsed_seconds": (time.time_ns() - started) / 1e9,
                        "stages": [stage.name for stage in pipeline.stages],
                        **usage[pipeline.run_id],
                    }
                    _publish_result(
                        pipeline.run_dir / f"run_result.{pipeline.stage}.json",
                        record,
                    )
                    completed.append(record)
            if pending or active:
                time.sleep(poll_seconds)
    except BaseException:
        _terminate_handles(tuple(item[2] for item in active.values()))
        raise
    return {"schema": SCHEMA, "version": 1, "runs": completed}


def run_lre_manifest(
    manifest_path: Path,
    *,
    output_root: Path,
    native_root: Path,
    gpus: str | Sequence[int],
    python_executable: str,
    compute_dpam: bool = True,
    trust_upstream_artifacts: bool = False,
    resume: bool = False,
    runner: ProcessRunner | None = None,
    gpu_query: Callable[[Sequence[int]], Mapping[int, object]] = query_idle_gpus,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_lre_run_manifest(manifest_path)
    devices = parse_gpus(gpus)
    gpu_status = dict(gpu_query(devices))
    pipelines = build_lre_pipelines(
        manifest,
        output_root=output_root,
        native_root=native_root,
        python_executable=python_executable,
        compute_dpam=compute_dpam,
        trust_upstream_artifacts=trust_upstream_artifacts,
        resume=resume,
    )
    result = execute_lre_pipelines(
        pipelines,
        gpus=devices,
        runner=runner,
        usage_sampler=sample_gpu_usage if runner is None else None,
    )
    result.update(
        source_manifest=str(manifest_path),
        source_manifest_sha256=_sha256(manifest_path),
        stage=manifest["stage"],
        gpus=list(devices),
        gpu_preflight=gpu_status,
    )
    _publish_result(
        Path(output_root).resolve() / f"runner_result.{manifest['stage']}.json",
        result,
    )
    return result


__all__ = [
    "LREPipeline",
    "LREStage",
    "build_lre_pipelines",
    "execute_lre_pipelines",
    "load_lre_run_manifest",
    "query_idle_gpus",
    "run_lre_manifest",
    "sample_gpu_usage",
]

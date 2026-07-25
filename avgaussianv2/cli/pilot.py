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
import stat
import subprocess
import sys
import tempfile
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


class PilotProcessError(RuntimeError):
    def __init__(self, failures: Sequence[tuple[str, int, Path]]) -> None:
        self.failures = tuple(failures)
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

    def wait(self) -> int:
        try:
            return self._process.wait()
        finally:
            self._stream.close()

    def terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()


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
    jobs: Sequence[tuple[str, ProcessHandle, Path]]
) -> None:
    failures: list[tuple[str, int, Path]] = []
    try:
        for name, handle, log in jobs:
            code = handle.wait()
            if code:
                failures.append((name, code, log))
    except BaseException:
        for _, handle, _ in jobs:
            handle.terminate()
        for _, handle, _ in jobs:
            try:
                handle.wait()
            except BaseException:
                pass
        raise
    if failures:
        raise PilotProcessError(failures)


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
            handle.terminate()
        for _, handle, _ in jobs:
            try:
                handle.wait()
            except BaseException:
                pass
        raise
    except BaseException:
        failures: list[tuple[str, int, Path]] = []
        for _, handle, _ in jobs:
            handle.terminate()
        for name, handle, log in jobs:
            try:
                code = handle.wait()
            except BaseException:
                code = -1
            if code:
                failures.append((name, code, log))
        failed_spec = specs[len(jobs)]
        failures.append((failed_spec.name, -1, failed_spec.log_path))
        raise PilotProcessError(failures)
    _wait_group(jobs)


def _command_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
    config_source = Path(config_path)
    config_bytes = _read_regular(config_source)
    config = load_project_config_bytes(config_bytes, base_dir=config_source.parent)
    if config.scene.scene_id != "scene1_opera":
        raise ValueError("three-GPU pilot requires exact scene_id scene1_opera")
    output = Path(output_dir).absolute()
    _verify_output_ancestors(output)
    if output.exists():
        metadata = output.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("pilot output must be a non-symlink directory")
        entries = [item for item in output.iterdir() if item.name != ".pilot-parent.lock"]
        if entries and not (resume or verify_only):
            raise FileExistsError("fresh pilot refuses nonempty output")
    else:
        if verify_only:
            raise FileNotFoundError("verify-only requires an existing output")
        output.mkdir(parents=True)
    if runner is None:
        runner = SubprocessRunner()
    if gpu_validator is None and not verify_only:
        import torch

        def gpu_validator(ids: tuple[int, int, int]) -> None:
            if not torch.cuda.is_available() or max(ids) >= torch.cuda.device_count():
                raise ValueError("requested CUDA GPU is unavailable")
    if not verify_only:
        gpu_validator(gpus)

    lock_fd = os.open(
        output / ".pilot-parent.lock",
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("pilot output already has an active parent") from error
        experiment_path = output / "experiment_manifest.json"
        shared_path = output / "shared_manifest.json"
        logs = output / "logs"
        workers_root = output / "workers"
        eval_root = output / "evaluations"
        if verify_only:
            for required in (logs, workers_root, eval_root):
                if not required.is_dir() or required.is_symlink():
                    raise FileNotFoundError(
                        f"verify-only requires existing directory: {required}"
                    )
        else:
            logs.mkdir(exist_ok=True)
            workers_root.mkdir(exist_ok=True)
            eval_root.mkdir(exist_ok=True)

        baseline_specs = (EvaluationSpec("baseline_imported", False),)
        baseline_dir = output / "baseline"
        baseline_manifest = baseline_dir / "evaluation_manifest.json"
        if not baseline_manifest.exists():
            if verify_only:
                raise FileNotFoundError("baseline evaluation is incomplete")
            command = [
                python, "-m", "avgaussianv2.cli.pilot_eval",
                "--config", str(config_source),
                "--system", "baseline_imported:off",
                "--output-dir", str(baseline_dir),
                "--device", "cuda:0",
            ]
            if trust_upstream_artifacts:
                command.append("--trust-upstream-artifacts")
            if resume and baseline_dir.exists():
                command.append("--resume")
            _run_one(runner, "baseline", command, gpus[0], logs / "baseline.log")
        baseline_job = _verify_existing(
            baseline_dir,
            baseline_specs,
            config_path=config_source,
        )
        if not verify_only:
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
        train_length = int(baseline_raw["train_length"])
        eval_length = int(baseline_raw["eval_length"])
        shared = _shared_indices(config.train.seed, train_length, pilot)
        heldout = _evenly_spaced(eval_length, pilot.quick_validation_samples)
        desired_shared = _worker_manifest(
            config=config,
            config_bytes=config_bytes,
            pilot=pilot,
            shared=shared,
            heldout=heldout,
            baseline_path=visual_baseline,
            baseline_summary=json.loads(_read_regular(visual_baseline).decode()),
            baseline_job=baseline_raw,
            train_length=train_length,
            eval_length=eval_length,
        )
        if shared_path.exists():
            existing = json.loads(_read_regular(shared_path).decode())
            if existing != desired_shared:
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
            "shared_manifest_path": str(shared_path.resolve()),
            "shared_manifest_sha256": loaded.sha256,
            "baseline_manifest_path": str(baseline_manifest.resolve()),
            "baseline_manifest_sha256": _sha(baseline_manifest),
            "source_hashes": desired_shared["source_hashes"],
        }
        if experiment_path.exists():
            if json.loads(_read_regular(experiment_path).decode()) != experiment:
                raise ValueError("experiment manifest identity mismatch")
        elif verify_only:
            raise FileNotFoundError("experiment manifest is missing")
        else:
            _atomic_json(experiment_path, experiment)

        worker_specs: list[LaunchSpec] = []
        verified_workers: dict[Variant, object] = {}
        for variant, gpu in zip(VARIANT_GPU_ORDER, gpus, strict=True):
            worker_dir = workers_root / variant.value
            has_entries = worker_dir.is_dir() and any(worker_dir.iterdir())
            if has_entries:
                try:
                    verified_workers[variant] = verify_worker_output(
                        config_source,
                        shared_path,
                        visual_baseline,
                        worker_dir,
                        variant,
                        trust_upstream_artifacts=trust_upstream_artifacts,
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
                        trust_upstream_artifacts=trust_upstream_artifacts,
                    )
            elif verify_only:
                raise FileNotFoundError(f"{variant.value} worker is incomplete")
            command = [
                python, "-m", "avgaussianv2.cli.pilot_worker",
                "--config", str(config_source),
                "--variant", variant.value,
                "--shared-indices", str(shared_path),
                "--visual-baseline", str(visual_baseline),
                "--output-dir", str(worker_dir),
                "--device", "cuda:0",
            ]
            if resume and has_entries:
                command.append("--resume")
            if trust_upstream_artifacts:
                command.append("--trust-upstream-artifacts")
            log = logs / f"worker-{variant.value}.log"
            worker_specs.append(
                LaunchSpec(variant.value, tuple(command), gpu, log)
            )
        if worker_specs:
            _launch_group(runner, worker_specs)
        for variant in VARIANT_GPU_ORDER:
            verified_workers[variant] = verify_worker_output(
                config_source,
                shared_path,
                visual_baseline,
                workers_root / variant.value,
                variant,
                trust_upstream_artifacts=trust_upstream_artifacts,
            )
        if not verify_only:
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
            destination = eval_root / variant.value
            specs = EVAL_SPECS[variant]
            if (destination / "evaluation_manifest.json").is_file():
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
                "--checkpoint", str(workers_root / variant.value / "best.pt"),
                "--manifest", str(shared_path),
                "--variant", variant.value,
                "--output-dir", str(destination),
                "--device", "cuda:0",
            ]
            for spec in specs:
                command.extend(
                    ["--system", f"{spec.system_name}:{'on' if spec.condition_enabled else 'off'}"]
                )
            if trust_upstream_artifacts:
                command.append("--trust-upstream-artifacts")
            if resume and destination.exists():
                command.append("--resume")
            log = logs / f"eval-{variant.value}.log"
            pending.append(
                LaunchSpec(variant.value, tuple(command), gpu, log)
            )
        if pending:
            _launch_group(runner, pending)
            for variant in VARIANT_GPU_ORDER:
                eval_jobs[variant] = _verify_existing(
                    eval_root / variant.value,
                    EVAL_SPECS[variant],
                    config_path=config_source,
                    shared_manifest=shared_path,
                    variant=variant,
                )
        if not verify_only:
            _stage_status(
                output,
                baseline="complete",
                workers="complete",
                evaluations="complete",
                report="pending",
            )

        if _read_regular(config_source) != config_bytes:
            raise ValueError("project config changed during orchestration")
        if _source_hashes(config_bytes, config) != desired_shared["source_hashes"]:
            raise ValueError("upstream source changed during orchestration")
        if load_worker_manifest(
            shared_path, config_path=config_source, config=config
        ).sha256 != loaded.sha256:
            raise ValueError("shared manifest changed during orchestration")

        if verify_only:
            comparison = verify_current_comparison(
                _report_inputs(
                    output, baseline_job, eval_jobs, verified_workers
                ),
                output / "report",
            )
            status_path = output / "status.json"
            status = json.loads(_read_regular(status_path).decode())
            if set(status) != {
                "schema", "version", "stages", "report_digest", "ready",
                "durability_warnings",
            }:
                raise ValueError("status fields mismatch")
            if status["schema"] != EXPERIMENT_SCHEMA or status["version"] != EXPERIMENT_VERSION:
                raise ValueError("status schema/version mismatch")
            if status["stages"] != {
                "baseline": "complete",
                "workers": "complete",
                "evaluations": "complete",
                "report": "complete",
            }:
                raise ValueError("status stages are incomplete")
            if status.get("report_digest") != comparison.content_digest:
                raise ValueError("status/report digest mismatch")
            ready = comparison.decision.ready
            if status.get("ready") is not ready:
                raise ValueError("status/report decision mismatch")
            if status["durability_warnings"] != list(
                comparison.durability_warnings
            ):
                raise ValueError("status/report durability warnings mismatch")
            return PilotOrchestrationResult(
                experiment_path.resolve(), comparison.generation_path, ready,
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

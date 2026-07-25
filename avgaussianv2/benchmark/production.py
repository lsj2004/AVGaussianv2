"""Production adapters for the strict dual-scene cam38 benchmark.

Training construction and evaluation construction intentionally live in
different subprocesses.  The former never asks the dataset for the held-out
camera; the latter is only entered after all fixed-budget workers are verified.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
import stat

import torch
import yaml

from avgaussianv2.benchmark.assets import audit_protocol_config
from avgaussianv2.benchmark.artifacts import canonical_json, publish_generation
from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluationRuntime,
    BenchmarkPrediction,
    EvaluationIdentity,
    TrainingEvidence,
)
from avgaussianv2.benchmark.native import verify_native_contract
from avgaussianv2.benchmark.runtime import (
    BenchmarkRuntime,
    build_production_runtime,
    load_audited_benchmark_config,
    state_sha256,
    upstream_source_inventory,
)
from avgaussianv2.benchmark.training import (
    BenchmarkCompatibility,
    BenchmarkConfig,
    BenchmarkMode,
    TEST_CAMERA,
    TRAIN_CAMERAS,
    build_worker_manifest,
    hash_shared_indices,
    make_shared_indices,
)
from avgaussianv2.config import load_project_config
from avgaussianv2.runtime import build_runtime


SCENE_COUNTS = {"scene1_opera": 130, "Scene7playing": 293}


class _PinnedInput:
    """Retain and snapshot one audited input while runtime code consumes it."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.path = path
        self.expected_sha256 = expected_sha256
        self.fd: int | None = None
        self.snapshot_fd: int | None = None
        self.identity: tuple[int, int] | None = None
        self.data = b""

    def __enter__(self) -> _PinnedInput:
        self.fd = os.open(
            self.path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(self.fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            self.close()
            raise ValueError(f"evaluation input is unsafe: {self.path}")
        self.identity = (metadata.st_dev, metadata.st_ino)
        with os.fdopen(os.dup(self.fd), "rb") as stream:
            self.data = stream.read()
        if hashlib.sha256(self.data).hexdigest() != self.expected_sha256:
            self.close()
            raise ValueError(f"evaluation input hash mismatch: {self.path}")
        descriptor, snapshot_name = tempfile.mkstemp(
            prefix="avgaussianv2-evaluation-input-"
        )
        snapshot_path = Path(snapshot_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.data)
                stream.flush()
                os.fsync(stream.fileno())
            snapshot_path.chmod(0o400)
            self.snapshot_fd = os.open(
                snapshot_path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        finally:
            snapshot_path.unlink(missing_ok=True)
        return self

    @property
    def proc_path(self) -> Path:
        if self.snapshot_fd is None:
            raise RuntimeError("evaluation input is not pinned")
        return Path(f"/proc/self/fd/{self.snapshot_fd}")

    def verify(self) -> None:
        if self.fd is None or self.identity is None:
            raise RuntimeError("evaluation input is not pinned")
        retained = os.fstat(self.fd)
        current = self.path.lstat()
        os.lseek(self.fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(self.fd), "rb") as stream:
            retained_data = stream.read()
        if (
            (retained.st_dev, retained.st_ino) != self.identity
            or (current.st_dev, current.st_ino) != self.identity
            or stat.S_ISLNK(current.st_mode)
            or hashlib.sha256(retained_data).hexdigest() != self.expected_sha256
        ):
            raise RuntimeError(f"evaluation input identity changed: {self.path}")

    def close(self) -> None:
        if self.snapshot_fd is not None:
            os.close(self.snapshot_fd)
            self.snapshot_fd = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if exc_type is None:
                self.verify()
        finally:
            self.close()
        return False


class _SnapshotSourceLoader(importlib.abc.Loader):
    def __init__(self, data: bytes, origin: str) -> None:
        self.data = data
        self.origin = origin

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        module.__file__ = self.origin
        exec(compile(self.data, self.origin, "exec"), module.__dict__)


class _SnapshotSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, modules: Mapping[str, tuple[bytes, str]]) -> None:
        self.modules = modules

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        record = self.modules.get(fullname)
        if record is None:
            return None
        data, origin = record
        return importlib.util.spec_from_loader(
            fullname, _SnapshotSourceLoader(data, origin), origin=origin
        )


@contextmanager
def _snapshot_source_imports(
    pins: list[_PinnedInput], upstream_roots: tuple[Path, Path]
):
    modules = {}
    for pin in pins:
        for root in upstream_roots:
            try:
                relative = pin.path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            if relative.suffix != ".py":
                break
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            if parts:
                modules[".".join(parts)] = (pin.data, str(pin.path))
            break
    loaded = set(modules).intersection(sys.modules)
    if loaded:
        raise RuntimeError(
            f"audited upstream modules were already loaded: {sorted(loaded)}"
        )
    finder = _SnapshotSourceFinder(modules)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def _seed_evaluation_runtime(seed: int) -> None:
    """Reproduce the worker's deterministic model-construction boundary."""
    if seed != 42:
        raise ValueError("strict benchmark seed must be 42")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_resolved_project_config(source: Path, destination: Path) -> str:
    """Materialize the frozen config with absolute paths.

    This prevents a worker's cwd from changing the meaning of a relative
    checkpoint or manifest path.
    """
    audit_protocol_config(source)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    for name in ("visual_checkpoint", "audio_checkpoint", "manifest", "visual_memmap"):
        value = raw["paths"].get(name)
        if value is not None and not Path(value).is_absolute():
            raw["paths"][name] = str((source.parent / value).resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_dump(raw, sort_keys=False).encode()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    origin = {
        "schema": "avgaussianv2.cam38-resolved-config-origin",
        "version": 1,
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "resolved_sha256": hashlib.sha256(data).hexdigest(),
    }
    origin_path = destination.with_name("resolved_project.origin.json")
    origin_data = (
        json.dumps(origin, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    descriptor, origin_name = tempfile.mkstemp(
        prefix=".resolved_project.origin.", suffix=".tmp", dir=destination.parent
    )
    origin_temporary = Path(origin_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(origin_data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(origin_temporary, origin_path)
    except BaseException:
        origin_temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def prepare_worker_manifests(
    *,
    config_path: Path,
    output_dir: Path,
    devices: tuple[torch.device | str, torch.device | str, torch.device | str],
    trusted_upstream_artifacts: bool,
    native_contract_dirs: Mapping[str, Path],
    runtime_builder=build_production_runtime,
    native_verifier=verify_native_contract,
) -> dict[str, object]:
    """Construct train-only runtime evidence on every assigned device.

    No eval dataset is constructed here.  All devices must independently
    produce the same immutable identity before manifests are published.
    """
    if len({str(torch.device(item)) for item in devices}) != 3:
        raise ValueError("preparation requires three distinct devices")
    if set(native_contract_dirs) != {"audiogs", "ftgspp"}:
        raise ValueError("preparation requires exact AudioGS/FTGS++ native contracts")
    source_config = load_project_config(config_path)
    source_sha256 = sha256_file(config_path)
    native_contracts = {}
    for kind in ("audiogs", "ftgspp"):
        contract_dir = Path(native_contract_dirs[kind]).absolute()
        contract = native_verifier(
            contract_dir,
            expected_scene=source_config.scene.scene_id,
            expected_model_kind=kind,
        )
        configured_checkpoint = getattr(
            source_config.paths,
            "audio_checkpoint" if kind == "audiogs" else "visual_checkpoint",
        ).resolve()
        if (
            contract["inputs"]["protocol_config"]["sha256"] != source_sha256
            or Path(contract["checkpoint"]["path"]).resolve() != configured_checkpoint
        ):
            raise ValueError("native contract does not bind the requested protocol")
        native_contracts[kind] = {
            "path": str(contract_dir),
            "manifest_sha256": contract["_manifest_sha256"],
            "checkpoint_sha256": contract["checkpoint"]["sha256"],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = output_dir / "resolved_project.yaml"
    write_resolved_project_config(config_path, resolved)
    runtimes: list[BenchmarkRuntime] = []
    for device in devices:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        runtime = runtime_builder(
            config_path=resolved,
            device=torch.device(device),
            trusted_upstream_artifacts=trusted_upstream_artifacts,
        )
        if not isinstance(runtime, BenchmarkRuntime):
            raise TypeError("runtime builder must return BenchmarkRuntime")
        runtimes.append(runtime)
    identity_names = (
        "config_sha256",
        "source_sha256",
        "visual_initialization_sha256",
        "audio_initialization_sha256",
        "model_initialization_sha256",
        "dataset_identity_sha256",
        "dataset_sample_ids",
    )
    reference = runtimes[0]
    if any(
        any(
            getattr(runtime, name) != getattr(reference, name)
            for name in identity_names
        )
        for runtime in runtimes[1:]
    ):
        raise RuntimeError("production runtime identity differs across assigned GPUs")
    training = BenchmarkConfig()
    indices = make_shared_indices(
        len(reference.train_samples), training.main_updates, training.seed
    )
    index_sha256 = hash_shared_indices(indices)
    manifests: dict[str, str] = {}
    for mode in BenchmarkMode:
        compatibility = BenchmarkCompatibility(
            scene_id=load_project_config(resolved).scene.scene_id,
            mode=mode.value,
            train_cameras=TRAIN_CAMERAS,
            test_camera=TEST_CAMERA,
            seed=training.seed,
            index_sha256=index_sha256,
            visual_initialization_sha256=reference.visual_initialization_sha256,
            audio_initialization_sha256=reference.audio_initialization_sha256,
            model_initialization_sha256=reference.model_initialization_sha256,
            source_sha256=reference.source_sha256,
            config_sha256=reference.config_sha256,
        )
        manifest = build_worker_manifest(
            config=training,
            compatibility=compatibility,
            shared_indices=indices,
        )
        path = output_dir / "worker_manifests" / f"{mode.value}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        manifests[mode.value] = str(path.absolute())
    evidence = {
        "schema": "avgaussianv2.cam38-production-preparation",
        "version": 1,
        "scene_id": load_project_config(resolved).scene.scene_id,
        "include_eval": False,
        "devices": [str(torch.device(item)) for item in devices],
        "resolved_config": str(resolved.absolute()),
        "resolved_config_sha256": sha256_file(resolved),
        "runtime": {
            name: (
                list(getattr(reference, name))
                if name == "dataset_sample_ids"
                else getattr(reference, name)
            )
            for name in identity_names
        },
        "worker_manifests": manifests,
        "native_contracts": native_contracts,
        "checkpoint_policy": {
            "checkpoint_every": training.checkpoint_every,
            "milestones": list(training.milestones),
            "retained_periodic": 2,
        },
    }
    path = output_dir / "preparation.json"
    preparation_bytes = canonical_json(evidence)
    path.write_bytes(preparation_bytes)
    publish_generation(
        output_dir / "immutable",
        schema="avgaussianv2.cam38-production-preparation",
        identity={"scene_id": evidence["scene_id"]},
        files={
            "resolved_project.yaml": resolved.read_bytes(),
            "resolved_project.origin.json": resolved.with_name(
                "resolved_project.origin.json"
            ).read_bytes(),
            "joint_conditioned.json": Path(manifests["joint_conditioned"]).read_bytes(),
            "audio_only.json": Path(manifests["audio_only"]).read_bytes(),
            "visual_only.json": Path(manifests["visual_only"]).read_bytes(),
            "preparation.json": preparation_bytes,
        },
    )
    return evidence


def expected_identity(
    scene_id: str, system: str, step: int | None
) -> EvaluationIdentity:
    count = SCENE_COUNTS[scene_id]
    return EvaluationIdentity(
        scene_id,
        system,
        step,
        tuple(f"{scene_id}/cam38/{index:06d}" for index in range(count)),
        count,
    )


def native_training_evidence(
    contract_dir: Path, *, scene_id: str, system: str
) -> TrainingEvidence:
    kind = "audiogs" if system == "native_audiogs" else "ftgspp"
    contract = verify_native_contract(
        contract_dir, expected_scene=scene_id, expected_model_kind=kind
    )
    checkpoint = contract["checkpoint"]
    derived = contract["derived_initialization"]
    if kind == "audiogs":
        updates = int(contract["budget"]["resolved_updates"])
        epochs = 61.0
    else:
        updates = int(contract["budget"]["iterations"])
        epochs = None
    return TrainingEvidence(
        system_name=system,
        scene_id=scene_id,
        role="native_reference",
        train_cameras=TRAIN_CAMERAS,
        test_camera=TEST_CAMERA,
        test_targets_read_during_training=False,
        seed=42,
        planned_updates=updates,
        completed_updates=updates,
        checkpoint_step=updates,
        checkpoint_path=checkpoint["path"],
        checkpoint_sha256=checkpoint["sha256"],
        config_sha256=contract["inputs"]["protocol_config"]["sha256"],
        source_sha256=contract["upstream"]["source_sha256"],
        visual_initialization_sha256=derived["visual_initialization_sha256"],
        audio_initialization_sha256=derived["audio_initialization_sha256"],
        model_initialization_sha256=derived["model_initialization_sha256"],
        index_sha256=None,
        batch_size=1,
        epochs=epochs,
        native_contract_path=str(contract_dir.absolute()),
        native_contract_sha256=contract["_manifest_sha256"],
    )


def continuation_training_evidence(
    worker_dir: Path, *, scene_id: str, system: str, step: int
) -> TrainingEvidence:
    contract = json.loads((worker_dir / "contract.json").read_text())
    compatibility = BenchmarkCompatibility.from_mapping(contract["compatibility"])
    checkpoint = worker_dir / "milestones" / f"step_{step:06d}.pt"
    runtime_contract = worker_dir / "runtime_contract.json"
    return TrainingEvidence(
        system_name=system,
        scene_id=scene_id,
        role="continuation",
        train_cameras=TRAIN_CAMERAS,
        test_camera=TEST_CAMERA,
        test_targets_read_during_training=False,
        seed=42,
        planned_updates=30_000,
        completed_updates=step,
        checkpoint_step=step,
        checkpoint_path=str(checkpoint.absolute()),
        checkpoint_sha256=sha256_file(checkpoint),
        config_sha256=compatibility.config_sha256,
        source_sha256=compatibility.source_sha256,
        visual_initialization_sha256=compatibility.visual_initialization_sha256,
        audio_initialization_sha256=compatibility.audio_initialization_sha256,
        model_initialization_sha256=compatibility.model_initialization_sha256,
        index_sha256=compatibility.index_sha256,
        batch_size=1,
        epochs=None,
        training_output_dir=str(worker_dir.absolute()),
        runtime_contract_path=str(runtime_contract.absolute()),
        runtime_contract_sha256=sha256_file(runtime_contract),
    )


def build_evaluation_adapters(
    *,
    resolved_config: Path,
    device: torch.device | str,
    evidence: TrainingEvidence,
    trusted_upstream_artifacts: bool,
):
    """Return lazy common-runtime and modality-exact predictor factories."""
    holder: dict[str, object] = {}

    def runtime_factory() -> BenchmarkEvaluationRuntime:
        resolved_sha256 = sha256_file(resolved_config)
        config, source_config_sha256 = load_audited_benchmark_config(resolved_config)
        if config.scene.scene_id != evidence.scene_id:
            raise ValueError("evaluation config/evidence scene mismatch")
        relevant_checkpoint: Path | None = None
        native_source_records: tuple[tuple[Path, str], ...] = ()
        training_runtime: BenchmarkRuntime | None = None
        if evidence.role == "continuation":
            if evidence.config_sha256 != resolved_sha256:
                raise ValueError("continuation config/evidence hash mismatch")
            checkpoint_bytes = Path(evidence.checkpoint_path).read_bytes()
            if (
                hashlib.sha256(checkpoint_bytes).hexdigest()
                != evidence.checkpoint_sha256
            ):
                raise ValueError("continuation checkpoint/evidence hash mismatch")
            holder["checkpoint_bytes"] = checkpoint_bytes
        elif evidence.system_name == "native_audiogs":
            relevant_checkpoint = config.paths.audio_checkpoint.resolve()
        elif evidence.system_name == "native_ftgspp":
            relevant_checkpoint = config.paths.visual_checkpoint.resolve()
        else:
            raise ValueError("unsupported production evaluation evidence")
        if relevant_checkpoint is not None and (
            relevant_checkpoint != Path(evidence.checkpoint_path).resolve()
            or source_config_sha256 != evidence.config_sha256
            or sha256_file(relevant_checkpoint) != evidence.checkpoint_sha256
        ):
            raise ValueError("native config/checkpoint evidence mismatch")
        if evidence.role == "native_reference":
            contract = verify_native_contract(
                Path(evidence.native_contract_path),
                expected_scene=evidence.scene_id,
                expected_model_kind=(
                    "audiogs" if evidence.system_name == "native_audiogs" else "ftgspp"
                ),
            )
            native_source_records = tuple(
                (Path(record["path"]), record["sha256"])
                for record in contract["inputs"]["source_audits"]
            )
            if any(
                not path.is_file() or path.is_symlink() or sha256_file(path) != digest
                for path, digest in native_source_records
            ):
                raise ValueError(
                    "native upstream execution source changed after training"
                )
        upstream_roots = (
            config.paths.audio_upstream_root.resolve(),
            config.paths.visual_upstream_root.resolve(),
        )
        roots_by_kind = {
            "audiogs": upstream_roots[0],
            "ftgspp": upstream_roots[1],
        }
        all_source_records = tuple(
            (
                roots_by_kind[key.split(":", 1)[0]] / key.split(":", 1)[1],
                digest,
            )
            for key, digest in sorted(upstream_source_inventory(config).items())
        )
        with ExitStack() as pinned_inputs:
            if relevant_checkpoint is not None:
                checkpoint_pin = pinned_inputs.enter_context(
                    _PinnedInput(relevant_checkpoint, evidence.checkpoint_sha256)
                )
                checkpoint_field = (
                    "audio_checkpoint"
                    if evidence.system_name == "native_audiogs"
                    else "visual_checkpoint"
                )
                config = replace(
                    config,
                    paths=replace(
                        config.paths, **{checkpoint_field: checkpoint_pin.proc_path}
                    ),
                )
            source_pins = [
                pinned_inputs.enter_context(_PinnedInput(path, digest))
                for path, digest in all_source_records
            ]
            with _snapshot_source_imports(source_pins, upstream_roots):
                if evidence.role == "continuation":
                    _seed_evaluation_runtime(evidence.seed)
                    training_runtime = build_production_runtime(
                        config_path=resolved_config,
                        device=torch.device(device),
                        trusted_upstream_artifacts=trusted_upstream_artifacts,
                    )
                    for name in (
                        "config_sha256",
                        "source_sha256",
                        "visual_initialization_sha256",
                        "audio_initialization_sha256",
                        "model_initialization_sha256",
                    ):
                        if getattr(training_runtime, name) != getattr(evidence, name):
                            raise ValueError(
                                f"continuation runtime {name} evidence mismatch"
                            )
                _seed_evaluation_runtime(evidence.seed)
                bundle = build_runtime(
                    config,
                    torch.device(device),
                    trusted_upstream_artifacts=trusted_upstream_artifacts,
                    include_eval=True,
                )
            for pinned in source_pins:
                pinned.verify()
        if (
            sha256_file(resolved_config) != resolved_sha256
            or (
                relevant_checkpoint is not None
                and sha256_file(relevant_checkpoint) != evidence.checkpoint_sha256
            )
            or (
                evidence.role == "continuation"
                and sha256_file(Path(evidence.checkpoint_path))
                != evidence.checkpoint_sha256
            )
        ):
            raise RuntimeError(
                "evaluation config/checkpoint changed during construction"
            )
        if any(sha256_file(path) != digest for path, digest in native_source_records):
            raise RuntimeError("native upstream source changed during construction")
        if training_runtime is not None and state_sha256(bundle.model) != state_sha256(
            training_runtime.model
        ):
            raise RuntimeError(
                "evaluation initialization differs from train-only runtime"
            )
        if bundle.eval_samples is None:
            raise RuntimeError("production evaluation did not construct cam38")
        holder["bundle"] = bundle
        return BenchmarkEvaluationRuntime(bundle.eval_samples, bundle.audio_loss_fn)

    def predictor_factory(_runtime: BenchmarkEvaluationRuntime):
        bundle = holder.get("bundle")
        if bundle is None:
            raise RuntimeError("evaluation runtime must be constructed first")
        model = bundle.model
        if evidence.role == "continuation":
            raw = torch.load(
                io.BytesIO(holder["checkpoint_bytes"]),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(raw["model"], strict=True)
            model.condition_enabled = evidence.system_name == "joint_conditioned"
        else:
            model.condition_enabled = False
        model.eval()

        def predict(sample):
            if evidence.system_name == "native_ftgspp":
                render = model.visual.render_rgbd(
                    sample.visual_time,
                    sample.w2c,
                    sample.intrinsic,
                    sample.image_size,
                )
                return BenchmarkPrediction(rendered_rgb=render.rgb)
            output = model(sample)
            if evidence.system_name == "native_audiogs":
                return BenchmarkPrediction(predicted_audio=output.predicted_audio)
            return BenchmarkPrediction(
                predicted_audio=output.predicted_audio,
                rendered_rgb=output.rgbd.rgb,
            )

        return predict

    return runtime_factory, predictor_factory


__all__ = [
    "build_evaluation_adapters",
    "continuation_training_evidence",
    "expected_identity",
    "native_training_evidence",
    "prepare_worker_manifests",
    "sha256_file",
    "write_resolved_project_config",
]

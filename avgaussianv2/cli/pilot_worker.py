"""Single-device worker for one bounded pilot variant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from avgaussianv2.config import ProjectConfig, load_project_config_bytes
from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotCompatibility,
    PilotResumeError,
    build_run_fingerprint,
    hash_index_manifest,
    inspect_pilot_checkpoint,
)
from avgaussianv2.experiment.contracts import (
    PilotConfig,
    SharedIndices,
    Variant,
)
from avgaussianv2.experiment.evaluation import METRIC_NAMES, Evaluator
from avgaussianv2.experiment.training import PilotTrainer, PilotTrainingResult
from avgaussianv2.train import (
    build_joint_optimizer,
    build_warmup_optimizer,
    condition_warmup_step,
    joint_train_step,
)


MANIFEST_SCHEMA = "avgaussianv2.single-gpu-pilot-worker"
MANIFEST_VERSION = 2
_ROOT_FIELDS = {
    "schema",
    "version",
    "scene_id",
    "seed",
    "pilot_config",
    "shared_indices",
    "quick_heldout_indices",
    "config_identity",
    "source_hashes",
    "compatibility",
    "component_identities",
    "runtime_identity",
    "dataset_lengths",
    "visual_baseline",
}
_SOURCE_HASH_FIELDS = {
    "project_config_sha256",
    "dataset_manifest_sha256",
    "visual_checkpoint_sha256",
    "audio_checkpoint_sha256",
    "camera_mapping_sha256",
}
_COMPONENT_IDENTITY_FIELDS = {
    "model_class",
    "warmup_optimizer_factory",
    "joint_optimizer_factory",
    "warmup_optimizer_class",
    "joint_optimizer_class",
    "warmup_step_fn",
    "joint_step_fn",
    "audio_loss_fn",
}
_AGGREGATE_FIELDS = {"mean", "std", "median"}
_NONNEGATIVE_BASELINE_METRICS = set(METRIC_NAMES) - {"rgb_ssim"}
MAX_SMALL_INPUT_BYTES = 16 * 1024 * 1024
MAX_SUMMARY_BYTES = 8 * 1024 * 1024
MAX_CURVE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class VerifiedWorkerOutput:
    manifest: WorkerManifest
    summary: dict[str, object]
    latest: Any
    best: Any
    worker_summary_sha256: str
    latest_sha256: str
    best_sha256: str


def _read_bounded_regular_bytes(path: Path, limit: int) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a non-symlink regular file: {path}")
    if before.st_size > limit:
        raise ValueError(f"input exceeds {limit} byte limit: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(data) > limit:
        raise ValueError(f"input exceeds {limit} byte limit: {path}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while being read: {path}")
    return data


def _snapshot_file(path: Path) -> FileSnapshot:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"artifact must be a non-symlink regular file: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"artifact changed while hashing: {path}")
    return FileSnapshot(
        path.resolve(),
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        digest.hexdigest(),
    )


def _verify_snapshot(snapshot: FileSnapshot) -> None:
    actual = _snapshot_file(snapshot.path)
    if actual != snapshot:
        raise ValueError(f"artifact changed after runtime load: {snapshot.path}")


def _exact(value: object, expected: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{name} fields mismatch: actual={sorted(value)} expected={sorted(expected)}"
        )
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _integer(
    value: object, name: str, *, minimum: int = 0, positive: bool = False
) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    lower = 1 if positive else minimum
    if result < lower:
        qualifier = "positive" if positive else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _indices(value: object, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    result = tuple(_integer(item, f"{name}[{position}]") for position, item in enumerate(value))
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _strict_json_bytes(data: bytes, name: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"cannot read strict JSON from {name}: {error}") from error


def _strict_json(path: Path, limit: int = MAX_SMALL_INPUT_BYTES) -> object:
    return _strict_json_bytes(_read_bounded_regular_bytes(path, limit), str(path))


def _normalized_manifest_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_manifest_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalized_manifest_value(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def _pilot_config(value: object) -> PilotConfig:
    mapping = _exact(value, set(PilotConfig.__dataclass_fields__), "pilot_config")
    integer_fields = {
        "warmup_steps",
        "joint_steps",
        "validation_interval",
        "minimum_joint_steps",
        "patience",
        "quick_validation_samples",
    }
    values: dict[str, object] = {}
    for field in fields(PilotConfig):
        raw = mapping[field.name]
        if field.name in integer_fields:
            positive = field.name not in {"warmup_steps", "minimum_joint_steps"}
            values[field.name] = _integer(
                raw, f"pilot_config.{field.name}", positive=positive
            )
        else:
            if not isinstance(raw, Real) or isinstance(raw, bool):
                raise TypeError(f"pilot_config.{field.name} must be numeric")
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"pilot_config.{field.name} must be finite")
            values[field.name] = number
    result = PilotConfig(**values)
    result.validate()
    if result.minimum_joint_steps > result.joint_steps:
        raise ValueError("pilot_config.minimum_joint_steps must not exceed joint_steps")
    if result.psnr_tolerance_db < 0 or result.ssim_tolerance < 0:
        raise ValueError("pilot visual tolerances must be nonnegative")
    return result


def _visual_baseline_summary(value: object, name: str) -> dict[str, object]:
    summary = _exact(value, set(METRIC_NAMES), f"{name} metric")
    result: dict[str, object] = {}
    for metric in METRIC_NAMES:
        aggregate = _exact(
            summary[metric],
            _AGGREGATE_FIELDS,
            f"{name}.{metric} aggregate",
        )
        values: dict[str, float] = {}
        for statistic in ("mean", "std", "median"):
            raw = aggregate[statistic]
            if not isinstance(raw, Real) or isinstance(raw, bool):
                raise TypeError(
                    f"{name}.{metric}.{statistic} must be a numeric real"
                )
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"{name}.{metric}.{statistic} must be finite")
            values[statistic] = number
        if values["std"] < 0:
            raise ValueError(f"{name}.{metric}.std must be nonnegative")
        if metric in _NONNEGATIVE_BASELINE_METRICS:
            for statistic in ("mean", "median"):
                if values[statistic] < 0:
                    raise ValueError(
                        f"{name}.{metric}.{statistic} must be nonnegative"
                    )
        if metric == "rgb_l1":
            for statistic in ("mean", "median"):
                if values[statistic] > 1:
                    raise ValueError(
                        f"{name}.rgb_l1.{statistic} must be at most 1"
                    )
        if metric == "rgb_ssim":
            for statistic in ("mean", "median"):
                if not -1 <= values[statistic] <= 1:
                    raise ValueError(
                        f"{name}.rgb_ssim.{statistic} must be in [-1, 1]"
                    )
        result[metric] = values
    return result


@dataclass(frozen=True)
class WorkerManifest:
    path: Path
    sha256: str
    scene_id: str
    seed: int
    pilot_config: PilotConfig
    shared_indices: SharedIndices
    quick_heldout_indices: tuple[int, ...]
    source_config_sha256: str
    runtime_config_sha256: str
    source_hashes: dict[str, str]
    compatibility: dict[Variant, PilotCompatibility]
    component_identities: dict[str, str]
    runtime_model_class: str
    runtime_model_format_version: str
    train_length: int
    eval_length: int
    visual_baseline_path: Path
    visual_baseline_sha256: str
    visual_baseline_summary: dict[str, object]

    def indices_for(self, variant: Variant):
        return self.shared_indices.for_variant(variant)

    def compatibility_for(self, variant: Variant) -> PilotCompatibility:
        return self.compatibility[Variant(variant)]


def pilot_index_hash(
    shared_indices: SharedIndices,
    quick_heldout_indices: Sequence[int],
    variant: Variant | str,
) -> str:
    """Return Task 6's canonical per-variant index compatibility digest."""
    selected = shared_indices.for_variant(Variant(variant))
    return hash_index_manifest(
        {
            "warmup": list(selected.warmup),
            "joint": list(selected.joint),
            "quick_heldout": list(quick_heldout_indices),
        }
    )


def load_worker_manifest(
    path: str | Path,
    *,
    config_path: str | Path,
    config: ProjectConfig,
    actual_source_hashes: Mapping[str, str] | None = None,
) -> WorkerManifest:
    """Load and fully validate the shared, JSON-safe worker contract."""
    source = Path(path)
    raw = _exact(_strict_json(source), _ROOT_FIELDS, "worker manifest")
    if raw["schema"] != MANIFEST_SCHEMA:
        raise ValueError("worker manifest schema mismatch")
    if _integer(raw["version"], "worker manifest version", positive=True) != MANIFEST_VERSION:
        raise ValueError("worker manifest version mismatch")
    scene_id = _text(raw["scene_id"], "scene_id")
    if scene_id != config.scene.scene_id:
        raise ValueError(
            f"scene mismatch: manifest={scene_id!r} config={config.scene.scene_id!r}"
        )
    seed = _integer(raw["seed"], "seed")
    pilot = _pilot_config(raw["pilot_config"])
    if seed != config.train.seed:
        raise ValueError(f"seed mismatch: manifest={seed} config={config.train.seed}")

    shared_raw = _exact(
        raw["shared_indices"], {"warmup", "joint"}, "shared_indices"
    )
    shared = SharedIndices(
        warmup=_indices(shared_raw["warmup"], "shared_indices.warmup"),
        joint=_indices(shared_raw["joint"], "shared_indices.joint", nonempty=True),
    )
    heldout = _indices(
        raw["quick_heldout_indices"], "quick_heldout_indices", nonempty=True
    )
    if len(shared.warmup) != pilot.warmup_steps:
        raise ValueError("shared warmup index count does not match PilotConfig")
    if len(shared.joint) != pilot.joint_steps:
        raise ValueError("shared joint index count does not match PilotConfig")
    if len(heldout) > pilot.quick_validation_samples:
        raise ValueError("quick held-out index count exceeds PilotConfig")
    if len(set(heldout)) != len(heldout):
        raise ValueError("quick held-out indices must not contain duplicates")

    config_identity = _exact(
        raw["config_identity"],
        {"source_config_sha256", "runtime_config_sha256"},
        "config_identity",
    )
    source_config_sha256 = _digest(
        config_identity["source_config_sha256"],
        "config_identity.source_config_sha256",
    )
    runtime_config_sha256 = _digest(
        config_identity["runtime_config_sha256"],
        "config_identity.runtime_config_sha256",
    )
    hashes_raw = _exact(raw["source_hashes"], _SOURCE_HASH_FIELDS, "source_hashes")
    source_hashes = {
        name: _digest(hashes_raw[name], f"source_hashes.{name}")
        for name in _SOURCE_HASH_FIELDS
    }
    if actual_source_hashes is None:
        config_bytes = _read_bounded_regular_bytes(
            Path(config_path), MAX_SMALL_INPUT_BYTES
        )
        dataset_snapshot = _snapshot_file(config.paths.manifest)
        visual_snapshot = _snapshot_file(config.paths.visual_checkpoint)
        audio_snapshot = _snapshot_file(config.paths.audio_checkpoint)
        actual_hashes = {
            "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "dataset_manifest_sha256": dataset_snapshot.sha256,
            "visual_checkpoint_sha256": visual_snapshot.sha256,
            "audio_checkpoint_sha256": audio_snapshot.sha256,
            "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
        }
    else:
        actual_hashes = dict(actual_source_hashes)
    _exact(actual_hashes, _SOURCE_HASH_FIELDS, "actual_source_hashes")
    labels = {
        "project_config_sha256": "project config",
        "dataset_manifest_sha256": "dataset manifest",
        "visual_checkpoint_sha256": "visual checkpoint",
        "audio_checkpoint_sha256": "audio checkpoint",
        "camera_mapping_sha256": "camera mapping",
    }
    for name, actual in actual_hashes.items():
        if source_hashes[name] != actual:
            raise ValueError(f"{labels[name]} hash mismatch")
    if runtime_config_sha256 != source_hashes["project_config_sha256"]:
        raise ValueError("runtime config identity/hash mismatch")

    compatibility_raw = _exact(
        raw["compatibility"], {item.value for item in Variant}, "compatibility"
    )
    compatibilities: dict[Variant, PilotCompatibility] = {}
    for variant in Variant:
        try:
            actual = PilotCompatibility.from_mapping(compatibility_raw[variant.value])
        except PilotResumeError as error:
            raise ValueError(f"compatibility.{variant.value} is invalid: {error}") from error
        expected = PilotCompatibility(
            scene_id=scene_id,
            variant=variant.value,
            seed=seed,
            index_hash=pilot_index_hash(shared, heldout, variant),
            visual_checkpoint_sha256=source_hashes["visual_checkpoint_sha256"],
            audio_checkpoint_sha256=source_hashes["audio_checkpoint_sha256"],
            camera_mapping_sha256=source_hashes["camera_mapping_sha256"],
            n_fft=config.model.n_fft,
            hop_length=config.model.hop_length,
            win_length=config.model.win_length,
            sample_rate=config.model.sample_rate,
        )
        if actual != expected:
            raise ValueError(f"compatibility mismatch for {variant.value}")
        compatibilities[variant] = actual

    identities_raw = _exact(
        raw["component_identities"],
        _COMPONENT_IDENTITY_FIELDS,
        "component_identities",
    )
    identities = {
        name: _text(identities_raw[name], f"component_identities.{name}")
        for name in _COMPONENT_IDENTITY_FIELDS
    }
    runtime_identity = _exact(
        raw["runtime_identity"],
        {"model_class", "model_format_version"},
        "runtime_identity",
    )
    runtime_model_class = _text(
        runtime_identity["model_class"], "runtime_identity.model_class"
    )
    runtime_model_format_version = _text(
        runtime_identity["model_format_version"],
        "runtime_identity.model_format_version",
    )
    lengths = _exact(raw["dataset_lengths"], {"train", "eval"}, "dataset_lengths")
    train_length = _integer(
        lengths["train"], "dataset_lengths.train", positive=True
    )
    eval_length = _integer(lengths["eval"], "dataset_lengths.eval", positive=True)
    for index in (*shared.warmup, *shared.joint):
        if index >= train_length:
            raise ValueError(f"training index {index} is out of expected range")
    for index in heldout:
        if index >= eval_length:
            raise ValueError(f"held-out index {index} is out of expected range")
    expected_heldout_count = min(eval_length, pilot.quick_validation_samples)
    if len(heldout) != expected_heldout_count:
        raise ValueError(
            "quick held-out index count does not match PilotConfig and expected "
            "evaluation dataset length"
        )

    baseline_raw = _exact(
        raw["visual_baseline"], {"path", "sha256", "summary"}, "visual_baseline"
    )
    baseline_path = Path(_text(baseline_raw["path"], "visual_baseline.path"))
    baseline_sha = _digest(
        baseline_raw["sha256"], "visual_baseline.sha256"
    )
    summary = _visual_baseline_summary(
        baseline_raw["summary"], "visual_baseline.summary"
    )
    return WorkerManifest(
        path=source.resolve(),
        sha256=hash_index_manifest(_normalized_manifest_value(raw)),
        scene_id=scene_id,
        seed=seed,
        pilot_config=pilot,
        shared_indices=shared,
        quick_heldout_indices=heldout,
        source_config_sha256=source_config_sha256,
        runtime_config_sha256=runtime_config_sha256,
        source_hashes=source_hashes,
        compatibility=compatibilities,
        component_identities=identities,
        runtime_model_class=runtime_model_class,
        runtime_model_format_version=runtime_model_format_version,
        train_length=train_length,
        eval_length=eval_length,
        visual_baseline_path=baseline_path.resolve(),
        visual_baseline_sha256=baseline_sha,
        visual_baseline_summary=summary,
    )


def _validate_baseline(manifest: WorkerManifest, path: Path) -> dict[str, object]:
    if path.resolve() != manifest.visual_baseline_path:
        raise ValueError("visual baseline path mismatch")
    data = _read_bounded_regular_bytes(path, MAX_SMALL_INPUT_BYTES)
    value = _visual_baseline_summary(
        _strict_json_bytes(data, str(path)), "visual baseline file"
    )
    if hashlib.sha256(data).hexdigest() != manifest.visual_baseline_sha256:
        raise ValueError("visual baseline hash mismatch")
    if value != manifest.visual_baseline_summary:
        raise ValueError("visual baseline summary mismatch")
    return value


def validate_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not isinstance(value, str) or not value.startswith("cuda:"):
        raise argparse.ArgumentTypeError("device must be cpu or cuda:<nonnegative-index>")
    suffix = value.removeprefix("cuda:")
    if not suffix.isdigit():
        raise argparse.ArgumentTypeError("device must be cpu or cuda:<nonnegative-index>")
    return torch.device(value)


def _preflight_output(output: Path, resume: bool) -> None:
    if resume:
        latest = output / "latest.pt"
        if not latest.is_file():
            raise FileNotFoundError(f"resume requires readable latest checkpoint: {latest}")
        return
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"pilot output is not a directory: {output}")
        existing = list(output.iterdir())
        symlinks = [path for path in existing if path.is_symlink()]
        if symlinks:
            raise ValueError(
                "pilot output contains unsafe symlink: "
                + ", ".join(path.name for path in symlinks)
            )
        existing = [path for path in existing if path.name != ".pilot.lock"]
        if existing:
            raise FileExistsError(
                "fresh pilot refuses nonempty output directory: "
                + ", ".join(path.name for path in existing)
            )


def _bound_component_identities(
    manifest: WorkerManifest,
    *,
    trust_upstream_artifacts: bool,
) -> dict[str, str]:
    """Bind canonical worker semantics and trust mode into Task 6."""
    identities = dict(manifest.component_identities)
    identities["worker_contract_sha256"] = build_worker_contract_sha256(
        manifest_sha256=manifest.sha256,
        source_hashes=manifest.source_hashes,
        trust_upstream_artifacts=trust_upstream_artifacts,
    )
    return identities


def build_worker_component_identities(
    model_class: str,
    *,
    audio_loss_fn: str = "runtime.audio_loss_fn-v1",
) -> dict[str, str]:
    """Return the single canonical producer contract for Task 7 manifests."""
    if not isinstance(model_class, str) or not model_class:
        raise TypeError("model_class must be a nonempty string")
    if not isinstance(audio_loss_fn, str) or not audio_loss_fn:
        raise TypeError("audio_loss_fn must be a nonempty string")
    return {
        "model_class": model_class,
        "warmup_optimizer_factory": "avgaussianv2.train.build_warmup_optimizer-v1",
        "joint_optimizer_factory": "avgaussianv2.train.build_joint_optimizer-v1",
        "warmup_optimizer_class": "torch.optim.adam.Adam",
        "joint_optimizer_class": "torch.optim.adam.Adam",
        "warmup_step_fn": "avgaussianv2.train.condition_warmup_step-v1",
        "joint_step_fn": "avgaussianv2.train.joint_train_step-v1",
        "audio_loss_fn": audio_loss_fn,
    }


def build_worker_contract_sha256(
    *,
    manifest_sha256: str,
    source_hashes: Mapping[str, str],
    trust_upstream_artifacts: bool,
) -> str:
    """Bind manifest version, sources, and explicit trust mode in one place."""
    if not isinstance(trust_upstream_artifacts, bool):
        raise TypeError("trust_upstream_artifacts must be boolean")
    _digest(manifest_sha256, "manifest_sha256")
    hashes = _exact(source_hashes, _SOURCE_HASH_FIELDS, "source_hashes")
    normalized = {
        name: _digest(hashes[name], f"source_hashes.{name}")
        for name in sorted(_SOURCE_HASH_FIELDS)
    }
    return hash_index_manifest(
        {
            "manifest_schema": MANIFEST_SCHEMA,
            "manifest_version": MANIFEST_VERSION,
            "manifest_sha256": manifest_sha256,
            "source_hashes": normalized,
            "trust_upstream_artifacts": trust_upstream_artifacts,
        }
    )


class _ManifestFingerprintModel(nn.Module):
    def __init__(self, format_version: str) -> None:
        super().__init__()
        self.checkpoint_format_version = format_version


def _fingerprint_audio_loss(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("manifest fingerprint placeholder must not execute")


def _expected_run_fingerprint(
    manifest: WorkerManifest,
    config: ProjectConfig,
    baseline: Mapping[str, object],
    *,
    trust_upstream_artifacts: bool,
) -> dict[str, object]:
    """Construct Task 6's exact fingerprint without loading a runtime."""
    return build_run_fingerprint(
        pilot_config=manifest.pilot_config,
        train_config=config.train,
        visual_baseline=baseline,
        model=_ManifestFingerprintModel(manifest.runtime_model_format_version),
        warmup_optimizer_factory=build_warmup_optimizer,
        joint_optimizer_factory=build_joint_optimizer,
        warmup_step_fn=condition_warmup_step,
        joint_step_fn=joint_train_step,
        audio_loss_fn=_fingerprint_audio_loss,
        component_identities=_bound_component_identities(
            manifest,
            trust_upstream_artifacts=trust_upstream_artifacts,
        ),
    )


def _validate_runtime_identity(model: nn.Module, manifest: WorkerManifest) -> None:
    model_type = model.__class__
    actual_class = f"{model_type.__module__}.{model_type.__qualname__}"
    if actual_class != manifest.runtime_model_class:
        raise ValueError(
            "runtime model class mismatch: "
            f"actual={actual_class!r} expected={manifest.runtime_model_class!r}"
        )
    actual_format = str(
        getattr(model, "checkpoint_format_version", "state-dict-v1")
    )
    if actual_format != manifest.runtime_model_format_version:
        raise ValueError(
            "runtime model format mismatch: "
            f"actual={actual_format!r} "
            f"expected={manifest.runtime_model_format_version!r}"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite(value: object, name: str = "result") -> None:
    if isinstance(value, Real) and not isinstance(value, (bool, Integral)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite value")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{name}[{index}]")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_worker_reports(
    output: Path,
    result: PilotTrainingResult,
    latest: Any,
    best: Any,
    store: PilotCheckpointStore,
) -> dict[str, object]:
    curve_data = _read_bounded_regular_bytes(
        output / "training_curve.csv", MAX_CURVE_BYTES
    )
    try:
        text = curve_data.decode("utf-8")
    except UnicodeError as error:
        raise RuntimeError("training curve must be UTF-8") from error
    reader = csv.DictReader(text.splitlines())
    expected_columns = [
        "stage",
        "step",
        "sample_index",
        "total",
        "audio_to_visual_grad_norm",
        "losses",
        "gradient_norms",
    ]
    if reader.fieldnames != expected_columns:
        raise RuntimeError("training curve columns mismatch")
    rows = list(reader)
    if len(rows) != len(result.training_history):
        raise RuntimeError("training curve row count mismatch")
    for actual, expected in zip(rows, result.training_history, strict=True):
        if (
            actual["stage"] != str(expected["stage"])
            or int(actual["step"]) != expected["step"]
            or int(actual["sample_index"]) != expected["sample_index"]
            or float(actual["total"]) != float(expected["total"])
            or float(actual["audio_to_visual_grad_norm"])
            != float(expected["audio_to_visual_grad_norm"])
            or _strict_json_bytes(
                actual["losses"].encode("utf-8"), "training curve losses"
            )
            != expected["losses"]
            or _strict_json_bytes(
                actual["gradient_norms"].encode("utf-8"),
                "training curve gradient_norms",
            )
            != expected["gradient_norms"]
        ):
            raise RuntimeError("training curve rows disagree with result")

    summary = _strict_json_bytes(
        _read_bounded_regular_bytes(
            output / "worker_summary.json", MAX_SUMMARY_BYTES
        ),
        "worker_summary.json",
    )
    expected_summary_fields = {
        "variant",
        "completed_warmup_steps",
        "completed_joint_steps",
        "best_step",
        "stop_reason",
        "training_history",
        "validation_history",
        "selector_state",
        "stopper_state",
        "checkpoint_io",
    }
    summary = dict(_exact(summary, expected_summary_fields, "worker summary"))
    if (
        summary["variant"] != result.variant.value
        or summary["completed_warmup_steps"] != result.completed_warmup_steps
        or summary["completed_joint_steps"] != result.completed_joint_steps
        or summary["best_step"] != result.best_step
        or summary["stop_reason"] != result.stop_reason
        or summary["training_history"] != list(result.training_history)
        or summary["validation_history"] != list(result.validation_history)
        or summary["selector_state"] != latest.selector.state_dict()
        or summary["stopper_state"] != latest.stopper.state_dict()
        or summary["checkpoint_io"] != store.metrics
    ):
        raise RuntimeError("worker summary disagrees with training result")
    if (
        latest.completed_warmup_steps != result.completed_warmup_steps
        or latest.completed_joint_steps != result.completed_joint_steps
        or latest.selector.best_step != result.best_step
        or latest.training_history != tuple(result.training_history)
        or latest.validation_history != tuple(result.validation_history)
        or latest.stop_reason != result.stop_reason
    ):
        raise RuntimeError("latest checkpoint disagrees with training result")
    selected_summary = next(
        row["summary"]
        for row in latest.validation_history
        if row["step"] == result.best_step
    )
    expected_best_training = tuple(
        row
        for row in latest.training_history
        if row["stage"] == "warmup"
        or (row["stage"] == "joint" and row["step"] <= result.best_step)
    )
    expected_best_validations = tuple(
        row
        for row in latest.validation_history
        if row["step"] <= result.best_step
    )
    if (
        best.checkpoint_kind != "best"
        or best.generation != latest.best_generation
        or best.selector.best_step != result.best_step
        or best.best_evaluation_summary != latest.best_evaluation_summary
        or best.validation_summary != selected_summary
        or best.training_history != expected_best_training
        or best.validation_history != expected_best_validations
    ):
        raise RuntimeError("best checkpoint disagrees with latest checkpoint")
    return summary


def verify_worker_output(
    config_path: str | Path,
    shared_indices_path: str | Path,
    visual_baseline_path: str | Path,
    output_dir: str | Path,
    variant: Variant | str,
    *,
    trust_upstream_artifacts: bool,
) -> VerifiedWorkerOutput:
    """Verify a completed worker output without constructing a runtime."""
    resolved_variant = Variant(variant)
    config_source = Path(config_path)
    config_bytes = _read_bounded_regular_bytes(config_source, MAX_SMALL_INPUT_BYTES)
    config = load_project_config_bytes(config_bytes, base_dir=config_source.parent)
    source_snapshots = (
        _snapshot_file(config.paths.visual_checkpoint),
        _snapshot_file(config.paths.audio_checkpoint),
        _snapshot_file(config.paths.manifest),
    )
    actual_hashes = {
        "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "visual_checkpoint_sha256": source_snapshots[0].sha256,
        "audio_checkpoint_sha256": source_snapshots[1].sha256,
        "dataset_manifest_sha256": source_snapshots[2].sha256,
        "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
    }
    manifest = load_worker_manifest(
        shared_indices_path,
        config_path=config_source,
        config=config,
        actual_source_hashes=actual_hashes,
    )
    baseline = _validate_baseline(manifest, Path(visual_baseline_path))
    expected_fingerprint = _expected_run_fingerprint(
        manifest,
        config,
        baseline,
        trust_upstream_artifacts=trust_upstream_artifacts,
    )
    indices = manifest.indices_for(resolved_variant)
    output = Path(output_dir)
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    summary_path = output / "worker_summary.json"
    curve_path = output / "training_curve.csv"
    latest_snapshot = _snapshot_file(latest_path)
    best_snapshot = _snapshot_file(best_path)
    summary_snapshot = _snapshot_file(summary_path)
    curve_snapshot = _snapshot_file(curve_path)
    latest = inspect_pilot_checkpoint(
        latest_path,
        expected_compatibility=manifest.compatibility_for(resolved_variant),
        indices=indices,
        expected_run_fingerprint=expected_fingerprint,
        active_resume=False,
        allow_complete=True,
    )
    best = inspect_pilot_checkpoint(
        best_path,
        expected_compatibility=manifest.compatibility_for(resolved_variant),
        indices=indices,
        expected_run_fingerprint=expected_fingerprint,
        active_resume=False,
    )
    if latest.checkpoint_kind != "latest" or latest.stage != "complete":
        raise PilotResumeError("worker latest checkpoint is not complete")
    if best.checkpoint_kind != "best" or best.generation != latest.best_generation:
        raise PilotResumeError("worker best checkpoint generation mismatch")
    summary_value = _strict_json_bytes(
        _read_bounded_regular_bytes(summary_path, MAX_SUMMARY_BYTES),
        "worker_summary.json",
    )
    fields_expected = {
        "variant", "completed_warmup_steps", "completed_joint_steps",
        "best_step", "stop_reason", "training_history", "validation_history",
        "selector_state", "stopper_state", "checkpoint_io", "worker",
    }
    summary = dict(_exact(summary_value, fields_expected, "worker summary"))
    worker = _exact(
        summary["worker"],
        {
            "variant", "device", "scene_id", "config_sha256",
            "source_config_sha256", "runtime_config_sha256",
            "manifest_sha256", "visual_baseline_sha256",
            "trusted_upstream_artifacts",
        },
        "worker summary identity",
    )
    expected_identity = {
        "variant": resolved_variant.value,
        "scene_id": manifest.scene_id,
        "config_sha256": manifest.source_hashes["project_config_sha256"],
        "source_config_sha256": manifest.source_config_sha256,
        "runtime_config_sha256": manifest.runtime_config_sha256,
        "manifest_sha256": manifest.sha256,
        "visual_baseline_sha256": manifest.visual_baseline_sha256,
        "trusted_upstream_artifacts": trust_upstream_artifacts,
    }
    for name, expected in expected_identity.items():
        if worker[name] != expected:
            raise ValueError(f"worker summary identity mismatch: {name}")
    if not isinstance(worker["device"], str) or not worker["device"]:
        raise TypeError("worker device identity must be nonempty")
    expected_summary = {
        "variant": resolved_variant.value,
        "completed_warmup_steps": latest.completed_warmup_steps,
        "completed_joint_steps": latest.completed_joint_steps,
        "best_step": latest.selector.best_step,
        "stop_reason": latest.stop_reason,
        "training_history": list(latest.training_history),
        "validation_history": list(latest.validation_history),
        "selector_state": latest.selector.state_dict(),
        "stopper_state": latest.stopper.state_dict(),
    }
    for name, expected in expected_summary.items():
        if summary[name] != expected:
            raise ValueError(f"worker summary/checkpoint mismatch: {name}")
    selected = next(
        (
            row["summary"]
            for row in latest.validation_history
            if row["step"] == latest.selector.best_step
        ),
        None,
    )
    if (
        selected is None
        or best.validation_summary != selected
        or best.best_evaluation_summary != latest.best_evaluation_summary
        or best.selector.best_step != latest.selector.best_step
    ):
        raise ValueError("worker best/latest selection mismatch")
    curve_data = _read_bounded_regular_bytes(curve_path, MAX_CURVE_BYTES)
    try:
        reader = csv.DictReader(curve_data.decode("utf-8").splitlines())
    except UnicodeError as error:
        raise ValueError("training curve must be UTF-8") from error
    expected_columns = [
        "stage", "step", "sample_index", "total",
        "audio_to_visual_grad_norm", "losses", "gradient_norms",
    ]
    if reader.fieldnames != expected_columns:
        raise ValueError("training curve columns mismatch")
    rows = list(reader)
    if len(rows) != len(latest.training_history):
        raise ValueError("training curve/checkpoint row count mismatch")
    for actual, expected in zip(rows, latest.training_history, strict=True):
        if (
            actual["stage"] != expected["stage"]
            or int(actual["step"]) != expected["step"]
            or int(actual["sample_index"]) != expected["sample_index"]
            or float(actual["total"]) != float(expected["total"])
            or float(actual["audio_to_visual_grad_norm"])
            != float(expected["audio_to_visual_grad_norm"])
            or _strict_json_bytes(actual["losses"].encode(), "curve losses")
            != expected["losses"]
            or _strict_json_bytes(
                actual["gradient_norms"].encode(), "curve gradients"
            )
            != expected["gradient_norms"]
        ):
            raise ValueError("training curve/checkpoint mismatch")
    _finite(summary, "worker summary")
    from avgaussianv2.experiment.report import _validate_worker

    report_system = {
        Variant.JOINT_CONDITIONED: "joint_conditioned_on",
        Variant.FROZEN_VISUAL: "frozen_visual_on",
        Variant.CONDITION_OFF: "condition_off",
    }[resolved_variant]
    _validate_worker(summary, report_system, manifest.scene_id)
    for snapshot in (
        *source_snapshots,
        latest_snapshot,
        best_snapshot,
        summary_snapshot,
        curve_snapshot,
    ):
        _verify_snapshot(snapshot)
    return VerifiedWorkerOutput(
        manifest=manifest,
        summary=summary,
        latest=latest,
        best=best,
        worker_summary_sha256=summary_snapshot.sha256,
        latest_sha256=latest_snapshot.sha256,
        best_sha256=best_snapshot.sha256,
    )


def verify_worker_resume_state(
    config_path: str | Path,
    shared_indices_path: str | Path,
    visual_baseline_path: str | Path,
    output_dir: str | Path,
    variant: Variant | str,
    *,
    trust_upstream_artifacts: bool,
) -> Any:
    """Verify that an incomplete output is compatible with Task 7 resume."""
    resolved_variant = Variant(variant)
    config_source = Path(config_path)
    config_bytes = _read_bounded_regular_bytes(config_source, MAX_SMALL_INPUT_BYTES)
    config = load_project_config_bytes(config_bytes, base_dir=config_source.parent)
    snapshots = (
        _snapshot_file(config.paths.visual_checkpoint),
        _snapshot_file(config.paths.audio_checkpoint),
        _snapshot_file(config.paths.manifest),
    )
    manifest = load_worker_manifest(
        shared_indices_path,
        config_path=config_source,
        config=config,
        actual_source_hashes={
            "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "visual_checkpoint_sha256": snapshots[0].sha256,
            "audio_checkpoint_sha256": snapshots[1].sha256,
            "dataset_manifest_sha256": snapshots[2].sha256,
            "camera_mapping_sha256": hash_index_manifest(
                config.scene.camera_mapping
            ),
        },
    )
    baseline = _validate_baseline(manifest, Path(visual_baseline_path))
    expected_fingerprint = _expected_run_fingerprint(
        manifest,
        config,
        baseline,
        trust_upstream_artifacts=trust_upstream_artifacts,
    )
    state = inspect_pilot_checkpoint(
        Path(output_dir) / "latest.pt",
        expected_compatibility=manifest.compatibility_for(resolved_variant),
        indices=manifest.indices_for(resolved_variant),
        expected_run_fingerprint=expected_fingerprint,
        active_resume=False,
        allow_complete=True,
    )
    for snapshot in snapshots:
        _verify_snapshot(snapshot)
    return state


RuntimeFactory = Callable[[ProjectConfig, torch.device], Any]


def _run_worker_locked(
    *,
    store: PilotCheckpointStore,
    config: ProjectConfig,
    manifest: WorkerManifest,
    baseline: dict[str, object],
    indices: Any,
    variant: Variant,
    device: torch.device,
    output: Path,
    resume: bool,
    trust_upstream_artifacts: bool,
    expected_run_fingerprint: Mapping[str, object],
    runtime_factory: RuntimeFactory | None,
    evaluator_factory: Callable[..., Any],
    trainer_factory: Callable[..., Any],
    artifact_snapshots: Sequence[FileSnapshot],
) -> PilotTrainingResult:
    store.bind_run_fingerprint(expected_run_fingerprint)
    resume_state = None
    if resume:
        store.recover()
        resume_state = inspect_pilot_checkpoint(
            store.latest_path,
            expected_compatibility=store.compatibility,
            indices=indices,
            expected_run_fingerprint=expected_run_fingerprint,
            model=None,
            allow_complete=True,
        )
        store.prepare(resume_state=resume_state, recover=False)
    else:
        store.prepare()

    _seed_everything(manifest.seed)
    if runtime_factory is None:
        from avgaussianv2.runtime import build_runtime

        if not trust_upstream_artifacts:
            raise PermissionError(
                "production runtime requires --trust-upstream-artifacts because "
                "configured upstream Python and unsafe legacy pickle are loaded"
            )
        bundle = build_runtime(
            config,
            device,
            trusted_upstream_artifacts=True,
            include_eval=True,
        )
    else:
        bundle = runtime_factory(config, device)
    for snapshot in artifact_snapshots:
        _verify_snapshot(snapshot)
    _validate_runtime_identity(bundle.model, manifest)
    if len(bundle.train_samples) != manifest.train_length:
        raise ValueError(
            "training dataset length mismatch: "
            f"actual={len(bundle.train_samples)} expected={manifest.train_length}"
        )
    if bundle.eval_samples is None:
        raise ValueError("worker runtime requires an evaluation dataset")
    if len(bundle.eval_samples) != manifest.eval_length:
        raise ValueError(
            "evaluation dataset length mismatch: "
            f"actual={len(bundle.eval_samples)} expected={manifest.eval_length}"
        )

    evaluator = evaluator_factory(bundle.model, bundle.audio_loss_fn, device)
    if getattr(evaluator, "model", bundle.model) is not bundle.model:
        raise ValueError("evaluator must be bound to the runtime model")
    trainer = trainer_factory(
        manifest.pilot_config, evaluator, train_config=config.train
    )
    result = trainer.run(
        model=bundle.model,
        train_samples=bundle.train_samples,
        heldout_samples=bundle.eval_samples,
        indices=indices,
        heldout_indices=manifest.quick_heldout_indices,
        variant=variant,
        visual_baseline=baseline,
        audio_loss_fn=bundle.audio_loss_fn,
        output_dir=output,
        checkpoint_store=store,
        preloaded_resume_state=resume_state,
        checkpoint_store_prepared=True,
    )
    preflight_stage = None if resume_state is None else resume_state.stage
    preflight_positions = (
        None
        if resume_state is None
        else (
            resume_state.completed_warmup_steps,
            resume_state.completed_joint_steps,
        )
    )
    if preflight_positions is not None and (
        result.completed_warmup_steps < preflight_positions[0]
        or result.completed_joint_steps < preflight_positions[1]
    ):
        raise PilotResumeError("resumed result regressed from preflight state")
    _finite(asdict(result))
    if result.best_step is None:
        (output / "worker_summary.json").unlink(missing_ok=True)
        raise RuntimeError(
            "pilot completed without a visually feasible best candidate; "
            "no best checkpoint was selected"
        )
    if store.run_fingerprint != expected_run_fingerprint:
        raise PilotResumeError(
            "runtime run fingerprint disagrees with preflight fingerprint"
        )

    store.verify_output_identity()
    if resume and preflight_stage == "complete" and store.save_count == 0:
        latest = resume_state
    else:
        latest = inspect_pilot_checkpoint(
            store.latest_path,
            expected_compatibility=store.compatibility,
            indices=indices,
            expected_run_fingerprint=store.run_fingerprint,
            model=bundle.model,
            allow_complete=True,
            active_resume=False,
        )
    if (
        resume
        and preflight_stage == "complete"
        and store.save_count == 0
        and store.inspected_best_state is not None
    ):
        best = store.inspected_best_state
    else:
        best = inspect_pilot_checkpoint(
            store.best_path,
            expected_compatibility=store.compatibility,
            indices=indices,
            expected_run_fingerprint=store.run_fingerprint,
            model=bundle.model,
            active_resume=False,
        )
    if latest is None or latest.stage != "complete":
        raise RuntimeError("pilot did not publish a complete readable checkpoint")

    summary_path = output / "worker_summary.json"
    summary = _verify_worker_reports(output, result, latest, best, store)
    summary["worker"] = {
        "variant": variant.value,
        "device": str(device),
        "scene_id": manifest.scene_id,
        "config_sha256": manifest.source_hashes["project_config_sha256"],
        "source_config_sha256": manifest.source_config_sha256,
        "runtime_config_sha256": manifest.runtime_config_sha256,
        "manifest_sha256": manifest.sha256,
        "visual_baseline_sha256": manifest.visual_baseline_sha256,
        "trusted_upstream_artifacts": trust_upstream_artifacts,
    }
    _finite(summary, "worker summary")
    store.verify_output_identity()
    _atomic_json(summary_path, summary)
    return result


def run_worker(
    config_path: str | Path,
    variant: Variant | str,
    shared_indices_path: str | Path,
    visual_baseline_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    resume: bool = False,
    trust_upstream_artifacts: bool = False,
    runtime_factory: RuntimeFactory | None = None,
    evaluator_factory: Callable[..., Any] = Evaluator,
    trainer_factory: Callable[..., Any] = PilotTrainer,
    checkpoint_store_factory: Callable[..., PilotCheckpointStore] = PilotCheckpointStore,
) -> PilotTrainingResult:
    """Validate and execute exactly one variant on exactly one device."""
    resolved_variant = Variant(variant)
    resolved_device = (
        validate_device(device) if isinstance(device, str) else torch.device(device)
    )
    config_source = Path(config_path)
    from avgaussianv2.cli.pilot_eval import _validate_orchestrator_proc_paths

    _validate_orchestrator_proc_paths(
        (
            config_source,
            Path(shared_indices_path),
            Path(visual_baseline_path),
            Path(output_dir),
        )
    )
    config_bytes = _read_bounded_regular_bytes(
        config_source, MAX_SMALL_INPUT_BYTES
    )
    config = load_project_config_bytes(
        config_bytes, base_dir=config_source.parent
    )
    if runtime_factory is None and not trust_upstream_artifacts:
        raise PermissionError(
            "production runtime requires --trust-upstream-artifacts because "
            "configured upstream Python and unsafe legacy pickle are loaded"
        )
    if resolved_device.type == "cuda":
        index = 0 if resolved_device.index is None else resolved_device.index
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is unavailable: {resolved_device}")
    artifact_snapshots = (
        _snapshot_file(config.paths.visual_checkpoint),
        _snapshot_file(config.paths.audio_checkpoint),
        _snapshot_file(config.paths.manifest),
    )
    # Runtime loaders receive the exact resolved paths that were snapshotted.
    # This prevents a configured parent symlink from being retargeted between
    # validation and backend/dataset construction.
    config = replace(
        config,
        paths=replace(
            config.paths,
            visual_checkpoint=artifact_snapshots[0].path,
            audio_checkpoint=artifact_snapshots[1].path,
            manifest=artifact_snapshots[2].path,
        ),
    )
    actual_source_hashes = {
        "project_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "visual_checkpoint_sha256": artifact_snapshots[0].sha256,
        "audio_checkpoint_sha256": artifact_snapshots[1].sha256,
        "dataset_manifest_sha256": artifact_snapshots[2].sha256,
        "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
    }
    manifest = load_worker_manifest(
        shared_indices_path,
        config_path=config_source,
        config=config,
        actual_source_hashes=actual_source_hashes,
    )
    baseline_source = Path(visual_baseline_path)
    baseline = _validate_baseline(manifest, baseline_source)
    output = Path(output_dir)
    _preflight_output(output, bool(resume))
    indices = manifest.indices_for(resolved_variant)
    expected_run_fingerprint = _expected_run_fingerprint(
        manifest,
        config,
        baseline,
        trust_upstream_artifacts=trust_upstream_artifacts,
    )
    store = checkpoint_store_factory(
        output,
        manifest.compatibility_for(resolved_variant),
        resume=bool(resume),
        component_identities=_bound_component_identities(
            manifest,
            trust_upstream_artifacts=trust_upstream_artifacts,
        ),
    )
    with store:
        return _run_worker_locked(
            store=store,
            config=config,
            manifest=manifest,
            baseline=baseline,
            indices=indices,
            variant=resolved_variant,
            device=resolved_device,
            output=store.pinned_output_dir,
            resume=bool(resume),
            trust_upstream_artifacts=trust_upstream_artifacts,
            expected_run_fingerprint=expected_run_fingerprint,
            runtime_factory=runtime_factory,
            evaluator_factory=evaluator_factory,
            trainer_factory=trainer_factory,
            artifact_snapshots=artifact_snapshots,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one AVGaussianFusionV2 pilot variant on one GPU"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--variant", required=True, type=Variant, choices=tuple(Variant))
    parser.add_argument("--shared-indices", required=True, type=Path)
    parser.add_argument("--visual-baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        type=lambda value: str(validate_device(value)),
        default="cuda:0",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trust-upstream-artifacts",
        action="store_true",
        help=(
            "trust configured upstream Python and unsafe legacy pickle "
            "checkpoints after independent review"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_worker(
        args.config,
        args.variant,
        args.shared_indices,
        args.visual_baseline,
        args.output_dir,
        device=args.device,
        resume=args.resume,
        trust_upstream_artifacts=args.trust_upstream_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "WorkerManifest",
    "build_parser",
    "load_worker_manifest",
    "main",
    "pilot_index_hash",
    "run_worker",
    "validate_device",
]

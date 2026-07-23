"""Single-device worker for one bounded pilot variant."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from avgaussianv2.config import ProjectConfig, load_project_config
from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotCompatibility,
    PilotResumeError,
    build_run_fingerprint,
    hash_index_manifest,
    inspect_pilot_checkpoint,
    sha256_file,
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


def _strict_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"cannot read strict JSON from {path}: {error}") from error


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

    hashes_raw = _exact(raw["source_hashes"], _SOURCE_HASH_FIELDS, "source_hashes")
    source_hashes = {
        name: _digest(hashes_raw[name], f"source_hashes.{name}")
        for name in _SOURCE_HASH_FIELDS
    }
    actual_hashes = {
        "project_config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(config.paths.manifest),
        "visual_checkpoint_sha256": sha256_file(config.paths.visual_checkpoint),
        "audio_checkpoint_sha256": sha256_file(config.paths.audio_checkpoint),
        "camera_mapping_sha256": hash_index_manifest(config.scene.camera_mapping),
    }
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
        sha256=sha256_file(source),
        scene_id=scene_id,
        seed=seed,
        pilot_config=pilot,
        shared_indices=shared,
        quick_heldout_indices=heldout,
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
    value = _visual_baseline_summary(
        _strict_json(path), "visual baseline file"
    )
    if sha256_file(path) != manifest.visual_baseline_sha256:
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
        if existing:
            raise FileExistsError(
                "fresh pilot refuses nonempty output directory: "
                + ", ".join(path.name for path in existing)
            )


def _bound_component_identities(manifest: WorkerManifest) -> dict[str, str]:
    """Bind the exact shared contract and all source hashes into Task 6."""
    identities = dict(manifest.component_identities)
    identities["model_class"] = (
        f"{identities['model_class']}|worker-manifest-v{MANIFEST_VERSION}:"
        f"{manifest.sha256}"
    )
    return identities


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
        component_identities=_bound_component_identities(manifest),
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
    finally:
        temporary.unlink(missing_ok=True)


RuntimeFactory = Callable[[ProjectConfig, torch.device], Any]


def run_worker(
    config_path: str | Path,
    variant: Variant | str,
    shared_indices_path: str | Path,
    visual_baseline_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    resume: bool = False,
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
    if resolved_device.type == "cuda":
        index = 0 if resolved_device.index is None else resolved_device.index
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is unavailable: {resolved_device}")
    config_source = Path(config_path)
    config = load_project_config(config_source)
    manifest = load_worker_manifest(
        shared_indices_path, config_path=config_source, config=config
    )
    baseline_source = Path(visual_baseline_path)
    baseline = _validate_baseline(manifest, baseline_source)
    output = Path(output_dir)
    _preflight_output(output, bool(resume))
    indices = manifest.indices_for(resolved_variant)
    expected_run_fingerprint = _expected_run_fingerprint(
        manifest, config, baseline
    )
    preflight_completed_positions: tuple[int, int] | None = None
    if resume:
        # This shape-independent pass uses weights_only loading and rejects
        # unsafe, corrupt, or incompatible state before expensive backends.
        preflight_state = inspect_pilot_checkpoint(
            output / "latest.pt",
            expected_compatibility=manifest.compatibility_for(resolved_variant),
            indices=indices,
            expected_run_fingerprint=expected_run_fingerprint,
            model=None,
            allow_complete=True,
        )
        preflight_completed_positions = (
            preflight_state.completed_warmup_steps,
            preflight_state.completed_joint_steps,
        )
        # Task 6 re-inspects under the store lock to close the TOCTOU window
        # and validate model shapes. Do not retain this tensor-bearing copy.
        del preflight_state

    _seed_everything(manifest.seed)
    if runtime_factory is None:
        # Keep backend/upstream imports out of parser/help and manifest-only paths.
        from avgaussianv2.runtime import build_runtime

        runtime_factory = build_runtime
    bundle = runtime_factory(config, resolved_device)
    _validate_runtime_identity(bundle.model, manifest)
    if len(bundle.train_samples) != manifest.train_length:
        raise ValueError(
            "training dataset length mismatch: "
            f"actual={len(bundle.train_samples)} expected={manifest.train_length}"
        )
    if len(bundle.eval_samples) != manifest.eval_length:
        raise ValueError(
            "evaluation dataset length mismatch: "
            f"actual={len(bundle.eval_samples)} expected={manifest.eval_length}"
        )
    for index in (*indices.warmup, *indices.joint):
        if index >= len(bundle.train_samples):
            raise ValueError(f"training index {index} is out of range")
    for index in manifest.quick_heldout_indices:
        if index >= len(bundle.eval_samples):
            raise ValueError(f"held-out index {index} is out of range")

    evaluator = evaluator_factory(
        bundle.model, bundle.audio_loss_fn, resolved_device
    )
    if getattr(evaluator, "model", bundle.model) is not bundle.model:
        raise ValueError("evaluator must be bound to the runtime model")
    trainer = trainer_factory(
        manifest.pilot_config, evaluator, train_config=config.train
    )
    store = checkpoint_store_factory(
        output,
        manifest.compatibility_for(resolved_variant),
        resume=bool(resume),
        component_identities=_bound_component_identities(manifest),
    )
    result = trainer.run(
        model=bundle.model,
        train_samples=bundle.train_samples,
        heldout_samples=bundle.eval_samples,
        indices=indices,
        heldout_indices=manifest.quick_heldout_indices,
        variant=resolved_variant,
        visual_baseline=baseline,
        audio_loss_fn=bundle.audio_loss_fn,
        output_dir=output,
        checkpoint_store=store,
    )
    if preflight_completed_positions is not None:
        if (
            result.completed_warmup_steps
            < preflight_completed_positions[0]
            or result.completed_joint_steps
            < preflight_completed_positions[1]
        ):
            raise PilotResumeError("resumed result regressed from preflight state")
    _finite(asdict(result))
    if result.best_step is None:
        (output / "worker_summary.json").unlink(missing_ok=True)
        raise RuntimeError(
            "pilot completed without a visually feasible best candidate; "
            "no best checkpoint was selected"
        )
    if store.run_fingerprint is None:
        raise RuntimeError("pilot checkpoint store did not bind a run fingerprint")
    if store.run_fingerprint != expected_run_fingerprint:
        raise PilotResumeError(
            "runtime run fingerprint disagrees with preflight fingerprint"
        )
    latest = inspect_pilot_checkpoint(
        store.latest_path,
        expected_compatibility=store.compatibility,
        indices=indices,
        expected_run_fingerprint=store.run_fingerprint,
        model=bundle.model,
        allow_complete=True,
        active_resume=False,
    )
    best = inspect_pilot_checkpoint(
        store.best_path,
        expected_compatibility=store.compatibility,
        indices=indices,
        expected_run_fingerprint=store.run_fingerprint,
        model=bundle.model,
        active_resume=False,
    )
    if latest.stage != "complete" or best.checkpoint_kind != "best":
        raise RuntimeError("pilot did not publish complete readable checkpoints")
    curve_path = output / "training_curve.csv"
    summary_path = output / "worker_summary.json"
    if not curve_path.is_file() or not curve_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("pilot training curve is missing or unreadable")
    summary = _strict_json(summary_path)
    if not isinstance(summary, dict):
        raise RuntimeError("worker summary must be a JSON object")
    summary["worker"] = {
        "variant": resolved_variant.value,
        "device": str(resolved_device),
        "scene_id": manifest.scene_id,
        "config_sha256": manifest.source_hashes["project_config_sha256"],
        "manifest_sha256": manifest.sha256,
        "visual_baseline_sha256": manifest.visual_baseline_sha256,
    }
    _finite(summary, "worker summary")
    _atomic_json(summary_path, summary)
    return result


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

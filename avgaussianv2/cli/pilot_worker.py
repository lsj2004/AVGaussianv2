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

from avgaussianv2.config import ProjectConfig, load_project_config
from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotCompatibility,
    PilotResumeError,
    hash_index_manifest,
    inspect_pilot_checkpoint,
    sha256_file,
)
from avgaussianv2.experiment.contracts import (
    PilotConfig,
    SharedIndices,
    Variant,
)
from avgaussianv2.experiment.evaluation import Evaluator
from avgaussianv2.experiment.training import PilotTrainer, PilotTrainingResult


MANIFEST_SCHEMA = "avgaussianv2.single-gpu-pilot-worker"
MANIFEST_VERSION = 1
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
    summary = baseline_raw["summary"]
    if not isinstance(summary, dict):
        raise TypeError("visual_baseline.summary must be a JSON object")
    # Round-trip also rejects non-string keys and non-finite values.
    summary = json.loads(
        json.dumps(summary, sort_keys=True, allow_nan=False)
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
        train_length=train_length,
        eval_length=eval_length,
        visual_baseline_path=baseline_path.resolve(),
        visual_baseline_sha256=baseline_sha,
        visual_baseline_summary=summary,
    )


def _validate_baseline(manifest: WorkerManifest, path: Path) -> dict[str, object]:
    if path.resolve() != manifest.visual_baseline_path:
        raise ValueError("visual baseline path mismatch")
    if sha256_file(path) != manifest.visual_baseline_sha256:
        raise ValueError("visual baseline hash mismatch")
    value = _strict_json(path)
    if value != manifest.visual_baseline_summary:
        raise ValueError("visual baseline summary mismatch")
    if not isinstance(value, dict):
        raise TypeError("visual baseline must be a JSON object")
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
    if resume:
        # This shape-independent pass uses weights_only loading and rejects
        # unsafe, corrupt, or incompatible state before expensive backends.
        inspect_pilot_checkpoint(
            output / "latest.pt",
            expected_compatibility=manifest.compatibility_for(resolved_variant),
            indices=indices,
            model=None,
            allow_complete=True,
        )

    _seed_everything(manifest.seed)
    if runtime_factory is None:
        # Keep backend/upstream imports out of parser/help and manifest-only paths.
        from avgaussianv2.runtime import build_runtime

        runtime_factory = build_runtime
    bundle = runtime_factory(config, resolved_device)
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
    _finite(asdict(result))
    if result.best_step is None:
        (output / "worker_summary.json").unlink(missing_ok=True)
        raise RuntimeError(
            "pilot completed without a visually feasible best candidate; "
            "no best checkpoint was selected"
        )
    if store.run_fingerprint is None:
        raise RuntimeError("pilot checkpoint store did not bind a run fingerprint")
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

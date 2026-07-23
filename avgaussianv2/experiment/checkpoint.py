"""Strict, crash-safe checkpoints for bounded pilot experiments.

Fingerprint discipline is part of the public checkpoint contract. Any change to
pilot behavior or checkpoint meaning must bump ``ALGORITHM_IDENTITY`` and/or
``SCHEMA_VERSION`` as appropriate. Callers that pass closures, dynamically
generated callables, or behavior whose qualified name is not a complete stable
identity must provide an explicit ``component_identities`` entry containing a
versioned identity or configuration/code digest. Source inspection is
intentionally avoided because it is fragile across packaging environments.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

from avgaussianv2.experiment.contracts import Variant, VariantIndices
from avgaussianv2.experiment.contracts import PilotConfig
from avgaussianv2.experiment.selection import BestSelector, EarlyStopper
from avgaussianv2.config import TrainConfig


SCHEMA_VERSION = 5
STAGES = frozenset({"warmup", "joint", "complete"})
ALGORITHM_IDENTITY = "avgaussianv2.pilot-training-v5"
LOSS_STEP_API_VERSION = "avgaussianv2.loss-step-api-v1"
MAX_PILOT_CHECKPOINT_BYTES = 256 * 1024 * 1024 * 1024
MAX_CHECKPOINT_NESTING = 64
MAX_CHECKPOINT_CONTAINER_ENTRIES = 2_000_000
MAX_CHECKPOINT_TENSORS = 200_000
MAX_CHECKPOINT_TENSOR_DIMENSIONS = 16
MAX_CHECKPOINT_TENSOR_NUMEL = 1 << 40
MAX_CHECKPOINT_LOGICAL_STORAGE_BYTES = MAX_PILOT_CHECKPOINT_BYTES


class PilotResumeError(RuntimeError):
    """Raised before any state is mutated when a pilot cannot resume exactly."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _digest(name: str, value: object) -> str:
    result = _text(name, value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


@dataclass(frozen=True)
class PilotCompatibility:
    scene_id: str
    variant: str
    seed: int
    index_hash: str
    visual_checkpoint_sha256: str
    audio_checkpoint_sha256: str
    camera_mapping_sha256: str
    n_fft: int
    hop_length: int
    win_length: int
    sample_rate: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _text("scene_id", self.scene_id))
        try:
            variant = Variant(self.variant).value
        except (TypeError, ValueError) as error:
            raise ValueError(f"variant is unsupported: {self.variant!r}") from error
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "seed", _positive_int("seed", self.seed, allow_zero=True))
        for name in (
            "index_hash",
            "visual_checkpoint_sha256",
            "audio_checkpoint_sha256",
            "camera_mapping_sha256",
        ):
            object.__setattr__(self, name, _digest(name, getattr(self, name)))
        for name in ("n_fft", "hop_length", "win_length", "sample_rate"):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))

    def to_mapping(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_mapping(cls, value: object) -> PilotCompatibility:
        if not isinstance(value, Mapping):
            raise PilotResumeError("compatibility must be a mapping")
        expected = [item.name for item in fields(cls)]
        if set(value) != set(expected):
            raise PilotResumeError(
                f"compatibility fields mismatch: actual={sorted(value)} expected={expected}"
            )
        try:
            return cls(**{name: value[name] for name in expected})
        except (TypeError, ValueError) as error:
            raise PilotResumeError(f"invalid compatibility: {error}") from error


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint input does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"checkpoint input is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_index_manifest(manifest: object) -> str:
    try:
        encoded = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("index manifest must be JSON-safe and finite") from error
    return hashlib.sha256(encoded).hexdigest()


def _qualified_identity(
    value: object,
    name: str,
    overrides: Mapping[str, str],
) -> str:
    if name in overrides:
        return _text(f"component_identities[{name!r}]", overrides[name])
    module = getattr(value, "__module__", None)
    qualified = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualified, str):
        raise ValueError(f"{name} requires an explicit stable identity")
    if "<lambda>" in qualified:
        raise ValueError(f"{name} lambda requires an explicit stable identity")
    return f"{module}.{qualified}"


def build_run_fingerprint(
    *,
    pilot_config: PilotConfig,
    train_config: TrainConfig,
    visual_baseline: object,
    model: nn.Module,
    warmup_optimizer_factory: object,
    joint_optimizer_factory: object,
    warmup_step_fn: object,
    joint_step_fn: object,
    audio_loss_fn: object,
    component_identities: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the canonical identity of every behavior-affecting pilot input.

    Behavior changes require an ``ALGORITHM_IDENTITY`` bump. Closures and
    dynamic callables must use a versioned explicit identity/config digest in
    ``component_identities``; qualified names alone are not sufficient for
    behavior that can change without the name changing.
    """
    overrides = {} if component_identities is None else dict(component_identities)
    allowed_overrides = {
        "model_class",
        "warmup_optimizer_factory",
        "joint_optimizer_factory",
        "warmup_optimizer_class",
        "joint_optimizer_class",
        "warmup_step_fn",
        "joint_step_fn",
        "audio_loss_fn",
        "worker_contract_sha256",
    }
    unexpected_overrides = sorted(set(overrides) - allowed_overrides)
    if unexpected_overrides:
        raise ValueError(
            f"unexpected component identities: {unexpected_overrides}"
        )
    inputs = {
        "pilot_config": asdict(pilot_config),
        "train_config": asdict(train_config),
        "visual_baseline": _json_clone(visual_baseline, "visual_baseline"),
        "model_class": _qualified_identity(model.__class__, "model_class", overrides),
        "model_format_version": str(
            getattr(model, "checkpoint_format_version", "state-dict-v1")
        ),
        "warmup_optimizer_factory": _qualified_identity(
            warmup_optimizer_factory, "warmup_optimizer_factory", overrides
        ),
        "joint_optimizer_factory": _qualified_identity(
            joint_optimizer_factory, "joint_optimizer_factory", overrides
        ),
        "warmup_optimizer_class": overrides.get(
            "warmup_optimizer_class", "torch.optim.adam.Adam"
        ),
        "joint_optimizer_class": overrides.get(
            "joint_optimizer_class", "torch.optim.adam.Adam"
        ),
        "warmup_step_fn": _qualified_identity(
            warmup_step_fn, "warmup_step_fn", overrides
        ),
        "joint_step_fn": _qualified_identity(
            joint_step_fn, "joint_step_fn", overrides
        ),
        "audio_loss_fn": _qualified_identity(
            audio_loss_fn, "audio_loss_fn", overrides
        ),
        "worker_contract_sha256": overrides.get(
            "worker_contract_sha256", "none"
        ),
        "loss_step_api_version": LOSS_STEP_API_VERSION,
    }
    encoded = json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "algorithm": ALGORITHM_IDENTITY,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "inputs": inputs,
    }


def _validate_run_fingerprint(value: object) -> dict[str, object]:
    mapping = _require_exact_keys(
        value, {"algorithm", "sha256", "inputs"}, "run_fingerprint"
    )
    if mapping["algorithm"] != ALGORITHM_IDENTITY:
        raise PilotResumeError("run_fingerprint algorithm mismatch")
    if not isinstance(mapping["inputs"], Mapping):
        raise PilotResumeError("run_fingerprint inputs must be a mapping")
    inputs = _json_clone(mapping["inputs"], "run_fingerprint.inputs")
    _require_exact_keys(
        inputs,
        {
            "pilot_config",
            "train_config",
            "visual_baseline",
            "model_class",
            "model_format_version",
            "warmup_optimizer_factory",
            "joint_optimizer_factory",
            "warmup_optimizer_class",
            "joint_optimizer_class",
            "warmup_step_fn",
            "joint_step_fn",
            "audio_loss_fn",
            "worker_contract_sha256",
            "loss_step_api_version",
        },
        "run_fingerprint.inputs",
    )
    _require_exact_keys(
        inputs["pilot_config"],
        set(PilotConfig.__dataclass_fields__),
        "run_fingerprint.inputs.pilot_config",
    )
    _require_exact_keys(
        inputs["train_config"],
        set(TrainConfig.__dataclass_fields__),
        "run_fingerprint.inputs.train_config",
    )
    encoded = json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    if mapping["sha256"] != expected:
        raise PilotResumeError("run_fingerprint sha256 mismatch")
    return {
        "algorithm": mapping["algorithm"],
        "sha256": mapping["sha256"],
        "inputs": inputs,
    }


def validate_compatibility(
    actual: PilotCompatibility, expected: PilotCompatibility
) -> None:
    if not isinstance(actual, PilotCompatibility) or not isinstance(
        expected, PilotCompatibility
    ):
        raise TypeError("actual and expected must be PilotCompatibility")
    mismatches = [
        f"{item.name}: actual={getattr(actual, item.name)!r}, "
        f"expected={getattr(expected, item.name)!r}"
        for item in fields(PilotCompatibility)
        if getattr(actual, item.name) != getattr(expected, item.name)
    ]
    if mismatches:
        raise PilotResumeError("pilot compatibility mismatch: " + "; ".join(mismatches))


def _json_clone(value: object, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise PilotResumeError(f"{name} must contain strict JSON-safe values") from error


@dataclass(frozen=True)
class PilotResumeState:
    checkpoint_kind: str
    generation: int
    best_generation: int | None
    run_fingerprint: dict[str, object]
    stage: str
    next_warmup_position: int
    next_joint_position: int
    completed_warmup_steps: int
    completed_joint_steps: int
    maximum_positive_audio_visual_gradient: float
    pending_validation: bool
    stop_requested: bool
    stop_reason: str | None
    training_history: tuple[dict[str, object], ...]
    validation_history: tuple[dict[str, object], ...]
    validation_summary: dict[str, object] | None
    best_evaluation_summary: dict[str, object] | None
    selector: BestSelector
    stopper: EarlyStopper
    optimizer_stage: str | None
    optimizer_state_dict: dict[str, Any] | None
    model_state_dict: dict[str, Any]


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    _validate_safe_checkpoint_value(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_safe_checkpoint_value(value: object, path: str = "payload") -> None:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if torch.is_tensor(value):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise PilotResumeError(
                    f"{path} contains unsupported mapping key {key!r}"
                )
            _validate_safe_checkpoint_value(item, f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_checkpoint_value(item, f"{path}[{index}]")
        return
    raise PilotResumeError(
        f"{path} must contain only tensors and safe primitive containers; "
        f"got {type(value).__name__}"
    )


def build_pilot_payload(
    *,
    model: nn.Module,
    compatibility: PilotCompatibility,
    run_fingerprint: Mapping[str, object],
    stage: str,
    next_warmup_position: int,
    next_joint_position: int,
    optimizer: torch.optim.Optimizer | None,
    optimizer_stage: str | None,
    selector: BestSelector,
    stopper: EarlyStopper,
    training_history: Sequence[Mapping[str, object]],
    validation_history: Sequence[Mapping[str, object]],
    maximum_positive_audio_visual_gradient: float,
    checkpoint_kind: str = "latest",
    generation: int = 0,
    best_generation: int | None = None,
    pending_validation: bool = False,
    stop_requested: bool = False,
    stop_reason: str | None = None,
    validation_summary: Mapping[str, object] | None = None,
    best_evaluation_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if checkpoint_kind not in {"latest", "best"}:
        raise ValueError("checkpoint_kind must be latest or best")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    if optimizer_stage not in {None, "warmup", "joint"}:
        raise ValueError("optimizer_stage must be warmup, joint, or None")
    if (optimizer is None) != (optimizer_stage is None):
        raise ValueError("optimizer and optimizer_stage must be present together")
    gradient = float(maximum_positive_audio_visual_gradient)
    if not math.isfinite(gradient) or gradient < 0:
        raise ValueError("maximum_positive_audio_visual_gradient must be finite nonnegative")
    warmup = _positive_int("next_warmup_position", next_warmup_position, allow_zero=True)
    joint = _positive_int("next_joint_position", next_joint_position, allow_zero=True)
    generation_value = _positive_int("generation", generation, allow_zero=True)
    best_generation_value = (
        None
        if best_generation is None
        else _positive_int("best_generation", best_generation, allow_zero=True)
    )
    fingerprint = _validate_run_fingerprint(run_fingerprint)
    if not isinstance(pending_validation, bool):
        raise TypeError("pending_validation must be a boolean")
    if not isinstance(stop_requested, bool):
        raise TypeError("stop_requested must be a boolean")
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "generation": generation_value,
        "best_generation": best_generation_value,
        "run_fingerprint": fingerprint,
        "compatibility": compatibility.to_mapping(),
        "provenance": {
            "compatibility": compatibility.to_mapping(),
            "variant": compatibility.variant,
        },
        "stage": stage,
        "variant": compatibility.variant,
        "next_warmup_position": warmup,
        "next_joint_position": joint,
        "completed_warmup_steps": warmup,
        "completed_joint_steps": joint,
        "model_state_dict": model.state_dict(),
        "optimizer_stage": optimizer_stage,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "selector_state": _json_clone(selector.state_dict(), "selector_state"),
        "stopper_state": _json_clone(stopper.state_dict(), "stopper_state"),
        "training_history": _json_clone(list(training_history), "training_history"),
        "validation_history": _json_clone(list(validation_history), "validation_history"),
        "maximum_positive_audio_visual_gradient": gradient,
        "pending_validation": pending_validation,
        "stop_requested": stop_requested,
        "stop_reason": stop_reason,
        "validation_summary": (
            None
            if validation_summary is None
            else _json_clone(dict(validation_summary), "validation_summary")
        ),
        "best_evaluation_summary": (
            None
            if best_evaluation_summary is None
            else _json_clone(
                dict(best_evaluation_summary), "best_evaluation_summary"
            )
        ),
    }


def save_pilot_checkpoint(path: str | Path, **kwargs: object) -> None:
    _atomic_torch_save(Path(path), build_pilot_payload(**kwargs))


def _load_checkpoint_metadata(descriptor: int, source: Path) -> object:
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except (ImportError, AttributeError) as error:
        raise PilotResumeError(
            "FakeTensorMode is required for allocation-safe checkpoint preflight"
        ) from error
    pinned = Path(f"/proc/self/fd/{descriptor}")
    try:
        with FakeTensorMode():
            return torch.load(
                pinned,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
    except Exception as error:
        raise PilotResumeError(
            f"cannot inspect pilot checkpoint metadata {source}: {error}"
        ) from error


def _load_checkpoint_real(descriptor: int) -> object:
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(descriptor, "rb", closefd=False) as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def _validate_checkpoint_metadata(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    container_entries = 0
    tensor_count = 0
    logical_storage_bytes = 0
    while stack:
        item, depth = stack.pop()
        if depth > MAX_CHECKPOINT_NESTING:
            raise PilotResumeError("checkpoint metadata nesting limit exceeded")
        if torch.is_tensor(item):
            tensor_count += 1
            if tensor_count > MAX_CHECKPOINT_TENSORS:
                raise PilotResumeError("checkpoint tensor count limit exceeded")
            if item.ndim > MAX_CHECKPOINT_TENSOR_DIMENSIONS:
                raise PilotResumeError("checkpoint tensor dimension limit exceeded")
            numel = item.numel()
            if numel > MAX_CHECKPOINT_TENSOR_NUMEL:
                raise PilotResumeError("checkpoint tensor numel limit exceeded")
            try:
                storage_bytes = item.untyped_storage().nbytes()
            except (AttributeError, RuntimeError):
                storage_bytes = numel * item.element_size()
            logical_storage_bytes += int(storage_bytes)
            if logical_storage_bytes > MAX_CHECKPOINT_LOGICAL_STORAGE_BYTES:
                raise PilotResumeError(
                    "checkpoint logical storage byte limit exceeded"
                )
            continue
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            container_entries += len(item)
            if container_entries > MAX_CHECKPOINT_CONTAINER_ENTRIES:
                raise PilotResumeError(
                    "checkpoint metadata container entry limit exceeded"
                )
            for key, child in item.items():
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            container_entries += len(item)
            if container_entries > MAX_CHECKPOINT_CONTAINER_ENTRIES:
                raise PilotResumeError(
                    "checkpoint metadata container entry limit exceeded"
                )
            stack.extend((child, depth + 1) for child in item)

    if not isinstance(value, Mapping):
        return
    try:
        pilot = value["run_fingerprint"]["inputs"]["pilot_config"]
        warmup_steps = int(pilot["warmup_steps"])
        joint_steps = int(pilot["joint_steps"])
        validation_interval = int(pilot["validation_interval"])
        training_history = value["training_history"]
        validation_history = value["validation_history"]
    except (KeyError, TypeError, ValueError):
        return
    if len(training_history) > warmup_steps + joint_steps:
        raise PilotResumeError(
            "checkpoint training history exceeds configured step count"
        )
    maximum_validations = (
        (joint_steps + validation_interval - 1) // validation_interval
        if validation_interval > 0
        else 0
    )
    if len(validation_history) > maximum_validations:
        raise PilotResumeError(
            "checkpoint validation history exceeds configured step count"
        )


def _payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        parts = source.parts
        if len(parts) == 5 and parts[:4] == ("/", "proc", "self", "fd"):
            descriptor = os.dup(int(parts[4]))
        else:
            descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"pilot checkpoint does not exist: {source}"
        ) from None
    except OSError as error:
        raise PilotResumeError(
            f"cannot securely open pilot checkpoint {source}: {error}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PilotResumeError(
                f"pilot checkpoint must be a single-link regular file: {source}"
            )
        if metadata.st_size > MAX_PILOT_CHECKPOINT_BYTES:
            raise PilotResumeError(
                "pilot checkpoint exceeds the configured "
                f"{MAX_PILOT_CHECKPOINT_BYTES} byte limit: {source}"
            )
        metadata_value = _load_checkpoint_metadata(descriptor, source)
        _validate_checkpoint_metadata(metadata_value)
        value = _load_checkpoint_real(descriptor)
    except Exception as error:
        raise PilotResumeError(f"cannot read pilot checkpoint {source}: {error}") from error
    finally:
        os.close(descriptor)
    _validate_safe_checkpoint_value(value)
    if not isinstance(value, dict):
        raise PilotResumeError("pilot checkpoint root must be a mapping")
    return value


def _require_exact_keys(
    value: object, expected: set[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PilotResumeError(f"{name} must be a mapping")
    actual = set(value)
    missing = sorted(expected - actual, key=repr)
    unexpected = sorted(actual - expected, key=repr)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise PilotResumeError(f"{name} keys mismatch: " + ", ".join(details))
    return value


def _resume_integer(name: str, value: object, *, allow_zero: bool = False) -> int:
    try:
        return _positive_int(name, value, allow_zero=allow_zero)
    except (TypeError, ValueError) as error:
        raise PilotResumeError(f"invalid {name}: {error}") from error


def inspect_pilot_checkpoint(
    path: str | Path,
    *,
    expected_compatibility: PilotCompatibility,
    indices: VariantIndices,
    expected_run_fingerprint: Mapping[str, object] | None = None,
    model: nn.Module | None = None,
    active_resume: bool = True,
    allow_complete: bool = False,
) -> PilotResumeState:
    payload = _payload(path)
    required = {
        "schema_version", "checkpoint_kind", "generation", "best_generation",
        "run_fingerprint",
        "compatibility", "provenance",
        "stage", "variant",
        "next_warmup_position", "next_joint_position", "completed_warmup_steps",
        "completed_joint_steps", "model_state_dict", "optimizer_stage",
        "optimizer_state_dict", "selector_state", "stopper_state",
        "training_history", "validation_history",
        "maximum_positive_audio_visual_gradient", "stop_reason",
        "pending_validation", "stop_requested",
        "validation_summary", "best_evaluation_summary",
    }
    _require_exact_keys(payload, required, "pilot checkpoint root")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise PilotResumeError(
            f"schema_version must be {SCHEMA_VERSION}, got {payload['schema_version']!r}"
        )
    generation = _resume_integer("generation", payload["generation"], allow_zero=True)
    best_generation = payload["best_generation"]
    if best_generation is not None:
        best_generation = _resume_integer(
            "best_generation", best_generation, allow_zero=True
        )
        if best_generation > generation:
            raise PilotResumeError("best_generation cannot exceed generation")
    run_fingerprint = _validate_run_fingerprint(payload["run_fingerprint"])
    if expected_run_fingerprint is not None:
        expected_fingerprint = _validate_run_fingerprint(expected_run_fingerprint)
        if run_fingerprint != expected_fingerprint:
            raise PilotResumeError(
                "run_fingerprint mismatch: "
                f"actual={run_fingerprint['sha256']!r}, "
                f"expected={expected_fingerprint['sha256']!r}"
            )
    actual = PilotCompatibility.from_mapping(payload["compatibility"])
    validate_compatibility(actual, expected_compatibility)
    _require_exact_keys(
        payload["provenance"], {"compatibility", "variant"}, "provenance"
    )
    expected_provenance = {
        "compatibility": actual.to_mapping(),
        "variant": actual.variant,
    }
    if payload["provenance"] != expected_provenance:
        raise PilotResumeError("checkpoint provenance is inconsistent")
    if payload["variant"] != expected_compatibility.variant:
        raise PilotResumeError("checkpoint variant disagrees with compatibility")
    checkpoint_kind = payload["checkpoint_kind"]
    if checkpoint_kind not in {"latest", "best"}:
        raise PilotResumeError("checkpoint_kind must be latest or best")
    if active_resume and checkpoint_kind != "latest":
        raise PilotResumeError("best checkpoint cannot be used for active resume")
    stage = payload["stage"]
    if stage not in STAGES:
        raise PilotResumeError(f"invalid checkpoint stage: {stage!r}")
    if active_resume and stage == "complete" and not allow_complete:
        raise PilotResumeError("complete checkpoint cannot be resumed as active")
    warmup = _resume_integer(
        "next_warmup_position", payload["next_warmup_position"], allow_zero=True
    )
    joint = _resume_integer(
        "next_joint_position", payload["next_joint_position"], allow_zero=True
    )
    if warmup > len(indices.warmup) or joint > len(indices.joint):
        raise PilotResumeError("checkpoint positions exceed configured shared indices")
    if expected_compatibility.variant == Variant.CONDITION_OFF.value:
        if stage == "warmup":
            raise PilotResumeError("condition_off checkpoint cannot have warmup stage")
        if warmup != 0:
            raise PilotResumeError("condition_off warmup position must be zero")
    if stage == "warmup" and joint != 0:
        raise PilotResumeError("warmup checkpoint joint position must be zero")
    if stage in {"joint", "complete"} and warmup != len(indices.warmup):
        raise PilotResumeError("joint/complete checkpoint requires completed warmup")
    completed_warmup = _resume_integer(
        "completed_warmup_steps",
        payload["completed_warmup_steps"],
        allow_zero=True,
    )
    completed_joint = _resume_integer(
        "completed_joint_steps",
        payload["completed_joint_steps"],
        allow_zero=True,
    )
    if completed_warmup != warmup or completed_joint != joint:
        raise PilotResumeError("completed counts must equal exact next positions")
    pending_validation = payload["pending_validation"]
    stop_requested = payload["stop_requested"]
    if not isinstance(pending_validation, bool) or not isinstance(stop_requested, bool):
        raise PilotResumeError(
            "pending_validation and stop_requested must be booleans"
        )
    if pending_validation and stop_requested:
        raise PilotResumeError(
            "pending_validation and stop_requested cannot both be true"
        )
    optimizer_stage = payload["optimizer_stage"]
    optimizer_state = payload["optimizer_state_dict"]
    expected_optimizer_stage = (
        None if checkpoint_kind == "best" or stage == "complete" else stage
    )
    if optimizer_stage != expected_optimizer_stage:
        raise PilotResumeError(
            f"optimizer/stage mismatch: actual={optimizer_stage!r}, expected={expected_optimizer_stage!r}"
        )
    if (optimizer_state is None) != (optimizer_stage is None):
        raise PilotResumeError("optimizer state presence does not match optimizer stage")
    if optimizer_state is not None:
        if (
            not isinstance(optimizer_state, dict)
            or not isinstance(optimizer_state.get("state"), dict)
            or not isinstance(optimizer_state.get("param_groups"), list)
        ):
            raise PilotResumeError("optimizer state must contain state and param_groups")
        _require_exact_keys(
            optimizer_state, {"state", "param_groups"}, "optimizer_state_dict"
        )
        for group in optimizer_state["param_groups"]:
            if (
                not isinstance(group, dict)
                or not isinstance(group.get("params"), list)
                or any(
                    not isinstance(parameter, Integral) or isinstance(parameter, bool)
                    for parameter in group["params"]
                )
            ):
                raise PilotResumeError("optimizer parameter groups are invalid")
    histories = []
    for name in ("training_history", "validation_history"):
        value = payload[name]
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise PilotResumeError(f"{name} must be a list of mappings")
        histories.append(tuple(_json_clone(value, name)))
    training_row_keys = {
        "stage",
        "step",
        "sample_index",
        "total",
        "audio_to_visual_grad_norm",
        "losses",
        "gradient_norms",
    }
    for index, row in enumerate(histories[0]):
        _require_exact_keys(
            row, training_row_keys, f"training_history[{index}]"
        )
    for index, row in enumerate(histories[1]):
        _require_exact_keys(
            row, {"step", "summary"}, f"validation_history[{index}]"
        )
    expected_rows = [
        ("warmup", position + 1, indices.warmup[position])
        for position in range(warmup)
    ] + [
        ("joint", position + 1, indices.joint[position])
        for position in range(joint)
    ]
    if len(histories[0]) != len(expected_rows):
        raise PilotResumeError(
            "training_history length does not match completed positions"
        )
    for row, (expected_stage, expected_step, expected_index) in zip(
        histories[0], expected_rows, strict=True
    ):
        if (
            row.get("stage") != expected_stage
            or row.get("step") != expected_step
            or row.get("sample_index") != expected_index
        ):
            raise PilotResumeError(
                "training_history does not match exact shared index order"
            )
    validation_steps: list[int] = []
    for row in histories[1]:
        step = row.get("step")
        if (
            not isinstance(step, Integral)
            or isinstance(step, bool)
            or int(step) <= 0
            or int(step) > joint
        ):
            raise PilotResumeError("validation_history contains an invalid step")
        validation_steps.append(int(step))
    if validation_steps != sorted(set(validation_steps)):
        raise PilotResumeError("validation_history steps must be strictly increasing")
    try:
        _require_exact_keys(
            payload["selector_state"],
            {
                "visual_baseline",
                "psnr_tolerance_db",
                "ssim_tolerance",
                "best_step",
                "best_audio_total",
                "last_step",
            },
            "selector_state",
        )
        selector_baseline = payload["selector_state"]["visual_baseline"]
        _require_exact_keys(
            selector_baseline,
            {"rgb_psnr", "rgb_ssim"},
            "selector_state.visual_baseline",
        )
        for metric in ("rgb_psnr", "rgb_ssim"):
            _require_exact_keys(
                selector_baseline[metric],
                {"mean"},
                f"selector_state.visual_baseline.{metric}",
            )
        _require_exact_keys(
            payload["stopper_state"],
            {
                "minimum_steps",
                "patience",
                "relative_delta",
                "best",
                "stale",
                "last_step",
            },
            "stopper_state",
        )
        selector = BestSelector.from_state_dict(payload["selector_state"])
        stopper = EarlyStopper.from_state_dict(payload["stopper_state"])
    except (KeyError, TypeError, ValueError) as error:
        raise PilotResumeError(f"invalid selector/stopper state: {error}") from error
    last_validation_step = validation_steps[-1] if validation_steps else None
    if selector.last_step != last_validation_step or stopper.last_step != last_validation_step:
        raise PilotResumeError(
            "selector/stopper last_step must match validation history"
        )
    try:
        replay_selector = BestSelector(
            payload["selector_state"]["visual_baseline"],
            payload["selector_state"]["psnr_tolerance_db"],
            payload["selector_state"]["ssim_tolerance"],
        )
        replay_stopper = EarlyStopper(
            payload["stopper_state"]["minimum_steps"],
            payload["stopper_state"]["patience"],
            payload["stopper_state"]["relative_delta"],
        )
        replay_stop_requested = False
        for validation_index, row in enumerate(histories[1]):
            replay_selector.consider(row["step"], row["summary"])
            replay_stop_requested = replay_stopper.update(
                row["step"], row["summary"]["audio_total"]["mean"]
            )
            if replay_stop_requested and validation_index != len(histories[1]) - 1:
                raise ValueError("validation history continues after early stop")
    except (KeyError, TypeError, ValueError) as error:
        raise PilotResumeError(
            f"validation history cannot reproduce selector/stopper: {error}"
        ) from error
    if replay_selector.state_dict() != selector.state_dict():
        raise PilotResumeError("selector_state does not match validation history")
    if replay_stopper.state_dict() != stopper.state_dict():
        raise PilotResumeError("stopper_state does not match validation history")
    if stop_requested != replay_stop_requested:
        raise PilotResumeError("stop_requested does not match validation history")

    fingerprint_inputs = run_fingerprint["inputs"]
    pilot_values = fingerprint_inputs.get("pilot_config")
    if not isinstance(pilot_values, Mapping):
        raise PilotResumeError("run_fingerprint pilot_config is invalid")
    try:
        total_joint = _resume_integer(
            "run_fingerprint joint_steps", pilot_values["joint_steps"]
        )
        validation_interval = _resume_integer(
            "run_fingerprint validation_interval",
            pilot_values["validation_interval"],
        )
    except KeyError as error:
        raise PilotResumeError(
            f"run_fingerprint pilot_config is missing {error.args[0]}"
        ) from error
    expected_selector_config = {
        "psnr_tolerance_db": pilot_values.get("psnr_tolerance_db"),
        "ssim_tolerance": pilot_values.get("ssim_tolerance"),
    }
    for name, expected_value in expected_selector_config.items():
        if payload["selector_state"][name] != expected_value:
            raise PilotResumeError(
                f"selector_state {name} disagrees with run_fingerprint"
            )
    fingerprint_baseline = fingerprint_inputs.get("visual_baseline")
    try:
        selector_fingerprint_baseline = {
            metric: {"mean": fingerprint_baseline[metric]["mean"]}
            for metric in ("rgb_psnr", "rgb_ssim")
        }
    except (KeyError, TypeError) as error:
        raise PilotResumeError(
            "run_fingerprint visual_baseline cannot configure selector"
        ) from error
    if (
        payload["selector_state"]["visual_baseline"]
        != selector_fingerprint_baseline
    ):
        raise PilotResumeError(
            "selector_state visual_baseline disagrees with run_fingerprint"
        )
    expected_stopper_config = {
        "minimum_steps": pilot_values.get("minimum_joint_steps"),
        "patience": pilot_values.get("patience"),
        "relative_delta": pilot_values.get("minimum_relative_improvement"),
    }
    for name, expected_value in expected_stopper_config.items():
        if payload["stopper_state"][name] != expected_value:
            raise PilotResumeError(
                f"stopper_state {name} disagrees with run_fingerprint"
            )
    scheduled = [
        step
        for step in range(1, joint + 1)
        if step % validation_interval == 0
        or (step == total_joint and joint == total_joint)
    ]
    expected_pending = bool(scheduled and scheduled[-1] == joint and joint not in validation_steps)
    if pending_validation != expected_pending:
        raise PilotResumeError(
            "pending_validation does not match scheduled validation completeness"
        )
    completed_schedule = scheduled[:-1] if pending_validation else scheduled
    if validation_steps != completed_schedule:
        raise PilotResumeError(
            "validation_history does not match the configured validation schedule"
        )
    gradient = payload["maximum_positive_audio_visual_gradient"]
    if not isinstance(gradient, Real) or isinstance(gradient, bool):
        raise PilotResumeError("maximum_positive_audio_visual_gradient must be numeric")
    gradient = float(gradient)
    if not math.isfinite(gradient) or gradient < 0:
        raise PilotResumeError("maximum_positive_audio_visual_gradient must be finite nonnegative")
    if not isinstance(payload["model_state_dict"], dict):
        raise PilotResumeError("model_state_dict must be a mapping")
    if model is not None:
        _validate_model_state(payload["model_state_dict"], model)
    stop_reason = payload["stop_reason"]
    if stop_reason not in {None, "max_steps", "early_stop"}:
        raise PilotResumeError("stop_reason is invalid")
    if stage == "complete" and stop_reason is None:
        raise PilotResumeError("complete checkpoint requires stop_reason")
    if stage == "complete":
        if pending_validation:
            raise PilotResumeError("complete checkpoint cannot have pending validation")
        if stop_reason == "max_steps" and joint != len(indices.joint):
            raise PilotResumeError(
                "complete max_steps checkpoint requires exhausted joint indices"
            )
        if stop_reason == "early_stop" and not stop_requested:
            raise PilotResumeError(
                "complete early_stop checkpoint requires stop_requested"
            )
        if stop_reason == "max_steps" and stop_requested:
            raise PilotResumeError(
                "complete max_steps checkpoint cannot request early stop"
            )
    validation_summary = payload["validation_summary"]
    if validation_summary is not None and not isinstance(validation_summary, dict):
        raise PilotResumeError("validation_summary must be a mapping or None")
    expected_validation_summary = (
        None if not histories[1] else histories[1][-1].get("summary")
    )
    if validation_summary != expected_validation_summary:
        raise PilotResumeError(
            "validation_summary must match the latest validation history row"
        )
    best_evaluation_summary = payload["best_evaluation_summary"]
    if best_evaluation_summary is not None and not isinstance(
        best_evaluation_summary, dict
    ):
        raise PilotResumeError("best_evaluation_summary must be a mapping or None")
    if checkpoint_kind == "best" and best_evaluation_summary is None:
        raise PilotResumeError("best checkpoint requires best_evaluation_summary")
    expected_best_summary = None
    if selector.best_step is not None:
        expected_best_summary = next(
            (
                row.get("summary")
                for row in histories[1]
                if row.get("step") == selector.best_step
            ),
            None,
        )
    if best_evaluation_summary != expected_best_summary:
        raise PilotResumeError(
            "best_evaluation_summary must match the selected validation row"
        )
    if selector.best_step is None and best_generation is not None:
        raise PilotResumeError("best_generation requires a selected best")
    if selector.best_step is not None and best_generation is None:
        raise PilotResumeError("selected best requires best_generation")
    if checkpoint_kind == "best" and best_generation != generation:
        raise PilotResumeError(
            "best checkpoint generation must equal best_generation"
        )
    return PilotResumeState(
        checkpoint_kind=checkpoint_kind,
        generation=generation,
        best_generation=best_generation,
        run_fingerprint=run_fingerprint,
        stage=stage,
        next_warmup_position=warmup,
        next_joint_position=joint,
        completed_warmup_steps=warmup,
        completed_joint_steps=joint,
        maximum_positive_audio_visual_gradient=gradient,
        pending_validation=pending_validation,
        stop_requested=stop_requested,
        stop_reason=stop_reason,
        training_history=histories[0],
        validation_history=histories[1],
        validation_summary=validation_summary,
        best_evaluation_summary=best_evaluation_summary,
        selector=selector,
        stopper=stopper,
        optimizer_stage=optimizer_stage,
        optimizer_state_dict=optimizer_state,
        model_state_dict=payload["model_state_dict"],
    )


def restore_pilot_checkpoint(
    state: PilotResumeState,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_stage: str,
) -> None:
    if state.optimizer_stage != optimizer_stage or state.optimizer_state_dict is None:
        raise PilotResumeError(
            f"optimizer/stage mismatch: checkpoint={state.optimizer_stage!r}, "
            f"active={optimizer_stage!r}"
        )
    _validate_model_state(state.model_state_dict, model)
    current_model_state = model.state_dict()
    original_model_state = {
        name: value.detach().clone() for name, value in current_model_state.items()
    }
    original_flags = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    original_condition = getattr(model, "condition_enabled", None)
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())
    optimizer_groups = state.optimizer_state_dict.get("param_groups")
    if not isinstance(optimizer_groups, list) or len(optimizer_groups) != len(
        optimizer.param_groups
    ):
        raise PilotResumeError("checkpoint optimizer parameter groups are incompatible")
    if any(
        not isinstance(saved.get("params"), list)
        or len(saved["params"]) != len(active["params"])
        for saved, active in zip(optimizer_groups, optimizer.param_groups, strict=True)
    ):
        raise PilotResumeError("checkpoint optimizer parameters are incompatible")
    try:
        optimizer.load_state_dict(state.optimizer_state_dict)
        model.load_state_dict(state.model_state_dict, strict=True)
        for parameter, optimizer_state in tuple(optimizer.state.items()):
            optimizer.state[parameter] = _move_optimizer_value(
                optimizer_state, parameter.device
            )
    except BaseException as error:
        try:
            model.load_state_dict(original_model_state, strict=True)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(original_flags[name])
            if hasattr(model, "condition_enabled"):
                model.condition_enabled = original_condition
            torch.optim.Optimizer.load_state_dict(optimizer, original_optimizer_state)
        except BaseException as rollback_error:
            raise PilotResumeError(
                f"checkpoint rollback failed after {error!r}: {rollback_error}"
            ) from rollback_error
        if isinstance(error, Exception):
            raise PilotResumeError(
                f"checkpoint state cannot be restored: {error}"
            ) from error
        raise


def _move_optimizer_value(value: object, device: torch.device) -> object:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_optimizer_value(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_optimizer_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_value(item, device) for item in value)
    return value


def _validate_model_state(state: object, model: nn.Module) -> None:
    if not isinstance(state, dict):
        raise PilotResumeError("model_state_dict must be a mapping")
    current_model_state = model.state_dict()
    if set(state) != set(current_model_state):
        raise PilotResumeError("checkpoint model state keys do not match active model")
    for name, value in state.items():
        expected = current_model_state[name]
        if (
            not torch.is_tensor(value)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise PilotResumeError(
                f"checkpoint model tensor {name!r} is incompatible with active model"
            )


def validate_pilot_resume_model(
    state: PilotResumeState, model: nn.Module
) -> None:
    """Validate a retained resume payload against the constructed model."""
    if not isinstance(state, PilotResumeState):
        raise TypeError("state must be PilotResumeState")
    _validate_model_state(state.model_state_dict, model)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _durable_copy(source: Path, destination: Path) -> None:
    source_fd = os.open(
        source,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            0o600,
        )
    except BaseException:
        os.close(source_fd)
        raise
    source_stat = os.fstat(source_fd)
    destination_stat = os.fstat(destination_fd)
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_nlink != 1
        or not stat.S_ISREG(destination_stat.st_mode)
        or destination_stat.st_nlink != 1
    ):
        os.close(source_fd)
        os.close(destination_fd)
        raise PilotResumeError(
            "checkpoint copy endpoints must be single-link regular files"
        )
    os.ftruncate(destination_fd, 0)
    with (
        os.fdopen(source_fd, "rb") as reader,
        os.fdopen(destination_fd, "wb") as writer,
    ):
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PilotCheckpointStore:
    """Own canonical latest/best paths and fresh/resume output policy.

    I/O counters are invocation-scoped: a new store starts from zero, including
    a finalize-only resume that performs no checkpoint save.
    """

    def __init__(
        self,
        output_dir: str | Path,
        compatibility: PilotCompatibility,
        *,
        resume: bool = False,
        overwrite: bool = False,
        component_identities: Mapping[str, str] | None = None,
        checkpoint_hook: Any | None = None,
    ) -> None:
        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        self.output_dir = Path(output_dir)
        self.compatibility = compatibility
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)
        self.component_identities = (
            {} if component_identities is None else dict(component_identities)
        )
        self.checkpoint_hook = checkpoint_hook
        self.run_fingerprint: dict[str, object] | None = None
        self.generation = 0
        self.best_generation: int | None = None
        self.save_count = 0
        self.save_attempt_count = 0
        self.failed_save_count = 0
        self.save_bytes = 0
        self.save_duration_seconds = 0.0
        self.backup_copy_count = 0
        self.backup_copy_attempt_count = 0
        self.backup_copy_failure_count = 0
        self.backup_copy_bytes = 0
        self.backup_copy_duration_seconds = 0.0
        self._lock_stream: Any | None = None
        self._directory_fd: int | None = None
        self._lock_depth = 0
        self.inspected_best_state: PilotResumeState | None = None

    @property
    def pinned_output_dir(self) -> Path:
        if self._directory_fd is None:
            return self.output_dir
        return Path(f"/proc/self/fd/{self._directory_fd}")

    @property
    def latest_path(self) -> Path:
        return self.pinned_output_dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.pinned_output_dir / "best.pt"

    @property
    def lock_path(self) -> Path:
        return self.pinned_output_dir / ".pilot.lock"

    @property
    def journal_path(self) -> Path:
        return self.pinned_output_dir / ".pilot-validation-transaction.json"

    @property
    def backup_path(self) -> Path:
        return self.pinned_output_dir / ".pilot-best-backup.pt"

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "save_count": self.save_count,
            "save_attempt_count": self.save_attempt_count,
            "failed_save_count": self.failed_save_count,
            "save_bytes": self.save_bytes,
            "save_duration_seconds": self.save_duration_seconds,
            "scope": "current_invocation",
            "backup_copy_count": self.backup_copy_count,
            "backup_copy_attempt_count": self.backup_copy_attempt_count,
            "backup_copy_failure_count": self.backup_copy_failure_count,
            "backup_copy_bytes": self.backup_copy_bytes,
            "backup_copy_duration_seconds": self.backup_copy_duration_seconds,
            "cadence": "every_completed_optimizer_step",
        }

    def bind_run_fingerprint(self, value: Mapping[str, object]) -> None:
        validated = _validate_run_fingerprint(value)
        if self.run_fingerprint is not None and self.run_fingerprint != validated:
            raise PilotResumeError("checkpoint store run_fingerprint changed")
        self.run_fingerprint = validated

    def acquire(self) -> None:
        if self._lock_stream is not None:
            self._lock_depth += 1
            return
        if self.output_dir.is_symlink():
            raise PilotResumeError("pilot output directory must not be a symlink")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.lock_path,
            self.journal_path,
            self.backup_path,
            self.latest_path,
            self.best_path,
            self.output_dir / "worker_summary.json",
            self.output_dir / "training_curve.csv",
        ):
            if path.is_symlink():
                raise PilotResumeError(
                    f"pilot control path must not be a symlink: {path}"
                )
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(
            self.output_dir,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        directory_stat = os.fstat(directory_fd)
        path_stat = self.output_dir.lstat()
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or (directory_stat.st_dev, directory_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            os.close(directory_fd)
            raise PilotResumeError("pilot output directory identity changed")
        try:
            descriptor = os.open(
                self.lock_path.name, flags, 0o600, dir_fd=directory_fd
            )
        except BaseException:
            os.close(directory_fd)
            raise
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            os.close(descriptor)
            os.close(directory_fd)
            raise PilotResumeError("pilot lock must be a single-link regular file")
        stream = os.fdopen(descriptor, "a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.close()
            os.close(directory_fd)
            raise PilotResumeError(
                f"pilot output is owned by another process: {self.output_dir}"
            ) from error
        metadata = {
            "pid": os.getpid(),
            "compatibility": self.compatibility.to_mapping(),
            "acquired_unix_seconds": time.time(),
        }
        stream.seek(0)
        stream.truncate()
        stream.write(
            (json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )
        stream.flush()
        os.fsync(stream.fileno())
        self._lock_stream = stream
        self._directory_fd = directory_fd
        self._lock_depth = 1

    def release(self) -> None:
        if self._lock_stream is None:
            return
        if self._lock_depth > 1:
            self._lock_depth -= 1
            return
        stream = self._lock_stream
        directory_fd = self._directory_fd
        self._lock_stream = None
        self._directory_fd = None
        self._lock_depth = 0
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
            if directory_fd is not None:
                os.close(directory_fd)

    def __enter__(self) -> PilotCheckpointStore:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def _require_lock(self) -> None:
        if self._lock_stream is None or self._directory_fd is None:
            raise RuntimeError("pilot checkpoint store must be acquired")
        pinned = os.fstat(self._directory_fd)
        try:
            current = self.output_dir.lstat()
        except OSError as error:
            raise PilotResumeError(
                "pilot output directory disappeared while locked"
            ) from error
        if (
            not stat.S_ISDIR(current.st_mode)
            or (pinned.st_dev, pinned.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise PilotResumeError(
                "pilot output directory changed while locked"
            )

    def verify_output_identity(self) -> None:
        """Require the locked output pathname to still name the pinned directory."""
        self._require_lock()

    def _record_save(self, path: Path, started: float, *, succeeded: bool) -> None:
        duration = time.perf_counter() - started
        size = 0
        if succeeded:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.save_count += 1
            self.save_bytes += size
        else:
            self.failed_save_count += 1
        self.save_attempt_count += 1
        self.save_duration_seconds += duration
        if self.checkpoint_hook is not None:
            try:
                self.checkpoint_hook(
                    {
                        "path": str(path),
                        "bytes": size,
                        "duration_seconds": duration,
                        "succeeded": succeeded,
                        "save_count": self.save_count,
                        "save_attempt_count": self.save_attempt_count,
                    }
                )
            except Exception:
                # Measurement must never change checkpoint durability semantics.
                pass

    def _save(self, path: Path, **kwargs: object) -> None:
        if self.run_fingerprint is None:
            raise RuntimeError("run_fingerprint must be bound before saving")
        started = time.perf_counter()
        succeeded = False
        try:
            kwargs.setdefault("best_generation", self.best_generation)
            save_pilot_checkpoint(
                path,
                run_fingerprint=self.run_fingerprint,
                **kwargs,
            )
            succeeded = True
        finally:
            self._record_save(path, started, succeeded=succeeded)

    def _backup_best(self) -> None:
        started = time.perf_counter()
        size = 0
        succeeded = False
        try:
            size = self.best_path.stat().st_size
            _durable_copy(self.best_path, self.backup_path)
            succeeded = True
        finally:
            self.backup_copy_attempt_count += 1
            if succeeded:
                self.backup_copy_count += 1
                self.backup_copy_bytes += size
            else:
                self.backup_copy_failure_count += 1
            self.backup_copy_duration_seconds += time.perf_counter() - started

    def publish_latest(self, **kwargs: object) -> None:
        self._require_lock()
        generation = self.generation + 1
        self._save(
            self.latest_path,
            checkpoint_kind="latest",
            generation=generation,
            **kwargs,
        )
        self.generation = generation

    def publish_validation(
        self,
        *,
        latest_kwargs: Mapping[str, object],
        best_kwargs: Mapping[str, object] | None,
    ) -> None:
        self._require_lock()
        if best_kwargs is None:
            self.publish_latest(**dict(latest_kwargs))
            return
        generation = self.generation + 1
        had_best = self.best_path.is_file()
        if had_best:
            self._backup_best()
        _atomic_json(
            self.journal_path,
            {"generation": generation, "had_best": had_best},
        )
        try:
            best_payload = dict(best_kwargs)
            best_payload["best_generation"] = generation
            self._save(
                self.best_path,
                checkpoint_kind="best",
                generation=generation,
                **best_payload,
            )
            latest_payload = dict(latest_kwargs)
            latest_payload["best_generation"] = generation
            self._save(
                self.latest_path,
                checkpoint_kind="latest",
                generation=generation,
                **latest_payload,
            )
            self.generation = generation
            self.best_generation = generation
        except BaseException as error:
            latest_generation = self._read_generation_safely(self.latest_path)
            if latest_generation == generation:
                try:
                    _fsync_directory(self.pinned_output_dir)
                except OSError:
                    # The canonical pair may be committed, but durability is
                    # ambiguous. Preserve the journal and backup for prepare().
                    if not isinstance(error, Exception):
                        raise error
                    raise
                self.generation = generation
                self.best_generation = generation
                try:
                    self._clear_transaction()
                except BaseException:
                    if not isinstance(error, Exception):
                        raise error
                    raise
                if not isinstance(error, Exception):
                    raise
                return
            if latest_generation is not None and latest_generation < generation:
                self._rollback_transaction(had_best)
            # Missing/corrupt/ahead latest is ambiguous: preserve recovery files.
            raise
        self._clear_transaction()

    @staticmethod
    def _read_generation_safely(path: Path) -> int | None:
        if not path.is_file():
            return -1
        try:
            return _resume_integer(
                "generation", _payload(path).get("generation"), allow_zero=True
            )
        except (OSError, PilotResumeError):
            return None

    def _clear_transaction(self) -> None:
        self.journal_path.unlink(missing_ok=True)
        self.backup_path.unlink(missing_ok=True)
        for temporary in self.pinned_output_dir.glob(".pilot-best-restore.*.tmp"):
            temporary.unlink(missing_ok=True)
        _fsync_directory(self.pinned_output_dir)

    def _rollback_transaction(self, had_best: bool) -> None:
        if had_best:
            if not self.backup_path.is_file():
                raise PilotResumeError(
                    "checkpoint transaction requires missing best backup"
                )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".pilot-best-restore.",
                suffix=".tmp",
                dir=self.pinned_output_dir,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                _durable_copy(self.backup_path, temporary)
                os.replace(temporary, self.best_path)
                _fsync_directory(self.pinned_output_dir)
            finally:
                temporary.unlink(missing_ok=True)
        elif not had_best:
            self.best_path.unlink(missing_ok=True)
            _fsync_directory(self.pinned_output_dir)
        # The rollback is durable before recovery metadata is removed.
        self.journal_path.unlink(missing_ok=True)
        self.backup_path.unlink(missing_ok=True)
        _fsync_directory(self.pinned_output_dir)

    def _recover_transaction(self) -> None:
        if not self.journal_path.exists():
            return
        try:
            journal = json.loads(self.journal_path.read_text(encoding="utf-8"))
            mapping = _require_exact_keys(
                journal, {"generation", "had_best"}, "transaction journal"
            )
            generation = _resume_integer(
                "transaction generation", mapping["generation"], allow_zero=True
            )
            if not isinstance(mapping["had_best"], bool):
                raise PilotResumeError("transaction had_best must be boolean")
            if mapping["had_best"] and not self.backup_path.is_file():
                raise PilotResumeError(
                    "checkpoint transaction requires missing best backup"
                )
            latest_generation = None
            if self.latest_path.is_file():
                latest_generation = _resume_integer(
                    "latest generation",
                    _payload(self.latest_path).get("generation"),
                    allow_zero=True,
                )
            if latest_generation == generation:
                _fsync_directory(self.pinned_output_dir)
                self._clear_transaction()
            elif latest_generation is None or latest_generation < generation:
                self._rollback_transaction(mapping["had_best"])
            else:
                raise PilotResumeError(
                    "latest checkpoint generation is ahead of transaction journal"
                )
        except (OSError, json.JSONDecodeError) as error:
            raise PilotResumeError(
                f"cannot recover checkpoint transaction: {error}"
            ) from error

    def recover(self) -> None:
        self._require_lock()
        if self.resume:
            self._recover_transaction()

    def prepare(
        self,
        *,
        resume_state: PilotResumeState | None = None,
        recover: bool = True,
    ) -> None:
        self._require_lock()
        if self.resume:
            if recover:
                self._recover_transaction()
            if not self.latest_path.is_file():
                raise FileNotFoundError(
                    f"resume requires readable latest checkpoint: {self.latest_path}"
                )
            if resume_state is None:
                payload = _payload(self.latest_path)
                self.generation = _resume_integer(
                    "generation", payload.get("generation"), allow_zero=True
                )
                raw_best_generation = payload.get("best_generation")
                self.best_generation = (
                    None
                    if raw_best_generation is None
                    else _resume_integer(
                        "best_generation", raw_best_generation, allow_zero=True
                    )
                )
            else:
                self.generation = resume_state.generation
                self.best_generation = resume_state.best_generation
            if resume_state is None and self.best_path.is_file():
                best_payload = _payload(self.best_path)
                best_generation = _resume_integer(
                    "best generation",
                    best_payload.get("generation"),
                    allow_zero=True,
                )
                if best_generation > self.generation:
                    raise PilotResumeError(
                        "best checkpoint generation is ahead of latest without journal"
                    )
            return
        existing = [
            path
            for path in self.pinned_output_dir.iterdir()
            if path.name != self.lock_path.name
        ]
        if existing and not self.overwrite:
            raise FileExistsError(
                "fresh pilot refuses nonempty output directory: "
                + ", ".join(path.name for path in existing)
            )
        if self.overwrite:
            for path in (
                self.latest_path,
                self.best_path,
                self.journal_path,
                self.backup_path,
                self.pinned_output_dir / "worker_summary.json",
                self.pinned_output_dir / "training_curve.csv",
            ):
                path.unlink(missing_ok=True)
            for pattern in (
                ".latest.pt.*.tmp",
                ".best.pt.*.tmp",
                ".worker_summary.json.*.tmp",
                ".training_curve.csv.*.tmp",
                ".pilot-best-restore.*.tmp",
            ):
                for temporary in self.pinned_output_dir.glob(pattern):
                    temporary.unlink(missing_ok=True)
            shutil.rmtree(self.pinned_output_dir / "validation", ignore_errors=True)
        self.generation = 0
        self.best_generation = None


__all__ = [
    "PilotCheckpointStore",
    "PilotCompatibility",
    "PilotResumeError",
    "PilotResumeState",
    "build_pilot_payload",
    "build_run_fingerprint",
    "hash_index_manifest",
    "inspect_pilot_checkpoint",
    "restore_pilot_checkpoint",
    "save_pilot_checkpoint",
    "sha256_file",
    "validate_compatibility",
    "validate_pilot_resume_model",
]

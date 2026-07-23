"""Strict, crash-safe checkpoints for bounded pilot experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

from avgaussianv2.experiment.contracts import Variant, VariantIndices
from avgaussianv2.experiment.selection import BestSelector, EarlyStopper


SCHEMA_VERSION = 2
STAGES = frozenset({"warmup", "joint", "complete"})


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
    stage: str
    next_warmup_position: int
    next_joint_position: int
    completed_warmup_steps: int
    completed_joint_steps: int
    maximum_positive_audio_visual_gradient: float
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
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
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


def _payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"pilot checkpoint does not exist: {source}")
    if not source.is_file():
        raise PilotResumeError(f"pilot checkpoint is not a regular file: {source}")
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PilotResumeError(f"cannot read pilot checkpoint {source}: {error}") from error
    if not isinstance(value, dict):
        raise PilotResumeError("pilot checkpoint root must be a mapping")
    return value


def inspect_pilot_checkpoint(
    path: str | Path,
    *,
    expected_compatibility: PilotCompatibility,
    indices: VariantIndices,
    model: nn.Module | None = None,
    active_resume: bool = True,
    allow_complete: bool = False,
) -> PilotResumeState:
    payload = _payload(path)
    required = {
        "schema_version", "checkpoint_kind", "compatibility", "provenance",
        "stage", "variant",
        "next_warmup_position", "next_joint_position", "completed_warmup_steps",
        "completed_joint_steps", "model_state_dict", "optimizer_stage",
        "optimizer_state_dict", "selector_state", "stopper_state",
        "training_history", "validation_history",
        "maximum_positive_audio_visual_gradient", "stop_reason",
        "validation_summary", "best_evaluation_summary",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise PilotResumeError(f"pilot checkpoint is missing {missing[0]}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise PilotResumeError(
            f"schema_version must be {SCHEMA_VERSION}, got {payload['schema_version']!r}"
        )
    actual = PilotCompatibility.from_mapping(payload["compatibility"])
    validate_compatibility(actual, expected_compatibility)
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
    warmup = _positive_int(
        "next_warmup_position", payload["next_warmup_position"], allow_zero=True
    )
    joint = _positive_int(
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
    if payload["completed_warmup_steps"] != warmup or payload["completed_joint_steps"] != joint:
        raise PilotResumeError("completed counts must equal exact next positions")
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
        selector = BestSelector.from_state_dict(payload["selector_state"])
        stopper = EarlyStopper.from_state_dict(payload["stopper_state"])
    except (KeyError, TypeError, ValueError) as error:
        raise PilotResumeError(f"invalid selector/stopper state: {error}") from error
    last_validation_step = validation_steps[-1] if validation_steps else None
    if selector.last_step != last_validation_step or stopper.last_step != last_validation_step:
        raise PilotResumeError(
            "selector/stopper last_step must match validation history"
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
    return PilotResumeState(
        checkpoint_kind=checkpoint_kind,
        stage=stage,
        next_warmup_position=warmup,
        next_joint_position=joint,
        completed_warmup_steps=warmup,
        completed_joint_steps=joint,
        maximum_positive_audio_visual_gradient=gradient,
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
        device = next(model.parameters()).device
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
    except (RuntimeError, ValueError, TypeError) as error:
        model.load_state_dict(original_model_state, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_flags[name])
        if hasattr(model, "condition_enabled"):
            model.condition_enabled = original_condition
        torch.optim.Optimizer.load_state_dict(optimizer, original_optimizer_state)
        raise PilotResumeError(f"checkpoint state cannot be restored: {error}") from error


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


class PilotCheckpointStore:
    """Own canonical latest/best paths and fresh/resume output policy."""

    def __init__(
        self,
        output_dir: str | Path,
        compatibility: PilotCompatibility,
        *,
        resume: bool = False,
        overwrite: bool = False,
    ) -> None:
        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        self.output_dir = Path(output_dir)
        self.compatibility = compatibility
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)
        self.latest_path = self.output_dir / "latest.pt"
        self.best_path = self.output_dir / "best.pt"

    def prepare(self) -> None:
        if self.resume:
            if not self.latest_path.is_file():
                raise FileNotFoundError(
                    f"resume requires readable latest checkpoint: {self.latest_path}"
                )
            return
        existing = list(self.output_dir.iterdir()) if self.output_dir.exists() else []
        if existing and not self.overwrite:
            raise FileExistsError(
                "fresh pilot refuses nonempty output directory: "
                + ", ".join(path.name for path in existing)
            )
        if self.overwrite:
            self.latest_path.unlink(missing_ok=True)
            self.best_path.unlink(missing_ok=True)


__all__ = [
    "PilotCheckpointStore",
    "PilotCompatibility",
    "PilotResumeError",
    "PilotResumeState",
    "build_pilot_payload",
    "hash_index_manifest",
    "inspect_pilot_checkpoint",
    "restore_pilot_checkpoint",
    "save_pilot_checkpoint",
    "sha256_file",
    "validate_compatibility",
]

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from avgaussianv2.config import ProjectConfig


SCHEMA_VERSION = 1


class CheckpointCompatibilityError(RuntimeError):
    """Raised before loading state that belongs to an incompatible experiment."""


@dataclass(frozen=True)
class CheckpointState:
    visual_state_dict: dict[str, Any]
    audio_state_dict: dict[str, Any]
    condition_state_dict: dict[str, Any]
    film_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    resolved_config: dict[str, Any]
    compatibility: dict[str, Any]
    provenance: dict[str, Any]
    stage: str
    step: int
    loss_history: list[dict[str, float]]


@dataclass(frozen=True)
class ResumeState:
    stage: str
    step: int
    loss_history: list[dict[str, float]]
    provenance: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _mapping_hash(mapping: Mapping[str, int]) -> str:
    encoded = json.dumps(dict(mapping), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compatibility(config: ProjectConfig) -> dict[str, Any]:
    return {
        "scene_id": config.scene.scene_id,
        "camera_mapping": _mapping_hash(config.scene.camera_mapping),
        "embedding_dim": config.model.embedding_dim,
        "n_fft": config.model.n_fft,
        "hop_length": config.model.hop_length,
        "win_length": config.model.win_length,
        "sample_rate": config.model.sample_rate,
        "audio_model_class": config.model.audio_model_class,
    }


def _film_module(model: nn.Module) -> nn.Module | None:
    try:
        return model.audio.conditioned_renderer.film
    except (AttributeError, RuntimeError):
        return None


def build_checkpoint_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ProjectConfig,
    provenance: Mapping[str, Any],
    stage: str,
    step: int,
    loss_history: Sequence[Mapping[str, float]],
) -> CheckpointState:
    film = _film_module(model)
    return CheckpointState(
        visual_state_dict=model.visual.state_dict(),
        audio_state_dict=model.audio.state_dict(),
        condition_state_dict=model.condition_encoder.state_dict(),
        film_state_dict={} if film is None else film.state_dict(),
        optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
        resolved_config=_json_safe(asdict(config)),
        compatibility=_compatibility(config),
        provenance=dict(provenance),
        stage=str(stage),
        step=int(step),
        loss_history=[dict(row) for row in loss_history],
    )


def save_checkpoint(path: str | Path, state: CheckpointState) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        **asdict(state),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_payload(payload: object, expected: ProjectConfig) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint root must be a mapping")
    required = (
        "visual_state_dict",
        "audio_state_dict",
        "condition_state_dict",
        "film_state_dict",
        "compatibility",
        "provenance",
        "stage",
        "step",
        "loss_history",
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"schema_version must be {SCHEMA_VERSION}, got {payload.get('schema_version')}"
        )
    for key in required:
        if key not in payload:
            raise CheckpointCompatibilityError(f"checkpoint is missing {key}")
    actual_compatibility = payload["compatibility"]
    expected_compatibility = _compatibility(expected)
    for key, expected_value in expected_compatibility.items():
        actual_value = actual_compatibility.get(key)
        if actual_value != expected_value:
            raise CheckpointCompatibilityError(
                f"checkpoint {key} mismatch: {actual_value!r} != {expected_value!r}"
            )
    return payload


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    expected_config: ProjectConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> ResumeState:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"fusion checkpoint does not exist: {checkpoint_path}")
    payload = _validate_payload(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        expected_config,
    )
    try:
        model.visual.load_state_dict(payload["visual_state_dict"], strict=True)
        model.audio.load_state_dict(payload["audio_state_dict"], strict=True)
        model.condition_encoder.load_state_dict(payload["condition_state_dict"], strict=True)
        film = _film_module(model)
        if payload["film_state_dict"]:
            if film is None:
                raise CheckpointCompatibilityError(
                    "checkpoint contains FiLM state but model has no FiLM adapters"
                )
            film.load_state_dict(payload["film_state_dict"], strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"checkpoint tensor shape mismatch: {error}") from error
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return ResumeState(
        stage=str(payload["stage"]),
        step=int(payload["step"]),
        loss_history=[dict(row) for row in payload["loss_history"]],
        provenance=dict(payload["provenance"]),
    )

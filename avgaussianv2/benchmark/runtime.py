"""Production construction and identity checks for the cam38 benchmark worker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from avgaussianv2.benchmark.assets import audit_protocol_config
from avgaussianv2.config import ProjectConfig, TrainConfig, load_project_config
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.losses import AudioLoss
from avgaussianv2.runtime import build_runtime

TRAIN_CAMERAS = tuple(f"cam{index:02d}" for index in range(38))
TEST_CAMERA = "cam38"


@dataclass(frozen=True)
class BenchmarkRuntime:
    model: nn.Module
    train_samples: Sequence[AlignedAVSample]
    train_config: TrainConfig
    audio_loss_fn: AudioLoss
    config_sha256: str
    source_sha256: str
    visual_initialization_sha256: str
    audio_initialization_sha256: str
    model_initialization_sha256: str
    dataset_identity_sha256: str
    dataset_sample_ids: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_sha256(model: nn.Module, prefix: str | None = None) -> str:
    """Hash named tensor values, not pickle/container representation."""
    digest = hashlib.sha256()
    selected = [
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if prefix is None or name.startswith(prefix)
    ]
    if not selected:
        label = "model" if prefix is None else prefix.removesuffix(".")
        raise ValueError(f"runtime model has no {label} state")
    for name, tensor in sorted(selected):
        value = tensor.detach().cpu().contiguous()
        metadata = {
            "dtype": str(value.dtype),
            "name": name,
            "shape": list(value.shape),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dataset_identity(
    samples: Sequence[AlignedAVSample], config: ProjectConfig
) -> tuple[tuple[str, ...], str]:
    records = getattr(samples, "records", None)
    if records is None:
        raise TypeError("production training dataset must expose ordered records")
    identities: list[str] = []
    order_keys: list[tuple[int, int]] = []
    for record in records:
        camera = str(record.camera)
        if camera not in config.scene.train_cameras:
            raise ValueError(
                f"training dataset contains non-training camera {camera!r}"
            )
        expected_index = config.scene.camera_mapping[camera]
        if int(record.camera_index) != expected_index:
            raise ValueError(f"training dataset camera mapping mismatch for {camera}")
        order_keys.append(
            (int(record.frame_index), config.scene.train_cameras.index(camera))
        )
        identity = {
            "camera": camera,
            "camera_index": int(record.camera_index),
            "frame_index": int(record.frame_index),
            "scene_id": config.scene.scene_id,
            "time_seconds": float(record.time_seconds),
        }
        identities.append(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    if not identities:
        raise ValueError("production training dataset is empty")
    if len(set(identities)) != len(identities):
        raise ValueError("production training dataset has duplicate sample IDs")
    if order_keys != sorted(order_keys):
        raise ValueError("production training dataset sample-ID order is not canonical")
    value = tuple(identities)
    return value, _json_sha256(list(value))


def build_production_runtime(
    *,
    config_path: Path,
    device: torch.device,
    trusted_upstream_artifacts: bool,
) -> BenchmarkRuntime:
    """Build train-only state and bind it to canonical source evidence.

    ``source_sha256`` is the SHA-256 of canonical JSON containing the real
    project-config, dataset-manifest, visual-checkpoint, audio-checkpoint,
    camera-mapping, and ordered training-sample identity digests.
    """
    config_path = Path(config_path)
    audit_protocol_config(config_path)
    config = load_project_config(config_path)
    if config.scene.train_cameras != TRAIN_CAMERAS:
        raise ValueError("benchmark config must train on exactly cam00 through cam37")
    if config.scene.eval_cameras != (TEST_CAMERA,):
        raise ValueError("benchmark config must reserve exactly cam38 for evaluation")
    if config.scene.camera_mapping != {f"cam{index:02d}": index for index in range(39)}:
        raise ValueError("benchmark config camera mapping must be exactly cam00..cam38")
    bundle = build_runtime(
        config,
        device,
        trusted_upstream_artifacts=trusted_upstream_artifacts,
        include_eval=False,
    )
    if bundle.eval_samples is not None:
        raise RuntimeError("benchmark production runtime constructed eval samples")
    sample_ids, dataset_identity_sha256 = _dataset_identity(
        bundle.train_samples, config
    )
    model_initialization_sha256 = _state_sha256(bundle.model)
    evidence = {
        "audio_checkpoint_sha256": _file_sha256(config.paths.audio_checkpoint),
        "camera_mapping_sha256": _json_sha256(config.scene.camera_mapping),
        "config_sha256": _file_sha256(config_path),
        "dataset_identity_sha256": dataset_identity_sha256,
        "dataset_manifest_sha256": _file_sha256(config.paths.manifest),
        "visual_checkpoint_sha256": _file_sha256(config.paths.visual_checkpoint),
        "model_initialization_sha256": model_initialization_sha256,
    }
    return BenchmarkRuntime(
        model=bundle.model,
        train_samples=bundle.train_samples,
        train_config=config.train,
        audio_loss_fn=bundle.audio_loss_fn,
        config_sha256=evidence["config_sha256"],
        source_sha256=_json_sha256(evidence),
        visual_initialization_sha256=_state_sha256(bundle.model, "visual."),
        audio_initialization_sha256=_state_sha256(bundle.model, "audio."),
        model_initialization_sha256=model_initialization_sha256,
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_sample_ids=sample_ids,
    )


__all__ = ["BenchmarkRuntime", "build_production_runtime"]

"""Production construction and identity checks for the cam38 benchmark worker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from avgaussianv2.benchmark.assets import AssetAuditError, audit_protocol_config
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


def state_sha256(model: nn.Module, prefix: str | None = None) -> str:
    return _state_sha256(model, prefix)


def upstream_source_inventory(config: ProjectConfig) -> dict[str, str]:
    """Hash every upstream source file imported by the production adapters."""
    roots = {
        "audiogs": config.paths.audio_upstream_root,
        "ftgspp": config.paths.visual_upstream_root,
    }
    relative = {
        "audiogs": (
            Path("configs/audio_3dgs_replaynvas_viewpoint.yaml"),
            Path("tools/train_audio_3dgs_viewpoint.py"),
            *(
                path.relative_to(roots["audiogs"])
                for path in sorted(roots["audiogs"].joinpath("libs").rglob("*.py"))
            ),
        ),
        "ftgspp": tuple(
            path.relative_to(roots["ftgspp"])
            for path in sorted(roots["ftgspp"].joinpath("ftgspp").rglob("*.py"))
        ),
    }
    inventory: dict[str, str] = {}
    for kind, names in relative.items():
        for relative_path in names:
            name = relative_path.as_posix()
            path = (roots[kind] / name).resolve()
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"unsafe or missing {kind} upstream source: {path}")
            inventory[f"{kind}:{name}"] = _file_sha256(path)
    return inventory


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


def load_audited_benchmark_config(config_path: Path) -> tuple[ProjectConfig, str]:
    """Load a canonical or orchestrator-resolved config and return source SHA."""
    config_path = Path(config_path)
    source_sha256 = _file_sha256(config_path)
    try:
        audit_protocol_config(config_path)
    except AssetAuditError:
        # Orchestration materializes an absolute-path copy outside configs/.
        # Bind it back to the audited immutable source instead of weakening the
        # canonical Task11 path checks.
        if config_path.name != "resolved_project.yaml":
            raise
        origin_path = config_path.with_name("resolved_project.origin.json")
        try:
            origin = json.loads(origin_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("resolved benchmark config origin is missing") from error
        if (
            not isinstance(origin, dict)
            or set(origin)
            != {
                "schema",
                "version",
                "source_path",
                "source_sha256",
                "resolved_sha256",
            }
            or origin["schema"] != "avgaussianv2.cam38-resolved-config-origin"
            or origin["version"] != 1
            or origin["resolved_sha256"] != _file_sha256(config_path)
        ):
            raise ValueError("resolved benchmark config origin contract mismatch")
        source = Path(origin["source_path"])
        if not source.is_absolute() or _file_sha256(source) != origin["source_sha256"]:
            raise ValueError("resolved benchmark source config hash mismatch")
        audit_protocol_config(source)
        source_sha256 = origin["source_sha256"]
        source_config = load_project_config(source)
        resolved_config = load_project_config(config_path)
        path_names = (
            "visual_upstream_root",
            "audio_upstream_root",
            "visual_checkpoint",
            "audio_checkpoint",
            "manifest",
            "visual_memmap",
        )
        if (
            source_config.scene != resolved_config.scene
            or source_config.model != resolved_config.model
            or source_config.train != resolved_config.train
            or any(
                (
                    getattr(source_config.paths, name) is None
                    or getattr(resolved_config.paths, name) is None
                )
                and getattr(source_config.paths, name)
                != getattr(resolved_config.paths, name)
                or (
                    getattr(source_config.paths, name) is not None
                    and getattr(resolved_config.paths, name) is not None
                    and getattr(source_config.paths, name).resolve()
                    != getattr(resolved_config.paths, name).resolve()
                )
                for name in path_names
            )
        ):
            raise ValueError("resolved benchmark config changes protocol semantics")
    config = load_project_config(config_path)
    if config.scene.train_cameras != TRAIN_CAMERAS:
        raise ValueError("benchmark config must train on exactly cam00 through cam37")
    if config.scene.eval_cameras != (TEST_CAMERA,):
        raise ValueError("benchmark config must reserve exactly cam38 for evaluation")
    if config.scene.camera_mapping != {f"cam{index:02d}": index for index in range(39)}:
        raise ValueError("benchmark config camera mapping must be exactly cam00..cam38")
    return config, source_sha256


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
    config, _ = load_audited_benchmark_config(config_path)
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
        "upstream_source_inventory": upstream_source_inventory(config),
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


__all__ = [
    "BenchmarkRuntime",
    "build_production_runtime",
    "load_audited_benchmark_config",
    "state_sha256",
    "upstream_source_inventory",
]

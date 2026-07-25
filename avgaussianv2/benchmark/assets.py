from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from avgaussianv2.config import load_project_config

PROTOCOL = "dual_dataset_cam38_v1"
TRAIN_CAMERAS = tuple(f"cam{index:02d}" for index in range(38))
TEST_CAMERA = "cam38"
EXPECTED = {
    "scene1_opera": {"test_samples": 130, "audio_updates": 2_318},
    "Scene7playing": {"test_samples": 293, "audio_updates": 6_954},
}
_IMAGE_KINDS = {"rgb", "image", "video", "frame", "depth", "depth_like"}
_INIT_USAGES = {
    "image_driven_initialization",
    "point_initialization",
    "sfm",
    "colmap",
    "temporal_flow",
}


class AssetAuditError(ValueError):
    """Raised when assets do not satisfy the immutable cam38 protocol."""


def _load_mapping(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AssetAuditError(f"{path}: root must be a mapping")
    return payload


def audit_protocol_config(path: str | Path) -> dict[str, Any]:
    """Fail closed unless a project config is the exact frozen cam38 protocol."""
    config_path = Path(path)
    project = load_project_config(config_path)
    raw = _load_mapping(config_path)
    benchmark = raw.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise AssetAuditError("benchmark section is required")
    if project.scene.scene_id not in EXPECTED:
        raise AssetAuditError(f"unsupported benchmark scene {project.scene.scene_id!r}")
    if project.scene.train_cameras != TRAIN_CAMERAS:
        raise AssetAuditError("training cameras must be exactly cam00 through cam37")
    if project.scene.eval_cameras != (TEST_CAMERA,):
        raise AssetAuditError("evaluation camera must be exactly cam38")
    expected = EXPECTED[project.scene.scene_id]
    requirements = {
        "protocol": PROTOCOL,
        "test_camera": TEST_CAMERA,
        "expected_test_samples": expected["test_samples"],
        "seed": 42,
        "continuation_updates": 30_000,
        "conditioner_warmup_steps": 2_000,
        "report_steps": [5_000, 10_000, 30_000],
    }
    for key, wanted in requirements.items():
        if benchmark.get(key) != wanted:
            raise AssetAuditError(f"benchmark.{key} must be {wanted!r}")
    budgets = benchmark.get("native_budgets")
    wanted_budgets = {
        "audiogs_epochs": 61,
        "audiogs_batch_size": 1,
        "audiogs_resolved_updates": expected["audio_updates"],
        "ftgspp_updates": 30_000,
        "ftgspp_batch_size": 1,
    }
    if budgets != wanted_budgets:
        raise AssetAuditError(f"benchmark.native_budgets must be {wanted_budgets!r}")

    namespace = f"cam38_strict/{project.scene.scene_id}"
    bound_paths = (
        project.paths.visual_checkpoint,
        project.paths.audio_checkpoint,
        project.paths.visual_memmap,
    )
    if any(value is None or namespace not in value.as_posix() for value in bound_paths):
        raise AssetAuditError(f"upstream assets must use independent namespace {namespace}")
    return dict(raw)


def audit_initialization_provenance(
    value: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Reject held-out image targets from every image-driven init input."""
    payload = _load_mapping(value)
    if payload.get("test_camera") != TEST_CAMERA:
        raise AssetAuditError("provenance test_camera must be cam38")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise AssetAuditError("provenance assets must be a list")
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise AssetAuditError(f"provenance asset {index} must be a mapping")
        kind = str(asset.get("kind", "")).lower()
        usage = str(asset.get("usage", "")).lower()
        camera = str(asset.get("camera", ""))
        if camera == TEST_CAMERA and kind in _IMAGE_KINDS and usage in _INIT_USAGES:
            raise AssetAuditError(
                f"cam38 RGB/depth target cannot enter image-driven initialization: "
                f"{asset.get('path', '<unknown>')}"
            )
    return dict(payload)


def audit_ftgspp_train_source(path: str | Path) -> dict[str, Any]:
    """Verify the actual image-driven source contains only cam00..cam37."""
    source = Path(path)
    videos = tuple(sorted(item.stem for item in source.glob("cam*.mp4")))
    if TEST_CAMERA in videos:
        raise AssetAuditError(f"cam38 RGB cannot enter FTGS++ train source {source}")
    if videos != TRAIN_CAMERAS:
        raise AssetAuditError(
            "FTGS++ train source must contain exactly cam00 through cam37"
        )
    return {"path": str(source), "cameras": list(videos)}


def audit_audiogs_conversion(
    value: str | Path | Mapping[str, Any],
    *,
    expected_clips: int,
    epochs: int,
) -> dict[str, Any]:
    """Bind viewpoint 39 to cam38 and resolve the shared-model update budget."""
    payload = _load_mapping(value)
    cameras = tuple(str(item) for item in payload.get("camera_names", ()))
    if cameras != (*TRAIN_CAMERAS, TEST_CAMERA):
        raise AssetAuditError("AudioGS conversion must map all 39 cameras in order")
    mapping = payload.get("viewpoint_mapping")
    if not isinstance(mapping, Mapping) or mapping.get("39") != TEST_CAMERA:
        raise AssetAuditError("AudioGS viewpoint 39 must map to cam38")
    clips = int(payload.get("num_clips", -1))
    if clips != expected_clips:
        raise AssetAuditError(
            f"AudioGS conversion must contain exactly {expected_clips} clips"
        )
    if epochs != 61:
        raise AssetAuditError("AudioGS native budget must be 61 epochs")
    return {
        "scene": str(payload.get("scene", "")),
        "clips": clips,
        "epochs": epochs,
        "batch_size": 1,
        "training_viewpoints": len(TRAIN_CAMERAS),
        "resolved_updates": clips * len(TRAIN_CAMERAS) * epochs,
    }

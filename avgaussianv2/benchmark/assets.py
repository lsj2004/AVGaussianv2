from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from avgaussianv2.config import load_project_config_bytes

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

PROTOCOL = "dual_dataset_cam38_v1"
TRAIN_CAMERAS = tuple(f"cam{index:02d}" for index in range(38))
ALL_CAMERAS = (*TRAIN_CAMERAS, "cam38")
TEST_CAMERA = "cam38"
MAX_METADATA_BYTES = 2 * 1024 * 1024
EXPECTED = {
    "scene1_opera": {
        "test_samples": 130,
        "audio_updates": 2_318,
        "audio_clips": 1,
        "audio_scene": "SC-scene1-opera-cam38-shared",
    },
    "Scene7playing": {
        "test_samples": 293,
        "audio_updates": 6_954,
        "audio_clips": 3,
        "audio_scene": "SC-scene7-playing-cam38-shared",
    },
}
BENCHMARK_KEYS = {
    "protocol",
    "test_camera",
    "expected_test_samples",
    "seed",
    "native_budgets",
    "continuation_updates",
    "conditioner_warmup_steps",
    "report_steps",
}
NATIVE_BUDGET_KEYS = {
    "audiogs_epochs",
    "audiogs_batch_size",
    "audiogs_resolved_updates",
    "ftgspp_updates",
    "ftgspp_batch_size",
}
PROVENANCE_KEYS = {"scene_id", "test_camera", "assets"}
ASSET_KEYS = {"kind", "camera", "usage", "path"}
IMAGE_KINDS = {"rgb", "image", "video", "frame", "depth", "depth_like"}
ASSET_KINDS = {*IMAGE_KINDS, "camera_pose", "camera_intrinsics"}
INIT_USAGES = {
    "image_driven_initialization",
    "point_initialization",
    "sfm",
    "colmap",
    "temporal_flow",
}
ASSET_USAGES = {*INIT_USAGES, "geometry"}
AUDIO_CONVERSION_KEYS = {
    "scene",
    "audio_root",
    "cameras_npz",
    "output_root",
    "format",
    "sample_rate",
    "clip_sec",
    "hop_sec",
    "num_clips",
    "camera_names",
    "viewpoint_mapping",
    "clips",
}
AUDIO_CLIP_KEYS = {
    "frame_id",
    "start_sample",
    "end_sample",
    "start_seconds",
    "end_seconds",
}
MARKER_NAME = ".cam38-audit.json"


class AssetAuditError(ValueError):
    """Raised when assets do not satisfy the immutable cam38 protocol."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AssetAuditError(
            f"{label} must have exact keys {sorted(expected)}; got {sorted(actual)}"
        )


def _equals_typed(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _equals_typed(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equals_typed(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _read_bounded_regular_nofollow(
    path: str | Path, *, maximum: int = MAX_METADATA_BYTES
) -> bytes:
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise AssetAuditError(f"cannot securely open {candidate}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AssetAuditError(f"{candidate} must be a regular file")
        if metadata.st_size > maximum:
            raise AssetAuditError(f"{candidate} exceeds {maximum} bytes")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise AssetAuditError(f"{candidate} exceeds {maximum} bytes")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_mapping(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    data = _read_bounded_regular_nofollow(path)
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(data.decode("utf-8"))
        else:
            payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise AssetAuditError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AssetAuditError(f"{path}: root must be a mapping")
    return payload


def _require_exact_path(value: Path, expected: Path, label: str) -> None:
    actual_normalized = Path(os.path.abspath(value))
    expected_normalized = Path(os.path.abspath(expected))
    if actual_normalized != expected_normalized:
        raise AssetAuditError(f"{label} must be exactly {expected}")


def audit_protocol_config(path: str | Path) -> dict[str, Any]:
    """Fail closed unless a project config is the exact frozen cam38 protocol."""
    config_path = Path(path)
    data = _read_bounded_regular_nofollow(config_path)
    try:
        project = load_project_config_bytes(data, base_dir=config_path.parent)
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise AssetAuditError(f"cannot parse {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AssetAuditError("configuration root must be a mapping")
    benchmark = raw.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise AssetAuditError("benchmark section is required")
    _exact_keys(benchmark, BENCHMARK_KEYS, "benchmark")
    if project.scene.scene_id not in EXPECTED:
        raise AssetAuditError(f"unsupported benchmark scene {project.scene.scene_id!r}")
    if project.scene.fps != 30.0:
        raise AssetAuditError("scene.fps must be exactly 30.0")
    if project.scene.train_cameras != TRAIN_CAMERAS:
        raise AssetAuditError("training cameras must be exactly cam00 through cam37")
    if project.scene.eval_cameras != (TEST_CAMERA,):
        raise AssetAuditError("evaluation camera must be exactly cam38")
    if project.scene.camera_mapping != {
        camera: index for index, camera in enumerate(ALL_CAMERAS)
    }:
        raise AssetAuditError("scene.camera_mapping must map exactly cam00..cam38")
    if project.train.seed != 42:
        raise AssetAuditError("train.seed must equal benchmark.seed 42")
    if project.train.warmup_steps != 2_000:
        raise AssetAuditError("train.warmup_steps must equal conditioner warmup 2000")
    if project.train.joint_steps != 30_000:
        raise AssetAuditError("train.joint_steps must equal continuation budget 30000")

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
        if not _equals_typed(benchmark.get(key), wanted):
            raise AssetAuditError(f"benchmark.{key} must be {wanted!r}")
    budgets = benchmark.get("native_budgets")
    if not isinstance(budgets, Mapping):
        raise AssetAuditError("benchmark.native_budgets must be a mapping")
    _exact_keys(budgets, NATIVE_BUDGET_KEYS, "benchmark.native_budgets")
    wanted_budgets = {
        "audiogs_epochs": 61,
        "audiogs_batch_size": 1,
        "audiogs_resolved_updates": expected["audio_updates"],
        "ftgspp_updates": 30_000,
        "ftgspp_batch_size": 1,
    }
    if not _equals_typed(dict(budgets), wanted_budgets):
        raise AssetAuditError(f"benchmark.native_budgets must be {wanted_budgets!r}")

    repository = config_path.parent.parent.parent.resolve()
    scene_root = repository / "runs" / "cam38_strict" / project.scene.scene_id
    _require_exact_path(
        project.paths.visual_checkpoint,
        scene_root
        / "ftgspp"
        / "native"
        / project.scene.scene_id
        / "00"
        / "gaussians.pt",
        "paths.visual_checkpoint",
    )
    _require_exact_path(
        project.paths.audio_checkpoint,
        scene_root
        / "audiogs"
        / "native"
        / "replayNVAS"
        / expected["audio_scene"]
        / "viewpoint_39"
        / "checkpoint_latest.pth",
        "paths.audio_checkpoint",
    )
    _require_exact_path(
        project.paths.manifest,
        scene_root / "protocol" / "scene_manifest.json",
        "paths.manifest",
    )
    if project.paths.visual_memmap is None:
        raise AssetAuditError("paths.visual_memmap is required")
    _require_exact_path(
        project.paths.visual_memmap,
        project.paths.visual_upstream_root
        / "_memmap"
        / "cam38_strict"
        / project.scene.scene_id,
        "paths.visual_memmap",
    )
    return dict(raw)


def audit_initialization_provenance(
    value: str | Path | Mapping[str, Any],
    *,
    expected_scene: str,
) -> dict[str, Any]:
    """Reject held-out image targets from every image-driven init input."""
    payload = _load_mapping(value)
    _exact_keys(payload, PROVENANCE_KEYS, "provenance")
    if payload.get("scene_id") != expected_scene:
        raise AssetAuditError(f"provenance scene_id must be {expected_scene!r}")
    if payload.get("test_camera") != TEST_CAMERA:
        raise AssetAuditError("provenance test_camera must be cam38")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise AssetAuditError("provenance assets must be a list")
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise AssetAuditError(f"provenance asset {index} must be a mapping")
        _exact_keys(asset, ASSET_KEYS, f"provenance asset {index}")
        kind = asset["kind"]
        usage = asset["usage"]
        camera = asset["camera"]
        path = asset["path"]
        if not isinstance(kind, str) or kind not in ASSET_KINDS:
            raise AssetAuditError(f"provenance asset {index} has unknown kind")
        if not isinstance(usage, str) or usage not in ASSET_USAGES:
            raise AssetAuditError(f"provenance asset {index} has unknown usage")
        if not isinstance(camera, str) or camera not in {*ALL_CAMERAS, "cam00-cam37"}:
            raise AssetAuditError(f"provenance asset {index} has invalid camera")
        if not isinstance(path, str) or not path:
            raise AssetAuditError(f"provenance asset {index} path must be non-empty")
        if camera == TEST_CAMERA and kind in IMAGE_KINDS and usage in INIT_USAGES:
            raise AssetAuditError(
                f"cam38 RGB/depth target cannot enter image-driven initialization: {path}"
            )
    return dict(payload)


def _secure_regular_metadata(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise AssetAuditError(f"{label} cannot be inspected: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise AssetAuditError(f"{label} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise AssetAuditError(f"{label} must be a regular file")
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise AssetAuditError(f"{label} cannot be opened without following links") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise AssetAuditError(f"{label} changed during audit")
    return metadata


def audit_ftgspp_train_source(
    path: str | Path,
    *,
    allowed_sampled_root: str | Path,
) -> dict[str, Any]:
    """Verify hard-linked train entries are identical to allowed sampled files."""
    source = Path(path)
    if source.is_symlink():
        raise AssetAuditError(f"FTGS++ train source {source} must not be a symlink")
    if not source.is_dir():
        raise AssetAuditError(f"FTGS++ train source {source} must be a directory")
    allowed_path = Path(allowed_sampled_root)
    if allowed_path.is_symlink():
        raise AssetAuditError(
            f"allowed sampled root {allowed_path} must not be a symlink"
        )
    allowed = Path(allowed_sampled_root).resolve(strict=True)
    if not allowed.is_dir():
        raise AssetAuditError(f"allowed sampled root {allowed} must be a directory")
    names = tuple(sorted(item.name for item in source.iterdir()))
    expected_names = tuple(f"{camera}.mp4" for camera in TRAIN_CAMERAS)
    if "cam38.mp4" in names:
        raise AssetAuditError(f"cam38 RGB cannot enter FTGS++ train source {source}")
    if names != expected_names:
        raise AssetAuditError(
            "FTGS++ train source must contain exactly cam00 through cam37"
        )
    for camera in TRAIN_CAMERAS:
        entry = source / f"{camera}.mp4"
        counterpart = allowed / f"{camera}.mp4"
        entry_stat = _secure_regular_metadata(entry, str(entry))
        source_stat = _secure_regular_metadata(counterpart, str(counterpart))
        if (entry_stat.st_dev, entry_stat.st_ino) != (
            source_stat.st_dev,
            source_stat.st_ino,
        ):
            raise AssetAuditError(
                f"{entry} must be a hard link to the matching allowed sampled camera"
            )
    return {
        "path": str(source.resolve()),
        "allowed_sampled_root": str(allowed),
        "cameras": list(TRAIN_CAMERAS),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def audit_audiogs_conversion(
    value: str | Path | Mapping[str, Any],
    *,
    expected_scene: str,
    expected_clips: int,
    epochs: int,
) -> dict[str, Any]:
    """Bind the exact upstream conversion schema and shared-model update budget."""
    payload = _load_mapping(value)
    _exact_keys(payload, AUDIO_CONVERSION_KEYS, "AudioGS conversion")
    expected_spec = next(
        (
            spec
            for spec in EXPECTED.values()
            if spec["audio_scene"] == expected_scene
        ),
        None,
    )
    if expected_spec is None:
        raise AssetAuditError(f"unsupported AudioGS scene {expected_scene!r}")
    if expected_clips != expected_spec["audio_clips"]:
        raise AssetAuditError(
            f"AudioGS expected clips must be {expected_spec['audio_clips']}"
        )
    if payload["scene"] != expected_scene:
        raise AssetAuditError(f"AudioGS conversion scene must be {expected_scene!r}")
    for key in ("audio_root", "cameras_npz", "output_root"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise AssetAuditError(f"AudioGS conversion {key} must be a string")
    if payload["format"] != "AudioGS ReplayNVAS-style viewpoint clips":
        raise AssetAuditError("AudioGS conversion format is invalid")
    if not _is_int(payload["sample_rate"]) or payload["sample_rate"] <= 0:
        raise AssetAuditError("AudioGS conversion sample_rate must be an integer")
    for key in ("clip_sec", "hop_sec"):
        if not isinstance(payload[key], float):
            raise AssetAuditError(f"AudioGS conversion {key} must be a float")
    if not _is_int(payload["num_clips"]):
        raise AssetAuditError("AudioGS conversion num_clips must be an integer")
    if payload["num_clips"] != expected_clips:
        raise AssetAuditError(
            f"AudioGS conversion must contain exactly {expected_clips} clips"
        )
    cameras = payload["camera_names"]
    if not isinstance(cameras, list) or tuple(cameras) != ALL_CAMERAS:
        raise AssetAuditError("AudioGS conversion must map all 39 cameras in order")
    mapping = payload["viewpoint_mapping"]
    expected_mapping = {
        str(index + 1): camera for index, camera in enumerate(ALL_CAMERAS)
    }
    if not isinstance(mapping, Mapping) or dict(mapping) != expected_mapping:
        raise AssetAuditError(
            "AudioGS viewpoint_mapping must contain exactly viewpoints 1 through 39"
        )
    clips = payload["clips"]
    if not isinstance(clips, list) or len(clips) != expected_clips:
        raise AssetAuditError("AudioGS conversion clips list is inconsistent")
    for index, clip in enumerate(clips):
        if not isinstance(clip, Mapping):
            raise AssetAuditError(f"AudioGS clip {index} must be a mapping")
        _exact_keys(clip, AUDIO_CLIP_KEYS, f"AudioGS clip {index}")
        for key in ("frame_id", "start_sample", "end_sample"):
            if not _is_int(clip[key]):
                raise AssetAuditError(f"AudioGS clip {index}.{key} must be an integer")
        for key in ("start_seconds", "end_seconds"):
            if not isinstance(clip[key], float):
                raise AssetAuditError(f"AudioGS clip {index}.{key} must be a float")
        if clip["frame_id"] != index:
            raise AssetAuditError("AudioGS clip frame_id sequence is invalid")
    if epochs != 61:
        raise AssetAuditError("AudioGS native budget must be 61 epochs")
    return {
        "scene": expected_scene,
        "clips": expected_clips,
        "epochs": epochs,
        "batch_size": 1,
        "training_viewpoints": len(TRAIN_CAMERAS),
        "resolved_updates": expected_clips * len(TRAIN_CAMERAS) * epochs,
    }


def _safe_toml_path(path: Path, label: str) -> str:
    text = str(path.resolve(strict=False))
    if any(character in text for character in ('"', "\n", "\r", "\x00")):
        raise AssetAuditError(f"{label} contains characters unsafe for TOML")
    return text


def render_ftgspp_config(
    template: str | Path,
    output: str | Path,
    *,
    repo_root: str | Path,
    ftgspp_root: str | Path,
    sampled_scene_root: str | Path,
) -> None:
    """Render the committed template atomically without shell interpolation."""
    template_path = Path(template)
    try:
        text = _read_bounded_regular_nofollow(template_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssetAuditError("FTGS++ template must be UTF-8") from exc
    replacements = {
        "@ROOT@": _safe_toml_path(Path(repo_root), "repo_root"),
        "@FTGSPP_ROOT@": _safe_toml_path(Path(ftgspp_root), "ftgspp_root"),
        "@SAMPLED_SCENE_ROOT@": _safe_toml_path(
            Path(sampled_scene_root), "sampled_scene_root"
        ),
    }
    for token, replacement in replacements.items():
        if token not in text:
            raise AssetAuditError(f"FTGS++ template is missing {token}")
        text = text.replace(token, replacement)
    if "@" in text:
        raise AssetAuditError("FTGS++ template contains unresolved tokens")
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AssetAuditError(f"rendered FTGS++ TOML is invalid: {exc}") from exc
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _interval(value: Any, label: str) -> None:
    if value != {"start": 0, "stop": 38}:
        raise AssetAuditError(f"{label} must be start=0, stop=38")


def audit_ftgspp_upstream_config(
    path: str | Path,
    *,
    protocol_config: str | Path,
    repo_root: str | Path,
    ftgspp_root: str | Path,
    sampled_scene_root: str | Path,
) -> dict[str, Any]:
    """Parse and bind the final rendered TOML immediately before launch."""
    raw_protocol = audit_protocol_config(protocol_config)
    scene = raw_protocol["scene"]["id"]
    expected = EXPECTED[scene]
    protocol_path = Path(protocol_config)
    expected_repository = protocol_path.parent.parent.parent.resolve()
    repository = Path(repo_root).resolve()
    if repository != expected_repository:
        raise AssetAuditError(f"repo_root must be exactly {expected_repository}")
    configured_upstream = Path(
        raw_protocol["paths"]["visual_upstream_root"]
    ).resolve()
    upstream = Path(ftgspp_root).resolve()
    if upstream != configured_upstream:
        raise AssetAuditError(f"ftgspp_root must be exactly {configured_upstream}")
    sampled = Path(sampled_scene_root).resolve()
    if sampled.name != scene:
        raise AssetAuditError(f"sampled_scene_root must end in scene {scene}")
    try:
        payload = tomllib.loads(
            _read_bounded_regular_nofollow(path).decode("utf-8")
        )
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AssetAuditError(f"cannot parse rendered FTGS++ config: {exc}") from exc
    for section in ("data", "init", "train"):
        if not isinstance(payload.get(section), Mapping):
            raise AssetAuditError(f"FTGS++ config requires [{section}]")
    data = payload["data"]
    init = payload["init"]
    train = payload["train"]
    exact_paths = {
        "data.video_path": (
            data.get("video_path"),
            repository
            / "runs"
            / "cam38_strict"
            / scene
            / "ftgspp"
            / "train_only_source",
        ),
        "data.calibration_path": (
            data.get("calibration_path"),
            sampled / "poses_bounds.npy",
        ),
        "data.extracted_path": (
            data.get("extracted_path"),
            upstream / "_extracted" / "cam38_strict" / scene,
        ),
        "data.memmap_path": (
            data.get("memmap_path"),
            upstream / "_memmap" / "cam38_strict" / scene,
        ),
        "data.colmap_path": (
            data.get("colmap_path"),
            upstream / "_colmap" / "cam38_strict" / scene,
        ),
        "init.points_path": (
            init.get("points_path"),
            upstream / "_points" / "cam38_strict" / scene,
        ),
        "init.temporal_flow_path": (
            init.get("temporal_flow_path"),
            upstream / "_flow" / "cam38_strict" / scene,
        ),
    }
    for label, (actual, wanted) in exact_paths.items():
        if not isinstance(actual, str):
            raise AssetAuditError(f"{label} must be a path string")
        _require_exact_path(Path(actual), wanted, label)
    extends = payload.get("extends")
    if not isinstance(extends, str):
        raise AssetAuditError("FTGS++ config extends must be a path string")
    _require_exact_path(
        Path(extends),
        upstream / "configs" / "dynerf" / "ftgs" / "coffee_martini.toml",
        "extends",
    )
    _secure_regular_metadata(sampled / "poses_bounds.npy", "data.calibration_path")
    if data.get("frames") != {"start": 0, "stop": expected["test_samples"]}:
        raise AssetAuditError("data.frames does not match the scene sample budget")
    if data.get("eval_cameras") != [38]:
        raise AssetAuditError("data.eval_cameras must be exactly [38]")
    _interval(data.get("train_cameras"), "data.train_cameras")
    _interval(init.get("temporal_flow_cameras"), "init.temporal_flow_cameras")
    if train.get("iterations") != 30_000:
        raise AssetAuditError("train.iterations must be exactly 30000")
    if train.get("batch_size") != 1:
        raise AssetAuditError("train.batch_size must be exactly 1")
    return {
        "scene_id": scene,
        "iterations": 30_000,
        "calibration_path": data["calibration_path"],
        "namespaces": [
            data["extracted_path"],
            data["memmap_path"],
            data["colmap_path"],
            init["points_path"],
            init["temporal_flow_path"],
        ],
    }


def prepare_fresh_ftgspp_namespaces(
    paths: Sequence[str | Path],
    *,
    scene_id: str,
    source_root: str | Path,
) -> None:
    """Refuse stale caches, then place an exact audit marker in each namespace."""
    if scene_id not in EXPECTED:
        raise AssetAuditError(f"unsupported benchmark scene {scene_id!r}")
    marker = {
        "protocol": PROTOCOL,
        "scene_id": scene_id,
        "test_camera": TEST_CAMERA,
        "source_root": str(Path(source_root).resolve(strict=False)),
    }
    for value in paths:
        namespace = Path(value)
        if namespace.is_symlink():
            raise AssetAuditError(f"namespace {namespace} must not be a symlink")
        if namespace.exists() and not namespace.is_dir():
            raise AssetAuditError(f"namespace {namespace} must be a directory")
        namespace.mkdir(parents=True, exist_ok=True)
        entries = list(namespace.iterdir())
        marker_path = namespace / MARKER_NAME
        if entries:
            if entries != [marker_path]:
                raise AssetAuditError(f"namespace {namespace} is stale/non-empty")
            existing = _load_mapping(marker_path)
            if dict(existing) != marker:
                raise AssetAuditError(f"namespace {namespace} audit marker mismatches")
            continue
        temporary = namespace / f".{MARKER_NAME}.tmp"
        descriptor = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = None
                stream.write(json.dumps(marker, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, marker_path)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

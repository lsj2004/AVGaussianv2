from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from avgaussianv2.benchmark.assets import (
    AssetAuditError,
    audit_audiogs_conversion,
    audit_ftgspp_train_source,
    audit_ftgspp_upstream_config,
    audit_initialization_provenance,
    audit_protocol_config,
    prepare_fresh_ftgspp_namespaces,
    render_ftgspp_config,
)
from avgaussianv2.config import load_project_config

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CAMERAS = tuple(f"cam{index:02d}" for index in range(38))


@pytest.mark.parametrize(
    ("name", "scene_id", "samples", "audio_updates"),
    (
        ("scene1_opera.yaml", "scene1_opera", 130, 2_318),
        ("Scene7playing.yaml", "Scene7playing", 293, 6_954),
    ),
)
def test_cam38_configs_freeze_strict_split_and_native_budgets(
    name: str, scene_id: str, samples: int, audio_updates: int
) -> None:
    path = ROOT / "configs" / "benchmark_cam38" / name
    project = load_project_config(path)
    raw = yaml.safe_load(path.read_text())

    assert project.scene.scene_id == scene_id
    assert project.scene.train_cameras == TRAIN_CAMERAS
    assert project.scene.eval_cameras == ("cam38",)
    assert raw["benchmark"] == {
        "protocol": "dual_dataset_cam38_v1",
        "test_camera": "cam38",
        "expected_test_samples": samples,
        "seed": 42,
        "native_budgets": {
            "audiogs_epochs": 61,
            "audiogs_batch_size": 1,
            "audiogs_resolved_updates": audio_updates,
            "ftgspp_updates": 30_000,
            "ftgspp_batch_size": 1,
        },
        "continuation_updates": 30_000,
        "conditioner_warmup_steps": 2_000,
        "report_steps": [5_000, 10_000, 30_000],
    }
    audit_protocol_config(path)

    namespace = f"cam38_strict/{scene_id}"
    assert namespace in str(project.paths.visual_checkpoint)
    assert namespace in str(project.paths.audio_checkpoint)
    assert namespace in str(project.paths.visual_memmap)
    assert all(
        "cam10" not in value.as_posix()
        for value in (
            project.paths.visual_checkpoint,
            project.paths.audio_checkpoint,
            project.paths.visual_memmap,
        )
    )


def test_config_audit_rejects_non_exact_cam38_split(tmp_path: Path) -> None:
    source = ROOT / "configs" / "benchmark_cam38" / "scene1_opera.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["scene"]["train_cameras"].append("cam38")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(AssetAuditError, match="exactly cam00 through cam37"):
        audit_protocol_config(path)

    raw = yaml.safe_load(source.read_text())
    raw["benchmark"]["unexpected"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssetAuditError, match="exact keys"):
        audit_protocol_config(path)

    raw = yaml.safe_load(source.read_text())
    raw["train"]["seed"] = 0
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssetAuditError, match="train.seed"):
        audit_protocol_config(path)


def test_provenance_allows_cam38_geometry_but_rejects_test_rgb_for_init(
    tmp_path: Path,
) -> None:
    allowed = {
        "scene_id": "scene1_opera",
        "test_camera": "cam38",
        "assets": [
            {
                "kind": "rgb",
                "camera": "cam37",
                "usage": "image_driven_initialization",
                "path": "/data/cam37.mp4",
            },
            {
                "kind": "camera_pose",
                "camera": "cam38",
                "usage": "geometry",
                "path": "/data/poses_bounds.npy",
            },
        ],
    }
    audit_initialization_provenance(allowed, expected_scene="scene1_opera")

    allowed["assets"].append(
        {
            "kind": "rgb",
            "camera": "cam38",
            "usage": "image_driven_initialization",
            "path": "/data/cam38.mp4",
        }
    )
    with pytest.raises(AssetAuditError, match="cam38 RGB"):
        audit_initialization_provenance(allowed, expected_scene="scene1_opera")

    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(allowed))
    with pytest.raises(AssetAuditError, match="cam38 RGB"):
        audit_initialization_provenance(path, expected_scene="scene1_opera")

    malformed = {
        "scene_id": "other",
        "test_camera": "cam38",
        "assets": [],
        "unknown": 1,
    }
    with pytest.raises(AssetAuditError, match="exact keys"):
        audit_initialization_provenance(malformed, expected_scene="scene1_opera")

    malformed.pop("unknown")
    with pytest.raises(AssetAuditError, match="scene_id"):
        audit_initialization_provenance(malformed, expected_scene="scene1_opera")


def test_runtime_asset_audits_reject_test_rgb_and_bind_shared_audio_updates(
    tmp_path: Path,
) -> None:
    sampled = tmp_path / "sampled"
    sampled.mkdir()
    visual = tmp_path / "visual"
    visual.mkdir()
    for camera in TRAIN_CAMERAS:
        source = sampled / f"{camera}.mp4"
        source.touch()
        os.link(source, visual / source.name)
    audit_ftgspp_train_source(visual, allowed_sampled_root=sampled)
    visual_link = tmp_path / "visual-link"
    visual_link.symlink_to(visual, target_is_directory=True)
    with pytest.raises(AssetAuditError, match="must not be a symlink"):
        audit_ftgspp_train_source(visual_link, allowed_sampled_root=sampled)
    (visual / "cam38.mp4").touch()
    with pytest.raises(AssetAuditError, match="cam38 RGB"):
        audit_ftgspp_train_source(visual, allowed_sampled_root=sampled)

    (visual / "cam38.mp4").unlink()
    (visual / "cam00.mp4").unlink()
    (visual / "cam00.mp4").symlink_to(sampled / "cam00.mp4")
    with pytest.raises(AssetAuditError, match="symlink"):
        audit_ftgspp_train_source(visual, allowed_sampled_root=sampled)
    (visual / "cam00.mp4").unlink()
    (visual / "cam00.mp4").symlink_to(sampled / "missing.mp4")
    with pytest.raises(AssetAuditError, match="symlink"):
        audit_ftgspp_train_source(visual, allowed_sampled_root=sampled)
    (visual / "cam00.mp4").unlink()
    (visual / "cam00.mp4").mkdir()
    with pytest.raises(AssetAuditError, match="regular file"):
        audit_ftgspp_train_source(visual, allowed_sampled_root=sampled)

    conversion = {
        "scene": "SC-scene7-playing-cam38-shared",
        "audio_root": "/audio",
        "cameras_npz": "/cameras.npz",
        "output_root": "/output",
        "format": "AudioGS ReplayNVAS-style viewpoint clips",
        "sample_rate": 16000,
        "clip_sec": 3.0,
        "hop_sec": 3.0,
        "num_clips": 3,
        "camera_names": [*TRAIN_CAMERAS, "cam38"],
        "viewpoint_mapping": {
            str(index + 1): camera
            for index, camera in enumerate((*TRAIN_CAMERAS, "cam38"))
        },
        "clips": [
            {
                "frame_id": index,
                "start_sample": index * 48000,
                "end_sample": (index + 1) * 48000,
                "start_seconds": float(index * 3),
                "end_seconds": float((index + 1) * 3),
            }
            for index in range(3)
        ],
    }
    result = audit_audiogs_conversion(
        conversion,
        expected_scene="SC-scene7-playing-cam38-shared",
        expected_clips=3,
        epochs=61,
    )
    assert result["resolved_updates"] == 6_954
    conversion["num_clips"] = 1
    with pytest.raises(AssetAuditError, match="clips"):
        audit_audiogs_conversion(
            conversion,
            expected_scene="SC-scene7-playing-cam38-shared",
            expected_clips=3,
            epochs=61,
        )


def test_audio_conversion_rejects_mapping_gaps_unknown_keys_and_wrong_scene() -> None:
    manifest = {
        "scene": "SC-scene1-opera-cam38-shared",
        "audio_root": "/audio",
        "cameras_npz": "/cameras.npz",
        "output_root": "/output",
        "format": "AudioGS ReplayNVAS-style viewpoint clips",
        "sample_rate": 16000,
        "clip_sec": 3.0,
        "hop_sec": 3.0,
        "num_clips": 1,
        "camera_names": [*TRAIN_CAMERAS, "cam38"],
        "viewpoint_mapping": {
            str(index + 1): camera
            for index, camera in enumerate((*TRAIN_CAMERAS, "cam38"))
        },
        "clips": [
            {
                "frame_id": 0,
                "start_sample": 0,
                "end_sample": 48000,
                "start_seconds": 0.0,
                "end_seconds": 3.0,
            }
        ],
    }
    kwargs = {
        "expected_scene": "SC-scene1-opera-cam38-shared",
        "expected_clips": 1,
        "epochs": 61,
    }
    broken = json.loads(json.dumps(manifest))
    broken["viewpoint_mapping"].pop("1")
    with pytest.raises(AssetAuditError, match="exactly viewpoints 1 through 39"):
        audit_audiogs_conversion(broken, **kwargs)
    broken = dict(manifest, unknown=True)
    with pytest.raises(AssetAuditError, match="exact keys"):
        audit_audiogs_conversion(broken, **kwargs)
    with pytest.raises(AssetAuditError, match="scene"):
        audit_audiogs_conversion(
            manifest, **dict(kwargs, expected_scene="wrong")
        )


@pytest.mark.parametrize("scene", ("scene1_opera", "Scene7playing"))
def test_rendered_ftgspp_config_is_parsed_and_audited(
    tmp_path: Path, scene: str
) -> None:
    template = ROOT / "configs" / "upstream" / "ftgspp_cam38" / f"{scene}.toml.in"
    output = tmp_path / f"{scene}.toml"
    ftgspp = Path("/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus")
    sampled = (
        Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf") / scene
    )
    render_ftgspp_config(
        template,
        output,
        repo_root=ROOT,
        ftgspp_root=ftgspp,
        sampled_scene_root=sampled,
    )
    result = audit_ftgspp_upstream_config(
        output,
        protocol_config=ROOT / "configs" / "benchmark_cam38" / f"{scene}.yaml",
        repo_root=ROOT,
        ftgspp_root=ftgspp,
        sampled_scene_root=sampled,
    )
    assert result["scene_id"] == scene
    assert result["iterations"] == 30_000
    assert result["calibration_path"] == str(
        sampled / "poses_bounds.npy"
    )

    text = output.read_text().replace(
        'temporal_flow_cameras = { "start" = 0, "stop" = 38 }',
        'temporal_flow_cameras = { "start" = 0, "stop" = 39 }',
    )
    output.write_text(text)
    with pytest.raises(AssetAuditError, match="temporal_flow_cameras"):
        audit_ftgspp_upstream_config(
            output,
            protocol_config=ROOT / "configs" / "benchmark_cam38" / f"{scene}.yaml",
            repo_root=ROOT,
            ftgspp_root=ftgspp,
            sampled_scene_root=sampled,
        )


def test_fresh_ftgspp_namespaces_reject_stale_content(tmp_path: Path) -> None:
    namespaces = [tmp_path / name for name in ("extracted", "memmap", "run")]
    prepare_fresh_ftgspp_namespaces(
        namespaces, scene_id="scene1_opera", source_root=tmp_path / "sampled"
    )
    for namespace in namespaces:
        marker = json.loads((namespace / ".cam38-audit.json").read_text())
        assert marker["scene_id"] == "scene1_opera"

    (namespaces[0] / "old-cache.bin").touch()
    with pytest.raises(AssetAuditError, match="stale/non-empty"):
        prepare_fresh_ftgspp_namespaces(
            namespaces, scene_id="scene1_opera", source_root=tmp_path / "sampled"
        )


def test_upstream_scripts_are_reproducible_and_do_not_launch_by_default() -> None:
    audio = (ROOT / "scripts" / "train_audiogs_cam38_baselines.sh").read_text()
    visual = (ROOT / "scripts" / "prepare_ftgspp_cam38_baselines.sh").read_text()

    for text in (audio, visual):
        assert "set -euo pipefail" in text
        assert "--execute" in text
        assert "audit_cam38_assets" in text
        assert "cam38_strict" in text

    assert "TEST_VIEWPOINT=39" in audio
    assert "NUM_VIEWPOINTS=39" in audio
    assert "train.max_epoch 61" in audio
    assert "train.batch_size 1" in audio
    assert "unset A3DGS_FRAME_ID" in audio
    assert "SC-scene7-playing-cam38-shared" in audio
    assert "A3DGS_FRAME_ID=" not in audio
    assert "--audiogs-conversion" in audio

    assert "eval_cameras = [38]" in visual
    assert 'train_cameras = { "start" = 0, "stop" = 38 }' in visual
    assert 'temporal_flow_cameras = { "start" = 0, "stop" = 38 }' in visual
    assert "iterations = 30000" in visual
    assert "batch_size = 1" in visual
    assert "--ftgspp-train-source" in visual


def test_readme_documents_cam38_native_budget_and_shared_scene7_model() -> None:
    text = (ROOT / "README.md").read_text()
    for phrase in (
        "Dual-dataset cam38 benchmark",
        "`cam00`–`cam37`",
        "`cam38`",
        "61 epochs",
        "2,318 updates",
        "6,954 updates",
        "single shared viewpoint-39 model",
        "30,000 updates",
        "scripts/train_audiogs_cam38_baselines.sh",
        "scripts/prepare_ftgspp_cam38_baselines.sh",
        "does not launch training unless `--execute` is supplied",
    ):
        assert phrase in text

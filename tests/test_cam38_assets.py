from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from avgaussianv2.benchmark.assets import (
    AssetAuditError,
    audit_audiogs_conversion,
    audit_ftgspp_train_source,
    audit_initialization_provenance,
    audit_protocol_config,
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
    audit_initialization_provenance(allowed)

    allowed["assets"].append(
        {
            "kind": "rgb",
            "camera": "cam38",
            "usage": "image_driven_initialization",
            "path": "/data/cam38.mp4",
        }
    )
    with pytest.raises(AssetAuditError, match="cam38 RGB"):
        audit_initialization_provenance(allowed)

    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(allowed))
    with pytest.raises(AssetAuditError, match="cam38 RGB"):
        audit_initialization_provenance(path)


def test_runtime_asset_audits_reject_test_rgb_and_bind_shared_audio_updates(
    tmp_path: Path,
) -> None:
    visual = tmp_path / "visual"
    visual.mkdir()
    for camera in TRAIN_CAMERAS:
        (visual / f"{camera}.mp4").touch()
    audit_ftgspp_train_source(visual)
    (visual / "cam38.mp4").touch()
    with pytest.raises(AssetAuditError, match="cam38 RGB"):
        audit_ftgspp_train_source(visual)

    conversion = {
        "scene": "SC-scene7-playing-cam38-shared",
        "num_clips": 3,
        "camera_names": [*TRAIN_CAMERAS, "cam38"],
        "viewpoint_mapping": {
            str(index + 1): camera
            for index, camera in enumerate((*TRAIN_CAMERAS, "cam38"))
        },
    }
    result = audit_audiogs_conversion(
        conversion, expected_clips=3, epochs=61
    )
    assert result["resolved_updates"] == 6_954
    conversion["num_clips"] = 1
    with pytest.raises(AssetAuditError, match="clips"):
        audit_audiogs_conversion(conversion, expected_clips=3, epochs=61)


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

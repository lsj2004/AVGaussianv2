from pathlib import Path

import pytest
import torch

from avgaussianv2.config import load_project_config
from avgaussianv2.contracts import RGBDRender


def test_load_project_config_rejects_missing_audio_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    path.write_text("scene:\n  id: scene1_opera\n")

    with pytest.raises(ValueError, match=r"paths\.audio_checkpoint"):
        load_project_config(path)


def test_load_project_config_parses_minimal_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    path.write_text(
        """
scene:
  id: scene1_opera
  fps: 30
  train_cameras: [cam00, cam01]
  eval_cameras: [cam10]
  camera_mapping: {cam00: 0, cam01: 1, cam10: 10}
paths:
  visual_upstream_root: /repos/FreeTimeGSPlusPlus
  audio_upstream_root: /repos/audioGS-replay
  visual_checkpoint: /runs/visual.pt
  audio_checkpoint: /runs/audio.pth
  manifest: /runs/manifest.json
model:
  embedding_dim: 64
  audio_model_class: Audio3DGS
  audio_render_strategy: direct_conditioned_unet
train:
  crop_seconds: 0.5
""".strip()
        + "\n"
    )

    config = load_project_config(path)

    assert config.scene.scene_id == "scene1_opera"
    assert config.scene.train_cameras == ("cam00", "cam01")
    assert config.paths.audio_checkpoint == Path("/runs/audio.pth")
    assert config.model.embedding_dim == 64
    assert config.model.audio_render_strategy == "direct_conditioned_unet"
    assert config.train.crop_seconds == pytest.approx(0.5)


def test_load_project_config_rejects_unknown_audio_render_strategy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene.yaml"
    path.write_text(
        """
scene:
  id: scene1_opera
  fps: 30
  train_cameras: [cam00]
  eval_cameras: [cam10]
  camera_mapping: {cam00: 0, cam10: 10}
paths:
  visual_upstream_root: /repos/FreeTimeGSPlusPlus
  audio_upstream_root: /repos/audioGS-replay
  visual_checkpoint: /runs/visual.pt
  audio_checkpoint: /runs/audio.pth
  manifest: /runs/manifest.json
model:
  audio_render_strategy: replace_everything
train: {}
""".strip()
        + "\n"
    )

    with pytest.raises(ValueError, match="model.audio_render_strategy"):
        load_project_config(path)


def test_load_project_config_resolves_relative_paths_from_config_directory(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs" / "scene1"
    config_dir.mkdir(parents=True)
    path = config_dir / "scene.yaml"
    path.write_text(
        """
scene:
  id: scene1_opera
  fps: 30
  train_cameras: [cam00]
  eval_cameras: [cam10]
  camera_mapping: {cam00: 0, cam10: 10}
paths:
  visual_upstream_root: upstream/visual
  audio_upstream_root: upstream/audio
  visual_checkpoint: checkpoints/visual.pt
  audio_checkpoint: checkpoints/audio.pt
  manifest: manifests/scene.json
  visual_memmap: cache/visual.dat
model: {}
train: {}
""".strip()
        + "\n"
    )

    config = load_project_config(path)

    assert config.paths.visual_upstream_root == config_dir / "upstream/visual"
    assert config.paths.audio_upstream_root == config_dir / "upstream/audio"
    assert config.paths.visual_checkpoint == config_dir / "checkpoints/visual.pt"
    assert config.paths.audio_checkpoint == config_dir / "checkpoints/audio.pt"
    assert config.paths.manifest == config_dir / "manifests/scene.json"
    assert config.paths.visual_memmap == config_dir / "cache/visual.dat"


def test_load_project_config_rejects_nonpositive_crop(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    path.write_text(
        """
scene:
  id: scene1_opera
  fps: 30
  train_cameras: [cam00]
  eval_cameras: [cam10]
  camera_mapping: {cam00: 0, cam10: 10}
paths:
  visual_upstream_root: /repos/FreeTimeGSPlusPlus
  audio_upstream_root: /repos/audioGS-replay
  visual_checkpoint: /runs/visual.pt
  audio_checkpoint: /runs/audio.pth
  manifest: /runs/manifest.json
model: {}
train:
  crop_seconds: 0
""".strip()
        + "\n"
    )

    with pytest.raises(ValueError, match="train.crop_seconds must be positive"):
        load_project_config(path)


def test_rgbd_render_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shared batch and image shape"):
        RGBDRender(
            rgb=torch.zeros(1, 8, 8, 3),
            depth=torch.zeros(1, 4, 4, 1),
            alpha=torch.zeros(1, 8, 8, 1),
        )


def test_rgbd_render_accepts_matching_channels() -> None:
    render = RGBDRender(
        rgb=torch.zeros(2, 8, 12, 3),
        depth=torch.ones(2, 8, 12, 1),
        alpha=torch.ones(2, 8, 12, 1),
    )

    assert render.image_size == (8, 12)
    assert render.batch_size == 2

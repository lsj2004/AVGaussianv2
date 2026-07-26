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
  audio_backend: cross_attention_tokens
  embedding_dim: 64
  audio_model_class: Audio3DGS
  audio_freq_patch: 8
  audio_time_patch: 2
  audio_transformer_layers: 2
  audio_transformer_heads: 4
  audio_pose_tokens: 1
  audio_loss_l1_weight: 0.9
  audio_loss_mse_weight: 0.2
  audio_loss_ild_weight: 0.3
  audio_loss_ipd_weight: 0.4
  audio_loss_lre_weight: 0.5
train:
  crop_seconds: 0.5
""".strip()
        + "\n"
    )

    config = load_project_config(path)

    assert config.scene.scene_id == "scene1_opera"
    assert config.scene.train_cameras == ("cam00", "cam01")
    assert config.paths.audio_checkpoint == Path("/runs/audio.pth")
    assert config.model.audio_backend == "cross_attention_tokens"
    assert config.model.embedding_dim == 64
    assert config.model.audio_freq_patch == 8
    assert config.model.audio_time_patch == 2
    assert config.model.audio_transformer_layers == 2
    assert config.model.audio_transformer_heads == 4
    assert config.model.audio_pose_tokens == 1
    assert config.model.audio_loss_l1_weight == pytest.approx(0.9)
    assert config.model.audio_loss_mse_weight == pytest.approx(0.2)
    assert config.model.audio_loss_ild_weight == pytest.approx(0.3)
    assert config.model.audio_loss_ipd_weight == pytest.approx(0.4)
    assert config.model.audio_loss_lre_weight == pytest.approx(0.5)
    assert config.train.crop_seconds == pytest.approx(0.5)


def test_load_project_config_rejects_unknown_audio_backend(tmp_path: Path) -> None:
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
  audio_backend: nope
train:
  crop_seconds: 0.5
""".strip()
        + "\n"
    )

    with pytest.raises(ValueError, match="model.audio_backend"):
        load_project_config(path)


def test_load_project_config_rejects_negative_spatial_loss_weight(tmp_path: Path) -> None:
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
  audio_backend: cross_attention_tokens
  audio_loss_ipd_weight: -0.1
train:
  crop_seconds: 0.5
""".strip()
        + "\n"
    )

    with pytest.raises(ValueError, match="audio_loss_ipd_weight"):
        load_project_config(path)


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

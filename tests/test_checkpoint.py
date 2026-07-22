from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from avgaussianv2.checkpoint import (
    CheckpointCompatibilityError,
    build_checkpoint_state,
    load_checkpoint,
    save_checkpoint,
)
from avgaussianv2.config import ModelConfig, PathConfig, ProjectConfig, SceneConfig, TrainConfig


class TinyCheckpointModel(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.visual = nn.Linear(width, width)
        self.condition_encoder = nn.Linear(width, width)
        self.audio = nn.Linear(width, width)

    def forward(self, value):
        return self.audio(self.condition_encoder(self.visual(value)))


def config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        scene=SceneConfig(
            scene_id="scene1_opera",
            fps=30.0,
            train_cameras=("cam00",),
            eval_cameras=("cam10",),
            camera_mapping={"cam00": 0, "cam10": 10},
        ),
        paths=PathConfig(
            visual_upstream_root=tmp_path / "ftgs",
            audio_upstream_root=tmp_path / "audiogs",
            visual_checkpoint=tmp_path / "visual.pt",
            audio_checkpoint=tmp_path / "audio.pth",
            manifest=tmp_path / "manifest.json",
        ),
        model=ModelConfig(embedding_dim=4),
        train=TrainConfig(crop_seconds=0.5),
    )


def test_checkpoint_roundtrip_reproduces_output(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = TinyCheckpointModel()
    value = torch.randn(2, 4)
    expected = model(value).detach().clone()
    path = tmp_path / "joint.pt"
    state = build_checkpoint_state(
        model,
        optimizer=None,
        config=config(tmp_path),
        provenance={"visual_commit": "abc", "audio_commit": "def"},
        stage="joint",
        step=4,
        loss_history=[{"total": 1.25}],
    )
    save_checkpoint(path, state)
    restored = TinyCheckpointModel()

    resume = load_checkpoint(path, restored, config(tmp_path))

    torch.testing.assert_close(restored(value), expected)
    assert resume.step == 4
    assert resume.stage == "joint"
    assert resume.loss_history == [{"total": 1.25}]


def test_checkpoint_rejects_stft_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "joint.pt"
    original = config(tmp_path)
    save_checkpoint(
        path,
        build_checkpoint_state(
            TinyCheckpointModel(),
            optimizer=None,
            config=original,
            provenance={},
            stage="warmup",
            step=1,
            loss_history=[],
        ),
    )
    mismatched = replace(original, model=replace(original.model, n_fft=1024))

    with pytest.raises(CheckpointCompatibilityError, match="n_fft"):
        load_checkpoint(path, TinyCheckpointModel(), mismatched)


def test_checkpoint_rejects_camera_mapping_change(tmp_path: Path) -> None:
    path = tmp_path / "joint.pt"
    original = config(tmp_path)
    save_checkpoint(
        path,
        build_checkpoint_state(
            TinyCheckpointModel(), None, original, {}, "joint", 2, []
        ),
    )
    changed_scene = replace(original.scene, camera_mapping={"cam00": 1, "cam10": 10})

    with pytest.raises(CheckpointCompatibilityError, match="camera_mapping"):
        load_checkpoint(path, TinyCheckpointModel(), replace(original, scene=changed_scene))


def test_checkpoint_rejects_missing_required_state(tmp_path: Path) -> None:
    path = tmp_path / "broken.pt"
    torch.save({"schema_version": 1}, path)

    with pytest.raises(CheckpointCompatibilityError, match="visual_state_dict"):
        load_checkpoint(path, TinyCheckpointModel(), config(tmp_path))

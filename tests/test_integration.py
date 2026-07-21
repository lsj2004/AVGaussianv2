import json
import math

import pytest
import soundfile as sf
import torch
from torch import nn

from avgaussianv2.cli.train import TrainingBundle, run_training
from avgaussianv2.config import ModelConfig, PathConfig, ProjectConfig, SceneConfig, TrainConfig
from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender


class ScalarModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))


class ScalarAudio(ScalarModule):
    def render(self, cam_pose, source_audio, condition=None):
        del cam_pose
        condition_gain = 0.0 if condition is None else condition.mean().reshape(1, 1, 1)
        return source_audio * (self.value + condition_gain)


class TinyFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = ScalarModule(0.2)
        self.audio = ScalarAudio(0.1)
        self.condition_encoder = ScalarModule(0.3)
        self.film = ScalarModule(0.4)
        self.audio_unet = ScalarModule(0.5)
        self.condition_enabled = True

    def forward(self, sample):
        value = self.visual.value
        rgb = value.sigmoid().expand(1, 8, 8, 3)
        depth = (value + 2.0).expand(1, 8, 8, 1)
        condition = value * self.condition_encoder.value
        if not self.condition_enabled:
            condition = condition * 0.0
        gain = self.audio.value + self.audio_unet.value + condition * self.film.value
        return FusionOutput(
            rgbd=RGBDRender(rgb, depth, torch.ones_like(depth)),
            condition=condition.reshape(1, 1),
            predicted_audio=sample.source_audio * gain,
        )

    def named_parameter_groups(self):
        return {
            "visual": list(self.visual.parameters()),
            "acoustic": list(self.audio.parameters()),
            "condition_encoder": list(self.condition_encoder.parameters()),
            "film": list(self.film.parameters()),
            "audio_unet": list(self.audio_unet.parameters()),
        }

    def freeze_pretrained(self):
        for group in (self.visual, self.audio, self.audio_unet):
            group.requires_grad_(False)
        for group in (self.condition_encoder, self.film):
            group.requires_grad_(True)

    def unfreeze_all(self):
        self.requires_grad_(True)


def sample() -> AlignedAVSample:
    return AlignedAVSample(
        scene_id="fixture",
        camera="cam00",
        frame_index=4,
        time_seconds=0.2,
        visual_time=torch.tensor([[0.1]]),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 12),
        source_audio=torch.full((1, 2, 32), 0.25),
        target_audio=torch.full((1, 2, 32), 0.5),
        target_rgb=torch.full((1, 8, 8, 3), 0.4),
        image_size=(8, 8),
    )


def fake_config(tmp_path) -> ProjectConfig:
    return ProjectConfig(
        scene=SceneConfig("fixture", 20.0, ("cam00",), ("cam01",), {"cam00": 0, "cam01": 1}),
        paths=PathConfig(tmp_path, tmp_path, tmp_path / "visual.pt", tmp_path / "audio.pt", tmp_path / "manifest.json"),
        model=ModelConfig(embedding_dim=8),
        train=TrainConfig(warmup_steps=2, joint_steps=2),
    )


def fake_backend_factory(config, device):
    del config
    model = TinyFusion().to(device)

    def audio_loss(predicted, target):
        return {"total_loss": torch.nn.functional.mse_loss(predicted, target)}

    return TrainingBundle(model=model, samples=[sample()], audio_loss_fn=audio_loss)


def test_cpu_warmup_then_joint_smoke_writes_reproducible_artifacts(tmp_path) -> None:
    result = run_training(
        fake_config(tmp_path),
        output_dir=tmp_path / "run",
        warmup_steps=2,
        joint_steps=2,
        backend_factory=fake_backend_factory,
        device="cpu",
    )

    assert result.completed_stage == "joint"
    assert len(result.history) == 4
    assert all(math.isfinite(row["total"]) for row in result.history)
    output = tmp_path / "run"
    for relative in (
        "checkpoint_latest.pt",
        "resolved_config.json",
        "loss_history.json",
        "gradient_norms.json",
        "selected_sample.json",
        "run_summary.json",
        "artifacts/sample_pred.wav",
        "artifacts/sample_condition_on.wav",
        "artifacts/sample_condition_off.wav",
        "artifacts/metrics.json",
        "artifacts/sample_rgb.ppm",
        "artifacts/sample_depth.pgm",
    ):
        assert (output / relative).is_file(), relative
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["completed_stage"] == "joint"
    assert summary["steps"] == 4
    assert summary["artifact_metrics"]["condition_delta_mean_abs"] > 0


def test_stage_warmup_does_not_run_joint(tmp_path) -> None:
    result = run_training(
        fake_config(tmp_path),
        stage="warmup",
        output_dir=tmp_path / "warmup",
        warmup_steps=1,
        joint_steps=9,
        backend_factory=fake_backend_factory,
        device="cpu",
    )

    assert result.completed_stage == "warmup"
    assert len(result.history) == 1


def test_condition_off_ablation_rejects_warmup(tmp_path) -> None:
    with pytest.raises(ValueError, match="joint-only"):
        run_training(
            fake_config(tmp_path),
            stage="all",
            condition_off=True,
            output_dir=tmp_path / "ablation",
            backend_factory=fake_backend_factory,
            device="cpu",
        )


def test_condition_off_joint_run_still_writes_a_true_paired_ablation(tmp_path) -> None:
    result = run_training(
        fake_config(tmp_path),
        stage="joint",
        condition_off=True,
        joint_steps=1,
        output_dir=tmp_path / "ablation",
        backend_factory=fake_backend_factory,
        device="cpu",
    )

    assert result.artifact_metrics["condition_delta_mean_abs"] > 0
    on, _ = sf.read(tmp_path / "ablation/artifacts/sample_condition_on.wav")
    off, _ = sf.read(tmp_path / "ablation/artifacts/sample_condition_off.wav")
    assert not torch.equal(torch.from_numpy(on), torch.from_numpy(off))

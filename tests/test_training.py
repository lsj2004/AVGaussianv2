import pytest
import torch
from torch import nn

from avgaussianv2.config import TrainConfig
from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender
from avgaussianv2.losses import JointLossWeights, capture_visual_anchor, compute_joint_loss
from avgaussianv2.train import (
    NonFiniteTrainingError,
    joint_train_step,
    run_condition_warmup,
)


def make_sample() -> AlignedAVSample:
    return AlignedAVSample(
        scene_id="scene1_opera",
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


class ScalarModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))


class TinyTrainFusion(nn.Module):
    def __init__(self, *, emit_nan: bool = False) -> None:
        super().__init__()
        self.visual = ScalarModule(0.2)
        self.acoustic = ScalarModule(0.1)
        self.condition_encoder = ScalarModule(0.3)
        self.film = ScalarModule(0.4)
        self.audio_unet = ScalarModule(0.5)
        self.emit_nan = emit_nan

    def forward(self, sample):
        visual_value = self.visual.value
        rgb = visual_value.sigmoid().expand(1, 8, 8, 3)
        depth = (visual_value + 2.0).expand(1, 8, 8, 1)
        condition = visual_value * self.condition_encoder.value
        audio_gain = (
            self.acoustic.value
            + self.audio_unet.value
            + condition * self.film.value
        )
        predicted = sample.source_audio * audio_gain
        if self.emit_nan:
            predicted = predicted * torch.tensor(float("nan"))
        return FusionOutput(
            rgbd=RGBDRender(rgb, depth, torch.ones_like(depth)),
            condition=condition.reshape(1, 1),
            predicted_audio=predicted,
        )

    def named_parameter_groups(self):
        return {
            "visual": list(self.visual.parameters()),
            "acoustic": list(self.acoustic.parameters()),
            "condition_encoder": list(self.condition_encoder.parameters()),
            "film": list(self.film.parameters()),
            "audio_unet": list(self.audio_unet.parameters()),
        }

    def freeze_pretrained(self):
        for group in (self.visual, self.acoustic, self.audio_unet):
            for parameter in group.parameters():
                parameter.requires_grad_(False)
        for group in (self.condition_encoder, self.film):
            for parameter in group.parameters():
                parameter.requires_grad_(True)

    def unfreeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad_(True)


def audio_loss(predicted, target):
    return {"total_loss": torch.nn.functional.mse_loss(predicted, target)}


def test_joint_loss_matches_weighted_components() -> None:
    model = TinyTrainFusion()
    sample = make_sample()
    output = model(sample)
    weights = JointLossWeights(audio=2.0, rgb=3.0, dssim=0.25, visual_anchor=4.0)
    anchor = capture_visual_anchor(model.visual)

    total, parts = compute_joint_loss(
        output,
        sample,
        visual_module=model.visual,
        visual_anchor=anchor,
        weights=weights,
        audio_loss_fn=audio_loss,
    )

    expected = 2.0 * parts["audio"] + 3.0 * parts["rgb"] + 4.0 * parts["visual_anchor"]
    torch.testing.assert_close(total, expected)
    assert parts["visual_anchor"] == 0


def test_joint_step_reports_all_gradient_groups() -> None:
    model = TinyTrainFusion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = TrainConfig(crop_seconds=0.5)

    stats = joint_train_step(
        model,
        make_sample(),
        optimizer,
        config,
        audio_loss_fn=audio_loss,
        visual_anchor=capture_visual_anchor(model.visual),
    )

    assert stats.gradient_norms.keys() >= {
        "visual",
        "acoustic",
        "condition_encoder",
        "film",
        "audio_unet",
    }
    assert all(value > 0 for value in stats.gradient_norms.values())
    assert stats.audio_to_visual_grad_norm > 0


def test_warmup_changes_only_condition_and_film_parameters() -> None:
    model = TinyTrainFusion()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    run_condition_warmup(
        model,
        [make_sample()],
        steps=2,
        learning_rate=0.05,
        config=TrainConfig(crop_seconds=0.5),
        audio_loss_fn=audio_loss,
    )

    after = dict(model.named_parameters())
    assert torch.equal(after["visual.value"], before["visual.value"])
    assert torch.equal(after["acoustic.value"], before["acoustic.value"])
    assert torch.equal(after["audio_unet.value"], before["audio_unet.value"])
    assert not torch.equal(after["condition_encoder.value"], before["condition_encoder.value"])
    assert not torch.equal(after["film.value"], before["film.value"])


def test_nonfinite_training_error_contains_sample_identity() -> None:
    model = TinyTrainFusion(emit_nan=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(
        NonFiniteTrainingError,
        match=r"scene1_opera.*cam00.*frame=4",
    ):
        joint_train_step(
            model,
            make_sample(),
            optimizer,
            TrainConfig(crop_seconds=0.5),
            audio_loss_fn=audio_loss,
            visual_anchor=capture_visual_anchor(model.visual),
        )

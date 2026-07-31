import pytest
import torch
from torch import nn

from avgaussianv2.config import TrainConfig
from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender
from avgaussianv2.losses import (
    JointLossWeights,
    capture_visual_anchor,
    compute_audio_objective,
    compute_joint_loss,
    signed_lre_db,
    signed_lre_loss,
)
from avgaussianv2.train import (
    NonFiniteTrainingError,
    build_joint_optimizer,
    build_warmup_optimizer,
    condition_warmup_step,
    joint_train_step,
    run_condition_warmup,
    run_joint_finetune,
    same_frame_camera_negative_indices,
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
        return self.forward_with_condition_sample(sample, sample)

    def forward_with_condition_sample(self, sample, condition_sample):
        visual_value = self.visual.value
        rgb = visual_value.sigmoid().expand(1, 8, 8, 3)
        depth = (visual_value + 2.0).expand(1, 8, 8, 1)
        frame_scale = 1.0 + 0.01 * float(condition_sample.frame_index)
        condition = visual_value * self.condition_encoder.value * frame_scale
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


def test_signed_lre_loss_breaks_diff_magnitude_sign_ambiguity() -> None:
    target = torch.stack(
        (torch.full((64,), 2.0), torch.ones(64)),
        dim=0,
    ).unsqueeze(0)
    swapped = target.flip(1).clone().requires_grad_(True)

    target_diff_magnitude = torch.fft.rfft(
        target[:, 0] - target[:, 1]
    ).abs()
    swapped_diff_magnitude = torch.fft.rfft(
        swapped[:, 0] - swapped[:, 1]
    ).abs()
    torch.testing.assert_close(swapped_diff_magnitude, target_diff_magnitude)

    loss = signed_lre_loss(swapped, target, scale_db=6.0)
    assert loss.item() > 0
    loss.backward()
    assert swapped.grad is not None
    assert swapped.grad[:, 0].abs().sum() > 0
    assert swapped.grad[:, 1].abs().sum() > 0


def test_signed_lre_db_preserves_left_right_direction() -> None:
    target = torch.stack(
        (torch.full((32,), 2.0), torch.ones(32)),
        dim=0,
    ).unsqueeze(0)

    expected = 10.0 * torch.log10(torch.tensor(4.0))
    torch.testing.assert_close(signed_lre_db(target), expected.reshape(1))
    torch.testing.assert_close(signed_lre_db(target.flip(1)), -expected.reshape(1))


def test_audio_objective_keeps_base_metric_separate_from_lre_regularizer() -> None:
    target = torch.stack(
        (torch.full((32,), 2.0), torch.ones(32)),
        dim=0,
    ).unsqueeze(0)
    predicted = target.flip(1)
    weights = JointLossWeights(audio=1.0, lre=0.02)

    total, parts = compute_audio_objective(
        predicted,
        target,
        weights=weights,
        audio_loss_fn=audio_loss,
    )

    torch.testing.assert_close(parts["audio"], parts["audio_base"])
    torch.testing.assert_close(
        total,
        parts["audio_base"] + parts["audio_lre_weighted"],
    )
    assert parts["audio_lre"] > 0
    assert parts["pred_lre_db"] < 0
    assert parts["target_lre_db"] > 0


@pytest.mark.parametrize(
    ("criterion", "error", "message"),
    [
        (lambda *_: torch.ones(2), ValueError, "scalar Tensor"),
        (lambda *_: torch.tensor(float("nan")), ValueError, "finite"),
        (lambda *_: {"wrong": torch.tensor(1.0)}, ValueError, "total_loss"),
        (lambda *_: 1.0, TypeError, "resolve to a Tensor"),
    ],
)
def test_audio_objective_rejects_invalid_base_criterion(
    criterion,
    error,
    message,
) -> None:
    audio = torch.ones(1, 2, 16)

    with pytest.raises(error, match=message):
        compute_audio_objective(
            audio,
            audio,
            weights=JointLossWeights(),
            audio_loss_fn=criterion,
        )


def test_same_frame_camera_negatives_are_reproducible_and_cross_camera() -> None:
    samples = []
    for frame in (1, 2):
        for camera in ("cam00", "cam01", "cam02"):
            value = make_sample()
            samples.append(
                type(value)(
                    **{
                        **vars(value),
                        "frame_index": frame,
                        "camera": camera,
                    }
                )
            )
    anchors = [0, 1, 3, 4]

    first = same_frame_camera_negative_indices(samples, anchors, seed=42)
    second = same_frame_camera_negative_indices(samples, anchors, seed=42)

    assert first == second
    for anchor, negative in zip(anchors, first):
        assert samples[anchor].frame_index == samples[negative].frame_index
        assert samples[anchor].camera != samples[negative].camera


def test_joint_step_applies_p1_camera_contrast() -> None:
    model = TinyTrainFusion()
    model.camera_contrast_weight = 0.5
    model.camera_contrast_margin = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    correct = make_sample()
    wrong = type(correct)(
        **{
            **vars(correct),
            "camera": "cam01",
            "frame_index": correct.frame_index,
        }
    )

    stats = joint_train_step(
        model,
        correct,
        optimizer,
        TrainConfig(),
        audio_loss,
        capture_visual_anchor(model.visual),
        contrast_sample=wrong,
    )

    assert stats.losses["camera_contrast"] >= 0
    assert stats.losses["camera_contrast_weighted"] == pytest.approx(
        0.5 * stats.losses["camera_contrast"]
    )


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

    expected = (
        2.0 * parts["audio_base"]
        + 3.0 * parts["rgb"]
        + 4.0 * parts["visual_anchor"]
    )
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


def test_reusable_optimizers_only_include_trainable_nonempty_groups() -> None:
    model = TinyTrainFusion()
    model.freeze_pretrained()

    warmup = build_warmup_optimizer(model, 0.01)
    assert len(warmup.param_groups) == 1
    assert {id(parameter) for parameter in warmup.param_groups[0]["params"]} == {
        id(model.condition_encoder.value),
        id(model.film.value),
    }

    model.unfreeze_all()
    model.visual.requires_grad_(False)
    joint = build_joint_optimizer(model, TrainConfig())
    assert len(joint.param_groups) == 4
    assert all(group["params"] for group in joint.param_groups)


def test_condition_warmup_step_accepts_scalar_criterion() -> None:
    model = TinyTrainFusion()
    model.freeze_pretrained()
    optimizer = build_warmup_optimizer(model, 0.01)

    stats = condition_warmup_step(
        model,
        make_sample(),
        optimizer,
        TrainConfig(),
        lambda predicted, target: torch.nn.functional.mse_loss(predicted, target),
    )

    assert stats.total > 0
    assert stats.losses["audio"] == stats.losses["audio_base"]
    assert stats.losses["audio_lre"] == 0
    assert stats.losses["audio_total_with_lre"] == stats.total
    assert stats.gradient_norms["condition_encoder"] > 0
    assert stats.audio_to_visual_grad_norm == 0


def test_joint_step_can_skip_audio_visual_gradient_probe(monkeypatch) -> None:
    model = TinyTrainFusion()
    model.visual.requires_grad_(False)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("autograd.grad must not be called")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    stats = joint_train_step(
        model,
        make_sample(),
        optimizer,
        TrainConfig(),
        audio_loss,
        capture_visual_anchor(model.visual),
        probe_audio_visual_gradient=False,
    )
    assert stats.audio_to_visual_grad_norm == 0
    assert not model.visual.value.requires_grad


def test_legacy_joint_finetune_skips_probe_when_not_required(monkeypatch) -> None:
    model = TinyTrainFusion()

    def forbidden(*args, **kwargs):
        raise AssertionError("autograd.grad must not be called")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    history = run_joint_finetune(
        model,
        [make_sample()],
        steps=1,
        config=TrainConfig(),
        audio_loss_fn=audio_loss,
        require_audio_visual_gradient=False,
    )

    assert len(history) == 1
    assert history[0].audio_to_visual_grad_norm == 0

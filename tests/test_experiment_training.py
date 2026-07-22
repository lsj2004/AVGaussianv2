from __future__ import annotations

import csv
import json

import pytest
import torch
from torch import nn

from avgaussianv2.config import TrainConfig
from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender
from avgaussianv2.experiment.contracts import (
    EvaluationResult,
    PilotConfig,
    Variant,
    VariantIndices,
)
from avgaussianv2.experiment.training import PilotTrainer, configure_variant
from avgaussianv2.train import DisconnectedAudioVisualGradient, TrainStepStats


class ScalarModule(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))


class TinyTrainFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = ScalarModule(0.2)
        self.acoustic = ScalarModule(0.1)
        self.condition_encoder = ScalarModule(0.3)
        self.film = ScalarModule(0.4)
        self.audio_unet = ScalarModule(0.5)

    def forward(self, sample):
        value = self.visual.value
        rgb = value.sigmoid().expand(1, 8, 8, 3)
        depth = (value + 2).expand(1, 8, 8, 1)
        condition = value * self.condition_encoder.value
        gain = self.acoustic.value + self.audio_unet.value + condition * self.film.value
        return FusionOutput(
            RGBDRender(rgb, depth, torch.ones_like(depth)),
            condition.reshape(1, 1),
            sample.source_audio * gain,
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
        self.visual.requires_grad_(False)
        self.acoustic.requires_grad_(False)
        self.audio_unet.requires_grad_(False)
        self.condition_encoder.requires_grad_(True)
        self.film.requires_grad_(True)

    def unfreeze_all(self):
        self.requires_grad_(True)


def make_sample() -> AlignedAVSample:
    return AlignedAVSample(
        scene_id="scene1_opera", camera="cam00", frame_index=4, time_seconds=0.2,
        visual_time=torch.tensor([[0.1]]), w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0), audio_cam_pose=torch.zeros(1, 12),
        source_audio=torch.full((1, 2, 32), 0.25),
        target_audio=torch.full((1, 2, 32), 0.5),
        target_rgb=torch.full((1, 8, 8, 3), 0.4), image_size=(8, 8),
    )


def audio_loss(predicted, target):
    return {"total_loss": torch.nn.functional.mse_loss(predicted, target)}


def _flags(model):
    return {
        name: tuple(parameter.requires_grad for parameter in parameters)
        for name, parameters in model.named_parameter_groups().items()
    }


@pytest.mark.parametrize(
    ("variant", "enabled"),
    [
        (Variant.JOINT_CONDITIONED, {"visual", "acoustic", "audio_unet", "condition_encoder", "film"}),
        (Variant.FROZEN_VISUAL, {"acoustic", "audio_unet", "condition_encoder", "film"}),
        (Variant.CONDITION_OFF, {"acoustic", "audio_unet"}),
    ],
)
def test_configure_variant_sets_exact_joint_policy(variant, enabled) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = True

    configure_variant(model, variant, "joint")

    flags = _flags(model)
    assert {name for name, values in flags.items() if all(values)} == enabled
    assert model.condition_enabled is (variant != Variant.CONDITION_OFF)


@pytest.mark.parametrize("variant", [Variant.JOINT_CONDITIONED, Variant.FROZEN_VISUAL])
def test_configure_variant_warmup_only_trains_condition_path(variant) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = False
    configure_variant(model, variant, "warmup")
    flags = _flags(model)
    assert {name for name, values in flags.items() if all(values)} == {
        "condition_encoder",
        "film",
    }
    assert model.condition_enabled


def test_condition_off_rejects_warmup_policy() -> None:
    model = TinyTrainFusion()
    model.condition_enabled = True
    with pytest.raises(ValueError, match="warmup"):
        configure_variant(model, Variant.CONDITION_OFF, "warmup")


class FakeEvaluator:
    def __init__(self, audios=(1.0, 0.9)) -> None:
        self.audios = iter(audios)
        self.calls = []

    def evaluate(self, samples, indices, system_name, condition_enabled, output_dir):
        self.calls.append((samples, tuple(indices), system_name, condition_enabled, output_dir))
        audio = next(self.audios)
        summary = {
            "audio_total": {"mean": audio},
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        }
        return EvaluationResult(system_name, len(indices), (), summary)


def _stats(probe: float = 0.0) -> TrainStepStats:
    return TrainStepStats(
        total=1.0,
        losses={"audio": 1.0},
        gradient_norms={"visual": probe, "acoustic": 1.0},
        audio_to_visual_grad_norm=probe,
    )


def _sample_with_frame(frame: int):
    sample = make_sample()
    return type(sample)(**{**vars(sample), "frame_index": frame})


def _baseline():
    return {"rgb_psnr": {"mean": 30.0}, "rgb_ssim": {"mean": 0.95}}


def test_pilot_uses_persistent_optimizers_exact_order_and_final_validation(tmp_path) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = True
    train_samples = [_sample_with_frame(index) for index in range(5)]
    heldout = [_sample_with_frame(100 + index) for index in range(3)]
    seen = []
    optimizer_builds = []

    def optimizer_factory(model, *args):
        optimizer_builds.append(args)
        return torch.optim.SGD([next(model.parameters())], lr=0.01)

    def warmup_step(model, sample, optimizer, criterion):
        seen.append(("warmup", sample.frame_index, id(optimizer)))
        return _stats()

    def joint_step(model, sample, optimizer, config, criterion, anchor, *, probe_audio_visual_gradient):
        seen.append(("joint", sample.frame_index, id(optimizer), probe_audio_visual_gradient))
        return _stats(0.5 if probe_audio_visual_gradient else 0.0)

    evaluator = FakeEvaluator((1.0, 0.9))
    events = []
    trainer = PilotTrainer(
        PilotConfig(warmup_steps=2, joint_steps=3, validation_interval=2, minimum_joint_steps=3),
        evaluator,
        train_config=TrainConfig(),
        warmup_optimizer_factory=optimizer_factory,
        joint_optimizer_factory=optimizer_factory,
        warmup_step_fn=warmup_step,
        joint_step_fn=joint_step,
    )
    result = trainer.run(
        model=model,
        train_samples=train_samples,
        heldout_samples=heldout,
        indices=VariantIndices((3, 1), (4, 0, 2)),
        heldout_indices=(0, 2),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        on_validation=lambda event: events.append(("validation", event)),
        on_best_candidate=lambda event: events.append(("best", event)),
    )

    assert [row[:2] for row in seen] == [
        ("warmup", 3), ("warmup", 1), ("joint", 4), ("joint", 0), ("joint", 2)
    ]
    assert len(optimizer_builds) == 2
    assert seen[0][2] == seen[1][2]
    assert seen[2][2] == seen[3][2] == seen[4][2]
    assert [call[4].name for call in evaluator.calls] == ["step_000002", "step_000003"]
    assert all(call[0] is heldout and call[1] == (0, 2) for call in evaluator.calls)
    assert [kind for kind, _ in events] == ["validation", "best", "validation", "best"]
    assert events[-1][1].selector_state["best_step"] == 3
    assert events[-1][1].optimizer is not None
    assert result.completed_warmup_steps == 2
    assert result.completed_joint_steps == 3
    assert result.stop_reason == "max_steps"
    assert result.best_step == 3

    with (tmp_path / "training_curve.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 5
    summary = json.loads((tmp_path / "worker_summary.json").read_text())
    assert summary["completed_joint_steps"] == 3
    assert [row["step"] for row in summary["validation_history"]] == [2, 3]
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("variant", [Variant.FROZEN_VISUAL, Variant.CONDITION_OFF])
def test_non_joint_variants_skip_probe_and_condition_off_skips_warmup(tmp_path, variant) -> None:
    probes = []
    warmups = []

    def step(model, sample, optimizer, config, criterion, anchor, *, probe_audio_visual_gradient):
        probes.append(probe_audio_visual_gradient)
        return _stats()

    trainer = PilotTrainer(
        PilotConfig(warmup_steps=1, joint_steps=1, validation_interval=1, minimum_joint_steps=1),
        FakeEvaluator((1.0,)),
        warmup_step_fn=lambda *args: warmups.append(True) or _stats(),
        joint_step_fn=step,
    )
    model = TinyTrainFusion()
    model.condition_enabled = True
    trainer.run(
        model=model,
        train_samples=[make_sample()],
        heldout_samples=[make_sample()],
        indices=VariantIndices((0,), (0,)),
        heldout_indices=(0,),
        variant=variant,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path / variant.value,
    )
    assert probes == [False]
    assert len(warmups) == (0 if variant == Variant.CONDITION_OFF else 1)


def test_joint_conditioned_requires_positive_probe_before_success(tmp_path) -> None:
    trainer = PilotTrainer(
        PilotConfig(warmup_steps=0, joint_steps=1, validation_interval=1, minimum_joint_steps=1),
        FakeEvaluator((1.0,)),
        joint_step_fn=lambda *args, **kwargs: _stats(0.0),
    )
    model = TinyTrainFusion()
    model.condition_enabled = True
    with pytest.raises(DisconnectedAudioVisualGradient, match="positive"):
        trainer.run(
            model=model,
            train_samples=[make_sample()], heldout_samples=[make_sample()],
            indices=VariantIndices((), (0,)), heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED, visual_baseline=_baseline(),
            audio_loss_fn=audio_loss, output_dir=tmp_path,
        )
    assert not (tmp_path / "worker_summary.json").exists()


def test_approved_plateau_validates_50_through_400_then_stops(tmp_path) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = True
    config = PilotConfig(warmup_steps=0, joint_steps=500)
    # 50 improves the infinity sentinel, 200 improves by one percent, then four stale.
    evaluator = FakeEvaluator((1.0, 0.995, 0.994, 0.99, 0.989, 0.988, 0.987, 0.986))
    trainer = PilotTrainer(
        config,
        evaluator,
        joint_step_fn=lambda *args, **kwargs: _stats(0.1),
    )
    result = trainer.run(
        model=model,
        train_samples=[make_sample()], heldout_samples=[make_sample()],
        indices=VariantIndices((), (0,) * 500), heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED, visual_baseline=_baseline(),
        audio_loss_fn=audio_loss, output_dir=tmp_path,
    )
    assert [row["step"] for row in result.validation_history] == [
        50, 100, 150, 200, 250, 300, 350, 400
    ]
    assert result.completed_joint_steps == 400
    assert result.stop_reason == "early_stop"

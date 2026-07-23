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
from avgaussianv2.experiment.evaluation import Evaluator
from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotCompatibility,
    PilotResumeError,
)
from avgaussianv2.experiment.training import PilotTrainer, configure_variant
from avgaussianv2.train import (
    DisconnectedAudioVisualGradient,
    TrainStepStats,
    build_joint_optimizer,
)


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


class BoundFakeEvaluator(FakeEvaluator):
    def __init__(self, model, audios=(1.0, 0.9)) -> None:
        super().__init__(audios)
        self.model = model


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


def _pilot_compatibility(variant=Variant.JOINT_CONDITIONED):
    return PilotCompatibility(
        scene_id="scene1_opera",
        variant=variant.value,
        seed=1,
        index_hash="1" * 64,
        visual_checkpoint_sha256="2" * 64,
        audio_checkpoint_sha256="3" * 64,
        camera_mapping_sha256="4" * 64,
        n_fft=512,
        hop_length=128,
        win_length=512,
        sample_rate=48_000,
    )


def _store(
    output_dir,
    compatibility,
    *,
    resume=False,
    overwrite=False,
    identities=None,
):
    stable = {
        "warmup_optimizer_factory": "tests.pilot.warmup_optimizer_factory",
        "joint_optimizer_factory": "tests.pilot.joint_optimizer_factory",
        "warmup_step_fn": "tests.pilot.warmup_step_fn",
        "joint_step_fn": "tests.pilot.joint_step_fn",
        "audio_loss_fn": "tests.pilot.audio_loss_fn",
    }
    if identities:
        stable.update(identities)
    return PilotCheckpointStore(
        output_dir,
        compatibility,
        resume=resume,
        overwrite=overwrite,
        component_identities=stable,
    )


def test_checkpoint_resume_after_validation_does_not_replay_joint_step(tmp_path) -> None:
    seen = []

    def joint_step(model, sample, optimizer, *args, **kwargs):
        seen.append(sample.frame_index)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return _stats(0.1)

    model = TinyTrainFusion()
    model.condition_enabled = True
    config = PilotConfig(
        warmup_steps=0,
        joint_steps=3,
        validation_interval=1,
        minimum_joint_steps=3,
    )
    store = _store(tmp_path, _pilot_compatibility())
    trainer = PilotTrainer(config, FakeEvaluator((1.0, 0.9, 0.8)), joint_step_fn=joint_step)

    with pytest.raises(RuntimeError, match="interrupt"):
        trainer.run(
            model=model,
            train_samples=[_sample_with_frame(i) for i in range(3)],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0, 1, 2)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=store,
            on_validation=lambda event: (
                (_ for _ in ()).throw(RuntimeError("interrupt"))
                if event.step == 1
                else None
            ),
        )

    resumed = TinyTrainFusion()
    resumed.condition_enabled = True
    result = PilotTrainer(
        config, FakeEvaluator((0.9, 0.8)), joint_step_fn=joint_step
    ).run(
        model=resumed,
        train_samples=[_sample_with_frame(i) for i in range(3)],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0, 1, 2)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(
            tmp_path, _pilot_compatibility(), resume=True
        ),
    )

    assert seen == [0, 1, 2]
    assert result.completed_joint_steps == 3
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["stage"] == "complete"
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["best_evaluation_summary"]["audio_total"]["mean"] == 0.8
    assert best["optimizer_state_dict"] is None
    for name, value in resumed.state_dict().items():
        assert torch.equal(best["model_state_dict"][name], value)


def test_checkpoint_condition_off_keeps_warmup_position_zero(tmp_path) -> None:
    variant = Variant.CONDITION_OFF
    model = TinyTrainFusion()
    model.condition_enabled = True
    PilotTrainer(
        PilotConfig(warmup_steps=2, joint_steps=1, validation_interval=1, minimum_joint_steps=1),
        FakeEvaluator((1.0,)),
        joint_step_fn=lambda *args, **kwargs: _stats(),
    ).run(
        model=model,
        train_samples=[make_sample()],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0,)),
        heldout_indices=(0,),
        variant=variant,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(
            tmp_path, _pilot_compatibility(variant)
        ),
    )
    payload = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert payload["next_warmup_position"] == 0


def test_checkpoint_resume_after_warmup_save_does_not_replay_step(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    seen = []
    real_save = checkpoint_module.save_pilot_checkpoint
    saves = 0

    def interrupt_after_first_save(*args, **kwargs):
        nonlocal saves
        real_save(*args, **kwargs)
        saves += 1
        if saves == 1:
            raise RuntimeError("interrupt")

    def warmup_step(model, sample, optimizer, criterion):
        seen.append(sample.frame_index)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return _stats()

    config = PilotConfig(
        warmup_steps=2,
        joint_steps=1,
        validation_interval=1,
        minimum_joint_steps=1,
    )
    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", interrupt_after_first_save)
    model = TinyTrainFusion()
    model.condition_enabled = True
    with pytest.raises(RuntimeError, match="interrupt"):
        PilotTrainer(
            config,
            FakeEvaluator((1.0,)),
            warmup_step_fn=warmup_step,
            joint_step_fn=lambda *args, **kwargs: _stats(0.1),
        ).run(
            model=model,
            train_samples=[_sample_with_frame(10), _sample_with_frame(11)],
            heldout_samples=[make_sample()],
            indices=VariantIndices((0, 1), (0,)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=_store(tmp_path, _pilot_compatibility()),
        )

    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", real_save)
    resumed = TinyTrainFusion()
    resumed.condition_enabled = True
    PilotTrainer(
        config,
        FakeEvaluator((1.0,)),
        warmup_step_fn=warmup_step,
        joint_step_fn=lambda *args, **kwargs: _stats(0.1),
    ).run(
        model=resumed,
        train_samples=[_sample_with_frame(10), _sample_with_frame(11)],
        heldout_samples=[make_sample()],
        indices=VariantIndices((0, 1), (0,)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(
            tmp_path, _pilot_compatibility(), resume=True
        ),
    )
    assert seen == [10, 11]


def test_checkpoint_resume_after_nonvalidation_joint_step_does_not_replay(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    seen = []
    real_save = checkpoint_module.save_pilot_checkpoint

    def interrupt_after_first_joint(*args, **kwargs):
        real_save(*args, **kwargs)
        if kwargs["stage"] == "joint" and kwargs["next_joint_position"] == 1:
            raise RuntimeError("interrupt nonvalidation")

    def joint_step(model, sample, optimizer, *args, **kwargs):
        seen.append(sample.frame_index)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return _stats(0.1)

    config = PilotConfig(
        warmup_steps=0,
        joint_steps=3,
        validation_interval=2,
        minimum_joint_steps=3,
    )
    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", interrupt_after_first_joint)
    model = TinyTrainFusion()
    model.condition_enabled = True
    with pytest.raises(RuntimeError, match="nonvalidation"):
        PilotTrainer(
            config, FakeEvaluator((1.0, 0.9)), joint_step_fn=joint_step
        ).run(
            model=model,
            train_samples=[_sample_with_frame(i) for i in range(3)],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0, 1, 2)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=_store(tmp_path, _pilot_compatibility()),
        )

    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", real_save)
    resumed = TinyTrainFusion()
    resumed.condition_enabled = True
    PilotTrainer(config, FakeEvaluator((1.0, 0.9)), joint_step_fn=joint_step).run(
        model=resumed,
        train_samples=[_sample_with_frame(i) for i in range(3)],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0, 1, 2)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(
            tmp_path, _pilot_compatibility(), resume=True
        ),
    )
    assert seen == [0, 1, 2]


def test_resume_finishes_pending_validation_without_replaying_joint_step(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    seen = []
    real_save = checkpoint_module.save_pilot_checkpoint

    def interrupt_before_validation(*args, **kwargs):
        real_save(*args, **kwargs)
        if kwargs["stage"] == "joint" and kwargs["next_joint_position"] == 2:
            raise RuntimeError("interrupt before validation")

    def joint_step(model, sample, optimizer, *args, **kwargs):
        seen.append(sample.frame_index)
        return _stats(0.1)

    config = PilotConfig(
        warmup_steps=0,
        joint_steps=3,
        validation_interval=2,
        minimum_joint_steps=3,
    )
    first_evaluator = FakeEvaluator((1.0,))
    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", interrupt_before_validation)
    model = TinyTrainFusion()
    model.condition_enabled = True
    with pytest.raises(RuntimeError, match="before validation"):
        PilotTrainer(config, first_evaluator, joint_step_fn=joint_step).run(
            model=model,
            train_samples=[_sample_with_frame(i) for i in range(3)],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0, 1, 2)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=_store(tmp_path, _pilot_compatibility()),
        )
    assert first_evaluator.calls == []

    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", real_save)
    resumed_evaluator = FakeEvaluator((1.0, 0.9))
    resumed = TinyTrainFusion()
    resumed.condition_enabled = True
    PilotTrainer(config, resumed_evaluator, joint_step_fn=joint_step).run(
        model=resumed,
        train_samples=[_sample_with_frame(i) for i in range(3)],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0, 1, 2)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(
            tmp_path, _pilot_compatibility(), resume=True
        ),
    )
    assert seen == [0, 1, 2]
    assert [call[2] for call in resumed_evaluator.calls] == [
        "joint_conditioned_step_000002",
        "joint_conditioned_step_000003",
    ]


def test_best_save_failure_leaves_latest_pending_and_resume_revalidates(
    tmp_path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    real_save = checkpoint_module.save_pilot_checkpoint

    def fail_best(path, **kwargs):
        if path.name == "best.pt":
            raise OSError("injected best failure")
        real_save(path, **kwargs)

    config = PilotConfig(
        warmup_steps=0,
        joint_steps=2,
        validation_interval=1,
        minimum_joint_steps=2,
    )
    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", fail_best)
    model = TinyTrainFusion()
    model.condition_enabled = True
    callbacks = []
    with pytest.raises(OSError, match="best failure"):
        PilotTrainer(
            config,
            FakeEvaluator((1.0,)),
            joint_step_fn=lambda *args, **kwargs: _stats(0.1),
        ).run(
            model=model,
            train_samples=[make_sample()],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0, 0)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=_store(tmp_path, _pilot_compatibility()),
            on_validation=lambda event: callbacks.append("validation"),
            on_best_candidate=lambda event: callbacks.append("best"),
        )
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert latest["next_joint_position"] == 1
    assert latest["validation_history"] == []
    assert latest["selector_state"]["last_step"] is None
    assert callbacks == []

    monkeypatch.setattr(checkpoint_module, "save_pilot_checkpoint", real_save)
    resumed = TinyTrainFusion()
    resumed.condition_enabled = True
    resume_callbacks = []
    PilotTrainer(
        config,
        FakeEvaluator((1.0, 0.9)),
        joint_step_fn=lambda *args, **kwargs: _stats(0.1),
    ).run(
        model=resumed,
        train_samples=[make_sample()],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0, 0)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(
            tmp_path, _pilot_compatibility(), resume=True
        ),
        on_validation=lambda event: resume_callbacks.append(
            ("validation", event.step)
        ),
        on_best_candidate=lambda event: resume_callbacks.append(("best", event.step)),
    )
    assert resume_callbacks == [
        ("validation", 1),
        ("best", 1),
        ("validation", 2),
        ("best", 2),
    ]
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert best["best_evaluation_summary"]["audio_total"]["mean"] == 0.9


def test_latest_preserves_earlier_best_evaluation_summary(tmp_path) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = True
    PilotTrainer(
        PilotConfig(
            warmup_steps=0,
            joint_steps=2,
            validation_interval=1,
            minimum_joint_steps=2,
        ),
        FakeEvaluator((1.0, 1.1)),
        joint_step_fn=lambda *args, **kwargs: _stats(0.1),
    ).run(
        model=model,
        train_samples=[make_sample()],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0, 0)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(tmp_path, _pilot_compatibility()),
    )
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert latest["best_evaluation_summary"]["audio_total"]["mean"] == 1.0
    summary = json.loads((tmp_path / "worker_summary.json").read_text())
    assert summary["checkpoint_io"]["save_count"] >= 5
    assert summary["checkpoint_io"]["save_bytes"] > 0
    assert summary["checkpoint_io"]["save_duration_seconds"] > 0
    assert summary["checkpoint_io"]["cadence"] == "every_completed_optimizer_step"


def test_stop_decision_is_durable_before_callback_and_resume_does_not_train(
    tmp_path,
) -> None:
    seen = []

    def step(model, sample, optimizer, *args, **kwargs):
        seen.append(sample.frame_index)
        return _stats(0.1)

    config = PilotConfig(
        warmup_steps=0,
        joint_steps=3,
        validation_interval=1,
        minimum_joint_steps=0,
        patience=1,
        minimum_relative_improvement=0.1,
    )
    model = TinyTrainFusion()
    model.condition_enabled = True
    with pytest.raises(RuntimeError, match="stop callback"):
        PilotTrainer(config, FakeEvaluator((1.0, 1.0)), joint_step_fn=step).run(
            model=model,
            train_samples=[_sample_with_frame(i) for i in range(3)],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0, 1, 2)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=_store(tmp_path, _pilot_compatibility()),
            on_validation=lambda event: (
                (_ for _ in ()).throw(RuntimeError("stop callback"))
                if event.should_stop
                else None
            ),
        )
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert latest["next_joint_position"] == 2
    assert latest["pending_validation"] is False
    assert latest["stop_requested"] is True

    resumed = TinyTrainFusion()
    resumed.condition_enabled = True
    evaluator = FakeEvaluator(())
    result = PilotTrainer(config, evaluator, joint_step_fn=step).run(
        model=resumed,
        train_samples=[_sample_with_frame(i) for i in range(3)],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0, 1, 2)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        checkpoint_store=_store(
            tmp_path, _pilot_compatibility(), resume=True
        ),
    )
    assert seen == [0, 1]
    assert evaluator.calls == []
    assert result.completed_joint_steps == 2
    assert result.stop_reason == "early_stop"


def test_failed_optimizer_restore_rolls_back_model_and_preserves_reports(
    tmp_path
) -> None:
    config = PilotConfig(
        warmup_steps=0,
        joint_steps=2,
        validation_interval=1,
        minimum_joint_steps=2,
    )
    source = TinyTrainFusion()
    source.condition_enabled = True
    with pytest.raises(RuntimeError, match="interrupt"):
        PilotTrainer(
            config,
            FakeEvaluator((1.0,)),
            joint_step_fn=lambda *args, **kwargs: _stats(0.1),
        ).run(
            model=source,
            train_samples=[make_sample()],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0, 0)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=_store(tmp_path, _pilot_compatibility()),
            on_validation=lambda event: (_ for _ in ()).throw(RuntimeError("interrupt")),
        )
    report = tmp_path / "worker_summary.json"
    report.write_text("preserve me")

    class FailingOptimizer(torch.optim.Adam):
        def load_state_dict(self, state_dict):
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(99)
            raise RuntimeError("injected optimizer failure")

    FailingOptimizer.__module__ = "torch.optim.adam"
    FailingOptimizer.__qualname__ = "Adam"

    resumed = TinyTrainFusion()
    resumed.condition_enabled = False
    before = {name: value.detach().clone() for name, value in resumed.state_dict().items()}
    before_flags = {
        name: parameter.requires_grad for name, parameter in resumed.named_parameters()
    }
    with pytest.raises(PilotResumeError, match="injected optimizer failure"):
        PilotTrainer(
            config,
            FakeEvaluator((0.9,)),
            joint_optimizer_factory=lambda model, config: FailingOptimizer(
                build_joint_optimizer(model, config).param_groups
            ),
            joint_step_fn=lambda *args, **kwargs: _stats(0.1),
        ).run(
            model=resumed,
            train_samples=[make_sample()],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0, 0)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            checkpoint_store=_store(
                tmp_path, _pilot_compatibility(), resume=True
            ),
        )
    assert resumed.condition_enabled is False
    for name, value in resumed.state_dict().items():
        assert torch.equal(value, before[name])
    assert {
        name: parameter.requires_grad for name, parameter in resumed.named_parameters()
    } == before_flags
    assert report.read_text() == "preserve me"


def test_pilot_rejects_evaluator_bound_to_different_model_before_mutation(tmp_path) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = False
    before = {
        name: (parameter.detach().clone(), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }
    evaluator = BoundFakeEvaluator(TinyTrainFusion(), (1.0,))
    trainer = PilotTrainer(
        PilotConfig(
            warmup_steps=0,
            joint_steps=1,
            validation_interval=1,
            minimum_joint_steps=1,
        ),
        evaluator,
    )

    with pytest.raises(ValueError, match="evaluator.*same model"):
        trainer.run(
            model=model,
            train_samples=[make_sample()],
            heldout_samples=[make_sample()],
            indices=VariantIndices((), (0,)),
            heldout_indices=(0,),
            variant=Variant.JOINT_CONDITIONED,
            visual_baseline=_baseline(),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path / "must_not_exist",
        )

    assert model.condition_enabled is False
    for name, parameter in model.named_parameters():
        old_value, old_requires_grad = before[name]
        assert torch.equal(parameter, old_value)
        assert parameter.requires_grad is old_requires_grad
    assert evaluator.calls == []
    assert not (tmp_path / "must_not_exist").exists()


def test_pilot_accepts_evaluator_bound_to_same_model(tmp_path) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = True
    evaluator = BoundFakeEvaluator(model, (1.0,))
    trainer = PilotTrainer(
        PilotConfig(
            warmup_steps=0,
            joint_steps=1,
            validation_interval=1,
            minimum_joint_steps=1,
        ),
        evaluator,
        joint_step_fn=lambda *args, **kwargs: _stats(0.1),
    )

    result = trainer.run(
        model=model,
        train_samples=[make_sample()],
        heldout_samples=[make_sample()],
        indices=VariantIndices((), (0,)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline=_baseline(),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
    )

    assert result.completed_joint_steps == 1
    assert len(evaluator.calls) == 1


def test_pilot_runs_real_training_and_same_model_evaluation_end_to_end(tmp_path) -> None:
    model = TinyTrainFusion()
    model.condition_enabled = True
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    def complete_audio_loss(predicted, target):
        error = predicted - target
        return {
            "total_loss": error.square().mean(),
            "mono_loss": error.mean().abs(),
            "diff_loss": (error[:, 0] - error[:, 1]).abs().mean(),
        }

    trainer = PilotTrainer(
        PilotConfig(
            warmup_steps=1,
            joint_steps=1,
            validation_interval=1,
            minimum_joint_steps=1,
        ),
        Evaluator(model, complete_audio_loss, "cpu"),
        train_config=TrainConfig(
            condition_lr=0.01,
            audio_lr=0.01,
            visual_lr=0.01,
        ),
    )

    result = trainer.run(
        model=model,
        train_samples=[make_sample()],
        heldout_samples=[make_sample()],
            indices=VariantIndices((0,), (0,)),
        heldout_indices=(0,),
        variant=Variant.JOINT_CONDITIONED,
        visual_baseline={
            "rgb_psnr": {"mean": 0.0},
            "rgb_ssim": {"mean": 0.0},
        },
        audio_loss_fn=complete_audio_loss,
        output_dir=tmp_path,
    )

    assert result.completed_warmup_steps == 1
    assert result.completed_joint_steps == 1
    assert result.best_step == 1
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )
    assert (tmp_path / "validation" / "step_000001" / "metrics_summary.json").exists()


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
        indices=VariantIndices(
            () if variant == Variant.CONDITION_OFF else (0,),
            (0,),
        ),
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

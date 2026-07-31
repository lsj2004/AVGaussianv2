from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

import avgaussianv2.benchmark.training as benchmark_training
from avgaussianv2.benchmark.training import (
    BenchmarkCompatibility,
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkResumeError,
    FixedBudgetTrainer,
    build_worker_manifest,
    configure_benchmark_mode,
    hash_shared_indices,
    make_shared_indices,
    verify_resume_artifacts,
)
from avgaussianv2.config import TrainConfig
from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender
from avgaussianv2.train import DisconnectedAudioVisualGradient


class Scalar(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))


class TinyFusion(nn.Module):
    checkpoint_format_version = "tiny-v1"

    def __init__(self) -> None:
        super().__init__()
        self.visual = Scalar(0.2)
        self.acoustic = Scalar(0.1)
        self.condition_encoder = Scalar(0.3)
        self.film = Scalar(0.4)
        self.audio_unet = Scalar(0.5)
        self.condition_enabled = True
        self.visual_forward_calls = 0
        self.audio_forward_calls = 0

    def render_rgbd(self, sample):
        self.visual_forward_calls += 1
        visual = self.visual.value
        rgb = visual.sigmoid().expand(1, 4, 4, 3)
        depth = (visual + 2).expand(1, 4, 4, 1)
        return RGBDRender(rgb, depth, torch.ones_like(depth))

    def forward_audio_only(self, sample):
        self.audio_forward_calls += 1
        gain = self.acoustic.value + self.audio_unet.value
        return sample.source_audio * gain

    def forward(self, sample):
        rgbd = self.render_rgbd(sample)
        condition = self.visual.value * self.condition_encoder.value
        gain = self.acoustic.value + self.audio_unet.value
        if self.condition_enabled:
            gain = gain + condition * self.film.value
        self.audio_forward_calls += 1
        return FusionOutput(
            rgbd,
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


class StochasticTinyFusion(TinyFusion):
    def forward(self, sample):
        output = super().forward(sample)
        noise = (
            torch.rand((), device=output.predicted_audio.device)
            + random.random()
            + float(np.random.random())
        ) * 0.01
        return FusionOutput(
            output.rgbd,
            output.condition,
            output.predicted_audio + noise,
        )


class DisconnectedTinyFusion(TinyFusion):
    def forward(self, sample):
        output = super().forward(sample)
        visual = self.visual.value
        disconnected_gain = (
            self.acoustic.value
            + self.audio_unet.value
            + visual.detach() * self.condition_encoder.value * self.film.value
        )
        return FusionOutput(
            output.rgbd,
            output.condition,
            sample.source_audio * disconnected_gain,
        )


def sample(frame: int) -> AlignedAVSample:
    return AlignedAVSample(
        scene_id="scene1_opera",
        camera="cam00",
        frame_index=frame,
        time_seconds=frame / 30,
        visual_time=torch.tensor([[frame / 30]]),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 12),
        source_audio=torch.full((1, 2, 16), 0.25),
        target_audio=torch.full((1, 2, 16), 0.5),
        target_rgb=torch.full((1, 4, 4, 3), 0.4),
        image_size=(4, 4),
    )


def audio_loss(predicted, target):
    return {"total_loss": torch.nn.functional.mse_loss(predicted, target)}


def flags(model):
    return {
        name: all(parameter.requires_grad for parameter in parameters)
        for name, parameters in model.named_parameter_groups().items()
    }


@pytest.mark.parametrize(
    ("mode", "enabled", "condition"),
    [
        (
            BenchmarkMode.JOINT_CONDITIONED,
            {"visual", "acoustic", "condition_encoder", "film", "audio_unet"},
            True,
        ),
        (BenchmarkMode.AUDIO_ONLY, {"acoustic", "audio_unet"}, False),
        (BenchmarkMode.VISUAL_ONLY, {"visual"}, False),
    ],
)
def test_benchmark_mode_has_exact_main_trainability(mode, enabled, condition) -> None:
    model = TinyFusion()
    configure_benchmark_mode(model, mode, "main")
    assert {name for name, value in flags(model).items() if value} == enabled
    assert model.condition_enabled is condition


def test_only_joint_mode_accepts_conditioner_warmup() -> None:
    model = TinyFusion()
    configure_benchmark_mode(model, BenchmarkMode.JOINT_CONDITIONED, "warmup")
    assert {name for name, value in flags(model).items() if value} == {
        "condition_encoder",
        "film",
    }
    for mode in (BenchmarkMode.AUDIO_ONLY, BenchmarkMode.VISUAL_ONLY):
        with pytest.raises(ValueError, match="warmup"):
            configure_benchmark_mode(model, mode, "warmup")


def test_shared_indices_are_exact_deterministic_and_mode_independent() -> None:
    first = make_shared_indices(dataset_length=7, updates=30_000, seed=42)
    second = make_shared_indices(dataset_length=7, updates=30_000, seed=42)
    assert first == second
    assert len(first) == 30_000
    assert min(first) == 0 and max(first) == 6


def test_strict_worker_manifest_persists_budget_split_and_exact_sequence() -> None:
    config = BenchmarkConfig()
    indices = make_shared_indices(7)
    value = compatibility(BenchmarkMode.JOINT_CONDITIONED, indices)
    manifest = build_worker_manifest(
        config=config, compatibility=value, shared_indices=indices
    )
    assert manifest["schema"] == "avgaussianv2.cam38-benchmark-worker"
    assert manifest["training"]["selection"] == "final"
    assert manifest["training"]["milestones"] == [5_000, 10_000, 30_000]
    assert manifest["compatibility"]["test_camera"] == "cam38"
    assert manifest["shared_indices"] == list(indices)


def tiny_config(**changes) -> BenchmarkConfig:
    return replace(
        BenchmarkConfig(
            main_updates=6,
            conditioner_warmup_steps=2,
            checkpoint_every=2,
            journal_every=1,
            milestones=(2, 4, 6),
        ),
        **changes,
    )


def compatibility(mode: BenchmarkMode, indices: tuple[int, ...]):
    return BenchmarkCompatibility(
        scene_id="scene1_opera",
        mode=mode.value,
        train_cameras=tuple(f"cam{i:02d}" for i in range(38)),
        test_camera="cam38",
        seed=42,
        index_sha256=hash_shared_indices(indices),
        visual_initialization_sha256="1" * 64,
        audio_initialization_sha256="2" * 64,
        model_initialization_sha256="5" * 64,
        source_sha256="3" * 64,
        config_sha256="4" * 64,
    )


class NoEvalSequence:
    def __init__(self, values):
        self.values = values
        self.seen = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        assert value.camera != "cam38"
        self.seen.append(value.frame_index)
        return value


def run(tmp_path, mode, *, interrupt=None, stop=None, resume=False, config=None):
    config = config or tiny_config()
    indices = (0, 1, 2, 0, 1, 2)
    model = TinyFusion()
    samples = NoEvalSequence([sample(index) for index in range(3)])
    trainer = FixedBudgetTrainer(config)
    result = trainer.run(
        model=model,
        train_samples=samples,
        shared_indices=indices,
        mode=mode,
        train_config=TrainConfig(
            seed=42, warmup_steps=2, joint_steps=6, gradient_probe_interval=100
        ),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        compatibility=compatibility(mode, indices),
        resume=resume,
        stop_after_main_step=stop,
        interrupt_after_main_step=interrupt,
    )
    return model, samples, result


def run_with_model(
    tmp_path,
    model,
    mode,
    *,
    interrupt=None,
    resume=False,
    config=None,
    train_config=None,
):
    config = config or tiny_config()
    indices = (0, 1, 2, 0, 1, 2)
    return FixedBudgetTrainer(config).run(
        model=model,
        train_samples=NoEvalSequence([sample(index) for index in range(3)]),
        shared_indices=indices,
        mode=mode,
        train_config=train_config
        or TrainConfig(
            seed=42,
            warmup_steps=2,
            joint_steps=6,
            gradient_probe_interval=100,
        ),
        audio_loss_fn=audio_loss,
        output_dir=tmp_path,
        compatibility=compatibility(mode, indices),
        resume=resume,
        interrupt_after_main_step=interrupt,
    )


def test_planned_budget_can_pause_at_exact_step_and_resume_same_contract(tmp_path):
    _, _, paused = run(tmp_path, BenchmarkMode.AUDIO_ONLY, stop=3)
    assert paused.selection == "paused"
    assert paused.completed_main_updates == 3
    assert paused.final_checkpoint.name == "main_step_000003.pt"
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["observed_main_step"] == progress["exact_main_step"] == 3

    _, _, completed = run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)
    assert completed.selection == "final"
    assert completed.resumed_from_main_step == 3
    assert completed.completed_main_updates == 6


@pytest.mark.parametrize("mode", list(BenchmarkMode))
def test_fixed_budget_updates_only_allowed_groups_and_never_accepts_eval_data(
    tmp_path, mode
) -> None:
    before = TinyFusion()
    model, samples, result = run(tmp_path / mode.value, mode)
    changed = {
        name
        for name, parameters in model.named_parameter_groups().items()
        if any(
            not torch.equal(left, right)
            for left, right in zip(parameters, before.named_parameter_groups()[name])
        )
    }
    expected = {
        BenchmarkMode.JOINT_CONDITIONED: {
            "visual",
            "acoustic",
            "condition_encoder",
            "film",
            "audio_unet",
        },
        BenchmarkMode.AUDIO_ONLY: {"acoustic", "audio_unet"},
        BenchmarkMode.VISUAL_ONLY: {"visual"},
    }[mode]
    assert changed == expected
    assert result.completed_main_updates == 6
    assert result.completed_warmup_steps == (
        2 if mode == BenchmarkMode.JOINT_CONDITIONED else 0
    )
    assert result.selection == "final"
    assert (tmp_path / mode.value / "final.pt").is_file()
    assert all(frame in {0, 1, 2} for frame in samples.seen)
    expected_calls = {
        BenchmarkMode.JOINT_CONDITIONED: (8, 8),
        BenchmarkMode.AUDIO_ONLY: (0, 6),
        BenchmarkMode.VISUAL_ONLY: (6, 0),
    }[mode]
    assert (model.visual_forward_calls, model.audio_forward_calls) == expected_calls


def test_audio_only_step_uses_lre_regularizer_and_preserves_base_metric() -> None:
    model = TinyFusion()
    configure_benchmark_mode(model, BenchmarkMode.AUDIO_ONLY, "main")
    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=0.01,
    )
    value = sample(0)
    value = type(value)(
        **{
            **vars(value),
            "source_audio": torch.stack(
                (torch.full((16,), 0.5), torch.full((16,), 0.25)),
                dim=0,
            ).unsqueeze(0),
            "target_audio": torch.stack(
                (torch.full((16,), 0.25), torch.full((16,), 0.5)),
                dim=0,
            ).unsqueeze(0),
        }
    )

    stats = benchmark_training._audio_only_step(
        model,
        value,
        optimizer,
        TrainConfig(lambda_lre=0.02),
        audio_loss,
    )

    assert stats.losses["audio"] == stats.losses["audio_base"]
    assert stats.losses["audio_lre"] > 0
    assert stats.losses["audio_lre_weighted"] > 0
    assert stats.total == pytest.approx(
        stats.losses["audio_base"] + stats.losses["audio_lre_weighted"]
    )
    assert stats.losses["pred_lre_db"] > 0
    assert stats.losses["target_lre_db"] < 0


@pytest.mark.parametrize(
    "mode", [BenchmarkMode.JOINT_CONDITIONED, BenchmarkMode.AUDIO_ONLY]
)
def test_cam38_target_is_rejected_in_warmup_and_main(tmp_path, mode) -> None:
    forbidden = sample(0)
    forbidden = type(forbidden)(**{**vars(forbidden), "camera": "cam38"})
    indices = (0,) * 6
    with pytest.raises(ValueError, match="outside cam00 through cam37"):
        FixedBudgetTrainer(tiny_config()).run(
            model=TinyFusion(),
            train_samples=[forbidden],
            shared_indices=indices,
            mode=mode,
            train_config=TrainConfig(
                seed=42, warmup_steps=2, joint_steps=6, gradient_probe_interval=100
            ),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path / mode.value,
            compatibility=compatibility(mode, indices),
        )


def test_mid_period_resume_replays_from_last_exact_checkpoint(tmp_path) -> None:
    config = tiny_config(checkpoint_every=3, milestones=(6,))
    with pytest.raises(RuntimeError, match="injected"):
        run(
            tmp_path,
            BenchmarkMode.AUDIO_ONLY,
            interrupt=5,
            config=config,
        )
    journal = json.loads((tmp_path / "progress.json").read_text())
    assert journal["observed_main_step"] == 5
    _, samples, result = run(
        tmp_path,
        BenchmarkMode.AUDIO_ONLY,
        resume=True,
        config=config,
    )
    assert result.resumed_from_main_step == 3
    assert result.redone_main_updates == 2
    assert result.redone_main_updates <= config.checkpoint_every - 1
    assert samples.seen == [0, 1, 2]


def test_joint_resume_matches_uninterrupted_final_state(tmp_path) -> None:
    config = tiny_config(checkpoint_every=3, milestones=(6,))
    with pytest.raises(RuntimeError, match="injected"):
        run(
            tmp_path / "resumed",
            BenchmarkMode.JOINT_CONDITIONED,
            interrupt=5,
            config=config,
        )
    resumed, _, _ = run(
        tmp_path / "resumed",
        BenchmarkMode.JOINT_CONDITIONED,
        resume=True,
        config=config,
    )
    uninterrupted, _, _ = run(
        tmp_path / "uninterrupted",
        BenchmarkMode.JOINT_CONDITIONED,
        config=config,
    )
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)


def test_atomic_final_milestones_retention_and_io_counters(tmp_path) -> None:
    _, _, result = run(
        tmp_path,
        BenchmarkMode.VISUAL_ONLY,
        config=tiny_config(checkpoint_every=1),
    )
    assert sorted(path.name for path in (tmp_path / "milestones").iterdir()) == [
        "step_000002.pt",
        "step_000004.pt",
        "step_000006.pt",
    ]
    periodic = sorted(path.name for path in (tmp_path / "checkpoints").iterdir())
    assert periodic == ["main_step_000005.pt", "main_step_000006.pt"]
    assert result.io.checkpoint_writes >= 6
    assert result.io.checkpoint_bytes > 0
    assert result.io.journal_writes >= 6
    assert not list(tmp_path.rglob("*.tmp"))


def test_resume_fails_closed_on_fingerprint_change(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    indices = (0, 1, 2, 0, 1, 2)
    bad = replace(
        compatibility(BenchmarkMode.AUDIO_ONLY, indices), config_sha256="9" * 64
    )
    with pytest.raises(BenchmarkResumeError, match="fingerprint"):
        FixedBudgetTrainer(tiny_config()).run(
            model=TinyFusion(),
            train_samples=[sample(i) for i in range(3)],
            shared_indices=indices,
            mode=BenchmarkMode.AUDIO_ONLY,
            train_config=TrainConfig(seed=42, warmup_steps=2, joint_steps=6),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            compatibility=bad,
            resume=True,
        )


def test_resume_fails_closed_when_effective_train_config_changes(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    indices = (0, 1, 2, 0, 1, 2)
    with pytest.raises(BenchmarkResumeError, match="fingerprint"):
        FixedBudgetTrainer(tiny_config()).run(
            model=TinyFusion(),
            train_samples=[sample(i) for i in range(3)],
            shared_indices=indices,
            mode=BenchmarkMode.AUDIO_ONLY,
            train_config=TrainConfig(
                seed=42,
                warmup_steps=2,
                joint_steps=6,
                audio_lr=9e-4,
            ),
            audio_loss_fn=audio_loss,
            output_dir=tmp_path,
            compatibility=compatibility(BenchmarkMode.AUDIO_ONLY, indices),
            resume=True,
        )


def test_contract_rejects_wrong_budget_split_and_milestones() -> None:
    BenchmarkConfig().validate()
    with pytest.raises(ValueError, match="30,000"):
        BenchmarkConfig(main_updates=10, milestones=(10,)).validate()
    with pytest.raises(ValueError, match="milestones"):
        BenchmarkConfig(milestones=(5_000, 30_000)).validate()
    indices = make_shared_indices(3, 30_000, 42)
    value = compatibility(BenchmarkMode.AUDIO_ONLY, indices)
    with pytest.raises(ValueError, match="train_cameras"):
        replace(value, train_cameras=("cam00",))


def test_resume_prevalidates_entire_payload_before_mutating_model(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    checkpoint = tmp_path / "checkpoints" / "main_step_000002.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["optimizer"] = {"state": payload["optimizer"]["state"]}
    torch.save(payload, checkpoint)
    model = TinyFusion()
    model.condition_enabled = True
    before_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(BenchmarkResumeError, match="optimizer fields"):
        run_with_model(
            tmp_path,
            model,
            BenchmarkMode.AUDIO_ONLY,
            resume=True,
        )
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    assert model.condition_enabled is True
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == before_requires_grad


def test_resume_rejects_malformed_rng_length_without_mutation(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    checkpoint = tmp_path / "checkpoints" / "main_step_000002.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["torch_rng_state"] = torch.zeros(3, dtype=torch.uint8)
    torch.save(payload, checkpoint)
    model = TinyFusion()
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    before_flags = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    with pytest.raises(BenchmarkResumeError, match="torch RNG"):
        run_with_model(tmp_path, model, BenchmarkMode.AUDIO_ONLY, resume=True)
    for name, expected in before_model.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == before_flags
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]


def test_resume_rolls_back_model_optimizer_rng_and_flags_on_late_failure(
    tmp_path, monkeypatch
) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    model = TinyFusion()
    original_builder = benchmark_training.build_joint_optimizer
    captured = {}

    def capture_optimizer(current_model, config):
        optimizer = original_builder(current_model, config)
        captured["optimizer"] = optimizer
        captured["before"] = copy.deepcopy(optimizer.state_dict())
        return optimizer

    monkeypatch.setattr(benchmark_training, "build_joint_optimizer", capture_optimizer)
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    before_flags = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    before_condition = model.condition_enabled
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    original_set_state = np.random.set_state
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected NumPy restore failure")
        original_set_state(state)

    monkeypatch.setattr(np.random, "set_state", fail_once)
    with pytest.raises(ValueError, match="injected NumPy"):
        run_with_model(tmp_path, model, BenchmarkMode.AUDIO_ONLY, resume=True)
    for name, expected in before_model.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    assert captured["optimizer"].state_dict() == captured["before"]
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == before_flags
    assert model.condition_enabled is before_condition
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]


def test_resume_rejects_tampered_checkpoint_io_sidecar(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    sidecar_path = tmp_path / "checkpoint_io.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["fingerprint_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(BenchmarkResumeError, match="sidecar metadata"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)


def test_resume_rejects_checkpoint_io_sidecar_counter_rollback(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    sidecar_path = tmp_path / "checkpoint_io.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["io"]["checkpoint_writes"] = 0
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(BenchmarkResumeError, match="sidecar counter rollback"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)


def test_resume_merges_journal_counters_newer_than_exact_checkpoint(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=3)
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "main_step_000002.pt",
        map_location="cpu",
        weights_only=True,
    )
    sidecar = json.loads((tmp_path / "checkpoint_io.json").read_text())
    assert sidecar["io"]["journal_writes"] > checkpoint["io"]["journal_writes"]
    _, _, result = run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)
    assert result.io.journal_writes == sidecar["io"]["journal_writes"] + 4


def test_checkpoint_sidecar_commit_marker_ignores_newer_uncommitted_checkpoint(
    tmp_path, monkeypatch
) -> None:
    config = tiny_config(checkpoint_every=2)
    original_persist = benchmark_training._persist_io_sidecar

    def crash_before_step_two_commit(output, fingerprint, counters, **kwargs):
        if (
            counters.checkpoint_writes > 1
            and (output / "checkpoints" / "main_step_000002.pt").is_file()
        ):
            raise RuntimeError("injected checkpoint commit-marker crash")
        original_persist(output, fingerprint, counters, **kwargs)

    monkeypatch.setattr(
        benchmark_training, "_persist_io_sidecar", crash_before_step_two_commit
    )
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    with pytest.raises(RuntimeError, match="commit-marker crash"):
        run_with_model(
            tmp_path / "resumed",
            StochasticTinyFusion(),
            BenchmarkMode.AUDIO_ONLY,
            config=config,
        )
    sidecar = json.loads((tmp_path / "resumed" / "checkpoint_io.json").read_text())
    assert set(sidecar["committed_checkpoints"]) == {"main_step_000000.pt"}
    assert (tmp_path / "resumed" / "checkpoints" / "main_step_000002.pt").is_file()

    monkeypatch.setattr(benchmark_training, "_persist_io_sidecar", original_persist)
    random.seed(901)
    np.random.seed(902)
    torch.manual_seed(903)
    resumed = StochasticTinyFusion()
    result = run_with_model(
        tmp_path / "resumed",
        resumed,
        BenchmarkMode.AUDIO_ONLY,
        resume=True,
        config=config,
    )
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    uninterrupted = StochasticTinyFusion()
    run_with_model(
        tmp_path / "uninterrupted",
        uninterrupted,
        BenchmarkMode.AUDIO_ONLY,
        config=config,
    )
    assert result.resumed_from_main_step == 0
    assert result.redone_main_updates == 2
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)


def test_checkpoint_sidecar_commit_marker_rejects_tampered_previous(
    tmp_path, monkeypatch
) -> None:
    config = tiny_config(checkpoint_every=2)
    original_persist = benchmark_training._persist_io_sidecar

    def crash_before_step_two_commit(output, fingerprint, counters, **kwargs):
        if (
            counters.checkpoint_writes > 1
            and (output / "checkpoints" / "main_step_000002.pt").is_file()
        ):
            raise RuntimeError("injected checkpoint commit-marker crash")
        original_persist(output, fingerprint, counters, **kwargs)

    monkeypatch.setattr(
        benchmark_training, "_persist_io_sidecar", crash_before_step_two_commit
    )
    with pytest.raises(RuntimeError, match="commit-marker crash"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, config=config)
    previous = tmp_path / "checkpoints" / "main_step_000000.pt"
    payload = torch.load(previous, map_location="cpu", weights_only=True)
    payload["model"]["acoustic.value"].add_(1)
    torch.save(payload, previous)
    monkeypatch.setattr(benchmark_training, "_persist_io_sidecar", original_persist)
    with pytest.raises(BenchmarkResumeError, match="committed checkpoint hash"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True, config=config)


def test_initial_journal_permits_fresh_resume_without_committed_checkpoint(
    tmp_path, monkeypatch
) -> None:
    original_persist = benchmark_training._persist_io_sidecar

    def crash_before_initial_commit(output, fingerprint, counters, **kwargs):
        if (
            counters.checkpoint_writes == 1
            and (output / "checkpoints" / "main_step_000000.pt").is_file()
        ):
            raise RuntimeError("injected initial commit-marker crash")
        original_persist(output, fingerprint, counters, **kwargs)

    monkeypatch.setattr(
        benchmark_training, "_persist_io_sidecar", crash_before_initial_commit
    )
    with pytest.raises(RuntimeError, match="initial commit-marker crash"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY)
    sidecar = json.loads((tmp_path / "checkpoint_io.json").read_text())
    assert sidecar["committed_checkpoints"] == {}
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["observed_main_step"] == 0
    monkeypatch.setattr(benchmark_training, "_persist_io_sidecar", original_persist)
    _, _, result = run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)
    assert result.resumed_from_main_step == 0
    assert result.completed_main_updates == 6


def test_stochastic_cpu_resume_is_bitwise_exact(tmp_path) -> None:
    config = tiny_config(checkpoint_every=3, milestones=(6,))
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    with pytest.raises(RuntimeError, match="injected"):
        run_with_model(
            tmp_path / "resumed",
            StochasticTinyFusion(),
            BenchmarkMode.AUDIO_ONLY,
            interrupt=5,
            config=config,
        )
    random.seed(901)
    np.random.seed(902)
    torch.manual_seed(903)
    resumed = StochasticTinyFusion()
    run_with_model(
        tmp_path / "resumed",
        resumed,
        BenchmarkMode.AUDIO_ONLY,
        resume=True,
        config=config,
    )

    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    uninterrupted = StochasticTinyFusion()
    run_with_model(
        tmp_path / "uninterrupted",
        uninterrupted,
        BenchmarkMode.AUDIO_ONLY,
        config=config,
    )
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)


def test_joint_zero_gradient_guard_survives_exact_resume(tmp_path) -> None:
    config = tiny_config(checkpoint_every=1)
    train_config = TrainConfig(
        seed=42,
        warmup_steps=2,
        joint_steps=6,
        gradient_probe_interval=1,
        max_zero_audio_visual_grad_steps=2,
    )
    with pytest.raises(RuntimeError, match="injected"):
        run_with_model(
            tmp_path,
            DisconnectedTinyFusion(),
            BenchmarkMode.JOINT_CONDITIONED,
            interrupt=1,
            config=config,
            train_config=train_config,
        )
    checkpoint = tmp_path / "checkpoints" / "main_step_000001.pt"
    guard = torch.load(checkpoint, map_location="cpu", weights_only=True)[
        "gradient_guard"
    ]
    assert guard == {
        "consecutive_zero_audio_visual_probes": 1,
        "audio_visual_probe_count": 1,
    }
    with pytest.raises(DisconnectedAudioVisualGradient, match="2 consecutive"):
        run_with_model(
            tmp_path,
            DisconnectedTinyFusion(),
            BenchmarkMode.JOINT_CONDITIONED,
            resume=True,
            config=config,
            train_config=train_config,
        )


@pytest.mark.parametrize(
    ("relative_path", "field", "value"),
    [
        ("final.pt", "main_step", 5),
        ("milestones/step_000002.pt", "stage", "warmup"),
    ],
)
def test_completed_resume_rejects_tampered_publications(
    tmp_path, relative_path, field, value
) -> None:
    run(tmp_path, BenchmarkMode.AUDIO_ONLY)
    artifact = tmp_path / relative_path
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload[field] = value
    torch.save(payload, artifact)
    with pytest.raises(BenchmarkResumeError, match="checkpoint|artifact"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)


def test_completed_resume_rejects_shape_valid_milestone_model_tamper(
    tmp_path,
) -> None:
    run(tmp_path, BenchmarkMode.AUDIO_ONLY)
    artifact = tmp_path / "milestones" / "step_000002.pt"
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["model"]["acoustic.value"].add_(1)
    torch.save(payload, artifact)
    with pytest.raises(BenchmarkResumeError, match="artifact hash mismatch"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)


def test_completed_resume_only_rebuilds_publication_at_authoritative_step(
    tmp_path,
) -> None:
    run(tmp_path, BenchmarkMode.AUDIO_ONLY)
    rolling = tmp_path / "checkpoints" / "main_step_000006.pt"
    authoritative_hash = hashlib.sha256(rolling.read_bytes()).hexdigest()
    (tmp_path / "final.pt").unlink()
    (tmp_path / "milestones" / "step_000006.pt").unlink()
    _, _, result = run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)
    assert hashlib.sha256(result.final_checkpoint.read_bytes()).hexdigest() == (
        authoritative_hash
    )
    assert hashlib.sha256(result.milestones[-1].read_bytes()).hexdigest() == (
        authoritative_hash
    )
    result.milestones[0].unlink()
    with pytest.raises(BenchmarkResumeError, match="required milestone is missing"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)


@pytest.mark.parametrize("partial_manifest", ["missing", "truncated", "valid"])
def test_final_publications_before_manifest_crash_self_heals_atomically(
    tmp_path, partial_manifest
) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(
            tmp_path,
            BenchmarkMode.AUDIO_ONLY,
            interrupt=6,
        )
    assert (tmp_path / "final.pt").is_file()
    assert (tmp_path / "artifact_journal.json").is_file()
    manifest = tmp_path / "artifact_hashes.json"
    assert not manifest.exists()
    if partial_manifest == "truncated":
        manifest.write_text('{"schema":')
    elif partial_manifest == "valid":
        journal = json.loads((tmp_path / "artifact_journal.json").read_text())
        first_name = "milestones/step_000002.pt"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "avgaussianv2.cam38-fixed-budget.artifacts",
                    "version": 1,
                    "fingerprint_sha256": journal["fingerprint_sha256"],
                    "sha256": {first_name: journal["sha256"][first_name]},
                }
            )
        )

    _, _, result = run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)

    published = json.loads(manifest.read_text())
    assert set(published["sha256"]) == {
        "milestones/step_000002.pt",
        "milestones/step_000004.pt",
        "milestones/step_000006.pt",
        "final.pt",
    }
    assert result.completed_main_updates == 6
    assert not list(tmp_path.glob(".artifact_hashes.json.*.tmp"))


def test_final_before_manifest_crash_still_rejects_publication_tamper(
    tmp_path,
) -> None:
    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, interrupt=6)
    final = tmp_path / "final.pt"
    payload = torch.load(final, map_location="cpu", weights_only=True)
    payload["model"]["acoustic.value"].add_(1)
    torch.save(payload, final)
    with pytest.raises(BenchmarkResumeError, match="artifact hash mismatch"):
        run(tmp_path, BenchmarkMode.AUDIO_ONLY, resume=True)


def test_checkpoint_io_snapshot_includes_its_own_final_write(tmp_path) -> None:
    _, _, result = run(
        tmp_path,
        BenchmarkMode.VISUAL_ONLY,
        config=tiny_config(checkpoint_every=1),
    )
    final_payload = torch.load(
        result.final_checkpoint, map_location="cpu", weights_only=True
    )
    persisted_json = json.loads((tmp_path / "checkpoint_io.json").read_text())
    assert persisted_json["schema"].endswith(".checkpoint-io")
    assert persisted_json["version"] == 1
    assert persisted_json["io"] == result.io.to_mapping()
    payload_io = dict(final_payload["io"])
    result_io = result.io.to_mapping()
    assert payload_io.pop("checkpoint_seconds") <= result_io.pop("checkpoint_seconds")
    assert payload_io == result_io
    assert persisted_json["io"]["checkpoint_seconds"] > 0
    assert final_payload["io"]["checkpoint_writes"] >= 3
    assert final_payload["io"]["checkpoint_bytes"] >= (
        result.final_checkpoint.stat().st_size * 3
    )


def _strict_resume_fixture(output, *, exact_main_step=0):
    training = BenchmarkConfig()
    compatibility = BenchmarkCompatibility(
        scene_id="scene1_opera",
        mode=BenchmarkMode.AUDIO_ONLY.value,
        train_cameras=benchmark_training.TRAIN_CAMERAS,
        test_camera=benchmark_training.TEST_CAMERA,
        seed=42,
        index_sha256="1" * 64,
        visual_initialization_sha256="2" * 64,
        audio_initialization_sha256="3" * 64,
        model_initialization_sha256="4" * 64,
        source_sha256="5" * 64,
        config_sha256="6" * 64,
    ).to_mapping()
    fingerprint_inputs = {"strict": True}
    fingerprint = {
        "inputs": fingerprint_inputs,
        "sha256": hashlib.sha256(
            json.dumps(
                fingerprint_inputs, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    manifest = {
        "training": {
            **benchmark_training.asdict(training),
            "milestones": list(training.milestones),
        },
        "compatibility": compatibility,
        "shared_indices": [0],
    }
    output.mkdir(exist_ok=True)
    (output / "contract.json").write_text(
        json.dumps(
            {
                "schema": f"{benchmark_training.SCHEMA}.contract",
                "version": benchmark_training.SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "compatibility": compatibility,
                "shared_indices": [0],
                "selection": "final",
                "milestones": list(training.milestones),
            }
        )
    )
    (output / "progress.json").write_text(
        json.dumps(
            {
                "schema": f"{benchmark_training.SCHEMA}.progress",
                "version": benchmark_training.SCHEMA_VERSION,
                "stage": "main",
                "observed_warmup_step": 0,
                "observed_main_step": exact_main_step,
                "exact_warmup_step": 0,
                "exact_main_step": exact_main_step,
                "maximum_replay_updates": training.checkpoint_every,
                "fingerprint_sha256": fingerprint["sha256"],
            }
        )
    )
    committed = {}
    if exact_main_step:
        checkpoints = output / "checkpoints"
        checkpoints.mkdir()
        name = f"main_step_{exact_main_step:06d}.pt"
        payload = {
            key: {}
            for key in benchmark_training._CHECKPOINT_KEYS
        }
        payload.update(
            schema=benchmark_training.SCHEMA,
            version=benchmark_training.SCHEMA_VERSION,
            fingerprint=fingerprint,
            compatibility=compatibility,
            stage="main",
            warmup_step=0,
            main_step=exact_main_step,
        )
        torch.save(payload, checkpoints / name)
        committed[name] = hashlib.sha256((checkpoints / name).read_bytes()).hexdigest()
    (output / "checkpoint_io.json").write_text(
        json.dumps(
            {
                "schema": f"{benchmark_training.SCHEMA}.checkpoint-io",
                "version": benchmark_training.SCHEMA_VERSION,
                "fingerprint_sha256": fingerprint["sha256"],
                "io": benchmark_training.CheckpointIO().to_mapping(),
                "committed_checkpoints": committed,
            }
        )
    )
    return manifest


def test_strict_resume_verifier_rejects_progress_without_exact_checkpoint(
    tmp_path,
) -> None:
    manifest = _strict_resume_fixture(tmp_path, exact_main_step=500)
    (tmp_path / "checkpoints" / "main_step_000500.pt").unlink()
    sidecar = json.loads((tmp_path / "checkpoint_io.json").read_text())
    sidecar["committed_checkpoints"] = {}
    (tmp_path / "checkpoint_io.json").write_text(json.dumps(sidecar))

    with pytest.raises(BenchmarkResumeError, match="no committed exact checkpoint"):
        verify_resume_artifacts(tmp_path, worker_manifest=manifest)


def test_strict_resume_verifier_rejects_corrupt_io_sidecar(tmp_path) -> None:
    manifest = _strict_resume_fixture(tmp_path)
    sidecar = json.loads((tmp_path / "checkpoint_io.json").read_text())
    sidecar["fingerprint_sha256"] = "f" * 64
    (tmp_path / "checkpoint_io.json").write_text(json.dumps(sidecar))

    with pytest.raises(BenchmarkResumeError, match="sidecar metadata"):
        verify_resume_artifacts(tmp_path, worker_manifest=manifest)


def test_strict_resume_verifier_rejects_corrupt_checkpoint_and_journal(
    tmp_path,
) -> None:
    manifest = _strict_resume_fixture(tmp_path, exact_main_step=500)
    checkpoint = tmp_path / "checkpoints" / "main_step_000500.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(BenchmarkResumeError, match="checkpoint hash"):
        verify_resume_artifacts(tmp_path, worker_manifest=manifest)

    manifest = _strict_resume_fixture(tmp_path / "journal")
    (tmp_path / "journal" / "published.pt").write_bytes(b"payload")
    (tmp_path / "journal" / "artifact_journal.json").write_text(
        json.dumps(
            {
                "schema": f"{benchmark_training.SCHEMA}.artifact-journal",
                "version": benchmark_training.SCHEMA_VERSION,
                "fingerprint_sha256": json.loads(
                    (tmp_path / "journal" / "contract.json").read_text()
                )["fingerprint"]["sha256"],
                "sha256": {"published.pt": "0" * 64},
            }
        )
    )
    with pytest.raises(BenchmarkResumeError, match="artifact transaction hash"):
        verify_resume_artifacts(tmp_path / "journal", worker_manifest=manifest)


def test_checkpoint_seconds_include_atomic_fsync_and_manifest_publication(
    tmp_path, monkeypatch
) -> None:
    original = benchmark_training._atomic_bytes
    measured_delay = 0.0

    def delayed(path, data):
        nonlocal measured_delay
        delay = 0.01 if path.name == "artifact_hashes.json" else 0.001
        started = time.monotonic()
        time.sleep(delay)
        result = original(path, data)
        measured_delay += time.monotonic() - started
        return result

    monkeypatch.setattr(benchmark_training, "_atomic_bytes", delayed)
    _, _, result = run(tmp_path, BenchmarkMode.VISUAL_ONLY)
    assert result.io.checkpoint_seconds >= 0.01
    # Journals and the final checkpoint_io sidecar are intentionally distinct
    # counters, so only checkpoint/publication writes must be covered.
    assert result.io.checkpoint_seconds < measured_delay

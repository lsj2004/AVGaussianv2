from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest
import torch
from torch import nn

from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender
from avgaussianv2.experiment.evaluation import Evaluator, move_sample


def sample(index: int, *, camera: str | None = None) -> AlignedAVSample:
    target_audio = torch.full((1, 2, 640), 0.2 + 0.01 * index)
    target_rgb = torch.full((1, 8, 8, 3), 0.3 + 0.01 * index)
    return AlignedAVSample(
        scene_id="fixture",
        camera=camera or f"cam{index:02d}",
        frame_index=index,
        time_seconds=index / 20.0,
        visual_time=torch.tensor([[index / 20.0]]),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 12),
        source_audio=target_audio - 0.03,
        target_audio=target_audio,
        target_rgb=target_rgb,
        image_size=(8, 8),
    )


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.02))
        self.condition_enabled = False
        self.raise_error = False

    def forward(self, item: AlignedAVSample) -> FusionOutput:
        if self.raise_error:
            raise RuntimeError("forced model failure")
        condition_offset = self.weight if self.condition_enabled else -self.weight
        rgb = item.target_rgb + condition_offset
        audio = item.target_audio + condition_offset
        depth = torch.ones((*rgb.shape[:-1], 1), device=rgb.device, dtype=rgb.dtype)
        return FusionOutput(
            rgbd=RGBDRender(rgb=rgb, depth=depth, alpha=torch.ones_like(depth)),
            condition=condition_offset.reshape(1, 1),
            predicted_audio=audio,
        )


def audio_loss(predicted: torch.Tensor, target: torch.Tensor):
    error = predicted - target
    return {
        "total_loss": error.square().mean(),
        "mono_loss": error.mean().abs(),
        "diff_loss": (error[:, 0] - error[:, 1]).abs().mean(),
    }


def test_evaluate_selected_indices_writes_exact_rows_and_summary(tmp_path) -> None:
    model = TinyModel()
    model.train()
    evaluator = Evaluator(model, audio_loss, "cpu")

    result = evaluator.evaluate(
        [sample(0), sample(1), sample(2)],
        [0, 2],
        system_name="joint_conditioned",
        condition_enabled=True,
        output_dir=tmp_path,
    )

    metric_keys = {
        "audio_total", "audio_mono", "audio_diff", "waveform_l1",
        "mono_lsd", "diff_lsd", "lre_error_db", "rgb_psnr", "rgb_ssim", "rgb_l1",
    }
    metadata_keys = {"sample_id", "scene_id", "camera", "frame_index", "time_seconds"}
    assert result.system_name == "joint_conditioned"
    assert result.count == 2
    assert all(set(row) == metadata_keys | metric_keys for row in result.rows)
    assert len({row["sample_id"] for row in result.rows}) == 2
    assert set(result.summary) == metric_keys
    assert {"audio_total", "rgb_psnr", "rgb_ssim"} <= set(result.summary)
    assert model.training is True
    assert model.condition_enabled is False

    rows = [json.loads(line) for line in (tmp_path / "metrics_per_sample.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert rows == list(result.rows)
    assert summary == result.summary


def test_sample_ids_are_stable_and_duplicate_identity_is_rejected(tmp_path) -> None:
    evaluator = Evaluator(TinyModel(), audio_loss, "cpu")
    first = evaluator.evaluate([sample(0)], [0], "a", True, tmp_path / "a")
    second = evaluator.evaluate([sample(0)], [0], "b", False, tmp_path / "b")
    assert first.rows[0]["sample_id"] == second.rows[0]["sample_id"]

    with pytest.raises(ValueError, match="duplicate sample ID"):
        evaluator.evaluate([sample(0), sample(0)], [0, 1], "a", True, tmp_path / "c")


def test_model_state_is_restored_after_model_exception(tmp_path) -> None:
    model = TinyModel()
    model.eval()
    model.condition_enabled = True
    model.raise_error = True

    with pytest.raises(RuntimeError, match="forced"):
        Evaluator(model, audio_loss, "cpu").evaluate(
            [sample(0)], [0], "broken", False, tmp_path
        )

    assert model.training is False
    assert model.condition_enabled is True
    assert not (tmp_path / "metrics_per_sample.jsonl").exists()
    assert not (tmp_path / "metrics_summary.json").exists()


@pytest.mark.parametrize("indices", [[], [-1], [2], [0, 0], [0.5], [True]])
def test_invalid_indices_are_rejected(indices, tmp_path) -> None:
    with pytest.raises((ValueError, TypeError)):
        Evaluator(TinyModel(), audio_loss, "cpu").evaluate(
            [sample(0), sample(1)], indices, "x", True, tmp_path
        )


@pytest.mark.parametrize(
    "bad_loss",
    [
        lambda p, t: {"total_loss": torch.tensor(1.0)},
        lambda p, t: {"total_loss": torch.tensor(float("nan")), "mono_loss": torch.tensor(1.0), "diff_loss": torch.tensor(1.0)},
        lambda p, t: {"total_loss": torch.ones(2), "mono_loss": torch.tensor(1.0), "diff_loss": torch.tensor(1.0)},
    ],
)
def test_malformed_or_nonfinite_losses_are_rejected_and_state_restored(bad_loss, tmp_path) -> None:
    model = TinyModel()
    model.train()
    model.condition_enabled = False
    with pytest.raises((TypeError, ValueError)):
        Evaluator(model, bad_loss, "cpu").evaluate([sample(0)], [0], "x", True, tmp_path)
    assert model.training is True
    assert model.condition_enabled is False


def test_nonfinite_metric_is_never_persisted(tmp_path) -> None:
    class ExactModel(TinyModel):
        def forward(self, item):
            output = super().forward(item)
            return replace(output, rgbd=replace(output.rgbd, rgb=item.target_rgb))

    with pytest.raises(ValueError, match="finite"):
        Evaluator(ExactModel(), audio_loss, "cpu").evaluate(
            [sample(0)], [0], "exact", True, tmp_path
        )
    assert not list(tmp_path.glob("metrics_*"))


def test_move_sample_moves_all_tensors_without_losing_metadata() -> None:
    original = sample(3)
    moved = move_sample(original, torch.device("meta"))
    assert all(value.device.type == "meta" for value in vars(moved).values() if isinstance(value, torch.Tensor))
    for name in ("scene_id", "camera", "frame_index", "time_seconds", "image_size"):
        assert getattr(moved, name) == getattr(original, name)


def test_failed_evaluation_preserves_existing_complete_metric_pair(tmp_path) -> None:
    rows_path = tmp_path / "metrics_per_sample.jsonl"
    summary_path = tmp_path / "metrics_summary.json"
    rows_path.write_text("old rows\n")
    summary_path.write_text("old summary\n")
    model = TinyModel()
    model.raise_error = True

    with pytest.raises(RuntimeError):
        Evaluator(model, audio_loss, "cpu").evaluate([sample(0)], [0], "x", True, tmp_path)

    assert rows_path.read_text() == "old rows\n"
    assert summary_path.read_text() == "old summary\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_failure_rolls_back_existing_metric_pair(monkeypatch, tmp_path) -> None:
    rows_path = tmp_path / "metrics_per_sample.jsonl"
    summary_path = tmp_path / "metrics_summary.json"
    rows_path.write_text("old rows\n")
    summary_path.write_text("old summary\n")
    real_replace = os.replace
    publish_calls = 0

    def fail_second_publish(source, destination):
        nonlocal publish_calls
        # Existing canonical paths must never be renamed out of the way while
        # preparing backups or installing either staged file.
        assert rows_path.exists()
        assert summary_path.exists()
        if str(source).endswith(".tmp"):
            publish_calls += 1
            if publish_calls == 2:
                raise OSError("forced replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="forced replacement"):
        Evaluator(TinyModel(), audio_loss, "cpu").evaluate(
            [sample(0)], [0], "x", True, tmp_path
        )

    assert rows_path.read_text() == "old rows\n"
    assert summary_path.read_text() == "old summary\n"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.backup"))

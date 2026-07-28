import math

import pytest
import torch

from avgaussianv2.losses import SpatialAudioLossWeights, spatial_audio_loss


def _stereo(
    *,
    left_gain: float = 1.0,
    right_gain: float = 0.5,
    right_phase: float = 0.4,
    samples: int = 2048,
) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float32) / 16_000
    left = left_gain * torch.sin(2 * math.pi * 440 * time)
    right = right_gain * torch.sin(2 * math.pi * 440 * time + right_phase)
    return torch.stack((left, right)).unsqueeze(0)


def test_spatial_audio_loss_is_zero_for_identical_audio() -> None:
    target = _stereo()

    result = spatial_audio_loss(target, target)

    assert result.keys() == {"total", "lre", "ild", "ipd", "diff"}
    for value in result.values():
        assert float(value) == pytest.approx(0.0, abs=2e-6)


def test_spatial_audio_loss_detects_energy_and_phase_errors() -> None:
    target = _stereo()
    wrong_energy = _stereo(left_gain=0.5, right_gain=1.0)
    wrong_phase = _stereo(right_phase=-0.8)

    energy = spatial_audio_loss(wrong_energy, target)
    phase = spatial_audio_loss(wrong_phase, target)

    assert energy["lre"] > 0
    assert energy["ild"] > 0
    assert phase["ipd"] > 0
    assert phase["diff"] > 0
    assert energy["total"] > 0
    assert phase["total"] > 0


def test_spatial_audio_loss_has_finite_low_energy_gradients() -> None:
    target = _stereo() * 1e-7
    predicted = (target * 0.8).detach().requires_grad_(True)

    result = spatial_audio_loss(predicted, target)
    result["total"].backward()

    assert torch.isfinite(result["total"])
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_spatial_audio_loss_weights_validate() -> None:
    with pytest.raises(ValueError, match="positive"):
        SpatialAudioLossWeights(0, 0, 0, 0).validate()
    with pytest.raises(ValueError, match="non-negative"):
        SpatialAudioLossWeights(lre=-1).validate()

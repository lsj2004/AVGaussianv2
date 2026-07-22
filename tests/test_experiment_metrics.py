import math

import pytest
import torch

from avgaussianv2.experiment.metrics import (
    aggregate_metrics,
    log_spectral_distance,
    lre_error_db,
    psnr,
    rgb_l1,
    ssim,
    waveform_l1,
)


def test_identical_audio_metrics_are_ideal() -> None:
    audio = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 640),
            torch.linspace(1.0, -1.0, 640),
        ),
        dim=0,
    ).unsqueeze(0)

    assert waveform_l1(audio, audio) == pytest.approx(0.0)
    assert lre_error_db(audio, audio) == pytest.approx(0.0)
    assert log_spectral_distance(audio, audio, "mono") == pytest.approx(0.0)
    assert log_spectral_distance(audio, audio, "diff") == pytest.approx(0.0)


def test_identical_bhwc_image_metrics_are_ideal() -> None:
    image = torch.rand(2, 12, 10, 3)

    assert rgb_l1(image, image) == pytest.approx(0.0)
    assert math.isinf(psnr(image, image)) and psnr(image, image) > 0
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_basic_l1_and_psnr_values() -> None:
    zeros = torch.zeros(1, 4, 5, 3)
    halves = torch.full_like(zeros, 0.5)

    assert rgb_l1(zeros, halves) == pytest.approx(0.5)
    assert waveform_l1(torch.zeros(1, 2, 8), torch.ones(1, 2, 8)) == 1.0
    assert psnr(zeros, halves) == pytest.approx(-10.0 * math.log10(0.25))


def test_lre_measures_left_right_energy_ratio_in_db() -> None:
    target = torch.ones(1, 2, 32)
    predicted = target.clone()
    predicted[:, 0] *= 2.0

    # Squaring a 2x amplitude produces a 4x energy ratio: 10 log10(4).
    assert lre_error_db(predicted, target) == pytest.approx(
        10.0 * math.log10(4.0), rel=1e-6
    )


def test_lsd_mono_and_diff_use_distinct_stereo_constructions() -> None:
    time = torch.arange(640, dtype=torch.float32)
    left = torch.sin(2.0 * math.pi * time / 40.0)
    right = torch.cos(2.0 * math.pi * time / 55.0)
    perturbation = 0.1 * torch.sin(2.0 * math.pi * time / 17.0)
    target = torch.stack((left, right), dim=0).unsqueeze(0)
    common_mode = target + torch.stack((perturbation, perturbation)).unsqueeze(0)
    anti_phase = target + torch.stack((perturbation, -perturbation)).unsqueeze(0)

    common_mono = log_spectral_distance(common_mode, target, "mono")
    common_diff = log_spectral_distance(common_mode, target, "diff")
    anti_mono = log_spectral_distance(anti_phase, target, "mono")
    anti_diff = log_spectral_distance(anti_phase, target, "diff")

    assert common_diff < 1e-3
    assert common_mono > 100.0 * common_diff
    assert anti_mono < 1e-3
    assert anti_diff > 100.0 * anti_mono


def test_invalid_lsd_component_and_stereo_shape_are_rejected() -> None:
    audio = torch.zeros(1, 2, 640)

    with pytest.raises(ValueError, match="component"):
        log_spectral_distance(audio, audio, "left")
    with pytest.raises(ValueError, match="stereo"):
        log_spectral_distance(torch.zeros(1, 1, 640), torch.zeros(1, 1, 640), "mono")


@pytest.mark.parametrize(
    "metric,args",
    [
        (waveform_l1, (torch.tensor([float("nan")]), torch.zeros(1))),
        (lre_error_db, (torch.full((1, 2, 8), float("inf")), torch.zeros(1, 2, 8))),
        (rgb_l1, (torch.full((1, 2, 2, 3), float("nan")), torch.zeros(1, 2, 2, 3))),
        (psnr, (torch.zeros(1, 2, 2, 3), torch.full((1, 2, 2, 3), float("inf")))),
        (ssim, (torch.zeros(1, 2, 2, 3), torch.full((1, 2, 2, 3), float("nan")))),
    ],
)
def test_nonfinite_tensor_input_is_rejected(metric, args) -> None:
    with pytest.raises(ValueError, match="finite"):
        metric(*args)


def test_aggregate_metrics_uses_population_statistics() -> None:
    result = aggregate_metrics([{"score": 1.0}, {"score": 3.0}])

    assert result == {"score": {"mean": 2.0, "std": 1.0, "median": 2.0}}


def test_aggregate_metrics_validates_rows() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_metrics([])
    with pytest.raises(ValueError, match="keys"):
        aggregate_metrics([{"a": 1.0}, {"b": 1.0}])
    with pytest.raises(ValueError, match="finite"):
        aggregate_metrics([{"a": 1.0}, {"a": float("nan")}])

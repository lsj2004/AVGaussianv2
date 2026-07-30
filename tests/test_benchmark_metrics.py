import math

import pytest
import torch

from avgaussianv2.benchmark.metrics import (
    aggregate_metrics,
    log_spectral_distance,
    lre_error_db,
    paper_envelope_distance,
    paper_lre_error_db,
    paper_magnitude_distance,
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


def test_psnr_promotes_cpu_bfloat16_for_mse() -> None:
    target = torch.zeros(1, 2, 2, 3, dtype=torch.bfloat16)
    predicted = torch.ones_like(target)

    assert psnr(predicted, target) == pytest.approx(0.0)


def test_psnr_preserves_small_nonzero_float16_error() -> None:
    target = torch.zeros(1, 2, 2, 3, dtype=torch.float16)
    predicted = torch.full_like(target, 1e-4)

    value = psnr(predicted, target)
    assert math.isfinite(value)
    assert value == pytest.approx(
        -10.0 * math.log10(float(predicted.flatten()[0]) ** 2),
        rel=1e-6,
    )


def test_lre_measures_left_right_energy_ratio_in_db() -> None:
    target = torch.ones(1, 2, 32)
    predicted = target.clone()
    predicted[:, 0] *= 2.0

    # Squaring a 2x amplitude produces a 4x energy ratio: 10 log10(4).
    assert lre_error_db(predicted, target) == pytest.approx(
        10.0 * math.log10(4.0), rel=1e-6
    )


def test_audio_paper_metrics_are_zero_for_identical_audio() -> None:
    audio = torch.randn(2, 2, 640)

    assert paper_magnitude_distance(audio, audio) == pytest.approx(0.0)
    assert paper_envelope_distance(audio, audio) == pytest.approx(0.0)
    assert paper_lre_error_db(audio, audio) == pytest.approx(0.0)


def test_paper_magnitude_distance_is_sum_of_per_ear_stft_l1() -> None:
    target = torch.zeros(1, 2, 32)
    predicted = target.clone()
    predicted[:, 0, 0] = 1.0
    window = torch.hamming_window(16)
    predicted_stft = torch.stft(
        predicted.flatten(0, 1),
        n_fft=16,
        hop_length=4,
        win_length=16,
        window=window,
        pad_mode="constant",
        return_complex=True,
    ).abs()
    target_stft = torch.stft(
        target.flatten(0, 1),
        n_fft=16,
        hop_length=4,
        win_length=16,
        window=window,
        pad_mode="constant",
        return_complex=True,
    ).abs()
    expected = (
        (predicted_stft - target_stft)
        .abs()
        .unflatten(0, (1, 2))
        .mean(dim=(-2, -1))
        .sum(dim=1)
        .mean()
    )

    assert paper_magnitude_distance(
        predicted,
        target,
        n_fft=16,
        hop_length=4,
        win_length=16,
    ) == pytest.approx(expected.item())


def test_paper_envelope_distance_matches_fft_analytic_signal_reference() -> None:
    target = torch.zeros(1, 2, 9)
    predicted = target.clone()
    predicted[0, 0] = torch.tensor([1.0, -0.5, 0.25, 0.0, 0.5, -1.0, 0.0, 0.2, -0.1])

    def reference_envelope(audio: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(audio, dim=-1)
        multiplier = torch.zeros(audio.shape[-1])
        multiplier[0] = 1
        multiplier[1 : (audio.shape[-1] + 1) // 2] = 2
        return torch.fft.ifft(spectrum * multiplier, dim=-1).abs()

    expected = (
        (reference_envelope(predicted) - reference_envelope(target))
        .square()
        .mean(dim=-1)
        .sqrt()
        .sum(dim=1)
        .mean()
    )

    assert paper_envelope_distance(predicted, target) == pytest.approx(
        expected.item()
    )


def test_paper_lre_uses_published_epsilon() -> None:
    target = torch.zeros(1, 2, 16)
    predicted = target.clone()
    predicted[:, 0] = 0.01
    expected = 10.0 * math.log10(
        (float(predicted[:, 0].square().sum()) + 1e-5) / 1e-5
    )

    assert paper_lre_error_db(predicted, target) == pytest.approx(expected)


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


@pytest.mark.parametrize("shape", [(2, 640), (1, 1, 640), (1, 2, 640, 1)])
@pytest.mark.parametrize(
    "metric",
    [
        waveform_l1,
        lre_error_db,
        lambda predicted, target: log_spectral_distance(predicted, target, "mono"),
    ],
)
def test_audio_metrics_require_batched_stereo_layout(metric, shape) -> None:
    audio = torch.zeros(shape)

    with pytest.raises(ValueError, match=r"\[B, 2, samples\]"):
        metric(audio, audio)


@pytest.mark.parametrize(
    "metric,inputs",
    [
        (waveform_l1, (torch.empty(0, 2, 8), torch.empty(0, 2, 8))),
        (lre_error_db, (torch.empty(1, 2, 0), torch.empty(1, 2, 0))),
        (rgb_l1, (torch.empty(0, 2, 2, 3), torch.empty(0, 2, 2, 3))),
    ],
)
def test_metrics_reject_empty_tensors(metric, inputs) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        metric(*inputs)


def test_metrics_reject_nonfloating_tensors() -> None:
    with pytest.raises(ValueError, match="floating"):
        waveform_l1(
            torch.zeros(1, 2, 8, dtype=torch.int64),
            torch.zeros(1, 2, 8, dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="floating"):
        rgb_l1(
            torch.zeros(1, 2, 2, 3, dtype=torch.int64),
            torch.zeros(1, 2, 2, 3, dtype=torch.int64),
        )


def test_rgb_metrics_require_matching_bhwc_rgb_layout() -> None:
    for metric in (rgb_l1, psnr, ssim):
        with pytest.raises(ValueError, match="BHWC"):
            metric(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4))
        with pytest.raises(ValueError, match="equal shapes"):
            metric(torch.zeros(1, 4, 4, 3), torch.zeros(1, 4, 5, 3))


def test_lsd_supports_short_cpu_float16_audio_via_constant_padding() -> None:
    audio = torch.rand(1, 2, 16, dtype=torch.float16)

    assert log_spectral_distance(audio, audio, "mono") == pytest.approx(0.0)


@pytest.mark.parametrize(
    "metric,args",
    [
        (waveform_l1, (torch.full((1, 2, 8), float("nan")), torch.zeros(1, 2, 8))),
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
    with pytest.raises(ValueError, match="scalar"):
        aggregate_metrics([{"a": [1.0, 2.0]}])

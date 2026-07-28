from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.contracts import AlignedAVSample, FusionOutput


AudioLoss = Callable[[Tensor, Tensor], Tensor | Mapping[str, Tensor]]


@dataclass(frozen=True)
class JointLossWeights:
    audio: float = 1.0
    rgb: float = 1.0
    dssim: float = 0.2
    visual_anchor: float = 1e-4


@dataclass(frozen=True)
class SpatialAudioLossWeights:
    """Weights for differentiable binaural geometry errors."""

    lre: float = 0.30
    ild: float = 0.25
    ipd: float = 0.25
    diff: float = 0.20

    def validate(self) -> None:
        values = (self.lre, self.ild, self.ipd, self.diff)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("spatial audio loss weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one spatial audio loss weight must be positive")


def _weighted_mean(value: Tensor, weight: Tensor, eps: float) -> Tensor:
    numerator = (value * weight).sum(dim=(-2, -1))
    denominator = weight.sum(dim=(-2, -1)).clamp_min(float(eps))
    return (numerator / denominator).mean()


def _charbonnier(value: Tensor, epsilon: float = 1e-3) -> Tensor:
    return (value.square() + float(epsilon) ** 2).sqrt() - float(epsilon)


def spatial_audio_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    weights: SpatialAudioLossWeights = SpatialAudioLossWeights(),
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    eps: float = 1e-7,
    energy_floor_ratio: float = 1e-3,
    lre_scale_db: float = 6.0,
) -> dict[str, Tensor]:
    """Return stable differentiable LRE/ILD/IPD/binaural-difference errors.

    STFT terms are weighted by target energy.  IPD additionally requires energy
    in both target channels, so silent or single-channel bins cannot dominate
    the phase gradient.
    """
    if (
        predicted.shape != target.shape
        or predicted.ndim != 3
        or predicted.shape[1] != 2
        or predicted.shape[-1] == 0
    ):
        raise ValueError(
            "spatial audio loss requires equal nonempty stereo tensors [B,2,N]"
        )
    if predicted.device != target.device:
        raise ValueError("spatial audio loss inputs must share a device")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise ValueError("spatial audio loss inputs must be floating point")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("spatial audio loss inputs must be finite")
    if (
        n_fft <= 0
        or hop_length <= 0
        or win_length <= 0
        or win_length > n_fft
    ):
        raise ValueError("invalid spatial audio STFT configuration")
    if (
        not math.isfinite(eps)
        or eps <= 0
        or not math.isfinite(energy_floor_ratio)
        or energy_floor_ratio <= 0
        or not math.isfinite(lre_scale_db)
        or lre_scale_db <= 0
    ):
        raise ValueError("spatial audio stability constants must be positive")
    weights.validate()

    reduction_dtype = torch.promote_types(predicted.dtype, target.dtype)
    if reduction_dtype in {torch.float16, torch.bfloat16}:
        reduction_dtype = torch.float32
    predicted_audio = predicted.to(reduction_dtype)
    target_audio = target.to(reduction_dtype)

    def lre_db(audio: Tensor) -> Tensor:
        channel_energy = audio.square().sum(dim=-1)
        return 10.0 * (
            torch.log(channel_energy[:, 0] + float(eps))
            - torch.log(channel_energy[:, 1] + float(eps))
        ) / math.log(10.0)

    lre = _charbonnier(
        (lre_db(predicted_audio) - lre_db(target_audio)) / float(lre_scale_db)
    ).mean()

    window = torch.hamming_window(
        win_length,
        device=predicted.device,
        dtype=reduction_dtype,
    )

    def stft(audio: Tensor) -> Tensor:
        batch, channels, samples = audio.shape
        flat = audio.reshape(batch * channels, samples)
        spectrum = torch.stft(
            flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="constant",
        )
        return spectrum.reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])

    predicted_stft = stft(predicted_audio)
    target_stft = stft(target_audio)
    predicted_magnitude = predicted_stft.abs()
    target_magnitude = target_stft.abs()
    target_energy = target_magnitude.square().sum(dim=1)
    maximum_energy = target_energy.amax(dim=(-2, -1), keepdim=True)
    energy_floor = maximum_energy * float(energy_floor_ratio) + float(eps)
    energy_weight = target_energy / (target_energy + energy_floor)

    predicted_ild = torch.log(predicted_magnitude[:, 0] + float(eps)) - torch.log(
        predicted_magnitude[:, 1] + float(eps)
    )
    target_ild = torch.log(target_magnitude[:, 0] + float(eps)) - torch.log(
        target_magnitude[:, 1] + float(eps)
    )
    ild = _weighted_mean(
        _charbonnier((predicted_ild - target_ild) / math.log(2.0)),
        energy_weight,
        eps,
    )

    def interaural_phase(spectrum: Tensor, magnitude: Tensor) -> Tensor:
        left_unit = spectrum[:, 0] / magnitude[:, 0].clamp_min(float(eps))
        right_unit = spectrum[:, 1] / magnitude[:, 1].clamp_min(float(eps))
        return left_unit * right_unit.conj()

    predicted_ipd = interaural_phase(predicted_stft, predicted_magnitude)
    target_ipd = interaural_phase(target_stft, target_magnitude)
    bilateral_weight = (
        2.0
        * target_magnitude[:, 0]
        * target_magnitude[:, 1]
        / target_energy.clamp_min(float(eps))
    )
    ipd_weight = energy_weight * bilateral_weight
    ipd = _weighted_mean(
        (1.0 - (predicted_ipd * target_ipd.conj()).real).clamp_min(0.0),
        ipd_weight,
        eps,
    )

    predicted_diff = torch.log1p((predicted_stft[:, 0] - predicted_stft[:, 1]).abs())
    target_diff = torch.log1p((target_stft[:, 0] - target_stft[:, 1]).abs())
    diff = _weighted_mean(
        _charbonnier(predicted_diff - target_diff),
        energy_weight,
        eps,
    )

    total_weight = weights.lre + weights.ild + weights.ipd + weights.diff
    total = (
        weights.lre * lre
        + weights.ild * ild
        + weights.ipd * ipd
        + weights.diff * diff
    ) / total_weight
    return {
        "total": total,
        "lre": lre,
        "ild": ild,
        "ipd": ipd,
        "diff": diff,
    }


def capture_visual_anchor(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def dssim(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("DSSIM inputs must have equal BHWC shapes")
    dimensions = (1, 2)
    mean_predicted = predicted.mean(dim=dimensions, keepdim=True)
    mean_target = target.mean(dim=dimensions, keepdim=True)
    centered_predicted = predicted - mean_predicted
    centered_target = target - mean_target
    variance_predicted = centered_predicted.square().mean(dim=dimensions, keepdim=True)
    variance_target = centered_target.square().mean(dim=dimensions, keepdim=True)
    covariance = (centered_predicted * centered_target).mean(dim=dimensions, keepdim=True)
    c1 = predicted.new_tensor(0.01**2)
    c2 = predicted.new_tensor(0.03**2)
    luminance = (2 * mean_predicted * mean_target + c1) / (
        mean_predicted.square() + mean_target.square() + c1
    )
    contrast = (2 * covariance + c2) / (variance_predicted + variance_target + c2)
    return 0.5 * (1.0 - (luminance * contrast).mean())


def _audio_total(loss: Tensor | Mapping[str, Tensor]) -> Tensor:
    if isinstance(loss, Tensor):
        return loss
    if "total_loss" not in loss:
        raise ValueError("AudioGS loss mapping is missing total_loss")
    return loss["total_loss"]


def _visual_anchor_loss(
    module: nn.Module,
    anchor: Mapping[str, Tensor],
) -> Tensor:
    current = dict(module.named_parameters())
    if current.keys() != anchor.keys():
        raise ValueError("visual anchor parameter names do not match visual module")
    losses = [
        (parameter - anchor[name].to(parameter)).square().mean()
        for name, parameter in current.items()
    ]
    if not losses:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def compute_joint_loss(
    output: FusionOutput,
    sample: AlignedAVSample,
    visual_module: nn.Module,
    visual_anchor: Mapping[str, Tensor],
    weights: JointLossWeights,
    audio_loss_fn: AudioLoss,
) -> tuple[Tensor, dict[str, Tensor]]:
    audio = _audio_total(audio_loss_fn(output.predicted_audio, sample.target_audio))
    rgb_l1 = F.l1_loss(output.rgbd.rgb, sample.target_rgb.to(output.rgbd.rgb))
    rgb = rgb_l1 + float(weights.dssim) * dssim(
        output.rgbd.rgb,
        sample.target_rgb.to(output.rgbd.rgb),
    )
    anchor = _visual_anchor_loss(visual_module, visual_anchor).to(audio)
    total = (
        float(weights.audio) * audio
        + float(weights.rgb) * rgb
        + float(weights.visual_anchor) * anchor
    )
    return total, {
        "audio": audio,
        "rgb": rgb,
        "rgb_l1": rgb_l1,
        "visual_anchor": anchor,
    }

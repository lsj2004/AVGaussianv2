from __future__ import annotations

from dataclasses import dataclass
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

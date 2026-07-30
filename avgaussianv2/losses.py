from __future__ import annotations

import math
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
    lre: float = 0.0
    lre_scale_db: float = 6.0
    lre_epsilon: float = 1e-8
    lre_smooth_l1_beta: float = 1.0
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


def signed_lre_db(audio: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Return signed left/right energy ratio in dB for ``[B, 2, samples]``."""
    if (
        audio.ndim != 3
        or audio.shape[0] == 0
        or audio.shape[1] != 2
        or audio.shape[-1] == 0
    ):
        raise ValueError("LRE audio must have shape [B,2,samples] and be nonempty")
    if not audio.is_floating_point():
        raise TypeError("LRE audio must be floating point")
    if not torch.isfinite(audio).all():
        raise ValueError("LRE audio must be finite")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("LRE epsilon must be finite and positive")
    reduction_dtype = (
        torch.float32
        if audio.dtype in {torch.float16, torch.bfloat16}
        else audio.dtype
    )
    energy = audio.to(reduction_dtype).square().sum(dim=-1)
    return 10.0 * torch.log10(
        (energy[:, 0] + float(eps)) / (energy[:, 1] + float(eps))
    )


def signed_lre_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    scale_db: float = 6.0,
    eps: float = 1e-8,
    beta: float = 1.0,
) -> Tensor:
    """Smooth-L1 loss on signed left/right energy ratios."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target audio must have equal shapes")
    if predicted.device != target.device:
        raise ValueError("predicted and target audio must share a device")
    for name, value in (
        ("scale_db", scale_db),
        ("beta", beta),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"LRE {name} must be finite and positive")
    predicted_lre = signed_lre_db(predicted, eps=eps) / float(scale_db)
    target_lre = signed_lre_db(target.to(predicted), eps=eps) / float(scale_db)
    return F.smooth_l1_loss(predicted_lre, target_lre, beta=float(beta))


def _audio_total(loss: Tensor | Mapping[str, Tensor]) -> Tensor:
    if isinstance(loss, Mapping):
        if "total_loss" not in loss:
            raise ValueError("AudioGS loss mapping is missing total_loss")
        loss = loss["total_loss"]
    if not isinstance(loss, Tensor):
        raise TypeError("AudioGS loss must resolve to a Tensor")
    if loss.ndim != 0:
        raise ValueError("AudioGS loss must resolve to a scalar Tensor")
    if not torch.isfinite(loss):
        raise ValueError("AudioGS loss must be finite")
    return loss


def compute_audio_objective(
    predicted: Tensor,
    target: Tensor,
    *,
    weights: JointLossWeights,
    audio_loss_fn: AudioLoss,
) -> tuple[Tensor, dict[str, Tensor]]:
    audio_base = _audio_total(audio_loss_fn(predicted, target))
    audio_lre = signed_lre_loss(
        predicted,
        target.to(predicted),
        scale_db=weights.lre_scale_db,
        eps=weights.lre_epsilon,
        beta=weights.lre_smooth_l1_beta,
    ).to(audio_base)
    weighted_lre = float(weights.lre) * audio_lre
    objective = float(weights.audio) * audio_base + weighted_lre
    predicted_lre_db = signed_lre_db(
        predicted,
        eps=weights.lre_epsilon,
    ).mean().to(audio_base)
    target_lre_db = signed_lre_db(
        target.to(predicted),
        eps=weights.lre_epsilon,
    ).mean().to(audio_base)
    return objective, {
        "audio": audio_base,
        "audio_base": audio_base,
        "audio_lre": audio_lre,
        "audio_lre_weighted": weighted_lre,
        "audio_total_with_lre": objective,
        "pred_lre_db": predicted_lre_db,
        "target_lre_db": target_lre_db,
    }


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
    audio_objective, audio_parts = compute_audio_objective(
        output.predicted_audio,
        sample.target_audio,
        weights=weights,
        audio_loss_fn=audio_loss_fn,
    )
    rgb_l1 = F.l1_loss(output.rgbd.rgb, sample.target_rgb.to(output.rgbd.rgb))
    rgb = rgb_l1 + float(weights.dssim) * dssim(
        output.rgbd.rgb,
        sample.target_rgb.to(output.rgbd.rgb),
    )
    anchor = _visual_anchor_loss(visual_module, visual_anchor).to(audio_objective)
    total = (
        audio_objective
        + float(weights.rgb) * rgb
        + float(weights.visual_anchor) * anchor
    )
    return total, {
        **audio_parts,
        "rgb": rgb,
        "rgb_l1": rgb_l1,
        "visual_anchor": anchor,
    }

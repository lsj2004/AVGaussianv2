from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from avgaussianv2.config import TrainConfig
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.losses import (
    AudioLoss,
    JointLossWeights,
    capture_visual_anchor,
    compute_joint_loss,
)


class NonFiniteTrainingError(RuntimeError):
    """Raised before an optimizer can be contaminated by NaN or infinity."""


class DisconnectedAudioVisualGradient(RuntimeError):
    """Raised when the joint audio condition path does not reach the visual field."""


@dataclass(frozen=True)
class TrainStepStats:
    total: float
    losses: dict[str, float]
    gradient_norms: dict[str, float]
    audio_to_visual_grad_norm: float


def _sample_identity(sample: AlignedAVSample) -> str:
    return (
        f"scene={sample.scene_id} camera={sample.camera} "
        f"frame={sample.frame_index} time={sample.time_seconds:.6f}"
    )


def _require_finite_tensor(name: str, value: Tensor, sample: AlignedAVSample) -> None:
    if not torch.isfinite(value).all():
        raise NonFiniteTrainingError(f"non-finite {name} at {_sample_identity(sample)}")


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _explicit_gradient_norm(gradients: Sequence[Tensor | None]) -> float:
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            total += float(gradient.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _weights(config: TrainConfig) -> JointLossWeights:
    return JointLossWeights(
        audio=config.lambda_audio,
        rgb=config.lambda_rgb,
        dssim=config.lambda_dssim,
        visual_anchor=config.lambda_visual_anchor,
    )


def joint_train_step(
    model: nn.Module,
    sample: AlignedAVSample,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    audio_loss_fn: AudioLoss,
    visual_anchor: Mapping[str, Tensor],
) -> TrainStepStats:
    model.unfreeze_all()
    optimizer.zero_grad(set_to_none=True)
    output = model(sample)
    _require_finite_tensor("predicted audio", output.predicted_audio, sample)
    _require_finite_tensor("rendered RGB", output.rgbd.rgb, sample)
    _require_finite_tensor("rendered depth", output.rgbd.depth, sample)
    _require_finite_tensor("condition", output.condition, sample)
    total, parts = compute_joint_loss(
        output,
        sample,
        visual_module=model.visual,
        visual_anchor=visual_anchor,
        weights=_weights(config),
        audio_loss_fn=audio_loss_fn,
    )
    _require_finite_tensor("total loss", total, sample)
    groups = model.named_parameter_groups()
    visual_parameters = [parameter for parameter in groups["visual"] if parameter.requires_grad]
    audio_visual_gradients = torch.autograd.grad(
        parts["audio"],
        visual_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    audio_to_visual = _explicit_gradient_norm(audio_visual_gradients)
    total.backward()
    gradient_norms = {
        name: _gradient_norm(parameters)
        for name, parameters in groups.items()
    }
    if not all(math.isfinite(value) for value in gradient_norms.values()):
        raise NonFiniteTrainingError(f"non-finite gradients at {_sample_identity(sample)}")
    optimizer.step()
    return TrainStepStats(
        total=float(total.detach().cpu()),
        losses={name: float(value.detach().cpu()) for name, value in parts.items()},
        gradient_norms=gradient_norms,
        audio_to_visual_grad_norm=audio_to_visual,
    )


def run_condition_warmup(
    model: nn.Module,
    samples: Sequence[AlignedAVSample],
    steps: int,
    learning_rate: float,
    config: TrainConfig,
    audio_loss_fn: AudioLoss,
) -> list[TrainStepStats]:
    del config
    if steps <= 0:
        return []
    if not samples:
        raise ValueError("warmup samples must not be empty")
    model.freeze_pretrained()
    groups = model.named_parameter_groups()
    trainable = [*groups["condition_encoder"], *groups["film"]]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history = []
    for step in range(steps):
        sample = samples[step % len(samples)]
        optimizer.zero_grad(set_to_none=True)
        output = model(sample)
        _require_finite_tensor("predicted audio", output.predicted_audio, sample)
        audio = audio_loss_fn(output.predicted_audio, sample.target_audio)
        if isinstance(audio, Mapping):
            audio = audio["total_loss"]
        _require_finite_tensor("warmup audio loss", audio, sample)
        audio.backward()
        gradient_norms = {
            name: _gradient_norm(parameters)
            for name, parameters in groups.items()
        }
        optimizer.step()
        history.append(
            TrainStepStats(
                total=float(audio.detach().cpu()),
                losses={"audio": float(audio.detach().cpu())},
                gradient_norms=gradient_norms,
                audio_to_visual_grad_norm=0.0,
            )
        )
    return history


def run_joint_finetune(
    model: nn.Module,
    samples: Sequence[AlignedAVSample],
    steps: int,
    config: TrainConfig,
    audio_loss_fn: AudioLoss,
    require_audio_visual_gradient: bool = True,
) -> list[TrainStepStats]:
    if steps <= 0:
        return []
    if not samples:
        raise ValueError("joint samples must not be empty")
    model.unfreeze_all()
    groups = model.named_parameter_groups()
    optimizer = torch.optim.Adam(
        [
            {"params": groups["visual"], "lr": config.visual_lr},
            {"params": groups["acoustic"], "lr": config.audio_lr},
            {"params": groups["audio_unet"], "lr": config.audio_lr},
            {"params": groups["condition_encoder"], "lr": config.condition_lr},
            {"params": groups["film"], "lr": config.condition_lr},
        ]
    )
    anchor = capture_visual_anchor(model.visual)
    history = []
    consecutive_zero = 0
    for step in range(steps):
        stats = joint_train_step(
            model,
            samples[step % len(samples)],
            optimizer,
            config,
            audio_loss_fn,
            anchor,
        )
        if require_audio_visual_gradient and (step + 1) % config.gradient_probe_interval == 0:
            consecutive_zero = consecutive_zero + 1 if stats.audio_to_visual_grad_norm == 0 else 0
            if consecutive_zero >= config.max_zero_audio_visual_grad_steps:
                raise DisconnectedAudioVisualGradient(
                    "audio loss did not reach visual parameters for "
                    f"{consecutive_zero} consecutive probes"
                )
        history.append(stats)
    return history

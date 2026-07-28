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


def _condition_tensor(condition) -> Tensor:
    if isinstance(condition, Tensor):
        return condition
    tokens = getattr(condition, "tokens", None)
    if isinstance(tokens, Tensor):
        return tokens
    raise TypeError("condition must be a tensor or expose tensor tokens")


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


def _trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def build_warmup_optimizer(
    model: nn.Module, learning_rate: float
) -> torch.optim.Optimizer:
    """Build the single optimizer used throughout condition warmup."""
    groups = model.named_parameter_groups()
    parameters = _trainable(
        [*groups["condition_encoder"], *groups["film"]]
    )
    if not parameters:
        raise ValueError("warmup has no trainable condition encoder or FiLM parameters")
    return torch.optim.Adam(parameters, lr=learning_rate)


def build_joint_optimizer(
    model: nn.Module, config: TrainConfig
) -> torch.optim.Optimizer:
    """Build a joint optimizer without passing empty or frozen groups to torch."""
    groups = model.named_parameter_groups()
    learning_rates = {
        "visual": config.visual_lr,
        "acoustic": config.audio_lr,
        "audio_unet": config.audio_lr,
        "condition_encoder": config.condition_lr,
        "film": config.condition_lr,
    }
    optimizer_groups = []
    for name, learning_rate in learning_rates.items():
        parameters = _trainable(groups[name])
        if parameters:
            optimizer_groups.append(
                {"params": parameters, "lr": learning_rate, "name": name}
            )
    if not optimizer_groups:
        raise ValueError("joint training has no trainable parameters")
    return torch.optim.Adam(optimizer_groups)


def _audio_loss_tensor(
    criterion_result: Tensor | Mapping[str, Tensor], sample: AlignedAVSample
) -> Tensor:
    if isinstance(criterion_result, Mapping):
        if "total_loss" not in criterion_result:
            raise KeyError("audio loss mapping is missing 'total_loss'")
        loss = criterion_result["total_loss"]
    else:
        loss = criterion_result
    if not isinstance(loss, Tensor):
        raise TypeError("audio loss must be a scalar tensor or mapping containing one")
    if loss.ndim != 0:
        raise ValueError("audio loss must be a scalar tensor")
    _require_finite_tensor("warmup audio loss", loss, sample)
    return loss


def same_frame_camera_negative_indices(
    samples: Sequence[AlignedAVSample],
    anchor_indices: Sequence[int],
    seed: int,
) -> list[int]:
    """Choose deterministic same-frame samples from another training camera."""
    records = getattr(samples, "records", None)
    identities = (
        [
            (int(record.frame_index), str(record.camera))
            for record in records
        ]
        if records is not None
        else [
            (int(samples[index].frame_index), str(samples[index].camera))
            for index in range(len(samples))
        ]
    )
    by_frame: dict[int, list[int]] = {}
    for index, (frame_index, _) in enumerate(identities):
        by_frame.setdefault(frame_index, []).append(index)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    negatives = []
    for anchor_index in anchor_indices:
        frame_index, camera = identities[anchor_index]
        candidates = [
            index
            for index in by_frame.get(frame_index, ())
            if identities[index][1] != camera
        ]
        if not candidates:
            raise ValueError(
                "camera contrast requires another camera at the same frame"
            )
        choice = int(
            torch.randint(len(candidates), (1,), generator=generator)
        )
        negatives.append(candidates[choice])
    return negatives


def _camera_contrast(
    model: nn.Module,
    sample: AlignedAVSample,
    contrast_sample: AlignedAVSample | None,
    correct_audio_loss: Tensor,
    audio_loss_fn: AudioLoss,
) -> tuple[Tensor, dict[str, Tensor]]:
    zero = correct_audio_loss.new_zeros(())
    weight = float(getattr(model, "camera_contrast_weight", 0.0))
    margin = float(getattr(model, "camera_contrast_margin", 0.0))
    if weight == 0:
        return zero, {}
    if contrast_sample is None:
        raise ValueError(
            "camera-contrast model requires a same-frame negative sample"
        )
    wrong = model.forward_with_condition_sample(sample, contrast_sample)
    _require_finite_tensor(
        "wrong-camera predicted audio",
        wrong.predicted_audio,
        sample,
    )
    wrong_loss = _audio_loss_tensor(
        audio_loss_fn(wrong.predicted_audio, sample.target_audio),
        sample,
    )
    observed = wrong_loss - correct_audio_loss
    contrast = torch.relu(correct_audio_loss.new_tensor(margin) - observed)
    return correct_audio_loss.new_tensor(weight) * contrast, {
        "camera_contrast": contrast,
        "camera_contrast_weight": correct_audio_loss.new_tensor(weight),
        "camera_contrast_margin_observed": observed,
    }


def condition_warmup_step(
    model: nn.Module,
    sample: AlignedAVSample,
    optimizer: torch.optim.Optimizer,
    audio_loss_fn: AudioLoss,
    contrast_sample: AlignedAVSample | None = None,
) -> TrainStepStats:
    optimizer.zero_grad(set_to_none=True)
    output = model(sample)
    _require_finite_tensor("predicted audio", output.predicted_audio, sample)
    audio = _audio_loss_tensor(
        audio_loss_fn(output.predicted_audio, sample.target_audio), sample
    )
    contrast, contrast_parts = _camera_contrast(
        model,
        sample,
        contrast_sample,
        audio,
        audio_loss_fn,
    )
    total = audio + contrast
    total.backward()
    groups = model.named_parameter_groups()
    gradient_norms = {
        name: _gradient_norm(parameters) for name, parameters in groups.items()
    }
    if not all(math.isfinite(value) for value in gradient_norms.values()):
        raise NonFiniteTrainingError(f"non-finite gradients at {_sample_identity(sample)}")
    optimizer.step()
    value = float(total.detach().cpu())
    return TrainStepStats(
        total=value,
        losses={
            "audio": float(audio.detach().cpu()),
            **{
                name: float(item.detach().cpu())
                for name, item in contrast_parts.items()
            },
        },
        gradient_norms=gradient_norms,
        audio_to_visual_grad_norm=0.0,
    )


def joint_train_step(
    model: nn.Module,
    sample: AlignedAVSample,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    audio_loss_fn: AudioLoss,
    visual_anchor: Mapping[str, Tensor],
    probe_audio_visual_gradient: bool = True,
    contrast_sample: AlignedAVSample | None = None,
) -> TrainStepStats:
    optimizer.zero_grad(set_to_none=True)
    output = model(sample)
    _require_finite_tensor("predicted audio", output.predicted_audio, sample)
    _require_finite_tensor("rendered RGB", output.rgbd.rgb, sample)
    _require_finite_tensor("rendered depth", output.rgbd.depth, sample)
    _require_finite_tensor(
        "condition",
        _condition_tensor(output.condition),
        sample,
    )
    total, parts = compute_joint_loss(
        output,
        sample,
        visual_module=model.visual,
        visual_anchor=visual_anchor,
        weights=_weights(config),
        audio_loss_fn=audio_loss_fn,
    )
    contrast, contrast_parts = _camera_contrast(
        model,
        sample,
        contrast_sample,
        parts["audio"],
        audio_loss_fn,
    )
    total = total + contrast
    parts = {**parts, **contrast_parts}
    _require_finite_tensor("total loss", total, sample)
    groups = model.named_parameter_groups()
    audio_to_visual = 0.0
    if probe_audio_visual_gradient:
        visual_parameters = _trainable(groups["visual"])
        if not visual_parameters:
            raise ValueError("audio-to-visual gradient probe has no trainable visual parameters")
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
    optimizer = build_warmup_optimizer(model, learning_rate)
    history = []
    for step in range(steps):
        sample = samples[step % len(samples)]
        history.append(condition_warmup_step(model, sample, optimizer, audio_loss_fn))
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
    optimizer = build_joint_optimizer(model, config)
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
            probe_audio_visual_gradient=require_audio_visual_gradient,
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

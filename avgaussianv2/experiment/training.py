"""Variant-aware training loop for the bounded Scene 1 pilot."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from avgaussianv2.config import TrainConfig
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.experiment.contracts import (
    EvaluationResult,
    PilotConfig,
    Variant,
    VariantIndices,
)
from avgaussianv2.experiment.selection import BestSelector, EarlyStopper
from avgaussianv2.losses import AudioLoss, capture_visual_anchor
from avgaussianv2.train import (
    DisconnectedAudioVisualGradient,
    NonFiniteTrainingError,
    TrainStepStats,
    build_joint_optimizer,
    build_warmup_optimizer,
    condition_warmup_step,
    joint_train_step,
)


@dataclass(frozen=True)
class PilotTrainingResult:
    variant: Variant
    completed_warmup_steps: int
    completed_joint_steps: int
    best_step: int | None
    stop_reason: str
    validation_history: tuple[dict[str, object], ...]
    training_history: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ValidationEvent:
    variant: Variant
    step: int
    summary: dict[str, object]
    evaluation: EvaluationResult
    completed_warmup_steps: int
    completed_joint_steps: int
    latest_training_row: dict[str, object]
    selector_state: dict[str, object]
    stopper_state: dict[str, object]
    is_best: bool
    should_stop: bool
    optimizer: Any = field(repr=False, compare=False)


ValidationCallback = Callable[[ValidationEvent], None]


def _set_enabled(parameters: Sequence[nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def configure_variant(model: nn.Module, variant: Variant, stage: str) -> None:
    """Apply the exact trainability and conditioning policy for one pilot stage."""
    try:
        resolved_variant = Variant(variant)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported pilot variant: {variant!r}") from error
    if stage not in {"warmup", "joint"}:
        raise ValueError("stage must be 'warmup' or 'joint'")
    if not hasattr(model, "condition_enabled"):
        raise TypeError("pilot model must expose condition_enabled")

    model.unfreeze_all()
    model.condition_enabled = resolved_variant != Variant.CONDITION_OFF
    if stage == "warmup":
        if resolved_variant == Variant.CONDITION_OFF:
            raise ValueError("condition_off does not support condition warmup")
        model.freeze_pretrained()
        return

    groups = model.named_parameter_groups()
    if resolved_variant == Variant.FROZEN_VISUAL:
        _set_enabled(groups["visual"], False)
    elif resolved_variant == Variant.CONDITION_OFF:
        for name in ("visual", "condition_encoder", "film"):
            _set_enabled(groups[name], False)


def _stats_row(stage: str, step: int, index: int, stats: TrainStepStats) -> dict[str, object]:
    values = [stats.total, stats.audio_to_visual_grad_norm]
    values.extend(stats.losses.values())
    values.extend(stats.gradient_norms.values())
    if not all(math.isfinite(float(value)) for value in values):
        raise NonFiniteTrainingError(f"non-finite {stage} statistics at step {step}")
    return {
        "stage": stage,
        "step": step,
        "sample_index": index,
        "total": stats.total,
        "audio_to_visual_grad_norm": stats.audio_to_visual_grad_norm,
        "losses": dict(stats.losses),
        "gradient_norms": dict(stats.gradient_norms),
    }


def _strict_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _curve_text(history: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "stage",
            "step",
            "sample_index",
            "total",
            "audio_to_visual_grad_norm",
            "losses",
            "gradient_norms",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in history:
        writer.writerow(
            {
                **row,
                "losses": _strict_json(row["losses"]),
                "gradient_norms": _strict_json(row["gradient_norms"]),
            }
        )
    return stream.getvalue()


class PilotTrainer:
    """Execute one variant using shared indices and periodic held-out validation."""

    def __init__(
        self,
        config: PilotConfig,
        evaluator: Any,
        *,
        train_config: TrainConfig | None = None,
        warmup_optimizer_factory: Callable[..., Any] = build_warmup_optimizer,
        joint_optimizer_factory: Callable[..., Any] = build_joint_optimizer,
        warmup_step_fn: Callable[..., TrainStepStats] = condition_warmup_step,
        joint_step_fn: Callable[..., TrainStepStats] = joint_train_step,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.train_config = TrainConfig() if train_config is None else train_config
        self.warmup_optimizer_factory = warmup_optimizer_factory
        self.joint_optimizer_factory = joint_optimizer_factory
        self.warmup_step_fn = warmup_step_fn
        self.joint_step_fn = joint_step_fn

    def run(
        self,
        *,
        model: nn.Module,
        train_samples: Sequence[AlignedAVSample],
        heldout_samples: Sequence[AlignedAVSample],
        indices: VariantIndices,
        heldout_indices: Sequence[int],
        variant: Variant,
        visual_baseline: object,
        audio_loss_fn: AudioLoss,
        output_dir: str | Path,
        on_validation: ValidationCallback | None = None,
        on_best_candidate: ValidationCallback | None = None,
    ) -> PilotTrainingResult:
        if hasattr(self.evaluator, "model") and self.evaluator.model is not model:
            raise ValueError("evaluator must be bound to the same model passed to PilotTrainer.run")
        self.config.validate()
        self.train_config.validate()
        resolved_variant = Variant(variant)
        expected_warmup = 0 if resolved_variant == Variant.CONDITION_OFF else self.config.warmup_steps
        if (
            resolved_variant != Variant.CONDITION_OFF
            and len(indices.warmup) != expected_warmup
        ):
            raise ValueError(
                f"variant indices contain {len(indices.warmup)} warmup steps; expected {expected_warmup}"
            )
        if len(indices.joint) != self.config.joint_steps:
            raise ValueError(
                f"variant indices contain {len(indices.joint)} joint steps; expected {self.config.joint_steps}"
            )
        if not train_samples:
            raise ValueError("training samples must not be empty")
        if not heldout_samples or not heldout_indices:
            raise ValueError("quick validation requires held-out samples and indices")
        for index in (*indices.warmup, *indices.joint):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("training indices must be integers")
            if not 0 <= index < len(train_samples):
                raise ValueError(f"training index {index} is out of range")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        # A failed replacement run must not leave an older success declaration.
        (output / "worker_summary.json").unlink(missing_ok=True)
        (output / "training_curve.csv").unlink(missing_ok=True)

        selector = BestSelector(
            visual_baseline,
            self.config.psnr_tolerance_db,
            self.config.ssim_tolerance,
        )
        stopper = EarlyStopper(
            self.config.minimum_joint_steps,
            self.config.patience,
            self.config.minimum_relative_improvement,
        )
        history: list[dict[str, object]] = []
        validation_history: list[dict[str, object]] = []

        if expected_warmup:
            configure_variant(model, resolved_variant, "warmup")
            optimizer = self.warmup_optimizer_factory(
                model, self.train_config.condition_lr
            )
            for step, index in enumerate(indices.warmup, start=1):
                stats = self.warmup_step_fn(
                    model, train_samples[index], optimizer, audio_loss_fn
                )
                history.append(_stats_row("warmup", step, index, stats))

        configure_variant(model, resolved_variant, "joint")
        joint_optimizer = self.joint_optimizer_factory(model, self.train_config)
        visual_anchor = capture_visual_anchor(model.visual)
        positive_probe_seen = False
        stop_reason = "max_steps"
        completed_joint_steps = 0
        for step, index in enumerate(indices.joint, start=1):
            should_probe = resolved_variant == Variant.JOINT_CONDITIONED
            stats = self.joint_step_fn(
                model,
                train_samples[index],
                joint_optimizer,
                self.train_config,
                audio_loss_fn,
                visual_anchor,
                probe_audio_visual_gradient=should_probe,
            )
            row = _stats_row("joint", step, index, stats)
            history.append(row)
            completed_joint_steps = step
            if should_probe and stats.audio_to_visual_grad_norm > 0:
                positive_probe_seen = True

            validate_now = (
                step % self.config.validation_interval == 0
                or step == len(indices.joint)
            )
            if not validate_now:
                continue
            validation_dir = output / "validation" / f"step_{step:06d}"
            validation_dir.mkdir(parents=True, exist_ok=True)
            evaluation = self.evaluator.evaluate(
                heldout_samples,
                heldout_indices,
                system_name=f"{resolved_variant.value}_step_{step:06d}",
                condition_enabled=bool(model.condition_enabled),
                output_dir=validation_dir,
            )
            summary = json.loads(_strict_json(evaluation.summary))
            selected = selector.consider(step, summary)
            audio_total = summary["audio_total"]["mean"]
            should_stop = stopper.update(step, audio_total)
            validation_row = {"step": step, "summary": summary}
            validation_history.append(validation_row)
            event = ValidationEvent(
                variant=resolved_variant,
                step=step,
                summary=summary,
                evaluation=evaluation,
                completed_warmup_steps=expected_warmup,
                completed_joint_steps=step,
                latest_training_row=dict(row),
                selector_state=selector.state_dict(),
                stopper_state=stopper.state_dict(),
                is_best=selected,
                should_stop=should_stop,
                optimizer=joint_optimizer,
            )
            if on_validation is not None:
                on_validation(event)
            if selected and on_best_candidate is not None:
                on_best_candidate(event)
            if should_stop:
                stop_reason = "early_stop"
                break

        if resolved_variant == Variant.JOINT_CONDITIONED and not positive_probe_seen:
            raise DisconnectedAudioVisualGradient(
                "joint_conditioned pilot completed without a finite positive "
                "audio-to-visual gradient probe"
            )

        result = PilotTrainingResult(
            variant=resolved_variant,
            completed_warmup_steps=expected_warmup,
            completed_joint_steps=completed_joint_steps,
            best_step=selector.best_step,
            stop_reason=stop_reason,
            validation_history=tuple(validation_history),
            training_history=tuple(history),
        )
        summary_payload = {
            "variant": resolved_variant.value,
            "completed_warmup_steps": result.completed_warmup_steps,
            "completed_joint_steps": result.completed_joint_steps,
            "best_step": result.best_step,
            "stop_reason": result.stop_reason,
            "training_history": history,
            "validation_history": validation_history,
            "selector_state": selector.state_dict(),
            "stopper_state": stopper.state_dict(),
        }
        _atomic_write_text(output / "training_curve.csv", _curve_text(history))
        _atomic_write_text(
            output / "worker_summary.json",
            json.dumps(summary_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        return result


__all__ = [
    "PilotTrainer",
    "PilotTrainingResult",
    "ValidationEvent",
    "configure_variant",
]

"""Variant-aware training loop for the bounded Scene 1 pilot."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
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
from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotResumeError,
    PilotResumeState,
    build_run_fingerprint,
    inspect_pilot_checkpoint,
    restore_pilot_checkpoint,
    validate_pilot_resume_model,
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


def _stage_text(path: Path, text: str) -> Path:
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
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rewrite_staged_text(staged: Path, text: str) -> None:
    with staged.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_report_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_staged_text(staged: Path, path: Path) -> None:
    replaced = False
    try:
        os.replace(staged, path)
        replaced = True
        _fsync_report_directory(path.parent)
    except BaseException as error:
        if replaced:
            try:
                _fsync_report_directory(path.parent)
            except OSError:
                raise
            if isinstance(error, Exception):
                return
        raise


def _duplicate_pinned_directory(path: Path) -> int | None:
    parts = path.parts
    if len(parts) == 5 and parts[:4] == ("/", "proc", "self", "fd"):
        try:
            source_fd = int(parts[4])
        except ValueError:
            return None
        descriptor = os.dup(source_fd)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"pinned output is not a directory: {path}")
        return descriptor
    return None


def _open_child_directory(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    metadata = os.fstat(descriptor)
    path_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino)
        != (path_metadata.st_dev, path_metadata.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"unsafe validation directory: {name}")
    return descriptor


@contextmanager
def _secure_validation_directory(output: Path, step: int):
    parent_fd = _duplicate_pinned_directory(output)
    if parent_fd is None:
        parent_fd = os.open(
            output,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    validation_fd: int | None = None
    step_fd: int | None = None
    try:
        validation_fd = _open_child_directory(parent_fd, "validation")
        step_fd = _open_child_directory(validation_fd, f"step_{step:06d}")
        yield Path(f"/proc/self/fd/{step_fd}")
    finally:
        if step_fd is not None:
            os.close(step_fd)
        if validation_fd is not None:
            os.close(validation_fd)
        os.close(parent_fd)


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


def _summary_payload(
    *,
    result: PilotTrainingResult,
    history: Sequence[Mapping[str, object]],
    validation_history: Sequence[Mapping[str, object]],
    selector: BestSelector,
    stopper: EarlyStopper,
    checkpoint_store: PilotCheckpointStore | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "variant": result.variant.value,
        "completed_warmup_steps": result.completed_warmup_steps,
        "completed_joint_steps": result.completed_joint_steps,
        "best_step": result.best_step,
        "stop_reason": result.stop_reason,
        "training_history": list(history),
        "validation_history": list(validation_history),
        "selector_state": selector.state_dict(),
        "stopper_state": stopper.state_dict(),
    }
    if checkpoint_store is not None:
        payload["checkpoint_io"] = checkpoint_store.metrics
    return payload


def _summary_text(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _publish_final_reports(
    *,
    output: Path,
    result: PilotTrainingResult,
    selector: BestSelector,
    stopper: EarlyStopper,
    checkpoint_store: PilotCheckpointStore | None,
    publish_complete: Callable[[], None] | None,
) -> None:
    history = result.training_history
    validation_history = result.validation_history
    curve_path = output / "training_curve.csv"
    summary_path = output / "worker_summary.json"
    curve_staged: Path | None = None
    summary_staged: Path | None = None
    try:
        curve_staged = _stage_text(curve_path, _curve_text(history))
        summary_staged = _stage_text(
            summary_path,
            _summary_text(
                _summary_payload(
                    result=result,
                    history=history,
                    validation_history=validation_history,
                    selector=selector,
                    stopper=stopper,
                    checkpoint_store=checkpoint_store,
                )
            ),
        )
        if publish_complete is not None:
            publish_complete()
            # The successful complete checkpoint belongs to this invocation's
            # metrics, so refresh only the noncanonical staged summary.
            _rewrite_staged_text(
                summary_staged,
                _summary_text(
                    _summary_payload(
                        result=result,
                        history=history,
                        validation_history=validation_history,
                        selector=selector,
                        stopper=stopper,
                        checkpoint_store=checkpoint_store,
                    )
                ),
            )
        # A previous/stale success declaration must not survive a failed curve
        # publication. The complete checkpoint remains the finalize source.
        summary_path.unlink(missing_ok=True)
        _fsync_report_directory(output)
        _publish_staged_text(curve_staged, curve_path)
        curve_staged = None
        # The summary is the success declaration and is always published last.
        _publish_staged_text(summary_staged, summary_path)
        summary_staged = None
    finally:
        if curve_staged is not None:
            curve_staged.unlink(missing_ok=True)
        if summary_staged is not None:
            summary_staged.unlink(missing_ok=True)


def _inspect_canonical_best(
    *,
    store: PilotCheckpointStore,
    latest: PilotResumeState,
    indices: VariantIndices,
    run_fingerprint: Mapping[str, object],
    model: nn.Module,
) -> PilotResumeState | None:
    selected_step = latest.selector.best_step
    if selected_step is None:
        if latest.best_generation is not None:
            raise PilotResumeError(
                "latest checkpoint tracks a best generation without a selection"
            )
        if store.best_path.exists():
            raise PilotResumeError(
                "stray best checkpoint exists without a selected best"
            )
        return None
    if latest.best_generation is None:
        raise PilotResumeError("latest selected best has no tracked best generation")
    if not store.best_path.is_file():
        raise PilotResumeError(
            f"selected best checkpoint is missing: {store.best_path}"
        )
    best = inspect_pilot_checkpoint(
        store.best_path,
        expected_compatibility=store.compatibility,
        indices=indices,
        expected_run_fingerprint=run_fingerprint,
        model=model,
        active_resume=False,
    )
    if best.checkpoint_kind != "best":
        raise PilotResumeError("canonical best path does not contain a best checkpoint")
    if best.generation != latest.best_generation:
        raise PilotResumeError(
            "best checkpoint generation does not match latest best_generation"
        )
    if best.selector.best_step != selected_step:
        raise PilotResumeError("best checkpoint selected step disagrees with latest")
    if best.best_evaluation_summary != latest.best_evaluation_summary:
        raise PilotResumeError("best checkpoint selected summary disagrees with latest")
    selected_summary = next(
        row["summary"]
        for row in latest.validation_history
        if row["step"] == selected_step
    )
    if best.validation_summary != selected_summary:
        raise PilotResumeError(
            "best checkpoint validation summary is not the selected row"
        )
    expected_validations = tuple(
        row for row in latest.validation_history if row["step"] <= selected_step
    )
    expected_training = tuple(
        row
        for row in latest.training_history
        if row["stage"] == "warmup"
        or (row["stage"] == "joint" and row["step"] <= selected_step)
    )
    if (
        best.next_joint_position != selected_step
        or best.validation_history != expected_validations
        or best.training_history != expected_training
    ):
        raise PilotResumeError(
            "best checkpoint history does not identify the selected training state"
        )
    latest_selector = latest.selector.state_dict()
    best_selector = best.selector.state_dict()
    for name in (
        "visual_baseline",
        "psnr_tolerance_db",
        "ssim_tolerance",
        "best_step",
        "best_audio_total",
    ):
        if best_selector[name] != latest_selector[name]:
            raise PilotResumeError(
                f"best checkpoint selector {name} disagrees with latest"
            )
    return best


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
        checkpoint_store: PilotCheckpointStore | None = None,
        preloaded_resume_state: PilotResumeState | None = None,
        checkpoint_store_prepared: bool = False,
    ) -> PilotTrainingResult:
        arguments = {
            "model": model,
            "train_samples": train_samples,
            "heldout_samples": heldout_samples,
            "indices": indices,
            "heldout_indices": heldout_indices,
            "variant": variant,
            "visual_baseline": visual_baseline,
            "audio_loss_fn": audio_loss_fn,
            "output_dir": output_dir,
            "on_validation": on_validation,
            "on_best_candidate": on_best_candidate,
            "checkpoint_store": checkpoint_store,
            "preloaded_resume_state": preloaded_resume_state,
            "checkpoint_store_prepared": checkpoint_store_prepared,
        }
        if checkpoint_store is None:
            return self._run_impl(**arguments)
        with checkpoint_store:
            return self._run_impl(**arguments)

    def _run_impl(
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
        checkpoint_store: PilotCheckpointStore | None = None,
        preloaded_resume_state: PilotResumeState | None = None,
        checkpoint_store_prepared: bool = False,
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
        if resolved_variant == Variant.CONDITION_OFF and indices.warmup:
            raise ValueError("condition_off requires empty warmup indices")
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
        resume_state: PilotResumeState | None = preloaded_resume_state
        if checkpoint_store is not None:
            if checkpoint_store.output_dir.resolve() != output.resolve():
                raise ValueError("checkpoint store output_dir must match trainer output_dir")
            run_fingerprint = build_run_fingerprint(
                pilot_config=self.config,
                train_config=self.train_config,
                visual_baseline=visual_baseline,
                model=model,
                warmup_optimizer_factory=self.warmup_optimizer_factory,
                joint_optimizer_factory=self.joint_optimizer_factory,
                warmup_step_fn=self.warmup_step_fn,
                joint_step_fn=self.joint_step_fn,
                audio_loss_fn=audio_loss_fn,
                component_identities=checkpoint_store.component_identities,
            )
            checkpoint_store.bind_run_fingerprint(run_fingerprint)
            if not checkpoint_store_prepared:
                checkpoint_store.prepare()
            if checkpoint_store.resume:
                if resume_state is None:
                    resume_state = inspect_pilot_checkpoint(
                        checkpoint_store.latest_path,
                        expected_compatibility=checkpoint_store.compatibility,
                        indices=indices,
                        expected_run_fingerprint=run_fingerprint,
                        model=model,
                        allow_complete=True,
                    )
                else:
                    validate_pilot_resume_model(resume_state, model)
                checkpoint_store.inspected_best_state = _inspect_canonical_best(
                    store=checkpoint_store,
                    latest=resume_state,
                    indices=indices,
                    run_fingerprint=run_fingerprint,
                    model=model,
                )
                expected_selector = BestSelector(
                    visual_baseline,
                    self.config.psnr_tolerance_db,
                    self.config.ssim_tolerance,
                ).state_dict()
                actual_selector = resume_state.selector.state_dict()
                for name in (
                    "visual_baseline",
                    "psnr_tolerance_db",
                    "ssim_tolerance",
                ):
                    if actual_selector[name] != expected_selector[name]:
                        raise PilotResumeError(
                            f"selector {name} mismatch: "
                            f"actual={actual_selector[name]!r}, "
                            f"expected={expected_selector[name]!r}"
                        )
                expected_stopper = EarlyStopper(
                    self.config.minimum_joint_steps,
                    self.config.patience,
                    self.config.minimum_relative_improvement,
                ).state_dict()
                actual_stopper = resume_state.stopper.state_dict()
                for name in ("minimum_steps", "patience", "relative_delta"):
                    if actual_stopper[name] != expected_stopper[name]:
                        raise PilotResumeError(
                            f"stopper {name} mismatch: "
                            f"actual={actual_stopper[name]!r}, "
                            f"expected={expected_stopper[name]!r}"
                        )
            elif resume_state is not None:
                raise ValueError(
                    "preloaded_resume_state requires a resume checkpoint store"
                )
        output_ready = False

        def prepare_output() -> None:
            nonlocal output_ready
            if output_ready:
                return
            output.mkdir(parents=True, exist_ok=True)
            # A failed replacement run must not leave an older success declaration.
            (output / "worker_summary.json").unlink(missing_ok=True)
            (output / "training_curve.csv").unlink(missing_ok=True)
            output_ready = True

        if resume_state is None:
            prepare_output()

        selector = (
            resume_state.selector
            if resume_state is not None
            else BestSelector(
                visual_baseline,
                self.config.psnr_tolerance_db,
                self.config.ssim_tolerance,
            )
        )
        stopper = (
            resume_state.stopper
            if resume_state is not None
            else EarlyStopper(
                self.config.minimum_joint_steps,
                self.config.patience,
                self.config.minimum_relative_improvement,
            )
        )
        history: list[dict[str, object]] = (
            [] if resume_state is None else [dict(row) for row in resume_state.training_history]
        )
        validation_history: list[dict[str, object]] = (
            []
            if resume_state is None
            else [dict(row) for row in resume_state.validation_history]
        )
        warmup_position = 0 if resume_state is None else resume_state.next_warmup_position
        joint_position = 0 if resume_state is None else resume_state.next_joint_position
        maximum_positive_probe = (
            0.0
            if resume_state is None
            else resume_state.maximum_positive_audio_visual_gradient
        )
        latest_validation_summary = (
            None if resume_state is None else resume_state.validation_summary
        )
        best_evaluation_summary = (
            None if resume_state is None else resume_state.best_evaluation_summary
        )
        checkpoint_pending_validation = (
            False if resume_state is None else resume_state.pending_validation
        )
        checkpoint_stop_requested = (
            False if resume_state is None else resume_state.stop_requested
        )

        if resume_state is not None and resume_state.stage == "complete":
            if resume_state.stop_reason is None:
                raise PilotResumeError(
                    "complete checkpoint is missing its stop reason"
                )
            result = PilotTrainingResult(
                variant=resolved_variant,
                completed_warmup_steps=resume_state.completed_warmup_steps,
                completed_joint_steps=resume_state.completed_joint_steps,
                best_step=selector.best_step,
                stop_reason=resume_state.stop_reason,
                validation_history=tuple(validation_history),
                training_history=tuple(history),
            )
            _publish_final_reports(
                output=output,
                result=result,
                selector=selector,
                stopper=stopper,
                checkpoint_store=checkpoint_store,
                publish_complete=None,
            )
            return result

        def checkpoint_kwargs(
            *,
            stage: str,
            optimizer: Any | None,
            optimizer_stage: str | None,
            stop_reason: str | None = None,
            pending_validation: bool = False,
            stop_requested: bool = False,
        ) -> dict[str, object]:
            assert checkpoint_store is not None
            return {
                "model": model,
                "compatibility": checkpoint_store.compatibility,
                "stage": stage,
                "next_warmup_position": warmup_position,
                "next_joint_position": joint_position,
                "optimizer": optimizer,
                "optimizer_stage": optimizer_stage,
                "selector": selector,
                "stopper": stopper,
                "training_history": history,
                "validation_history": validation_history,
                "maximum_positive_audio_visual_gradient": maximum_positive_probe,
                "stop_reason": stop_reason,
                "validation_summary": latest_validation_summary,
                "best_evaluation_summary": best_evaluation_summary,
                "pending_validation": pending_validation,
                "stop_requested": stop_requested,
            }

        def save_latest(
            *,
            stage: str,
            optimizer: Any | None,
            optimizer_stage: str | None,
            stop_reason: str | None = None,
            pending_validation: bool = False,
            stop_requested: bool = False,
        ) -> None:
            if checkpoint_store is None:
                return
            checkpoint_store.publish_latest(
                **checkpoint_kwargs(
                    stage=stage,
                    optimizer=optimizer,
                    optimizer_stage=optimizer_stage,
                    stop_reason=stop_reason,
                    pending_validation=pending_validation,
                    stop_requested=stop_requested,
                )
            )

        def validate_optimizer_identity(optimizer: Any, stage: str) -> None:
            if checkpoint_store is None or checkpoint_store.run_fingerprint is None:
                return
            optimizer_type = optimizer.__class__
            actual = f"{optimizer_type.__module__}.{optimizer_type.__qualname__}"
            expected = checkpoint_store.run_fingerprint["inputs"][
                f"{stage}_optimizer_class"
            ]
            if actual != expected:
                raise PilotResumeError(
                    f"{stage} optimizer class mismatch: "
                    f"actual={actual!r}, expected={expected!r}"
                )

        if expected_warmup and (
            resume_state is None or resume_state.stage == "warmup"
        ):
            if resume_state is None:
                configure_variant(model, resolved_variant, "warmup")
                optimizer = self.warmup_optimizer_factory(
                    model, self.train_config.condition_lr
                )
                validate_optimizer_identity(optimizer, "warmup")
            else:
                original_model_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                original_flags = {
                    name: parameter.requires_grad
                    for name, parameter in model.named_parameters()
                }
                original_condition = getattr(model, "condition_enabled", None)
                try:
                    configure_variant(model, resolved_variant, "warmup")
                    optimizer = self.warmup_optimizer_factory(
                        model, self.train_config.condition_lr
                    )
                    validate_optimizer_identity(optimizer, "warmup")
                    restore_pilot_checkpoint(
                        resume_state,
                        model=model,
                        optimizer=optimizer,
                        optimizer_stage="warmup",
                    )
                except BaseException:
                    model.load_state_dict(original_model_state, strict=True)
                    for name, parameter in model.named_parameters():
                        parameter.requires_grad_(original_flags[name])
                    if hasattr(model, "condition_enabled"):
                        model.condition_enabled = original_condition
                    raise
            prepare_output()
            for step, index in enumerate(
                indices.warmup[warmup_position:], start=warmup_position + 1
            ):
                stats = self.warmup_step_fn(
                    model, train_samples[index], optimizer, audio_loss_fn
                )
                history.append(_stats_row("warmup", step, index, stats))
                warmup_position = step
                save_latest(
                    stage="warmup",
                    optimizer=optimizer,
                    optimizer_stage="warmup",
                )

        if resume_state is None or resume_state.stage != "joint":
            configure_variant(model, resolved_variant, "joint")
            joint_optimizer = self.joint_optimizer_factory(model, self.train_config)
            validate_optimizer_identity(joint_optimizer, "joint")
        else:
            original_model_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            original_flags = {
                name: parameter.requires_grad
                for name, parameter in model.named_parameters()
            }
            original_condition = getattr(model, "condition_enabled", None)
            try:
                configure_variant(model, resolved_variant, "joint")
                joint_optimizer = self.joint_optimizer_factory(
                    model, self.train_config
                )
                validate_optimizer_identity(joint_optimizer, "joint")
                restore_pilot_checkpoint(
                    resume_state,
                    model=model,
                    optimizer=joint_optimizer,
                    optimizer_stage="joint",
                )
            except BaseException:
                model.load_state_dict(original_model_state, strict=True)
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(original_flags[name])
                if hasattr(model, "condition_enabled"):
                    model.condition_enabled = original_condition
                raise
        prepare_output()
        visual_anchor = capture_visual_anchor(model.visual)
        positive_probe_seen = maximum_positive_probe > 0
        stop_reason = (
            "early_stop" if checkpoint_stop_requested else "max_steps"
        )
        completed_joint_steps = joint_position

        def validate_joint(step: int, row: Mapping[str, object]) -> bool:
            nonlocal latest_validation_summary, best_evaluation_summary
            nonlocal checkpoint_pending_validation, checkpoint_stop_requested
            with _secure_validation_directory(output, step) as validation_dir:
                evaluation = self.evaluator.evaluate(
                    heldout_samples,
                    heldout_indices,
                    system_name=f"{resolved_variant.value}_step_{step:06d}",
                    condition_enabled=bool(model.condition_enabled),
                    output_dir=validation_dir,
                )
            summary = json.loads(_strict_json(evaluation.summary))
            selected = selector.consider(step, summary)
            latest_validation_summary = summary
            if selected:
                best_evaluation_summary = summary
            audio_total = summary["audio_total"]["mean"]
            should_stop = stopper.update(step, audio_total)
            checkpoint_pending_validation = False
            checkpoint_stop_requested = should_stop
            validation_history.append({"step": step, "summary": summary})
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
            if checkpoint_store is not None:
                latest_kwargs = checkpoint_kwargs(
                    stage="joint",
                    optimizer=joint_optimizer,
                    optimizer_stage="joint",
                    pending_validation=False,
                    stop_requested=should_stop,
                )
                best_kwargs = (
                    checkpoint_kwargs(
                        stage="joint",
                        optimizer=None,
                        optimizer_stage=None,
                        pending_validation=False,
                        stop_requested=should_stop,
                    )
                    if selected
                    else None
                )
                checkpoint_store.publish_validation(
                    latest_kwargs=latest_kwargs,
                    best_kwargs=best_kwargs,
                )
            if on_validation is not None:
                on_validation(event)
            if selected and on_best_candidate is not None:
                on_best_candidate(event)
            return should_stop

        pending_validation = checkpoint_pending_validation
        stop_requested = checkpoint_stop_requested
        if pending_validation:
            stop_requested = validate_joint(joint_position, history[-1])
            if stop_requested:
                stop_reason = "early_stop"

        for step, index in enumerate(
            indices.joint[joint_position:], start=joint_position + 1
        ):
            if stop_requested:
                break
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
            joint_position = step
            if should_probe and stats.audio_to_visual_grad_norm > 0:
                positive_probe_seen = True
                maximum_positive_probe = max(
                    maximum_positive_probe, float(stats.audio_to_visual_grad_norm)
                )
            validate_now = (
                step % self.config.validation_interval == 0
                or step == len(indices.joint)
            )
            checkpoint_pending_validation = validate_now
            checkpoint_stop_requested = False
            save_latest(
                stage="joint",
                optimizer=joint_optimizer,
                optimizer_stage="joint",
                pending_validation=validate_now,
            )
            if not validate_now:
                continue
            if validate_joint(step, row):
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
        _publish_final_reports(
            output=output,
            result=result,
            selector=selector,
            stopper=stopper,
            checkpoint_store=checkpoint_store,
            publish_complete=lambda: save_latest(
                stage="complete",
                optimizer=None,
                optimizer_stage=None,
                stop_reason=stop_reason,
                stop_requested=stop_reason == "early_stop",
            ),
        )
        return result


__all__ = [
    "PilotTrainer",
    "PilotTrainingResult",
    "ValidationEvent",
    "configure_variant",
]

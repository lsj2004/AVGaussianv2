from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from numbers import Integral, Real
from pathlib import Path

import torch
from torch import Tensor, nn

from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.experiment.contracts import EvaluationResult
from avgaussianv2.experiment.metrics import (
    aggregate_metrics,
    log_spectral_distance,
    lre_error_db,
    psnr,
    rgb_l1,
    ssim,
    waveform_l1,
)


METRIC_NAMES = (
    "audio_total",
    "audio_mono",
    "audio_diff",
    "waveform_l1",
    "mono_lsd",
    "diff_lsd",
    "lre_error_db",
    "rgb_psnr",
    "rgb_ssim",
    "rgb_l1",
)
_LOSS_KEYS = {
    "audio_total": "total_loss",
    "audio_mono": "mono_loss",
    "audio_diff": "diff_loss",
}


def move_sample(
    sample: AlignedAVSample, device: torch.device | str
) -> AlignedAVSample:
    """Move every tensor field while preserving all sample metadata."""
    destination = torch.device(device)
    changes = {
        name: value.to(destination)
        for name, value in vars(sample).items()
        if isinstance(value, Tensor)
    }
    return replace(sample, **changes)


def _sample_id(sample: AlignedAVSample) -> str:
    if not math.isfinite(sample.time_seconds):
        raise ValueError("sample time_seconds must be finite")
    identity = {
        "camera": sample.camera,
        "frame_index": sample.frame_index,
        "scene_id": sample.scene_id,
        "time_seconds": sample.time_seconds,
    }
    return json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _scalar(name: str, value: object) -> float:
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise ValueError(f"loss {name!r} must be a scalar tensor")
        result = float(value.detach().item())
    elif isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        raise TypeError(f"loss {name!r} must be a numeric scalar")
    if not math.isfinite(result):
        raise ValueError(f"loss {name!r} must be finite")
    return result


def _stage_text(path: Path, content: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_metric_pair(output_dir: Path, rows: tuple[dict[str, object], ...], summary: dict[str, dict[str, float]]) -> None:
    """Publish a staged metric pair, rolling back ordinary replacement failures.

    Each individual file is atomic and replacement failures are rolled back. No
    filesystem API can make two filenames crash-atomic as one transaction, so a
    process or machine crash between the two replacements remains a narrow limit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = (
        output_dir / "metrics_per_sample.jsonl",
        output_dir / "metrics_summary.json",
    )
    row_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    summary_text = json.dumps(
        summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    staged: list[Path] = []
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        # Append immediately so the first stage is still known and cleaned if
        # constructing the second stage fails.
        staged.append(_stage_text(destinations[0], row_text))
        staged.append(_stage_text(destinations[1], summary_text))
        for destination in destinations:
            if destination.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".backup", dir=output_dir
                )
                os.close(descriptor)
                backup = Path(backup_name)
                backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
        for temporary, destination in zip(staged, destinations):
            os.replace(temporary, destination)
            installed.append(destination)
        _fsync_directory(output_dir)
    except BaseException:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        _fsync_directory(output_dir)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        audio_loss_fn: Callable[[Tensor, Tensor], Mapping[str, object]],
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.audio_loss_fn = audio_loss_fn
        self.device = torch.device(device)

    def evaluate(
        self,
        samples: Sequence[AlignedAVSample],
        indices: Sequence[int],
        system_name: str,
        condition_enabled: bool,
        output_dir: Path | str,
    ) -> EvaluationResult:
        selected = tuple(indices)
        if not selected:
            raise ValueError("evaluation indices must be nonempty")
        for index in selected:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("evaluation indices must be integers")
            if index < 0 or index >= len(samples):
                raise ValueError(f"evaluation index {index} is out of range")
        if len(set(selected)) != len(selected):
            raise ValueError("evaluation indices must not contain duplicates")
        if not hasattr(self.model, "condition_enabled"):
            raise ValueError("evaluated model must expose condition_enabled")

        original_training = self.model.training
        original_condition = self.model.condition_enabled
        rows: list[dict[str, object]] = []
        sample_ids: set[str] = set()
        try:
            self.model.eval()
            self.model.condition_enabled = bool(condition_enabled)
            with torch.no_grad():
                for index in selected:
                    sample = move_sample(samples[index], self.device)
                    identity = _sample_id(sample)
                    if identity in sample_ids:
                        raise ValueError(f"duplicate sample ID: {identity}")
                    sample_ids.add(identity)
                    output = self.model(sample)
                    losses = self.audio_loss_fn(output.predicted_audio, sample.target_audio)
                    if not isinstance(losses, Mapping):
                        raise TypeError("audio loss criterion must return a mapping")
                    missing = set(_LOSS_KEYS.values()) - set(losses)
                    if missing:
                        raise ValueError(f"audio loss mapping missing required keys: {sorted(missing)}")

                    rendered_rgb = output.rgbd.rgb
                    target_rgb = sample.target_rgb
                    if rendered_rgb.device != target_rgb.device:
                        raise ValueError("rendered and target RGB must be on the same device")
                    if rendered_rgb.dtype != target_rgb.dtype:
                        raise ValueError("rendered and target RGB must have the same dtype")
                    if rendered_rgb.shape != target_rgb.shape:
                        raise ValueError(
                            "rendered and target RGB must have equal BHWC shapes, got "
                            f"{tuple(rendered_rgb.shape)} and {tuple(target_rgb.shape)}"
                        )

                    metrics = {
                        name: _scalar(loss_key, losses[loss_key])
                        for name, loss_key in _LOSS_KEYS.items()
                    }
                    metrics.update(
                        waveform_l1=waveform_l1(output.predicted_audio, sample.target_audio),
                        mono_lsd=log_spectral_distance(output.predicted_audio, sample.target_audio, "mono"),
                        diff_lsd=log_spectral_distance(output.predicted_audio, sample.target_audio, "diff"),
                        lre_error_db=lre_error_db(output.predicted_audio, sample.target_audio),
                        rgb_psnr=psnr(rendered_rgb, target_rgb),
                        rgb_ssim=ssim(rendered_rgb, target_rgb),
                        rgb_l1=rgb_l1(rendered_rgb, target_rgb),
                    )
                    for name in METRIC_NAMES:
                        if not math.isfinite(metrics[name]):
                            raise ValueError(f"metric {name!r} must be finite")
                    rows.append(
                        {
                            "sample_id": identity,
                            "scene_id": sample.scene_id,
                            "camera": sample.camera,
                            "frame_index": sample.frame_index,
                            "time_seconds": sample.time_seconds,
                            **metrics,
                        }
                    )
        finally:
            self.model.condition_enabled = original_condition
            self.model.train(original_training)

        metric_rows = [{name: float(row[name]) for name in METRIC_NAMES} for row in rows]
        summary = aggregate_metrics(metric_rows)
        result = EvaluationResult(system_name, len(rows), tuple(rows), summary)
        _write_metric_pair(Path(output_dir), result.rows, result.summary)
        return result

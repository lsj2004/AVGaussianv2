from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import stat
import tempfile
import warnings
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


def _open_pinned_output_directory(output_dir: Path) -> int:
    parts = output_dir.parts
    if len(parts) == 5 and parts[:4] == ("/", "proc", "self", "fd"):
        try:
            descriptor = os.dup(int(parts[4]))
        except (ValueError, OSError) as error:
            raise ValueError(f"invalid pinned metric directory: {output_dir}") from error
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"metric output is not a directory: {output_dir}")
        return descriptor
    output_dir.mkdir(parents=True, exist_ok=True)
    before = output_dir.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(
            f"metric output must be a non-symlink directory: {output_dir}"
        )
    descriptor = os.open(
        output_dir,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise ValueError(f"metric output directory identity changed: {output_dir}")
    return descriptor


def _snapshot_without_removing(destination: Path, suffix: str = ".backup") -> Path:
    source_fd = os.open(
        destination,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    source_metadata = os.fstat(source_fd)
    if (
        not stat.S_ISREG(source_metadata.st_mode)
        or source_metadata.st_nlink != 1
    ):
        os.close(source_fd)
        raise ValueError(
            f"metric backup source must be a single-link regular file: {destination}"
        )
    backup_fd, backup_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(backup_fd)
    backup = Path(backup_name)
    backup.unlink()
    try:
        os.link(destination, backup, follow_symlinks=False)
        backup_metadata = backup.lstat()
        if (
            not stat.S_ISREG(backup_metadata.st_mode)
            or (backup_metadata.st_dev, backup_metadata.st_ino)
            != (source_metadata.st_dev, source_metadata.st_ino)
        ):
            raise ValueError("metric backup source changed during hard-link snapshot")
    except OSError:
        try:
            target_fd = os.open(
                backup,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with (
                os.fdopen(source_fd, "rb", closefd=False) as source,
                os.fdopen(target_fd, "wb") as target,
            ):
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            backup.unlink(missing_ok=True)
            raise
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
    return backup


def _record_recovery_errors(primary: BaseException, errors: list[BaseException]) -> None:
    if not errors:
        return
    details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
    existing = getattr(primary, "publication_recovery_errors", ())
    setattr(primary, "publication_recovery_errors", (*existing, *errors))
    if hasattr(primary, "add_note"):
        primary.add_note(f"metric publication recovery errors: {details}")
    else:  # pragma: no cover - Python 3.10 compatibility
        warnings.warn(f"metric publication recovery errors: {details}", RuntimeWarning)


def _remove_paths(paths: Sequence[Path]) -> list[BaseException]:
    errors: list[BaseException] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except BaseException as error:
            errors.append(error)
    return errors


def _publish_metric_pair(
    output_dir: Path,
    destinations: tuple[Path, Path],
    contents: tuple[str, str],
) -> None:
    staged: list[Path] = []
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination in destinations:
            try:
                metadata = destination.lstat()
            except FileNotFoundError:
                continue
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ValueError(
                    "metric destination must be a single-link "
                    f"non-symlink regular file: {destination.name}"
                )
        for destination, content in zip(destinations, contents):
            staged.append(_stage_text(destination, content))
        for destination in destinations:
            if destination.exists():
                backups[destination] = _snapshot_without_removing(destination)
        _fsync_directory(output_dir)
        for temporary, destination in zip(staged, destinations):
            os.replace(temporary, destination)
            installed.append(destination)
        _fsync_directory(output_dir)
    except BaseException as primary:
        recovery_errors: list[BaseException] = []
        restoration_temps: list[Path] = []
        for destination in installed:
            backup = backups.get(destination)
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    restoration = _snapshot_without_removing(backup, suffix=".restore")
                    restoration_temps.append(restoration)
                    os.replace(restoration, destination)
            except BaseException as error:
                recovery_errors.append(error)
        if installed:
            try:
                _fsync_directory(output_dir)
            except BaseException as error:
                recovery_errors.append(error)

        recovery_errors.extend(_remove_paths(staged))
        recovery_errors.extend(_remove_paths(restoration_temps))
        # Backups are expendable only when no canonical file changed, or after
        # every changed destination was restored and made directory-durable.
        if not recovery_errors:
            recovery_errors.extend(_remove_paths(list(backups.values())))
            try:
                _fsync_directory(output_dir)
            except BaseException as error:
                recovery_errors.append(error)
        _record_recovery_errors(primary, recovery_errors)
        raise
    else:
        cleanup_errors = _remove_paths(staged)
        cleanup_errors.extend(_remove_paths(list(backups.values())))
        try:
            _fsync_directory(output_dir)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            failure = RuntimeError("metric publication succeeded but cleanup failed")
            _record_recovery_errors(failure, cleanup_errors)
            raise failure from cleanup_errors[0]


def _write_metric_pair(output_dir: Path, rows: tuple[dict[str, object], ...], summary: dict[str, dict[str, float]]) -> None:
    """Publish a staged metric pair, rolling back ordinary replacement failures.

    The caller must give each system an exclusive output directory. A persistent
    lock file carries a kernel advisory lock during publication; file presence
    alone never blocks a later process after a crash. Canonical paths remain
    present while durable backups are created, and each individual replacement is
    atomic. Ordinary replacement failures are rolled back. No filesystem API can
    make two filenames crash-atomic, so a crash can expose mixed versions.
    """
    row_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    summary_text = json.dumps(
        summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    directory_fd = _open_pinned_output_directory(output_dir)
    pinned_output = Path(f"/proc/self/fd/{directory_fd}")
    destinations = (
        pinned_output / "metrics_per_sample.jsonl",
        pinned_output / "metrics_summary.json",
    )
    lock_path = pinned_output / ".metrics-publication.lock"
    lock_descriptor: int | None = None
    primary: BaseException | None = None
    try:
        lock_descriptor = os.open(
            lock_path.name,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        lock_metadata = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_nlink != 1
        ):
            raise ValueError(
                "metric publication lock must be a single-link regular file"
            )
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(lock_descriptor)
            lock_descriptor = None
            raise RuntimeError(
                f"metric output directory already has an active writer: {output_dir}"
            ) from error
        os.ftruncate(lock_descriptor, 0)
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_descriptor)
        _fsync_directory(pinned_output)
        _publish_metric_pair(pinned_output, destinations, (row_text, summary_text))
    except BaseException as error:
        primary = error
        raise
    finally:
        release_errors: list[BaseException] = []
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            except BaseException as error:
                release_errors.append(error)
            try:
                os.close(lock_descriptor)
            except BaseException as error:
                release_errors.append(error)
        try:
            os.close(directory_fd)
        except BaseException as error:
            release_errors.append(error)
        if release_errors:
            if primary is not None:
                _record_recovery_errors(primary, release_errors)
            else:
                failure = RuntimeError("metric publication lock cleanup failed")
                _record_recovery_errors(failure, release_errors)
                raise failure from release_errors[0]


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
        """Evaluate selected samples and publish into a system-exclusive directory.

        Concurrent writers to one directory are rejected across processes. Pilot
        callers must use a distinct output directory for every evaluated system.
        """
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

        original_training = tuple(
            (module, module.training) for module in self.model.modules()
        )
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
            for module, training in original_training:
                module.training = training

        metric_rows = [{name: float(row[name]) for name in METRIC_NAMES} for row in rows]
        summary = aggregate_metrics(metric_rows)
        result = EvaluationResult(system_name, len(rows), tuple(rows), summary)
        _write_metric_pair(Path(output_dir), result.rows, result.summary)
        return result

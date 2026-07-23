"""Strict, deterministic comparison reports for the five-system pilot."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from avgaussianv2.experiment.contracts import EvaluationResult
from avgaussianv2.experiment.evaluation import METRIC_NAMES
from avgaussianv2.experiment.metrics import aggregate_metrics
from avgaussianv2.experiment.selection import visual_feasible


REQUIRED_SYSTEMS = (
    "baseline_imported",
    "joint_conditioned_on",
    "joint_conditioned_off",
    "frozen_visual_on",
    "condition_off",
)
STATISTICS = ("mean", "std", "median")
METRIC_DIRECTIONS = {
    metric: (
        "higher_is_better" if metric in {"rgb_psnr", "rgb_ssim"} else "lower_is_better"
    )
    for metric in METRIC_NAMES
}
_ROW_METADATA = ("sample_id", "scene_id", "camera", "frame_index", "time_seconds")
_ROW_FIELDS = set(_ROW_METADATA) | set(METRIC_NAMES)
_AGGREGATE_FIELDS = set(STATISTICS)
_PROVENANCE_FIELDS = {
    "scene_id",
    "checkpoint_sha256",
    "checkpoint_generation",
    "condition_enabled",
}
_WORKER_FIELDS = {
    "variant",
    "completed_warmup_steps",
    "completed_joint_steps",
    "best_step",
    "stop_reason",
    "training_history",
    "validation_history",
    "selector_state",
    "stopper_state",
    "checkpoint_io",
    "worker",
}
_TRAINING_ROW_FIELDS = {
    "stage",
    "step",
    "sample_index",
    "total",
    "audio_to_visual_grad_norm",
    "losses",
    "gradient_norms",
}
_WORKER_IDENTITY_FIELDS = {
    "variant",
    "device",
    "scene_id",
    "config_sha256",
    "manifest_sha256",
    "visual_baseline_sha256",
    "trusted_upstream_artifacts",
}
_SELECTOR_FIELDS = {
    "visual_baseline",
    "psnr_tolerance_db",
    "ssim_tolerance",
    "best_step",
    "best_audio_total",
    "last_step",
}
_STOPPER_FIELDS = {
    "minimum_steps",
    "patience",
    "relative_delta",
    "best",
    "stale",
    "last_step",
}
_CHECKPOINT_IO_FIELDS = {
    "save_count",
    "save_attempt_count",
    "failed_save_count",
    "save_bytes",
    "save_duration_seconds",
    "scope",
    "backup_copy_count",
    "backup_copy_attempt_count",
    "backup_copy_failure_count",
    "backup_copy_bytes",
    "backup_copy_duration_seconds",
    "cadence",
}
_EXPECTED_VARIANTS = {
    "joint_conditioned_on": "joint_conditioned",
    "joint_conditioned_off": "joint_conditioned",
    "frozen_visual_on": "frozen_visual",
    "condition_off": "condition_off",
}
_EXPECTED_CONDITION = {
    "baseline_imported": False,
    "joint_conditioned_on": True,
    "joint_conditioned_off": False,
    "frozen_visual_on": True,
    "condition_off": False,
}


@dataclass(frozen=True)
class SystemReportInput:
    name: str
    evaluation: EvaluationResult
    worker_summary: Mapping[str, object] | None
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class PilotDecision:
    ready: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonResult:
    systems: dict[str, dict[str, object]]
    rows: dict[str, tuple[dict[str, object], ...]]
    paired_condition: dict[str, object]
    descriptive_comparisons: dict[str, dict[str, object]]
    decision: PilotDecision

    @property
    def pair_records(self) -> tuple[dict[str, object], ...]:
        return tuple(self.paired_condition["records"])


def _exact(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(fields - actual)} "
            f"extra={sorted(actual - fields)}"
        )
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _finite_tree(value: object, name: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, Real):
        _scalar(value, name)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
        return
    raise TypeError(f"{name} contains unsupported value type")


def _metric_value(metric: str, value: object, name: str) -> float:
    result = _scalar(value, name)
    if metric != "rgb_ssim" and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    if metric == "rgb_ssim" and not -1 <= result <= 1:
        raise ValueError(f"{name} must be in [-1, 1]")
    if metric == "rgb_l1" and result > 1:
        raise ValueError(f"{name} must be at most 1")
    return result


def _validate_row(row: object, name: str) -> dict[str, object]:
    value = _exact(row, _ROW_FIELDS, name)
    result: dict[str, object] = {
        "sample_id": _text(value["sample_id"], f"{name}.sample_id"),
        "scene_id": _text(value["scene_id"], f"{name}.scene_id"),
        "camera": _text(value["camera"], f"{name}.camera"),
        "frame_index": _integer(value["frame_index"], f"{name}.frame_index"),
        "time_seconds": _scalar(value["time_seconds"], f"{name}.time_seconds"),
    }
    for metric in METRIC_NAMES:
        result[metric] = _metric_value(metric, value[metric], f"{name}.{metric}")
    return result


def _aggregate_validated_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    return aggregate_metrics(
        [
            {metric: float(row[metric]) for metric in METRIC_NAMES}
            for row in rows
        ]
    )


def _validate_summary(
    summary: object,
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> dict[str, dict[str, float]]:
    value = _exact(summary, set(METRIC_NAMES), f"{name} metric")
    result: dict[str, dict[str, float]] = {}
    for metric in METRIC_NAMES:
        aggregate = _exact(
            value[metric], _AGGREGATE_FIELDS, f"{name}.{metric} aggregate"
        )
        result[metric] = {}
        for statistic in STATISTICS:
            number = _metric_value(
                metric, aggregate[statistic], f"{name}.{metric}.{statistic}"
            )
            if statistic == "std" and number < 0:
                raise ValueError(f"{name}.{metric}.std must be nonnegative")
            result[metric][statistic] = number
    recalculated = _aggregate_validated_rows(rows)
    for metric in METRIC_NAMES:
        for statistic in STATISTICS:
            if result[metric][statistic] != recalculated[metric][statistic]:
                raise ValueError(
                    f"{name}.{metric}.{statistic} is inconsistent with per-sample rows"
                )
    return result


def _validate_evaluation(
    evaluation: object, expected_name: str
) -> tuple[dict[str, dict[str, float]], tuple[dict[str, object], ...]]:
    if not isinstance(evaluation, EvaluationResult):
        raise TypeError("evaluation must be an EvaluationResult")
    if evaluation.system_name != expected_name:
        raise ValueError(
            f"evaluation system_name mismatch for {expected_name!r}"
        )
    count = _integer(evaluation.count, f"{expected_name}.count", minimum=1)
    if not isinstance(evaluation.rows, (list, tuple)):
        raise TypeError(f"{expected_name}.rows must be a sequence")
    rows = tuple(
        _validate_row(row, f"{expected_name}.rows[{index}]")
        for index, row in enumerate(evaluation.rows)
    )
    if count != len(rows):
        raise ValueError(f"{expected_name} sample count does not match rows")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{expected_name} contains duplicate sample IDs")
    return _validate_summary(evaluation.summary, rows, expected_name), rows


def _validate_numeric_mapping(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{name} must be a nonempty mapping")
    for key, item in value.items():
        _text(key, f"{name} key")
        _scalar(item, f"{name}.{key}")


def _validate_worker(
    worker: object, system_name: str, scene_id: str
) -> dict[str, object]:
    value = _exact(worker, _WORKER_FIELDS, f"{system_name} worker summary")
    expected_variant = _EXPECTED_VARIANTS[system_name]
    if value["variant"] != expected_variant:
        raise ValueError(f"{system_name} worker variant mismatch")
    completed_warmup = _integer(
        value["completed_warmup_steps"], f"{system_name}.completed_warmup_steps"
    )
    completed_joint = _integer(
        value["completed_joint_steps"],
        f"{system_name}.completed_joint_steps",
        minimum=1,
    )
    best_step = _integer(value["best_step"], f"{system_name}.best_step", minimum=1)
    if best_step > completed_joint:
        raise ValueError(f"{system_name}.best_step exceeds completed joint steps")
    stop_reason = _text(value["stop_reason"], f"{system_name}.stop_reason")
    if stop_reason not in {"max_steps", "early_stop"}:
        raise ValueError(f"{system_name}.stop_reason is invalid")

    history = value["training_history"]
    if not isinstance(history, list) or not history:
        raise ValueError(f"{system_name}.training_history must be a nonempty list")
    gradients: list[float] = []
    previous: dict[str, int] = {}
    for index, raw_row in enumerate(history):
        row = _exact(
            raw_row,
            _TRAINING_ROW_FIELDS,
            f"{system_name}.training_history[{index}]",
        )
        stage = row["stage"]
        if stage not in {"warmup", "joint"}:
            raise ValueError(f"{system_name} training stage is invalid")
        step = _integer(row["step"], f"{system_name}.training_history[{index}].step", minimum=1)
        if step <= previous.get(str(stage), 0):
            raise ValueError(f"{system_name} training steps must increase within stage")
        previous[str(stage)] = step
        _integer(row["sample_index"], f"{system_name}.training_history[{index}].sample_index")
        _scalar(row["total"], f"{system_name}.training_history[{index}].total")
        gradient = _scalar(
            row["audio_to_visual_grad_norm"],
            f"{system_name}.training_history[{index}].audio_to_visual_grad_norm",
        )
        if gradient < 0:
            raise ValueError(f"{system_name} gradient norm must be nonnegative")
        if stage == "joint":
            gradients.append(gradient)
        _validate_numeric_mapping(
            row["losses"], f"{system_name}.training_history[{index}].losses"
        )
        _validate_numeric_mapping(
            row["gradient_norms"],
            f"{system_name}.training_history[{index}].gradient_norms",
        )
    if len(history) != completed_warmup + completed_joint:
        raise ValueError(
            f"{system_name}.training_history length does not match completed steps"
        )
    if sum(row["stage"] == "warmup" for row in history) != completed_warmup:
        raise ValueError(
            f"{system_name}.training_history warmup count does not match completed steps"
        )
    if sum(row["stage"] == "joint" for row in history) != completed_joint:
        raise ValueError(
            f"{system_name}.training_history joint count does not match completed steps"
        )

    validation_history = value["validation_history"]
    if not isinstance(validation_history, list):
        raise TypeError(f"{system_name}.validation_history must be a list")
    validation_steps: list[int] = []
    for index, raw_validation in enumerate(validation_history):
        validation = _exact(
            raw_validation,
            {"step", "summary"},
            f"{system_name}.validation_history[{index}]",
        )
        step = _integer(
            validation["step"],
            f"{system_name}.validation_history[{index}].step",
            minimum=1,
        )
        if step > completed_joint:
            raise ValueError(f"{system_name} validation step exceeds completed steps")
        validation_steps.append(step)
        summary = _exact(
            validation["summary"],
            set(METRIC_NAMES),
            f"{system_name}.validation_history[{index}].summary metric",
        )
        for metric in METRIC_NAMES:
            aggregate = _exact(
                summary[metric],
                _AGGREGATE_FIELDS,
                f"{system_name}.validation_history[{index}].summary.{metric} aggregate",
            )
            for statistic in STATISTICS:
                _metric_value(
                    metric,
                    aggregate[statistic],
                    f"{system_name}.validation_history[{index}].summary."
                    f"{metric}.{statistic}",
                )
    if validation_steps != sorted(set(validation_steps)):
        raise ValueError(f"{system_name} validation steps must strictly increase")
    if best_step not in validation_steps:
        raise ValueError(f"{system_name}.best_step is absent from validation_history")
    selector = _exact(
        value["selector_state"], _SELECTOR_FIELDS, f"{system_name}.selector_state"
    )
    selector_baseline = _exact(
        selector["visual_baseline"],
        {"rgb_psnr", "rgb_ssim"},
        f"{system_name}.selector_state.visual_baseline",
    )
    for metric in ("rgb_psnr", "rgb_ssim"):
        mean = _exact(
            selector_baseline[metric],
            {"mean"},
            f"{system_name}.selector_state.visual_baseline.{metric}",
        )
        _metric_value(
            metric,
            mean["mean"],
            f"{system_name}.selector_state.visual_baseline.{metric}.mean",
        )
    for field in ("psnr_tolerance_db", "ssim_tolerance", "best_audio_total"):
        if _scalar(
            selector[field], f"{system_name}.selector_state.{field}"
        ) < 0:
            raise ValueError(f"{system_name}.selector_state.{field} must be nonnegative")
    if _integer(
        selector["best_step"],
        f"{system_name}.selector_state.best_step",
        minimum=1,
    ) != best_step:
        raise ValueError(f"{system_name} selector best_step mismatch")
    selector_last = _integer(
        selector["last_step"],
        f"{system_name}.selector_state.last_step",
        minimum=1,
    )
    if selector_last != validation_steps[-1]:
        raise ValueError(f"{system_name} selector last_step mismatch")
    stopper = _exact(
        value["stopper_state"], _STOPPER_FIELDS, f"{system_name}.stopper_state"
    )
    _integer(
        stopper["minimum_steps"],
        f"{system_name}.stopper_state.minimum_steps",
    )
    _integer(
        stopper["patience"],
        f"{system_name}.stopper_state.patience",
        minimum=1,
    )
    _integer(stopper["stale"], f"{system_name}.stopper_state.stale")
    relative_delta = _scalar(
        stopper["relative_delta"],
        f"{system_name}.stopper_state.relative_delta",
    )
    if not 0 < relative_delta < 1:
        raise ValueError(f"{system_name}.stopper_state.relative_delta must be in (0, 1)")
    if _scalar(stopper["best"], f"{system_name}.stopper_state.best") < 0:
        raise ValueError(f"{system_name}.stopper_state.best must be nonnegative")
    stopper_last = _integer(
        stopper["last_step"],
        f"{system_name}.stopper_state.last_step",
        minimum=1,
    )
    if stopper_last != selector_last:
        raise ValueError(f"{system_name} stopper last_step mismatch")
    checkpoint_io = _exact(
        value["checkpoint_io"],
        _CHECKPOINT_IO_FIELDS,
        f"{system_name}.checkpoint_io",
    )
    for field in (
        "save_count",
        "save_attempt_count",
        "failed_save_count",
        "save_bytes",
        "backup_copy_count",
        "backup_copy_attempt_count",
        "backup_copy_failure_count",
        "backup_copy_bytes",
    ):
        _integer(
            checkpoint_io[field],
            f"{system_name}.checkpoint_io.{field}",
        )
    for field in ("save_duration_seconds", "backup_copy_duration_seconds"):
        if _scalar(
            checkpoint_io[field], f"{system_name}.checkpoint_io.{field}"
        ) < 0:
            raise ValueError(f"{system_name}.checkpoint_io.{field} must be nonnegative")
    if checkpoint_io["scope"] != "current_invocation":
        raise ValueError(f"{system_name}.checkpoint_io.scope is invalid")
    if checkpoint_io["cadence"] != "every_completed_optimizer_step":
        raise ValueError(f"{system_name}.checkpoint_io.cadence is invalid")
    identity = _exact(
        value["worker"], _WORKER_IDENTITY_FIELDS, f"{system_name}.worker"
    )
    if identity["variant"] != expected_variant:
        raise ValueError(f"{system_name} nested worker variant mismatch")
    if identity["scene_id"] != scene_id:
        raise ValueError(f"{system_name} worker scene mismatch")
    _text(identity["device"], f"{system_name}.worker.device")
    for field in ("config_sha256", "manifest_sha256", "visual_baseline_sha256"):
        _digest(identity[field], f"{system_name}.worker.{field}")
    if not isinstance(identity["trusted_upstream_artifacts"], bool):
        raise TypeError(
            f"{system_name}.worker.trusted_upstream_artifacts must be boolean"
        )
    return {
        "variant": expected_variant,
        "completed_warmup_steps": completed_warmup,
        "completed_joint_steps": completed_joint,
        "best_step": best_step,
        "stop_reason": stop_reason,
        "max_audio_to_visual_grad_norm": max(gradients),
        "config_sha256": identity["config_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
    }


def _validate_provenance(
    value: object, system_name: str
) -> dict[str, object]:
    source = _exact(value, _PROVENANCE_FIELDS, f"{system_name} provenance")
    condition = source["condition_enabled"]
    if not isinstance(condition, bool):
        raise TypeError(f"{system_name}.condition_enabled must be boolean")
    if condition != _EXPECTED_CONDITION[system_name]:
        raise ValueError(f"{system_name} condition_enabled mismatch")
    return {
        "scene_id": _text(source["scene_id"], f"{system_name}.scene_id"),
        "checkpoint_sha256": _digest(
            source["checkpoint_sha256"], f"{system_name}.checkpoint_sha256"
        ),
        "checkpoint_generation": _integer(
            source["checkpoint_generation"],
            f"{system_name}.checkpoint_generation",
        ),
        "condition_enabled": condition,
    }


def paired_audio_deltas(
    on_rows: Sequence[Mapping[str, object]],
    off_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Pair exact samples and return ``on - off`` audio-total deltas."""
    if not isinstance(on_rows, Sequence) or isinstance(on_rows, (str, bytes)):
        raise TypeError("on_rows must be a sequence")
    if not isinstance(off_rows, Sequence) or isinstance(off_rows, (str, bytes)):
        raise TypeError("off_rows must be a sequence")

    def index_rows(
        values: Sequence[Mapping[str, object]], label: str
    ) -> dict[str, Mapping[str, object]]:
        indexed: dict[str, Mapping[str, object]] = {}
        for position, row in enumerate(values):
            if not isinstance(row, Mapping):
                raise TypeError(f"{label}[{position}] must be a mapping")
            sample_id = _text(row.get("sample_id"), f"{label}[{position}].sample_id")
            if sample_id in indexed:
                raise ValueError(f"{label} contains duplicate sample ID {sample_id!r}")
            for field in _ROW_METADATA[1:]:
                if field not in row:
                    raise ValueError(f"{label}[{position}] is missing metadata {field!r}")
            _scalar(row.get("audio_total"), f"{label}[{position}].audio_total")
            indexed[sample_id] = row
        return indexed

    on = index_rows(on_rows, "on_rows")
    off = index_rows(off_rows, "off_rows")
    if not on:
        raise ValueError("paired rows must be nonempty")
    if set(on) != set(off):
        raise ValueError("on/off sample IDs differ")
    records: list[dict[str, object]] = []
    for sample_id in sorted(on):
        on_row, off_row = on[sample_id], off[sample_id]
        for field in _ROW_METADATA[1:]:
            if on_row[field] != off_row[field]:
                raise ValueError(
                    f"paired sample {sample_id!r} metadata mismatch for {field}"
                )
        on_audio = _scalar(on_row["audio_total"], f"{sample_id}.on.audio_total")
        off_audio = _scalar(off_row["audio_total"], f"{sample_id}.off.audio_total")
        delta = on_audio - off_audio
        if not math.isfinite(delta):
            raise ValueError(f"paired sample {sample_id!r} delta must be finite")
        records.append(
            {
                "sample_id": sample_id,
                "scene_id": on_row["scene_id"],
                "camera": on_row["camera"],
                "frame_index": on_row["frame_index"],
                "time_seconds": on_row["time_seconds"],
                "on_audio_total": on_audio,
                "off_audio_total": off_audio,
                "audio_total_delta": delta,
            }
        )
    values = [float(record["audio_total_delta"]) for record in records]
    aggregate = aggregate_metrics([{"delta": value} for value in values])["delta"]
    return {
        "sample_count": len(records),
        "mean": aggregate["mean"],
        "std": aggregate["std"],
        "median": aggregate["median"],
        "records": records,
    }


def _normalize_systems(
    systems: Sequence[SystemReportInput],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, tuple[dict[str, object], ...]],
]:
    if not isinstance(systems, Sequence) or isinstance(systems, (str, bytes)):
        raise TypeError("systems must be a sequence")
    if any(not isinstance(item, SystemReportInput) for item in systems):
        raise TypeError("every system must be a SystemReportInput")
    names = [item.name for item in systems]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate system names: {duplicates}")
    if set(names) != set(REQUIRED_SYSTEMS) or len(names) != len(REQUIRED_SYSTEMS):
        raise ValueError(
            f"systems must contain exactly {list(REQUIRED_SYSTEMS)}"
        )
    by_name = {item.name: item for item in systems}
    normalized: dict[str, dict[str, object]] = {}
    rows: dict[str, tuple[dict[str, object], ...]] = {}
    for name in REQUIRED_SYSTEMS:
        item = by_name[name]
        summary, system_rows = _validate_evaluation(item.evaluation, name)
        provenance = _validate_provenance(item.provenance, name)
        if any(row["scene_id"] != provenance["scene_id"] for row in system_rows):
            raise ValueError(
                f"{name} row scene_id does not match provenance scene_id"
            )
        if name == "baseline_imported":
            if item.worker_summary is not None:
                raise ValueError("baseline_imported worker_summary must be None")
            worker = None
        else:
            if item.worker_summary is None:
                raise ValueError(f"{name} requires a worker summary")
            worker = _validate_worker(
                item.worker_summary, name, str(provenance["scene_id"])
            )
        normalized[name] = {
            "summary": summary,
            "provenance": provenance,
            "worker": worker,
        }
        rows[name] = system_rows

    baseline_rows = {str(row["sample_id"]): row for row in rows["baseline_imported"]}
    baseline_scene = normalized["baseline_imported"]["provenance"]["scene_id"]
    for name in REQUIRED_SYSTEMS:
        if normalized[name]["provenance"]["scene_id"] != baseline_scene:
            raise ValueError(f"{name} provenance scene identity mismatch")
        current = {str(row["sample_id"]): row for row in rows[name]}
        if set(current) != set(baseline_rows):
            raise ValueError(f"{name} sample IDs do not match imported baseline")
        for sample_id in sorted(current):
            for field in _ROW_METADATA[1:]:
                if current[sample_id][field] != baseline_rows[sample_id][field]:
                    raise ValueError(
                        f"{name} scene/camera/frame/time identity mismatch "
                        f"for sample {sample_id!r}"
                    )

    on_provenance = normalized["joint_conditioned_on"]["provenance"]
    off_provenance = normalized["joint_conditioned_off"]["provenance"]
    if (
        on_provenance["checkpoint_sha256"]
        != off_provenance["checkpoint_sha256"]
        or on_provenance["checkpoint_generation"]
        != off_provenance["checkpoint_generation"]
    ):
        raise ValueError(
            "joint_conditioned_on/off must use the same checkpoint SHA-256 "
            "and checkpoint generation"
        )
    trained_workers = [
        normalized[name]["worker"] for name in REQUIRED_SYSTEMS[1:]
    ]
    if len({worker["config_sha256"] for worker in trained_workers}) != 1:
        raise ValueError("trained worker config identity mismatch")
    if len({worker["manifest_sha256"] for worker in trained_workers}) != 1:
        raise ValueError("trained worker manifest identity mismatch")
    return normalized, rows


def _decide_normalized(
    systems: Mapping[str, Mapping[str, object]],
    paired_condition: Mapping[str, object],
) -> PilotDecision:
    """Apply only the four approved long-training gates."""
    if set(systems) != set(REQUIRED_SYSTEMS):
        return PilotDecision(False, ("required five-system comparison is incomplete",))
    reasons: list[str] = []
    try:
        baseline = systems["baseline_imported"]["summary"]
        joint = systems["joint_conditioned_on"]
        joint_summary = joint["summary"]
        off_summary = systems["joint_conditioned_off"]["summary"]
        psnr_drop = (
            baseline["rgb_psnr"]["mean"] - joint_summary["rgb_psnr"]["mean"]
        )
        ssim_drop = (
            baseline["rgb_ssim"]["mean"] - joint_summary["rgb_ssim"]["mean"]
        )
        if (
            joint_summary["rgb_psnr"]["mean"]
            < baseline["rgb_psnr"]["mean"] - 0.5
        ):
            reasons.append(
                "joint_conditioned_on PSNR drop exceeds 0.5 dB"
            )
        if (
            joint_summary["rgb_ssim"]["mean"]
            < baseline["rgb_ssim"]["mean"] - 0.01
        ):
            reasons.append(
                "joint_conditioned_on SSIM drop exceeds 0.01"
            )
        if not (
            joint_summary["audio_total"]["mean"]
            < off_summary["audio_total"]["mean"]
        ):
            reasons.append(
                "joint_conditioned_on audio_total mean is not strictly lower "
                "than joint_conditioned_off"
            )
        finite_gate_values = (
            psnr_drop,
            ssim_drop,
            joint_summary["audio_total"]["mean"],
            off_summary["audio_total"]["mean"],
            paired_condition["median"],
        )
        if not all(math.isfinite(float(value)) for value in finite_gate_values):
            raise ValueError("nonfinite decision input")
        if not (paired_condition["median"] < 0):
            reasons.append(
                "paired audio_total median delta is not strictly negative"
            )
        worker = joint["worker"]
        if worker is None or not (
            worker["max_audio_to_visual_grad_norm"] > 0
        ):
            reasons.append(
                "joint_conditioned_on max_audio_to_visual_grad_norm "
                "is not strictly positive"
            )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return PilotDecision(False, ("comparison data is missing, nonfinite, or inconsistent",))
    return PilotDecision(not reasons, tuple(reasons))


def decide_long_training(
    systems: Sequence[SystemReportInput],
) -> PilotDecision:
    """Validate inputs and apply exactly the approved long-training gates."""
    try:
        normalized, rows = _normalize_systems(systems)
        paired = paired_audio_deltas(
            rows["joint_conditioned_on"], rows["joint_conditioned_off"]
        )
        return _decide_normalized(_system_output(normalized), paired)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return PilotDecision(
            False, ("comparison data is missing, nonfinite, or inconsistent",)
        )


def _system_output(
    normalized: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    baseline = normalized["baseline_imported"]["summary"]
    result: dict[str, dict[str, object]] = {}
    for name in REQUIRED_SYSTEMS:
        source = normalized[name]
        summary = source["summary"]
        deltas = {
            metric: {
                statistic: summary[metric][statistic] - baseline[metric][statistic]
                for statistic in STATISTICS
            }
            for metric in METRIC_NAMES
        }
        result[name] = {
            "summary": summary,
            "deltas_vs_baseline": deltas,
            "metric_directions": dict(METRIC_DIRECTIONS),
            "visual_feasible": visual_feasible(summary, baseline, 0.5, 0.01),
            "provenance": source["provenance"],
            "worker": source["worker"],
        }
    return result


def _descriptive_comparisons(
    systems: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    condition_off = systems["condition_off"]["summary"]["audio_total"]["mean"]
    joint_off = systems["joint_conditioned_off"]["summary"]["audio_total"]["mean"]
    return {
        "condition_off_vs_joint_conditioned_off": {
            "condition_off_audio_total_mean": condition_off,
            "joint_conditioned_off_audio_total_mean": joint_off,
            "audio_total_mean_delta": condition_off - joint_off,
            "delta_definition": "condition_off - joint_conditioned_off",
            "interpretation": "negative favors separately trained condition_off",
            "decision_gate": False,
        }
    }


def _json_text(value: object, *, indent: int | None = 2) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
        allow_nan=False,
    ) + "\n"


def _csv_text(systems: Mapping[str, Mapping[str, object]]) -> str:
    fields = ["system"]
    for metric in METRIC_NAMES:
        for statistic in STATISTICS:
            fields.extend(
                (f"{metric}_{statistic}", f"{metric}_{statistic}_delta_vs_baseline")
            )
    fields.extend(
        (
            "visual_feasible",
            "completed_joint_steps",
            "best_step",
            "stop_reason",
            "max_audio_to_visual_grad_norm",
            "checkpoint_sha256",
            "checkpoint_generation",
            "condition_enabled",
        )
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for name in REQUIRED_SYSTEMS:
        system = systems[name]
        row: dict[str, object] = {"system": name}
        for metric in METRIC_NAMES:
            for statistic in STATISTICS:
                row[f"{metric}_{statistic}"] = system["summary"][metric][statistic]
                row[f"{metric}_{statistic}_delta_vs_baseline"] = (
                    system["deltas_vs_baseline"][metric][statistic]
                )
        worker = system["worker"]
        row.update(
            visual_feasible=str(system["visual_feasible"]).lower(),
            completed_joint_steps="" if worker is None else worker["completed_joint_steps"],
            best_step="" if worker is None else worker["best_step"],
            stop_reason="" if worker is None else worker["stop_reason"],
            max_audio_to_visual_grad_norm=(
                "" if worker is None else worker["max_audio_to_visual_grad_norm"]
            ),
            checkpoint_sha256=system["provenance"]["checkpoint_sha256"],
            checkpoint_generation=system["provenance"]["checkpoint_generation"],
            condition_enabled=str(
                system["provenance"]["condition_enabled"]
            ).lower(),
        )
        writer.writerow(row)
    return stream.getvalue()


def _markdown_text(
    systems: Mapping[str, Mapping[str, object]],
    paired: Mapping[str, object],
    descriptive: Mapping[str, Mapping[str, object]],
    decision: PilotDecision,
) -> str:
    status = "READY" if decision.ready else "NOT READY"
    lines = [
        f"# Pilot comparison: {status}",
        "",
        "Metric directions: RGB PSNR and RGB SSIM — Higher is better. "
        "All audio metrics and RGB L1 — Lower is better. "
        "Every baseline delta is system minus imported baseline.",
        "",
        "| System | Audio total mean | Δ baseline | PSNR mean | Δ baseline | "
        "SSIM mean | Δ baseline | Visual feasible | Stop |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for name in REQUIRED_SYSTEMS:
        system = systems[name]
        summary = system["summary"]
        deltas = system["deltas_vs_baseline"]
        worker = system["worker"]
        stop = (
            "imported"
            if worker is None
            else f"{worker['stop_reason']} at {worker['completed_joint_steps']} joint steps"
        )
        lines.append(
            f"| {name} | {summary['audio_total']['mean']:.12g} | "
            f"{deltas['audio_total']['mean']:.12g} | "
            f"{summary['rgb_psnr']['mean']:.12g} | "
            f"{deltas['rgb_psnr']['mean']:.12g} | "
            f"{summary['rgb_ssim']['mean']:.12g} | "
            f"{deltas['rgb_ssim']['mean']:.12g} | "
            f"{str(system['visual_feasible']).lower()} | {stop} |"
        )
    joint = systems["joint_conditioned_on"]
    frozen = systems["frozen_visual_on"]
    separate_comparison = descriptive["condition_off_vs_joint_conditioned_off"]
    lines.extend(
        [
            "",
            "## Visual constraint and paired conditioning",
            "",
            "The joint conditioned system must remain within a 0.5 dB PSNR drop "
            "and a 0.01 SSIM drop from baseline.",
            f"Paired on-minus-off audio_total: mean {paired['mean']:.12g}, "
            f"median {paired['median']:.12g}, std {paired['std']:.12g}, "
            f"n={paired['sample_count']}. Negative means conditioning improves.",
            "",
            "## Descriptive comparisons",
            "",
            f"Frozen visual vs joint audio_total mean: "
            f"{frozen['summary']['audio_total']['mean']:.12g} vs "
            f"{joint['summary']['audio_total']['mean']:.12g}.",
            "Separately trained condition_off vs joint_conditioned_off: "
            f"audio_total mean "
            f"{separate_comparison['condition_off_audio_total_mean']:.12g} vs "
            f"{separate_comparison['joint_conditioned_off_audio_total_mean']:.12g}; "
            "signed delta (condition_off - joint_conditioned_off) "
            f"{separate_comparison['audio_total_mean_delta']:.12g}. "
            "Negative favors separately trained condition_off. "
            "This is descriptive and is not a decision gate.",
            "",
            f"## Decision: {status}",
            "",
        ]
    )
    if decision.reasons:
        lines.extend(f"- {reason}" for reason in decision.reasons)
    else:
        lines.append("All approved acceptance gates passed.")
    lines.append("")
    return "\n".join(lines)


def _stage(path: Path, content: str) -> Path:
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(output_dir: Path, contents: Mapping[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = output_dir.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("report output must be a non-symlink directory")
    ordered = (
        "paired_condition_deltas.jsonl",
        "comparison.csv",
        "comparison.json",
        "comparison.md",
    )
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    installed: list[str] = []
    try:
        for name in ordered:
            destination = output_dir / name
            try:
                existing = destination.lstat()
            except FileNotFoundError:
                pass
            else:
                if (
                    stat.S_ISLNK(existing.st_mode)
                    or not stat.S_ISREG(existing.st_mode)
                    or existing.st_nlink != 1
                ):
                    raise ValueError(
                        f"report destination must be a single-link regular file: {name}"
                    )
            staged[name] = _stage(destination, contents[name])
            if destination.exists():
                backup = _stage(destination, destination.read_text(encoding="utf-8"))
                backups[name] = backup
        _fsync_directory(output_dir)
        for name in ordered:
            os.replace(staged[name], output_dir / name)
            installed.append(name)
        _fsync_directory(output_dir)
    except BaseException:
        for name in reversed(installed):
            destination = output_dir / name
            backup = backups.get(name)
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
        _fsync_directory(output_dir)
        raise
    finally:
        for temporary in (*staged.values(), *backups.values()):
            temporary.unlink(missing_ok=True)


def build_comparison(
    systems: Sequence[SystemReportInput],
    output_dir: str | Path,
) -> ComparisonResult:
    """Validate five systems, decide acceptance, and atomically publish reports."""
    normalized, rows = _normalize_systems(systems)
    paired = paired_audio_deltas(
        rows["joint_conditioned_on"], rows["joint_conditioned_off"]
    )
    output_systems = _system_output(normalized)
    descriptive = _descriptive_comparisons(output_systems)
    decision = _decide_normalized(output_systems, paired)
    result = ComparisonResult(
        systems=output_systems,
        rows=rows,
        paired_condition=paired,
        descriptive_comparisons=descriptive,
        decision=decision,
    )
    json_payload: dict[str, Any] = {
        "schema": "avgaussianv2.pilot-comparison",
        "version": 1,
        "systems": output_systems,
        "rows": {name: list(rows[name]) for name in REQUIRED_SYSTEMS},
        "paired_condition": paired,
        "descriptive_comparisons": descriptive,
        "decision": {
            "ready": decision.ready,
            "reasons": list(decision.reasons),
        },
    }
    paired_jsonl = "".join(
        _json_text(record, indent=None) for record in paired["records"]
    )
    _publish(
        Path(output_dir),
        {
            "comparison.json": _json_text(json_payload),
            "comparison.csv": _csv_text(output_systems),
            "comparison.md": _markdown_text(
                output_systems, paired, descriptive, decision
            ),
            "paired_condition_deltas.jsonl": paired_jsonl,
        },
    )
    return result


__all__ = [
    "ComparisonResult",
    "PilotDecision",
    "REQUIRED_SYSTEMS",
    "SystemReportInput",
    "build_comparison",
    "decide_long_training",
    "paired_audio_deltas",
]

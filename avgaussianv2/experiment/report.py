"""Strict, deterministic comparison reports for the five-system pilot."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from avgaussianv2.experiment.checkpoint import (
    PilotCompatibility,
    hash_index_manifest,
    inspect_pilot_checkpoint,
)
from avgaussianv2.experiment.contracts import (
    EvaluationResult,
    PilotConfig,
    VariantIndices,
)
from avgaussianv2.experiment.evaluation import METRIC_NAMES
from avgaussianv2.experiment.metrics import aggregate_metrics
from avgaussianv2.experiment.selection import (
    BestSelector,
    EarlyStopper,
    visual_feasible,
)


REQUIRED_SYSTEMS = (
    "baseline_imported",
    "joint_conditioned_on",
    "joint_conditioned_off",
    "frozen_visual_on",
    "condition_off",
)
PSNR_TOLERANCE_DB = 0.5
SSIM_TOLERANCE = 0.01
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
class EvaluationProvenance:
    scene_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_generation: int
    run_fingerprint: Mapping[str, object]
    evaluation_indices_hash: str
    condition_enabled: bool
    evaluation_run_id: str
    compatibility: PilotCompatibility | None
    variant_indices: VariantIndices | None
    pilot_config: PilotConfig | None


@dataclass(frozen=True)
class EvaluationArtifactProvenance:
    metrics_per_sample_path: Path
    metrics_per_sample_sha256: str
    metrics_summary_path: Path
    metrics_summary_sha256: str
    manifest_sha256: str
    system_name: str
    condition_enabled: bool
    count: int
    evaluation_indices: tuple[int, ...]
    evaluation_indices_hash: str
    checkpoint: EvaluationProvenance
    evaluation_run_id: str


@dataclass(frozen=True)
class WorkerArtifactProvenance:
    worker_summary_path: Path
    worker_summary_sha256: str
    latest_checkpoint_path: Path
    latest_checkpoint_sha256: str
    latest_checkpoint_generation: int
    run_fingerprint: Mapping[str, object]


@dataclass(frozen=True)
class CheckpointArtifactIdentity:
    checkpoint_kind: str
    generation: int
    run_fingerprint: Mapping[str, object]
    best_step: int
    visual_baseline: Mapping[str, object]
    psnr_tolerance_db: float
    ssim_tolerance: float
    best_evaluation_summary: Mapping[str, object]


@dataclass(frozen=True)
class SystemReportInput:
    name: str
    evaluation: EvaluationResult
    worker_summary: Mapping[str, object] | None
    provenance: EvaluationArtifactProvenance
    worker_provenance: WorkerArtifactProvenance | None


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
    content_digest: str
    generation_path: Path
    committed: bool
    durability_warnings: tuple[str, ...]

    @property
    def pair_records(self) -> tuple[dict[str, object], ...]:
        return tuple(self.paired_condition["records"])


@dataclass(frozen=True)
class ResolvedReport:
    generation_path: Path
    durability_warnings: tuple[str, ...]


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


def _canonical_json(value: object, name: str) -> object:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be strict JSON-safe data") from error


def _run_fingerprint(value: object, name: str) -> dict[str, object]:
    mapping = _exact(
        value, {"algorithm", "sha256", "inputs"}, name
    )
    algorithm = _text(mapping["algorithm"], f"{name}.algorithm")
    digest = _digest(mapping["sha256"], f"{name}.sha256")
    inputs = _canonical_json(mapping["inputs"], f"{name}.inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError(f"{name}.inputs must be a mapping")
    encoded = json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError(f"{name}.sha256 does not match inputs")
    return {"algorithm": algorithm, "sha256": digest, "inputs": inputs}


def _canonical_pilot_config(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, PilotConfig):
        raise TypeError(f"{name} must be a PilotConfig")
    for field_name, field in PilotConfig.__dataclass_fields__.items():
        field_value = getattr(value, field_name)
        if type(field_value) is not type(field.default):
            raise TypeError(
                f"{name}.{field_name} must have exact type "
                f"{type(field.default).__name__}"
            )
    value.validate()
    canonical = _canonical_json(asdict(value), name)
    if not isinstance(canonical, dict):  # pragma: no cover - dataclass invariant
        raise TypeError(f"{name} must canonicalize to a mapping")
    return canonical


def _fingerprint_pilot_config(
    fingerprint: Mapping[str, object], name: str
) -> Mapping[str, object]:
    inputs = fingerprint["inputs"]
    if not isinstance(inputs, Mapping):  # guarded by _run_fingerprint
        raise TypeError(f"{name}.inputs must be a mapping")
    config = _exact(
        inputs.get("pilot_config"),
        set(PilotConfig.__dataclass_fields__),
        f"{name}.inputs.pilot_config",
    )
    return config


def _require_matching_pilot_config(
    canonical: Mapping[str, object],
    fingerprint: Mapping[str, object],
    name: str,
) -> None:
    inspected = _fingerprint_pilot_config(fingerprint, name)
    if not _strict_json_equal(dict(canonical), dict(inspected)):
        raise ValueError(f"{name}.inputs.pilot_config mismatch")


def build_evaluation_run_id(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    checkpoint_generation: int,
    run_fingerprint: Mapping[str, object],
    evaluation_indices_hash: str,
) -> str:
    """Return the condition-independent identity of one checkpoint evaluation."""
    if not isinstance(checkpoint_path, (str, Path)):
        raise TypeError("checkpoint_path must be path-like")
    payload = {
        "checkpoint_sha256": _digest(
            checkpoint_sha256, "checkpoint_sha256"
        ),
        "checkpoint_generation": _integer(
            checkpoint_generation, "checkpoint_generation"
        ),
        "run_fingerprint": _run_fingerprint(
            run_fingerprint, "run_fingerprint"
        ),
        "evaluation_indices_hash": _digest(
            evaluation_indices_hash, "evaluation_indices_hash"
        ),
    }
    return hash_index_manifest(payload)


def build_evaluation_manifest_sha256(
    *,
    metrics_per_sample_sha256: str,
    metrics_summary_sha256: str,
    system_name: str,
    condition_enabled: bool,
    count: int,
    evaluation_indices: Sequence[int],
    evaluation_indices_hash: str,
    checkpoint_sha256: str,
    checkpoint_generation: int,
    evaluation_run_id: str,
) -> str:
    if not isinstance(condition_enabled, bool):
        raise TypeError("condition_enabled must be boolean")
    indices = tuple(
        _integer(index, f"evaluation_indices[{position}]")
        for position, index in enumerate(evaluation_indices)
    )
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("evaluation_indices must be nonempty and unique")
    payload = {
        "metrics_per_sample_sha256": _digest(
            metrics_per_sample_sha256, "metrics_per_sample_sha256"
        ),
        "metrics_summary_sha256": _digest(
            metrics_summary_sha256, "metrics_summary_sha256"
        ),
        "system_name": _text(system_name, "system_name"),
        "condition_enabled": condition_enabled,
        "count": _integer(count, "count", minimum=1),
        "evaluation_indices": list(indices),
        "evaluation_indices_hash": _digest(
            evaluation_indices_hash, "evaluation_indices_hash"
        ),
        "checkpoint_sha256": _digest(
            checkpoint_sha256, "checkpoint_sha256"
        ),
        "checkpoint_generation": _integer(
            checkpoint_generation, "checkpoint_generation"
        ),
        "evaluation_run_id": _digest(
            evaluation_run_id, "evaluation_run_id"
        ),
    }
    return hash_index_manifest(payload)


def _artifact_identity_tuple(path: Path) -> tuple[int, int, int, int, int]:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError(
            f"evaluation artifact must be a single-link non-symlink regular file: {path}"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_verified_artifact(
    path: Path, expected_sha256: str, *, limit: int, name: str
) -> bytes:
    before = _artifact_identity_tuple(path)
    if before[2] > limit:
        raise ValueError(f"{name} exceeds {limit} byte limit")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != before[:2]:
            raise ValueError(f"{name} identity changed while opening")
        total = 0
        while block := os.read(descriptor, min(1024 * 1024, limit + 1 - total)):
            total += len(block)
            if total > limit:
                raise ValueError(f"{name} exceeds {limit} byte limit")
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != before:
            raise ValueError(f"{name} changed while reading")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != _digest(expected_sha256, f"{name} SHA-256"):
        raise ValueError(f"{name} SHA-256 mismatch")
    return b"".join(chunks)


def resolve_pilot_checkpoint_identity(
    path: Path, provenance: EvaluationProvenance
) -> CheckpointArtifactIdentity:
    """Safely load only the checkpoint identity metadata needed by this report."""
    if provenance.compatibility is None or provenance.variant_indices is None:
        raise ValueError("pilot checkpoint provenance requires compatibility and indices")
    state = inspect_pilot_checkpoint(
        path,
        expected_compatibility=provenance.compatibility,
        indices=provenance.variant_indices,
        expected_run_fingerprint=provenance.run_fingerprint,
        active_resume=False,
    )
    if state.checkpoint_kind != "best":
        raise ValueError("pilot evaluation checkpoint must have kind 'best'")
    selector_state = state.selector.state_dict()
    return CheckpointArtifactIdentity(
        checkpoint_kind=state.checkpoint_kind,
        generation=state.generation,
        run_fingerprint=state.run_fingerprint,
        best_step=int(selector_state["best_step"]),
        visual_baseline=selector_state["visual_baseline"],
        psnr_tolerance_db=float(selector_state["psnr_tolerance_db"]),
        ssim_tolerance=float(selector_state["ssim_tolerance"]),
        best_evaluation_summary=state.best_evaluation_summary,
    )


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
        if _scalar(item, f"{name}.{key}") < 0:
            raise ValueError(f"{name}.{key} must be nonnegative")


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
    observed_stage_steps: list[tuple[str, int]] = []
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
        observed_stage_steps.append((str(stage), step))
        _integer(row["sample_index"], f"{system_name}.training_history[{index}].sample_index")
        if _scalar(
            row["total"], f"{system_name}.training_history[{index}].total"
        ) < 0:
            raise ValueError(
                f"{system_name}.training_history[{index}].total must be nonnegative"
            )
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
    expected_stage_steps = [
        ("warmup", step) for step in range(1, completed_warmup + 1)
    ] + [
        ("joint", step) for step in range(1, completed_joint + 1)
    ]
    if observed_stage_steps != expected_stage_steps:
        raise ValueError(
            f"{system_name}.training_history must contain contiguous warmup then "
            "joint stage steps matching completed counts"
        )

    validation_history = value["validation_history"]
    if not isinstance(validation_history, list):
        raise TypeError(f"{system_name}.validation_history must be a list")
    validation_steps: list[int] = []
    validated_validations: list[dict[str, object]] = []
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
        normalized_summary: dict[str, dict[str, float]] = {}
        for metric in METRIC_NAMES:
            aggregate = _exact(
                summary[metric],
                _AGGREGATE_FIELDS,
                f"{system_name}.validation_history[{index}].summary.{metric} aggregate",
            )
            normalized_summary[metric] = {}
            for statistic in STATISTICS:
                normalized_summary[metric][statistic] = _metric_value(
                    metric,
                    aggregate[statistic],
                    f"{system_name}.validation_history[{index}].summary."
                    f"{metric}.{statistic}",
                )
        validated_validations.append(
            {"step": step, "summary": normalized_summary}
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
    if selector["psnr_tolerance_db"] != PSNR_TOLERANCE_DB:
        raise ValueError(
            f"{system_name} selector PSNR tolerance must equal "
            f"{PSNR_TOLERANCE_DB}"
        )
    if selector["ssim_tolerance"] != SSIM_TOLERANCE:
        raise ValueError(
            f"{system_name} selector SSIM tolerance must equal {SSIM_TOLERANCE}"
        )
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
    replay_selector = BestSelector(
        selector["visual_baseline"],
        selector["psnr_tolerance_db"],
        selector["ssim_tolerance"],
    )
    replay_stopper = EarlyStopper(
        stopper["minimum_steps"],
        stopper["patience"],
        stopper["relative_delta"],
    )
    replay_should_stop = False
    for validation_index, validation in enumerate(validated_validations):
        replay_selector.consider(validation["step"], validation["summary"])
        replay_should_stop = replay_stopper.update(
            validation["step"],
            validation["summary"]["audio_total"]["mean"],
        )
        if (
            replay_should_stop
            and validation_index != len(validated_validations) - 1
        ):
            raise ValueError(
                f"{system_name} validation history continues after early stop"
            )
    if replay_selector.state_dict() != dict(selector):
        raise ValueError(f"{system_name} selector_state does not replay")
    if replay_stopper.state_dict() != dict(stopper):
        raise ValueError(f"{system_name} stopper_state does not replay")
    expected_stop_reason = "early_stop" if replay_should_stop else "max_steps"
    if stop_reason != expected_stop_reason:
        raise ValueError(f"{system_name} stop_reason does not replay")
    replay_best_step = replay_selector.best_step
    if replay_best_step != best_step:
        raise ValueError(f"{system_name} replayed best_step mismatch")
    best_validation = next(
        validation
        for validation in validated_validations
        if validation["step"] == best_step
    )
    quick_best_summary = best_validation["summary"]
    quick_best_visual_feasible = visual_feasible(
        quick_best_summary,
        selector["visual_baseline"],
        selector["psnr_tolerance_db"],
        selector["ssim_tolerance"],
    )
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
    if (
        checkpoint_io["save_count"] + checkpoint_io["failed_save_count"]
        != checkpoint_io["save_attempt_count"]
    ):
        raise ValueError(f"{system_name}.checkpoint_io save counters are incoherent")
    if (
        checkpoint_io["backup_copy_count"]
        + checkpoint_io["backup_copy_failure_count"]
        != checkpoint_io["backup_copy_attempt_count"]
    ):
        raise ValueError(f"{system_name}.checkpoint_io backup counters are incoherent")
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
        "quick_best_summary": quick_best_summary,
        "quick_visual_baseline": selector["visual_baseline"],
        "quick_psnr_tolerance_db": selector["psnr_tolerance_db"],
        "quick_ssim_tolerance": selector["ssim_tolerance"],
        "quick_best_visual_feasible": quick_best_visual_feasible,
    }


def _validate_provenance(
    value: object,
    system_name: str,
    *,
    hash_cache: dict[Path, tuple[str, tuple[int, int, int, int, int]]],
    identity_cache: dict[Path, CheckpointArtifactIdentity],
    checkpoint_identity_resolver: Callable[
        [Path, EvaluationProvenance], CheckpointArtifactIdentity
    ],
) -> dict[str, object]:
    if not isinstance(value, EvaluationProvenance):
        raise TypeError(
            f"{system_name} provenance must be an EvaluationProvenance"
        )
    condition = value.condition_enabled
    if not isinstance(condition, bool):
        raise TypeError(f"{system_name}.condition_enabled must be boolean")
    if condition != _EXPECTED_CONDITION[system_name]:
        raise ValueError(f"{system_name} condition_enabled mismatch")
    scene_id = _text(value.scene_id, f"{system_name}.scene_id")
    if not isinstance(value.checkpoint_path, Path):
        raise TypeError(f"{system_name}.checkpoint_path must be a Path")
    artifact_path = Path(os.path.abspath(value.checkpoint_path))
    expected_sha = _digest(
        value.checkpoint_sha256, f"{system_name}.checkpoint_sha256"
    )
    before_identity = _artifact_identity_tuple(artifact_path)
    resolved_path = artifact_path.resolve(strict=True)
    cached = hash_cache.get(resolved_path)
    if cached is None:
        descriptor = os.open(
            artifact_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != before_identity[:2]:
                raise ValueError(f"{system_name} checkpoint identity changed")
            actual_sha = _hash_fd(descriptor)
            if actual_sha != expected_sha:
                raise ValueError(f"{system_name} checkpoint SHA-256 mismatch")
            pinned_path = Path(f"/proc/self/fd/{descriptor}")
            if system_name != "baseline_imported":
                identity = checkpoint_identity_resolver(pinned_path, value)
                if not isinstance(identity, CheckpointArtifactIdentity):
                    raise TypeError(
                        "checkpoint identity resolver must return "
                        "CheckpointArtifactIdentity"
                    )
                identity_cache[resolved_path] = identity
            after = os.fstat(descriptor)
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if after_identity != before_identity:
                raise ValueError(
                    f"{system_name} checkpoint changed during pinned inspection"
                )
        finally:
            os.close(descriptor)
        hash_cache[resolved_path] = (actual_sha, before_identity)
    else:
        actual_sha, cached_identity = cached
        if before_identity != cached_identity:
            raise ValueError(f"{system_name} checkpoint identity changed between uses")
    if actual_sha != expected_sha:
        raise ValueError(f"{system_name} checkpoint SHA-256 mismatch")
    generation = _integer(
        value.checkpoint_generation,
        f"{system_name}.checkpoint_generation",
    )
    fingerprint = _run_fingerprint(
        value.run_fingerprint, f"{system_name}.run_fingerprint"
    )
    indices_hash = _digest(
        value.evaluation_indices_hash,
        f"{system_name}.evaluation_indices_hash",
    )
    expected_run_id = build_evaluation_run_id(
        checkpoint_path=resolved_path,
        checkpoint_sha256=actual_sha,
        checkpoint_generation=generation,
        run_fingerprint=fingerprint,
        evaluation_indices_hash=indices_hash,
    )
    run_id = _digest(value.evaluation_run_id, f"{system_name}.evaluation_run_id")
    if run_id != expected_run_id:
        raise ValueError(f"{system_name} evaluation_run_id mismatch")
    if system_name != "baseline_imported":
        canonical_pilot_config = _canonical_pilot_config(
            value.pilot_config, f"{system_name}.pilot_config"
        )
        _require_matching_pilot_config(
            canonical_pilot_config,
            fingerprint,
            f"{system_name}.run_fingerprint",
        )
        if value.pilot_config.psnr_tolerance_db != PSNR_TOLERANCE_DB:
            raise ValueError(
                f"{system_name} pilot PSNR tolerance must equal {PSNR_TOLERANCE_DB}"
            )
        if value.pilot_config.ssim_tolerance != SSIM_TOLERANCE:
            raise ValueError(
                f"{system_name} pilot SSIM tolerance must equal {SSIM_TOLERANCE}"
            )
        identity = identity_cache.get(resolved_path)
        if identity is None:
            raise RuntimeError("checkpoint identity cache is incomplete")
        if _artifact_identity_tuple(artifact_path) != hash_cache[resolved_path][1]:
            raise ValueError(
                f"{system_name} checkpoint changed during metadata inspection"
            )
        if identity.checkpoint_kind != "best":
            raise ValueError(f"{system_name} checkpoint kind must be best")
        if identity.generation != generation:
            raise ValueError(f"{system_name} checkpoint generation mismatch")
        if _run_fingerprint(
            identity.run_fingerprint, "resolved checkpoint run_fingerprint"
        ) != fingerprint:
            raise ValueError(f"{system_name} checkpoint run_fingerprint mismatch")
        inspected_fingerprint = _run_fingerprint(
            identity.run_fingerprint, "resolved checkpoint run_fingerprint"
        )
        _require_matching_pilot_config(
            canonical_pilot_config,
            inspected_fingerprint,
            f"{system_name} inspected best run_fingerprint",
        )
    result = {
        "scene_id": scene_id,
        "checkpoint_path": str(resolved_path),
        "checkpoint_sha256": actual_sha,
        "checkpoint_generation": generation,
        "run_fingerprint": fingerprint,
        "evaluation_indices_hash": indices_hash,
        "condition_enabled": condition,
        "evaluation_run_id": run_id,
    }
    if system_name != "baseline_imported":
        result["_checkpoint_identity"] = identity
        result["_checkpoint_best_step"] = identity.best_step
        result["_checkpoint_visual_baseline"] = identity.visual_baseline
        result["_checkpoint_psnr_tolerance_db"] = identity.psnr_tolerance_db
        result["_checkpoint_ssim_tolerance"] = identity.ssim_tolerance
    return result


def _load_evaluation_artifact(
    value: object, system_name: str
) -> tuple[EvaluationResult, EvaluationProvenance]:
    if not isinstance(value, EvaluationArtifactProvenance):
        raise TypeError(
            f"{system_name} provenance must be an "
            "EvaluationArtifactProvenance"
        )
    if value.system_name != system_name:
        raise ValueError(f"{system_name} evaluation artifact system_name mismatch")
    if value.condition_enabled != _EXPECTED_CONDITION[system_name]:
        raise ValueError(f"{system_name} evaluation artifact condition mismatch")
    if value.condition_enabled != value.checkpoint.condition_enabled:
        raise ValueError(f"{system_name} artifact/checkpoint condition mismatch")
    count = _integer(value.count, f"{system_name} artifact count", minimum=1)
    indices = tuple(
        _integer(index, f"{system_name}.evaluation_indices[{position}]")
        for position, index in enumerate(value.evaluation_indices)
    )
    if len(indices) != count or len(indices) != len(set(indices)):
        raise ValueError(f"{system_name} evaluation indices/count mismatch")
    indices_hash = _digest(
        value.evaluation_indices_hash,
        f"{system_name}.evaluation_indices_hash",
    )
    if hash_index_manifest(list(indices)) != indices_hash:
        raise ValueError(f"{system_name} evaluation indices hash mismatch")
    if indices_hash != value.checkpoint.evaluation_indices_hash:
        raise ValueError(f"{system_name} artifact/checkpoint indices hash mismatch")
    if value.evaluation_run_id != value.checkpoint.evaluation_run_id:
        raise ValueError(f"{system_name} artifact/checkpoint run ID mismatch")
    if not isinstance(value.metrics_per_sample_path, Path) or not isinstance(
        value.metrics_summary_path, Path
    ):
        raise TypeError(f"{system_name} metric artifact paths must be Paths")
    rows_data = _read_verified_artifact(
        value.metrics_per_sample_path,
        value.metrics_per_sample_sha256,
        limit=256 * 1024 * 1024,
        name=f"{system_name} metrics_per_sample.jsonl",
    )
    summary_data = _read_verified_artifact(
        value.metrics_summary_path,
        value.metrics_summary_sha256,
        limit=16 * 1024 * 1024,
        name=f"{system_name} metrics_summary.json",
    )
    try:
        row_lines = rows_data.decode("utf-8").splitlines()
        if len(row_lines) != count or any(not line for line in row_lines):
            raise ValueError(f"{system_name} metrics JSONL row count mismatch")
        rows = tuple(
            json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {token}")
                ),
            )
            for line in row_lines
        )
        summary = json.loads(
            summary_data.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{system_name} metric artifacts are invalid JSON") from error
    expected_manifest = build_evaluation_manifest_sha256(
        metrics_per_sample_sha256=value.metrics_per_sample_sha256,
        metrics_summary_sha256=value.metrics_summary_sha256,
        system_name=system_name,
        condition_enabled=value.condition_enabled,
        count=count,
        evaluation_indices=indices,
        evaluation_indices_hash=indices_hash,
        checkpoint_sha256=value.checkpoint.checkpoint_sha256,
        checkpoint_generation=value.checkpoint.checkpoint_generation,
        evaluation_run_id=value.evaluation_run_id,
    )
    if _digest(value.manifest_sha256, f"{system_name}.manifest_sha256") != expected_manifest:
        raise ValueError(f"{system_name} evaluation manifest hash mismatch")
    return EvaluationResult(system_name, count, rows, summary), value.checkpoint


def _load_worker_summary_artifact(
    value: WorkerArtifactProvenance,
    system_name: str,
    cache: dict[Path, tuple[str, tuple[int, int, int, int, int], dict[str, object]]],
) -> tuple[dict[str, object], Path]:
    if not isinstance(value.worker_summary_path, Path):
        raise TypeError(f"{system_name} worker_summary_path must be a Path")
    path = Path(os.path.abspath(value.worker_summary_path))
    expected_sha = _digest(
        value.worker_summary_sha256,
        f"{system_name}.worker_summary_sha256",
    )
    identity = _artifact_identity_tuple(path)
    resolved = path.resolve(strict=True)
    cached = cache.get(resolved)
    if cached is None:
        data = _read_verified_artifact(
            path,
            expected_sha,
            limit=64 * 1024 * 1024,
            name=f"{system_name} worker_summary.json",
        )
        try:
            parsed = json.loads(
                data.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {token}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{system_name} worker_summary.json is invalid JSON"
            ) from error
        if not isinstance(parsed, dict):
            raise TypeError(f"{system_name} worker_summary.json must be an object")
        cache[resolved] = (expected_sha, identity, parsed)
    else:
        cached_sha, cached_identity, parsed = cached
        if cached_sha != expected_sha or cached_identity != identity:
            raise ValueError(
                f"{system_name} worker summary identity disagrees between uses"
            )
    return parsed, resolved


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _load_latest_checkpoint_artifact(
    value: WorkerArtifactProvenance,
    checkpoint: EvaluationProvenance,
    system_name: str,
    cache: dict[
        Path,
        tuple[
            str,
            tuple[int, int, int, int, int],
            object,
        ],
    ],
) -> tuple[object, Path]:
    if not isinstance(value.latest_checkpoint_path, Path):
        raise TypeError(f"{system_name} latest_checkpoint_path must be a Path")
    if checkpoint.compatibility is None or checkpoint.variant_indices is None:
        raise ValueError(f"{system_name} latest checkpoint expectations are incomplete")
    path = Path(os.path.abspath(value.latest_checkpoint_path))
    expected_sha = _digest(
        value.latest_checkpoint_sha256,
        f"{system_name}.latest_checkpoint_sha256",
    )
    generation = _integer(
        value.latest_checkpoint_generation,
        f"{system_name}.latest_checkpoint_generation",
    )
    run_fingerprint = _run_fingerprint(
        value.run_fingerprint, f"{system_name}.worker run_fingerprint"
    )
    checkpoint_fingerprint = _run_fingerprint(
        checkpoint.run_fingerprint, f"{system_name}.checkpoint run_fingerprint"
    )
    if not _strict_json_equal(run_fingerprint, checkpoint_fingerprint):
        raise ValueError(f"{system_name} latest/best run_fingerprint mismatch")
    canonical_pilot_config = _canonical_pilot_config(
        checkpoint.pilot_config, f"{system_name}.pilot_config"
    )
    _require_matching_pilot_config(
        canonical_pilot_config,
        run_fingerprint,
        f"{system_name}.worker run_fingerprint",
    )
    before = _artifact_identity_tuple(path)
    resolved = path.resolve(strict=True)
    cached = cache.get(resolved)
    if cached is None:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != before[:2]:
                raise ValueError(f"{system_name} latest checkpoint identity changed")
            actual_sha = _hash_fd(descriptor)
            if actual_sha != expected_sha:
                raise ValueError(f"{system_name} latest checkpoint SHA-256 mismatch")
            state = inspect_pilot_checkpoint(
                Path(f"/proc/self/fd/{descriptor}"),
                expected_compatibility=checkpoint.compatibility,
                indices=checkpoint.variant_indices,
                expected_run_fingerprint=run_fingerprint,
                active_resume=False,
            )
            after = os.fstat(descriptor)
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if after_identity != before:
                raise ValueError(
                    f"{system_name} latest checkpoint changed during inspection"
                )
        finally:
            os.close(descriptor)
        cache[resolved] = (actual_sha, before, state)
    else:
        cached_sha, cached_identity, state = cached
        if cached_sha != expected_sha or cached_identity != before:
            raise ValueError(
                f"{system_name} latest checkpoint identity disagrees between uses"
            )
    if state.checkpoint_kind != "latest":
        raise ValueError(f"{system_name} complete checkpoint must have kind latest")
    if state.stage != "complete":
        raise ValueError(f"{system_name} latest checkpoint stage must be complete")
    if state.generation != generation:
        raise ValueError(f"{system_name} latest checkpoint generation mismatch")
    inspected_fingerprint = _run_fingerprint(
        state.run_fingerprint, f"{system_name} inspected latest run_fingerprint"
    )
    if not _strict_json_equal(inspected_fingerprint, run_fingerprint):
        raise ValueError(f"{system_name} inspected latest run_fingerprint mismatch")
    _require_matching_pilot_config(
        canonical_pilot_config,
        inspected_fingerprint,
        f"{system_name} inspected latest run_fingerprint",
    )
    return state, resolved


def _bind_verified_worker_evidence(
    *,
    system_name: str,
    in_memory: Mapping[str, object],
    provenance: WorkerArtifactProvenance,
    checkpoint_provenance: EvaluationProvenance,
    best_identity: CheckpointArtifactIdentity,
    scene_id: str,
    worker_cache: dict[
        Path, tuple[str, tuple[int, int, int, int, int], dict[str, object]]
    ],
    latest_cache: dict[
        Path, tuple[str, tuple[int, int, int, int, int], object]
    ],
) -> tuple[dict[str, object], tuple[object, ...]]:
    actual_worker, worker_path = _load_worker_summary_artifact(
        provenance, system_name, worker_cache
    )
    if not _strict_json_equal(dict(in_memory), actual_worker):
        raise ValueError(
            f"{system_name} in-memory worker summary disagrees with hashed artifact"
        )
    worker = _validate_worker(actual_worker, system_name, scene_id)
    latest, latest_path = _load_latest_checkpoint_artifact(
        provenance,
        checkpoint_provenance,
        system_name,
        latest_cache,
    )
    expected_variant = _EXPECTED_VARIANTS[system_name]
    if latest.checkpoint_kind != "latest" or latest.stage != "complete":
        raise ValueError(f"{system_name} latest checkpoint is not complete")
    if checkpoint_provenance.compatibility.variant != expected_variant:
        raise ValueError(f"{system_name} latest checkpoint variant mismatch")
    exact_pairs = (
        ("completed_warmup_steps", latest.completed_warmup_steps),
        ("completed_joint_steps", latest.completed_joint_steps),
        ("best_step", latest.selector.best_step),
        ("stop_reason", latest.stop_reason),
        ("training_history", list(latest.training_history)),
        ("validation_history", list(latest.validation_history)),
        ("selector_state", latest.selector.state_dict()),
        ("stopper_state", latest.stopper.state_dict()),
    )
    for field, expected in exact_pairs:
        if actual_worker[field] != expected:
            raise ValueError(
                f"{system_name} worker/latest {field} mismatch"
            )
    if latest.best_generation != best_identity.generation:
        raise ValueError(f"{system_name} latest/best generation mismatch")
    if latest.best_evaluation_summary != best_identity.best_evaluation_summary:
        raise ValueError(f"{system_name} latest/best summary mismatch")
    gradients = [
        _scalar(
            row["audio_to_visual_grad_norm"],
            f"{system_name} latest joint gradient",
        )
        for row in latest.training_history
        if row["stage"] == "joint"
    ]
    if not gradients:
        raise ValueError(f"{system_name} latest checkpoint has no joint history")
    if latest.maximum_positive_audio_visual_gradient != max(gradients):
        raise ValueError(
            f"{system_name} latest checkpoint gradient summary mismatch"
        )
    worker["max_audio_to_visual_grad_norm"] = max(gradients)
    identity = (
        worker_path,
        provenance.worker_summary_sha256,
        latest_path,
        provenance.latest_checkpoint_sha256,
        provenance.latest_checkpoint_generation,
        _run_fingerprint(provenance.run_fingerprint, "worker run_fingerprint"),
    )
    return worker, identity


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
    *,
    checkpoint_identity_resolver: Callable[
        [Path, EvaluationProvenance], CheckpointArtifactIdentity
    ] = resolve_pilot_checkpoint_identity,
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
    hash_cache: dict[
        Path, tuple[str, tuple[int, int, int, int, int]]
    ] = {}
    identity_cache: dict[Path, CheckpointArtifactIdentity] = {}
    worker_cache: dict[
        Path, tuple[str, tuple[int, int, int, int, int], dict[str, object]]
    ] = {}
    latest_cache: dict[
        Path, tuple[str, tuple[int, int, int, int, int], object]
    ] = {}
    worker_identities: dict[str, tuple[object, ...]] = {}
    for name in REQUIRED_SYSTEMS:
        item = by_name[name]
        artifact_evaluation, checkpoint_provenance = _load_evaluation_artifact(
            item.provenance, name
        )
        if artifact_evaluation != item.evaluation:
            raise ValueError(
                f"{name} in-memory evaluation disagrees with metric artifacts"
            )
        summary, system_rows = _validate_evaluation(
            artifact_evaluation, name
        )
        provenance = _validate_provenance(
            checkpoint_provenance,
            name,
            hash_cache=hash_cache,
            identity_cache=identity_cache,
            checkpoint_identity_resolver=checkpoint_identity_resolver,
        )
        if any(row["scene_id"] != provenance["scene_id"] for row in system_rows):
            raise ValueError(
                f"{name} row scene_id does not match provenance scene_id"
            )
        if name == "baseline_imported":
            if item.worker_summary is not None:
                raise ValueError("baseline_imported worker_summary must be None")
            if item.worker_provenance is not None:
                raise ValueError("baseline_imported worker_provenance must be None")
            worker = None
        else:
            if item.worker_summary is None:
                raise ValueError(f"{name} requires a worker summary")
            if not isinstance(item.worker_provenance, WorkerArtifactProvenance):
                raise TypeError(
                    f"{name} requires a WorkerArtifactProvenance"
                )
            worker, worker_identity = _bind_verified_worker_evidence(
                system_name=name,
                in_memory=item.worker_summary,
                provenance=item.worker_provenance,
                checkpoint_provenance=checkpoint_provenance,
                best_identity=provenance["_checkpoint_identity"],
                scene_id=str(provenance["scene_id"]),
                worker_cache=worker_cache,
                latest_cache=latest_cache,
            )
            worker_identities[name] = worker_identity
            if worker["best_step"] != provenance["_checkpoint_best_step"]:
                raise ValueError(f"{name} worker/checkpoint best_step mismatch")
            if (
                worker["quick_visual_baseline"]
                != provenance["_checkpoint_visual_baseline"]
            ):
                raise ValueError(
                    f"{name} worker/checkpoint visual baseline mismatch"
                )
            if (
                worker["quick_psnr_tolerance_db"]
                != provenance["_checkpoint_psnr_tolerance_db"]
                or worker["quick_ssim_tolerance"]
                != provenance["_checkpoint_ssim_tolerance"]
            ):
                raise ValueError(
                    f"{name} worker/checkpoint visual tolerances mismatch"
                )
            for internal_field in (
                "_checkpoint_best_step",
                "_checkpoint_identity",
                "_checkpoint_visual_baseline",
                "_checkpoint_psnr_tolerance_db",
                "_checkpoint_ssim_tolerance",
            ):
                provenance.pop(internal_field)
        normalized[name] = {
            "summary": summary,
            "provenance": provenance,
            "worker": worker,
        }
        rows[name] = system_rows

    if (
        worker_identities["joint_conditioned_on"]
        != worker_identities["joint_conditioned_off"]
    ):
        raise ValueError(
            "joint conditioned on/off must share identical worker/latest artifacts"
        )

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
        on_provenance["checkpoint_path"]
        != off_provenance["checkpoint_path"]
        or
        on_provenance["checkpoint_sha256"]
        != off_provenance["checkpoint_sha256"]
        or on_provenance["checkpoint_generation"]
        != off_provenance["checkpoint_generation"]
        or on_provenance["run_fingerprint"]
        != off_provenance["run_fingerprint"]
        or on_provenance["evaluation_run_id"]
        != off_provenance["evaluation_run_id"]
    ):
        raise ValueError(
            "joint_conditioned_on/off must use the same actual checkpoint path, "
            "SHA-256, generation, run fingerprint, and evaluation_run_id"
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
        joint = systems["joint_conditioned_on"]
        joint_summary = joint["summary"]
        off_summary = systems["joint_conditioned_off"]["summary"]
        worker = joint["worker"]
        quick_summary = worker["quick_best_summary"]
        quick_baseline = worker["quick_visual_baseline"]
        psnr_tolerance = worker["quick_psnr_tolerance_db"]
        ssim_tolerance = worker["quick_ssim_tolerance"]
        psnr_drop = (
            quick_baseline["rgb_psnr"]["mean"]
            - quick_summary["rgb_psnr"]["mean"]
        )
        ssim_drop = (
            quick_baseline["rgb_ssim"]["mean"]
            - quick_summary["rgb_ssim"]["mean"]
        )
        if (
            quick_summary["rgb_psnr"]["mean"]
            < quick_baseline["rgb_psnr"]["mean"] - psnr_tolerance
        ):
            reasons.append(
                "joint_conditioned_on PSNR drop exceeds 0.5 dB"
            )
        if (
            quick_summary["rgb_ssim"]["mean"]
            < quick_baseline["rgb_ssim"]["mean"] - ssim_tolerance
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
            psnr_tolerance,
            ssim_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in finite_gate_values):
            raise ValueError("nonfinite decision input")
        if not (paired_condition["median"] < 0):
            reasons.append(
                "paired audio_total median delta is not strictly negative"
            )
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
            "acceptance_visual_feasible": (
                None
                if source["worker"] is None
                else source["worker"]["quick_best_visual_feasible"]
            ),
            "full_split_visual_feasible": visual_feasible(
                summary, baseline, PSNR_TOLERANCE_DB, SSIM_TOLERANCE
            ),
            "provenance": {
                key: value
                for key, value in source["provenance"].items()
                if key != "checkpoint_path"
            },
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
            "acceptance_visual_feasible",
            "full_split_visual_feasible",
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
            acceptance_visual_feasible=(
                ""
                if system["acceptance_visual_feasible"] is None
                else str(system["acceptance_visual_feasible"]).lower()
            ),
            full_split_visual_feasible=str(
                system["full_split_visual_feasible"]
            ).lower(),
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
        "SSIM mean | Δ baseline | Quick-best feasible | Full-split feasible | Stop |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|",
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
            f"{str(system['acceptance_visual_feasible']).lower()} | "
            f"{str(system['full_split_visual_feasible']).lower()} | {stop} |"
        )
    joint = systems["joint_conditioned_on"]
    frozen = systems["frozen_visual_on"]
    separate_comparison = descriptive["condition_off_vs_joint_conditioned_off"]
    lines.extend(
        [
            "",
            "## Visual constraint and paired conditioning",
            "",
            "The acceptance gate uses the joint worker's replayed quick-validation "
            "best step and its stored baseline/tolerances. Full-split visual "
            "feasibility is descriptive only and is never an additional gate.",
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


_REPORT_FILES = (
    "comparison.json",
    "comparison.csv",
    "comparison.md",
    "paired_condition_deltas.jsonl",
)
_GENERATION_MANIFEST = "generation_manifest.json"


def _open_secure_directory(path: Path, *, create: bool) -> tuple[int, Path]:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(
        "/",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        for part in absolute.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )


def _open_lock(output_fd: int) -> int:
    descriptor = os.open(
        ".report.lock",
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=output_fd,
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("report lock must be a single-link regular file")
    return descriptor


def _hash_fd(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _write_generation_file(
    generation_fd: int, name: str, content: bytes
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=generation_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o400)
    finally:
        os.close(descriptor)


def _read_regular_file(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"report generation file is unsafe: {name}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _generation_metadata(contents: Mapping[str, bytes]) -> dict[str, object]:
    hashes = {
        name: hashlib.sha256(contents[name]).hexdigest()
        for name in _REPORT_FILES
    }
    content_digest = hash_index_manifest(
        {
            "schema": "avgaussianv2.pilot-report-generation",
            "version": 1,
            "files": hashes,
        }
    )
    return {
        "schema": "avgaussianv2.pilot-report-generation",
        "version": 1,
        "content_digest": content_digest,
        "files": hashes,
    }


def _verify_generation(generation_fd: int, expected_digest: str) -> None:
    manifest_data = _read_regular_file(generation_fd, _GENERATION_MANIFEST)
    try:
        manifest = json.loads(
            manifest_data.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid report generation manifest") from error
    manifest = _exact(
        manifest,
        {"schema", "version", "content_digest", "files"},
        "report generation manifest",
    )
    if (
        manifest["schema"] != "avgaussianv2.pilot-report-generation"
        or manifest["version"] != 1
        or manifest["content_digest"] != expected_digest
    ):
        raise ValueError("report generation manifest identity mismatch")
    hashes = _exact(
        manifest["files"], set(_REPORT_FILES), "report generation hashes"
    )
    for name in _REPORT_FILES:
        data = _read_regular_file(generation_fd, name)
        if hashlib.sha256(data).hexdigest() != _digest(
            hashes[name], f"generation hash for {name}"
        ):
            raise ValueError(f"report generation hash mismatch: {name}")


def _remove_temporary_generation(generations_fd: int, name: str) -> None:
    try:
        generation_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=generations_fd,
        )
    except FileNotFoundError:
        return
    try:
        os.fchmod(generation_fd, 0o700)
        for child in (*_REPORT_FILES, _GENERATION_MANIFEST):
            try:
                os.unlink(child, dir_fd=generation_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(generation_fd)
    os.rmdir(name, dir_fd=generations_fd)


def _cleanup_stale_generations(generations_fd: int) -> None:
    for name in os.listdir(generations_fd):
        if name.startswith(".") and name.endswith(".tmp"):
            _remove_temporary_generation(generations_fd, name)


def _cleanup_stale_output_entries(output_fd: int) -> None:
    discovery_prefixes = tuple(f".{name}." for name in _REPORT_FILES)
    for name in os.listdir(output_fd):
        is_pointer_temp = name.startswith(".current.") and name.endswith(".tmp")
        is_discovery_temp = (
            name.endswith(".tmp")
            and any(name.startswith(prefix) for prefix in discovery_prefixes)
        )
        if not (is_pointer_temp or is_discovery_temp):
            continue
        metadata = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        if not stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"unsafe stale report temporary entry: {name}")
        os.unlink(name, dir_fd=output_fd)


def _current_pointer_state(
    output_fd: int,
) -> tuple[int, int, str] | None:
    try:
        metadata = os.stat("current", dir_fd=output_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISLNK(metadata.st_mode):
        raise ValueError("report current pointer must be a symlink")
    target = os.readlink("current", dir_fd=output_fd)
    prefix = ".report-generations/"
    if not target.startswith(prefix) or "/" in target[len(prefix):]:
        raise ValueError("report current pointer is unsafe")
    _digest(target[len(prefix):], "current generation digest")
    return metadata.st_dev, metadata.st_ino, target


def _verify_output_identity(output_fd: int, absolute_output: Path) -> None:
    pinned = os.fstat(output_fd)
    try:
        current = absolute_output.lstat()
    except FileNotFoundError as error:
        raise ValueError("report output directory was replaced") from error
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino)
        != (pinned.st_dev, pinned.st_ino)
    ):
        raise ValueError("report output directory identity changed")


def _ensure_discovery_symlinks(output_fd: int) -> None:
    for name in _REPORT_FILES:
        expected = f"current/{name}"
        try:
            actual = os.readlink(name, dir_fd=output_fd)
        except FileNotFoundError:
            temporary = f".{name}.{uuid.uuid4().hex}.tmp"
            try:
                os.symlink(expected, temporary, dir_fd=output_fd)
                os.rename(
                    temporary,
                    name,
                    src_dir_fd=output_fd,
                    dst_dir_fd=output_fd,
                )
            finally:
                try:
                    os.unlink(temporary, dir_fd=output_fd)
                except FileNotFoundError:
                    pass
        else:
            if actual != expected:
                raise ValueError(f"unsafe report discovery link: {name}")


def _publish_generation(
    output_dir: Path, contents: Mapping[str, str]
) -> tuple[str, Path, tuple[str, ...]]:
    encoded = {name: contents[name].encode("utf-8") for name in _REPORT_FILES}
    metadata = _generation_metadata(encoded)
    digest = str(metadata["content_digest"])
    output_fd, absolute_output = _open_secure_directory(output_dir, create=True)
    lock_fd: int | None = None
    generations_fd: int | None = None
    temporary_name: str | None = None
    pointer_temporary: str | None = None
    committed = False
    durability_warnings: list[str] = []
    try:
        lock_fd = _open_lock(output_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _cleanup_stale_output_entries(output_fd)
        desired_target = f".report-generations/{digest}"
        previous_pointer = _current_pointer_state(output_fd)
        generations_fd = _open_or_create_directory(
            output_fd, ".report-generations"
        )
        _cleanup_stale_generations(generations_fd)
        try:
            existing_fd = os.open(
                digest,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=generations_fd,
            )
        except FileNotFoundError:
            temporary_name = f".{digest}.{uuid.uuid4().hex}.tmp"
            os.mkdir(temporary_name, mode=0o700, dir_fd=generations_fd)
            generation_fd = os.open(
                temporary_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=generations_fd,
            )
            try:
                for name in _REPORT_FILES:
                    _write_generation_file(generation_fd, name, encoded[name])
                manifest_text = _json_text(metadata).encode("utf-8")
                _write_generation_file(
                    generation_fd, _GENERATION_MANIFEST, manifest_text
                )
                os.fsync(generation_fd)
                os.fchmod(generation_fd, 0o500)
            finally:
                os.close(generation_fd)
            os.rename(
                temporary_name,
                digest,
                src_dir_fd=generations_fd,
                dst_dir_fd=generations_fd,
            )
            temporary_name = None
            os.fsync(generations_fd)
            existing_fd = os.open(
                digest,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=generations_fd,
            )
        try:
            _verify_generation(existing_fd, digest)
        finally:
            os.close(existing_fd)
        _ensure_discovery_symlinks(output_fd)
        if _current_pointer_state(output_fd) != previous_pointer:
            raise ValueError("report current pointer identity changed during publication")
        _verify_output_identity(output_fd, absolute_output)
        pointer_temporary = f".current.{uuid.uuid4().hex}.tmp"
        os.symlink(
            desired_target,
            pointer_temporary,
            dir_fd=output_fd,
        )
        os.rename(
            pointer_temporary,
            "current",
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
        )
        pointer_temporary = None
        committed = True
        try:
            os.fsync(output_fd)
        except OSError as error:
            durability_warnings.append(
                f"current pointer committed but output-directory fsync failed: {error}"
            )
        try:
            _verify_output_identity(output_fd, absolute_output)
        except (OSError, ValueError) as error:
            durability_warnings.append(
                f"current pointer committed but output identity recheck failed: {error}"
            )
    finally:
        if pointer_temporary is not None:
            try:
                os.unlink(pointer_temporary, dir_fd=output_fd)
            except FileNotFoundError:
                pass
        if temporary_name is not None and generations_fd is not None:
            _remove_temporary_generation(generations_fd, temporary_name)
        if generations_fd is not None:
            if committed:
                try:
                    os.close(generations_fd)
                except Exception as error:
                    durability_warnings.append(
                        f"current pointer committed but generations fd close failed: {error}"
                    )
            else:
                os.close(generations_fd)
        if lock_fd is not None:
            if committed:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except Exception as error:
                    durability_warnings.append(
                        f"current pointer committed but lock release failed: {error}"
                    )
                try:
                    os.close(lock_fd)
                except Exception as error:
                    durability_warnings.append(
                        f"current pointer committed but lock fd close failed: {error}"
                    )
            else:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
        if committed:
            try:
                os.close(output_fd)
            except Exception as error:
                durability_warnings.append(
                    f"current pointer committed but output fd close failed: {error}"
                )
        else:
            os.close(output_fd)
    generation_path = absolute_output / ".report-generations" / digest
    if not committed:
        raise RuntimeError("report publication returned without committing current")
    if durability_warnings:
        try:
            _persist_durability_warnings(
                absolute_output, digest, tuple(durability_warnings)
            )
        except Exception as error:
            durability_warnings.append(
                f"current pointer committed but durability warning persistence "
                f"failed: {error}"
            )
    return digest, generation_path, tuple(durability_warnings)


def _persist_durability_warnings(
    output_dir: Path, digest: str, warnings: tuple[str, ...]
) -> None:
    output_fd, _ = _open_secure_directory(output_dir, create=False)
    warnings_fd: int | None = None
    temporary: str | None = None
    failure: Exception | None = None
    try:
        warnings_fd = _open_or_create_directory(output_fd, ".report-warnings")
        temporary = f".{digest}.{uuid.uuid4().hex}.tmp"
        payload = _json_text(
            {
                "schema": "avgaussianv2.report-durability-warnings",
                "version": 1,
                "content_digest": digest,
                "warnings": list(warnings),
            }
        ).encode("utf-8")
        _write_generation_file(warnings_fd, temporary, payload)
        os.rename(
            temporary,
            f"{digest}.json",
            src_dir_fd=warnings_fd,
            dst_dir_fd=warnings_fd,
        )
        temporary = None
        os.fsync(warnings_fd)
        os.fsync(output_fd)
    except Exception as error:
        failure = error
    finally:
        if temporary is not None and warnings_fd is not None:
            try:
                os.unlink(temporary, dir_fd=warnings_fd)
            except Exception as error:
                if failure is None:
                    failure = error
        if warnings_fd is not None:
            try:
                os.close(warnings_fd)
            except Exception as error:
                if failure is None:
                    failure = error
        try:
            os.close(output_fd)
        except Exception as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def _read_durability_warnings(output_fd: int, digest: str) -> tuple[str, ...]:
    try:
        warnings_fd = os.open(
            ".report-warnings",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_fd,
        )
    except FileNotFoundError:
        return ()
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                f"{digest}.json",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=warnings_fd,
            )
        except FileNotFoundError:
            return ()
        metadata = os.fstat(descriptor)
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_nlink,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > 1024 * 1024
        ):
            raise ValueError("durability warning sidecar is unsafe")
        data = b""
        while block := os.read(descriptor, 64 * 1024):
            data += block
            if len(data) > 1024 * 1024:
                raise ValueError("durability warning sidecar is too large")
        after_metadata = os.fstat(descriptor)
        after = (
            after_metadata.st_dev,
            after_metadata.st_ino,
            after_metadata.st_size,
            after_metadata.st_mtime_ns,
            after_metadata.st_ctime_ns,
            after_metadata.st_nlink,
        )
        if after != before or after_metadata.st_nlink != 1:
            raise ValueError("durability warning sidecar changed while reading")
        payload = json.loads(data.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema")
            != "avgaussianv2.report-durability-warnings"
            or payload.get("version") != 1
            or payload.get("content_digest") != digest
            or not isinstance(payload.get("warnings"), list)
            or any(not isinstance(item, str) for item in payload["warnings"])
        ):
            raise ValueError("durability warning sidecar is invalid")
        return tuple(payload["warnings"])
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("durability warning sidecar is invalid") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(warnings_fd)


def resolve_current_report(
    output_dir: str | Path, *, include_warnings: bool = False
) -> Path | ResolvedReport:
    """Pin and verify the authoritative immutable report generation."""
    output_fd, absolute_output = _open_secure_directory(
        Path(output_dir), create=False
    )
    lock_fd = _open_lock(output_fd)
    generation_fd: int | None = None
    generations_fd: int | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        try:
            target = os.readlink("current", dir_fd=output_fd)
        except FileNotFoundError as error:
            raise ValueError("report has no authoritative current generation") from error
        prefix = ".report-generations/"
        if not target.startswith(prefix) or "/" in target[len(prefix):]:
            raise ValueError("report current pointer is unsafe")
        digest = _digest(target[len(prefix):], "current generation digest")
        generations_fd = os.open(
            ".report-generations",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_fd,
        )
        generation_fd = os.open(
            digest,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=generations_fd,
        )
        _verify_generation(generation_fd, digest)
        _verify_output_identity(output_fd, absolute_output)
        generation_path = absolute_output / target
        if include_warnings:
            return ResolvedReport(
                generation_path=generation_path,
                durability_warnings=_read_durability_warnings(output_fd, digest),
            )
        return generation_path
    finally:
        if generation_fd is not None:
            os.close(generation_fd)
        if generations_fd is not None:
            os.close(generations_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        os.close(output_fd)


def _prepare_comparison(
    systems: Sequence[SystemReportInput],
    *,
    checkpoint_identity_resolver: Callable[
        [Path, EvaluationProvenance], CheckpointArtifactIdentity
    ] = resolve_pilot_checkpoint_identity,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, tuple[dict[str, object], ...]],
    dict[str, object],
    dict[str, dict[str, object]],
    PilotDecision,
    dict[str, str],
]:
    normalized, rows = _normalize_systems(
        systems,
        checkpoint_identity_resolver=checkpoint_identity_resolver,
    )
    paired = paired_audio_deltas(
        rows["joint_conditioned_on"], rows["joint_conditioned_off"]
    )
    output_systems = _system_output(normalized)
    descriptive = _descriptive_comparisons(output_systems)
    decision = _decide_normalized(output_systems, paired)
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
    contents = {
        "comparison.json": _json_text(json_payload),
        "comparison.csv": _csv_text(output_systems),
        "comparison.md": _markdown_text(
            output_systems, paired, descriptive, decision
        ),
        "paired_condition_deltas.jsonl": paired_jsonl,
    }
    return output_systems, rows, paired, descriptive, decision, contents


def build_comparison(
    systems: Sequence[SystemReportInput],
    output_dir: str | Path,
    *,
    checkpoint_identity_resolver: Callable[
        [Path, EvaluationProvenance], CheckpointArtifactIdentity
    ] = resolve_pilot_checkpoint_identity,
) -> ComparisonResult:
    """Validate five systems, decide acceptance, and atomically publish reports."""
    output_systems, rows, paired, descriptive, decision, contents = (
        _prepare_comparison(
            systems,
            checkpoint_identity_resolver=checkpoint_identity_resolver,
        )
    )
    digest, generation_path, durability_warnings = _publish_generation(
        Path(output_dir), contents,
    )
    return ComparisonResult(
        systems=output_systems,
        rows=rows,
        paired_condition=paired,
        descriptive_comparisons=descriptive,
        decision=decision,
        content_digest=digest,
        generation_path=generation_path,
        committed=True,
        durability_warnings=durability_warnings,
    )


def verify_current_comparison(
    systems: Sequence[SystemReportInput],
    output_dir: str | Path,
    *,
    checkpoint_identity_resolver: Callable[
        [Path, EvaluationProvenance], CheckpointArtifactIdentity
    ] = resolve_pilot_checkpoint_identity,
) -> ComparisonResult:
    """Recompute and byte-verify the authoritative report without publication."""
    output_systems, rows, paired, descriptive, decision, contents = (
        _prepare_comparison(
            systems,
            checkpoint_identity_resolver=checkpoint_identity_resolver,
        )
    )
    resolved = resolve_current_report(output_dir, include_warnings=True)
    if not isinstance(resolved, ResolvedReport):  # pragma: no cover - type guard
        raise RuntimeError("report resolver did not return metadata")
    encoded = {name: contents[name].encode("utf-8") for name in _REPORT_FILES}
    expected_digest = str(_generation_metadata(encoded)["content_digest"])
    if resolved.generation_path.name != expected_digest:
        raise ValueError("authoritative report digest disagrees with recomputed inputs")
    for name, expected in encoded.items():
        actual = _read_verified_artifact(
            resolved.generation_path / name,
            hashlib.sha256(expected).hexdigest(),
            limit=max(len(expected), 1),
            name=f"report {name}",
        )
        if actual != expected:
            raise ValueError(f"authoritative report content mismatch: {name}")
    return ComparisonResult(
        systems=output_systems,
        rows=rows,
        paired_condition=paired,
        descriptive_comparisons=descriptive,
        decision=decision,
        content_digest=expected_digest,
        generation_path=resolved.generation_path,
        committed=True,
        durability_warnings=resolved.durability_warnings,
    )


__all__ = [
    "ComparisonResult",
    "CheckpointArtifactIdentity",
    "EvaluationArtifactProvenance",
    "EvaluationProvenance",
    "PilotDecision",
    "REQUIRED_SYSTEMS",
    "ResolvedReport",
    "SystemReportInput",
    "WorkerArtifactProvenance",
    "build_comparison",
    "build_evaluation_manifest_sha256",
    "build_evaluation_run_id",
    "decide_long_training",
    "paired_audio_deltas",
    "resolve_current_report",
    "resolve_pilot_checkpoint_identity",
    "verify_current_comparison",
]

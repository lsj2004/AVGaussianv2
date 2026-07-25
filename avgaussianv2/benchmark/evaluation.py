"""Strict common evaluator for the dual-scene cam38 benchmark."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import torch
from torch import Tensor

from avgaussianv2.benchmark.artifacts import (
    ArtifactError,
    canonical_json,
    load_generation,
    publish_generation,
)
from avgaussianv2.benchmark.native import verify_native_contract
from avgaussianv2.benchmark.output import (
    BenchmarkOutputError,
    BenchmarkOutputReadLock,
    validate_output_children,
)
from avgaussianv2.benchmark.training import (
    ALGORITHM as TRAINING_ALGORITHM,
    SCHEMA as TRAINING_SCHEMA,
    SCHEMA_VERSION as TRAINING_SCHEMA_VERSION,
    BenchmarkCompatibility,
    BenchmarkConfig,
    TEST_CAMERA,
    TRAIN_CAMERAS,
    hash_shared_indices,
)
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.experiment.evaluation import move_sample
from avgaussianv2.experiment.metrics import (
    aggregate_metrics,
    log_spectral_distance,
    lre_error_db,
    psnr,
    rgb_l1,
    ssim,
    waveform_l1,
)

SCHEMA = "avgaussianv2.cam38-benchmark-evaluation"
AUDIO_METRICS = (
    "audio_total",
    "audio_mono",
    "audio_diff",
    "waveform_l1",
    "mono_lsd",
    "diff_lsd",
    "lre_error_db",
)
VIDEO_METRICS = ("rgb_psnr", "rgb_ssim", "rgb_l1")
ALL_METRICS = (*AUDIO_METRICS, *VIDEO_METRICS)
METRIC_DIRECTIONS = {
    name: "higher_is_better" if name in {"rgb_psnr", "rgb_ssim"} else "lower_is_better"
    for name in ALL_METRICS
}
SCENE_SAMPLE_COUNTS = {"scene1_opera": 130, "Scene7playing": 293}
REPORTING_STEPS = (5_000, 10_000, 30_000)
# Exact RGB matches have mathematical +inf PSNR.  The strict benchmark caps
# them at 100 dB so every published value remains finite and JSON-portable.
PSNR_CAP_DB = 100.0
CONTINUATION_SYSTEMS = {"joint_conditioned", "audio_only", "visual_only"}
NATIVE_SYSTEMS = {"native_audiogs", "native_ftgspp"}
NATIVE_AUDIO_UPDATES = {"scene1_opera": 2_318, "Scene7playing": 6_954}
_ROW_METADATA = {"sample_id", "scene_id", "camera", "frame_index", "time_seconds"}
_RESERVED_EXTRA_METRICS = {*_ROW_METADATA, *ALL_METRICS, "rgb_lpips"}
_TASK12_CHECKPOINT_FIELDS = {
    "schema",
    "version",
    "fingerprint",
    "compatibility",
    "stage",
    "warmup_step",
    "main_step",
    "model",
    "optimizer",
    "visual_anchor",
    "torch_rng_state",
    "cuda_rng_states",
    "python_rng_state",
    "numpy_rng_state",
    "io",
    "gradient_guard",
}


class BenchmarkEvaluationError(RuntimeError):
    pass


def _digest(value: str | None, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TrainingEvidence:
    system_name: str
    scene_id: str
    role: str
    train_cameras: tuple[str, ...]
    test_camera: str
    test_targets_read_during_training: bool
    seed: int
    planned_updates: int | None
    completed_updates: int | None
    checkpoint_step: int | None
    checkpoint_path: str
    checkpoint_sha256: str
    config_sha256: str
    source_sha256: str
    visual_initialization_sha256: str
    audio_initialization_sha256: str
    model_initialization_sha256: str
    index_sha256: str | None
    batch_size: int
    epochs: float | None
    training_output_dir: str | None = None
    runtime_contract_path: str | None = None
    runtime_contract_sha256: str | None = None
    native_contract_path: str | None = None
    native_contract_sha256: str | None = None

    def validate(self, identity: EvaluationIdentity) -> None:
        if self.system_name != identity.system_name or self.scene_id != identity.scene_id:
            raise BenchmarkEvaluationError("training/evaluation identity mismatch")
        if self.train_cameras != TRAIN_CAMERAS or self.test_camera != TEST_CAMERA:
            raise BenchmarkEvaluationError("benchmark split must be cam00..cam37/cam38")
        if self.test_targets_read_during_training is not False:
            raise BenchmarkEvaluationError("test target was read during training")
        if self.seed != 42 or self.batch_size != 1:
            raise BenchmarkEvaluationError("benchmark seed/batch size mismatch")
        if not isinstance(self.checkpoint_path, str) or not Path(
            self.checkpoint_path
        ).is_absolute():
            raise BenchmarkEvaluationError("checkpoint path must be absolute")
        for name in (
            "checkpoint_sha256",
            "config_sha256",
            "source_sha256",
            "visual_initialization_sha256",
            "audio_initialization_sha256",
            "model_initialization_sha256",
        ):
            try:
                _digest(getattr(self, name), name)
            except ValueError as error:
                raise BenchmarkEvaluationError(str(error)) from error
        if self.role == "continuation":
            if self.native_contract_path is not None or self.native_contract_sha256 is not None:
                raise BenchmarkEvaluationError(
                    "continuation must not carry native training evidence"
                )
            if self.system_name not in CONTINUATION_SYSTEMS:
                raise BenchmarkEvaluationError("unknown continuation system")
            if (
                identity.reporting_step not in REPORTING_STEPS
                or self.planned_updates != 30_000
                or self.completed_updates != identity.reporting_step
                or self.checkpoint_step != identity.reporting_step
            ):
                raise BenchmarkEvaluationError("continuation budget/final step mismatch")
            try:
                _digest(self.index_sha256, "index_sha256")
            except ValueError as error:
                raise BenchmarkEvaluationError(str(error)) from error
        elif self.role == "native_reference":
            if any(
                value is not None
                for value in (
                    self.training_output_dir,
                    self.runtime_contract_path,
                    self.runtime_contract_sha256,
                )
            ):
                raise BenchmarkEvaluationError(
                    "native reference must not carry Task12 continuation evidence"
                )
            if (
                self.native_contract_path is None
            ) != (self.native_contract_sha256 is None):
                raise BenchmarkEvaluationError(
                    "native contract path/hash must be provided together"
                )
            if self.native_contract_path is not None:
                if not Path(self.native_contract_path).is_absolute():
                    raise BenchmarkEvaluationError(
                        "native contract path must be absolute"
                    )
                try:
                    _digest(self.native_contract_sha256, "native_contract_sha256")
                except ValueError as error:
                    raise BenchmarkEvaluationError(str(error)) from error
            if self.system_name not in NATIVE_SYSTEMS or identity.reporting_step is not None:
                raise BenchmarkEvaluationError("native reference identity mismatch")
            if self.index_sha256 is not None:
                raise BenchmarkEvaluationError("native reference is not update-matched")
            if self.system_name == "native_audiogs":
                expected_updates = NATIVE_AUDIO_UPDATES.get(self.scene_id)
                if (
                    self.epochs != 61.0
                    or self.completed_updates != expected_updates
                    or self.planned_updates != expected_updates
                    or self.checkpoint_step != expected_updates
                ):
                    raise BenchmarkEvaluationError(
                        "native AudioGS must record its resolved 61-epoch update budget"
                    )
            elif (
                self.epochs is not None
                or self.planned_updates != 30_000
                or self.completed_updates != 30_000
                or self.checkpoint_step != 30_000
            ):
                raise BenchmarkEvaluationError(
                    "native FreeTimeGS++ must be the strict 30,000-update reference"
                )
        else:
            raise BenchmarkEvaluationError("unsupported training evidence role")


@dataclass(frozen=True)
class EvaluationIdentity:
    scene_id: str
    system_name: str
    reporting_step: int | None
    expected_sample_ids: tuple[str, ...]
    expected_sample_count: int

    def __post_init__(self) -> None:
        if not self.scene_id or not self.system_name:
            raise ValueError("evaluation scene/system must be nonempty")
        if (
            not isinstance(self.expected_sample_count, int)
            or isinstance(self.expected_sample_count, bool)
            or self.expected_sample_count <= 0
            or len(self.expected_sample_ids) != self.expected_sample_count
            or len(set(self.expected_sample_ids)) != self.expected_sample_count
        ):
            raise ValueError("evaluation sample identity/count mismatch")
        if self.reporting_step is not None and self.reporting_step not in REPORTING_STEPS:
            raise ValueError("reporting step must be 5000, 10000, 30000, or null")

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["expected_sample_ids"] = list(self.expected_sample_ids)
        return value


@dataclass(frozen=True)
class BenchmarkPrediction:
    predicted_audio: Tensor | None = None
    rendered_rgb: Tensor | None = None


@dataclass(frozen=True)
class BenchmarkEvaluationRuntime:
    samples: Sequence[AlignedAVSample]
    audio_loss_fn: Callable[[Tensor, Tensor], Mapping[str, object]]
    extra_metric_fns: Mapping[
        str, Callable[[BenchmarkPrediction, AlignedAVSample], object]
    ] | None = None
    extra_metric_directions: Mapping[str, str] | None = None
    extra_metric_modalities: Mapping[str, str] | None = None
    lpips_fn: Callable[[Tensor, Tensor], object] | None = None
    lpips_implementation_sha256: str | None = None


@dataclass(frozen=True)
class BenchmarkEvaluationResult:
    identity: EvaluationIdentity
    count: int
    rows: tuple[dict[str, object], ...]
    summary: dict[str, dict[str, float]]
    metric_directions: dict[str, str]
    metric_protocol: dict[str, object]
    provenance: dict[str, object]
    content_sha256: str
    generation_path: Path | None


def benchmark_sample_id(sample: AlignedAVSample) -> str:
    return f"{sample.scene_id}/{sample.camera}/{sample.frame_index:06d}"


def _scalar(value: object, name: str) -> float:
    if isinstance(value, Tensor):
        if value.ndim:
            raise BenchmarkEvaluationError(f"{name} must be scalar")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BenchmarkEvaluationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkEvaluationError(f"{name} must be finite")
    return result


def _identity_from_mapping(value: object) -> EvaluationIdentity:
    if not isinstance(value, Mapping):
        raise BenchmarkEvaluationError("evaluation identity must be a mapping")
    try:
        return EvaluationIdentity(
            scene_id=value["scene_id"],
            system_name=value["system_name"],
            reporting_step=value["reporting_step"],
            expected_sample_ids=tuple(value["expected_sample_ids"]),
            expected_sample_count=value["expected_sample_count"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkEvaluationError(f"invalid evaluation identity: {error}") from error


def _parse_result(
    generation: Path,
    files: Mapping[str, bytes],
    manifest_digest: str,
) -> BenchmarkEvaluationResult:
    try:
        summary_document = json.loads(files["metrics_summary.json"])
        rows = tuple(
            json.loads(line)
            for line in files["metrics_per_sample.jsonl"].decode().splitlines()
        )
    except (KeyError, UnicodeDecodeError, ValueError) as error:
        raise BenchmarkEvaluationError(f"invalid evaluation artifact: {error}") from error
    if (
        not isinstance(summary_document, Mapping)
        or set(summary_document)
        != {
            "schema",
            "version",
            "identity",
            "count",
            "summary",
            "provenance",
            "metric_directions",
            "metric_protocol",
        }
        or summary_document["schema"] != SCHEMA
        or summary_document["version"] != 1
    ):
        raise BenchmarkEvaluationError("evaluation summary schema mismatch")
    identity = _identity_from_mapping(summary_document["identity"])
    if len(rows) != identity.expected_sample_count or summary_document["count"] != len(rows):
        raise BenchmarkEvaluationError("evaluation artifact count mismatch")
    result = BenchmarkEvaluationResult(
        identity=identity,
        count=len(rows),
        rows=rows,
        summary=dict(summary_document["summary"]),
        metric_directions=dict(summary_document["metric_directions"]),
        metric_protocol=dict(summary_document["metric_protocol"]),
        provenance=dict(summary_document["provenance"]),
        content_sha256=manifest_digest,
        generation_path=generation,
    )
    _validate_loaded_result(result)
    return result


def _validate_loaded_result(result: BenchmarkEvaluationResult) -> None:
    metadata = {"sample_id", "scene_id", "camera", "frame_index", "time_seconds"}
    expected_ids = result.identity.expected_sample_ids
    if tuple(row.get("sample_id") for row in result.rows) != expected_ids:
        raise BenchmarkEvaluationError("evaluation artifact sample IDs/order mismatch")
    if any(
        row.get("scene_id") != result.identity.scene_id
        or row.get("camera") != TEST_CAMERA
        or set(row) - metadata != set(result.summary)
        for row in result.rows
    ):
        raise BenchmarkEvaluationError("evaluation artifact row schema mismatch")
    recalculated = aggregate_metrics(
        [
            {name: _scalar(row[name], name) for name in result.summary}
            for row in result.rows
        ]
    )
    if recalculated != result.summary:
        raise BenchmarkEvaluationError("evaluation summary does not match rows")
    if (
        set(result.metric_directions) != set(result.summary)
        or any(
            direction not in {"lower_is_better", "higher_is_better"}
            for direction in result.metric_directions.values()
        )
        or set(result.metric_protocol)
        != {
            "psnr_cap_db",
            "lpips_implementation_sha256",
            "extra_metric_registry",
        }
        or result.metric_protocol["psnr_cap_db"] != PSNR_CAP_DB
    ):
        raise BenchmarkEvaluationError("evaluation metric schema/directions mismatch")
    registry = result.metric_protocol["extra_metric_registry"]
    if not isinstance(registry, Mapping):
        raise BenchmarkEvaluationError("extra metric registry schema mismatch")
    for name, specification in registry.items():
        if (
            not isinstance(name, str)
            or name in _RESERVED_EXTRA_METRICS
            or "depth" in name.lower()
            or not isinstance(specification, Mapping)
            or set(specification) != {"direction", "modality"}
            or specification["direction"]
            not in {"lower_is_better", "higher_is_better"}
            or specification["modality"] not in {"audio", "video"}
            or result.metric_directions.get(name) != specification["direction"]
        ):
            raise BenchmarkEvaluationError("extra metric registry schema mismatch")
    lpips_identity = result.metric_protocol["lpips_implementation_sha256"]
    if lpips_identity is not None:
        try:
            _digest(lpips_identity, "lpips_implementation_sha256")
        except ValueError as error:
            raise BenchmarkEvaluationError(str(error)) from error
    if ("rgb_lpips" in result.summary) != (lpips_identity is not None):
        # Audio-only native references may bind the common implementation
        # without producing an RGB metric.
        if not (
            result.identity.system_name == "native_audiogs"
            and lpips_identity is not None
            and "rgb_lpips" not in result.summary
        ):
            raise BenchmarkEvaluationError("LPIPS metric/protocol mismatch")
    for name, direction in result.metric_directions.items():
        expected = (
            "lower_is_better"
            if name == "rgb_lpips"
            else registry[name]["direction"]
            if name in registry
            else METRIC_DIRECTIONS.get(name)
        )
        if expected is None or direction != expected:
            raise BenchmarkEvaluationError(
                f"metric direction mismatch for {name!r}"
            )
    if "rgb_psnr" in result.summary and any(
        float(row["rgb_psnr"]) > PSNR_CAP_DB for row in result.rows
    ):
        raise BenchmarkEvaluationError("RGB PSNR exceeds the documented finite cap")
    provenance = result.provenance
    try:
        fields = set(TrainingEvidence.__dataclass_fields__)
        evidence = TrainingEvidence(
            **{
                **{name: provenance[name] for name in fields},
                "train_cameras": tuple(provenance["train_cameras"]),
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkEvaluationError(
            f"evaluation provenance schema mismatch: {error}"
        ) from error
    if set(provenance) != {*fields, "update_matched"}:
        raise BenchmarkEvaluationError("evaluation provenance fields mismatch")
    evidence.validate(result.identity)
    if provenance["update_matched"] is not (evidence.role == "continuation"):
        raise BenchmarkEvaluationError("evaluation update-matched label mismatch")


def load_evaluation(
    output_dir: Path | str, *, identity: EvaluationIdentity | None = None
) -> BenchmarkEvaluationResult:
    try:
        generation, files, _, digest = load_generation(
            Path(output_dir),
            schema=SCHEMA,
            expected_identity=identity.to_mapping() if identity else None,
        )
        return _parse_result(generation, files, digest)
    except ArtifactError as error:
        raise BenchmarkEvaluationError(str(error)) from error


def verify_evaluation(
    output_dir: Path | str,
    *,
    identity: EvaluationIdentity | None = None,
    verify_checkpoint: bool = True,
    strict_training_evidence: bool = True,
) -> BenchmarkEvaluationResult:
    """Verify an existing evaluation without runtime construction or mutation."""
    result = load_evaluation(output_dir, identity=identity)
    evidence_fields = set(TrainingEvidence.__dataclass_fields__)
    evidence = TrainingEvidence(
        **{
            **{name: result.provenance[name] for name in evidence_fields},
            "train_cameras": tuple(result.provenance["train_cameras"]),
        }
    )
    if strict_training_evidence:
        audit_training_evidence(evidence, result.identity)
    elif verify_checkpoint:
        _verify_checkpoint(
            evidence.checkpoint_path,
            evidence.checkpoint_sha256,
        )
    return result


def _verify_checkpoint(
    path_value: object,
    expected_digest: object,
    *,
    reject_components: bool = True,
) -> bytes:
    checkpoint = Path(str(path_value))
    if (
        not isinstance(path_value, (str, Path))
        or not checkpoint.is_absolute()
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
    ):
        raise BenchmarkEvaluationError(
            "strict benchmark checkpoint is missing or unsafe"
        )
    if reject_components:
        _reject_symlink_components(checkpoint)
    data = checkpoint.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise BenchmarkEvaluationError("strict benchmark checkpoint hash mismatch")
    return data


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                raise BenchmarkEvaluationError(
                    f"strict evidence path contains a symlink component: {current}"
                )
        except OSError as error:
            raise BenchmarkEvaluationError(
                f"cannot inspect strict evidence path: {current}"
            ) from error


def _load_exact_json(path: Path, fields: set[str], name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BenchmarkEvaluationError(f"cannot read {name}: {error}") from error
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BenchmarkEvaluationError(f"{name} schema fields mismatch")
    return value


def _load_exact_json_with_bytes(
    path: Path, fields: set[str], name: str
) -> tuple[Mapping[str, object], bytes]:
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, ValueError) as error:
        raise BenchmarkEvaluationError(f"cannot read {name}: {error}") from error
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BenchmarkEvaluationError(f"{name} schema fields mismatch")
    return value, data


def _audit_continuation_evidence(
    evidence: TrainingEvidence, identity: EvaluationIdentity
) -> None:
    if (
        not evidence.training_output_dir
        or not evidence.runtime_contract_path
        or not evidence.runtime_contract_sha256
    ):
        raise BenchmarkEvaluationError(
            "continuation requires Task12 output and train-only runtime contract"
        )
    output = Path(evidence.training_output_dir)
    runtime_path = Path(evidence.runtime_contract_path)
    checkpoint = Path(evidence.checkpoint_path)
    if (
        not output.is_absolute()
        or not runtime_path.is_absolute()
        or runtime_path != output / "runtime_contract.json"
        or checkpoint.parent != output / "milestones"
        or checkpoint.name != f"step_{identity.reporting_step:06d}.pt"
    ):
        raise BenchmarkEvaluationError("Task12 artifact path/layout mismatch")
    _reject_symlink_components(output)
    _reject_symlink_components(runtime_path)
    try:
        with BenchmarkOutputReadLock(output) as pinned:
            _audit_continuation_snapshot(evidence, identity, pinned)
    except BenchmarkOutputError as error:
        raise BenchmarkEvaluationError(
            f"unsafe or changing Task12 output: {error}"
        ) from error


def _audit_continuation_snapshot(
    evidence: TrainingEvidence,
    identity: EvaluationIdentity,
    output: Path,
) -> None:
    """Read every Task12 byte through one retained directory descriptor."""
    try:
        validate_output_children(output)
    except BenchmarkOutputError as error:
        raise BenchmarkEvaluationError(f"unsafe Task12 output child: {error}") from error
    runtime_path = output / "runtime_contract.json"
    checkpoint = output / "milestones" / f"step_{identity.reporting_step:06d}.pt"
    contract = _load_exact_json(
        output / "contract.json",
        {
            "schema",
            "version",
            "fingerprint",
            "compatibility",
            "shared_indices",
            "selection",
            "milestones",
        },
        "Task12 contract",
    )
    if (
        contract["schema"] != f"{TRAINING_SCHEMA}.contract"
        or contract["version"] != TRAINING_SCHEMA_VERSION
        or contract["selection"] != "final"
        or contract["milestones"] != list(REPORTING_STEPS)
    ):
        raise BenchmarkEvaluationError("Task12 contract protocol mismatch")
    compatibility = BenchmarkCompatibility.from_mapping(contract["compatibility"])
    if compatibility.to_mapping() != {
        "scene_id": evidence.scene_id,
        "mode": evidence.system_name,
        "train_cameras": list(evidence.train_cameras),
        "test_camera": evidence.test_camera,
        "seed": evidence.seed,
        "index_sha256": evidence.index_sha256,
        "visual_initialization_sha256": evidence.visual_initialization_sha256,
        "audio_initialization_sha256": evidence.audio_initialization_sha256,
        "model_initialization_sha256": evidence.model_initialization_sha256,
        "source_sha256": evidence.source_sha256,
        "config_sha256": evidence.config_sha256,
    }:
        raise BenchmarkEvaluationError("Task12 compatibility/evidence mismatch")
    shared_indices = contract["shared_indices"]
    if (
        not isinstance(shared_indices, list)
        or len(shared_indices) != 30_000
        or hash_shared_indices(shared_indices) != evidence.index_sha256
    ):
        raise BenchmarkEvaluationError("Task12 shared index contract mismatch")
    fingerprint = contract["fingerprint"]
    if (
        not isinstance(fingerprint, Mapping)
        or set(fingerprint) != {"sha256", "inputs"}
        or not isinstance(fingerprint["inputs"], Mapping)
        or set(fingerprint["inputs"])
        != {
            "schema",
            "version",
            "algorithm",
            "config",
            "compatibility",
            "train_config",
            "model_class",
            "model_format_version",
        }
        or fingerprint["inputs"].get("schema") != TRAINING_SCHEMA
        or fingerprint["inputs"].get("version") != TRAINING_SCHEMA_VERSION
        or fingerprint["inputs"].get("algorithm") != TRAINING_ALGORITHM
        or fingerprint["inputs"].get("compatibility") != compatibility.to_mapping()
        or hashlib.sha256(
            json.dumps(
                fingerprint["inputs"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        != fingerprint["sha256"]
    ):
        raise BenchmarkEvaluationError("Task12 fingerprint mismatch")
    try:
        BenchmarkConfig.from_mapping(fingerprint["inputs"]["config"]).validate()
    except (TypeError, ValueError) as error:
        raise BenchmarkEvaluationError(
            f"Task12 fingerprint training config mismatch: {error}"
        ) from error
    runtime, runtime_bytes = _load_exact_json_with_bytes(
        runtime_path,
        {
            "schema",
            "version",
            "include_eval",
            "train_cameras",
            "test_camera",
            "dataset_identity_sha256",
            "dataset_sample_ids_sha256",
            "config_sha256",
            "source_sha256",
            "visual_initialization_sha256",
            "audio_initialization_sha256",
            "model_initialization_sha256",
        },
        "production train-only runtime contract",
    )
    if hashlib.sha256(runtime_bytes).hexdigest() != evidence.runtime_contract_sha256:
        raise BenchmarkEvaluationError("train-only runtime contract hash mismatch")
    if (
        runtime["schema"] != "avgaussianv2.cam38-production-train-only-runtime"
        or runtime["version"] != 1
        or runtime["include_eval"] is not False
        or runtime["train_cameras"] != list(TRAIN_CAMERAS)
        or runtime["test_camera"] != TEST_CAMERA
    ):
        raise BenchmarkEvaluationError(
            "production runtime is not a strict train-only cam38 contract"
        )
    for field in (
        "config_sha256",
        "source_sha256",
        "visual_initialization_sha256",
        "audio_initialization_sha256",
        "model_initialization_sha256",
    ):
        if runtime[field] != getattr(evidence, field):
            raise BenchmarkEvaluationError(
                f"production runtime {field} evidence mismatch"
            )
    for field in ("dataset_identity_sha256", "dataset_sample_ids_sha256"):
        try:
            _digest(runtime[field], field)
        except ValueError as error:
            raise BenchmarkEvaluationError(str(error)) from error
    artifact_manifest = _load_exact_json(
        output / "artifact_hashes.json",
        {"schema", "version", "fingerprint_sha256", "sha256"},
        "Task12 artifact hash manifest",
    )
    expected_artifacts = {
        *(f"milestones/step_{step:06d}.pt" for step in REPORTING_STEPS),
        "final.pt",
    }
    if (
        artifact_manifest["schema"] != f"{TRAINING_SCHEMA}.artifacts"
        or artifact_manifest["version"] != TRAINING_SCHEMA_VERSION
        or artifact_manifest["fingerprint_sha256"] != fingerprint["sha256"]
        or not isinstance(artifact_manifest["sha256"], Mapping)
        or set(artifact_manifest["sha256"]) != expected_artifacts
    ):
        raise BenchmarkEvaluationError("Task12 artifact hash manifest mismatch")
    relative = f"milestones/{checkpoint.name}"
    if artifact_manifest["sha256"][relative] != evidence.checkpoint_sha256:
        raise BenchmarkEvaluationError("Task12 milestone artifact hash mismatch")
    checkpoint_bytes = _verify_checkpoint(
        checkpoint, evidence.checkpoint_sha256, reject_components=False
    )
    try:
        checkpoint_payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise BenchmarkEvaluationError(
            f"cannot parse Task12 checkpoint: {error}"
        ) from error
    if (
        not isinstance(checkpoint_payload, Mapping)
        or set(checkpoint_payload) != _TASK12_CHECKPOINT_FIELDS
        or checkpoint_payload.get("schema") != TRAINING_SCHEMA
        or checkpoint_payload.get("version") != TRAINING_SCHEMA_VERSION
        or checkpoint_payload.get("fingerprint") != fingerprint
        or checkpoint_payload.get("compatibility") != compatibility.to_mapping()
        or checkpoint_payload.get("stage") != "main"
        or checkpoint_payload.get("warmup_step")
        != (2_000 if evidence.system_name == "joint_conditioned" else 0)
        or checkpoint_payload.get("main_step") != identity.reporting_step
    ):
        raise BenchmarkEvaluationError(
            "Task12 checkpoint schema/fingerprint/stage/step mismatch"
        )


def audit_training_evidence(
    evidence: TrainingEvidence, identity: EvaluationIdentity
) -> None:
    """Validate immutable training evidence without constructing eval runtime."""
    evidence.validate(identity)
    if evidence.role == "continuation":
        _audit_continuation_evidence(evidence, identity)
    else:
        if not evidence.native_contract_path or not evidence.native_contract_sha256:
            raise BenchmarkEvaluationError(
                "native reference requires an immutable native training contract"
            )
        contract_path = Path(evidence.native_contract_path)
        try:
            _reject_symlink_components(contract_path)
            if hashlib.sha256(contract_path.read_bytes()).hexdigest() != (
                evidence.native_contract_sha256
            ):
                raise BenchmarkEvaluationError("native contract hash mismatch")
            kind = (
                "audiogs"
                if evidence.system_name == "native_audiogs"
                else "ftgspp"
            )
            contract = verify_native_contract(
                contract_path,
                expected_scene=evidence.scene_id,
                expected_model_kind=kind,
            )
        except Exception as error:
            if isinstance(error, BenchmarkEvaluationError):
                raise
            raise BenchmarkEvaluationError(
                f"native training contract verification failed: {error}"
            ) from error
        checkpoint = contract["checkpoint"]
        if (
            checkpoint["path"] != str(Path(evidence.checkpoint_path).resolve())
            or checkpoint["sha256"] != evidence.checkpoint_sha256
            or contract["inputs"]["protocol_config"]["sha256"]
            != evidence.config_sha256
            or contract["upstream"]["source_sha256"] != evidence.source_sha256
        ):
            raise BenchmarkEvaluationError(
                "native training contract/evaluation evidence mismatch"
            )


class BenchmarkEvaluator:
    def __init__(
        self,
        device: torch.device | str,
        *,
        strict_protocol: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.strict_protocol = strict_protocol

    def evaluate(
        self,
        *,
        identity: EvaluationIdentity,
        evidence: TrainingEvidence,
        runtime_factory: Callable[[], BenchmarkEvaluationRuntime],
        predictor_factory: Callable[
            [BenchmarkEvaluationRuntime], Callable[[AlignedAVSample], BenchmarkPrediction]
        ],
        output_dir: Path | str,
        resume: bool = False,
        overwrite: bool = False,
    ) -> BenchmarkEvaluationResult:
        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        evidence.validate(identity)  # Must happen before cam38 construction.
        if self.strict_protocol:
            expected = SCENE_SAMPLE_COUNTS.get(identity.scene_id)
            if expected is None or identity.expected_sample_count != expected:
                raise BenchmarkEvaluationError(
                    "strict benchmark sample count must be 130/293 by scene"
                )
            if evidence.role == "continuation":
                _audit_continuation_evidence(evidence, identity)
            else:
                audit_training_evidence(evidence, identity)
        output = Path(output_dir)
        if resume:
            result = load_evaluation(output, identity=identity)
            if result.provenance != self._provenance(evidence):
                raise BenchmarkEvaluationError("resumed evaluation provenance mismatch")
            return result
        if (output / "current.json").exists() and not overwrite:
            raise BenchmarkEvaluationError("evaluation already exists; use resume or overwrite")

        runtime = runtime_factory()
        if not isinstance(runtime, BenchmarkEvaluationRuntime):
            raise BenchmarkEvaluationError(
                "runtime_factory must return BenchmarkEvaluationRuntime"
            )
        predictor = predictor_factory(runtime)
        if not callable(predictor) or not callable(runtime.audio_loss_fn):
            raise BenchmarkEvaluationError("evaluation runtime factories are invalid")
        registry, directions, modalities = self._validate_metric_runtime(runtime)
        samples = tuple(runtime.samples)
        if len(samples) != identity.expected_sample_count:
            raise BenchmarkEvaluationError("cam38 sample count mismatch")
        actual_ids = tuple(benchmark_sample_id(sample) for sample in samples)
        if actual_ids != identity.expected_sample_ids:
            raise BenchmarkEvaluationError("cam38 sample IDs/order mismatch")
        with torch.no_grad():
            rows = tuple(
                self._evaluate_one(
                    sample,
                    predictor,
                    audio_loss_fn=runtime.audio_loss_fn,
                    extra_metric_fns=registry,
                    extra_metric_modalities=modalities,
                    lpips_fn=runtime.lpips_fn,
                )
                for sample in samples
            )
        metadata = {"sample_id", "scene_id", "camera", "frame_index", "time_seconds"}
        metric_names = tuple(name for name in rows[0] if name not in metadata)
        if not metric_names:
            raise BenchmarkEvaluationError("evaluation produced no common metrics")
        if any(
            set(row) - {"sample_id", "scene_id", "camera", "frame_index", "time_seconds"}
            != set(metric_names)
            for row in rows
        ):
            raise BenchmarkEvaluationError("per-sample metric availability differs")
        expected_core = (
            set(ALL_METRICS)
            if evidence.role == "continuation"
            else set(AUDIO_METRICS)
            if evidence.system_name == "native_audiogs"
            else set(VIDEO_METRICS)
        )
        available_modalities: set[str] = set()
        if set(AUDIO_METRICS).issubset(expected_core):
            available_modalities.add("audio")
        if set(VIDEO_METRICS).issubset(expected_core):
            available_modalities.add("video")
        expected_metrics = {
            *expected_core,
            *(
                name
                for name in registry
                if modalities[name] in available_modalities
            ),
            *(
                ("rgb_lpips",)
                if runtime.lpips_fn is not None and "video" in available_modalities
                else ()
            ),
        }
        if set(metric_names) != expected_metrics:
            raise BenchmarkEvaluationError(
                "system did not produce its exact required metric schema"
            )
        summary = aggregate_metrics(
            [{name: float(row[name]) for name in metric_names} for row in rows]
        )
        provenance = self._provenance(evidence)
        metric_directions = {
            name: (
                "lower_is_better"
                if name == "rgb_lpips"
                else directions[name]
                if name in directions
                else METRIC_DIRECTIONS[name]
            )
            for name in metric_names
        }
        metric_protocol = {
            "psnr_cap_db": PSNR_CAP_DB,
            "lpips_implementation_sha256": runtime.lpips_implementation_sha256,
            "extra_metric_registry": {
                name: {
                    "direction": directions[name],
                    "modality": modalities[name],
                }
                for name in sorted(registry)
            },
        }
        document = {
            "schema": SCHEMA,
            "version": 1,
            "identity": identity.to_mapping(),
            "count": len(rows),
            "summary": summary,
            "provenance": provenance,
            "metric_directions": metric_directions,
            "metric_protocol": metric_protocol,
        }
        candidate = BenchmarkEvaluationResult(
            identity,
            len(rows),
            rows,
            summary,
            metric_directions,
            metric_protocol,
            provenance,
            "",
            None,
        )
        _validate_loaded_result(candidate)
        jsonl = b"".join(canonical_json(row) for row in rows)
        csv_stream = io.StringIO(newline="")
        writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        generation, digest = publish_generation(
            output,
            schema=SCHEMA,
            identity=identity.to_mapping(),
            files={
                "metrics_per_sample.jsonl": jsonl,
                "metrics_per_sample.csv": csv_stream.getvalue().encode(),
                "metrics_summary.json": canonical_json(document),
            },
            overwrite=overwrite,
        )
        return BenchmarkEvaluationResult(
            identity,
            len(rows),
            rows,
            summary,
            metric_directions,
            metric_protocol,
            provenance,
            digest,
            generation,
        )

    @staticmethod
    def _provenance(evidence: TrainingEvidence) -> dict[str, object]:
        value = asdict(evidence)
        value["train_cameras"] = list(evidence.train_cameras)
        value["update_matched"] = evidence.role == "continuation"
        return value

    @staticmethod
    def _validate_metric_runtime(
        runtime: BenchmarkEvaluationRuntime,
    ) -> tuple[
        Mapping[str, Callable[[BenchmarkPrediction, AlignedAVSample], object]],
        Mapping[str, str],
        Mapping[str, str],
    ]:
        registry = runtime.extra_metric_fns or {}
        directions = runtime.extra_metric_directions or {}
        modalities = runtime.extra_metric_modalities or {}
        if set(registry) != set(directions) or set(registry) != set(modalities):
            raise BenchmarkEvaluationError(
                "extra metric registry and direction allowlist must match exactly"
            )
        for name, function in registry.items():
            if (
                not isinstance(name, str)
                or not name
                or name in _RESERVED_EXTRA_METRICS
                or "depth" in name.lower()
                or not callable(function)
                or directions[name] not in {"lower_is_better", "higher_is_better"}
                or modalities[name] not in {"audio", "video"}
            ):
                raise BenchmarkEvaluationError(
                    f"invalid or reserved extra metric registration: {name!r}"
                )
        if (runtime.lpips_fn is None) != (
            runtime.lpips_implementation_sha256 is None
        ):
            raise BenchmarkEvaluationError(
                "LPIPS function and implementation hash must be configured together"
            )
        if runtime.lpips_fn is not None:
            if not callable(runtime.lpips_fn):
                raise BenchmarkEvaluationError("LPIPS implementation must be callable")
            try:
                _digest(
                    runtime.lpips_implementation_sha256,
                    "lpips_implementation_sha256",
                )
            except ValueError as error:
                raise BenchmarkEvaluationError(str(error)) from error
        return registry, directions, modalities

    def _evaluate_one(
        self,
        sample: AlignedAVSample,
        predictor: Callable[[AlignedAVSample], BenchmarkPrediction],
        *,
        audio_loss_fn: Callable[[Tensor, Tensor], Mapping[str, object]],
        extra_metric_fns: Mapping[
            str, Callable[[BenchmarkPrediction, AlignedAVSample], object]
        ],
        extra_metric_modalities: Mapping[str, str],
        lpips_fn: Callable[[Tensor, Tensor], object] | None,
    ) -> dict[str, object]:
        if sample.camera != TEST_CAMERA:
            raise BenchmarkEvaluationError("final evaluator accepts only cam38")
        sample = move_sample(sample, self.device)
        prediction = predictor(sample)
        if not isinstance(prediction, BenchmarkPrediction):
            raise BenchmarkEvaluationError("predictor must return BenchmarkPrediction")
        metrics: dict[str, float] = {}
        if prediction.predicted_audio is not None:
            losses = audio_loss_fn(prediction.predicted_audio, sample.target_audio)
            required = {"total_loss", "mono_loss", "diff_loss"}
            if not isinstance(losses, Mapping) or not required.issubset(losses):
                raise BenchmarkEvaluationError("audio loss mapping is incomplete")
            metrics.update(
                audio_total=_scalar(losses["total_loss"], "audio_total"),
                audio_mono=_scalar(losses["mono_loss"], "audio_mono"),
                audio_diff=_scalar(losses["diff_loss"], "audio_diff"),
                waveform_l1=waveform_l1(prediction.predicted_audio, sample.target_audio),
                mono_lsd=log_spectral_distance(
                    prediction.predicted_audio, sample.target_audio, "mono"
                ),
                diff_lsd=log_spectral_distance(
                    prediction.predicted_audio, sample.target_audio, "diff"
                ),
                lre_error_db=lre_error_db(
                    prediction.predicted_audio, sample.target_audio
                ),
            )
        if prediction.rendered_rgb is not None:
            metrics.update(
                rgb_psnr=min(
                    psnr(prediction.rendered_rgb, sample.target_rgb),
                    PSNR_CAP_DB,
                ),
                rgb_ssim=ssim(prediction.rendered_rgb, sample.target_rgb),
                rgb_l1=rgb_l1(prediction.rendered_rgb, sample.target_rgb),
            )
            if lpips_fn is not None:
                metrics["rgb_lpips"] = _scalar(
                    lpips_fn(prediction.rendered_rgb, sample.target_rgb),
                    "rgb_lpips",
                )
        for name, function in extra_metric_fns.items():
            if (
                extra_metric_modalities[name] == "audio"
                and prediction.predicted_audio is not None
            ) or (
                extra_metric_modalities[name] == "video"
                and prediction.rendered_rgb is not None
            ):
                metrics[name] = _scalar(function(prediction, sample), name)
        if not metrics:
            raise BenchmarkEvaluationError("prediction has neither audio nor RGB")
        if any(not math.isfinite(value) for value in metrics.values()):
            raise BenchmarkEvaluationError("all metrics must be finite")
        return {
            "sample_id": benchmark_sample_id(sample),
            "scene_id": sample.scene_id,
            "camera": sample.camera,
            "frame_index": sample.frame_index,
            "time_seconds": sample.time_seconds,
            **metrics,
        }

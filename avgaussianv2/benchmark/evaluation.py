"""Strict common evaluator for the dual-scene cam38 benchmark."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
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
from avgaussianv2.benchmark.training import TEST_CAMERA, TRAIN_CAMERAS
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
CONTINUATION_SYSTEMS = {"joint_conditioned", "audio_only", "visual_only"}
NATIVE_SYSTEMS = {"native_audiogs", "native_ftgspp"}


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
            if self.system_name not in NATIVE_SYSTEMS or identity.reporting_step is not None:
                raise BenchmarkEvaluationError("native reference identity mismatch")
            if self.index_sha256 is not None:
                raise BenchmarkEvaluationError("native reference is not update-matched")
            if self.system_name == "native_audiogs":
                if (
                    self.epochs != 61.0
                    or not isinstance(self.completed_updates, int)
                    or self.completed_updates <= 0
                    or self.planned_updates != self.completed_updates
                    or self.checkpoint_step != self.completed_updates
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
    extra_audio_metrics: Mapping[str, object] | None = None
    lpips: object | None = None


@dataclass(frozen=True)
class BenchmarkEvaluationResult:
    identity: EvaluationIdentity
    count: int
    rows: tuple[dict[str, object], ...]
    summary: dict[str, dict[str, float]]
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
        != {"schema", "version", "identity", "count", "summary", "provenance", "metric_directions"}
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
) -> BenchmarkEvaluationResult:
    """Verify an existing evaluation without runtime construction or mutation."""
    result = load_evaluation(output_dir, identity=identity)
    if verify_checkpoint:
        _verify_checkpoint(
            result.provenance["checkpoint_path"],
            result.provenance["checkpoint_sha256"],
        )
    return result


def _verify_checkpoint(path_value: object, expected_digest: object) -> None:
    checkpoint = Path(str(path_value))
    if (
        not isinstance(path_value, str)
        or not checkpoint.is_absolute()
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
    ):
        raise BenchmarkEvaluationError(
            "strict benchmark checkpoint is missing or unsafe"
        )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise BenchmarkEvaluationError("strict benchmark checkpoint hash mismatch")


class BenchmarkEvaluator:
    def __init__(
        self,
        audio_loss_fn: Callable[[Tensor, Tensor], Mapping[str, object]],
        device: torch.device | str,
        *,
        strict_protocol: bool = True,
    ) -> None:
        self.audio_loss_fn = audio_loss_fn
        self.device = torch.device(device)
        self.strict_protocol = strict_protocol

    def evaluate(
        self,
        *,
        identity: EvaluationIdentity,
        evidence: TrainingEvidence,
        sample_factory: Callable[[], Sequence[AlignedAVSample]],
        predictor: Callable[[AlignedAVSample], BenchmarkPrediction],
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
            _verify_checkpoint(evidence.checkpoint_path, evidence.checkpoint_sha256)
        output = Path(output_dir)
        if resume:
            result = load_evaluation(output, identity=identity)
            if result.provenance != self._provenance(evidence):
                raise BenchmarkEvaluationError("resumed evaluation provenance mismatch")
            return result
        if (output / "current.json").exists() and not overwrite:
            raise BenchmarkEvaluationError("evaluation already exists; use resume or overwrite")

        samples = tuple(sample_factory())
        if len(samples) != identity.expected_sample_count:
            raise BenchmarkEvaluationError("cam38 sample count mismatch")
        actual_ids = tuple(benchmark_sample_id(sample) for sample in samples)
        if actual_ids != identity.expected_sample_ids:
            raise BenchmarkEvaluationError("cam38 sample IDs/order mismatch")
        with torch.no_grad():
            rows = tuple(self._evaluate_one(sample, predictor) for sample in samples)
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
        summary = aggregate_metrics(
            [{name: float(row[name]) for name in metric_names} for row in rows]
        )
        provenance = self._provenance(evidence)
        document = {
            "schema": SCHEMA,
            "version": 1,
            "identity": identity.to_mapping(),
            "count": len(rows),
            "summary": summary,
            "provenance": provenance,
            "metric_directions": {
                name: METRIC_DIRECTIONS.get(name, "lower_is_better")
                for name in metric_names
            },
        }
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
        )
        return BenchmarkEvaluationResult(
            identity, len(rows), rows, summary, provenance, digest, generation
        )

    @staticmethod
    def _provenance(evidence: TrainingEvidence) -> dict[str, object]:
        value = asdict(evidence)
        value["train_cameras"] = list(evidence.train_cameras)
        value["update_matched"] = evidence.role == "continuation"
        return value

    def _evaluate_one(
        self,
        sample: AlignedAVSample,
        predictor: Callable[[AlignedAVSample], BenchmarkPrediction],
    ) -> dict[str, object]:
        if sample.camera != TEST_CAMERA:
            raise BenchmarkEvaluationError("final evaluator accepts only cam38")
        sample = move_sample(sample, self.device)
        prediction = predictor(sample)
        if not isinstance(prediction, BenchmarkPrediction):
            raise BenchmarkEvaluationError("predictor must return BenchmarkPrediction")
        metrics: dict[str, float] = {}
        if prediction.predicted_audio is not None:
            losses = self.audio_loss_fn(prediction.predicted_audio, sample.target_audio)
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
            if prediction.extra_audio_metrics:
                for name, value in prediction.extra_audio_metrics.items():
                    if name in metrics or name in VIDEO_METRICS:
                        raise BenchmarkEvaluationError("duplicate/reserved audio metric")
                    metrics[name] = _scalar(value, name)
        if prediction.rendered_rgb is not None:
            metrics.update(
                rgb_psnr=psnr(prediction.rendered_rgb, sample.target_rgb),
                rgb_ssim=ssim(prediction.rendered_rgb, sample.target_rgb),
                rgb_l1=rgb_l1(prediction.rendered_rgb, sample.target_rgb),
            )
            if prediction.lpips is not None:
                metrics["rgb_lpips"] = _scalar(prediction.lpips, "rgb_lpips")
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

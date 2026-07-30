"""Strict per-scene and two-scene reports for the cam38 benchmark."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from avgaussianv2.benchmark.artifacts import (
    ArtifactError,
    canonical_json,
    load_generation,
    publish_generation,
    sha256,
)
from avgaussianv2.benchmark.evaluation import (
    ALL_METRICS,
    AUDIO_METRICS,
    CONTINUATION_SYSTEMS,
    NATIVE_SYSTEMS,
    REPORTING_STEPS,
    SCENE_SAMPLE_COUNTS,
    VIDEO_METRICS,
    BenchmarkEvaluationResult,
    TrainingEvidence,
    verify_evaluation,
)
from avgaussianv2.benchmark.training import TEST_CAMERA, TRAIN_CAMERAS
from avgaussianv2.benchmark.metrics import aggregate_metrics

SCENE_SCHEMA = "avgaussianv2.cam38-benchmark-scene-report"
SUITE_SCHEMA = "avgaussianv2.cam38-benchmark-suite-report"
PRIMARY_STEP = 30_000


class BenchmarkReportError(RuntimeError):
    pass


def _key(result: BenchmarkEvaluationResult) -> tuple[str, int | None]:
    return result.identity.system_name, result.identity.reporting_step


def _metric_names(result: BenchmarkEvaluationResult) -> tuple[str, ...]:
    if not result.rows:
        raise BenchmarkReportError("evaluation rows must not be empty")
    metadata = {"sample_id", "scene_id", "camera", "frame_index", "time_seconds"}
    return tuple(name for name in result.rows[0] if name not in metadata)


def _validate_result(
    result: BenchmarkEvaluationResult, scene_id: str, count: int
) -> None:
    if result.identity.scene_id != scene_id or result.count != count:
        raise BenchmarkReportError("evaluation scene/count mismatch")
    if tuple(row["sample_id"] for row in result.rows) != result.identity.expected_sample_ids:
        raise BenchmarkReportError("evaluation sample IDs/order mismatch")
    if any(
        row.get("scene_id") != scene_id or row.get("camera") != TEST_CAMERA
        for row in result.rows
    ):
        raise BenchmarkReportError("evaluation must contain only scene cam38 rows")
    for row in result.rows:
        for name in _metric_names(result):
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise BenchmarkReportError("all report metrics must be finite")
    recalculated = aggregate_metrics(
        [
            {name: float(row[name]) for name in _metric_names(result)}
            for row in result.rows
        ]
    )
    if recalculated != result.summary:
        raise BenchmarkReportError("evaluation summary does not match per-sample rows")
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
    ):
        raise BenchmarkReportError("evaluation metric schema is incomplete")
    provenance = result.provenance
    if (
        provenance.get("test_camera") != TEST_CAMERA
        or tuple(provenance.get("train_cameras", ())) != TRAIN_CAMERAS
        or provenance.get("test_targets_read_during_training") is not False
        or provenance.get("seed") != 42
    ):
        raise BenchmarkReportError("split/no-test-pretraining evidence failed")
    try:
        evidence_fields = set(TrainingEvidence.__dataclass_fields__)
        evidence = TrainingEvidence(
            **{
                **{name: provenance[name] for name in evidence_fields},
                "train_cameras": tuple(provenance["train_cameras"]),
            }
        )
        evidence.validate(result.identity)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BenchmarkReportError(
            f"evaluation provenance gate failed: {error}"
        ) from error


def _validate_fairness(
    indexed: Mapping[tuple[str, int | None], BenchmarkEvaluationResult],
) -> None:
    shared_fields = (
        "visual_initialization_sha256",
        "audio_initialization_sha256",
        "model_initialization_sha256",
        "index_sha256",
        "seed",
        "planned_updates",
        "batch_size",
        "source_sha256",
        "config_sha256",
        "runtime_contract_sha256",
    )
    reference = indexed[("joint_conditioned", REPORTING_STEPS[0])].provenance
    for step in REPORTING_STEPS:
        results = [indexed[(system, step)] for system in sorted(CONTINUATION_SYSTEMS)]
        for result in results:
            provenance = result.provenance
            if provenance.get("role") != "continuation" or provenance.get(
                "update_matched"
            ) is not True:
                raise BenchmarkReportError("continuation update-matched label missing")
            if (
                provenance.get("completed_updates") != step
                or provenance.get("checkpoint_step") != step
                or result.identity.reporting_step != step
            ):
                raise BenchmarkReportError("continuation final step mismatch")
            for field in shared_fields:
                if provenance.get(field) != reference.get(field):
                    label = "index" if field == "index_sha256" else field
                    raise BenchmarkReportError(f"continuation {label} mismatch")
    for system in NATIVE_SYSTEMS:
        provenance = indexed[(system, None)].provenance
        if (
            provenance.get("role") != "native_reference"
            or provenance.get("update_matched") is not False
        ):
            raise BenchmarkReportError(
                "native references must be labeled non-update-matched"
            )


def _paired(
    left: BenchmarkEvaluationResult,
    right: BenchmarkEvaluationResult,
    metrics: Sequence[str],
) -> dict[str, dict[str, object]]:
    left_rows = {row["sample_id"]: row for row in left.rows}
    right_rows = {row["sample_id"]: row for row in right.rows}
    if tuple(left_rows) != tuple(right_rows):
        raise BenchmarkReportError("paired evaluations require identical sample IDs")
    result: dict[str, dict[str, object]] = {}
    for metric in metrics:
        if metric not in _metric_names(left) or metric not in _metric_names(right):
            continue
        deltas = [
            float(left_rows[sample_id][metric]) - float(right_rows[sample_id][metric])
            for sample_id in left_rows
        ]
        if left.metric_directions[metric] != right.metric_directions[metric]:
            raise BenchmarkReportError("paired metric direction mismatch")
        direction = left.metric_directions[metric]
        wins = [
            delta > 0 if direction == "higher_is_better" else delta < 0
            for delta in deltas
        ]
        statistics = aggregate_metrics([{"delta": delta} for delta in deltas])["delta"]
        result[metric] = {
            "direction": direction,
            "mean_delta": statistics["mean"],
            "std_delta": statistics["std"],
            "median_delta": statistics["median"],
            "win_rate": sum(wins) / len(wins),
            "count": len(deltas),
        }
    return result


def _report_identity(
    kind: str,
    inputs: Sequence[BenchmarkEvaluationResult] | Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if kind == "scene":
        return {
            "kind": kind,
            "inputs": [
                {
                    "system_name": item.identity.system_name,
                    "reporting_step": item.identity.reporting_step,
                    "sha256": item.content_sha256,
                }
                for item in sorted(inputs, key=lambda value: str(_key(value)))  # type: ignore[arg-type]
            ],
        }
    return {
        "kind": kind,
        "inputs": [
            {
                "scene_id": item["scene_id"],  # type: ignore[index]
                "sha256": item["content_sha256"],  # type: ignore[index]
            }
            for item in sorted(inputs, key=lambda value: value["scene_id"])  # type: ignore[index]
        ],
    }


def _markdown_scene(report: Mapping[str, object]) -> str:
    lines = [
        f"# Cam38 benchmark: {report['scene_id']}",
        "",
        f"Test samples: {report['sample_count']}; primary step: {report['primary_step']}.",
        "",
        "Native AudioGS/FreeTimeGS++ references are descriptive and are not update-matched.",
        "",
        "| Step | System | Audio total | PSNR | SSIM | RGB L1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    scaling = report["scaling"]
    for step in ("5000", "10000", "30000"):
        for system in sorted(CONTINUATION_SYSTEMS):
            summary = scaling[step][system]["summary"]

            def value(name: str) -> object:
                return summary.get(name, {}).get("mean", "")

            lines.append(
                f"| {step} | {system} | {value('audio_total')} | "
                f"{value('rgb_psnr')} | {value('rgb_ssim')} | {value('rgb_l1')} |"
            )
    lines.append("")
    return "\n".join(lines)


def _flat_rows(
    evaluations: Sequence[BenchmarkEvaluationResult],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for evaluation in sorted(evaluations, key=lambda value: str(_key(value))):
        for row in evaluation.rows:
            rows.append(
                {
                    "system_name": evaluation.identity.system_name,
                    "reporting_step": evaluation.identity.reporting_step,
                    **row,
                }
            )
    return tuple(rows)


def _load_report(output_dir: Path, schema: str) -> dict[str, object]:
    try:
        _, files, _, _ = load_generation(output_dir, schema=schema)
        report = json.loads(files["report.json"])
    except (ArtifactError, KeyError, ValueError) as error:
        raise BenchmarkReportError(f"cannot load report: {error}") from error
    if not isinstance(report, dict) or report.get("schema") != schema:
        raise BenchmarkReportError("report schema mismatch")
    content = dict(report)
    digest = content.pop("content_sha256", None)
    if digest != sha256(canonical_json(content)):
        raise BenchmarkReportError("report content hash mismatch")
    return report


def load_report(output_dir: Path | str) -> dict[str, object]:
    output = Path(output_dir)
    for schema in (SCENE_SCHEMA, SUITE_SCHEMA):
        try:
            return _load_report(output, schema)
        except BenchmarkReportError:
            continue
    raise BenchmarkReportError("no valid scene or suite report")


def verify_scene_report(
    output_dir: Path | str, *, verify_inputs: bool = True
) -> dict[str, object]:
    """Verify an existing scene report without evaluating or mutating outputs."""
    report = _load_report(Path(output_dir), SCENE_SCHEMA)
    if verify_inputs:
        for artifact in report["evaluation_artifacts"]:
            if not isinstance(artifact.get("root"), str):
                raise BenchmarkReportError("scene report lacks evaluation artifact root")
            verified = verify_evaluation(Path(artifact["root"]))
            if verified.content_sha256 != artifact["sha256"]:
                raise BenchmarkReportError("scene report evaluation hash mismatch")
    return report


def verify_suite_report(
    output_dir: Path | str, *, verify_inputs: bool = True
) -> dict[str, object]:
    """Verify an existing suite report without evaluating or mutating outputs."""
    report = _load_report(Path(output_dir), SUITE_SCHEMA)
    if verify_inputs:
        for scene in report["scenes"].values():
            verified = verify_scene_report(Path(scene["report_root"]))
            if verified["content_sha256"] != scene["content_sha256"]:
                raise BenchmarkReportError("suite scene report hash mismatch")
    return report


def build_scene_report(
    *,
    scene_id: str,
    evaluations: Sequence[BenchmarkEvaluationResult],
    expected_sample_count: int | None = None,
    output_dir: Path | str,
    resume: bool = False,
    overwrite: bool = False,
    strict_protocol: bool = True,
) -> dict[str, object]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if expected_sample_count is None:
        try:
            expected_sample_count = SCENE_SAMPLE_COUNTS[scene_id]
        except KeyError as error:
            raise BenchmarkReportError("unsupported benchmark scene") from error
    if strict_protocol and SCENE_SAMPLE_COUNTS.get(scene_id) != expected_sample_count:
        raise BenchmarkReportError(
            "strict scene sample count must be 130/293 by scene"
        )
    indexed = {_key(result): result for result in evaluations}
    expected_keys = {
        *((system, step) for system in CONTINUATION_SYSTEMS for step in REPORTING_STEPS),
        *((system, None) for system in NATIVE_SYSTEMS),
    }
    if set(indexed) != expected_keys or len(indexed) != len(evaluations):
        raise BenchmarkReportError("scene report requires exact declared systems/steps")
    for result in indexed.values():
        _validate_result(result, scene_id, expected_sample_count)
        if strict_protocol:
            if result.generation_path is None:
                raise BenchmarkReportError(
                    "strict report inputs must be immutable evaluation artifacts"
                )
            verified = verify_evaluation(
                result.generation_path.parent.parent,
                identity=result.identity,
            )
            if (
                verified.content_sha256 != result.content_sha256
                or verified.rows != result.rows
                or verified.summary != result.summary
                or verified.metric_directions != result.metric_directions
                or verified.metric_protocol != result.metric_protocol
                or verified.provenance != result.provenance
            ):
                raise BenchmarkReportError(
                    "strict report evaluation artifact identity mismatch"
                )
    common_ids = indexed[("joint_conditioned", PRIMARY_STEP)].identity.expected_sample_ids
    if any(result.identity.expected_sample_ids != common_ids for result in indexed.values()):
        raise BenchmarkReportError("all systems require identical sample IDs")
    _validate_fairness(indexed)
    reference_protocol = indexed[
        ("joint_conditioned", REPORTING_STEPS[0])
    ].metric_protocol
    registry = reference_protocol["extra_metric_registry"]
    if not isinstance(registry, Mapping):
        raise BenchmarkReportError("extra metric registry must be a mapping")
    for (system, _), result in indexed.items():
        if result.metric_protocol != reference_protocol:
            raise BenchmarkReportError(
                "all systems/steps must use one exact metric protocol"
            )
        core = (
            set(ALL_METRICS)
            if system in CONTINUATION_SYSTEMS
            else set(AUDIO_METRICS)
            if system == "native_audiogs"
            else set(VIDEO_METRICS)
        )
        modalities: set[str] = set()
        if set(AUDIO_METRICS).issubset(core):
            modalities.add("audio")
        if set(VIDEO_METRICS).issubset(core):
            modalities.add("video")
        extras = {
            name
            for name, specification in registry.items()
            if specification["modality"] in modalities
        }
        if "video" in modalities and reference_protocol[
            "lpips_implementation_sha256"
        ] is not None:
            extras.add("rgb_lpips")
        if set(result.summary) != core | extras:
            raise BenchmarkReportError(
                "system/step metric set is incomplete or contains unknown metrics"
            )
    video_results = [
        result
        for result in indexed.values()
        if set(VIDEO_METRICS).issubset(_metric_names(result))
    ]
    lpips_presence = {"rgb_lpips" in _metric_names(result) for result in video_results}
    if len(lpips_presence) > 1:
        raise BenchmarkReportError(
            "LPIPS must use one identically available implementation for all visual systems"
        )
    identity = _report_identity("scene", evaluations)
    output = Path(output_dir)
    if resume:
        try:
            _, _, manifest, _ = load_generation(
                output, schema=SCENE_SCHEMA, expected_identity=identity
            )
        except ArtifactError as error:
            raise BenchmarkReportError(str(error)) from error
        return _load_report(output, SCENE_SCHEMA)
    if (output / "current.json").exists() and not overwrite:
        raise BenchmarkReportError("scene report exists; use resume or overwrite")

    scaling = {
        str(step): {
            system: {
                "summary": indexed[(system, step)].summary,
                "checkpoint_sha256": indexed[(system, step)].provenance[
                    "checkpoint_sha256"
                ],
            }
            for system in sorted(CONTINUATION_SYSTEMS)
        }
        for step in REPORTING_STEPS
    }
    systems = {
        system: {
            "comparison_class": (
                "native_reference_non_update_matched"
                if system in NATIVE_SYSTEMS
                else "update_matched_continuation"
            ),
            "primary_summary": indexed[
                (system, None if system in NATIVE_SYSTEMS else PRIMARY_STEP)
            ].summary,
            "provenance": indexed[
                (system, None if system in NATIVE_SYSTEMS else PRIMARY_STEP)
            ].provenance,
            "metric_directions": indexed[
                (system, None if system in NATIVE_SYSTEMS else PRIMARY_STEP)
            ].metric_directions,
            "metric_protocol": indexed[
                (system, None if system in NATIVE_SYSTEMS else PRIMARY_STEP)
            ].metric_protocol,
        }
        for system in sorted((*CONTINUATION_SYSTEMS, *NATIVE_SYSTEMS))
    }
    paired = {
        "joint_vs_audio_only": _paired(
            indexed[("joint_conditioned", PRIMARY_STEP)],
            indexed[("audio_only", PRIMARY_STEP)],
            AUDIO_METRICS,
        ),
        "joint_vs_visual_only": _paired(
            indexed[("joint_conditioned", PRIMARY_STEP)],
            indexed[("visual_only", PRIMARY_STEP)],
            VIDEO_METRICS,
        ),
    }
    paired_by_step = {
        str(step): {
            "joint_vs_audio_only": _paired(
                indexed[("joint_conditioned", step)],
                indexed[("audio_only", step)],
                AUDIO_METRICS,
            ),
            "joint_vs_visual_only": _paired(
                indexed[("joint_conditioned", step)],
                indexed[("visual_only", step)],
                (*VIDEO_METRICS, "rgb_lpips"),
            ),
        }
        for step in REPORTING_STEPS
    }
    descriptive_native = {
        "joint_vs_native_audiogs": {
            "comparison_class": "descriptive_non_update_matched",
            "metrics": _paired(
                indexed[("joint_conditioned", PRIMARY_STEP)],
                indexed[("native_audiogs", None)],
                AUDIO_METRICS,
            ),
        },
        "joint_vs_native_ftgspp": {
            "comparison_class": "descriptive_non_update_matched",
            "metrics": _paired(
                indexed[("joint_conditioned", PRIMARY_STEP)],
                indexed[("native_ftgspp", None)],
                (*VIDEO_METRICS, "rgb_lpips"),
            ),
        },
    }
    base: dict[str, object] = {
        "schema": SCENE_SCHEMA,
        "version": 1,
        "scene_id": scene_id,
        "test_camera": TEST_CAMERA,
        "train_cameras": list(TRAIN_CAMERAS),
        "sample_count": expected_sample_count,
        "sample_ids": list(common_ids),
        "reporting_steps": list(REPORTING_STEPS),
        "primary_step": PRIMARY_STEP,
        "systems": systems,
        "scaling": scaling,
        "paired": paired,
        "paired_by_step": paired_by_step,
        "descriptive_native_comparisons": descriptive_native,
        "artifact_root": str(output.absolute()),
        "evaluation_artifacts": [
            {
                "system_name": result.identity.system_name,
                "reporting_step": result.identity.reporting_step,
                "root": (
                    str(result.generation_path.parent.parent)
                    if result.generation_path is not None
                    else None
                ),
                "sha256": result.content_sha256,
            }
            for result in sorted(evaluations, key=lambda value: str(_key(value)))
        ],
        "depth_accuracy_reported": False,
        "lpips_policy": "reported_only_when_identically_available_to_all_compared_systems",
    }
    report = {**base, "content_sha256": sha256(canonical_json(base))}
    flat = _flat_rows(evaluations)
    csv_stream = io.StringIO(newline="")
    fieldnames = list(
        dict.fromkeys(name for row in flat for name in row)
    )
    writer = csv.DictWriter(csv_stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flat)
    publish_generation(
        output,
        schema=SCENE_SCHEMA,
        identity=identity,
        files={
            "report.json": canonical_json(report),
            "report.md": _markdown_scene(report).encode(),
            "per_sample.jsonl": b"".join(canonical_json(row) for row in flat),
            "per_sample.csv": csv_stream.getvalue().encode(),
        },
        overwrite=overwrite,
    )
    return report


def _aggregate_suite(
    scene_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    accumulator: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for report in scene_reports:
        count = int(report["sample_count"])
        for step, systems in report["scaling"].items():
            for system, values in systems.items():
                for metric, stats in values["summary"].items():
                    accumulator[(step, system, metric)].append(
                        (float(stats["mean"]), count)
                    )
        for system, values in report["systems"].items():
            if values["comparison_class"] != "native_reference_non_update_matched":
                continue
            for metric, stats in values["primary_summary"].items():
                accumulator[("native", system, metric)].append(
                    (float(stats["mean"]), count)
                )
    macro: dict[str, object] = {}
    micro: dict[str, object] = {}
    for (step, system, metric), values in sorted(accumulator.items()):
        if len(values) != len(SCENE_SAMPLE_COUNTS):
            raise BenchmarkReportError(
                "suite aggregate must contain exactly both scenes"
            )
        macro.setdefault(step, {}).setdefault(system, {})[metric] = sum(
            value for value, _ in values
        ) / len(values)
        total = sum(count for _, count in values)
        micro.setdefault(step, {}).setdefault(system, {})[metric] = sum(
            value * count for value, count in values
        ) / total
    return {"macro": macro, "micro": micro}


def _aggregate_suite_pairs(
    scene_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {"macro": {}, "micro": {}}
    comparisons = {
        "joint_vs_audio_only": AUDIO_METRICS,
        "joint_vs_visual_only": VIDEO_METRICS,
    }
    for comparison, expected_metrics in comparisons.items():
        available = [
            set(report["paired"][comparison]) for report in scene_reports
        ]
        if any(metrics != available[0] for metrics in available[1:]):
            raise BenchmarkReportError("suite paired metric set mismatch")
        if not set(expected_metrics).issubset(available[0]):
            raise BenchmarkReportError("suite paired metrics are incomplete")
        for metric in sorted(available[0]):
            values = [
                report["paired"][comparison][metric] for report in scene_reports
            ]
            counts = [int(value["count"]) for value in values]
            if counts != [int(report["sample_count"]) for report in scene_reports]:
                raise BenchmarkReportError("suite paired sample count mismatch")
            result["macro"].setdefault(comparison, {})[metric] = {
                "mean_delta": sum(value["mean_delta"] for value in values)
                / len(values),
                "win_rate": sum(value["win_rate"] for value in values) / len(values),
            }
            total = sum(counts)
            result["micro"].setdefault(comparison, {})[metric] = {
                "mean_delta": sum(
                    value["mean_delta"] * count
                    for value, count in zip(values, counts)
                )
                / total,
                "win_rate": sum(
                    value["win_rate"] * count
                    for value, count in zip(values, counts)
                )
                / total,
            }
    return result


def build_suite_report(
    *,
    scene_reports: Sequence[Mapping[str, object]],
    output_dir: Path | str,
    resume: bool = False,
    overwrite: bool = False,
    strict_protocol: bool = True,
) -> dict[str, object]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    indexed = {report.get("scene_id"): report for report in scene_reports}
    if set(indexed) != set(SCENE_SAMPLE_COUNTS) or len(indexed) != len(scene_reports):
        raise BenchmarkReportError("suite requires exactly both benchmark scenes")
    for scene, count in SCENE_SAMPLE_COUNTS.items():
        report = indexed[scene]
        if (
            report.get("schema") != SCENE_SCHEMA
            or report.get("sample_count") != count
            or report.get("test_camera") != TEST_CAMERA
        ):
            raise BenchmarkReportError("suite scene report contract mismatch")
        content = dict(report)
        digest = content.pop("content_sha256", None)
        if digest != sha256(canonical_json(content)):
            raise BenchmarkReportError("suite input scene report hash mismatch")
        if strict_protocol:
            root = report.get("artifact_root")
            if not isinstance(root, str):
                raise BenchmarkReportError("suite input scene report root is missing")
            verified = verify_scene_report(Path(root))
            if verified["content_sha256"] != report["content_sha256"]:
                raise BenchmarkReportError("suite input scene artifact mismatch")
    first, second = (indexed[scene] for scene in sorted(indexed))
    if (
        set(first["scaling"]) != {str(step) for step in REPORTING_STEPS}
        or set(second["scaling"]) != set(first["scaling"])
        or set(first["systems"])
        != {*CONTINUATION_SYSTEMS, *NATIVE_SYSTEMS}
        or set(second["systems"]) != set(first["systems"])
    ):
        raise BenchmarkReportError("suite system/step set mismatch")
    for step in first["scaling"]:
        if (
            set(first["scaling"][step]) != CONTINUATION_SYSTEMS
            or set(second["scaling"][step]) != CONTINUATION_SYSTEMS
        ):
            raise BenchmarkReportError("suite continuation set mismatch")
        for system in CONTINUATION_SYSTEMS:
            if set(first["scaling"][step][system]["summary"]) != set(
                second["scaling"][step][system]["summary"]
            ):
                raise BenchmarkReportError("suite metric set mismatch")
    for system in NATIVE_SYSTEMS:
        if set(first["systems"][system]["primary_summary"]) != set(
            second["systems"][system]["primary_summary"]
        ):
            raise BenchmarkReportError("suite native metric set mismatch")
    identity = _report_identity("suite", scene_reports)
    output = Path(output_dir)
    if resume:
        try:
            load_generation(output, schema=SUITE_SCHEMA, expected_identity=identity)
        except ArtifactError as error:
            raise BenchmarkReportError(str(error)) from error
        return _load_report(output, SUITE_SCHEMA)
    if (output / "current.json").exists() and not overwrite:
        raise BenchmarkReportError("suite report exists; use resume or overwrite")
    base: dict[str, object] = {
        "schema": SUITE_SCHEMA,
        "version": 1,
        "scenes": {
            scene: {
                "content_sha256": indexed[scene]["content_sha256"],
                "sample_count": indexed[scene]["sample_count"],
                "paired": indexed[scene]["paired"],
                "report_root": indexed[scene]["artifact_root"],
            }
            for scene in sorted(indexed)
        },
        "scene_sample_counts": {
            scene: SCENE_SAMPLE_COUNTS[scene] for scene in sorted(SCENE_SAMPLE_COUNTS)
        },
        "total_sample_count": sum(SCENE_SAMPLE_COUNTS.values()),
        "primary_step": PRIMARY_STEP,
        "aggregates": _aggregate_suite(scene_reports),
        "paired_aggregates": _aggregate_suite_pairs(scene_reports),
        "aggregation_notes": {
            "macro": "unweighted mean of scene means",
            "micro": "sample-count-weighted mean of scene means",
            "native_references": "descriptive; not update-matched",
        },
    }
    report = {**base, "content_sha256": sha256(canonical_json(base))}
    markdown = (
        "# Dual-scene cam38 benchmark\n\n"
        f"Scenes: {', '.join(sorted(indexed))}; total samples: "
        f"{report['total_sample_count']}; primary step: {PRIMARY_STEP}.\n\n"
        "Macro averages weight scenes equally; micro averages weight aligned samples.\n"
    )
    publish_generation(
        output,
        schema=SUITE_SCHEMA,
        identity=identity,
        files={
            "report.json": canonical_json(report),
            "report.md": markdown.encode(),
            "report.csv": _suite_csv(report).encode(),
        },
        overwrite=overwrite,
    )
    return report


def _suite_csv(report: Mapping[str, object]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("aggregation", "step", "system", "metric", "mean"))
    for aggregation, values in report["aggregates"].items():
        for step, systems in values.items():
            for system, metrics in systems.items():
                for metric, mean in metrics.items():
                    writer.writerow((aggregation, step, system, metric, mean))
    return stream.getvalue()

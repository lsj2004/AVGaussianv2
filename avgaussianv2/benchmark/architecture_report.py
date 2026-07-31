"""Paired report for the plain AudioGS U-Net architecture ablation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from avgaussianv2.benchmark.artifacts import canonical_json, publish_generation, sha256
from avgaussianv2.benchmark.evaluation import BenchmarkEvaluationResult
from avgaussianv2.benchmark.report import BenchmarkReportError, _paired, _validate_result

SCHEMA = "avgaussianv2.audio-architecture-scene-report"


def build_architecture_scene_report(
    *,
    scene_id: str,
    evaluations: Sequence[BenchmarkEvaluationResult],
    expected_sample_count: int,
    output_dir: Path | str,
    preparation: Mapping[str, object],
) -> dict[str, object]:
    system = str(preparation.get("evaluation_system", preparation.get("strategy", "")))
    if preparation.get("scene_id") != scene_id or system != "plain_unet":
        raise BenchmarkReportError("plain-U-Net report requires its verified preparation")
    indexed = {
        (result.identity.system_name, result.identity.reporting_step): result
        for result in evaluations
    }
    steps = tuple(sorted(step for name, step in indexed if name == "audio_only" and step))
    expected = {
        *(("audio_only", step) for step in steps),
        *((system, step) for step in steps),
    }
    if not steps or set(indexed) != expected or len(indexed) != len(evaluations):
        raise BenchmarkReportError("plain-U-Net and audio-only require identical steps")
    for result in evaluations:
        _validate_result(result, scene_id, expected_sample_count)
    for step in steps:
        baseline, candidate = indexed[("audio_only", step)], indexed[(system, step)]
        if baseline.identity.expected_sample_ids != candidate.identity.expected_sample_ids:
            raise BenchmarkReportError("architecture comparison sample IDs differ")
        for field in ("seed", "index_sha256", "planned_updates", "batch_size"):
            if baseline.provenance.get(field) != candidate.provenance.get(field):
                raise BenchmarkReportError(f"architecture fairness mismatch: {field}")
        if baseline.metric_protocol != candidate.metric_protocol:
            raise BenchmarkReportError("architecture metric protocols differ")
    primary = steps[-1]
    base: dict[str, object] = {
        "schema": SCHEMA,
        "version": 1,
        "scene_id": scene_id,
        "sample_count": expected_sample_count,
        "reporting_steps": list(steps),
        "primary_step": primary,
        "systems": ["audio_only", system],
        "comparison_scope": "audio_postprocessor_main_update_matched",
        "total_optimizer_updates_matched": True,
        "total_compute_matched": False,
        "scaling": {
            str(step): {name: indexed[(name, step)].summary for name in ("audio_only", system)}
            for step in steps
        },
        "paired_by_step": {
            str(step): {
                "plain_unet_vs_native_residual": _paired(
                    indexed[(system, step)],
                    indexed[("audio_only", step)],
                    tuple(sorted(indexed[(system, step)].summary)),
                )
            }
            for step in steps
        },
        "inputs": [
            {
                "system": result.identity.system_name,
                "step": result.identity.reporting_step,
                "content_sha256": result.content_sha256,
            }
            for result in sorted(evaluations, key=lambda item: str(item.identity.to_mapping()))
        ],
    }
    base["content_sha256"] = sha256(canonical_json(base))
    report_bytes = canonical_json(base)
    publish_generation(
        Path(output_dir),
        schema=SCHEMA,
        identity={"scene_id": scene_id, "input_sha256": sha256(canonical_json(base["inputs"]))},
        files={"report.json": report_bytes},
    )
    return json.loads(report_bytes)


__all__ = ["SCHEMA", "build_architecture_scene_report"]

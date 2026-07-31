"""Same-checkpoint causal report for FiLM RGBD conditioning."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from avgaussianv2.benchmark.artifacts import canonical_json, publish_generation, sha256
from avgaussianv2.benchmark.evaluation import BenchmarkEvaluationResult
from avgaussianv2.benchmark.report import BenchmarkReportError, _paired, _validate_result

SCHEMA = "avgaussianv2.film-causal-scene-report"
SYSTEMS = (
    "joint_conditioned",
    "joint_conditioned_no_rgbd",
    "joint_conditioned_wrong_camera",
)


def build_film_causal_report(
    *, scene_id: str, evaluations: Sequence[BenchmarkEvaluationResult],
    expected_sample_count: int, output_dir: Path | str,
) -> dict[str, object]:
    indexed = {
        (value.identity.system_name, value.identity.reporting_step): value
        for value in evaluations
    }
    steps = tuple(sorted(step for system, step in indexed if system == SYSTEMS[0] and step))
    expected = {(system, step) for system in SYSTEMS for step in steps}
    if not steps or set(indexed) != expected or len(indexed) != len(evaluations):
        raise BenchmarkReportError("FiLM causal report requires all systems at identical steps")
    for value in evaluations:
        _validate_result(value, scene_id, expected_sample_count)
    paired = {}
    for step in steps:
        values = [indexed[(system, step)] for system in SYSTEMS]
        if len({value.provenance["checkpoint_sha256"] for value in values}) != 1:
            raise BenchmarkReportError("FiLM causal variants must reuse one checkpoint")
        if len({canonical_json(value.metric_protocol) for value in values}) != 1:
            raise BenchmarkReportError("FiLM causal variants require one metric protocol")
        metrics = tuple(sorted(values[0].summary))
        paired[str(step)] = {
            "rgbd_on_vs_off": _paired(values[0], values[1], metrics),
            "correct_vs_wrong_camera": _paired(values[0], values[2], metrics),
        }
    base: dict[str, object] = {
        "schema": SCHEMA, "version": 1, "scene_id": scene_id,
        "sample_count": expected_sample_count, "reporting_steps": list(steps),
        "systems": list(SYSTEMS), "same_checkpoint": True,
        "paired_by_step": paired,
        "inputs": [
            {"system": value.identity.system_name, "step": value.identity.reporting_step,
             "content_sha256": value.content_sha256}
            for value in sorted(evaluations, key=lambda item: str(item.identity.to_mapping()))
        ],
    }
    base["content_sha256"] = sha256(canonical_json(base))
    report_bytes = canonical_json(base)
    publish_generation(
        Path(output_dir), schema=SCHEMA,
        identity={"scene_id": scene_id, "input_sha256": sha256(canonical_json(base["inputs"]))},
        files={"report.json": report_bytes},
    )
    return json.loads(report_bytes)


__all__ = ["SCHEMA", "SYSTEMS", "build_film_causal_report"]

"""Paired AudioGS postprocessor report: cross-attention versus FiLM+U-Net."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from avgaussianv2.benchmark.artifacts import (
    canonical_json,
    publish_generation,
    sha256,
)
from avgaussianv2.benchmark.cross_attention_ablation import (
    CAUSAL_EVALUATION_SYSTEMS,
)
from avgaussianv2.benchmark.evaluation import (
    ALL_METRICS,
    REPORTING_STEPS,
    BenchmarkEvaluationResult,
)
from avgaussianv2.benchmark.report import (
    BenchmarkReportError,
    _paired,
    _validate_result,
)


SCHEMA = "avgaussianv2.cross-attention-scene-report"
FILM_SYSTEM = "joint_conditioned"
PRIMARY_STEP = 30_000


def _key(result: BenchmarkEvaluationResult) -> tuple[str, int | None]:
    return result.identity.system_name, result.identity.reporting_step


def _require_fair_protocol(
    indexed: Mapping[tuple[str, int | None], BenchmarkEvaluationResult],
) -> None:
    reference = indexed[(FILM_SYSTEM, REPORTING_STEPS[0])]
    for step in REPORTING_STEPS:
        film = indexed[(FILM_SYSTEM, step)]
        cross = indexed[("cross_attention", step)]
        for field in (
            "index_sha256",
            "visual_initialization_sha256",
            "seed",
            "planned_updates",
            "completed_updates",
            "checkpoint_step",
            "batch_size",
        ):
            if film.provenance.get(field) != cross.provenance.get(field):
                raise BenchmarkReportError(
                    f"FiLM/cross-attention fairness mismatch: {field}"
                )
        causal = [
            indexed[(system, step)]
            for system in CAUSAL_EVALUATION_SYSTEMS
        ]
        checkpoint_hashes = {
            result.provenance.get("checkpoint_sha256") for result in causal
        }
        model_hashes = {
            result.provenance.get("model_initialization_sha256")
            for result in causal
        }
        if len(checkpoint_hashes) != 1 or len(model_hashes) != 1:
            raise BenchmarkReportError(
                "cross-attention causal ablations must reuse one checkpoint"
            )
        if any(
            result.metric_protocol != reference.metric_protocol
            for result in (film, *causal)
        ):
            raise BenchmarkReportError("all systems must use one metric protocol")


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        f"# Cross-attention vs FiLM+U-Net: {report['scene_id']}",
        "",
        "This is an update-matched postprocessor comparison. Both systems start "
        "from the same audited AudioGS acoustic-Gaussian checkpoint and use its "
        "criterion; only the RGBD-conditioned postprocessor differs.",
        "",
        "| Step | System | Audio total | PSNR | SSIM | RGB L1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for step in REPORTING_STEPS:
        for system in (FILM_SYSTEM, *CAUSAL_EVALUATION_SYSTEMS):
            summary = report["scaling"][str(step)][system]

            def mean(name: str) -> object:
                return summary.get(name, {}).get("mean", "")

            lines.append(
                f"| {step} | {system} | {mean('audio_total')} | "
                f"{mean('rgb_psnr')} | {mean('rgb_ssim')} | {mean('rgb_l1')} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_cross_attention_scene_report(
    *,
    scene_id: str,
    evaluations: Sequence[BenchmarkEvaluationResult],
    expected_sample_count: int,
    output_dir: Path | str,
    preparation: Mapping[str, object],
) -> dict[str, object]:
    alignment = preparation.get("alignment")
    audio_contract = preparation.get("audiogs_contract")
    if (
        preparation.get("scene_id") != scene_id
        or not isinstance(alignment, Mapping)
        or alignment.get("comparison_scope")
        != "shared_audiogs_gaussians_postprocessor_ablation"
        or alignment.get("same_audiogs_checkpoint_as_a") is not True
        or alignment.get("audio_criterion")
        != "native_audiogs_checkpoint_criterion"
        or not isinstance(audio_contract, Mapping)
        or not isinstance(audio_contract.get("checkpoint_sha256"), str)
    ):
        raise BenchmarkReportError(
            "report requires verified shared-AudioGS cross-attention preparation"
        )
    indexed = {_key(result): result for result in evaluations}
    expected = {
        *((FILM_SYSTEM, step) for step in REPORTING_STEPS),
        *(
            (system, step)
            for system in CAUSAL_EVALUATION_SYSTEMS
            for step in REPORTING_STEPS
        ),
    }
    if set(indexed) != expected or len(indexed) != len(evaluations):
        raise BenchmarkReportError(
            "cross-attention report requires FiLM and all causal systems at 5k/10k/30k"
        )
    for result in indexed.values():
        _validate_result(result, scene_id, expected_sample_count)
    sample_ids = indexed[(FILM_SYSTEM, PRIMARY_STEP)].identity.expected_sample_ids
    if any(result.identity.expected_sample_ids != sample_ids for result in indexed.values()):
        raise BenchmarkReportError("all comparison systems require identical sample IDs")
    _require_fair_protocol(indexed)

    paired_by_step = {}
    for step in REPORTING_STEPS:
        cross = indexed[("cross_attention", step)]
        paired_by_step[str(step)] = {
            "cross_attention_vs_film_unet": _paired(
                cross,
                indexed[(FILM_SYSTEM, step)],
                ALL_METRICS,
            ),
            "rgbd_on_vs_off": _paired(
                cross,
                indexed[("cross_attention_no_rgbd", step)],
                ALL_METRICS,
            ),
            "rgbd_on_vs_shuffled": _paired(
                cross,
                indexed[("cross_attention_shuffled_rgbd", step)],
                ALL_METRICS,
            ),
        }
    scaling = {
        str(step): {
            system: indexed[(system, step)].summary
            for system in (FILM_SYSTEM, *CAUSAL_EVALUATION_SYSTEMS)
        }
        for step in REPORTING_STEPS
    }
    base: dict[str, object] = {
        "schema": SCHEMA,
        "version": 1,
        "scene_id": scene_id,
        "sample_count": expected_sample_count,
        "primary_step": PRIMARY_STEP,
        "comparison_scope": "shared_audiogs_gaussians_postprocessor_update_matched",
        "shared": {
            "train_cameras": list(
                indexed[(FILM_SYSTEM, PRIMARY_STEP)].provenance["train_cameras"]
            ),
            "test_camera": "cam38",
            "seed": 42,
            "batch_size": 1,
            "index_sha256": indexed[
                (FILM_SYSTEM, PRIMARY_STEP)
            ].provenance["index_sha256"],
            "visual_initialization_sha256": indexed[
                (FILM_SYSTEM, PRIMARY_STEP)
            ].provenance["visual_initialization_sha256"],
            "audiogs_checkpoint_sha256": audio_contract["checkpoint_sha256"],
            "audio_criterion": alignment["audio_criterion"],
        },
        "different_by_design": [
            "rgbd_conditioned_postprocessor",
            "postprocessor_parameterization",
        ],
        "scaling": scaling,
        "paired_by_step": paired_by_step,
        "inputs": [
            {
                "system": result.identity.system_name,
                "step": result.identity.reporting_step,
                "content_sha256": result.content_sha256,
            }
            for result in sorted(evaluations, key=lambda item: str(_key(item)))
        ],
    }
    base["content_sha256"] = sha256(canonical_json(base))
    report_bytes = canonical_json(base)
    publish_generation(
        Path(output_dir),
        schema=SCHEMA,
        identity={
            "scene_id": scene_id,
            "input_sha256": sha256(canonical_json(base["inputs"])),
        },
        files={
            "report.json": report_bytes,
            "report.md": _markdown(base).encode(),
        },
    )
    return json.loads(report_bytes)


__all__ = ["SCHEMA", "build_cross_attention_scene_report"]

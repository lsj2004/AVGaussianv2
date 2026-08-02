from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avgaussianv2.benchmark.evaluation import verify_evaluation  # noqa: E402
from avgaussianv2.benchmark.lre_orchestration import (  # noqa: E402
    load_lre_run_manifest,
)


SCHEMA = "avgaussianv2.final-fair-comparison"
SCENES = ("scene1_opera", "Scene7playing")
SEEDS = (17, 42, 73)
MODEL_METRICS = (
    "audio_total",
    "audio_mono",
    "audio_diff",
    "waveform_l1",
    "mono_lsd",
    "diff_lsd",
    "lre_error_db",
    "paper_mag",
    "paper_env",
    "paper_lre_db",
    "ild_error_db",
    "ipd_error_rad",
    "paper_dpam",
)
DISPLAY_METRICS = (
    "waveform_l1",
    "paper_mag",
    "paper_env",
    "paper_dpam",
    "paper_lre_db",
    "ild_error_db",
    "ipd_error_rad",
)
DISPLAY_LABELS = {
    "audio_total": "Audio total",
    "audio_mono": "Audio mono",
    "audio_diff": "Audio diff",
    "waveform_l1": "Waveform",
    "mono_lsd": "Mono LSD",
    "diff_lsd": "Diff LSD",
    "lre_error_db": "Native LRE",
    "paper_mag": "MAG",
    "paper_env": "ENV",
    "paper_dpam": "DPAM",
    "paper_lre_db": "LRE",
    "ild_error_db": "ILD",
    "ipd_error_rad": "IPD",
}
PRIMARY_MODEL_METRICS = ("audio_total", *DISPLAY_METRICS)
DIAGNOSTIC_MODEL_METRICS = (
    "audio_mono",
    "audio_diff",
    "mono_lsd",
    "diff_lsd",
    "lre_error_db",
)
ARCHITECTURE_SYSTEMS = (
    "audio_only",
    "query_dependent_p1",
    "joint_conditioned",
    "cross_attention_masks",
)
ARCHITECTURE_MECHANISMS = {
    "audio_only": "AudioGS native residual；无视觉条件",
    "query_dependent_p1": "query-dependent P1 视觉条件残差",
    "joint_conditioned": "联合优化 AudioGS 与视觉条件器",
    "cross_attention_masks": "音频 token 与视觉条件 cross-attention mask",
}
PARAMETER_FIELDS = (
    "checkpoint_5k_total_parameter_elements",
    "checkpoint_5k_declared_trainable_parameter_elements",
    "checkpoint_5k_optimizer_active_parameter_elements",
    "checkpoint_5k_optimizer_dormant_parameter_elements",
    "checkpoint_5k_orphan_trainable_parameter_elements",
    "checkpoint_5k_frozen_parameter_elements",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_artifact_pointer(pointer: Any, path: Path, *, label: str) -> None:
    if not isinstance(pointer, Mapping):
        raise ValueError(f"{label} artifact pointer is missing")
    if pointer.get("path") != str(path.resolve()) or pointer.get("sha256") != _sha256(
        path
    ):
        raise ValueError(f"{label} artifact pointer/hash mismatch")


def _integer_range(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    by_scene: dict[str, int] = {}
    for record in records:
        scene = record.get("scene")
        value = record.get(field)
        if scene not in SCENES or isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"invalid parameter audit field: {field}")
        if value < 0 or scene in by_scene:
            raise ValueError(f"invalid parameter audit scene/value: {field}/{scene}")
        by_scene[str(scene)] = value
    if set(by_scene) != set(SCENES):
        raise ValueError(f"parameter audit lacks the exact scene matrix: {field}")
    values = list(by_scene.values())
    return {"min": min(values), "max": max(values), "by_scene": by_scene}


def _architecture_summary(
    architecture_report_path: Path,
    parameter_audit_path: Path,
    resource_report_path: Path,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    architecture_report = _load_json(architecture_report_path)
    parameter_audit = _load_json(parameter_audit_path)
    resource_report = _load_json(resource_report_path)
    expected_identities = (
        (architecture_report, "avgaussianv2.p2-architecture-config-report"),
        (parameter_audit, "avgaussianv2.p2-cuda-parameter-audit"),
        (resource_report, "avgaussianv2.p2-stitched-resource-report"),
    )
    for document, schema in expected_identities:
        if (
            document.get("schema") != schema
            or document.get("version") != 1
            or document.get("repository") != repository
        ):
            raise ValueError(f"architecture evidence identity mismatch: {schema}")

    if tuple(architecture_report.get("systems", ())) != ARCHITECTURE_SYSTEMS:
        raise ValueError("architecture report does not contain the exact P2 systems")
    architectures = architecture_report.get("architectures")
    if not isinstance(architectures, Mapping) or set(architectures) != set(
        ARCHITECTURE_SYSTEMS
    ):
        raise ValueError("architecture report system mapping mismatch")
    _verify_artifact_pointer(
        architecture_report.get("cuda_parameter_audit"),
        parameter_audit_path,
        label="parameter audit",
    )
    _verify_artifact_pointer(
        architecture_report.get("resource_report"),
        resource_report_path,
        label="resource report",
    )
    _verify_artifact_pointer(
        parameter_audit.get("resource_report"),
        resource_report_path,
        label="parameter-audit resource report",
    )

    raw_parameter_records = parameter_audit.get("records")
    if not isinstance(raw_parameter_records, list) or len(raw_parameter_records) != 8:
        raise ValueError("parameter audit must contain four systems x two scenes")
    parameter_records: dict[str, list[Mapping[str, Any]]] = {
        system: [] for system in ARCHITECTURE_SYSTEMS
    }
    for record in raw_parameter_records:
        if (
            not isinstance(record, Mapping)
            or record.get("system") not in parameter_records
        ):
            raise ValueError("invalid parameter audit record")
        parameter_records[str(record["system"])].append(record)

    raw_resource_records = resource_report.get("records")
    raw_resource_aggregates = resource_report.get("aggregates")
    if not isinstance(raw_resource_records, list) or len(raw_resource_records) != 32:
        raise ValueError("resource report must contain the exact 32-run P2 matrix")
    if (
        not isinstance(raw_resource_aggregates, list)
        or len(raw_resource_aggregates) != 4
    ):
        raise ValueError("resource report must contain four system aggregates")
    resource_records: dict[str, list[Mapping[str, Any]]] = {
        system: [] for system in ARCHITECTURE_SYSTEMS
    }
    for record in raw_resource_records:
        if (
            not isinstance(record, Mapping)
            or record.get("system") not in resource_records
        ):
            raise ValueError("invalid P2 resource record")
        resource_records[str(record["system"])].append(record)
    resource_aggregates: dict[str, Mapping[str, Any]] = {}
    for aggregate in raw_resource_aggregates:
        if (
            not isinstance(aggregate, Mapping)
            or aggregate.get("system") not in resource_records
            or aggregate["system"] in resource_aggregates
        ):
            raise ValueError("invalid P2 resource aggregate")
        resource_aggregates[str(aggregate["system"])] = aggregate
    if set(resource_aggregates) != set(ARCHITECTURE_SYSTEMS):
        raise ValueError("P2 resource aggregate system mismatch")

    systems: dict[str, Any] = {}
    for system in ARCHITECTURE_SYSTEMS:
        architecture_entry = architectures[system]
        if not isinstance(architecture_entry, Mapping):
            raise ValueError(f"invalid architecture entry: {system}")
        architecture = architecture_entry.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError(f"architecture mapping missing: {system}")
        signature = architecture_entry.get("architecture_signature_sha256")
        if signature != _canonical_sha256(architecture):
            raise ValueError(f"architecture signature mismatch: {system}")
        active_hyperparameters = architecture.get("active_hyperparameters")
        expected_worker = (
            "audio_only" if system == "audio_only" else "joint_conditioned"
        )
        if (
            not isinstance(active_hyperparameters, Mapping)
            or architecture.get("worker_mode") != expected_worker
            or architecture.get("evaluation_system") != system
            or active_hyperparameters.get("worker_mode") != expected_worker
        ):
            raise ValueError(f"architecture runtime identity mismatch: {system}")

        records = parameter_records[system]
        if len(records) != 2 or {record.get("scene") for record in records} != set(
            SCENES
        ):
            raise ValueError(f"parameter audit scene matrix mismatch: {system}")
        embedded_records = architecture_entry.get("scene_records")
        if not isinstance(embedded_records, list) or len(embedded_records) != 2:
            raise ValueError(f"architecture scene records mismatch: {system}")
        embedded_by_scene = {
            record.get("scene"): record
            for record in embedded_records
            if isinstance(record, Mapping)
        }
        if set(embedded_by_scene) != set(SCENES):
            raise ValueError(f"architecture embedded scene matrix mismatch: {system}")
        for record in records:
            embedded = embedded_by_scene[record["scene"]]
            if (
                embedded.get("architecture_signature_sha256") != signature
                or embedded.get("architecture") != architecture
                or any(
                    embedded.get("cuda_parameter_audit", {}).get(field)
                    != record.get(field)
                    for field in PARAMETER_FIELDS
                )
            ):
                raise ValueError(f"architecture/parameter audit mismatch: {system}")

        p2_records = resource_records[system]
        expected_cells = {
            (scene, weight) for scene in SCENES for weight in (0.0, 0.01, 0.02, 0.05)
        }
        observed_cells = {
            (record.get("scene"), float(record.get("lambda_lre", -1.0)))
            for record in p2_records
        }
        if len(p2_records) != 8 or observed_cells != expected_cells:
            raise ValueError(f"P2 resource run matrix mismatch: {system}")
        aggregate = resource_aggregates[system]
        elapsed = [float(record["pipeline_elapsed_seconds"]) for record in p2_records]
        gpu_hours = [float(record["pipeline_gpu_hours"]) for record in p2_records]
        expected_resource_values = {
            "run_count": 8,
            "maximum_peak_memory_used_mib": max(
                int(record["peak_memory_used_mib"]) for record in p2_records
            ),
            "maximum_gpu_utilization_percent": max(
                int(record["maximum_gpu_utilization_percent"]) for record in p2_records
            ),
        }
        if any(
            aggregate.get(key) != value
            for key, value in expected_resource_values.items()
        ):
            raise ValueError(f"P2 resource aggregate mismatch: {system}")
        if not math.isclose(
            float(aggregate.get("mean_pipeline_elapsed_seconds", math.nan)),
            statistics.fmean(elapsed),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(aggregate.get("total_pipeline_gpu_hours", math.nan)),
            sum(gpu_hours),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"P2 resource aggregate arithmetic mismatch: {system}")

        systems[system] = {
            "mechanism": ARCHITECTURE_MECHANISMS[system],
            "architecture_signature_sha256": signature,
            "worker_mode": architecture["worker_mode"],
            "evaluation_system": architecture["evaluation_system"],
            "active_hyperparameters": dict(active_hyperparameters),
            "parameter_elements": {
                field.removeprefix("checkpoint_5k_").removesuffix(
                    "_parameter_elements"
                ): _integer_range(records, field)
                for field in PARAMETER_FIELDS
            },
            "p2_5k_resource": {
                key: aggregate[key]
                for key in (
                    "run_count",
                    "maximum_peak_memory_used_mib",
                    "maximum_gpu_utilization_percent",
                    "mean_pipeline_elapsed_seconds",
                    "total_pipeline_gpu_hours",
                )
            },
        }

    return {
        "scope": {
            "included_systems": list(ARCHITECTURE_SYSTEMS),
            "excluded_before_p2": {
                "plain_unet": "eliminated before the formal P2 32-run matrix"
            },
            "resource_boundary": "P2 5k screening only; not final 30k runtime",
        },
        "systems": systems,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile requires values and probability in [0,1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def hierarchical_bootstrap(
    cells: Mapping[tuple[int, str], Sequence[float]],
    *,
    resamples: int,
    rng_seed: int,
) -> dict[str, float | int]:
    expected = {(seed, scene) for seed in SEEDS for scene in SCENES}
    if set(cells) != expected or any(not cells[key] for key in expected):
        raise ValueError(
            "hierarchical bootstrap requires the exact 3-seed/2-scene matrix"
        )
    if resamples < 1_000:
        raise ValueError("hierarchical bootstrap requires at least 1000 resamples")
    rng = random.Random(rng_seed)
    draws = []
    for _ in range(resamples):
        seed_means = []
        for seed in rng.choices(SEEDS, k=len(SEEDS)):
            scene_means = []
            for scene in SCENES:
                values = list(cells[(seed, scene)])
                scene_means.append(statistics.fmean(rng.choices(values, k=len(values))))
            seed_means.append(statistics.fmean(scene_means))
        draws.append(statistics.fmean(seed_means))
    return {
        "lower_95": _percentile(draws, 0.025),
        "median": _percentile(draws, 0.5),
        "upper_95": _percentile(draws, 0.975),
        "resamples": resamples,
        "rng_seed": rng_seed,
    }


def _manifest(path: Path) -> dict[str, Any]:
    return load_lre_run_manifest(path)


def select_exact_matrix(
    *,
    gate: Mapping[str, Any],
    candidate_main: Mapping[str, Any],
    candidate_seeds: Mapping[str, Any],
    audio_main: Mapping[str, Any],
    audio_seeds: Mapping[str, Any],
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], str, float]:
    finalist = gate.get("selected_finalist")
    if gate.get("schema") != "avgaussianv2.p3-30k-gate" or not isinstance(
        finalist, dict
    ):
        raise ValueError("30k gate has no selected finalist")
    system = finalist.get("system")
    treatment = finalist.get("lambda_lre")
    if (
        not isinstance(system, str)
        or isinstance(treatment, bool)
        or not isinstance(treatment, (int, float))
    ):
        raise ValueError("invalid selected finalist identity")
    treatment = float(treatment)

    repositories = {
        json.dumps(document.get("repository"), sort_keys=True)
        for document in (candidate_main, candidate_seeds, audio_main, audio_seeds)
    }
    repositories.add(json.dumps(gate.get("repository"), sort_keys=True))
    if len(repositories) != 1:
        raise ValueError("repository mismatch across final comparison inputs")

    runs: dict[tuple[str, int, str], dict[str, Any]] = {}
    for document in (candidate_main, candidate_seeds):
        for raw in document["runs"]:
            if not isinstance(raw, dict) or raw.get("system") != system:
                continue
            seed = raw.get("seed")
            scene = raw.get("scene")
            weight = raw.get("lambda_lre")
            if (
                seed not in SEEDS
                or scene not in SCENES
                or float(weight)
                not in {
                    0.0,
                    treatment,
                }
            ):
                continue
            role = "candidate" if float(weight) == treatment else "architecture_control"
            key = (role, int(seed), str(scene))
            if key in runs:
                raise ValueError(f"duplicate final candidate run: {key}")
            runs[key] = raw
    for document in (audio_main, audio_seeds):
        for raw in document["runs"]:
            if (
                not isinstance(raw, dict)
                or raw.get("system") != "audio_only"
                or raw.get("seed") not in SEEDS
                or raw.get("scene") not in SCENES
                or float(raw.get("lambda_lre", -1.0)) != 0.0
            ):
                continue
            key = ("audio_only", int(raw["seed"]), str(raw["scene"]))
            if key in runs:
                raise ValueError(f"duplicate final Audio-only run: {key}")
            runs[key] = raw
    expected = {
        (role, seed, scene)
        for role in ("candidate", "architecture_control", "audio_only")
        for seed in SEEDS
        for scene in SCENES
    }
    if set(runs) != expected:
        missing = sorted(expected - set(runs))
        extra = sorted(set(runs) - expected)
        raise ValueError(
            f"final fair matrix mismatch; missing={missing}, extra={extra}"
        )
    return runs, system, treatment


def _complete_run(run_dir: Path, stage: str) -> None:
    result = _load_json(run_dir / f"run_result.{stage}.json")
    if result.get("status") != "succeeded" or result.get(
        "completed_stages"
    ) != result.get("planned_stages"):
        raise ValueError(f"incomplete run result: {run_dir.name}/{stage}")


def _load_model_run(
    run_root: Path,
    run: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = run_root / str(run["continuation_id"])
    identity = _load_json(run_dir / "continuation_identity.json")
    for field in ("continuation_id", "scene", "system", "seed", "lambda_lre"):
        if identity.get(field) != run.get(field):
            raise ValueError(f"continuation identity mismatch: {run_dir.name}/{field}")
    if identity.get("repository") != repository:
        raise ValueError(f"continuation repository mismatch: {run_dir.name}")
    _complete_run(run_dir, str(run["stage"]))
    # A final report must re-audit the immutable evaluation *and* the bound
    # training output.  Merely loading the evaluation generation would allow a
    # checkpoint/runtime-contract mutation after evaluation to go unnoticed.
    evaluation = verify_evaluation(run_dir / "evaluations/step_030000")
    if (
        evaluation.identity.scene_id != run["scene"]
        or evaluation.identity.system_name != run["system"]
        or evaluation.identity.reporting_step != 30_000
        or evaluation.provenance.get("seed") != run["seed"]
        or evaluation.provenance.get("checkpoint_step") != 30_000
        or evaluation.provenance.get("main_update_matched") is not True
        or any(metric not in evaluation.summary for metric in MODEL_METRICS)
    ):
        raise ValueError(f"30k evaluation identity mismatch: {run_dir.name}")
    rows = [dict(row) for row in evaluation.rows]
    return {
        "content_sha256": evaluation.content_sha256,
        "sample_ids": list(evaluation.identity.expected_sample_ids),
        "metric_protocol": evaluation.metric_protocol,
        "metrics": {
            metric: float(evaluation.summary[metric]["mean"])
            for metric in MODEL_METRICS
        },
        "rows": rows,
    }


def aggregate_final_models(
    evaluations: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    resamples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    roles = ("candidate", "architecture_control", "audio_only")
    expected = {
        (role, seed, scene) for role in roles for seed in SEEDS for scene in SCENES
    }
    if set(evaluations) != expected:
        raise ValueError("evaluation matrix is incomplete")

    per_seed: dict[str, Any] = {}
    for role in roles:
        seed_records = {}
        for seed in SEEDS:
            scenes = {
                scene: dict(evaluations[(role, seed, scene)]["metrics"])
                for scene in SCENES
            }
            seed_records[str(seed)] = {
                "scenes": scenes,
                "scene_equal_macro": {
                    metric: statistics.fmean(scenes[scene][metric] for scene in SCENES)
                    for metric in MODEL_METRICS
                },
            }
        per_seed[role] = seed_records

    absolute = {}
    for role in roles:
        absolute[role] = {}
        for metric in MODEL_METRICS:
            values = [
                per_seed[role][str(seed)]["scene_equal_macro"][metric] for seed in SEEDS
            ]
            absolute[role][metric] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "per_seed": dict(
                    zip((str(seed) for seed in SEEDS), values, strict=True)
                ),
            }

    comparisons = {}
    for comparison, left_role in (
        ("candidate_vs_architecture_control", "architecture_control"),
        ("candidate_vs_audio_only", "audio_only"),
    ):
        metrics = {}
        for metric_index, metric in enumerate(MODEL_METRICS):
            cells: dict[tuple[int, str], list[float]] = {}
            win_rates = []
            seed_macro_deltas = []
            for seed in SEEDS:
                scene_deltas = []
                for scene in SCENES:
                    left = evaluations[(left_role, seed, scene)]
                    right = evaluations[("candidate", seed, scene)]
                    if left["sample_ids"] != right["sample_ids"]:
                        raise ValueError(
                            f"paired sample mismatch: {comparison}/{seed}/{scene}"
                        )
                    deltas = [
                        float(candidate_row[metric]) - float(control_row[metric])
                        for control_row, candidate_row in zip(
                            left["rows"], right["rows"], strict=True
                        )
                    ]
                    cells[(seed, scene)] = deltas
                    wins = sum(delta < 0.0 for delta in deltas)
                    ties = sum(delta == 0.0 for delta in deltas)
                    win_rates.append((wins + 0.5 * ties) / len(deltas))
                    scene_deltas.append(statistics.fmean(deltas))
                seed_macro_deltas.append(statistics.fmean(scene_deltas))
            ci = hierarchical_bootstrap(
                cells,
                resamples=resamples,
                rng_seed=20260731 + metric_index,
            )
            conclusion = "uncertain"
            if ci["upper_95"] < 0:
                conclusion = "candidate_improves"
            elif ci["lower_95"] > 0:
                conclusion = "candidate_degrades"
            metrics[metric] = {
                "mean_delta": statistics.fmean(seed_macro_deltas),
                "seed_sample_std": statistics.stdev(seed_macro_deltas),
                "scene_seed_equal_paired_win_rate": statistics.fmean(win_rates),
                "hierarchical_paired_bootstrap_95_ci": ci,
                "conclusion": conclusion,
            }
        comparisons[comparison] = metrics
    return {"per_seed": per_seed, "across_seed": absolute}, comparisons


def _format(value: float) -> str:
    return f"{value:.6f}"


def _format_integer_range(value: Mapping[str, Any]) -> str:
    lower = int(value["min"])
    upper = int(value["max"])
    return f"{lower:,}" if lower == upper else f"{lower:,}–{upper:,}"


def _render_architecture_section(report: Mapping[str, Any], number: int) -> list[str]:
    summary = report["architecture_summary"]
    systems = summary["systems"]
    lines = [
        f"## {number}. 架构、参数量与 P2 资源成本",
        "",
        "参数来自两个正式场景的 5k CUDA checkpoint 审计；资源来自 P2 的 32-run "
        "screening。资源数字仅描述 5k 筛选成本，不代表最终 30k 运行成本。",
        "",
        "| 架构 | 核心机制 | 总参数量 | optimizer-active | orphan-trainable | "
        "峰值显存 MiB | 平均耗时 s | P2 GPU·h |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in ARCHITECTURE_SYSTEMS:
        record = systems[system]
        parameters = record["parameter_elements"]
        resource = record["p2_5k_resource"]
        lines.append(
            f"| {system} | {record['mechanism']} | "
            f"{_format_integer_range(parameters['total'])} | "
            f"{_format_integer_range(parameters['optimizer_active'])} | "
            f"{_format_integer_range(parameters['orphan_trainable'])} | "
            f"{int(resource['maximum_peak_memory_used_mib']):,} | "
            f"{float(resource['mean_pipeline_elapsed_seconds']):.2f} | "
            f"{float(resource['total_pipeline_gpu_hours']):.3f} |"
        )
    lines.extend(
        [
            "",
            "### 完整 active hyperparameters",
            "",
        ]
    )
    for system in ARCHITECTURE_SYSTEMS:
        config = json.dumps(
            systems[system]["active_hyperparameters"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(", ", ": "),
        )
        lines.append(f"- `{system}`：`{config}`")
        lines.append("")
    lines.extend(
        [
            "`plain_unet` 已在正式 P2 32-run 矩阵前淘汰，不得伪装成 P2 完成架构。",
            "",
        ]
    )
    return lines


def _render_no_finalist_markdown(report: Mapping[str, Any]) -> str:
    audio = report["audio_only_seed42_30k"]
    candidates = report["rejected_candidates"]
    references = report["absolute_references"]
    lines = [
        "# 最终公平比较：无候选通过 30k 门禁",
        "",
        "30k、逐场景、causal 与 guardrail 门禁没有选出 finalist；因此未启动候选多 seed，",
        "也不能把任一候选报告为优于 Audio-only。下表为 seed42、30k 的门禁证据。",
        "",
        "## 1. 30k 候选与 Audio-only",
        "",
        "| 模型 | "
        + " | ".join(DISPLAY_LABELS[m] for m in PRIMARY_MODEL_METRICS)
        + " | 门禁 |",
        "|---|" + "---:|" * len(PRIMARY_MODEL_METRICS) + "---|",
        "| audio_only/lambda=0 | "
        + " | ".join(_format(audio[metric]) for metric in PRIMARY_MODEL_METRICS)
        + " | 公平主基线 |",
    ]
    for candidate in candidates:
        label = f"{candidate['system']}/lambda={candidate['lambda_lre']}"
        reasons = ", ".join(candidate["reasons"])
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                _format(candidate["treatment_macro"][metric])
                for metric in PRIMARY_MODEL_METRICS
            )
            + f" | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## 2. Source/Mono/native 绝对参照",
            "",
            "该表只做 metric-matched 描述，不是 update-/seed-matched 排名。",
            "",
            "| 方法 | " + " | ".join(DISPLAY_LABELS[m] for m in DISPLAY_METRICS) + " |",
            "|---|" + "---:|" * len(DISPLAY_METRICS),
            "| audio_only/lambda=0，seed42/30k | "
            + " | ".join(_format(audio[metric]) for metric in DISPLAY_METRICS)
            + " |",
        ]
    )
    for name in ("source_binaural", "mono", "native_audiogs"):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                _format(references[name][metric]) for metric in DISPLAY_METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Mono 的低 LRE/ILD/IPD 受通道对称性影响，不能单独证明空间定位正确。",
            "",
        ]
    )
    lines.extend(_render_architecture_section(report, 3))
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    if report.get("status") == "no_finalist":
        return _render_no_finalist_markdown(report)
    finalist = report["selected_finalist"]
    models = report["models"]["across_seed"]
    comparison = report["paired_comparisons"]["candidate_vs_audio_only"]
    references = report["absolute_references"]
    labels = {
        "candidate": f"{finalist['system']}/lambda={finalist['lambda_lre']}",
        "architecture_control": f"{finalist['system']}/lambda=0",
        "audio_only": "audio_only/lambda=0",
    }
    lines = [
        "# 最终公平模型与绝对参考对比",
        "",
        "所有误差指标均为越低越好。严格主榜为 30k、3 seeds、2 scenes、相同样本与 evaluator；",
        "Source/Mono/native AudioGS 只进入绝对参照榜。",
        "",
        "## 1. 严格公平主榜",
        "",
        "| 模型 | "
        + " | ".join(DISPLAY_LABELS[m] for m in PRIMARY_MODEL_METRICS)
        + " |",
        "|---|" + "---:|" * len(PRIMARY_MODEL_METRICS),
    ]
    for role in ("candidate", "architecture_control", "audio_only"):
        lines.append(
            "| "
            + labels[role]
            + " | "
            + " | ".join(
                _format(models[role][metric]["mean"])
                for metric in PRIMARY_MODEL_METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 严格主榜诊断指标",
            "",
            "| 模型 | "
            + " | ".join(DISPLAY_LABELS[m] for m in DIAGNOSTIC_MODEL_METRICS)
            + " |",
            "|---|" + "---:|" * len(DIAGNOSTIC_MODEL_METRICS),
        ]
    )
    for role in ("candidate", "architecture_control", "audio_only"):
        lines.append(
            "| "
            + labels[role]
            + " | "
            + " | ".join(
                _format(models[role][metric]["mean"])
                for metric in DIAGNOSTIC_MODEL_METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 候选相对 Audio-only 的配对差值与 95% CI",
            "",
            "| 指标 | mean delta | 95% CI | 配对胜率 | 判断 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric in MODEL_METRICS:
        item = comparison[metric]
        ci = item["hierarchical_paired_bootstrap_95_ci"]
        lines.append(
            f"| {DISPLAY_LABELS[metric]} | {_format(item['mean_delta'])} | "
            f"[{_format(ci['lower_95'])}, {_format(ci['upper_95'])}] | "
            f"{item['scene_seed_equal_paired_win_rate']:.1%} | {item['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "## 2. Source/Mono/native 绝对参照榜",
            "",
            "该表只做 metric-matched 描述，不是 update-/seed-matched 排名。",
            "",
            "| 方法 | " + " | ".join(DISPLAY_LABELS[m] for m in DISPLAY_METRICS) + " |",
            "|---|" + "---:|" * len(DISPLAY_METRICS),
        ]
    )
    for role in ("candidate", "audio_only"):
        lines.append(
            "| "
            + labels[role]
            + " | "
            + " | ".join(
                _format(models[role][metric]["mean"]) for metric in DISPLAY_METRICS
            )
            + " |"
        )
    for name in ("source_binaural", "mono", "native_audiogs"):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                _format(references[name][metric]) for metric in DISPLAY_METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 最终候选相对绝对参照的差值",
            "",
            "差值为候选减参照；负值表示候选误差更低。",
            "",
            "| 参照 | "
            + " | ".join(f"Δ{DISPLAY_LABELS[m]}" for m in DISPLAY_METRICS)
            + " |",
            "|---|" + "---:|" * len(DISPLAY_METRICS),
        ]
    )
    for name in ("source_binaural", "mono", "native_audiogs", "audio_only"):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                _format(report["candidate_reference_deltas"][name][metric])
                for metric in DISPLAY_METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Mono 的通道对称性会天然压低部分空间误差；低 LRE/ILD/IPD 不能单独证明空间定位正确。",
            "",
        ]
    )
    lines.extend(_render_architecture_section(report, 3))
    return "\n".join(lines)


def _reference_macro(
    p2_fair_report_path: Path, repository: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    p2 = _load_json(p2_fair_report_path)
    if (
        p2.get("schema") != "avgaussianv2.p2-fair-baseline-report"
        or p2.get("repository") != repository
    ):
        raise ValueError("P2 reference report is not bound to the final repository")
    evidence = p2.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("P2 reference report lacks reference artifact evidence")
    reference_dir = p2_fair_report_path.parent / "audio_references"
    aggregate_path = reference_dir / "aggregate.json"
    verification_path = reference_dir / "verification.json"
    if evidence.get("reference_aggregate_sha256") != _sha256(
        aggregate_path
    ) or evidence.get("reference_verification_sha256") != _sha256(verification_path):
        raise ValueError("P2 reference artifact hash mismatch")
    verification = _load_json(verification_path)
    files = verification.get("files")
    if (
        verification.get("schema")
        != "avgaussianv2.audiogs-paper-reference-baselines.verification"
        or verification.get("version") != 1
        or not isinstance(files, Mapping)
        or files.get("aggregate.json") != evidence["reference_aggregate_sha256"]
    ):
        raise ValueError("P2 reference verification manifest mismatch")
    for name, expected_sha256 in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected_sha256, str)
            or _sha256(reference_dir / name) != expected_sha256
        ):
            raise ValueError(f"P2 reference verification failed: {name}")
    aggregate = _load_json(aggregate_path)
    if (
        aggregate.get("schema") != "avgaussianv2.audiogs-paper-reference-baselines"
        or aggregate.get("version") != 1
        or aggregate.get("repository") != repository
    ):
        raise ValueError("P2 reference aggregate identity mismatch")
    reference_macro = p2.get("scene_macro")
    if not isinstance(reference_macro, dict):
        raise ValueError("P2 reference report lacks scene macro values")
    references = {}
    for name in ("source_binaural", "mono", "native_audiogs"):
        values = reference_macro.get(name)
        if not isinstance(values, dict) or any(
            metric not in values for metric in DISPLAY_METRICS
        ):
            raise ValueError(f"P2 reference metrics missing: {name}")
        references[name] = {metric: float(values[metric]) for metric in DISPLAY_METRICS}
    aggregate_combined = aggregate.get("combined")
    if not isinstance(aggregate_combined, Mapping):
        raise ValueError("P2 reference aggregate lacks combined values")
    for name in ("source_binaural", "mono"):
        try:
            aggregate_values = {
                metric: float(aggregate_combined[name]["scene_macro"][metric]["mean"])
                for metric in DISPLAY_METRICS
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"P2 reference aggregate metrics missing: {name}"
            ) from error
        if aggregate_values != references[name]:
            raise ValueError(f"P2 reference report/aggregate mismatch: {name}")
    return references


def _build_no_finalist(
    *,
    gate: Mapping[str, Any],
    gate_path: Path,
    candidate_main_path: Path,
    audio_main_path: Path,
    run_root: Path,
    p2_fair_report_path: Path,
    architecture_summary: Mapping[str, Any],
    architecture_report_path: Path,
    parameter_audit_path: Path,
    resource_report_path: Path,
) -> dict[str, Any]:
    if (
        gate.get("schema") != "avgaussianv2.p3-30k-gate"
        or gate.get("selected_finalist") is not None
    ):
        raise ValueError(
            "no-finalist report requires a completed gate without finalist"
        )
    candidate_main = _manifest(candidate_main_path)
    audio_main = _manifest(audio_main_path)
    repository = gate.get("repository")
    if (
        not isinstance(repository, dict)
        or candidate_main.get("repository") != repository
        or audio_main.get("repository") != repository
        or gate.get("manifest_30k")
        != {
            "path": str(candidate_main_path.resolve()),
            "sha256": _sha256(candidate_main_path),
        }
    ):
        raise ValueError("no-finalist inputs are not bound to one formal repository")
    audio_runs = {
        str(run["scene"]): run
        for run in audio_main["runs"]
        if run.get("system") == "audio_only"
        and run.get("seed") == 42
        and float(run.get("lambda_lre", -1.0)) == 0.0
    }
    if set(audio_runs) != set(SCENES) or len(audio_main["runs"]) != 2:
        raise ValueError(
            "no-finalist report requires the exact two-scene Audio-only seed42 matrix"
        )
    audio_evaluations = {
        scene: _load_model_run(run_root, audio_runs[scene], repository)
        for scene in SCENES
    }
    protocols = {
        json.dumps(value["metric_protocol"], sort_keys=True)
        for value in audio_evaluations.values()
    }
    if len(protocols) != 1:
        raise ValueError("Audio-only metric protocol differs across no-finalist scenes")
    audio_macro = {
        metric: statistics.fmean(
            audio_evaluations[scene]["metrics"][metric] for scene in SCENES
        )
        for metric in MODEL_METRICS
    }
    raw_candidates = gate.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("no-finalist gate lacks rejected candidate evidence")
    candidates = []
    for raw in raw_candidates:
        if (
            not isinstance(raw, dict)
            or raw.get("passed") is not False
            or not isinstance(raw.get("reasons"), list)
            or not raw["reasons"]
            or not isinstance(raw.get("treatment_macro"), dict)
            or any(
                metric not in raw["treatment_macro"] for metric in PRIMARY_MODEL_METRICS
            )
        ):
            raise ValueError("no-finalist gate contains an invalid rejected candidate")
        candidates.append(raw)
    references = _reference_macro(p2_fair_report_path, repository)
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "no_finalist",
        "repository": repository,
        "selected_finalist": None,
        "protocol": {
            "candidate_boundary": "30k seed42 gate; no candidate passed causal and guardrails",
            "audio_only_boundary": "30k seed42 x two scenes; scene-equal macro",
            "multi_seed": "not run because the preregistered 30k gate selected no finalist",
            "reference_boundary": "Source/Mono/native are metric-matched descriptive references only",
        },
        "audio_only_seed42_30k": audio_macro,
        "rejected_candidates": candidates,
        "absolute_references": references,
        "architecture_summary": dict(architecture_summary),
        "evidence": {
            "gate_30k": {
                "path": str(gate_path.resolve()),
                "sha256": _sha256(gate_path),
            },
            "candidate_main_manifest": {
                "path": str(candidate_main_path.resolve()),
                "sha256": _sha256(candidate_main_path),
            },
            "audio_main_manifest": {
                "path": str(audio_main_path.resolve()),
                "sha256": _sha256(audio_main_path),
            },
            "p2_fair_report": {
                "path": str(p2_fair_report_path.resolve()),
                "sha256": _sha256(p2_fair_report_path),
            },
            "architecture_report": {
                "path": str(architecture_report_path.resolve()),
                "sha256": _sha256(architecture_report_path),
            },
            "parameter_audit": {
                "path": str(parameter_audit_path.resolve()),
                "sha256": _sha256(parameter_audit_path),
            },
            "resource_report": {
                "path": str(resource_report_path.resolve()),
                "sha256": _sha256(resource_report_path),
            },
            "audio_evaluation_content_sha256": {
                scene: audio_evaluations[scene]["content_sha256"] for scene in SCENES
            },
        },
    }


def build(
    *,
    gate_path: Path,
    candidate_main_path: Path,
    candidate_seed_path: Path | None,
    audio_main_path: Path,
    audio_seed_path: Path | None,
    run_root: Path,
    p2_fair_report_path: Path,
    architecture_report_path: Path,
    parameter_audit_path: Path,
    resource_report_path: Path,
    resamples: int,
) -> dict[str, Any]:
    gate = _load_json(gate_path)
    repository = gate.get("repository")
    if not isinstance(repository, Mapping):
        raise ValueError("30k gate lacks a formal repository identity")
    architecture_summary = _architecture_summary(
        architecture_report_path,
        parameter_audit_path,
        resource_report_path,
        repository,
    )
    if gate.get("selected_finalist") is None:
        return _build_no_finalist(
            gate=gate,
            gate_path=gate_path,
            candidate_main_path=candidate_main_path,
            audio_main_path=audio_main_path,
            run_root=run_root,
            p2_fair_report_path=p2_fair_report_path,
            architecture_summary=architecture_summary,
            architecture_report_path=architecture_report_path,
            parameter_audit_path=parameter_audit_path,
            resource_report_path=resource_report_path,
        )
    if candidate_seed_path is None or audio_seed_path is None:
        raise ValueError(
            "finalist report requires candidate and Audio-only seed manifests"
        )
    candidate_main = _manifest(candidate_main_path)
    candidate_seeds = _manifest(candidate_seed_path)
    audio_main = _manifest(audio_main_path)
    audio_seeds = _manifest(audio_seed_path)
    if gate.get("manifest_30k") != {
        "path": str(candidate_main_path.resolve()),
        "sha256": _sha256(candidate_main_path),
    }:
        raise ValueError("30k gate does not bind the candidate main manifest")
    if candidate_seeds.get("p3_30k_gate_sha256") != _sha256(
        gate_path
    ) or candidate_seeds.get("source_manifest_sha256") != _sha256(candidate_main_path):
        raise ValueError(
            "candidate seed manifest is not transitively bound to the gate"
        )
    runs, system, treatment = select_exact_matrix(
        gate=gate,
        candidate_main=candidate_main,
        candidate_seeds=candidate_seeds,
        audio_main=audio_main,
        audio_seeds=audio_seeds,
    )
    evaluations = {
        key: _load_model_run(run_root, run, repository) for key, run in runs.items()
    }
    protocols = {
        json.dumps(value["metric_protocol"], sort_keys=True)
        for value in evaluations.values()
    }
    if len(protocols) != 1:
        raise ValueError("metric protocol mismatch across strict final models")
    models, comparisons = aggregate_final_models(evaluations, resamples=resamples)

    references = _reference_macro(p2_fair_report_path, repository)

    candidate_means = {
        metric: models["across_seed"]["candidate"][metric]["mean"]
        for metric in DISPLAY_METRICS
    }
    delta_references = {
        name: {
            metric: candidate_means[metric] - values[metric]
            for metric in DISPLAY_METRICS
        }
        for name, values in {
            **references,
            "audio_only": {
                metric: models["across_seed"]["audio_only"][metric]["mean"]
                for metric in DISPLAY_METRICS
            },
        }.items()
    }

    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "finalist",
        "repository": repository,
        "selected_finalist": {"system": system, "lambda_lre": treatment},
        "protocol": {
            "strict_model_matrix": "30k x 3 seeds x 2 scenes; scene and seed equal weighting",
            "pairing": "within (seed, scene, sample_id), candidate minus control",
            "bootstrap": "seeds and paired samples resampled; both scene strata retained equally",
            "reference_boundary": "Source/Mono/native are metric-matched descriptive references only",
        },
        "models": models,
        "paired_comparisons": comparisons,
        "absolute_references": references,
        "candidate_reference_deltas": delta_references,
        "architecture_summary": architecture_summary,
        "evidence": {
            "gate_30k": {
                "path": str(gate_path.resolve()),
                "sha256": _sha256(gate_path),
            },
            "candidate_main_manifest": {
                "path": str(candidate_main_path.resolve()),
                "sha256": _sha256(candidate_main_path),
            },
            "candidate_seed_manifest": {
                "path": str(candidate_seed_path.resolve()),
                "sha256": _sha256(candidate_seed_path),
            },
            "audio_main_manifest": {
                "path": str(audio_main_path.resolve()),
                "sha256": _sha256(audio_main_path),
            },
            "audio_seed_manifest": {
                "path": str(audio_seed_path.resolve()),
                "sha256": _sha256(audio_seed_path),
            },
            "p2_fair_report": {
                "path": str(p2_fair_report_path.resolve()),
                "sha256": _sha256(p2_fair_report_path),
            },
            "architecture_report": {
                "path": str(architecture_report_path.resolve()),
                "sha256": _sha256(architecture_report_path),
            },
            "parameter_audit": {
                "path": str(parameter_audit_path.resolve()),
                "sha256": _sha256(parameter_audit_path),
            },
            "resource_report": {
                "path": str(resource_report_path.resolve()),
                "sha256": _sha256(resource_report_path),
            },
            "evaluation_content_sha256": {
                "/".join((role, str(seed), scene)): value["content_sha256"]
                for (role, seed, scene), value in evaluations.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-30k", type=Path, required=True)
    parser.add_argument("--candidate-main-manifest", type=Path, required=True)
    parser.add_argument("--candidate-seed-manifest", type=Path)
    parser.add_argument("--audio-main-manifest", type=Path, required=True)
    parser.add_argument("--audio-seed-manifest", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--p2-fair-report", type=Path, required=True)
    parser.add_argument("--architecture-report", type=Path, required=True)
    parser.add_argument("--parameter-audit", type=Path, required=True)
    parser.add_argument("--resource-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    report = build(
        gate_path=args.gate_30k,
        candidate_main_path=args.candidate_main_manifest,
        candidate_seed_path=args.candidate_seed_manifest,
        audio_main_path=args.audio_main_manifest,
        audio_seed_path=args.audio_seed_manifest,
        run_root=args.run_root,
        p2_fair_report_path=args.p2_fair_report,
        architecture_report_path=args.architecture_report,
        parameter_audit_path=args.parameter_audit,
        resource_report_path=args.resource_report,
        resamples=args.bootstrap_resamples,
    )
    _atomic_write(
        args.output_json.resolve(),
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(args.output_markdown.resolve(), render_markdown(report).encode())


if __name__ == "__main__":
    main()

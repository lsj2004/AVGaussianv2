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

from avgaussianv2.benchmark.evaluation import load_evaluation  # noqa: E402
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
    evaluation = load_evaluation(run_dir / "evaluations/step_030000")
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


def render_markdown(report: Mapping[str, Any]) -> str:
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
    return "\n".join(lines)


def build(
    *,
    gate_path: Path,
    candidate_main_path: Path,
    candidate_seed_path: Path,
    audio_main_path: Path,
    audio_seed_path: Path,
    run_root: Path,
    p2_fair_report_path: Path,
    resamples: int,
) -> dict[str, Any]:
    gate = _load_json(gate_path)
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
    repository = gate["repository"]
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

    p2 = _load_json(p2_fair_report_path)
    if (
        p2.get("schema") != "avgaussianv2.p2-fair-baseline-report"
        or p2.get("repository") != repository
    ):
        raise ValueError("P2 reference report is not bound to the final repository")
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
    parser.add_argument("--candidate-seed-manifest", type=Path, required=True)
    parser.add_argument("--audio-main-manifest", type=Path, required=True)
    parser.add_argument("--audio-seed-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--p2-fair-report", type=Path, required=True)
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
        resamples=args.bootstrap_resamples,
    )
    _atomic_write(
        args.output_json.resolve(),
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(args.output_markdown.resolve(), render_markdown(report).encode())


if __name__ == "__main__":
    main()

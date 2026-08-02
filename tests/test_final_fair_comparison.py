from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_final_fair_comparison.py"
REPOSITORY = {"root": "/formal", "commit": "1" * 40, "clean": True}


def _load_module():
    spec = importlib.util.spec_from_file_location("final_fair_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _documents():
    finalist = {"system": "query_dependent_p1", "lambda_lre": 0.02}
    gate = {
        "schema": "avgaussianv2.p3-30k-gate",
        "repository": REPOSITORY,
        "selected_finalist": finalist,
    }

    def document(stage, runs):
        return {"repository": REPOSITORY, "stage": stage, "runs": runs}

    candidate_main = []
    candidate_seeds = []
    audio_main = []
    audio_seeds = []
    for seed in (17, 42, 73):
        for scene in ("scene1_opera", "Scene7playing"):
            for weight in (0.0, 0.02):
                run = {
                    "continuation_id": f"query__{scene}__seed{seed}__{weight}",
                    "system": "query_dependent_p1",
                    "scene": scene,
                    "seed": seed,
                    "lambda_lre": weight,
                }
                (candidate_main if seed == 42 else candidate_seeds).append(run)
            audio = {
                "continuation_id": f"audio__{scene}__seed{seed}",
                "system": "audio_only",
                "scene": scene,
                "seed": seed,
                "lambda_lre": 0.0,
            }
            (audio_main if seed == 42 else audio_seeds).append(audio)
    return (
        gate,
        document("confirmation", candidate_main),
        document("robustness", candidate_seeds),
        document("confirmation", audio_main),
        document("robustness", audio_seeds),
    )


def _evaluations(module):
    values = {}
    for role_index, role in enumerate(
        ("candidate", "architecture_control", "audio_only")
    ):
        for seed in module.SEEDS:
            for scene in module.SCENES:
                rows = []
                for index in range(4):
                    rows.append(
                        {
                            metric: float(role_index + seed / 1000 + index / 10000)
                            for metric in module.MODEL_METRICS
                        }
                    )
                values[(role, seed, scene)] = {
                    "sample_ids": [f"{scene}/{index}" for index in range(4)],
                    "metrics": {
                        metric: sum(row[metric] for row in rows) / len(rows)
                        for metric in module.MODEL_METRICS
                    },
                    "rows": rows,
                }
    return values


def test_select_exact_matrix_accepts_only_three_by_two_by_three() -> None:
    module = _load_module()
    gate, candidate_main, candidate_seeds, audio_main, audio_seeds = _documents()

    runs, system, treatment = module.select_exact_matrix(
        gate=gate,
        candidate_main=candidate_main,
        candidate_seeds=candidate_seeds,
        audio_main=audio_main,
        audio_seeds=audio_seeds,
    )

    assert len(runs) == 18
    assert system == "query_dependent_p1"
    assert treatment == pytest.approx(0.02)


def test_select_exact_matrix_rejects_missing_audio_seed() -> None:
    module = _load_module()
    gate, candidate_main, candidate_seeds, audio_main, audio_seeds = _documents()
    audio_seeds["runs"].pop()

    with pytest.raises(ValueError, match="final fair matrix mismatch"):
        module.select_exact_matrix(
            gate=gate,
            candidate_main=candidate_main,
            candidate_seeds=candidate_seeds,
            audio_main=audio_main,
            audio_seeds=audio_seeds,
        )


def test_aggregation_is_paired_and_bootstrap_is_deterministic() -> None:
    module = _load_module()
    evaluations = _evaluations(module)

    first = module.aggregate_final_models(evaluations, resamples=1_000)
    second = module.aggregate_final_models(evaluations, resamples=1_000)

    assert first == second
    comparison = first[1]["candidate_vs_audio_only"]["paper_lre_db"]
    assert comparison["mean_delta"] == pytest.approx(-2.0)
    assert comparison["scene_seed_equal_paired_win_rate"] == pytest.approx(1.0)
    assert comparison["conclusion"] == "candidate_improves"


def test_markdown_has_separate_fair_and_reference_tables() -> None:
    module = _load_module()
    models, comparisons = module.aggregate_final_models(
        _evaluations(module), resamples=1_000
    )
    reference = {
        name: {metric: 0.5 for metric in module.DISPLAY_METRICS}
        for name in ("source_binaural", "mono", "native_audiogs")
    }
    report = {
        "selected_finalist": {
            "system": "query_dependent_p1",
            "lambda_lre": 0.02,
        },
        "models": models,
        "paired_comparisons": comparisons,
        "absolute_references": reference,
        "candidate_reference_deltas": {
            name: {metric: -0.5 for metric in module.DISPLAY_METRICS}
            for name in (
                "source_binaural",
                "mono",
                "native_audiogs",
                "audio_only",
            )
        },
    }

    text = module.render_markdown(report)

    assert "严格公平主榜" in text
    assert "Source/Mono/native 绝对参照榜" in text
    assert "candidate_vs_audio_only" not in text
    assert "source_binaural" in text
    assert "mono" in text
    assert "最终候选相对绝对参照的差值" in text


def test_no_finalist_markdown_reports_rejection_without_multiseed_claim() -> None:
    module = _load_module()
    metrics = {metric: 0.5 for metric in module.MODEL_METRICS}
    report = {
        "status": "no_finalist",
        "audio_only_seed42_30k": metrics,
        "rejected_candidates": [
            {
                "system": "query_dependent_p1",
                "lambda_lre": 0.02,
                "passed": False,
                "reasons": ["causal_gate_failed"],
                "treatment_macro": metrics,
            }
        ],
        "absolute_references": {
            name: {metric: 0.5 for metric in module.DISPLAY_METRICS}
            for name in ("source_binaural", "mono", "native_audiogs")
        },
    }

    text = module.render_markdown(report)

    assert "无候选通过 30k 门禁" in text
    assert "causal_gate_failed" in text
    assert "未启动候选多 seed" in text
    assert "Source/Mono/native 绝对参照" in text


def test_no_finalist_builder_requires_audio_only_and_preserves_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    candidate_path = tmp_path / "candidate.json"
    audio_path = tmp_path / "audio.json"
    gate_path = tmp_path / "gate.json"
    p2_path = tmp_path / "p2.json"
    candidate_path.write_text("{}\n")
    audio_path.write_text("{}\n")
    gate_path.write_text("{}\n")
    metrics = {metric: 0.5 for metric in module.MODEL_METRICS}
    audio_runs = [
        {
            "continuation_id": f"audio__{scene}",
            "stage": "confirmation",
            "system": "audio_only",
            "scene": scene,
            "seed": 42,
            "lambda_lre": 0.0,
        }
        for scene in module.SCENES
    ]
    candidate_document = {"repository": REPOSITORY, "runs": []}
    audio_document = {"repository": REPOSITORY, "runs": audio_runs}
    monkeypatch.setattr(
        module,
        "_manifest",
        lambda path: candidate_document if path == candidate_path else audio_document,
    )
    monkeypatch.setattr(
        module,
        "_load_model_run",
        lambda *_args: {
            "content_sha256": "a" * 64,
            "metric_protocol": {"fixed": True},
            "metrics": metrics,
        },
    )
    p2_path.write_text(
        json.dumps(
            {
                "schema": "avgaussianv2.p2-fair-baseline-report",
                "repository": REPOSITORY,
                "scene_macro": {
                    name: {metric: 0.5 for metric in module.DISPLAY_METRICS}
                    for name in ("source_binaural", "mono", "native_audiogs")
                },
            }
        )
    )
    gate = {
        "schema": "avgaussianv2.p3-30k-gate",
        "repository": REPOSITORY,
        "manifest_30k": {
            "path": str(candidate_path.resolve()),
            "sha256": module._sha256(candidate_path),
        },
        "selected_finalist": None,
        "candidates": [
            {
                "system": "query_dependent_p1",
                "lambda_lre": 0.02,
                "passed": False,
                "reasons": ["lre_not_improved_in_both_scenes"],
                "treatment_macro": metrics,
            }
        ],
    }

    report = module._build_no_finalist(
        gate=gate,
        gate_path=gate_path,
        candidate_main_path=candidate_path,
        audio_main_path=audio_path,
        run_root=tmp_path,
        p2_fair_report_path=p2_path,
    )

    assert report["status"] == "no_finalist"
    assert report["selected_finalist"] is None
    assert report["rejected_candidates"][0]["reasons"] == [
        "lre_not_improved_in_both_scenes"
    ]
    assert report["audio_only_seed42_30k"]["paper_lre_db"] == pytest.approx(0.5)

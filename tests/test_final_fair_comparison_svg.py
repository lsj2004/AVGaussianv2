from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_final_fair_comparison_svg.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("final_fair_svg", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _architecture_summary(module):
    systems = {}
    for index, system in enumerate(module.ARCHITECTURE_SYSTEMS):
        systems[system] = {
            "parameter_elements": {
                "optimizer_active": {
                    "min": 1_000_000 + index,
                    "max": 1_000_000 + index,
                },
                "total": {"min": 2_000_000 + index, "max": 2_000_000 + index},
                "orphan_trainable": {"min": 0, "max": 0},
            },
            "p2_5k_resource": {
                "run_count": 8,
                "maximum_peak_memory_used_mib": 1200 + index * 100,
                "mean_pipeline_elapsed_seconds": 1000.0 + index * 100,
                "total_pipeline_gpu_hours": 2.0 + index,
            },
        }
    return {"systems": systems}


def _metrics(module, value: float):
    return {
        metric: value + index / 100
        for index, metric in enumerate(module.PRIMARY_METRICS)
    }


def _finalist_report(module):
    return {
        "schema": module.SCHEMA,
        "version": 1,
        "status": "finalist",
        "selected_finalist": {
            "system": "query_dependent_p1",
            "lambda_lre": 0.02,
        },
        "architecture_summary": _architecture_summary(module),
        "models": {
            "across_seed": {
                role: {
                    metric: {"mean": value}
                    for metric, value in _metrics(module, base).items()
                }
                for role, base in (
                    ("candidate", 0.8),
                    ("architecture_control", 1.1),
                    ("audio_only", 1.0),
                )
            }
        },
        "absolute_references": {
            name: {
                metric: value
                for metric, value in _metrics(module, base).items()
                if metric in module.REFERENCE_METRICS
            }
            for name, base in (
                ("source_binaural", 1.2),
                ("mono", 0.9),
                ("native_audiogs", 1.05),
            )
        },
    }


def _no_finalist_report(module):
    metrics = _metrics(module, 1.0)
    return {
        "schema": module.SCHEMA,
        "version": 1,
        "status": "no_finalist",
        "architecture_summary": _architecture_summary(module),
        "audio_only_seed42_30k": metrics,
        "rejected_candidates": [
            {
                "system": "query_dependent_p1",
                "lambda_lre": 0.02,
                "treatment_macro": _metrics(module, 1.2),
            }
        ],
        "absolute_references": {
            name: {
                metric: value
                for metric, value in _metrics(module, base).items()
                if metric in module.REFERENCE_METRICS
            }
            for name, base in (
                ("source_binaural", 1.2),
                ("mono", 0.9),
                ("native_audiogs", 1.05),
            )
        },
    }


@pytest.mark.parametrize("factory", (_finalist_report, _no_finalist_report))
def test_renderer_outputs_three_self_contained_svg_documents(factory) -> None:
    module = _load_module()
    rendered = module.render_all(factory(module))

    assert set(rendered) == {
        "strict-model-metric-ratios.svg",
        "absolute-reference-ratios.svg",
        "p2-architecture-resource-pareto.svg",
    }
    for content in rendered.values():
        assert content.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
        assert content.endswith("</svg>\n")
        assert "aria-labelledby" in content
        assert "Audio-only" in content or "P2 architecture" in content


def test_resource_chart_marks_only_non_dominated_points() -> None:
    module = _load_module()
    svg = module.render_architecture_resource_pareto(_finalist_report(module))

    assert "Thick outline = non-dominated" in svg
    assert 'stroke-width="4"' in svg
    assert svg.count('stroke-width="4"') == 1
    for system in module.ARCHITECTURE_SYSTEMS:
        assert system in svg


def test_renderer_accepts_perfect_candidate_but_rejects_zero_audio_denominator() -> (
    None
):
    module = _load_module()
    report = _finalist_report(module)
    report["models"]["across_seed"]["candidate"]["paper_lre_db"]["mean"] = 0

    rendered = module.render_all(report)

    assert "strict-model-metric-ratios.svg" in rendered
    report["models"]["across_seed"]["audio_only"]["paper_lre_db"]["mean"] = 0

    with pytest.raises(ValueError, match="finite and positive"):
        module.render_all(report)

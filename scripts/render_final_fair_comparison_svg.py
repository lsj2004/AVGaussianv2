from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "avgaussianv2.final-fair-comparison"
ARCHITECTURE_SYSTEMS = (
    "audio_only",
    "query_dependent_p1",
    "joint_conditioned",
    "cross_attention_masks",
)
PRIMARY_METRICS = (
    "audio_total",
    "waveform_l1",
    "paper_mag",
    "paper_env",
    "paper_dpam",
    "paper_lre_db",
    "ild_error_db",
    "ipd_error_rad",
)
REFERENCE_METRICS = PRIMARY_METRICS[1:]
METRIC_LABELS = {
    "audio_total": "Audio total",
    "waveform_l1": "Waveform",
    "paper_mag": "MAG",
    "paper_env": "ENV",
    "paper_dpam": "DPAM",
    "paper_lre_db": "LRE",
    "ild_error_db": "ILD",
    "ipd_error_rad": "IPD",
}
SERIES_COLORS = ("#3366cc", "#dc3912", "#109618", "#990099", "#ff9900")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read final report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("final report must be a JSON object")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _validate_architecture_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = report.get("architecture_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("final report lacks architecture summary")
    systems = summary.get("systems")
    if not isinstance(systems, Mapping) or set(systems) != set(ARCHITECTURE_SYSTEMS):
        raise ValueError("final report architecture system mismatch")
    for system in ARCHITECTURE_SYSTEMS:
        record = systems[system]
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid architecture summary: {system}")
        parameters = record.get("parameter_elements")
        resource = record.get("p2_5k_resource")
        if not isinstance(parameters, Mapping) or not isinstance(resource, Mapping):
            raise ValueError(f"architecture evidence missing: {system}")
        for field in ("optimizer_active", "total", "orphan_trainable"):
            value = parameters.get(field)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"architecture parameter range missing: {system}/{field}"
                )
            if field == "orphan_trainable":
                _finite_nonnegative(value.get("max"), label=f"{system}/{field}/max")
            else:
                _finite_positive(value.get("max"), label=f"{system}/{field}/max")
        for field in (
            "run_count",
            "maximum_peak_memory_used_mib",
            "mean_pipeline_elapsed_seconds",
            "total_pipeline_gpu_hours",
        ):
            _finite_positive(resource.get(field), label=f"{system}/{field}")
    return systems


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA or report.get("version") != 1:
        raise ValueError("unsupported final report identity")
    if report.get("status") not in {"finalist", "no_finalist"}:
        raise ValueError("unsupported final report status")
    _validate_architecture_summary(report)


def _svg_document(title: str, body: Sequence[str], *, width: int, height: int) -> str:
    escaped_title = html.escape(title)
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escaped_title}</title>',
            '<desc id="desc">All plotted error metrics are lower-is-better. Values are derived from the verified final report.</desc>',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            "<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#5f6368;stroke-width:1}.grid{stroke:#dadce0;stroke-width:1}.label{font-size:13px}.small{font-size:11px}.title{font-size:18px;font-weight:600}.note{font-size:12px;fill:#5f6368}</style>",
            *body,
            "</svg>",
            "",
        ]
    )


def _model_series(report: Mapping[str, Any]) -> list[tuple[str, dict[str, float]]]:
    if report["status"] == "finalist":
        finalist = report.get("selected_finalist")
        models = report.get("models", {}).get("across_seed", {})
        if not isinstance(finalist, Mapping) or not isinstance(models, Mapping):
            raise ValueError("finalist report lacks strict model aggregates")
        labels = {
            "candidate": f"{finalist['system']}/lambda={finalist['lambda_lre']}",
            "architecture_control": f"{finalist['system']}/lambda=0",
            "audio_only": "audio_only/lambda=0",
        }
        series = []
        for role in ("candidate", "architecture_control", "audio_only"):
            values = models.get(role)
            if not isinstance(values, Mapping):
                raise ValueError(f"strict model aggregate missing: {role}")
            series.append(
                (
                    labels[role],
                    {
                        metric: _finite_nonnegative(
                            values.get(metric, {}).get("mean"),
                            label=f"{role}/{metric}",
                        )
                        for metric in PRIMARY_METRICS
                    },
                )
            )
        return series

    audio = report.get("audio_only_seed42_30k")
    rejected = report.get("rejected_candidates")
    if not isinstance(audio, Mapping) or not isinstance(rejected, list):
        raise ValueError("no-finalist report lacks seed42 model evidence")
    series = [
        (
            "audio_only/lambda=0",
            {
                metric: _finite_nonnegative(audio.get(metric), label=f"audio/{metric}")
                for metric in PRIMARY_METRICS
            },
        )
    ]
    for candidate in rejected:
        if not isinstance(candidate, Mapping) or not isinstance(
            candidate.get("treatment_macro"), Mapping
        ):
            raise ValueError("invalid rejected candidate evidence")
        label = f"{candidate['system']}/lambda={candidate['lambda_lre']}"
        series.append(
            (
                label,
                {
                    metric: _finite_nonnegative(
                        candidate["treatment_macro"].get(metric),
                        label=f"{label}/{metric}",
                    )
                    for metric in PRIMARY_METRICS
                },
            )
        )
    return series


def render_fair_metric_ratios(report: Mapping[str, Any]) -> str:
    series = _model_series(report)
    audio = next(
        (values for label, values in series if label.startswith("audio_only")), None
    )
    if audio is None:
        raise ValueError("strict model chart requires Audio-only")
    for metric in PRIMARY_METRICS:
        _finite_positive(audio[metric], label=f"audio denominator/{metric}")
    ratios = {
        label: {metric: values[metric] / audio[metric] for metric in PRIMARY_METRICS}
        for label, values in series
    }
    maximum = max(value for values in ratios.values() for value in values.values())
    upper = max(1.1, math.ceil(maximum * 10.0) / 10.0)
    width, height = 1240, 620
    left, top, plot_width, plot_height = 90, 90, 1090, 400
    group_width = plot_width / len(PRIMARY_METRICS)
    bar_width = min(28.0, group_width * 0.72 / len(series))
    body = [
        '<text class="title" x="30" y="35">Strict model metrics normalized to Audio-only</text>',
        '<text class="note" x="30" y="58">Ratio &lt; 1 improves on Audio-only; all metrics are lower-is-better.</text>',
    ]
    for tick_index in range(6):
        value = upper * tick_index / 5
        y = top + plot_height * (1 - tick_index / 5)
        body.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        body.append(
            f'<text class="small" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>'
        )
    baseline_y = top + plot_height * (1 - 1.0 / upper)
    body.append(
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_width}" y2="{baseline_y:.2f}" stroke="#202124" stroke-width="1.5"/>'
    )
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        center = left + (metric_index + 0.5) * group_width
        for series_index, (label, _) in enumerate(series):
            value = ratios[label][metric]
            height_value = plot_height * value / upper
            x = center - len(series) * bar_width / 2 + series_index * bar_width
            y = top + plot_height - height_value
            color = SERIES_COLORS[series_index]
            body.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" height="{height_value:.2f}" fill="{color}" opacity="0.85"/>'
            )
            body.append(
                f'<text class="small" x="{x + (bar_width - 2) / 2:.2f}" y="{max(top + 10, y - 4):.2f}" text-anchor="middle">{value:.2f}</text>'
            )
        body.append(
            f'<text class="label" x="{center:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{html.escape(METRIC_LABELS[metric])}</text>'
        )
    legend_y = 550
    for index, (label, _) in enumerate(series):
        x = 90 + index * 360
        body.append(
            f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" fill="{SERIES_COLORS[index]}"/>'
        )
        body.append(
            f'<text class="label" x="{x + 22}" y="{legend_y}">{html.escape(label)}</text>'
        )
    return _svg_document("Strict model metric ratios", body, width=width, height=height)


def render_architecture_resource_pareto(report: Mapping[str, Any]) -> str:
    systems = _validate_architecture_summary(report)
    points = []
    for system in ARCHITECTURE_SYSTEMS:
        record = systems[system]
        active = (
            _finite_positive(
                record["parameter_elements"]["optimizer_active"]["max"],
                label=f"{system}/optimizer_active",
            )
            / 1_000_000
        )
        elapsed = _finite_positive(
            record["p2_5k_resource"]["mean_pipeline_elapsed_seconds"],
            label=f"{system}/elapsed",
        )
        memory = _finite_positive(
            record["p2_5k_resource"]["maximum_peak_memory_used_mib"],
            label=f"{system}/memory",
        )
        points.append((system, active, elapsed, memory))
    frontier = {
        system
        for system, active, elapsed, _ in points
        if not any(
            (other_active <= active and other_elapsed <= elapsed)
            and (other_active < active or other_elapsed < elapsed)
            for other_system, other_active, other_elapsed, _ in points
            if other_system != system
        )
    }
    width, height = 1000, 650
    left, top, plot_width, plot_height = 100, 90, 790, 440
    max_x = max(point[1] for point in points) * 1.1
    min_y = min(point[2] for point in points) * 0.9
    max_y = max(point[2] for point in points) * 1.08
    min_memory = min(point[3] for point in points)
    max_memory = max(point[3] for point in points)
    body = [
        '<text class="title" x="30" y="35">P2 architecture resource Pareto</text>',
        '<text class="note" x="30" y="58">Lower-left is better. Runtime and memory are P2 5k screening evidence, not 30k cost.</text>',
    ]
    for index in range(6):
        x_value = max_x * index / 5
        x = left + plot_width * index / 5
        body.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>'
        )
        body.append(
            f'<text class="small" x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle">{x_value:.1f}</text>'
        )
        y_value = min_y + (max_y - min_y) * index / 5
        y = top + plot_height * (1 - index / 5)
        body.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        body.append(
            f'<text class="small" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{y_value:.0f}</text>'
        )
    body.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text class="label" x="{left + plot_width / 2}" y="{height - 55}" text-anchor="middle">Optimizer-active parameters (million)</text>',
            f'<text class="label" transform="translate(28 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">Mean P2 5k pipeline seconds</text>',
        ]
    )
    for index, (system, active, elapsed, memory) in enumerate(points):
        x = left + plot_width * active / max_x
        y = top + plot_height * (1 - (elapsed - min_y) / (max_y - min_y))
        radius = 10 + 10 * (memory - min_memory) / max(1.0, max_memory - min_memory)
        stroke_width = 4 if system in frontier else 1.5
        body.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{SERIES_COLORS[index]}" fill-opacity="0.72" stroke="#202124" stroke-width="{stroke_width}"/>'
        )
        body.append(
            f'<text class="label" x="{x + radius + 7:.2f}" y="{y - 3:.2f}">{html.escape(system)}</text>'
        )
        body.append(
            f'<text class="small" x="{x + radius + 7:.2f}" y="{y + 13:.2f}">{active:.2f}M, {elapsed:.0f}s, {memory:.0f}MiB</text>'
        )
    body.append(
        '<text class="note" x="100" y="620">Thick outline = non-dominated on optimizer-active parameters and mean 5k runtime. Bubble size = peak memory.</text>'
    )
    return _svg_document(
        "P2 architecture resource Pareto", body, width=width, height=height
    )


def render_reference_ratios(report: Mapping[str, Any]) -> str:
    references = report.get("absolute_references")
    if not isinstance(references, Mapping):
        raise ValueError("final report lacks absolute references")
    model_series = _model_series(report)
    audio = next(
        (values for label, values in model_series if label.startswith("audio_only")),
        None,
    )
    if audio is None:
        raise ValueError("reference chart requires Audio-only")
    for metric in REFERENCE_METRICS:
        _finite_positive(audio[metric], label=f"audio denominator/{metric}")
    series = [("audio_only", {metric: audio[metric] for metric in REFERENCE_METRICS})]
    if report["status"] == "finalist":
        series.insert(0, ("final_candidate", model_series[0][1]))
    for name in ("source_binaural", "mono", "native_audiogs"):
        values = references.get(name)
        if not isinstance(values, Mapping):
            raise ValueError(f"absolute reference missing: {name}")
        series.append(
            (
                name,
                {
                    metric: _finite_nonnegative(
                        values.get(metric), label=f"{name}/{metric}"
                    )
                    for metric in REFERENCE_METRICS
                },
            )
        )
    ratios = {
        label: {metric: values[metric] / audio[metric] for metric in REFERENCE_METRICS}
        for label, values in series
    }
    width, height = 1240, 650
    left, top, plot_width, plot_height = 100, 90, 1060, 420
    maximum = max(value for values in ratios.values() for value in values.values())
    upper = max(1.1, math.ceil(maximum * 10) / 10)
    body = [
        '<text class="title" x="30" y="35">Absolute references normalized to Audio-only</text>',
        '<text class="note" x="30" y="58">Metric-matched descriptive ratios only; Source/Mono/native are not update- and seed-matched models.</text>',
    ]
    for tick_index in range(6):
        value = upper * tick_index / 5
        y = top + plot_height * (1 - tick_index / 5)
        body.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        body.append(
            f'<text class="small" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>'
        )
    for metric_index, metric in enumerate(REFERENCE_METRICS):
        x = left + plot_width * metric_index / (len(REFERENCE_METRICS) - 1)
        body.append(
            f'<text class="label" x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle">{html.escape(METRIC_LABELS[metric])}</text>'
        )
    for series_index, (label, _) in enumerate(series):
        coordinates = []
        for metric_index, metric in enumerate(REFERENCE_METRICS):
            x = left + plot_width * metric_index / (len(REFERENCE_METRICS) - 1)
            y = top + plot_height * (1 - ratios[label][metric] / upper)
            coordinates.append((x, y))
        color = SERIES_COLORS[series_index]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, y in coordinates:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    legend_y = 585
    for index, (label, _) in enumerate(series):
        x = 70 + (index % 3) * 390
        y = legend_y + (index // 3) * 28
        body.append(
            f'<line x1="{x}" y1="{y - 4}" x2="{x + 22}" y2="{y - 4}" stroke="{SERIES_COLORS[index]}" stroke-width="3"/>'
        )
        body.append(
            f'<text class="label" x="{x + 30}" y="{y}">{html.escape(label)}</text>'
        )
    return _svg_document(
        "Absolute reference metric ratios", body, width=width, height=height
    )


def render_all(report: Mapping[str, Any]) -> dict[str, str]:
    validate_report(report)
    return {
        "strict-model-metric-ratios.svg": render_fair_metric_ratios(report),
        "absolute-reference-ratios.svg": render_reference_ratios(report),
        "p2-architecture-resource-pareto.svg": render_architecture_resource_pareto(
            report
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = _load_json(args.report)
    for name, content in render_all(report).items():
        _atomic_write(args.output_dir.resolve() / name, content)


if __name__ == "__main__":
    main()

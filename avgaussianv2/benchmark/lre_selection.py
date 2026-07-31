"""Fail-closed screening selection for the LRE loss ablation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluationResult,
    load_evaluation,
)
from avgaussianv2.benchmark.lre_orchestration import load_lre_run_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _relative_improvement(candidate: float, control: float) -> float:
    denominator = abs(control)
    if denominator == 0:
        return 0.0 if candidate == control else -1.0
    return (control - candidate) / denominator


def _relative_degradation(candidate: float, control: float) -> float:
    denominator = abs(control)
    if denominator == 0:
        return 0.0 if candidate == control else 1.0
    return (candidate - control) / denominator


def _mean(result: BenchmarkEvaluationResult, metric: str) -> float:
    try:
        value = float(result.summary[metric]["mean"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"screening evaluation lacks metric {metric}") from error
    if not math.isfinite(value):
        raise ValueError(f"screening metric is nonfinite: {metric}")
    return value


def select_screening_winners(
    manifest_path: Path,
    run_root: Path,
    output_path: Path,
    *,
    evaluation_loader: Callable[[Path], BenchmarkEvaluationResult] = load_evaluation,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_lre_run_manifest(manifest_path)
    if manifest["stage"] != "screening":
        raise ValueError("winner selection requires a screening manifest")
    source_path = Path(str(manifest["source_manifest"])).resolve()
    if _sha256(source_path) != manifest["source_manifest_sha256"]:
        raise ValueError("LRE ablation source manifest hash mismatch")
    source = yaml.safe_load(source_path.read_text())
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != "avgaussianv2.lre-loss-ablation"
        or source.get("version") != 1
    ):
        raise ValueError("unsupported LRE ablation source manifest")
    screening = source["screening"]
    gates = screening["gates"]
    if int(gates["nonfinite_samples_max"]) != 0:
        raise ValueError("screening requires a zero nonfinite-sample tolerance")
    maximum = int(screening["keep_nonzero_candidates_max"])

    by_id = {str(run["run_id"]): run for run in manifest["runs"]}
    weights = sorted({float(run["lambda_lre"]) for run in manifest["runs"]})
    expected_units = {
        (run["scene"], run["system"], run["training_mode"], run["seed"])
        for run in manifest["runs"]
        if float(run["lambda_lre"]) == 0.0
    }
    candidates = []
    for weight in weights:
        if weight == 0.0:
            continue
        units = []
        for run in manifest["runs"]:
            if float(run["lambda_lre"]) != weight:
                continue
            control_id = run["control_run_id"]
            if not isinstance(control_id, str) or control_id not in by_id:
                raise ValueError(f"missing paired control for {run['run_id']}")
            control_run = by_id[control_id]
            for field in ("scene", "system", "training_mode", "seed"):
                if run[field] != control_run[field]:
                    raise ValueError(
                        f"control pairing mismatch for {run['run_id']}: {field}"
                    )
            step = int(run["max_steps"])
            candidate = evaluation_loader(
                Path(run_root).resolve()
                / str(run["continuation_id"])
                / "evaluations"
                / f"step_{step:06d}"
            )
            control = evaluation_loader(
                Path(run_root).resolve()
                / str(control_run["continuation_id"])
                / "evaluations"
                / f"step_{step:06d}"
            )
            expected = (run["scene"], run["system"], step)
            if (
                candidate.identity.scene_id,
                candidate.identity.system_name,
                candidate.identity.reporting_step,
            ) != expected:
                raise ValueError(
                    f"candidate evaluation identity mismatch: {run['run_id']}"
                )
            if (
                control.identity.scene_id,
                control.identity.system_name,
                control.identity.reporting_step,
            ) != expected:
                raise ValueError(f"control evaluation identity mismatch: {control_id}")
            if candidate.identity.expected_sample_ids != control.identity.expected_sample_ids:
                raise ValueError(f"candidate/control samples differ: {run['run_id']}")
            candidate_lre = _mean(candidate, "lre_error_db")
            lre = _relative_improvement(
                candidate_lre, _mean(control, "lre_error_db")
            )
            audio = _relative_degradation(
                _mean(candidate, "audio_total"), _mean(control, "audio_total")
            )
            waveform = _relative_degradation(
                _mean(candidate, "waveform_l1"), _mean(control, "waveform_l1")
            )
            passed = (
                lre >= float(gates["lre_relative_improvement_min"])
                and audio <= float(gates["audio_total_relative_degradation_max"])
                and waveform <= float(gates["waveform_l1_relative_degradation_max"])
            )
            units.append(
                {
                    "scene": run["scene"],
                    "system": run["system"],
                    "training_mode": run["training_mode"],
                    "seed": run["seed"],
                    "lre_error_db": candidate_lre,
                    "lre_relative_improvement": lre,
                    "audio_total_relative_degradation": audio,
                    "waveform_l1_relative_degradation": waveform,
                    "nonfinite_samples": 0,
                    "passed": passed,
                }
            )
        if not units:
            raise ValueError(f"screening weight has no runs: {weight}")
        observed_units = {
            (
                unit["scene"],
                unit["system"],
                unit["training_mode"],
                unit["seed"],
            )
            for unit in units
        }
        if observed_units != expected_units:
            raise ValueError(f"screening candidate coverage mismatch: {weight}")
        candidates.append(
            {
                "lambda_lre": weight,
                "passed": all(bool(unit["passed"]) for unit in units),
                "macro_lre_relative_improvement": sum(
                    float(unit["lre_relative_improvement"]) for unit in units
                )
                / len(units),
                "macro_lre_error_db": sum(
                    float(unit["lre_error_db"]) for unit in units
                )
                / len(units),
                "units": units,
            }
        )
    ranked = sorted(
        (record for record in candidates if record["passed"]),
        key=lambda record: (
            float(record["macro_lre_error_db"]),
            float(record["lambda_lre"]),
        ),
    )
    selected = [float(record["lambda_lre"]) for record in ranked[:maximum]]
    result = {
        "schema": "avgaussianv2.lre-loss-screening-selection",
        "version": 1,
        "source_screening_manifest": str(manifest_path),
        "source_screening_manifest_sha256": _sha256(manifest_path),
        "selected_lambda_lre": selected,
        "gates": dict(gates),
        "candidates": candidates,
    }
    _atomic_json(Path(output_path).resolve(), result)
    return result


__all__ = ["select_screening_winners"]

#!/usr/bin/env python3
"""Run the frozen P3 gate with strict evaluation and training-evidence audits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avgaussianv2.benchmark.evaluation import verify_evaluation  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_gate_module(path: Path, expected_sha256: str) -> ModuleType:
    if _sha256(path) != expected_sha256:
        raise ValueError("P3 gate implementation hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_p3_30k_gate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load P3 gate implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StrictEvaluationLoader:
    """Drop-in replacement for the legacy gate's weak evaluation loader."""

    def __init__(self) -> None:
        self._metric_protocol: str | None = None

    def __call__(
        self,
        run_dir: Path,
        step: int,
        main: str,
        system: str,
        metrics: Sequence[str],
    ) -> dict[str, object]:
        identity = _load_json(run_dir / "continuation_identity.json")
        evaluation = verify_evaluation(
            (
                run_dir
                / "evaluations"
                / (Path() if system == main else system)
                / f"step_{step:06d}"
            )
        )
        if (
            identity.get("continuation_id") != run_dir.name
            or identity.get("system") != main
            or identity.get("scene") != evaluation.identity.scene_id
            or identity.get("seed") != 42
            or evaluation.identity.system_name != system
            or evaluation.identity.reporting_step != step
            or evaluation.provenance.get("seed") != 42
            or evaluation.provenance.get("checkpoint_step") != step
            or evaluation.provenance.get("main_update_matched") is not True
        ):
            raise ValueError(
                f"strict evaluation identity mismatch: {run_dir.name}/{system}/{step}"
            )
        rows = [dict(row) for row in evaluation.rows]
        sample_ids = [str(row.get("sample_id")) for row in rows]
        if (
            evaluation.count != len(rows)
            or sample_ids != list(evaluation.identity.expected_sample_ids)
        ):
            raise ValueError(
                f"strict evaluation sample mismatch: {run_dir.name}/{system}/{step}"
            )
        means: dict[str, float] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            if any(not math.isfinite(value) for value in values):
                raise ValueError(
                    f"nonfinite {metric}: {run_dir.name}/{system}/{step}"
                )
            mean = statistics.fmean(values)
            if (
                metric not in evaluation.summary
                or not math.isclose(
                    mean,
                    float(evaluation.summary[metric]["mean"]),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"row/summary mismatch: {run_dir.name}/{system}/{step}/{metric}"
                )
            means[metric] = mean
        protocol = json.dumps(
            evaluation.metric_protocol,
            sort_keys=True,
            separators=(",", ":"),
        )
        if self._metric_protocol is None:
            self._metric_protocol = protocol
        elif protocol != self._metric_protocol:
            raise ValueError("metric protocol mismatch across P3 gate evaluations")
        checkpoint_sha256 = evaluation.provenance.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise ValueError(
                f"checkpoint evidence missing: {run_dir.name}/{system}/{step}"
            )
        return {
            "manifest_sha256": evaluation.content_sha256,
            "summary": {
                "identity": evaluation.identity.to_mapping(),
                "provenance": evaluation.provenance,
            },
            "rows": rows,
            "sample_ids": sample_ids,
            "metrics": means,
            "checkpoint_sha256": checkpoint_sha256,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-implementation", type=Path, required=True)
    parser.add_argument("--gate-implementation-sha256", required=True)
    parser.add_argument("--gate-10k", type=Path, required=True)
    parser.add_argument("--manifest-30k", type=Path, required=True)
    parser.add_argument("--causal-manifest", type=Path, required=True)
    parser.add_argument("--noise-report", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = _load_gate_module(
        args.gate_implementation,
        args.gate_implementation_sha256,
    )
    module.load_evaluation = StrictEvaluationLoader()
    original_argv = sys.argv
    try:
        sys.argv = [
            str(args.gate_implementation),
            "--gate-10k",
            str(args.gate_10k),
            "--manifest-30k",
            str(args.manifest_30k),
            "--causal-manifest",
            str(args.causal_manifest),
            "--noise-report",
            str(args.noise_report),
            "--run-root",
            str(args.run_root),
            "--output",
            str(args.output),
        ]
        module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

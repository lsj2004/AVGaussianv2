"""Shared CLI execution for strict benchmark ablations.

Preparation remains experiment-specific. Once a preparation has been verified,
architecture and cross-attention ablations use the same worker and evaluator
pipelines; keeping that flow here makes the thin CLI modules describe only
their actual policy differences.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


PreparationVerifier = Callable[..., Mapping[str, Any]]

_DETERMINISTIC_ENVIRONMENT = {
    "PYTHONHASHSEED": "42",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def ensure_deterministic_environment(
    *,
    marker: str,
    module_name: str,
    experiment_name: str,
) -> None:
    """Re-exec a worker before Torch import when deterministic settings differ."""
    if all(
        os.environ.get(name) == value
        for name, value in _DETERMINISTIC_ENVIRONMENT.items()
    ):
        return
    if os.environ.get(marker) == "1":
        raise RuntimeError(f"{experiment_name} worker deterministic re-exec failed")
    environment = dict(os.environ)
    environment.update(_DETERMINISTIC_ENVIRONMENT)
    environment[marker] = "1"
    os.execve(
        sys.executable,
        [sys.executable, "-m", module_name, *sys.argv[1:]],
        environment,
    )


def run_ablation_worker(
    verify_preparation: PreparationVerifier,
    *,
    result_label: str,
) -> None:
    """Run a prepared ablation through the common fixed-budget worker."""
    import torch

    from avgaussianv2.cli.benchmark_worker import run_worker

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()

    preparation = verify_preparation(args.protocol_dir)
    result = run_worker(
        manifest_path=args.protocol_dir / "worker_manifest.json",
        output_dir=args.output_dir,
        config_path=args.protocol_dir / "resolved_project.yaml",
        device=torch.device(args.device),
        trust_upstream_artifacts=args.trust_upstream_artifacts,
        resume=args.resume,
        stop_after_step=args.stop_after_step,
    )
    print(
        json.dumps(
            {
                "scene_id": preparation["scene_id"],
                result_label: preparation[result_label],
                "warmup_step": result["completed_warmup_steps"],
                "main_step": result["completed_main_updates"],
                "runtime_contract_sha256": result["runtime_contract_sha256"],
            },
            sort_keys=True,
        )
    )


def run_ablation_evaluation(
    verify_preparation: PreparationVerifier,
    *,
    result_label: str,
    fixed_system: str | None = None,
    system_from_preparation: str | None = None,
    system_choices: Sequence[str] = (),
    allowed_systems_from_preparation: str | None = None,
    verify_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Evaluate one prepared ablation milestone through the common evaluator."""
    from avgaussianv2.benchmark.evaluation import BenchmarkEvaluator
    from avgaussianv2.benchmark.production import (
        build_evaluation_adapters,
        continuation_training_evidence,
        expected_identity,
    )

    configured_sources = sum(
        (
            fixed_system is not None,
            system_from_preparation is not None,
            bool(system_choices),
        )
    )
    if configured_sources != 1:
        raise ValueError("configure exactly one evaluation system source")

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    if system_choices:
        parser.add_argument(
            "--system",
            choices=tuple(system_choices),
            default=system_choices[0],
        )
    parser.add_argument(
        "--step",
        type=int,
        choices=(5_000, 10_000, 30_000),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    parser.add_argument("--compute-dpam", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preparation = verify_preparation(
        args.protocol_dir,
        **dict(verify_kwargs or {}),
    )
    scene_id = str(preparation["scene_id"])
    if fixed_system is not None:
        system = fixed_system
    elif system_from_preparation is not None:
        system = str(preparation[system_from_preparation])
    else:
        system = args.system
    if allowed_systems_from_preparation is not None:
        allowed = tuple(preparation[allowed_systems_from_preparation])
        if system not in allowed:
            raise ValueError(
                f"evaluation system {system!r} is not allowed by preparation"
            )
    identity = expected_identity(scene_id, system, args.step)
    evidence = continuation_training_evidence(
        args.worker_dir,
        scene_id=scene_id,
        system=system,
        step=args.step,
    )
    runtime_factory, predictor_factory = build_evaluation_adapters(
        resolved_config=args.protocol_dir / "resolved_project.yaml",
        device=args.device,
        evidence=evidence,
        trusted_upstream_artifacts=args.trust_upstream_artifacts,
        compute_dpam=args.compute_dpam,
    )
    result = BenchmarkEvaluator(args.device).evaluate(
        identity=identity,
        evidence=evidence,
        runtime_factory=runtime_factory,
        predictor_factory=predictor_factory,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    label_value = system if result_label == "system" else preparation[result_label]
    print(
        json.dumps(
            {
                "scene_id": scene_id,
                result_label: label_value,
                "step": args.step,
                "count": result.count,
                "content_sha256": result.content_sha256,
            },
            sort_keys=True,
        )
    )


__all__ = [
    "ensure_deterministic_environment",
    "run_ablation_evaluation",
    "run_ablation_worker",
]

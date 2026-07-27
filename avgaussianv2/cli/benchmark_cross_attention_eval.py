"""Evaluate cross-attention and its causal RGBD ablations on cam38."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.cross_attention_ablation import (
    ALL_CROSS_ATTENTION_EVALUATION_SYSTEMS,
    verify_cross_attention_preparation,
)
from avgaussianv2.benchmark.evaluation import BenchmarkEvaluator
from avgaussianv2.benchmark.production import (
    build_evaluation_adapters,
    continuation_training_evidence,
    expected_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--system",
        choices=ALL_CROSS_ATTENTION_EVALUATION_SYSTEMS,
        default="cross_attention",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=(5_000, 10_000, 30_000),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Training remains bound to the immutable preparation revision and worker
    # fingerprint. Evaluation code may receive audit-only fixes afterwards;
    # the strict checkpoint/runtime evidence below still rejects model,
    # configuration, source-inventory or data changes.
    preparation = verify_cross_attention_preparation(
        args.protocol_dir,
        require_repository_match=False,
    )
    allowed_systems = tuple(preparation["causal_evaluation_systems"])
    if args.system not in allowed_systems:
        parser.error(
            f"--system {args.system!r} is incompatible with prepared backend; "
            f"choose one of {allowed_systems}"
        )
    scene_id = str(preparation["scene_id"])
    identity = expected_identity(scene_id, args.system, args.step)
    evidence = continuation_training_evidence(
        args.worker_dir,
        scene_id=scene_id,
        system=args.system,
        step=args.step,
    )
    runtime_factory, predictor_factory = build_evaluation_adapters(
        resolved_config=args.protocol_dir / "resolved_project.yaml",
        device=args.device,
        evidence=evidence,
        trusted_upstream_artifacts=args.trust_upstream_artifacts,
    )
    result = BenchmarkEvaluator(args.device).evaluate(
        identity=identity,
        evidence=evidence,
        runtime_factory=runtime_factory,
        predictor_factory=predictor_factory,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "scene_id": scene_id,
                "system": args.system,
                "step": args.step,
                "count": result.count,
                "content_sha256": result.content_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Evaluate one native reference or one continuation milestone on cam38."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluator,
    verify_evaluation,
)
from avgaussianv2.benchmark.production import (
    build_evaluation_adapters,
    continuation_training_evidence,
    expected_identity,
    native_training_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    identity = expected_identity(args.scene, args.system, args.step)
    if args.verify_only:
        result = verify_evaluation(args.output_dir, identity=identity)
    else:
        if args.system.startswith("native_"):
            if args.step is not None:
                parser.error("native systems do not accept --step")
            evidence = native_training_evidence(
                args.source, scene_id=args.scene, system=args.system
            )
        else:
            if args.step is None:
                parser.error("continuation systems require --step")
            evidence = continuation_training_evidence(
                args.source,
                scene_id=args.scene,
                system=args.system,
                step=args.step,
            )
        runtime_factory, predictor_factory = build_evaluation_adapters(
            resolved_config=args.resolved_config,
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
                "scene_id": result.identity.scene_id,
                "system": result.identity.system_name,
                "step": result.identity.reporting_step,
                "count": result.count,
                "content_sha256": result.content_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

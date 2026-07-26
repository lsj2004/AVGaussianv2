"""Build one strict cross-attention versus FiLM scene report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.cross_attention_ablation import (
    CAUSAL_EVALUATION_SYSTEMS,
    verify_cross_attention_preparation,
)
from avgaussianv2.benchmark.cross_attention_report import (
    build_cross_attention_scene_report,
)
from avgaussianv2.benchmark.evaluation import REPORTING_STEPS, verify_evaluation
from avgaussianv2.benchmark.production import expected_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--film-eval-root", type=Path, required=True)
    parser.add_argument("--cross-eval-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    args = parser.parse_args()

    evaluations = []
    for step in REPORTING_STEPS:
        evaluations.append(
            verify_evaluation(
                args.film_eval_root / f"step_{step:06d}",
                identity=expected_identity(
                    args.scene_id,
                    "joint_conditioned",
                    step,
                ),
            )
        )
        for system in CAUSAL_EVALUATION_SYSTEMS:
            evaluations.append(
                verify_evaluation(
                    args.cross_eval_root / system / f"step_{step:06d}",
                    identity=expected_identity(args.scene_id, system, step),
                )
            )
    result = build_cross_attention_scene_report(
        scene_id=args.scene_id,
        evaluations=evaluations,
        expected_sample_count=args.expected_samples,
        output_dir=args.output_dir,
        preparation=verify_cross_attention_preparation(args.protocol_dir),
    )
    print(
        json.dumps(
            {
                "scene_id": result["scene_id"],
                "sample_count": result["sample_count"],
                "content_sha256": result["content_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

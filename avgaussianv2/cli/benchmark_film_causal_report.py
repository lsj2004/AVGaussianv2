"""Build the FiLM same-checkpoint causal report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.evaluation import REPORTING_STEPS, verify_evaluation
from avgaussianv2.benchmark.film_causal_report import SYSTEMS, build_film_causal_report
from avgaussianv2.benchmark.production import expected_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--evaluations-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--steps", type=int, nargs="+", choices=REPORTING_STEPS, default=REPORTING_STEPS)
    args = parser.parse_args()
    evaluations = []
    for system in SYSTEMS:
        for step in tuple(sorted(set(args.steps))):
            evaluations.append(verify_evaluation(
                args.evaluations_root / system / f"step_{step:06d}",
                identity=expected_identity(args.scene, system, step),
            ))
    report = build_film_causal_report(
        scene_id=args.scene, evaluations=evaluations,
        expected_sample_count=args.expected_samples, output_dir=args.output_dir,
    )
    print(json.dumps({"content_sha256": report["content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

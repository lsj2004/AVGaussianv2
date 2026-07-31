"""Build a paired plain-U-Net versus AudioGS residual report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.architecture_ablation import verify_architecture_preparation
from avgaussianv2.benchmark.architecture_report import build_architecture_scene_report
from avgaussianv2.benchmark.evaluation import REPORTING_STEPS, verify_evaluation
from avgaussianv2.benchmark.production import expected_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--audio-only-eval-root", type=Path, required=True)
    parser.add_argument("--architecture-eval-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--steps", type=int, nargs="+", choices=REPORTING_STEPS, default=REPORTING_STEPS)
    args = parser.parse_args()
    preparation = verify_architecture_preparation(args.protocol_dir)
    evaluations = []
    for step in tuple(sorted(set(args.steps))):
        for system, root in (
            ("audio_only", args.audio_only_eval_root),
            ("plain_unet", args.architecture_eval_root),
        ):
            evaluations.append(
                verify_evaluation(
                    root / f"step_{step:06d}",
                    identity=expected_identity(args.scene_id, system, step),
                )
            )
    report = build_architecture_scene_report(
        scene_id=args.scene_id,
        evaluations=evaluations,
        expected_sample_count=args.expected_samples,
        output_dir=args.output_dir,
        preparation=preparation,
    )
    print(json.dumps({"scene_id": report["scene_id"], "content_sha256": report["content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

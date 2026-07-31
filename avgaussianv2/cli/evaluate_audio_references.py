"""Evaluate AudioGS-paper Source Binaural and Mono references."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

from avgaussianv2.benchmark.audio_references import (
    CDPAMMetric,
    evaluate_reference_baselines,
    verify_reference_evaluation,
    write_reference_evaluation,
)
from avgaussianv2.benchmark.artifacts import repository_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-dpam",
        action="store_true",
        help="write an explicitly incomplete MAG/ENV/LRE-only report",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify_only:
        report = verify_reference_evaluation(args.output_dir)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "rows": report["row_count"],
                    "verified": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.config:
        parser.error("--config is required unless --verify-only is used")
    context = nullcontext(None) if args.skip_dpam else CDPAMMetric()
    with context as dpam_metric:
        evaluation = evaluate_reference_baselines(
            args.config,
            dpam_metric=dpam_metric,
            repository=repository_identity(),
        )
    write_reference_evaluation(
        evaluation,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "rows": len(evaluation.rows),
                "dpam": "skipped" if args.skip_dpam else "computed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

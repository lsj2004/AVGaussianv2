"""Run or independently verify both strict cam38 benchmark scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from avgaussianv2.benchmark.orchestration import parse_gpus, run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-native-training", action="store_true")
    args = parser.parse_args()
    report = run_benchmark_suite(
        repository=args.repository,
        output_dir=args.output_dir,
        gpus=parse_gpus(args.gpus),
        python_executable=args.python,
        resume=args.resume,
        verify_only=args.verify_only,
        skip_native_training=args.skip_native_training,
    )
    print(json.dumps({"content_sha256": report["content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

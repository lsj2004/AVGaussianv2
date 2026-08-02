"""Execute one generated LRE stage across one or more idle GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.lre_orchestration import run_lre_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--native-root",
        type=Path,
        required=True,
        help="root containing <scene>/{audiogs,ftgspp}/native_contract",
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--python",
        required=True,
        help="audited production Python containing AVGaussianV2 and FTGS++ dependencies",
    )
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    parser.add_argument("--skip-dpam", action="store_true")
    parser.add_argument(
        "--dpam-python",
        help="isolated Python containing cdpam for formal main evaluations",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-repository-relocation",
        action="store_true",
        help=(
            "allow an absolute worktree-root change only when the manifest and "
            "execution worktrees are clean at the exact same commit"
        ),
    )
    args = parser.parse_args()
    if args.dpam_python and args.skip_dpam:
        parser.error("--dpam-python cannot be combined with --skip-dpam")
    result = run_lre_manifest(
        args.manifest,
        output_root=args.output_root,
        native_root=args.native_root,
        gpus=args.gpus,
        python_executable=args.python,
        compute_dpam=not args.skip_dpam,
        trust_upstream_artifacts=args.trust_upstream_artifacts,
        resume=args.resume,
        dpam_python=args.dpam_python,
        allow_repository_relocation=args.allow_repository_relocation,
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "gpus": result["gpus"],
                "runs": len(result["runs"]),
                "result": str(
                    (
                        args.output_root
                        / f"runner_result.{result['stage']}.json"
                    ).resolve()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

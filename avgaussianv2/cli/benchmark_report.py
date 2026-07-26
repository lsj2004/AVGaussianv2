"""Build or independently verify strict cam38 scene/suite reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.evaluation import (
    CONTINUATION_SYSTEMS,
    NATIVE_SYSTEMS,
    REPORTING_STEPS,
    verify_evaluation,
)
from avgaussianv2.benchmark.production import expected_identity
from avgaussianv2.benchmark.report import (
    build_scene_report,
    build_suite_report,
    verify_scene_report,
    verify_suite_report,
)


def _scene_evaluations(scene: str, root: Path):
    values = []
    for system in sorted(NATIVE_SYSTEMS):
        identity = expected_identity(scene, system, None)
        values.append(verify_evaluation(root / system / "native", identity=identity))
    for system in sorted(CONTINUATION_SYSTEMS):
        for step in REPORTING_STEPS:
            identity = expected_identity(scene, system, step)
            values.append(
                verify_evaluation(root / system / f"step_{step:06d}", identity=identity)
            )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("scene", "suite"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--evaluations-root", type=Path)
    parser.add_argument("--scene-report", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.kind == "scene":
        if not args.scene or args.evaluations_root is None:
            parser.error("scene report requires --scene and --evaluations-root")
        if args.verify_only:
            result = verify_scene_report(args.output_dir)
        else:
            result = build_scene_report(
                scene_id=args.scene,
                evaluations=_scene_evaluations(args.scene, args.evaluations_root),
                output_dir=args.output_dir,
                resume=args.resume,
                overwrite=args.overwrite,
            )
    else:
        if args.verify_only:
            result = verify_suite_report(args.output_dir)
        else:
            if len(args.scene_report) != 2:
                parser.error("suite report requires exactly two --scene-report values")
            reports = [verify_scene_report(path) for path in args.scene_report]
            result = build_suite_report(
                scene_reports=reports,
                output_dir=args.output_dir,
                resume=args.resume,
                overwrite=args.overwrite,
            )
    print(json.dumps({"content_sha256": result["content_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

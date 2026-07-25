"""Finalize or verify immutable native cam38 training evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.native import (
    finalize_native_contract,
    verify_native_contract,
    write_audiogs_seed_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--write-audiogs-seed-record", action="store_true")
    parser.add_argument("--scene-id")
    parser.add_argument("--upstream-scene")
    parser.add_argument("--model-kind", choices=("audiogs", "ftgspp"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--conversion-manifest", type=Path)
    parser.add_argument("--rendered-config", type=Path)
    parser.add_argument("--sampled-scene-root", type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--seed-record", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.write_audiogs_seed_record:
        if (
            args.verify is not None
            or args.scene_id is None
            or args.upstream_scene is None
            or args.output is None
        ):
            parser.error(
                "seed-record mode requires --scene-id, --upstream-scene, and --output"
            )
        forbidden = (
            args.model_kind,
            args.config,
            args.provenance,
            args.checkpoint,
            args.upstream_root,
            args.conversion_manifest,
            args.rendered_config,
            args.sampled_scene_root,
            args.train_log,
            *args.seed_record,
        )
        if any(value is not None for value in forbidden):
            parser.error("seed-record mode cannot be combined with finalization arguments")
        record = write_audiogs_seed_record(
            args.output,
            scene_id=args.scene_id,
            upstream_scene=args.upstream_scene,
        )
        print(json.dumps({"status": "ok", **record}, sort_keys=True))
        return
    if args.scene_id is not None or args.upstream_scene is not None:
        parser.error("--scene-id/--upstream-scene require seed-record mode")
    if args.verify is not None:
        forbidden = (
            args.write_audiogs_seed_record,
            args.model_kind,
            args.config,
            args.provenance,
            args.checkpoint,
            args.upstream_root,
            args.output,
            args.conversion_manifest,
            args.rendered_config,
            args.sampled_scene_root,
            args.train_log,
            *args.seed_record,
        )
        if any(value not in (None, False) for value in forbidden):
            parser.error("--verify cannot be combined with finalization arguments")
        contract = verify_native_contract(args.verify)
        digest = contract["_manifest_sha256"]
    else:
        required = {
            "--model-kind": args.model_kind,
            "--config": args.config,
            "--provenance": args.provenance,
            "--checkpoint": args.checkpoint,
            "--upstream-root": args.upstream_root,
            "--output": args.output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"finalization requires {', '.join(missing)}")
        contract = finalize_native_contract(
            model_kind=args.model_kind,
            config_path=args.config,
            provenance_path=args.provenance,
            checkpoint_path=args.checkpoint,
            upstream_root=args.upstream_root,
            output_path=args.output,
            conversion_manifest=args.conversion_manifest,
            rendered_config=args.rendered_config,
            sampled_scene_root=args.sampled_scene_root,
            train_log=args.train_log,
            seed_records=args.seed_record,
        )
        digest = contract["_manifest_sha256"]
    print(
        json.dumps(
            {
                "status": "ok",
                "scene_id": contract["scene_id"],
                "model_kind": contract["model_kind"],
                "contract_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

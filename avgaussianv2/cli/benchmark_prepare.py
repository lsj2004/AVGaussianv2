"""Prepare strict train-only Task12 manifests on the assigned GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.production import (
    materialize_strict_scene_manifest,
    prepare_worker_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--native-audiogs-contract", type=Path, required=True)
    parser.add_argument("--native-ftgspp-contract", type=Path, required=True)
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    args = parser.parse_args()
    devices = tuple(item.strip() for item in args.devices.split(","))
    if not devices or any(not item for item in devices) or len(set(devices)) != len(devices):
        parser.error("--devices requires one or more distinct comma-separated devices")
    native_contract_dirs = {
        "audiogs": args.native_audiogs_contract,
        "ftgspp": args.native_ftgspp_contract,
    }
    materialize_strict_scene_manifest(
        config_path=args.config,
        native_contract_dirs=native_contract_dirs,
    )
    result = prepare_worker_manifests(
        config_path=args.config,
        output_dir=args.output_dir,
        devices=devices,
        trusted_upstream_artifacts=args.trust_upstream_artifacts,
        native_contract_dirs=native_contract_dirs,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Prepare one strict B/C audio-render architecture worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.architecture_ablation import (
    ABLATION_STRATEGIES,
    prepare_architecture_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--derived-config", type=Path, required=True)
    parser.add_argument("--base-protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=sorted(ABLATION_STRATEGIES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audiogs-contract", type=Path, required=True)
    parser.add_argument("--ftgspp-contract", type=Path, required=True)
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    args = parser.parse_args()
    result = prepare_architecture_run(
        base_config=args.base_config,
        derived_config=args.derived_config,
        base_protocol_dir=args.base_protocol_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        device=args.device,
        trusted_upstream_artifacts=args.trust_upstream_artifacts,
        native_contract_dirs={
            "audiogs": args.audiogs_contract,
            "ftgspp": args.ftgspp_contract,
        },
    )
    print(
        json.dumps(
            {
                "scene_id": result["scene_id"],
                "strategy": result["strategy"],
                "alignment": result["alignment"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

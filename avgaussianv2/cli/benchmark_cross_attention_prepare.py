"""Prepare one strict cross-attention cam38 worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.cross_attention_ablation import (
    prepare_cross_attention_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--derived-config", type=Path, required=True)
    parser.add_argument("--base-protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ftgspp-contract", type=Path, required=True)
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    args = parser.parse_args()
    result = prepare_cross_attention_run(
        base_config=args.base_config,
        derived_config=args.derived_config,
        base_protocol_dir=args.base_protocol_dir,
        output_dir=args.output_dir,
        device=args.device,
        trusted_upstream_artifacts=args.trust_upstream_artifacts,
        ftgspp_contract_dir=args.ftgspp_contract,
    )
    print(
        json.dumps(
            {
                "scene_id": result["scene_id"],
                "system": result["system"],
                "alignment": result["alignment"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

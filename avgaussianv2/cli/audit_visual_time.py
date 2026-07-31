"""Audit post-fix held-out visual-time semantics before formal experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.visual_time_audit import audit_visual_time_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    args = parser.parse_args()
    result = audit_visual_time_configs(
        args.config,
        output=args.output,
        device=args.device,
        trusted_upstream_artifacts=args.trust_upstream_artifacts,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scenes": [record["scene"] for record in result["scenes"]],
                "visual_time_semantics": result["visual_time_semantics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

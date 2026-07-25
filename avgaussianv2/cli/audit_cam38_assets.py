from __future__ import annotations

import argparse
import json

from avgaussianv2.benchmark.assets import (
    audit_audiogs_conversion,
    audit_ftgspp_train_source,
    audit_initialization_provenance,
    audit_protocol_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict cam38 benchmark assets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--provenance")
    parser.add_argument("--ftgspp-train-source")
    parser.add_argument("--audiogs-conversion")
    parser.add_argument("--expected-clips", type=int)
    args = parser.parse_args()
    raw = audit_protocol_config(args.config)
    if args.provenance:
        audit_initialization_provenance(args.provenance)
    details = {}
    if args.ftgspp_train_source:
        details["ftgspp"] = audit_ftgspp_train_source(args.ftgspp_train_source)
    if args.audiogs_conversion:
        if args.expected_clips is None:
            parser.error("--audiogs-conversion requires --expected-clips")
        details["audiogs"] = audit_audiogs_conversion(
            args.audiogs_conversion,
            expected_clips=args.expected_clips,
            epochs=61,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "scene_id": raw["scene"]["id"],
                "test_camera": "cam38",
                **details,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

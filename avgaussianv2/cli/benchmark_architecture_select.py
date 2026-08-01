"""Select at most two architecture-screening survivors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.architecture_selection import (
    select_architecture_winners,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-postprocessing-revision",
        action="store_true",
        help=(
            "allow a clean ancestor experiment only when every intervening change "
            "is restricted to the architecture selector/generator and their tests"
        ),
    )
    args = parser.parse_args()
    result = select_architecture_winners(
        args.manifest,
        args.run_root,
        args.output,
        allow_postprocessing_revision=args.allow_postprocessing_revision,
    )
    print(
        json.dumps(
            {
                "selected_systems": result["selected_systems"],
                "exploratory_systems": result["exploratory_systems"],
                "screening_systems": result["screening_systems"],
            }
        )
    )


if __name__ == "__main__":
    main()

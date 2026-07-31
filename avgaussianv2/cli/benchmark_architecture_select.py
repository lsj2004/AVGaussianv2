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
    args = parser.parse_args()
    result = select_architecture_winners(args.manifest, args.run_root, args.output)
    print(json.dumps({"selected_systems": result["selected_systems"]}))


if __name__ == "__main__":
    main()

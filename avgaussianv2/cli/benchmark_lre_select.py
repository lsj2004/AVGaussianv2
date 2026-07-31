from __future__ import annotations

import argparse
import json
from pathlib import Path

from avgaussianv2.benchmark.lre_selection import select_screening_winners


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_screening_winners(args.manifest, args.run_root, args.output)
    print(json.dumps({"selected_lambda_lre": result["selected_lambda_lre"]}))


if __name__ == "__main__":
    main()

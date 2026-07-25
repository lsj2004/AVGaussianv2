from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, argv: list[str]) -> dict[str, Any]:
    if seed != 42:
        raise ValueError("strict cam38 FTGS++ seed must be 42")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("PYTHONHASHSEED=42 must be set before interpreter startup")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "schema": "ftgspp_seed_v1",
        "seed": seed,
        "pythonhashseed": "42",
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "argv": argv,
    }


def _write_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed FTGS++ deterministically before executing upstream code"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--record", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--script", type=Path)
    target.add_argument("--module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    target_name = str(args.script) if args.script else str(args.module)
    argv = [target_name, *arguments]
    record = seed_everything(args.seed, argv)
    _write_record(args.record, record)
    sys.argv = argv
    if args.script:
        runpy.run_path(str(args.script), run_name="__main__")
    else:
        runpy.run_module(str(args.module), run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()

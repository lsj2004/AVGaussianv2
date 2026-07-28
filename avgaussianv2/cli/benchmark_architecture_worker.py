"""Run one strict B/C worker only under its pinned preparation revision."""

# ruff: noqa: E402 -- deterministic process settings precede Torch imports.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_MARKER = "AVGAUSSIANV2_ARCHITECTURE_WORKER_REEXEC"
_ENVIRONMENT = {
    "PYTHONHASHSEED": "42",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def _ensure_environment() -> None:
    if all(os.environ.get(name) == value for name, value in _ENVIRONMENT.items()):
        return
    if os.environ.get(_MARKER) == "1":
        raise RuntimeError("architecture worker deterministic re-exec failed")
    environment = dict(os.environ)
    environment.update(_ENVIRONMENT)
    environment[_MARKER] = "1"
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-m",
            "avgaussianv2.cli.benchmark_architecture_worker",
            *sys.argv[1:],
        ],
        environment,
    )


if __name__ == "__main__":
    _ensure_environment()

import torch

from avgaussianv2.benchmark.architecture_ablation import (
    verify_architecture_preparation,
)
from avgaussianv2.cli.benchmark_worker import run_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evidence = verify_architecture_preparation(args.protocol_dir)
    result = run_worker(
        manifest_path=args.protocol_dir / "worker_manifest.json",
        output_dir=args.output_dir,
        config_path=args.protocol_dir / "resolved_project.yaml",
        device=torch.device(args.device),
        trust_upstream_artifacts=args.trust_upstream_artifacts,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "scene_id": evidence["scene_id"],
                "strategy": evidence["strategy"],
                "mode": evidence.get("mode", "joint_conditioned"),
                "warmup_step": result["completed_warmup_steps"],
                "main_step": result["completed_main_updates"],
                "runtime_contract_sha256": result["runtime_contract_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

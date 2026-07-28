"""Worker-level CLI for one strict cam38 fixed-budget continuation."""

# ruff: noqa: E402 -- the executable bootstrap must run before third-party imports.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path


_REEXEC_MARKER = "AVGAUSSIANV2_BENCHMARK_ENV_REEXEC"
_REQUIRED_PROCESS_ENVIRONMENT = {
    "PYTHONHASHSEED": "42",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def _ensure_process_environment() -> None:
    mismatches = {
        name: os.environ.get(name)
        for name, expected in _REQUIRED_PROCESS_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if not mismatches:
        return
    if os.environ.get(_REEXEC_MARKER) == "1":
        raise RuntimeError(
            f"benchmark process environment remained invalid after re-exec: {mismatches}"
        )
    environment = dict(os.environ)
    environment.update(_REQUIRED_PROCESS_ENVIRONMENT)
    environment[_REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-m",
            "avgaussianv2.cli.benchmark_worker",
            *sys.argv[1:],
        ],
        environment,
    )


# When executed as a worker, enforce process-start settings before importing
# NumPy, Torch, or any runtime/upstream module. Imports remain side-effect free
# for unit-test injection.
if __name__ == "__main__":
    _ensure_process_environment()

import numpy as np
import torch

from avgaussianv2.benchmark.runtime import (
    BenchmarkRuntime,
    build_production_runtime,
)
from avgaussianv2.benchmark.output import (
    BenchmarkOutputLock,
    validate_output_children,
)
from avgaussianv2.benchmark.training import (
    BenchmarkCompatibility,
    BenchmarkConfig,
    BenchmarkMode,
    FixedBudgetTrainer,
)
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.experiment.evaluation import move_sample


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


RuntimeBuilder = Callable[..., BenchmarkRuntime]


class _DeviceSampleSequence(Sequence[AlignedAVSample]):
    def __init__(
        self, samples: Sequence[AlignedAVSample], device: torch.device
    ) -> None:
        self.samples = samples
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def records(self):
        return getattr(self.samples, "records", None)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                move_sample(sample, self.device) for sample in self.samples[index]
            ]
        return move_sample(self.samples[index], self.device)


def _atomic_runtime_contract(
    output: Path,
    *,
    runtime: BenchmarkRuntime,
    compatibility: BenchmarkCompatibility,
) -> str:
    payload = {
        "schema": "avgaussianv2.cam38-production-train-only-runtime",
        "version": 1,
        "include_eval": False,
        "train_cameras": list(compatibility.train_cameras),
        "test_camera": compatibility.test_camera,
        "dataset_identity_sha256": runtime.dataset_identity_sha256,
        "dataset_sample_ids_sha256": hashlib.sha256(
            json.dumps(
                list(runtime.dataset_sample_ids),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "config_sha256": runtime.config_sha256,
        "source_sha256": runtime.source_sha256,
        "visual_initialization_sha256": runtime.visual_initialization_sha256,
        "audio_initialization_sha256": runtime.audio_initialization_sha256,
        "model_initialization_sha256": runtime.model_initialization_sha256,
    }
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(
        prefix=".runtime_contract.", suffix=".tmp", dir=output
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "runtime_contract.json")
        directory = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def _seed_everything(seed: int) -> None:
    if seed != 42:
        raise ValueError("strict benchmark seed must be 42")
    mismatches = {
        name: os.environ.get(name)
        for name, expected in _REQUIRED_PROCESS_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"benchmark process environment is invalid: {mismatches}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_worker(
    *,
    manifest_path: Path,
    output_dir: Path,
    config_path: Path,
    device: torch.device | str,
    trust_upstream_artifacts: bool,
    resume: bool,
    _runtime_builder: RuntimeBuilder = build_production_runtime,
) -> dict[str, object]:
    raw = _load_json(manifest_path)
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema",
        "version",
        "mode",
        "training",
        "compatibility",
        "shared_indices",
    }:
        raise ValueError("benchmark worker manifest fields mismatch")
    if raw["schema"] != "avgaussianv2.cam38-benchmark-worker" or raw["version"] != 1:
        raise ValueError("unsupported benchmark worker manifest")
    mode = BenchmarkMode(raw["mode"])
    config = BenchmarkConfig.from_mapping(raw["training"])
    compatibility = BenchmarkCompatibility.from_mapping(raw["compatibility"])
    if compatibility.mode != mode.value:
        raise ValueError("manifest mode and compatibility mode differ")
    indices = tuple(raw["shared_indices"])
    original_output = Path(output_dir)
    with BenchmarkOutputLock(original_output) as pinned_output:
        validate_output_children(pinned_output)
        # Seed before importing/constructing any production model or dataset state.
        _seed_everything(config.seed)
        runtime = _runtime_builder(
            config_path=Path(config_path),
            device=torch.device(device),
            trusted_upstream_artifacts=trust_upstream_artifacts,
        )
        if not isinstance(runtime, BenchmarkRuntime):
            raise TypeError("internal runtime builder must return BenchmarkRuntime")
        with runtime:
            expected_identity = {
                "config_sha256": compatibility.config_sha256,
                "source_sha256": compatibility.source_sha256,
                "visual_initialization_sha256": (
                    compatibility.visual_initialization_sha256
                ),
                "audio_initialization_sha256": (
                    compatibility.audio_initialization_sha256
                ),
                "model_initialization_sha256": (
                    compatibility.model_initialization_sha256
                ),
            }
            for name, expected in expected_identity.items():
                if getattr(runtime, name) != expected:
                    raise ValueError(f"runtime {name} mismatch")
            if len(runtime.train_samples) <= 0:
                raise ValueError("runtime training dataset is empty")
            if len(runtime.dataset_sample_ids) != len(runtime.train_samples):
                raise ValueError("runtime dataset sample-ID sequence length mismatch")
            if any(
                index < 0 or index >= len(runtime.train_samples) for index in indices
            ):
                raise ValueError("shared sample index is outside the training dataset")
            training_samples = _DeviceSampleSequence(
                runtime.train_samples, torch.device(device)
            )
            result = FixedBudgetTrainer(config).run(
                model=runtime.model,
                train_samples=training_samples,
                shared_indices=indices,
                mode=mode,
                train_config=runtime.train_config,
                audio_loss_fn=runtime.audio_loss_fn,
                output_dir=pinned_output,
                compatibility=compatibility,
                resume=resume,
            )
            runtime_contract_sha256 = _atomic_runtime_contract(
                pinned_output,
                runtime=runtime,
                compatibility=compatibility,
            )
            return {
                "mode": result.mode.value,
                "completed_warmup_steps": result.completed_warmup_steps,
                "completed_main_updates": result.completed_main_updates,
                "resumed_from_main_step": result.resumed_from_main_step,
                "redone_main_updates": result.redone_main_updates,
                "selection": result.selection,
                "final_checkpoint": str(original_output / "final.pt"),
                "milestones": [
                    str(original_output / "milestones" / path.name)
                    for path in result.milestones
                ],
                "io": asdict(result.io),
                "runtime_contract": str(original_output / "runtime_contract.json"),
                "runtime_contract_sha256": runtime_contract_sha256,
            }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--environment-probe":
        print(
            json.dumps(
                {
                    "hash": hash(sys.argv[2]),
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    "cublas_workspace_config": os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    "reexec_marker": os.environ.get(_REEXEC_MARKER),
                },
                sort_keys=True,
            )
        )
        return
    parser = argparse.ArgumentParser(
        description="Train one strict cam38 fixed-budget benchmark continuation"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_worker(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                config_path=args.config,
                device=args.device,
                trust_upstream_artifacts=args.trust_upstream_artifacts,
                resume=args.resume,
            ),
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

"""Run a bounded real-data diagnostic without mutating strict training state."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from avgaussianv2.benchmark.cross_attention_ablation import (
    verify_cross_attention_preparation,
)
from avgaussianv2.benchmark.runtime import BenchmarkRuntime, build_production_runtime
from avgaussianv2.benchmark.training import make_shared_indices
from avgaussianv2.cli.benchmark_worker import _DeviceSampleSequence
from avgaussianv2.losses import capture_visual_anchor
from avgaussianv2.train import (
    build_joint_optimizer,
    build_warmup_optimizer,
    condition_warmup_step,
    joint_train_step,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(
            json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stats(rows) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]


def run_diagnostic(
    *,
    protocol_dir: Path,
    output: Path,
    device: torch.device,
    steps: int,
    trusted_upstream_artifacts: bool,
) -> dict[str, object]:
    if steps <= 0:
        raise ValueError("diagnostic steps must be positive")
    preparation = verify_cross_attention_preparation(protocol_dir)
    worker_manifest = json.loads(
        (protocol_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    seed = int(worker_manifest["training"]["seed"])
    _seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    runtime = build_production_runtime(
        config_path=protocol_dir / "resolved_project.yaml",
        device=device,
        trusted_upstream_artifacts=trusted_upstream_artifacts,
    )
    if not isinstance(runtime, BenchmarkRuntime):
        raise TypeError("diagnostic runtime must be BenchmarkRuntime")
    with runtime:
        samples = _DeviceSampleSequence(runtime.train_samples, device)
        warmup_indices = make_shared_indices(len(samples), steps, seed)
        main_indices = tuple(worker_manifest["shared_indices"][:steps])

        runtime.model.freeze_pretrained()
        warmup_optimizer = build_warmup_optimizer(
            runtime.model,
            runtime.train_config.condition_lr,
        )
        warmup = [
            condition_warmup_step(
                runtime.model,
                samples[index],
                warmup_optimizer,
                runtime.audio_loss_fn,
            )
            for index in warmup_indices
        ]

        runtime.model.unfreeze_all()
        joint_optimizer = build_joint_optimizer(runtime.model, runtime.train_config)
        visual_anchor = capture_visual_anchor(runtime.model.visual)
        joint = [
            joint_train_step(
                runtime.model,
                samples[index],
                joint_optimizer,
                runtime.train_config,
                runtime.audio_loss_fn,
                visual_anchor,
                probe_audio_visual_gradient=True,
            )
            for index in main_indices
        ]

        probe = samples[main_indices[-1]]
        runtime.model.eval()
        with torch.no_grad():
            runtime.model.condition_enabled = True
            conditioned = runtime.model(probe).predicted_audio
            runtime.model.condition_enabled = False
            native = runtime.model(probe).predicted_audio
            runtime.model.condition_enabled = True
        difference = (conditioned - native).abs()
        result = {
            "schema": "avgaussianv2.cross-attention-diagnostic",
            "version": 1,
            "scene_id": preparation["scene_id"],
            "repository": preparation["repository"],
            "steps_per_stage": steps,
            "warmup": _stats(warmup),
            "joint": _stats(joint),
            "condition_effect": {
                "mean_absolute": float(difference.mean().cpu()),
                "max_absolute": float(difference.max().cpu()),
            },
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.type == "cuda"
                else 0
            ),
        }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--trust-upstream-artifacts", action="store_true")
    args = parser.parse_args()
    result = run_diagnostic(
        protocol_dir=args.protocol_dir,
        output=args.output,
        device=torch.device(args.device),
        steps=args.steps,
        trusted_upstream_artifacts=args.trust_upstream_artifacts,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

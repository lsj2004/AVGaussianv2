import copy
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from avgaussianv2.benchmark.architecture_ablation import (
    build_aligned_worker_manifest,
    common_model_initialization_sha256,
    validate_strategy_only_delta,
)
from avgaussianv2.benchmark.training import (
    BenchmarkCompatibility,
    BenchmarkConfig,
    hash_shared_indices,
    make_shared_indices,
)


def _config(strategy: str | None = None) -> dict:
    value = {
        "scene": {
            "id": "scene1_opera",
            "fps": 30.0,
            "train_cameras": ["cam00"],
            "eval_cameras": ["cam38"],
            "camera_mapping": {"cam00": 0, "cam38": 38},
        },
        "paths": {
            "visual_upstream_root": "/visual",
            "audio_upstream_root": "/audio",
            "visual_checkpoint": "/visual.pt",
            "audio_checkpoint": "/audio.pt",
            "manifest": "/manifest.json",
        },
        "model": {"audio_model_class": "Audio3DGSMonoDiffGSOnly"},
        "train": {"seed": 42},
        "benchmark": {"protocol": "dual_dataset_cam38_v1"},
    }
    if strategy is not None:
        value["model"]["audio_render_strategy"] = strategy
    return value


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_strategy_delta_rejects_every_other_protocol_change(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    derived = tmp_path / "derived.yaml"
    _write_yaml(base, _config())
    _write_yaml(derived, _config("direct_conditioned_unet"))

    result = validate_strategy_only_delta(
        base, derived, expected_strategy="direct_conditioned_unet"
    )
    assert result["strategy"] == "direct_conditioned_unet"

    changed = _config("direct_conditioned_unet")
    changed["train"]["seed"] = 7
    _write_yaml(derived, changed)
    with pytest.raises(ValueError, match="only audio_render_strategy"):
        validate_strategy_only_delta(
            base, derived, expected_strategy="direct_conditioned_unet"
        )


class TinyState(nn.Module):
    def __init__(self, *, gated: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        if gated:
            self.residual_gate_logit = nn.Parameter(torch.zeros(()))


def test_common_initialization_hash_excludes_only_the_gate() -> None:
    base = TinyState(gated=False)
    gated = TinyState(gated=True)

    assert common_model_initialization_sha256(base) == (
        common_model_initialization_sha256(gated)
    )

    gated.weight.data.add_(1)
    assert common_model_initialization_sha256(base) != (
        common_model_initialization_sha256(gated)
    )


def test_aligned_manifest_reuses_exact_a_sample_sequence() -> None:
    config = BenchmarkConfig()
    indices = make_shared_indices(17, config.main_updates, config.seed)
    base = {
        "schema": "avgaussianv2.cam38-benchmark-worker",
        "version": 1,
        "mode": "joint_conditioned",
        "training": {
            "main_updates": 30_000,
            "conditioner_warmup_steps": 2_000,
            "checkpoint_every": 500,
            "journal_every": 10,
            "milestones": [5_000, 10_000, 30_000],
            "seed": 42,
            "batch_size": 1,
            "selection": "final",
        },
        "compatibility": {
            "scene_id": "scene1_opera",
            "mode": "joint_conditioned",
            "train_cameras": [f"cam{index:02d}" for index in range(38)],
            "test_camera": "cam38",
            "seed": 42,
            "index_sha256": hash_shared_indices(indices),
            "visual_initialization_sha256": "1" * 64,
            "audio_initialization_sha256": "2" * 64,
            "model_initialization_sha256": "3" * 64,
            "source_sha256": "4" * 64,
            "config_sha256": "5" * 64,
        },
        "shared_indices": list(indices),
    }
    compatibility = BenchmarkCompatibility(
        scene_id="scene1_opera",
        mode="joint_conditioned",
        train_cameras=tuple(f"cam{index:02d}" for index in range(38)),
        test_camera="cam38",
        seed=42,
        index_sha256=hash_shared_indices(indices),
        visual_initialization_sha256="a" * 64,
        audio_initialization_sha256="b" * 64,
        model_initialization_sha256="c" * 64,
        source_sha256="d" * 64,
        config_sha256="e" * 64,
    )

    manifest = build_aligned_worker_manifest(
        base_manifest=copy.deepcopy(base),
        compatibility=compatibility,
        config=config,
    )

    assert manifest["shared_indices"] == list(indices)
    assert manifest["compatibility"] == compatibility.to_mapping()
    assert manifest["mode"] == "joint_conditioned"
    assert json.dumps(manifest, sort_keys=True)


def test_architecture_worker_prints_generic_worker_result_keys() -> None:
    worker = (
        Path(__file__).resolve().parents[1]
        / "avgaussianv2"
        / "cli"
        / "benchmark_architecture_worker.py"
    ).read_text()

    assert 'result["completed_warmup_steps"]' in worker
    assert 'result["completed_main_updates"]' in worker
    assert 'result["warmup_step"]' not in worker
    assert 'result["main_step"]' not in worker

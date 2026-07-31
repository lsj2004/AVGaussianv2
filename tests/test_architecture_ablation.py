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
    BenchmarkMode,
    build_worker_manifest,
    hash_shared_indices,
    make_shared_indices,
)
from avgaussianv2.benchmark.evaluation import _training_mode_for_evaluation_system


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


@pytest.mark.parametrize(
    ("base_name", "derived_name"),
    [
        ("scene1_opera.yaml", "scene1_opera_plain_unet.yaml"),
        ("Scene7playing.yaml", "Scene7playing_plain_unet.yaml"),
    ],
)
def test_plain_unet_configs_are_strategy_only_deltas(
    base_name: str,
    derived_name: str,
) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs/benchmark_cam38"

    result = validate_strategy_only_delta(
        config_dir / base_name,
        config_dir / derived_name,
        expected_strategy="plain_unet",
    )

    assert result["strategy"] == "plain_unet"


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


def test_aligned_manifest_supports_plain_unet_audio_only_mode() -> None:
    config = BenchmarkConfig()
    indices = make_shared_indices(17, config.main_updates, config.seed)
    compatibility = BenchmarkCompatibility(
        scene_id="scene1_opera",
        mode=BenchmarkMode.AUDIO_ONLY.value,
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
    base = build_worker_manifest(
        config=config,
        compatibility=compatibility,
        shared_indices=indices,
    )

    manifest = build_aligned_worker_manifest(
        base_manifest=base,
        compatibility=compatibility,
        config=config,
        mode=BenchmarkMode.AUDIO_ONLY,
    )

    assert manifest["mode"] == BenchmarkMode.AUDIO_ONLY.value
    assert manifest["shared_indices"] == list(indices)
    assert _training_mode_for_evaluation_system("plain_unet") == "audio_only"


def test_architecture_worker_prints_generic_worker_result_keys() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "avgaussianv2"
        / "cli"
        / "ablation_runner.py"
    ).read_text()

    assert 'result["completed_warmup_steps"]' in runner
    assert 'result["completed_main_updates"]' in runner
    assert 'result["warmup_step"]' not in runner
    assert 'result["main_step"]' not in runner


def test_architecture_cli_delegates_to_shared_ablation_runner() -> None:
    cli_dir = Path(__file__).resolve().parents[1] / "avgaussianv2" / "cli"
    worker = (cli_dir / "benchmark_architecture_worker.py").read_text()
    evaluator = (cli_dir / "benchmark_architecture_eval.py").read_text()

    assert "run_ablation_worker(" in worker
    assert 'result_label="strategy"' in worker
    assert "run_ablation_evaluation(" in evaluator
    assert 'system_from_preparation="evaluation_system"' in evaluator

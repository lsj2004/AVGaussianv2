from pathlib import Path

import pytest
import yaml

from avgaussianv2.benchmark.cross_attention_ablation import (
    ALL_CAUSAL_EVALUATION_SYSTEMS,
    MASK_CAUSAL_EVALUATION_SYSTEMS,
    P1_CAUSAL_EVALUATION_SYSTEMS,
    validate_backend_only_delta,
)
from avgaussianv2.benchmark.evaluation import EVALUATION_CONTINUATION_SYSTEMS
from avgaussianv2.benchmark.evaluation import _training_mode_for_evaluation_system
from avgaussianv2.config import load_project_config


ROOT = Path(__file__).resolve().parents[1]


def _config(audio_backend: str | None = None) -> dict:
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
        "model": {"embedding_dim": 128},
        "train": {"seed": 42},
        "benchmark": {"protocol": "dual_dataset_cam38_v1"},
    }
    if audio_backend is not None:
        value["model"]["audio_backend"] = audio_backend
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_cross_attention_delta_rejects_every_other_change(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    derived = tmp_path / "derived.yaml"
    _write(base, _config())
    _write(derived, _config("cross_attention_tokens"))

    result = validate_backend_only_delta(base, derived)
    assert len(result["base_config_sha256"]) == 64

    changed = _config("cross_attention_tokens")
    changed["train"]["seed"] = 7
    _write(derived, changed)
    with pytest.raises(ValueError, match="only model.audio_backend"):
        validate_backend_only_delta(base, derived)


@pytest.mark.parametrize(
    ("base_name", "derived_name", "backend"),
    [
        (
            "scene1_opera.yaml",
            "scene1_opera_cross_attention.yaml",
            "cross_attention_tokens",
        ),
        (
            "Scene7playing.yaml",
            "Scene7playing_cross_attention.yaml",
            "cross_attention_tokens",
        ),
        (
            "scene1_opera.yaml",
            "scene1_opera_cross_attention_masks.yaml",
            "cross_attention_masks",
        ),
        (
            "Scene7playing.yaml",
            "Scene7playing_cross_attention_masks.yaml",
            "cross_attention_masks",
        ),
        (
            "scene1_opera.yaml",
            "scene1_opera_query_dependent_p1.yaml",
            "query_dependent_p1",
        ),
        (
            "Scene7playing.yaml",
            "Scene7playing_query_dependent_p1.yaml",
            "query_dependent_p1",
        ),
    ],
)
def test_repository_cross_configs_are_backend_only_deltas(
    base_name: str,
    derived_name: str,
    backend: str,
) -> None:
    config_dir = ROOT / "configs" / "benchmark_cam38"
    validate_backend_only_delta(
        config_dir / base_name,
        config_dir / derived_name,
        expected_backend=backend,
    )
    config = load_project_config(config_dir / derived_name)
    assert config.model.audio_backend == backend
    assert config.model.audio_render_strategy == "native_residual"


def test_causal_evaluation_systems_are_registered_continuations() -> None:
    assert set(ALL_CAUSAL_EVALUATION_SYSTEMS) <= EVALUATION_CONTINUATION_SYSTEMS
    assert all(
        _training_mode_for_evaluation_system(system) == "joint_conditioned"
        for system in ALL_CAUSAL_EVALUATION_SYSTEMS
    )


@pytest.mark.parametrize(
    ("backend", "systems"),
    [
        ("cross_attention_masks", MASK_CAUSAL_EVALUATION_SYSTEMS),
        ("query_dependent_p1", P1_CAUSAL_EVALUATION_SYSTEMS),
    ],
)
def test_unified_cross_attention_backends_are_backend_only_deltas(
    tmp_path: Path,
    backend: str,
    systems: tuple[str, ...],
) -> None:
    base = tmp_path / "base.yaml"
    derived = tmp_path / "derived.yaml"
    _write(base, _config())
    _write(derived, _config(backend))

    result = validate_backend_only_delta(
        base,
        derived,
        expected_backend=backend,
    )

    assert result["audio_backend"] == backend
    assert set(systems) <= EVALUATION_CONTINUATION_SYSTEMS


def test_cross_attention_cli_delegates_to_shared_ablation_runner() -> None:
    cli_dir = ROOT / "avgaussianv2" / "cli"
    worker = (cli_dir / "benchmark_cross_attention_worker.py").read_text()
    evaluator = (cli_dir / "benchmark_cross_attention_eval.py").read_text()

    assert "run_ablation_worker(" in worker
    assert 'result_label="system"' in worker
    assert "run_ablation_evaluation(" in evaluator
    assert "system_choices=ALL_CAUSAL_EVALUATION_SYSTEMS" in evaluator
    assert 'allowed_systems_from_preparation="causal_evaluation_systems"' in evaluator
    assert '"require_repository_match": False' in evaluator

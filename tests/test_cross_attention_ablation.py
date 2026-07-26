from pathlib import Path

import pytest
import yaml

from avgaussianv2.benchmark.cross_attention_ablation import (
    CAUSAL_EVALUATION_SYSTEMS,
    validate_backend_only_delta,
)
from avgaussianv2.benchmark.evaluation import EVALUATION_CONTINUATION_SYSTEMS
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
    ("base_name", "derived_name"),
    [
        ("scene1_opera.yaml", "scene1_opera_cross_attention.yaml"),
        ("Scene7playing.yaml", "Scene7playing_cross_attention.yaml"),
    ],
)
def test_repository_cross_configs_are_backend_only_deltas(
    base_name: str,
    derived_name: str,
) -> None:
    config_dir = ROOT / "configs" / "benchmark_cam38"
    validate_backend_only_delta(
        config_dir / base_name,
        config_dir / derived_name,
    )
    config = load_project_config(config_dir / derived_name)
    assert config.model.audio_backend == "cross_attention_tokens"
    assert config.model.audio_render_strategy == "native_residual"


def test_causal_evaluation_systems_are_registered_continuations() -> None:
    assert set(CAUSAL_EVALUATION_SYSTEMS) <= EVALUATION_CONTINUATION_SYSTEMS

from pathlib import Path

from avgaussianv2.config import load_project_config


ROOT = Path(__file__).resolve().parents[1]


def test_real_smoke_scripts_are_explicit_and_fail_fast() -> None:
    expected = {
        "smoke_scene1_opera.sh": ("configs/scene1_opera.yaml", "smoke_scene1_opera"),
        "smoke_Scene7playing.sh": ("configs/Scene7playing.yaml", "smoke_Scene7playing"),
    }
    for name, (config, output) in expected.items():
        path = ROOT / "scripts" / name
        text = path.read_text()
        assert "set -euo pipefail" in text
        assert config in text
        assert output in text
        assert "--stage all" in text
        assert "--warmup-steps 1" in text
        assert "--joint-steps 1" in text
        assert "verify_smoke.py" in text


def test_readme_documents_required_training_contracts() -> None:
    text = (ROOT / "README.md").read_text()
    for phrase in (
        "independent Gaussian",
        "FreeTimeGS++",
        "AudioGS",
        "condition warmup",
        "joint fine-tuning",
        "condition-off",
        "scene1_opera",
        "Scene7playing",
        "checkpoint_latest.pt",
    ):
        assert phrase in text


def test_real_scene_configs_match_checkpoint_model_family() -> None:
    for name in ("scene1_opera.yaml", "Scene7playing.yaml"):
        config = load_project_config(ROOT / "configs" / name)
        assert config.model.audio_model_class == "Audio3DGSMonoDiffGSOnly"

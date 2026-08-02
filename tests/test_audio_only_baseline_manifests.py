from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from avgaussianv2.benchmark.lre_orchestration import load_lre_run_manifest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_audio_only_baseline_manifests.py"
SCENES = ("scene1_opera", "Scene7playing")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "audio_only_baseline_generator", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    configs = []
    runs = []
    for scene in SCENES:
        config_id = f"audio_only__{scene}__seed42__lre0000"
        config_path = tmp_path / "configs" / f"{config_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "paths": {"audio_upstream_root": "/fixed/audio"},
            "train": {
                "seed": 42,
                "lambda_lre": 0.0,
                "joint_steps": 30_000,
            },
            "benchmark": {
                "seed": 42,
                "continuation_updates": 30_000,
                "report_steps": [5_000, 10_000, 30_000],
            },
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs.append(
            {
                "config_id": config_id,
                "system": "audio_only",
                "training_mode": "audio_only",
                "scene": scene,
                "seed": 42,
                "lambda_lre": 0.0,
                "base_config": str(tmp_path / f"{scene}.yaml"),
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
            }
        )
        runs.append(
            {
                "run_id": f"screening__{config_id}",
                "continuation_id": config_id,
                "stage": "screening",
                "config_id": config_id,
                "scene": scene,
                "system": "audio_only",
                "training_mode": "audio_only",
                "seed": 42,
                "lambda_lre": 0.0,
                "control_run_id": None,
                "evaluation_systems": ["audio_only"],
                "report_steps": [5_000],
                "max_steps": 5_000,
                "stop_after_step": 5_000,
            }
        )
    manifest = {
        "schema": "avgaussianv2.lre-loss-run-manifest",
        "version": 1,
        "stage": "screening",
        "repository": {
            "root": str(tmp_path.resolve()),
            "commit": "1" * 40,
            "clean": True,
        },
        "configs": configs,
        "runs": runs,
    }
    path = tmp_path / "screening" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def test_generates_seed42_continuation_and_two_robustness_seeds(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path)
    output = tmp_path / "baseline"

    index = _load_generator().generate(source, output)

    assert index["repository"]["commit"] == "1" * 40
    confirmation = load_lre_run_manifest(output / "confirmation/manifest.json")
    robustness = load_lre_run_manifest(output / "robustness/manifest.json")
    assert len(confirmation["runs"]) == 2
    assert {run["seed"] for run in confirmation["runs"]} == {42}
    assert all(
        run["report_steps"] == [5_000, 10_000, 30_000] for run in confirmation["runs"]
    )
    assert len(robustness["runs"]) == 4
    assert {run["seed"] for run in robustness["runs"]} == {17, 73}
    assert all(run["report_steps"] == [30_000] for run in robustness["runs"])


def test_seed42_reuses_exact_config_and_new_seeds_change_only_seed(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path)
    source_value = json.loads(source.read_text())
    source_paths = {
        record["scene"]: Path(record["config"]) for record in source_value["configs"]
    }

    output = tmp_path / "baseline"
    _load_generator().generate(source, output)
    confirmation = json.loads((output / "confirmation/manifest.json").read_text())
    robustness = json.loads((output / "robustness/manifest.json").read_text())

    assert {Path(record["config"]) for record in confirmation["configs"]} == set(
        source_paths.values()
    )
    for record in robustness["configs"]:
        source_config = yaml.safe_load(source_paths[record["scene"]].read_text())
        derived = yaml.safe_load(Path(record["config"]).read_text())
        assert derived["train"]["seed"] == record["seed"]
        assert derived["benchmark"]["seed"] == record["seed"]
        derived["train"]["seed"] = 42
        derived["benchmark"]["seed"] = 42
        assert derived == source_config


def test_generation_is_byte_deterministic(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    output = tmp_path / "baseline"
    module = _load_generator()

    first = module.generate(source, output)
    first_bytes = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*.json")
    }
    second = module.generate(source, output)

    assert first == second
    assert first_bytes == {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*.json")
    }


def test_rejects_tampered_source_config(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    manifest = json.loads(source.read_text())
    Path(manifest["configs"][0]["config"]).write_text("tampered: true\n")

    with pytest.raises(ValueError, match="source config hash mismatch"):
        _load_generator().generate(source, tmp_path / "baseline")

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from avgaussianv2.benchmark.lre_selection import select_screening_winners


ROOT = Path(__file__).resolve().parents[1]
FAKE_REPOSITORY = {"root": str(ROOT), "commit": "1" * 40, "clean": True}


def _generator():
    script = ROOT / "scripts/generate_lre_ablation_configs.py"
    specification = importlib.util.spec_from_file_location("lre_generator", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _generate(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    module = _generator()
    value = module.generate(
        ROOT / "configs/experiments/lre_loss_ablation.yaml",
        tmp_path / "generated",
        systems=("audio_only", "joint_conditioned"),
        _repository_identity=lambda: FAKE_REPOSITORY,
    )
    return tmp_path / "generated/screening/manifest.json", value


def test_screening_selector_applies_all_unit_gates_and_ranks_survivors(tmp_path):
    manifest_path, manifest = _generate(tmp_path)
    run_root = tmp_path / "runs"
    by_path = {}
    improvements = {0.0: 0.0, 0.01: 0.16, 0.02: 0.25, 0.05: 0.30}
    for run in manifest["runs"]:
        weight = float(run["lambda_lre"])
        audio_degradation = 0.04 if weight == 0.05 else 0.02
        identity = SimpleNamespace(
            scene_id=run["scene"],
            system_name=run["system"],
            reporting_step=5_000,
            expected_sample_ids=(f"{run['scene']}/cam38/000000",),
        )
        by_path[
            (
                run_root
                / run["continuation_id"]
                / "evaluations/step_005000"
            ).resolve()
        ] = SimpleNamespace(
            identity=identity,
            summary={
                "lre_error_db": {"mean": 1.0 - improvements[weight]},
                "audio_total": {"mean": 1.0 + audio_degradation if weight else 1.0},
                "waveform_l1": {"mean": 0.102 if weight else 0.1},
            },
        )

    output = tmp_path / "winners.json"
    selected = select_screening_winners(
        manifest_path,
        run_root,
        output,
        evaluation_loader=lambda path: by_path[path.resolve()],
    )

    assert selected["selected_lambda_lre"] == [0.02, 0.01]
    assert json.loads(output.read_text()) == selected
    rejected = next(
        record for record in selected["candidates"] if record["lambda_lre"] == 0.05
    )
    assert rejected["passed"] is False
    confirmation = _generator().generate(
        ROOT / "configs/experiments/lre_loss_ablation.yaml",
        tmp_path / "generated",
        stage="confirmation",
        winners_path=output,
        systems=("audio_only", "joint_conditioned"),
        _repository_identity=lambda: FAKE_REPOSITORY,
    )
    assert {record["lambda_lre"] for record in confirmation["runs"]} == {
        0.0,
        0.01,
        0.02,
    }

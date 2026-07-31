import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_lre_ablation_configs.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("lre_ablation_generator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> Path:
    return ROOT / "configs/experiments/lre_loss_ablation.yaml"


def _generate(module, output: Path, **kwargs):
    systems = kwargs.pop("systems", ("audio_only", "joint_conditioned"))
    return module.generate(
        _manifest(),
        output,
        systems=systems,
        **kwargs,
    )


def test_screening_generation_only_expands_screening_seed(tmp_path: Path) -> None:
    generated = _generate(_load_generator(), tmp_path)

    assert generated["stage"] == "screening"
    assert len(generated["configs"]) == 16
    assert len(generated["runs"]) == 16
    assert {record["seed"] for record in generated["runs"]} == {42}
    assert {record["lambda_lre"] for record in generated["runs"]} == {
        0.0,
        0.01,
        0.02,
        0.05,
    }
    assert {record["system"] for record in generated["runs"]} == {
        "audio_only",
        "joint_conditioned",
    }
    assert {tuple(record["report_steps"]) for record in generated["runs"]} == {
        (5_000,)
    }
    assert all(record["max_steps"] == 5_000 for record in generated["runs"])

    scene1 = next(
        record
        for record in generated["configs"]
        if record["scene"] == "scene1_opera"
        and record["lambda_lre"] == pytest.approx(0.02)
    )
    config = yaml.safe_load(Path(scene1["config"]).read_text())
    assert Path(config["paths"]["visual_upstream_root"]).is_absolute()
    assert Path(config["paths"]["audio_upstream_root"]).is_absolute()
    assert Path(config["paths"]["visual_memmap"]).is_absolute()
    assert config["train"]["joint_steps"] == 30_000
    assert config["benchmark"]["continuation_updates"] == 30_000
    assert config["benchmark"]["report_steps"] == [5_000, 10_000, 30_000]
    assert all(record["stop_after_step"] == 5_000 for record in generated["runs"])
    assert (
        Path(scene1["config"]).parent / config["paths"]["visual_checkpoint"]
    ).resolve() == (
        ROOT / "runs/cam38_strict/scene1_opera/ftgspp/native/"
        "scene1_opera/00/gaussians.pt"
    ).resolve()


def test_smoke_generation_is_strict_isolated_and_nonzero(tmp_path: Path) -> None:
    generated = _generate(
        _load_generator(),
        tmp_path,
        stage="smoke",
        systems=("query_dependent_p1",),
    )

    assert len(generated["runs"]) == 1
    run = generated["runs"][0]
    assert run["scene"] == "scene1_opera"
    assert run["training_mode"] == "joint_conditioned"
    assert run["lambda_lre"] == pytest.approx(0.02)
    assert run["report_steps"] == [5_000]
    assert run["stop_after_step"] == 5_000
    assert run["continuation_id"].startswith("smoke__")
    config = yaml.safe_load(Path(generated["configs"][0]["config"]).read_text())
    assert config["train"]["warmup_steps"] == 2_000
    assert config["train"]["joint_steps"] == 30_000
    assert config["benchmark"]["conditioner_warmup_steps"] == 2_000
    assert config["benchmark"]["continuation_updates"] == 30_000
    assert config["benchmark"]["report_steps"] == [5_000, 10_000, 30_000]


def test_architecture_generation_is_fair_and_schedules_causal_checks(
    tmp_path: Path,
) -> None:
    systems = (
        "audio_only",
        "plain_unet",
        "joint_conditioned",
        "cross_attention_masks",
        "query_dependent_p1",
    )
    generated = _generate(
        _load_generator(), tmp_path, stage="architecture", systems=systems
    )

    assert len(generated["runs"]) == 10
    assert {record["seed"] for record in generated["runs"]} == {42}
    assert {record["lambda_lre"] for record in generated["runs"]} == {0.0}
    assert all(record["report_steps"] == [5_000] for record in generated["runs"])
    by_system = {
        record["system"]: tuple(record["evaluation_systems"])
        for record in generated["runs"]
        if record["scene"] == "scene1_opera"
    }
    assert by_system["audio_only"] == ("audio_only",)
    assert by_system["joint_conditioned"] == (
        "joint_conditioned",
        "joint_conditioned_no_rgbd",
        "joint_conditioned_wrong_camera",
    )
    assert by_system["cross_attention_masks"] == (
        "cross_attention_masks",
        "cross_attention_masks_no_rgbd",
        "cross_attention_masks_shuffled_rgbd",
    )
    assert by_system["query_dependent_p1"] == (
        "query_dependent_p1",
        "query_dependent_p1_no_rgbd",
        "query_dependent_p1_wrong_camera",
    )


def test_architecture_generation_rejects_non_axis_config_delta(
    tmp_path: Path,
) -> None:
    module = _load_generator()
    manifest = yaml.safe_load(_manifest().read_text())
    source = ROOT / "configs/benchmark_cam38/scene1_opera_plain_unet.yaml"
    invalid = tmp_path / "invalid.yaml"
    config = yaml.safe_load(source.read_text())
    config["train"]["seed"] = 73
    invalid.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest["architecture_configs"]["audio_only"]["scene1_opera"] = str(
        ROOT / "configs/benchmark_cam38/scene1_opera.yaml"
    )
    manifest["architecture_configs"]["plain_unet"]["scene1_opera"] = str(invalid)
    local_manifest = tmp_path / "manifest.yaml"
    local_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="only audio_render_strategy"):
        module.generate(
            local_manifest,
            tmp_path / "generated",
            stage="architecture",
            systems=("plain_unet",),
        )


def test_confirmation_requires_winners_bound_to_screening_manifest(
    tmp_path: Path,
) -> None:
    module = _load_generator()

    with pytest.raises(ValueError, match="requires --winners"):
        _generate(module, tmp_path, stage="confirmation")

    screening = _generate(module, tmp_path, stage="screening")
    screening_manifest = tmp_path / "screening/manifest.json"
    winners = tmp_path / "winners.json"
    winners.write_text(
        json.dumps(
            {
                "schema": "avgaussianv2.lre-loss-screening-selection",
                "version": 1,
                "source_screening_manifest_sha256": hashlib.sha256(
                    screening_manifest.read_bytes()
                ).hexdigest(),
                "selected_lambda_lre": [0.01, 0.02],
            }
        )
    )

    generated = _generate(
        module,
        tmp_path,
        stage="confirmation",
        winners_path=winners,
    )

    assert len(generated["configs"]) == 12
    assert len(generated["runs"]) == 12
    assert {record["seed"] for record in generated["runs"]} == {42}
    assert {record["lambda_lre"] for record in generated["runs"]} == {
        0.0,
        0.01,
        0.02,
    }
    assert {tuple(record["report_steps"]) for record in generated["runs"]} == {
        (5_000, 10_000, 30_000)
    }
    assert all(record["max_steps"] == 30_000 for record in generated["runs"])
    assert all(
        record["control_run_id"] is None
        for record in generated["runs"]
        if record["lambda_lre"] == 0.0
    )
    assert all(
        record["control_run_id"] is not None
        for record in generated["runs"]
        if record["lambda_lre"] != 0.0
    )
    screening_configs = {
        record["config_id"]: (record["config"], record["config_sha256"])
        for record in screening["configs"]
    }
    assert all(
        screening_configs[record["config_id"]]
        == (record["config"], record["config_sha256"])
        for record in generated["configs"]
    )
    assert all(
        record["continuation_id"] == record["config_id"]
        for record in generated["runs"]
    )


def test_generation_can_bind_configs_to_external_strict_asset_root(tmp_path: Path) -> None:
    external = tmp_path / "external-strict-runs"
    generated = _generate(
        _load_generator(),
        tmp_path / "generated",
        strict_run_root=external,
    )
    scene1 = next(
        record
        for record in generated["configs"]
        if record["scene"] == "scene1_opera"
        and record["system"] == "audio_only"
        and record["lambda_lre"] == 0.0
    )
    config = yaml.safe_load(Path(scene1["config"]).read_text())
    checkpoint = (
        Path(scene1["config"]).parent / config["paths"]["visual_checkpoint"]
    ).resolve()
    assert checkpoint.is_relative_to(external.resolve())


@pytest.mark.parametrize(
    ("selected", "expected_configs", "expected_runs", "expected_weights"),
    [
        ([], 4, 4, {0.0}),
        ([0.02], 8, 8, {0.0, 0.02}),
    ],
)
def test_confirmation_supports_dynamic_survivor_count(
    tmp_path: Path,
    selected: list[float],
    expected_configs: int,
    expected_runs: int,
    expected_weights: set[float],
) -> None:
    module = _load_generator()
    _generate(module, tmp_path, stage="screening")
    screening_manifest = tmp_path / "screening/manifest.json"
    winners = tmp_path / "winners.json"
    winners.write_text(
        json.dumps(
            {
                "schema": "avgaussianv2.lre-loss-screening-selection",
                "version": 1,
                "source_screening_manifest_sha256": hashlib.sha256(
                    screening_manifest.read_bytes()
                ).hexdigest(),
                "selected_lambda_lre": selected,
            }
        )
    )

    generated = _generate(
        module,
        tmp_path,
        stage="confirmation",
        winners_path=winners,
    )

    assert len(generated["configs"]) == expected_configs
    assert len(generated["runs"]) == expected_runs
    assert {record["lambda_lre"] for record in generated["runs"]} == expected_weights


def test_confirmation_rejects_unbound_or_too_many_winners(tmp_path: Path) -> None:
    module = _load_generator()
    _generate(module, tmp_path, stage="screening")
    winners = tmp_path / "winners.json"
    winners.write_text(
        json.dumps(
            {
                "schema": "avgaussianv2.lre-loss-screening-selection",
                "version": 1,
                "source_screening_manifest_sha256": "0" * 64,
                "selected_lambda_lre": [0.01, 0.02],
            }
        )
    )

    with pytest.raises(ValueError, match="does not bind"):
        _generate(
            module,
            tmp_path,
            stage="confirmation",
            winners_path=winners,
        )

    screening_manifest = tmp_path / "screening/manifest.json"
    winners.write_text(
        json.dumps(
            {
                "schema": "avgaussianv2.lre-loss-screening-selection",
                "version": 1,
                "source_screening_manifest_sha256": hashlib.sha256(
                    screening_manifest.read_bytes()
                ).hexdigest(),
                "selected_lambda_lre": [0.01, 0.02, 0.05],
            }
        )
    )
    with pytest.raises(ValueError, match="at most 2"):
        _generate(
            module,
            tmp_path,
            stage="confirmation",
            winners_path=winners,
        )


def test_generation_binds_each_survivor_to_its_architecture_config(
    tmp_path: Path,
) -> None:
    module = _load_generator()

    generated = module.generate(
        _manifest(),
        tmp_path,
        systems=("plain_unet", "query_dependent_p1"),
    )

    plain = next(
        record
        for record in generated["configs"]
        if record["system"] == "plain_unet"
        and record["scene"] == "scene1_opera"
        and record["lambda_lre"] == 0.0
    )
    p1 = next(
        record
        for record in generated["configs"]
        if record["system"] == "query_dependent_p1"
        and record["scene"] == "scene1_opera"
        and record["lambda_lre"] == 0.0
    )
    plain_config = yaml.safe_load(Path(plain["config"]).read_text())
    p1_config = yaml.safe_load(Path(p1["config"]).read_text())

    assert plain["training_mode"] == "audio_only"
    assert plain_config["model"]["audio_render_strategy"] == "plain_unet"
    assert p1["training_mode"] == "joint_conditioned"
    assert p1_config["model"]["audio_backend"] == "query_dependent_p1"


def test_generation_requires_explicit_architecture_survivors(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="explicitly supplied"):
        _load_generator().generate(_manifest(), tmp_path)

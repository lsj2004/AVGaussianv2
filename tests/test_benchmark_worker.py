from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from avgaussianv2.benchmark.runtime import BenchmarkRuntime, build_production_runtime
from avgaussianv2.benchmark.training import (
    BenchmarkCompatibility,
    BenchmarkConfig,
    BenchmarkMode,
    CheckpointIO,
    TRAIN_CAMERAS,
    build_worker_manifest,
    configure_benchmark_mode,
    make_shared_indices,
)
from avgaussianv2.cli import benchmark_worker
from avgaussianv2.config import (
    ModelConfig,
    PathConfig,
    ProjectConfig,
    SceneConfig,
    TrainConfig,
)
from avgaussianv2.models.fusion import AVGaussianFusionV2
from avgaussianv2.models.rgbd import RGBDConditionEncoder


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Linear(1, 1)
        self.audio = nn.Linear(1, 1)


class _Samples:
    def __init__(self) -> None:
        self.records = [
            SimpleNamespace(
                camera="cam00",
                camera_index=0,
                frame_index=2,
                time_seconds=0.25,
            )
        ]

    def __len__(self) -> int:
        return len(self.records)


class _Audio(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.acoustic = nn.Linear(1, 1)
        self.unet = nn.Linear(1, 1)
        self.film = nn.Linear(1, 1)

    def acoustic_parameters(self):
        return self.acoustic.parameters()

    def audio_unet_parameters(self):
        return self.unet.parameters()

    def film_parameters(self):
        return self.film.parameters()


def _project(tmp_path: Path) -> ProjectConfig:
    files = {}
    for name in ("visual.pt", "audio.pt", "dataset.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    return ProjectConfig(
        scene=SceneConfig(
            scene_id="scene1_opera",
            fps=30,
            train_cameras=tuple(f"cam{i:02d}" for i in range(38)),
            eval_cameras=("cam38",),
            camera_mapping={f"cam{i:02d}": i for i in range(39)},
        ),
        paths=PathConfig(
            visual_upstream_root=tmp_path,
            audio_upstream_root=tmp_path,
            visual_checkpoint=files["visual.pt"],
            audio_checkpoint=files["audio.pt"],
            manifest=files["dataset.json"],
            visual_memmap=tmp_path,
        ),
        model=ModelConfig(),
        train=TrainConfig(seed=42, warmup_steps=2_000, joint_steps=30_000),
    )


def test_production_runtime_is_train_only_and_hashes_real_inputs(
    tmp_path, monkeypatch
) -> None:
    project = _project(tmp_path)
    config_path = tmp_path / "project.yaml"
    config_path.write_bytes(b"project-config")
    calls = []
    model = _Model()
    samples = _Samples()
    criterion = nn.L1Loss()

    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.load_project_config", lambda _: project
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.audit_protocol_config", lambda _: {}
    )

    def fake_build(config, device, **kwargs):
        calls.append((config, device, kwargs))
        return SimpleNamespace(
            model=model,
            train_samples=samples,
            eval_samples=None,
            audio_loss_fn=criterion,
        )

    monkeypatch.setattr("avgaussianv2.benchmark.runtime.build_runtime", fake_build)
    first = build_production_runtime(
        config_path=config_path,
        device=torch.device("cpu"),
        trusted_upstream_artifacts=True,
    )
    second = build_production_runtime(
        config_path=config_path,
        device=torch.device("cpu"),
        trusted_upstream_artifacts=True,
    )

    assert calls[0][2] == {
        "trusted_upstream_artifacts": True,
        "include_eval": False,
    }
    assert first.config_sha256 == hashlib.sha256(b"project-config").hexdigest()
    assert first.source_sha256 == second.source_sha256
    assert first.dataset_identity_sha256 == second.dataset_identity_sha256
    assert first.visual_initialization_sha256 == second.visual_initialization_sha256
    assert first.audio_initialization_sha256 == second.audio_initialization_sha256
    assert first.model_initialization_sha256 == second.model_initialization_sha256
    assert first.model is model
    assert first.audio_loss_fn is criterion
    identity = json.loads(first.dataset_sample_ids[0])
    assert identity == {
        "camera": "cam00",
        "camera_index": 0,
        "frame_index": 2,
        "scene_id": "scene1_opera",
        "time_seconds": 0.25,
    }


def _write_worker_manifest(path: Path, identity: dict[str, str]) -> None:
    indices = make_shared_indices(1)
    compatibility = BenchmarkCompatibility(
        scene_id="scene1_opera",
        mode=BenchmarkMode.AUDIO_ONLY.value,
        train_cameras=tuple(f"cam{i:02d}" for i in range(38)),
        test_camera="cam38",
        seed=42,
        index_sha256=hashlib.sha256(
            json.dumps(list(indices), separators=(",", ":")).encode()
        ).hexdigest(),
        **identity,
    )
    path.write_text(
        json.dumps(
            build_worker_manifest(
                config=BenchmarkConfig(),
                compatibility=compatibility,
                shared_indices=indices,
            )
        )
    )


def test_worker_seeds_before_internal_builder_and_checks_identity(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    identity = {
        "visual_initialization_sha256": "1" * 64,
        "audio_initialization_sha256": "2" * 64,
        "model_initialization_sha256": "6" * 64,
        "source_sha256": "3" * 64,
        "config_sha256": "4" * 64,
    }
    manifest = tmp_path / "worker.json"
    _write_worker_manifest(manifest, identity)
    observed = {}

    def builder(**kwargs):
        observed["rng"] = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        observed["kwargs"] = kwargs
        return BenchmarkRuntime(
            model=_Model(),
            train_samples=[object()],
            train_config=TrainConfig(seed=42, warmup_steps=2_000, joint_steps=30_000),
            audio_loss_fn=nn.L1Loss(),
            dataset_identity_sha256="5" * 64,
            dataset_sample_ids=("sample-0",),
            **identity,
        )

    expected_python = random.Random(42).random()
    expected_numpy = float(np.random.RandomState(42).random_sample())
    generator = torch.Generator().manual_seed(42)
    expected_torch = float(torch.rand((), generator=generator))
    fake_result = SimpleNamespace(
        mode=BenchmarkMode.AUDIO_ONLY,
        completed_warmup_steps=0,
        completed_main_updates=30_000,
        resumed_from_main_step=0,
        redone_main_updates=0,
        selection="final",
        final_checkpoint=tmp_path / "final.pt",
        milestones=(),
        io=CheckpointIO(),
    )

    class Trainer:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            observed["run"] = kwargs
            return fake_result

    monkeypatch.setattr(benchmark_worker, "FixedBudgetTrainer", Trainer)
    result = benchmark_worker.run_worker(
        manifest_path=manifest,
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "output",
        device="cpu",
        trust_upstream_artifacts=True,
        resume=False,
        _runtime_builder=builder,
    )

    assert observed["rng"] == pytest.approx(
        (expected_python, expected_numpy, expected_torch)
    )
    assert observed["kwargs"]["trusted_upstream_artifacts"] is True
    assert len(observed["run"]["train_samples"]) == 1
    assert str(observed["run"]["output_dir"]).startswith("/proc/self/fd/")
    assert result["completed_main_updates"] == 30_000
    assert result["final_checkpoint"] == str(tmp_path / "output" / "final.pt")
    runtime_contract = json.loads(
        (tmp_path / "output" / "runtime_contract.json").read_text()
    )
    assert runtime_contract["schema"] == (
        "avgaussianv2.cam38-production-train-only-runtime"
    )
    assert runtime_contract["include_eval"] is False
    assert runtime_contract["train_cameras"] == list(TRAIN_CAMERAS)
    assert runtime_contract["test_camera"] == "cam38"
    assert result["runtime_contract_sha256"] == hashlib.sha256(
        (tmp_path / "output" / "runtime_contract.json").read_bytes()
    ).hexdigest()


def test_cli_does_not_expose_external_runtime_factory(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark-worker", "--help"])
    with pytest.raises(SystemExit) as stopped:
        benchmark_worker.main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--runtime-factory" not in help_text
    assert "--config" in help_text


def test_cli_reexec_sets_real_hash_seed_and_cublas_environment() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONHASHSEED", None)
    environment["CUBLAS_WORKSPACE_CONFIG"] = "wrong"
    environment.pop("AVGAUSSIANV2_BENCHMARK_ENV_REEXEC", None)
    command = [
        sys.executable,
        "-m",
        "avgaussianv2.cli.benchmark_worker",
        "--environment-probe",
        "benchmark-secret",
    ]
    first = subprocess.run(
        command,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    second = subprocess.run(
        command,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    first_value = json.loads(first.stdout)
    second_value = json.loads(second.stdout)
    assert first_value == second_value
    assert first_value["pythonhashseed"] == "42"
    assert first_value["cublas_workspace_config"] == ":4096:8"
    assert first_value["reexec_marker"] == "1"


def test_cli_rejects_invalid_environment_after_single_reexec() -> None:
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": "wrong",
        "AVGAUSSIANV2_BENCHMARK_ENV_REEXEC": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "avgaussianv2.cli.benchmark_worker",
            "--environment-probe",
            "secret",
        ],
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "remained invalid after re-exec" in result.stderr


@pytest.mark.parametrize(
    ("mode", "enabled"),
    [
        (BenchmarkMode.JOINT_CONDITIONED, {"visual", "condition", "audio"}),
        (BenchmarkMode.AUDIO_ONLY, {"audio"}),
        (BenchmarkMode.VISUAL_ONLY, {"visual"}),
    ],
)
def test_real_fusion_model_parameter_groups_follow_benchmark_mode(
    mode, enabled
) -> None:
    model = AVGaussianFusionV2(
        visual=nn.Linear(1, 1),
        condition_encoder=RGBDConditionEncoder(embedding_dim=8),
        audio=_Audio(),
    )
    configure_benchmark_mode(model, mode, "main")
    actual = set()
    if any(parameter.requires_grad for parameter in model.visual.parameters()):
        actual.add("visual")
    if any(
        parameter.requires_grad for parameter in model.condition_encoder.parameters()
    ):
        actual.add("condition")
    if any(parameter.requires_grad for parameter in model.audio.parameters()):
        actual.add("audio")
    assert actual == enabled
    assert callable(nn.L1Loss())

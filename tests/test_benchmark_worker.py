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
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.data.tensor import DeviceSampleSequence


def test_benchmark_worker_moves_every_training_sample_tensor_to_device() -> None:
    sample = AlignedAVSample(
        scene_id="scene1_opera",
        camera="cam00",
        frame_index=0,
        time_seconds=0.25,
        visual_time=torch.zeros(1, 1),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 12),
        source_audio=torch.zeros(1, 2, 8),
        target_audio=torch.zeros(1, 2, 8),
        target_rgb=torch.zeros(1, 2, 3, 3),
        image_size=(2, 3),
    )

    moved = DeviceSampleSequence(
        (sample,), torch.device("meta")
    )[0]

    assert moved.scene_id == sample.scene_id
    assert moved.camera == sample.camera
    assert moved.frame_index == sample.frame_index
    assert moved.time_seconds == sample.time_seconds
    assert moved.image_size == sample.image_size
    assert {
        value.device.type
        for value in vars(moved).values()
        if isinstance(value, torch.Tensor)
    } == {"meta"}


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
    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.upstream_source_inventory",
        lambda _: {"fixture": "f" * 64},
    )

    class Snapshot:
        def __init__(self, *_args, **_kwargs):
            self._active = True
            self.config = project
            self.config_sha256 = hashlib.sha256(b"project-config").hexdigest()
            self.audio_checkpoint_sha256 = hashlib.sha256(b"audio.pt").hexdigest()
            self.visual_checkpoint_sha256 = hashlib.sha256(b"visual.pt").hexdigest()
            self.manifest_sha256 = hashlib.sha256(b"dataset.json").hexdigest()
            self.source_inventory = {"fixture": "f" * 64}

        def __enter__(self):
            return self

        def verify(self):
            return None

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.ProductionRuntimeSnapshot", Snapshot
    )
    with build_production_runtime(
        config_path=config_path,
        device=torch.device("cpu"),
        trusted_upstream_artifacts=True,
    ) as first:
        pass
    with build_production_runtime(
        config_path=config_path,
        device=torch.device("cpu"),
        trusted_upstream_artifacts=True,
    ) as second:
        pass

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


def test_production_runtime_executes_only_pinned_inputs_during_live_replacement(
    tmp_path, monkeypatch
) -> None:
    audio_root = tmp_path / "audio"
    visual_root = tmp_path / "visual"
    (audio_root / "libs").mkdir(parents=True)
    (visual_root / "ftgspp").mkdir(parents=True)
    sources = {
        audio_root / "libs/__init__.py": b"",
        audio_root / "libs/snapshot_probe.py": b"VALUE = 'pinned-source'\n",
        visual_root / "ftgspp/__init__.py": b"",
    }
    for path, data in sources.items():
        path.write_bytes(data)
    visual_checkpoint = tmp_path / "visual.pt"
    audio_checkpoint = tmp_path / "audio.pt"
    manifest = tmp_path / "manifest.json"
    config_path = tmp_path / "resolved_project.yaml"
    source_config_path = tmp_path / "source_project.yaml"
    origin_path = tmp_path / "resolved_project.origin.json"
    project = _project(tmp_path)
    source_config_data = b"pinned-source-config"
    config_data = b"pinned-config"
    origin_data = json.dumps(
        {
            "schema": "avgaussianv2.cam38-resolved-config-origin",
            "version": 1,
            "source_path": str(source_config_path),
            "source_sha256": hashlib.sha256(source_config_data).hexdigest(),
            "resolved_sha256": hashlib.sha256(config_data).hexdigest(),
        }
    ).encode()
    originals = {
        visual_checkpoint: b"pinned-visual",
        audio_checkpoint: b"pinned-audio",
        manifest: b"pinned-manifest",
        config_path: config_data,
        source_config_path: source_config_data,
        origin_path: origin_data,
        **sources,
    }
    for path, data in originals.items():
        path.write_bytes(data)
    project = ProjectConfig(
        scene=project.scene,
        paths=PathConfig(
            visual_upstream_root=visual_root,
            audio_upstream_root=audio_root,
            visual_checkpoint=visual_checkpoint,
            audio_checkpoint=audio_checkpoint,
            manifest=manifest,
            visual_memmap=tmp_path,
        ),
        model=project.model,
        train=project.train,
    )

    def inventory(_config):
        return {
            "audiogs:libs/__init__.py": hashlib.sha256(
                sources[audio_root / "libs/__init__.py"]
            ).hexdigest(),
            "audiogs:libs/snapshot_probe.py": hashlib.sha256(
                (audio_root / "libs/snapshot_probe.py").read_bytes()
            ).hexdigest(),
            "ftgspp:ftgspp/__init__.py": hashlib.sha256(
                sources[visual_root / "ftgspp/__init__.py"]
            ).hexdigest(),
        }

    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.load_project_config", lambda _: project
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.load_audited_benchmark_config",
        lambda *_args, **_kwargs: (
            project,
            hashlib.sha256(originals[source_config_path]).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.upstream_source_inventory", inventory
    )
    observed = {}

    class LazyModel(_Model):
        def forward(self, _sample):
            from libs import snapshot_probe

            observed["source"] = snapshot_probe.VALUE
            return snapshot_probe.VALUE

    def fake_build(config, _device, **_kwargs):
        live_paths = (
            config_path,
            source_config_path,
            origin_path,
            visual_checkpoint,
            audio_checkpoint,
            manifest,
            audio_root / "libs/snapshot_probe.py",
        )
        try:
            for path in live_paths:
                path.write_bytes(b"live-replacement")
            observed.update(
                visual=config.paths.visual_checkpoint.read_bytes(),
                audio=config.paths.audio_checkpoint.read_bytes(),
                manifest=config.paths.manifest.read_bytes(),
                scene=config.scene.scene_id,
            )
        finally:
            for path in live_paths:
                path.write_bytes(originals[path])
        return SimpleNamespace(
            model=LazyModel(),
            train_samples=_Samples(),
            eval_samples=None,
            audio_loss_fn=nn.L1Loss(),
        )

    monkeypatch.setattr(
        "avgaussianv2.benchmark.runtime.build_runtime", fake_build
    )
    for name in tuple(sys.modules):
        if name == "libs" or name.startswith("libs."):
            monkeypatch.delitem(sys.modules, name)
    with build_production_runtime(
        config_path=config_path,
        device=torch.device("cpu"),
        trusted_upstream_artifacts=True,
        config_origin_path=origin_path,
    ) as result:
        assert any(
            finder.__class__.__name__ == "_SnapshotSourceFinder"
            for finder in sys.meta_path
        )
        lazy_source = audio_root / "libs/snapshot_probe.py"
        lazy_source.write_bytes(b"VALUE = 'active-attacker'\n")
        try:
            assert result.model(object()) == "pinned-source"
        finally:
            lazy_source.write_bytes(originals[lazy_source])

    assert observed == {
        "visual": b"pinned-visual",
        "audio": b"pinned-audio",
        "manifest": b"pinned-manifest",
        "source": "pinned-source",
        "scene": "scene1_opera",
    }
    assert not any(
        finder.__class__.__name__ == "_SnapshotSourceFinder"
        for finder in sys.meta_path
    )
    assert result.config_sha256 == hashlib.sha256(b"pinned-config").hexdigest()


def _write_worker_manifest(
    path: Path, identity: dict[str, str], *, seed: int = 42
) -> None:
    indices = make_shared_indices(1, seed=seed)
    compatibility = BenchmarkCompatibility(
        scene_id="scene1_opera",
        mode=BenchmarkMode.AUDIO_ONLY.value,
        train_cameras=tuple(f"cam{i:02d}" for i in range(38)),
        test_camera="cam38",
        seed=seed,
        index_sha256=hashlib.sha256(
            json.dumps(list(indices), separators=(",", ":")).encode()
        ).hexdigest(),
        **identity,
    )
    path.write_text(
        json.dumps(
            build_worker_manifest(
                config=BenchmarkConfig(seed=seed),
                compatibility=compatibility,
                shared_indices=indices,
            )
        )
    )


@pytest.mark.parametrize("seed", [42, 73])
def test_worker_seeds_before_internal_builder_and_checks_identity(
    tmp_path, monkeypatch, seed
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
    _write_worker_manifest(manifest, identity, seed=seed)
    observed = {}

    class Lease:
        active = True

        def __exit__(self, *_):
            self.active = False
            return False

    lease = Lease()

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
            train_config=TrainConfig(seed=seed, warmup_steps=2_000, joint_steps=30_000),
            audio_loss_fn=nn.L1Loss(),
            dataset_identity_sha256="5" * 64,
            dataset_sample_ids=("sample-0",),
            _input_snapshot=lease,
            **identity,
        )

    expected_python = random.Random(seed).random()
    expected_numpy = float(np.random.RandomState(seed).random_sample())
    generator = torch.Generator().manual_seed(seed)
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
            assert lease.active
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
        stop_after_step=100,
        _runtime_builder=builder,
    )

    assert observed["rng"] == pytest.approx(
        (expected_python, expected_numpy, expected_torch)
    )
    assert observed["kwargs"]["trusted_upstream_artifacts"] is True
    assert len(observed["run"]["train_samples"]) == 1
    assert observed["run"]["stop_after_main_step"] == 100
    assert lease.active is False
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
    assert (
        result["runtime_contract_sha256"]
        == hashlib.sha256(
            (tmp_path / "output" / "runtime_contract.json").read_bytes()
        ).hexdigest()
    )


def test_runtime_lease_closes_on_training_exception() -> None:
    observed = {}

    class Lease:
        def __exit__(self, exc_type, exc, _traceback):
            observed["exception"] = (exc_type, exc)
            return False

    runtime = BenchmarkRuntime(
        model=_Model(),
        train_samples=(object(),),
        train_config=TrainConfig(),
        audio_loss_fn=nn.L1Loss(),
        config_sha256="1" * 64,
        source_sha256="2" * 64,
        visual_initialization_sha256="3" * 64,
        audio_initialization_sha256="4" * 64,
        model_initialization_sha256="5" * 64,
        dataset_identity_sha256="6" * 64,
        dataset_sample_ids=("sample",),
        _input_snapshot=Lease(),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        with runtime:
            raise RuntimeError("training failed")

    assert observed["exception"][0] is RuntimeError
    assert str(observed["exception"][1]) == "training failed"


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


def test_plain_unet_audio_only_mode_enables_acoustic_and_unet() -> None:
    model = AVGaussianFusionV2(
        visual=nn.Linear(1, 1),
        condition_encoder=RGBDConditionEncoder(embedding_dim=8),
        audio=_Audio(),
    )
    model.audio.render_strategy = SimpleNamespace(value="plain_unet")

    configure_benchmark_mode(model, BenchmarkMode.AUDIO_ONLY, "main")

    groups = model.named_parameter_groups()
    assert all(parameter.requires_grad for parameter in groups["acoustic"])
    assert all(parameter.requires_grad for parameter in groups["audio_unet"])
    assert callable(nn.L1Loss())

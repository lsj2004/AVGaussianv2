from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import wave
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from avgaussianv2.benchmark.evaluation import TrainingEvidence
from avgaussianv2.benchmark.production import (
    _PinnedInput,
    _SnapshotSourceFinder,
    _snapshot_source_imports,
    build_evaluation_adapters,
    materialize_strict_scene_manifest,
    prepare_worker_manifests,
    write_resolved_project_config,
)
from avgaussianv2.benchmark.runtime import BenchmarkRuntime
from avgaussianv2.benchmark.runtime import state_sha256
from avgaussianv2.benchmark.training import TRAIN_CAMERAS
from avgaussianv2.config import load_project_config
from avgaussianv2.config import (
    ModelConfig,
    PathConfig,
    ProjectConfig,
    SceneConfig,
    TrainConfig,
)


ROOT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


@pytest.mark.parametrize(
    ("time_shape", "valid"),
    [([130, 38, 1], True), ([130, 37, 1], False)],
)
def test_materialize_strict_manifest_uses_verified_cam38_native_inputs(
    tmp_path, monkeypatch, time_shape, valid
):
    scene = "scene1_opera"
    config_path = tmp_path / "configs/benchmark_cam38/scene1_opera.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(b"strict protocol\n")
    visual_root = tmp_path / "visual"
    audio_root = tmp_path / "audio"
    aligned_audio = audio_root / "aligned_16k_stereo"
    visual_root.mkdir()
    aligned_audio.mkdir(parents=True)
    cameras = tuple(f"cam{index:02d}" for index in range(39))
    for camera in cameras:
        (visual_root / f"{camera}.mp4").write_bytes(b"video")
    os.link(visual_root / "cam00.mp4", tmp_path / "train-only-cam00.mp4")
    (visual_root / "manifest.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "num_cameras": 39,
                "num_frames": 150,
                "camera_names": list(cameras),
            }
        )
    )
    for name in ("near", *cameras):
        with wave.open(str(aligned_audio / f"{name}.wav"), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(1_000)
            stream.writeframes(b"\0" * (5_000 * 2 * 2))
    memmap = tmp_path / "memmap"
    memmap.mkdir()
    (memmap / "meta.json").write_text(
        json.dumps(
            {
                "rgb": {"shape": [130, 38, 8, 12, 3]},
                "time": {"shape": time_shape},
            }
        )
    )
    manifest = tmp_path / "runs/cam38_strict/scene1_opera/protocol/scene_manifest.json"
    config = ProjectConfig(
        scene=SceneConfig(
            scene_id=scene,
            fps=30.0,
            train_cameras=cameras[:-1],
            eval_cameras=("cam38",),
            camera_mapping={camera: index for index, camera in enumerate(cameras)},
        ),
        paths=PathConfig(
            visual_upstream_root=tmp_path / "ftgspp",
            audio_upstream_root=tmp_path / "audiogs",
            visual_checkpoint=tmp_path / "visual.pt",
            audio_checkpoint=tmp_path / "audio.pt",
            manifest=manifest,
            visual_memmap=memmap,
        ),
        model=ModelConfig(sample_rate=1_000),
        train=TrainConfig(crop_seconds=0.5),
    )
    conversion = tmp_path / "conversion.json"
    conversion.write_text(json.dumps({"audio_root": str(audio_root)}))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def verify(_path, *, expected_scene, expected_model_kind):
        assert expected_scene == scene
        inputs = {
            "protocol_config": {"sha256": config_sha256},
        }
        if expected_model_kind == "audiogs":
            inputs["conversion_manifest"] = {"path": str(conversion)}
        else:
            inputs["sampled_scene_root"] = str(visual_root)
        return {"inputs": inputs}

    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.audit_protocol_config",
        lambda _path: {"benchmark": {"expected_test_samples": 130}},
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.load_project_config",
        lambda _path: config,
    )

    arguments = {
        "config_path": config_path,
        "native_contract_dirs": {
            "audiogs": tmp_path / "audiogs-contract",
            "ftgspp": tmp_path / "ftgspp-contract",
        },
        "native_verifier": verify,
    }
    if not valid:
        with pytest.raises(ValueError, match="RGB/model-time"):
            materialize_strict_scene_manifest(**arguments)
        return

    result = materialize_strict_scene_manifest(**arguments)

    payload = json.loads(manifest.read_text())
    assert result == manifest
    assert payload["train_cameras"] == list(cameras[:-1])
    assert payload["eval_cameras"] == ["cam38"]
    assert payload["num_frames"] == 130
    assert payload["frame_times"][0] == pytest.approx(0.25)
    assert payload["frame_times"][-1] == pytest.approx(0.25 + 129 / 30)
    assert payload["cameras"]["cam38"]["audio_path"].endswith("/cam38.wav")

    before = manifest.read_bytes()
    materialize_strict_scene_manifest(**arguments)
    assert manifest.read_bytes() == before


@pytest.fixture(autouse=True)
def _restore_torch_random_and_deterministic_state():
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    yield
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = {"audio_only": 0, "visual_only": 0, "full": 0}
        self.visual = SimpleNamespace(
            render_rgbd=lambda *_: SimpleNamespace(rgb=torch.ones(1, 2, 2, 3))
        )
        self.condition_enabled = True

    def forward_audio_only(self, _sample):
        self.calls["audio_only"] += 1
        return torch.ones(1, 2, 8)

    def render_rgbd(self, _sample):
        self.calls["visual_only"] += 1
        return SimpleNamespace(rgb=torch.ones(1, 2, 2, 3))

    def forward(self, _sample):
        self.calls["full"] += 1
        return SimpleNamespace(
            predicted_audio=torch.ones(1, 2, 8),
            rgbd=SimpleNamespace(rgb=torch.ones(1, 2, 2, 3)),
        )


def _evidence(
    system: str,
    *,
    checkpoint: Path = Path("/tmp/native.pt"),
    config_sha256: str = DIGEST,
) -> TrainingEvidence:
    return TrainingEvidence(
        system_name=system,
        scene_id="scene1_opera",
        role="native_reference",
        train_cameras=TRAIN_CAMERAS,
        test_camera="cam38",
        test_targets_read_during_training=False,
        seed=42,
        planned_updates=2_318 if system == "native_audiogs" else 30_000,
        completed_updates=2_318 if system == "native_audiogs" else 30_000,
        checkpoint_step=2_318 if system == "native_audiogs" else 30_000,
        checkpoint_path=str(checkpoint.absolute()),
        checkpoint_sha256=(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if checkpoint.is_file()
            else DIGEST
        ),
        config_sha256=config_sha256,
        source_sha256=DIGEST,
        visual_initialization_sha256=DIGEST,
        audio_initialization_sha256=DIGEST,
        model_initialization_sha256=DIGEST,
        index_sha256=None,
        batch_size=1,
        epochs=61.0 if system == "native_audiogs" else None,
        native_contract_path="/tmp/native-contract",
        native_contract_sha256=DIGEST,
    )


def test_prepare_builds_train_only_identity_on_each_assigned_device(
    tmp_path, monkeypatch
):
    source = ROOT / "configs" / "benchmark_cam38" / "scene1_opera.yaml"
    calls = []
    loaded = load_project_config(source)

    class Snapshot:
        instances = []

        def __init__(self, path):
            self.path = path
            self.config = loaded
            self.audited_config = loaded
            self.source_config_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            self.audio_checkpoint_sha256 = DIGEST
            self.visual_checkpoint_sha256 = DIGEST
            self.source_inventory = {
                "audiogs:source.py": DIGEST,
                "ftgspp:source.py": DIGEST,
            }
            self.verify_calls = 0
            self.instances.append(self)

        def __enter__(self):
            return self

        def verify(self):
            self.verify_calls += 1

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.ProductionRuntimeSnapshot", Snapshot
    )

    def build(**kwargs):
        calls.append(kwargs)
        resolved_hash = hashlib.sha256(kwargs["config_path"].read_bytes()).hexdigest()
        return BenchmarkRuntime(
            model=_Model(),
            train_samples=(object(), object()),
            train_config=SimpleNamespace(),
            audio_loss_fn=SimpleNamespace(),
            config_sha256=resolved_hash,
            source_sha256=DIGEST,
            visual_initialization_sha256=DIGEST,
            audio_initialization_sha256=DIGEST,
            model_initialization_sha256=DIGEST,
            dataset_identity_sha256=DIGEST,
            dataset_sample_ids=("sample-a", "sample-b"),
        )

    def native_contract(_path, *, expected_scene, expected_model_kind):
        del expected_scene
        root = (
            loaded.paths.audio_upstream_root
            if expected_model_kind == "audiogs"
            else loaded.paths.visual_upstream_root
        )
        checkpoint = (
            loaded.paths.audio_checkpoint
            if expected_model_kind == "audiogs"
            else loaded.paths.visual_checkpoint
        )
        return {
            "inputs": {
                "protocol_config": {
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
                },
                "source_audits": [
                    {"path": str(root / "source.py"), "sha256": DIGEST}
                ],
            },
            "checkpoint": {"path": str(checkpoint), "sha256": DIGEST},
            "_manifest_sha256": DIGEST,
        }

    result = prepare_worker_manifests(
        config_path=source,
        output_dir=tmp_path,
        devices=("cuda:0", "cuda:1", "cuda:2"),
        trusted_upstream_artifacts=True,
        native_contract_dirs={
            "audiogs": tmp_path / "audiogs",
            "ftgspp": tmp_path / "ftgspp",
        },
        runtime_builder=build,
        native_verifier=native_contract,
    )

    assert [str(call["device"]) for call in calls] == ["cuda:0", "cuda:1", "cuda:2"]
    assert len({id(call["input_snapshot"]) for call in calls}) == 1
    assert calls[0]["input_snapshot"] is Snapshot.instances[0]
    assert Snapshot.instances[0].verify_calls == 4
    assert all(call["trusted_upstream_artifacts"] for call in calls)
    assert result["include_eval"] is False
    assert set(result["worker_manifests"]) == {
        "joint_conditioned",
        "audio_only",
        "visual_only",
    }
    for mode, path in result["worker_manifests"].items():
        manifest = json.loads(Path(path).read_text())
        assert manifest["mode"] == mode
        assert len(manifest["shared_indices"]) == 30_000
        assert set(manifest["shared_indices"]) <= {0, 1}

    def mismatched_native(*args, **kwargs):
        contract = native_contract(*args, **kwargs)
        contract["inputs"]["source_audits"][0]["sha256"] = "b" * 64
        return contract

    with pytest.raises(ValueError, match="does not bind"):
        prepare_worker_manifests(
            config_path=source,
            output_dir=tmp_path / "mismatched",
            devices=("cuda:0", "cuda:1", "cuda:2"),
            trusted_upstream_artifacts=True,
            native_contract_dirs={
                "audiogs": tmp_path / "audiogs",
                "ftgspp": tmp_path / "ftgspp",
            },
            runtime_builder=build,
            native_verifier=mismatched_native,
        )


@pytest.mark.parametrize(
    ("system", "has_audio", "has_rgb"),
    [
        ("native_audiogs", True, False),
        ("native_ftgspp", False, True),
    ],
)
def test_native_production_adapter_exposes_only_its_modality(
    tmp_path, monkeypatch, system, has_audio, has_rgb
):
    source = ROOT / "configs" / "benchmark_cam38" / "scene1_opera.yaml"
    resolved = tmp_path / "resolved_project.yaml"
    write_resolved_project_config(source, resolved)
    checkpoint = tmp_path / "native.pt"
    checkpoint.write_bytes(b"native")
    model = _Model()
    calls = []
    consumed_checkpoints = []
    config = load_project_config(resolved)
    config = replace(
        config,
        paths=replace(
            config.paths,
            audio_checkpoint=checkpoint,
            visual_checkpoint=checkpoint,
        ),
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    def build(config, device, *, trusted_upstream_artifacts, include_eval):
        calls.append((config, device, trusted_upstream_artifacts, include_eval))
        checkpoint_field = (
            "audio_checkpoint" if system == "native_audiogs" else "visual_checkpoint"
        )
        checkpoint.write_bytes(b"temporarily replaced")
        consumed_checkpoints.append(
            Path(getattr(config.paths, checkpoint_field)).read_bytes()
        )
        checkpoint.write_bytes(b"native")
        return SimpleNamespace(
            model=model,
            eval_samples=(object(),),
            audio_loss_fn=lambda *_: {},
        )

    monkeypatch.setattr("avgaussianv2.benchmark.production.build_runtime", build)
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.load_audited_benchmark_config",
        lambda _path, **_kwargs: (config, source_sha256),
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.verify_native_contract",
        lambda *_args, **_kwargs: {"inputs": {"source_audits": []}},
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.upstream_source_inventory",
        lambda _config: {},
    )
    runtime_factory, predictor_factory = build_evaluation_adapters(
        resolved_config=resolved,
        device="cpu",
        evidence=_evidence(system, checkpoint=checkpoint, config_sha256=source_sha256),
        trusted_upstream_artifacts=True,
    )
    assert calls == []
    runtime = runtime_factory()
    predictor = predictor_factory(runtime)
    sample = SimpleNamespace(
        visual_time=torch.zeros(1),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        image_size=(2, 2),
    )
    checkpoint_field = (
        "audio_checkpoint" if system == "native_audiogs" else "visual_checkpoint"
    )
    pinned_checkpoint = Path(getattr(calls[0][0].paths, checkpoint_field))
    pinned_target = os.readlink(pinned_checkpoint)
    with predictor:
        assert pinned_checkpoint.read_bytes() == b"native"
        assert any(isinstance(finder, _SnapshotSourceFinder) for finder in sys.meta_path)
        prediction = predictor(sample)
        assert pinned_checkpoint.read_bytes() == b"native"
    assert not pinned_checkpoint.exists() or os.readlink(pinned_checkpoint) != pinned_target
    assert not any(
        isinstance(finder, _SnapshotSourceFinder) for finder in sys.meta_path
    )

    assert calls[0][2:] == (True, True)
    assert str(getattr(calls[0][0].paths, checkpoint_field)).startswith(
        "/proc/self/fd/"
    )
    assert consumed_checkpoints == [b"native"]
    assert (prediction.predicted_audio is not None) is has_audio
    assert (prediction.rendered_rgb is not None) is has_rgb
    expected_calls = (
        {"audio_only": 1, "visual_only": 0, "full": 0}
        if system == "native_audiogs"
        else {"audio_only": 0, "visual_only": 1, "full": 0}
    )
    assert model.calls == expected_calls


def test_snapshot_importer_preserves_nested_package_semantics(
    tmp_path, monkeypatch
):
    audio_root = tmp_path / "audio"
    visual_root = tmp_path / "visual"
    sources = {
        visual_root / "ftgspp/__init__.py": b"from .models import GAUSSIAN\n",
        visual_root
        / "ftgspp/models/__init__.py": b"from .gaussians import GAUSSIAN\n",
        visual_root / "ftgspp/models/gaussians.py": b"GAUSSIAN = 'snapshot-ftgs'\n",
        audio_root / "libs/__init__.py": b"from .models import AUDIO\n",
        audio_root / "libs/models/__init__.py": b"from .networks import AUDIO\n",
        audio_root / "libs/models/networks/__init__.py": b"AUDIO = 'snapshot-audio'\n",
    }
    for path, data in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in tuple(sys.modules):
        if (
            name == "ftgspp"
            or name.startswith("ftgspp.")
            or name == "libs"
            or name.startswith("libs.")
        ):
            monkeypatch.delitem(sys.modules, name)
    with ExitStack() as stack:
        pins = [
            stack.enter_context(
                _PinnedInput(path, hashlib.sha256(data).hexdigest())
            )
            for path, data in sources.items()
        ]
        with _snapshot_source_imports(pins, (audio_root, visual_root)):
            ftgspp = importlib.import_module("ftgspp")
            gaussians = importlib.import_module("ftgspp.models.gaussians")
            libs_models = importlib.import_module("libs.models")
            networks = importlib.import_module("libs.models.networks")

            assert ftgspp.GAUSSIAN == gaussians.GAUSSIAN == "snapshot-ftgs"
            assert libs_models.AUDIO == networks.AUDIO == "snapshot-audio"
            assert ftgspp.__spec__.submodule_search_locations
            assert libs_models.__spec__.submodule_search_locations
            assert gaussians.__spec__.submodule_search_locations is None
    assert not {
        "ftgspp",
        "ftgspp.models",
        "ftgspp.models.gaussians",
        "libs",
        "libs.models",
        "libs.models.networks",
    }.intersection(sys.modules)


def test_snapshot_import_failure_is_not_hidden_by_filesystem_fallback(tmp_path):
    audio_root = tmp_path / "audio"
    visual_root = tmp_path / "visual"
    sources = {
        visual_root / "ftgspp/__init__.py": b"",
        visual_root / "ftgspp/models/__init__.py": b"",
        visual_root / "ftgspp/models/gaussians.py": b"this is invalid python !\n",
    }
    for source, data in sources.items():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
    with ExitStack() as stack:
        pins = [
            stack.enter_context(
                _PinnedInput(source, hashlib.sha256(data).hexdigest())
            )
            for source, data in sources.items()
        ]
        fallback = tmp_path / "fallback/ftgspp/models/gaussians.py"
        fallback.parent.mkdir(parents=True)
        fallback.write_bytes(b"VALUE = 'filesystem fallback'\n")
        stack.callback(sys.path.remove, str(fallback.parents[2]))
        sys.path.append(str(fallback.parents[2]))
        with _snapshot_source_imports(pins, (audio_root, visual_root)) as finder:
            with pytest.raises(SyntaxError):
                importlib.import_module("ftgspp.models.gaussians")
            with pytest.raises(RuntimeError, match="snapshot import failed"):
                finder.assert_no_failures()


def test_pinned_input_rejects_symlinked_ancestor(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    source = real / "source.py"
    source.write_bytes(b"trusted")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        with _PinnedInput(alias / "source.py", None):
            pass


@pytest.mark.parametrize("process_component", ["self", str(os.getpid())])
def test_pinned_input_accepts_owned_retained_fd_descendant(
    tmp_path, process_component
):
    root = tmp_path / "retained"
    child = root / "protocol"
    child.mkdir(parents=True)
    source = child / "resolved_project.yaml"
    source.write_bytes(b"trusted")
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        retained = (
            Path(f"/proc/{process_component}/fd/{root_fd}")
            / "protocol"
            / "resolved_project.yaml"
        )
        with _PinnedInput(retained, hashlib.sha256(b"trusted").hexdigest()) as pin:
            assert pin.data == b"trusted"
            assert pin.proc_path.read_bytes() == b"trusted"
    finally:
        os.close(root_fd)


def test_pinned_input_retained_fd_descendant_still_rejects_symlink(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (real / "source.py").write_bytes(b"trusted")
    (root / "alias").symlink_to(real, target_is_directory=True)
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        with pytest.raises(OSError):
            with _PinnedInput(
                Path(f"/proc/self/fd/{root_fd}") / "alias" / "source.py", None
            ):
                pass
    finally:
        os.close(root_fd)


def test_pinned_input_accepts_exact_retained_regular_fd(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"trusted")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        with _PinnedInput(Path(f"/proc/self/fd/{source_fd}"), None) as pin:
            assert pin.data == b"trusted"
    finally:
        os.close(source_fd)


def test_pinned_input_detects_ancestor_swap_and_keeps_snapshot_bytes(tmp_path):
    ancestor = tmp_path / "source"
    ancestor.mkdir()
    source = ancestor / "module.py"
    source.write_bytes(b"trusted")
    moved = tmp_path / "moved"

    with pytest.raises(RuntimeError, match="identity changed"):
        with _PinnedInput(source, None) as pinned:
            ancestor.rename(moved)
            ancestor.mkdir()
            (ancestor / "module.py").write_bytes(b"attacker")
            assert pinned.proc_path.read_bytes() == b"trusted"


def test_pinned_input_consumes_snapshot_during_final_swap_and_restore(tmp_path):
    source = tmp_path / "module.py"
    source.write_bytes(b"trusted")
    retained = tmp_path / "retained.py"

    with _PinnedInput(source, None) as pinned:
        source.rename(retained)
        source.write_bytes(b"attacker")
        assert pinned.proc_path.read_bytes() == b"trusted"
        source.unlink()
        retained.rename(source)


def test_snapshot_source_mapping_never_resolves_paths_after_pin(
    tmp_path, monkeypatch
):
    root = tmp_path / "upstream"
    source = root / "package/module.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")

    with _PinnedInput(source, None) as pinned:
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda *_args, **_kwargs: pytest.fail("resolved after pin"),
        )
        with _snapshot_source_imports([pinned], (root, tmp_path / "other")):
            assert any(
                isinstance(finder, _SnapshotSourceFinder) for finder in sys.meta_path
            )


def test_native_adapter_rejects_changed_upstream_source_before_runtime(
    tmp_path, monkeypatch
):
    source = ROOT / "configs" / "benchmark_cam38" / "scene1_opera.yaml"
    resolved = tmp_path / "resolved_project.yaml"
    write_resolved_project_config(source, resolved)
    checkpoint = tmp_path / "native.pt"
    checkpoint.write_bytes(b"native")
    upstream = tmp_path / "upstream.py"
    upstream.write_bytes(b"trusted source")
    config = load_project_config(resolved)
    config = replace(config, paths=replace(config.paths, audio_checkpoint=checkpoint))
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.load_audited_benchmark_config",
        lambda _path, **_kwargs: (config, source_sha256),
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.verify_native_contract",
        lambda *_args, **_kwargs: {
            "inputs": {
                "source_audits": [
                    {
                        "path": str(upstream),
                        "sha256": hashlib.sha256(b"old").hexdigest(),
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.build_runtime",
        lambda *_args, **_kwargs: pytest.fail(
            "constructed runtime from changed source"
        ),
    )
    runtime_factory, _ = build_evaluation_adapters(
        resolved_config=resolved,
        device="cpu",
        evidence=_evidence(
            "native_audiogs",
            checkpoint=checkpoint,
            config_sha256=source_sha256,
        ),
        trusted_upstream_artifacts=True,
    )

    with pytest.raises(ValueError, match="execution source changed"):
        runtime_factory()


def test_continuation_adapter_reseeds_both_random_runtime_constructions(
    tmp_path, monkeypatch
):
    source = ROOT / "configs" / "benchmark_cam38" / "scene1_opera.yaml"
    resolved = tmp_path / "resolved_project.yaml"
    write_resolved_project_config(source, resolved)
    checkpoint = tmp_path / "step.pt"
    config_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()

    def random_model():
        model = _Model()
        model.weight = nn.Parameter(torch.rand(()))
        return model

    torch.manual_seed(42)
    reference = random_model()
    torch.save({"model": reference.state_dict()}, checkpoint)
    model_sha256 = state_sha256(reference)
    evidence = replace(
        _evidence(
            "joint_conditioned",
            checkpoint=checkpoint,
            config_sha256=config_sha256,
        ),
        role="continuation",
        planned_updates=30_000,
        completed_updates=1_000,
        checkpoint_step=1_000,
        epochs=None,
        source_sha256=DIGEST,
        visual_initialization_sha256=DIGEST,
        audio_initialization_sha256=DIGEST,
        model_initialization_sha256=model_sha256,
        index_sha256=DIGEST,
        training_output_dir=str(tmp_path),
        runtime_contract_path=str(tmp_path / "runtime_contract.json"),
        runtime_contract_sha256=DIGEST,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = load_project_config(resolved)
    config = replace(
        config,
        paths=replace(
            config.paths,
            visual_checkpoint=checkpoint,
            audio_checkpoint=checkpoint,
            manifest=manifest,
        ),
    )
    consumed_configs = []

    def training_runtime(**_kwargs):
        original_config = resolved.read_bytes()
        resolved.write_bytes(b"temporarily replaced")
        consumed_configs.append(Path(_kwargs["config_path"]).read_bytes())
        resolved.write_bytes(original_config)
        model = random_model()
        return BenchmarkRuntime(
            model=model,
            train_samples=(object(),),
            train_config=SimpleNamespace(),
            audio_loss_fn=SimpleNamespace(),
            config_sha256=config_sha256,
            source_sha256=DIGEST,
            visual_initialization_sha256=DIGEST,
            audio_initialization_sha256=DIGEST,
            model_initialization_sha256=state_sha256(model),
            dataset_identity_sha256=DIGEST,
            dataset_sample_ids=("sample",),
        )

    def evaluation_runtime(*_args, **_kwargs):
        return SimpleNamespace(
            model=random_model(),
            eval_samples=(object(),),
            audio_loss_fn=SimpleNamespace(),
        )

    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.load_audited_benchmark_config",
        lambda _path, **_kwargs: (config, config_sha256),
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.build_production_runtime",
        training_runtime,
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.build_runtime", evaluation_runtime
    )
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.upstream_source_inventory",
        lambda _config: {},
    )
    runtime_factory, _ = build_evaluation_adapters(
        resolved_config=resolved,
        device="cpu",
        evidence=evidence,
        trusted_upstream_artifacts=True,
    )

    runtime_factory()
    assert consumed_configs == [resolved.read_bytes()]

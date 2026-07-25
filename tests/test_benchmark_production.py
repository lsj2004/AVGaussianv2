from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from avgaussianv2.benchmark.evaluation import TrainingEvidence
from avgaussianv2.benchmark.production import (
    build_evaluation_adapters,
    prepare_worker_manifests,
    write_resolved_project_config,
)
from avgaussianv2.benchmark.runtime import BenchmarkRuntime
from avgaussianv2.benchmark.training import TRAIN_CAMERAS
from avgaussianv2.config import load_project_config


ROOT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.visual = SimpleNamespace(
            render_rgbd=lambda *_: SimpleNamespace(rgb=torch.ones(1, 2, 2, 3))
        )
        self.condition_enabled = True

    def forward(self, _sample):
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
    )


def test_prepare_builds_train_only_identity_on_each_assigned_device(tmp_path):
    source = ROOT / "configs" / "benchmark_cam38" / "scene1_opera.yaml"
    calls = []

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
        native_verifier=lambda path, *, expected_scene, expected_model_kind: {
            "inputs": {
                "protocol_config": {
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
                }
            },
            "checkpoint": {
                "path": str(
                    (
                        ROOT
                        / "runs"
                        / "cam38_strict"
                        / "scene1_opera"
                        / expected_model_kind
                        / "native"
                        / (
                            "replayNVAS/SC-scene1-opera-cam38-shared/viewpoint_39/checkpoint_latest.pth"
                            if expected_model_kind == "audiogs"
                            else "scene1_opera/00/gaussians.pt"
                        )
                    ).resolve()
                ),
                "sha256": DIGEST,
            },
            "_manifest_sha256": DIGEST,
        },
    )

    assert [str(call["device"]) for call in calls] == ["cuda:0", "cuda:1", "cuda:2"]
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
        return SimpleNamespace(
            model=model,
            eval_samples=(object(),),
            audio_loss_fn=lambda *_: {},
        )

    monkeypatch.setattr("avgaussianv2.benchmark.production.build_runtime", build)
    monkeypatch.setattr(
        "avgaussianv2.benchmark.production.load_audited_benchmark_config",
        lambda _path: (config, source_sha256),
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
    prediction = predictor(sample)

    assert calls[0][2:] == (True, True)
    assert (prediction.predicted_audio is not None) is has_audio
    assert (prediction.rendered_rgb is not None) is has_rgb

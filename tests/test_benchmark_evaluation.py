from __future__ import annotations

import hashlib
import json

import pytest
import torch

from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluationError,
    BenchmarkEvaluator,
    BenchmarkPrediction,
    EvaluationIdentity,
    TrainingEvidence,
    load_evaluation,
    verify_evaluation,
)
from avgaussianv2.contracts import AlignedAVSample


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sample(index: int, scene: str = "scene1_opera") -> AlignedAVSample:
    audio = torch.linspace(-0.2, 0.2, 640).repeat(1, 2, 1)
    rgb = torch.full((1, 4, 5, 3), 0.2 + index / 100)
    return AlignedAVSample(
        scene_id=scene,
        camera="cam38",
        frame_index=index,
        time_seconds=index / 10,
        visual_time=torch.tensor([index / 10]),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 3),
        source_audio=audio,
        target_audio=audio,
        target_rgb=rgb,
        image_size=(4, 5),
    )


def _evidence(system: str, step: int = 30_000) -> TrainingEvidence:
    return TrainingEvidence(
        system_name=system,
        scene_id="scene1_opera",
        role="continuation",
        train_cameras=tuple(f"cam{x:02d}" for x in range(38)),
        test_camera="cam38",
        test_targets_read_during_training=False,
        seed=42,
        planned_updates=30_000,
        completed_updates=step,
        checkpoint_step=step,
        checkpoint_path=f"/tmp/{system}-{step}.pt",
        checkpoint_sha256=_sha(f"checkpoint-{system}-{step}"),
        config_sha256=_sha("config"),
        source_sha256=_sha("source"),
        visual_initialization_sha256=_sha("visual"),
        audio_initialization_sha256=_sha("audio"),
        model_initialization_sha256=_sha("model"),
        index_sha256=_sha("indices"),
        batch_size=1,
        epochs=None,
    )


def _loss(predicted, target):
    error = (predicted - target).abs().mean()
    return {"total_loss": error, "mono_loss": error, "diff_loss": error}


def test_evaluator_constructs_cam38_only_after_training_gate_and_publishes(tmp_path):
    calls = []

    def samples():
        calls.append("samples")
        return [_sample(0), _sample(1)]

    def predict(sample):
        return BenchmarkPrediction(
            predicted_audio=sample.target_audio + 0.01,
            rendered_rgb=sample.target_rgb + 0.01,
        )

    identity = EvaluationIdentity(
        scene_id="scene1_opera",
        system_name="joint_conditioned",
        reporting_step=30_000,
        expected_sample_ids=(
            "scene1_opera/cam38/000000",
            "scene1_opera/cam38/000001",
        ),
        expected_sample_count=2,
    )
    result = BenchmarkEvaluator(_loss, "cpu", strict_protocol=False).evaluate(
        identity=identity,
        evidence=_evidence("joint_conditioned"),
        sample_factory=samples,
        predictor=predict,
        output_dir=tmp_path,
    )
    assert calls == ["samples"]
    assert result.count == 2
    assert set(result.summary) == {
        "audio_total",
        "audio_mono",
        "audio_diff",
        "waveform_l1",
        "mono_lsd",
        "diff_lsd",
        "lre_error_db",
        "rgb_psnr",
        "rgb_ssim",
        "rgb_l1",
    }
    assert all(set(stats) == {"mean", "std", "median"} for stats in result.summary.values())
    assert (tmp_path / "current.json").is_file()
    assert load_evaluation(tmp_path, identity=identity).rows == result.rows


def test_resume_does_not_construct_test_samples_and_tamper_fails(tmp_path):
    identity = EvaluationIdentity(
        "scene1_opera",
        "audio_only",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    evaluator = BenchmarkEvaluator(_loss, "cpu", strict_protocol=False)
    evaluator.evaluate(
        identity=identity,
        evidence=_evidence("audio_only"),
        sample_factory=lambda: [_sample(0)],
        predictor=lambda sample: BenchmarkPrediction(
            sample.target_audio + 0.001, sample.target_rgb + 0.001
        ),
        output_dir=tmp_path,
    )
    resumed = evaluator.evaluate(
        identity=identity,
        evidence=_evidence("audio_only"),
        sample_factory=lambda: pytest.fail("resume constructed cam38"),
        predictor=lambda _: pytest.fail("resume rendered cam38"),
        output_dir=tmp_path,
        resume=True,
    )
    assert resumed.count == 1
    pointer = json.loads((tmp_path / "current.json").read_text())
    rows = tmp_path / pointer["generation"] / "metrics_per_sample.jsonl"
    rows.write_text(rows.read_text() + "{}\n")
    with pytest.raises(BenchmarkEvaluationError, match="hash"):
        load_evaluation(tmp_path, identity=identity)


def test_gate_rejects_test_leak_and_wrong_step_before_sample_construction(tmp_path):
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        10_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    evidence = _evidence("joint_conditioned", 10_000)
    evidence = TrainingEvidence(
        **{**evidence.__dict__, "test_targets_read_during_training": True}
    )
    called = False

    def samples():
        nonlocal called
        called = True
        return [_sample(0)]

    with pytest.raises(BenchmarkEvaluationError, match="test target"):
        BenchmarkEvaluator(_loss, "cpu", strict_protocol=False).evaluate(
            identity=identity,
            evidence=evidence,
            sample_factory=samples,
            predictor=lambda _: None,
            output_dir=tmp_path,
        )
    assert not called


def test_native_audio_reference_allows_audio_only_and_labels_non_update_matched(tmp_path):
    native = TrainingEvidence(
        **{
            **_evidence("native_audiogs").__dict__,
            "role": "native_reference",
            "planned_updates": 61,
            "completed_updates": 61,
            "checkpoint_step": 61,
            "index_sha256": None,
            "epochs": 61.0,
        }
    )
    identity = EvaluationIdentity(
        "scene1_opera",
        "native_audiogs",
        None,
        ("scene1_opera/cam38/000000",),
        1,
    )
    result = BenchmarkEvaluator(_loss, "cpu", strict_protocol=False).evaluate(
        identity=identity,
        evidence=native,
        sample_factory=lambda: [_sample(0)],
        predictor=lambda sample: BenchmarkPrediction(
            predicted_audio=sample.target_audio + 0.001
        ),
        output_dir=tmp_path,
    )
    assert set(result.summary) == {
        "audio_total",
        "audio_mono",
        "audio_diff",
        "waveform_l1",
        "mono_lsd",
        "diff_lsd",
        "lre_error_db",
    }
    assert result.provenance["update_matched"] is False


def test_overwrite_publishes_new_generation_and_verify_binds_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    evidence = TrainingEvidence(
        **{
            **_evidence("audio_only").__dict__,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        }
    )
    identity = EvaluationIdentity(
        "scene1_opera",
        "audio_only",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    evaluator = BenchmarkEvaluator(_loss, "cpu", strict_protocol=False)
    arguments = {
        "identity": identity,
        "evidence": evidence,
        "sample_factory": lambda: [_sample(0)],
        "predictor": lambda sample: BenchmarkPrediction(
            sample.target_audio + 0.001, sample.target_rgb + 0.001
        ),
        "output_dir": tmp_path / "evaluation",
    }
    first = evaluator.evaluate(**arguments)
    second = evaluator.evaluate(**arguments, overwrite=True)
    assert first.generation_path != second.generation_path
    assert first.generation_path.is_dir()
    assert verify_evaluation(tmp_path / "evaluation").count == 1
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(BenchmarkEvaluationError, match="checkpoint hash"):
        verify_evaluation(tmp_path / "evaluation")

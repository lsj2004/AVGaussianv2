from __future__ import annotations

import hashlib
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import avgaussianv2.benchmark.evaluation as evaluation_module
from avgaussianv2.benchmark.evaluation import (
    ALL_METRICS,
    AUDIO_METRICS,
    VIDEO_METRICS,
    BenchmarkEvaluationError,
    BenchmarkEvaluationRuntime,
    BenchmarkEvaluator,
    BenchmarkPrediction,
    EvaluationIdentity,
    TrainingEvidence,
    audit_training_evidence,
    load_evaluation,
    verify_evaluation,
)
from avgaussianv2.benchmark.artifacts import ArtifactError, publish_generation
from avgaussianv2.benchmark.output import BenchmarkOutputLock
from avgaussianv2.benchmark.training import (
    ALGORITHM,
    SCHEMA,
    SCHEMA_VERSION,
    BenchmarkCompatibility,
    CheckpointIO,
    hash_shared_indices,
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


def _runtime(samples):
    return BenchmarkEvaluationRuntime(samples=samples, audio_loss_fn=_loss)


def _hold_artifact_lock(path, ready, release):
    with BenchmarkOutputLock(path):
        ready.set()
        release.wait(10)


def _task12_evidence(tmp_path, *, step=5_000, paused=False):
    output = tmp_path / "worker"
    milestones = output / "milestones"
    milestones.mkdir(parents=True)
    (output / ".benchmark.lock").write_bytes(b"")
    indices = [0] * 30_000
    compatibility = BenchmarkCompatibility(
        scene_id="scene1_opera",
        mode="joint_conditioned",
        train_cameras=tuple(f"cam{x:02d}" for x in range(38)),
        test_camera="cam38",
        seed=42,
        index_sha256=hash_shared_indices(indices),
        visual_initialization_sha256=_sha("visual"),
        audio_initialization_sha256=_sha("audio"),
        model_initialization_sha256=_sha("model"),
        source_sha256=_sha("source"),
        config_sha256=_sha("config"),
    )
    inputs = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "config": {
            "main_updates": 30_000,
            "conditioner_warmup_steps": 2_000,
            "checkpoint_every": 500,
            "journal_every": 10,
            "milestones": [5_000, 10_000, 30_000],
            "seed": 42,
            "batch_size": 1,
            "selection": "final",
        },
        "compatibility": compatibility.to_mapping(),
        "train_config": {},
        "model_class": "tests.Tiny",
        "model_format_version": "tiny-v1",
    }
    encoded = json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    fingerprint = {"sha256": hashlib.sha256(encoded).hexdigest(), "inputs": inputs}
    contract = {
        "schema": f"{SCHEMA}.contract",
        "version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "compatibility": compatibility.to_mapping(),
        "shared_indices": indices,
        "selection": "final",
        "milestones": [5_000, 10_000, 30_000],
    }
    (output / "contract.json").write_text(json.dumps(contract))
    runtime = {
        "schema": "avgaussianv2.cam38-production-train-only-runtime",
        "version": 1,
        "include_eval": False,
        "train_cameras": [f"cam{x:02d}" for x in range(38)],
        "test_camera": "cam38",
        "dataset_identity_sha256": _sha("dataset"),
        "dataset_sample_ids_sha256": _sha("sample-ids"),
        "config_sha256": compatibility.config_sha256,
        "source_sha256": compatibility.source_sha256,
        "visual_initialization_sha256": compatibility.visual_initialization_sha256,
        "audio_initialization_sha256": compatibility.audio_initialization_sha256,
        "model_initialization_sha256": compatibility.model_initialization_sha256,
    }
    runtime_path = output / "runtime_contract.json"
    runtime_path.write_text(json.dumps(runtime))
    hashes = {}
    for milestone_step in (5_000, 10_000, 30_000):
        payload = {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "compatibility": compatibility.to_mapping(),
            "stage": "main",
            "warmup_step": 2_000,
            "main_step": milestone_step,
            "model": {},
            "optimizer": {},
            "visual_anchor": {},
            "torch_rng_state": torch.zeros(1, dtype=torch.uint8),
            "cuda_rng_states": [],
            "python_rng_state": (),
            "numpy_rng_state": {},
            "io": {},
            "gradient_guard": {},
        }
        path = milestones / f"step_{milestone_step:06d}.pt"
        torch.save(payload, path)
        hashes[f"milestones/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if milestone_step == 30_000:
            (output / "final.pt").write_bytes(path.read_bytes())
            hashes["final.pt"] = hashes[f"milestones/{path.name}"]
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "schema": f"{SCHEMA}.artifacts",
                "version": SCHEMA_VERSION,
                "fingerprint_sha256": fingerprint["sha256"],
                "sha256": hashes,
            }
        )
    )
    if paused:
        if step != 5_000:
            raise ValueError("paused Task12 fixture supports only step 5000")
        for stale_step in (10_000, 30_000):
            (milestones / f"step_{stale_step:06d}.pt").unlink()
        (output / "final.pt").unlink()
        (output / "artifact_hashes.json").unlink()
        milestone_name = "milestones/step_005000.pt"
        (output / "artifact_journal.json").write_text(
            json.dumps(
                {
                    "schema": f"{SCHEMA}.artifact-journal",
                    "version": SCHEMA_VERSION,
                    "fingerprint_sha256": fingerprint["sha256"],
                    "sha256": {milestone_name: hashes[milestone_name]},
                }
            )
        )
        checkpoints = output / "checkpoints"
        checkpoints.mkdir()
        rolling = checkpoints / "main_step_005000.pt"
        rolling.write_bytes((milestones / "step_005000.pt").read_bytes())
        rolling_sha256 = hashlib.sha256(rolling.read_bytes()).hexdigest()
        (output / "checkpoint_io.json").write_text(
            json.dumps(
                {
                    "schema": f"{SCHEMA}.checkpoint-io",
                    "version": SCHEMA_VERSION,
                    "fingerprint_sha256": fingerprint["sha256"],
                    "io": CheckpointIO().to_mapping(),
                    "committed_checkpoints": {rolling.name: rolling_sha256},
                }
            )
        )
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "schema": f"{SCHEMA}.progress",
                    "version": SCHEMA_VERSION,
                    "stage": "main",
                    "observed_warmup_step": 2_000,
                    "observed_main_step": 5_000,
                    "exact_warmup_step": 2_000,
                    "exact_main_step": 5_000,
                    "maximum_replay_updates": 500,
                    "fingerprint_sha256": fingerprint["sha256"],
                }
            )
        )
    checkpoint = milestones / f"step_{step:06d}.pt"
    return TrainingEvidence(
        system_name="joint_conditioned",
        scene_id="scene1_opera",
        role="continuation",
        train_cameras=compatibility.train_cameras,
        test_camera="cam38",
        test_targets_read_during_training=False,
        seed=42,
        planned_updates=30_000,
        completed_updates=step,
        checkpoint_step=step,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=hashes[f"milestones/{checkpoint.name}"],
        config_sha256=compatibility.config_sha256,
        source_sha256=compatibility.source_sha256,
        visual_initialization_sha256=compatibility.visual_initialization_sha256,
        audio_initialization_sha256=compatibility.audio_initialization_sha256,
        model_initialization_sha256=compatibility.model_initialization_sha256,
        index_sha256=compatibility.index_sha256,
        batch_size=1,
        epochs=None,
        training_output_dir=str(output),
        runtime_contract_path=str(runtime_path),
        runtime_contract_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
    )


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
    result = BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
        identity=identity,
        evidence=_evidence("joint_conditioned"),
        runtime_factory=lambda: _runtime(samples()),
        predictor_factory=lambda _: predict,
        output_dir=tmp_path,
    )
    assert calls == ["samples"]
    assert result.count == 2
    assert set(result.summary) == set(ALL_METRICS)
    assert all(set(stats) == {"mean", "std", "median"} for stats in result.summary.values())
    assert (tmp_path / "current.json").is_file()
    assert load_evaluation(tmp_path, identity=identity).rows == result.rows


def test_evaluator_keeps_context_managed_predictor_open_through_forwards(tmp_path):
    events = []

    class Predictor:
        def __enter__(self):
            events.append("enter")
            return self

        def __call__(self, sample):
            events.append(f"predict-{sample.frame_index}")
            return BenchmarkPrediction(
                predicted_audio=sample.target_audio + 0.01,
                rendered_rgb=sample.target_rgb + 0.01,
            )

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")
            return False

    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        30_000,
        (
            "scene1_opera/cam38/000000",
            "scene1_opera/cam38/000001",
        ),
        2,
    )
    BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
        identity=identity,
        evidence=_evidence("joint_conditioned"),
        runtime_factory=lambda: _runtime([_sample(0), _sample(1)]),
        predictor_factory=lambda _: Predictor(),
        output_dir=tmp_path,
    )

    assert events == ["enter", "predict-0", "predict-1", "exit"]


def test_resume_does_not_construct_test_samples_and_tamper_fails(tmp_path):
    identity = EvaluationIdentity(
        "scene1_opera",
        "audio_only",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    evaluator = BenchmarkEvaluator("cpu", strict_protocol=False)
    evaluator.evaluate(
        identity=identity,
        evidence=_evidence("audio_only"),
        runtime_factory=lambda: _runtime([_sample(0)]),
        predictor_factory=lambda _: lambda sample: BenchmarkPrediction(
            predicted_audio=sample.target_audio + 0.001
        ),
        output_dir=tmp_path,
    )
    lock_before = (tmp_path / ".benchmark.lock").read_bytes()
    lock_mtime = (tmp_path / ".benchmark.lock").stat().st_mtime_ns
    pointer_mtime = (tmp_path / "current.json").stat().st_mtime_ns
    resumed = evaluator.evaluate(
        identity=identity,
        evidence=_evidence("audio_only"),
        runtime_factory=lambda: pytest.fail("resume constructed cam38"),
        predictor_factory=lambda _: pytest.fail("resume rendered cam38"),
        output_dir=tmp_path,
        resume=True,
    )
    assert resumed.count == 1
    assert (tmp_path / ".benchmark.lock").read_bytes() == lock_before
    assert (tmp_path / ".benchmark.lock").stat().st_mtime_ns == lock_mtime
    assert (tmp_path / "current.json").stat().st_mtime_ns == pointer_mtime
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
        BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
            identity=identity,
            evidence=evidence,
            runtime_factory=lambda: _runtime(samples()),
            predictor_factory=lambda _: lambda _: None,
            output_dir=tmp_path,
        )
    assert not called


def test_training_evidence_accepts_nonnegative_multiseed_run() -> None:
    identity = EvaluationIdentity(
        "scene1_opera",
        "audio_only",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    replace(_evidence("audio_only"), seed=73).validate(identity)


def test_native_audio_reference_allows_audio_only_and_labels_non_update_matched(tmp_path):
    native = TrainingEvidence(
        **{
            **_evidence("native_audiogs").__dict__,
            "role": "native_reference",
            "planned_updates": 2_318,
            "completed_updates": 2_318,
            "checkpoint_step": 2_318,
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
    result = BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
        identity=identity,
        evidence=native,
        runtime_factory=lambda: _runtime([_sample(0)]),
        predictor_factory=lambda _: lambda sample: BenchmarkPrediction(
            predicted_audio=sample.target_audio + 0.001
        ),
        output_dir=tmp_path,
    )
    assert set(result.summary) == set(AUDIO_METRICS)
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
    evaluator = BenchmarkEvaluator("cpu", strict_protocol=False)
    arguments = {
        "identity": identity,
        "evidence": evidence,
        "runtime_factory": lambda: _runtime([_sample(0)]),
        "predictor_factory": lambda _: lambda sample: BenchmarkPrediction(
            predicted_audio=sample.target_audio + 0.001
        ),
        "output_dir": tmp_path / "evaluation",
    }
    first = evaluator.evaluate(**arguments)
    second = evaluator.evaluate(**arguments, overwrite=True)
    assert first.generation_path != second.generation_path
    assert first.generation_path.is_dir()
    assert (
        verify_evaluation(
            tmp_path / "evaluation", strict_training_evidence=False
        ).count
        == 1
    )
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(BenchmarkEvaluationError, match="checkpoint hash"):
        verify_evaluation(
            tmp_path / "evaluation", strict_training_evidence=False
        )


def test_strict_task12_contract_adapter_accepts_real_checkpoint_layout(tmp_path):
    evidence = _task12_evidence(tmp_path)
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        5_000,
        tuple(f"scene1_opera/cam38/{index:06d}" for index in range(130)),
        130,
    )
    audit_training_evidence(evidence, identity)


def test_strict_task12_accepts_audited_paused_milestone_without_final(tmp_path):
    evidence = _task12_evidence(tmp_path, paused=True)
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        5_000,
        tuple(f"scene1_opera/cam38/{index:06d}" for index in range(130)),
        130,
    )

    audit_training_evidence(evidence, identity)
    assert not (Path(evidence.training_output_dir) / "final.pt").exists()

    journal = Path(evidence.training_output_dir) / "artifact_journal.json"
    value = json.loads(journal.read_text())
    value["sha256"] = {}
    journal.write_text(json.dumps(value))
    with pytest.raises(BenchmarkEvaluationError, match="transaction hash|manifest"):
        audit_training_evidence(evidence, identity)


def test_cross_attention_causal_label_reuses_joint_training_contract(tmp_path):
    evidence = replace(
        _task12_evidence(tmp_path),
        system_name="cross_attention_no_gaussians",
    )
    identity = EvaluationIdentity(
        "scene1_opera",
        "cross_attention_no_gaussians",
        5_000,
        tuple(f"scene1_opera/cam38/{index:06d}" for index in range(130)),
        130,
    )

    audit_training_evidence(evidence, identity)


def test_strict_task12_audit_rejects_concurrent_writer_before_runtime(tmp_path):
    evidence = _task12_evidence(tmp_path)
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        5_000,
        tuple(f"scene1_opera/cam38/{index:06d}" for index in range(130)),
        130,
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_artifact_lock,
        args=(evidence.training_output_dir, ready, release),
    )
    process.start()
    assert ready.wait(10)
    try:
        with pytest.raises(BenchmarkEvaluationError, match="locked"):
            audit_training_evidence(evidence, identity)
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0


def test_strict_task12_audit_detects_rename_after_pinned_reads(
    tmp_path, monkeypatch
):
    evidence = _task12_evidence(tmp_path)
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        5_000,
        tuple(f"scene1_opera/cam38/{index:06d}" for index in range(130)),
        130,
    )
    output = Path(evidence.training_output_dir)
    moved = tmp_path / "moved-worker"
    original = evaluation_module._load_exact_json
    renamed = False

    def rename_after_contract(path, fields, name):
        nonlocal renamed
        value = original(path, fields, name)
        if name == "Task12 contract" and not renamed:
            output.rename(moved)
            renamed = True
        return value

    monkeypatch.setattr(
        evaluation_module, "_load_exact_json", rename_after_contract
    )
    with pytest.raises(BenchmarkEvaluationError, match="changed|disappeared"):
        audit_training_evidence(evidence, identity)
    assert renamed


def test_strict_task12_audit_rejects_symlinked_output_component(tmp_path):
    evidence = _task12_evidence(tmp_path)
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        5_000,
        tuple(f"scene1_opera/cam38/{index:06d}" for index in range(130)),
        130,
    )
    link = tmp_path / "worker-link"
    link.symlink_to(Path(evidence.training_output_dir), target_is_directory=True)
    checkpoint = link / "milestones" / "step_005000.pt"
    linked = TrainingEvidence(
        **{
            **evidence.__dict__,
            "training_output_dir": str(link),
            "runtime_contract_path": str(link / "runtime_contract.json"),
            "checkpoint_path": str(checkpoint),
        }
    )
    with pytest.raises(BenchmarkEvaluationError, match="symlink"):
        audit_training_evidence(linked, identity)


def test_strict_evidence_failure_constructs_no_runtime_predictor_or_loss(tmp_path):
    evidence = _task12_evidence(tmp_path)
    runtime_path = tmp_path / "worker" / "runtime_contract.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["include_eval"] = True
    runtime_path.write_text(json.dumps(runtime))
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        5_000,
        tuple(f"scene1_opera/cam38/{index:06d}" for index in range(130)),
        130,
    )
    calls = {"runtime": 0, "predictor": 0, "loss": 0}

    def runtime_factory():
        calls["runtime"] += 1

        def loss(*_):
            calls["loss"] += 1

        return BenchmarkEvaluationRuntime([], loss)

    def predictor_factory(_):
        calls["predictor"] += 1

    with pytest.raises(BenchmarkEvaluationError, match="runtime contract"):
        BenchmarkEvaluator("cpu").evaluate(
            identity=identity,
            evidence=evidence,
            runtime_factory=runtime_factory,
            predictor_factory=predictor_factory,
            output_dir=tmp_path / "evaluation",
        )
    assert calls == {"runtime": 0, "predictor": 0, "loss": 0}


@pytest.mark.parametrize(
    "name",
    [
        "sample_id",
        "rgb_lpips",
        "depth_accuracy",
        "audio_total",
    ],
)
def test_extra_metric_registry_rejects_reserved_and_depth_names(tmp_path, name):
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    runtime = BenchmarkEvaluationRuntime(
        samples=[_sample(0)],
        audio_loss_fn=_loss,
        extra_metric_fns={name: lambda *_: 0.0},
        extra_metric_directions={name: "lower_is_better"},
        extra_metric_modalities={name: "audio"},
    )
    with pytest.raises(BenchmarkEvaluationError, match="reserved"):
        BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
            identity=identity,
            evidence=_evidence("joint_conditioned"),
            runtime_factory=lambda: runtime,
            predictor_factory=lambda _: lambda sample: BenchmarkPrediction(
                sample.target_audio + 0.001, sample.target_rgb + 0.001
            ),
            output_dir=tmp_path,
        )


def test_registered_extra_metric_persists_explicit_direction_and_modality(tmp_path):
    identity = EvaluationIdentity(
        "scene1_opera",
        "joint_conditioned",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    runtime = BenchmarkEvaluationRuntime(
        samples=[_sample(0)],
        audio_loss_fn=_loss,
        extra_metric_fns={"audio_custom": lambda *_: 0.25},
        extra_metric_directions={"audio_custom": "higher_is_better"},
        extra_metric_modalities={"audio_custom": "audio"},
    )
    result = BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
        identity=identity,
        evidence=_evidence("joint_conditioned"),
        runtime_factory=lambda: runtime,
        predictor_factory=lambda _: lambda sample: BenchmarkPrediction(
            sample.target_audio + 0.001, sample.target_rgb + 0.001
        ),
        output_dir=tmp_path,
    )
    assert result.metric_directions["audio_custom"] == "higher_is_better"
    assert result.metric_protocol["extra_metric_registry"]["audio_custom"] == {
        "direction": "higher_is_better",
        "modality": "audio",
        "protocol": None,
    }


def test_perfect_rgb_uses_documented_finite_psnr_cap(tmp_path):
    identity = EvaluationIdentity(
        "scene1_opera",
        "visual_only",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    result = BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
        identity=identity,
        evidence=_evidence("visual_only"),
        runtime_factory=lambda: _runtime([_sample(0)]),
        predictor_factory=lambda _: lambda sample: BenchmarkPrediction(
            rendered_rgb=sample.target_rgb
        ),
        output_dir=tmp_path,
    )
    assert result.rows[0]["rgb_psnr"] == 100.0
    assert result.metric_protocol["psnr_cap_db"] == 100.0


def test_continuation_rejects_unexpected_audio_for_visual_only(tmp_path):
    identity = EvaluationIdentity(
        "scene1_opera",
        "visual_only",
        30_000,
        ("scene1_opera/cam38/000000",),
        1,
    )
    with pytest.raises(BenchmarkEvaluationError, match="exact required metric"):
        BenchmarkEvaluator("cpu", strict_protocol=False).evaluate(
            identity=identity,
            evidence=_evidence("visual_only"),
            runtime_factory=lambda: _runtime([_sample(0)]),
            predictor_factory=lambda _: lambda sample: BenchmarkPrediction(
                predicted_audio=sample.target_audio + 0.001,
                rendered_rgb=sample.target_rgb + 0.001,
            ),
            output_dir=tmp_path,
        )


def test_artifact_publisher_rejects_cross_process_writer_and_symlink_parent(tmp_path):
    output = tmp_path / "locked"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_artifact_lock,
        args=(str(output), ready, release),
    )
    process.start()
    assert ready.wait(10)
    try:
        with pytest.raises(ArtifactError, match="locked"):
            publish_generation(
                output,
                schema="test",
                files={"value.json": b"{}\n"},
                identity={"value": 1},
            )
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0

    parent = tmp_path / "parent-link"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        publish_generation(
            parent / "output",
            schema="test",
            files={"value.json": b"{}\n"},
            identity={"value": 1},
        )

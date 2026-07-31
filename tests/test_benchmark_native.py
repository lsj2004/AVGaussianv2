from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

import avgaussianv2.benchmark.evaluation as evaluation_module
import avgaussianv2.benchmark.native as native_module
from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluationRuntime,
    BenchmarkEvaluator,
    BenchmarkPrediction,
    EvaluationIdentity,
    TrainingEvidence,
    audit_training_evidence,
    verify_evaluation,
)
from avgaussianv2.benchmark.native import (
    NativeContractError,
    _checkpoint_metadata,
    finalize_native_contract,
    verify_native_contract,
    write_audiogs_seed_record,
)
from avgaussianv2.benchmark.report import (
    build_scene_report,
    build_suite_report,
    verify_suite_report,
)
from avgaussianv2.contracts import AlignedAVSample


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


FTGSPP_GAUSSIAN_PARAMETERS = {
    "means",
    "scales",
    "quats",
    "opacities",
    "sh_0",
    "sh_n",
    "times",
    "durations",
    "marginal_gates",
}
FTGSPP_GAUSSIAN_GLOBALS = {
    "ftgspp.models.gaussians Gaussians",
    "torch._utils _rebuild_parameter",
    "torch._utils _rebuild_tensor_v2",
    "torch FloatStorage",
    "collections OrderedDict",
    "__builtin__ set",
}


def _save_ftgspp_gaussians_fixture(path: Path) -> None:
    module_names = ("ftgspp", "ftgspp.models", "ftgspp.models.gaussians")
    previous = {name: sys.modules.get(name) for name in module_names}
    package = types.ModuleType("ftgspp")
    package.__path__ = []
    models = types.ModuleType("ftgspp.models")
    models.__path__ = []
    gaussians = types.ModuleType("ftgspp.models.gaussians")
    cls = type(
        "Gaussians",
        (torch.nn.Module,),
        {"__module__": "ftgspp.models.gaussians"},
    )
    gaussians.Gaussians = cls
    package.models = models
    models.gaussians = gaussians
    sys.modules.update(
        {
            "ftgspp": package,
            "ftgspp.models": models,
            "ftgspp.models.gaussians": gaussians,
        }
    )
    try:
        value = cls()
        for name in (*sorted(FTGSPP_GAUSSIAN_PARAMETERS), "velocity_model"):
            setattr(value, name, torch.nn.Parameter(torch.ones(1)))
        value.max_duration = 1.0
        value.sh_degree = 3
        torch.save(value, path)
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_ftgspp_checkpoint_requires_real_pickle_global_and_exact_state_schema() -> None:
    metadata = _checkpoint_metadata(
        {"sh_degree": 3},
        {
            *FTGSPP_GAUSSIAN_PARAMETERS,
            "velocity_model",
            "max_duration",
            "sh_degree",
        },
        FTGSPP_GAUSSIAN_GLOBALS,
        model_kind="ftgspp",
        scene_id="scene1_opera",
    )
    assert metadata["model_class"] == "ftgspp.models.gaussians.Gaussians"

    with pytest.raises(NativeContractError, match="schema mismatch"):
        _checkpoint_metadata(
            {"sh_degree": 3},
            {
                *FTGSPP_GAUSSIAN_PARAMETERS,
                "velocity_model",
                "max_duration",
                "sh_degree",
                # A Unicode string must not impersonate a pickle GLOBAL opcode.
                "ftgspp.models.gaussians Gaussians",
            },
            FTGSPP_GAUSSIAN_GLOBALS - {"ftgspp.models.gaussians Gaussians"},
            model_kind="ftgspp",
            scene_id="scene1_opera",
        )

    for globals_, strings in (
        (
            FTGSPP_GAUSSIAN_GLOBALS
            - {"ftgspp.models.gaussians Gaussians"}
            | {"attacker.models Gaussians"},
            {
                *FTGSPP_GAUSSIAN_PARAMETERS,
                "velocity_model",
                "max_duration",
                "sh_degree",
            },
        ),
        (
            FTGSPP_GAUSSIAN_GLOBALS,
            {
                *(FTGSPP_GAUSSIAN_PARAMETERS - {"means"}),
                "velocity_model",
                "max_duration",
                "sh_degree",
            },
        ),
    ):
        with pytest.raises(NativeContractError, match="schema mismatch"):
            _checkpoint_metadata(
                {"sh_degree": 3},
                strings,
                globals_,
                model_kind="ftgspp",
                scene_id="scene1_opera",
            )


def _install_native_audit_stubs(monkeypatch):
    protocols: dict[str, dict[str, object]] = {}
    flow_roots: dict[str, Path] = {}

    def protocol(path):
        return protocols[str(Path(path).resolve())]

    def conversion(payload, **_):
        clips = int(payload["num_clips"])
        return {"resolved_updates": clips * 38 * 61, "num_clips": clips}

    def upstream_config(rendered, **_):
        root = flow_roots[str(Path(rendered).resolve())]
        return {"namespaces": [str(root)], "iterations": 30_000}

    def seed_record(path, **_):
        name = path["stage"] if isinstance(path, dict) else Path(path).stem
        argv = (
            ["run", "--from", "extract", "--to", "prep"]
            if name == "prep"
            else ["run", "--from", "points", "--to", "train"]
            if name == "train"
            else ["flow"]
        )
        return {"argv": argv, "seed": 42}

    monkeypatch.setattr(native_module, "audit_protocol_config", protocol)
    monkeypatch.setattr(
        native_module, "audit_initialization_provenance", lambda *_, **__: None
    )
    monkeypatch.setattr(native_module, "audit_audiogs_conversion", conversion)
    monkeypatch.setattr(native_module, "audit_ftgspp_upstream_config", upstream_config)
    monkeypatch.setattr(native_module, "audit_ftgspp_seed_record", seed_record)
    monkeypatch.setattr(
        native_module,
        "audit_ftgspp_flow_cache",
        lambda *_, **__: {
            "pairs": 0,
            "cameras": 38,
            "files": 0,
            "complete": True,
        },
    )
    monkeypatch.setattr(
        native_module, "_source_files", lambda _, root: (Path(root) / "source.py",)
    )
    monkeypatch.setattr(
        native_module,
        "_git_identity",
        lambda root: {
            "root": str(Path(root).resolve()),
            "commit": "a" * 40,
            "tree": "b" * 40,
            "tracked_worktree_status_sha256": _sha_bytes(b""),
        },
    )
    return protocols, flow_roots


def _make_native_contract(
    tmp_path: Path,
    *,
    scene: str,
    kind: str,
    protocols,
    flow_roots,
):
    root = tmp_path / f"{scene}-{kind}"
    upstream = root / "upstream"
    upstream.mkdir(parents=True)
    (upstream / "source.py").write_text("source = 1\n")
    config = root / "config.yaml"
    config.write_text("{}\n")
    provenance = root / "provenance.json"
    provenance.write_text(
        json.dumps(
            {"scene_id": scene, "test_camera": "cam38", "assets": []}
        )
        + "\n"
    )
    checkpoint = root / "checkpoint.pt"
    if kind == "audiogs":
        updates = 2_318 if scene == "scene1_opera" else 6_954
        torch.save(
            {
                "epoch": 60,
                "iter": updates,
                "max_epoch": 61,
                "test_viewpoint": 39,
                "model_file": "audio_3dgs_mono_diff_gs_only",
                "model_state_dict": {"weight": torch.ones(1)},
            },
            checkpoint,
        )
    else:
        _save_ftgspp_gaussians_fixture(checkpoint)
    protocols[str(config.resolve())] = {
        "scene": {
            "id": scene,
            "train_cameras": [f"cam{x:02d}" for x in range(38)],
            "eval_cameras": ["cam38"],
        },
        "paths": {
            "audio_checkpoint": str(checkpoint),
            "visual_checkpoint": str(checkpoint),
            "audio_upstream_root": str(upstream),
            "visual_upstream_root": str(upstream),
        },
        "benchmark": {
            "expected_test_samples": 130 if scene == "scene1_opera" else 293,
            "protocol": "dual_dataset_cam38_v1",
            "seed": 42,
            "native_budgets": {
                "audiogs_epochs": 61,
                "audiogs_batch_size": 1,
                "audiogs_resolved_updates": (
                    2_318 if scene == "scene1_opera" else 6_954
                ),
                "ftgspp_updates": 30_000,
                "ftgspp_batch_size": 1,
            },
        },
    }
    config.write_text(json.dumps(protocols[str(config.resolve())]))
    contract_path = root / "native_contract"
    kwargs = {}
    if kind == "audiogs":
        conversion = root / "conversion.json"
        conversion.write_text(
            json.dumps(
                {
                    "num_clips": 1 if scene == "scene1_opera" else 3,
                    "audio_root": str(root / "audio"),
                    "cameras_npz": str(root / "cameras.npz"),
                    "output_root": str(root / "converted"),
                }
            )
        )
        seed = root / "seed.json"
        write_audiogs_seed_record(
            seed,
            scene_id=scene,
            upstream_scene=native_module.EXPECTED[scene]["audio_scene"],
        )
        kwargs = {"conversion_manifest": conversion, "seed_records": [seed]}
    else:
        rendered = root / "rendered.toml"
        rendered.write_text(
            "[data]\n"
            f'frames = {{ start = 0, stop = {130 if scene == "scene1_opera" else 293} }}\n'
            "eval_cameras = [37]\n"
            "train_cameras = { start = 0, stop = 38 }\n"
            "[init]\n"
            "temporal_flow_cameras = { start = 0, stop = 38 }\n"
            "keyframe_stride = 10\n"
            "temporal_motion_adapted = true\n"
            "[train]\n"
            "iterations = 30000\n"
            "batch_size = 1\n"
        )
        flow_root = root / "flow"
        flow_root.mkdir()
        flow_roots[str(rendered.resolve())] = flow_root
        train_log = root / "train.log"
        train_log.write_text("Starting training\nDone training\n")
        seeds = []
        for name in ("prep", "flow", "train"):
            seed = root / f"{name}.json"
            seed.write_text(json.dumps({"stage": name}) + "\n")
            seeds.append(seed)
        kwargs = {
            "rendered_config": rendered,
            "sampled_scene_root": root / "sampled",
            "train_log": train_log,
            "seed_records": seeds,
        }
    contract = finalize_native_contract(
        model_kind=kind,
        config_path=config,
        provenance_path=provenance,
        checkpoint_path=checkpoint,
        upstream_root=upstream,
        output_path=contract_path,
        **kwargs,
    )
    return contract_path, contract


@pytest.mark.parametrize("kind", ["audiogs", "ftgspp"])
def test_native_contract_finalizer_verifies_every_bound_byte(
    tmp_path, monkeypatch, kind
):
    protocols, flow_roots = _install_native_audit_stubs(monkeypatch)
    path, contract = _make_native_contract(
        tmp_path,
        scene="scene1_opera",
        kind=kind,
        protocols=protocols,
        flow_roots=flow_roots,
    )
    assert verify_native_contract(path) == contract
    source = Path(contract["inputs"]["source_audits"][0]["path"])
    source.write_text("source = 2\n")
    assert verify_native_contract(path) == contract
    pointer = json.loads((path / "current.json").read_text())
    snapshot = (
        path
        / pointer["generation"]
        / contract["inputs"]["source_audits"][0]["snapshot"]
    )
    snapshot.write_text("tampered snapshot\n")
    with pytest.raises(NativeContractError, match="hash|audit"):
        verify_native_contract(path)


def test_native_contract_rejects_non_torch_and_checkpoint_tamper(
    tmp_path, monkeypatch
):
    protocols, flow_roots = _install_native_audit_stubs(monkeypatch)
    path, contract = _make_native_contract(
        tmp_path,
        scene="scene1_opera",
        kind="audiogs",
        protocols=protocols,
        flow_roots=flow_roots,
    )
    checkpoint = Path(contract["checkpoint"]["path"])
    checkpoint.write_bytes(b"not a torch archive")
    with pytest.raises(NativeContractError, match="hash|Torch"):
        verify_native_contract(path)


def test_native_snapshot_and_checkpoint_replacement_races_are_rejected(
    tmp_path, monkeypatch
):
    protocols, flow_roots = _install_native_audit_stubs(monkeypatch)
    path, contract = _make_native_contract(
        tmp_path,
        scene="scene1_opera",
        kind="audiogs",
        protocols=protocols,
        flow_roots=flow_roots,
    )
    pointer = json.loads((path / "current.json").read_text())
    snapshot = (
        path
        / pointer["generation"]
        / contract["inputs"]["source_audits"][0]["snapshot"]
    )
    original_read = native_module._read_fd
    replaced = False

    def replace_snapshot_after_read(fd, label):
        nonlocal replaced
        data = original_read(fd, label)
        if "source_00.snapshot" in label and not replaced:
            replacement = snapshot.with_suffix(".replacement")
            replacement.write_bytes(data)
            replacement.replace(snapshot)
            replaced = True
        return data

    monkeypatch.setattr(native_module, "_read_fd", replace_snapshot_after_read)
    with pytest.raises(NativeContractError, match="identity changed"):
        verify_native_contract(path)
    assert replaced

    path, contract = _make_native_contract(
        tmp_path / "checkpoint-race",
        scene="scene1_opera",
        kind="audiogs",
        protocols=protocols,
        flow_roots=flow_roots,
    )
    monkeypatch.setattr(native_module, "_read_fd", original_read)
    checkpoint = Path(contract["checkpoint"]["path"])
    original_hash = native_module._hash_fd
    checkpoint_replaced = False

    def replace_checkpoint_after_hash(fd, label):
        nonlocal checkpoint_replaced
        digest = original_hash(fd, label)
        if label == "native checkpoint" and not checkpoint_replaced:
            replacement = checkpoint.with_suffix(".replacement")
            replacement.write_bytes(checkpoint.read_bytes())
            replacement.replace(checkpoint)
            checkpoint_replaced = True
        return digest

    monkeypatch.setattr(native_module, "_hash_fd", replace_checkpoint_after_hash)
    with pytest.raises(NativeContractError, match="identity changed"):
        verify_native_contract(path)
    assert checkpoint_replaced


def test_native_evidence_rejects_self_reported_initialization_hash(
    tmp_path, monkeypatch
):
    protocols, flow_roots = _install_native_audit_stubs(monkeypatch)
    path, contract = _make_native_contract(
        tmp_path,
        scene="scene1_opera",
        kind="audiogs",
        protocols=protocols,
        flow_roots=flow_roots,
    )
    evidence = _native_evidence(
        "scene1_opera", "native_audiogs", path, contract
    )
    evidence = TrainingEvidence(
        **{
            **evidence.__dict__,
            "visual_initialization_sha256": _sha_bytes(b"self-reported"),
        }
    )
    identity = EvaluationIdentity(
        "scene1_opera",
        "native_audiogs",
        None,
        ("scene1_opera/cam38/000000",),
        1,
    )
    with pytest.raises(
        evaluation_module.BenchmarkEvaluationError,
        match="contract/evaluation evidence mismatch",
    ):
        audit_training_evidence(evidence, identity)


def _sample(scene: str) -> AlignedAVSample:
    audio = torch.linspace(-0.2, 0.2, 640).repeat(1, 2, 1)
    return AlignedAVSample(
        scene_id=scene,
        camera="cam38",
        frame_index=0,
        time_seconds=0.0,
        visual_time=torch.zeros(1),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 3),
        source_audio=audio,
        target_audio=audio,
        target_rgb=torch.full((1, 4, 5, 3), 0.2),
        image_size=(4, 5),
    )


def _loss(predicted, target):
    error = (predicted - target).abs().mean()
    return {"total_loss": error, "mono_loss": error, "diff_loss": error}


def _native_evidence(scene, system, path, contract):
    audio = system == "native_audiogs"
    updates = (2_318 if scene == "scene1_opera" else 6_954) if audio else 30_000
    return TrainingEvidence(
        system_name=system,
        scene_id=scene,
        role="native_reference",
        train_cameras=tuple(f"cam{x:02d}" for x in range(38)),
        test_camera="cam38",
        test_targets_read_during_training=False,
        seed=42,
        planned_updates=updates,
        completed_updates=updates,
        checkpoint_step=updates,
        checkpoint_path=contract["checkpoint"]["path"],
        checkpoint_sha256=contract["checkpoint"]["sha256"],
        config_sha256=contract["inputs"]["protocol_config"]["sha256"],
        source_sha256=contract["upstream"]["source_sha256"],
        visual_initialization_sha256=contract["derived_initialization"][
            "visual_initialization_sha256"
        ],
        audio_initialization_sha256=contract["derived_initialization"][
            "audio_initialization_sha256"
        ],
        model_initialization_sha256=contract["derived_initialization"][
            "model_initialization_sha256"
        ],
        index_sha256=None,
        batch_size=1,
        epochs=61.0 if audio else None,
        native_contract_path=str(path.resolve()),
        native_contract_sha256=contract["_manifest_sha256"],
    )


def _continuation_evidence(scene, system, step):
    digest = _sha_bytes(b"shared")
    return TrainingEvidence(
        system_name=system,
        scene_id=scene,
        role="continuation",
        train_cameras=tuple(f"cam{x:02d}" for x in range(38)),
        test_camera="cam38",
        test_targets_read_during_training=False,
        seed=42,
        planned_updates=30_000,
        completed_updates=step,
        checkpoint_step=step,
        checkpoint_path=f"/tmp/{scene}-{system}-{step}.pt",
        checkpoint_sha256=_sha_bytes(f"{scene}-{system}-{step}".encode()),
        config_sha256=digest,
        source_sha256=digest,
        visual_initialization_sha256=digest,
        audio_initialization_sha256=digest,
        model_initialization_sha256=digest,
        index_sha256=digest,
        batch_size=1,
        epochs=None,
    )


def test_strict_native_evidence_to_eval_scene_and_suite_chain(
    tmp_path, monkeypatch
):
    protocols, flow_roots = _install_native_audit_stubs(monkeypatch)
    monkeypatch.setitem(evaluation_module.SCENE_SAMPLE_COUNTS, "scene1_opera", 1)
    monkeypatch.setitem(evaluation_module.SCENE_SAMPLE_COUNTS, "Scene7playing", 1)
    monkeypatch.setattr(
        evaluation_module, "_audit_continuation_evidence", lambda *_, **__: None
    )
    scene_reports = []
    for scene in ("scene1_opera", "Scene7playing"):
        sample = _sample(scene)
        evaluations = []
        for system in ("joint_conditioned", "audio_only", "visual_only"):
            for step in (5_000, 10_000, 30_000):
                identity = EvaluationIdentity(
                    scene, system, step, (f"{scene}/cam38/000000",), 1
                )
                output = tmp_path / "eval" / scene / system / str(step)
                output.parent.mkdir(parents=True, exist_ok=True)
                result = BenchmarkEvaluator("cpu").evaluate(
                    identity=identity,
                    evidence=_continuation_evidence(scene, system, step),
                    runtime_factory=lambda sample=sample: BenchmarkEvaluationRuntime(
                        [sample], _loss
                    ),
                        predictor_factory=lambda _, system=system: lambda value: BenchmarkPrediction(
                            predicted_audio=(
                                value.target_audio + 0.001
                                if system != "visual_only"
                                else None
                            ),
                            rendered_rgb=(
                                value.target_rgb + 0.001
                                if system != "audio_only"
                                else None
                            ),
                        ),
                    output_dir=output,
                )
                evaluations.append(result)
        for kind, system in (
            ("audiogs", "native_audiogs"),
            ("ftgspp", "native_ftgspp"),
        ):
            path, contract = _make_native_contract(
                tmp_path,
                scene=scene,
                kind=kind,
                protocols=protocols,
                flow_roots=flow_roots,
            )
            identity = EvaluationIdentity(
                scene, system, None, (f"{scene}/cam38/000000",), 1
            )
            native_output = tmp_path / "eval" / scene / system
            native_output.parent.mkdir(parents=True, exist_ok=True)
            result = BenchmarkEvaluator("cpu").evaluate(
                identity=identity,
                evidence=_native_evidence(scene, system, path, contract),
                runtime_factory=lambda sample=sample: BenchmarkEvaluationRuntime(
                    [sample], _loss
                ),
                predictor_factory=lambda _, system=system: (
                    lambda value: BenchmarkPrediction(
                        predicted_audio=value.target_audio + 0.001
                    )
                    if system == "native_audiogs"
                    else BenchmarkPrediction(rendered_rgb=value.target_rgb + 0.001)
                ),
                output_dir=native_output,
            )
            assert verify_evaluation(result.generation_path.parent.parent) == result
            evaluations.append(result)
        scene_output = tmp_path / "reports" / scene
        scene_output.parent.mkdir(parents=True, exist_ok=True)
        scene_reports.append(
            build_scene_report(
                scene_id=scene,
                evaluations=evaluations,
                expected_sample_count=1,
                output_dir=scene_output,
            )
        )
    suite_output = tmp_path / "reports" / "suite"
    build_suite_report(
        scene_reports=scene_reports,
        output_dir=suite_output,
    )
    assert verify_suite_report(suite_output)["scene_sample_counts"] == {
        "Scene7playing": 1,
        "scene1_opera": 1,
    }


def test_native_protocol_projection_ignores_continuation_only_fields(tmp_path):
    base = {
        "scene": {"id": "scene1_opera"},
        "paths": {"audio_checkpoint": "audio.pt"},
        "model": {"sample_rate": 16_000},
        "train": {"crop_seconds": 0.5, "seed": 42, "lambda_lre": 0.0},
        "benchmark": {
            "protocol": "dual_dataset_cam38_v1",
            "seed": 42,
            "continuation_updates": 30_000,
            "conditioner_warmup_steps": 2_000,
            "report_steps": [5_000, 10_000, 30_000],
        },
    }
    derived = json.loads(json.dumps(base))
    derived["train"].update(seed=73, lambda_lre=0.02)
    derived["benchmark"]["seed"] = 73
    assert native_module.native_protocol_projection_sha256(
        base, base_dir=tmp_path
    ) == native_module.native_protocol_projection_sha256(
        derived, base_dir=tmp_path
    )

    derived["model"]["sample_rate"] = 48_000
    assert native_module.native_protocol_projection_sha256(
        base, base_dir=tmp_path
    ) != native_module.native_protocol_projection_sha256(
        derived, base_dir=tmp_path
    )

    derived = json.loads(json.dumps(base))
    derived["train"]["crop_seconds"] = 1.0
    assert native_module.native_protocol_projection_sha256(
        base, base_dir=tmp_path
    ) != native_module.native_protocol_projection_sha256(
        derived, base_dir=tmp_path
    )

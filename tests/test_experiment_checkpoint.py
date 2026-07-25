from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotCompatibility,
    PilotResumeError,
    build_run_fingerprint,
    inspect_pilot_checkpoint,
    restore_pilot_checkpoint,
    save_pilot_checkpoint,
    hash_index_manifest,
    sha256_file,
    validate_compatibility,
)
from avgaussianv2.experiment.contracts import VariantIndices
from avgaussianv2.experiment.contracts import PilotConfig
from avgaussianv2.experiment.selection import BestSelector, EarlyStopper
from avgaussianv2.config import TrainConfig


class Dangerous:
    pass


def compatibility() -> PilotCompatibility:
    return PilotCompatibility(
        scene_id="scene1_opera",
        variant="joint_conditioned",
        seed=7,
        index_hash="1" * 64,
        visual_checkpoint_sha256="2" * 64,
        audio_checkpoint_sha256="3" * 64,
        camera_mapping_sha256="4" * 64,
        n_fft=512,
        hop_length=128,
        win_length=512,
        sample_rate=48_000,
    )


def test_sha256_file_hashes_streamed_content(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sha256_file(tmp_path / "missing")
    with pytest.raises(ValueError, match="regular file"):
        sha256_file(tmp_path)


def test_index_manifest_hash_is_canonical_and_order_sensitive() -> None:
    first = {"warmup": [2, 1], "joint": [3, 4]}
    reordered_keys = {"joint": [3, 4], "warmup": [2, 1]}
    changed_order = {"warmup": [1, 2], "joint": [3, 4]}
    assert hash_index_manifest(first) == hash_index_manifest(reordered_keys)
    assert hash_index_manifest(first) != hash_index_manifest(changed_order)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_id", "other"),
        ("variant", "frozen_visual"),
        ("seed", 8),
        ("index_hash", "a" * 64),
        ("visual_checkpoint_sha256", "b" * 64),
        ("audio_checkpoint_sha256", "c" * 64),
        ("camera_mapping_sha256", "d" * 64),
        ("n_fft", 1024),
        ("hop_length", 256),
        ("win_length", 1024),
        ("sample_rate", 44_100),
    ],
)
def test_validate_compatibility_rejects_every_mismatch(field: str, value: object) -> None:
    expected = compatibility()
    actual = replace(expected, **{field: value})
    with pytest.raises(PilotResumeError, match=field) as caught:
        validate_compatibility(actual, expected)
    assert "actual=" in str(caught.value)
    assert "expected=" in str(caught.value)


def test_compatibility_mapping_is_deterministic_and_strict() -> None:
    value = compatibility()
    assert PilotCompatibility.from_mapping(value.to_mapping()) == value
    assert list(value.to_mapping()) == [
        "scene_id",
        "variant",
        "seed",
        "index_hash",
        "visual_checkpoint_sha256",
        "audio_checkpoint_sha256",
        "camera_mapping_sha256",
        "n_fft",
        "hop_length",
        "win_length",
        "sample_rate",
    ]
    with pytest.raises(TypeError, match="seed"):
        replace(value, seed=True)
    with pytest.raises(ValueError, match="index_hash"):
        replace(value, index_hash="not-a-digest")
    payload = value.to_mapping()
    payload["extra"] = 1
    with pytest.raises(PilotResumeError, match="fields"):
        PilotCompatibility.from_mapping(payload)


def _checkpoint_kwargs(model, optimizer):
    selector = BestSelector(
        {"rgb_psnr": {"mean": 30.0}, "rgb_ssim": {"mean": 0.95}}, 0.5, 0.01
    )
    selector.consider(
        1,
        {
            "audio_total": {"mean": 1.0},
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
    )
    stopper = EarlyStopper(0, 2, 0.1)
    stopper.update(1, 1.0)
    fingerprint = build_run_fingerprint(
        pilot_config=PilotConfig(
            warmup_steps=2,
            joint_steps=3,
            validation_interval=1,
            minimum_joint_steps=0,
            patience=2,
            minimum_relative_improvement=0.1,
        ),
        train_config=TrainConfig(),
        visual_baseline={
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
        model=model,
        warmup_optimizer_factory=compatibility,
        joint_optimizer_factory=compatibility,
        warmup_step_fn=compatibility,
        joint_step_fn=compatibility,
        audio_loss_fn=compatibility,
    )
    return dict(
        model=model,
        compatibility=compatibility(),
        run_fingerprint=fingerprint,
        stage="joint",
        next_warmup_position=2,
        next_joint_position=1,
        optimizer=optimizer,
        optimizer_stage="joint",
        selector=selector,
        stopper=stopper,
        training_history=[
            {
                "stage": "warmup",
                "step": 1,
                "sample_index": 0,
                "total": 1.0,
                "audio_to_visual_grad_norm": 0.0,
                "losses": {"audio": 1.0},
                "gradient_norms": {"visual": 0.0},
            },
            {
                "stage": "warmup",
                "step": 2,
                "sample_index": 1,
                "total": 1.0,
                "audio_to_visual_grad_norm": 0.0,
                "losses": {"audio": 1.0},
                "gradient_norms": {"visual": 0.0},
            },
            {
                "stage": "joint",
                "step": 1,
                "sample_index": 4,
                "total": 1.0,
                "audio_to_visual_grad_norm": 0.25,
                "losses": {"audio": 1.0},
                "gradient_norms": {"visual": 0.25},
            }
        ],
        validation_history=[
            {
                "step": 1,
                "summary": {
                    "audio_total": {"mean": 1.0},
                    "rgb_psnr": {"mean": 30.0},
                    "rgb_ssim": {"mean": 0.95},
                },
            }
        ],
        maximum_positive_audio_visual_gradient=0.25,
        best_generation=0,
        validation_summary={
            "audio_total": {"mean": 1.0},
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
        best_evaluation_summary={
            "audio_total": {"mean": 1.0},
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
    )


def test_checkpoint_roundtrip_restores_all_state(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "latest.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))

    state = inspect_pilot_checkpoint(
        path,
        expected_compatibility=compatibility(),
        indices=VariantIndices((0, 1), (4, 5, 6)),
    )
    restored = nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    restore_pilot_checkpoint(
        state,
        model=restored,
        optimizer=restored_optimizer,
        optimizer_stage="joint",
    )

    assert state.next_warmup_position == 2
    assert state.next_joint_position == 1
    assert state.maximum_positive_audio_visual_gradient == 0.25
    assert state.selector.best_step == 1
    assert state.stopper.last_step == 1
    assert state.training_history[-1]["sample_index"] == 4
    assert restored_optimizer.state
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])


def test_incompatible_checkpoint_is_rejected_before_model_mutation(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))
    target = nn.Linear(2, 1)
    before = {name: value.clone() for name, value in target.state_dict().items()}

    with pytest.raises(PilotResumeError, match="variant"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=replace(
                compatibility(), variant="frozen_visual"
            ),
            indices=VariantIndices((0, 1), (4, 5, 6)),
        )
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name])


def test_atomic_save_failure_preserves_old_checkpoint(tmp_path: Path, monkeypatch) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))
    old = path.read_bytes()
    monkeypatch.setattr(
        "avgaussianv2.experiment.checkpoint.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))
    assert path.read_bytes() == old
    assert not list(tmp_path.glob(".latest.pt.*.tmp"))


def test_store_enforces_fresh_resume_output_policy(tmp_path: Path) -> None:
    store = PilotCheckpointStore(tmp_path, compatibility())
    store.latest_path.write_bytes(b"old")
    with store, pytest.raises(FileExistsError, match="refuses"):
        store.prepare()
    overwrite = PilotCheckpointStore(tmp_path, compatibility(), overwrite=True)
    with overwrite:
        overwrite.prepare()
    model = nn.Linear(2, 1)
    save_pilot_checkpoint(
        store.latest_path,
        **_checkpoint_kwargs(model, torch.optim.SGD(model.parameters(), lr=0.1)),
    )
    resume = PilotCheckpointStore(tmp_path, compatibility(), resume=True)
    with resume:
        resume.prepare()
    missing = PilotCheckpointStore(tmp_path / "missing", compatibility(), resume=True)
    with missing, pytest.raises(FileNotFoundError, match="latest"):
        missing.prepare()


def test_fresh_store_rejects_any_nonempty_output_directory(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").write_text("occupied")
    fresh = PilotCheckpointStore(tmp_path, compatibility())
    with fresh, pytest.raises(FileExistsError, match="nonempty"):
        fresh.prepare()
    overwrite = PilotCheckpointStore(tmp_path, compatibility(), overwrite=True)
    with overwrite:
        overwrite.prepare()
    assert (tmp_path / "unrelated.txt").read_text() == "occupied"


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda payload: payload.pop("model_state_dict"), "model_state_dict"),
        (
            lambda payload: payload.__setitem__("optimizer_stage", "warmup"),
            "optimizer/stage",
        ),
        (
            lambda payload: payload.__setitem__(
                "training_history", payload["training_history"][:-1]
            ),
            "training_history",
        ),
    ],
)
def test_partial_and_corrupt_payloads_are_rejected(
    tmp_path: Path, corrupt, message: str
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))
    payload = torch.load(path, weights_only=False)
    corrupt(payload)
    torch.save(payload, path)
    with pytest.raises(PilotResumeError, match=message):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            indices=VariantIndices((0, 1), (4, 5, 6)),
        )


def test_complete_checkpoint_is_not_resumed_as_active(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    kwargs = _checkpoint_kwargs(model, optimizer)
    kwargs.update(
        stage="complete",
        optimizer=None,
        optimizer_stage=None,
        stop_reason="max_steps",
    )
    save_pilot_checkpoint(path, **kwargs)
    with pytest.raises(PilotResumeError, match="complete"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            indices=VariantIndices((0, 1), (4,)),
        )


def test_bad_model_shape_is_rejected_before_any_model_mutation(tmp_path: Path) -> None:
    source = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(source, optimizer))
    state = inspect_pilot_checkpoint(
        path,
        expected_compatibility=compatibility(),
        indices=VariantIndices((0, 1), (4, 5, 6)),
    )
    state.model_state_dict["weight"] = torch.zeros(2, 2)
    target = nn.Linear(2, 1)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(PilotResumeError, match="weight"):
        restore_pilot_checkpoint(
            state,
            model=target,
            optimizer=target_optimizer,
            optimizer_stage="joint",
        )
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name])


def test_condition_off_warmup_checkpoint_is_rejected_at_inspection(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    condition_off = replace(compatibility(), variant="condition_off")
    path = tmp_path / "latest.pt"
    kwargs = _checkpoint_kwargs(model, optimizer)
    kwargs.update(
        compatibility=condition_off,
        stage="warmup",
        next_warmup_position=0,
        next_joint_position=0,
        training_history=[],
        validation_history=[],
        optimizer_stage="warmup",
    )
    save_pilot_checkpoint(path, **kwargs)
    with pytest.raises(PilotResumeError, match="condition_off.*warmup"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=condition_off,
            indices=VariantIndices((), (4, 5, 6)),
        )


def test_malicious_custom_object_checkpoint_is_not_loaded(tmp_path: Path) -> None:
    path = tmp_path / "malicious.pt"
    torch.save({"payload": Dangerous()}, path)
    with pytest.raises(PilotResumeError, match="cannot read"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            indices=VariantIndices((0, 1), (4, 5, 6)),
        )


def test_save_rejects_custom_object_in_optimizer_state(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.state[next(model.parameters())]["unsafe"] = Dangerous()
    with pytest.raises(PilotResumeError, match="safe primitive"):
        save_pilot_checkpoint(
            tmp_path / "unsafe.pt", **_checkpoint_kwargs(model, optimizer)
        )
    assert not (tmp_path / "unsafe.pt").exists()


def test_loaded_checkpoint_rejects_nonprimitive_allowlisted_extra(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "device-extra.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))
    payload = torch.load(path, weights_only=True)
    payload["unexpected_device"] = torch.device("cpu")
    torch.save(payload, path)
    with pytest.raises(PilotResumeError, match="safe primitive"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            indices=VariantIndices((0, 1), (4, 5, 6)),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("unexpected_scalar", 1),
        lambda payload: payload["provenance"].__setitem__("unexpected_scalar", 1),
        lambda payload: payload["selector_state"].__setitem__("unexpected_scalar", 1),
        lambda payload: payload["stopper_state"].__setitem__("unexpected_scalar", 1),
        lambda payload: payload["training_history"][0].__setitem__(
            "unexpected_scalar", 1
        ),
        lambda payload: payload["validation_history"][0].__setitem__(
            "unexpected_scalar", 1
        ),
    ],
)
def test_loaded_checkpoint_rejects_unexpected_schema_keys(
    tmp_path: Path, mutate
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "unexpected.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))
    payload = torch.load(path, weights_only=True)
    mutate(payload)
    torch.save(payload, path)
    with pytest.raises(PilotResumeError, match="unexpected"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            indices=VariantIndices((0, 1), (4, 5, 6)),
        )


def test_best_checkpoint_self_inspects_but_cannot_be_active_resume(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    path = tmp_path / "best.pt"
    kwargs = _checkpoint_kwargs(model, None)
    kwargs.update(
        checkpoint_kind="best",
        optimizer=None,
        optimizer_stage=None,
    )
    save_pilot_checkpoint(path, **kwargs)
    artifact = inspect_pilot_checkpoint(
        path,
        expected_compatibility=compatibility(),
        indices=VariantIndices((0, 1), (4, 5, 6)),
        active_resume=False,
    )
    assert artifact.checkpoint_kind == "best"
    assert artifact.best_evaluation_summary["audio_total"] == {"mean": 1.0}
    with pytest.raises(PilotResumeError, match="best.*active"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            indices=VariantIndices((0, 1), (4, 5, 6)),
        )


def test_store_lock_rejects_concurrent_owner_and_stale_file_is_harmless(
    tmp_path: Path,
) -> None:
    first = PilotCheckpointStore(tmp_path, compatibility())
    second = PilotCheckpointStore(tmp_path, compatibility())
    first.acquire()
    try:
        with pytest.raises(PilotResumeError, match="another process"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()
    assert (tmp_path / ".pilot.lock").is_file()


def test_store_accepts_live_inherited_parent_proc_fd_output(tmp_path: Path) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        inherited_path = f"/proc/{os.getpid()}/fd/{directory_fd}"
        script = """
import json
import sys
from avgaussianv2.experiment.checkpoint import PilotCheckpointStore, PilotCompatibility

store = PilotCheckpointStore(
    sys.argv[1],
    PilotCompatibility.from_mapping(json.loads(sys.argv[2])),
)
with store:
    store.verify_output_identity()
    store.prepare()
"""
        environment = dict(os.environ)
        environment["AVGAUSSIANV2_ORCHESTRATOR_PID"] = str(os.getpid())
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                inherited_path,
                json.dumps(compatibility().to_mapping()),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
        )
    finally:
        os.close(directory_fd)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".pilot.lock").is_file()


def test_run_fingerprint_detects_behavior_changes_before_model_mutation(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    kwargs = _checkpoint_kwargs(model, optimizer)
    save_pilot_checkpoint(path, **kwargs)
    changed = build_run_fingerprint(
        pilot_config=PilotConfig(
            warmup_steps=2,
            joint_steps=3,
            validation_interval=2,
            minimum_joint_steps=0,
            patience=2,
            minimum_relative_improvement=0.1,
        ),
        train_config=TrainConfig(audio_lr=0.2, lambda_rgb=0.3),
        visual_baseline={
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
        model=nn.Sequential(nn.Linear(2, 1)),
        warmup_optimizer_factory=compatibility,
        joint_optimizer_factory=compatibility,
        warmup_step_fn=compatibility,
        joint_step_fn=compatibility,
        audio_loss_fn=compatibility,
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(PilotResumeError, match="run_fingerprint"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            expected_run_fingerprint=changed,
            indices=VariantIndices((0, 1), (4, 5, 6)),
            model=model,
        )
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize(
    ("pilot_config", "train_config", "model", "joint_factory"),
    [
        (
            PilotConfig(validation_interval=25),
            TrainConfig(),
            nn.Linear(2, 1),
            compatibility,
        ),
        (
            PilotConfig(),
            TrainConfig(audio_lr=0.2),
            nn.Linear(2, 1),
            compatibility,
        ),
        (
            PilotConfig(),
            TrainConfig(lambda_rgb=0.3),
            nn.Linear(2, 1),
            compatibility,
        ),
        (
            PilotConfig(),
            TrainConfig(),
            nn.Sequential(nn.Linear(2, 1)),
            compatibility,
        ),
        (
            PilotConfig(),
            TrainConfig(),
            nn.Linear(2, 1),
            sha256_file,
        ),
    ],
    ids=[
        "validation_interval",
        "learning_rate",
        "loss_weight",
        "model_identity",
        "factory_identity",
    ],
)
def test_run_fingerprint_changes_for_behavior_affecting_inputs(
    pilot_config, train_config, model, joint_factory
) -> None:
    base = build_run_fingerprint(
        pilot_config=PilotConfig(),
        train_config=TrainConfig(),
        visual_baseline={
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
        model=nn.Linear(2, 1),
        warmup_optimizer_factory=compatibility,
        joint_optimizer_factory=compatibility,
        warmup_step_fn=compatibility,
        joint_step_fn=compatibility,
        audio_loss_fn=compatibility,
    )
    changed = build_run_fingerprint(
        pilot_config=pilot_config,
        train_config=train_config,
        visual_baseline={
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
        model=model,
        warmup_optimizer_factory=compatibility,
        joint_optimizer_factory=joint_factory,
        warmup_step_fn=compatibility,
        joint_step_fn=compatibility,
        audio_loss_fn=compatibility,
    )
    assert changed["sha256"] != base["sha256"]


def test_checkpoint_enabled_lambda_requires_explicit_identity() -> None:
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="lambda.*explicit"):
        build_run_fingerprint(
            pilot_config=PilotConfig(),
            train_config=TrainConfig(),
            visual_baseline={
                "rgb_psnr": {"mean": 30.0},
                "rgb_ssim": {"mean": 0.95},
            },
            model=model,
            warmup_optimizer_factory=compatibility,
            joint_optimizer_factory=compatibility,
            warmup_step_fn=compatibility,
            joint_step_fn=lambda: None,
            audio_loss_fn=compatibility,
        )


def test_explicit_component_identity_changes_run_fingerprint() -> None:
    common = {
        "pilot_config": PilotConfig(),
        "train_config": TrainConfig(),
        "visual_baseline": {
            "rgb_psnr": {"mean": 30.0},
            "rgb_ssim": {"mean": 0.95},
        },
        "model": nn.Linear(2, 1),
        "warmup_optimizer_factory": compatibility,
        "joint_optimizer_factory": compatibility,
        "warmup_step_fn": compatibility,
        "joint_step_fn": compatibility,
        "audio_loss_fn": compatibility,
    }
    first = build_run_fingerprint(
        **common,
        component_identities={"joint_step_fn": "pilot.joint_step.v1"},
    )
    second = build_run_fingerprint(
        **common,
        component_identities={"joint_step_fn": "pilot.joint_step.v2"},
    )
    assert first["sha256"] != second["sha256"]


def test_baseexception_restore_rolls_back_model_and_optimizer(tmp_path: Path) -> None:
    class Abort(BaseException):
        pass

    class AbortingSGD(torch.optim.SGD):
        def load_state_dict(self, state_dict):
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(10)
            self.state["partial"] = {"nested": [torch.tensor(2.0)]}
            raise Abort("abort restore")

    source = nn.Linear(2, 1)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    path = tmp_path / "latest.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(source, source_optimizer))
    state = inspect_pilot_checkpoint(
        path,
        expected_compatibility=compatibility(),
        indices=VariantIndices((0, 1), (4, 5, 6)),
    )
    target = nn.Linear(2, 1)
    optimizer = AbortingSGD(target.parameters(), lr=0.1)
    before_model = {name: value.clone() for name, value in target.state_dict().items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(Abort, match="abort restore"):
        restore_pilot_checkpoint(
            state, model=target, optimizer=optimizer, optimizer_stage="joint"
        )
    for name, value in target.state_dict().items():
        assert torch.equal(value, before_model[name])
    assert optimizer.state_dict() == before_optimizer


def test_prepare_rolls_back_ahead_best_from_incomplete_transaction(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    save_pilot_checkpoint(latest, generation=1, **kwargs)
    best_kwargs = {
        **kwargs,
        "checkpoint_kind": "best",
        "optimizer": None,
        "optimizer_stage": None,
    }
    save_pilot_checkpoint(best, generation=1, **best_kwargs)
    old_best = best.read_bytes()
    (tmp_path / ".pilot-best-backup.pt").write_bytes(old_best)
    with torch.no_grad():
        model.weight.add_(10)
    save_pilot_checkpoint(best, generation=2, **best_kwargs)
    (tmp_path / ".pilot-validation-transaction.json").write_text(
        json.dumps({"generation": 2, "had_best": True})
    )

    store = PilotCheckpointStore(tmp_path, compatibility(), resume=True)
    with store:
        store.prepare()

    assert best.read_bytes() == old_best
    assert not (tmp_path / ".pilot-validation-transaction.json").exists()
    assert not (tmp_path / ".pilot-best-backup.pt").exists()


def test_validation_commit_survives_one_shot_directory_sync_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    store = PilotCheckpointStore(tmp_path, compatibility())
    store.bind_run_fingerprint(kwargs["run_fingerprint"])
    with store:
        store.prepare()
        checkpoint_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"run_fingerprint", "checkpoint_kind", "generation"}
        }
        store.publish_validation(
            latest_kwargs=checkpoint_kwargs,
            best_kwargs={
                **checkpoint_kwargs,
                "optimizer": None,
                "optimizer_stage": None,
            },
        )
        real_open = checkpoint_module.os.open
        latest_replaced = False
        failed = False
        real_replace = checkpoint_module.os.replace

        def track_replace(source, destination):
            nonlocal latest_replaced
            real_replace(source, destination)
            if Path(destination) == store.latest_path:
                latest_replaced = True

        def fail_first_directory_open(path, flags, *args, **named):
            nonlocal failed
            if (
                latest_replaced
                and not failed
                and Path(path).resolve() == tmp_path.resolve()
            ):
                failed = True
                raise OSError("injected directory open failure")
            return real_open(path, flags, *args, **named)

        monkeypatch.setattr(checkpoint_module.os, "replace", track_replace)
        monkeypatch.setattr(checkpoint_module.os, "open", fail_first_directory_open)
        store.publish_validation(
            latest_kwargs=checkpoint_kwargs,
            best_kwargs={
                **checkpoint_kwargs,
                "optimizer": None,
                "optimizer_stage": None,
            },
        )

    latest = torch.load(store.latest_path, weights_only=True)
    best = torch.load(store.best_path, weights_only=True)
    assert latest["generation"] == best["generation"] == 2
    assert not store.journal_path.exists()
    assert not store.backup_path.exists()
    assert store.metrics["backup_copy_count"] == 1
    assert store.metrics["backup_copy_bytes"] > 0


def test_validation_commit_retains_journal_when_durability_is_ambiguous(
    tmp_path: Path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    checkpoint_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"run_fingerprint", "checkpoint_kind", "generation"}
    }
    store = PilotCheckpointStore(tmp_path, compatibility())
    store.bind_run_fingerprint(kwargs["run_fingerprint"])
    with store:
        store.prepare()
        store.publish_validation(
            latest_kwargs=checkpoint_kwargs,
            best_kwargs={
                **checkpoint_kwargs,
                "optimizer": None,
                "optimizer_stage": None,
            },
        )
        real_open = checkpoint_module.os.open
        real_replace = checkpoint_module.os.replace
        latest_replaced = False

        def track_replace(source, destination):
            nonlocal latest_replaced
            real_replace(source, destination)
            if Path(destination) == store.latest_path:
                latest_replaced = True

        def fail_directory_open(path, flags, *args, **named):
            if latest_replaced and Path(path).resolve() == tmp_path.resolve():
                raise OSError("persistent directory open failure")
            return real_open(path, flags, *args, **named)

        monkeypatch.setattr(checkpoint_module.os, "replace", track_replace)
        monkeypatch.setattr(checkpoint_module.os, "open", fail_directory_open)
        with pytest.raises(OSError, match="persistent directory"):
            store.publish_validation(
                latest_kwargs=checkpoint_kwargs,
                best_kwargs={
                    **checkpoint_kwargs,
                    "optimizer": None,
                    "optimizer_stage": None,
                },
            )
        assert store.journal_path.is_file()

    monkeypatch.setattr(checkpoint_module.os, "open", real_open)
    resumed = PilotCheckpointStore(tmp_path, compatibility(), resume=True)
    with resumed:
        resumed.prepare()
    latest = torch.load(resumed.latest_path, weights_only=True)
    best = torch.load(resumed.best_path, weights_only=True)
    assert latest["generation"] == best["generation"] == 2
    assert not resumed.journal_path.exists()


@pytest.mark.parametrize("case", ["depth", "tensor_count", "storage_bytes"])
def test_checkpoint_metadata_limits_reject_before_real_materialization(
    tmp_path, monkeypatch, case
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    checkpoint_path = tmp_path / "latest.pt"
    checkpoint_path.write_bytes(b"metadata placeholder")
    if case == "depth":
        nested: object = 0
        for _ in range(checkpoint_module.MAX_CHECKPOINT_NESTING + 1):
            nested = [nested]
        metadata = {"nested": nested}
    elif case == "tensor_count":
        monkeypatch.setattr(checkpoint_module, "MAX_CHECKPOINT_TENSORS", 1)
        metadata = {"tensors": [torch.empty(0), torch.empty(0)]}
    else:
        monkeypatch.setattr(
            checkpoint_module, "MAX_CHECKPOINT_LOGICAL_STORAGE_BYTES", 8
        )
        metadata = {"tensor": torch.empty(3, dtype=torch.float32)}

    real_loads = []
    monkeypatch.setattr(
        checkpoint_module,
        "_load_checkpoint_metadata",
        lambda descriptor, source: metadata,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_load_checkpoint_real",
        lambda descriptor: real_loads.append(True),
    )

    with pytest.raises(
        checkpoint_module.PilotResumeError,
        match="metadata|nesting|tensor|storage|limit",
    ):
        checkpoint_module._payload(checkpoint_path)

    assert real_loads == []


def test_committed_validation_reraises_keyboard_interrupt(
    tmp_path: Path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    checkpoint_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"run_fingerprint", "checkpoint_kind", "generation"}
    }
    store = PilotCheckpointStore(tmp_path, compatibility())
    store.bind_run_fingerprint(kwargs["run_fingerprint"])
    with store:
        store.prepare()
        store.publish_latest(**checkpoint_kwargs)
        real_save = checkpoint_module.save_pilot_checkpoint

        def interrupt_after_committed_latest(path, **save_kwargs):
            real_save(path, **save_kwargs)
            if path.name == "latest.pt" and save_kwargs["generation"] == 2:
                raise KeyboardInterrupt

        monkeypatch.setattr(
            checkpoint_module,
            "save_pilot_checkpoint",
            interrupt_after_committed_latest,
        )
        with pytest.raises(KeyboardInterrupt):
            store.publish_validation(
                latest_kwargs=checkpoint_kwargs,
                best_kwargs={
                    **checkpoint_kwargs,
                    "optimizer": None,
                    "optimizer_stage": None,
                },
            )
    assert torch.load(store.latest_path, weights_only=True)["generation"] == 2
    assert torch.load(store.best_path, weights_only=True)["generation"] == 2
    assert not store.journal_path.exists()


def test_keyboard_interrupt_wins_when_durability_retry_fails(
    tmp_path: Path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    checkpoint_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"run_fingerprint", "checkpoint_kind", "generation"}
    }
    store = PilotCheckpointStore(tmp_path, compatibility())
    store.bind_run_fingerprint(kwargs["run_fingerprint"])
    with store:
        store.prepare()
        store.publish_latest(**checkpoint_kwargs)
        real_save = checkpoint_module.save_pilot_checkpoint

        def interrupt_after_committed_latest(path, **save_kwargs):
            real_save(path, **save_kwargs)
            if path.name == "latest.pt" and save_kwargs["generation"] == 2:
                raise KeyboardInterrupt

        monkeypatch.setattr(
            checkpoint_module,
            "save_pilot_checkpoint",
            interrupt_after_committed_latest,
        )
        monkeypatch.setattr(
            checkpoint_module,
            "_fsync_directory",
            lambda path: (_ for _ in ()).throw(
                OSError("injected durability retry failure")
            ),
        )
        with pytest.raises(KeyboardInterrupt):
            store.publish_validation(
                latest_kwargs=checkpoint_kwargs,
                best_kwargs={
                    **checkpoint_kwargs,
                    "optimizer": None,
                    "optimizer_stage": None,
                },
            )
    assert store.journal_path.is_file()
    assert torch.load(store.latest_path, weights_only=True)["generation"] == 2


def test_backup_copy_failure_metrics_distinguish_attempt_success_and_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    checkpoint_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"run_fingerprint", "checkpoint_kind", "generation"}
    }
    store = PilotCheckpointStore(tmp_path, compatibility())
    store.bind_run_fingerprint(kwargs["run_fingerprint"])
    with store:
        store.prepare()
        store.publish_validation(
            latest_kwargs=checkpoint_kwargs,
            best_kwargs={
                **checkpoint_kwargs,
                "optimizer": None,
                "optimizer_stage": None,
            },
        )
        monkeypatch.setattr(
            checkpoint_module,
            "_durable_copy",
            lambda *args: (_ for _ in ()).throw(OSError("backup copy failure")),
        )
        with pytest.raises(OSError, match="backup copy"):
            store.publish_validation(
                latest_kwargs=checkpoint_kwargs,
                best_kwargs={
                    **checkpoint_kwargs,
                    "optimizer": None,
                    "optimizer_stage": None,
                },
            )
    assert store.metrics["backup_copy_attempt_count"] == 1
    assert store.metrics["backup_copy_count"] == 0
    assert store.metrics["backup_copy_failure_count"] == 1


@pytest.mark.parametrize("latest_generation", [1, 2])
def test_transaction_with_missing_required_best_backup_is_unrecoverable(
    tmp_path: Path, latest_generation: int
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    save_pilot_checkpoint(
        tmp_path / "latest.pt", generation=latest_generation, **kwargs
    )
    (tmp_path / ".pilot-validation-transaction.json").write_text(
        json.dumps({"generation": 2, "had_best": True})
    )
    store = PilotCheckpointStore(tmp_path, compatibility(), resume=True)
    with store, pytest.raises(PilotResumeError, match="backup"):
        store.prepare()
    assert store.journal_path.is_file()
    assert (tmp_path / "latest.pt").is_file()


def test_transaction_recovery_rejects_ahead_latest_and_retains_journal(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    save_pilot_checkpoint(tmp_path / "latest.pt", generation=3, **kwargs)
    (tmp_path / ".pilot-validation-transaction.json").write_text(
        json.dumps({"generation": 2, "had_best": False})
    )
    store = PilotCheckpointStore(tmp_path, compatibility(), resume=True)
    with store, pytest.raises(PilotResumeError, match="ahead"):
        store.prepare()
    assert store.journal_path.is_file()
    assert torch.load(store.latest_path, weights_only=True)["generation"] == 3


def test_rollback_sync_failure_retains_retryable_journal_and_backup(
    tmp_path: Path, monkeypatch
) -> None:
    import avgaussianv2.experiment.checkpoint as checkpoint_module

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = _checkpoint_kwargs(model, optimizer)
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    backup = tmp_path / ".pilot-best-backup.pt"
    journal = tmp_path / ".pilot-validation-transaction.json"
    save_pilot_checkpoint(latest, generation=1, **kwargs)
    save_pilot_checkpoint(best, generation=1, **kwargs)
    backup.write_bytes(best.read_bytes())
    with torch.no_grad():
        model.weight.add_(10)
    save_pilot_checkpoint(best, generation=2, **kwargs)
    journal.write_text(json.dumps({"generation": 2, "had_best": True}))

    real_fsync_directory = checkpoint_module._fsync_directory
    failed = False

    def fail_once(path):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected rollback sync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(checkpoint_module, "_fsync_directory", fail_once)
    store = PilotCheckpointStore(tmp_path, compatibility(), resume=True)
    with store, pytest.raises(PilotResumeError, match="rollback sync"):
        store.prepare()
    assert journal.is_file()
    assert backup.is_file()

    monkeypatch.setattr(
        checkpoint_module, "_fsync_directory", real_fsync_directory
    )
    retry = PilotCheckpointStore(tmp_path, compatibility(), resume=True)
    with retry:
        retry.prepare()
    assert not journal.exists()
    assert not backup.exists()


def test_derived_stopper_corruption_is_rejected(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "corrupt-derived.pt"
    save_pilot_checkpoint(path, **_checkpoint_kwargs(model, optimizer))
    payload = torch.load(path, weights_only=True)
    payload["stopper_state"]["best"] = 0.5
    torch.save(payload, path)
    with pytest.raises(PilotResumeError, match="stopper_state.*history"):
        inspect_pilot_checkpoint(
            path,
            expected_compatibility=compatibility(),
            indices=VariantIndices((0, 1), (4, 5, 6)),
        )


def test_overwrite_removes_all_owned_outputs_but_keeps_unowned_file(
    tmp_path: Path,
) -> None:
    for name in (
        "latest.pt",
        "best.pt",
        "worker_summary.json",
        "training_curve.csv",
        ".pilot-validation-transaction.json",
        ".pilot-best-backup.pt",
    ):
        (tmp_path / name).write_text("old")
    (tmp_path / "validation" / "step_000001").mkdir(parents=True)
    (tmp_path / "validation" / "step_000001" / "metrics.json").write_text("old")
    (tmp_path / "keep.txt").write_text("keep")
    store = PilotCheckpointStore(tmp_path, compatibility(), overwrite=True)
    with store:
        store.prepare()
    for name in (
        "latest.pt",
        "best.pt",
        "worker_summary.json",
        "training_curve.csv",
        ".pilot-validation-transaction.json",
        ".pilot-best-backup.pt",
        "validation",
    ):
        assert not (tmp_path / name).exists()
    assert (tmp_path / "keep.txt").read_text() == "keep"

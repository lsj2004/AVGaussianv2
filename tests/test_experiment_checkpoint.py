from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotCompatibility,
    PilotResumeError,
    inspect_pilot_checkpoint,
    restore_pilot_checkpoint,
    save_pilot_checkpoint,
    hash_index_manifest,
    sha256_file,
    validate_compatibility,
)
from avgaussianv2.experiment.contracts import VariantIndices
from avgaussianv2.experiment.selection import BestSelector, EarlyStopper


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
    return dict(
        model=model,
        compatibility=compatibility(),
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
        validation_history=[{"step": 1, "summary": {"audio_total": {"mean": 1.0}}}],
        maximum_positive_audio_visual_gradient=0.25,
        evaluation_summary={"audio_total": {"mean": 1.0}},
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
    with pytest.raises(FileExistsError, match="refuses"):
        store.prepare()
    PilotCheckpointStore(tmp_path, compatibility(), overwrite=True).prepare()
    store.latest_path.write_bytes(b"resume")
    PilotCheckpointStore(tmp_path, compatibility(), resume=True).prepare()
    with pytest.raises(FileNotFoundError, match="latest"):
        PilotCheckpointStore(tmp_path / "missing", compatibility(), resume=True).prepare()


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

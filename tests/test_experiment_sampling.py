from dataclasses import FrozenInstanceError

import pytest

from avgaussianv2.experiment.contracts import (
    PilotConfig,
    SharedIndices,
    Variant,
    VariantIndices,
)
from avgaussianv2.experiment.sampling import (
    build_shared_indices,
    evenly_spaced_indices,
)


def test_pilot_config_has_approved_defaults() -> None:
    config = PilotConfig()

    assert config.warmup_steps == 200
    assert config.joint_steps == 500
    assert config.validation_interval == 50
    assert config.minimum_joint_steps == 200
    assert config.patience == 4
    assert config.minimum_relative_improvement == pytest.approx(0.005)
    assert config.quick_validation_samples == 32
    assert config.psnr_tolerance_db == pytest.approx(0.5)
    assert config.ssim_tolerance == pytest.approx(0.01)


def test_variants_have_exact_approved_values() -> None:
    assert tuple(Variant) == (
        Variant.JOINT_CONDITIONED,
        Variant.FROZEN_VISUAL,
        Variant.CONDITION_OFF,
    )
    assert tuple(variant.value for variant in Variant) == (
        "joint_conditioned",
        "frozen_visual",
        "condition_off",
    )
    assert str(Variant.CONDITION_OFF) == "condition_off"


def test_pilot_config_is_frozen() -> None:
    config = PilotConfig()

    with pytest.raises(FrozenInstanceError):
        config.joint_steps = 1  # type: ignore[misc]


def test_index_contracts_are_frozen() -> None:
    shared = SharedIndices(warmup=(0,), joint=(1,))
    variant = VariantIndices(warmup=(0,), joint=(1,))

    with pytest.raises(FrozenInstanceError):
        shared.warmup = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        variant.joint = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"joint_steps": 0}, "joint_steps"),
        ({"validation_interval": 0}, "validation_interval"),
        ({"minimum_relative_improvement": 0.0}, "minimum_relative_improvement"),
        ({"minimum_relative_improvement": 1.0}, "minimum_relative_improvement"),
    ],
)
def test_pilot_config_validation_rejects_invalid_values(
    overrides: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PilotConfig(**overrides).validate()


def test_shared_indices_are_identical_where_variants_share_phases() -> None:
    shared = build_shared_indices(dataset_size=11, warmup_steps=14, joint_steps=25, seed=9)

    conditioned = shared.for_variant(Variant.JOINT_CONDITIONED)
    frozen = shared.for_variant(Variant.FROZEN_VISUAL)
    condition_off = shared.for_variant(Variant.CONDITION_OFF)

    assert len(shared.warmup) == 14
    assert len(shared.joint) == 25
    assert conditioned.warmup == frozen.warmup == shared.warmup
    assert condition_off.warmup == ()
    assert conditioned.joint == frozen.joint == condition_off.joint == shared.joint


def test_build_shared_indices_is_deterministic_across_seeds() -> None:
    first = build_shared_indices(dataset_size=7, warmup_steps=14, joint_steps=14, seed=123)
    second = build_shared_indices(dataset_size=7, warmup_steps=14, joint_steps=14, seed=123)
    different = build_shared_indices(
        dataset_size=7, warmup_steps=14, joint_steps=14, seed=124
    )

    assert first == second
    assert first != different


def test_build_shared_indices_uses_complete_permutations_before_repeating() -> None:
    shared = build_shared_indices(
        dataset_size=4, warmup_steps=10, joint_steps=10, seed=4
    )

    for indices in (shared.warmup, shared.joint):
        assert sorted(indices[:4]) == [0, 1, 2, 3]
        assert sorted(indices[4:8]) == [0, 1, 2, 3]
        assert len(indices) == 10


@pytest.mark.parametrize("size", [0, -1])
def test_build_shared_indices_rejects_nonpositive_dataset_size(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        build_shared_indices(dataset_size=size, warmup_steps=1, joint_steps=1, seed=0)


def test_build_shared_indices_allows_zero_counts() -> None:
    shared = build_shared_indices(
        dataset_size=3, warmup_steps=0, joint_steps=0, seed=0
    )

    assert shared == SharedIndices(warmup=(), joint=())


@pytest.mark.parametrize(("warmup_steps", "joint_steps"), [(-1, 1), (1, -1)])
def test_build_shared_indices_rejects_negative_counts(
    warmup_steps: int, joint_steps: int
) -> None:
    with pytest.raises(ValueError, match="count"):
        build_shared_indices(
            dataset_size=3,
            warmup_steps=warmup_steps,
            joint_steps=joint_steps,
            seed=0,
        )


def test_evenly_spaced_indices_matches_acceptance_example() -> None:
    assert evenly_spaced_indices(dataset_size=101, count=5) == (0, 25, 50, 75, 100)


def test_evenly_spaced_indices_caps_count_to_dataset_size() -> None:
    assert evenly_spaced_indices(dataset_size=3, count=10) == (0, 1, 2)


@pytest.mark.parametrize(("dataset_size", "count"), [(0, 1), (-1, 1), (3, 0), (3, -1)])
def test_evenly_spaced_indices_rejects_nonpositive_arguments(
    dataset_size: int, count: int
) -> None:
    with pytest.raises(ValueError):
        evenly_spaced_indices(dataset_size=dataset_size, count=count)

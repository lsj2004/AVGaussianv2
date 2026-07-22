from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from avgaussianv2.experiment.contracts import PilotConfig, Variant
from avgaussianv2.experiment.sampling import (
    _draw_epochs,
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


def test_pilot_config_is_frozen() -> None:
    config = PilotConfig()

    with pytest.raises(FrozenInstanceError):
        config.joint_steps = 1  # type: ignore[misc]


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


def test_build_shared_indices_is_deterministic_and_draws_separate_phases() -> None:
    first = build_shared_indices(dataset_size=7, warmup_steps=7, joint_steps=7, seed=123)
    second = build_shared_indices(dataset_size=7, warmup_steps=7, joint_steps=7, seed=123)

    assert first == second
    assert sorted(first.warmup) == list(range(7))
    assert sorted(first.joint) == list(range(7))
    expected_rng = np.random.default_rng(123)
    assert first.warmup == tuple(int(index) for index in expected_rng.permutation(7))
    assert first.joint == tuple(int(index) for index in expected_rng.permutation(7))


def test_draw_epochs_uses_complete_permutations_before_repeating() -> None:
    indices = _draw_epochs(size=4, count=10, rng=np.random.default_rng(4))

    assert sorted(indices[:4]) == [0, 1, 2, 3]
    assert sorted(indices[4:8]) == [0, 1, 2, 3]
    assert len(indices) == 10


@pytest.mark.parametrize("size", [0, -1])
def test_draw_epochs_rejects_nonpositive_dataset_size(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        _draw_epochs(size=size, count=1, rng=np.random.default_rng(0))


def test_draw_epochs_allows_zero_count_and_rejects_negative_count() -> None:
    rng = np.random.default_rng(0)

    assert _draw_epochs(size=3, count=0, rng=rng) == ()
    with pytest.raises(ValueError, match="count"):
        _draw_epochs(size=3, count=-1, rng=rng)


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

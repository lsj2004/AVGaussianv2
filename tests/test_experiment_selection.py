from __future__ import annotations

import math
import json

import pytest

from avgaussianv2.experiment.selection import BestSelector, EarlyStopper, visual_feasible


def summary(*, psnr: float = 30.0, ssim: float = 0.95, audio: float = 1.0):
    return {
        "rgb_psnr": {"mean": psnr},
        "rgb_ssim": {"mean": ssim},
        "audio_total": {"mean": audio},
    }


def test_visual_feasibility_includes_exact_tolerance_boundaries() -> None:
    baseline = summary()

    assert visual_feasible(summary(psnr=29.5, ssim=0.94), baseline, 0.5, 0.01)
    assert not visual_feasible(summary(psnr=29.49, ssim=0.94), baseline, 0.5, 0.01)
    assert not visual_feasible(summary(psnr=29.5, ssim=0.939), baseline, 0.5, 0.01)


@pytest.mark.parametrize("tolerances", [(-0.1, 0.01), (0.5, -0.01)])
def test_visual_feasibility_rejects_negative_tolerances(tolerances) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        visual_feasible(summary(), summary(), *tolerances)


@pytest.mark.parametrize("bad", [math.nan, math.inf, "0.5", True])
def test_visual_feasibility_rejects_invalid_tolerances(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        visual_feasible(summary(), summary(), bad, 0.01)


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"rgb_psnr": {}},
        {"rgb_psnr": {"mean": 30.0}},
        {"rgb_psnr": {"mean": math.nan}, "rgb_ssim": {"mean": 0.95}},
        {"rgb_psnr": {"mean": [30.0]}, "rgb_ssim": {"mean": 0.95}},
        {"rgb_psnr": 30.0, "rgb_ssim": {"mean": 0.95}},
    ],
)
def test_visual_feasibility_rejects_missing_malformed_or_nonfinite_fields(metrics) -> None:
    with pytest.raises(
        (KeyError, TypeError, ValueError), match="rgb_psnr|rgb_ssim|mean|finite"
    ):
        visual_feasible(metrics, summary(), 0.5, 0.01)

    with pytest.raises(
        (KeyError, TypeError, ValueError), match="rgb_psnr|rgb_ssim|mean|finite"
    ):
        visual_feasible(summary(), metrics, 0.5, 0.01)


def test_early_stopper_matches_pilot_schedule_and_counts_boundary_as_stale() -> None:
    stopper = EarlyStopper(minimum_steps=200, patience=4, relative_delta=0.005)

    assert not stopper.update(50, 1.0)
    assert not stopper.update(200, 0.99)
    assert not stopper.update(250, 0.989)
    assert stopper.stale == 1
    assert not stopper.update(300, 0.988)
    assert not stopper.update(350, 0.987)
    assert stopper.update(400, 0.986)
    assert stopper.best == pytest.approx(0.99)
    assert stopper.stale == 4
    assert stopper.last_step == 400


def test_early_stopper_accepts_improvement_at_exact_relative_boundary() -> None:
    stopper = EarlyStopper(minimum_steps=0, patience=1, relative_delta=0.1)
    assert not stopper.update(0, 10.0)
    assert not stopper.update(1, 9.0)
    assert stopper.best == 9.0
    assert stopper.stale == 0


def test_early_stopper_does_not_count_stale_updates_before_minimum_steps() -> None:
    stopper = EarlyStopper(minimum_steps=10, patience=1, relative_delta=0.1)
    assert not stopper.update(0, 1.0)
    assert not stopper.update(5, 1.0)
    assert stopper.stale == 0
    assert stopper.update(10, 1.0)


def test_early_stopper_zero_plateau_exhausts_patience() -> None:
    stopper = EarlyStopper(minimum_steps=0, patience=2, relative_delta=0.1)
    assert not stopper.update(0, 0.0)
    assert not stopper.update(1, 0.0)
    assert stopper.update(2, 0.0)
    assert stopper.best == 0.0
    assert stopper.stale == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_steps": -1, "patience": 1, "relative_delta": 0.1},
        {"minimum_steps": 0, "patience": 0, "relative_delta": 0.1},
        {"minimum_steps": 0, "patience": 1, "relative_delta": 0.0},
        {"minimum_steps": 0, "patience": 1, "relative_delta": 1.0},
        {"minimum_steps": 0, "patience": 1, "relative_delta": math.nan},
        {"minimum_steps": 0.5, "patience": 1, "relative_delta": 0.1},
        {"minimum_steps": True, "patience": 1, "relative_delta": 0.1},
        {"minimum_steps": 0, "patience": 1.5, "relative_delta": 0.1},
    ],
)
def test_early_stopper_rejects_invalid_configuration(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        EarlyStopper(**kwargs)


def test_early_stopper_rejects_bad_values_and_nonincreasing_steps() -> None:
    stopper = EarlyStopper(minimum_steps=0, patience=2, relative_delta=0.1)
    with pytest.raises(ValueError, match="finite"):
        stopper.update(0, math.nan)
    with pytest.raises(ValueError, match="nonnegative"):
        stopper.update(-1, 1.0)

    assert not stopper.update(0, 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        stopper.update(0, 0.8)
    with pytest.raises(ValueError, match="nonnegative"):
        stopper.update(-1, 0.8)
    with pytest.raises(TypeError, match="integer"):
        stopper.update("1", 0.8)


@pytest.mark.parametrize("step", [1.0, 1.5, True])
def test_early_stopper_rejects_nonintegral_or_bool_steps(step) -> None:
    with pytest.raises(TypeError, match="integer"):
        EarlyStopper(0, 1, 0.1).update(step, 1.0)


def test_best_selector_ignores_visual_violation_and_only_replaces_strictly_lower() -> None:
    selector = BestSelector(summary(), psnr_tolerance_db=0.5, ssim_tolerance=0.01)

    assert not selector.consider(10, summary(psnr=29.49, audio=0.1))
    assert selector.best_step is None
    assert selector.last_step == 10
    assert selector.consider(20, summary(psnr=29.5, ssim=0.94, audio=0.8))
    assert selector.best_step == 20
    assert selector.best_audio_total == 0.8
    assert not selector.consider(30, summary(psnr=30.0, audio=0.8))
    assert selector.consider(40, summary(psnr=30.0, audio=0.79))
    assert selector.best_step == 40
    assert selector.last_step == 40


def test_best_selector_requires_increasing_steps_for_every_valid_candidate() -> None:
    selector = BestSelector(summary(), 0.5, 0.01)
    assert not selector.consider(10, summary(psnr=20.0, audio=0.1))
    with pytest.raises(ValueError, match="strictly increasing"):
        selector.consider(10, summary())
    with pytest.raises(ValueError, match="strictly increasing"):
        selector.consider(9, summary())


@pytest.mark.parametrize(
    "audio", [math.nan, math.inf, -math.inf, -0.1, "low", [0.1]]
)
def test_best_selector_rejects_invalid_audio_mean(audio) -> None:
    selector = BestSelector(summary(), 0.5, 0.01)
    with pytest.raises((TypeError, ValueError), match="audio_total|finite|scalar"):
        selector.consider(0, summary(audio=audio))


@pytest.mark.parametrize("candidate", [{}, {"audio_total": {}}, {"audio_total": 0.5}])
def test_best_selector_rejects_missing_or_malformed_summary(candidate) -> None:
    selector = BestSelector(summary(), 0.5, 0.01)
    with pytest.raises((KeyError, TypeError), match="audio_total|rgb_psnr|mean"):
        selector.consider(0, candidate)


@pytest.mark.parametrize("step", [-1, 1.0, 1.5, math.inf, True, "1"])
def test_best_selector_rejects_invalid_step(step) -> None:
    selector = BestSelector(summary(), 0.5, 0.01)
    with pytest.raises((TypeError, ValueError), match="step"):
        selector.consider(step, summary())


def test_best_selector_validates_baseline_and_configuration_at_construction() -> None:
    with pytest.raises(KeyError, match="rgb_psnr"):
        BestSelector({}, 0.5, 0.01)
    with pytest.raises(ValueError, match="nonnegative"):
        BestSelector(summary(), -0.5, 0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"best_step": None, "best_audio_total": 0.1},
        {"best_step": 1, "best_audio_total": math.inf},
        {"best_step": 1, "best_audio_total": -0.1},
        {"best_step": 2, "best_audio_total": 0.1, "last_step": 1},
        {"best_step": 1.0, "best_audio_total": 0.1, "last_step": 1},
    ],
)
def test_best_selector_rejects_corrupt_resume_state(kwargs) -> None:
    with pytest.raises((TypeError, ValueError), match="best|last_step|coherent"):
        BestSelector(summary(), 0.5, 0.01, **kwargs)


def test_external_baseline_and_configuration_mutation_cannot_change_selector() -> None:
    baseline = summary()
    selector = BestSelector(baseline, 0.5, 0.01)
    baseline["rgb_psnr"]["mean"] = 100.0
    assert selector.consider(1, summary(psnr=29.5))
    with pytest.raises(TypeError):
        selector.visual_baseline["rgb_psnr"]["mean"] = 100.0
    with pytest.raises(AttributeError):
        selector.psnr_tolerance_db = 100.0
    with pytest.raises(AttributeError):
        selector.visual_baseline = summary(psnr=0.0)

    stopper = EarlyStopper(0, 2, 0.1)
    with pytest.raises(AttributeError):
        stopper.minimum_steps = 100
    with pytest.raises(AttributeError):
        stopper.relative_delta = 0.9


def test_runtime_state_mutation_is_validated_before_operation() -> None:
    stopper = EarlyStopper(0, 2, 0.1)
    stopper.stale = -1
    with pytest.raises(ValueError, match="stale"):
        stopper.update(0, 1.0)

    selector = BestSelector(summary(), 0.5, 0.01)
    selector.best_audio_total = 0.1
    with pytest.raises(ValueError, match="coherent"):
        selector.consider(1, summary())


def test_early_stopper_rejects_corrupt_resume_state() -> None:
    with pytest.raises(ValueError, match="coherent"):
        EarlyStopper(0, 2, 0.1, best=1.0)
    with pytest.raises(ValueError, match="minimum_steps"):
        EarlyStopper(10, 2, 0.1, best=1.0, stale=1, last_step=5)


@pytest.mark.parametrize("selected", [False, True])
def test_selection_state_has_strict_json_roundtrip(selected) -> None:
    stopper = EarlyStopper(10, 2, 0.1)
    selector = BestSelector(summary(), 0.5, 0.01)
    if selected:
        stopper.update(5, 1.0)
        selector.consider(10, summary(audio=0.8))

    stopper_payload = json.dumps(stopper.state_dict(), allow_nan=False)
    selector_payload = json.dumps(selector.state_dict(), allow_nan=False)
    restored_stopper = EarlyStopper.from_state_dict(json.loads(stopper_payload))
    restored_selector = BestSelector.from_state_dict(json.loads(selector_payload))

    assert restored_stopper.state_dict() == stopper.state_dict()
    assert restored_selector.state_dict() == selector.state_dict()

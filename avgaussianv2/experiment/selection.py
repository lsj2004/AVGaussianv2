"""Visual feasibility, checkpoint selection, and pilot early stopping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real


def _finite_scalar(path: str, value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{path} must be a numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _nonnegative_scalar(name: str, value: object) -> float:
    result = _finite_scalar(name, value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _metric_mean(summary_name: str, summary: object, metric: str) -> float:
    if not isinstance(summary, Mapping):
        raise TypeError(f"{summary_name} must be a mapping")
    if metric not in summary:
        raise KeyError(f"{summary_name} is missing {metric!r}")
    aggregate = summary[metric]
    if not isinstance(aggregate, Mapping):
        raise TypeError(f"{summary_name}[{metric!r}] must be a mapping")
    if "mean" not in aggregate:
        raise KeyError(f"{summary_name}[{metric!r}] is missing 'mean'")
    return _finite_scalar(f"{summary_name}[{metric!r}]['mean']", aggregate["mean"])


def _step(value: object) -> int | float:
    result: int | float
    if isinstance(value, Integral) and not isinstance(value, bool):
        result = int(value)
    else:
        result = _finite_scalar("step", value)
    if result < 0:
        raise ValueError("step must be nonnegative")
    return result


def visual_feasible(
    candidate: object,
    baseline: object,
    psnr_tolerance_db: object,
    ssim_tolerance: object,
) -> bool:
    """Return whether both candidate visual means remain within tolerance."""
    psnr_tolerance = _nonnegative_scalar("psnr_tolerance_db", psnr_tolerance_db)
    ssim_tolerance_value = _nonnegative_scalar("ssim_tolerance", ssim_tolerance)
    candidate_psnr = _metric_mean("candidate", candidate, "rgb_psnr")
    candidate_ssim = _metric_mean("candidate", candidate, "rgb_ssim")
    baseline_psnr = _metric_mean("baseline", baseline, "rgb_psnr")
    baseline_ssim = _metric_mean("baseline", baseline, "rgb_ssim")
    return (
        candidate_psnr >= baseline_psnr - psnr_tolerance
        and candidate_ssim >= baseline_ssim - ssim_tolerance_value
    )


@dataclass
class EarlyStopper:
    """Track relative improvements and signal when patience is exhausted."""

    minimum_steps: float
    patience: int
    relative_delta: float
    best: float = math.inf
    stale: int = 0
    last_step: int | float | None = None

    def __post_init__(self) -> None:
        self.minimum_steps = _nonnegative_scalar("minimum_steps", self.minimum_steps)

        if not isinstance(self.patience, Integral) or isinstance(self.patience, bool):
            raise TypeError("patience must be an integer")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        self.patience = int(self.patience)

        delta = _finite_scalar("relative_delta", self.relative_delta)
        if not 0 < delta < 1:
            raise ValueError("relative_delta must be between 0 and 1")
        self.relative_delta = delta

        if not isinstance(self.best, Real) or isinstance(self.best, bool):
            raise TypeError("best must be a numeric scalar")
        self.best = float(self.best)
        if math.isnan(self.best) or self.best < 0:
            raise ValueError("best must be nonnegative and not NaN")
        if not isinstance(self.stale, Integral) or isinstance(self.stale, bool):
            raise TypeError("stale must be an integer")
        if self.stale < 0:
            raise ValueError("stale must be nonnegative")
        self.stale = int(self.stale)
        if self.last_step is not None:
            self.last_step = _step(self.last_step)

    def update(self, step: object, value: object) -> bool:
        current_step = _step(step)
        current_value = _nonnegative_scalar("value", value)
        if self.last_step is not None and current_step <= self.last_step:
            raise ValueError("step must be strictly increasing")

        self.last_step = current_step
        improved = math.isinf(self.best) or current_value <= self.best * (
            1 - self.relative_delta
        )
        if improved:
            self.best = current_value
            self.stale = 0
        elif current_step >= self.minimum_steps:
            self.stale += 1

        return current_step >= self.minimum_steps and self.stale >= self.patience


@dataclass
class BestSelector:
    """Select the lowest-audio checkpoint among visually feasible candidates."""

    visual_baseline: object
    psnr_tolerance_db: float
    ssim_tolerance: float
    best_step: int | float | None = None
    best_audio_total: float = math.inf

    def __post_init__(self) -> None:
        self.psnr_tolerance_db = _nonnegative_scalar(
            "psnr_tolerance_db", self.psnr_tolerance_db
        )
        self.ssim_tolerance = _nonnegative_scalar("ssim_tolerance", self.ssim_tolerance)
        # Comparing the baseline with itself validates all required visual fields.
        visual_feasible(
            self.visual_baseline,
            self.visual_baseline,
            self.psnr_tolerance_db,
            self.ssim_tolerance,
        )
        if self.best_step is not None:
            self.best_step = _step(self.best_step)
        if not isinstance(self.best_audio_total, Real) or isinstance(
            self.best_audio_total, bool
        ):
            raise TypeError("best_audio_total must be a numeric scalar")
        self.best_audio_total = float(self.best_audio_total)
        if not math.isfinite(self.best_audio_total) and self.best_audio_total != math.inf:
            raise ValueError("best_audio_total must be finite or positive infinity")

    def consider(self, step: object, summary: object) -> bool:
        candidate_step = _step(step)
        audio_total = _metric_mean("summary", summary, "audio_total")
        feasible = visual_feasible(
            summary,
            self.visual_baseline,
            self.psnr_tolerance_db,
            self.ssim_tolerance,
        )
        if not feasible or audio_total >= self.best_audio_total:
            return False
        self.best_step = candidate_step
        self.best_audio_total = audio_total
        return True

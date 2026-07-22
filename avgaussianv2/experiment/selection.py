"""Visual feasibility, checkpoint selection, and pilot early stopping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import ClassVar


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


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _step(value: object) -> int:
    return _nonnegative_integer("step", value)


def _state_mapping(state: object) -> Mapping[str, object]:
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    return state


def _immutable_baseline(baseline: object) -> Mapping[str, Mapping[str, float]]:
    psnr = _metric_mean("visual_baseline", baseline, "rgb_psnr")
    ssim_value = _metric_mean("visual_baseline", baseline, "rgb_ssim")
    return MappingProxyType(
        {
            "rgb_psnr": MappingProxyType({"mean": psnr}),
            "rgb_ssim": MappingProxyType({"mean": ssim_value}),
        }
    )


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

    minimum_steps: int
    patience: int
    relative_delta: float
    best: float = math.inf
    stale: int = 0
    last_step: int | None = None

    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"minimum_steps", "patience", "relative_delta", "_config_locked"}
    )

    def __setattr__(self, name: str, value: object) -> None:
        if (
            self.__dict__.get("_config_locked", False)
            and name in self._IMMUTABLE_FIELDS
        ):
            raise AttributeError(f"{name} is immutable")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_steps",
            _nonnegative_integer("minimum_steps", self.minimum_steps),
        )

        if not isinstance(self.patience, Integral) or isinstance(self.patience, bool):
            raise TypeError("patience must be an integer")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        object.__setattr__(self, "patience", int(self.patience))

        delta = _finite_scalar("relative_delta", self.relative_delta)
        if not 0 < delta < 1:
            raise ValueError("relative_delta must be between 0 and 1")
        object.__setattr__(self, "relative_delta", delta)

        self._validate_runtime_state()
        object.__setattr__(self, "_config_locked", True)

    def _validate_runtime_state(self) -> None:
        if not isinstance(self.best, Real) or isinstance(self.best, bool):
            raise TypeError("best must be a numeric scalar")
        best = float(self.best)
        if (not math.isfinite(best) and best != math.inf) or best < 0:
            raise ValueError("best must be finite nonnegative or positive infinity")
        if not isinstance(self.stale, Integral) or isinstance(self.stale, bool):
            raise TypeError("stale must be an integer")
        if self.stale < 0:
            raise ValueError("stale must be nonnegative")
        last_step = None if self.last_step is None else _step(self.last_step)
        if last_step is None:
            if best != math.inf or self.stale != 0:
                raise ValueError("untouched early-stopper state is not coherent")
        elif not math.isfinite(best):
            raise ValueError("updated early-stopper state must have a finite best")
        elif last_step < self.minimum_steps and self.stale != 0:
            raise ValueError("stale must be zero before minimum_steps")
        self.best = best
        self.stale = int(self.stale)
        if self.last_step is not None:
            self.last_step = last_step

    def update(self, step: object, value: object) -> bool:
        self._validate_runtime_state()
        current_step = _step(step)
        current_value = _nonnegative_scalar("value", value)
        if self.last_step is not None and current_step <= self.last_step:
            raise ValueError("step must be strictly increasing")

        self.last_step = current_step
        improved = math.isinf(self.best) or (
            self.best > 0
            and current_value <= self.best * (1 - self.relative_delta)
        )
        if improved:
            self.best = current_value
            self.stale = 0
        elif current_step >= self.minimum_steps:
            self.stale += 1

        return current_step >= self.minimum_steps and self.stale >= self.patience

    def state_dict(self) -> dict[str, object]:
        """Return state containing no non-standard JSON infinity values."""
        self._validate_runtime_state()
        return {
            "minimum_steps": self.minimum_steps,
            "patience": self.patience,
            "relative_delta": self.relative_delta,
            "best": None if self.best == math.inf else self.best,
            "stale": self.stale,
            "last_step": self.last_step,
        }

    @classmethod
    def from_state_dict(cls, state: object) -> EarlyStopper:
        values = _state_mapping(state)
        best = values["best"]
        return cls(
            minimum_steps=values["minimum_steps"],
            patience=values["patience"],
            relative_delta=values["relative_delta"],
            best=math.inf if best is None else best,
            stale=values["stale"],
            last_step=values["last_step"],
        )


@dataclass
class BestSelector:
    """Select the lowest-audio checkpoint among visually feasible candidates."""

    visual_baseline: object
    psnr_tolerance_db: float
    ssim_tolerance: float
    best_step: int | None = None
    best_audio_total: float = math.inf
    last_step: int | None = None

    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "visual_baseline",
            "psnr_tolerance_db",
            "ssim_tolerance",
            "_config_locked",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        if (
            self.__dict__.get("_config_locked", False)
            and name in self._IMMUTABLE_FIELDS
        ):
            raise AttributeError(f"{name} is immutable")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        baseline = _immutable_baseline(self.visual_baseline)
        object.__setattr__(self, "visual_baseline", baseline)
        object.__setattr__(
            self,
            "psnr_tolerance_db",
            _nonnegative_scalar("psnr_tolerance_db", self.psnr_tolerance_db),
        )
        object.__setattr__(
            self,
            "ssim_tolerance",
            _nonnegative_scalar("ssim_tolerance", self.ssim_tolerance),
        )
        self._validate_runtime_state()
        object.__setattr__(self, "_config_locked", True)

    def _validate_runtime_state(self) -> None:
        best_step = (
            None
            if self.best_step is None
            else _nonnegative_integer("best_step", self.best_step)
        )
        last_step = (
            None
            if self.last_step is None
            else _nonnegative_integer("last_step", self.last_step)
        )
        if not isinstance(self.best_audio_total, Real) or isinstance(
            self.best_audio_total, bool
        ):
            raise TypeError("best_audio_total must be a numeric scalar")
        best_audio = float(self.best_audio_total)
        if best_step is None:
            if best_audio != math.inf:
                raise ValueError("best selector state is not coherent")
        elif not math.isfinite(best_audio) or best_audio < 0:
            raise ValueError("selected best_audio_total must be finite and nonnegative")
        if best_step is not None and (last_step is None or best_step > last_step):
            raise ValueError("best_step must not exceed last_step")
        self.best_step = best_step
        self.best_audio_total = best_audio
        self.last_step = last_step

    def consider(self, step: object, summary: object) -> bool:
        self._validate_runtime_state()
        candidate_step = _step(step)
        if self.last_step is not None and candidate_step <= self.last_step:
            raise ValueError("step must be strictly increasing")
        audio_total = _nonnegative_scalar(
            "summary['audio_total']['mean']",
            _metric_mean("summary", summary, "audio_total"),
        )
        feasible = visual_feasible(
            summary,
            self.visual_baseline,
            self.psnr_tolerance_db,
            self.ssim_tolerance,
        )
        self.last_step = candidate_step
        if not feasible or audio_total >= self.best_audio_total:
            return False
        self.best_step = candidate_step
        self.best_audio_total = audio_total
        return True

    def state_dict(self) -> dict[str, object]:
        """Return state containing no non-standard JSON infinity values."""
        self._validate_runtime_state()
        return {
            "visual_baseline": {
                metric: {"mean": aggregate["mean"]}
                for metric, aggregate in self.visual_baseline.items()
            },
            "psnr_tolerance_db": self.psnr_tolerance_db,
            "ssim_tolerance": self.ssim_tolerance,
            "best_step": self.best_step,
            "best_audio_total": (
                None if self.best_audio_total == math.inf else self.best_audio_total
            ),
            "last_step": self.last_step,
        }

    @classmethod
    def from_state_dict(cls, state: object) -> BestSelector:
        values = _state_mapping(state)
        best_audio = values["best_audio_total"]
        return cls(
            visual_baseline=values["visual_baseline"],
            psnr_tolerance_db=values["psnr_tolerance_db"],
            ssim_tolerance=values["ssim_tolerance"],
            best_step=values["best_step"],
            best_audio_total=math.inf if best_audio is None else best_audio,
            last_step=values["last_step"],
        )

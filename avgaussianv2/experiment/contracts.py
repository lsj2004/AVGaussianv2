from __future__ import annotations

from dataclasses import dataclass

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility with Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        __str__ = str.__str__
        __format__ = str.__format__


class Variant(StrEnum):
    JOINT_CONDITIONED = "joint_conditioned"
    FROZEN_VISUAL = "frozen_visual"
    CONDITION_OFF = "condition_off"


@dataclass(frozen=True)
class PilotConfig:
    warmup_steps: int = 200
    joint_steps: int = 500
    validation_interval: int = 50
    minimum_joint_steps: int = 200
    patience: int = 4
    minimum_relative_improvement: float = 0.005
    quick_validation_samples: int = 32
    psnr_tolerance_db: float = 0.5
    ssim_tolerance: float = 0.01

    def validate(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.joint_steps <= 0:
            raise ValueError("joint_steps must be positive")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if not 0 < self.minimum_relative_improvement < 1:
            raise ValueError("minimum_relative_improvement must be between 0 and 1")


@dataclass(frozen=True)
class VariantIndices:
    warmup: tuple[int, ...]
    joint: tuple[int, ...]


@dataclass(frozen=True)
class SharedIndices:
    warmup: tuple[int, ...]
    joint: tuple[int, ...]

    def for_variant(self, variant: Variant) -> VariantIndices:
        warmup = () if variant == Variant.CONDITION_OFF else self.warmup
        return VariantIndices(warmup=warmup, joint=self.joint)


@dataclass(frozen=True)
class EvaluationResult:
    system_name: str
    count: int
    rows: tuple[dict[str, object], ...]
    summary: dict[str, dict[str, float]]

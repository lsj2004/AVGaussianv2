"""Contracts and deterministic sampling for diagnostic experiments."""

from avgaussianv2.experiment.contracts import (
    EvaluationResult,
    PilotConfig,
    SharedIndices,
    Variant,
    VariantIndices,
)
from avgaussianv2.experiment.sampling import build_shared_indices, evenly_spaced_indices
from avgaussianv2.experiment.selection import BestSelector, EarlyStopper, visual_feasible

__all__ = [
    "EvaluationResult",
    "PilotConfig",
    "SharedIndices",
    "Variant",
    "VariantIndices",
    "BestSelector",
    "EarlyStopper",
    "build_shared_indices",
    "evenly_spaced_indices",
    "visual_feasible",
]

"""Contracts and deterministic sampling for diagnostic experiments."""

from avgaussianv2.experiment.contracts import (
    EvaluationResult,
    PilotConfig,
    SharedIndices,
    Variant,
    VariantIndices,
)
from avgaussianv2.experiment.sampling import build_shared_indices, evenly_spaced_indices

__all__ = [
    "EvaluationResult",
    "PilotConfig",
    "SharedIndices",
    "Variant",
    "VariantIndices",
    "build_shared_indices",
    "evenly_spaced_indices",
]

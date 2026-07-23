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
from avgaussianv2.experiment.checkpoint import (
    PilotCheckpointStore,
    PilotCompatibility,
    PilotResumeError,
    build_run_fingerprint,
    hash_index_manifest,
    sha256_file,
    validate_compatibility,
)
from avgaussianv2.experiment.report import (
    ComparisonResult,
    PilotDecision,
    REQUIRED_SYSTEMS,
    SystemReportInput,
    build_comparison,
    decide_long_training,
    paired_audio_deltas,
)

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
    "PilotCheckpointStore",
    "PilotCompatibility",
    "PilotResumeError",
    "build_run_fingerprint",
    "hash_index_manifest",
    "sha256_file",
    "validate_compatibility",
    "ComparisonResult",
    "PilotDecision",
    "REQUIRED_SYSTEMS",
    "SystemReportInput",
    "build_comparison",
    "decide_long_training",
    "paired_audio_deltas",
]

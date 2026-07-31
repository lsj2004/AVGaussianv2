"""Evaluate cross-attention and its causal RGBD ablations on cam38."""

from __future__ import annotations

from avgaussianv2.benchmark.cross_attention_ablation import (
    ALL_CAUSAL_EVALUATION_SYSTEMS,
    verify_cross_attention_preparation,
)
from avgaussianv2.cli.ablation_runner import run_ablation_evaluation


def main() -> None:
    # Training remains bound to the immutable preparation revision and worker
    # fingerprint. Evaluation code may receive audit-only fixes afterwards;
    # strict checkpoint/runtime evidence still rejects model,
    # configuration, source-inventory or data changes.
    run_ablation_evaluation(
        verify_cross_attention_preparation,
        result_label="system",
        system_choices=ALL_CAUSAL_EVALUATION_SYSTEMS,
        allowed_systems_from_preparation="causal_evaluation_systems",
        verify_kwargs={"require_repository_match": False},
    )


if __name__ == "__main__":
    main()

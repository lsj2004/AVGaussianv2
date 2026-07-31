"""Evaluate one B/C milestone under the exact preparation Git revision."""

from __future__ import annotations

from avgaussianv2.benchmark.architecture_ablation import (
    verify_architecture_preparation,
)
from avgaussianv2.cli.ablation_runner import run_ablation_evaluation


def main() -> None:
    run_ablation_evaluation(
        verify_architecture_preparation,
        result_label="strategy",
        system_from_preparation="evaluation_system",
    )


if __name__ == "__main__":
    main()

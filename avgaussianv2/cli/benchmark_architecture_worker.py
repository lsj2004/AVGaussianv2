"""Run one audio-render architecture worker under its pinned preparation."""

# ruff: noqa: E402 -- deterministic process settings precede Torch imports.

from __future__ import annotations

from avgaussianv2.cli.ablation_runner import (
    ensure_deterministic_environment,
    run_ablation_worker,
)


if __name__ == "__main__":
    ensure_deterministic_environment(
        marker="AVGAUSSIANV2_ARCHITECTURE_WORKER_REEXEC",
        module_name="avgaussianv2.cli.benchmark_architecture_worker",
        experiment_name="architecture",
    )

from avgaussianv2.benchmark.architecture_ablation import (
    verify_architecture_preparation,
)


def main() -> None:
    run_ablation_worker(
        verify_architecture_preparation,
        result_label="strategy",
    )


if __name__ == "__main__":
    main()

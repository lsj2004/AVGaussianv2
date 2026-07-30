"""Run a cross-attention worker under its pinned preparation revision."""

# ruff: noqa: E402 -- deterministic process settings precede Torch imports.

from __future__ import annotations

from avgaussianv2.cli.ablation_runner import (
    ensure_deterministic_environment,
    run_ablation_worker,
)


if __name__ == "__main__":
    ensure_deterministic_environment(
        marker="AVGAUSSIANV2_CROSS_ATTENTION_WORKER_REEXEC",
        module_name="avgaussianv2.cli.benchmark_cross_attention_worker",
        experiment_name="cross-attention",
    )

from avgaussianv2.benchmark.cross_attention_ablation import (
    verify_cross_attention_preparation,
)


def main() -> None:
    run_ablation_worker(
        verify_cross_attention_preparation,
        result_label="system",
    )


if __name__ == "__main__":
    main()

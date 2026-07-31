from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import avgaussianv2.benchmark.evaluation as evaluation
import avgaussianv2.benchmark.production as production
from avgaussianv2.cli.ablation_runner import run_ablation_evaluation


def test_ablation_evaluation_accepts_and_forwards_compute_dpam(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark-architecture-eval",
            "--protocol-dir",
            str(tmp_path / "protocol"),
            "--worker-dir",
            str(tmp_path / "worker"),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--step",
            "5000",
            "--compute-dpam",
        ],
    )
    monkeypatch.setattr(production, "expected_identity", lambda *args: args)
    monkeypatch.setattr(
        production,
        "continuation_training_evidence",
        lambda *args, **kwargs: (args, kwargs),
    )

    def adapters(**kwargs):
        captured.update(kwargs)
        return lambda: None, lambda _: None

    monkeypatch.setattr(production, "build_evaluation_adapters", adapters)

    class Evaluator:
        def __init__(self, device):
            captured["device"] = device

        def evaluate(self, **kwargs):
            captured["evaluate"] = kwargs
            return SimpleNamespace(count=130, content_sha256="a" * 64)

    monkeypatch.setattr(evaluation, "BenchmarkEvaluator", Evaluator)
    run_ablation_evaluation(
        lambda _path: {
            "scene_id": "scene1_opera",
            "evaluation_system": "plain_unet",
            "strategy": "plain_unet",
        },
        result_label="strategy",
        system_from_preparation="evaluation_system",
    )

    assert captured["compute_dpam"] is True

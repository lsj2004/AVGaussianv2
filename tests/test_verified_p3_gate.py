from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_verified_p3_30k_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verified_p3_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    run_dir = tmp_path / "candidate__scene1_opera__seed42"
    run_dir.mkdir()
    (run_dir / "continuation_identity.json").write_text(
        json.dumps(
            {
                "continuation_id": run_dir.name,
                "system": "query_dependent_p1",
                "scene": "scene1_opera",
                "seed": 42,
            }
        )
    )
    rows = (
        {"sample_id": "scene1_opera/cam38/000000", "audio_total": 0.2},
        {"sample_id": "scene1_opera/cam38/000001", "audio_total": 0.4},
    )
    evaluation = SimpleNamespace(
        identity=SimpleNamespace(
            scene_id="scene1_opera",
            system_name="query_dependent_p1",
            reporting_step=30_000,
            expected_sample_ids=tuple(row["sample_id"] for row in rows),
            to_mapping=lambda: {"fixed": True},
        ),
        provenance={
            "seed": 42,
            "checkpoint_step": 30_000,
            "checkpoint_sha256": "a" * 64,
            "main_update_matched": True,
        },
        count=2,
        rows=rows,
        summary={"audio_total": {"mean": 0.3}},
        metric_protocol={"fixed": True},
        content_sha256="b" * 64,
    )
    return run_dir, evaluation


def test_strict_loader_uses_full_evaluation_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    run_dir, evaluation = _fixture(tmp_path)
    expected_dir = run_dir / "evaluations/step_030000"

    def verify(path: Path):
        assert path == expected_dir
        return evaluation

    monkeypatch.setattr(module, "verify_evaluation", verify)

    loaded = module.StrictEvaluationLoader()(
        run_dir,
        30_000,
        "query_dependent_p1",
        "query_dependent_p1",
        ("audio_total",),
    )

    assert loaded["metrics"] == {"audio_total": pytest.approx(0.3)}
    assert loaded["checkpoint_sha256"] == "a" * 64


def test_strict_loader_propagates_training_evidence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    run_dir, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_evaluation",
        lambda _path: (_ for _ in ()).throw(RuntimeError("checkpoint mismatch")),
    )

    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        module.StrictEvaluationLoader()(
            run_dir,
            30_000,
            "query_dependent_p1",
            "query_dependent_p1",
            ("audio_total",),
        )


def test_strict_loader_rejects_metric_protocol_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    run_dir, evaluation = _fixture(tmp_path)
    monkeypatch.setattr(module, "verify_evaluation", lambda _path: evaluation)
    loader = module.StrictEvaluationLoader()
    loader(
        run_dir,
        30_000,
        "query_dependent_p1",
        "query_dependent_p1",
        ("audio_total",),
    )
    evaluation.metric_protocol = {"changed": True}

    with pytest.raises(ValueError, match="metric protocol mismatch"):
        loader(
            run_dir,
            30_000,
            "query_dependent_p1",
            "query_dependent_p1",
            ("audio_total",),
        )


def test_gate_implementation_is_content_pinned(tmp_path: Path) -> None:
    module = _load_module()
    implementation = tmp_path / "gate.py"
    implementation.write_text("def main(): pass\n")

    with pytest.raises(ValueError, match="implementation hash mismatch"):
        module._load_gate_module(implementation, "0" * 64)

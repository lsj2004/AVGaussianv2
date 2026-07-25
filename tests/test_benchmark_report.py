from __future__ import annotations

import hashlib
import json

import pytest

from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluationResult,
    EvaluationIdentity,
)
from avgaussianv2.benchmark.report import (
    BenchmarkReportError,
    build_scene_report,
    build_suite_report,
    load_report,
)
from avgaussianv2.experiment.metrics import aggregate_metrics


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _result(scene, system, step, count, offset=0.0, role="continuation"):
    ids = tuple(f"{scene}/cam38/{index:06d}" for index in range(count))
    rows = tuple(
        {
            "sample_id": sample_id,
            "scene_id": scene,
            "camera": "cam38",
            "frame_index": index,
            "time_seconds": index / 10,
            "audio_total": 1.0 + offset + index / 100,
            "audio_mono": 1.1 + offset,
            "audio_diff": 1.2 + offset,
            "waveform_l1": 0.1 + offset,
            "mono_lsd": 0.2 + offset,
            "diff_lsd": 0.3 + offset,
            "lre_error_db": 0.4 + offset,
            "rgb_psnr": 20.0 - offset,
            "rgb_ssim": 0.8 - offset / 10,
            "rgb_l1": 0.05 + offset,
        }
        for index, sample_id in enumerate(ids)
    )
    metadata = {"sample_id", "scene_id", "camera", "frame_index", "time_seconds"}
    metrics = tuple(metric for metric in rows[0] if metric not in metadata)
    summary = aggregate_metrics(
        [{metric: float(row[metric]) for metric in metrics} for row in rows]
    )
    native_updates = 61 if system == "native_audiogs" else 30_000
    return BenchmarkEvaluationResult(
        identity=EvaluationIdentity(scene, system, step, ids, count),
        count=count,
        rows=rows,
        summary=summary,
        provenance={
            "system_name": system,
            "scene_id": scene,
            "role": role,
            "update_matched": role == "continuation",
            "test_camera": "cam38",
            "train_cameras": [f"cam{x:02d}" for x in range(38)],
            "test_targets_read_during_training": False,
            "seed": 42,
            "planned_updates": (
                30_000 if role == "continuation" else native_updates
            ),
            "completed_updates": (
                step if role == "continuation" else native_updates
            ),
            "checkpoint_step": (
                step if role == "continuation" else native_updates
            ),
            "checkpoint_path": f"/tmp/{scene}-{system}-{step}.pt",
            "checkpoint_sha256": _sha(f"{scene}-{system}-{step}"),
            "config_sha256": _sha(f"{scene}-config"),
            "source_sha256": _sha("source"),
            "visual_initialization_sha256": _sha(f"{scene}-visual"),
            "audio_initialization_sha256": _sha(f"{scene}-audio"),
            "model_initialization_sha256": _sha(f"{scene}-model"),
            "index_sha256": _sha(f"{scene}-indices") if role == "continuation" else None,
            "batch_size": 1,
            "epochs": 61.0 if system == "native_audiogs" else None,
        },
        content_sha256=_sha(f"{scene}-{system}-{step}-content"),
        generation_path=None,
    )


def _scene_inputs(scene, count):
    values = []
    for step in (5_000, 10_000, 30_000):
        values.extend(
            [
                _result(scene, "joint_conditioned", step, count, 0.0),
                _result(scene, "audio_only", step, count, 0.2),
                _result(scene, "visual_only", step, count, 0.1),
            ]
        )
    values.extend(
        [
            _result(scene, "native_audiogs", None, count, 0.3, "native_reference"),
            _result(scene, "native_ftgspp", None, count, 0.4, "native_reference"),
        ]
    )
    return values


def test_scene_report_has_scaling_paired_deltas_win_rates_and_native_labels(tmp_path):
    report = build_scene_report(
        scene_id="scene1_opera",
        evaluations=_scene_inputs("scene1_opera", 2),
        expected_sample_count=2,
        output_dir=tmp_path,
        strict_protocol=False,
    )
    assert report["primary_step"] == 30_000
    assert set(report["scaling"]) == {"5000", "10000", "30000"}
    audio = report["paired"]["joint_vs_audio_only"]["audio_total"]
    assert audio["mean_delta"] == pytest.approx(-0.2)
    assert audio["win_rate"] == 1.0
    video = report["paired"]["joint_vs_visual_only"]["rgb_psnr"]
    assert video["win_rate"] == 1.0
    assert report["systems"]["native_audiogs"]["comparison_class"] == "native_reference_non_update_matched"
    assert (
        report["descriptive_native_comparisons"]["joint_vs_native_audiogs"][
            "comparison_class"
        ]
        == "descriptive_non_update_matched"
    )
    loaded = load_report(tmp_path)
    assert loaded["content_sha256"] == report["content_sha256"]
    assert (tmp_path / "current.json").read_bytes()


def test_scene_report_rejects_different_ids_or_fairness_identity(tmp_path):
    values = _scene_inputs("scene1_opera", 2)
    broken = values[1]
    broken.provenance["index_sha256"] = _sha("different")
    with pytest.raises(BenchmarkReportError, match="index"):
        build_scene_report(
            scene_id="scene1_opera",
            evaluations=values,
            expected_sample_count=2,
            output_dir=tmp_path,
            strict_protocol=False,
        )


def test_suite_reports_macro_and_sample_weighted_micro(tmp_path):
    first = build_scene_report(
        scene_id="scene1_opera",
        evaluations=_scene_inputs("scene1_opera", 130),
        expected_sample_count=130,
        output_dir=tmp_path / "first",
        strict_protocol=False,
    )
    second = build_scene_report(
        scene_id="Scene7playing",
        evaluations=_scene_inputs("Scene7playing", 293),
        expected_sample_count=293,
        output_dir=tmp_path / "second",
        strict_protocol=False,
    )
    suite = build_suite_report(
        scene_reports=[first, second],
        output_dir=tmp_path / "suite",
        strict_protocol=False,
    )
    assert suite["scene_sample_counts"] == {
        "Scene7playing": 293,
        "scene1_opera": 130,
    }
    assert "macro" in suite["aggregates"]
    assert "micro" in suite["aggregates"]


def test_report_resume_is_zero_mutation_and_tamper_is_rejected(tmp_path):
    args = dict(
        scene_id="scene1_opera",
        evaluations=_scene_inputs("scene1_opera", 2),
        expected_sample_count=2,
        output_dir=tmp_path,
        strict_protocol=False,
    )
    first = build_scene_report(**args)
    before = (tmp_path / "current.json").stat().st_mtime_ns
    second = build_scene_report(**args, resume=True)
    assert second["content_sha256"] == first["content_sha256"]
    assert (tmp_path / "current.json").stat().st_mtime_ns == before
    pointer = json.loads((tmp_path / "current.json").read_text())
    report_path = tmp_path / pointer["generation"] / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(BenchmarkReportError, match="report"):
        load_report(tmp_path)

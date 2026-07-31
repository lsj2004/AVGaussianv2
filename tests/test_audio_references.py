import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import soundfile as sf
import torch
import yaml

from avgaussianv2.benchmark.audio_references import (
    CDPAMMetric,
    evaluate_reference_baselines,
    paper_audio_metrics,
    reference_prediction,
    verify_reference_evaluation,
    write_reference_evaluation,
)


class _CDPAMState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class _CDPAMWrapper:
    def __init__(self) -> None:
        self.model = _CDPAMState()


def test_cdpam_protocol_hashes_wrapped_model_state() -> None:
    metric = CDPAMMetric.__new__(CDPAMMetric)
    metric._model = _CDPAMWrapper()
    metric._cdpam = SimpleNamespace(__file__=__file__)

    protocol = metric.protocol

    assert protocol["implementation"].endswith("._CDPAMWrapper")
    assert protocol["weight_module"].endswith("._CDPAMState")
    assert len(protocol["model_state_sha256"]) == 64


def _write_config(root: Path, scene: str, offset: float) -> Path:
    root.mkdir()
    sample_rate = 1_000
    length = 500
    time = torch.arange(length, dtype=torch.float32) / sample_rate
    source = torch.stack(
        (
            0.5 * torch.sin(2.0 * torch.pi * 30.0 * time),
            0.25 * torch.cos(2.0 * torch.pi * 40.0 * time),
        ),
        dim=1,
    ).numpy()
    target = source.copy()
    target[:, 0] *= 0.5 + offset
    target[:, 1] *= 1.5 - offset
    source_path = root / "source.wav"
    target_path = root / "cam38.wav"
    sf.write(source_path, source, sample_rate, subtype="FLOAT")
    sf.write(target_path, target, sample_rate, subtype="FLOAT")
    manifest = {
        "scene_id": scene,
        "frame_times": [0.1, 0.2, 0.3],
        "audio_times": [0.1, 0.2, 0.3],
        "cameras": {
            "cam00": {
                "audio_path": str(target_path),
                "video_path": str(root / "missing-cam00.mp4"),
            },
            "cam38": {
                "audio_path": str(target_path),
                "video_path": str(root / "missing-cam38.mp4"),
            },
        },
        "audio": {
            "sample_rate": sample_rate,
            "source_path": str(source_path),
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "scene": {
            "id": scene,
            "fps": 10.0,
            "train_cameras": ["cam00"],
            "eval_cameras": ["cam38"],
            "camera_mapping": {"cam00": 0, "cam38": 38},
        },
        "paths": {
            "visual_upstream_root": str(root / "visual"),
            "audio_upstream_root": str(root / "audio"),
            "visual_checkpoint": str(root / "visual.pt"),
            "audio_checkpoint": str(root / "audio.pt"),
            "manifest": str(manifest_path),
        },
        "model": {"sample_rate": sample_rate},
        "train": {"crop_seconds": 0.02},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def test_reference_predictions_match_paper_baselines() -> None:
    source = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    passthrough = reference_prediction(source, "source_binaural")
    mono = reference_prediction(source, "mono")

    torch.testing.assert_close(passthrough, source)
    torch.testing.assert_close(mono[:, 0], torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(mono[:, 1], mono[:, 0])
    with pytest.raises(ValueError, match="unsupported"):
        reference_prediction(source, "unknown")


def test_paper_audio_metrics_requires_finite_dpam() -> None:
    audio = torch.ones(1, 2, 640)

    with pytest.raises(ValueError, match="DPAM"):
        paper_audio_metrics(
            audio,
            audio,
            sample_rate=16_000,
            dpam_metric=lambda *_: float("nan"),
        )


def test_reference_evaluation_writes_paired_rows_aggregates_and_hashes(
    tmp_path: Path,
) -> None:
    first = _write_config(tmp_path / "scene1", "scene1_opera", 0.0)
    second = _write_config(tmp_path / "scene7", "Scene7playing", 0.1)
    calls = []

    def dpam(predicted, target, sample_rate):
        calls.append(sample_rate)
        return float((predicted - target).abs().mean())

    evaluation = evaluate_reference_baselines(
        [first, second],
        dpam_metric=dpam,
        repository={"root": str(tmp_path), "commit": "1" * 40, "clean": True},
    )

    assert len(evaluation.rows) == 12
    assert len(calls) == 12
    assert set(evaluation.report["per_scene"]) == {
        "source_binaural",
        "mono",
    }
    assert evaluation.report["metric_protocol"]["paper_dpam"]["status"] == "computed"
    assert evaluation.report["repository"]["clean"] is True
    for row in evaluation.rows:
        assert set(row) >= {
            "sample_id",
            "scene_id",
            "baseline",
            "paper_mag",
            "paper_env",
            "paper_lre_db",
            "paper_dpam",
        }
    paired = {}
    for row in evaluation.rows:
        paired.setdefault(row["sample_id"], {})[row["baseline"]] = row
    assert all(set(value) == {"source_binaural", "mono"} for value in paired.values())
    assert all(
        value["mono"]["paper_lre_db"] != value["source_binaural"]["paper_lre_db"]
        for value in paired.values()
    )

    output = tmp_path / "output"
    write_reference_evaluation(evaluation, output)
    verification = json.loads((output / "verification.json").read_text())
    assert set(verification["files"]) == {
        "metrics_per_sample.jsonl",
        "metrics_per_sample.csv",
        "aggregate.json",
    }
    for name, digest in verification["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    verified = verify_reference_evaluation(output)
    assert verified["row_count"] == 12
    with pytest.raises(FileExistsError, match="already exists"):
        write_reference_evaluation(evaluation, output)
    write_reference_evaluation(evaluation, output, overwrite=True)

    (output / "metrics_per_sample.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_reference_evaluation(output)


def test_skipped_dpam_is_explicit_and_not_present_in_rows(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "scene", "scene1_opera", 0.0)

    evaluation = evaluate_reference_baselines([config], dpam_metric=None)

    assert evaluation.report["metric_protocol"]["paper_dpam"]["status"] == "skipped"
    assert all("paper_dpam" not in row for row in evaluation.rows)

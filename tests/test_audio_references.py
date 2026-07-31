import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch
import yaml

from avgaussianv2.benchmark.audio_references import (
    CDPAMMetric,
    ExternalCDPAMMetric,
    evaluate_reference_baselines,
    paper_audio_metrics,
    reference_prediction,
    serve_cdpam_worker,
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


def test_cdpam_worker_protocol_is_persistent_and_fail_closed(tmp_path) -> None:
    class Metric:
        protocol = {"model_state_sha256": "a" * 64}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def evaluate_files(self, predicted, target):
            assert predicted == tmp_path / "predicted.wav"
            assert target == tmp_path / "target.wav"
            return 0.25

    source = io.StringIO(
        "\n".join(
            (
                json.dumps(
                    {
                        "request_id": 7,
                        "predicted_path": str(tmp_path / "predicted.wav"),
                        "target_path": str(tmp_path / "target.wav"),
                    }
                ),
                "[]",
                json.dumps({"command": "close"}),
            )
        )
        + "\n"
    )
    output = io.StringIO()

    assert serve_cdpam_worker(
        input_stream=source, output_stream=output, metric_factory=Metric
    ) == 0

    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[0]["status"] == "ready"
    assert rows[1] == {"request_id": 7, "status": "ok", "value": 0.25}
    assert rows[2]["status"] == "error"
    assert rows[2]["error"]["type"] == "TypeError"


def test_external_cdpam_metric_uses_isolated_interpreter(tmp_path, monkeypatch) -> None:
    (tmp_path / "yaml.py").write_text("# import stub for isolated worker test\n")
    stub = tmp_path / "cdpam.py"
    stub.write_text(
        "import soundfile\n"
        "import torch\n"
        "class _State(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.weight = torch.nn.Parameter(torch.tensor([1.0]))\n"
        "class CDPAM:\n"
        "    def __init__(self):\n"
        "        self.model = _State()\n"
        "    def forward(self, reference, output):\n"
        "        return (reference - output).abs().mean().reshape(1)\n"
        "def load_audio(path):\n"
        "    value, _ = soundfile.read(path, always_2d=True)\n"
        "    return torch.tensor(value.T, dtype=torch.float32).unsqueeze(0)\n"
    )
    inherited = os.pathsep.join(path for path in sys.path if path)
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + (os.pathsep + inherited if inherited else "")
    )
    predicted = torch.zeros(1, 2, 64)
    target = torch.ones(1, 2, 64) * 0.5

    with ExternalCDPAMMetric(sys.executable) as metric:
        value = metric(predicted, target, 16_000)
        protocol = metric.protocol

    assert value == pytest.approx(0.5, abs=2e-4)
    assert protocol["execution_mode"] == "persistent_external_worker"
    assert protocol["python_executable"] == str(Path(sys.executable).resolve())
    assert len(protocol["python_executable_sha256"]) == 64
    assert len(protocol["bridge_source_sha256"]) == 64
    assert len(protocol["worker_runtime_source_sha256"]) == 64
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

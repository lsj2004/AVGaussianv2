from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

import avgaussianv2.benchmark.metrics as metrics_module
from avgaussianv2.benchmark.artifacts import atomic_write, canonical_json
from avgaussianv2.benchmark.metrics import (
    aggregate_metrics,
    paper_envelope_distance,
    paper_lre_error_db,
    paper_magnitude_distance,
)
from avgaussianv2.config import load_project_config
from avgaussianv2.data.aligned import AlignedAVDataset


SCHEMA = "avgaussianv2.audiogs-paper-reference-baselines"
BASELINES = ("source_binaural", "mono")
METRICS = ("paper_mag", "paper_env", "paper_lre_db", "paper_dpam")
DPAMMetric = Callable[[Tensor, Tensor, int], float]


@dataclass(frozen=True)
class ReferenceEvaluation:
    rows: tuple[dict[str, object], ...]
    report: dict[str, object]


def reference_prediction(source_audio: Tensor, baseline: str) -> Tensor:
    if source_audio.ndim != 3 or source_audio.shape[1] != 2:
        raise ValueError("reference source audio must have shape [B,2,samples]")
    if baseline == "source_binaural":
        return source_audio
    if baseline == "mono":
        return source_audio.mean(dim=1, keepdim=True).expand(-1, 2, -1)
    raise ValueError(f"unsupported audio reference baseline: {baseline!r}")


def paper_audio_metrics(
    predicted: Tensor,
    target: Tensor,
    *,
    dpam_metric: DPAMMetric | None = None,
    sample_rate: int,
) -> dict[str, float]:
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
    ):
        raise ValueError("paper metric sample rate must be a positive integer")
    result = {
        "paper_mag": paper_magnitude_distance(predicted, target),
        "paper_env": paper_envelope_distance(predicted, target),
        "paper_lre_db": paper_lre_error_db(predicted, target),
    }
    if dpam_metric is not None:
        value = float(dpam_metric(predicted, target, sample_rate))
        if not math.isfinite(value):
            raise ValueError("paper DPAM must be finite")
        result["paper_dpam"] = value
    return result


class CDPAMMetric:
    """Paper-compatible CDPAM adapter with one shared model and isolated WAV I/O."""

    def __init__(self) -> None:
        try:
            import cdpam
            import soundfile
        except ImportError as error:
            raise RuntimeError(
                "DPAM requires the cdpam and soundfile packages; "
                "use --skip-dpam only for an explicitly incomplete report"
            ) from error
        self._cdpam = cdpam
        self._soundfile = soundfile
        self._model = cdpam.CDPAM()
        self._temporary = tempfile.TemporaryDirectory(
            prefix="avgaussianv2-paper-dpam-"
        )
        self._index = 0

    @property
    def protocol(self) -> dict[str, object]:
        import inspect

        state_digest = hashlib.sha256()
        for name, value in sorted(self._model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            state_digest.update(name.encode())
            state_digest.update(str(tensor.dtype).encode())
            state_digest.update(canonical_json(list(tensor.shape)))
            state_digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        module_path = Path(self._cdpam.__file__).resolve()
        class_path_value = inspect.getsourcefile(type(self._model))
        if class_path_value is None:
            raise RuntimeError("cannot locate CDPAM implementation source")
        class_path = Path(class_path_value).resolve()
        return {
            "implementation": f"{type(self._model).__module__}.{type(self._model).__qualname__}",
            "module_path": str(module_path),
            "module_sha256": _sha256(module_path),
            "class_source_path": str(class_path),
            "class_source_sha256": _sha256(class_path),
            "model_state_sha256": state_digest.hexdigest(),
        }

    def __call__(self, predicted: Tensor, target: Tensor, sample_rate: int) -> float:
        self._index += 1
        root = Path(self._temporary.name)
        predicted_path = root / f"predicted-{self._index:08d}.wav"
        target_path = root / f"target-{self._index:08d}.wav"
        self._soundfile.write(
            predicted_path,
            predicted.detach().cpu().squeeze(0).transpose(0, 1).numpy(),
            sample_rate,
        )
        self._soundfile.write(
            target_path,
            target.detach().cpu().squeeze(0).transpose(0, 1).numpy(),
            sample_rate,
        )
        reference = self._cdpam.load_audio(str(target_path))
        output = self._cdpam.load_audio(str(predicted_path))
        with torch.no_grad():
            value = self._model.forward(reference, output)
        predicted_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)
        return float(value.detach().cpu().reshape(-1)[0])

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> CDPAMMetric:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(scene_id: str, camera: str, frame_index: int) -> str:
    return f"{scene_id}/{camera}/{frame_index:06d}"


def _scene_aggregate(
    rows: Sequence[dict[str, object]],
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    scenes = tuple(dict.fromkeys(str(row["scene_id"]) for row in rows))
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for baseline in BASELINES:
        result[baseline] = {}
        for scene in scenes:
            selected = [
                {
                    name: float(row[name])
                    for name in metric_names
                }
                for row in rows
                if row["baseline"] == baseline and row["scene_id"] == scene
            ]
            if not selected:
                raise ValueError(f"missing {baseline} rows for scene {scene}")
            result[baseline][scene] = aggregate_metrics(selected)
    return result


def _combined_aggregate(
    rows: Sequence[dict[str, object]],
    per_scene: dict[str, dict[str, dict[str, dict[str, float]]]],
    metric_names: tuple[str, ...],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for baseline in BASELINES:
        baseline_rows = [
            {name: float(row[name]) for name in metric_names}
            for row in rows
            if row["baseline"] == baseline
        ]
        micro = aggregate_metrics(baseline_rows)
        macro = aggregate_metrics(
            [
                {
                    name: scene_metrics[name]["mean"]
                    for name in metric_names
                }
                for scene_metrics in per_scene[baseline].values()
            ]
        )
        result[baseline] = {
            "sample_weighted_micro": micro,
            "scene_macro": macro,
        }
    return result


def evaluate_reference_baselines(
    config_paths: Sequence[Path],
    *,
    dpam_metric: DPAMMetric | None,
) -> ReferenceEvaluation:
    if not config_paths:
        raise ValueError("at least one reference config is required")
    rows: list[dict[str, object]] = []
    inputs = []
    seen_scenes: set[str] = set()
    for raw_path in config_paths:
        config_path = Path(raw_path).resolve()
        config = load_project_config(config_path)
        if config.scene.scene_id in seen_scenes:
            raise ValueError(f"duplicate scene config: {config.scene.scene_id}")
        seen_scenes.add(config.scene.scene_id)
        dataset = AlignedAVDataset(config, split="eval", audio_only=True)
        sample_ids = []
        for index in range(len(dataset)):
            sample = dataset.audio_reference(index)
            sample_id = _sample_id(
                sample.scene_id,
                sample.camera,
                sample.frame_index,
            )
            sample_ids.append(sample_id)
            for baseline in BASELINES:
                predicted = reference_prediction(sample.source_audio, baseline)
                metrics = paper_audio_metrics(
                    predicted,
                    sample.target_audio,
                    dpam_metric=dpam_metric,
                    sample_rate=sample.sample_rate,
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "scene_id": sample.scene_id,
                        "camera": sample.camera,
                        "frame_index": sample.frame_index,
                        "time_seconds": sample.time_seconds,
                        "baseline": baseline,
                        **metrics,
                    }
                )
        manifest = json.loads(config.paths.manifest.read_text())
        target_paths = sorted(
            {record.target_audio_path.resolve() for record in dataset.records}
        )
        inputs.append(
            {
                "scene_id": config.scene.scene_id,
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "manifest": str(config.paths.manifest.resolve()),
                "manifest_sha256": _sha256(config.paths.manifest),
                "source_audio": str(dataset.source_audio_path.resolve()),
                "source_audio_sha256": _sha256(dataset.source_audio_path),
                "target_audio": [
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for path in target_paths
                ],
                "manifest_audio_source_matches": (
                    Path(manifest["audio"]["source_path"]).resolve()
                    == dataset.source_audio_path.resolve()
                ),
                "sample_rate": config.model.sample_rate,
                "crop_seconds": config.train.crop_seconds,
                "sample_count": len(dataset),
                "sample_ids_sha256": hashlib.sha256(
                    canonical_json(sample_ids)
                ).hexdigest(),
            }
        )
    metric_names = (
        METRICS
        if dpam_metric is not None
        else tuple(name for name in METRICS if name != "paper_dpam")
    )
    expected_rows = sum(int(item["sample_count"]) for item in inputs) * len(BASELINES)
    if len(rows) != expected_rows or any(
        set(metric_names) - set(row) for row in rows
    ):
        raise RuntimeError("reference baseline row count/schema mismatch")
    per_scene = _scene_aggregate(rows, metric_names)
    implementation_digest = hashlib.sha256()
    implementation_digest.update(Path(__file__).read_bytes())
    implementation_digest.update(Path(metrics_module.__file__).read_bytes())
    dpam_protocol = (
        getattr(dpam_metric, "protocol", None)
        if dpam_metric is not None
        else None
    )
    report = {
        "schema": SCHEMA,
        "version": 1,
        "baselines": {
            "source_binaural": "unprocessed source-view stereo waveform",
            "mono": "mean of source left/right channels, duplicated to both ears",
        },
        "metric_protocol": {
            "source": "AudioGS arXiv:2604.08967v1, Section IV-B",
            "implementation_sha256": implementation_digest.hexdigest(),
            "paper_mag": {
                "definition": "sum of per-ear mean absolute magnitude-STFT error",
                "n_fft": 512,
                "hop_length": 160,
                "win_length": 400,
                "window": "hamming",
                "center": True,
                "pad_mode": "constant",
            },
            "paper_env": {
                "definition": "sum of per-ear Hilbert-envelope RMSE",
            },
            "paper_lre_db": {
                "definition": "absolute left/right energy-ratio error in dB",
                "epsilon": 1e-5,
            },
            "paper_dpam": {
                "definition": "CDPAM distance after WAV I/O and cdpam.load_audio",
                "status": "computed" if dpam_metric is not None else "skipped",
                "implementation": dpam_protocol,
            },
        },
        "inputs": inputs,
        "row_count": len(rows),
        "per_scene": per_scene,
        "combined": _combined_aggregate(rows, per_scene, metric_names),
    }
    return ReferenceEvaluation(tuple(rows), report)


def write_reference_evaluation(
    evaluation: ReferenceEvaluation,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "metrics_per_sample.jsonl",
        "metrics_per_sample.csv",
        "aggregate.json",
        "verification.json",
    )
    existing = [name for name in names if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"reference output already exists: {output / existing[0]}"
        )
    jsonl = b"".join(canonical_json(row) for row in evaluation.rows)
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=list(evaluation.rows[0]))
    writer.writeheader()
    writer.writerows(evaluation.rows)
    files = {
        "metrics_per_sample.jsonl": jsonl,
        "metrics_per_sample.csv": csv_stream.getvalue().encode(),
        "aggregate.json": canonical_json(evaluation.report),
    }
    for name, data in files.items():
        atomic_write(output / name, data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    atomic_write(
        output / "verification.json",
        canonical_json(
            {
                "schema": f"{SCHEMA}.verification",
                "version": 1,
                "files": hashes,
            }
        ),
    )


def verify_reference_evaluation(output_dir: Path) -> dict[str, object]:
    output = Path(output_dir)
    verification = json.loads((output / "verification.json").read_text())
    expected_files = {
        "metrics_per_sample.jsonl",
        "metrics_per_sample.csv",
        "aggregate.json",
    }
    if (
        not isinstance(verification, dict)
        or set(verification) != {"schema", "version", "files"}
        or verification["schema"] != f"{SCHEMA}.verification"
        or verification["version"] != 1
        or not isinstance(verification["files"], dict)
        or set(verification["files"]) != expected_files
    ):
        raise ValueError("reference verification schema mismatch")
    for name, expected in verification["files"].items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(output / name) != expected
        ):
            raise ValueError(f"reference output hash mismatch: {name}")
    report = json.loads((output / "aggregate.json").read_text())
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("version") != 1
        or set(report.get("baselines", {})) != set(BASELINES)
        or not isinstance(report.get("row_count"), int)
        or report["row_count"] <= 0
    ):
        raise ValueError("reference aggregate schema mismatch")
    jsonl_rows = [
        json.loads(line)
        for line in (output / "metrics_per_sample.jsonl").read_text().splitlines()
        if line
    ]
    with (output / "metrics_per_sample.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if (
        len(jsonl_rows) != report["row_count"]
        or len(csv_rows) != report["row_count"]
        or any(row.get("baseline") not in BASELINES for row in jsonl_rows)
    ):
        raise ValueError("reference output row count/schema mismatch")
    metric_names = {
        name for name in METRICS if any(name in row for row in jsonl_rows)
    }
    dpam_status = report.get("metric_protocol", {}).get("paper_dpam", {}).get("status")
    if (
        set(metric_names)
        != (
            set(METRICS)
            if dpam_status == "computed"
            else set(METRICS) - {"paper_dpam"}
            if dpam_status == "skipped"
            else set()
        )
        or any(set(metric_names) - set(row) for row in jsonl_rows)
    ):
        raise ValueError("reference metric completeness mismatch")
    return report


__all__ = [
    "BASELINES",
    "CDPAMMetric",
    "ReferenceEvaluation",
    "evaluate_reference_baselines",
    "paper_audio_metrics",
    "reference_prediction",
    "verify_reference_evaluation",
    "write_reference_evaluation",
]

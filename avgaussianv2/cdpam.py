"""Minimal paper-compatible CDPAM runtime without benchmark-package imports."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
from pathlib import Path

import torch
from torch import Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        state_owner = getattr(self._model, "model", self._model)
        state_dict = getattr(state_owner, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError("cannot locate CDPAM model state for hashing")
        state_digest = hashlib.sha256()
        for name, value in sorted(state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            state_digest.update(name.encode())
            state_digest.update(str(tensor.dtype).encode())
            state_digest.update(
                (
                    json.dumps(list(tensor.shape), separators=(",", ":")) + "\n"
                ).encode()
            )
            state_digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        module_path = Path(self._cdpam.__file__).resolve()
        class_path_value = inspect.getsourcefile(type(self._model))
        if class_path_value is None:
            raise RuntimeError("cannot locate CDPAM implementation source")
        class_path = Path(class_path_value).resolve()
        return {
            "implementation": (
                f"{type(self._model).__module__}.{type(self._model).__qualname__}"
            ),
            "weight_module": (
                f"{type(state_owner).__module__}.{type(state_owner).__qualname__}"
            ),
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
        try:
            return self.evaluate_files(predicted_path, target_path)
        finally:
            predicted_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)

    def evaluate_files(self, predicted_path: Path, target_path: Path) -> float:
        reference = self._cdpam.load_audio(str(target_path))
        output = self._cdpam.load_audio(str(predicted_path))
        with torch.no_grad():
            value = self._model.forward(reference, output)
        return float(value.detach().cpu().reshape(-1)[0])

    def close(self) -> None:
        self._temporary.cleanup()

    def __enter__(self) -> CDPAMMetric:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def serve_cdpam_worker(
    *,
    input_stream=None,
    output_stream=None,
    metric_factory=CDPAMMetric,
) -> int:
    """Serve line-delimited requests for a persistent external metric process."""
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    with metric_factory() as metric:
        output_stream.write(
            json.dumps(
                {
                    "status": "ready",
                    "protocol": {
                        **metric.protocol,
                        "worker_runtime_source_sha256": _sha256(
                            Path(__file__).resolve()
                        ),
                    },
                    "python_version": sys.version,
                },
                sort_keys=True,
            )
            + "\n"
        )
        output_stream.flush()
        for line in input_stream:
            request: object = None
            try:
                request = json.loads(line)
                if request == {"command": "close"}:
                    return 0
                if not isinstance(request, dict):
                    raise TypeError("DPAM request must be a JSON object")
                request_id = request["request_id"]
                value = metric.evaluate_files(
                    Path(request["predicted_path"]), Path(request["target_path"])
                )
                response = {
                    "status": "ok",
                    "request_id": request_id,
                    "value": value,
                }
            except Exception as error:
                response = {
                    "status": "error",
                    "request_id": (
                        request.get("request_id")
                        if isinstance(request, dict)
                        else None
                    ),
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            output_stream.write(json.dumps(response, sort_keys=True) + "\n")
            output_stream.flush()
    return 0


__all__ = ["CDPAMMetric", "serve_cdpam_worker"]

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor, nn

from avgaussianv2.models.film_unet import FiLMConditionedAudioUNet


class AudioCheckpointError(RuntimeError):
    """Raised when an AudioGS checkpoint cannot be reconstructed safely."""


ModelFactory = Callable[[object], nn.Module]


@contextmanager
def _temporary_import_root(root: Path) -> Iterator[None]:
    root_text = str(root.resolve())
    sys.path.insert(0, root_text)
    try:
        yield
    finally:
        try:
            sys.path.remove(root_text)
        except ValueError:
            pass


def _upstream_model_factory(upstream_root: Path, model_class: str) -> ModelFactory:
    locations = {
        "Audio3DGS": ("libs.models.audio_3dgs", "Audio3DGS"),
        "Audio3DGSMonoDiff": (
            "libs.models.audio_3dgs_mono_diff",
            "Audio3DGSMonoDiff",
        ),
        "Audio3DGSMonoDiffGSOnly": (
            "libs.models.audio_3dgs_mono_diff_gs_only",
            "Audio3DGSMonoDiffGSOnly",
        ),
    }
    if model_class not in locations:
        raise AudioCheckpointError(f"unsupported AudioGS model class {model_class!r}")
    module_name, class_name = locations[model_class]

    def factory(config: object) -> nn.Module:
        with _temporary_import_root(upstream_root):
            module = importlib.import_module(module_name)
            model_type = getattr(module, class_name)
            return model_type(config)

    return factory


class AudioGSBackend(nn.Module):
    def __init__(self, model: nn.Module, source_path: Path) -> None:
        super().__init__()
        if not isinstance(getattr(model, "renderer", None), FiLMConditionedAudioUNet):
            raise TypeError("AudioGS model renderer must be wrapped by FiLMConditionedAudioUNet")
        self.model = model
        self.source_path = Path(source_path)

    @property
    def conditioned_renderer(self) -> FiLMConditionedAudioUNet:
        renderer = self.model.renderer
        if not isinstance(renderer, FiLMConditionedAudioUNet):
            raise RuntimeError("AudioGS renderer wrapper was replaced after backend construction")
        return renderer

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        model_factory: ModelFactory | None = None,
        embedding_dim: int = 128,
        upstream_root: str | Path | None = None,
        model_class: str = "Audio3DGS",
    ) -> "AudioGSBackend":
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"AudioGS checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise AudioCheckpointError("AudioGS checkpoint is missing model_state_dict")
        if model_factory is None:
            if upstream_root is None:
                raise AudioCheckpointError(
                    "upstream_root is required when model_factory is not provided"
                )
            model_factory = _upstream_model_factory(Path(upstream_root), model_class)
        model = model_factory(payload.get("cfg"))
        if not isinstance(model, nn.Module):
            raise AudioCheckpointError("AudioGS model factory must return a torch module")
        if not hasattr(model, "renderer"):
            raise AudioCheckpointError("AudioGS model is missing renderer")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.renderer = FiLMConditionedAudioUNet(model.renderer, embedding_dim=embedding_dim)
        return cls(model, source_path=checkpoint_path)

    def render(
        self,
        cam_pose: Tensor,
        source_audio: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        if source_audio.ndim != 3 or source_audio.shape[1] != 2:
            raise ValueError("source_audio must have shape (B,2,samples)")
        if cam_pose.ndim != 2 or cam_pose.shape[0] != source_audio.shape[0]:
            raise ValueError("cam_pose must have shape (B,features) and match source audio")
        if condition is None:
            return self.model(cam_pose, source_audio)
        with self.conditioned_renderer.use_condition(condition):
            return self.model(cam_pose, source_audio)

    def acoustic_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("renderer.")
        ]

    def film_parameters(self) -> list[nn.Parameter]:
        return list(self.conditioned_renderer.film.parameters())

    def audio_unet_parameters(self) -> list[nn.Parameter]:
        return list(self.conditioned_renderer.base.parameters())

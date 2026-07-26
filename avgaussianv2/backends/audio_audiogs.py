from __future__ import annotations

import importlib
import sys
from types import ModuleType
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor, nn

from avgaussianv2.models.film_unet import FiLMConditionedAudioUNet


class AudioCheckpointError(RuntimeError):
    """Raised when an AudioGS checkpoint cannot be reconstructed safely."""


ModelFactory = Callable[[object], nn.Module]
ForwardOverride = Callable[[nn.Module, Tensor, Tensor], Tensor]


def _deterministic_stft_magnitude(
    waveform: Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: Tensor,
) -> Tensor:
    """Match centered reflect STFT without CUDA reflection-pad backward."""
    waveform = torch.nan_to_num(
        waveform, nan=0.0, posinf=0.0, neginf=0.0
    )
    padding = int(fft_size) // 2
    if waveform.shape[-1] <= padding:
        raise ValueError("waveform is too short for reflected STFT padding")
    reflected = torch.cat(
        (
            waveform[..., 1 : padding + 1].flip(-1),
            waveform,
            waveform[..., -padding - 1 : -1].flip(-1),
        ),
        dim=-1,
    )
    spectrum = torch.stft(
        reflected,
        n_fft=int(fft_size),
        hop_length=int(hop_size),
        win_length=int(win_length),
        window=window.to(waveform.device),
        center=False,
        return_complex=True,
    )
    return torch.sqrt(torch.clamp(spectrum.abs().square(), min=1e-7))


def _install_lightweight_scene_package(upstream_root: Path) -> None:
    """Avoid importing AudioGS dataset readers when only model math is needed."""
    module_name = "libs.datasets.scene"
    if module_name in sys.modules:
        return
    scene_path = upstream_root / "libs" / "datasets" / "scene"
    if not scene_path.is_dir():
        return
    module = ModuleType(module_name)
    module.__package__ = module_name
    module.__path__ = [str(scene_path)]
    sys.modules[module_name] = module


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
            _install_lightweight_scene_package(upstream_root)
            module = importlib.import_module(module_name)
            build_model = getattr(module, "build_model", None)
            if callable(build_model):
                return build_model(config)
            model_type = getattr(module, class_name)
            return model_type(config)

    return factory


def _upstream_forward_override(
    upstream_root: Path,
    model_class: str,
) -> ForwardOverride | None:
    if model_class != "Audio3DGSMonoDiffGSOnly":
        return None
    with _temporary_import_root(upstream_root):
        module = importlib.import_module("libs.models.audio_3dgs_mono_diff")
        parent_type = getattr(module, "Audio3DGSMonoDiff")
    return parent_type.forward


class AudioGSBackend(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        source_path: Path,
        checkpoint_config: object | None = None,
        upstream_root: Path | None = None,
        forward_override: ForwardOverride | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(getattr(model, "renderer", None), FiLMConditionedAudioUNet):
            raise TypeError("AudioGS model renderer must be wrapped by FiLMConditionedAudioUNet")
        self.model = model
        self.source_path = Path(source_path)
        self.checkpoint_config = checkpoint_config
        self.upstream_root = None if upstream_root is None else Path(upstream_root)
        self.forward_override = forward_override

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
        checkpoint_config = payload.get("cfg")
        model = model_factory(checkpoint_config)
        if not isinstance(model, nn.Module):
            raise AudioCheckpointError("AudioGS model factory must return a torch module")
        if not hasattr(model, "renderer"):
            raise AudioCheckpointError("AudioGS model is missing renderer")
        checkpoint_state = dict(payload["model_state_dict"])
        initialized_state = model.state_dict()
        for cache_name in (
            "static_source_mag",
            "static_phase_L",
            "static_phase_R",
        ):
            if cache_name in checkpoint_state and cache_name in initialized_state:
                checkpoint_state[cache_name] = initialized_state[cache_name]
        model.load_state_dict(checkpoint_state, strict=True)
        model.renderer = FiLMConditionedAudioUNet(model.renderer, embedding_dim=embedding_dim)
        forward_override = None
        if upstream_root is not None:
            forward_override = _upstream_forward_override(Path(upstream_root), model_class)
        return cls(
            model,
            source_path=checkpoint_path,
            checkpoint_config=checkpoint_config,
            upstream_root=None if upstream_root is None else Path(upstream_root),
            forward_override=forward_override,
        )

    def build_criterion(self) -> nn.Module:
        """Recreate the loss selected by Audio3DGSTrainer for this checkpoint."""
        if self.checkpoint_config is None:
            raise AudioCheckpointError("AudioGS checkpoint is missing cfg for criterion creation")
        if self.upstream_root is None:
            raise AudioCheckpointError("AudioGS upstream_root is required for criterion creation")
        train_config = getattr(self.checkpoint_config, "train", object())
        enhanced_weight = float(getattr(train_config, "enhanced_weight", 0.0) or 0.0)
        if enhanced_weight > 0:
            raise AudioCheckpointError(
                "AudioGS checkpoints with train.enhanced_weight > 0 are not supported"
            )
        model_config = getattr(self.checkpoint_config, "model", object())
        model_file = str(getattr(model_config, "file", "") or "")
        mono_diff_models = {
            "audio_3dgs_mono_diff",
            "audio_3dgs_mono_diff_gs_only",
            "audio_3dgs_mono_diff_field",
            "audio_3dgs_shared_gaussians_gs_only",
        }
        mono_only_models = {"audio_3dgs_mono_only", "audio_3dgs_mono_gs_only"}
        if model_file in mono_diff_models:
            module_name, class_name = (
                "libs.criterions.MonoDiffMSECriterion",
                "MonoDiffMSECriterion",
            )
        elif model_file in mono_only_models:
            module_name, class_name = (
                "libs.criterions.MonoOnlyMSECriterion",
                "MonoOnlyMSECriterion",
            )
        else:
            module_name, class_name = "libs.criterions.Criterion_2", "Criterion"
        with _temporary_import_root(self.upstream_root):
            module = importlib.import_module(module_name)
            module.stft = _deterministic_stft_magnitude
            criterion_type = getattr(module, class_name)
            criterion = criterion_type(self.checkpoint_config)
        if not isinstance(criterion, nn.Module):
            raise AudioCheckpointError("AudioGS criterion must be a torch module")
        return criterion

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
        if self.forward_override is None:
            if condition is None:
                return self.model(cam_pose, source_audio)
            with self.conditioned_renderer.use_condition(condition):
                return self.model(cam_pose, source_audio)

        native = self.model(cam_pose, source_audio)
        if condition is None:
            return native
        # GS-only checkpoints bypass their saved renderer. Use the inherited
        # U-Net only as a conditional residual so zero-init FiLM preserves the
        # native pretrained function exactly.
        plain_unet = self.forward_override(self.model, cam_pose, source_audio)
        with self.conditioned_renderer.use_condition(condition):
            conditioned_unet = self.forward_override(self.model, cam_pose, source_audio)
        return native + (conditioned_unet - plain_unet)

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

from __future__ import annotations

import importlib
import sys
from enum import Enum
from types import ModuleType
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor, nn

from avgaussianv2.models.film_unet import FiLMConditionedAudioUNet
from avgaussianv2.models.mask_cross_attention import (
    AudioFeatureMaskCrossAttention,
)
from avgaussianv2.models.p1_audio import AlignedComplexCrossAttention
from avgaussianv2.models.p1_visual import VisualMemory


class AudioCheckpointError(RuntimeError):
    """Raised when an AudioGS checkpoint cannot be reconstructed safely."""


ModelFactory = Callable[[object], nn.Module]
ForwardOverride = Callable[[nn.Module, Tensor, Tensor], Tensor]


class AudioRenderStrategy(str, Enum):
    """How RGBD-conditioned U-Net output is combined with AudioGS."""

    NATIVE_RESIDUAL = "native_residual"
    PLAIN_UNET = "plain_unet"
    DIRECT_CONDITIONED_UNET = "direct_conditioned_unet"
    GATED_NATIVE_RESIDUAL = "gated_native_residual"


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


def _load_audiogs_model(
    checkpoint: str | Path,
    *,
    model_factory: ModelFactory | None,
    upstream_root: str | Path | None,
    model_class: str,
) -> tuple[nn.Module, object | None, Path]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"AudioGS checkpoint does not exist: {checkpoint_path}"
        )
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
    return model, checkpoint_config, checkpoint_path


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


def build_audiogs_criterion(
    checkpoint_config: object | None,
    upstream_root: Path | None,
) -> nn.Module:
    """Recreate the criterion selected by the upstream AudioGS checkpoint."""
    if checkpoint_config is None:
        raise AudioCheckpointError("AudioGS checkpoint is missing cfg for criterion creation")
    if upstream_root is None:
        raise AudioCheckpointError("AudioGS upstream_root is required for criterion creation")
    train_config = getattr(checkpoint_config, "train", object())
    enhanced_weight = float(getattr(train_config, "enhanced_weight", 0.0) or 0.0)
    if enhanced_weight > 0:
        raise AudioCheckpointError(
            "AudioGS checkpoints with train.enhanced_weight > 0 are not supported"
        )
    model_config = getattr(checkpoint_config, "model", object())
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
    with _temporary_import_root(upstream_root):
        module = importlib.import_module(module_name)
        module.stft = _deterministic_stft_magnitude
        criterion_type = getattr(module, class_name)
        criterion = criterion_type(checkpoint_config)
    if not isinstance(criterion, nn.Module):
        raise AudioCheckpointError("AudioGS criterion must be a torch module")
    return criterion


class AudioGSBackend(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        source_path: Path,
        checkpoint_config: object | None = None,
        upstream_root: Path | None = None,
        forward_override: ForwardOverride | None = None,
        render_strategy: AudioRenderStrategy | str = AudioRenderStrategy.NATIVE_RESIDUAL,
        complex_renderer: AlignedComplexCrossAttention | None = None,
    ) -> None:
        super().__init__()
        renderer = getattr(model, "renderer", None)
        required_renderer_methods = (
            "use_condition",
            "conditioning_parameters",
            "base_parameters",
        )
        if complex_renderer is None and (
            not isinstance(renderer, nn.Module)
            or any(
                not callable(getattr(renderer, name, None))
                for name in required_renderer_methods
            )
        ):
            raise TypeError(
                "AudioGS model renderer must implement the conditioned renderer protocol"
            )
        self.model = model
        self.complex_renderer = complex_renderer
        self.source_path = Path(source_path)
        self.checkpoint_config = checkpoint_config
        self.upstream_root = None if upstream_root is None else Path(upstream_root)
        self.forward_override = forward_override
        self.render_strategy = AudioRenderStrategy(render_strategy)
        self.residual_gate_logit = (
            nn.Parameter(torch.zeros(()))
            if self.render_strategy is AudioRenderStrategy.GATED_NATIVE_RESIDUAL
            else None
        )

    @property
    def conditioned_renderer(self) -> nn.Module:
        renderer = self.model.renderer
        if not isinstance(renderer, nn.Module) or not callable(
            getattr(renderer, "use_condition", None)
        ):
            raise RuntimeError(
                "AudioGS conditioned renderer was replaced after backend construction"
            )
        return renderer

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        model_factory: ModelFactory | None = None,
        embedding_dim: int = 128,
        upstream_root: str | Path | None = None,
        model_class: str = "Audio3DGS",
        render_strategy: AudioRenderStrategy | str = AudioRenderStrategy.NATIVE_RESIDUAL,
        renderer_kind: str = "film_unet",
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        freq_patch: int = 16,
        time_patch: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        max_log_magnitude: float = 0.15,
        max_phase: float = 0.25,
        additive_scale: float = 0.01,
        geometry_rank: int = 16,
        geometry_bias_scale: float = 1.0,
    ) -> "AudioGSBackend":
        model, checkpoint_config, checkpoint_path = _load_audiogs_model(
            checkpoint,
            model_factory=model_factory,
            upstream_root=upstream_root,
            model_class=model_class,
        )
        complex_renderer = None
        if renderer_kind == "film_unet":
            model.renderer = FiLMConditionedAudioUNet(
                model.renderer,
                embedding_dim=embedding_dim,
            )
        elif renderer_kind == "mask_cross_attention":
            model.renderer = AudioFeatureMaskCrossAttention(
                d_model=embedding_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                freq_patch=freq_patch,
                time_patch=time_patch,
                dropout=dropout,
                cross_gate_init=cross_gate_init,
            )
        elif renderer_kind == "p1_query_geometry":
            if model_class != "Audio3DGSMonoDiffGSOnly":
                raise AudioCheckpointError(
                    "query-dependent P1 requires Audio3DGSMonoDiffGSOnly"
                )
            complex_renderer = AlignedComplexCrossAttention(
                d_model=embedding_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                freq_patch=freq_patch,
                time_patch=time_patch,
                dropout=dropout,
                cross_gate_init=cross_gate_init,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                max_log_magnitude=max_log_magnitude,
                max_phase=max_phase,
                additive_scale=additive_scale,
                geometry_rank=geometry_rank,
                geometry_bias_scale=geometry_bias_scale,
            )
        else:
            raise ValueError(
                "renderer_kind must be film_unet, mask_cross_attention, "
                "or p1_query_geometry"
            )
        forward_override = None
        if upstream_root is not None:
            forward_override = _upstream_forward_override(Path(upstream_root), model_class)
        return cls(
            model,
            source_path=checkpoint_path,
            checkpoint_config=checkpoint_config,
            upstream_root=None if upstream_root is None else Path(upstream_root),
            forward_override=forward_override,
            render_strategy=render_strategy,
            complex_renderer=complex_renderer,
        )

    def build_criterion(self) -> nn.Module:
        return build_audiogs_criterion(self.checkpoint_config, self.upstream_root)

    def render(
        self,
        cam_pose: Tensor,
        source_audio: Tensor,
        condition: Tensor | VisualMemory | None = None,
    ) -> Tensor:
        if source_audio.ndim != 3 or source_audio.shape[1] != 2:
            raise ValueError("source_audio must have shape (B,2,samples)")
        if cam_pose.ndim != 2 or cam_pose.shape[0] != source_audio.shape[0]:
            raise ValueError("cam_pose must have shape (B,features) and match source audio")
        if self.complex_renderer is not None:
            if condition is None:
                return self.model(cam_pose, source_audio)
            native_outputs = self.model(
                cam_pose,
                source_audio,
                return_masks=True,
            )
            if not isinstance(native_outputs, tuple) or len(native_outputs) != 5:
                raise RuntimeError(
                    "query-dependent P1 requires AudioGS return_masks output "
                    "(audio, mono, diff, source_magnitude, distance)"
                )
            native, mono, diff, source_magnitude, distance = native_outputs
            return self.complex_renderer(
                native,
                mono,
                diff,
                source_magnitude,
                distance,
                cam_pose,
                condition,
            )
        if self.forward_override is None:
            plain = self.model(cam_pose, source_audio)
            if condition is None:
                return plain
            with self.conditioned_renderer.use_condition(condition):
                conditioned = self.model(cam_pose, source_audio)
            if self.render_strategy is AudioRenderStrategy.GATED_NATIVE_RESIDUAL:
                return plain + self.residual_gate_scale() * (conditioned - plain)
            return conditioned

        if self.render_strategy is AudioRenderStrategy.PLAIN_UNET:
            return self.forward_override(self.model, cam_pose, source_audio)

        if self.render_strategy is AudioRenderStrategy.DIRECT_CONDITIONED_UNET:
            if condition is None:
                return self.forward_override(self.model, cam_pose, source_audio)
            with self.conditioned_renderer.use_condition(condition):
                return self.forward_override(self.model, cam_pose, source_audio)

        native = self.model(cam_pose, source_audio)
        if condition is None:
            return native
        # GS-only checkpoints bypass their saved renderer. For residual
        # strategies, isolate only the visual-condition delta so zero-init
        # FiLM preserves the native pretrained function exactly.
        plain_unet = self.forward_override(self.model, cam_pose, source_audio)
        with self.conditioned_renderer.use_condition(condition):
            conditioned_unet = self.forward_override(self.model, cam_pose, source_audio)
        residual = conditioned_unet - plain_unet
        if self.render_strategy is AudioRenderStrategy.GATED_NATIVE_RESIDUAL:
            residual = self.residual_gate_scale() * residual
        return native + residual

    def residual_gate_scale(self) -> Tensor:
        """Return a bounded (0, 2) gate initialized exactly to one."""
        if self.residual_gate_logit is None:
            raise RuntimeError("audio render strategy has no learnable residual gate")
        return 2.0 * torch.sigmoid(self.residual_gate_logit)

    def acoustic_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("renderer.")
        ]

    def film_parameters(self) -> list[nn.Parameter]:
        if self.complex_renderer is not None:
            return list(self.complex_renderer.parameters())
        parameters = list(self.conditioned_renderer.conditioning_parameters())
        if self.residual_gate_logit is not None:
            parameters.append(self.residual_gate_logit)
        return parameters

    def audio_unet_parameters(self) -> list[nn.Parameter]:
        if self.complex_renderer is not None:
            return []
        return list(self.conditioned_renderer.base_parameters())

    def audio_only_parameter_groups(self) -> tuple[str, ...]:
        """Return groups consumed by the actual ``condition=None`` path."""
        if self.complex_renderer is not None:
            return ("acoustic",)
        if self.forward_override is None:
            return ("acoustic", "audio_unet")
        if self.render_strategy in {
            AudioRenderStrategy.PLAIN_UNET,
            AudioRenderStrategy.DIRECT_CONDITIONED_UNET,
        }:
            return ("acoustic", "audio_unet")
        return ("acoustic",)

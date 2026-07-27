from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn

from avgaussianv2.backends.audio_audiogs import (
    AudioCheckpointError,
    ModelFactory,
    _upstream_model_factory,
    build_audiogs_criterion,
)
from avgaussianv2.models.acoustic_gaussian_tokens import (
    AudioGSGaussianAttributeAdapter,
    GaussianTokenEncoder,
    PoseTokenEncoder,
)
from avgaussianv2.models.audio_tokens import AudioSpectrogramHead, AudioSTFTTokenizer


def _validate_token_tensor(name: str, value: Tensor, d_model: int) -> None:
    if value.ndim != 3 or value.shape[-1] != d_model:
        raise ValueError(f"{name} must have shape (B,N,{d_model})")


def _parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [parameter for module in modules for parameter in module.parameters()]


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        if not 0 <= cross_gate_init <= 1:
            raise ValueError("cross_gate_init must be in [0,1]")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        hidden_dim = self.d_model * int(ffn_multiplier)

        self.self_norm = nn.LayerNorm(self.d_model)
        self.self_attention = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.d_model)
        self.memory_norm = nn.LayerNorm(self.d_model)
        self.cross_attention = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.d_model),
        )

        self.self_gate = nn.Parameter(torch.zeros(self.d_model))
        self.cross_gate = nn.Parameter(
            torch.full((self.d_model,), float(cross_gate_init))
        )
        self.ffn_gate = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, audio_tokens: Tensor, memory_tokens: Tensor | None = None) -> Tensor:
        _validate_token_tensor("audio_tokens", audio_tokens, self.d_model)
        tokens = audio_tokens
        self_tokens = self.self_norm(tokens)
        self_update, _ = self.self_attention(
            self_tokens,
            self_tokens,
            self_tokens,
            need_weights=False,
        )
        tokens = tokens + self.self_gate.view(1, 1, -1) * self_update

        if memory_tokens is not None:
            _validate_token_tensor("memory_tokens", memory_tokens, self.d_model)
            if memory_tokens.shape[0] != tokens.shape[0]:
                raise ValueError("memory_tokens must match audio_tokens batch size")
            if memory_tokens.shape[1] > 0:
                cross_update, _ = self.cross_attention(
                    self.cross_norm(tokens),
                    self.memory_norm(memory_tokens),
                    self.memory_norm(memory_tokens),
                    need_weights=False,
                )
                tokens = tokens + self.cross_gate.view(1, 1, -1) * cross_update

        ffn_update = self.ffn(self.ffn_norm(tokens))
        return tokens + self.ffn_gate.view(1, 1, -1) * ffn_update

    def audio_parameters(self) -> list[nn.Parameter]:
        return [
            self.self_gate,
            self.ffn_gate,
            *_parameters([self.self_norm, self.self_attention, self.ffn_norm, self.ffn]),
        ]

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return [
            self.cross_gate,
            *_parameters([self.cross_norm, self.memory_norm, self.cross_attention]),
        ]


class AudioVisualTokenTransformer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.d_model = int(d_model)
        self.blocks = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    cross_gate_init=cross_gate_init,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, audio_tokens: Tensor, memory_tokens: Tensor | None = None) -> Tensor:
        _validate_token_tensor("audio_tokens", audio_tokens, self.d_model)
        tokens = audio_tokens
        for block in self.blocks:
            tokens = block(tokens, memory_tokens)
        return tokens

    def audio_parameters(self) -> list[nn.Parameter]:
        return [parameter for block in self.blocks for parameter in block.audio_parameters()]

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return [parameter for block in self.blocks for parameter in block.conditioning_parameters()]


class AudioVisualTokenAudioBackend(nn.Module):
    """Primitive-aware RGBD/pose/AudioGS cross-attention residual.

    Source-audio TF tokens query visual, pose and explicit acoustic-Gaussian
    tokens. The predicted complex-spectrogram residual is anchored on the
    native AudioGS Gaussian render. The upstream U-Net is never used.
    """

    def __init__(
        self,
        model: nn.Module,
        source_path: Path,
        *,
        checkpoint_config: object | None = None,
        upstream_root: Path | None = None,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        freq_patch: int = 16,
        time_patch: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
        residual_scale: float = 0.05,
        gaussian_token_rows: int = 16,
        gaussian_token_columns: int = 16,
        gaussian_token_hidden_dim: int = 32,
        pose_tokens: int = 2,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(model, nn.Module):
            raise TypeError("AudioGS model must be a torch module")
        if not hasattr(model, "renderer"):
            raise TypeError("AudioGS model is missing renderer")
        # The GS-only forward never calls renderer. Removing it makes accidental
        # U-Net use impossible and avoids carrying unrelated trainable weights.
        model.renderer = nn.Identity()
        self.model = model
        self.source_path = Path(source_path)
        self.checkpoint_config = checkpoint_config
        self.upstream_root = None if upstream_root is None else Path(upstream_root)
        self.d_model = int(d_model)
        self.gaussian_adapter = AudioGSGaussianAttributeAdapter(self.model)
        self.gaussian_encoder = GaussianTokenEncoder(
            self.gaussian_adapter.feature_dim,
            d_model=d_model,
            hidden_dim=gaussian_token_hidden_dim,
            token_grid=(gaussian_token_rows, gaussian_token_columns),
        )
        self.pose_encoder = PoseTokenEncoder(
            d_model=d_model,
            num_tokens=pose_tokens,
        )
        # visual, pose, acoustic Gaussian
        self.memory_modality_embedding = nn.Parameter(torch.zeros(3, self.d_model))
        self.gaussian_tokens_enabled = True
        self.pose_tokens_enabled = True
        self.tokenizer = AudioSTFTTokenizer(
            d_model=d_model,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            freq_patch=freq_patch,
            time_patch=time_patch,
        )
        self.transformer = AudioVisualTokenTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            cross_gate_init=cross_gate_init,
        )
        self.head = AudioSpectrogramHead(
            d_model=d_model,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            freq_patch=freq_patch,
            time_patch=time_patch,
            residual_scale=residual_scale,
        )
        nn.init.zeros_(self.head.projection.bias)

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        model_factory: ModelFactory | None = None,
        upstream_root: str | Path | None = None,
        model_class: str = "Audio3DGSMonoDiffGSOnly",
        **kwargs,
    ) -> "AudioVisualTokenAudioBackend":
        if model_class != "Audio3DGSMonoDiffGSOnly":
            raise AudioCheckpointError(
                "cross-attention backend requires Audio3DGSMonoDiffGSOnly "
                "so its base prediction is guaranteed to come from AudioGS gaussians"
            )
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
        return cls(
            model,
            checkpoint_path,
            checkpoint_config=checkpoint_config,
            upstream_root=None if upstream_root is None else Path(upstream_root),
            **kwargs,
        )

    def _condition_tokens(self, condition: Tensor, batch_size: int) -> Tensor:
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        _validate_token_tensor("condition", condition, self.d_model)
        if condition.shape[0] != batch_size:
            raise ValueError("condition must match source_audio batch size")
        return condition

    def _memory_tokens(
        self,
        cam_pose: Tensor,
        condition: Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        batch_size = int(cam_pose.shape[0])
        memory: list[Tensor] = []
        if condition is not None:
            visual = self._condition_tokens(condition, batch_size).to(
                device=device, dtype=dtype
            )
            memory.append(visual + self.memory_modality_embedding[0])
        if self.pose_tokens_enabled:
            pose = self.pose_encoder(cam_pose).to(device=device, dtype=dtype)
            memory.append(pose + self.memory_modality_embedding[1])
        if self.gaussian_tokens_enabled:
            gaussian_batch = self.gaussian_adapter(cam_pose)
            gaussian = self.gaussian_encoder(gaussian_batch).to(
                device=device, dtype=dtype
            )
            memory.append(gaussian + self.memory_modality_embedding[2])
        if not memory:
            raise RuntimeError("cross-attention requires at least one memory modality")
        return torch.cat(memory, dim=1)

    def render(
        self,
        cam_pose: Tensor,
        source_audio: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        if source_audio.ndim != 3 or source_audio.shape[1] != 2:
            raise ValueError("source_audio must have shape (B,2,samples)")
        if cam_pose.ndim != 2 or cam_pose.shape[0] != source_audio.shape[0]:
            raise ValueError(
                "cam_pose must have shape (B,features) and match source audio"
            )
        native = self.model(cam_pose, source_audio)
        # Queries come from the source audio; the native Gaussian render is only
        # the shared residual anchor used for a fair comparison with FiLM+U-Net.
        batch = self.tokenizer(source_audio)
        memory_tokens = self._memory_tokens(
            cam_pose,
            condition,
            dtype=batch.tokens.dtype,
            device=batch.tokens.device,
        )
        conditioned_tokens = self.transformer(batch.tokens, memory_tokens)
        condition_delta = conditioned_tokens - batch.tokens
        native_stft = self.tokenizer.stft(native)
        return self.head(
            condition_delta,
            batch.grid_size,
            native_stft,
            length=batch.original_samples,
        )

    def forward(
        self,
        cam_pose: Tensor,
        source_audio: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        return self.render(cam_pose, source_audio, condition=condition)

    def build_criterion(self) -> nn.Module:
        return build_audiogs_criterion(
            self.checkpoint_config,
            self.upstream_root,
        )

    def acoustic_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("renderer.")
        ]

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return [
            self.memory_modality_embedding,
            *_parameters(
                [
                    self.tokenizer,
                    self.head,
                    self.gaussian_encoder,
                    self.pose_encoder,
                ]
            ),
            *self.transformer.audio_parameters(),
            *self.transformer.conditioning_parameters(),
        ]

    def film_parameters(self) -> list[nn.Parameter]:
        return self.conditioning_parameters()

    def audio_unet_parameters(self) -> list[nn.Parameter]:
        return []

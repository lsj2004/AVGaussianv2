from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.models.positional import add_grid_position_encoding


def _validate_features(name: str, value: Tensor, channels: int) -> None:
    if value.ndim != 4 or value.shape[1] != channels:
        raise ValueError(
            f"{name} must have shape (B,{channels},F,T), got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


class _AudioVisualCrossBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ffn_multiplier: int,
        dropout: float,
        cross_gate_init: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model) * int(ffn_multiplier)
        self.audio_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.self_gate = nn.Parameter(torch.zeros(d_model))
        self.cross_gate = nn.Parameter(
            torch.full((d_model,), float(cross_gate_init))
        )
        self.ffn_gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, audio: Tensor, visual: Tensor | None) -> Tensor:
        normalized_audio = self.audio_norm(audio)
        self_update, _ = self.self_attention(
            normalized_audio,
            normalized_audio,
            normalized_audio,
            need_weights=False,
        )
        audio = audio + self.self_gate.view(1, 1, -1) * self_update
        if visual is not None:
            memory = self.memory_norm(visual)
            cross_update, _ = self.cross_attention(
                self.query_norm(audio),
                memory,
                memory,
                need_weights=False,
            )
            audio = audio + self.cross_gate.view(1, 1, -1) * cross_update
        return audio + self.ffn_gate.view(1, 1, -1) * self.ffn(
            self.ffn_norm(audio)
        )


class AudioFeatureMaskCrossAttention(nn.Module):
    """AudioGS-compatible mask renderer conditioned by RGBD tokens."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        freq_patch: int = 16,
        time_patch: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        positive = {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "freq_patch": freq_patch,
            "time_patch": time_patch,
            "ffn_multiplier": ffn_multiplier,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        if not 0 <= cross_gate_init <= 1:
            raise ValueError("cross_gate_init must be in [0,1]")

        self.d_model = int(d_model)
        self.freq_patch = int(freq_patch)
        self.time_patch = int(time_patch)
        patch = (self.freq_patch, self.time_patch)
        self.mono_patch_embedding = nn.Conv2d(
            3, self.d_model, kernel_size=patch, stride=patch
        )
        self.diff_patch_embedding = nn.Conv2d(
            2, self.d_model, kernel_size=patch, stride=patch
        )
        self.modality_fusion = nn.Linear(2 * self.d_model, self.d_model)
        self.blocks = nn.ModuleList(
            [
                _AudioVisualCrossBlock(
                    self.d_model,
                    int(num_heads),
                    ffn_multiplier=int(ffn_multiplier),
                    dropout=float(dropout),
                    cross_gate_init=float(cross_gate_init),
                )
                for _ in range(int(num_layers))
            ]
        )
        patch_area = self.freq_patch * self.time_patch
        self.output_norm = nn.LayerNorm(self.d_model)
        self.mono_head = nn.Linear(self.d_model, patch_area)
        self.diff_head = nn.Linear(self.d_model, patch_area)
        self._active_condition: Tensor | None = None

    @property
    def active_condition(self) -> Tensor | None:
        return self._active_condition

    @contextmanager
    def use_condition(self, condition: Tensor) -> Iterator[None]:
        if self._active_condition is not None:
            raise RuntimeError("nested AudioGS condition contexts are not supported")
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if condition.ndim != 3 or condition.shape[-1] != self.d_model:
            raise ValueError(
                "condition must have shape "
                f"(B,N,{self.d_model}) or (B,{self.d_model}), "
                f"got {tuple(condition.shape)}"
            )
        self._active_condition = condition
        try:
            yield
        finally:
            self._active_condition = None

    def _pad(self, value: Tensor) -> tuple[Tensor, tuple[int, int]]:
        height, width = (int(value.shape[-2]), int(value.shape[-1]))
        pad_height = (-height) % self.freq_patch
        pad_width = (-width) % self.time_patch
        return F.pad(value, (0, pad_width, 0, pad_height)), (height, width)

    def _audio_tokens(
        self,
        mono_features: Tensor,
        diff_features: Tensor,
    ) -> tuple[Tensor, tuple[int, int], tuple[int, int]]:
        mono, original_size = self._pad(mono_features)
        diff, diff_original_size = self._pad(diff_features)
        if diff_original_size != original_size:
            raise ValueError("mono_features and diff_features must share F,T dimensions")
        mono_tokens = self.mono_patch_embedding(mono)
        diff_tokens = self.diff_patch_embedding(diff)
        if mono_tokens.shape[-2:] != diff_tokens.shape[-2:]:
            raise RuntimeError("mono and difference patch grids do not match")
        grid_size = (int(mono_tokens.shape[-2]), int(mono_tokens.shape[-1]))
        mono_tokens = mono_tokens.flatten(2).transpose(1, 2)
        diff_tokens = diff_tokens.flatten(2).transpose(1, 2)
        tokens = self.modality_fusion(torch.cat([mono_tokens, diff_tokens], dim=-1))
        return add_grid_position_encoding(tokens, grid_size), grid_size, original_size

    def _unpatchify(
        self,
        patches: Tensor,
        grid_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> Tensor:
        batch = int(patches.shape[0])
        grid_height, grid_width = grid_size
        value = patches.reshape(
            batch,
            grid_height,
            grid_width,
            self.freq_patch,
            self.time_patch,
        )
        value = value.permute(0, 1, 3, 2, 4).reshape(
            batch,
            1,
            grid_height * self.freq_patch,
            grid_width * self.time_patch,
        )
        height, width = original_size
        return value[..., :height, :width]

    def forward(
        self,
        mono_features: Tensor,
        diff_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _validate_features("mono_features", mono_features, 3)
        _validate_features("diff_features", diff_features, 2)
        if mono_features.shape[0] != diff_features.shape[0]:
            raise ValueError("mono_features and diff_features batches must match")

        tokens, grid_size, original_size = self._audio_tokens(
            mono_features,
            diff_features,
        )
        condition = self._active_condition
        if condition is not None:
            if condition.shape[0] != tokens.shape[0]:
                raise ValueError("condition and AudioGS feature batches must match")
            condition = condition.to(device=tokens.device, dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, condition)
        tokens = self.output_norm(tokens)
        mono_logits = self._unpatchify(
            self.mono_head(tokens), grid_size, original_size
        )
        diff_logits = self._unpatchify(
            self.diff_head(tokens), grid_size, original_size
        )
        return F.softplus(mono_logits) + 0.1, torch.tanh(diff_logits)

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def base_parameters(self) -> list[nn.Parameter]:
        return []

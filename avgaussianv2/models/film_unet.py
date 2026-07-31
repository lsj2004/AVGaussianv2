from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FiLM(nn.Module):
    def __init__(self, embedding_dim: int, channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(embedding_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=-1)
        return value * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


def _block_output_channels(block: nn.Module, name: str) -> int:
    modules = list(block.modules())
    for module in reversed(modules):
        if isinstance(module, nn.Conv2d):
            return int(module.out_channels)
    raise TypeError(f"AudioGS renderer block {name} has no Conv2d output")


class FiLMConditionedAudioUNet(nn.Module):
    _BASE_BLOCKS = {
        "e1": "enc1",
        "e2": "enc2",
        "e3": "enc3",
        "e4": "enc4",
        "d4": "dec4",
        "d3": "dec3",
        "d2": "dec2",
        "d1": "dec1",
    }

    def __init__(self, base: nn.Module, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.base = base
        self.embedding_dim = int(embedding_dim)
        required = (
            "enc1",
            "diff_enc1",
            "enc2",
            "enc3",
            "enc4",
            "maxpool",
            "upconv4",
            "dec4",
            "upconv3",
            "dec3",
            "upconv2",
            "dec2",
            "upconv1",
            "dec1",
            "out_mono",
            "out_diff",
        )
        missing = [name for name in required if not hasattr(base, name)]
        if missing:
            raise TypeError(f"AudioGS renderer is missing {missing[0]}")
        self.film = nn.ModuleDict(
            {
                name: FiLM(
                    self.embedding_dim,
                    _block_output_channels(getattr(base, block_name), block_name),
                )
                for name, block_name in self._BASE_BLOCKS.items()
            }
        )
        self._active_condition: Tensor | None = None

    @property
    def active_condition(self) -> Tensor | None:
        return self._active_condition

    @contextmanager
    def use_condition(self, condition: Tensor) -> Iterator[None]:
        if self._active_condition is not None:
            raise RuntimeError("nested AudioGS condition contexts are not supported")
        if condition.ndim != 2 or condition.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"condition must have shape (B,{self.embedding_dim}), got {tuple(condition.shape)}"
            )
        self._active_condition = condition
        try:
            yield
        finally:
            self._active_condition = None

    @staticmethod
    def _resize_like(value: Tensor, reference: Tensor) -> Tensor:
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        return F.interpolate(
            value,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _modulate(self, name: str, value: Tensor, condition: Tensor) -> Tensor:
        return self.film[name](value, condition)

    def forward(self, mono_features: Tensor, diff_features: Tensor) -> tuple[Tensor, Tensor]:
        condition = self._active_condition
        if condition is None:
            return self.base(mono_features, diff_features)
        if condition.shape[0] != mono_features.shape[0]:
            raise ValueError("condition and AudioGS feature batches must match")

        e1_mono = self._modulate("e1", self.base.enc1(mono_features), condition)
        e1_pool_mono = self.base.maxpool(e1_mono)
        e1_diff = self.base.diff_enc1(diff_features)
        e1_pool_diff = self.base.maxpool(e1_diff)
        e1_pool = 0.5 * (e1_pool_mono + e1_pool_diff)

        e2 = self._modulate("e2", self.base.enc2(e1_pool), condition)
        e3 = self._modulate("e3", self.base.enc3(self.base.maxpool(e2)), condition)
        e4 = self._modulate("e4", self.base.enc4(self.base.maxpool(e3)), condition)

        d4_up = self._resize_like(self.base.upconv4(e4), e3)
        d4 = self._modulate("d4", self.base.dec4(torch.cat([d4_up, e3], dim=1)), condition)
        d3_up = self._resize_like(self.base.upconv3(d4), e2)
        d3 = self._modulate("d3", self.base.dec3(torch.cat([d3_up, e2], dim=1)), condition)
        d2_up = self._resize_like(self.base.upconv2(d3), e1_mono)
        d2 = self._modulate(
            "d2", self.base.dec2(torch.cat([d2_up, e1_mono], dim=1)), condition
        )
        d1_up = self._resize_like(self.base.upconv1(d2), e1_mono)
        d1 = self._modulate(
            "d1", self.base.dec1(torch.cat([d1_up, e1_mono], dim=1)), condition
        )

        mono_mask = F.softplus(self.base.out_mono(d1)) + 0.1
        diff_mask = torch.tanh(self.base.out_diff(d1))
        return mono_mask, diff_mask

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return list(self.film.parameters())

    def base_parameters(self) -> list[nn.Parameter]:
        return list(self.base.parameters())

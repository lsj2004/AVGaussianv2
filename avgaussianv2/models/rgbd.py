from __future__ import annotations

import torch
from torch import Tensor, nn

from avgaussianv2.contracts import RGBDRender


def _masked_batch_median(values: Tensor, valid: Tensor) -> Tensor:
    medians = []
    for batch_index in range(values.shape[0]):
        selected = values[batch_index][valid[batch_index]]
        if selected.numel() == 0:
            medians.append(values[batch_index].sum() * 0.0)
        else:
            medians.append(selected.median())
    return torch.stack(medians)


def normalize_depth(
    depth: Tensor,
    alpha: Tensor,
    alpha_threshold: float = 1e-3,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise ValueError("depth must have shape (B,H,W,1)")
    if alpha.shape != depth.shape:
        raise ValueError("alpha must match depth shape")
    if not 0 <= alpha_threshold <= 1:
        raise ValueError("alpha_threshold must be in [0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")

    valid = (alpha >= alpha_threshold) & torch.isfinite(depth) & (depth > 0)
    safe_depth = torch.where(valid, depth.clamp_min(eps), torch.ones_like(depth))
    log_depth = torch.log1p(safe_depth)
    flat = log_depth.flatten(1)
    flat_valid = valid.flatten(1)
    center = _masked_batch_median(flat, flat_valid).view(-1, 1, 1, 1)
    deviation = (log_depth - center).abs()
    scale = _masked_batch_median(deviation.flatten(1), flat_valid).view(-1, 1, 1, 1)
    normalized = torch.where(valid, (log_depth - center) / scale.clamp_min(eps), 0.0)
    return torch.nan_to_num(normalized), valid


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
    )


class RGBDConditionEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        alpha_threshold: float = 1e-3,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.alpha_threshold = float(alpha_threshold)
        self.features = nn.Sequential(
            _conv_block(5, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(128, self.embedding_dim)

    def forward(self, render: RGBDRender) -> Tensor:
        depth, mask = normalize_depth(
            render.depth,
            render.alpha,
            alpha_threshold=self.alpha_threshold,
        )
        features = torch.cat([render.rgb, depth, mask.to(depth)], dim=-1)
        features = features.permute(0, 3, 1, 2).contiguous()
        encoded = self.features(features)
        return self.projection(self.pool(encoded).flatten(1))

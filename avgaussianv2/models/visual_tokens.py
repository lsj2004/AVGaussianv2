from __future__ import annotations

import torch
from torch import Tensor, nn

from avgaussianv2.contracts import RGBDRender
from avgaussianv2.models.positional import add_grid_position_encoding
from avgaussianv2.models.rgbd import normalize_depth


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


class RGBDTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        channels: tuple[int, ...] = (32, 64, 128),
        alpha_threshold: float = 1e-3,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must be a nonempty tuple of positive integers")
        self.d_model = int(d_model)
        self.alpha_threshold = float(alpha_threshold)
        layers = []
        in_channels = 5
        for out_channels in channels:
            layers.append(_conv_block(in_channels, int(out_channels)))
            in_channels = int(out_channels)
        self.features = nn.Sequential(*layers)
        self.projection = nn.Conv2d(in_channels, self.d_model, kernel_size=1)

    def _content_tokens(
        self,
        render: RGBDRender,
    ) -> tuple[Tensor, tuple[int, int]]:
        depth, mask = normalize_depth(
            render.depth,
            render.alpha,
            alpha_threshold=self.alpha_threshold,
        )
        features = torch.cat([render.rgb, depth, mask.to(depth)], dim=-1)
        features = features.permute(0, 3, 1, 2).contiguous()
        encoded = self.projection(self.features(features))
        grid_size = (int(encoded.shape[-2]), int(encoded.shape[-1]))
        tokens = encoded.flatten(2).transpose(1, 2).contiguous()
        return tokens, grid_size

    def forward(self, render: RGBDRender) -> Tensor:
        tokens, grid_size = self._content_tokens(render)
        return add_grid_position_encoding(tokens, grid_size)

    def forward_with_content_permutation(
        self,
        render: RGBDRender,
        permutation: Tensor | tuple[int, ...],
    ) -> Tensor:
        """Shuffle visual content while retaining its destination position code."""
        tokens, grid_size = self._content_tokens(render)
        indices = torch.as_tensor(permutation, device=tokens.device, dtype=torch.long)
        if (
            indices.ndim != 1
            or indices.numel() != tokens.shape[1]
            or not torch.equal(
                torch.sort(indices).values,
                torch.arange(tokens.shape[1], device=tokens.device),
            )
        ):
            raise ValueError("content permutation must contain every token index once")
        return add_grid_position_encoding(tokens[:, indices], grid_size)

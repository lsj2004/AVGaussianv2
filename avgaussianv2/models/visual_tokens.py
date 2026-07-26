from __future__ import annotations

import torch
from torch import Tensor, nn

from avgaussianv2.contracts import RGBDRender
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

    def forward(self, render: RGBDRender) -> Tensor:
        depth, mask = normalize_depth(
            render.depth,
            render.alpha,
            alpha_threshold=self.alpha_threshold,
        )
        features = torch.cat([render.rgb, depth, mask.to(depth)], dim=-1)
        features = features.permute(0, 3, 1, 2).contiguous()
        encoded = self.projection(self.features(features))
        return encoded.flatten(2).transpose(1, 2).contiguous()


class PoseTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int = 12,
        d_model: int = 128,
        num_tokens: int = 2,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pose_dim <= 0:
            raise ValueError("pose_dim must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        hidden = int(hidden_dim or max(d_model, pose_dim * 2))
        self.pose_dim = int(pose_dim)
        self.d_model = int(d_model)
        self.num_tokens = int(num_tokens)
        self.network = nn.Sequential(
            nn.Linear(self.pose_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.num_tokens * self.d_model),
        )

    def forward(self, cam_pose: Tensor) -> Tensor:
        if cam_pose.ndim != 2 or cam_pose.shape[-1] != self.pose_dim:
            raise ValueError(f"cam_pose must have shape (B,{self.pose_dim})")
        return self.network(cam_pose).view(cam_pose.shape[0], self.num_tokens, self.d_model)

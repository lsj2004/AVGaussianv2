from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.contracts import RGBDRender
from avgaussianv2.models.positional import add_grid_position_encoding
from avgaussianv2.models.rgbd import normalize_depth


@dataclass(frozen=True)
class VisualMemory:
    tokens: Tensor
    key_padding_mask: Tensor
    world_positions: Tensor
    world_normals: Tensor
    confidence: Tensor
    camera_position: Tensor
    camera_rotation: Tensor

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("visual tokens must have shape (B,N,C)")
        batch, count = self.tokens.shape[:2]
        if self.key_padding_mask.shape != (batch, count):
            raise ValueError("visual key_padding_mask must have shape (B,N)")
        if self.world_positions.shape != (batch, count, 3):
            raise ValueError("visual world_positions must have shape (B,N,3)")
        if self.world_normals.shape != (batch, count, 3):
            raise ValueError("visual world_normals must have shape (B,N,3)")
        if self.confidence.shape != (batch, count):
            raise ValueError("visual confidence must have shape (B,N)")
        if self.camera_position.shape != (batch, 3):
            raise ValueError("visual camera_position must have shape (B,3)")
        if self.camera_rotation.shape != (batch, 3, 3):
            raise ValueError("visual camera_rotation must have shape (B,3,3)")


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


def _camera_geometry(
    depth: Tensor,
    intrinsic: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, height, width = depth.shape[:3]
    dtype, device = depth.dtype, depth.device
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    columns = columns.view(1, height, width)
    rows = rows.view(1, height, width)
    z = depth[..., 0]
    fx = intrinsic[:, 0, 0].view(batch, 1, 1).clamp_min(1e-6)
    fy = intrinsic[:, 1, 1].view(batch, 1, 1).clamp_min(1e-6)
    cx = intrinsic[:, 0, 2].view(batch, 1, 1)
    cy = intrinsic[:, 1, 2].view(batch, 1, 1)
    camera_xyz = torch.stack(
        ((columns - cx) * z / fx, (rows - cy) * z / fy, z),
        dim=-1,
    )
    dx = F.pad(
        camera_xyz[:, :, 2:] - camera_xyz[:, :, :-2],
        (0, 0, 1, 1),
        mode="replicate",
    )
    dy = F.pad(
        camera_xyz[:, 2:] - camera_xyz[:, :-2],
        (0, 0, 0, 0, 1, 1),
        mode="replicate",
    )
    normals = F.normalize(torch.linalg.cross(dx, dy, dim=-1), dim=-1, eps=1e-6)
    return camera_xyz, torch.nan_to_num(normals)


def _world_geometry(
    camera_xyz: Tensor,
    camera_normals: Tensor,
    w2c: Tensor,
) -> tuple[Tensor, Tensor]:
    c2w = torch.linalg.inv(w2c)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3]
    world_xyz = torch.einsum("bij,bhwj->bhwi", rotation, camera_xyz)
    world_xyz = world_xyz + translation[:, None, None]
    world_normals = torch.einsum("bij,bhwj->bhwi", rotation, camera_normals)
    return world_xyz, F.normalize(world_normals, dim=-1, eps=1e-6)


class GeometricVisualTokenEncoder(nn.Module):
    requires_camera_geometry = True

    def __init__(
        self,
        *,
        d_model: int = 128,
        channels: tuple[int, ...] = (32, 64, 128),
        alpha_threshold: float = 1e-3,
        scene_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or scene_scale <= 0:
            raise ValueError("d_model and scene_scale must be positive")
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("visual encoder channels must be positive")
        if not 0 <= alpha_threshold <= 1:
            raise ValueError("alpha_threshold must be in [0,1]")
        self.d_model = int(d_model)
        self.alpha_threshold = float(alpha_threshold)
        self.scene_scale = float(scene_scale)
        layers = []
        in_channels = 16
        for out_channels in channels:
            layers.append(_conv_block(in_channels, int(out_channels)))
            in_channels = int(out_channels)
        self.features = nn.Sequential(*layers)
        self.projection = nn.Conv2d(in_channels, self.d_model, kernel_size=1)

    def forward(
        self,
        render: RGBDRender,
        w2c: Tensor,
        intrinsic: Tensor,
    ) -> VisualMemory:
        depth = render.depth
        valid = (
            (render.alpha >= self.alpha_threshold)
            & torch.isfinite(depth)
            & (depth > 0)
        )
        safe_depth = torch.where(valid, depth.clamp_min(1e-6), torch.ones_like(depth))
        local_depth, _ = normalize_depth(
            depth,
            render.alpha,
            alpha_threshold=self.alpha_threshold,
        )
        camera_xyz, camera_normals = _camera_geometry(safe_depth, intrinsic)
        world_xyz, world_normals = _world_geometry(camera_xyz, camera_normals, w2c)
        validity = valid.to(depth)
        absolute_depth = torch.where(
            valid,
            torch.log1p(safe_depth / self.scene_scale),
            torch.zeros_like(depth),
        )
        inverse_depth = torch.where(
            valid,
            self.scene_scale / safe_depth,
            torch.zeros_like(depth),
        ).clamp(max=100.0)
        camera_scaled = camera_xyz / self.scene_scale
        world_scaled = world_xyz / self.scene_scale
        features = torch.cat(
            (
                render.rgb,
                absolute_depth,
                inverse_depth,
                camera_scaled,
                world_scaled,
                camera_normals,
                render.alpha.clamp(0, 1),
                local_depth,
            ),
            dim=-1,
        )
        features = torch.where(validity.expand_as(features) > 0, features, 0.0)
        encoded = self.projection(
            self.features(features.permute(0, 3, 1, 2).contiguous())
        )
        grid_size = (int(encoded.shape[-2]), int(encoded.shape[-1]))
        tokens = add_grid_position_encoding(
            encoded.flatten(2).transpose(1, 2).contiguous(),
            grid_size,
        )

        def resize_geometry(value: Tensor) -> Tensor:
            return F.interpolate(
                value.permute(0, 3, 1, 2),
                size=grid_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        positions = resize_geometry(world_xyz).flatten(1, 2)
        normals = F.normalize(
            resize_geometry(world_normals).flatten(1, 2),
            dim=-1,
            eps=1e-6,
        )
        confidence_map = F.interpolate(
            render.alpha.permute(0, 3, 1, 2),
            size=grid_size,
            # CUDA adaptive-area backward is nondeterministic. The strict
            # cam38 runner requires deterministic algorithms for exact resume.
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        confidence = confidence_map.flatten(1).clamp(0, 1)
        key_padding_mask = confidence < self.alpha_threshold
        key_padding_mask = key_padding_mask.clone()
        for batch_index in range(key_padding_mask.shape[0]):
            if bool(key_padding_mask[batch_index].all()):
                best = int(confidence[batch_index].argmax())
                key_padding_mask[batch_index, best] = False
        camera_to_world = torch.linalg.inv(w2c)
        return VisualMemory(
            tokens=tokens,
            key_padding_mask=key_padding_mask,
            world_positions=positions,
            world_normals=normals,
            confidence=confidence,
            camera_position=camera_to_world[:, :3, 3],
            camera_rotation=w2c[:, :3, :3],
        )

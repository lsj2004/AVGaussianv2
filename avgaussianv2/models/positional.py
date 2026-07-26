"""Deterministic positional encodings for variable-size token grids."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _axis_encoding(
    length: int,
    channels: int,
    *,
    device: torch.device,
) -> Tensor:
    if length <= 0 or channels <= 0:
        raise ValueError("position-encoding dimensions must be positive")
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, channels, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / channels)
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, channels, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if channels > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : channels // 2])
    return encoding


def grid_position_encoding(
    grid_size: tuple[int, int],
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return one ``(1, height*width, channels)`` row-major 2-D encoding."""
    height, width = (int(grid_size[0]), int(grid_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("grid_size entries must be positive")
    if channels < 2:
        raise ValueError("2-D position encoding requires at least two channels")
    row_channels = channels // 2
    column_channels = channels - row_channels
    rows = _axis_encoding(height, row_channels, device=device)
    columns = _axis_encoding(width, column_channels, device=device)
    grid = torch.cat(
        [
            rows[:, None, :].expand(height, width, row_channels),
            columns[None, :, :].expand(height, width, column_channels),
        ],
        dim=-1,
    )
    return grid.reshape(1, height * width, channels).to(dtype=dtype)


def add_grid_position_encoding(
    tokens: Tensor,
    grid_size: tuple[int, int],
) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape (B,N,C)")
    expected_tokens = int(grid_size[0]) * int(grid_size[1])
    if tokens.shape[1] != expected_tokens:
        raise ValueError("tokens length does not match grid_size")
    return tokens + grid_position_encoding(
        grid_size,
        int(tokens.shape[-1]),
        device=tokens.device,
        dtype=tokens.dtype,
    )


__all__ = ["add_grid_position_encoding", "grid_position_encoding"]

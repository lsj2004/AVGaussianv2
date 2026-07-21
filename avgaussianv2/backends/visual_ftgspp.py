from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import Tensor, nn

from avgaussianv2.contracts import RGBDRender


class UnsupportedFTGSCheckpoint(TypeError):
    """Raised when a checkpoint is not a FreeTimeGS++ Gaussian module."""


Rasterize = Callable[..., tuple[Tensor, Tensor, dict[str, Any]]]


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


def _single_shared_time(time: Tensor | float) -> Tensor:
    value = torch.as_tensor(time)
    first = value.reshape(-1)[0]
    if value.numel() > 1 and not torch.allclose(value, first.expand_as(value)):
        raise ValueError("RGBD rendering requires a single shared time for all cameras")
    return first.reshape(1, 1)


class FTGSVisualBackend(nn.Module):
    _REQUIRED_ATTRIBUTES = (
        "means_t",
        "opacities_t",
        "means",
        "quats",
        "scales",
        "sh_0",
        "sh_n",
        "sh_degree",
    )

    def __init__(self, gaussians: nn.Module, rasterize: Rasterize):
        super().__init__()
        self._validate_gaussians(gaussians)
        self.gaussians = gaussians
        self._rasterize = rasterize

    @classmethod
    def _validate_gaussians(cls, gaussians: object) -> None:
        for attribute in cls._REQUIRED_ATTRIBUTES:
            if not hasattr(gaussians, attribute):
                raise UnsupportedFTGSCheckpoint(
                    f"FreeTimeGS++ checkpoint payload is missing {attribute}"
                )
        if not isinstance(gaussians, nn.Module):
            raise UnsupportedFTGSCheckpoint("FreeTimeGS++ payload must be a torch module")

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        upstream_root: str | Path,
    ) -> "FTGSVisualBackend":
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"FreeTimeGS++ checkpoint does not exist: {checkpoint_path}")
        root = Path(upstream_root)
        if not root.exists():
            raise FileNotFoundError(f"FreeTimeGS++ upstream root does not exist: {root}")
        with _temporary_import_root(root):
            importlib.import_module("ftgspp.models.gaussians")
            gsplat = importlib.import_module("gsplat")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "gaussians" in payload:
            payload = payload["gaussians"]
        return cls(payload, rasterize=gsplat.rasterization)

    def render_rgbd(
        self,
        time: Tensor | float,
        w2c: Tensor,
        intrinsic: Tensor,
        image_size: tuple[int, int],
    ) -> RGBDRender:
        render_time = _single_shared_time(time).to(self.gaussians.means)
        if w2c.ndim != 3 or w2c.shape[-2:] != (4, 4):
            raise ValueError("w2c must have shape (B,4,4)")
        if intrinsic.shape != (w2c.shape[0], 3, 3):
            raise ValueError("intrinsic must have shape (B,3,3) and match w2c batch")
        height, width = (int(image_size[0]), int(image_size[1]))
        if height <= 0 or width <= 0:
            raise ValueError("image_size values must be positive")
        rendered, alpha, _ = self._rasterize(
            means=self.gaussians.means_t(render_time),
            quats=self.gaussians.quats,
            scales=self.gaussians.scales.exp(),
            opacities=self.gaussians.opacities_t(render_time).squeeze(-1),
            colors=torch.cat([self.gaussians.sh_0, self.gaussians.sh_n], dim=1),
            viewmats=w2c,
            Ks=intrinsic,
            width=width,
            height=height,
            sh_degree=int(self.gaussians.sh_degree),
            render_mode="RGB+ED",
        )
        expected_shape = (w2c.shape[0], height, width, 4)
        if rendered.shape != expected_shape:
            raise RuntimeError(
                f"gsplat RGB+ED returned shape {tuple(rendered.shape)}, expected {expected_shape}"
            )
        return RGBDRender(
            rgb=rendered[..., :3].clamp(0, 1),
            depth=rendered[..., 3:4],
            alpha=alpha,
        )

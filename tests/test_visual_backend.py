from pathlib import Path

import pytest
import torch
from torch import nn

from avgaussianv2.backends.visual_ftgspp import (
    FTGSVisualBackend,
    UnsupportedFTGSCheckpoint,
)


class FakeGaussians(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.means = nn.Parameter(torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]]))
        self.quats = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
        self.scales = nn.Parameter(torch.zeros(2, 3))
        self.opacities = nn.Parameter(torch.zeros(2, 1))
        self.sh_0 = nn.Parameter(torch.zeros(2, 1, 3))
        self.sh_n = nn.Parameter(torch.zeros(2, 3, 3))
        self.sh_degree = 1

    def means_t(self, time: torch.Tensor) -> torch.Tensor:
        return self.means + time.reshape(1, 1)

    def opacities_t(self, time: torch.Tensor) -> torch.Tensor:
        del time
        return self.opacities.sigmoid()


def fake_rgb_ed_rasterize(**kwargs):
    batch = kwargs["viewmats"].shape[0]
    height = kwargs["height"]
    width = kwargs["width"]
    means = kwargs["means"]
    rgb = means.mean().expand(batch, height, width, 3)
    depth = means[:, 2].mean().expand(batch, height, width, 1)
    rendered = torch.cat([rgb, depth], dim=-1)
    alpha = means.new_full((batch, height, width, 1), 0.75)
    return rendered, alpha, {"render_mode": kwargs["render_mode"]}


def render_inputs() -> dict[str, object]:
    return {
        "time": torch.tensor([[0.25]]),
        "w2c": torch.eye(4).unsqueeze(0),
        "intrinsic": torch.eye(3).unsqueeze(0),
        "image_size": (8, 12),
    }


def test_render_rgbd_splits_rgb_depth_and_preserves_gradient() -> None:
    gaussians = FakeGaussians()
    backend = FTGSVisualBackend(gaussians, rasterize=fake_rgb_ed_rasterize)

    result = backend.render_rgbd(**render_inputs())

    assert result.rgb.shape == (1, 8, 12, 3)
    assert result.depth.shape == (1, 8, 12, 1)
    assert result.alpha.shape == (1, 8, 12, 1)
    result.depth.sum().backward()
    assert gaussians.means.grad is not None
    assert gaussians.means.grad.abs().sum() > 0


def test_render_rgbd_requests_expected_depth_mode() -> None:
    observed = {}

    def rasterize(**kwargs):
        observed.update(kwargs)
        return fake_rgb_ed_rasterize(**kwargs)

    backend = FTGSVisualBackend(FakeGaussians(), rasterize=rasterize)
    backend.render_rgbd(**render_inputs())

    assert observed["render_mode"] == "RGB+ED"


def test_render_rgbd_rejects_multiple_times() -> None:
    backend = FTGSVisualBackend(FakeGaussians(), rasterize=fake_rgb_ed_rasterize)
    inputs = render_inputs()
    inputs["time"] = torch.tensor([[0.25], [0.5]])

    with pytest.raises(ValueError, match="single shared time"):
        backend.render_rgbd(**inputs)


def test_backend_rejects_unsupported_gaussian_payload() -> None:
    with pytest.raises(UnsupportedFTGSCheckpoint, match="means_t"):
        FTGSVisualBackend(nn.Linear(2, 2), rasterize=fake_rgb_ed_rasterize)


def test_load_rejects_missing_checkpoint_before_import(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        FTGSVisualBackend.load(
            tmp_path / "missing.pt",
            upstream_root=tmp_path / "FreeTimeGSPlusPlus",
        )

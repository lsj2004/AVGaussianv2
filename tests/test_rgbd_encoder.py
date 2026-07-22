import torch

from avgaussianv2.contracts import RGBDRender
from avgaussianv2.models.rgbd import RGBDConditionEncoder, normalize_depth


def test_normalize_depth_masks_background_and_keeps_gradient() -> None:
    depth = torch.tensor(
        [[[[1.0], [3.0]], [[9.0], [100.0]]]],
        requires_grad=True,
    )
    alpha = torch.tensor([[[[1.0], [1.0]], [[1.0], [0.0]]]])

    normalized, mask = normalize_depth(depth, alpha, alpha_threshold=0.01)

    assert normalized[0, 1, 1, 0] == 0
    assert not bool(mask[0, 1, 1, 0])
    assert torch.isfinite(normalized).all()
    normalized.square().sum().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert depth.grad[0, 1, 1, 0] == 0


def test_normalize_depth_returns_zero_for_fully_invalid_frame() -> None:
    depth = torch.full((2, 4, 5, 1), float("nan"))
    alpha = torch.zeros_like(depth)

    normalized, mask = normalize_depth(depth, alpha)

    torch.testing.assert_close(normalized, torch.zeros_like(normalized))
    assert not mask.any()


def make_render(batch: int = 2, height: int = 32, width: int = 48) -> RGBDRender:
    depth = torch.linspace(1.0, 10.0, height * width).reshape(1, height, width, 1)
    depth = depth.repeat(batch, 1, 1, 1).requires_grad_()
    return RGBDRender(
        rgb=torch.rand(batch, height, width, 3),
        depth=depth,
        alpha=torch.ones(batch, height, width, 1),
    )


def test_encoder_returns_fixed_embedding() -> None:
    encoder = RGBDConditionEncoder(embedding_dim=128)

    embedding = encoder(make_render())

    assert embedding.shape == (2, 128)
    assert torch.isfinite(embedding).all()


def test_encoder_audio_condition_gradient_reaches_depth() -> None:
    render = make_render(batch=1)
    encoder = RGBDConditionEncoder(embedding_dim=32)

    encoder(render).square().mean().backward()

    assert render.depth.grad is not None
    assert torch.isfinite(render.depth.grad).all()
    assert render.depth.grad.abs().sum() > 0

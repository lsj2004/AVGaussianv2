from pathlib import Path

import pytest
import torch
from torch import nn

from avgaussianv2.backends.audio_audiogs import AudioGSBackend
from avgaussianv2.contracts import RGBDRender
from avgaussianv2.models.p1_audio import AlignedComplexCrossAttention
from avgaussianv2.models.p1_visual import (
    GeometricVisualTokenEncoder,
    VisualMemory,
)


def _memory(*, requires_grad: bool = False) -> VisualMemory:
    return VisualMemory(
        tokens=torch.randn(1, 8, 32, requires_grad=requires_grad),
        key_padding_mask=torch.zeros(1, 8, dtype=torch.bool),
        world_positions=torch.randn(1, 8, 3),
        world_normals=torch.nn.functional.normalize(
            torch.randn(1, 8, 3), dim=-1
        ),
        confidence=torch.ones(1, 8),
        camera_position=torch.zeros(1, 3),
        camera_rotation=torch.eye(3).unsqueeze(0),
    )


def _renderer() -> AlignedComplexCrossAttention:
    return AlignedComplexCrossAttention(
        d_model=32,
        num_layers=1,
        num_heads=4,
        freq_patch=4,
        time_patch=2,
        n_fft=64,
        hop_length=16,
        win_length=32,
    )


def test_geometric_encoder_emits_physical_visual_memory() -> None:
    encoder = GeometricVisualTokenEncoder(
        d_model=32,
        channels=(16, 32),
        alpha_threshold=0.5,
    )
    rgb = torch.rand(1, 16, 20, 3, requires_grad=True)
    depth = torch.full((1, 16, 20, 1), 2.0, requires_grad=True)
    alpha = torch.ones(1, 16, 20, 1)
    alpha[:, :8, :10] = 0
    intrinsic = torch.tensor(
        [[[20.0, 0.0, 10.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]]
    )

    memory = encoder(
        RGBDRender(rgb, depth, alpha),
        torch.eye(4).unsqueeze(0),
        intrinsic,
    )

    assert memory.tokens.shape == (1, 20, 32)
    assert memory.world_positions.shape == (1, 20, 3)
    assert memory.world_normals.shape == (1, 20, 3)
    assert memory.key_padding_mask.any()
    assert not memory.key_padding_mask.all()
    memory.tokens.square().mean().backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    assert depth.grad is not None and torch.isfinite(depth.grad).all()


def test_p1_is_near_native_at_initialization_and_reaches_visual_tokens() -> None:
    renderer = _renderer()
    native = torch.randn(1, 2, 256)
    fields = [
        torch.rand(1, 33, 17, requires_grad=True),
        torch.randn(1, 33, 17, requires_grad=True),
        torch.rand(1, 33, 17),
        torch.rand(1, 33, 17),
    ]
    memory = _memory(requires_grad=True)

    predicted = renderer(
        native,
        *fields,
        torch.zeros(1, 12),
        memory,
    )

    assert predicted.shape == native.shape
    assert torch.isfinite(predicted).all()
    assert (predicted - native).abs().mean() < 0.01
    predicted.square().mean().backward()
    assert memory.tokens.grad is not None
    assert memory.tokens.grad.abs().sum() > 0
    assert fields[0].grad is not None and fields[0].grad.abs().sum() > 0
    assert fields[1].grad is not None and fields[1].grad.abs().sum() > 0


class _NativeMaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.renderer = nn.Identity()
        self.gain = nn.Parameter(torch.tensor(0.75))

    def forward(self, pose, source, return_masks=False):
        del pose
        native = source * self.gain
        if not return_masks:
            return native
        spectrum = torch.stft(
            source[:, 0],
            n_fft=64,
            hop_length=16,
            win_length=32,
            window=torch.hamming_window(32),
            return_complex=True,
        )
        shape = spectrum.shape
        return (
            native,
            torch.ones(shape),
            torch.zeros(shape),
            spectrum.abs(),
            torch.ones(shape),
        )


def test_p1_backend_uses_mask_fields_only_when_conditioned() -> None:
    model = _NativeMaskModel()
    backend = AudioGSBackend(
        model,
        Path("audio.pth"),
        complex_renderer=_renderer(),
    )
    source = torch.randn(1, 2, 256)
    pose = torch.zeros(1, 12)

    native = backend.render(pose, source)
    conditioned = backend.render(pose, source, condition=_memory())

    torch.testing.assert_close(native, source * model.gain)
    assert conditioned.shape == native.shape
    assert not torch.equal(conditioned, native)
    assert backend.audio_unet_parameters() == []
    assert backend.film_parameters()


def test_visual_memory_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="key_padding_mask"):
        VisualMemory(
            tokens=torch.zeros(1, 3, 8),
            key_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
            world_positions=torch.zeros(1, 3, 3),
            world_normals=torch.zeros(1, 3, 3),
            confidence=torch.ones(1, 3),
            camera_position=torch.zeros(1, 3),
            camera_rotation=torch.eye(3).unsqueeze(0),
        )

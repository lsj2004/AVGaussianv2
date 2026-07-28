from dataclasses import replace
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


def visual_memory(
    batch: int = 1,
    count: int = 12,
    channels: int = 32,
    *,
    requires_grad: bool = False,
) -> VisualMemory:
    return VisualMemory(
        tokens=torch.randn(
            batch, count, channels, requires_grad=requires_grad
        ),
        key_padding_mask=torch.zeros(batch, count, dtype=torch.bool),
        world_positions=torch.randn(batch, count, 3),
        world_normals=torch.nn.functional.normalize(
            torch.randn(batch, count, 3), dim=-1
        ),
        confidence=torch.ones(batch, count),
        camera_position=torch.zeros(batch, 3),
        camera_rotation=torch.eye(3).expand(batch, -1, -1).clone(),
    )


def p1_renderer(**kwargs) -> AlignedComplexCrossAttention:
    defaults = {
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "freq_patch": 4,
        "time_patch": 2,
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 32,
    }
    defaults.update(kwargs)
    return AlignedComplexCrossAttention(**defaults)


def test_geometric_visual_encoder_emits_physical_tokens_and_alpha_mask() -> None:
    encoder = GeometricVisualTokenEncoder(
        d_model=32,
        channels=(16, 32),
        alpha_threshold=0.5,
        scene_scale=2.0,
    )
    rgb = torch.rand(1, 16, 20, 3, requires_grad=True)
    depth = torch.full((1, 16, 20, 1), 2.0, requires_grad=True)
    alpha = torch.ones(1, 16, 20, 1)
    alpha[:, :8, :10] = 0.0
    render = RGBDRender(rgb, depth, alpha)
    intrinsic = torch.tensor(
        [[[20.0, 0.0, 10.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]]
    )
    w2c = torch.eye(4).unsqueeze(0)

    memory = encoder(render, w2c, intrinsic)

    assert memory.tokens.shape == (1, 20, 32)
    assert memory.key_padding_mask.shape == (1, 20)
    assert memory.world_positions.shape == (1, 20, 3)
    assert memory.world_normals.shape == (1, 20, 3)
    assert memory.camera_position.shape == (1, 3)
    assert memory.camera_rotation.shape == (1, 3, 3)
    assert memory.key_padding_mask.any()
    assert not memory.key_padding_mask.all()
    memory.tokens.square().mean().backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    assert depth.grad is not None and torch.isfinite(depth.grad).all()


def test_geometric_visual_encoder_keeps_one_key_for_empty_render() -> None:
    encoder = GeometricVisualTokenEncoder(
        d_model=16,
        channels=(16,),
        alpha_threshold=0.5,
    )
    render = RGBDRender(
        torch.zeros(1, 8, 8, 3),
        torch.zeros(1, 8, 8, 1),
        torch.zeros(1, 8, 8, 1),
    )

    memory = encoder(
        render,
        torch.eye(4).unsqueeze(0),
        torch.eye(3).unsqueeze(0),
    )

    assert (~memory.key_padding_mask).sum() == 1
    assert torch.isfinite(memory.tokens).all()


def test_p1_complex_renderer_is_near_identity_and_backpropagates_to_vision() -> None:
    renderer = p1_renderer()
    native = torch.randn(1, 2, 256)
    mono = torch.rand(1, 33, 17, requires_grad=True)
    diff = torch.randn(1, 33, 17, requires_grad=True)
    source_magnitude = torch.rand(1, 33, 17)
    distance = torch.rand(1, 33, 17)
    pose = torch.zeros(1, 12)
    memory = visual_memory(requires_grad=True)

    predicted = renderer(
        native,
        mono,
        diff,
        source_magnitude,
        distance,
        pose,
        memory,
    )

    assert predicted.shape == native.shape
    assert torch.isfinite(predicted).all()
    assert (predicted - native).abs().mean() < 0.01
    predicted.square().mean().backward()
    assert memory.tokens.grad is not None
    assert memory.tokens.grad.abs().sum() > 0
    assert mono.grad is not None and mono.grad.abs().sum() > 0
    assert diff.grad is not None and diff.grad.abs().sum() > 0


def test_p1_uses_pose_and_respects_visual_padding_mask() -> None:
    renderer = p1_renderer()
    native = torch.randn(1, 2, 256)
    fields = [torch.rand(1, 33, 17) for _ in range(4)]
    memory = visual_memory(count=5)
    masked = VisualMemory(
        tokens=memory.tokens.clone(),
        key_padding_mask=torch.tensor([[False, True, True, True, True]]),
        world_positions=memory.world_positions,
        world_normals=memory.world_normals,
        confidence=memory.confidence,
        camera_position=memory.camera_position,
        camera_rotation=memory.camera_rotation,
    )
    altered_tokens = masked.tokens.clone()
    altered_tokens[:, 1:] = 1e4
    altered = VisualMemory(
        tokens=altered_tokens,
        key_padding_mask=masked.key_padding_mask,
        world_positions=masked.world_positions,
        world_normals=masked.world_normals,
        confidence=masked.confidence,
        camera_position=masked.camera_position,
        camera_rotation=masked.camera_rotation,
    )

    first = renderer(native, *fields, torch.zeros(1, 12), masked)
    second = renderer(native, *fields, torch.zeros(1, 12), altered)

    torch.testing.assert_close(first, second)


def test_geometry_bias_varies_by_time_frequency_query_and_head_pose() -> None:
    renderer = p1_renderer()
    memory = visual_memory(count=5)
    query = torch.randn(1, 6, 32)
    pose = torch.cat(
        (
            torch.zeros(1, 3),
            torch.eye(3).reshape(1, 9),
        ),
        dim=1,
    )

    bias = renderer._geometry_attention_bias(
        memory,
        pose,
        query,
        (2, 3),
    )
    moved_pose = pose.clone()
    moved_pose[:, 0] = 1.0
    moved_bias = renderer._geometry_attention_bias(
        memory,
        moved_pose,
        query,
        (2, 3),
    )
    alternate_view_bias = renderer._geometry_attention_bias(
        replace(
            memory,
            camera_position=torch.ones_like(memory.camera_position),
        ),
        pose,
        query,
        (2, 3),
    )

    assert bias.shape == (1, 4, 6, 5)
    assert not torch.allclose(bias[:, :, 0], bias[:, :, -1])
    assert not torch.allclose(bias, moved_bias)
    assert not torch.allclose(bias, alternate_view_bias)


class NativeMaskModel(nn.Module):
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


def test_backend_uses_native_return_masks_only_when_conditioned() -> None:
    model = NativeMaskModel()
    cross = p1_renderer()
    backend = AudioGSBackend(
        model,
        source_path=Path("audio.pth"),
        complex_renderer=cross,
    )
    source = torch.randn(1, 2, 256)
    pose = torch.zeros(1, 12)

    native = backend.render(pose, source)
    conditioned = backend.render(
        pose,
        source,
        condition=visual_memory(),
    )

    torch.testing.assert_close(native, source * model.gain)
    assert conditioned.shape == native.shape
    assert not torch.equal(conditioned, native)
    assert backend.audio_unet_parameters() == []
    assert {
        id(parameter) for parameter in backend.film_parameters()
    } == {id(parameter) for parameter in cross.parameters()}


def test_visual_memory_validates_contract() -> None:
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

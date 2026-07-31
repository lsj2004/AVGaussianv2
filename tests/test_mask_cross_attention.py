from pathlib import Path

import pytest
import torch
from torch import nn

from avgaussianv2.backends.audio_audiogs import AudioGSBackend
from avgaussianv2.models.mask_cross_attention import (
    AudioFeatureMaskCrossAttention,
)


def _renderer(**changes) -> AudioFeatureMaskCrossAttention:
    options = {
        "d_model": 32,
        "num_layers": 1,
        "num_heads": 4,
        "freq_patch": 4,
        "time_patch": 2,
        "cross_gate_init": 0.1,
    }
    options.update(changes)
    return AudioFeatureMaskCrossAttention(**options)


def test_mask_renderer_matches_audiogs_protocol_and_backpropagates() -> None:
    model = _renderer()
    mono = torch.randn(2, 3, 17, 10, requires_grad=True)
    diff = torch.randn(2, 2, 17, 10, requires_grad=True)
    visual = torch.randn(2, 7, 32, requires_grad=True)

    plain = model(mono, diff)
    with model.use_condition(visual):
        conditioned = model(mono, diff)

    assert conditioned[0].shape == (2, 1, 17, 10)
    assert conditioned[1].shape == (2, 1, 17, 10)
    assert torch.all(conditioned[0] >= 0.1)
    assert torch.all(conditioned[1].abs() <= 1.0)
    assert not torch.allclose(conditioned[0], plain[0])
    (conditioned[0].mean() + conditioned[1].square().mean()).backward()
    for value in (mono, diff, visual):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_mask_condition_context_always_clears() -> None:
    model = _renderer()
    condition = torch.randn(1, 5, 32)

    with model.use_condition(condition):
        with pytest.raises(RuntimeError, match="nested"):
            with model.use_condition(condition):
                pass

    assert model.active_condition is None


class _NativeGSOnly(nn.Module):
    def __init__(self, renderer: nn.Module) -> None:
        super().__init__()
        self.renderer = renderer
        self.gain = nn.Parameter(torch.tensor(0.75))

    def forward(self, _pose, source):
        return source * self.gain


def _parent_synthesis(model, _pose, source):
    batch = source.shape[0]
    mono = source.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
    mono = mono.unsqueeze(-1).expand(batch, 3, 17, 10)
    diff = (source[:, :1] - source[:, 1:]).mean(dim=-1, keepdim=True)
    diff = diff.unsqueeze(-1).expand(batch, 2, 17, 10)
    mono_mask, diff_mask = model.renderer(mono, diff)
    gain = mono_mask.mean((1, 2, 3)) + diff_mask.mean((1, 2, 3))
    return source * gain[:, None, None]


def test_mask_backend_preserves_native_residual_control_rule() -> None:
    renderer = _renderer()
    native = _NativeGSOnly(renderer)
    backend = AudioGSBackend(
        native,
        Path("audio.pth"),
        forward_override=_parent_synthesis,
    )
    pose = torch.zeros(1, 12)
    source = torch.randn(1, 2, 160)
    condition = torch.randn(1, 7, 32)

    plain = _parent_synthesis(native, pose, source)
    with renderer.use_condition(condition):
        conditioned = _parent_synthesis(native, pose, source)

    torch.testing.assert_close(backend.render(pose, source), native(pose, source))
    torch.testing.assert_close(
        backend.render(pose, source, condition),
        native(pose, source) + conditioned - plain,
    )
    assert backend.audio_unet_parameters() == []

from pathlib import Path

import pytest
import torch
from torch import nn

from avgaussianv2.backends.audio_audiogs import AudioGSBackend
from avgaussianv2.models.mask_cross_attention import (
    AudioFeatureMaskCrossAttention,
)


def renderer(**kwargs) -> AudioFeatureMaskCrossAttention:
    defaults = {
        "d_model": 32,
        "num_layers": 1,
        "num_heads": 4,
        "freq_patch": 16,
        "time_patch": 4,
        "cross_gate_init": 0.1,
    }
    defaults.update(kwargs)
    return AudioFeatureMaskCrossAttention(**defaults)


def test_renderer_matches_exact_audiogs_mask_protocol() -> None:
    model = renderer()
    mono_features = torch.randn(1, 3, 257, 160)
    diff_features = torch.randn(1, 2, 257, 160)
    visual_tokens = torch.randn(1, 12 * 21, 32)

    with model.use_condition(visual_tokens):
        mono_mask, diff_mask = model(mono_features, diff_features)

    assert mono_mask.shape == (1, 1, 257, 160)
    assert diff_mask.shape == (1, 1, 257, 160)
    assert torch.isfinite(mono_mask).all()
    assert torch.isfinite(diff_mask).all()
    assert torch.all(mono_mask >= 0.1)
    assert torch.all(diff_mask >= -1.0)
    assert torch.all(diff_mask <= 1.0)


def test_visual_condition_changes_masks_and_receives_gradient() -> None:
    model = renderer(freq_patch=4, time_patch=2)
    mono_features = torch.randn(2, 3, 17, 10, requires_grad=True)
    diff_features = torch.randn(2, 2, 17, 10, requires_grad=True)
    visual_tokens = torch.randn(2, 7, 32, requires_grad=True)

    plain_masks = model(mono_features, diff_features)
    with model.use_condition(visual_tokens):
        conditioned_masks = model(mono_features, diff_features)

    assert not torch.allclose(conditioned_masks[0], plain_masks[0])
    assert not torch.allclose(conditioned_masks[1], plain_masks[1])
    loss = conditioned_masks[0].mean() + conditioned_masks[1].square().mean()
    loss.backward()
    for value in (mono_features, diff_features, visual_tokens):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_condition_context_validates_batch_and_always_clears() -> None:
    model = renderer(freq_patch=4, time_patch=2)
    mono_features = torch.randn(2, 3, 17, 10)
    diff_features = torch.randn(2, 2, 17, 10)
    wrong_batch = torch.randn(1, 5, 32)

    with pytest.raises(ValueError, match="batches must match"):
        with model.use_condition(wrong_batch):
            model(mono_features, diff_features)
    assert model.active_condition is None

    condition = torch.randn(2, 5, 32)
    with model.use_condition(condition):
        with pytest.raises(RuntimeError, match="nested"):
            with model.use_condition(condition):
                pass
    assert model.active_condition is None


class _NativeGSOnlyModel(nn.Module):
    def __init__(self, conditioned_renderer: nn.Module) -> None:
        super().__init__()
        self.renderer = conditioned_renderer
        self.native_gain = nn.Parameter(torch.tensor(0.75))

    def forward(self, _pose: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return source * self.native_gain


def _parent_synthesis(
    model: _NativeGSOnlyModel,
    _pose: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    batch = source.shape[0]
    mono_features = source.mean(dim=1, keepdim=True).mean(
        dim=-1, keepdim=True
    ).unsqueeze(-1)
    mono_features = mono_features.expand(batch, 3, 17, 10)
    diff_features = (source[:, :1] - source[:, 1:]).mean(
        dim=-1, keepdim=True
    ).unsqueeze(-1)
    diff_features = diff_features.expand(batch, 2, 17, 10)
    mono_mask, diff_mask = model.renderer(mono_features, diff_features)
    gain = mono_mask.mean(dim=(1, 2, 3)) + diff_mask.mean(dim=(1, 2, 3))
    return source * gain[:, None, None]


def test_backend_keeps_native_plus_conditioned_minus_plain_control_rule() -> None:
    conditioned_renderer = renderer(freq_patch=4, time_patch=2)
    native_model = _NativeGSOnlyModel(conditioned_renderer)
    backend = AudioGSBackend(
        native_model,
        Path("audio.pth"),
        forward_override=_parent_synthesis,
    )
    pose = torch.zeros(2, 12)
    source = torch.randn(2, 2, 160)
    condition = torch.randn(2, 7, 32)

    native = native_model(pose, source)
    plain = _parent_synthesis(native_model, pose, source)
    with conditioned_renderer.use_condition(condition):
        conditioned = _parent_synthesis(native_model, pose, source)
    actual = backend.render(pose, source, condition=condition)

    torch.testing.assert_close(backend.render(pose, source), native)
    torch.testing.assert_close(actual, native + conditioned - plain)
    assert backend.audio_unet_parameters() == []
    assert {
        id(parameter) for parameter in backend.film_parameters()
    } == {
        id(parameter) for parameter in conditioned_renderer.parameters()
    }

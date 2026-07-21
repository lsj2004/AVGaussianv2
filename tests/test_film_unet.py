import torch
import torch.nn.functional as F
from torch import nn

from avgaussianv2.models.film_unet import FiLMConditionedAudioUNet


def layer(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )


class TinyDualBranchAudioUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = layer(3, 8)
        self.diff_enc1 = layer(1, 8)
        self.enc2 = layer(8, 16)
        self.enc3 = layer(16, 24)
        self.enc4 = layer(24, 32)
        self.maxpool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec4 = layer(48, 24)
        self.upconv3 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec3 = layer(32, 16)
        self.upconv2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec2 = layer(16, 8)
        self.upconv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.dec1 = layer(16, 8)
        self.out_mono = nn.Conv2d(8, 1, 1)
        self.out_diff = nn.Conv2d(8, 1, 1)

    @staticmethod
    def resize_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, mono_features, diff_features):
        e1 = self.enc1(mono_features)
        diff_e1 = self.diff_enc1(diff_features)
        e2 = self.enc2(0.5 * (self.maxpool(e1) + self.maxpool(diff_e1)))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        d4 = self.dec4(torch.cat([self.resize_like(self.upconv4(e4), e3), e3], dim=1))
        d3 = self.dec3(torch.cat([self.resize_like(self.upconv3(d4), e2), e2], dim=1))
        d2 = self.dec2(torch.cat([self.resize_like(self.upconv2(d3), e1), e1], dim=1))
        d1 = self.dec1(torch.cat([self.resize_like(self.upconv1(d2), e1), e1], dim=1))
        return F.softplus(self.out_mono(d1)) + 0.1, torch.tanh(self.out_diff(d1))


def inputs():
    torch.manual_seed(7)
    return (
        torch.randn(2, 3, 17, 19),
        torch.randn(2, 1, 17, 19),
        torch.randn(2, 12),
    )


def test_zero_init_is_identical_to_base() -> None:
    mono, diff, condition = inputs()
    base = TinyDualBranchAudioUNet()
    expected = base(mono, diff)
    wrapped = FiLMConditionedAudioUNet(base, embedding_dim=condition.shape[-1])

    with wrapped.use_condition(condition):
        actual = wrapped(mono, diff)

    torch.testing.assert_close(actual[0], expected[0], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-7, rtol=1e-7)


def test_no_condition_delegates_to_base() -> None:
    mono, diff, _ = inputs()
    base = TinyDualBranchAudioUNet()
    wrapped = FiLMConditionedAudioUNet(base, embedding_dim=12)

    expected = base(mono, diff)
    actual = wrapped(mono, diff)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_nonzero_film_changes_output_and_reaches_condition() -> None:
    mono, diff, condition = inputs()
    condition.requires_grad_()
    wrapped = FiLMConditionedAudioUNet(TinyDualBranchAudioUNet(), embedding_dim=12)
    with torch.no_grad():
        wrapped.film["e1"].to_scale_shift.weight.fill_(0.02)

    baseline = wrapped(mono, diff)[0]
    with wrapped.use_condition(condition):
        conditioned = wrapped(mono, diff)[0]

    assert not torch.allclose(conditioned, baseline)
    conditioned.mean().backward()
    assert condition.grad is not None
    assert torch.isfinite(condition.grad).all()
    assert condition.grad.abs().sum() > 0


def test_condition_context_is_cleared_after_exception() -> None:
    _, _, condition = inputs()
    wrapped = FiLMConditionedAudioUNet(TinyDualBranchAudioUNet(), embedding_dim=12)

    try:
        with wrapped.use_condition(condition):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert wrapped.active_condition is None


def test_nested_condition_context_is_rejected() -> None:
    _, _, condition = inputs()
    wrapped = FiLMConditionedAudioUNet(TinyDualBranchAudioUNet(), embedding_dim=12)

    with wrapped.use_condition(condition):
        try:
            with wrapped.use_condition(condition):
                raise AssertionError("nested context unexpectedly entered")
        except RuntimeError as error:
            assert "nested" in str(error)

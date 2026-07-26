from pathlib import Path

import pytest
import torch
from torch import nn

import avgaussianv2.models.cross_attention_audio as cross_module
from avgaussianv2.backends.audio_audiogs import AudioCheckpointError
from avgaussianv2.contracts import RGBDRender
from avgaussianv2.models.audio_tokens import AudioSTFTTokenizer, AudioSpectrogramHead
from avgaussianv2.models.cross_attention_audio import (
    AudioVisualTokenAudioBackend,
    GatedCrossAttentionBlock,
)
from avgaussianv2.models.positional import grid_position_encoding
from avgaussianv2.models.visual_tokens import RGBDTokenEncoder


class TinyNativeAudioGS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gaussian_gain = nn.Parameter(torch.tensor(0.75))
        self.renderer = nn.Linear(1, 1)

    def forward(self, cam_pose, source_audio):
        del cam_pose
        return source_audio * self.gaussian_gain


def cross_backend(**kwargs) -> AudioVisualTokenAudioBackend:
    defaults = {
        "d_model": 32,
        "num_layers": 1,
        "num_heads": 4,
        "n_fft": 32,
        "hop_length": 8,
        "win_length": 16,
        "freq_patch": 4,
        "time_patch": 2,
    }
    defaults.update(kwargs)
    return AudioVisualTokenAudioBackend(
        TinyNativeAudioGS(),
        Path("native-audiogs.pth"),
        **defaults,
    )


def test_audio_stft_tokenizer_returns_time_frequency_tokens() -> None:
    tokenizer = AudioSTFTTokenizer(
        d_model=32,
        n_fft=32,
        hop_length=8,
        win_length=16,
        freq_patch=4,
        time_patch=2,
    )
    source_audio = torch.randn(2, 2, 160)

    batch = tokenizer(source_audio)

    assert batch.tokens.ndim == 3
    assert batch.tokens.shape[0] == 2
    assert batch.tokens.shape[-1] == 32
    assert batch.tokens.shape[1] == batch.grid_size[0] * batch.grid_size[1]
    assert batch.source_stft.shape[:2] == (2, 2)
    assert batch.source_stft.is_complex()
    assert torch.isfinite(batch.tokens).all()


def test_spectrogram_head_returns_binaural_waveform() -> None:
    tokenizer = AudioSTFTTokenizer(
        d_model=24,
        n_fft=32,
        hop_length=8,
        win_length=16,
        freq_patch=4,
        time_patch=2,
    )
    head = AudioSpectrogramHead(
        d_model=24,
        n_fft=32,
        hop_length=8,
        win_length=16,
        freq_patch=4,
        time_patch=2,
    )
    source_audio = torch.randn(2, 2, 160)
    batch = tokenizer(source_audio)

    predicted = head(
        batch.tokens,
        batch.grid_size,
        batch.source_stft,
        length=batch.original_samples,
    )

    assert predicted.shape == source_audio.shape
    assert torch.isfinite(predicted).all()


def test_rgbd_token_encoder_returns_tokens_and_reaches_depth() -> None:
    encoder = RGBDTokenEncoder(d_model=32, channels=(16, 32))
    depth = torch.linspace(1.0, 5.0, 2 * 32 * 48).reshape(2, 32, 48, 1)
    depth.requires_grad_()
    render = RGBDRender(
        rgb=torch.rand(2, 32, 48, 3),
        depth=depth,
        alpha=torch.ones(2, 32, 48, 1),
    )

    tokens = encoder(render)

    assert tokens.ndim == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[-1] == 32
    tokens.square().mean().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert depth.grad.abs().sum() > 0


def test_rgbd_content_shuffle_retains_positions_but_changes_condition() -> None:
    encoder = RGBDTokenEncoder(d_model=32, channels=(16, 32))
    render = RGBDRender(
        rgb=torch.rand(1, 32, 48, 3),
        depth=torch.rand(1, 32, 48, 1) + 0.1,
        alpha=torch.ones(1, 32, 48, 1),
    )
    original = encoder(render)
    permutation = tuple(reversed(range(original.shape[1])))

    shuffled = encoder.forward_with_content_permutation(render, permutation)

    assert shuffled.shape == original.shape
    assert not torch.allclose(shuffled, original)
    with pytest.raises(ValueError, match="every token index"):
        encoder.forward_with_content_permutation(
            render,
            tuple(0 for _ in range(original.shape[1])),
        )


def test_gated_cross_attention_block_is_identity_at_zero_gates() -> None:
    block = GatedCrossAttentionBlock(
        d_model=32,
        num_heads=4,
        cross_gate_init=0.0,
    )
    audio_tokens = torch.randn(2, 7, 32)
    memory_tokens = torch.randn(2, 5, 32)

    output = block(audio_tokens, memory_tokens)

    torch.testing.assert_close(output, audio_tokens)


def test_grid_position_encoding_identifies_every_row_and_column() -> None:
    encoding = grid_position_encoding(
        (3, 5),
        32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).squeeze(0)

    assert encoding.shape == (15, 32)
    assert torch.unique(encoding, dim=0).shape[0] == 15
    assert not torch.allclose(encoding[0], encoding[1])
    assert not torch.allclose(encoding[0], encoding[5])


def test_cross_attention_condition_path_has_gradient_on_first_step() -> None:
    backend = cross_backend(cross_gate_init=0.01)
    condition = torch.randn(2, 6, 32, requires_grad=True)
    prediction = backend.render(
        torch.randn(2, 12),
        torch.randn(2, 2, 160),
        condition=condition,
    )

    prediction.square().mean().backward()

    assert condition.grad is not None
    assert torch.isfinite(condition.grad).all()
    assert condition.grad.abs().sum() > 0
    cross_attention_grad = sum(
        parameter.grad.abs().sum()
        for parameter in backend.transformer.blocks[0].cross_attention.parameters()
        if parameter.grad is not None
    )
    assert cross_attention_grad > 0


def test_gated_cross_attention_block_reaches_memory_when_gate_opens() -> None:
    block = GatedCrossAttentionBlock(d_model=32, num_heads=4)
    with torch.no_grad():
        block.cross_gate.fill_(1.0)
    audio_tokens = torch.randn(2, 7, 32)
    memory_tokens = torch.randn(2, 5, 32, requires_grad=True)

    block(audio_tokens, memory_tokens).square().mean().backward()

    assert memory_tokens.grad is not None
    assert torch.isfinite(memory_tokens.grad).all()
    assert memory_tokens.grad.abs().sum() > 0


def test_audio_visual_token_backend_uses_condition_only_when_gate_opens() -> None:
    backend = cross_backend(cross_gate_init=0.0)
    source_audio = torch.randn(2, 2, 160)
    cam_pose = torch.randn(2, 12)
    condition_a = torch.randn(2, 6, 32)
    condition_b = torch.randn(2, 6, 32)

    closed_a = backend.render(cam_pose, source_audio, condition=condition_a)
    closed_b = backend.render(cam_pose, source_audio, condition=condition_b)
    torch.testing.assert_close(closed_a, closed_b)

    with torch.no_grad():
        backend.transformer.blocks[0].cross_gate.fill_(1.0)
    opened_a = backend.render(cam_pose, source_audio, condition=condition_a)
    opened_b = backend.render(cam_pose, source_audio, condition=condition_b)
    assert not torch.allclose(opened_a, opened_b)

    condition = condition_a.detach().clone().requires_grad_()
    backend.render(cam_pose, source_audio, condition=condition).square().mean().backward()
    assert condition.grad is not None
    assert torch.isfinite(condition.grad).all()
    assert condition.grad.abs().sum() > 0


def test_cross_backend_starts_from_native_audiogs_and_removes_unet() -> None:
    backend = cross_backend(cross_gate_init=0.01)
    source = torch.randn(2, 2, 160)
    pose = torch.randn(2, 12)
    native = source * backend.model.gaussian_gain

    torch.testing.assert_close(backend.render(pose, source), native)
    assert isinstance(backend.model.renderer, nn.Identity)

    conditioned = backend.render(
        pose,
        source,
        condition=torch.randn(2, 6, 32),
    )
    assert conditioned.shape == native.shape
    assert not torch.allclose(conditioned, source)
    assert all(
        parameter is not backend.model.gaussian_gain
        for parameter in backend.film_parameters()
    )
    assert backend.audio_unet_parameters() == []


def test_cross_backend_reuses_audiogs_checkpoint_criterion(monkeypatch) -> None:
    backend = cross_backend()
    backend.checkpoint_config = object()
    backend.upstream_root = Path("/audio-upstream")
    criterion = nn.L1Loss()
    seen = []
    monkeypatch.setattr(
        cross_module,
        "build_audiogs_criterion",
        lambda config, root: seen.append((config, root)) or criterion,
    )

    assert backend.build_criterion() is criterion
    assert seen == [(backend.checkpoint_config, backend.upstream_root)]


def test_cross_backend_loader_rejects_non_gs_only_model_before_io() -> None:
    with pytest.raises(AudioCheckpointError, match="Audio3DGSMonoDiffGSOnly"):
        AudioVisualTokenAudioBackend.load(
            Path("/missing.pth"),
            model_class="Audio3DGS",
        )

from pathlib import Path

import pytest
import torch
from torch import nn

import avgaussianv2.models.cross_attention_audio as cross_module
from avgaussianv2.backends.audio_audiogs import AudioCheckpointError
from avgaussianv2.contracts import RGBDRender
from avgaussianv2.models.audio_tokens import AudioSTFTTokenizer, AudioSpectrogramHead
from avgaussianv2.models.acoustic_gaussian_tokens import (
    AudioGSGaussianAttributeAdapter,
    GaussianTokenEncoder,
)
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
        self.freq_num = 4
        self.time_num = 5
        self.n_points = self.freq_num * self.time_num
        self.max_norm = 2.0
        self.normalize_world_coords = False
        self.use_cam_rotation = True
        self.flip_cam_y_for_sh = False
        self._xyz = nn.Parameter(torch.randn(self.n_points, 3) * 0.1)
        self._rotation = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.n_points, 1)
        )
        self._sh_mono = nn.Parameter(torch.randn(self.n_points, 1, 4) * 0.1)
        self._sh_diff = nn.Parameter(torch.randn(self.n_points, 1, 4) * 0.1)
        freq, time = torch.meshgrid(
            torch.arange(self.freq_num),
            torch.arange(self.time_num),
            indexing="ij",
        )
        self.register_buffer(
            "tf_coords",
            torch.stack([freq.flatten(), time.flatten()], dim=-1).float(),
        )

    def forward(self, cam_pose, source_audio):
        del cam_pose
        return source_audio * self.gaussian_gain

    def _safe_xyz(self):
        return self._xyz

    def _safe_rotation_quaternions(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    def _safe_sh_mono(self):
        return self._sh_mono

    def _safe_sh_diff(self):
        return self._sh_diff

    def compute_relative_positions(self, cam_pose):
        rel_world = (
            cam_pose[:, None, :3] - self._xyz[None]
        ) / self.max_norm
        if cam_pose.shape[1] == 12:
            rotation = cam_pose[:, 3:].reshape(-1, 3, 3)
            rel_cam = torch.matmul(rel_world, rotation.transpose(1, 2))
        else:
            rel_cam = rel_world
        return rel_world, -rel_world * self.max_norm, rel_cam

    def eval_mono_diff_fields(self, relative_pos):
        direction = torch.nn.functional.normalize(relative_pos, dim=-1)
        basis = torch.cat(
            [torch.ones_like(direction[..., :1]), direction],
            dim=-1,
        )
        mono = (basis * self._sh_mono[:, 0][None]).sum(dim=-1)
        diff = (basis * self._sh_diff[:, 0][None]).sum(dim=-1)
        return mono, diff


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


def test_audio_stft_matches_centered_reflection_reference() -> None:
    tokenizer = AudioSTFTTokenizer(
        d_model=16,
        n_fft=32,
        hop_length=8,
        win_length=16,
        freq_patch=4,
        time_patch=2,
    )
    source_audio = torch.randn(2, 2, 160)
    actual = tokenizer.stft(source_audio)
    reference = torch.stft(
        source_audio.reshape(4, 160),
        n_fft=32,
        hop_length=8,
        win_length=16,
        window=tokenizer.window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    ).reshape_as(actual)

    torch.testing.assert_close(actual, reference)


def test_gaussian_encoder_uses_real_attributes_and_compresses_tf_grid() -> None:
    model = TinyNativeAudioGS()
    adapter = AudioGSGaussianAttributeAdapter(model)
    encoder = GaussianTokenEncoder(
        adapter.feature_dim,
        d_model=32,
        hidden_dim=8,
        token_grid=(2, 3),
    )
    pose = torch.eye(3).reshape(1, 9)
    pose = torch.cat([torch.tensor([[0.2, -0.1, 0.3]]), pose], dim=1)

    batch = adapter(pose)
    tokens = encoder(batch)

    assert batch.features.shape == (1, model.n_points, 29)
    assert tokens.shape == (1, 6, 32)
    tokens.square().mean().backward()
    for parameter in (
        model._xyz,
        model._rotation,
        model._sh_mono,
        model._sh_diff,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gaussian_grid_pool_preserves_constant_boundaries() -> None:
    encoder = GaussianTokenEncoder(
        3,
        d_model=8,
        hidden_dim=4,
        token_grid=(3, 4),
    )
    hidden = torch.full((2, 4, 7, 11), 2.5, requires_grad=True)

    pooled = encoder._deterministic_grid_pool(hidden)

    assert pooled.shape == (2, 4, 3, 4)
    torch.testing.assert_close(pooled, torch.full_like(pooled, 2.5))
    pooled.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


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
    backend = cross_backend(cross_gate_init=0.0)
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


def test_cross_backend_queries_source_and_attends_visual_pose_gaussians() -> None:
    backend = cross_backend(
        cross_gate_init=1.0,
        gaussian_token_rows=2,
        gaussian_token_columns=3,
        gaussian_token_hidden_dim=8,
    )
    source = torch.randn(1, 2, 160)
    pose = torch.cat(
        [torch.randn(1, 3), torch.eye(3).reshape(1, 9)],
        dim=1,
    )
    condition = torch.randn(1, 7, 32, requires_grad=True)

    prediction = backend.render(pose, source, condition)
    prediction.square().mean().backward()

    assert prediction.shape == source.shape
    assert condition.grad is not None and condition.grad.abs().sum() > 0
    assert backend.model._xyz.grad is not None
    assert backend.model._xyz.grad.abs().sum() > 0
    assert backend.pose_encoder.encoder[1].weight.grad is not None


def test_cross_backend_component_toggles_are_same_checkpoint_ablations() -> None:
    backend = cross_backend(
        cross_gate_init=1.0,
        gaussian_token_rows=2,
        gaussian_token_columns=3,
    )
    source = torch.randn(1, 2, 160)
    pose = torch.cat(
        [torch.randn(1, 3), torch.eye(3).reshape(1, 9)],
        dim=1,
    )
    condition = torch.randn(1, 7, 32)
    state_keys = set(backend.state_dict())
    assert not any(key.startswith("gaussian_adapter.model.") for key in state_keys)

    full = backend.render(pose, source, condition)
    backend.gaussian_tokens_enabled = False
    no_gaussians = backend.render(pose, source, condition)
    backend.gaussian_tokens_enabled = True
    backend.pose_tokens_enabled = False
    no_pose = backend.render(pose, source, condition)

    assert not torch.allclose(full, no_gaussians)
    assert not torch.allclose(full, no_pose)


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

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.models.positional import add_grid_position_encoding


@dataclass(frozen=True)
class AudioTokenBatch:
    tokens: Tensor
    grid_size: tuple[int, int]
    source_stft: Tensor
    feature_size: tuple[int, int]
    original_samples: int


def _validate_stft_config(
    n_fft: int,
    hop_length: int,
    win_length: int,
    freq_patch: int,
    time_patch: int,
) -> None:
    values = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "freq_patch": freq_patch,
        "time_patch": time_patch,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(win_length) > int(n_fft):
        raise ValueError("win_length must not exceed n_fft")


def _hann_window(module: nn.Module, win_length: int) -> None:
    module.register_buffer("window", torch.hann_window(int(win_length)), persistent=False)


def _stft(
    source_audio: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: Tensor,
) -> Tensor:
    if source_audio.ndim != 3 or source_audio.shape[1] != 2:
        raise ValueError("source_audio must have shape (B,2,samples)")
    batch, channels, samples = source_audio.shape
    if samples < win_length:
        raise ValueError("source_audio is shorter than win_length")
    flat = source_audio.reshape(batch * channels, samples)
    spectrum = torch.stft(
        flat,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window.to(device=source_audio.device, dtype=source_audio.dtype),
        return_complex=True,
    )
    return spectrum.reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])


def _binaural_features(source_stft: Tensor, eps: float) -> Tensor:
    if source_stft.ndim != 4 or source_stft.shape[1] != 2 or not source_stft.is_complex():
        raise ValueError("source_stft must have shape (B,2,F,T) and complex dtype")
    left = source_stft[:, 0]
    right = source_stft[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    left_mag = left.abs().clamp_min(float(eps))
    right_mag = right.abs().clamp_min(float(eps))
    mid_mag = torch.log1p(mid.abs())
    side_mag = torch.log1p(side.abs())
    ild = torch.log(left_mag) - torch.log(right_mag)
    phase_ratio = left / left_mag * (right / right_mag).conj()
    ipd_sin = phase_ratio.imag
    ipd_cos = phase_ratio.real
    return torch.stack([mid_mag, side_mag, ild, ipd_sin, ipd_cos], dim=1)


def _pad_to_patch_grid(
    features: Tensor,
    freq_patch: int,
    time_patch: int,
) -> tuple[Tensor, tuple[int, int]]:
    freq_bins, frames = int(features.shape[-2]), int(features.shape[-1])
    pad_freq = (-freq_bins) % int(freq_patch)
    pad_time = (-frames) % int(time_patch)
    padded = F.pad(features, (0, pad_time, 0, pad_freq))
    return padded, (padded.shape[-2] // int(freq_patch), padded.shape[-1] // int(time_patch))


class AudioSTFTTokenizer(nn.Module):
    feature_channels = 5

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        freq_patch: int = 16,
        time_patch: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        _validate_stft_config(n_fft, hop_length, win_length, freq_patch, time_patch)
        self.d_model = int(d_model)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.freq_patch = int(freq_patch)
        self.time_patch = int(time_patch)
        self.eps = float(eps)
        _hann_window(self, self.win_length)
        self.patch_embed = nn.Conv2d(
            self.feature_channels,
            self.d_model,
            kernel_size=(self.freq_patch, self.time_patch),
            stride=(self.freq_patch, self.time_patch),
        )

    def stft(self, source_audio: Tensor) -> Tensor:
        return _stft(source_audio, self.n_fft, self.hop_length, self.win_length, self.window)

    def features(self, source_audio: Tensor) -> tuple[Tensor, Tensor]:
        source_stft = self.stft(source_audio)
        return _binaural_features(source_stft, self.eps), source_stft

    def forward(self, source_audio: Tensor) -> AudioTokenBatch:
        features, source_stft = self.features(source_audio)
        padded, grid_size = _pad_to_patch_grid(features, self.freq_patch, self.time_patch)
        embedded = self.patch_embed(padded)
        tokens = embedded.flatten(2).transpose(1, 2).contiguous()
        tokens = add_grid_position_encoding(tokens, grid_size)
        return AudioTokenBatch(
            tokens=tokens,
            grid_size=(int(grid_size[0]), int(grid_size[1])),
            source_stft=source_stft,
            feature_size=(int(features.shape[-2]), int(features.shape[-1])),
            original_samples=int(source_audio.shape[-1]),
        )


class AudioSpectrogramHead(nn.Module):
    output_channels = 4

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        freq_patch: int = 16,
        time_patch: int = 4,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        _validate_stft_config(n_fft, hop_length, win_length, freq_patch, time_patch)
        self.d_model = int(d_model)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.freq_patch = int(freq_patch)
        self.time_patch = int(time_patch)
        self.residual_scale = float(residual_scale)
        _hann_window(self, self.win_length)
        self.projection = nn.Linear(
            self.d_model,
            self.output_channels * self.freq_patch * self.time_patch,
        )

    def _unpatchify(self, tokens: Tensor, grid_size: tuple[int, int]) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_model:
            raise ValueError(f"tokens must have shape (B,N,{self.d_model})")
        grid_freq, grid_time = int(grid_size[0]), int(grid_size[1])
        if tokens.shape[1] != grid_freq * grid_time:
            raise ValueError("tokens length does not match grid_size")
        batch = tokens.shape[0]
        patches = self.projection(tokens)
        patches = patches.view(
            batch,
            grid_freq,
            grid_time,
            self.output_channels,
            self.freq_patch,
            self.time_patch,
        )
        return (
            patches.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(
                batch,
                self.output_channels,
                grid_freq * self.freq_patch,
                grid_time * self.time_patch,
            )
        )

    def forward(
        self,
        tokens: Tensor,
        grid_size: tuple[int, int],
        source_stft: Tensor,
        *,
        length: int,
    ) -> Tensor:
        if source_stft.ndim != 4 or source_stft.shape[1] != 2 or not source_stft.is_complex():
            raise ValueError("source_stft must have shape (B,2,F,T) and complex dtype")
        residual = self._unpatchify(tokens, grid_size)
        freq_bins, frames = int(source_stft.shape[-2]), int(source_stft.shape[-1])
        residual = torch.tanh(residual[..., :freq_bins, :frames]) * self.residual_scale
        left = source_stft[:, 0] + torch.complex(residual[:, 0], residual[:, 1])
        right = source_stft[:, 1] + torch.complex(residual[:, 2], residual[:, 3])
        predicted_stft = torch.stack([left, right], dim=1)
        flat = predicted_stft.reshape(-1, freq_bins, frames)
        waveform = torch.istft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=source_stft.device, dtype=source_stft.real.dtype),
            length=int(length),
        )
        return waveform.reshape(source_stft.shape[0], 2, int(length))

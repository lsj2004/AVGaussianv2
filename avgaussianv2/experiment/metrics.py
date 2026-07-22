from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from avgaussianv2.losses import dssim


def _finite(name: str, *values: Tensor) -> None:
    for value in values:
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} inputs must contain only finite values")


def _matching_shape(name: str, predicted: Tensor, target: Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            f"{name} inputs must have equal shapes, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )


def _values(name: str, predicted: Tensor, target: Tensor) -> None:
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise ValueError(f"{name} inputs must be floating-point tensors")
    if predicted.numel() == 0 or target.numel() == 0:
        raise ValueError(f"{name} inputs must be nonempty")
    if predicted.device != target.device:
        raise ValueError(f"{name} inputs must be on the same device")
    _finite(name, predicted, target)


def _audio(name: str, predicted: Tensor, target: Tensor) -> None:
    _matching_shape(name, predicted, target)
    if predicted.ndim != 3 or predicted.shape[1] != 2:
        raise ValueError(
            f"{name} inputs must be batched stereo tensors with shape [B, 2, samples]"
        )
    _values(name, predicted, target)


def _rgb(name: str, predicted: Tensor, target: Tensor) -> None:
    _matching_shape(name, predicted, target)
    if predicted.ndim != 4 or predicted.shape[-1] != 3:
        raise ValueError(
            f"{name} inputs must be BHWC RGB tensors with shape [B, H, W, 3]"
        )
    _values(name, predicted, target)


def waveform_l1(predicted: Tensor, target: Tensor) -> float:
    """Return L1 error for nonempty floating audio shaped ``[B, 2, samples]``."""
    _audio("waveform L1", predicted, target)
    return float(F.l1_loss(predicted, target).item())


def lre_error_db(predicted: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    """Return left/right energy-ratio error for audio shaped ``[B, 2, samples]``."""
    _audio("LRE", predicted, target)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("LRE eps must be a positive finite value")

    def ratio_db(audio: Tensor) -> Tensor:
        left_energy = audio[:, 0, :].square().sum(dim=-1)
        right_energy = audio[:, 1, :].square().sum(dim=-1)
        return 10.0 * torch.log10((left_energy + eps) / (right_energy + eps))

    return float((ratio_db(predicted) - ratio_db(target)).abs().mean().item())


def rgb_l1(predicted: Tensor, target: Tensor) -> float:
    """Return L1 error for nonempty floating BHWC RGB tensors."""
    _rgb("RGB L1", predicted, target)
    return float(F.l1_loss(predicted, target).item())


def psnr(predicted: Tensor, target: Tensor) -> float:
    """Return PSNR for nonempty floating BHWC RGB tensors."""
    _rgb("PSNR", predicted, target)
    mse = F.mse_loss(predicted, target)
    if mse.item() == 0.0:
        return float("inf")
    return float((-10.0 * torch.log10(mse)).item())


def ssim(predicted: Tensor, target: Tensor) -> float:
    """Return SSIM for nonempty floating BHWC RGB tensors."""
    _rgb("SSIM", predicted, target)
    return float((1.0 - 2.0 * dssim(predicted, target)).item())


def log_spectral_distance(
    predicted: Tensor,
    target: Tensor,
    component: str,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    eps: float = 1e-7,
) -> float:
    """Return mono/diff LSD for nonempty floating audio shaped ``[B, 2, samples]``.

    Centered STFTs use constant padding, including when the input is shorter than
    ``n_fft``. Half-precision inputs are promoted to float32 for the STFT.
    """
    if component not in {"mono", "diff"}:
        raise ValueError("LSD component must be 'mono' or 'diff'")
    _audio("LSD", predicted, target)
    if n_fft <= 0 or hop_length <= 0 or win_length <= 0 or win_length > n_fft:
        raise ValueError(
            "LSD FFT and window lengths must be positive and win_length <= n_fft"
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("LSD eps must be a positive finite value")

    sign = 1.0 if component == "mono" else -1.0
    stft_dtype = torch.promote_types(predicted.dtype, target.dtype)
    if stft_dtype in {torch.float16, torch.bfloat16}:
        stft_dtype = torch.float32
    predicted_stft = predicted.to(stft_dtype)
    target_stft = target.to(stft_dtype)
    predicted_component = predicted_stft[:, 0, :] + sign * predicted_stft[:, 1, :]
    target_component = target_stft[:, 0, :] + sign * target_stft[:, 1, :]
    window = torch.hamming_window(
        win_length,
        device=predicted.device,
        dtype=stft_dtype,
    )

    def log_magnitude(audio: Tensor) -> Tensor:
        spectrum = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            pad_mode="constant",
            return_complex=True,
        )
        return spectrum.abs().clamp_min(eps).log()

    squared_difference = (
        log_magnitude(predicted_component) - log_magnitude(target_component)
    ).square()
    per_frame = squared_difference.mean(dim=-2).sqrt()
    return float(per_frame.mean().item())


def aggregate_metrics(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("cannot aggregate empty metric rows")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("metric rows must have consistent keys")

    result: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        scalar_values: list[float] = []
        for row in rows:
            value = row[key]
            if isinstance(value, Tensor):
                if value.ndim != 0:
                    raise ValueError(f"metric {key!r} values must be scalar")
                value = value.item()
            else:
                try:
                    array = np.asarray(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"metric {key!r} values must be numeric scalars") from error
                if array.ndim != 0:
                    raise ValueError(f"metric {key!r} values must be scalar")
            try:
                scalar_values.append(float(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"metric {key!r} values must be numeric scalars") from error
        values = np.asarray(scalar_values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"metric {key!r} values must be finite")
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "median": float(np.median(values)),
        }
    return result

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.models.audio_tokens import AudioSpectrogramHead, AudioSTFTTokenizer
from avgaussianv2.models.visual_tokens import PoseTokenEncoder


def _validate_token_tensor(name: str, value: Tensor, d_model: int) -> None:
    if value.ndim != 3 or value.shape[-1] != d_model:
        raise ValueError(f"{name} must have shape (B,N,{d_model})")


def _parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [parameter for module in modules for parameter in module.parameters()]


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        if not 0 <= cross_gate_init <= 1:
            raise ValueError("cross_gate_init must be in [0,1]")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        hidden_dim = self.d_model * int(ffn_multiplier)

        self.self_norm = nn.LayerNorm(self.d_model)
        self.self_attention = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.d_model)
        self.memory_norm = nn.LayerNorm(self.d_model)
        self.cross_attention = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.d_model),
        )

        self.self_gate = nn.Parameter(torch.zeros(self.d_model))
        self.cross_gate = nn.Parameter(
            torch.full((self.d_model,), float(cross_gate_init))
        )
        self.ffn_gate = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, audio_tokens: Tensor, memory_tokens: Tensor | None = None) -> Tensor:
        _validate_token_tensor("audio_tokens", audio_tokens, self.d_model)
        tokens = audio_tokens
        self_tokens = self.self_norm(tokens)
        self_update, _ = self.self_attention(
            self_tokens,
            self_tokens,
            self_tokens,
            need_weights=False,
        )
        tokens = tokens + self.self_gate.view(1, 1, -1) * self_update

        if memory_tokens is not None:
            _validate_token_tensor("memory_tokens", memory_tokens, self.d_model)
            if memory_tokens.shape[0] != tokens.shape[0]:
                raise ValueError("memory_tokens must match audio_tokens batch size")
            if memory_tokens.shape[1] > 0:
                cross_update, _ = self.cross_attention(
                    self.cross_norm(tokens),
                    self.memory_norm(memory_tokens),
                    self.memory_norm(memory_tokens),
                    need_weights=False,
                )
                tokens = tokens + self.cross_gate.view(1, 1, -1) * cross_update

        ffn_update = self.ffn(self.ffn_norm(tokens))
        return tokens + self.ffn_gate.view(1, 1, -1) * ffn_update

    def audio_parameters(self) -> list[nn.Parameter]:
        return [
            self.self_gate,
            self.ffn_gate,
            *_parameters([self.self_norm, self.self_attention, self.ffn_norm, self.ffn]),
        ]

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return [
            self.cross_gate,
            *_parameters([self.cross_norm, self.memory_norm, self.cross_attention]),
        ]


class AudioVisualTokenTransformer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.d_model = int(d_model)
        self.blocks = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    cross_gate_init=cross_gate_init,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, audio_tokens: Tensor, memory_tokens: Tensor | None = None) -> Tensor:
        _validate_token_tensor("audio_tokens", audio_tokens, self.d_model)
        tokens = audio_tokens
        for block in self.blocks:
            tokens = block(tokens, memory_tokens)
        return tokens

    def audio_parameters(self) -> list[nn.Parameter]:
        return [parameter for block in self.blocks for parameter in block.audio_parameters()]

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return [parameter for block in self.blocks for parameter in block.conditioning_parameters()]


class AudioVisualTokenAudioBackend(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        pose_dim: int = 12,
        pose_tokens: int = 2,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        freq_patch: int = 16,
        time_patch: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
        residual_scale: float = 0.05,
        loss_l1_weight: float = 1.0,
        loss_mse_weight: float = 0.1,
        loss_ild_weight: float = 0.1,
        loss_ipd_weight: float = 0.1,
        loss_lre_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.loss_l1_weight = float(loss_l1_weight)
        self.loss_mse_weight = float(loss_mse_weight)
        self.loss_ild_weight = float(loss_ild_weight)
        self.loss_ipd_weight = float(loss_ipd_weight)
        self.loss_lre_weight = float(loss_lre_weight)
        self.tokenizer = AudioSTFTTokenizer(
            d_model=d_model,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            freq_patch=freq_patch,
            time_patch=time_patch,
        )
        self.pose_encoder = PoseTokenEncoder(
            pose_dim=pose_dim,
            d_model=d_model,
            num_tokens=pose_tokens,
        )
        self.transformer = AudioVisualTokenTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            cross_gate_init=cross_gate_init,
        )
        self.head = AudioSpectrogramHead(
            d_model=d_model,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            freq_patch=freq_patch,
            time_patch=time_patch,
            residual_scale=residual_scale,
        )

    def _condition_tokens(self, condition: Tensor, batch_size: int) -> Tensor:
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        _validate_token_tensor("condition", condition, self.d_model)
        if condition.shape[0] != batch_size:
            raise ValueError("condition must match source_audio batch size")
        return condition

    def _memory_tokens(
        self,
        cam_pose: Tensor,
        audio_tokens: Tensor,
        condition: Tensor | None,
    ) -> Tensor:
        if cam_pose.ndim != 2 or cam_pose.shape[0] != audio_tokens.shape[0]:
            raise ValueError(
                "cam_pose must have shape (B,features) and match source_audio batch size"
            )
        pose_tokens = self.pose_encoder(
            cam_pose.to(device=audio_tokens.device, dtype=audio_tokens.dtype)
        )
        memory = [pose_tokens]
        if condition is not None:
            condition_tokens = self._condition_tokens(condition, audio_tokens.shape[0])
            memory.insert(
                0,
                condition_tokens.to(device=audio_tokens.device, dtype=audio_tokens.dtype),
            )
        return torch.cat(memory, dim=1)

    def render(
        self,
        cam_pose: Tensor,
        source_audio: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        batch = self.tokenizer(source_audio)
        memory_tokens = self._memory_tokens(cam_pose, batch.tokens, condition)
        tokens = self.transformer(batch.tokens, memory_tokens)
        return self.head(
            tokens,
            batch.grid_size,
            batch.source_stft,
            length=batch.original_samples,
        )

    def forward(
        self,
        cam_pose: Tensor,
        source_audio: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        return self.render(cam_pose, source_audio, condition=condition)

    def build_criterion(self) -> nn.Module:
        return WaveformReconstructionLoss(
            n_fft=self.tokenizer.n_fft,
            hop_length=self.tokenizer.hop_length,
            win_length=self.tokenizer.win_length,
            l1_weight=self.loss_l1_weight,
            mse_weight=self.loss_mse_weight,
            ild_weight=self.loss_ild_weight,
            ipd_weight=self.loss_ipd_weight,
            lre_weight=self.loss_lre_weight,
        )

    def acoustic_parameters(self) -> list[nn.Parameter]:
        return [
            *_parameters([self.tokenizer, self.head]),
            *self.transformer.audio_parameters(),
        ]

    def conditioning_parameters(self) -> list[nn.Parameter]:
        return [
            *_parameters([self.pose_encoder]),
            *self.transformer.conditioning_parameters(),
        ]

    def film_parameters(self) -> list[nn.Parameter]:
        return self.conditioning_parameters()

    def audio_unet_parameters(self) -> list[nn.Parameter]:
        return []


class WaveformReconstructionLoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        mse_weight: float = 0.1,
        ild_weight: float = 0.1,
        ipd_weight: float = 0.1,
        lre_weight: float = 0.1,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        weights = {
            "l1_weight": l1_weight,
            "mse_weight": mse_weight,
            "ild_weight": ild_weight,
            "ipd_weight": ipd_weight,
            "lre_weight": lre_weight,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if sum(float(value) for value in weights.values()) == 0:
            raise ValueError("at least one loss weight must be positive")
        if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
            raise ValueError("STFT parameters must be positive")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.l1_weight = float(l1_weight)
        self.mse_weight = float(mse_weight)
        self.ild_weight = float(ild_weight)
        self.ipd_weight = float(ipd_weight)
        self.lre_weight = float(lre_weight)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.eps = float(eps)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def _stft(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError("waveform must have shape (B,2,samples)")
        if waveform.shape[-1] < self.win_length:
            raise ValueError("waveform is shorter than win_length")
        batch, channels, samples = waveform.shape
        flat = waveform.reshape(batch * channels, samples)
        spectrum = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            return_complex=True,
        )
        return spectrum.reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])

    def _spatial_losses(self, predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
        predicted_stft = self._stft(predicted)
        target_stft = self._stft(target)
        pred_left, pred_right = predicted_stft[:, 0], predicted_stft[:, 1]
        target_left, target_right = target_stft[:, 0], target_stft[:, 1]

        pred_left_mag = pred_left.abs().clamp_min(self.eps)
        pred_right_mag = pred_right.abs().clamp_min(self.eps)
        target_left_mag = target_left.abs().clamp_min(self.eps)
        target_right_mag = target_right.abs().clamp_min(self.eps)

        pred_ild = torch.log(pred_left_mag) - torch.log(pred_right_mag)
        target_ild = torch.log(target_left_mag) - torch.log(target_right_mag)
        ild_loss = F.l1_loss(pred_ild, target_ild)

        pred_phase = (pred_left / pred_left_mag) * (pred_right / pred_right_mag).conj()
        target_phase = (target_left / target_left_mag) * (target_right / target_right_mag).conj()
        ipd_loss = F.mse_loss(pred_phase.real, target_phase.real) + F.mse_loss(
            pred_phase.imag,
            target_phase.imag,
        )

        pred_lre = self._left_right_energy_db(pred_left_mag, pred_right_mag)
        target_lre = self._left_right_energy_db(target_left_mag, target_right_mag)
        lre_loss = F.l1_loss(pred_lre, target_lre)
        return {"ild_loss": ild_loss, "ipd_loss": ipd_loss, "lre_loss": lre_loss}

    def _left_right_energy_db(self, left_mag: Tensor, right_mag: Tensor) -> Tensor:
        left_energy = left_mag.square().mean(dim=(-2, -1)).clamp_min(self.eps)
        right_energy = right_mag.square().mean(dim=(-2, -1)).clamp_min(self.eps)
        return 10.0 * (torch.log(left_energy) - torch.log(right_energy)) / torch.log(
            left_energy.new_tensor(10.0)
        )

    def forward(self, predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[1] != 2:
            raise ValueError("predicted and target must share shape (B,2,samples)")
        target = target.to(predicted)
        wave_l1 = F.l1_loss(predicted, target)
        wave_mse = F.mse_loss(predicted, target)
        spatial = self._spatial_losses(predicted, target)
        total = self.l1_weight * wave_l1 + self.mse_weight * wave_mse
        total = total + self.ild_weight * spatial["ild_loss"]
        total = total + self.ipd_weight * spatial["ipd_loss"]
        total = total + self.lre_weight * spatial["lre_loss"]
        return {"total_loss": total, "wave_l1": wave_l1, "wave_mse": wave_mse, **spatial}

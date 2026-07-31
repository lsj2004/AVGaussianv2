from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.models.p1_visual import VisualMemory
from avgaussianv2.models.positional import add_grid_position_encoding


def _stft(audio: Tensor, n_fft: int, hop_length: int, win_length: int) -> Tensor:
    batch, channels, samples = audio.shape
    window = torch.hamming_window(
        win_length,
        periodic=True,
        device=audio.device,
        dtype=audio.dtype,
    )
    spectrum = torch.stft(
        audio.reshape(batch * channels, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    return spectrum.reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])


class _GeometryBiasedCrossBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float,
        cross_gate_init: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        ratio = min(max(float(cross_gate_init) / 0.1, -0.999), 0.999)
        self.self_gate = nn.Parameter(torch.zeros(()))
        self.cross_gate = nn.Parameter(torch.tensor(math.atanh(ratio)))
        self.ffn_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _bounded_gate(raw: Tensor) -> Tensor:
        return 0.1 * torch.tanh(raw)

    def forward(
        self,
        query: Tensor,
        memory: VisualMemory,
        geometry_bias: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(query)
        update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        query = query + self._bounded_gate(self.self_gate) * update
        batch, query_count = query.shape[:2]
        key_count = memory.tokens.shape[1]
        expected_shape = (batch, self.num_heads, query_count, key_count)
        if geometry_bias.shape != expected_shape:
            raise ValueError(
                "geometry bias must have shape (B,heads,queries,keys)"
            )
        attention_mask = geometry_bias.reshape(
            batch * self.num_heads, query_count, key_count
        )
        padding_mask = torch.zeros(
            memory.key_padding_mask.shape,
            device=query.device,
            dtype=query.dtype,
        ).masked_fill(memory.key_padding_mask, float("-inf"))
        normalized_memory = self.memory_norm(memory.tokens)
        update, _ = self.cross_attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        query = query + self._bounded_gate(self.cross_gate) * update
        return query + self._bounded_gate(self.ffn_gate) * self.ffn(
            self.ffn_norm(query)
        )


class AlignedComplexCrossAttention(nn.Module):
    """P1 target-view AudioGS query with geometry-aware visual memory."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        freq_patch: int = 8,
        time_patch: int = 2,
        dropout: float = 0.0,
        cross_gate_init: float = 0.01,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        max_log_magnitude: float = 0.15,
        max_phase: float = 0.25,
        additive_scale: float = 0.01,
        geometry_rank: int = 16,
        geometry_bias_scale: float = 1.0,
    ) -> None:
        super().__init__()
        positive = {
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "freq_patch": freq_patch,
            "time_patch": time_patch,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "max_log_magnitude": max_log_magnitude,
            "max_phase": max_phase,
            "additive_scale": additive_scale,
            "geometry_rank": geometry_rank,
            "geometry_bias_scale": geometry_bias_scale,
        }
        if any(float(value) <= 0 for value in positive.values()):
            raise ValueError("P1 cross-attention dimensions and scales must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0 <= cross_gate_init <= 0.1:
            raise ValueError("cross_gate_init must be in [0,0.1]")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.freq_patch = int(freq_patch)
        self.time_patch = int(time_patch)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.max_log_magnitude = float(max_log_magnitude)
        self.max_phase = float(max_phase)
        self.additive_scale = float(additive_scale)
        self.geometry_rank = int(geometry_rank)
        self.geometry_bias_scale = float(geometry_bias_scale)
        self.query_embedding = nn.Conv2d(
            10,
            self.d_model,
            kernel_size=(self.freq_patch, self.time_patch),
            stride=(self.freq_patch, self.time_patch),
        )
        self.pose_projection = nn.Sequential(
            nn.Linear(12, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.query_geometry = nn.Linear(
            self.d_model,
            self.num_heads * self.geometry_rank,
        )
        self.query_coordinate_geometry = nn.Sequential(
            nn.Linear(6, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(
                self.d_model // 2,
                self.num_heads * self.geometry_rank,
            ),
        )
        self.key_geometry = nn.Sequential(
            nn.Linear(16, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(
                self.d_model // 2,
                self.num_heads * self.geometry_rank,
            ),
        )
        self.blocks = nn.ModuleList(
            [
                _GeometryBiasedCrossBlock(
                    self.d_model,
                    self.num_heads,
                    dropout=float(dropout),
                    cross_gate_init=float(cross_gate_init),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.decoder = nn.ConvTranspose2d(
            self.d_model,
            8,
            kernel_size=(self.freq_patch + 4, self.time_patch + 2),
            stride=(self.freq_patch, self.time_patch),
            padding=(2, 1),
        )
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.decoder.bias)

    def _query_features(
        self,
        native_spectrum: Tensor,
        mono_field: Tensor,
        diff_field: Tensor,
        source_magnitude: Tensor,
        distance_attenuation: Tensor,
    ) -> Tensor:
        frequency, frames = native_spectrum.shape[-2:]

        def resize(value: Tensor) -> Tensor:
            if value.shape[-2:] == (frequency, frames):
                return value
            return F.interpolate(
                value.unsqueeze(1),
                size=(frequency, frames),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        mono_field = resize(mono_field)
        diff_field = resize(diff_field)
        source_magnitude = resize(source_magnitude)
        distance_attenuation = resize(distance_attenuation)
        phase = torch.angle(native_spectrum)
        features = torch.stack(
            (
                torch.log1p(native_spectrum[:, 0].abs()),
                torch.log1p(native_spectrum[:, 1].abs()),
                torch.cos(phase[:, 0]),
                torch.sin(phase[:, 0]),
                torch.cos(phase[:, 1]),
                torch.sin(phase[:, 1]),
                torch.log1p(mono_field.abs()),
                torch.tanh(diff_field),
                torch.log1p(source_magnitude.clamp_min(0)),
                torch.log1p(distance_attenuation.clamp_min(0)),
            ),
            dim=1,
        )
        return torch.nan_to_num(features)

    def _geometry_attention_bias(
        self,
        memory: VisualMemory,
        cam_pose: Tensor,
        query: Tensor,
        grid_size: tuple[int, int],
    ) -> Tensor:
        if cam_pose.ndim != 2 or cam_pose.shape[-1] != 12:
            raise ValueError("P1 cam_pose must have shape (B,12)")
        listener = cam_pose[:, None, :3]
        relative = memory.world_positions - listener
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        direction_world = relative / distance.clamp_min(1e-6)
        world_to_head = cam_pose[:, 3:].reshape(-1, 3, 3)
        direction_head = torch.einsum(
            "bij,bkj->bki",
            world_to_head,
            direction_world,
        )
        normal_head = torch.einsum(
            "bij,bkj->bki",
            world_to_head,
            memory.world_normals,
        )
        facing = (normal_head * -direction_head).sum(dim=-1, keepdim=True)
        condition_offset = memory.camera_position[:, None] - listener
        condition_distance = torch.linalg.vector_norm(
            condition_offset,
            dim=-1,
            keepdim=True,
        )
        condition_direction_world = (
            condition_offset / condition_distance.clamp_min(1e-6)
        )
        condition_direction_head = torch.einsum(
            "bij,bkj->bki",
            world_to_head,
            condition_direction_world,
        )
        condition_view_direction = memory.world_positions - (
            memory.camera_position[:, None]
        )
        condition_view_direction = F.normalize(
            condition_view_direction,
            dim=-1,
            eps=1e-6,
        )
        view_parallax = (
            condition_view_direction * direction_world
        ).sum(dim=-1, keepdim=True)
        relative_rotation = torch.matmul(
            memory.camera_rotation,
            world_to_head.transpose(1, 2),
        )
        orientation_match = (
            relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            - 1.0
        ).div(2.0).clamp(-1, 1)[:, None]
        key_features = torch.cat(
            (
                torch.log1p(distance),
                distance.clamp_min(1e-3).reciprocal().clamp(max=100.0),
                direction_head,
                normal_head,
                facing,
                memory.confidence.unsqueeze(-1),
                torch.log1p(condition_distance).expand(
                    -1,
                    memory.tokens.shape[1],
                    -1,
                ),
                condition_direction_head.expand(
                    -1,
                    memory.tokens.shape[1],
                    -1,
                ),
                view_parallax,
                orientation_match.expand(
                    -1,
                    memory.tokens.shape[1],
                    -1,
                ),
            ),
            dim=-1,
        )
        batch, query_count = query.shape[:2]
        frequency_count, time_count = grid_size
        if frequency_count * time_count != query_count:
            raise ValueError("query count must match its time-frequency grid")
        frequency, time = torch.meshgrid(
            torch.linspace(
                0,
                1,
                frequency_count,
                device=query.device,
                dtype=query.dtype,
            ),
            torch.linspace(
                -1,
                1,
                time_count,
                device=query.device,
                dtype=query.dtype,
            ),
            indexing="ij",
        )
        log_frequency = torch.log1p(9.0 * frequency) / math.log(10.0)
        query_coordinates = torch.stack(
            (
                frequency,
                log_frequency,
                time,
                torch.sin(math.pi * frequency),
                torch.cos(math.pi * time),
                torch.sin(math.pi * time),
            ),
            dim=-1,
        ).reshape(1, query_count, 6).expand(batch, -1, -1)
        query_geometry = (
            self.query_geometry(query)
            + self.query_coordinate_geometry(query_coordinates)
        ).reshape(
            batch,
            query_count,
            self.num_heads,
            self.geometry_rank,
        )
        key_geometry = self.key_geometry(
            torch.nan_to_num(key_features)
        ).reshape(
            batch,
            memory.tokens.shape[1],
            self.num_heads,
            self.geometry_rank,
        )
        query_geometry = F.normalize(query_geometry, dim=-1, eps=1e-6)
        key_geometry = F.normalize(key_geometry, dim=-1, eps=1e-6)
        bias = torch.einsum(
            "bqhr,bkhr->bhqk",
            query_geometry,
            key_geometry,
        )
        return self.geometry_bias_scale * bias

    def forward(
        self,
        native_audio: Tensor,
        mono_field: Tensor,
        diff_field: Tensor,
        source_magnitude: Tensor,
        distance_attenuation: Tensor,
        cam_pose: Tensor,
        condition: VisualMemory,
    ) -> Tensor:
        if not isinstance(condition, VisualMemory):
            raise TypeError("P1 condition must be VisualMemory")
        native_spectrum = _stft(
            native_audio, self.n_fft, self.hop_length, self.win_length
        )
        features = self._query_features(
            native_spectrum,
            mono_field,
            diff_field,
            source_magnitude,
            distance_attenuation,
        )
        original_size = features.shape[-2:]
        pad_frequency = (-original_size[0]) % self.freq_patch
        pad_time = (-original_size[1]) % self.time_patch
        features = F.pad(features, (0, pad_time, 0, pad_frequency))
        encoded = self.query_embedding(features)
        grid_size = (int(encoded.shape[-2]), int(encoded.shape[-1]))
        tokens = encoded.flatten(2).transpose(1, 2)
        tokens = add_grid_position_encoding(tokens, grid_size)
        tokens = tokens + self.pose_projection(cam_pose).unsqueeze(1)
        geometry_bias = self._geometry_attention_bias(
            condition,
            cam_pose,
            tokens,
            grid_size,
        )
        for block in self.blocks:
            tokens = block(tokens, condition, geometry_bias)
        decoded_tokens = self.output_norm(tokens).transpose(1, 2).reshape(
            tokens.shape[0], self.d_model, *grid_size
        )
        residual = self.decoder(decoded_tokens)
        residual = F.interpolate(
            residual,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        delta_log_magnitude = self.max_log_magnitude * torch.tanh(residual[:, :2])
        delta_phase = self.max_phase * torch.tanh(residual[:, 2:4])
        phase_rotation = torch.polar(
            torch.ones_like(delta_phase),
            delta_phase,
        )
        corrected = (
            native_spectrum
            * torch.exp(delta_log_magnitude)
            * phase_rotation
        )
        native_scale = native_spectrum.abs().mean(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-6)
        additive = torch.complex(
            torch.tanh(residual[:, 4:6]),
            torch.tanh(residual[:, 6:8]),
        )
        corrected = corrected + self.additive_scale * native_scale * additive
        window = torch.hamming_window(
            self.win_length,
            periodic=True,
            device=native_audio.device,
            dtype=native_audio.dtype,
        )
        waveform = torch.istft(
            corrected.reshape(-1, *corrected.shape[-2:]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=native_audio.shape[-1],
        )
        return waveform.reshape_as(native_audio)

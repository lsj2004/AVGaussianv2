from __future__ import annotations

from dataclasses import dataclass
from weakref import ref

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avgaussianv2.models.positional import add_grid_position_encoding


ACOUSTIC_GAUSSIAN_SCHEMA = "audiogs_mono_diff_v1"


@dataclass(frozen=True)
class AcousticGaussianBatch:
    """Pose-conditioned AudioGS attributes on the native time-frequency grid."""

    features: Tensor
    grid_size: tuple[int, int]
    schema: str = ACOUSTIC_GAUSSIAN_SCHEMA


def _required_tensor(model: nn.Module, name: str) -> Tensor:
    value = getattr(model, name, None)
    if not isinstance(value, Tensor):
        raise TypeError(f"AudioGS model is missing tensor attribute {name}")
    return value


def _safe_value(model: nn.Module, method_name: str, attribute_name: str) -> Tensor:
    method = getattr(model, method_name, None)
    if callable(method):
        value = method()
        if not isinstance(value, Tensor):
            raise TypeError(f"AudioGS {method_name}() must return a tensor")
        return value
    return _required_tensor(model, attribute_name)


class AudioGSGaussianAttributeAdapter(nn.Module):
    """Expose the actual AudioGS mono/diff Gaussian fields through one schema.

    This adapter deliberately does not invent visual-3DGS attributes such as
    opacity or scale: the AudioGS mono/diff implementation does not own them.
    """

    schema = ACOUSTIC_GAUSSIAN_SCHEMA

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # The backend owns the AudioGS module. Keep only a weak reference here
        # so its parameters occur once in state_dict()/optimizer groups.
        object.__setattr__(self, "_model_ref", ref(model))
        self.freq_num = int(getattr(model, "freq_num", 0))
        self.time_num = int(getattr(model, "time_num", 0))
        self.n_points = int(getattr(model, "n_points", 0))
        self.max_norm = float(getattr(model, "max_norm", 1.0))
        if self.freq_num <= 0 or self.time_num <= 0:
            raise TypeError("AudioGS model must expose positive freq_num and time_num")
        if self.n_points != self.freq_num * self.time_num:
            raise ValueError("AudioGS point count must equal freq_num * time_num")
        if self.max_norm <= 0:
            raise ValueError("AudioGS max_norm must be positive")

        xyz = _required_tensor(model, "_xyz")
        rotation = _required_tensor(model, "_rotation")
        sh_mono = _required_tensor(model, "_sh_mono")
        sh_diff = _required_tensor(model, "_sh_diff")
        tf_coords = _required_tensor(model, "tf_coords")
        if xyz.shape != (self.n_points, 3):
            raise ValueError("AudioGS _xyz must have shape (N,3)")
        if rotation.shape != (self.n_points, 4):
            raise ValueError("AudioGS _rotation must have shape (N,4)")
        if sh_mono.ndim != 3 or sh_mono.shape[:2] != (self.n_points, 1):
            raise ValueError("AudioGS _sh_mono must have shape (N,1,C)")
        if sh_diff.shape != sh_mono.shape:
            raise ValueError("AudioGS _sh_diff must match _sh_mono")
        if tf_coords.shape != (self.n_points, 2):
            raise ValueError("AudioGS tf_coords must have shape (N,2)")
        if not callable(getattr(model, "compute_relative_positions", None)):
            raise TypeError("AudioGS model is missing compute_relative_positions()")
        if not callable(getattr(model, "eval_mono_diff_fields", None)):
            raise TypeError("AudioGS model is missing eval_mono_diff_fields()")
        self.num_sh_coeffs = int(sh_mono.shape[-1])
        # xyz, quaternion, mono/diff SH, TF coordinate, world/camera relative
        # vectors, point direction, log-distance, evaluated mono/diff response.
        self.feature_dim = 21 + 2 * self.num_sh_coeffs

    @property
    def model(self) -> nn.Module:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("AudioGS model was released before its attribute adapter")
        return model

    def _tf_coordinates(self, batch_size: int, dtype: torch.dtype) -> Tensor:
        tf = _required_tensor(self.model, "tf_coords").to(dtype=dtype)
        denominators = tf.new_tensor(
            [max(self.freq_num - 1, 1), max(self.time_num - 1, 1)]
        )
        tf = 2.0 * tf / denominators - 1.0
        return tf.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, cam_pose: Tensor) -> AcousticGaussianBatch:
        if cam_pose.ndim != 2 or cam_pose.shape[1] < 3:
            raise ValueError("cam_pose must have shape (B,features>=3)")
        batch_size = int(cam_pose.shape[0])
        dtype = cam_pose.dtype
        xyz = _safe_value(self.model, "_safe_xyz", "_xyz").to(dtype=dtype)
        if not bool(getattr(self.model, "normalize_world_coords", False)):
            xyz = xyz / self.max_norm
        rotation = _safe_value(
            self.model, "_safe_rotation_quaternions", "_rotation"
        ).to(dtype=dtype)
        rotation = F.normalize(rotation, dim=-1, eps=1e-8)
        sh_mono = _safe_value(self.model, "_safe_sh_mono", "_sh_mono")
        sh_diff = _safe_value(self.model, "_safe_sh_diff", "_sh_diff")
        sh_mono = sh_mono[:, 0].to(dtype=dtype)
        sh_diff = sh_diff[:, 0].to(dtype=dtype)

        rel_world, point_direction, rel_cam = self.model.compute_relative_positions(
            cam_pose
        )
        rel_for_sh = (
            rel_cam
            if bool(getattr(self.model, "use_cam_rotation", False))
            else rel_world
        )
        if bool(getattr(self.model, "use_cam_rotation", False)) and bool(
            getattr(self.model, "flip_cam_y_for_sh", False)
        ):
            rel_for_sh = rel_for_sh.clone()
            rel_for_sh[..., 1] = -rel_for_sh[..., 1]
        mono_response, diff_response = self.model.eval_mono_diff_fields(rel_for_sh)

        point_direction = point_direction.to(dtype=dtype) / self.max_norm
        direction_unit = F.normalize(point_direction, dim=-1, eps=1e-8)
        log_distance = torch.log1p(torch.linalg.vector_norm(point_direction, dim=-1))
        static = torch.cat(
            [
                xyz,
                rotation,
                sh_mono,
                sh_diff,
                self._tf_coordinates(batch_size, dtype)[0],
            ],
            dim=-1,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat(
            [
                static,
                rel_world.to(dtype=dtype),
                rel_cam.to(dtype=dtype),
                direction_unit,
                log_distance.unsqueeze(-1),
                mono_response.to(dtype=dtype).unsqueeze(-1),
                diff_response.to(dtype=dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        if features.shape != (batch_size, self.n_points, self.feature_dim):
            raise RuntimeError("AudioGS Gaussian feature schema produced an invalid shape")
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        return AcousticGaussianBatch(
            features=features,
            grid_size=(self.freq_num, self.time_num),
        )


class GaussianTokenEncoder(nn.Module):
    """Compress the native AudioGS TF field before cross-attention."""

    def __init__(
        self,
        feature_dim: int,
        *,
        d_model: int = 128,
        hidden_dim: int = 32,
        token_grid: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or d_model <= 0 or hidden_dim <= 0:
            raise ValueError("Gaussian token dimensions must be positive")
        if token_grid[0] <= 0 or token_grid[1] <= 0:
            raise ValueError("Gaussian token grid entries must be positive")
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.token_grid = (int(token_grid[0]), int(token_grid[1]))
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.point_projection = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
        )
        self.grid_projection = nn.Sequential(
            nn.Conv2d(int(hidden_dim), self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
        )

    def _deterministic_grid_pool(self, hidden: Tensor) -> Tensor:
        """Pool to the configured grid without CUDA adaptive-pool atomics."""
        rows, columns = self.token_grid
        height, width = int(hidden.shape[-2]), int(hidden.shape[-1])
        block_height = (height + rows - 1) // rows
        block_width = (width + columns - 1) // columns
        padded_height = block_height * rows
        padded_width = block_width * columns
        pad_height = padded_height - height
        pad_width = padded_width - width
        padded = F.pad(hidden, (0, pad_width, 0, pad_height))
        kernel = (block_height, block_width)
        area = block_height * block_width
        pooled_sum = F.avg_pool2d(
            padded,
            kernel_size=kernel,
            stride=kernel,
        ) * area
        # Correct boundary cells so constant-zero padding does not attenuate
        # valid Gaussian attributes. This mask carries no gradient.
        valid = hidden.new_ones((1, 1, height, width))
        valid = F.pad(valid, (0, pad_width, 0, pad_height))
        counts = F.avg_pool2d(
            valid,
            kernel_size=kernel,
            stride=kernel,
        ) * area
        return pooled_sum / counts.clamp_min(1.0)

    def forward(self, batch: AcousticGaussianBatch) -> Tensor:
        if batch.schema != ACOUSTIC_GAUSSIAN_SCHEMA:
            raise ValueError(f"unsupported acoustic Gaussian schema: {batch.schema}")
        features = batch.features
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Gaussian features must have shape (B,N,{self.feature_dim})"
            )
        freq_num, time_num = batch.grid_size
        if features.shape[1] != freq_num * time_num:
            raise ValueError("Gaussian feature length does not match its TF grid")
        hidden = self.point_projection(self.input_norm(features))
        hidden = hidden.transpose(1, 2).reshape(
            features.shape[0], -1, freq_num, time_num
        )
        pooled = self._deterministic_grid_pool(hidden)
        encoded = self.grid_projection(pooled)
        tokens = encoded.flatten(2).transpose(1, 2).contiguous()
        return add_grid_position_encoding(tokens, self.token_grid)


class PoseTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int = 12,
        d_model: int = 128,
        num_tokens: int = 2,
    ) -> None:
        super().__init__()
        if pose_dim <= 0 or d_model <= 0 or num_tokens <= 0:
            raise ValueError("pose token dimensions must be positive")
        self.pose_dim = int(pose_dim)
        self.d_model = int(d_model)
        self.num_tokens = int(num_tokens)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.pose_dim),
            nn.Linear(self.pose_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.num_tokens * self.d_model),
        )
        self.token_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.d_model)
        )

    def forward(self, cam_pose: Tensor) -> Tensor:
        if cam_pose.ndim != 2 or cam_pose.shape[1] != self.pose_dim:
            raise ValueError(f"cam_pose must have shape (B,{self.pose_dim})")
        tokens = self.encoder(cam_pose).reshape(
            cam_pose.shape[0], self.num_tokens, self.d_model
        )
        return tokens + self.token_embedding


__all__ = [
    "ACOUSTIC_GAUSSIAN_SCHEMA",
    "AcousticGaussianBatch",
    "AudioGSGaussianAttributeAdapter",
    "GaussianTokenEncoder",
    "PoseTokenEncoder",
]

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{path}.{key} is required")
    return mapping[key]


@dataclass(frozen=True)
class SceneConfig:
    scene_id: str
    fps: float
    train_cameras: tuple[str, ...]
    eval_cameras: tuple[str, ...]
    camera_mapping: dict[str, int]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SceneConfig":
        return cls(
            scene_id=str(_required(raw, "id", "scene")),
            fps=float(_required(raw, "fps", "scene")),
            train_cameras=tuple(str(value) for value in _required(raw, "train_cameras", "scene")),
            eval_cameras=tuple(str(value) for value in _required(raw, "eval_cameras", "scene")),
            camera_mapping={
                str(name): int(index)
                for name, index in _mapping(
                    _required(raw, "camera_mapping", "scene"), "scene.camera_mapping"
                ).items()
            },
        )

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("scene.fps must be positive")
        if not self.train_cameras:
            raise ValueError("scene.train_cameras must not be empty")
        if not self.eval_cameras:
            raise ValueError("scene.eval_cameras must not be empty")
        requested = (*self.train_cameras, *self.eval_cameras)
        missing = [camera for camera in requested if camera not in self.camera_mapping]
        if missing:
            raise ValueError(f"scene.camera_mapping is missing {missing[0]}")


@dataclass(frozen=True)
class PathConfig:
    visual_upstream_root: Path
    audio_upstream_root: Path
    visual_checkpoint: Path
    audio_checkpoint: Path
    manifest: Path
    visual_memmap: Path | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PathConfig":
        memmap = raw.get("visual_memmap")
        return cls(
            visual_upstream_root=Path(_required(raw, "visual_upstream_root", "paths")),
            audio_upstream_root=Path(_required(raw, "audio_upstream_root", "paths")),
            visual_checkpoint=Path(_required(raw, "visual_checkpoint", "paths")),
            audio_checkpoint=Path(_required(raw, "audio_checkpoint", "paths")),
            manifest=Path(_required(raw, "manifest", "paths")),
            visual_memmap=None if memmap is None else Path(memmap),
        )


@dataclass(frozen=True)
class ModelConfig:
    audio_backend: str = "audiogs"
    embedding_dim: int = 128
    alpha_threshold: float = 1e-3
    audio_model_class: str = "Audio3DGS"
    audio_render_strategy: str = "native_residual"
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    audio_freq_patch: int = 16
    audio_time_patch: int = 4
    audio_transformer_layers: int = 4
    audio_transformer_heads: int = 4
    audio_dropout: float = 0.0
    audio_cross_gate_init: float = 0.01
    audio_residual_scale: float = 0.05
    audio_gaussian_token_rows: int = 16
    audio_gaussian_token_columns: int = 16
    audio_gaussian_token_hidden_dim: int = 32
    audio_pose_tokens: int = 2
    p1_transformer_layers: int = 2
    p1_transformer_heads: int = 4
    p1_freq_patch: int = 8
    p1_time_patch: int = 2
    p1_dropout: float = 0.0
    p1_cross_gate_init: float = 0.01
    p1_max_log_magnitude: float = 0.15
    p1_max_phase: float = 0.25
    p1_additive_scale: float = 0.01
    p1_geometry_rank: int = 16
    p1_geometry_bias_scale: float = 1.0
    p1_visual_scene_scale: float = 1.0
    p1_camera_contrast_weight: float = 0.5
    p1_camera_contrast_margin: float = 0.05
    sample_rate: int = 16_000
    condition_height: int = 64
    condition_width: int = 96

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        defaults = cls()
        return cls(
            audio_backend=str(raw.get("audio_backend", defaults.audio_backend)),
            embedding_dim=int(raw.get("embedding_dim", defaults.embedding_dim)),
            alpha_threshold=float(raw.get("alpha_threshold", defaults.alpha_threshold)),
            audio_model_class=str(raw.get("audio_model_class", defaults.audio_model_class)),
            audio_render_strategy=str(
                raw.get("audio_render_strategy", defaults.audio_render_strategy)
            ),
            n_fft=int(raw.get("n_fft", defaults.n_fft)),
            hop_length=int(raw.get("hop_length", defaults.hop_length)),
            win_length=int(raw.get("win_length", defaults.win_length)),
            audio_freq_patch=int(raw.get("audio_freq_patch", defaults.audio_freq_patch)),
            audio_time_patch=int(raw.get("audio_time_patch", defaults.audio_time_patch)),
            audio_transformer_layers=int(
                raw.get("audio_transformer_layers", defaults.audio_transformer_layers)
            ),
            audio_transformer_heads=int(
                raw.get("audio_transformer_heads", defaults.audio_transformer_heads)
            ),
            audio_dropout=float(raw.get("audio_dropout", defaults.audio_dropout)),
            audio_cross_gate_init=float(
                raw.get("audio_cross_gate_init", defaults.audio_cross_gate_init)
            ),
            audio_residual_scale=float(
                raw.get("audio_residual_scale", defaults.audio_residual_scale)
            ),
            audio_gaussian_token_rows=int(
                raw.get(
                    "audio_gaussian_token_rows",
                    defaults.audio_gaussian_token_rows,
                )
            ),
            audio_gaussian_token_columns=int(
                raw.get(
                    "audio_gaussian_token_columns",
                    defaults.audio_gaussian_token_columns,
                )
            ),
            audio_gaussian_token_hidden_dim=int(
                raw.get(
                    "audio_gaussian_token_hidden_dim",
                    defaults.audio_gaussian_token_hidden_dim,
                )
            ),
            audio_pose_tokens=int(
                raw.get("audio_pose_tokens", defaults.audio_pose_tokens)
            ),
            p1_transformer_layers=int(
                raw.get("p1_transformer_layers", defaults.p1_transformer_layers)
            ),
            p1_transformer_heads=int(
                raw.get("p1_transformer_heads", defaults.p1_transformer_heads)
            ),
            p1_freq_patch=int(raw.get("p1_freq_patch", defaults.p1_freq_patch)),
            p1_time_patch=int(raw.get("p1_time_patch", defaults.p1_time_patch)),
            p1_dropout=float(raw.get("p1_dropout", defaults.p1_dropout)),
            p1_cross_gate_init=float(
                raw.get("p1_cross_gate_init", defaults.p1_cross_gate_init)
            ),
            p1_max_log_magnitude=float(
                raw.get("p1_max_log_magnitude", defaults.p1_max_log_magnitude)
            ),
            p1_max_phase=float(raw.get("p1_max_phase", defaults.p1_max_phase)),
            p1_additive_scale=float(
                raw.get("p1_additive_scale", defaults.p1_additive_scale)
            ),
            p1_geometry_rank=int(
                raw.get("p1_geometry_rank", defaults.p1_geometry_rank)
            ),
            p1_geometry_bias_scale=float(
                raw.get(
                    "p1_geometry_bias_scale",
                    defaults.p1_geometry_bias_scale,
                )
            ),
            p1_visual_scene_scale=float(
                raw.get("p1_visual_scene_scale", defaults.p1_visual_scene_scale)
            ),
            p1_camera_contrast_weight=float(
                raw.get(
                    "p1_camera_contrast_weight",
                    defaults.p1_camera_contrast_weight,
                )
            ),
            p1_camera_contrast_margin=float(
                raw.get(
                    "p1_camera_contrast_margin",
                    defaults.p1_camera_contrast_margin,
                )
            ),
            sample_rate=int(raw.get("sample_rate", defaults.sample_rate)),
            condition_height=int(raw.get("condition_height", defaults.condition_height)),
            condition_width=int(raw.get("condition_width", defaults.condition_width)),
        )

    def validate(self) -> None:
        positive = {
            "embedding_dim": self.embedding_dim,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "sample_rate": self.sample_rate,
            "condition_height": self.condition_height,
            "condition_width": self.condition_width,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"model.{name} must be positive")
        if not 0 <= self.alpha_threshold <= 1:
            raise ValueError("model.alpha_threshold must be in [0, 1]")
        cross_backends = {
            "cross_attention_tokens",
            "cross_attention_masks",
            "query_dependent_p1",
        }
        if self.audio_backend not in {"audiogs", *cross_backends}:
            raise ValueError(
                "model.audio_backend must be audiogs, cross_attention_tokens, "
                "cross_attention_masks, or query_dependent_p1"
            )
        if self.audio_render_strategy not in {
            "native_residual",
            "plain_unet",
            "direct_conditioned_unet",
            "gated_native_residual",
        }:
            raise ValueError(
                "model.audio_render_strategy must be native_residual, plain_unet, "
                "direct_conditioned_unet, or gated_native_residual"
            )
        if (
            self.audio_backend in cross_backends
            and self.audio_render_strategy != "native_residual"
        ):
            raise ValueError(
                "cross-attention backends do not use AudioGS render strategies; "
                "model.audio_render_strategy must remain native_residual"
            )
        if (
            self.audio_render_strategy == "plain_unet"
            and self.audio_model_class != "Audio3DGSMonoDiffGSOnly"
        ):
            raise ValueError(
                "plain_unet requires "
                "model.audio_model_class=Audio3DGSMonoDiffGSOnly"
            )
        if self.audio_backend in cross_backends:
            if self.audio_model_class != "Audio3DGSMonoDiffGSOnly":
                raise ValueError(
                    "cross-attention backends require "
                    "model.audio_model_class=Audio3DGSMonoDiffGSOnly"
                )
        if self.audio_backend in {
            "cross_attention_tokens",
            "cross_attention_masks",
        }:
            transformer_positive = {
                "audio_freq_patch": self.audio_freq_patch,
                "audio_time_patch": self.audio_time_patch,
                "audio_transformer_layers": self.audio_transformer_layers,
                "audio_transformer_heads": self.audio_transformer_heads,
            }
            for name, value in transformer_positive.items():
                if value <= 0:
                    raise ValueError(f"model.{name} must be positive")
            if self.embedding_dim % self.audio_transformer_heads != 0:
                raise ValueError(
                    "model.embedding_dim must be divisible by audio_transformer_heads"
                )
            if self.audio_dropout < 0:
                raise ValueError("model.audio_dropout must be non-negative")
            if not 0 <= self.audio_cross_gate_init <= 1:
                raise ValueError("model.audio_cross_gate_init must be in [0,1]")
        if self.audio_backend == "cross_attention_tokens":
            gaussian_positive = {
                "audio_gaussian_token_rows": self.audio_gaussian_token_rows,
                "audio_gaussian_token_columns": self.audio_gaussian_token_columns,
                "audio_gaussian_token_hidden_dim": self.audio_gaussian_token_hidden_dim,
                "audio_pose_tokens": self.audio_pose_tokens,
                "audio_residual_scale": self.audio_residual_scale,
            }
            for name, value in gaussian_positive.items():
                if value <= 0:
                    raise ValueError(f"model.{name} must be positive")
        if self.audio_backend == "query_dependent_p1":
            p1_positive = {
                "p1_transformer_layers": self.p1_transformer_layers,
                "p1_transformer_heads": self.p1_transformer_heads,
                "p1_freq_patch": self.p1_freq_patch,
                "p1_time_patch": self.p1_time_patch,
                "p1_max_log_magnitude": self.p1_max_log_magnitude,
                "p1_max_phase": self.p1_max_phase,
                "p1_additive_scale": self.p1_additive_scale,
                "p1_geometry_rank": self.p1_geometry_rank,
                "p1_geometry_bias_scale": self.p1_geometry_bias_scale,
                "p1_visual_scene_scale": self.p1_visual_scene_scale,
            }
            for name, value in p1_positive.items():
                if value <= 0:
                    raise ValueError(f"model.{name} must be positive")
            if self.embedding_dim % self.p1_transformer_heads != 0:
                raise ValueError(
                    "model.embedding_dim must be divisible by p1_transformer_heads"
                )
            if self.p1_dropout < 0:
                raise ValueError("model.p1_dropout must be non-negative")
            if not 0 <= self.p1_cross_gate_init <= 0.1:
                raise ValueError("model.p1_cross_gate_init must be in [0,0.1]")
            if self.p1_camera_contrast_weight < 0:
                raise ValueError(
                    "model.p1_camera_contrast_weight must be non-negative"
                )
            if self.p1_camera_contrast_margin < 0:
                raise ValueError(
                    "model.p1_camera_contrast_margin must be non-negative"
                )


@dataclass(frozen=True)
class TrainConfig:
    crop_seconds: float = 0.5
    warmup_steps: int = 2
    joint_steps: int = 2
    seed: int = 0
    audio_lr: float = 1e-4
    visual_lr: float = 1e-5
    condition_lr: float = 1e-4
    lambda_audio: float = 1.0
    lambda_lre: float = 0.0
    lre_scale_db: float = 6.0
    lre_epsilon: float = 1e-8
    lre_smooth_l1_beta: float = 1.0
    lambda_rgb: float = 1.0
    lambda_dssim: float = 0.2
    lambda_visual_anchor: float = 1e-4
    gradient_probe_interval: int = 1
    max_zero_audio_visual_grad_steps: int = 3

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainConfig":
        defaults = cls()
        values = {
            name: raw.get(name, getattr(defaults, name))
            for name in defaults.__dataclass_fields__
        }
        return cls(
            crop_seconds=float(values["crop_seconds"]),
            warmup_steps=int(values["warmup_steps"]),
            joint_steps=int(values["joint_steps"]),
            seed=int(values["seed"]),
            audio_lr=float(values["audio_lr"]),
            visual_lr=float(values["visual_lr"]),
            condition_lr=float(values["condition_lr"]),
            lambda_audio=float(values["lambda_audio"]),
            lambda_lre=float(values["lambda_lre"]),
            lre_scale_db=float(values["lre_scale_db"]),
            lre_epsilon=float(values["lre_epsilon"]),
            lre_smooth_l1_beta=float(values["lre_smooth_l1_beta"]),
            lambda_rgb=float(values["lambda_rgb"]),
            lambda_dssim=float(values["lambda_dssim"]),
            lambda_visual_anchor=float(values["lambda_visual_anchor"]),
            gradient_probe_interval=int(values["gradient_probe_interval"]),
            max_zero_audio_visual_grad_steps=int(values["max_zero_audio_visual_grad_steps"]),
        )

    def validate(self) -> None:
        if self.crop_seconds <= 0:
            raise ValueError("train.crop_seconds must be positive")
        if self.warmup_steps < 0 or self.joint_steps < 0:
            raise ValueError("train stage steps must be nonnegative")
        for name in ("audio_lr", "visual_lr", "condition_lr"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"train.{name} must be finite and positive")
        for name in (
            "lambda_audio",
            "lambda_lre",
            "lambda_rgb",
            "lambda_dssim",
            "lambda_visual_anchor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"train.{name} must be finite and non-negative")
        for name in ("lre_scale_db", "lre_epsilon", "lre_smooth_l1_beta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"train.{name} must be finite and positive")
        if self.lambda_audio == 0 and self.lambda_rgb == 0:
            raise ValueError("train must enable at least one primary loss")
        if self.gradient_probe_interval <= 0:
            raise ValueError("train.gradient_probe_interval must be positive")
        if self.max_zero_audio_visual_grad_steps <= 0:
            raise ValueError("train.max_zero_audio_visual_grad_steps must be positive")


@dataclass(frozen=True)
class ProjectConfig:
    scene: SceneConfig
    paths: PathConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProjectConfig":
        scene = _mapping(_required(raw, "scene", "config"), "scene")
        paths = _mapping(_required(raw, "paths", "config"), "paths")
        model = _mapping(raw.get("model", {}), "model")
        train = _mapping(raw.get("train", {}), "train")
        return cls(
            scene=SceneConfig.from_dict(scene),
            paths=PathConfig.from_dict(paths),
            model=ModelConfig.from_dict(model),
            train=TrainConfig.from_dict(train),
        )

    def validate(self) -> None:
        self.scene.validate()
        self.model.validate()
        self.train.validate()


def load_project_config_bytes(
    data: bytes,
    *,
    base_dir: str | Path,
) -> ProjectConfig:
    """Parse config bytes, resolving relative paths against the config directory.

    This preserves file-based loading semantics when callers migrate from
    ``load_project_config(path)`` to a read-once byte snapshot.
    """
    raw = yaml.safe_load(data.decode("utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    if "paths" not in raw:
        raise ValueError("paths.audio_checkpoint is required")
    paths = _mapping(raw["paths"], "paths")
    if "audio_checkpoint" not in paths:
        raise ValueError("paths.audio_checkpoint is required")
    paths = dict(paths)
    root = Path(base_dir)
    for name in (
        "visual_upstream_root",
        "audio_upstream_root",
        "visual_checkpoint",
        "audio_checkpoint",
        "manifest",
        "visual_memmap",
    ):
        value = paths.get(name)
        if value is not None:
            candidate = Path(value)
            paths[name] = candidate if candidate.is_absolute() else root / candidate
    normalized = dict(raw)
    normalized["paths"] = paths
    config = ProjectConfig.from_dict(normalized)
    config.validate()
    return config


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    return load_project_config_bytes(
        config_path.read_bytes(), base_dir=config_path.parent
    )

from __future__ import annotations

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
    embedding_dim: int = 128
    alpha_threshold: float = 1e-3
    audio_model_class: str = "Audio3DGS"
    audio_render_strategy: str = "native_residual"
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    sample_rate: int = 16_000
    condition_height: int = 64
    condition_width: int = 96

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        defaults = cls()
        return cls(
            embedding_dim=int(raw.get("embedding_dim", defaults.embedding_dim)),
            alpha_threshold=float(raw.get("alpha_threshold", defaults.alpha_threshold)),
            audio_model_class=str(raw.get("audio_model_class", defaults.audio_model_class)),
            audio_render_strategy=str(
                raw.get("audio_render_strategy", defaults.audio_render_strategy)
            ),
            n_fft=int(raw.get("n_fft", defaults.n_fft)),
            hop_length=int(raw.get("hop_length", defaults.hop_length)),
            win_length=int(raw.get("win_length", defaults.win_length)),
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
        if self.audio_render_strategy not in {
            "native_residual",
            "direct_conditioned_unet",
            "gated_native_residual",
        }:
            raise ValueError(
                "model.audio_render_strategy must be native_residual, "
                "direct_conditioned_unet, or gated_native_residual"
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

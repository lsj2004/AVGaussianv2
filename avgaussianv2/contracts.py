from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class RGBDRender:
    rgb: Tensor
    depth: Tensor
    alpha: Tensor

    def __post_init__(self) -> None:
        if self.rgb.ndim != 4 or self.rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape (B,H,W,3)")
        spatial = self.rgb.shape[:3]
        if self.depth.shape != (*spatial, 1) or self.alpha.shape != (*spatial, 1):
            raise ValueError("rgb, depth, and alpha must have shared batch and image shape")

    @property
    def batch_size(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.rgb.shape[1]), int(self.rgb.shape[2])


@dataclass(frozen=True)
class AlignedAVSample:
    scene_id: str
    camera: str
    frame_index: int
    time_seconds: float
    visual_time: Tensor
    w2c: Tensor
    intrinsic: Tensor
    audio_cam_pose: Tensor
    source_audio: Tensor
    target_audio: Tensor
    target_rgb: Tensor
    image_size: tuple[int, int]


@dataclass(frozen=True)
class FusionOutput:
    rgbd: RGBDRender
    condition: Tensor
    predicted_audio: Tensor

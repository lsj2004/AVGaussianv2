"""Fusion-specific neural network components."""

from avgaussianv2.models.fusion import AVGaussianFusionV2
from avgaussianv2.models.rgbd import RGBDConditionEncoder, normalize_depth
from avgaussianv2.models.mask_cross_attention import (
    AudioFeatureMaskCrossAttention,
)

__all__ = [
    "AVGaussianFusionV2",
    "RGBDConditionEncoder",
    "AudioFeatureMaskCrossAttention",
    "normalize_depth",
]

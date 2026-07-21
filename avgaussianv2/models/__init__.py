"""Fusion-specific neural network components."""

from avgaussianv2.models.rgbd import RGBDConditionEncoder, normalize_depth

__all__ = ["RGBDConditionEncoder", "normalize_depth"]

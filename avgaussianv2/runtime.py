"""Public construction boundary for production AVGaussianFusionV2 runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from avgaussianv2.backends.audio_audiogs import AudioGSBackend
from avgaussianv2.backends.visual_ftgspp import FTGSVisualBackend
from avgaussianv2.config import ProjectConfig
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.data.aligned import AlignedAVDataset
from avgaussianv2.losses import AudioLoss
from avgaussianv2.models.fusion import AVGaussianFusionV2
from avgaussianv2.models.rgbd import RGBDConditionEncoder


@dataclass(frozen=True, init=False)
class TrainingBundle:
    model: nn.Module
    train_samples: Sequence[AlignedAVSample]
    eval_samples: Sequence[AlignedAVSample]
    audio_loss_fn: AudioLoss

    def __init__(
        self,
        model: nn.Module,
        train_samples: Sequence[AlignedAVSample] | None = None,
        eval_samples: Sequence[AlignedAVSample] = (),
        audio_loss_fn: AudioLoss | None = None,
        *,
        samples: Sequence[AlignedAVSample] | None = None,
    ) -> None:
        """Create a bundle, accepting the old ``samples=`` seam during migration."""
        if audio_loss_fn is None and callable(eval_samples):
            # Original positional form: TrainingBundle(model, samples, loss).
            audio_loss_fn = eval_samples
            eval_samples = ()
        if samples is not None:
            if train_samples is not None:
                raise TypeError("pass train_samples or samples, not both")
            train_samples = samples
        if train_samples is None:
            raise TypeError("train_samples is required")
        if audio_loss_fn is None:
            raise TypeError("audio_loss_fn is required")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "train_samples", train_samples)
        object.__setattr__(self, "eval_samples", eval_samples)
        object.__setattr__(self, "audio_loss_fn", audio_loss_fn)

    @property
    def samples(self) -> Sequence[AlignedAVSample]:
        """Compatibility alias for the original training CLI injection API."""
        return self.train_samples


def build_runtime(config: ProjectConfig, device: torch.device) -> TrainingBundle:
    """Build the model, aligned train/eval datasets, and audio criterion once."""
    config.validate()
    resolved_device = torch.device(device)
    visual = FTGSVisualBackend.load(
        config.paths.visual_checkpoint,
        config.paths.visual_upstream_root,
    )
    audio = AudioGSBackend.load(
        config.paths.audio_checkpoint,
        embedding_dim=config.model.embedding_dim,
        upstream_root=config.paths.audio_upstream_root,
        model_class=config.model.audio_model_class,
    )
    model = AVGaussianFusionV2(
        visual=visual,
        condition_encoder=RGBDConditionEncoder(
            embedding_dim=config.model.embedding_dim,
            alpha_threshold=config.model.alpha_threshold,
        ),
        audio=audio,
    ).to(resolved_device)
    train_samples = AlignedAVDataset(config, split="train")
    eval_samples = AlignedAVDataset(config, split="eval")
    criterion = audio.build_criterion().to(resolved_device)
    return TrainingBundle(model, train_samples, eval_samples, criterion)


__all__ = ["TrainingBundle", "build_runtime"]

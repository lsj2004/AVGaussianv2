"""Public construction boundary for production AVGaussianFusionV2 runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from avgaussianv2.config import ProjectConfig
from avgaussianv2.contracts import AlignedAVSample
from avgaussianv2.losses import AudioLoss


FTGSVisualBackend = None
AudioGSBackend = None
AlignedAVDataset = None
AVGaussianFusionV2 = None
RGBDConditionEncoder = None


def _load_runtime_components() -> tuple[object, ...]:
    global FTGSVisualBackend, AudioGSBackend, AlignedAVDataset
    global AVGaussianFusionV2, RGBDConditionEncoder
    if FTGSVisualBackend is None:
        from avgaussianv2.backends.visual_ftgspp import (
            FTGSVisualBackend as visual_backend,
        )
        from avgaussianv2.backends.audio_audiogs import AudioGSBackend as audio_backend
        from avgaussianv2.data.aligned import AlignedAVDataset as aligned_dataset
        from avgaussianv2.models.fusion import AVGaussianFusionV2 as fusion_model
        from avgaussianv2.models.rgbd import RGBDConditionEncoder as condition_encoder

        FTGSVisualBackend = visual_backend
        AudioGSBackend = audio_backend
        AlignedAVDataset = aligned_dataset
        AVGaussianFusionV2 = fusion_model
        RGBDConditionEncoder = condition_encoder
    return (
        FTGSVisualBackend,
        AudioGSBackend,
        AlignedAVDataset,
        AVGaussianFusionV2,
        RGBDConditionEncoder,
    )


@dataclass(frozen=True)
class TrainingBundle:
    model: nn.Module
    train_samples: Sequence[AlignedAVSample]
    eval_samples: Sequence[AlignedAVSample] | None
    audio_loss_fn: AudioLoss


def build_runtime(
    config: ProjectConfig,
    device: torch.device,
    *,
    trusted_upstream_artifacts: bool = False,
    include_eval: bool = True,
) -> TrainingBundle:
    """Build a runtime only after explicitly trusting legacy upstream artifacts.

    FTGS and AudioGS loaders import Python from configured upstream roots and
    load legacy pickle checkpoints. Content hashes establish identity, not
    safety; callers must independently trust these artifacts.
    """
    if not trusted_upstream_artifacts:
        raise PermissionError(
            "refusing untrusted upstream artifacts: FTGS/AudioGS loaders import "
            "configured Python and use unsafe legacy pickle; pass "
            "trusted_upstream_artifacts=True only after independently trusting "
            "the upstream code and checkpoints"
        )
    (
        visual_backend,
        audio_backend,
        aligned_dataset,
        fusion_model,
        condition_encoder,
    ) = _load_runtime_components()
    config.validate()
    resolved_device = torch.device(device)
    visual = visual_backend.load(
        config.paths.visual_checkpoint,
        config.paths.visual_upstream_root,
    )
    if config.model.audio_backend == "cross_attention_tokens":
        from avgaussianv2.models.cross_attention_audio import (
            AudioVisualTokenAudioBackend,
        )
        from avgaussianv2.models.visual_tokens import RGBDTokenEncoder

        audio = AudioVisualTokenAudioBackend.load(
            config.paths.audio_checkpoint,
            upstream_root=config.paths.audio_upstream_root,
            model_class=config.model.audio_model_class,
            d_model=config.model.embedding_dim,
            num_layers=config.model.audio_transformer_layers,
            num_heads=config.model.audio_transformer_heads,
            n_fft=config.model.n_fft,
            hop_length=config.model.hop_length,
            win_length=config.model.win_length,
            freq_patch=config.model.audio_freq_patch,
            time_patch=config.model.audio_time_patch,
            dropout=config.model.audio_dropout,
            cross_gate_init=config.model.audio_cross_gate_init,
            residual_scale=config.model.audio_residual_scale,
            gaussian_token_rows=config.model.audio_gaussian_token_rows,
            gaussian_token_columns=config.model.audio_gaussian_token_columns,
            gaussian_token_hidden_dim=config.model.audio_gaussian_token_hidden_dim,
            pose_tokens=config.model.audio_pose_tokens,
        )
        condition = RGBDTokenEncoder(
            d_model=config.model.embedding_dim,
            alpha_threshold=config.model.alpha_threshold,
        )
    else:
        audio = audio_backend.load(
            config.paths.audio_checkpoint,
            embedding_dim=config.model.embedding_dim,
            upstream_root=config.paths.audio_upstream_root,
            model_class=config.model.audio_model_class,
            render_strategy=config.model.audio_render_strategy,
        )
        condition = condition_encoder(
            embedding_dim=config.model.embedding_dim,
            alpha_threshold=config.model.alpha_threshold,
        )
    model = fusion_model(visual=visual, condition_encoder=condition, audio=audio)
    criterion = audio.build_criterion()
    train_samples = aligned_dataset(config, split="train")
    eval_samples = (
        aligned_dataset(config, split="eval") if include_eval else None
    )
    model = model.to(resolved_device)
    criterion = criterion.to(resolved_device)
    return TrainingBundle(
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        audio_loss_fn=criterion,
    )


__all__ = ["TrainingBundle", "build_runtime"]

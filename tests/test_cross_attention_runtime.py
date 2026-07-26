from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import avgaussianv2.runtime as runtime_module
from avgaussianv2.config import (
    ModelConfig,
    PathConfig,
    ProjectConfig,
    SceneConfig,
    TrainConfig,
)
from avgaussianv2.models.cross_attention_audio import (
    AudioVisualTokenAudioBackend,
)
from avgaussianv2.models.fusion import AVGaussianFusionV2
from avgaussianv2.models.visual_tokens import RGBDTokenEncoder


def test_cross_attention_runtime_does_not_construct_audiogs_unet(monkeypatch) -> None:
    visual = nn.Linear(1, 1)
    monkeypatch.setattr(
        runtime_module,
        "FTGSVisualBackend",
        SimpleNamespace(load=lambda *_: visual),
    )
    monkeypatch.setattr(
        runtime_module,
        "AudioGSBackend",
        SimpleNamespace(
            load=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("AudioGS must not be constructed")
            )
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "AlignedAVDataset",
        lambda _config, split: (f"{split}-sample",),
    )
    monkeypatch.setattr(runtime_module, "AVGaussianFusionV2", AVGaussianFusionV2)
    monkeypatch.setattr(runtime_module, "RGBDConditionEncoder", nn.Identity)
    config = ProjectConfig(
        scene=SceneConfig(
            "scene",
            30.0,
            ("cam00",),
            ("cam38",),
            {"cam00": 0, "cam38": 38},
        ),
        paths=PathConfig(
            Path("/visual"),
            Path("/audio"),
            Path("/visual.pt"),
            Path("/audio.pt"),
            Path("/manifest.json"),
        ),
        model=ModelConfig(
            audio_backend="cross_attention_tokens",
            embedding_dim=32,
            n_fft=32,
            hop_length=8,
            win_length=16,
            audio_freq_patch=4,
            audio_time_patch=2,
            audio_transformer_layers=1,
            audio_transformer_heads=4,
            audio_pose_tokens=1,
        ),
        train=TrainConfig(),
    )

    bundle = runtime_module.build_runtime(
        config,
        torch.device("cpu"),
        trusted_upstream_artifacts=True,
    )

    assert isinstance(bundle.model.audio, AudioVisualTokenAudioBackend)
    assert isinstance(bundle.model.condition_encoder, RGBDTokenEncoder)
    assert bundle.train_samples == ("train-sample",)
    assert bundle.eval_samples == ("eval-sample",)

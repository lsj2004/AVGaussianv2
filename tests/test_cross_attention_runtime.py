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


def test_cross_attention_runtime_loads_native_audiogs_without_unet(monkeypatch) -> None:
    visual = nn.Linear(1, 1)
    native_model = nn.Module()
    native_model.renderer = nn.Linear(1, 1)
    native_model.gaussian = nn.Parameter(torch.tensor(1.0))
    native_model.forward = lambda _pose, source: source * native_model.gaussian
    backend = AudioVisualTokenAudioBackend(
        native_model,
        Path("/audio.pt"),
        d_model=32,
        num_layers=1,
        num_heads=4,
        n_fft=32,
        hop_length=8,
        win_length=16,
        freq_patch=4,
        time_patch=2,
    )
    monkeypatch.setattr(
        AudioVisualTokenAudioBackend,
        "load",
        classmethod(lambda cls, *_args, **_kwargs: backend),
    )
    monkeypatch.setattr(backend, "build_criterion", lambda: nn.MSELoss())
    monkeypatch.setattr(
        runtime_module,
        "FTGSVisualBackend",
        SimpleNamespace(load=lambda *_: visual),
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
            audio_model_class="Audio3DGSMonoDiffGSOnly",
        ),
        train=TrainConfig(),
    )

    bundle = runtime_module.build_runtime(
        config,
        torch.device("cpu"),
        trusted_upstream_artifacts=True,
    )

    assert isinstance(bundle.model.audio, AudioVisualTokenAudioBackend)
    assert isinstance(bundle.model.audio.model.renderer, nn.Identity)
    assert hasattr(bundle.model.audio.model, "gaussian")
    assert isinstance(bundle.model.condition_encoder, RGBDTokenEncoder)
    assert bundle.train_samples == ("train-sample",)
    assert bundle.eval_samples == ("eval-sample",)

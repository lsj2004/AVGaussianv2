from pathlib import Path
from types import SimpleNamespace

import pytest
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
from avgaussianv2.models.p1_visual import GeometricVisualTokenEncoder
from avgaussianv2.models.visual_tokens import RGBDTokenEncoder


class TinyRuntimeAudioGS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.renderer = nn.Linear(1, 1)
        self.gaussian_gain = nn.Parameter(torch.tensor(1.0))
        self.freq_num = 2
        self.time_num = 3
        self.n_points = 6
        self.max_norm = 1.0
        self.normalize_world_coords = False
        self.use_cam_rotation = False
        self.flip_cam_y_for_sh = False
        self._xyz = nn.Parameter(torch.zeros(6, 3))
        self._rotation = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(6, 1)
        )
        self._sh_mono = nn.Parameter(torch.zeros(6, 1, 4))
        self._sh_diff = nn.Parameter(torch.zeros(6, 1, 4))
        self.register_buffer(
            "tf_coords",
            torch.tensor(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
                dtype=torch.float32,
            ),
        )

    def forward(self, _pose, source):
        return source * self.gaussian_gain

    def compute_relative_positions(self, pose):
        relative = pose[:, None, :3] - self._xyz[None]
        return relative, -relative, relative

    def eval_mono_diff_fields(self, relative):
        zeros = relative[..., 0] * 0
        return zeros, zeros


class TinyMaskBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def build_criterion(self):
        return nn.MSELoss()

    def acoustic_parameters(self):
        return []

    def film_parameters(self):
        return [self.value]

    def audio_unet_parameters(self):
        return []


def test_cross_attention_runtime_loads_native_audiogs_without_unet(monkeypatch) -> None:
    visual = nn.Linear(1, 1)
    native_model = TinyRuntimeAudioGS()
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
    assert hasattr(bundle.model.audio.model, "_sh_mono")
    assert isinstance(bundle.model.condition_encoder, RGBDTokenEncoder)
    assert bundle.train_samples == ("train-sample",)
    assert bundle.eval_samples == ("eval-sample",)


def test_mask_cross_attention_runtime_selects_renderer_protocol(monkeypatch) -> None:
    visual = nn.Linear(1, 1)
    backend = TinyMaskBackend()
    captured = {}

    def load_audio(*_args, **kwargs):
        captured.update(kwargs)
        return backend

    monkeypatch.setattr(
        runtime_module,
        "FTGSVisualBackend",
        SimpleNamespace(load=lambda *_: visual),
    )
    monkeypatch.setattr(
        runtime_module,
        "AudioGSBackend",
        SimpleNamespace(load=load_audio),
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
            audio_backend="cross_attention_masks",
            embedding_dim=32,
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

    assert bundle.model.audio is backend
    assert isinstance(bundle.model.condition_encoder, RGBDTokenEncoder)
    assert captured["renderer_kind"] == "mask_cross_attention"
    assert captured["transformer_layers"] == 1
    assert captured["freq_patch"] == 4


@pytest.mark.parametrize(
    ("audio_backend", "contrast_mode"),
    [
        ("query_dependent_p1", "audio_total"),
        ("query_dependent_p1_spatial", "spatial"),
    ],
)
def test_query_dependent_p1_runtime_selects_geometry_protocol(
    monkeypatch,
    audio_backend: str,
    contrast_mode: str,
) -> None:
    visual = nn.Linear(1, 1)
    backend = TinyMaskBackend()
    captured = {}

    def load_audio(*_args, **kwargs):
        captured.update(kwargs)
        return backend

    monkeypatch.setattr(
        runtime_module,
        "FTGSVisualBackend",
        SimpleNamespace(load=lambda *_: visual),
    )
    monkeypatch.setattr(
        runtime_module,
        "AudioGSBackend",
        SimpleNamespace(load=load_audio),
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
            audio_backend=audio_backend,
            embedding_dim=32,
            p1_transformer_layers=1,
            p1_transformer_heads=4,
            p1_freq_patch=4,
            p1_time_patch=2,
            audio_model_class="Audio3DGSMonoDiffGSOnly",
        ),
        train=TrainConfig(),
    )

    bundle = runtime_module.build_runtime(
        config,
        torch.device("cpu"),
        trusted_upstream_artifacts=True,
    )

    assert bundle.model.audio is backend
    assert isinstance(
        bundle.model.condition_encoder,
        GeometricVisualTokenEncoder,
    )
    assert captured["renderer_kind"] == "p1_query_geometry"
    assert captured["transformer_layers"] == 1
    assert captured["freq_patch"] == 4
    assert bundle.model.camera_contrast_weight == 0.5
    assert bundle.model.camera_contrast_margin == 0.05
    assert bundle.model.camera_contrast_mode == contrast_mode
    assert bundle.model.camera_spatial_supervision_weight == 0.1

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from avgaussianv2.backends.audio_audiogs import (
    AudioCheckpointError,
    AudioGSBackend,
    _upstream_model_factory,
)
from avgaussianv2.contracts import AlignedAVSample, RGBDRender
from avgaussianv2.models.film_unet import FiLMConditionedAudioUNet
from avgaussianv2.models.fusion import AVGaussianFusionV2
from avgaussianv2.models.rgbd import RGBDConditionEncoder


def layer(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )


class TinyAudioUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = layer(3, 4)
        self.diff_enc1 = layer(1, 4)
        self.enc2 = layer(4, 8)
        self.enc3 = layer(8, 12)
        self.enc4 = layer(12, 16)
        self.maxpool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(16, 12, 2, stride=2)
        self.dec4 = layer(24, 12)
        self.upconv3 = nn.ConvTranspose2d(12, 8, 2, stride=2)
        self.dec3 = layer(16, 8)
        self.upconv2 = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.dec2 = layer(8, 4)
        self.upconv1 = nn.ConvTranspose2d(4, 4, 2, stride=2)
        self.dec1 = layer(8, 4)
        self.out_mono = nn.Conv2d(4, 1, 1)
        self.out_diff = nn.Conv2d(4, 1, 1)

    @staticmethod
    def resize(value, reference):
        if value.shape[-2:] == reference.shape[-2:]:
            return value
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, mono, diff):
        e1 = self.enc1(mono)
        diff_e1 = self.diff_enc1(diff)
        e2 = self.enc2(0.5 * (self.maxpool(e1) + self.maxpool(diff_e1)))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        d4 = self.dec4(torch.cat([self.resize(self.upconv4(e4), e3), e3], 1))
        d3 = self.dec3(torch.cat([self.resize(self.upconv3(d4), e2), e2], 1))
        d2 = self.dec2(torch.cat([self.resize(self.upconv2(d3), e1), e1], 1))
        d1 = self.dec1(torch.cat([self.resize(self.upconv1(d2), e1), e1], 1))
        return F.softplus(self.out_mono(d1)) + 0.1, torch.tanh(self.out_diff(d1))


class TinyAudioModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.acoustic_gain = nn.Parameter(torch.tensor(0.1))
        self.renderer = TinyAudioUNet()

    def forward(self, cam_pose, source_audio):
        del cam_pose
        content = source_audio.mean(dim=1, keepdim=True)
        mono = content.unsqueeze(2).expand(-1, 3, 16, -1)
        diff = content.unsqueeze(2).expand(-1, 1, 16, -1)
        mono_mask, _ = self.renderer(mono, diff)
        gain = mono_mask.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
        return source_audio * (gain + self.acoustic_gain)


class TinyGSOnlyAudioModel(TinyAudioModel):
    def forward(self, cam_pose, source_audio):
        del cam_pose
        return source_audio * self.acoustic_gain

    def inherited_unet_forward(self, cam_pose, source_audio):
        return super().forward(cam_pose, source_audio)


def test_audio_backend_rejects_checkpoint_without_model_state(tmp_path: Path) -> None:
    path = tmp_path / "bad.pth"
    torch.save({"optimizer_state_dict": {}}, path)

    with pytest.raises(AudioCheckpointError, match="model_state_dict"):
        AudioGSBackend.load(path, model_factory=lambda _: TinyAudioModel(), embedding_dim=8)


def test_upstream_model_import_skips_heavy_scene_package_init(tmp_path: Path) -> None:
    for relative in ("libs", "libs/models", "libs/datasets"):
        package = tmp_path / relative
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
    scene = tmp_path / "libs/datasets/scene"
    scene.mkdir()
    (scene / "__init__.py").write_text("raise RuntimeError('heavy optional dependencies')\n")
    (scene / "colmap_loader.py").write_text("VALUE = 7\n")
    (tmp_path / "libs/models/audio_3dgs.py").write_text(
        "from torch import nn\n"
        "from libs.datasets.scene.colmap_loader import VALUE\n"
        "class Audio3DGS(nn.Module):\n"
        "    def __init__(self, cfg):\n"
        "        raise RuntimeError('factory must use build_model')\n"
        "class Built(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.value = VALUE\n"
        "def build_model(cfg):\n"
        "    return Built()\n"
    )

    model = _upstream_model_factory(tmp_path, "Audio3DGS")(object())

    assert model.value == 7


def test_audio_backend_strict_load_reinitializes_only_static_stft_caches(tmp_path: Path) -> None:
    original = TinyAudioModel()
    original.register_buffer("static_source_mag", torch.ones(1, 257, 3))
    path = tmp_path / "audio-static.pth"
    torch.save({"model_state_dict": original.state_dict(), "cfg": {}}, path)

    def factory(_):
        model = TinyAudioModel()
        model.register_buffer("static_source_mag", torch.zeros(1, 257, 1))
        return model

    backend = AudioGSBackend.load(path, model_factory=factory, embedding_dim=8)

    assert backend.model.static_source_mag.shape == (1, 257, 1)
    assert torch.count_nonzero(backend.model.static_source_mag) == 0


def test_audio_backend_loads_original_weights_before_wrapping(tmp_path: Path) -> None:
    original = TinyAudioModel()
    original.acoustic_gain.data.fill_(0.75)
    path = tmp_path / "audio.pth"
    torch.save({"model_state_dict": original.state_dict(), "cfg": {"name": "tiny"}}, path)

    backend = AudioGSBackend.load(
        path,
        model_factory=lambda cfg: TinyAudioModel(),
        embedding_dim=8,
    )

    assert isinstance(backend.conditioned_renderer, FiLMConditionedAudioUNet)
    torch.testing.assert_close(backend.model.acoustic_gain, torch.tensor(0.75))


@pytest.mark.parametrize(
    ("model_file", "expected_module", "expected_class"),
    [
        ("audio_3dgs", "libs.criterions.Criterion_2", "Criterion"),
        (
            "audio_3dgs_mono_diff",
            "libs.criterions.MonoDiffMSECriterion",
            "MonoDiffMSECriterion",
        ),
    ],
)
def test_audio_backend_builds_the_same_criterion_as_upstream_trainer(
    monkeypatch, model_file, expected_module, expected_class
) -> None:
    config = SimpleNamespace(model=SimpleNamespace(file=model_file))
    model = TinyAudioModel()
    model.renderer = FiLMConditionedAudioUNet(model.renderer, embedding_dim=8)
    backend = AudioGSBackend(
        model,
        source_path=Path("audio.pth"),
        checkpoint_config=config,
        upstream_root=Path("/upstream"),
    )
    imported = []

    class FakeCriterion(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(**{expected_class: FakeCriterion})

    monkeypatch.setattr("avgaussianv2.backends.audio_audiogs.importlib.import_module", fake_import)

    criterion = backend.build_criterion()

    assert imported == [expected_module]
    assert criterion.cfg is config


def test_audio_backend_rejects_unimplemented_enhanced_criterion() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(file="audio_3dgs"),
        train=SimpleNamespace(enhanced_weight=0.5),
    )
    model = TinyAudioModel()
    model.renderer = FiLMConditionedAudioUNet(model.renderer, embedding_dim=8)
    backend = AudioGSBackend(
        model,
        source_path=Path("audio.pth"),
        checkpoint_config=config,
        upstream_root=Path("/upstream"),
    )

    with pytest.raises(AudioCheckpointError, match="enhanced_weight"):
        backend.build_criterion()


def test_audio_backend_applies_condition_only_inside_render_scope() -> None:
    model = TinyAudioModel()
    model.renderer = FiLMConditionedAudioUNet(model.renderer, embedding_dim=8)
    backend = AudioGSBackend(model, source_path=Path("audio.pth"))
    with torch.no_grad():
        backend.conditioned_renderer.film["e1"].to_scale_shift.weight.fill_(0.02)
    source = torch.randn(1, 2, 32)
    pose = torch.zeros(1, 12)

    plain = backend.render(pose, source)
    conditioned = backend.render(pose, source, condition=torch.ones(1, 8))

    assert not torch.allclose(conditioned, plain)
    assert backend.conditioned_renderer.active_condition is None


def test_gs_only_bridge_forces_inherited_unet_forward() -> None:
    model = TinyGSOnlyAudioModel()
    model.renderer = FiLMConditionedAudioUNet(model.renderer, embedding_dim=8)
    backend = AudioGSBackend(
        model,
        source_path=Path("audio.pth"),
        forward_override=TinyGSOnlyAudioModel.inherited_unet_forward,
    )
    source = torch.randn(1, 2, 32)
    pose = torch.zeros(1, 12)

    native_gs_only = model(pose, source)
    plain = backend.render(pose, source)
    zero_init_conditioned = backend.render(pose, source, condition=torch.ones(1, 8))
    torch.testing.assert_close(plain, native_gs_only)
    torch.testing.assert_close(zero_init_conditioned, native_gs_only)

    with torch.no_grad():
        backend.conditioned_renderer.film["e1"].to_scale_shift.weight.fill_(0.02)
    conditioned_unet = backend.render(pose, source, condition=torch.ones(1, 8))

    torch.testing.assert_close(backend.render(pose, source), native_gs_only)
    assert not torch.allclose(conditioned_unet, native_gs_only)


class FakeVisualBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gaussian_parameter = nn.Parameter(torch.tensor(0.2))

    def render_rgbd(self, time, w2c, intrinsic, image_size):
        del time, intrinsic
        batch = w2c.shape[0]
        height, width = image_size
        grid = torch.linspace(0.1, 1.0, height * width).reshape(1, height, width, 1)
        grid = grid.to(self.gaussian_parameter)
        value = torch.sigmoid(self.gaussian_parameter * grid)
        return RGBDRender(
            rgb=value.expand(batch, height, width, 3),
            depth=(2.0 + self.gaussian_parameter * grid).expand(batch, height, width, 1),
            alpha=torch.ones(batch, height, width, 1),
        )


class FakeAudioBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.acoustic_parameter = nn.Parameter(torch.tensor(0.3))
        self.film_parameter = nn.Parameter(torch.tensor(0.1))
        self.unet_parameter = nn.Parameter(torch.tensor(0.2))

    def render(self, cam_pose, source_audio, condition=None):
        del cam_pose
        condition_gain = 0.0 if condition is None else condition.mean(dim=1).view(-1, 1, 1)
        return source_audio * (1.0 + self.acoustic_parameter) + condition_gain

    def acoustic_parameters(self):
        return [self.acoustic_parameter]

    def film_parameters(self):
        return [self.film_parameter]

    def audio_unet_parameters(self):
        return [self.unet_parameter]


def sample() -> AlignedAVSample:
    return AlignedAVSample(
        scene_id="scene1_opera",
        camera="cam00",
        frame_index=3,
        time_seconds=0.1,
        visual_time=torch.tensor([[0.25]]),
        w2c=torch.eye(4).unsqueeze(0),
        intrinsic=torch.eye(3).unsqueeze(0),
        audio_cam_pose=torch.zeros(1, 12),
        source_audio=torch.randn(1, 2, 32),
        target_audio=torch.zeros(1, 2, 32),
        target_rgb=torch.zeros(1, 16, 16, 3),
        image_size=(16, 16),
    )


def test_fusion_audio_loss_reaches_visual_parameter() -> None:
    visual = FakeVisualBackend()
    model = AVGaussianFusionV2(visual, RGBDConditionEncoder(8), FakeAudioBackend())

    output = model(sample())
    output.predicted_audio.square().mean().backward()

    assert visual.gaussian_parameter.grad is not None
    assert visual.gaussian_parameter.grad.abs().sum() > 0


def test_fusion_condition_off_ablation_bypasses_rgbd_embedding() -> None:
    model = AVGaussianFusionV2(
        FakeVisualBackend(),
        RGBDConditionEncoder(8),
        FakeAudioBackend(),
    )
    aligned = sample()
    conditioned = model(aligned).predicted_audio

    model.condition_enabled = False
    unconditioned = model(aligned).predicted_audio

    assert not torch.allclose(conditioned, unconditioned)
    torch.testing.assert_close(
        unconditioned,
        aligned.source_audio * (1.0 + model.audio.acoustic_parameter),
    )


def test_fusion_freeze_policy_keeps_only_condition_and_film_trainable() -> None:
    model = AVGaussianFusionV2(
        FakeVisualBackend(),
        RGBDConditionEncoder(8),
        FakeAudioBackend(),
    )

    model.freeze_pretrained()

    assert not any(parameter.requires_grad for parameter in model.visual.parameters())
    assert all(parameter.requires_grad for parameter in model.condition_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.audio.film_parameters())
    assert not any(parameter.requires_grad for parameter in model.audio.acoustic_parameters())
    assert not any(parameter.requires_grad for parameter in model.audio.audio_unet_parameters())

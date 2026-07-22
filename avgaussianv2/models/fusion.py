from __future__ import annotations

from torch import nn

from avgaussianv2.contracts import AlignedAVSample, FusionOutput


def _set_requires_grad(parameters, enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


class AVGaussianFusionV2(nn.Module):
    def __init__(
        self,
        visual: nn.Module,
        condition_encoder: nn.Module,
        audio: nn.Module,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.condition_encoder = condition_encoder
        self.audio = audio
        self.condition_enabled = True

    def forward(self, sample: AlignedAVSample) -> FusionOutput:
        rgbd = self.visual.render_rgbd(
            sample.visual_time,
            sample.w2c,
            sample.intrinsic,
            sample.image_size,
        )
        condition = self.condition_encoder(rgbd)
        predicted_audio = self.audio.render(
            sample.audio_cam_pose,
            sample.source_audio,
            condition=condition if self.condition_enabled else None,
        )
        return FusionOutput(
            rgbd=rgbd,
            condition=condition,
            predicted_audio=predicted_audio,
        )

    def freeze_pretrained(self) -> None:
        _set_requires_grad(self.visual.parameters(), False)
        _set_requires_grad(self.audio.parameters(), False)
        _set_requires_grad(self.condition_encoder.parameters(), True)
        _set_requires_grad(self.audio.film_parameters(), True)

    def unfreeze_all(self) -> None:
        _set_requires_grad(self.parameters(), True)

    def named_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "visual": list(self.visual.parameters()),
            "acoustic": list(self.audio.acoustic_parameters()),
            "condition_encoder": list(self.condition_encoder.parameters()),
            "film": list(self.audio.film_parameters()),
            "audio_unet": list(self.audio.audio_unet_parameters()),
        }

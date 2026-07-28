from __future__ import annotations

from torch import Tensor, nn

from avgaussianv2.contracts import AlignedAVSample, FusionOutput, RGBDRender


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
        self.condition_content_permutation: tuple[int, ...] | None = None

    def render_rgbd(self, sample: AlignedAVSample) -> RGBDRender:
        return self.visual.render_rgbd(
            sample.visual_time,
            sample.w2c,
            sample.intrinsic,
            sample.image_size,
        )

    def forward_audio_only(self, sample: AlignedAVSample) -> Tensor:
        return self.audio.render(
            sample.audio_cam_pose,
            sample.source_audio,
            condition=None,
        )

    def _encode_condition(
        self,
        rgbd: RGBDRender,
        condition_sample: AlignedAVSample,
    ):
        if bool(
            getattr(self.condition_encoder, "requires_camera_geometry", False)
        ):
            return self.condition_encoder(
                rgbd,
                condition_sample.w2c,
                condition_sample.intrinsic,
            )
        if self.condition_content_permutation is None:
            condition = self.condition_encoder(rgbd)
        else:
            permuted_encoder = getattr(
                self.condition_encoder,
                "forward_with_content_permutation",
                None,
            )
            if permuted_encoder is None:
                raise TypeError(
                    "condition encoder does not support content permutation"
                )
            condition = permuted_encoder(
                rgbd,
                self.condition_content_permutation,
            )
        return condition

    def forward_with_condition_sample(
        self,
        sample: AlignedAVSample,
        condition_sample: AlignedAVSample,
    ) -> FusionOutput:
        rgbd = self.render_rgbd(condition_sample)
        condition = self._encode_condition(rgbd, condition_sample)
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

    def forward(self, sample: AlignedAVSample) -> FusionOutput:
        return self.forward_with_condition_sample(sample, sample)

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

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from avgaussianv2.config import ProjectConfig
from avgaussianv2.contracts import AlignedAVSample


_NUMPY_DTYPES = {
    "torch.uint8": np.uint8,
    "torch.float32": np.float32,
}


@dataclass(frozen=True)
class AlignedRecord:
    camera: str
    camera_index: int
    frame_index: int
    time_seconds: float
    target_audio_path: Path


def _open_memmaps(root: Path) -> dict[str, np.memmap]:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"visual memmap metadata does not exist: {meta_path}")
    meta = json.loads(meta_path.read_text())
    arrays = {}
    for name in ("rgb", "w2c", "intrinsic", "time"):
        if name not in meta:
            raise ValueError(f"visual memmap metadata is missing {name}")
        spec = meta[name]
        dtype_name = spec["dtype"]
        if dtype_name not in _NUMPY_DTYPES:
            raise ValueError(f"unsupported visual memmap dtype {dtype_name}")
        arrays[name] = np.memmap(
            root / f"{name}.memmap",
            mode="r",
            dtype=_NUMPY_DTYPES[dtype_name],
            shape=tuple(int(value) for value in spec["shape"]),
        )
    return arrays


def _read_audio_crop(path: Path, start: int, frames: int, expected_rate: int) -> Tensor:
    info = sf.info(str(path))
    if int(info.samplerate) != expected_rate:
        raise ValueError(
            f"audio sample rate mismatch for {path}: {info.samplerate} != {expected_rate}"
        )
    if int(info.channels) != 2:
        raise ValueError(f"audio must be binaural for {path}, got {info.channels} channels")
    with sf.SoundFile(str(path)) as handle:
        handle.seek(start)
        audio = handle.read(frames, dtype="float32", always_2d=True)
    if audio.shape != (frames, 2):
        raise ValueError(f"audio crop from {path} returned {audio.shape}, expected {(frames, 2)}")
    return torch.from_numpy(audio.T.copy()).unsqueeze(0)


def _audiogs_pose(w2c: Tensor) -> Tensor:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    center = -(rotation.transpose(0, 1) @ translation)
    return torch.cat([center, rotation.reshape(-1)]).unsqueeze(0)


def _resize_rgb_and_intrinsic(
    rgb: Tensor,
    intrinsic: Tensor,
    image_size: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    target_height, target_width = image_size
    source_height, source_width = int(rgb.shape[0]), int(rgb.shape[1])
    if (source_height, source_width) == image_size:
        return rgb, intrinsic
    resized = F.interpolate(
        rgb.permute(2, 0, 1).unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).contiguous()
    scaled = intrinsic.clone()
    scaled[0] *= target_width / source_width
    scaled[1] *= target_height / source_height
    scaled[2, 2] = 1.0
    return resized, scaled


class AlignedAVDataset(Dataset[AlignedAVSample]):
    def __init__(self, config: ProjectConfig, split: str) -> None:
        if split not in {"train", "eval"}:
            raise ValueError("split must be train or eval")
        self.config = config
        self.split = split
        manifest = json.loads(config.paths.manifest.read_text())
        if manifest.get("scene_id") != config.scene.scene_id:
            raise ValueError(
                f"scene ID mismatch: manifest={manifest.get('scene_id')} config={config.scene.scene_id}"
            )
        manifest_rate = int(manifest["audio"]["sample_rate"])
        if manifest_rate != config.model.sample_rate:
            raise ValueError(
                f"audio sample rate mismatch: manifest={manifest_rate} config={config.model.sample_rate}"
            )
        if config.paths.visual_memmap is None:
            raise ValueError("paths.visual_memmap is required for aligned RGB/time loading")
        self.arrays = _open_memmaps(config.paths.visual_memmap)
        self.sample_rate = manifest_rate
        self.crop_samples = int(round(config.train.crop_seconds * self.sample_rate))
        if self.crop_samples <= 0:
            raise ValueError("configured audio crop has zero samples")
        self.source_audio_path = Path(manifest["audio"]["source_path"])
        self.image_size = (
            int(config.model.condition_height),
            int(config.model.condition_width),
        )

        camera_names = config.scene.train_cameras if split == "train" else config.scene.eval_cameras
        manifest_cameras = manifest["cameras"]
        for camera in camera_names:
            if camera not in config.scene.camera_mapping:
                raise ValueError(f"{camera} has no camera mapping")
            if camera not in manifest_cameras:
                raise ValueError(f"{camera} is missing from scene manifest")

        frame_times = [float(value) for value in manifest["frame_times"]]
        audio_times = [float(value) for value in manifest.get("audio_times", frame_times)]
        if len(audio_times) != len(frame_times):
            raise ValueError("audio_times and frame_times must have equal length")
        tolerance = 0.5 / float(config.scene.fps)
        for frame_index, (frame_time, audio_time) in enumerate(zip(frame_times, audio_times)):
            if abs(frame_time - audio_time) > tolerance + 1e-9:
                raise ValueError(
                    f"timestamp mismatch at frame {frame_index}: {frame_time} vs {audio_time}"
                )

        source_frames = int(sf.info(str(self.source_audio_path)).frames)
        records = []
        for frame_index, time_seconds in enumerate(frame_times):
            center = int(round(time_seconds * self.sample_rate))
            start = center - self.crop_samples // 2
            end = start + self.crop_samples
            for camera in camera_names:
                target_path = Path(manifest_cameras[camera]["audio_path"])
                target_frames = int(sf.info(str(target_path)).frames)
                if start < 0 or end > min(source_frames, target_frames):
                    continue
                camera_index = int(config.scene.camera_mapping[camera])
                if camera_index < 0 or camera_index >= self.arrays["rgb"].shape[1]:
                    raise ValueError(f"{camera} camera mapping index {camera_index} is out of range")
                records.append(
                    AlignedRecord(
                        camera=camera,
                        camera_index=camera_index,
                        frame_index=frame_index,
                        time_seconds=time_seconds,
                        target_audio_path=target_path,
                    )
                )
        self.records = records
        if not self.records:
            raise ValueError(f"no valid aligned {split} samples for {config.scene.scene_id}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AlignedAVSample:
        record = self.records[index]
        center = int(round(record.time_seconds * self.sample_rate))
        start = center - self.crop_samples // 2
        source_audio = _read_audio_crop(
            self.source_audio_path,
            start,
            self.crop_samples,
            self.sample_rate,
        )
        target_audio = _read_audio_crop(
            record.target_audio_path,
            start,
            self.crop_samples,
            self.sample_rate,
        )
        frame = record.frame_index
        camera = record.camera_index
        rgb = torch.tensor(np.array(self.arrays["rgb"][frame, camera]), dtype=torch.float32) / 255.0
        w2c = torch.tensor(np.array(self.arrays["w2c"][frame, camera]), dtype=torch.float32)
        intrinsic = torch.tensor(
            np.array(self.arrays["intrinsic"][frame, camera]), dtype=torch.float32
        )
        rgb, intrinsic = _resize_rgb_and_intrinsic(rgb, intrinsic, self.image_size)
        visual_time = torch.tensor(
            np.array(self.arrays["time"][frame, camera]).reshape(1, 1),
            dtype=torch.float32,
        )
        return AlignedAVSample(
            scene_id=self.config.scene.scene_id,
            camera=record.camera,
            frame_index=frame,
            time_seconds=record.time_seconds,
            visual_time=visual_time,
            w2c=w2c.unsqueeze(0),
            intrinsic=intrinsic.unsqueeze(0),
            audio_cam_pose=_audiogs_pose(w2c),
            source_audio=source_audio,
            target_audio=target_audio,
            target_rgb=rgb.unsqueeze(0),
            image_size=self.image_size,
        )

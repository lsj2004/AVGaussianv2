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


@dataclass(frozen=True)
class AlignedAudioReference:
    scene_id: str
    camera: str
    frame_index: int
    time_seconds: float
    source_audio: Tensor
    target_audio: Tensor
    sample_rate: int


def _load_heldout_calibration(video_path: Path, camera: str) -> tuple[Tensor, Tensor]:
    calibration_path = video_path.parent / "cameras.npz"
    if not calibration_path.is_file() or calibration_path.is_symlink():
        raise ValueError(f"held-out calibration is missing or unsafe: {calibration_path}")
    with np.load(calibration_path, allow_pickle=False) as calibration:
        names = tuple(str(value) for value in calibration["names"])
        if camera not in names:
            raise ValueError(f"{camera} is missing from held-out calibration")
        index = names.index(camera)
        w2c = torch.tensor(calibration["w2c"][index], dtype=torch.float32)
        intrinsic = torch.tensor(calibration["intrinsics"][index], dtype=torch.float32)
    return w2c, intrinsic


def _read_heldout_frame(video_path: Path, frame_index: int) -> Tensor:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - production dependency probe
        raise RuntimeError("OpenCV is required for held-out RGB evaluation") from error
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open held-out video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(
                f"cannot decode held-out frame {frame_index}: {video_path}"
            )
    finally:
        capture.release()
    return torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1]))


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


def _shared_visual_time(time_array: np.memmap, frame_index: int) -> Tensor:
    """Read the frame's model time without substituting physical seconds."""
    values = np.asarray(time_array[frame_index], dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"visual model time is invalid at frame {frame_index}")
    if not np.allclose(values, values[0], rtol=1e-6, atol=1e-7):
        raise ValueError(
            f"visual model time differs across cameras at frame {frame_index}"
        )
    return torch.tensor([[float(values[0])]], dtype=torch.float32)


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
    def __init__(
        self,
        config: ProjectConfig,
        split: str,
        *,
        audio_only: bool = False,
    ) -> None:
        if split not in {"train", "eval"}:
            raise ValueError("split must be train or eval")
        self.config = config
        self.split = split
        self.audio_only = audio_only
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
        if audio_only:
            self.arrays = None
        else:
            if config.paths.visual_memmap is None:
                raise ValueError(
                    "paths.visual_memmap is required for aligned RGB/time loading"
                )
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
        self.heldout_sources: dict[str, tuple[Path, Tensor, Tensor]] = {}

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
                if self.arrays is not None and (
                    camera_index < 0
                    or camera_index >= self.arrays["rgb"].shape[1]
                ):
                    if split != "eval":
                        raise ValueError(
                            f"{camera} camera mapping index {camera_index} is out of range"
                        )
                    camera_record = manifest_cameras[camera]
                    video_path = Path(camera_record["video_path"])
                    if not video_path.is_file() or video_path.is_symlink():
                        raise ValueError(
                            f"{camera} held-out video is missing or unsafe: {video_path}"
                        )
                    if camera not in self.heldout_sources:
                        w2c, intrinsic = _load_heldout_calibration(video_path, camera)
                        self.heldout_sources[camera] = (video_path, w2c, intrinsic)
                    camera_index = -1
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

    def audio_reference(self, index: int) -> AlignedAudioReference:
        """Read the exact aligned audio crop without loading visual targets."""
        record = self.records[index]
        center = int(round(record.time_seconds * self.sample_rate))
        start = center - self.crop_samples // 2
        return AlignedAudioReference(
            scene_id=self.config.scene.scene_id,
            camera=record.camera,
            frame_index=record.frame_index,
            time_seconds=record.time_seconds,
            source_audio=_read_audio_crop(
                self.source_audio_path,
                start,
                self.crop_samples,
                self.sample_rate,
            ),
            target_audio=_read_audio_crop(
                record.target_audio_path,
                start,
                self.crop_samples,
                self.sample_rate,
            ),
            sample_rate=self.sample_rate,
        )

    def __getitem__(self, index: int) -> AlignedAVSample:
        if self.audio_only:
            raise RuntimeError(
                "audio-only aligned dataset exposes audio_reference(), not visual samples"
            )
        assert self.arrays is not None
        record = self.records[index]
        audio = self.audio_reference(index)
        frame = record.frame_index
        camera = record.camera_index
        if camera >= 0:
            rgb = (
                torch.tensor(
                    np.array(self.arrays["rgb"][frame, camera]), dtype=torch.float32
                )
                / 255.0
            )
            w2c = torch.tensor(
                np.array(self.arrays["w2c"][frame, camera]), dtype=torch.float32
            )
            intrinsic = torch.tensor(
                np.array(self.arrays["intrinsic"][frame, camera]), dtype=torch.float32
            )
            visual_time = torch.tensor(
                np.array(self.arrays["time"][frame, camera]).reshape(1, 1),
                dtype=torch.float32,
            )
        else:
            video_path, w2c, intrinsic = self.heldout_sources[record.camera]
            rgb = _read_heldout_frame(video_path, frame).to(torch.float32) / 255.0
            source_height, source_width = int(rgb.shape[0]), int(rgb.shape[1])
            calibration_width = float(intrinsic[0, 2]) * 2.0
            calibration_height = float(intrinsic[1, 2]) * 2.0
            if calibration_width <= 0.0 or calibration_height <= 0.0:
                raise ValueError("held-out intrinsic has invalid principal point")
            intrinsic = intrinsic.clone()
            intrinsic[0] *= source_width / calibration_width
            intrinsic[1] *= source_height / calibration_height
            intrinsic[2, 2] = 1.0
            # The strict memmap excludes held-out RGB, but its frame time is a
            # camera-independent FreeTimeGS++ coordinate and remains safe to use.
            visual_time = _shared_visual_time(self.arrays["time"], frame)
        rgb, intrinsic = _resize_rgb_and_intrinsic(rgb, intrinsic, self.image_size)
        return AlignedAVSample(
            scene_id=self.config.scene.scene_id,
            camera=record.camera,
            frame_index=frame,
            time_seconds=record.time_seconds,
            visual_time=visual_time,
            w2c=w2c.unsqueeze(0),
            intrinsic=intrinsic.unsqueeze(0),
            audio_cam_pose=_audiogs_pose(w2c),
            source_audio=audio.source_audio,
            target_audio=audio.target_audio,
            target_rgb=rgb.unsqueeze(0),
            image_size=self.image_size,
        )

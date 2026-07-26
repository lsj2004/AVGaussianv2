import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import avgaussianv2.data.aligned as aligned
from avgaussianv2.config import ModelConfig, PathConfig, ProjectConfig, SceneConfig, TrainConfig
from avgaussianv2.data.aligned import AlignedAVDataset


def write_memmap(root: Path) -> None:
    root.mkdir(parents=True)
    specs = {
        "rgb": ((5, 2, 8, 12, 3), np.uint8, "torch.uint8"),
        "w2c": ((5, 2, 4, 4), np.float32, "torch.float32"),
        "intrinsic": ((5, 2, 3, 3), np.float32, "torch.float32"),
        "time": ((5, 2, 1), np.float32, "torch.float32"),
    }
    meta = {}
    for name, (shape, dtype, torch_dtype) in specs.items():
        array = np.memmap(root / f"{name}.memmap", mode="w+", dtype=dtype, shape=shape)
        if name == "rgb":
            for frame in range(5):
                array[frame] = 10 + frame
        elif name == "w2c":
            array[:] = np.eye(4, dtype=np.float32)
            array[:, 0, 0, 3] = 1.0
            array[:, 1, 1, 3] = 2.0
        elif name == "intrinsic":
            array[:] = np.eye(3, dtype=np.float32)
        elif name == "time":
            times = np.array([-1.0, -0.8, -0.6, -0.4, -0.2], dtype=np.float32)
            array[:] = times[:, None, None]
        array.flush()
        meta[name] = {"shape": list(shape), "dtype": torch_dtype}
    (root / "meta.json").write_text(json.dumps(meta))


def make_scene(tmp_path: Path, *, mismatched_audio_times: bool = False) -> ProjectConfig:
    sample_rate = 1_000
    source = np.stack([np.linspace(-1, 1, 500), np.linspace(1, -1, 500)], axis=1)
    source_path = tmp_path / "near.wav"
    target0 = tmp_path / "cam00.wav"
    target10 = tmp_path / "cam10.wav"
    sf.write(source_path, source, sample_rate, subtype="FLOAT")
    sf.write(target0, 0.5 * source, sample_rate, subtype="FLOAT")
    sf.write(target10, 0.25 * source, sample_rate, subtype="FLOAT")
    memmap_root = tmp_path / "memmap"
    write_memmap(memmap_root)
    frame_times = [0.0, 0.1, 0.2, 0.3, 0.4]
    manifest = {
        "scene_id": "scene1_opera",
        "fps": 10.0,
        "num_frames": 5,
        "frame_times": frame_times,
        "audio_times": [0.0, 0.2, 0.3, 0.4, 0.5] if mismatched_audio_times else frame_times,
        "cameras": {
            "cam00": {"name": "cam00", "index": 0, "video_path": "cam00.mp4", "audio_path": str(target0)},
            "cam10": {"name": "cam10", "index": 1, "video_path": "cam10.mp4", "audio_path": str(target10)},
        },
        "train_cameras": ["cam00"],
        "eval_cameras": ["cam10"],
        "audio": {
            "sample_rate": sample_rate,
            "channels": 2,
            "crop_seconds": 0.02,
            "crop_samples": 20,
            "source_path": str(source_path),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return ProjectConfig(
        scene=SceneConfig(
            scene_id="scene1_opera",
            fps=10.0,
            train_cameras=("cam00",),
            eval_cameras=("cam10",),
            camera_mapping={"cam00": 0, "cam10": 1},
        ),
        paths=PathConfig(
            visual_upstream_root=tmp_path / "FreeTimeGSPlusPlus",
            audio_upstream_root=tmp_path / "audioGS-replay",
            visual_checkpoint=tmp_path / "visual.pt",
            audio_checkpoint=tmp_path / "audio.pth",
            manifest=manifest_path,
            visual_memmap=memmap_root,
        ),
        model=ModelConfig(
            embedding_dim=16,
            n_fft=16,
            hop_length=4,
            win_length=16,
            sample_rate=sample_rate,
            condition_height=8,
            condition_width=12,
        ),
        train=TrainConfig(crop_seconds=0.02),
    )


def test_dataset_keeps_physical_and_visual_time_separate(tmp_path: Path) -> None:
    dataset = AlignedAVDataset(make_scene(tmp_path), split="train")

    sample = dataset[0]

    assert sample.frame_index == 1
    assert sample.time_seconds == pytest.approx(0.1)
    assert sample.visual_time.item() == pytest.approx(-0.8)
    assert sample.source_audio.shape == (1, 2, 20)
    assert sample.target_audio.shape == (1, 2, 20)
    assert sample.target_rgb.shape == (1, 8, 12, 3)
    assert sample.image_size == (8, 12)


def test_dataset_builds_audiogs_pose_from_same_w2c(tmp_path: Path) -> None:
    sample = AlignedAVDataset(make_scene(tmp_path), split="train")[0]

    assert sample.audio_cam_pose.shape == (1, 12)
    np.testing.assert_allclose(sample.audio_cam_pose[0, :3].numpy(), [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(sample.audio_cam_pose[0, 3:].reshape(3, 3).numpy(), np.eye(3))


def test_training_dataset_excludes_windows_requiring_padding(tmp_path: Path) -> None:
    dataset = AlignedAVDataset(make_scene(tmp_path), split="train")

    assert [record.frame_index for record in dataset.records] == [1, 2, 3, 4]


def test_eval_reads_heldout_video_when_train_memmap_excludes_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_scene(tmp_path)
    config.scene.camera_mapping["cam10"] = 2
    video = tmp_path / "cam10.mp4"
    video.write_bytes(b"heldout")
    manifest = json.loads(config.paths.manifest.read_text())
    manifest["cameras"]["cam10"]["video_path"] = str(video)
    config.paths.manifest.write_text(json.dumps(manifest))
    intrinsic = np.array(
        [[24.0, 0.0, 12.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        aligned,
        "_load_heldout_calibration",
        lambda *_args: (
            aligned.torch.eye(4, dtype=aligned.torch.float32),
            aligned.torch.tensor(intrinsic),
        ),
    )
    monkeypatch.setattr(
        aligned,
        "_read_heldout_frame",
        lambda _path, frame: aligned.torch.full(
            (8, 12, 3), frame, dtype=aligned.torch.uint8
        ),
    )

    dataset = AlignedAVDataset(config, split="eval")
    sample = dataset[0]

    assert dataset.records[0].camera_index == -1
    assert sample.camera == "cam10"
    assert sample.frame_index == 1
    assert sample.visual_time.item() == pytest.approx(0.1)
    assert sample.target_rgb.mean().item() == pytest.approx(1.0 / 255.0)


def test_missing_camera_mapping_is_fatal(tmp_path: Path) -> None:
    config = make_scene(tmp_path)
    config.scene.camera_mapping.pop("cam10")

    with pytest.raises(ValueError, match="cam10.*camera mapping"):
        AlignedAVDataset(config, split="eval")


def test_timestamp_mismatch_greater_than_half_frame_is_fatal(tmp_path: Path) -> None:
    config = make_scene(tmp_path, mismatched_audio_times=True)

    with pytest.raises(ValueError, match="timestamp mismatch"):
        AlignedAVDataset(config, split="train")


def test_dataset_rejects_configured_sample_rate_mismatch(tmp_path: Path) -> None:
    config = make_scene(tmp_path)
    config = replace(config, model=replace(config.model, sample_rate=16_000))

    with pytest.raises(ValueError, match="sample rate"):
        AlignedAVDataset(config, split="train")

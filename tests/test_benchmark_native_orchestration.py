from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _run_script(
    name: str,
    *arguments: str,
    gpu: int = 0,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        AVGAUSSIANV2_PYTHON="/bin/true",
        CUDA_VISIBLE_DEVICES=str(gpu),
    )
    return subprocess.run(
        ["bash", str(REPOSITORY / "scripts" / name), *arguments],
        cwd=REPOSITORY,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("script", "training_marker"),
    [
        ("prepare_ftgspp_cam38_baselines.sh", "--from points --to train"),
        (
            "train_audiogs_cam38_baselines.sh",
            "train_audio_3dgs_replaynvas_viewpoint_per_scene.sh",
        ),
    ],
)
def test_native_preflight_only_checks_both_scenes_before_launching_nothing(
    script: str,
    training_marker: str,
) -> None:
    result = _run_script(script, "--preflight-only", gpu=7)

    assert "Preflight complete: scene1_opera" in result.stdout
    assert "Preflight complete: Scene7playing" in result.stdout
    assert training_marker not in result.stdout


def test_ftgspp_native_maps_physical_gpu_to_local_cuda_zero_and_names_scene() -> None:
    result = _run_script(
        "prepare_ftgspp_cam38_baselines.sh",
        "--scene",
        "Scene7playing",
        gpu=11,
    )

    assert "CUDA_VISIBLE_DEVICES=11" in result.stdout
    assert "--device cuda:0" in result.stdout
    assert "--scenes Scene7playing" in result.stdout
    assert "--scenes scene1_opera" not in result.stdout


def test_audiogs_native_maps_physical_gpu_to_local_cuda_zero_and_names_scene() -> None:
    result = _run_script(
        "train_audiogs_cam38_baselines.sh",
        "--scene",
        "scene1_opera",
        gpu=13,
    )

    assert "CUDA_VISIBLE_DEVICES=13" in result.stdout
    assert "device cuda:0" in result.stdout
    assert "SC-scene1-opera-cam38-shared" in result.stdout
    assert "SC-scene7-playing-cam38-shared" not in result.stdout


@pytest.mark.parametrize(
    ("script", "first_launch"),
    [
        ("prepare_ftgspp_cam38_baselines.sh", "--from extract --to prep"),
        (
            "train_audiogs_cam38_baselines.sh",
            "create_sampled_scene_audiogs_replay.py",
        ),
    ],
)
def test_all_scene_native_preflight_finishes_before_first_launch(
    script: str,
    first_launch: str,
) -> None:
    result = _run_script(script, gpu=3)

    scene1 = result.stdout.index("Preflight complete: scene1_opera")
    scene7 = result.stdout.index("Preflight complete: Scene7playing")
    launch = result.stdout.index(first_launch)
    assert scene1 < scene7 < launch

"""Machine-verifiable post-fix cam38 visual-time and RGBD gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from avgaussianv2.benchmark.artifacts import (
    atomic_write,
    canonical_json,
    repository_identity,
)
from avgaussianv2.benchmark.runtime import BenchmarkRuntime, build_production_runtime
from avgaussianv2.config import ProjectConfig, load_project_config
from avgaussianv2.contracts import AlignedAVSample, RGBDRender
from avgaussianv2.data.aligned import AlignedAVDataset
from avgaussianv2.data.tensor import move_sample


SCHEMA = "avgaussianv2.post-fix-visual-time-audit"
SEMANTICS = "shared_memmap_model_time_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_time_memmap(config: ProjectConfig) -> tuple[np.memmap, Path, Path]:
    root = config.paths.visual_memmap
    if root is None:
        raise ValueError("visual-time audit requires a visual memmap")
    meta_path = root / "meta.json"
    time_path = root / "time.memmap"
    metadata = json.loads(meta_path.read_text())
    specification = metadata.get("time")
    if (
        not isinstance(specification, dict)
        or specification.get("dtype") != "torch.float32"
        or not isinstance(specification.get("shape"), list)
        or len(specification["shape"]) != 3
    ):
        raise ValueError("visual-time memmap metadata is invalid")
    shape = tuple(int(value) for value in specification["shape"])
    if shape[1:] != (38, 1):
        raise ValueError("visual-time memmap must contain exactly cam00..cam37")
    return (
        np.memmap(time_path, mode="r", dtype=np.float32, shape=shape),
        meta_path,
        time_path,
    )


def _audit_render(render: RGBDRender) -> dict[str, float]:
    values = {"rgb": render.rgb, "depth": render.depth, "alpha": render.alpha}
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("post-fix held-out RGBD render contains nonfinite values")
    magnitudes = {
        name: float(value.detach().abs().sum().cpu())
        for name, value in values.items()
    }
    if magnitudes["rgb"] <= 0.0 or magnitudes["alpha"] <= 0.0:
        raise ValueError("post-fix held-out RGBD render is empty")
    return magnitudes


def audit_visual_time_configs(
    config_paths: Sequence[Path],
    *,
    output: Path,
    device: torch.device | str,
    trusted_upstream_artifacts: bool,
    dataset_builder: Callable[..., object] = AlignedAVDataset,
    runtime_builder: Callable[..., BenchmarkRuntime] = build_production_runtime,
    repository_identity_getter: Callable[[], dict[str, object]] = repository_identity,
    config_loader: Callable[[Path], ProjectConfig] = load_project_config,
) -> dict[str, object]:
    if not config_paths:
        raise ValueError("visual-time audit requires at least one config")
    destination = torch.device(device)
    scenes = []
    seen: set[str] = set()
    for raw_path in config_paths:
        config_path = Path(raw_path).resolve()
        config = config_loader(config_path)
        scene = config.scene.scene_id
        if scene in seen:
            raise ValueError(f"duplicate visual-time audit scene: {scene}")
        seen.add(scene)
        dataset = dataset_builder(config, split="eval", audio_only=False)
        if len(dataset) <= 0:
            raise ValueError(f"visual-time audit dataset is empty: {scene}")
        times, meta_path, time_path = _load_time_memmap(config)
        if times.shape[0] < max(record.frame_index for record in dataset.records) + 1:
            raise ValueError("visual-time memmap is shorter than the eval inventory")
        if not np.isfinite(times).all():
            raise ValueError("visual-time memmap contains nonfinite values")
        maximum_camera_spread = float(
            np.max(np.ptp(np.asarray(times[:, :, 0]), axis=1))
        )
        if maximum_camera_spread > 1.0e-6:
            raise ValueError("visual model time differs across training cameras")
        offsets = []
        for record in dataset.records:
            visual = float(times[record.frame_index, 0, 0])
            offset = float(record.time_seconds) - visual
            if not math.isfinite(offset) or abs(offset) <= 1.0e-6:
                raise ValueError("physical time and visual model time are conflated")
            offsets.append(offset)
        indices = tuple(dict.fromkeys((0, len(dataset) // 2, len(dataset) - 1)))
        samples: list[AlignedAVSample] = [dataset[index] for index in indices]
        for sample in samples:
            expected = float(times[sample.frame_index, 0, 0])
            if not math.isclose(
                float(sample.visual_time.item()), expected, abs_tol=1.0e-6
            ):
                raise ValueError("held-out sample did not use shared memmap model time")
            if (
                not torch.isfinite(sample.target_rgb).all()
                or float(sample.target_rgb.abs().sum()) <= 0.0
            ):
                raise ValueError("held-out RGB input is nonfinite or empty")
        runtime = runtime_builder(
            config_path=config_path,
            device=destination,
            trusted_upstream_artifacts=trusted_upstream_artifacts,
        )
        with runtime as active_runtime, torch.no_grad():
            render = active_runtime.model.render_rgbd(
                move_sample(samples[0], destination)
            )
            render_magnitudes = _audit_render(render)
        scenes.append(
            {
                "scene": scene,
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "sample_count": len(dataset),
                "sampled_indices": list(indices),
                "physical_minus_visual_time_min": min(offsets),
                "physical_minus_visual_time_max": max(offsets),
                "maximum_training_camera_time_spread": maximum_camera_spread,
                "visual_memmap_meta_sha256": _sha256(meta_path),
                "visual_time_memmap_sha256": _sha256(time_path),
                "render_abs_sum": render_magnitudes,
            }
        )
    result = {
        "schema": SCHEMA,
        "version": 1,
        "visual_time_semantics": SEMANTICS,
        "repository": repository_identity_getter(),
        "device": str(destination),
        "scenes": scenes,
    }
    atomic_write(Path(output), canonical_json(result))
    return result


__all__ = ["SCHEMA", "SEMANTICS", "audit_visual_time_configs"]

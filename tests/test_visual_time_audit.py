from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from avgaussianv2.benchmark.visual_time_audit import (
    SEMANTICS,
    audit_visual_time_configs,
)
from avgaussianv2.contracts import RGBDRender
from avgaussianv2.data.aligned import AlignedAVDataset
from tests.test_aligned_dataset import make_scene


class _Model:
    def render_rgbd(self, sample):
        del sample
        return RGBDRender(
            rgb=torch.ones(1, 8, 12, 3),
            depth=torch.ones(1, 8, 12, 1),
            alpha=torch.ones(1, 8, 12, 1),
        )


def test_visual_time_audit_binds_shared_time_and_finite_rgbd(tmp_path: Path) -> None:
    config = make_scene(tmp_path)
    meta_path = config.paths.visual_memmap / "meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata["time"]["shape"] = [5, 38, 1]
    meta_path.write_text(json.dumps(metadata))
    time = np.memmap(
        config.paths.visual_memmap / "time.memmap",
        mode="w+",
        dtype=np.float32,
        shape=(5, 38, 1),
    )
    time[:] = np.array([-1.0, -0.8, -0.6, -0.4, -0.2])[:, None, None]
    time.flush()
    config_path = tmp_path / "project.yaml"
    config_path.write_text("fixture\n")

    result = audit_visual_time_configs(
        [config_path],
        output=tmp_path / "audit.json",
        device="cpu",
        trusted_upstream_artifacts=True,
        dataset_builder=lambda *_args, **_kwargs: AlignedAVDataset(
            config, split="eval", audio_only=False
        ),
        runtime_builder=lambda **_kwargs: nullcontext(
            SimpleNamespace(model=_Model())
        ),
        repository_identity_getter=lambda: {
            "root": str(tmp_path),
            "commit": "1" * 40,
            "clean": True,
        },
        config_loader=lambda _path: config,
    )

    assert result["visual_time_semantics"] == SEMANTICS
    assert result["scenes"][0]["maximum_training_camera_time_spread"] == 0.0
    assert result["scenes"][0]["physical_minus_visual_time_min"] > 0.0
    assert result["scenes"][0]["render_abs_sum"]["alpha"] > 0.0
    assert (tmp_path / "audit.json").is_file()

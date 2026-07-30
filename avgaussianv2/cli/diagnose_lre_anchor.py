"""Evaluate fixed native-LRE projections without retraining or test-label fitting."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from avgaussianv2.backends.audio_audiogs import AudioGSBackend
from avgaussianv2.config import load_project_config
from avgaussianv2.experiment.evaluation import move_sample
from avgaussianv2.experiment.metrics import (
    aggregate_metrics,
    log_spectral_distance,
    lre_error_db,
    waveform_l1,
)
from avgaussianv2.runtime import build_runtime


def _metrics(criterion, predicted, target) -> dict[str, float]:
    losses = criterion(predicted, target)
    if not isinstance(losses, Mapping):
        raise TypeError("AudioGS criterion must return a mapping")
    return {
        "audio_total": float(losses["total_loss"].item()),
        "audio_mono": float(losses["mono_loss"].item()),
        "audio_diff": float(losses["diff_loss"].item()),
        "waveform_l1": waveform_l1(predicted, target),
        "mono_lsd": log_spectral_distance(predicted, target, "mono"),
        "diff_lsd": log_spectral_distance(predicted, target, "diff"),
        "lre_error_db": lre_error_db(predicted, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--conditioned",
        action="store_true",
        help="evaluate the FiLM-conditioned path; omit for plain U-Net",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.strengths):
        parser.error("all strengths must be in [0,1]")

    device = torch.device(args.device)
    config = load_project_config(args.resolved_config)
    bundle = build_runtime(
        config,
        device,
        trusted_upstream_artifacts=True,
        include_eval=True,
    )
    if bundle.eval_samples is None:
        raise RuntimeError("resolved benchmark config has no evaluation samples")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bundle.model.load_state_dict(payload["model"], strict=True)
    bundle.model.condition_enabled = bool(args.conditioned)
    bundle.model.eval()
    criterion = bundle.audio_loss_fn
    rows = {str(value): [] for value in args.strengths}

    with torch.no_grad():
        for original in bundle.eval_samples:
            sample = move_sample(original, device)
            if args.conditioned:
                rgbd = bundle.model.render_rgbd(sample)
                condition = bundle.model._encode_condition(rgbd, sample)
            else:
                condition = None
            previous = bundle.model.audio.native_lre_anchor_strength
            bundle.model.audio.native_lre_anchor_strength = 0.0
            try:
                rendered = bundle.model.audio.render(
                    sample.audio_cam_pose,
                    sample.source_audio,
                    condition=condition,
                )
            finally:
                bundle.model.audio.native_lre_anchor_strength = previous
            native = bundle.model.audio.model(
                sample.audio_cam_pose,
                sample.source_audio,
            )
            for strength in args.strengths:
                projected = AudioGSBackend.project_lre(
                    rendered,
                    native,
                    strength,
                )
                rows[str(strength)].append(
                    _metrics(criterion, projected, sample.target_audio)
                )

    result = {
        "schema": "avgaussianv2.native-lre-anchor-diagnostic",
        "version": 1,
        "scene_id": config.scene.scene_id,
        "conditioned": bool(args.conditioned),
        "checkpoint": str(args.checkpoint.resolve()),
        "selection_uses_test_targets": False,
        "strengths": {
            strength: aggregate_metrics(values)
            for strength, values in rows.items()
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

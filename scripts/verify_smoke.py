#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def verify(output: Path) -> None:
    required = (
        "checkpoint_latest.pt",
        "resolved_config.json",
        "loss_history.json",
        "gradient_norms.json",
        "selected_sample.json",
        "run_summary.json",
        "artifacts/sample_pred.wav",
        "artifacts/sample_condition_on.wav",
        "artifacts/sample_condition_off.wav",
        "artifacts/sample_rgb.ppm",
        "artifacts/sample_depth.pgm",
        "artifacts/metrics.json",
    )
    missing = [str(output / relative) for relative in required if not (output / relative).is_file()]
    if missing:
        raise SystemExit("missing smoke artifact: " + missing[0])
    history = json.loads((output / "loss_history.json").read_text())
    if not history or not all(math.isfinite(float(row["total"])) for row in history):
        raise SystemExit(f"non-finite or empty loss history: {output / 'loss_history.json'}")
    stages = {row.get("stage") for row in history}
    if stages != {"warmup", "joint"}:
        raise SystemExit(f"smoke must contain warmup and joint rows, got {sorted(stages)}")
    decoded = {}
    for name in ("sample_condition_on.wav", "sample_condition_off.wav"):
        audio, rate = sf.read(output / "artifacts" / name, always_2d=True, dtype="float32")
        if rate <= 0 or audio.shape[1] != 2 or not np.isfinite(audio).all():
            raise SystemExit(f"invalid binaural WAV: {output / 'artifacts' / name}")
        decoded[name] = audio
    if np.array_equal(decoded["sample_condition_on.wav"], decoded["sample_condition_off.wav"]):
        raise SystemExit("serialized condition-on and condition-off WAV files are identical")
    metrics = json.loads((output / "artifacts" / "metrics.json").read_text())
    delta = float(metrics.get("condition_delta_mean_abs", 0.0))
    if not math.isfinite(delta) or delta <= 0:
        raise SystemExit(f"condition-on output equals condition-off output: delta={delta}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.output)
    print(f"smoke verification passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

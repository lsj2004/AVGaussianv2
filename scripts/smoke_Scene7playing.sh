#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AVGAUSSIANV2_PYTHON:-/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python}"
CONFIG="$ROOT/configs/Scene7playing.yaml"
OUTPUT="${AVGAUSSIANV2_OUTPUT:-$ROOT/runs/smoke_Scene7playing}"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m avgaussianv2.cli.train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT" \
  --stage all \
  --warmup-steps 1 \
  --joint-steps 1 \
  --device cuda

"$PYTHON" "$ROOT/scripts/verify_smoke.py" "$OUTPUT"

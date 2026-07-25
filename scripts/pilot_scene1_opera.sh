#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AVGAUSSIANV2_PYTHON:-/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m avgaussianv2.cli.pilot \
  --config "${ROOT}/configs/scene1_opera.yaml" \
  --output-dir "${ROOT}/runs/pilot_scene1_opera" \
  --gpus 0,1,2 \
  --trust-upstream-artifacts

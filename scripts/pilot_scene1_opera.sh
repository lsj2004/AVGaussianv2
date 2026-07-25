#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m avgaussianv2.cli.pilot \
  --config "${ROOT}/configs/scene1_opera.yaml" \
  --output-dir "${ROOT}/runs/scene1_opera_pilot" \
  --gpus 0,1,2 \
  --trust-upstream-artifacts

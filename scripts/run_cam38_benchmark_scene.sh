#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AVGAUSSIANV2_PYTHON:-${ROOT}/.venv/bin/python}"
SCENE="${1:?usage: run_cam38_benchmark_scene.sh SCENE [extra args]}"
shift

exec "${PYTHON}" -m avgaussianv2.cli.benchmark_scene \
  --config "${ROOT}/configs/benchmark_cam38/${SCENE}.yaml" \
  --output-dir "${ROOT}/runs/cam38_benchmark/${SCENE}" \
  --python "${PYTHON}" \
  "$@"

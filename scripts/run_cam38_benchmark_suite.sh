#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AVGAUSSIANV2_PYTHON:-${ROOT}/.venv/bin/python}"

exec "${PYTHON}" -m avgaussianv2.cli.benchmark_suite \
  --repository "${ROOT}" \
  --output-dir "${ROOT}/runs/cam38_benchmark" \
  --python "${PYTHON}" \
  "$@"

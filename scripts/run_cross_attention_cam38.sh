#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <scene1_opera|Scene7playing> <prepare|diagnose|train|eval|report> [gpu] [system] [step]" >&2
  exit 2
fi

SCENE="$1"
ACTION="$2"
GPU="${3:-0}"
SYSTEM="${4:-cross_attention}"
STEP="${5:-30000}"

case "${SCENE}" in
  scene1_opera)
    EXPECTED_SAMPLES=130
    ;;
  Scene7playing)
    EXPECTED_SAMPLES=293
    ;;
  *)
    echo "unsupported scene: ${SCENE}" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${ROOT}/configs/benchmark_cam38/${SCENE}.yaml"
CROSS_CONFIG="${ROOT}/configs/benchmark_cam38/${SCENE}_cross_attention.yaml"
BASE_PROTOCOL="${ROOT}/runs/cam38_benchmark/${SCENE}/protocol"
FTGSPP_CONTRACT="${ROOT}/runs/cam38_strict/${SCENE}/ftgspp/native_contract"
AUDIOGS_CONTRACT="${ROOT}/runs/cam38_strict/${SCENE}/audiogs/native_contract"
OUTPUT="${ROOT}/runs/cross_attention_ablation/${SCENE}"
PROTOCOL="${OUTPUT}/protocol"
WORKER="${OUTPUT}/worker"
EVALUATIONS="${OUTPUT}/evaluations"
FILM_EVALUATIONS="${ROOT}/runs/cam38_benchmark/${SCENE}/evaluations/joint_conditioned"
PYTHON="${AVGAUSSIANV2_PYTHON:-/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "missing AVGaussianFusionv2 Python: ${PYTHON}" >&2
  exit 2
fi

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

case "${ACTION}" in
  prepare)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      -m avgaussianv2.cli.benchmark_cross_attention_prepare \
      --base-config "${BASE_CONFIG}" \
      --derived-config "${CROSS_CONFIG}" \
      --base-protocol-dir "${BASE_PROTOCOL}" \
      --output-dir "${OUTPUT}" \
      --device cuda:0 \
      --ftgspp-contract "${FTGSPP_CONTRACT}" \
      --audiogs-contract "${AUDIOGS_CONTRACT}" \
      --trust-upstream-artifacts
    ;;
  diagnose)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      -m avgaussianv2.cli.benchmark_cross_attention_diagnostic \
      --protocol-dir "${PROTOCOL}" \
      --output "${OUTPUT}/diagnostic.json" \
      --device cuda:0 \
      --steps "${DIAGNOSTIC_STEPS:-8}" \
      --trust-upstream-artifacts
    ;;
  train)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      -m avgaussianv2.cli.benchmark_cross_attention_worker \
      --protocol-dir "${PROTOCOL}" \
      --output-dir "${WORKER}" \
      --device cuda:0 \
      --trust-upstream-artifacts
    ;;
  eval)
    case "${SYSTEM}" in
      cross_attention|cross_attention_no_rgbd|cross_attention_shuffled_rgbd) ;;
      *)
        echo "unsupported causal system: ${SYSTEM}" >&2
        exit 2
        ;;
    esac
    case "${STEP}" in
      5000|10000|30000) ;;
      *)
        echo "step must be 5000, 10000, or 30000" >&2
        exit 2
        ;;
    esac
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      -m avgaussianv2.cli.benchmark_cross_attention_eval \
      --protocol-dir "${PROTOCOL}" \
      --worker-dir "${WORKER}" \
      --output-dir "${EVALUATIONS}/${SYSTEM}/step_$(printf '%06d' "${STEP}")" \
      --system "${SYSTEM}" \
      --step "${STEP}" \
      --device cuda:0 \
      --trust-upstream-artifacts
    ;;
  report)
    "${PYTHON}" -m avgaussianv2.cli.benchmark_cross_attention_report \
      --scene-id "${SCENE}" \
      --film-eval-root "${FILM_EVALUATIONS}" \
      --cross-eval-root "${EVALUATIONS}" \
      --protocol-dir "${PROTOCOL}" \
      --output-dir "${OUTPUT}/report" \
      --expected-samples "${EXPECTED_SAMPLES}"
    ;;
  *)
    echo "unsupported action: ${ACTION}" >&2
    exit 2
    ;;
esac

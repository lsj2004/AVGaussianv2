#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
SAMPLED_ROOT="/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf"
PYTHON="${AVGAUSSIANV2_PYTHON:-${ROOT}/.venv/bin/python}"
EXECUTE=0
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

# Frozen upstream TOML contract:
# eval_cameras = [37]  # train-only monitor, never the benchmark report
# train_cameras = { "start" = 0, "stop" = 38 }
# temporal_flow_cameras = { "start" = 0, "stop" = 38 }
# iterations = 30000
# batch_size = 1

run_command() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${EXECUTE}" -eq 1 ]]; then
    "$@"
  fi
}

prepare_scene() {
  local scene="$1"
  local config="${ROOT}/configs/benchmark_cam38/${scene}.yaml"
  local provenance="${ROOT}/configs/benchmark_cam38/provenance/${scene}.json"
  local template="${ROOT}/configs/upstream/ftgspp_cam38/${scene}.toml.in"
  local source="${SAMPLED_ROOT}/${scene}"
  local train_source="${ROOT}/runs/cam38_strict/${scene}/ftgspp/train_only_source"
  local run_root="${ROOT}/runs/cam38_strict/${scene}/ftgspp/native"
  local generated_config_dir="${ROOT}/runs/cam38_strict/${scene}/protocol/ftgspp_config"
  local generated_config="${generated_config_dir}/${scene}.toml"
  local marker_root="${ROOT}/runs/cam38_strict/${scene}/protocol/namespace_markers"
  local seed_root="${ROOT}/runs/cam38_strict/${scene}/protocol/seed_records"
  local seeded_wrapper="${ROOT}/avgaussianv2/cli/run_seeded_ftgspp.py"

  "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
    --config "${config}" --provenance "${provenance}"

  echo "Train-only source for ${scene}: cam00..cam37; cam38 RGB is not linked."
  if [[ "${EXECUTE}" -eq 1 ]]; then
    mkdir -p "${train_source}"
    for index in {0..37}; do
      camera="$(printf 'cam%02d' "${index}")"
      if [[ -L "${source}/${camera}.mp4" || ! -f "${source}/${camera}.mp4" ]]; then
        echo "refusing non-regular sampled input: ${source}/${camera}.mp4" >&2
        exit 1
      fi
      if [[ -e "${train_source}/${camera}.mp4" || -L "${train_source}/${camera}.mp4" ]]; then
        echo "refusing pre-existing train-only entry: ${train_source}/${camera}.mp4" >&2
        exit 1
      fi
      ln "${source}/${camera}.mp4" "${train_source}/${camera}.mp4"
    done
    "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
      --config "${config}" \
      --ftgspp-train-source "${train_source}" \
      --allowed-sampled-root "${source}" \
      --ftgspp-template "${template}" \
      --ftgspp-output "${generated_config}" \
      --repo-root "${ROOT}" \
      --ftgspp-root "${FTGSPP_ROOT}" \
      --prepare-ftgspp-namespaces \
      --ftgspp-run-root "${run_root}" \
      --ftgspp-marker-root "${marker_root}"
  else
    echo "pre-launch audit: --ftgspp-train-source ${train_source} --allowed-sampled-root ${source}"
    echo "safe Python render+audit: ${template} -> ${generated_config}"
  fi

  (
    cd "${FTGSPP_ROOT}"
    run_command env PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      .venv/bin/python "${seeded_wrapper}" \
      --seed 42 --record "${seed_root}/prep.json" \
      --script "${FTGSPP_ROOT}/run" -- dynerf \
      "${generated_config_dir}" \
      "${run_root}" \
      --scenes "${scene}" --from extract --to prep
    run_command env PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      .venv/bin/python "${seeded_wrapper}" \
      --seed 42 --record "${seed_root}/flow.json" \
      --module ftgspp.data.flow -- \
      "${generated_config}" --cameras 0-37
  )

  if [[ "${EXECUTE}" -eq 1 ]]; then
    "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
      --config "${config}" \
      --allowed-sampled-root "${source}" \
      --ftgspp-template "${template}" \
      --ftgspp-output "${generated_config}" \
      --repo-root "${ROOT}" \
      --ftgspp-root "${FTGSPP_ROOT}" \
      --audit-ftgspp-flow \
      --ftgspp-seed-record "${seed_root}/flow.json"
  else
    echo "pre-init audit: --audit-ftgspp-flow --ftgspp-seed-record ${seed_root}/flow.json"
  fi

  (
    cd "${FTGSPP_ROOT}"
    run_command env PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      .venv/bin/python "${seeded_wrapper}" \
      --seed 42 --record "${seed_root}/train.json" \
      --script "${FTGSPP_ROOT}/run" -- dynerf \
      "${generated_config_dir}" \
      "${run_root}" \
      --scenes "${scene}" --from points --to train
  )

  if [[ "${EXECUTE}" -eq 1 ]]; then
    "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
      --config "${config}" --ftgspp-seed-record "${seed_root}/train.json"
    "${PYTHON}" -m avgaussianv2.cli.native_contract \
      --model-kind ftgspp \
      --config "${config}" \
      --provenance "${provenance}" \
      --checkpoint "${run_root}/${scene}/00/gaussians.pt" \
      --upstream-root "${FTGSPP_ROOT}" \
      --rendered-config "${generated_config}" \
      --sampled-scene-root "${source}" \
      --train-log "${run_root}/${scene}/00/train.log" \
      --seed-record "${seed_root}/prep.json" \
      --seed-record "${seed_root}/flow.json" \
      --seed-record "${seed_root}/train.json" \
      --output "${ROOT}/runs/cam38_strict/${scene}/ftgspp/native_contract"
  else
    echo "post-success native contract: ${scene}/ftgspp/native_contract"
  fi
}

echo "Mode: $([[ ${EXECUTE} -eq 1 ]] && echo execute || echo dry-run)"
echo "FreeTimeGS++ seed 42 is bound by the deterministic wrapper for every executed stage; batch_size=1, iterations=30000."
prepare_scene scene1_opera
prepare_scene Scene7playing

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FTGSPP_ROOT="${FTGSPP_ROOT:-/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus}"
SAMPLED_ROOT="${SAMPLED_ROOT:-/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf}"
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
# eval_cameras = [38]
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
      --ftgspp-run-root "${run_root}"
  else
    echo "pre-launch audit: --ftgspp-train-source ${train_source} --allowed-sampled-root ${source}"
    echo "safe Python render+audit: ${template} -> ${generated_config}"
  fi

  (
    cd "${FTGSPP_ROOT}"
    run_command .venv/bin/python run dynerf \
      "${generated_config_dir}" \
      "${run_root}" \
      --scenes "${scene}" --from extract --to train
  )
}

echo "Mode: $([[ ${EXECUTE} -eq 1 ]] && echo execute || echo dry-run)"
echo "FreeTimeGS++ seed is bound by the upstream config; batch_size=1, iterations=30000."
prepare_scene scene1_opera
prepare_scene Scene7playing

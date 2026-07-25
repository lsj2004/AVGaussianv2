#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIOGS_ROOT="${AUDIOGS_ROOT:-/mnt/sda/lisujing/Dataset/audioGS-replay}"
SAMPLED_ROOT="${SAMPLED_ROOT:-/mnt/sda/lisujing/Dataset/Sampled_data}"
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

TEST_VIEWPOINT=39
NUM_VIEWPOINTS=39
CAMERAS="$(printf 'cam%02d,' {0..37})cam38"
export A3DGS_USE_METADATA=1
export A3DGS_INPUT_SOURCE=near
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/audiogs-cam38-numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/audiogs-cam38-matplotlib}"
# The Scene7 baseline must be one model over all three frame directories.
unset A3DGS_FRAME_ID
unset A3DGS_TRAIN_VP

run_command() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${EXECUTE}" -eq 1 ]]; then
    "$@"
  fi
}

audit_scene() {
  local scene="$1"
  "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
    --config "${ROOT}/configs/benchmark_cam38/${scene}.yaml" \
    --provenance "${ROOT}/configs/benchmark_cam38/provenance/${scene}.json"
}

prepare_and_train() {
  local config_name="$1"
  local source_scene="$2"
  local upstream_scene="$3"
  local max_clips="$4"
  local data_root="${ROOT}/runs/cam38_strict/${source_scene}/audiogs/dataset"

  audit_scene "${config_name}"
  run_command conda run -n avcloud python \
    "${AUDIOGS_ROOT}/scripts/create_sampled_scene_audiogs_replay.py" \
    --audio-root "${SAMPLED_ROOT}/v5_0630_audiogs_audio/${source_scene}" \
    --cameras-npz "${SAMPLED_ROOT}/v5_0630_dynerf/${source_scene}/cameras.npz" \
    --output-root "${data_root}" \
    --scene "${upstream_scene}" \
    --camera-names "${CAMERAS}" \
    --clip-sec 3 --hop-sec 3 --max-clips "${max_clips}"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
      --config "${ROOT}/configs/benchmark_cam38/${config_name}.yaml" \
      --audiogs-conversion "${data_root}/conversion_manifest.json" \
      --expected-clips "${max_clips}"
  else
    echo "pre-launch audit: --audiogs-conversion ${data_root}/conversion_manifest.json --expected-clips ${max_clips}"
  fi

  (
    cd "${AUDIOGS_ROOT}"
    run_command env \
      A3DGS_RESULT_ROOT="${ROOT}/runs/cam38_strict/${source_scene}/audiogs/native" \
      conda run -n avcloud bash \
      train_audio_3dgs_replaynvas_viewpoint_per_scene.sh \
      "${TEST_VIEWPOINT}" "${upstream_scene}" \
      dataset.data_root "${data_root}" \
      dataset.num_viewpoints "${NUM_VIEWPOINTS}" \
      dataset.pose_source fixed_rotation \
      dataset.fixed_rotation_mode lookat \
      model.xyz_anchor_radius 0.5 \
      train.lre_loss_weight 0.075 \
      train.diff_weight 0.5 \
      train.max_epoch 61 \
      train.batch_size 1
  )
}

echo "Mode: $([[ ${EXECUTE} -eq 1 ]] && echo execute || echo dry-run)"
echo "AudioGS seed=42, batch_size=1, epochs=61; viewpoint 39 (cam38) held out."
prepare_and_train scene1_opera scene1_opera SC-scene1-opera-cam38-shared 1
# No A3DGS_FRAME_ID is set: 3 clips x 38 training viewpoints x 61 = 6954 updates.
prepare_and_train Scene7playing Scene7playing SC-scene7-playing-cam38-shared 3

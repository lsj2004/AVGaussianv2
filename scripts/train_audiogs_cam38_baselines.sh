#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIOGS_ROOT="/mnt/sda/lisujing/Dataset/audioGS-replay"
SAMPLED_ROOT="/mnt/sda/lisujing/Dataset/Sampled_data"
PYTHON="${AVGAUSSIANV2_PYTHON:-${ROOT}/.venv/bin/python}"
EXECUTE=0
PREFLIGHT_ONLY=0
SCENE_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --scene) SCENE_FILTER="${2:-}"; shift 2 ;;
    *) echo "usage: $0 [--execute] [--preflight-only] [--scene scene1_opera|Scene7playing]" >&2; exit 2 ;;
  esac
done
if [[ "${EXECUTE}" -eq 1 && "${PREFLIGHT_ONLY}" -eq 1 ]]; then
  echo "--execute and --preflight-only are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${SCENE_FILTER}" && "${SCENE_FILTER}" != "scene1_opera" && "${SCENE_FILTER}" != "Scene7playing" ]]; then
  echo "unsupported scene: ${SCENE_FILTER}" >&2
  exit 2
fi
PHYSICAL_GPU="${CUDA_VISIBLE_DEVICES:-0}"
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one nonnegative GPU ID" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
LOCAL_DEVICE="cuda:0"
if [[ -n "${SCENE_FILTER}" ]]; then
  SCENES=("${SCENE_FILTER}")
else
  SCENES=("scene1_opera" "Scene7playing")
fi

TEST_VIEWPOINT=39
NUM_VIEWPOINTS=39
CAMERAS="$(printf 'cam%02d,' {0..37})cam38"
export A3DGS_USE_METADATA=1
export A3DGS_INPUT_SOURCE=near
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

preflight_scene() {
  local scene="$1"
  [[ -f "${AUDIOGS_ROOT}/train_audio_3dgs_replaynvas_viewpoint_per_scene.sh" ]] || {
    echo "missing AudioGS entrypoint" >&2
    exit 1
  }
  [[ -f "${AUDIOGS_ROOT}/scripts/create_sampled_scene_audiogs_replay.py" ]] || {
    echo "missing AudioGS dataset converter" >&2
    exit 1
  }
  command -v conda >/dev/null || {
    echo "conda is required for the AudioGS avcloud environment" >&2
    exit 1
  }
  "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
    --config "${ROOT}/configs/benchmark_cam38/${scene}.yaml" \
    --provenance "${ROOT}/configs/benchmark_cam38/provenance/${scene}.json"
  echo "Preflight complete: ${scene} (physical GPU ${PHYSICAL_GPU} -> ${LOCAL_DEVICE})"
}

prepare_and_train() {
  local config_name="$1"
  local source_scene="$2"
  local upstream_scene="$3"
  local max_clips="$4"
  local data_root="${ROOT}/runs/cam38_strict/${source_scene}/audiogs/dataset"
  local seed_record="${ROOT}/runs/cam38_strict/${source_scene}/protocol/audiogs_seed_record.json"

  run_command conda run -n avcloud python \
    "${AUDIOGS_ROOT}/scripts/create_sampled_scene_audiogs_replay.py" \
    --audio-root "${SAMPLED_ROOT}/v5_0630_audiogs_audio/${source_scene}" \
    --cameras-npz "${SAMPLED_ROOT}/v5_0630_dynerf/${source_scene}/cameras.npz" \
    --output-root "${data_root}" \
    --scene "${upstream_scene}" \
    --camera-names "${CAMERAS}" \
    --clip-sec 3 --hop-sec 3 --max-clips "${max_clips}" \
    --sample-rate 16000
  if [[ "${EXECUTE}" -eq 1 ]]; then
    "${PYTHON}" -m avgaussianv2.cli.audit_cam38_assets \
      --config "${ROOT}/configs/benchmark_cam38/${config_name}.yaml" \
      --audiogs-conversion "${data_root}/conversion_manifest.json" \
      --expected-clips "${max_clips}" \
      --expected-audio-root "${SAMPLED_ROOT}/v5_0630_audiogs_audio/${source_scene}" \
      --expected-cameras-npz "${SAMPLED_ROOT}/v5_0630_dynerf/${source_scene}/cameras.npz" \
      --expected-output-root "${data_root}"
  else
    echo "pre-launch audit: --audiogs-conversion ${data_root}/conversion_manifest.json --expected-clips ${max_clips}"
  fi
  run_command "${PYTHON}" -m avgaussianv2.cli.native_contract \
    --write-audiogs-seed-record \
    --scene-id "${source_scene}" \
    --upstream-scene "${upstream_scene}" \
    --output "${seed_record}"

  (
    cd "${AUDIOGS_ROOT}"
    run_command env \
      CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
      A3DGS_RESULT_ROOT="${ROOT}/runs/cam38_strict/${source_scene}/audiogs/native" \
      conda run -n avcloud bash \
      train_audio_3dgs_replaynvas_viewpoint_per_scene.sh \
      "${TEST_VIEWPOINT}" "${upstream_scene}" \
      dataset.data_root "${data_root}" \
      seed 42 \
      dataset.num_viewpoints "${NUM_VIEWPOINTS}" \
      device "${LOCAL_DEVICE}" \
      dataset.pose_source fixed_rotation \
      dataset.fixed_rotation_mode lookat \
      model.xyz_anchor_radius 0.5 \
      train.lre_loss_weight 0.075 \
      train.diff_weight 0.5 \
      train.max_epoch 61 \
      train.batch_size 1
  )
  if [[ "${EXECUTE}" -eq 1 ]]; then
    "${PYTHON}" -m avgaussianv2.cli.native_contract \
      --model-kind audiogs \
      --config "${ROOT}/configs/benchmark_cam38/${config_name}.yaml" \
      --provenance "${ROOT}/configs/benchmark_cam38/provenance/${config_name}.json" \
      --checkpoint "${ROOT}/runs/cam38_strict/${source_scene}/audiogs/native/replayNVAS/${upstream_scene}/viewpoint_39/checkpoint_latest.pth" \
      --upstream-root "${AUDIOGS_ROOT}" \
      --conversion-manifest "${data_root}/conversion_manifest.json" \
      --seed-record "${seed_record}" \
      --output "${ROOT}/runs/cam38_strict/${source_scene}/audiogs/native_contract"
  else
    echo "post-success native contract: ${source_scene}/audiogs/native_contract"
  fi
}

echo "Mode: $([[ ${EXECUTE} -eq 1 ]] && echo execute || ([[ ${PREFLIGHT_ONLY} -eq 1 ]] && echo preflight-only || echo dry-run))"
echo "AudioGS seed=42, batch_size=1, epochs=61; viewpoint 39 (cam38) held out."
for scene in "${SCENES[@]}"; do
  preflight_scene "${scene}"
done
if [[ "${PREFLIGHT_ONLY}" -eq 1 ]]; then
  exit 0
fi
for scene in "${SCENES[@]}"; do
  if [[ "${scene}" == "scene1_opera" ]]; then
    prepare_and_train scene1_opera scene1_opera SC-scene1-opera-cam38-shared 1
  else
    # No A3DGS_FRAME_ID is set: 3 clips x 38 training viewpoints x 61 = 6954 updates.
    prepare_and_train Scene7playing Scene7playing SC-scene7-playing-cam38-shared 3
  fi
done

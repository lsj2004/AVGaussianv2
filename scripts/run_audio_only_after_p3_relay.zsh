#!/usr/bin/env zsh

set -u

if (( $# != 7 )); then
  print -u2 -- "usage: $0 UPSTREAM_PID UPSTREAM_IDENTITY P3_RUN_TOKEN TOOL_ROOT TOOL_COMMIT FORMAL_ROOT RUN_TOKEN"
  exit 2
fi

upstream_pid=$1
upstream_identity=$2
p3_run_token=$3
tool_root=$4
tool_commit=$5
formal_root=$6
run_token=$7
frozen_root=/mnt/sda/lisujing/Dataset/AVGaussianFusionv2/.worktrees/p3-formal-frozen-98d
run_root=$formal_root/runs/lre_loss_ablation_visual_time_v3
result_root=$formal_root/results/lre_loss_ablation_visual_time_v3
manifest_root=$formal_root/configs/generated/audio_only_final_baseline
confirmation_manifest=$manifest_root/confirmation/manifest.json
robustness_manifest=$manifest_root/robustness/manifest.json
confirmation_sha256=5ff107105afdb699fe844d36f76a1a02b44bfb640686cd063069a2c624b1ca2b
robustness_sha256=503c711a1560411e2642cf44e25d0674c496d6769fc5b0212283a4a08c8012b0
p3_receipt=$result_root/p3_pipeline_receipt.json
gate_30k=$result_root/p3_30k_gate.json
multiseed_report=$result_root/p3_multiseed_report.json
receipt=$result_root/audio_only_pipeline_receipt.json
python_executable=/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python
adaptive_runner=/tmp/avgf_run_adaptive_relocated_manifest.zsh
adaptive_runner_sha256=1934e0d9a3a13248263d009d670469cec2c49c76aa033052701fdccb92871534
verify_lre=/tmp/avgf_verify_lre_manifest_step.zsh
verify_lre_sha256=ce4888d24a53a0a281f7dc518dd7585cc1add024bec99fdf919e1c7939232c42
verify_causal=/tmp/avgf_verify_causal_manifest_step.zsh
verify_causal_sha256=82a59c9aaa259458034ca79fc0c5b6f622cc0539b50fad207f1e0d683e787f61
relocated_runner=/tmp/avgf_benchmark_lre_run_relocated.py
relocated_runner_sha256=6b7811138c1db48c79deb93cf6440da372ff9005985e08836d31c09a8e78abcd
gpu_watchdog=/tmp/avgf_gpu_pid_watchdog_logged.zsh
gpu_watchdog_sha256=d5c824c76da88c0e798aa2aa578ea5730c912c2d585762b285ca5995477f0115
gpu_watchdog_python=/tmp/avgf_gpu_descendant_watchdog.py
gpu_watchdog_python_sha256=6f3dfc6d41f534d5e7ea6c78e713a7d76a4696974f584ec2a082a44679ef0bf7
shard_generator=/tmp/avgf_shard_lre_manifest.py
shard_generator_sha256=634e0cb0be4aa253f1691f41a7784aa27114f0d20a7a1248362468e362991c85
log=/tmp/avgf-audio-only-after-p3-versioned.log

log_event() {
  print -r -- "$(date -Ins) $*" | tee -a "$log"
}

validate_manifest() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$frozen_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_executable" -c \
    'from pathlib import Path; from avgaussianv2.benchmark.lre_orchestration import load_lre_run_manifest; import sys; load_lre_run_manifest(Path(sys.argv[1]))' \
    "$1" >> "$log" 2>&1
}

publish_receipt() {
  local finalist=$1 robustness_sha=$2 temporary=$receipt.tmp.$run_token
  jq -n \
    --arg token "$run_token" \
    --arg p3_token "$p3_run_token" \
    --arg tool_commit "$tool_commit" \
    --arg gate_sha "$(sha256sum "$gate_30k" | cut -d' ' -f1)" \
    --arg confirmation_sha "$confirmation_sha256" \
    --arg robustness_sha "$robustness_sha" \
    --argjson finalist "$finalist" \
    '{schema:"avgaussianv2.audio-only-pipeline-receipt",version:1,status:"succeeded",run_token:$token,p3_run_token:$p3_token,tool_commit:$tool_commit,gate_30k_sha256:$gate_sha,selected_finalist:$finalist,confirmation_manifest_sha256:$confirmation_sha,robustness_manifest_sha256:(if $robustness_sha == "" then null else $robustness_sha end)}' \
    > "$temporary" || return 1
  mv "$temporary" "$receipt"
}

cd "$tool_root" || exit 2
observed_commit=$(git rev-parse HEAD)
observed_dirty=$(git status --porcelain --untracked-files=normal)
log_event "relay_started upstream_pid=$upstream_pid run_token=$run_token p3_run_token=$p3_run_token tool_commit=$observed_commit dirty_length=${#observed_dirty}"
if [[ "$observed_commit" != "$tool_commit" ]] || [[ -n "$observed_dirty" ]]; then
  log_event "tool_identity_failed expected_commit=$tool_commit"
  exit 3
fi
if [[ "$(sha256sum "$adaptive_runner" | cut -d' ' -f1)" != "$adaptive_runner_sha256" ]] \
  || [[ "$(sha256sum "$verify_lre" | cut -d' ' -f1)" != "$verify_lre_sha256" ]] \
  || [[ "$(sha256sum "$verify_causal" | cut -d' ' -f1)" != "$verify_causal_sha256" ]] \
  || [[ "$(sha256sum "$relocated_runner" | cut -d' ' -f1)" != "$relocated_runner_sha256" ]] \
  || [[ "$(sha256sum "$gpu_watchdog" | cut -d' ' -f1)" != "$gpu_watchdog_sha256" ]] \
  || [[ "$(sha256sum "$gpu_watchdog_python" | cut -d' ' -f1)" != "$gpu_watchdog_python_sha256" ]] \
  || [[ "$(sha256sum "$shard_generator" | cut -d' ' -f1)" != "$shard_generator_sha256" ]] \
  || [[ "$(sha256sum "$confirmation_manifest" | cut -d' ' -f1)" != "$confirmation_sha256" ]] \
  || [[ "$(sha256sum "$robustness_manifest" | cut -d' ' -f1)" != "$robustness_sha256" ]] \
  || ! validate_manifest "$confirmation_manifest" \
  || ! validate_manifest "$robustness_manifest"; then
  log_event "dependency_or_manifest_validation_failed"
  exit 4
fi

while kill -0 "$upstream_pid" 2>/dev/null; do
  command_line=$(ps -o args= -p "$upstream_pid" 2>/dev/null)
  if [[ "$command_line" != *"$upstream_identity"* ]]; then
    log_event "upstream_pid_identity_changed command=$command_line"
    exit 5
  fi
  sleep 30
done
log_event "upstream_exited"

if [[ ! -f "$p3_receipt" ]] \
  || [[ "$(jq -r .schema "$p3_receipt")" != avgaussianv2.p3-pipeline-receipt ]] \
  || [[ "$(jq -r .status "$p3_receipt")" != succeeded ]] \
  || [[ "$(jq -r .run_token "$p3_receipt")" != "$p3_run_token" ]] \
  || [[ "$(jq -r .gate_30k_sha256 "$p3_receipt")" != "$(sha256sum "$gate_30k" | cut -d' ' -f1)" ]]; then
  log_event "missing_or_invalid_p3_receipt"
  exit 6
fi

zsh "$adaptive_runner" "$confirmation_manifest" audio-only-seed42-versioned main 30000 >> "$log" 2>&1
status_code=$?
log_event "audio_only_seed42_exited status=$status_code"
if (( status_code != 0 )); then exit 7; fi
zsh "$verify_lre" "$confirmation_manifest" "$run_root" 30000 2 >> "$log" 2>&1
status_code=$?
log_event "audio_only_seed42_verified status=$status_code"
if (( status_code != 0 )); then exit 8; fi

finalist=$(jq -c .selected_finalist "$gate_30k")
if [[ "$finalist" != "$(jq -c .selected_finalist "$p3_receipt")" ]]; then
  log_event "gate_receipt_finalist_mismatch"
  exit 9
fi
if [[ "$finalist" == null ]]; then
  publish_receipt "$finalist" "" || exit 10
  log_event "no_finalist_audio_receipt_published sha256=$(sha256sum "$receipt" | cut -d' ' -f1)"
  exit 0
fi

expected_multiseed_sha=$(jq -r .multiseed_report_sha256 "$p3_receipt")
if [[ ! -f "$multiseed_report" ]] \
  || [[ "$expected_multiseed_sha" == null ]] \
  || [[ "$expected_multiseed_sha" != "$(sha256sum "$multiseed_report" | cut -d' ' -f1)" ]] \
  || [[ "$(jq -r .schema "$multiseed_report")" != avgaussianv2.p3-multiseed-report ]]; then
  log_event "candidate_multiseed_report_or_receipt_invalid"
  exit 11
fi

zsh "$adaptive_runner" "$robustness_manifest" audio-only-seeds-17-73-versioned main 30000 >> "$log" 2>&1
status_code=$?
log_event "audio_only_seed17_73_exited status=$status_code"
if (( status_code != 0 )); then exit 12; fi
zsh "$verify_lre" "$robustness_manifest" "$run_root" 30000 4 >> "$log" 2>&1
status_code=$?
log_event "audio_only_seed17_73_verified status=$status_code"
if (( status_code != 0 )); then exit 13; fi
publish_receipt "$finalist" "$robustness_sha256" || exit 14
log_event "audio_receipt_published sha256=$(sha256sum "$receipt" | cut -d' ' -f1)"

#!/usr/bin/env zsh

set -u

if (( $# != 6 )); then
  print -u2 -- "usage: $0 UPSTREAM_PID UPSTREAM_IDENTITY TOOL_ROOT TOOL_COMMIT FORMAL_ROOT RUN_TOKEN"
  exit 2
fi

upstream_pid=$1
upstream_identity=$2
tool_root=$3
tool_commit=$4
formal_root=$5
run_token=$6
frozen_root=/mnt/sda/lisujing/Dataset/AVGaussianFusionv2/.worktrees/p3-formal-frozen-98d
run_root=$formal_root/runs/lre_loss_ablation_visual_time_v3
result_root=$formal_root/results/lre_loss_ablation_visual_time_v3
gate_10k=$result_root/p3_10k_gate.json
manifest_30k=$formal_root/configs/generated/lre_loss_ablation_p3/confirmation_30k/manifest.json
causal_manifest=$formal_root/configs/generated/lre_loss_ablation_p3/causal_30k/manifest.json
gate_30k=$result_root/p3_30k_gate.json
seed_config_dir=$formal_root/configs/generated/lre_loss_ablation_p3/robustness_seed_configs
seed_manifest=$formal_root/configs/generated/lre_loss_ablation_p3/robustness_seeds/manifest.json
multiseed_report=$result_root/p3_multiseed_report.json
receipt=$result_root/p3_pipeline_receipt.json
python_executable=/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python
adaptive_runner=/tmp/avgf_run_adaptive_relocated_manifest.zsh
causal_generator=/tmp/avgf_generate_p3_causal_manifest.py
gate_implementation=/tmp/avgf_p3_30k_gate.py
seed_generator=/tmp/avgf_generate_p3_seed_manifest.py
multiseed_builder=/tmp/avgf_p3_multiseed_report.py
verify_lre=/tmp/avgf_verify_lre_manifest_step.zsh
strict_gate=$tool_root/scripts/run_verified_p3_30k_gate.py
log=/tmp/avgf-p3-after-30k-versioned.log

typeset -A expected_sha256
expected_sha256[$adaptive_runner]=1934e0d9a3a13248263d009d670469cec2c49c76aa033052701fdccb92871534
expected_sha256[$causal_generator]=56eea1a93057d01da6d9a87314df66b03d9240b8318a93217d1234cf5c6d09ee
expected_sha256[$gate_implementation]=1fac351e3d8baa444034b91fd25760ce44554ee8a8340a8485d27b92290f2525
expected_sha256[$seed_generator]=535db7b450ebfa2cd565af371750bfa73abeca6886dbec6a5b443c7a886157a5
expected_sha256[$multiseed_builder]=cd3322b2eb3ee7a4cceade889f640d00ec213186e23159b27e8dcc37a0be3ac3
expected_sha256[$verify_lre]=ce4888d24a53a0a281f7dc518dd7585cc1add024bec99fdf919e1c7939232c42

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
  local finalist=$1 multiseed_sha=$2 temporary=$receipt.tmp.$run_token
  jq -n \
    --arg token "$run_token" \
    --arg tool_commit "$tool_commit" \
    --arg gate_sha "$(sha256sum "$gate_30k" | cut -d' ' -f1)" \
    --arg manifest_sha "$(sha256sum "$manifest_30k" | cut -d' ' -f1)" \
    --argjson finalist "$finalist" \
    --arg multiseed_sha "$multiseed_sha" \
    '{schema:"avgaussianv2.p3-pipeline-receipt",version:1,status:"succeeded",run_token:$token,tool_commit:$tool_commit,manifest_30k_sha256:$manifest_sha,gate_30k_sha256:$gate_sha,selected_finalist:$finalist,multiseed_report_sha256:(if $multiseed_sha == "" then null else $multiseed_sha end)}' \
    > "$temporary" || return 1
  mv "$temporary" "$receipt"
}

cd "$tool_root" || exit 2
observed_commit=$(git rev-parse HEAD)
observed_dirty=$(git status --porcelain --untracked-files=normal)
log_event "relay_started upstream_pid=$upstream_pid run_token=$run_token tool_commit=$observed_commit dirty_length=${#observed_dirty}"
if [[ "$observed_commit" != "$tool_commit" ]] || [[ -n "$observed_dirty" ]]; then
  log_event "tool_identity_failed expected_commit=$tool_commit"
  exit 3
fi
for dependency in ${(k)expected_sha256}; do
  expected=${expected_sha256[$dependency]}
  observed=$(sha256sum "$dependency" | cut -d' ' -f1)
  if [[ "$observed" != "$expected" ]]; then
    log_event "dependency_hash_failed path=$dependency expected=$expected observed=$observed"
    exit 4
  fi
done
strict_gate_sha=$(sha256sum "$strict_gate" | cut -d' ' -f1)
log_event "dependencies_verified strict_gate_sha256=$strict_gate_sha"

while kill -0 "$upstream_pid" 2>/dev/null; do
  command_line=$(ps -o args= -p "$upstream_pid" 2>/dev/null)
  if [[ "$command_line" != *"$upstream_identity"* ]]; then
    log_event "upstream_pid_identity_changed command=$command_line"
    exit 5
  fi
  sleep 30
done
log_event "upstream_exited"

manifest_30k_sha=$(sha256sum "$manifest_30k" | cut -d' ' -f1)
pointer=$run_root/runner_result.confirmation.json
if [[ ! -f "$pointer" ]] \
  || [[ "$(jq -r .status "$pointer")" != succeeded ]] \
  || [[ "$(jq -r .source_manifest_sha256 "$pointer")" != "$manifest_30k_sha" ]]; then
  log_event "invalid_30k_full_manifest_pointer"
  exit 6
fi
zsh "$verify_lre" "$manifest_30k" "$run_root" 30000 8 >> "$log" 2>&1
status_code=$?
log_event "30k_independent_reverification_exited status=$status_code"
if (( status_code != 0 )); then exit 7; fi

causal_pending=$causal_manifest.pending.$run_token
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$frozen_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_executable" "$causal_generator" \
  --gate "$gate_10k" --manifest-30k "$manifest_30k" --output "$causal_pending" >> "$log" 2>&1
status_code=$?
if (( status_code != 0 )) || ! validate_manifest "$causal_pending"; then
  log_event "causal_manifest_generation_or_validation_failed status=$status_code"
  exit 8
fi
mv "$causal_pending" "$causal_manifest"
log_event "causal_manifest_validated sha256=$(sha256sum "$causal_manifest" | cut -d' ' -f1) runs=$(jq '.runs | length' "$causal_manifest")"
zsh "$adaptive_runner" "$causal_manifest" p3-30k-causal-versioned causal 30000 >> "$log" 2>&1
status_code=$?
log_event "causal_stage_exited status=$status_code"
if (( status_code != 0 )); then exit 9; fi

gate_pending=$gate_30k.pending.$run_token
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$tool_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_executable" "$strict_gate" \
  --gate-implementation "$gate_implementation" \
  --gate-implementation-sha256 "${expected_sha256[$gate_implementation]}" \
  --gate-10k "$gate_10k" --manifest-30k "$manifest_30k" \
  --causal-manifest "$causal_manifest" \
  --noise-report "$result_root/noise_retest1_report.json" \
  --run-root "$run_root" --output "$gate_pending" >> "$log" 2>&1
status_code=$?
if (( status_code != 0 )); then
  log_event "strict_30k_gate_failed status=$status_code"
  exit 10
fi
mv "$gate_pending" "$gate_30k"
finalist=$(jq -c .selected_finalist "$gate_30k")
log_event "30k_gate_complete finalist=$finalist pareto=$(jq -c .pareto_finalists "$gate_30k")"
if [[ "$finalist" == null ]]; then
  publish_receipt "$finalist" "" || exit 11
  log_event "no_30k_finalist_receipt_published sha256=$(sha256sum "$receipt" | cut -d' ' -f1)"
  exit 0
fi

seed_pending=$seed_manifest.pending.$run_token
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$frozen_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_executable" "$seed_generator" \
  --gate-30k "$gate_30k" --manifest-30k "$manifest_30k" \
  --config-dir "$seed_config_dir" --output "$seed_pending" >> "$log" 2>&1
status_code=$?
if (( status_code != 0 )) || ! validate_manifest "$seed_pending"; then
  log_event "seed_manifest_generation_or_validation_failed status=$status_code"
  exit 12
fi
mv "$seed_pending" "$seed_manifest"
log_event "seed_manifest_validated sha256=$(sha256sum "$seed_manifest" | cut -d' ' -f1) runs=$(jq '.runs | length' "$seed_manifest")"
zsh "$adaptive_runner" "$seed_manifest" p3-seeds-17-73-versioned main 30000 >> "$log" 2>&1
status_code=$?
log_event "seed_stage_exited status=$status_code"
if (( status_code != 0 )); then exit 13; fi

multiseed_pending=$multiseed_report.pending.$run_token
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$frozen_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_executable" "$multiseed_builder" \
  --gate-30k "$gate_30k" --manifest-30k "$manifest_30k" \
  --seed-manifest "$seed_manifest" --run-root "$run_root" \
  --output "$multiseed_pending" --bootstrap-resamples 10000 >> "$log" 2>&1
status_code=$?
if (( status_code != 0 )); then
  log_event "multiseed_report_failed status=$status_code"
  exit 14
fi
mv "$multiseed_pending" "$multiseed_report"
multiseed_sha=$(sha256sum "$multiseed_report" | cut -d' ' -f1)
publish_receipt "$finalist" "$multiseed_sha" || exit 15
log_event "multiseed_receipt_published report_sha256=$multiseed_sha receipt_sha256=$(sha256sum "$receipt" | cut -d' ' -f1)"

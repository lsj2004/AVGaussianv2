#!/usr/bin/env zsh

set -u

if (( $# != 5 )); then
  print -u2 -- "usage: $0 UPSTREAM_PID UPSTREAM_IDENTITY AUDIO_RUN_TOKEN REPORT_ROOT FORMAL_ROOT"
  exit 2
fi

upstream_pid=$1
upstream_identity=$2
audio_run_token=$3
report_root=$4
formal_root=$5

expected_report_commit=912b08b12dfeaeda2fa9095c7e7ea0ab0e4cef36
expected_builder_sha256=bdf2ed0e2838b9866b42194592828d3b7515da8d4919256bce507a031e65c31a
run_root=$formal_root/runs/lre_loss_ablation_visual_time_v3
result_root=$formal_root/results/lre_loss_ablation_visual_time_v3
gate_30k=$result_root/p3_30k_gate.json
candidate_main=$formal_root/configs/generated/lre_loss_ablation_p3/confirmation_30k/manifest.json
candidate_seeds=$formal_root/configs/generated/lre_loss_ablation_p3/robustness_seeds/manifest.json
audio_main=$formal_root/configs/generated/audio_only_final_baseline/confirmation/manifest.json
audio_seeds=$formal_root/configs/generated/audio_only_final_baseline/robustness/manifest.json
p2_fair=$result_root/fair_baseline_report.json
architecture_report=$result_root/architecture_config_report.json
parameter_audit=$result_root/p2_cuda_parameter_audit.json
resource_report=$result_root/p2_resource_report.json
reference_aggregate=$result_root/audio_references/aggregate.json
reference_verification=$result_root/audio_references/verification.json
expected_p2_fair_sha256=920364e134896d2e495ba8985936d5798048bc5b4b2ed763604569db3e722eb0
expected_architecture_report_sha256=dc4bb138457e300098709a7e08757f8c38a0a7261fb5632309faa4a8be6dad6c
expected_parameter_audit_sha256=85299ea4f04435f05125ed22a10134152fd854651b7bf31e4d962a5819848356
expected_resource_report_sha256=4db3fbc9ff8dacc848e567fcea5c9462b52bc0c60c681c2fa1fc7f4a67c24cd9
expected_reference_aggregate_sha256=d10e68a5f90df29c538e66fdc889b4e3be46b01fd4b530af4effdce7dcf3e5e6
expected_reference_verification_sha256=38aab786429fe56e4d73f163de9e7a1059f0c4d09fd8acd2385b5aeb54fd9958
audio_receipt=$result_root/audio_only_pipeline_receipt.json
output_json=$result_root/final_fair_comparison.json
output_markdown=$result_root/FINAL_FAIR_COMPARISON.zh-CN.md
python_executable=/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python
builder=$report_root/scripts/build_final_fair_comparison.py
log=/tmp/avgf-final-report-after-audio-baseline.log

log_event() {
  print -r -- "$(date -Ins) $*" | tee -a "$log"
}

cd "$report_root" || exit 2
observed_dirty=$(git status --porcelain --untracked-files=normal)
observed_head=$(git rev-parse HEAD)
observed_builder_sha256=$(sha256sum "$builder" | cut -d' ' -f1)
log_event "relay_started upstream_pid=$upstream_pid audio_run_token=$audio_run_token report_commit=$observed_head dirty_length=${#observed_dirty} builder_sha256=$observed_builder_sha256"
if [[ -n "$observed_dirty" ]] \
  || [[ "$observed_head" != "$expected_report_commit" ]] \
  || [[ "$observed_builder_sha256" != "$expected_builder_sha256" ]]; then
  log_event "report_builder_identity_failed expected_commit=$expected_report_commit expected_builder_sha256=$expected_builder_sha256"
  exit 3
fi

while kill -0 "$upstream_pid" 2>/dev/null; do
  command_line=$(ps -o args= -p "$upstream_pid" 2>/dev/null)
  if [[ "$command_line" != *"$upstream_identity"* ]]; then
    log_event "upstream_pid_identity_changed command=$command_line"
    exit 4
  fi
  sleep 30
done
log_event "upstream_exited"

if [[ ! -f "$gate_30k" ]] \
  || [[ "$(jq -r .schema "$gate_30k")" != avgaussianv2.p3-30k-gate ]]; then
  log_event "missing_or_invalid_30k_gate"
  exit 5
fi
if [[ ! -f "$audio_receipt" ]] \
  || [[ ! -f "$audio_main" ]] \
  || [[ "$(jq -r .schema "$audio_receipt")" != avgaussianv2.audio-only-pipeline-receipt ]] \
  || [[ "$(jq -r .status "$audio_receipt")" != succeeded ]] \
  || [[ "$(jq -r .run_token "$audio_receipt")" != "$audio_run_token" ]] \
  || [[ "$(jq -r .gate_30k_sha256 "$audio_receipt")" != "$(sha256sum "$gate_30k" | cut -d' ' -f1)" ]] \
  || [[ "$(jq -c .selected_finalist "$audio_receipt")" != "$(jq -c .selected_finalist "$gate_30k")" ]] \
  || [[ "$(jq -r .confirmation_manifest_sha256 "$audio_receipt")" != "$(sha256sum "$audio_main" | cut -d' ' -f1)" ]]; then
  log_event "missing_or_invalid_audio_pipeline_receipt"
  exit 6
fi
for required in "$candidate_main" "$audio_main" "$p2_fair" \
  "$architecture_report" "$parameter_audit" "$resource_report"; do
  if [[ ! -f "$required" ]]; then
    log_event "required_input_missing path=$required"
    exit 7
  fi
done
if [[ "$(sha256sum "$p2_fair" | cut -d' ' -f1)" != "$expected_p2_fair_sha256" ]] \
  || [[ "$(sha256sum "$architecture_report" | cut -d' ' -f1)" != "$expected_architecture_report_sha256" ]] \
  || [[ "$(sha256sum "$parameter_audit" | cut -d' ' -f1)" != "$expected_parameter_audit_sha256" ]] \
  || [[ "$(sha256sum "$resource_report" | cut -d' ' -f1)" != "$expected_resource_report_sha256" ]] \
  || [[ "$(sha256sum "$reference_aggregate" | cut -d' ' -f1)" != "$expected_reference_aggregate_sha256" ]] \
  || [[ "$(sha256sum "$reference_verification" | cut -d' ' -f1)" != "$expected_reference_verification_sha256" ]]; then
  log_event "p2_reference_root_hash_mismatch"
  exit 8
fi

report_args=(
  --gate-30k "$gate_30k"
  --candidate-main-manifest "$candidate_main"
  --audio-main-manifest "$audio_main"
  --run-root "$run_root"
  --p2-fair-report "$p2_fair"
  --architecture-report "$architecture_report"
  --parameter-audit "$parameter_audit"
  --resource-report "$resource_report"
  --output-json "$output_json"
  --output-markdown "$output_markdown"
  --bootstrap-resamples 10000
)
if [[ "$(jq -c .selected_finalist "$gate_30k")" != null ]]; then
  if [[ ! -f "$audio_seeds" ]] \
    || [[ "$(jq -r .robustness_manifest_sha256 "$audio_receipt")" != "$(sha256sum "$audio_seeds" | cut -d' ' -f1)" ]]; then
    log_event "audio_robustness_receipt_mismatch"
    exit 9
  fi
  for required in "$candidate_seeds" "$audio_seeds"; do
    if [[ ! -f "$required" ]]; then
      log_event "required_multiseed_input_missing path=$required"
      exit 7
    fi
  done
  report_args+=(
    --candidate-seed-manifest "$candidate_seeds"
    --audio-seed-manifest "$audio_seeds"
  )
else
  log_event "no_finalist; generating_formal_rejection_report_without_multiseed"
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$report_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_executable" "$builder" "${report_args[@]}" >> "$log" 2>&1
status_code=$?
if (( status_code != 0 )); then
  log_event "final_report_failed status=$status_code"
  exit 10
fi
log_event "final_report_complete json_sha256=$(sha256sum "$output_json" | cut -d' ' -f1) markdown_sha256=$(sha256sum "$output_markdown" | cut -d' ' -f1)"

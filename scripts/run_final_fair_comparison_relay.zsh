#!/usr/bin/env zsh

set -u

if (( $# != 4 )); then
  print -u2 -- "usage: $0 UPSTREAM_PID UPSTREAM_IDENTITY REPORT_ROOT FORMAL_ROOT"
  exit 2
fi

upstream_pid=$1
upstream_identity=$2
report_root=$3
formal_root=$4

expected_report_commit=ab184f9e6c1296de655c8b093e625d2fa6c6c6f3
expected_builder_sha256=6e48d3c6dbb6b0ce39b15bb5a1bca08de48143bf0e8dab9de11b9273d42fe927
run_root=$formal_root/runs/lre_loss_ablation_visual_time_v3
result_root=$formal_root/results/lre_loss_ablation_visual_time_v3
gate_30k=$result_root/p3_30k_gate.json
candidate_main=$formal_root/configs/generated/lre_loss_ablation_p3/confirmation_30k/manifest.json
candidate_seeds=$formal_root/configs/generated/lre_loss_ablation_p3/robustness_seeds/manifest.json
audio_main=$formal_root/configs/generated/audio_only_final_baseline/confirmation/manifest.json
audio_seeds=$formal_root/configs/generated/audio_only_final_baseline/robustness/manifest.json
p2_fair=$result_root/fair_baseline_report.json
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
log_event "relay_started upstream_pid=$upstream_pid report_commit=$observed_head dirty_length=${#observed_dirty} builder_sha256=$observed_builder_sha256"
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
for required in "$candidate_main" "$audio_main" "$p2_fair"; do
  if [[ ! -f "$required" ]]; then
    log_event "required_input_missing path=$required"
    exit 6
  fi
done

report_args=(
  --gate-30k "$gate_30k"
  --candidate-main-manifest "$candidate_main"
  --audio-main-manifest "$audio_main"
  --run-root "$run_root"
  --p2-fair-report "$p2_fair"
  --output-json "$output_json"
  --output-markdown "$output_markdown"
  --bootstrap-resamples 10000
)
if [[ "$(jq -c .selected_finalist "$gate_30k")" != null ]]; then
  for required in "$candidate_seeds" "$audio_seeds"; do
    if [[ ! -f "$required" ]]; then
      log_event "required_multiseed_input_missing path=$required"
      exit 6
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
  exit 7
fi
log_event "final_report_complete json_sha256=$(sha256sum "$output_json" | cut -d' ' -f1) markdown_sha256=$(sha256sum "$output_markdown" | cut -d' ' -f1)"

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POST_P3 = ROOT / "scripts/run_p3_post_30k_relay.zsh"
AUDIO_ONLY = ROOT / "scripts/run_audio_only_after_p3_relay.zsh"
FINAL = ROOT / "scripts/run_final_fair_comparison_relay.zsh"


def test_versioned_pipeline_relays_have_valid_zsh_syntax() -> None:
    for script in (POST_P3, AUDIO_ONLY, FINAL):
        subprocess.run(["zsh", "-n", str(script)], check=True)


def test_post_p3_relay_uses_strict_gate_and_publishes_token_bound_receipt() -> None:
    text = POST_P3.read_text()

    assert "run_verified_p3_30k_gate.py" in text
    assert "--gate-implementation-sha256" in text
    assert "avgf_benchmark_lre_run_relocated.py" in text
    assert "avgf_gpu_pid_watchdog_logged.zsh" in text
    assert "avgf_shard_lre_manifest.py" in text
    assert "avgf_verify_causal_manifest_step.zsh" in text
    assert "p3-pipeline-receipt" in text
    assert 'run_token:$token' in text
    assert 'gate_30k_sha256:$gate_sha' in text
    assert 'mv "$gate_pending" "$gate_30k"' in text


def test_audio_relay_requires_p3_receipt_and_verifies_both_manifests() -> None:
    text = AUDIO_ONLY.read_text()

    assert "missing_or_invalid_p3_receipt" in text
    assert "relocated_runner_sha256" in text
    assert "gpu_watchdog_sha256" in text
    assert "shard_generator_sha256" in text
    assert "verify_causal_sha256" in text
    assert '[[ "$(jq -r .run_token "$p3_receipt")" != "$p3_run_token" ]]' in text
    assert '"$confirmation_manifest" "$run_root" 30000 2' in text
    assert '"$robustness_manifest" "$run_root" 30000 4' in text
    assert "audio-only-pipeline-receipt" in text


def test_final_relay_requires_exact_audio_receipt_token_and_hashes() -> None:
    text = FINAL.read_text()

    assert "missing_or_invalid_audio_pipeline_receipt" in text
    assert '[[ "$(jq -r .run_token "$audio_receipt")" != "$audio_run_token" ]]' in text
    assert "confirmation_manifest_sha256" in text
    assert "robustness_manifest_sha256" in text
    assert "expected_p2_fair_sha256" in text
    assert "expected_reference_aggregate_sha256" in text
    assert "expected_reference_verification_sha256" in text
    assert "p2_reference_root_hash_mismatch" in text

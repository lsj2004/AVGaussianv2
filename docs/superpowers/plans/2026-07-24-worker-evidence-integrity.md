# Archived Implementation Plan

This plan extends the retired Scene1 pilot workflow. Its implementation remains
available in Git history on `origin/agent/scene1-pilot`; the current clean branch
uses the strict cam38 benchmark evidence pipeline instead.

# Worker Evidence Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pilot acceptance depend only on hash-pinned worker summaries and complete latest checkpoints, while making every failure after the `current` rename non-raising and resolver-visible.

**Architecture:** Extend report inputs with a frozen worker artifact provenance record. Read worker JSON and latest checkpoints once from pinned descriptors, canonically inspect latest/best Task6 state, and compare all training evidence before deciding readiness. Treat the `current` rename as the publication commit point and store post-commit warnings in a generation-keyed sidecar.

**Tech Stack:** Python dataclasses, descriptor-relative POSIX file operations, PyTorch safe checkpoint inspection, pytest.

## Global Constraints

- Worker summaries and latest checkpoints are untrusted until SHA-256 verification and pinned-fd parsing complete.
- Joint conditioned on/off must use the identical trained worker/latest artifact identity.
- Acceptance gradient evidence comes only from the canonical latest checkpoint joint history.
- No `Exception` may escape after the `current` pointer rename; `BaseException` cancellation semantics remain unchanged.
- Report generation contents remain immutable.

---

### Task 1: Verified worker and latest provenance

**Files:**
- Modify: `avgaussianv2/experiment/report.py`
- Modify: `avgaussianv2/experiment/__init__.py`
- Test: `tests/test_experiment_report.py`

**Interfaces:**
- Produces: frozen `WorkerArtifactProvenance` attached to every trained `SystemReportInput`.
- Produces: secure worker JSON loader and pinned latest checkpoint canonical inspector.

- [ ] Add failing fixtures that write actual `worker_summary.json` and `latest.pt`, hash them, and construct `WorkerArtifactProvenance`.
- [ ] Run focused tests and confirm the missing type/validation failures.
- [ ] Add the frozen provenance type and require baseline `None`, trained systems non-`None`.
- [ ] Hash/read bounded worker JSON from one `O_NOFOLLOW` descriptor and inspect latest from the same descriptor used for hashing.
- [ ] Require complete latest state, exact variant, generation/fingerprint, histories/counts/stop state/best generation, and exact agreement with worker JSON and best identity.
- [ ] Derive maximum positive audio-to-visual gradient from verified latest joint rows only.
- [ ] Require joint on/off identical worker/latest identities and add once-per-distinct-artifact cache assertions.
- [ ] Run focused tests and confirm all worker/latest cases pass.

### Task 2: Irreversible post-commit cleanup and persisted warnings

**Files:**
- Modify: `avgaussianv2/experiment/report.py`
- Test: `tests/test_experiment_report.py`

**Interfaces:**
- Produces: generation-keyed durability warning sidecar outside immutable generation contents.
- Changes: `resolve_current_report` returns a path-compatible result exposing persisted warnings.

- [ ] Add failing tests for post-rename unlock and each directory/lock/output close failure.
- [ ] Add failing tests for warning-sidecar persistence and persistence failure fallback.
- [ ] Guard every post-commit `Exception`, attempt all cleanup operations, and append deterministic operation-labelled warnings.
- [ ] Persist warnings best-effort using atomic write/rename and make the resolver load them without changing current/generation content.
- [ ] Run the post-commit fault matrix and confirm the new generation remains authoritative.

### Task 3: Final evidence validation and verification

**Files:**
- Modify: `avgaussianv2/experiment/report.py`
- Test: `tests/test_experiment_report.py`

**Interfaces:**
- Enforces: nonnegative training totals and exact success/failure/attempt counter equations.

- [ ] Add failing tests for negative totals and independently incoherent save/backup counters.
- [ ] Add the minimal validation rules.
- [ ] Run `pytest tests/test_experiment_report.py -q`.
- [ ] Run the complete pytest suite.
- [ ] Run `git diff --check`, review the final diff, and commit as `fix: verify pilot worker evidence for acceptance`.

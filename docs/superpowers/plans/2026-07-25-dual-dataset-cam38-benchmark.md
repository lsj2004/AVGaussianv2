# Dual-Dataset Cam38 Benchmark Implementation Plan

## Task 11: Freeze protocol and upstream assets

- Add strict cam38 benchmark configs for both scenes.
- Add split validation and asset-audit contracts.
- Add reproducible AudioGS viewpoint-39 baseline commands.
- Add reproducible FreeTimeGS++ cam38 strict-holdout preparation/training
  commands with independent cache namespaces.
- Record native budgets: AudioGS 61 epochs and FreeTimeGS++ 30,000 updates.
- Test that cam38 target files cannot enter image-driven training
  initialization.

## Task 12: Fixed-budget benchmark training

- Add a benchmark-specific experiment schema without changing the existing
  cam10 pilot schema.
- Add `joint_conditioned`, `audio_only`, and `visual_only` trainability modes.
- Resolve and persist a fixed 30,000-update budget and a shared sample-index
  sequence.
- Add final-checkpoint selection with no evaluation dataset construction during
  training.
- Add exact periodic resume, atomic 5k/10k/30k milestones, bounded retention,
  and checkpoint I/O accounting.
- Cover exact resume and trainability with CPU tests.

## Task 13: Evaluation and reporting

- Evaluate native initialization references and all three update-matched
  continuations on the exact cam38 aligned sample IDs.
- Report the full supported audio metric family and PSNR/SSIM/RGB-L1.
- Add paired audio comparison against `audio_only` and paired video comparison
  against `visual_only`.
- Add predeclared 5k/10k/30k scaling tables.
- Add strict provenance, finite-value, budget, split, initialization, and
  sample-pairing gates.
- Add two-scene macro/micro aggregation.

## Task 14: Orchestration and verification

- Add a resumable per-scene three-GPU orchestrator and a two-scene suite
  orchestrator.
- Ensure final evaluation is the first consumer of cam38 targets.
- Add immutable status, logs, process supervision, `--resume`, and
  zero-mutation `--verify-only`.
- Add shell entrypoints and README instructions.
- Run focused tests, the complete offline suite, Ruff, shell syntax,
  compilation, and diff checks.

## Task 15: Real baselines

- Preflight three GPUs, disk, data, dependencies, and all source hashes.
- In parallel, train strict cam38 FreeTimeGS++ baselines for both scenes and the
  AudioGS viewpoint-39 baselines.
- Verify native checkpoint provenance and common-evaluator metrics.

## Task 16: Real update-matched benchmark

- Run the three fixed-budget variants for `scene1_opera`.
- Run the three fixed-budget variants for `Scene7playing`.
- Resume only from compatible periodic checkpoints.
- Evaluate the 5k/10k/30k milestones and generate per-scene reports.
- Generate the two-scene macro/micro report and run independent
  `--verify-only`.

## Task 17: Final review and delivery

- Review specification compliance and code quality.
- Confirm no test-camera leakage and no running GPU processes.
- Push the benchmark branch and deliver report/checkpoint links.

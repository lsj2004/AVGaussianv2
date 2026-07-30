# Archived Design

This document describes the retired Scene1 pilot workflow. Its implementation
remains available in Git history on `origin/agent/scene1-pilot`; the current
clean branch uses the strict cam38 benchmark instead.

# Scene1 Pilot Training and Evaluation Design

## Scope

This milestone runs a controlled, fast diagnostic on `scene1_opera` before any long training.
It adds reproducible pilot orchestration, held-out evaluation, early stopping, best-checkpoint
selection, and comparison reporting. It does not run `Scene7playing`, search large
hyperparameter spaces, or claim an upper-bound result. Those actions follow only after this
pilot demonstrates stable convergence and a useful audio improvement without unacceptable
visual degradation.

The pilot may use GPUs 0, 1, and 2 concurrently and should target roughly 15–30 minutes of
wall-clock time. Runtime is a target rather than a correctness condition; correctness is
defined by finite metrics, valid checkpoints, completed comparisons, and the acceptance rules
below.

## Goals

The diagnostic must answer four questions:

1. Does RGBD conditioning improve held-out `cam10` audio relative to the imported AudioGS
   checkpoint and a condition-off continuation baseline?
2. Does allowing audio loss to update visual Gaussians improve audio over freezing the visual
   field?
3. Does visual quality remain within the configured PSNR and SSIM tolerances?
4. Is any improvement broadly distributed across held-out samples rather than caused by a few
   outliers?

## Architecture

The experiment layer is additive and does not change the upstream-specific visual or audio
backend interfaces.

- `PilotTrainer` owns deterministic sampling, stage execution, periodic validation, early
  stopping, and best/latest checkpoint lifecycle.
- `Evaluator` evaluates a model on a fixed subset or the full held-out split and emits
  per-sample and aggregate audio/visual metrics.
- `ExperimentVariant` defines trainable parameter groups and condition state for each controlled
  experiment.
- `ComparisonReporter` combines baselines and trained variants into JSON, CSV, and Markdown
  summaries.
- The pilot CLI creates the experiment manifest, evaluates baselines, launches the three GPU
  workers, verifies their outputs, runs final full-split evaluation, and writes the comparison.

The model data path remains:

```text
aligned time/camera sample
  -> differentiable FreeTimeGS++ RGBD render
  -> RGBD condition encoder
  -> AudioGS U-Net conditional residual
  -> binaural prediction and RGB render
  -> audio + visual losses
```
For imported GS-only AudioGS checkpoints, condition-off remains the native GS-only output.
Condition-on uses the already implemented residual
`native + conditioned U-Net - plain U-Net`, preserving the imported function at zero-init.

## Controlled Variants

The experiment uses one immutable training index sequence, validation subset, seed, upstream
checkpoint pair, and metric implementation across all variants.

| Variant | Warmup | Joint | Condition | Visual updates | Audio updates |
|---|---:|---:|---|---|---|
| `joint_conditioned` | 200 | up to 500 | on | on during joint | on during joint |
| `frozen_visual` | 200 | up to 500 | on | off | on during joint |
| `condition_off` | 0 | up to 500 | off | off | on during joint |

The condition-off variant has no warmup because disabling the condition removes the warmup
gradient path. All variants still perform the same maximum number of AudioGS/acoustic updates:
500 joint steps. The two conditioned variants use 200 additional condition-only warmup updates.

Before training, the evaluator records an immutable baseline from the original AudioGS and
FreeTimeGS++ checkpoints. For the trained `joint_conditioned` checkpoint, final evaluation also
runs condition-on and condition-off through the same weights to measure the net effect of the
condition without conflating it with continued AudioGS training.

## Sampling and Reproducibility

Training must not consume only the first N manifest records. A seeded generator creates a
deterministic random index sequence long enough for warmup and joint stages. The sequence is
stored in the experiment manifest and shared by all applicable variants. Because
`condition_off` has no warmup, its joint sequence matches the joint portion used by the other
variants.

The quick validation subset contains 32 deterministic, evenly distributed `cam10` records.
The subset covers the held-out timeline rather than taking the first 32 records. Final
evaluation uses every valid `cam10` record.

Each run records:

- resolved project and variant configuration;
- random seed and exact train/validation indices;
- current Git commit;
- upstream visual and audio checkpoint paths and SHA-256 hashes;
- PyTorch, CUDA, and GPU identity;
- stage, step, validation history, and stop reason.

## Training and Early Stopping

Warmup always executes 200 steps for conditioned variants. Joint training validates every 50
steps and executes at least 200 steps. After the minimum, it stops when the held-out AudioGS
total loss fails to improve by at least 0.5% for four consecutive validation events. The
relative improvement is measured against the best prior validation value, not against the
immediately preceding noisy value.

Every validation event atomically updates `latest.pt`. `best.pt` changes only when the candidate
satisfies visual constraints and improves held-out AudioGS total loss. Non-finite predictions,
losses, metrics, or gradients terminate the worker with the exact scene, camera, frame, time,
variant, stage, and step. The full conditioned variant also retains the existing invariant that
audio loss must reach visual Gaussian parameters during joint training.

## Metrics

### Audio

- upstream AudioGS `total_loss`;
- upstream `mono_loss` and `diff_loss`;
- binaural waveform L1;
- mono log-spectral distance;
- diff log-spectral distance;
- absolute left/right energy-ratio error in dB (LRE error).

Log-spectral distance uses the configured Hamming-window STFT and an epsilon-protected log
magnitude. LRE uses per-channel waveform energy with an epsilon before the base-10 logarithm.
All calculations reject non-finite inputs rather than replacing an invalid sample silently.

### Visual

- RGB PSNR;
- RGB SSIM;
- RGB L1.

The original imported FreeTimeGS++ checkpoint establishes the visual baseline on the same
held-out samples.

For every metric, evaluation writes the per-sample value plus mean, standard deviation, and
median. Lower is better for all listed audio metrics and RGB L1; higher is better for PSNR and
SSIM.

## Best-Checkpoint and Pilot Acceptance Rules

A checkpoint is visually feasible when both conditions hold on quick validation:

- PSNR is no more than 0.5 dB below the original visual baseline;
- SSIM is no more than 0.01 below the original visual baseline.

Among feasible checkpoints, the one with the lowest held-out AudioGS total loss is `best.pt`.
If no checkpoint is feasible, the run is explicitly marked `visual_degradation`; its numerically
lowest audio loss is retained for diagnosis but is not reported as an accepted improvement.

The pilot is considered ready for long training only when:

1. all three workers and all final evaluations exit successfully;
2. the full conditioned run has finite metrics and a nonzero audio-to-visual gradient during
   joint training;
3. its best checkpoint satisfies both visual constraints;
4. condition-on improves held-out AudioGS total loss over the same checkpoint with condition
   off;
5. the median per-sample audio change agrees in direction with the mean change.

Comparisons against `frozen_visual` and the separately trained `condition_off` variant are
reported even if the pilot does not pass the long-training gate.

## Evaluation and Reports

Training-time validation emits a compact curve. After all workers finish, each best checkpoint
is evaluated over the complete held-out split. The reporter produces:

- `metrics_per_sample.jsonl` for every evaluated system;
- `metrics_summary.json` with aggregate statistics;
- `training_curve.csv` per worker;
- `comparison.csv` for machine-readable cross-system comparison;
- `comparison.md` with the baseline, variants, visual feasibility, convergence, and acceptance
  decision;
- fixed-sample condition-on/off WAVs and RGB/depth previews;
- `best.pt` and `latest.pt` per trainable variant.

The report must distinguish three effects:

- imported checkpoint versus continued AudioGS training;
- condition-on versus condition-off with identical trained weights;
- joint visual updates versus frozen visual Gaussians.

## CLI and Process Orchestration

The user-facing entry point is:

```bash
python -m avgaussianv2.cli.pilot \
  --config configs/scene1_opera.yaml \
  --output-dir runs/pilot_scene1_opera \
  --gpus 0,1,2
```

The orchestrator evaluates the baseline, writes shared indices, launches one worker per GPU,
waits for every worker, validates required artifacts, performs full evaluations, and writes the
comparison. It never overwrites a nonempty output directory unless an explicit compatible
resume flag is supplied.

Resume checks scene ID, variant, camera mapping hash, STFT settings, seed, index sequence,
upstream checkpoint hashes, and model tensor shapes. A mismatch is fatal and names the first
incompatible field.

## Failure Handling

- A missing upstream checkpoint reports its exact path and stops before GPU workers launch.
- A worker failure terminates the overall pilot as failed after collecting the other workers'
  exit status and logs; partial results remain available for diagnosis.
- Missing, duplicated, or changed shared sample indices are fatal.
- Empty evaluation splits and non-finite metric aggregates are fatal.
- Disk writes for checkpoints and summary JSON use temporary files plus atomic replacement.
- Reports never substitute zero for a missing metric and never label an infeasible checkpoint as
  accepted.

## Verification

Automated CPU tests use tiny fake backends and cover:

- deterministic random sampling and identical joint sequences across variants;
- evenly distributed held-out subset selection;
- audio and visual metric formulas, directions, and finite-value validation;
- visual-feasibility filtering and best-checkpoint selection;
- minimum joint length, patience, and relative-improvement early stopping;
- variant parameter-freezing and condition policies;
- paired condition-on/off evaluation;
- strict resume compatibility;
- three-worker result validation and comparison reporting;
- end-to-end pilot orchestration without real CUDA.

Real acceptance runs the single `scene1_opera` pilot command with GPUs 0, 1, and 2. It must
produce three successful workers, finite training and full-split metrics, valid best/latest
checkpoints, and the complete comparison report. Only after reviewing that report will the
project proceed to a separate long-training design for upper-bound results and expansion to
`Scene7playing`.

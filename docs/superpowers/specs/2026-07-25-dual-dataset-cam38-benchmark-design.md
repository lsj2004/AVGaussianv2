# Dual-Dataset Cam38 Benchmark Design

## Goal

Evaluate AVGaussianFusionv2 on `scene1_opera` and `Scene7playing` against
standalone audio and visual reconstruction baselines. In both scenes, `cam38`
is the only test camera and `cam00` through `cam37` are training cameras.

The benchmark must be resumable, must not use test targets for initialization
or model selection, and must make training-budget differences explicit.

## Strict split

- Train cameras: `cam00` ... `cam37`.
- Test camera: `cam38`.
- Valid aligned test samples:
  - `scene1_opera`: 130.
  - `Scene7playing`: 293.
- Camera poses and intrinsics may be known for all cameras. Test RGB, depth-like
  renderer targets, and test audio may only be read by the final evaluator.
- FreeTimeGS++ extraction, point initialization, temporal flow, COLMAP caches,
  and memmaps use a new cam38-specific namespace. Temporal-flow camera ranges
  stop before camera 38.
- Existing cam10 checkpoints and caches are historical references only and are
  not valid benchmark initialization.

## Native initialization baselines

Each scene receives newly trained, cam38-holdout upstream checkpoints:

- AudioGS-replay: viewpoint 39 (`cam38`) held out, 39-view metadata, batch size
  1, seed 42, 61 epochs. The resolved update count is recorded.
- FreeTimeGS++: camera 38 held out, batch size 1, 30,000 iterations, with fresh
  strict-holdout caches.

Both checkpoints are evaluated once on the common aligned cam38 windows and
their paths, hashes, upstream configurations, split, seed, and budgets are
bound into the benchmark manifest.

## Update-matched comparison

The native checkpoints are a shared initialization. Three continuations use
the same seed and exact sample-index sequence:

1. `joint_conditioned`: update AudioGS, the RGB/depth conditioner, and the
   visual carrier with the joint loss.
2. `audio_only`: update only AudioGS acoustic reconstruction parameters with
   the audio loss; visual and conditioning parameters are frozen.
3. `visual_only`: update only the FreeTimeGS++ carrier with the same visual loss
   used by the joint system; audio and conditioning parameters are frozen.

Each continuation performs exactly 30,000 main updates. A 2,000-step
conditioner warmup is permitted only for `joint_conditioned`; it freezes both
pretrained modality models and is reported separately rather than counted as a
modality update. Checkpoints at 5,000, 10,000, and 30,000 are evaluated as
predeclared reporting points. The primary comparison is the 30,000-step final
checkpoint.

Because upstream projects and the fusion trainer define an epoch differently,
optimizer updates are the primary fairness axis. Reports also include epochs,
dataset length, batch size, and sample exposures where those values exist.

## Selection and evaluation

- Training never reads cam38 targets.
- There is no test-set early stopping or best-checkpoint selection.
- The final checkpoint at the fixed budget is authoritative.
- Intermediate 5k/10k results are a predeclared scaling curve, not candidates
  from which the primary result is selected.
- All systems are evaluated with the same aligned dataset, sample IDs, crop,
  and metric implementation.

Audio metrics include `audio_total`, mono and difference reconstruction losses,
waveform L1, mono/difference LSD, LRE error, and every already-supported common
audio metric. Lower is better unless the metric schema explicitly says
otherwise.

Video metrics include RGB PSNR, SSIM, and RGB L1. LPIPS is included only if the
same local implementation and weights are available to every system. No depth
accuracy metric is reported because these datasets do not provide an
independently verified depth ground truth.

For an exact RGB match, mathematical PSNR is positive infinity. The benchmark
reports a finite 100 dB cap for that case; the cap is persisted in every metric
artifact so JSON, CSV, paired comparisons, and aggregate reports remain finite.

Each scene report contains per-sample JSONL/CSV, mean, standard deviation,
median, paired deltas, and win rates. A suite report contains per-scene tables
plus macro and sample-weighted micro aggregates.

## Checkpointing and recovery

- Full exact-resume checkpoints are written periodically, not every step.
- Required reporting checkpoints are atomically published at 5k, 10k, and 30k.
- The default periodic interval is 500 steps.
- A lightweight durable progress journal is updated more frequently.
- Resume fingerprints bind scene, split, budgets, shared sample sequence,
  initialization hashes, source/config hashes, and checkpoint policy.
- Existing incompatible output fails closed; it is never overwritten by
  `--resume`.
- `--verify-only` performs no runtime construction, process launch, or output
  mutation.

## Acceptance

A scene is complete only when:

- the native AudioGS and FreeTimeGS++ baselines satisfy the cam38 provenance;
- all three continuations finish the exact fixed budget;
- every system has exactly the expected cam38 sample IDs and finite metrics;
- paired comparisons use identical sample IDs;
- no test target was read before final evaluation;
- independent `--verify-only` succeeds.

The suite is complete only when both scenes pass and the aggregate report binds
the two immutable scene reports. Results are reported without requiring the
fusion model to beat both baselines.

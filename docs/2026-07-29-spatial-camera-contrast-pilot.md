# Query-dependent P1 spatial camera contrast pilot

Date: 2026-07-29

## Objective

The original query-dependent P1 camera contrast ranks correct-camera and
wrong-camera predictions with the native AudioGS total loss. That objective is
not guaranteed to align with spatial evaluation metrics. This variant directly
optimizes differentiable binaural geometry errors.

## Training objective

For target binaural audio `y`, correct-camera prediction `p+`, and same-frame
wrong-camera prediction `p-`, define:

```text
S(p, y) = 0.30 LRE + 0.25 ILD + 0.25 IPD + 0.20 binaural-difference

L_spatial = 0.10 S(p+, y)
          + 0.50 max(0, 0.05 - (S(p-, y) - S(p+, y)))
```

The complete training loss retains the native AudioGS criterion and adds
`L_spatial`. The positive spatial anchor is necessary: ranking alone starts
from almost identical correct/wrong predictions and provides a weak
anti-symmetric gradient.

The component definitions are:

- LRE: Charbonnier error between global left/right energy ratios in dB,
  normalized by 6 dB.
- ILD: target-energy-weighted STFT log-magnitude-ratio error, normalized by
  one octave (`ln 2`).
- IPD: circular complex-phase distance with bilateral target-energy weighting.
- Binaural difference: target-energy-weighted error between
  `log1p(|STFT(left) - STFT(right)|)` features.

Low-energy bins are continuously down-weighted. IPD additionally requires
target energy in both channels. This avoids unstable phase gradients from
silent or single-channel bins.

## Integration

The new backend is `query_dependent_p1_spatial`. It uses the same architecture,
initialization, data order, update budget, and native AudioGS loss as
`query_dependent_p1`; only the camera-loss objective changes.

It is integrated into:

- both cam38 scene configurations;
- strict preparation and training;
- correct-condition, no-RGBD, and wrong-camera evaluation variants;
- cam38 reporting.

## 500 + 500 step pilot

Each scene was trained for 500 condition-warmup steps followed by 500 joint
steps. Measurements use the same fixed 32 training-camera probe pairs before
training, after warmup, and after joint training. These are optimization
diagnostics, not held-out benchmark results.

### Correct-camera spatial errors

| Scene | Stage | Total | LRE | ILD | IPD | Diff |
|---|---:|---:|---:|---:|---:|---:|
| scene1_opera | initial | 0.483663 | 0.448608 | 0.883520 | 0.348684 | 0.205148 |
| scene1_opera | post-joint | 0.391733 | 0.219034 | 0.795780 | 0.344991 | 0.204149 |
| Scene7playing | initial | 0.257064 | 0.179125 | 0.570924 | 0.162302 | 0.100100 |
| Scene7playing | post-joint | 0.216878 | 0.096705 | 0.512182 | 0.159352 | 0.099916 |

Relative changes:

- scene1_opera: spatial total -19.0%, LRE -51.2%, ILD -9.9%.
- Scene7playing: spatial total -15.6%, LRE -46.0%, ILD -10.3%.

### Camera discrimination

| Scene | Stage | Mean `wrong - correct` spatial loss | Correct-camera win rate |
|---|---:|---:|---:|
| scene1_opera | initial | 0.000000001 | 34.4% |
| scene1_opera | post-joint | 0.003729 | 78.1% |
| Scene7playing | initial | -0.000000015 | 34.4% |
| Scene7playing | post-joint | 0.000250 | 43.8% |

Peak allocated CUDA memory was 1.43 GB for scene1 and 1.44 GB for Scene7.
No NaNs or out-of-memory failures occurred.

## Decision

The positive spatial objective is effective in both scenes, with the largest
gain in LRE. Camera discrimination improves clearly in scene1 but remains weak
in Scene7 at this short horizon. Therefore:

1. keep the implementation and run the strict production diagnostic;
2. evaluate the full 5k/10k/30k held-out benchmark before claiming a spatial
   generalization improvement;
3. do not increase the contrast weight yet—Scene7 may need more optimization
   time, and a larger rank-only weight can improve the gap by degrading the
   wrong-camera branch rather than improving correct-camera audio.


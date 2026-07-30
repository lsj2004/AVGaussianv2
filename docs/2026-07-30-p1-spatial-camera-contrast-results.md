# Spatial camera-contrast P1: cam38 30k results

Date: 2026-07-30

## Protocol

- Compared systems:
  - `query_dependent_p1`
  - `query_dependent_p1_spatial`
- Training budget per scene: 2,000 condition-warmup updates followed by
  30,000 joint updates.
- Evaluation checkpoints: 5k, 10k, and 30k joint updates.
- Evaluation conditions:
  - correct camera and RGBD condition;
  - RGBD disabled;
  - same-frame wrong-camera condition.
- Held-out camera: cam38.
- Samples: 130 for `scene1_opera`, 293 for `Scene7playing`.
- All metrics in this report are lower-is-better.
- Direct P1-versus-spatial win rates are paired by sample ID.

The spatial objective was:

```text
S = 0.30 LRE + 0.25 ILD + 0.25 IPD + 0.20 binaural difference

L_spatial = 0.10 S(correct)
          + 0.50 max(0, 0.05 - [S(wrong) - S(correct)])
```

## Correct-condition result

### Direct paired comparison

`Delta` is `spatial - original`; negative is better. `Win` is the fraction of
held-out samples on which spatial P1 is lower than original P1.

| Scene | Step | Audio total delta | Audio win | LRE delta (dB) | LRE win | Wave L1 delta | Wave win |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene1 | 5k | -0.035892 | 71.5% | -0.575301 | 82.3% | +0.000294 | 24.6% |
| scene1 | 10k | -0.012744 | 37.7% | +0.406375 | 0.8% | -0.000118 | 71.5% |
| scene1 | 30k | +0.024945 | 36.2% | +0.164326 | 42.3% | +0.000302 | 64.6% |
| Scene7 | 5k | -0.001155 | 64.5% | -0.089800 | 68.6% | +0.000181 | 8.9% |
| Scene7 | 10k | +0.020447 | 29.4% | +0.119817 | 46.4% | +0.000677 | 1.0% |
| Scene7 | 30k | +0.007809 | 19.5% | +0.033174 | 37.2% | +0.000115 | 28.0% |

At 30k:

- scene1 audio total worsened by 2.1%, LRE by 39.1%, and waveform L1 by
  0.6%;
- Scene7 audio total worsened by 6.0%, LRE by 15.2%, and waveform L1 by
  0.8%.

The 5k improvement is not sustained. The sharp scene1 LRE reversal between 5k
and 10k also shows that the spatial objective is not producing a stable
held-out optimization trajectory.

## All requested condition metrics

### scene1_opera

| Step | System | Condition | Audio total | LRE error dB | Waveform L1 |
|---|---|---|---:|---:|---:|
| 5k | P1 | correct | 1.450007 | 1.050473 | 0.051537 |
| 5k | Spatial | correct | 1.414115 | 0.475171 | 0.051831 |
| 5k | P1 | no RGBD | 1.628102 | 0.422801 | 0.051002 |
| 5k | Spatial | no RGBD | 1.627915 | 0.420109 | 0.051011 |
| 5k | P1 | wrong camera | 1.410710 | 0.389552 | 0.052146 |
| 5k | Spatial | wrong camera | 1.431700 | 0.486788 | 0.052312 |
| 10k | P1 | correct | 1.400133 | 0.983819 | 0.051093 |
| 10k | Spatial | correct | 1.387389 | 1.390194 | 0.050975 |
| 10k | P1 | no RGBD | 1.563565 | 0.402389 | 0.050786 |
| 10k | Spatial | no RGBD | 1.566697 | 0.398636 | 0.050820 |
| 10k | P1 | wrong camera | 1.343020 | 0.409151 | 0.051796 |
| 10k | Spatial | wrong camera | 1.347228 | 0.545437 | 0.051700 |
| 30k | P1 | correct | 1.174828 | 0.420556 | 0.051751 |
| 30k | Spatial | correct | 1.199774 | 0.584882 | 0.052053 |
| 30k | P1 | no RGBD | 1.392284 | 0.404662 | 0.050395 |
| 30k | Spatial | no RGBD | 1.398719 | 0.396554 | 0.050379 |
| 30k | P1 | wrong camera | 1.215284 | 0.409186 | 0.051806 |
| 30k | Spatial | wrong camera | 1.279671 | 0.838185 | 0.051075 |

### Scene7playing

| Step | System | Condition | Audio total | LRE error dB | Waveform L1 |
|---|---|---|---:|---:|---:|
| 5k | P1 | correct | 0.172491 | 0.474940 | 0.015722 |
| 5k | Spatial | correct | 0.171336 | 0.385140 | 0.015903 |
| 5k | P1 | no RGBD | 0.175715 | 0.267257 | 0.016380 |
| 5k | Spatial | no RGBD | 0.181197 | 0.302427 | 0.016448 |
| 5k | P1 | wrong camera | 0.274335 | 1.700481 | 0.015886 |
| 5k | Spatial | wrong camera | 0.215862 | 0.961703 | 0.015947 |
| 10k | P1 | correct | 0.146171 | 0.263449 | 0.015258 |
| 10k | Spatial | correct | 0.166617 | 0.383266 | 0.015935 |
| 10k | P1 | no RGBD | 0.172251 | 0.285851 | 0.016269 |
| 10k | Spatial | no RGBD | 0.177852 | 0.301068 | 0.016374 |
| 10k | P1 | wrong camera | 0.251301 | 1.519282 | 0.015553 |
| 10k | Spatial | wrong camera | 0.228528 | 1.148084 | 0.016054 |
| 30k | P1 | correct | 0.130710 | 0.218987 | 0.015067 |
| 30k | Spatial | correct | 0.138520 | 0.252161 | 0.015182 |
| 30k | P1 | no RGBD | 0.163706 | 0.312136 | 0.016035 |
| 30k | Spatial | no RGBD | 0.164994 | 0.273957 | 0.016129 |
| 30k | P1 | wrong camera | 0.260348 | 1.732244 | 0.015422 |
| 30k | Spatial | wrong camera | 0.215206 | 1.140864 | 0.015528 |

## 30k condition causality

The paired mean delta below is `correct - ablated`. Negative is desirable for
lower-is-better metrics. Win rate is the fraction of samples on which the
correct condition is better than the ablated condition.

| Scene | System | Contrast | Audio delta | Audio win | LRE delta | LRE win |
|---|---|---|---:|---:|---:|---:|
| scene1 | P1 | correct vs wrong camera | -0.040456 | 57.7% | +0.011370 | 43.8% |
| scene1 | Spatial | correct vs wrong camera | -0.079897 | 56.2% | -0.253304 | 56.2% |
| scene1 | P1 | RGBD on vs off | -0.217456 | 76.2% | +0.015893 | 46.2% |
| scene1 | Spatial | RGBD on vs off | -0.198946 | 69.2% | +0.188328 | 36.9% |
| Scene7 | P1 | correct vs wrong camera | -0.129638 | 100.0% | -1.513257 | 100.0% |
| Scene7 | Spatial | correct vs wrong camera | -0.076686 | 97.6% | -0.888703 | 97.3% |
| Scene7 | P1 | RGBD on vs off | -0.032995 | 99.0% | -0.093149 | 78.2% |
| Scene7 | Spatial | RGBD on vs off | -0.026474 | 99.7% | -0.021796 | 59.7% |

Spatial P1 improves the scene1 wrong-camera LRE distinction, but:

- its scene1 correct-camera LRE is itself worse;
- scene1 RGBD causality weakens;
- both wrong-camera and RGBD causality weaken in Scene7.

Therefore the stronger scene1 wrong-camera gap is not evidence of a general
spatial-audio improvement.

## Secondary spectral metrics at 30k correct condition

| Scene | System | Mono LSD | Diff LSD |
|---|---|---:|---:|
| scene1 | P1 | 0.944733 | 1.035957 |
| scene1 | Spatial | 0.946622 | 1.057800 |
| Scene7 | P1 | 0.943969 | 1.202784 |
| Scene7 | Spatial | 0.947562 | 1.203144 |

Spatial P1 is worse on all four 30k spectral comparisons.

## Decision

Do not replace the original P1 objective with this spatial objective.

The direct LRE/ILD/IPD/difference supervision produces useful early
optimization pressure, but the benefit does not generalize through 30k:

1. correct-condition 30k audio and LRE are worse in both scenes;
2. direct paired win rates at 30k are below 50% for audio total and LRE in both
   scenes;
3. Scene7, which had the strongest original camera causality, becomes less
   camera-discriminative;
4. the scene1 LRE trajectory is unstable between 5k and 10k.

The next experiment should not be a simple weight increase. A better-scoped
follow-up is to use the spatial term only as an early, decaying auxiliary
objective and remove it before the original P1 reaches its strong 30k regime.
The 5k checkpoints provide direct evidence for this hypothesis, but it still
requires a controlled schedule ablation.


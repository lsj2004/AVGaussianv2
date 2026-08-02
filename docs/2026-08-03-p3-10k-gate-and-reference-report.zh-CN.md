# P3 10k 门禁与 Source/Mono/Audio-only 参考对比

更新日期：2026-08-03

状态：P3 10k 已完成并通过独立复核；30k、causal、多 seed 与 update-matched
Audio-only 对照尚未完成。本文是中期报告，不是最终冠军结论。

执行更新：首条 30k continuation 遇到新的外部 GPU PID 后，所有权 watchdog 已在
精确 18k checkpoint 安全停止。恢复 supervisor、后续 causal/multi-seed relay 及
Audio-only 公平基线 relay 均已重新排队，不会复用受污染的非精确进度。

## 1. 结论摘要

1. `cross_attention_masks/lambda_lre=0.01` 与
   `query_dependent_p1/lambda_lre=0.02` 均通过 10k 架构内门禁，继续到 30k。
2. 10k 时，query P1/0.02 相对本架构 lambda=0 control 的 scene-macro
   paper LRE 降低 38.18%，其余重建指标满足经验噪声感知 guardrail；这是当前更强的
   综合候选。
3. cross-attention/0.01 的架构内 paper LRE 改善为 10.07%，但其 treatment 自身从
   5k 到 10k 的 scene1 LRE 明显变差，因此必须等 30k 和逐场景结果，不能只看
   treatment/control 的宏平均。
4. Source Binaural 与 Mono 已正式纳入共同输出指标表，但它们不是训练模型，不进入
   “公平架构排行榜”。Mono 的通道对称性会天然压低部分 LRE/ILD/IPD 数值。
5. 当前 Audio-only 只有 5k 正式结果。10k 候选与 Audio-only 5k 的数字只能作描述性
   参照；最终报告必须补齐相同 update、scene、seed 和 evaluator 的 Audio-only 对照。

所有下表指标均为两个场景等权 scene macro，且均为越低越好。

## 2. 证据与可复现边界

正式训练代码冻结在 clean commit
`98d158e4e9f1389aa924422faef901b2673ca3bc`。P3 10k 使用原 5k checkpoint 原地
continuation，不是重新初始化训练。

| 证据 | SHA-256 |
|---|---|
| P3 10k manifest | `596a6fc2bef3c1c79a99ff3f2994a4a94313f767493a2272cc18e17bc55257e4` |
| P3 10k gate | `9bf310ce014de90fb6a6bc856dc85c2d8049aa633a57302b7c08a3d143339196` |
| test-retest noise report | `963dd195487654664e125d8a9d802d5223563fd30de37a389e16e1f15ac0aaac` |
| fair baseline report | `920364e134896d2e495ba8985936d5798048bc5b4b2ed763604569db3e722eb0` |
| Source/Mono aggregate | `d10e68a5f90df29c538e66fdc889b4e3be46b01fd4b530af4effdce7dcf3e5e6` |

10k manifest 包含 2 架构 × 2 权重（control/treatment）× 2 场景 = 8 个
continuation；8/8 均达到精确 10k checkpoint 并通过独立 evaluation verifier。

## 3. 严格公平的 10k 架构内比较

本表只比较同一架构、相同 10k update、相同 seed42、相同两个场景、相同 sample 集和
相同 evaluator 的 control/treatment。

| 架构 | lambda | Audio total | Waveform | MAG | ENV | DPAM | LRE | ILD | IPD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-attention masks | 0 | 0.864450 | 0.038075 | 0.169815 | 0.076022 | 0.197714 | 1.429911 | 7.254996 | 0.587835 |
| Cross-attention masks | 0.01 | **0.698441** | **0.037267** | **0.161032** | 0.076531 | **0.189913** | **1.285877** | **7.063714** | **0.550463** |
| Query-dependent P1 | 0 | 0.815716 | **0.033110** | 0.155442 | 0.078782 | 0.235559 | 0.754178 | 6.038957 | **0.472309** |
| Query-dependent P1 | 0.02 | **0.737075** | 0.033158 | **0.152412** | **0.077246** | **0.229058** | **0.466237** | **6.001998** | 0.477682 |

### 3.1 treatment 相对同架构 control 的变化

负值表示误差降低；正值表示误差增大。

| 候选 | Audio total | Waveform | MAG | ENV | DPAM | LRE | ILD | IPD | 门禁 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cross-attention/0.01 | -19.20% | -2.12% | -5.17% | +0.67% | -3.95% | -10.07% | -2.64% | -6.36% | 通过 |
| Query P1/0.02 | -9.64% | +0.15% | -1.95% | -1.95% | -2.76% | -38.18% | -0.61% | +1.14% | 通过 |

该结论只回答“加入 LRE 约束后，同一架构是否改善”，不回答“哪个架构已经超过
Audio-only”。

## 4. 5k 到 10k 的逐场景纵向轨迹

| 架构/权重 | 场景 | LRE@5k | LRE@10k | 10k−5k | 判断 |
|---|---|---:|---:|---:|---|
| Cross-attention/0 | Scene7 | 0.916681 | 0.719460 | -0.197221 | 改善 |
| Cross-attention/0 | scene1 | 2.037805 | 2.140363 | +0.102558 | 退化 |
| Cross-attention/0.01 | Scene7 | 0.780524 | 0.804654 | +0.024129 | 轻微退化 |
| Cross-attention/0.01 | scene1 | 1.121956 | 1.767101 | +0.645145 | 明显退化 |
| Query P1/0 | Scene7 | 0.300282 | 0.337963 | +0.037681 | 退化 |
| Query P1/0 | scene1 | 1.430670 | 1.170392 | -0.260278 | 改善 |
| Query P1/0.02 | Scene7 | 0.380260 | 0.257853 | -0.122407 | 改善 |
| Query P1/0.02 | scene1 | 0.636666 | 0.674621 | +0.037955 | 轻微退化 |

宏平均会隐藏两个场景方向相反的问题。Query P1/0.02 在 10k 的两场景 LRE 更均衡；
cross-attention/0.01 虽通过架构内门禁，但绝对收敛轨迹不稳定。

## 5. Source、Mono、native AudioGS 与模型的绝对参考表

本表统一使用七个共同输出指标。Source/Mono 没有训练 loss；native AudioGS 的两场景
训练更新数为 2,318/6,954；Audio-only 是 5k，而两个 P3 候选是 10k。因此本表是
metric-matched 的绝对参照，不是 update-matched 排名。

| 方法 | Waveform | MAG | ENV | DPAM | LRE | ILD | IPD | 比较边界 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Source Binaural | 0.039082 | 0.187144 | 0.086367 | 0.191029 | 0.902396 | 6.712646 | 0.515112 | 输入参考，非预测模型 |
| Mono | 0.036714 | 0.173993 | 0.081296 | 0.191029 | 0.428337 | 4.993183 | 0.446930 | 通道对称参考 |
| native AudioGS | 0.033913 | 0.166064 | 0.081180 | 0.248325 | 0.445566 | 5.864855 | 0.504390 | 非 update-matched |
| Audio-only/0，5k | 0.033675 | 0.162428 | 0.080271 | 0.247448 | 0.337543 | 5.835723 | 0.505838 | 当前主基线，仅 5k |
| Cross-attention/0.01，10k | 0.037267 | 0.161032 | 0.076531 | 0.189913 | 1.285877 | 7.063714 | 0.550463 | P3 候选 |
| Query P1/0.02，10k | 0.033158 | 0.152412 | 0.077246 | 0.229058 | 0.466237 | 6.001998 | 0.477682 | P3 候选 |

### 5.1 P3 候选相对参考的绝对差值

差值定义为“候选 − 参考”；负值表示候选误差更低。这里仍然不是 update-matched
显著性检验。

| 候选 | 参考 | ΔWaveform | ΔMAG | ΔENV | ΔDPAM | ΔLRE | ΔILD | ΔIPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Query P1/0.02 10k | Source | -0.005924 | -0.034732 | -0.009122 | +0.038028 | -0.436159 | -0.710648 | -0.037430 |
| Query P1/0.02 10k | Mono | -0.003556 | -0.021581 | -0.004050 | +0.038029 | +0.037900 | +1.008815 | +0.030752 |
| Query P1/0.02 10k | Audio-only 5k | -0.000517 | -0.010016 | -0.003025 | -0.018391 | +0.128694 | +0.166275 | -0.028156 |
| Cross-attention/0.01 10k | Source | -0.001816 | -0.026111 | -0.009836 | -0.001116 | +0.383482 | +0.351068 | +0.035351 |
| Cross-attention/0.01 10k | Mono | +0.000552 | -0.012961 | -0.004764 | -0.001116 | +0.857541 | +2.070531 | +0.103533 |
| Cross-attention/0.01 10k | Audio-only 5k | +0.003591 | -0.001395 | -0.003740 | -0.057535 | +0.948335 | +1.227991 | +0.044625 |

Query P1/0.02 已在 waveform、MAG、ENV、DPAM、IPD 上低于 Audio-only 5k，但其 LRE
仍高 0.128694dB、ILD 仍高 0.166275dB。由于训练步数不同，不能据此宣布总体超过或
未超过 Audio-only。

Mono 的 LRE/ILD/IPD 不能单独作为空间正确性的目标：把左右通道做成相同会消除大量
双耳差异，可能获得较低误差，却同时丢失方向线索。最终判断必须联合重建质量、空间
指标、逐场景结果和 causal 评估。

## 6. 剩余闭环

最终结论发布前必须完成：

1. 两个候选及各自 lambda=0 control 的 30k continuation；
2. 条件模型 main/no-RGBD/wrong-camera 或 shuffled-RGBD 的 causal 门禁；
3. Audio-only/0 的 seed42 5k→10k→30k 同 checkpoint continuation；
4. 最终候选、同架构 control 与 Audio-only 的 seed17/42/73、两场景 30k 对齐比较；
5. across-seed mean/std 与 95% paired hierarchical bootstrap CI；
6. 最终报告保持两张主表：严格公平模型榜，以及 Source/Mono/native 的绝对参考榜。

在这些证据齐全前，当前最强表述只能是：Query P1/0.02 是 10k 阶段综合趋势最好的
条件候选；尚未证明它在全部指标或统计意义上超过 Audio-only。

Audio-only 两份正式 manifest 已生成并通过冻结 loader：seed42 confirmation 为
2 runs，SHA-256
`5ff107105afdb699fe844d36f76a1a02b44bfb640686cd063069a2c624b1ca2b`；seed17/73
robustness 为 4 runs，SHA-256
`503c711a1560411e2642cf44e25d0674c496d6769fc5b0212283a4a08c8012b0`。它们仍属于
待执行证据，不能因 manifest 已就绪而标记为实验完成。

# Query-dependent P1：cam38/30k 正式实验报告

## 1. 结论摘要

本次实验完成了两个场景、三个 checkpoint、三种条件系统，共 18 组
held-out cam38 评估：

- 训练相机：cam00–cam37；
- 测试相机：cam38；
- seed：42；
- 训练预算：2,000-step warmup + 30,000-step main；
- checkpoint：5k、10k、30k；
- 系统：correct、no-RGBD、wrong-camera；
- Scene1：130 个测试窗口；
- Scene7：293 个测试窗口。

主要结论：

1. P1 在 Scene1 获得当前项目内最低的 `audio_total=1.1748`，相对 FiLM
   降低 3.89%，同时把 FiLM 的 LRE 从 2.4659 dB 降到 0.4206 dB。
2. P1 在 Scene7 得到 `audio_total=0.1307`、`LRE=0.2190 dB`，不是任一
   单指标最优，但处于 audio-total/LRE 的强 Pareto 区域。
3. RGBD 条件在两个场景都显著降低 30k audio total；Scene7 同时获得明确
   LRE 收益，Scene1 的 LRE 则没有受益。
4. wrong-camera 因果性在 Scene7 极强：correct 在 audio total 和 LRE 上
   对 293/293 样本全部获胜。Scene1 只在 audio total 上有较弱优势，LRE
   方向仍然错误。
5. 因此，query-dependent 几何对应是有效方向，但当前 camera contrast
   仍具有明显场景依赖，不能宣称已经得到稳定、跨场景的相机几何泛化。

## 2. P1 checkpoint 曲线

### 2.1 Scene1 Opera

| Step | 条件 | Audio total ↓ | LRE dB ↓ | Wave L1 ↓ |
|---:|---|---:|---:|---:|
| 5k | correct | 1.4500 | 1.0505 | 0.05154 |
| 5k | no-RGBD | 1.6281 | 0.4228 | 0.05100 |
| 5k | wrong-camera | 1.4107 | 0.3896 | 0.05215 |
| 10k | correct | 1.4001 | 0.9838 | 0.05109 |
| 10k | no-RGBD | 1.5636 | 0.4024 | 0.05079 |
| 10k | wrong-camera | 1.3430 | 0.4092 | 0.05180 |
| 30k | **correct** | **1.1748** | 0.4206 | 0.05175 |
| 30k | no-RGBD | 1.3923 | **0.4047** | **0.05040** |
| 30k | wrong-camera | 1.2153 | 0.4092 | 0.05181 |

Scene1 的条件分支在 10k 后才发生关键收敛：correct 的 audio total 从
10k 到 30k 再降低 16.1%，LRE 降低 57.3%。但 30k correct 的 LRE 仍略差于
no-RGBD 和 wrong-camera。

### 2.2 Scene7 Playing

| Step | 条件 | Audio total ↓ | LRE dB ↓ | Wave L1 ↓ |
|---:|---|---:|---:|---:|
| 5k | correct | **0.1725** | 0.4749 | **0.01572** |
| 5k | no-RGBD | 0.1757 | **0.2673** | 0.01638 |
| 5k | wrong-camera | 0.2743 | 1.7005 | 0.01589 |
| 10k | **correct** | **0.1462** | **0.2634** | **0.01526** |
| 10k | no-RGBD | 0.1723 | 0.2859 | 0.01627 |
| 10k | wrong-camera | 0.2513 | 1.5193 | 0.01555 |
| 30k | **correct** | **0.1307** | **0.2190** | **0.01507** |
| 30k | no-RGBD | 0.1637 | 0.3121 | 0.01603 |
| 30k | wrong-camera | 0.2603 | 1.7322 | 0.01542 |

Scene7 从 10k 开始形成方向一致的视觉和相机收益，并在 30k 进一步增强。

## 3. 30k 因果消融

以下 paired delta 定义为 `correct - comparison`；对这些误差指标，负数表示
correct 更好。

### 3.1 Scene1

| 对比 | Audio-total delta | Win rate | LRE delta | LRE win rate |
|---|---:|---:|---:|---:|
| correct vs no-RGBD | -0.2175 | 76.15% | +0.0159 dB | 46.15% |
| correct vs wrong-camera | -0.0405 | 57.69% | +0.0114 dB | 43.85% |

Scene1 证明 RGBD 能改善总体音频重建，但没有证明正确相机能改善 LRE。
camera contrast 在该场景只学到较弱的总体重建偏好。

### 3.2 Scene7

| 对比 | Audio-total delta | Win rate | LRE delta | LRE win rate |
|---|---:|---:|---:|---:|
| correct vs no-RGBD | -0.0330 | 98.98% | -0.0931 dB | 78.16% |
| correct vs wrong-camera | -0.1296 | 100.00% | -1.5133 dB | 100.00% |

Scene7 的结论很强：RGBD 有效，正确相机有效，而且 wrong-camera 对全部
293 个样本都造成 audio-total 和 LRE 退化。

## 4. 与 AVGaussianFusionv2 内部历史分支比较

所有结果均为相同 cam38、seed 42、30k update 协议。

### 4.1 Scene1

| 方法 | Audio total ↓ | LRE dB ↓ | Wave L1 ↓ |
|---|---:|---:|---:|
| native AudioGS | 1.7182 | 0.6380 | 0.05128 |
| audio-only | 1.4102 | 0.4207 | **0.05059** |
| FiLM + U-Net | 1.2224 | 2.4659 | 0.05487 |
| direct conditioned U-Net | 1.4974 | 5.3492 | 0.05377 |
| gated native residual | 1.5749 | 5.1284 | 0.05818 |
| aligned-mask cross-attention | 1.3334 | 1.7871 | 0.06044 |
| Gaussian-token cross-attention v3 | 1.4074 | **0.4049** | 0.05063 |
| **query-dependent P1** | **1.1748** | 0.4206 | 0.05175 |

P1 的 Scene1 audio total 排名第一；LRE 与 audio-only 基本相同，略逊于
Gaussian-token v3。与 Gaussian-token v3 不同，P1 的 no-RGBD 消融有明显
audio-total 退化，说明视觉条件形成了真实因果贡献。

### 4.2 Scene7

| 方法 | Audio total ↓ | LRE dB ↓ | Wave L1 ↓ |
|---|---:|---:|---:|
| native AudioGS | 0.1817 | 0.2531 | 0.01654 |
| audio-only | 0.1529 | 0.2130 | 0.01598 |
| FiLM + U-Net | **0.1237** | 0.5792 | 0.01527 |
| direct conditioned U-Net | 0.1828 | 1.2909 | 0.01636 |
| gated native residual | 0.1264 | 0.3128 | **0.01456** |
| aligned-mask cross-attention | 0.1404 | 0.7366 | 0.01550 |
| Gaussian-token cross-attention v3 | 0.1521 | **0.2059** | 0.01606 |
| **query-dependent P1** | 0.1307 | 0.2190 | 0.01507 |

P1 的 Scene7 audio total 排名第三、LRE 排名第三、Wave L1 排名第二。FiLM
和 gated residual 的 audio total 略低，但空间误差明显更高；Gaussian-token
的 LRE 略低，但 audio total 和 Wave L1 更差。

### 4.3 双场景综合

按场景等权宏平均：

| 方法 | Macro audio total ↓ | Macro LRE dB ↓ |
|---|---:|---:|
| FiLM + U-Net | 0.6731 | 1.5226 |
| audio-only | 0.7816 | 0.3169 |
| Gaussian-token v3 | 0.7798 | **0.3054** |
| **query-dependent P1** | **0.6528** | 0.3198 |

P1 得到最低的双场景 macro audio total；macro LRE 与 audio-only 只差约
0.003 dB。它是当前分支里最均衡的总体重建/空间定位折中，但不是每个场景、
每个单项指标都最优。

## 5. 相对 FiLM 的逐样本结果

| 场景 | Audio-total delta | Win rate | LRE delta | LRE win rate | Wave-L1 delta |
|---|---:|---:|---:|---:|---:|
| Scene1 | -0.0476 | 70.00% | -2.0453 dB | 94.62% | -0.00312 |
| Scene7 | +0.0070 | 55.29% | -0.3602 dB | 89.42% | -0.00021 |

Scene7 的 mean audio total 略逊于 FiLM，但 P1 的逐样本 win rate 仍为
55.29%，表明少量较大退化样本拉高了均值；LRE 和 waveform 则稳定优于 FiLM。

## 6. 限制与下一步

当前 P1 仍有三个明确问题：

1. **Scene1 相机空间因果性失败。** correct 的 audio total 优于
   wrong-camera，但 LRE 没有改善；相机对比目标和最终空间指标没有完全对齐。
2. **差分频谱仍有权衡。** Scene7 30k correct 的 diff-LSD 为 1.2028，
   高于 no-RGBD 的 1.1859 和 wrong-camera 的 1.1099；模型可能通过更好的
   单声道重建和能量关系获得 audio-total/LRE 优势，但没有改善所有双耳频谱指标。
3. **只有两个场景和单 seed。** 当前结果足以进行项目内方法判断，但不足以
   声称跨场景统计泛化。

建议下一步优先级：

1. 保留当前 P1 作为新的主干候选；
2. 不增加 wrong-time，先修正 camera contrast，使其直接约束 LRE、ILD/IPD
   或条件预测差异，而不只对总 audio loss 做 margin；
3. 针对 Scene1 检查相机基线分布、同帧负相机距离和视觉几何可辨识度；
4. 增加至少两个 seed，重点复核 Scene1 correct-vs-wrong 的方向；
5. 在不牺牲 Scene7 因果性的前提下，再优化 diff-LSD。

## 7. 原始证据

- Scene1 report：
  `runs/query_dependent_p1_cam38/scene1_opera/report/current.json`
- Scene7 report：
  `runs/query_dependent_p1_cam38/Scene7playing/report/current.json`
- Scene1 evaluations：
  `runs/query_dependent_p1_cam38/scene1_opera/evaluations/`
- Scene7 evaluations：
  `runs/query_dependent_p1_cam38/Scene7playing/evaluations/`

两个报告分别覆盖 130 和 293 个 held-out 样本；所有 correct/no-RGBD/
wrong-camera 评估复用同一 checkpoint。

# P2 架构、参数与 LRE 权重实验报告

状态：**P2 已完成**。本报告只总结 5k / seed42 的 P2 阶段证据；进入
P3 的配置是候选而不是最终冠军。

实验代码冻结在提交
`98d158e4e9f1389aa924422faef901b2673ca3bc`。正式矩阵包含
4 个架构 × 4 个 `lambda_lre` × 2 个场景，共 `32/32` 个 run；另有
4 个架构 × 2 个场景的独立 lambda=0 test-retest，共 `8/8` 个 run。
全部 run、evaluation、样本顺序和指标协议均通过独立验证。

## 1. 这轮实验回答什么

P2 分开回答两个问题：

1. 在 `lambda_lre=0`、相同 5k update、相同 seed 和 evaluator 下，四个
   架构的音频、空间与 RGB 表现有什么差异；
2. 对每个架构分别加入 `lambda_lre∈{0.01,0.02,0.05}` 后，相对该架构
   自己的 lambda=0 control 是否产生超过经验噪声的稳定收益。

第二个问题不能用 `audio_only` 作为所有架构的 control。不同架构的
lambda 效应必须和同架构、同场景的 lambda=0 逐样本配对。

主聚合口径为两个场景等权的 scene macro；Scene7 的 293 个样本不能因数量
更多而压过 scene1 的 130 个样本。423 样本 micro 只作补充。误差指标越低
越好，PSNR/SSIM 越高越好。

## 2. 统一实验协议

| 项目 | 固定值 |
|---|---|
| 场景 | `scene1_opera`、`Scene7playing` |
| 训练/评估相机 | cam00–cam37 / cam38 |
| held-out 样本 | 130 + 293 = 423 |
| seed | 42 |
| conditioner warmup | 2,000 updates |
| P2 主训练 | exact 5,000 updates |
| batch / crop | 1 / 0.5 s |
| AudioGS 类 | `Audio3DGSMonoDiffGSOnly` |
| sample rate | 16 kHz |
| condition map | 94 × 166，embedding dim 128 |
| audio / condition / visual LR | 1e-4 / 1e-4 / 1e-5 |
| loss | `lambda_audio=1`、`lambda_rgb=1`、`lambda_dssim=0.2` |
| visual anchor | `1e-4` |
| LRE | sign-sensitive，scale 6 dB，Smooth-L1 beta 1 |

所有 P2 cell 共享相同 native 初始化、ordered sample inventory、5k index
sequence、metric directions 和 DPAM 权重。允许变化的字段只有架构定义和
`train.lambda_lre`。

## 3. 四个架构

`plain_unet` 已在前一阶段淘汰，不进入 P2。

| system | worker mode | 条件/融合机制 | 关键配置 | signature |
|---|---|---|---|---|
| `audio_only` | audio_only | GS-only AudioGS，无视觉条件 | native residual | `f579d8f465bc` |
| `query_dependent_p1` | joint_conditioned | query/geometry-dependent complex cross attention | 2 layers、4 heads、F/T patch 8/2、geometry rank 16、gate 0.01 | `266174006c39` |
| `joint_conditioned` | joint_conditioned | FiLM-conditioned AudioGS U-Net | embedding 128 | `d7efff407e81` |
| `cross_attention_masks` | joint_conditioned | 音频 feature-mask cross attention | 4 layers、4 heads、F/T patch 16/4、gate 0.01、dropout 0 | `47fc58f3c221` |

P1 还固定使用 STFT `n_fft=512 / hop=160 / win=400`，最大 log-magnitude
和 phase 修正分别为 0.15/0.25，additive scale 0.01，camera contrast
weight/margin 为 0.5/0.05。

## 4. 实验时真实参数语义

下表来自真实 CUDA checkpoint 和 Adam state，而不是仅统计
`requires_grad=True`。单位为 parameter elements。

| system | total | declared trainable | optimizer declared | Adam active | optimizer dormant | orphan trainable | frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| audio_only | 42,653,534 | 9,469,474 | 9,469,474 | 1,603,680 | 7,865,794 | 0 | 33,184,060 |
| query_dependent_p1 | 42,912,908 | 42,912,908 | 35,047,114 | 35,047,114 | 0 | 7,865,794 | 0 |
| joint_conditioned | 42,653,534 | 42,653,534 | 42,653,534 | 42,653,534 | 0 | 0 | 0 |
| cross_attention_masks | 35,559,708 | 35,559,708 | 35,559,708 | 35,559,708 | 0 | 0 | 0 |

解释：

- audio-only 的保存 U-Net 被列入 optimizer，但 GS-only condition=None
  forward 不使用它，因此没有梯度或 Adam state；实际更新的是 1,603,680
  个 acoustic 参数；
- query P1 的保存上游 U-Net 被标记为 trainable，但不在 optimizer 中，也
  不参与 P1 forward；实际更新的是 35,047,114 个参数；
- 这两个问题会在正式实验结束后清理，并用一步更新 bitwise equivalence
  证明不改变本轮已训练模型的数值更新；
- 因此本轮是相同数据、update 数和 evaluator 下的系统级公平比较，但不是
  parameter-count-matched 比较。不能把更大模型的收益解释为纯架构效率收益。

修复后预期 active/frozen 分别为：audio-only
`1,603,680 / 41,049,854`，query P1
`35,047,114 / 7,865,794`；最终报告会用修复后真实 CUDA 审计替换“预期”。

## 5. lambda=0 架构横向结果

以下均为两个场景等权 macro。

| system | audio total ↓ | waveform L1 ↓ | MAG ↓ | ENV ↓ | DPAM ↓ | paper LRE dB ↓ | ILD dB ↓ | IPD rad ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audio_only | 0.898737 | 0.033675 | 0.162428 | 0.080271 | 0.247448 | **0.337543** | **5.835723** | 0.505838 |
| query_dependent_p1 | 0.868384 | **0.033540** | **0.159269** | 0.079678 | 0.236640 | 0.865476 | 5.974141 | **0.471407** |
| joint_conditioned | 0.944582 | 0.034252 | 0.164525 | 0.078555 | 0.210722 | 2.497107 | 7.323302 | 0.535390 |
| cross_attention_masks | **0.736041** | 0.036901 | 0.168562 | **0.077098** | **0.199307** | 1.477243 | 7.207063 | 0.570948 |

不能从这张表选出一个“所有指标都最好”的架构：

- `audio_only` 的 LRE 和 ILD 最稳，是空间误差的强 control；
- `query_dependent_p1` 是 lambda=0 时最平衡的条件架构：audio total、
  waveform、MAG、DPAM 和 IPD 优于 audio-only，但 paper LRE 明显更差；
- `cross_attention_masks` 的 audio total、ENV、DPAM 最好，但 waveform、MAG
  和全部主要空间指标明显退化；
- `joint_conditioned` 改善 ENV/DPAM，但 audio total 和空间误差没有形成
  有竞争力的整体折中。

### 5.1 lambda=0 分场景结果

| system | scene | audio total ↓ | MAG ↓ | DPAM ↓ | paper LRE dB ↓ |
|---|---|---:|---:|---:|---:|
| audio_only | scene1 | 1.621367 | 0.230962 | 0.324198 | **0.402580** |
| audio_only | Scene7 | 0.176108 | 0.093894 | 0.170699 | **0.272505** |
| query P1 | scene1 | 1.577354 | 0.228009 | 0.309973 | 1.430670 |
| query P1 | Scene7 | **0.159414** | **0.090530** | 0.163306 | 0.300282 |
| joint | scene1 | 1.720018 | 0.238932 | 0.283994 | 4.531749 |
| joint | Scene7 | 0.169147 | 0.090119 | 0.137450 | 0.462465 |
| cross-attention | scene1 | **1.279935** | 0.243359 | **0.267461** | 2.037805 |
| cross-attention | Scene7 | 0.192146 | 0.093765 | **0.131154** | 0.916681 |

分场景表说明 scene macro 并没有掩盖一个方向完全相反的架构结论：条件模型的
空间 LRE 问题在 scene1 尤其明显；同时 query P1 在 Scene7 的重建指标最有竞争力。

### 5.2 相对 audio_only 的逐样本配对效果

win 是方向感知后的逐样本胜率；两个场景先分别计算，再等权平均。

| system | Δ audio total | win | Δ paper LRE | win | Δ DPAM | win |
|---|---:|---:|---:|---:|---:|---:|
| query_dependent_p1 | -0.030354 | 72.8% | +0.527933 | 27.2% | -0.010809 | 73.8% |
| joint_conditioned | +0.045845 | 61.6% | +2.159564 | 12.4% | -0.036727 | 78.4% |
| cross_attention_masks | -0.162697 | 51.3% | +1.139700 | 14.1% | -0.048141 | 82.3% |

均值和胜率回答的问题不同。例如 cross-attention 的 audio total 均值改善很大，
但胜率只有 51.3%，说明收益可能集中在部分样本；最终报告必须同时保留均值、
median delta 和 win rate。

### 5.3 RGB 指标

只报告真实输出 RGB 的条件架构；audio-only 不填零也不做插值。

| system | RGB L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| query_dependent_p1 | 0.102392 | 16.854656 | **0.389416** |
| joint_conditioned | 0.101784 | 16.814332 | 0.387115 |
| cross_attention_masks | **0.101660** | **16.863532** | 0.388893 |

三者差异很小，且当前还缺 native FTGS++ 的相同样本描述性参考。因此不能据此
宣称某个音频条件架构改善了视觉重建；native FTGS++ 评测将在 P3 core 后补齐。

## 6. 每个架构内部的 lambda_lre sweep

星号表示经验噪声校正后进入 10k 的 treatment，不表示最终冠军。

| system | lambda | audio total ↓ | waveform L1 ↓ | MAG ↓ | ENV ↓ | DPAM ↓ | paper LRE dB ↓ | LRE paired win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audio_only | 0.00 | 0.898737 | 0.033675 | 0.162428 | 0.080271 | 0.247448 | 0.337543 | 50.0% |
| audio_only | 0.01 | 0.898758 | 0.033676 | 0.162429 | 0.080272 | 0.247447 | 0.337594 | 45.6% |
| audio_only | 0.02 | 0.898929 | 0.033676 | 0.162431 | 0.080274 | 0.247460 | 0.339448 | 47.7% |
| audio_only | 0.05 | 0.899069 | 0.033677 | 0.162434 | 0.080276 | 0.247447 | 0.340286 | 42.4% |
| query_dependent_p1 | 0.00 | 0.868384 | 0.033540 | 0.159269 | 0.079678 | 0.236640 | 0.865476 | 50.0% |
| query_dependent_p1 | 0.01 | 0.817561 | 0.033772 | 0.157459 | 0.078620 | 0.231201 | 0.713028 | 44.3% |
| query_dependent_p1 | 0.02 * | **0.769633** | **0.033598** | **0.155587** | **0.077655** | 0.227907 | 0.508463 | 60.4% |
| query_dependent_p1 | 0.05 | 0.788424 | 0.033679 | 0.156552 | 0.077933 | **0.226241** | **0.477003** | 66.9% |
| joint_conditioned | 0.00 | 0.944582 | 0.034252 | 0.164525 | 0.078555 | 0.210722 | 2.497107 | 50.0% |
| joint_conditioned | 0.01 | 0.888211 | **0.034120** | 0.159451 | 0.078941 | 0.207664 | **1.003290** | 44.0% |
| joint_conditioned | 0.02 | **0.844959** | 0.034986 | 0.163516 | 0.078046 | **0.207119** | 1.300150 | 77.7% |
| joint_conditioned | 0.05 | 0.864020 | 0.034188 | **0.159381** | **0.077902** | 0.211894 | 1.035353 | 63.7% |
| cross_attention_masks | 0.00 | 0.736041 | 0.036901 | 0.168562 | 0.077098 | 0.199307 | 1.477243 | 50.0% |
| cross_attention_masks | 0.01 * | **0.674539** | **0.036703** | 0.165749 | 0.075696 | 0.199023 | **0.951240** | **86.8%** |
| cross_attention_masks | 0.02 | 0.700546 | 0.036885 | 0.167551 | **0.075336** | 0.200698 | 1.194916 | 50.2% |
| cross_attention_masks | 0.05 | 0.710447 | 0.037291 | **0.165696** | 0.076910 | **0.191155** | 1.177874 | 61.2% |

### 6.1 架构内结论

- **audio_only**：三个正 lambda 的变化都没有超过有意义的经验噪声范围，
  paper LRE 还略微变差。空间约束对该 forward 没有稳定收益；
- **query P1**：0.02 和 0.05 都形成有效折中。0.05 的 LRE/DPAM 更低，
  但 0.02 的 audio total、waveform、MAG、ENV 更好；预注册的重建优先
  tie-break 选择 0.02；
- **joint conditioned**：0.01 将 LRE 从 2.497 降至 1.003，成为该架构代表；
  0.02 的 paired win 更高但宏观 LRE 和 waveform 更差。该代表仍因最多两个
  treatment 的预算上限没有进入 10k；
- **cross-attention masks**：0.01 同时改善 audio total、waveform、MAG、ENV
  和 LRE，且 LRE paired win 86.8%，是该架构最清晰的 5k 候选。

## 7. 与 Source、Mono、native AudioGS 的描述性对比

以下也是两个场景等权 macro，但不是同一种公平性：Source/Mono 没有训练，
native AudioGS 的两场景训练更新分别为 2,318/6,954，并非 5k update-matched。
它们只能作为绝对指标参考，不能和 P2 模型混成单一排行榜。

| reference/system | waveform L1 ↓ | MAG ↓ | ENV ↓ | DPAM ↓ | paper LRE dB ↓ | ILD dB ↓ | IPD rad ↓ | 边界 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Source Binaural | 0.039082 | 0.187144 | 0.086367 | 0.191029 | 0.902396 | 6.712646 | 0.515112 | 输入参考，不是预测模型 |
| Mono | 0.036714 | 0.173993 | 0.081296 | 0.191029 | 0.428337 | 4.993183 | 0.446930 | 通道对称参考，低 LRE 不等于空间正确 |
| native AudioGS | 0.033913 | 0.166064 | 0.081180 | 0.248325 | 0.445566 | 5.864855 | 0.504390 | metric-matched，非 update-matched |
| audio_only 5k | 0.033675 | 0.162428 | 0.080271 | 0.247448 | 0.337543 | 5.835723 | 0.505838 | P2 主 update-matched control |
| query P1 5k, λ=0 | 0.033540 | 0.159269 | 0.079678 | 0.236640 | 0.865476 | 5.974141 | 0.471407 | 架构横向比较 |
| query P1 5k, λ=0.02 | 0.033598 | 0.155587 | 0.077655 | 0.227907 | 0.508463 | 5.878570 | 0.482958 | P2 入选 treatment |
| joint 5k, λ=0 | 0.034252 | 0.164525 | 0.078555 | 0.210722 | 2.497107 | 7.323302 | 0.535390 | 架构横向比较 |
| joint 5k, λ=0.01 | 0.034120 | 0.159451 | 0.078941 | 0.207664 | 1.003290 | 5.807644 | 0.519926 | 架构内 Pareto，后续预算截断 |
| cross-attention 5k, λ=0 | 0.036901 | 0.168562 | 0.077098 | 0.199307 | 1.477243 | 7.207063 | 0.570948 | 架构横向比较 |
| cross-attention 5k, λ=0.01 | 0.036703 | 0.165749 | 0.075696 | 0.199023 | 0.951240 | 7.102835 | 0.566417 | P2 入选 treatment |

`audio_total/audio_mono/audio_diff` 是模型训练损失，Source/Mono 没有这些字段，
不能补零或用于跨边界排名。

这张扩展表给出了 Source/Mono 与 P2 代表 treatment 的直接数值参照：query
P1/0.02 的 waveform、MAG、ENV 均低于 Source 和 Mono，paper LRE 低于 Source
但仍高于 Mono 和 audio-only；cross-attention/0.01 的 ENV 最低、DPAM 最接近
Source/Mono，但 LRE/ILD 仍明显较差。Mono 的左右通道对称性会天然压低部分
能量比误差，因此其低 LRE/ILD 不能解释为空间定位优于双耳预测模型。

## 8. P2 选择结果

每个架构先按其 test-retest 经验噪声做 LRE 改善和重建 guardrail，再形成
架构内 Pareto 代表。三个条件架构代表均进入跨架构 Pareto front：

| system | representative lambda | P3 状态 |
|---|---:|---|
| query_dependent_p1 | 0.02 | 进入 10k |
| joint_conditioned | 0.01 | Pareto，但受最多两个 treatment 上限截断 |
| cross_attention_masks | 0.01 | 进入 10k |

最终进入 P3 的两个 treatment 是：

1. `cross_attention_masks / lambda_lre=0.01`；
2. `query_dependent_p1 / lambda_lre=0.02`。

二者都携带各自 lambda=0 control、两个场景，从同一 5k checkpoint 精确续训
到 10k，共 8 个 continuation。只有通过 10k gate 的候选才继续 30k。

### 8.1 两个入选 treatment 的分场景效果

| system | lambda | scene | audio total ↓ | MAG ↓ | DPAM ↓ | paper LRE dB ↓ |
|---|---:|---|---:|---:|---:|---:|
| cross-attention | 0.00 | scene1 | 1.279935 | 0.243359 | 0.267461 | 2.037805 |
| cross-attention | 0.01 | scene1 | **1.168969** | **0.239091** | **0.265467** | **1.121956** |
| cross-attention | 0.00 | Scene7 | 0.192146 | 0.093765 | **0.131154** | 0.916681 |
| cross-attention | 0.01 | Scene7 | **0.180108** | **0.092407** | 0.132579 | **0.780524** |
| query P1 | 0.00 | scene1 | 1.577354 | 0.228009 | 0.309973 | 1.430670 |
| query P1 | 0.02 | scene1 | **1.373673** | **0.220699** | **0.288773** | **0.636666** |
| query P1 | 0.00 | Scene7 | **0.159414** | 0.090530 | **0.163306** | **0.300282** |
| query P1 | 0.02 | Scene7 | 0.165593 | **0.090475** | 0.167041 | 0.380260 |

cross-attention/0.01 在 5k 的两个场景都降低 LRE 和 audio total；query P1/0.02
的主要收益来自 scene1，而 Scene7 的 audio total、DPAM 和 LRE 均轻微变差。
因此 query P1 必须经过 10k 的两场景门禁，不能只依据 scene macro 晋级到最终结论。

## 9. 当前能够和不能够下的结论

### 可以下的结论

1. “其他架构都不行”是错误表述。lambda=0 时 audio-only 的空间指标最稳，
   但条件架构在重建、感知和部分空间指标上形成不同 Pareto 折中；
2. 对 audio-only 加 LRE loss 没有稳定收益；
3. 条件架构的空间退化可以被 LRE loss 显著修复，且不是单一架构特例；
4. 5k 最值得继续的是 cross-attention/0.01 与 query P1/0.02；joint/0.01
   仍是有效 Pareto 证据，只是未获得后续预算；
5. 评价架构必须同时看重建、空间、感知和逐样本分布，不能只看一个最低均值。

### 还不能下的结论

1. 不能把两个 P3 候选称为最终冠军；
2. 不能证明 5k 排序会持续到 30k；
3. 不能证明 seed42 的效果跨 seed 稳定；
4. 不能把 Source、Mono、native AudioGS 和 5k 模型放入同一个“公平排行榜”；
5. 不能声称参数效率更高，因为四个系统的实际 active 参数量不同；
6. 在 native FTGS++ 相同样本评测完成前，不能下视觉基线结论。

## 10. 权威证据

| 证据 | SHA-256 |
|---|---|
| P2 primary manifest | `e4b450025b14439f64df2b73b2e3d2a3191821399fc662cbc00ddd783db74272` |
| P2 expanded manifest | `5b8215ceaf453bf8a0cf9e765386cb724eb7305a27bbbf9272a66705704eac69` |
| noise report | `963dd195487654664e125d8a9d802d5223563fd30de37a389e16e1f15ac0aaac` |
| empirical selection | `06cee9efa02432d2d79299b7f843b82eabddc2b93ce13df06770d58ff8d78d72` |
| CUDA parameter audit | `85299ea4f04435f05125ed22a10134152fd854651b7bf31e4d962a5819848356` |
| architecture/config report | `dc4bb138457e300098709a7e08757f8c38a0a7261fb5632309faa4a8be6dad6c` |
| fair baseline report | `920364e134896d2e495ba8985936d5798048bc5b4b2ed763604569db3e722eb0` |
| P2 authoritative-table preview JSON | `9fe3c8f6bbd97a10141100165987a634208b7e77a9a9ca0551c1676bbe1b38fa` |
| independent 700-value cross-check script | `9dea955e011131b46f8ab0cbaf4972a03c4b9511aeb6558599e6c30b7cd213b7` |

P2 JSON 逐 cell 重新验证了配置字节、resolved-config origin、continuation
identity、run completion、evaluation content hash、具体 5k checkpoint hash、
130/293 ordered sample IDs、row/summary mean、metric direction 和共同 protocol。
独立交叉检查进一步确认 P2 表、fair baseline 表和 selector 的 700 个共享聚合值
在 `1e-12` 容差内一致。

## 11. 后续更新方式

本文件不会把尚未完成的 P3 数字写成结论。P3 完成后，最终实验计划和总报告将新增：

- treatment/control 的 5k→10k→30k 纵向轨迹；
- 30k same-checkpoint no-RGBD / wrong-camera / shuffled-RGBD 因果评测；
- seed17/42/73 的 scene-equal paired effect 与 95% hierarchical bootstrap CI；
- native FTGS++ 的 RGB L1/PSNR/SSIM 描述性参考；
- 修复后真实参数审计及与冻结实验的一步更新等价证明。

# LRE loss 精简实验计划

## 1. 核心目标

本轮先回答三个问题：

1. GS-only、Plain U-Net、FiLM、Cross-Attention、P1 中哪些架构更好；
2. 胜出架构加 sign-sensitive LRE loss 后，能否降低 LRE，同时不明显损害
   MAG、ENV、DPAM 和原 AudioGS loss；
3. 哪些方法或 loss 权重值得继续训练，哪些可以尽早停止。

`configs/experiments/lre_loss_ablation.yaml` 中的组合是**候选池上限**，不是必须
全部跑完的清单。Agent 应根据中间结果动态淘汰，把算力留给更有希望的配置。
不要直接展开“全部架构 × 全部 loss 权重”的笛卡尔积。

## 2. 服务器与并发

服务器预计有两张空闲的 48 GB GPU。先为每个架构各跑一个短 smoke，记录单进程
峰值显存和吞吐，再决定并发：

- 初始每张卡启动 2 个训练进程；
- 只有估算总峰值低于显存的 80%、没有 OOM，且单任务吞吐下降不超过 20% 时，
  才可提高到每卡 3-4 个；
- 两张卡尽量同时保持有任务，不要串行等待；
- 同一组 control / treatment 尽量分配到相同型号 GPU；
- OOM 时先降低单卡并发，不修改 batch size 或实验语义。

Agent 必须记录每个进程的 GPU、峰值显存、运行时间和失败原因。并发数量应动态
调整，不要求固定。

## 3. 实验优先级

### P0：先建立可比较的参考线

优先完成：

- 计算 `Source Binaural` 和 `Mono` 的 paper MAG / ENV / LRE / DPAM；
- 验证 `lambda_lre=0` 与旧 objective 一致；
- 建立统一架构 runner，保证所有方法使用同一 split、初始化、seed、预算和 evaluator；
- 验证 clean 分支中的 Plain U-Net、Mask Cross-Attention、P1 统一 runner；
- 对所有可运行架构做短程 smoke，确认 loss、梯度和指标有限；
- 检查两张 GPU 的可用显存，并确定每卡并发数。

P0 失败时不启动大规模训练。

架构来源：

| 架构 | 当前来源 | 初始处理 |
|---|---|---|
| GS-only | clean 分支 `audio_only` | 必跑基线 |
| FiLM residual | clean 分支 `joint_conditioned` | 必跑 |
| Plain U-Net | clean 分支 `plain_unet` strategy | 必跑 |
| Mask Cross-Attention | clean 分支 `cross_attention_masks` | 必跑 |
| Query-dependent P1 | clean 分支 `query_dependent_p1` | 必跑主候选 |
| Gaussian-token Cross-Attention | clean 分支 `cross_attention_tokens` | 条件复核 |
| Direct / Gated / Spatial P1 | 已有历史负结果 | 默认淘汰，仅 smoke 异常优秀时恢复 |

Visual-only 不产生音频，不进入这轮音频架构排名。

历史结果只用于安排优先级。它们来自 `visual_time` 修复前或不同分支，不能直接
替代本轮统一 runner 的重评估。

Plain 复用现有 architecture runner，训练 mode 为 `audio_only`，评估名为
`plain_unet`；Mask、P1 与 Gaussian-token 复用同一个 cross-attention runner，
由派生 config 的 `audio_backend` 决定具体实现。Spatial P1 不合入 clean，本轮
先验证原始 P1，避免同时增加架构和额外 spatial objective。

运行入口保持最少：

- Plain：`benchmark_architecture_*`，strategy 为 `plain_unet`；
- Gaussian-token / Mask / P1：`run_cross_attention_cam38.sh`，variant 分别为
  `cross_attention`、`cross_attention_masks`、`query_dependent_p1`；
- LRE 搜索配置必须显式传入 P1 survivor system，生成器不再默认回退到某个架构。

### P1：先做架构赛马

所有架构先固定：

```text
seed: 42
lambda_lre: 0.0
scenes: scene1_opera, Scene7playing
```

首次正式比较统一使用 5k checkpoint，不为更早观察点修改 strict runner。重点排序：

1. MAG / ENV / DPAM、`audio_total` 和 waveform L1；
2. LRE 是否接近或优于 GS-only；
3. 两个场景的趋势是否一致；
4. 视觉方法的 correct RGBD 是否优于 no-RGBD / shuffled-RGBD。

P1 目标是保留最多 1-2 个架构。明显失败或没有使用视觉的配置应及时停止。

### P2：只在胜出架构上搜索 LRE loss

只对 P1 胜出的最多 1-2 个架构测试：

```text
lambda_lre: 0.0, 0.01, 0.02, 0.05
seed: 42
```

- 每个架构最多保留 1-2 个非零权重；
- 先训练到 10k；
- 10k 仍有稳定收益的候选再训练到 30k；
- 输出逐样本指标和 Source Binaural / Mono / control / treatment 对比表。

P2 的目标是选出“架构 + loss 权重”，不是补齐搜索网格。

### P3：最后再做随机种子

多 seed 优先级最低。只有 P2 出现明确候选后才运行：

```text
seeds: 17, 73
configs: 最佳架构的 control + 最佳 LRE 权重
```

如果 P2 没有候选通过保护指标，就不跑多 seed，直接结论为当前 LRE loss 配置
不值得继续。

## 4. 动态淘汰规则

满足任一条件可以停止候选，但必须保存已有 checkpoint、指标和淘汰原因：

- loss、梯度或输出出现非有限值；
- 同一配置连续 OOM，降低并发后仍无法稳定运行；
- 架构在两个场景、连续两个观测点都被更简单方法稳定支配；
- 视觉架构的 correct RGBD 与 no-RGBD / shuffled-RGBD 基本相同；
- LRE loss 配置在两个场景连续两个观测点都没有 LRE 改善；
- LRE 没有改善，同时 MAG、ENV、DPAM、`audio_total` 或 waveform L1 明显变差；
- 候选在主要指标上被另一个更小权重配置稳定支配。

不要只凭一个场景或一次短程波动淘汰。边界不清楚时，多跑一个观测点再决定。

## 5. Agent 停止条件

Agent 可以在以下条件满足后停止本轮实验：

1. P0 参考线和运行环境验证完成；
2. P1 候选均有结果或明确淘汰原因；
3. 入选候选完成 P2，或已确认没有候选值得推进；
4. 只有存在明确最佳候选时才完成 P3 多 seed；
5. 最终报告列出最佳配置、被砍配置、指标对比和下一轮建议；
6. 所有已完成结果通过独立 verifier，没有缺失、重复或 hash 不一致。

本轮不要求机械完成整个候选池。详细统计协议和更多候选配置放到下一轮再补。

# LRE loss 精简实验计划

## 1. 核心目标

本轮先回答三个问题：

1. `Source Binaural`、`Mono`、`audio_only`、`joint_conditioned` 谁的
   MAG / ENV / LRE / DPAM 更好；
2. 加 sign-sensitive LRE loss 后，能否降低 LRE，同时不明显损害
   MAG、ENV、DPAM 和原 AudioGS loss；
3. 哪些方法或 loss 权重值得继续训练，哪些可以尽早停止。

`configs/experiments/lre_loss_ablation.yaml` 中的组合是**候选池上限**，不是必须
全部跑完的清单。Agent 应根据中间结果动态淘汰，把算力留给更有希望的配置。

## 2. 服务器与并发

服务器预计有两张空闲的 48 GB GPU。允许一张卡同时运行多个实验：

- 初始每张卡启动 2 个训练进程；
- 运行稳定后，如果显存峰值低于 80%、没有 OOM，且吞吐没有明显下降，可提高到
  每卡 3-4 个；
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
- 对 `audio_only`、`joint_conditioned` 做短程 smoke，确认 loss、梯度和指标有限；
- 检查两张 GPU 的可用显存，并确定每卡并发数。

P0 失败时不启动大规模训练。

### P1：快速探索方法和 loss 配置

使用固定 seed 42，优先比较：

```text
systems: audio_only, joint_conditioned
lambda_lre: 0.0, 0.01, 0.02, 0.05
scenes: scene1_opera, Scene7playing
```

先看最早可评估 checkpoint（建议 1k）的指标，再把有希望的配置推进到 5k。
如果当前 runner 只能在 5k 评估，则直接使用 5k，不为此改变训练语义。重点排序：

1. LRE 是否稳定下降；
2. MAG / ENV / DPAM 是否优于或接近 control；
3. `audio_total`、waveform L1 是否没有明显退化；
4. 两个场景的趋势是否一致。

P1 不要求全量跑完。明显失败的配置应及时停止。

### P2：只确认少量候选

从 P1 中保留最多 1-2 个非零 `lambda_lre`，与 `lambda_lre=0` 对照：

- 两个系统都有效时都保留；只有一个系统有效时只推进该系统；
- 先训练到 10k；
- 10k 仍有稳定收益的候选再训练到 30k；
- 输出逐样本指标和 Source Binaural / Mono / control / treatment 对比表。

P2 的目标是选出当前最佳方法和 loss 配置，不是补齐搜索网格。

### P3：最后再做随机种子

多 seed 优先级最低。只有 P2 出现明确候选后才运行：

```text
seeds: 17, 73
configs: control + 最佳 1 个候选
```

如果 P2 没有候选通过保护指标，就不跑多 seed，直接结论为当前 LRE loss 配置
不值得继续。

## 4. 动态淘汰规则

满足任一条件可以停止候选，但必须保存已有 checkpoint、指标和淘汰原因：

- loss、梯度或输出出现非有限值；
- 同一配置连续 OOM，降低并发后仍无法稳定运行；
- 两个场景在连续两个观测点都没有 LRE 改善；
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

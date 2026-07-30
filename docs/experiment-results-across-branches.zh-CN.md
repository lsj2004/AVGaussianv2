# AVGaussianV2 跨分支实验结果分析

> **历史结果提示**
>
> 本文汇总的是 cam38 `visual_time` 修复前的结果。旧 evaluator 在 held-out
> camera 不在 train-only memmap 时使用 `frame_index / fps` 代替 FreeTimeGS++
> model time。当前代码已经改为复用同帧 train-camera 的共享 `time.memmap`
> model time。本文数字保留用于历史比较，在完成
> [`post-visual-time-fix-rerun-plan.zh-CN.md`](post-visual-time-fix-rerun-plan.zh-CN.md)
> 前，不应作为修复后代码的有效排名。

本文汇总截至 2026-07-30 已提交到各远端分支的实验结果，回答三个问题：

1. 哪些分支真的产生了新结果，哪些只是继承了相同结果或提供运行代码；
2. 在统一 cam38/30k 协议下，各模型路线的结果如何；
3. 下一步应该保留、复验或停止哪些方向。

## 1. 结论摘要

目前最值得保留的三个参照是：

- **Query-dependent P1**：综合折中最好。复算的双场景 micro
  `audio_total` 约为 **0.451597**，略优于 FiLM 的 0.461376；micro LRE
  约为 0.280935 dB，接近 GS-only 的 0.276815 dB。
- **FiLM residual**：最稳定、证据最完整的视觉条件基线。它在两个场景都降低
  `audio_total`，但 LRE 从 GS-only 的 0.276815 dB 恶化到 1.159030 dB。
- **GS-only 30k**：空间指标和 waveform 的强基线。它的主重建损失不是最好，
  但不能被只看 `audio_total` 的模型替代。

其他路线的判断：

- **Plain U-Net** 证明 U-Net 本身能贡献重建收益，但跨场景不稳定，且 LRE
  明显恶化。它是必要对照，不是当前最佳系统。
- **Mask cross-attention** 确实使用了 RGBD，但不稳定地使用空间排列，整体仍
  弱于 FiLM 和 P1。
- **Gaussian-token complex cross-attention** 几乎退化为 audio-only：
  去掉 RGBD 后结果几乎不变。
- **Spatial P1 objective** 在 5k 有短期收益，但 30k 时两个场景都弱于原 P1，
  不应替换原目标。
- **Direct U-Net** 和 **gated residual** 跨场景表现差，现有形式没有继续跑
  完整实验的优先级。

这些结论只适用于项目内两个场景、单 seed 42。它们足以指导下一轮工程选择，
不足以支持跨数据集统计泛化声明。

## 2. 结果证据等级

分支名不等于独立结果集。多个分支继承了同一份基础 benchmark，而有些模型结果
只存在于后续分支的汇总报告中。

| 等级 | 证据 | 当前结果 |
|---|---|---|
| A | 提交了机器可读 JSON/CSV、内容哈希和人类可读报告 | 基础 cam38：native、GS-only、FiLM、visual-only |
| B+ | 提交了完整报告和各 checkpoint evaluation hash，但未提交逐样本文件 | Plain U-Net、Mask cross-attention |
| B | 提交了完整数值报告和本地证据路径，未提交逐样本文件 | P1、Spatial P1 |
| C | 数值只出现在后续汇总文档，原分支没有正式结果产物 | Gaussian-token、direct U-Net、gated residual |
| D | 只有代码、设计或短程 diagnostic | `main` smoke、scene1 pilot、早期 codex token 原型 |

因此，基础 FiLM 结果最容易独立复核。P1 是当前最有希望的结果，但在升为默认
主线前，仍应把逐样本评估和多 seed 复验纳入正式产物。

`plain-unet-baseline` 中的总实验汇总生成时 P1 尚未完成，所以仍把 P1 标为
“进行中”。本分析使用之后提交到 `p1-spatial-camera-contrast` 的完整 30k
报告覆盖该旧状态。

## 3. 统一实验协议

下面的主表只使用相同协议下的结果：

| 项目 | 设置 |
|---|---|
| 场景 | `scene1_opera`、`Scene7playing` |
| 训练相机 | cam00-cam37 |
| 测试相机 | cam38 |
| 测试样本 | 130 + 293 = 423 |
| Seed | 42 |
| Main budget | 30,000 updates，batch size 1 |
| 条件模型 warmup | 2,000 updates |
| 报告节点 | 5k、10k、30k |
| 主损失 | AudioGS checkpoint 对应的原生 criterion |

`native_audiogs` 的原始训练定义与 30k continuation 不同，只能作描述性参考。
P1/Spatial P1 的 micro 数字是按 `130/293` 样本数从报告中的场景均值复算；表中
使用 `~` 标识。其他 micro 数字来自已提交报告。

## 4. 30k 主结果排名

### 4.1 `audio_total`

越低越好。

| 排名 | 系统 | scene1 | Scene7 | 423-sample micro | 解释 |
|---:|---|---:|---:|---:|---|
| 1 | Query-dependent P1 | **1.174828** | 0.130710 | **~0.451597** | 当前综合候选 |
| 2 | FiLM residual | 1.222404 | **0.123718** | 0.461376 | 最稳定视觉条件基线 |
| 3 | Spatial P1 | 1.199774 | 0.138520 | ~0.464674 | 30k 弱于原 P1 |
| 4 | Plain U-Net | 1.225181 | 0.169448 | 0.493905 | 长训练后才出现收益 |
| 5 | Mask cross-attention | 1.333412 | 0.140418 | 0.507059 | RGBD 有效但空间对应不稳 |
| 6 | Gaussian-token cross-attention | 1.407427 | 0.152107 | 0.537903 | 几乎等于 audio-only |
| 7 | GS-only 30k | 1.410239 | 0.152879 | 0.539302 | update-matched 强基线 |
| 8 | Gated residual | 1.574867 | 0.126429 | 0.571575 | 场景分化严重 |
| 9 | Direct conditioned U-Net | 1.497358 | 0.182761 | 0.586774 | 去掉 native 锚点后退化 |
| - | Native AudioGS | 1.718166 | 0.181735 | 0.653924 | 非 update-matched 参考 |

P1 相对 FiLM 的 micro `audio_total` 约低 2.12%，相对 GS-only 约低 16.26%。
差距不算大，因此多 seed 复验可能改变 P1 与 FiLM 的排序。

### 4.2 不应只按主损失选模型

几个关键系统的空间和波形指标如下。全部越低越好。

| 系统 | Micro audio_total | Micro waveform L1 | Micro LRE dB | 判断 |
|---|---:|---:|---:|---|
| GS-only 30k | 0.539302 | 0.026613 | **0.276815** | 空间/波形强基线 |
| Gaussian-token | 0.537903 | ~0.026681 | **~0.267022** | LRE 好，但基本没用 RGBD |
| Query-dependent P1 | **~0.451597** | **~0.026341** | ~0.280935 | 当前最佳折中 |
| Spatial P1 | ~0.464674 | ~0.026514 | ~0.354416 | 全面弱于原 P1 |
| FiLM residual | 0.461376 | 0.027442 | 1.159030 | 主损失强，空间误差差 |
| Plain U-Net | 0.493905 | 0.029086 | 1.166815 | 频谱强，波形/空间差 |
| Mask cross-attention | 0.507059 | ~0.029313 | ~1.059460 | 弱于 P1 和 FiLM |

这里揭示了项目最重要的目标冲突：

- FiLM 把 `audio_total` 从 GS-only 的 0.539302 降到 0.461376，却把 LRE
  从 0.276815 dB 提高到 1.159030 dB。
- Plain U-Net 的 mono/diff LSD 最好，但 waveform 和 LRE 明显变差。
- Gaussian-token 的空间指标很好，不代表视觉融合成功；no-RGBD 消融几乎不变，
  说明它主要维持了 AudioGS/audio-only 解。
- P1 是目前唯一同时明显改善主损失、又把 LRE 保持在 GS-only 附近的路线。

## 5. 各分支到底产生了什么

### 5.1 `main` / `agent/rgbd-conditioning`

作用是建立 RGBD -> condition encoder -> FiLM AudioGS 的可微链路。提交了
smoke pipeline，没有提交统一 cam38 数值结果。

结论：这是实现基线，不是结果分支。

### 5.2 `agent/scene1-pilot`

新增三变体 pilot、quick/full evaluation、早停和恢复框架，但没有把一次正式
pilot 数值报告提交进仓库。

结论：它证明了实验基础设施，不应出现在 30k 方法排名中。

### 5.3 `agent/dual-dataset-benchmark`

提交了证据最完整的基础结果：

- GS-only 30k；
- FiLM joint conditioned；
- visual-only；
- native AudioGS / FreeTimeGS++ 描述性参考。

FiLM 相对 GS-only：

- `audio_total` 改善 14.45%；
- mono/diff LSD 改善；
- waveform L1 退化；
- LRE 大幅退化。

视觉侧，FiLM 相对 visual-only：

| 指标 | Visual-only | FiLM joint | 判断 |
|---|---:|---:|---|
| RGB-L1 | 0.085769 | **0.084584** | 改善 |
| PSNR | **18.2161** | 18.1340 | 退化 |
| SSIM | **0.541144** | 0.534050 | 退化 |

所以不能声称 FiLM 同时提升了音频和视频质量。

### 5.4 `agent/cross-attention-benchmark` / `cross-attention-run`

两条分支的文件树完全一致。它们主要提交 strict runner 和 diagnostic，没有单独
提交正式结果文件。

后续汇总显示，这条 source-STFT/Gaussian-token complex residual 路线 30k
micro `audio_total=0.537903`，与 GS-only 的 0.539302 几乎相同。更关键的是：

| 场景 | Correct RGBD | No RGBD | 差值 |
|---|---:|---:|---:|
| scene1 | 1.407427 | 1.407489 | 0.000062 |
| Scene7 | 0.152107 | 0.152114 | 0.000007 |

结论：模型几乎没有使用视觉条件。显式加入 Gaussian token 并不自动产生视觉
因果贡献。

### 5.5 `agent/gaussian-token-cross-attention`

加入真实 AudioGS Gaussian 属性和 pose token，是 P1/plain 分支的共同祖先。
正式数字没有作为独立结果文件提交，而是由后续实验汇总记录。

结论：它是重要结构节点和空间强基线，但不是成功的视觉融合模型。

### 5.6 Mask-protocol cross-attention

该路线在后续 P1/plain 分支中加入，严格对齐 AudioGS U-Net 的 mono/diff mask
输入输出协议。

结果比旧 Gaussian-token 路线更好：

- micro `audio_total=0.507059`；
- 相对 GS-only 改善约 5.98%；
- 相对 FiLM 仍差约 9.90%。

RGBD 因果消融：

| 场景 | Correct | No RGBD | Shuffled RGBD |
|---|---:|---:|---:|
| scene1 | **1.333412** | 1.730626 | 1.399436 |
| Scene7 | 0.140418 | 0.178275 | **0.138678** |

结论：模型确实使用视觉内容，但 Scene7 中 shuffled RGBD 略优于正确排列，
说明它没有稳定学会空间对应关系。

### 5.7 `agent/plain-unet-baseline`

补齐了此前缺失的“无视觉 plain U-Net”对照。

主要结果：

- 30k micro `audio_total=0.493905`；
- 比 GS-only 好 8.42%；
- 比 FiLM 差 7.05%；
- Scene7 上比 GS-only 更差，75.43% 样本输给 GS-only；
- mono/diff LSD 最好，但 waveform、audio_diff 和 LRE 明显更差。

它说明 U-Net 的局部时频归纳偏置本身能解释一部分收益。因此 FiLM 与 GS-only
的全部差值不能简单归因于 RGBD；严格归因还需要在同一个 U-Net 输出规则下比较
plain 与 conditioned。

### 5.8 Query-dependent P1

P1 加入显式世界几何、相机射线、listener-relative attention bias，并用
same-frame wrong-camera 条件做训练和因果评估。

30k 结果：

- scene1 `audio_total=1.174828`，项目内最低；
- Scene7 `audio_total=0.130710`，略逊于 FiLM；
- 双场景 macro `audio_total=0.652769`，低于 FiLM 的 0.673061；
- micro `audio_total~0.451597`；
- micro LRE `~0.280935` dB，接近 GS-only。

因果性：

| 场景 | 对比 | Audio delta | Audio win | LRE delta | LRE win |
|---|---|---:|---:|---:|---:|
| scene1 | correct vs no RGBD | -0.217456 | 76.2% | +0.015893 | 46.2% |
| scene1 | correct vs wrong camera | -0.040456 | 57.7% | +0.011370 | 43.8% |
| Scene7 | correct vs no RGBD | -0.032995 | 99.0% | -0.093149 | 78.2% |
| Scene7 | correct vs wrong camera | -0.129638 | 100.0% | -1.513257 | 100.0% |

Scene7 的视觉和相机因果性很强；scene1 只证明 RGBD 改善总体重建，没有证明
正确相机改善空间方向指标。这是 P1 升主线前最需要复验的问题。

### 5.9 Spatial P1

Spatial P1 在原 P1 上加入 LRE、ILD、IPD 和 binaural-difference 辅助目标。

短程 500+500 pilot 显示 LRE 和空间 loss 明显改善，但正式 30k 结果反转：

- scene1 audio total 比原 P1 差 2.1%，LRE 差 39.1%；
- Scene7 audio total 差 6.0%，LRE 差 15.2%；
- 两个场景 30k 的 audio/LRE paired win rate 都低于 50%；
- mono/diff LSD 四项比较全部弱于原 P1。

结论：不要用当前 spatial objective 替换 P1。5k 的短期收益提示它可以作为
衰减到零的早期辅助项，但这需要新的 schedule ablation。

### 5.10 早期 codex token 原型

`codex/cross-attention-audio-tokens` 和
`codex/hybrid-audiogs-gaussian-tokens` 指向同一提交。它们提供原型代码和设计
文档，没有正式 cam38 结果。

结论：保留历史参考即可，不进入性能比较。

## 6. 为什么不同场景会给出不同答案

### 6.1 `scene1_opera`

- P1 的几何建模明显改善 `audio_total`；
- FiLM、mask cross-attention 和 gated residual 的 LRE 容易严重退化；
- mask cross-attention 能区分 correct 与 shuffled RGBD；
- P1 的 correct-camera LRE 仍没有优于 wrong-camera。

这个场景更能暴露训练不稳定和空间目标错配。

### 6.2 `Scene7playing`

- FiLM 的 `audio_total` 最低；
- P1 的 LRE 和 waveform 明显优于 FiLM；
- P1 对 wrong-camera 的 293 个样本全部获胜；
- mask cross-attention 的 shuffled RGBD 略优于 correct；
- gated residual 在该场景很好，但在 scene1 失败。

这个场景更容易得到主损失收益，但也更容易让全局视觉统计掩盖错误的空间对应。

因此任何新方法都必须同时报告：

- 每个场景；
- macro 和 micro；
- correct/no-RGBD/shuffled/wrong-camera；
- mean delta 和 paired win rate；
- `audio_total`、waveform、LSD、LRE/ILD/IPD。

## 7. 建议的分支决策

| 路线 | 决策 | 原因 |
|---|---|---|
| GS-only 30k | 保留强基线 | 空间和 waveform 不能被主损失掩盖 |
| FiLM residual | 保留正式 baseline | 稳定、证据完整、主损失强 |
| Query-dependent P1 | 主候选，先复验 | 当前综合最好，但只有两场景单 seed |
| Plain U-Net | 保留必要对照 | 分离 U-Net 本身与视觉条件贡献 |
| Mask cross-attention | 暂停扩展 | 使用 RGBD，但空间排列不稳定 |
| Gaussian-token complex | 归档为负结果 | 几乎旁路视觉 |
| Spatial P1 | 不合入；可做衰减 schedule | 5k 好、30k 反转 |
| Direct / gated | 归档当前形式 | 跨场景不稳定或总体弱 |
| 早期 codex 原型 | 归档 | 无正式结果，已被后续路线吸收 |

## 8. 下一轮最有信息量的实验

### 8.1 先复验 P1，不急着继续加结构

用至少 3 个 seed 重跑 P1、FiLM、GS-only：

- 保留同一 cam38 协议；
- 提交逐样本结果，不只提交 Markdown；
- 把 scene1 correct-vs-wrong-camera LRE 作为预注册 gate；
- 报告均值、标准差、bootstrap confidence interval 和 paired win rate。

### 8.2 严格隔离视觉条件的贡献

当前 Plain U-Net 与 FiLM 的外层输出规则不同。下一组应固定同一 backbone 和
同一输出规则，只改变 condition injection：

```text
plain U-Net
vs
RGBD + FiLM U-Net
vs
RGBD + cross-attention U-Net
```

这样才能把“视觉条件”“U-Net 归纳偏置”和“native residual anchor”拆开。

### 8.3 使用多目标 checkpoint 选择

不能只按 `audio_total` 选 checkpoint。建议至少要求：

- `audio_total` 优于 GS-only；
- LRE 不劣于 GS-only 的预设容差；
- correct 优于 no-RGBD；
- correct 优于 shuffled/wrong-camera；
- 两个场景方向一致。

Spatial loss 若继续，应只测试早期衰减 schedule，而不是增加固定权重。

## 9. 证据来源

- 基础机器可读结果：
  `results/cam38_benchmark/2026-07-26/aggregate_report.json`
- 基础人类可读报告：
  `results/cam38_benchmark/2026-07-26/README.md`
- Plain U-Net：
  `origin/agent/plain-unet-baseline:results/plain_unet_cam38/2026-07-30/README.md`
- Cross-attention 路线汇总：
  `origin/agent/plain-unet-baseline:docs/avgaussianv2-experiments-summary.zh-CN.md`
- P1：
  `origin/agent/p1-spatial-camera-contrast:docs/2026-07-28-query-dependent-p1-cam38-results.md`
- Spatial P1：
  `origin/agent/p1-spatial-camera-contrast:docs/2026-07-30-p1-spatial-camera-contrast-results.md`

基础 suite 内容哈希为：

```text
cc57caeaa1ab7e170c27b57be9eae80d5b11d4965d978b4eeee21a835e483c4a
```

Plain U-Net 报告列出了每个场景、每个 milestone 的 evaluation hash；Mask
cross-attention 报告列出了两个场景的 report content hash。P1/Spatial P1
报告引用了本地严格产物路径，但当前分支没有提交对应逐样本文件。

# AudioGS plain U-Net：双数据集 cam38 严格基准

> 完成日期：2026-07-30
>
> 状态：训练与 5k/10k/30k held-out 评估全部完成

## 1. 实验问题

本实验补齐此前缺失的 `AudioGS + plain U-Net` 无视觉条件基线。实际音频前向为：

```text
AudioGS acoustic Gaussians / pose-conditioned mono-diff features
    -> inherited AudioGS mono/diff U-Net
    -> mono/diff masks
    -> original AudioGS binaural synthesis
```

它不使用 RGBD condition，不执行 conditioner warmup，也不采用
`native + conditioned_U-Net - plain_U-Net` 外层残差。

## 2. 严格对齐协议

| 项目 | 设置 |
|---|---|
| 数据集 | `scene1_opera`、`Scene7playing` |
| 训练相机 | cam00–cam37 |
| 测试相机 | cam38 |
| 测试样本 | 130 + 293 = 423 |
| 训练预算 | 30,000 updates，batch size 1 |
| 报告节点 | 5k、10k、30k |
| Seed | 42 |
| 初始化 | 与 GS-only、FiLM 相同的 AudioGS/FreeTimeGS++ checkpoint |
| 样本顺序 | 精确复用原 benchmark 的 `audio_only` 有序索引 |
| 音频 criterion | checkpoint 对应的原生 AudioGS criterion |
| 视觉条件 | 关闭 |

准备器验证了相同模型初始化、相同有序数据集和相同样本索引序列。配置相对
FiLM 基础协议只改变 `model.audio_render_strategy`，训练 mode 为
`audio_only`。

## 3. `audio_total` 收敛结果

### 3.1 分场景

| 场景 | 系统 | 5k | 10k | 30k |
|---|---|---:|---:|---:|
| scene1_opera | GS-only | 1.621367 | 1.560745 | 1.410239 |
| scene1_opera | Plain U-Net | 1.757658 | 1.491162 | 1.225181 |
| scene1_opera | FiLM residual | **1.588761** | **1.472095** | **1.222404** |
| Scene7playing | GS-only | **0.176108** | **0.169252** | 0.152879 |
| Scene7playing | Plain U-Net | 0.185872 | 0.206652 | 0.169448 |
| Scene7playing | FiLM residual | 0.201951 | 0.170732 | **0.123718** |

### 3.2 423-sample micro

| 系统 | 5k | 10k | 30k |
|---|---:|---:|---:|
| GS-only | **0.620278** | 0.596897 | 0.539302 |
| Plain U-Net | 0.668927 | 0.601418 | 0.493905 |
| FiLM residual | 0.628157 | **0.570678** | **0.461376** |

Plain U-Net 的优势只在长训练后出现。它在 5k/10k 都不优于 GS-only，
30k 才使 micro `audio_total` 相对 GS-only 改善 8.42%。

## 4. 30k 完整音频指标

以下为 423 个 cam38 样本的 micro 均值；全部越低越好。

| 系统 | audio_total | audio_mono | audio_diff | Waveform L1 | Mono LSD | Diff LSD | LRE dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native AudioGS | 0.653924 | 0.583717 | 0.084702 | 0.027220 | 0.982832 | 1.167756 | 0.371422 |
| GS-only 30k | 0.539302 | 0.477085 | **0.082913** | **0.026613** | 0.963523 | 1.146853 | **0.276815** |
| Plain U-Net 30k | 0.493905 | 0.356977 | 0.098835 | 0.029086 | **0.899464** | **1.041216** | 1.166815 |
| FiLM residual 30k | **0.461376** | **0.333581** | **0.081738** | 0.027442 | 0.916021 | 1.061590 | 1.159030 |

相对关系：

- Plain U-Net 比 GS-only 的 `audio_total` 低 8.42%；
- FiLM residual 比 Plain U-Net 低 6.59%；
- Plain U-Net 比 FiLM residual 高 7.05%；
- Plain U-Net 的 mono/diff LSD 最好，但 waveform、audio_diff 和 LRE 明显
  弱于 GS-only；
- FiLM 在主要 `audio_total`、mono、diff 上最好，但空间 LRE 仍明显弱于
  GS-only。

## 5. 逐样本配对结果

### 5.1 Plain U-Net 相对 GS-only

| 场景 | audio_total mean delta | Plain 胜率 |
|---|---:|---:|
| scene1_opera | -0.185058 | 51.54% |
| Scene7playing | +0.016569 | 24.57% |
| 423-sample micro | -0.045397 | 32.86% |

micro 均值改善主要由 scene1 中少数大幅改善样本驱动。Plain U-Net 在
Scene7 的均值和 75.43% 样本上都弱于 GS-only，因此不能称为稳定提升。

### 5.2 Plain U-Net 相对 FiLM residual

| 场景 | audio_total mean delta | Plain 胜率 |
|---|---:|---:|
| scene1_opera | +0.002777 | 56.15% |
| Scene7playing | +0.045730 | 2.05% |
| 423-sample micro | +0.032529 | 18.68% |

scene1 两者均值几乎持平，但误差分布不同；Scene7 上 FiLM 具有压倒性优势。
综合两个场景，FiLM residual 是更强的已完成系统。

## 6. 应如何解释

本实验回答了“直接使用 AudioGS plain U-Net 能达到什么水平”，但不能把
Plain U-Net 与 FiLM residual 的全部差值解释成视觉收益，因为两者的外层合成
规则不同：

```text
Plain U-Net:
    audio = plain_U-Net

FiLM residual:
    audio = native_AudioGS
          + conditioned_U-Net
          - plain_U-Net
```

因此目前可以确认：

1. U-Net 局部时频建模本身能改善长程 micro `audio_total`；
2. 该改善跨场景不稳定，并牺牲 waveform/LRE；
3. 当前完整 FiLM residual 系统优于 Plain U-Net；
4. 若要严格分离“视觉条件本身”的贡献，还应比较同一 direct-U-Net 外层规则下
   的 `plain_U-Net` 与 `conditioned-U-Net`，或在同一 U-Net 内只替换
   condition injection。

## 7. 可复核产物

本地严格产物：

```text
runs/plain_unet_cam38/scene1_opera/
runs/plain_unet_cam38/Scene7playing/
```

评估内容 SHA-256：

| 场景 | 5k | 10k | 30k |
|---|---|---|---|
| scene1_opera | `90b3ec91760083aca449536bffc535aaf17bf773d791589da3ca5dbb7310468b` | `b88a6d3b8f93ad76271013f420b0390fc1da22b74de6d3edd1ad0427ef5001c0` | `f37a83b3e61d8c5aa279bbd3552821ed55601f67ba8b9a8810c669011cf80bf6` |
| Scene7playing | `fe9a609fe3ddfe3e290bdbf8b302ec35fbfb10028a14ba3fb478972b1cae5ffe` | `95fcc13ace848c02da7125375e7fb72375b36926249b25260dc61080932c6f1d` | `8c98ff26db12ebae2d0431f469b6f959a86658e76296ffd41c6ed3abc7a5a228` |

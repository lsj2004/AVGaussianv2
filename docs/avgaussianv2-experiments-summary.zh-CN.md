# AVGaussianFusionV2：现有实验路径、结果与分析

> 更新日期：2026-07-28
>
> 范围：`scene1_opera`、`Scene7playing`，cam38 held-out benchmark
>
> 状态：除 Query-dependent P1 外，本文列出的正式主实验均已完成

## 1. 摘要

AVGaussianFusionV2 的目标是把视觉高斯和声学高斯分开建模，同时让视觉渲染
结果参与双耳音频渲染。到目前为止，项目依次探索了以下路径：

1. **FiLM + AudioGS U-Net**：用单帧 RGBD 全局条件调制 AudioGS U-Net；
2. **音频输出组合消融**：比较 native residual、direct U-Net 和 gated
   residual；
3. **Source-STFT / Gaussian-token Cross-Attention**：用 source-audio
   时频 token 查询 RGBD、pose 和显式声学高斯 token，输出 complex-STFT
   residual；
4. **AudioGS mask-protocol Cross-Attention**：严格复用 AudioGS U-Net 的
   输入输出协议，只替换 mask renderer；
5. **Query-dependent P1**：加入世界几何、射线、监听者坐标系注意力偏置和
   wrong-camera 排序约束；该路线截至本文生成时仍在训练。

完成实验支持的主要结论是：

- 视觉条件能够稳定改善主要音频重建损失 `audio_total`，但尚未同时改善所有
  波形和空间音频指标；
- 30k 时，**FiLM + AudioGS U-Net 仍是已完成方案中综合最优的正式模型**，
  两场景 micro `audio_total=0.461376`；
- 对齐 AudioGS mask 协议后的 Cross-Attention 明显优于 no-RGBD，也优于旧
  complex-residual Cross-Attention，但 30k micro `audio_total=0.507059`，
  仍比 FiLM 差约 9.90%；
- `scene1_opera` 中正确 RGBD 优于 shuffled-RGBD，说明模型使用了视觉空间
  内容；`Scene7playing` 中 shuffled-RGBD 略优于正确 RGBD，说明当前空间
  对应关系仍不可靠；
- 简单增加训练时长不是答案。多个方案存在 5k 较好、10k/30k 退化的现象，
  更需要稳定的视觉门控、局部时频归纳偏置和几何感知条件，而不是更深的纯
  Transformer。

## 2. 统一实验协议

正式比较使用同一套 cam38 协议：

| 项目 | 设置 |
|---|---|
| 数据集 | `scene1_opera`、`Scene7playing` |
| 训练相机 | cam00–cam37 |
| 测试相机 | cam38 |
| 测试样本数 | 130 + 293 = 423 |
| 随机种子 | 42 |
| Batch size | 1 |
| 条件 warmup | 2,000 updates |
| Main continuation | 30,000 updates |
| 报告节点 | 5k、10k、30k |
| 视觉初始化 | 同一 FreeTimeGS++ checkpoint |
| 声学初始化 | 同一 `Audio3DGSMonoDiffGSOnly` checkpoint |
| 音频损失 | checkpoint 对应的原生 AudioGS criterion |

严格 runner 会固定并验证：

- 有序训练样本索引；
- 原生 AudioGS/FTGS++ checkpoint 路径及 SHA-256；
- 数据集身份、split、seed、batch size 和训练预算；
- 5k/10k/30k milestone；
- cam38 target 不在训练时读取；
- 因果消融复用同一个 checkpoint，不重新训练。

原生 AudioGS 和原生 FreeTimeGS++ 是描述性参考；它们的原始训练定义与
30k continuation 并不完全 update-matched，因此不应作为主因果比较对象。

### 2.1 Clip 和视觉条件

音频仍以 clip 为单位训练和评估：

```text
0.5 秒音频 clip
    -> 取 clip 对应的中心 frame_index
    -> 使用该时刻、目标相机外参与内参
    -> 从动态视觉高斯渲染一张 RGB / expected depth / alpha
    -> 编码为该 clip 的 visual condition
```

condition 不是直接读取真实视频帧；真实帧仅用于视觉监督和指标。当前也没有
使用相邻多帧、光流或与 160 个 STFT 时间格逐一对应的视觉序列。

相邻 clip 中心相隔约 1/30 秒，因此 0.5 秒音频窗口高度重叠。正式报告对各
clip 独立评估，没有先 overlap-add 为完整连续音轨。

### 2.2 主要指标

除 PSNR、SSIM 越高越好外，其他指标均越低越好：

- `audio_total`：AudioGS 原生 criterion 的总损失，是主要选择指标；
- `audio_mono`、`audio_diff`：mono/difference 分支损失；
- `waveform_l1`：双耳波形 L1；
- `mono_lsd`、`diff_lsd`：log-spectral distance；
- `lre_error_db`：左右耳能量比误差；
- `rgb_psnr`、`rgb_ssim`、`rgb_l1`：视觉重建指标。

`audio_total` 改善并不自动代表空间音频质量改善，因此 waveform、LSD 和 LRE
必须共同报告。

### 2.3 基线系统定义与当前覆盖

这里必须区分“是否使用 U-Net”和“是否经过相同 30k continuation”。当前
配置的音频模型是 `Audio3DGSMonoDiffGSOnly`；当 `condition=None` 时，后端
直接返回 native GS-only 前向结果，继承的 U-Net 不参与音频输出。因此报告中
的 `audio_only` 是 **AudioGS GS-only 的 30k audio-only continuation**，不是
“AudioGS + U-Net、无视觉条件”。

| 系统 | 实际音频前向 | 视觉分支 | 视觉条件 | 当前状态 |
|---|---|---|---|---|
| Native AudioGS | GS-only，不使用 U-Net | 无 | 无 | 已评估；原始 61-epoch checkpoint，描述性参考 |
| Audio-only 30k | GS-only，不使用 U-Net | 冻结且不参与音频 | 无 | 已完成；与联合模型 update-matched |
| AudioGS + plain U-Net | 继承的 mono/diff U-Net | 无或冻结 | 无 | **尚未完成严格双数据集 baseline** |
| AudioGS U-Net + FiLM | `native + conditioned_U-Net - plain_U-Net` | FreeTimeGS++ RGBD | 有 | 已完成；联合模型 |
| FreeTimeGS++ visual-only | 不产生音频 | FreeTimeGS++ | 不适用 | 已完成；只比较视觉指标 |

30k cam38 micro 指标总表如下。`—` 表示该系统不能产生对应模态，`未跑` 表示
代码路径存在，但缺少严格对齐的训练与评估结果。

| 系统 | audio_total ↓ | waveform L1 ↓ | mono LSD ↓ | diff LSD ↓ | LRE dB ↓ |
|---|---:|---:|---:|---:|---:|
| Native AudioGS（GS-only） | 0.653924 | 0.027220 | 0.982832 | 1.167756 | 0.371422 |
| Audio-only 30k（GS-only） | 0.539302 | **0.026613** | 0.963523 | 1.146853 | **0.276815** |
| AudioGS + plain U-Net（无视觉） | **未跑** | **未跑** | **未跑** | **未跑** | **未跑** |
| AudioGS U-Net + FiLM | **0.461376** | 0.027442 | **0.916021** | **1.061590** | 1.159030 |
| FreeTimeGS++ visual-only | — | — | — | — | — |

| 系统 | RGB-L1 ↓ | PSNR dB ↑ | SSIM ↑ |
|---|---:|---:|---:|
| Native FreeTimeGS++ checkpoint | **0.076549** | **18.2343** | 0.527436 |
| FreeTimeGS++ visual-only 30k | 0.085769 | 18.2161 | **0.541144** |
| AudioGS U-Net + FiLM 联合模型 30k | 0.084584 | 18.1340 | 0.534050 |

Native AudioGS/FreeTimeGS++ 的训练预算与 30k continuation 不同，因此只能
描述性比较。严格因果比较目前只覆盖 FiLM 联合模型与 GS-only audio
continuation、FreeTimeGS++ visual-only continuation；plain U-Net 无视觉
baseline 是现有实验矩阵中的真实缺口。

## 3. 路径 A：FiLM + AudioGS U-Net

### 3.1 动机和实现

第一版保留 AudioGS 的 mono/diff U-Net，用视觉高斯渲染的单帧 RGBD 生成
128 维全局 embedding，并通过 FiLM 调制 U-Net 的 encoder/decoder block。

当前 GS-only checkpoint 的组合规则是：

```text
audio =
    native_AudioGS
    + conditioned_U-Net
    - plain_U-Net
```

这样能够以原生声学高斯渲染为锚点，只加入由视觉条件导致的 U-Net 差值。

### 3.2 相对 AudioGS GS-only continuation 的正式结果

30k、423 个 cam38 样本的 micro 结果：

| 指标 | Audio-only GS-only | FiLM + U-Net | 差值 | 结论 |
|---|---:|---:|---:|---|
| audio_total | 0.539302 | **0.461376** | -0.077926 | 改善 14.45% |
| audio_mono | 0.477085 | **0.333581** | -0.143504 | 改善 |
| audio_diff | 0.082913 | **0.081738** | -0.001175 | 均值小幅改善 |
| waveform_l1 | **0.026613** | 0.027442 | +0.000829 | 退化 |
| mono_lsd | 0.963523 | **0.916021** | -0.047502 | 改善 |
| diff_lsd | 1.146853 | **1.061590** | -0.085263 | 改善 |
| lre_error_db | **0.276815** | 1.159030 | +0.882215 | 明显退化 |

分场景 30k：

| 场景 | 系统 | audio_total | waveform_l1 | lre_error_db |
|---|---|---:|---:|---:|
| scene1_opera | Audio-only GS-only | 1.410239 | **0.050588** | **0.420740** |
| scene1_opera | FiLM + U-Net | **1.222404** | 0.054872 | 2.465855 |
| Scene7playing | Audio-only GS-only | 0.152879 | 0.015976 | **0.212957** |
| Scene7playing | FiLM + U-Net | **0.123718** | **0.015272** | 0.579210 |

FiLM 在两个场景的 `audio_total` 都改善，但 `scene1_opera` 的 waveform 和
LRE 明显变差。它证明了视觉条件有用，但没有解决空间指标与主损失不一致的
问题。

### 3.3 收敛趋势

| 场景 | 5k | 10k | 30k |
|---|---:|---:|---:|
| scene1_opera | 1.588761 | 1.472095 | **1.222404** |
| Scene7playing | 0.201951 | 0.170732 | **0.123718** |

FiLM 是现有正式方案中收敛最稳定的一条路线。

### 3.4 视觉重建结果

联合模型与 update-matched visual-only 的 30k micro 对比：

| 指标 | Visual-only | Joint RGBD-conditioned | Joint − baseline | 结论 |
|---|---:|---:|---:|---|
| RGB-L1 | 0.085769 | **0.084584** | -0.001185 | 改善 |
| PSNR (dB) | **18.2161** | 18.1340 | -0.0821 | 退化 |
| SSIM | **0.541144** | 0.534050 | -0.007095 | 退化 |

RGB-L1 的逐样本胜率为 70.45%，但 PSNR 和 SSIM 胜率只有 1.42% 和
13.48%。分场景看，scene1 的三个视觉指标全部退化；Scene7 只有 RGB-L1
改善。由于 Scene7 样本更多，总体 RGB-L1 掩盖了 scene1 的退化。现有联合
训练不能被描述为“同时提升音频和视频”，更准确的结论是：音频主损失改善，
视觉误差分布发生变化，但视觉保真度没有一致提高。

## 4. 路径 B：音频输出组合策略消融

这一组实验研究视觉条件 renderer 的输出应如何与原生 AudioGS 合成。

### 4.1 三种策略

1. **Native residual / FiLM**

   ```text
   native + conditioned_renderer - plain_renderer
   ```

2. **Direct conditioned U-Net**

   ```text
   audio = conditioned_U-Net
   ```

3. **Gated native residual**

   ```text
   audio = native + learned_gate * (conditioned - plain)
   ```

### 4.2 30k 对比

| 场景 | FiLM residual | Direct U-Net | Gated residual |
|---|---:|---:|---:|
| scene1_opera | **1.222404** | 1.497358 | 1.574867 |
| Scene7playing | **0.123718** | 0.182761 | 0.126429 |
| 423-sample micro | **0.461376** | 0.586774 | 0.571575 |

### 4.3 关键现象

- Direct U-Net 在 scene1 从 5k 的 2.389912、10k 的 2.706469 到 30k 的
  1.497358，最终仍明显弱于 native residual；
- Gated residual 在 Scene7 的 waveform L1 为 **0.014557**、LRE 为
  **0.312796**，优于 FiLM 的对应指标；
- 但 gated residual 在 scene1 从 5k 的 1.286092 退化到 30k 的
  1.574867，跨场景稳定性较差。

这组实验支持保留 native AudioGS 作为锚点。直接让条件 U-Net 接管完整音频
输出风险较大；一个全局可学习 gate 也不足以处理不同时间、频率和 mono/diff
区域的条件可靠性。

## 5. 路径 C：Source-STFT + Gaussian-token Cross-Attention

### 5.1 动机

在正式引入 Gaussian tokens 前，项目先验证了一个以 source-audio
时频表示为 query、RGBD 为 memory、输出 spectrogram residual 的短程原型。
该原型仅证明 condition 链路可训练，没有完成统一 cam38 主实验，因此不进入
正式排名。它随后演化为显式接入声学高斯属性的 Cross-Attention：

```text
source audio
    -> complex-STFT patch queries

AudioGS xyz / quaternion / mono-diff SH / pose response
    -> Gaussian token encoder
    -> 16x16 acoustic Gaussian tokens

RGBD -> visual tokens
pose -> pose tokens

audio queries cross-attend
    [visual + pose + acoustic Gaussian tokens]
    -> bounded complex-STFT residual
    -> native AudioGS + residual
```

两个实际 AudioGS checkpoint 的原生网格是 `257×160=41,120` 个点，再结构
池化为 256 个 Gaussian tokens。早期路线文档中的 `257×348=89,436` 是历史
配置数字，不适用于本轮两个 checkpoint。

### 5.2 正式结果

| 场景 | Step | audio_total | waveform_l1 | lre_error_db |
|---|---:|---:|---:|---:|
| scene1_opera | 5k | 1.621773 | 0.051005 | 0.409911 |
| scene1_opera | 30k | 1.407427 | 0.050627 | 0.404890 |
| Scene7playing | 5k | 0.175498 | 0.016480 | 0.252352 |
| Scene7playing | 30k | 0.152107 | 0.016056 | 0.205852 |

30k micro `audio_total≈0.537903`，几乎等于 audio-only 的 0.539302，明显弱于
FiLM 的 0.461376。

关键因果结果：

| 场景 | 正确 RGBD | no-RGBD | 差异 |
|---|---:|---:|---:|
| scene1_opera 30k | 1.407427 | 1.407489 | 0.000062 |
| Scene7playing 30k | 0.152107 | 0.152114 | 0.000007 |

去掉 RGBD 后几乎不变，说明该结构主要退化成 audio-only / native AudioGS
修正器。尽管它显式接入了声学高斯属性，视觉 condition 基本被旁路。

### 5.3 为什么这条路径不适合作为最终 Cross-Attention 比较

它同时改变了三件事：

- 输入从 AudioGS mono/diff features 变成 source complex STFT；
- 输出从 mono/diff masks 变成 complex spectrogram residual；
- 合成路径绕开了 AudioGS U-Net mask 控制。

因此它不能回答“只替换 U-Net 网络后 Cross-Attention 是否更好”，只能作为
一个独立 complex-residual 后处理器实验。

## 6. 路径 D：AudioGS mask-protocol Cross-Attention

### 6.1 协议对齐

新版 Cross-Attention 严格对齐 AudioGS U-Net 的输入输出：

```text
mono_features [B,3,257,160]
  channel 0 = source magnitude
  channel 1 = pose-conditioned mono SH response
  channel 2 = inverse distance

diff_features [B,2,257,160]
  channel 0 = pose-conditioned diff SH response
  channel 1 = inverse distance

+ single-frame RGBD visual tokens

-> mono_mask [B,1,257,160] = softplus(logits) + 0.1
-> diff_mask [B,1,257,160] = tanh(logits)
```

网络内部不是用单个 raw STFT bin 作为 query，而是：

```text
AudioGS mono/diff feature maps
    -> 独立 16x4 patch embedding
    -> 融合为 17x40 = 680 个 AudioGS feature-patch queries
    -> cross-attend RGBD tokens
    -> linear patch decoder
    -> crop 回 257x160 masks
```

AudioGS 的 mask resize、mono/diff 控制、source phase 和 iSTFT 完全复用。
GS-only checkpoint 仍采用：

```text
native AudioGS + conditioned renderer - plain renderer
```

### 6.2 5k/10k/30k 主结果

| 场景 | 5k | 10k | 30k |
|---|---:|---:|---:|
| scene1_opera | **1.232753** | 1.608218 | 1.333412 |
| Scene7playing | 0.201659 | 0.168785 | **0.140418** |

scene1 在 5k 已接近最终 FiLM，但随后明显退化；Scene7 则总体随训练改善。
这说明当前 Cross-Attention 的优化行为具有明显场景依赖性。

### 6.3 30k 与 FiLM 的主比较

| 场景 | FiLM + U-Net | Mask Cross-Attn | Cross − FiLM | Cross 胜率 |
|---|---:|---:|---:|---:|
| scene1_opera | **1.222404** | 1.333412 | +0.111007 | 44.62% |
| Scene7playing | **0.123718** | 0.140418 | +0.016700 | 23.55% |
| 423-sample micro | **0.461376** | 0.507059 | +0.045683 | — |

Mask Cross-Attention 相比 audio-only 的 micro 0.539302 有约 5.98% 改善，但
相比 FiLM 仍差约 9.90%。

### 6.4 RGBD 因果消融

30k `audio_total`：

| 场景 | 正确 RGBD | no-RGBD | shuffled-RGBD |
|---|---:|---:|---:|
| scene1_opera | **1.333412** | 1.730626 | 1.399436 |
| Scene7playing | 0.140418 | 0.178275 | **0.138678** |

逐样本配对统计：

| 场景 | 比较 | mean delta | 正确 RGBD 胜率 |
|---|---|---:|---:|
| scene1_opera | correct − no-RGBD | -0.397215 | 76.92% |
| scene1_opera | correct − shuffled | -0.066024 | 65.38% |
| Scene7playing | correct − no-RGBD | -0.037857 | 74.06% |
| Scene7playing | correct − shuffled | +0.001740 | 21.50% |

可以下的结论：

- 两个场景中，正确 RGBD 都显著优于 no-RGBD，视觉 condition 不再像旧
  complex-residual 路线一样被完全旁路；
- scene1 的正确空间排列优于 shuffled；
- Scene7 的 shuffled 反而略好，表明模型使用了视觉总体内容，但没有可靠学习
  token 的空间对应关系，或依赖了与排列无关的全局统计。

### 6.5 指标冲突

scene1 30k：

| 系统 | audio_total | waveform_l1 | lre_error_db |
|---|---:|---:|---:|
| FiLM | **1.222404** | **0.054872** | 2.465855 |
| Mask Cross-Attn | 1.333412 | 0.060445 | **1.787137** |
| no-RGBD | 1.730626 | **0.051033** | **0.888650** |

视觉条件改善 `audio_total`，却可能增大 waveform 和 LRE 误差。它更积极地
修正 mono/diff 频谱，但修正幅度和方向未受到足够约束。

## 7. 路径 E：Query-dependent P1（进行中）

P1 尝试解决“二维 visual tokens 缺少明确声学对应关系”的问题：

```text
source audio + listener pose
    -> AudioGS native render
    -> complex-STFT audio queries

RGB / depth / alpha + camera rays
    -> metric visual memory

audio-query geometry × visual metric geometry
    -> listener/head-relative attention bias
    -> bounded complex residual
```

训练时为每个正确条件配对同 frame、错误训练相机的负条件，并加入：

```text
max(0, margin + loss(correct_camera) - loss(wrong_camera))
```

同 checkpoint 将评估：

- correct RGBD/camera；
- no-RGBD；
- wrong-camera。

截至 2026-07-28 16:40，最近一次持久化进度为：

| 场景 | Warmup | 已持久化 main 进度 | 正式结果 |
|---|---:|---:|---|
| scene1_opera | 2,000 | 24,500 / 30,000 | 尚无 |
| Scene7playing | 2,000 | 21,500 / 30,000 | 尚无 |

短程 diagnostic 中固定 probe loss 有下降，且 condition effect 非零，但这只能
证明链路可训练，不能作为性能结论。P1 必须完成 30k 和 correct/no-RGBD/
wrong-camera 的 5k/10k/30k 评估后再进入正式排名。

## 8. 已完成方案的 30k 总览

以下 micro `audio_total` 按 423 个 cam38 样本加权；越低越好：

| 排名 | 系统 | scene1 | Scene7 | Micro | 备注 |
|---:|---|---:|---:|---:|---|
| 1 | FiLM + AudioGS U-Net | **1.222404** | **0.123718** | **0.461376** | 当前综合最优 |
| 2 | Mask-protocol Cross-Attn | 1.333412 | 0.140418 | 0.507059 | 视觉有效但不稳定 |
| 3 | Gaussian-token complex Cross-Attn | 1.407427 | 0.152107 | 0.537903 | 几乎退化为 audio-only |
| 4 | Audio-only GS-only | 1.410239 | 0.152879 | 0.539302 | update-matched baseline |
| 5 | Gated native residual | 1.574867 | 0.126429 | 0.571575 | 场景差异大 |
| 6 | Direct conditioned U-Net | 1.497358 | 0.182761 | 0.586774 | 去掉 native 锚点后退化 |
| — | Native AudioGS | 1.718166 | 0.181735 | 0.653924 | 描述性参考，非 update-matched |

这个排序只针对 `audio_total`。例如 gated residual 在 Scene7 的 waveform 和
LRE 很强，但在 scene1 明显退化；因此不能把单一 micro 排名解释为所有音频
属性的绝对优劣。

## 9. 跨实验分析

### 9.1 FiLM 为什么暂时更强

FiLM 保留了 AudioGS U-Net 的局部时频归纳偏置和 skip connection，并以
native AudioGS 外层残差作为保护。视觉条件只做低风险的 feature
modulation。

当前 Mask Cross-Attention 则从头学习：

- `16×4` patch tokenizer；
- 全局 self/cross attention；
- 线性 unpatchify mask decoder。

它同时承担局部频谱重建和跨模态选择，样本效率更低，也更容易产生 patch 内
细节损失和训练波动。

### 9.2 旧 Cross-Attention 为什么几乎忽略视觉

旧路线以 source STFT 为 query，以 native AudioGS 为 residual anchor。仅靠
音频 query 和声学高斯 memory 已能逼近 audio-only 解，视觉 memory 不是完成
目标所必需的，因此优化会选择条件旁路这一更容易的局部最优。

显式加入更多 Gaussian tokens 并不能自动保证视觉被使用；no-RGBD 的几乎零
差异直接证明了这一点。

### 9.3 Mask 对齐解决了什么，又没有解决什么

Mask 版解决了协议问题：

- 输入确实来自 AudioGS acoustic Gaussian / pose-conditioned features；
- 输出确实是 AudioGS mono/diff masks；
- AudioGS 双耳合成算法保持不变。

但它仍存在：

- mono/diff 过早融合；
- `16×4` 非重叠 patch 频率压缩较强；
- visual token 主要是 RGB/depth/alpha + 二维位置，而不是世界坐标和表面
  法向；
- 每个 0.5 秒 clip 只有一个静态中心帧条件；
- 所有时频区域接受同类视觉修正，缺少内容相关的可靠性门控；
- loss 对 waveform/LRE 的约束不足。

### 9.4 数据集异质性不能被 micro 均值掩盖

- scene1 更能体现正确 RGBD 与 shuffled-RGBD 的差异，但训练曲线波动大；
- Scene7 上视觉总体内容有用，空间排列却没有形成可靠收益；
- gated residual 在 Scene7 较好、在 scene1 退化；
- 因此所有后续方案必须同时报告逐场景、macro/micro 和 paired win rate。

## 10. 建议的下一步

### 10.1 若目标是纯比较条件注入算法

最严格的设计应使用同一个 AudioGS U-Net backbone：

```text
A: AudioGS U-Net + FiLM
B: AudioGS U-Net + Cross-Attention
```

只替换 condition injection。否则“预训练 U-Net vs 从头训练 Transformer”和
“FiLM vs Cross-Attention”两个变量混在一起，无法归因。

### 10.2 若目标是最终性能

建议实现独立但具有局部归纳偏置的双流门控结构：

```text
mono_features -> normalized overlapping Conv encoder -> mono queries
diff_features -> normalized overlapping Conv encoder -> diff queries

RGBD -> world xyz / normal / depth / alpha / RGB / ray tokens

mono/diff queries
    -> separate zero-init visual gates
    -> geometry-aware cross-attention
    -> local convolution decoder
    -> mono/diff masks
    -> original AudioGS synthesis
```

优先级：

1. mono/diff 双流，避免过早融合；
2. 每个时频 patch 独立的零初始化视觉 gate；
3. overlapping Conv tokenizer 和卷积 mask decoder；
4. 世界坐标、法向、ray direction、listener-relative geometry；
5. 在单帧版本稳定后，再扩展为 0.5 秒内约 15 帧的时空 visual tokens。

### 10.3 高效实验门槛

下一版先在两个数据集跑严格 5k：

- 主模型必须优于 no-RGBD；
- 主模型必须在两个场景都不劣于 shuffled-RGBD；
- 相比当前 Mask Cross-Attention 至少改善 3%；
- waveform/LRE 不应出现大幅反向退化；
- 只有满足以上条件才继续 10k/30k。

这样能够避免再次为明显不稳定的结构支付完整 30k 成本。

## 11. 可复核材料

版本库内：

- [正式双数据集基准结果](../results/cam38_benchmark/2026-07-26/README.md)
- [Gaussian-token Cross-Attention 路线说明](cross-attention-audio-token-roadmap.zh-CN.md)
- [Query-dependent P1 协议](query-dependent-p1-cam38-benchmark.zh-CN.md)

本地严格产物：

```text
runs/cam38_benchmark/<scene>/
runs/audio_architecture_ablation/<scene>/
runs/cross_attention_masks_ablation/<scene>/
runs/query_dependent_p1_cam38/<scene>/
```

Mask Cross-Attention 报告内容哈希：

| 场景 | 报告 content SHA-256 |
|---|---|
| scene1_opera | `6b7d0effd99cfa55da7b87c268f36edfc83b47bbe890a93b737ceadf3c9554f3` |
| Scene7playing | `b9c1cc56414aa6eb969bbcbead1e7ccbdb1884cf6162f0594236116f94f46925` |

基础双数据集 suite 内容哈希：

```text
cc57caeaa1ab7e170c27b57be9eae80d5b11d4965d978b4eeee21a835e483c4a
```

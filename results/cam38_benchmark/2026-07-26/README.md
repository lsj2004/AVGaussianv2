# 双数据集 cam38 基准结果（2026-07-26）

## 结论

在 `scene1_opera` 和 `Scene7playing` 共 423 个 cam38 测试样本上，30k
update-matched 对比表明：

- RGBD 条件使联合模型的 micro `audio_total` 从 audio-only 的
  0.539302 降至 0.461376，相对改善 14.45%；`audio_mono`、`diff_lsd` 和
  `mono_lsd` 也改善。
- 联合模型并非全面优于 audio-only：micro `waveform_l1` 从 0.026613
  升至 0.027442，`lre_error_db` 从 0.276815 升至 1.159030。尤其 LRE
  明显退化，说明当前条件分支损害了空间方向线索。
- 相对 visual-only，联合模型的 micro RGB-L1 从 0.085769 降至
  0.084584，但 PSNR 从 18.2161 dB 降至 18.1340 dB，SSIM 从 0.541144
  降至 0.534050。不能据此声称视频质量整体提升。

因此，这轮实验支持的最窄结论是：**视觉条件能够改善主要音频重建损失，
但当前实现尚未在空间音频和整体视频质量上形成一致收益。**

## 实验协议

| 项目 | 设置 |
|---|---|
| 场景 | `scene1_opera`、`Scene7playing` |
| 训练视角 | cam00–cam37 |
| 测试视角 | cam38 |
| 测试样本 | 130 + 293 = 423 |
| 随机种子 | 42 |
| 公平主比较 | `joint_conditioned` vs `audio_only` / `visual_only` |
| continuation 预算 | 每个系统 30,000 updates，batch size 1 |
| 联合模型额外阶段 | 2,000-step conditioner warmup |
| 报告节点 | 5k、10k、30k |
| 聚合 | macro 为场景等权；micro 按样本数加权 |

训练缓存仅包含 cam00–cam37。cam38 目标只在训练结束后的独立评估路径中
读取；运行时合约记录 `test_targets_read_during_training=false`。原生 AudioGS
和 FreeTimeGS++ 结果属于描述性参考，其训练定义与 continuation 不完全
update-matched，因此不作为主因果比较。

没有可用的深度真值，所以不报告深度准确率。LPIPS 也未纳入本轮表格，因为
它没有以相同实现同时覆盖所有被比较系统。

## 30k 总体结果

以下为 423 个样本的 micro 均值。除 PSNR、SSIM 越高越好外，其余指标均
越低越好。

### 音频：联合模型与 audio-only

| 指标 | Audio-only | Joint RGBD-conditioned | Joint − baseline | 判断 |
|---|---:|---:|---:|---|
| audio_total | 0.539302 | **0.461376** | -0.077926 | 改善 |
| audio_mono | 0.477085 | **0.333581** | -0.143504 | 改善 |
| audio_diff | 0.082913 | **0.081738** | -0.001175 | 均值小幅改善 |
| waveform_l1 | **0.026613** | 0.027442 | +0.000829 | 退化 |
| mono_lsd | 0.963523 | **0.916021** | -0.047502 | 改善 |
| diff_lsd | 1.146853 | **1.061590** | -0.085263 | 改善 |
| lre_error_db | **0.276815** | 1.159030 | +0.882215 | 明显退化 |

配对样本上，联合模型在 `audio_total`、`audio_mono`、`diff_lsd` 和
`mono_lsd` 的胜率分别为 70.45%、93.62%、94.80% 和 82.51%。但是
`audio_diff` 的胜率只有 17.73%，说明其均值改善由少数幅度较大的样本驱动，
不应只看均值。

### 视频：联合模型与 visual-only

| 指标 | Visual-only | Joint RGBD-conditioned | Joint − baseline | 判断 |
|---|---:|---:|---:|---|
| RGB-L1 | 0.085769 | **0.084584** | -0.001185 | 改善 |
| PSNR (dB) | **18.2161** | 18.1340 | -0.0821 | 退化 |
| SSIM | **0.541144** | 0.534050 | -0.007095 | 退化 |

RGB-L1 的配对胜率为 70.45%，但 PSNR、SSIM 的配对胜率仅为 1.42% 和
13.48%。三个指标不一致，说明联合训练改变了误差分布，而不是稳定提高视觉
保真度。

## 分场景 30k 结果

### 音频

| 场景 | 系统 | audio_total | waveform_l1 | mono_lsd | diff_lsd | lre_error_db |
|---|---|---:|---:|---:|---:|---:|
| scene1_opera | Audio-only | 1.410239 | **0.050588** | **0.977741** | 1.067826 | **0.420740** |
| scene1_opera | Joint | **1.222404** | 0.054872 | 0.978140 | **1.048034** | 2.465855 |
| Scene7playing | Audio-only | 0.152879 | 0.015976 | 0.957215 | 1.181916 | **0.212957** |
| Scene7playing | Joint | **0.123718** | **0.015272** | **0.888460** | **1.067605** | 0.579210 |

两个场景的 `audio_total` 都改善，但退化模式并不相同：
`scene1_opera` 的 waveform、mono-LSD 和 LRE 同时变差；
`Scene7playing` 只有 audio-diff 和 LRE 变差。场景异质性意味着后续优化
不能只针对总体 micro 指标。

### 视频

| 场景 | 系统 | RGB-L1 | PSNR (dB) | SSIM |
|---|---|---:|---:|---:|
| scene1_opera | Visual-only | **0.153169** | **13.5421** | **0.028474** |
| scene1_opera | Joint | 0.155002 | 13.4788 | 0.026406 |
| Scene7playing | Visual-only | 0.055864 | **20.2898** | **0.768609** |
| Scene7playing | Joint | **0.053340** | 20.1994 | 0.759284 |

`scene1_opera` 的三个视频指标全部退化；总体 RGB-L1 的改善来自样本更多的
`Scene7playing`。因此必须同时保留 macro、micro 和逐场景结果。

## 训练节点趋势

| Step | Audio-only audio_total | Joint audio_total | Visual-only RGB-L1 | Joint RGB-L1 | Visual-only PSNR | Joint PSNR |
|---:|---:|---:|---:|---:|---:|---:|
| 5k | **0.620278** | 0.628157 | 0.082203 | **0.081002** | **18.1613** | 18.1161 |
| 10k | 0.596897 | **0.570678** | 0.083150 | **0.081541** | **18.1807** | 18.1362 |
| 30k | 0.539302 | **0.461376** | 0.085769 | **0.084584** | **18.2161** | 18.1340 |

联合模型的 `audio_total` 优势在 10k 后出现并随训练扩大；视频侧则始终表现为
RGB-L1 较好、PSNR 较差。下一轮应优先针对 LRE 和视觉多指标约束做消融，
而不是简单延长训练。

## 可复核产物

- [`aggregate_report.json`](aggregate_report.json)：完整 suite 聚合、配对统计、
  场景报告哈希和 provenance 引用。
- [`aggregate_metrics.csv`](aggregate_metrics.csv)：5k/10k/30k、native、
  macro/micro 的长表指标。
- [`scene_30k_metrics.csv`](scene_30k_metrics.csv)：两个场景在 30k 的
  update-matched 系统绝对均值，用于复核逐场景表。

校验信息：

| 对象 | SHA-256 |
|---|---|
| suite 内容哈希 | `cc57caeaa1ab7e170c27b57be9eae80d5b11d4965d978b4eeee21a835e483c4a` |
| aggregate_report.json 文件 | `4db9385d22bd6459890c7df137b2350beb1684cb9eacd3bf38953dce884a4d4b` |
| aggregate_metrics.csv 文件 | `cd688e58f44b2cf388b82b0d5ce9ed97c037837a7c148925ab0608ab38cc7f3c` |
| scene_30k_metrics.csv 文件 | `2b1dcd20d1cbe74bb072ea1aeaffa002c145561569d82359a005597a1a908f0c` |
| scene1_opera 内容哈希 | `e430279e02a5852b5ddcbf4039ce519ffc9b248c343ec5de3fbb801057931807` |
| Scene7playing 内容哈希 | `043e5d4d7c1dc18717311368ee3a2452480c2ce9088cdd7a5609b42cc5940654` |

正式流水线的独立 `--verify-only` 返回相同 suite 内容哈希，项目完整测试为
648/648 通过。JSON 中的本地绝对路径仅用于记录本次运行的 provenance；
版本库内的结果阅读和数值复核不依赖这些路径。

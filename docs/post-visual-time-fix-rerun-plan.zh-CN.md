# cam38 visual-time 修复后的重跑计划

本文记录 held-out cam38 `visual_time` 修复后的结果失效范围、重跑优先级和执行
方式。它是后续服务器实验的操作清单。

## 1. 修复内容

修复前，训练相机从 FreeTimeGS++ `time.memmap` 读取模型时间，但不在 train-only
memmap 中的 held-out cam38 使用：

```text
visual_time = frame_index / fps
```

这把物理秒数替代成了 FreeTimeGS++ 模型时间。两者在实际数据中可能不同，例如
测试夹具中：

```text
time_seconds = 0.1
visual_time = -0.8
```

修复后，held-out camera 使用同一帧 cam00-cam37 共享的 `time.memmap` 值。
model time 是 frame-specific、camera-independent 的 FreeTimeGS++ 坐标，因此：

- 不读取 cam38 RGB、depth 或其他 held-out image target；
- 不改变 cam00-cam37 的训练样本；
- 不把 physical time 与 model time 混用；
- 如果同一帧的 train-camera model time 不一致或非有限，立即失败。

本次同时修复：

- audio-only 训练直接调用 `forward_audio_only()`，不再运行视觉后端；
- visual-only 训练直接调用 `render_rgbd()`，不再运行音频后端；
- learning rate 必须有限且为正；
- loss weight 必须有限且非负；
- `lambda_audio` 和 `lambda_rgb` 不能同时为零。

## 2. 哪些旧结果失效

所有使用 held-out cam38 视觉渲染的结果都必须视为 **pre-fix historical
results**，不能继续作为当前代码的有效排名：

- FiLM `joint_conditioned`；
- `visual_only` 和 native FreeTimeGS++ cam38 视觉指标；
- Direct conditioned U-Net；
- Gated native residual；
- Gaussian-token Cross-Attention；
- Mask-protocol Cross-Attention；
- Query-dependent P1；
- Spatial P1；
- correct/no-RGBD/shuffled-RGBD/wrong-camera 等所有 cam38 因果评估。

需要注意：

- `no-RGBD` 系统虽然不把视觉 condition 传给音频后端，但现有评估报告仍包含
  rendered RGB，因此其完整评估产物也应重建。
- 旧数值可以保留用于追踪修复影响，但新报告必须使用新的输出根目录和新的内容
  hash，不能覆盖旧结果后继续沿用旧哈希。

## 3. 哪些内容不需要重跑

### 3.1 不需要重训上游 native 模型

下面的上游训练没有使用错误的 held-out `visual_time`：

- Native AudioGS 训练；
- Native FreeTimeGS++ 训练；
- cam00-cam37 资产准备、COLMAP、flow 和 native contracts。

因此可以复用已经验证的 native contracts，并在基础 suite 中使用：

```bash
--skip-native-training
```

### 3.2 从科学计算角度，旧 continuation checkpoint 参数可复用

30k continuation 训练只使用 cam00-cam37，这些样本一直从 `time.memmap` 读取
正确 model time。因此从模型参数角度，以下 checkpoint 不因本次时间修复而改变：

- GS-only/audio-only 30k；
- visual-only 30k；
- FiLM 30k；
- Direct/Gated 30k；
- Gaussian-token、Mask、P1、Spatial P1 和 Plain U-Net 30k。

理论上的最小工作量是：复用 checkpoint，重新执行所有 5k/10k/30k cam38
evaluation 和 report。

但是当前 strict runner 将 Git revision、source inventory 和 preparation
evidence 绑定到 checkpoint。修复源码后，它会正确拒绝把旧 checkpoint 冒充为
当前代码产物。因此不要绕过校验或手工改 hash。

## 4. 实际可执行的两套方案

### 4.1 推荐方案：按严格 runner 全量重建 continuation

优点：

- 不绕过 evidence；
- checkpoint、evaluation 和 report 都绑定修复后的源码；
- 可独立复核。

代价：

- 所有 continuation 需要重新训练；
- 但 native AudioGS/FTGS++ 可复用，不需要重新训练上游。

### 4.2 受控迁移方案：实现 audited checkpoint migration 后只重评估

只有在新增一个正式 migration 工具后才允许使用。该工具至少需要验证：

1. 旧 checkpoint 的训练 sample IDs 全部属于 cam00-cam37；
2. 训练 memmap 的 time hash 未变化；
3. 模型、optimizer 和训练配置未变化；
4. 本次源码变化只影响 held-out time lookup 和单模态执行隔离；
5. 生成新的 evidence，保留旧 checkpoint hash 和迁移说明；
6. 新 evaluation 使用独立输出目录。

当前仓库没有这个 migration 工具，所以现阶段不要手工复制 checkpoint 后跳过
strict verification。

## 5. 重跑前置工作

### 5.1 建立真正的统一实验分支

当前 `clean/codebase-structure` 包含：

- FiLM；
- Direct/Gated；
- Gaussian-token Cross-Attention；
- 基础 cam38 benchmark。

以下实现仍在其他分支：

- Mask Cross-Attention、P1、Spatial P1：
  `origin/agent/p1-spatial-camera-contrast`
- Plain U-Net：
  `origin/agent/plain-unet-baseline`

重跑这些系统前，必须把本次修复和 clean 文档提交移植到对应分支，或者先建立
统一模型 registry 分支。禁止直接在旧实验分支上运行，因为旧分支仍包含错误的
cam38 time fallback。

### 5.2 使用新输出目录

不要覆盖：

```text
runs/cam38_benchmark/
runs/audio_architecture_ablation/
runs/cross_attention_ablation/
runs/cross_attention_masks_ablation/
runs/query_dependent_p1_cam38/
runs/query_dependent_p1_spatial_cam38/
runs/plain_unet_cam38/
```

建议统一追加：

```text
_visual_time_v2
```

例如：

```text
runs/cam38_benchmark_visual_time_v2/
runs/query_dependent_p1_cam38_visual_time_v2/
```

## 6. 必须重跑的优先级

### P0：数据和渲染正确性 smoke

在启动 GPU 长任务前验证两个场景：

1. cam38 sample 的 `time_seconds` 与 `visual_time` 不再被当作同一数值；
2. 同一 frame 的 cam00-cam37 model time 完全一致；
3. cam38 RGBD render 有限且非空；
4. 修复前后使用同一个 checkpoint 时，cam38 render 确实发生预期变化；
5. cam00-cam37 render 在修复前后保持一致。

建议输出：

```text
results/visual_time_v2_smoke/<scene>/comparison.json
results/visual_time_v2_smoke/<scene>/before.png
results/visual_time_v2_smoke/<scene>/after.png
```

### P1：三条主结论路线

先重跑：

1. GS-only 30k；
2. FiLM residual；
3. Query-dependent P1。

原因：

- GS-only 是 waveform/LRE 基线；
- FiLM 是证据最完整的视觉条件 baseline；
- P1 是修复前综合结果最好的候选。

每个场景、每个系统必须重新评估：

```text
5k / 10k / 30k
```

P1 还必须重新评估：

```text
correct
no-RGBD
wrong-camera
```

只有 P1 在至少 3 个 seed 下仍同时满足以下条件，才允许称为主候选：

- `audio_total` 优于 GS-only；
- LRE 不超过预注册容差；
- correct 优于 no-RGBD；
- correct 优于 wrong-camera；
- scene1 和 Scene7 方向一致。

### P2：必要对照

重跑：

- Plain U-Net；
- Mask-protocol Cross-Attention；
- visual-only；
- native FreeTimeGS++ cam38 evaluation。

它们分别回答：

- U-Net 本身贡献多少；
- mask-protocol cross-attention 是否真的使用空间排列；
- 联合训练是否损害视觉；
- 修复后的 FTGS++ held-out render 基线是多少。

Mask Cross-Attention 必须重新做：

```text
correct / no-RGBD / shuffled-RGBD
```

### P3：负结果复核

资源允许时再重跑：

- Gaussian-token complex Cross-Attention；
- Direct conditioned U-Net；
- Gated native residual；
- Spatial P1。

这些路线修复前已显示明显问题。它们的价值主要是确认负结论是否仍成立，而不是
争夺默认模型。

Spatial P1 如果继续，不建议直接重复固定权重版本；优先测试只在早期启用、随后
衰减到零的 schedule。

## 7. 基础 suite 的执行方式

先在服务器设置生产 Python：

```bash
export AVGAUSSIANV2_PYTHON=/absolute/path/to/production/python
```

使用新输出根运行基础 suite，并复用 native contracts：

```bash
"${AVGAUSSIANV2_PYTHON}" -m avgaussianv2.cli.benchmark_suite \
  --repository "$(pwd)" \
  --output-dir "$(pwd)/runs/cam38_benchmark_visual_time_v2" \
  --python "${AVGAUSSIANV2_PYTHON}" \
  --gpus 0,1,2 \
  --skip-native-training
```

完成后只读验证：

```bash
"${AVGAUSSIANV2_PYTHON}" -m avgaussianv2.cli.benchmark_suite \
  --repository "$(pwd)" \
  --output-dir "$(pwd)/runs/cam38_benchmark_visual_time_v2" \
  --python "${AVGAUSSIANV2_PYTHON}" \
  --gpus 0,1,2 \
  --verify-only
```

不要对旧 `runs/cam38_benchmark` 使用 `--resume`。旧目录绑定 pre-fix source
identity，应保留为历史结果。

## 8. Cross-Attention / P1 / Plain 的执行要求

这些实验的 prepare 阶段依赖基础 suite 的 protocol 和 FiLM evaluations。使用
新根目录时，现有脚本中的固定路径也需要参数化或复制为 v2 runner，不能让新实验
继续引用旧：

```text
runs/cam38_benchmark/<scene>/protocol
runs/cam38_benchmark/<scene>/evaluations/joint_conditioned
```

v2 runner 必须改为引用：

```text
runs/cam38_benchmark_visual_time_v2/<scene>/protocol
runs/cam38_benchmark_visual_time_v2/<scene>/evaluations/joint_conditioned
```

每个实验遵循：

```text
prepare -> diagnose -> train -> eval(5k/10k/30k) -> report
```

P1/Mask/Plain 的实际命令应在统一实现分支落地后，由对应 runner 的 `--help` 生成，
不要从旧分支文档直接复制固定输出路径。

## 9. 新报告必须包含

### 9.1 核心指标

- `audio_total`
- `audio_mono`
- `audio_diff`
- waveform L1
- mono/diff LSD
- LRE
- RGB-L1、PSNR、SSIM

P1/Spatial 额外报告：

- ILD
- IPD
- binaural difference

### 9.2 聚合和因果统计

- scene1、Scene7 分开；
- macro 与 sample-weighted micro；
- mean、median、standard deviation；
- paired delta 与 win rate；
- 至少 3 个 seed 的均值、标准差和 bootstrap confidence interval；
- correct/no-RGBD/shuffled/wrong-camera 使用同一 checkpoint。

### 9.3 Provenance

- Git commit；
- config SHA-256；
- native and continuation checkpoint SHA-256；
- memmap metadata 和 time.memmap SHA-256；
- evaluation content SHA-256；
- 明确标记 `visual_time_semantics=shared_memmap_model_time_v2`。

## 10. 完成判定

满足以下条件后，旧排名才可以被新排名替换：

1. 两个场景 P0 smoke 通过；
2. P1 的 GS-only、FiLM、P1 全部完成 3 seeds；
3. 所有新 evaluation 通过独立 verify；
4. 新结果使用独立输出根和内容 hash；
5. 逐样本产物进入可复核存储，不只提交 Markdown；
6. README 和跨分支结果文档更新为 post-fix 数值；
7. pre-fix 结果保留历史标签，不与 post-fix 数字混合聚合。

## 11. 当前结果状态

在完成上述重跑前：

- `results/cam38_benchmark/2026-07-26/` 是历史 pre-fix 结果；
- `docs/experiment-results-across-branches.zh-CN.md` 中的排名是历史分析；
- P1、Plain、Mask、Spatial 报告同样属于 pre-fix；
- 不应再将 `0.451597`、`0.461376` 等数字描述为当前修复代码的已验证性能。

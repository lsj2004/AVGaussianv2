# post-fix 双 48GB GPU 实验计划（执行版）

更新日期：2026-08-01

## 1. 本轮要回答的问题

本轮只在统一 post-fix 代码、相同数据、相同初始化和相同 evaluator 下回答四个问题：

1. `audio_only`、`plain_unet`、`joint_conditioned`、
   `cross_attention_masks`、`query_dependent_p1` 中，哪些架构在两个场景上稳定占优；
2. 胜出架构加入 sign-sensitive LRE loss 后，能否降低左右能量比误差，同时守住
   AudioGS 重建质量、波形、频谱和视觉指标；
3. 5k 的早期排序是否能延续到 10k/30k，最终结论是否跨随机种子稳定；
4. 模型相对 Source Binaural、Mono、native AudioGS、GS-only 和视觉基线处于什么位置。

候选矩阵是计算上限，不是必须跑满的网格。每阶段结束立即聚合、淘汰和续训，禁止
在架构尚未确定时展开“全部架构 × 全部 LRE 权重 × 全部 seed”。

## 2. 结论边界

cam38 同时用于本轮模型选择和最终数字，因此本轮结论是固定 benchmark 上的
**工程级内部选择结论**，不是未参与选择的无偏测试性能。若需要发表级结论，必须另建
validation/test 分离协议，并确认 native AudioGS/FTGS++ 也没有使用 validation/test
相机；这会要求重训上游，不纳入本轮预算。

不同基线只能在其定义域内比较：

| 基线 | 比较性质 | 可比较指标 | 不能声称 |
|---|---|---|---|
| Source Binaural | 无训练的输入参考 | 音频、paper MAG/ENV/LRE/DPAM | 不是可部署预测模型 |
| Mono | 无训练的单声道参考 | 音频、paper MAG/ENV/LRE/DPAM | 不代表空间音频模型上限 |
| native AudioGS | 原生训练参考 | 音频指标 | 与 30k continuation update-matched |
| `audio_only` | 主 GS-only control | 音频指标；与 continuation update-matched | 不报告无意义的视觉质量 |
| `plain_unet` | 音频架构对照 | 音频指标；与 `audio_only` update-matched | 不报告视觉因果性 |
| 条件音频模型 | 主候选 | 音频 + 条件因果差值 | 不能用冻结载体 RGB 冒充视觉训练收益 |
| native FTGS++ / `visual_only` | 视觉参考 | RGB L1/PSNR/SSIM | 不报告音频指标 |

主模型表必须包含 `audio_only`；只和 Source/Mono/native AudioGS 比较是不充分且不公平的。

## 3. 固定协议与唯一变量

| 项目 | 固定值 |
|---|---|
| 场景 | `scene1_opera`、`Scene7playing` |
| split | cam00-cam37 训练，cam38 评估 |
| 样本数 | 130 + 293 = 423 |
| 主筛选 seed | 42 |
| 稳健性 seeds | 17、42、73 |
| batch / crop | 1 / 0.5 秒 |
| 条件 warmup | 2,000 updates |
| 主节点 | 5k、10k、30k main updates |
| 架构筛选 LRE | `lambda_lre=0` |
| native 初始化 | 两场景已验证的 AudioGS/FTGS++ contracts |
| 数值环境 | `PYTHONHASHSEED=42`、`CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| 主统计 | per-scene + scene macro；sample-weighted micro 仅补充 |

派生配置只能改变当前阶段授权的字段：

- 架构阶段：只允许 `model.audio_backend` 或 `model.audio_render_strategy`；
- LRE 阶段：只允许 `train.lambda_lre`；
- 多种子阶段：只允许 `train.seed == benchmark.seed`；
- 其余 config、native checkpoint、ordered dataset、sample-index sequence、预算和 evaluator
  必须相同。

生成器已对架构源配置做精确 delta 校验，架构、screening、confirmation 使用稳定的
`continuation_id`。同一配置的 5k checkpoint 必须原地续训到 10k/30k，禁止重新起跑。

## 4. GPU 与调度策略

用户授权 GPU 1、GPU 2，均为 48GB。不能假设两张卡始终空闲：2026-08-01 两次真实
smoke 时 GPU 1 被其他任务占用约 46GB，因此只安全使用 GPU 2。

启动规则：

1. 每次启动前查询目标 GPU；空闲显存至少 8GiB、利用率不超过 10%，否则 fail closed；
2. 当前每张卡最多 1 个活动 pipeline；空闲卡从中央队列领取下一个已满足依赖的任务；
3. 两卡都空闲时使用 `--gpus 1,2`，只有 GPU 2 空闲时使用 `--gpus 2`，不等待或抢占
   GPU 1 上的外部任务；
4. control/treatment 尽量在同一型号 GPU 上交错调度，两卡任务错开约 60 秒，避免
   500-step checkpoint 同时写盘；
5. OOM 只降低并发，不修改 batch、crop、模型或指标协议；同一基础设施失败最多自动
   恢复一次，第二次进入审计；
6. 只有完成单卡 1×/2× pipeline 吞吐 A/B，且总显存低于 38.4GB、无 OOM、总吞吐提升、
   单任务吞吐下降不超过 20%，才允许单卡 2 pipeline。

已测得 FiLM 5k smoke 峰值至少约 2.9GiB，P1 smoke 训练中约 2.3GiB；这只证明显存
充足，不证明多开能提速。P1 利用率明显低于显存占比，CPU、数据和渲染可能是瓶颈。

## 5. 已完成 smoke 与剩余开跑门禁

### 5.1 已通过的两个真实 smoke

Smoke 仅验证工程链路，均不进入性能表。

1. FiLM：`scene1_opera / joint_conditioned / seed42 / lambda_lre=0.02`，2k warmup +
   5k main，130 样本无 DPAM 评测通过；evaluation SHA-256：
   `d0791e59d77f47ae9321836d9794d49b897f5274550868fc54399e2f39d7e20b`。
2. P1：`scene1_opera / query_dependent_p1 / seed42 / lambda_lre=0.02`，2k warmup +
   5k main；主、no-RGBD、wrong-camera 各 130 样本并通过 verify-only。content SHA-256：
   - main：`fe496fc8d5d21df13937621c16ad94cc111fe42be28394cac9efd41847da9a3a`
   - no-RGBD：`8f4262e1bba61f44cf9658a6df3968f13f33a0abee5d44f7c2d20c8c20a95cff`
   - wrong-camera：`a169a6cd0d75604d118be5f4748a2d056fbfda4e749240b48a07d9604c57b292`

P1 smoke 真实验证了 5k 暂停、三路因果评测、失败后恢复和 verify-only。它暴露并修复：

- continuation 架构字段错误地使 native contract 失效；
- 嵌套因果 evaluation 父目录未创建；
- 已完成主评测恢复时必须 verify-only，不能覆盖。

### 5.2 正式长任务前仍必须完成

- 全量测试、`git diff --check` 通过，冻结单一 commit；
- generated config 不能使 worktree dirty；run manifest 必须绑定 clean Git revision；
- 失败 attempt 也必须保留阶段、耗时和资源证据，不能只发布成功/verify-only 最新指针；
- 两场景 visual-time 机器断言通过，pre-fix/post-fix 输出根完全隔离；
- Source Binaural 与 Mono 的 846 行参考结果完成 MAG/ENV/LRE/DPAM 并通过 verifier；
- `lambda_lre=0` 与无 LRE objective 的 loss、gradient、短程参数更新一致性测试通过；
- 主评测启用 DPAM；smoke 和因果反事实允许显式跳过 DPAM，但报告必须标注 incomplete。

任一门禁失败，不启动正式架构队列。

## 6. 分阶段实验矩阵

### P1：架构横向筛选

固定两个场景、seed42、`lambda_lre=0`、5k。

| system | 训练 mode | 正式因果评测 |
|---|---|---|
| `audio_only` | audio_only | main |
| `plain_unet` | audio_only | main |
| `joint_conditioned` | joint_conditioned | main / no-RGBD / wrong-camera |
| `cross_attention_masks` | joint_conditioned | main / no-RGBD / shuffled-RGBD |
| `query_dependent_p1` | joint_conditioned | main / no-RGBD / wrong-camera |

基础上限为 5 架构 × 2 场景 = 10 个 5k run。Gaussian-token
`cross_attention` 是条件项：只有短程因果检查显示 correct 与 no-RGBD/shuffled 有实质
差异，或五个必跑架构均未形成有效视觉候选时，才增加 2 个场景 run。

架构晋级条件：

- 两场景均完成，所有样本/loss/gradient 有限，evaluation verifier 通过；
- 主音频质量不能被 `audio_only` 在 MAG、ENV、DPAM、`audio_total`、waveform L1 上
  全面支配；
- 条件模型 correct 必须总体优于相应 no-RGBD，并且不能在两个场景都被 shuffled 或
  wrong-camera 稳定击败；
- LRE、ILD、IPD 不出现不可接受退化；
- Pareto 近似等价时优先参数更少、GPU-hours 更低的模型。

最多保留 2 个架构。5k 边界不清楚时只把边界候选延长到 10k，不凭微小均值差淘汰。

### P2：LRE 权重筛选

只对 P1 的 1–2 个胜出架构运行：

```text
lambda_lre = 0.00, 0.01, 0.02, 0.05
seed = 42
scenes = scene1_opera, Scene7playing
step = 5k
```

训练上限 16 个 5k run；架构阶段已有的 `lambda=0` 目录和 checkpoint 直接复用。非零
权重晋级要求：

- 两场景 scene-macro LRE 相对本架构 control 改善至少 15%；
- `audio_total` 与 waveform L1 相对退化各不超过 3%；
- MAG、ENV、DPAM 无预注册明显退化，非有限样本为 0；
- 不被更小权重在所有主指标上支配。

每架构最多保留 1–2 个非零权重，允许一个都不保留。

### P3：纵向收敛与多种子

先将胜出架构的 `lambda=0` 和非零候选从同一 5k checkpoint 续训到 10k。只有 10k
仍保持 Pareto 优势时才续训到 30k。报告 5k→10k→30k 的绝对值、paired delta、
paired win rate、因果差值和 guardrail 轨迹。

30k 门槛：两场景 macro LRE 相对 control 改善至少 10%，`audio_total`/waveform L1
退化不超过 3%，LRE paired win rate 至少 55%，且条件模型在两个场景总体优于因果
对照。不能用 Scene7 的 293 样本 micro 均值掩盖 scene1 的失败。

只有出现明确最终候选时，才为“最佳非零权重 + 同架构 lambda=0 control”增加 seed17、
seed73；新增上限 8 个 30k run，与 seed42 合成 3 seeds。最终报告 across-seed mean/std
和 95% paired hierarchical bootstrap CI。CI 跨零时结论写为“不确定”。

## 7. 指标与横纵向报告

横向比较固定 `(scene, seed, step, sample_id)`：

- 原生训练目标：`audio_total`、`audio_mono`、`audio_diff`；
- 波形/频谱：waveform L1、mono LSD、diff LSD；
- 论文口径：MAG、ENV、LRE、DPAM；
- 空间：signed/absolute LRE、ILD、IPD；
- 视觉：RGB L1、PSNR、SSIM，仅对确实输出/训练视觉的系统报告；
- 因果：correct−noRGBD、correct−shuffled、correct−wrong-camera；
- 资源：总参数/可训练参数、峰值显存、step time、GPU-hours、训练/评测吞吐。

纵向比较固定 `(scene, system, seed, lambda_lre, sample_id)`，只比较同一 continuation 的
5k/10k/30k。禁止拼接不同 seed、sample inventory、配置哈希或 evaluator 的节点。

主结论优先 per-scene 与两个 scene 等权 macro；423 样本 micro 只作补充。选择不构造
任意加权总分：先剔除证据不完整和 guardrail 失败项，再形成音频质量、空间质量、因果性、
资源成本的 Pareto front。

## 8. 统一命令与复用关系

```bash
PRODUCTION_PYTHON=/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python
STRICT_RUN_ROOT=/mnt/sda/lisujing/Dataset/AVGaussianFusionv2/.worktrees/scene1-pilot/runs/cam38_strict
GENERATED_ROOT=configs/generated/lre_loss_ablation
RUN_ROOT=runs/lre_loss_ablation_visual_time_v2

# 架构阶段：5 systems × 2 scenes，lambda=0，5k
$PRODUCTION_PYTHON scripts/generate_lre_ablation_configs.py \
  --stage architecture \
  --system audio_only \
  --system plain_unet \
  --system joint_conditioned \
  --system cross_attention_masks \
  --system query_dependent_p1 \
  --output-dir "$GENERATED_ROOT" \
  --strict-run-root "$STRICT_RUN_ROOT"

$PRODUCTION_PYTHON -m avgaussianv2.cli.benchmark_lre_run \
  --manifest "$GENERATED_ROOT/architecture/manifest.json" \
  --output-root "$RUN_ROOT" \
  --native-root "$STRICT_RUN_ROOT" \
  --gpus 1,2 \
  --python "$PRODUCTION_PYTHON" \
  --trust-upstream-artifacts

# P1 选出 survivor 后生成 LRE screening；示例中的 system 必须替换为真实 survivor。
$PRODUCTION_PYTHON scripts/generate_lre_ablation_configs.py \
  --stage screening \
  --system query_dependent_p1 \
  --output-dir "$GENERATED_ROOT" \
  --strict-run-root "$STRICT_RUN_ROOT"

$PRODUCTION_PYTHON -m avgaussianv2.cli.benchmark_lre_run \
  --manifest "$GENERATED_ROOT/screening/manifest.json" \
  --output-root "$RUN_ROOT" \
  --native-root "$STRICT_RUN_ROOT" \
  --gpus 1,2 \
  --python "$PRODUCTION_PYTHON" \
  --trust-upstream-artifacts \
  --resume
```

若 GPU 1 未通过预检，将 `--gpus 1,2` 改为 `--gpus 2`；禁止关闭预检。selection、
confirmation、robustness 继续使用 `benchmark_lre_select` 和同一 `RUN_ROOT`，确保稳定
`continuation_id` 原地恢复。

Source/Mono 参考单独运行，并在正式报告前 verify-only：

```bash
$PRODUCTION_PYTHON -m avgaussianv2.cli.evaluate_audio_references \
  --config configs/benchmark_cam38/scene1_opera.yaml \
  --config configs/benchmark_cam38/Scene7playing.yaml \
  --output-dir results/lre_loss_ablation_visual_time_v2/audio_references

$PRODUCTION_PYTHON -m avgaussianv2.cli.evaluate_audio_references \
  --output-dir results/lre_loss_ablation_visual_time_v2/audio_references \
  --verify-only
```

## 9. 产物与停止规则

每个结果必须绑定 clean Git commit、config/manifest/sample-sequence/checkpoint SHA-256、
native contract、GPU、CUDA/driver/PyTorch、evaluation content SHA-256 和完整 attempt history。
顶层 `runner_result.*.json` 只是最新指针；资源统计必须读取不可变 history，不能把
verify-only 的 14MiB/数十秒误作训练成本。

候选满足以下任一条件可停止，但必须保存 checkpoint、指标和机器可读原因：

- 非有限 loss、gradient 或输出；
- 降低并发后仍重复 OOM；
- 两场景、连续两个节点均被更简单模型支配；
- 条件模型与反事实条件基本相同；
- 非零 LRE 连续两个节点无改善且至少一个 guardrail 退化；
- 被更小 LRE 权重全面支配。

不能仅凭一个场景、一个 5k 波动或 CI 跨零的均值差自动淘汰。最终交付必须包含冻结
manifest、逐样本结果、per-scene/macro/micro/跨 seed 统计、横向表、纵向曲线、Pareto
图、资源报告、淘汰原因和一条命令可复核的 verifier 入口。

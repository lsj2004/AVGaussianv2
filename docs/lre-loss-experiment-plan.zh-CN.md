# Sign-sensitive LRE loss 实验与智能体执行协议

本文是 `configs/experiments/lre_loss_ablation.yaml` 的人类可读执行协议。目标是
验证 sign-sensitive LRE loss 是否能解决原 AudioGS magnitude-only diff loss 对
`L-R` / `R-L` 符号不敏感的问题，同时不显著损害原生 AudioGS 重建指标。

## 1. Goal 与硬停止条件

当前 goal：

> 完成 sign-sensitive LRE loss 的实现与验证，制定并执行多配置、多场景、
> 多 seed 的 LRE 消融实验；产出逐样本结果、聚合统计、置信区间、因果对照和
> 最终模型建议，所有预注册实验与完整统计完成后才停止。

执行智能体必须持续推进、恢复或记录阻塞，不能因单次失败、会话切换、GPU
抢占或已有趋势明显而提前结束。只有以下条件全部成立，才允许将 goal 标记为
complete 并停止：

1. 本文第 4 节的 16 个 screening run 全部完成并评估；
2. winner JSON 与 screening manifest 的 SHA-256 绑定，且按第 5 节规则选出；
3. 本文第 6 节的 36 个 confirmation run 全部完成；
4. 所有 5k/10k/30k milestone 都有完整 checkpoint 和逐样本评估；
5. 3,384 条 screening 和 22,842 条 confirmation 逐样本记录数量核对一致；
6. 所有 control/treatment pairing 均按
   `(scene, system, seed, step, sample_id)` 一一对应；
7. 第 8 节全部聚合、paired delta、win rate 和 10,000 次 bootstrap 已生成；
8. 所有非有限值、缺失值、重复样本、配置漂移和 hash 不一致均为零；
9. 最终报告同时给出通过与未通过 gate 的结果，不能只报告最佳配置；
10. 给出最终权重建议，或基于预注册 gate 明确得出“不启用 LRE loss”。
11. Source Binaural 与 Mono reference 共 846 条逐样本记录完整；
12. reference 和所有模型结果都计算同一协议的 MAG/ENV/LRE/DPAM。

如果服务器资产、权限或 GPU 长期不可用，智能体可以记录 `blocked` 状态和可复现
证据，但 goal 必须保持 active，不能把未运行的实验解释为完成。

## 2. Loss 定义

对 `[B, 2, T]` stereo waveform：

```text
E_left  = sum_t(left[t] ** 2)
E_right = sum_t(right[t] ** 2)
LRE_dB  = 10 * log10((E_left + epsilon) / (E_right + epsilon))
```

训练正则项：

```text
pred_scaled = pred_LRE_dB / 6.0
gt_scaled   = gt_LRE_dB / 6.0
loss_lre    = SmoothL1(pred_scaled, gt_scaled, beta=1.0)

audio_total_with_lre =
    lambda_audio * audio_base + lambda_lre * loss_lre
```

固定参数：

| 参数 | 值 | 原因 |
|---|---:|---|
| `lre_scale_db` | 6.0 | 约一个左右能量 4:1 的尺度，避免 dB 数值主导梯度 |
| `lre_epsilon` | `1e-8` | 防止静音声道产生 `log10(0)` |
| `lre_smooth_l1_beta` | 1.0 | 小误差二次、大误差一次，降低极端样本影响 |

`audio_base` 必须继续表示原 AudioGS criterion。不得把 LRE 项写回或重命名成
历史 `audio_total`；训练日志必须分别保留：

```text
audio / audio_base
audio_lre
audio_lre_weighted
audio_total_with_lre
pred_lre_db
target_lre_db
```

`lambda_lre=0` 是严格 control，优化目标应与修改前一致。LRE 数值仍可计算并记录，
但不得影响梯度。

## 3. 公平性与不变量

两个实验系统：

- `audio_only`：AudioGS Gaussian continuation，不使用视觉条件；
- `joint_conditioned`：当前 FiLM residual 视觉条件路线。

两个场景：

- `scene1_opera`：held-out cam38 共 130 个样本；
- `Scene7playing`：held-out cam38 共 293 个样本。

所有 treatment 必须与同一
`(scene, system, seed, step)` 的 `lambda_lre=0` control 配对，并保持以下内容
完全相同：

- cam00-cam37 train、cam38 test split；
- 修复后的 `visual_time` 路径；
- AudioGS/FreeTimeGS++ 初始化 checkpoint 及其 SHA-256；
- 模型结构、optimizer、learning rate、batch size；
- warmup、训练 sample 顺序和报告 checkpoint；
- evaluation sample IDs、顺序和 metric 实现；
- 除 LRE 四个配置字段外的所有有效配置。

已发布的 `configs/benchmark_cam38/*.yaml` 是冻结基线，不能修改。所有实验必须
使用派生 YAML 和独立输出根：

```text
configs/generated/lre_loss_ablation/
runs/lre_loss_ablation_visual_time_v2/
results/lre_loss_ablation_visual_time_v2/
```

### 3.1 AudioGS 论文参考线

除训练系统外，固定计算两条不训练的 reference baseline：

| Reference | 预测定义 |
|---|---|
| `source_binaural` | source viewpoint 的原始左右双声道直接作为 target-view prediction |
| `mono` | `mean(source_left, source_right)` 后复制到左右两个输出声道 |

两者使用与模型评估完全相同的 held-out cam38 sample IDs、0.5 秒 crop、采样率、
归一化和 target waveform。它们不依赖 seed，也不应为每个训练 seed 重复计算：

```text
(130 + 293) samples * 2 references = 846 per-sample rows
```

运行完整论文指标：

```bash
conda run -n avcloud python -m \
  avgaussianv2.cli.evaluate_audio_references \
  --config configs/benchmark_cam38/scene1_opera.yaml \
  --config configs/benchmark_cam38/Scene7playing.yaml \
  --output-dir \
  results/lre_loss_ablation_visual_time_v2/references
```

该命令默认要求 CDPAM 可用。`--skip-dpam` 只允许用于 smoke，会把报告明确标为
DPAM skipped；这种产物不能满足最终停止条件。这里使用 AudioGS 的 `avcloud`
环境，是因为项目轻量 `.venv` 默认不安装 CDPAM。

计算完成后独立验收，不加载数据、模型或 CDPAM，也不写文件：

```bash
UV_OFFLINE=1 uv run python -m \
  avgaussianv2.cli.evaluate_audio_references \
  --output-dir \
  results/lre_loss_ablation_visual_time_v2/references \
  --verify-only
```

### 3.2 论文指标协议

为避免同名指标混算，产物字段固定为：

| 字段 | AudioGS 论文口径 |
|---|---|
| `paper_mag` | 左右耳 magnitude STFT 各自 mean L1 后求和 |
| `paper_env` | 左右耳 Hilbert envelope RMSE 求和 |
| `paper_lre_db` | 左右能量比 dB 绝对误差，`epsilon=1e-5` |
| `paper_dpam` | CDPAM，WAV 写入后由 `cdpam.load_audio` 加载 |

MAG 固定使用 `n_fft=512`、`hop_length=160`、`win_length=400`、Hamming window、
centered STFT 和 constant padding。它不是上游脚本可选的 complex-STFT MSE。

现有 `lre_error_db` 是仓库历史 evaluator 指标，使用 `epsilon=1e-8`；主实验 gate
继续沿用它以保持历史可比性。`paper_lre_db` 使用论文 `epsilon=1e-5`，用于
Source Binaural / Mono / AudioGS 表格横向比较。两列必须同时保留，不能互相
覆盖。

所有 screening/confirmation 模型 checkpoint 在已有保护指标之外，还必须对
每个 sample 计算这四个 `paper_*` 指标。reference 和模型只有调用同一仓库
metric 函数并记录相同 protocol/hash 后，才允许进入同一张对比表。

这里的 “paper” 表示**指标定义与 AudioGS 论文一致**，不表示数值可直接与
论文 Table I 横向排名。本实验只覆盖本仓库的两个场景、cam38 split 和 0.5 秒
crop；除非数据集、场景、source viewpoint、归一化、时间窗和 sample 集合也与
论文完全一致，否则论文原表只能作为定义来源，不能作为数值基线。

## 4. 阶段 A：Screening

固定 seed `42`，训练到 5k：

| 维度 | 取值 |
|---|---|
| Scene | `scene1_opera`, `Scene7playing` |
| System | `audio_only`, `joint_conditioned` |
| `lambda_lre` | `0.0`, `0.01`, `0.02`, `0.05` |
| Seed | `42` |
| Report step | `5000` |

总计：

```text
2 scenes * 2 systems * 4 weights = 16 training runs
(130 + 293) * 2 systems * 4 weights = 3,384 per-sample rows
```

生成配置：

```bash
UV_OFFLINE=1 uv run python scripts/generate_lre_ablation_configs.py \
  --stage screening
```

预期输出为 `8 configs / 16 runs`。每份 config 被两个 system 使用，run manifest
位于：

```text
configs/generated/lre_loss_ablation/screening/manifest.json
```

### Screening gate

每个非零 candidate 均与同 scene/system 的 `lambda_lre=0` 比较：

```text
LRE relative improvement =
    (control_lre - candidate_lre) / control_lre

guardrail relative degradation =
    (candidate_metric - control_metric) / control_metric
```

基本 gate：

- 两个场景、两个系统均有结果；
- macro LRE 至少改善 15%；
- `audio_total` 相对退化不超过 3%；
- waveform L1 相对退化不超过 3%；
- 非有限样本为 0。

## 5. Winner 选择与冻结

必须选择恰好两个非零权重进入 confirmation。不能看过其他 seed 或 10k/30k
结果后再改候选。

对每个 candidate，按下面的固定键升序排序：

1. 非有限样本数；
2. 四个 scene/system 单元中的 gate violation 数；
3. 超过 guardrail 阈值的最坏幅度；
4. 双场景、双系统 macro `lre_error_db`；
5. `lambda_lre`，较小者优先。

即使不足两个 candidate 通过全部 gate，也按该排序选满两个，以便 confirmation
量化失败是否稳定；被选中不等于通过 gate。生成：

```text
results/lre_loss_ablation_visual_time_v2/screening/winners.json
```

严格 schema：

```json
{
  "schema": "avgaussianv2.lre-loss-screening-selection",
  "version": 1,
  "source_screening_manifest_sha256": "<64-char sha256>",
  "selected_lambda_lre": [0.01, 0.02]
}
```

这里的示例权重仅展示格式，实际值必须由 screening 排序产生。

## 6. 阶段 B：Confirmation

确认阶段只运行：

```text
lambda_lre = 0.0 + 两个 screening winners
```

矩阵：

| 维度 | 取值 |
|---|---|
| Scene | `scene1_opera`, `Scene7playing` |
| System | `audio_only`, `joint_conditioned` |
| `lambda_lre` | control + 2 winners |
| Seed | `17`, `42`, `73` |
| Report step | `5000`, `10000`, `30000` |

总计：

```text
2 scenes * 2 systems * 3 weights * 3 seeds = 36 training runs
36 * 3 milestones = 108 evaluations
(130 + 293) * 2 systems * 3 weights * 3 seeds * 3 steps
    = 22,842 per-sample rows
```

生成配置：

```bash
UV_OFFLINE=1 uv run python scripts/generate_lre_ablation_configs.py \
  --stage confirmation \
  --winners \
  results/lre_loss_ablation_visual_time_v2/screening/winners.json
```

预期输出为 `18 configs / 36 runs`。

30k 是确认性主结果。5k/10k 只用于分析收敛速度和训练稳定性，不能用早期最好值
替代 30k，也不能根据中途结果停止较差配置。

## 7. 多 seed runner 前置修复

当前已发布 strict cam38 runner 在以下位置将 seed `42` 写入协议：

- `BenchmarkConfig` 和 `BenchmarkCompatibility`；
- worker 的 `PYTHONHASHSEED` 与 `_seed_everything()`；
- production preparation/evaluation；
- asset audit 和 report provenance。

因此 `17/73` 派生配置目前不能直接交给 strict runner。执行智能体在启动
screening 前必须增加一个**独立的 LRE ablation 协议入口**：

1. seed 从派生 config/manifest 注入子进程启动环境；
2. Python、NumPy、Torch 和 CUDA 使用同一个 run seed；
3. shared sample indices 由 run seed 生成并写入 compatibility；
4. screening 接受 5k budget，confirmation 接受 30k 与 5k/10k/30k milestones；
5. evaluation/report 接受并核验该 run 的 seed/budget，而不是常量 42/30k；
6. frozen benchmark 入口仍只允许 seed 42/30k，原测试和 config hash 不变；
7. 新入口拥有独立 schema、output root、resume verification 和测试；
8. `--verify-only` 必须在不构造模型/数据、不写文件的情况下验证完整运行树。

禁止通过删除 strict seed 校验、手工改 manifest/hash 或复用 seed-42 sample order
来伪装多 seed。

## 8. 统计方案

主指标：

```text
lre_error_db = abs(pred_LRE_dB - target_LRE_dB)
```

保护指标：

- `audio_total`, `audio_mono`, `audio_diff`；
- `waveform_l1`, `mono_lsd`, `diff_lsd`；
- `rgb_l1`, `rgb_psnr`, `rgb_ssim`。

论文对比指标：

- `paper_mag`, `paper_env`, `paper_lre_db`, `paper_dpam`；
- 不作为 LRE weight 的筛选 gate，但必须逐样本保存；
- 用于回答模型相对 Source Binaural/Mono 是否真正改善。

必须输出：

1. 每个 sample 的原始 metric；
2. 每个 scene/system/seed/step/weight 的 mean、std、median；
3. scene macro 与按样本数加权的 micro；
4. 每个 seed 的结果及 across-seed mean/std；
5. treatment-control 的逐样本 paired delta；
6. `lre_error_db` paired win rate；
7. 95% paired hierarchical bootstrap CI。

Bootstrap 固定：

```text
seed = 20260731
resamples = 10000
hierarchy = scene -> seed -> paired sample
```

每次 resample 必须对 control/treatment 使用同一 paired sample 索引。报告至少给出：

- LRE absolute delta 和 relative improvement 的 CI；
- `audio_total`、waveform L1 relative degradation 的 CI；
- 每个 scene/system 分层 CI；
- 双场景 macro 和 sample-weighted micro CI。

不得把 423 个样本当成跨 seed 的独立模型重复。模型层面的不确定性必须同时展示
三个 seed 的离散结果；bootstrap 只补充 paired sample uncertainty。

### Confirmation gate

在 30k 上，每个 system 都必须满足：

- 两个场景都有完整结果；
- across-seed macro LRE 相对 control 至少改善 10%；
- LRE paired win rate 至少 55%；
- `audio_total` 相对退化不超过 3%；
- waveform L1 相对退化不超过 3%；
- 非有限样本为 0。

最终推荐优先选择通过全部 gate 的较小 `lambda_lre`。如果两个候选均未通过，
默认保持 `lambda_lre=0`，并把失败原因写入结论，不能从未预注册权重中临时挑选
结果。

## 9. 运行、恢复与审计

每个 run 都必须保存：

```text
resolved config + SHA-256
source revision/tree state
initialization checkpoint paths + SHA-256
seed and sample-index SHA-256
periodic/milestone checkpoints
training loss history
gradient norms
evaluation metrics_per_sample.jsonl/csv
evaluation summary and provenance
stdout/stderr attempt logs
completion/verification status
```

智能体调度规则：

1. 先运行 CPU tests、config generation tests 和单步 GPU smoke；
2. screening 全部完成并验证后才允许生成 winners；
3. confirmation config 生成后冻结，不因中途结果更改；
4. GPU 抢占或进程失败时从已验证 checkpoint `--resume`；
5. 同一 run 连续失败三次时生成 incident 记录，定位并修复根因后继续；
6. 修复若改变 loss、数据、metric 或 sample order，旧阶段结果全部失效并重跑；
7. 仅影响日志/调度且不改变数值语义的修复，需记录 diff/hash 后可恢复；
8. 每轮调度后运行独立 verifier，不能只依赖进程退出码。

## 10. 最终产物

结果根目录至少包含：

```text
screening/
  manifest.json
  winners.json
  per_sample.jsonl
  aggregate.json
confirmation/
  manifest.json
  per_sample.jsonl
  aggregate.json
  bootstrap.json
  paired_deltas.csv
verification.json
README.md
references/
  metrics_per_sample.jsonl
  metrics_per_sample.csv
  aggregate.json
  verification.json
```

最终 `README.md` 必须回答：

1. magnitude-only diff 符号歧义是否被训练 loss 消除；
2. audio-only 和 joint-conditioned 是否得到一致结论；
3. 改善是否跨两个场景、三个 seed 稳定；
4. 代价是否超过 AudioGS reconstruction/waveform guardrail；
5. 推荐的 `lambda_lre`、适用范围和仍未覆盖的风险；
6. 与历史 pre-visual-time-fix 数字的关系，且不得混为同一结果集。
7. Source Binaural、Mono、control 与 LRE treatment 的
   `paper_mag/paper_env/paper_lre_db/paper_dpam` 同口径表格。

## 11. 代码提交前的本地验证

不依赖服务器数据的最低验证：

```bash
UV_OFFLINE=1 uv run python scripts/generate_lre_ablation_configs.py \
  --stage screening \
  --output-dir /tmp/avgaussianv2-lre-configs

UV_OFFLINE=1 uv run --with pytest pytest -q \
  tests/test_audio_references.py \
  tests/test_benchmark_metrics.py \
  tests/test_aligned_dataset.py \
  tests/test_training.py \
  tests/test_config_and_contracts.py \
  tests/test_lre_ablation_configs.py \
  tests/test_benchmark_training.py \
  tests/test_benchmark_worker.py

UV_OFFLINE=1 uv run --with ruff ruff check .
git diff --check
```

完整 GPU 实验不在本地伪造。缺少 `/mnt/sda` 资产或 CUDA 时，应明确记录为尚未
执行，而不是用 toy test 代替实验结论。

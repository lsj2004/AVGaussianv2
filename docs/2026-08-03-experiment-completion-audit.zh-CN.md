# 实验计划逐要求完成审计

更新日期：2026-08-03 07:30（Asia/Shanghai）

本文把 `2026-07-31-post-fix-two-gpu-experiment-plan.zh-CN.md` 的要求逐项绑定到权威
证据。状态只允许“完成”“进行中”“条件待执行”“待执行”；目录存在、manifest 已生成或
进程退出均不能替代 verifier 成功。

## 1. 代码与协议门禁

| 要求 | 状态 | 权威证据 |
|---|---|---|
| 正式训练代码冻结且 clean | 完成 | commit `98d158e4e9f1389aa924422faef901b2673ca3bc` |
| 两个真实 GPU smoke | 完成 | 计划 §5.1；P1 smoke 含三路 causal、恢复与 verify-only |
| post-freeze 全量回归 | 完成 | `452 passed`；Ruff、format、zsh、diff-check 通过 |
| P2/P3 continuation 身份与 evaluator 对齐 | 完成 | manifest loader、evaluation verifier、checkpoint/runtime-contract 复核 |
| 最终报告证据链冻结 | 完成 | report root `d39c81e`；builder `bdf2ed0e...`；renderer `be5918bb...`；relay `e907f7d`；token-bound receipts |

代码门禁完成不等于实验目标完成；下表中的运行与最终产物仍必须闭环。

## 2. 实验矩阵

| 阶段 | 要求 | 当前状态 | 完成证据/剩余条件 |
|---|---|---|---|
| P2 | 4 架构 × 4 lambda × 2 场景 × seed42，5k | 完成 32/32 | `p2_empirical_selection.json` 与独立 matrix verifier |
| P2 噪声 | 4 架构 × 2 场景 test-retest | 完成 8/8 | `noise_retest1_report.json` |
| P3 10k | 2 架构 × treatment/control × 2 场景 | 完成 8/8 | `p3_10k_gate.json`，8 个精确 10k evaluation |
| P3 30k | 同上 8 个 continuation | 进行中 2/8 | manifest SHA `002a74d...`；第三条正在训练 |
| causal gate | finalist 候选的 correct/no-RGBD/wrong/shuffled 条件验证 | 待执行 | P3 30k 完成后由 strict relay 运行并验证 |
| 候选多 seed | finalist/control 的 seed17/73、两场景 30k | 条件待执行 | 仅在 30k+causal gate 选出 finalist 时运行 |
| Audio-only seed42 | 两场景同 checkpoint 5k→10k→30k | 待执行（必跑） | P3 候选链结束后由 Audio-only relay 运行 |
| Audio-only seed17/73 | 两 seed × 两场景 30k | 条件待执行 | 有 finalist 时用于严格 3-seed 主榜；无 finalist 时不伪造候选多 seed |

P3 30k 当前正式完成项均为 `cross_attention_masks/lambda=0`：

- `scene1_opera`：130 samples，evaluation SHA `93f3f9e3...`；
- `Scene7playing`：293 samples，evaluation SHA `7da858b5...`。

## 3. 基线与比较边界

| 比较对象 | 状态 | 公平边界 |
|---|---|---|
| Audio-only | 5k 完成；30k/多 seed 待执行 | 最终主榜必须 update/seed/scene/sample/evaluator matched |
| Source Binaural | 完成并校验 | 输入参考，不是训练模型；只比较 7 个公共输出指标 |
| Mono | 完成并校验 | 通道对称退化参考；低 LRE/ILD/IPD 不等于空间定位正确 |
| native AudioGS | 完成 | metric-matched 描述性参考，不是 update-matched 主榜 |
| GS-only/视觉基线 | P2 相关证据存在 | 只报告其有定义的模态指标，不补造音频或视觉字段 |

Source/Mono 的正式根哈希：aggregate `d10e68a5...`、verification `38aab786...`；P2 fair
report 为 `920364e1...`。最终生成器会重新验证原始文件而不是只信任汇总数值。

## 4. 架构、参数与资源

| 要求 | 状态 | 权威证据 |
|---|---|---|
| 四架构完整 active hyperparameters | 完成 | `architecture_config_report.json`，SHA `dc4bb138...` |
| 两场景 CUDA 参数审计 | 完成 | `p2_cuda_parameter_audit.json`，SHA `85299ea4...` |
| P2 32-run 资源成本 | 完成 | `p2_resource_report.json`，SHA `4db3fbc...` |
| Query P1 orphan-trainable 披露 | 完成 | 7,865,794 elements；不计入 optimizer-active |
| 最终 JSON/Markdown 架构段 | 实现完成、产物待生成 | Audio-only relay 完成后自动生成 |

P2 资源只代表 5k screening，不得外推成 30k 最终成本。`plain_unet` 在正式 P2 32-run
矩阵前淘汰，只能作为早期证据，不能写成 P2 完成架构。

## 5. 最终交付物审计

| 交付物 | 状态 | 完成条件 |
|---|---|---|
| 冻结 manifests 与逐样本结果 | 进行中 | P3/causal/seeds/Audio-only 所有实际运行项均 verify |
| per-scene、macro、micro、跨 seed 统计 | 待执行 | 最终矩阵完整后生成；无 finalist 时明确跳过候选多 seed |
| 横向公平模型榜 | 待执行 | candidate/control/Audio-only 的 30k、3 seeds、2 scenes（若有 finalist） |
| Source/Mono/native 绝对参照榜 | 生成器完成、最终产物待执行 | 与模型公共指标及逐指标 delta 完整输出 |
| 5k→10k→30k 纵向曲线 | 进行中 | 当前已记录两个 control；等待其余 P3 与 Audio-only |
| 横向指标图与资源 Pareto 图 | 实现完成、最终执行待完成 | 从最终 verified JSON 自动生成，不混入不公平训练预算 |
| 5k→10k→30k 纵向图 | 待实现并执行 | 等全部适用 continuation 节点完成后从 verified evaluation 生成 |
| 淘汰原因 | 进行中 | 30k/causal gate 机器可读 reasons 完整保存 |
| 一条命令 verifier | 实现完成、最终执行待完成 | 最终 report receipt 与全部根哈希一致 |

## 6. 完成判定

当前总目标**未完成**。直接阻塞完成判定的缺口依次是：

1. P3 30k 余下 6/8；
2. causal gate；
3. 条件候选 multi-seed（若有 finalist）；
4. 必跑 Audio-only seed42/30k，以及有 finalist 时的 seed17/73；
5. 最终报告、Pareto/纵向产物与逐要求独立复核。

只有上述实际适用项完成且最终 verifier/receipt 成功后，才能把目标标记为完成。

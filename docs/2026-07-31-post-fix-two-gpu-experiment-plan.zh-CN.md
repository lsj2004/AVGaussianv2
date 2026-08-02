# post-fix 双 48GB GPU 实验计划（执行版）

更新日期：2026-08-03（P2 完成、P3 10k 门禁后执行版）

当前状态：P2 的 32/32 个 5k run、8/8 个 test-retest run 和 P3 的 8/8 个 10k
continuation 均已完成并独立复核。P3 10k 门禁保留
`cross_attention_masks/lambda_lre=0.01` 与
`query_dependent_p1/lambda_lre=0.02`。首条 30k continuation 在外部 GPU 进程进入时由
所有权 watchdog 安全终止于精确 18k checkpoint；污染恢复 supervisor 已重新排队，待
GPU 1/2 任一卡连续三次 clean 后从 18k 恢复。
本文同时保留最初预注册规则与执行后修订，不能把探索性扩展事后表述成预注册实验。

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

执行中经用户明确批准增加了一次有边界的 P2 探索性扩展：淘汰 `plain_unet`，对其余
四个架构统一跑四个 LRE 权重、两个场景和 seed42。该扩展用于回答“空间约束加入后哪个
架构最好”，不改变 P3 仍只允许最多两个 treatment 继续消耗 10k/30k 预算的原则。

## 2. 结论边界

cam38 同时用于本轮模型选择和最终数字，因此本轮结论是固定 benchmark 上的
**工程级内部选择结论**，不是未参与选择的无偏测试性能。若需要发表级结论，必须另建
validation/test 分离协议，并确认 native AudioGS/FTGS++ 也没有使用 validation/test
相机；这会要求重训上游，不纳入本轮预算。

不同基线只能在其定义域内比较：

| 基线 | 比较性质 | 可比较指标 | 不能声称 |
|---|---|---|---|
| Source Binaural | 无训练的输入参考 | waveform、MAG/ENV/DPAM/LRE、ILD/IPD | 不是可部署预测模型；没有训练 loss |
| Mono | 无训练的通道对称参考 | waveform、MAG/ENV/DPAM/LRE、ILD/IPD | 低 LRE/ILD 不等于空间定位正确；没有训练 loss |
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

用户授权 GPU 1、GPU 2，均为 48GB。不能假设两张卡始终空闲：2026-08-01 前两次真实
smoke 时 GPU 1 被其他任务占用约 46GB，DPAM smoke 前两张卡均已空闲。正式启动仍以
即时 preflight 为准，不把任一时刻状态写死到调度配置。

启动规则：

1. 每次启动前查询目标 GPU；空闲显存至少 8GiB、利用率不超过 10%，否则 fail closed；
2. 当前每张卡最多 1 个活动 pipeline；空闲卡从中央队列领取下一个已满足依赖的任务；
3. 两卡都空闲时使用 `--gpus 1,2`；只有一张授权卡空闲时使用该卡，不把 GPU 1 或
   GPU 2 设为必需卡，也不等待或抢占另一张卡上的外部任务；
4. control/treatment 尽量在同一型号 GPU 上交错调度，两卡任务错开约 60 秒，避免
   500-step checkpoint 同时写盘；
5. OOM 只降低并发，不修改 batch、crop、模型或指标协议；同一基础设施失败最多自动
   恢复一次，第二次进入审计；
6. 只有完成单卡 1×/2× pipeline 吞吐 A/B，且总显存低于 38.4GB、无 OOM、总吞吐提升、
   单任务吞吐下降不超过 20%，才允许单卡 2 pipeline。

P3 采用逐 shard 自适应调度：每次资源选择要求连续三次 clean sample；双卡均 clean
时领取两个 run，否则任一 clean 卡领取一个 run；每个 shard 完成后重新选择。GPU
所有权按 UUID 和 PID/starttime 后代树监控，忽略已经没有 `/proc` 身份的 stale NVML
记录，但对真实外来 PID fail closed。OOM 或资源变化只改变并发，不改变科学配置。

FiLM/P1 训练期间人工观测的设备占用约 2.9/2.3GiB；P1 三路评测的首个完整 runner
attempt 用时 836.22 秒，采样到的设备占用峰值 1,033MiB、利用率峰值 16%。随后
verify-only 仅用 27.00 秒、14MiB，因此不能用最新指针估算训练或完整评测成本。这些
数据只证明显存充足，不证明多开能提速；CPU、数据读取和渲染可能是主要瓶颈。

正式 P1 的 10 个 run 均成功。scene1/Scene7 单 pipeline 总用时分别为：`audio_only`
882.67/1491.09 秒、`plain_unet` 777.67/1196.84 秒、`joint_conditioned`
2612.20/4209.60 秒、`cross_attention_masks` 2466.50/4404.60 秒、
`query_dependent_p1` 3034.10/4632.00 秒；峰值显存范围 1,418--2,997MiB。22 份
主/因果 evaluation 已逐份独立 verify-only。由此确认瓶颈是训练后多路 DPAM 评测，P2
仍保持每卡一个 pipeline，不做未经 A/B 验证的单卡多开。

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

2026-08-01 再次对三路现有产物执行独立 verifier，样本数和上述三个 content SHA-256
完全一致。前两项 smoke 明确没有覆盖 DPAM，也不能用于比较架构性能。

3. DPAM 双运行时：复用 P1 5k checkpoint，在 FreeTimeGS++ Python 加载模型、由持久化
   `avcloud` worker 计算 CDPAM；130/130 行均含 `paper_dpam` 并通过独立 verifier。
   evaluation content SHA-256：
   `98e7508bfa98f0871aba46c6e3d219b1a3fdfe72099851f6bff01a0436002e5b`；
   CDPAM 权重 SHA-256：
   `2841b384b2423a34e282a66ea69dd608c9e585584f60d13337220d4cb69f08cb`。

DPAM smoke 首次诊断命令遗漏 `CUBLAS_WORKSPACE_CONFIG`，第二次在全部计算结束后因输出
父目录不存在而无法发布；两项均已修正。正式 runner 原本已强制注入确定性环境，独立
evaluator 现在也会在昂贵计算前创建输出父目录。

### 5.2 门禁状态与剩余工作

| 门禁 | 当前状态 | 正式开跑前动作 |
|---|---|---|
| clean revision / manifest / continuation identity | 正式 commit `98d158e4` 已冻结 | P3 全程保持该 commit |
| success、failure、peer-abort 的不可变 attempt history | 已实现并经历真实恢复 | 最终报告汇总所有 attempt |
| 两场景 visual-time 机器断言 | 已通过 | 最终 commit 再绑定一次审计结果 |
| Source/Mono 846 行完整参考指标 | 已在 frozen commit 覆盖重跑并通过 verifier | 最终表分开标注比较边界 |
| `lambda_lre=0` 等价于 legacy 更新 | loss、gradient、连续 10 次更新精确一致 | 全量测试复核 |
| P1 三路真实 GPU smoke | 已完成 | 不再重复训练 |
| 正式主评测 DPAM | 双运行时 130 样本 smoke 通过 | 全量测试复核 |
| P1 自动化筛选 | fail-closed selector 与生成器绑定已实现 | 全量测试复核 |
| frozen 正式代码质量 | 400 项 pytest、全仓 Ruff、diff-check 通过 | 已完成 |
| post-freeze 流水线修复 | 431/431 pytest、全仓 Ruff check、新增文件 Ruff format check、diff-check 通过 | P3 后再做最终合并 review |
| P2 主矩阵 | 32/32 5k + 8/8 test-retest，均独立复核 | 已完成 |
| P3 10k | 8/8 continuation，门禁与独立复核通过 | 已完成 |
| P3 30k / causal / seeds | 首条 continuation 精确 18k；污染恢复队列已运行 | 依门禁顺序继续 |
| update-matched Audio-only 闭环 | 2-run confirmation 与 4-run robustness manifest 已生成并由 frozen loader 验证 | P3 候选链结束后自动执行 |

正式训练 Python 没有 `cdpam`；已有 CDPAM 环境又没有 `gsplat/tinycudann`。主评测因此
使用已验证的持久化双运行时，并把解释器、实现、权重哈希写入 metric protocol。禁止
临时给训练环境安装未经锁定的依赖，也禁止因环境问题在正式主表跳过 DPAM。正式训练
始终使用冻结 commit；post-freeze 修复分支不得在 P3 中途替换训练代码。

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
- paper LRE、native LRE、ILD、IPD 的两场景 macro 相对 `audio_only` 退化均不超过 20%；
- Pareto 近似等价时优先参数更少、GPU-hours 更低的模型。

自动 selector 按以下固定顺序决策，不构造加权总分：先检查覆盖、身份、样本集合和有限性；
再执行条件因果门禁，其中 correct 相对 no-RGBD 的两场景 macro `audio_total` 改善至少
0.1%，且 alternate 不能在两个场景都优于 correct；随后剔除被 `audio_only` 全面支配的
候选并形成 Pareto front；若前沿
超过 2 个，依次用两场景等权 macro `audio_total`、MAG、ENV、DPAM、waveform L1、
paper LRE、native LRE、ILD、IPD 作词典序 tie-break，完全相同时按稳定 system ID 排序。
资源成本单独报告，不用于自动
打破指标差异，避免因不同数量的因果评测污染训练成本。selector 必须输出全部候选和至多 2 个
`selected_systems`；任何输入缺失都 fail closed，不允许人工补齐默认值。

最多保留 2 个架构；P3 再检验其 5k 排序能否延续到 10k/30k。P1 不在看到结果后
临时增加第三个 confirmatory 候选，以免破坏预注册预算和引入选择自由度。

P1 实测后，严格门禁唯一胜者是 `audio_only`。`query_dependent_p1` 的五个主音频质量
指标整体优于 `audio_only` 且因果门禁通过，但 paper LRE scene-macro 从 0.338dB 增至
0.912dB，未通过 20% 相对空间 guardrail。若只继续严格胜者，P2 将无法回答 LRE loss
能否修复最佳条件架构。因此额外保留一个明确标注为 **post-hoc exploratory** 的 rescue
槽：只有没有条件架构通过严格门禁时，才在“因果门禁通过、未被 `audio_only` 在质量
指标全面支配、paper LRE 绝对退化不超过 1dB”的条件架构中，按 paper LRE 绝对退化、
五个质量指标、稳定 system ID 排序取一个。本轮该槽为 `query_dependent_p1`。它进入 P2
但不能被报告为预注册胜者；最终 confirmatory 与 exploratory 结论必须分表。

### P2：LRE 权重筛选

最初计划只对 P1 的 1–2 个胜出架构运行。为避免把“`lambda=0` 时 audio-only 最稳”
错误推广为“其他架构加入空间约束后仍不行”，用户批准以下统一探索性矩阵：

```text
lambda_lre = 0.00, 0.01, 0.02, 0.05
seed = 42
scenes = scene1_opera, Scene7playing
step = 5k
systems = audio_only, query_dependent_p1, joint_conditioned, cross_attention_masks
```

实际规模为 4 架构 × 4 权重 × 2 场景 = 32 个 5k run。由于 selector 修复产生了新的
clean commit，P2 使用新的
`RUN_ROOT_V3`，并在新提交下重新训练 `lambda=0` control；禁止跨提交 continuation。
非零权重晋级要求：

- 两场景 scene-macro LRE 相对本架构 control 改善至少 15%；
- `audio_total` 与 waveform L1 相对退化各不超过 3%；
- MAG、ENV、DPAM 无预注册明显退化，非有限样本为 0；
- 不被更小权重在所有主指标上支配。

每架构最多保留 1–2 个非零权重，允许一个都不保留。

P2 已完成。`audio_only` 的正 lambda 没有稳定收益；四个架构内代表分别是
`audio_only/0`、`query_dependent_p1/0.02`、`joint_conditioned/0.01`、
`cross_attention_masks/0.01`。受预注册 P3 最多两个 treatment 的预算约束，10k 只继续
query P1/0.02、cross-attention/0.01 及各自 lambda=0 control；joint/0.01 保留为有效
5k Pareto 证据，而不是被描述为“架构失败”。

### P3：纵向收敛与多种子

先将胜出架构的 `lambda=0` 和非零候选从同一 5k checkpoint 续训到 10k。只有 10k
仍保持 Pareto 优势时才续训到 30k。报告 5k→10k→30k 的绝对值、paired delta、
paired win rate、因果差值和 guardrail 轨迹。

30k 门槛：两场景 macro LRE 相对 control 改善至少 10%，`audio_total`/waveform L1
退化不超过 3%，LRE paired win rate 至少 55%，且条件模型在两个场景总体优于因果
对照。不能用 Scene7 的 293 样本 micro 均值掩盖 scene1 的失败。

只有出现明确最终候选时，才为“最佳非零权重 + 同架构 lambda=0 control”增加 seed17、
seed73；这部分新增上限 8 个 30k run，与 seed42 合成 3 seeds。最终报告 across-seed
mean/std 和 95% paired hierarchical bootstrap CI。CI 跨零时结论写为“不确定”。

### P3-B：Audio-only 公平基线闭环

P2 的 Audio-only 正式结果停在 5k，而 P3 候选会训练到 10k/30k。跨步数比较只能作为
描述性参考，不能用于“超过 Audio-only”的主结论。为闭合最终公平主榜，增加以下必跑
基线，不以候选表现为触发条件：

1. 将 `audio_only/lambda_lre=0/seed42` 的两个 5k checkpoint 原地续训到 10k 和 30k；
2. 最终进入多 seed 阶段时，同步训练 `audio_only/lambda_lre=0` 的 seed17、seed73，
   两个场景均达到 30k；
3. Audio-only 与最终候选必须使用相同 sample inventory、数值环境、evaluator 和 30k
   main-update 口径；Audio-only 没有条件输入，不伪造 causal 指标；
4. 最终模型主榜以 3 seeds × 2 scenes 的 30k 结果比较候选、同架构 lambda=0 control
   和 Audio-only。seed42 的 10k 结果只用于补齐纵向曲线与早期收敛诊断。

该闭环新增 2 个 seed42 continuation 和 4 个新 seed 30k run。它是公平性所必需的
实验，不是看到结果后的超参数搜索；不得用 Audio-only 5k 数字替代。

生成器已在 post-freeze commit `74a8545` 加入，新增与相邻测试 16/16；最终公平报告
生成器在 commit `fee48d1` 加入。全量 pytest 431/431 通过。正式内容寻址产物为：

- seed42 confirmation：
  `configs/generated/audio_only_final_baseline/confirmation/manifest.json`，2 runs，
  SHA-256 `5ff107105afdb699fe844d36f76a1a02b44bfb640686cd063069a2c624b1ca2b`；
- seed17/73 robustness：
  `configs/generated/audio_only_final_baseline/robustness/manifest.json`，4 runs，
  SHA-256 `503c711a1560411e2642cf44e25d0674c496d6769fc5b0212283a4a08c8012b0`。

两份 manifest 均绑定正式 clean commit `98d158e4`；seed42 复用原 config SHA 和
`continuation_id`，seed17/73 的派生 config 机器断言只改变
`train.seed == benchmark.seed`。冻结 runner loader 已独立接受 2/2 与 4/4 run。

10k 门禁已完成：cross-attention/0.01 相对同架构 control 的 scene-macro LRE 改善
10.07%，query P1/0.02 改善 38.18%，二者均通过 noise-aware guardrail 并进入 30k。
这些是架构内 treatment/control 结论，不能解释为已经超过 `audio_only`、Source 或 Mono。
P3 10k 中期报告见
`docs/2026-08-03-p3-10k-gate-and-reference-report.zh-CN.md`。

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

最终报告固定拆成两张并排表，禁止混成单一排行榜：

1. **严格公平主榜**：只含 update-/seed-/sample-/metric-matched 的模型，报告绝对值、
   `Delta vs audio_only`、逐样本胜率和 Pareto 状态，用于模型选择；
2. **绝对参照榜**：含 Source Binaural、Mono、native AudioGS、audio-only 和最终候选，
   只比较共同定义的 waveform、MAG、ENV、DPAM、LRE、ILD、IPD，并分别给出
   `Delta vs Source`、`Delta vs Mono`、`Delta vs audio_only`。Source/Mono 缺失的
   `audio_total/audio_mono/audio_diff` 保持缺失，禁止补零。

Mono 左右通道对称会天然压低部分能量比误差；若 Mono 的 LRE/ILD 低于双耳模型，只能
陈述该数值事实，不能据此宣称其空间定位更好。Source/Mono 的 DPAM 若相同，也必须明确
标注该指标对两种参照缺乏区分力，不能用它单独支持空间结论。

## 8. 统一命令与复用关系

```bash
PRODUCTION_PYTHON=/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/.venv/bin/python
STRICT_RUN_ROOT=/mnt/sda/lisujing/Dataset/AVGaussianFusionv2/.worktrees/scene1-pilot/runs/cam38_strict
GENERATED_ROOT=configs/generated/lre_loss_ablation
P3_GENERATED_ROOT=configs/generated/lre_loss_ablation_p3
RUN_ROOT=runs/lre_loss_ablation_visual_time_v2
RUN_ROOT_V3=runs/lre_loss_ablation_visual_time_v3

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

# 仅在 DPAM 双运行时集成 smoke 通过后执行；正式主评测不得使用 --skip-dpam。
$PRODUCTION_PYTHON -m avgaussianv2.cli.benchmark_lre_run \
  --manifest "$GENERATED_ROOT/architecture/manifest.json" \
  --output-root "$RUN_ROOT" \
  --native-root "$STRICT_RUN_ROOT" \
  --gpus 1,2 \
  --python "$PRODUCTION_PYTHON" \
  --dpam-python /home/lisujing/miniconda3/envs/avcloud/bin/python \
  --trust-upstream-artifacts

# P1 selector fail closed；输出最多两个 survivor。
$PRODUCTION_PYTHON -m avgaussianv2.cli.benchmark_architecture_select \
  --manifest "$GENERATED_ROOT/architecture/manifest.json" \
  --run-root "$RUN_ROOT" \
  --output "$GENERATED_ROOT/architecture/selection.json" \
  --allow-postprocessing-revision

# 按 selection.json 的 selected_systems 原顺序逐项传入 --system。
$PRODUCTION_PYTHON scripts/generate_lre_ablation_configs.py \
  --stage screening \
  --system audio_only \
  --system query_dependent_p1 \
  --architecture-selection "$GENERATED_ROOT/architecture/selection.json" \
  --output-dir "$GENERATED_ROOT" \
  --strict-run-root "$STRICT_RUN_ROOT"

$PRODUCTION_PYTHON -m avgaussianv2.cli.benchmark_lre_run \
  --manifest "$GENERATED_ROOT/screening/manifest.json" \
  --output-root "$RUN_ROOT_V3" \
  --native-root "$STRICT_RUN_ROOT" \
  --gpus 1,2 \
  --python "$PRODUCTION_PYTHON" \
  --dpam-python /home/lisujing/miniconda3/envs/avcloud/bin/python \
  --trust-upstream-artifacts
```

若双卡未通过预检，改为当前唯一通过连续三次 clean sample 的授权卡；禁止关闭预检。selection、
confirmation、robustness 继续使用 `benchmark_lre_select` 和同一 `RUN_ROOT`，确保稳定
`continuation_id` 原地恢复。

当前 P3 10k/30k 权威 manifest 分别为：

```text
configs/generated/lre_loss_ablation_p3/confirmation_10k/manifest.json
configs/generated/lre_loss_ablation_p3/confirmation_30k/manifest.json
```

30k manifest SHA-256 为
`002a74dedd07d658c63516a2d60b93d5057bd3cbb18faaabe88547a4414b2a2d`。
因原正式 worktree 被另一个开发任务推进，P3 执行迁移到新的 clean detached worktree，
但仍使用相同 `98d158e4` commit、原 manifest 字节、原 continuation identity 和原输出目录。
这种迁移必须使用显式 audited relocation；不能手工改 continuation identity 或覆盖旧目录。
Audio-only 10k/30k 与后续 seed17/73 已生成独立、内容寻址的 manifest，并沿用
P2 Audio-only 的稳定 `continuation_id`。它们在 P3 候选 30k/causal/多 seed 链结束后
由独立 relay 执行，避免覆盖仍被候选 gate 使用的 legacy stage pointer；完成独立
verifier 前不得进入最终公平主榜。

候选 seed manifest 与 causal manifest 将直接发布到工作区持久目录，不再只保存在
`/tmp`。Audio-only relay 完成后，冻结于 `fee48d1` 的报告 worktree 自动生成：

```text
results/lre_loss_ablation_visual_time_v3/final_fair_comparison.json
results/lre_loss_ablation_visual_time_v3/FINAL_FAIR_COMPARISON.zh-CN.md
```

生成器强制要求精确的 3 systems（final candidate、同架构 lambda=0、Audio-only）×
3 seeds × 2 scenes × 30k 矩阵，逐样本配对并输出分层 bootstrap 95% CI；任一 manifest、
repository、config、sample order、metric protocol 或 run completion 不一致都 fail closed。

Source/Mono 参考单独运行，并在正式报告前 verify-only：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" \
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache \
uv run --no-project \
  --python /home/lisujing/miniconda3/envs/avcloud/bin/python \
  --with tomli python -m avgaussianv2.cli.evaluate_audio_references \
  --config configs/benchmark_cam38/scene1_opera.yaml \
  --config configs/benchmark_cam38/Scene7playing.yaml \
  --output-dir results/lre_loss_ablation_visual_time_v3/audio_references

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" \
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache \
uv run --no-project \
  --python /home/lisujing/miniconda3/envs/avcloud/bin/python \
  --with tomli python -m avgaussianv2.cli.evaluate_audio_references \
  --output-dir results/lre_loss_ablation_visual_time_v3/audio_references \
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

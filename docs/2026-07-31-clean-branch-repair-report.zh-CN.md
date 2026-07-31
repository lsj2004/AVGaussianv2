# clean 分支实验公平性修复报告

日期：2026-07-31  
基线：`origin/clean/codebase-structure` (`3a8b81b`)  
修复分支：`fix/clean-experiment-fairness`

## 1. 结论

本次修复覆盖了会使实验无法运行、指标不可横向比较、因果结论证据不足，以及两卡机器利用率不正确的代码路径。修复后的正式评测满足：相同 cam38 样本与顺序、相同裁剪、相同指标实现与方向、单模态只报告其真实输出、条件模型具有同检查点反事实对照，并记录主训练步数与总优化器步数。

架构实现本身是完整的：FiLM+U-Net、plain U-Net、token cross-attention、mask cross-attention 和 query-dependent P1 都有配置、构建、训练、检查点恢复和评测入口。运行仍依赖仓库外的 FTGS++、AudioGS、数据集和原生检查点；这些属于显式上游依赖，不是本仓库缺失的架构代码。

## 2. 已修复问题

### 2.1 运行与调度

- Python 3.10 不再因无条件导入 `tomllib` 而失败；3.10 使用 `tomli` 回退，并在项目依赖中声明条件依赖。
- 准备阶段和 CLI 不再要求恰好三张 GPU；接受任意非空、互异 GPU 列表，默认改为 `0,1`。
- 并行器改为每张物理卡最多一个活跃任务。同一卡的任务排队，空闲后立即补位；失败任务及同批兄弟进程都会被清理。
- 三个 continuation worker 在两卡上按 `GPU0/GPU1/GPU0(排队)` 分配；评测任务也以两卡流水执行，不会在同一卡上盲目叠进程导致 OOM。
- cross-attention 脚本不再固定引用旧的 `runs/cam38_benchmark`，默认指向修复后的 `cam38_benchmark_visual_time_v2`，并允许通过环境变量覆盖根目录。

### 2.2 指标协议

模型、Native AudioGS、Source Binaural、Mono 现在共享以下可直接比较的波形指标：

- `waveform_l1`
- `mono_lsd`、`diff_lsd`
- `lre_error_db`（工程协议，epsilon=1e-8）
- `paper_mag`、`paper_env`、`paper_lre_db`（AudioGS 论文协议，LRE epsilon=1e-5）
- `ild_error_db`
- `ipd_error_rad`
- `paper_dpam`（正式评测通过 `--compute-dpam`；记录实现、源文件和模型权重 SHA-256）

`audio_total/audio_mono/audio_diff` 依赖 AudioGS checkpoint criterion，只对加载该 criterion 的模型有定义，不能伪装成 Source/Mono 的通用指标。报告比较 Source/Mono 时必须使用上述公共波形指标。

修复了两类误导输出：

- `audio_only`、`plain_unet`、`native_audiogs` 只报告音频指标；不再顺带报告未训练/冻结视觉模块的指标。
- `visual_only`、`native_ftgspp` 只报告视觉指标；不再顺带报告未训练/冻结音频模块的指标。

指标扩展注册表现在同时保存方向、模态和协议身份，防止两个同名外部指标实现被错误合并。

### 2.3 公平性与因果证据

- 新增 `joint_conditioned_no_rgbd`：关闭条件输入但复用同一个 FiLM checkpoint。
- 新增 `joint_conditioned_wrong_camera`：用同帧训练相机 RGBD 替换 cam38 RGBD，并复用同一个 FiLM checkpoint。
- 新增 FiLM 因果配对报告，强制三种推理模式使用同一 checkpoint、同一指标协议和同一测试样本。
- plain U-Net 新增独立架构报告，可在共同 step 上与 AudioGS native-residual 后处理器做逐样本配对。
- cross-attention 报告支持 `{5k}`、`{5k,10k}` 或 `{5k,10k,30k}` 的共同 step，因此架构筛选阶段不必为得到报告强制跑满 30k。
- 报告文字由笼统的 `update-matched` 改为 `main-update-matched`。每条评测证据新增 `warmup_updates` 和 `total_optimizer_updates`；条件模型 30k 主训练对应 32k 总优化器步，非条件模型为 30k，不能宣称总计算量完全匹配。
- Native AudioGS 继续明确标记为 `native_reference_non_update_matched`；它与 30k continuation 的比较只能是描述性基线，不能作为训练预算相同的消融。

### 2.4 多 seed 与 LRE 流程

- seed 不再在训练合同、兼容性对象、评测运行时和证据生成中固定为 42；允许任意非负整数。
- 准备阶段从配置的 `benchmark.seed` 构造共享样本序列，并强制其与 `train.seed` 一致。
- continuation 评测证据从 checkpoint fingerprint 读取 seed 与计划步数，不再伪造为 42/30000。
- LRE 生成器新增 `robustness` 阶段；可在 screening winner + zero 上使用 seed 17、73 做最后稳健性复验。

### 2.5 第二轮 P0 修复：合同内筛选、暂停恢复与 Native 复用

- worker、训练合同和资产审计现在接受任意非负 seed，并强制 `train.seed == benchmark.seed`；seed 42 与 73 已通过相同 worker 回归。
- LRE screening 配置不再把正式 `joint_steps/continuation_updates` 篡改为 5k。所有候选保留 30k 最终合同和 `[5k,10k,30k]` 里程碑；screening run manifest 仅声明 `stop_after_step=5000`。
- worker CLI 新增 `--stop-after-step N`。暂停时会原子发布精确的 `main_step_N` checkpoint、I/O sidecar 与 progress journal，但不发布会被误认为正式结果的 `final.pt`；后续用 `--resume` 在同一 30k 合同中继续。
- 恢复时强制 `stop_after_step` 大于已提交进度，防止参数写错后意外跑满 30k。
- Native 合同不再绑定 continuation-only 字段的完整 YAML SHA。新增 Native-affecting 投影：保留场景、路径、模型、`train.crop_seconds` 和 Native 预算，排除 continuation 优化器/损失、seed、步数、warmup 和报告里程碑；因此 seed/LRE continuation 变体可以安全复用同一份经过验证的 AudioGS/FTGS++ Native 合同，架构、裁剪长度或资产变化仍会被拒绝。
- resolved config 即使自身能通过资产审计，也必须验证 origin 合同、原始配置 SHA 和 resolved SHA，修复了绝对路径物化后错误使用 resolved SHA 充当 source SHA 的问题。
- Source Audio 的 Hilbert envelope 在 FFT 前显式物化连续张量，修复 PyTorch/oneMKL 对 `expand` 零步长布局报错而导致 `paper_env` 无法计算的问题。
- 真实暂停恢复 smoke 进一步发现 rolling checkpoint 删除后，sidecar 仍保留已删除文件名。写入顺序现改为先淘汰旧 checkpoint，再原子发布仅包含实际保留文件的 committed inventory；否则严格恢复验证会拒绝有效暂停目录。

精确暂停/恢复单测验证了：3 步暂停后从同一 checkpoint 恢复至 6 步，不重启、不改合同、最终状态与连续执行一致。Native 投影测试验证 seed/LRE 变化可复用合同，而模型配置变化必然改变投影。

### 2.6 LRE 两卡流水线闭环

- 新增统一 LRE runner，严格按 `prepare → train → eval` 依赖顺序执行，每张 GPU 同时最多一个 pipeline；两张卡并行不同候选，单个阶段失败会终止其他活动阶段。
- GPU 预检从“显存大于零”收紧为至少 8 GiB 空闲且利用率不超过 10%，运行结果记录峰值已用显存和最高利用率。
- screening 的 5k 暂停 checkpoint 现在可以通过 progress、resume sidecar、artifact journal 与 SHA-256 独立验证并评测；30k 正式结果仍必须具有完整 5k/10k/30k 里程碑、`final.pt` 和最终资产清单。
- screening 与 confirmation 使用稳定 `continuation_id`、共享配置字节和同一输出目录。每个续训目录新增不可变身份文件，绑定配置 SHA、场景、架构、seed 和 LRE 权重，拒绝无身份或身份不匹配的目录复用。
- 配置生成器新增 `--strict-run-root`，可把配置中的规范 `runs/cam38_strict` 路径显式映射到本机已验证资产根，避免在准备阶段才发现 worktree 路径不存在。
- 新增 screening selector：候选必须在所有 scene/system 单元分别满足 LRE 相对改善、`audio_total` 与 waveform L1 退化门槛，才按配置约定的宏平均 LRE error 排序保留最多两个。winner 文件与 screening manifest SHA-256 绑定。
- prepare CLI 不再向终端展开 4,940 个样本 ID，只输出样本数；完整 ID 序列仍保存在可审计 preparation artifact 中。

## 3. 公平比较边界

### 可以严格横向比较

- 各模型之间：公共音频指标；联合模型之间还可比较公共视觉指标。
- 模型与 Native AudioGS：公共音频指标，但报告必须保留训练预算不匹配标签。
- 模型与 Source Binaural / Mono：公共波形指标和论文指标；不能比较 AudioGS criterion loss。
- FiLM 条件开/关/错误相机：同检查点逐样本配对，可支持“模型是否使用视觉条件”的因果判断。

### 仍不能声称的结论

- cam38 同时承担开发选择与最终汇报时，它是内部 held-out benchmark，不是无偏最终测试集。对外论文结论需要预先冻结方案后增加独立场景/相机测试集。
- RTE/RT60 不能从任意节目音频片段可靠推出。没有干净激励信号、房间脉冲响应或经过验证的预训练估计器时，加入一个自制 RTE 数字反而会降低可信度，因此本次没有伪造该指标。
- P1 每次更新包含额外错误相机前向，虽然主优化器更新数相同，但 FLOPs 不同；报告必须保留 `different_by_design`，不能称为计算量严格匹配。

## 4. 新增报告入口

```bash
python -m avgaussianv2.cli.benchmark_architecture_report \
  --scene-id scene1_opera \
  --audio-only-eval-root <audio-only-root> \
  --architecture-eval-root <plain-unet-root> \
  --protocol-dir <plain-unet-protocol> \
  --output-dir <report-root> \
  --expected-samples 130 --steps 5000

python -m avgaussianv2.cli.benchmark_film_causal_report \
  --scene scene1_opera \
  --evaluations-root <scene-evaluations-root> \
  --output-dir <causal-report-root> \
  --expected-samples 130 --steps 5000 10000 30000
```

正式模型评测必须添加 `--compute-dpam`。cross-attention 运行脚本和主 benchmark orchestration 已自动添加该参数。

## 5. 验证

- Python compileall：通过。
- Bash 语法检查：通过。
- `git diff --check`：通过。
- P0 定向单元/合同测试：88 项通过。
- 全量单元/合同测试：383 项通过（最终复跑 45.19 秒）；包括任意 seed、精确暂停/恢复、Native 投影、30k/5k LRE 合同、两卡 runner、screening selector，以及 Source Audio 论文指标。
- 两份真实 scene1 配置（seed 42、73）均通过资产审计，且其 Native-affecting 投影与现有 AudioGS Native 合同一致。
- 使用本机外部严格资产根生成 16 份 LRE screening 配置，逐项检查 96 个路径，缺失数为 0。

## 6. 真实 GPU smoke

执行环境：NVIDIA RTX 5880 Ada 48GB，FTGS++ Python、PyTorch 2.9.1+cu128、gsplat、tiny-cuda-nn。执行时物理 GPU 1 被其他任务占用 46.1GB 且利用率 100%，因此没有争抢；以下两条 smoke 均在空闲物理 GPU 2 上顺序执行。

| smoke | seed | 30k 合同 | 实际路径 | 结果 |
|---|---:|---|---|---|
| 暂停/恢复 | 42 | 保持 | `0→100→200→300` | 每次精确恢复，最终 `resumed_from_main_step=200`、`redone_main_updates=0` |
| 多 seed / Native 复用 | 73 | 保持 | `0→100` | AudioGS/FTGS++ Native 合同投影匹配并成功训练 |

最终只读核验结果：

- 两个目录均通过 `verify_resume_artifacts`，checkpoint 内容、SHA、progress、sidecar committed inventory 和 fingerprint 一致。
- seed 42 保留 `main_step_000200.pt` 与 `main_step_000300.pt`，精确进度为 300。
- seed 73 保留 `main_step_000000.pt` 与 `main_step_000100.pt`，精确进度为 100。
- 两个 smoke 均为 `selection=paused`，且均不存在 `final.pt`，不会进入正式结果汇总。
- seed 42 三次恢复的 runtime contract SHA-256 始终为 `6812100985def93ef11f9976dc2c95b9045f54793a256eccc02a3b2e0d927c6d`。

未执行真实 30k GPU 训练。smoke 证明真实 CUDA、Native 合同复用、非默认 seed、精确暂停和无重做恢复链路闭环；正式性能结论仍必须按修复后的协议重新运行。

### 6.1 统一 LRE runner 真实端到端 smoke（2026-08-01）

- 配置：`scene1_opera / joint_conditioned / seed=42 / lambda_lre=0.02`。
- 设备：物理 GPU 2；GPU 1 当时占用约 46 GiB，预检规则拒绝抢占。
- 合同：保持正式 2,000 warmup + 30,000 main 合同，在精确 5,000 main milestone 暂停；不存在 `final.pt`。
- 训练：约 22 分 38 秒；joint 阶段人工观测显存至少 2.9 GiB；单个 joint checkpoint 约 613 MiB，暂停目录约 1.8 GiB。
- 评测：130 个 cam38 样本，smoke 按协议跳过 DPAM；成功 generation 的 content SHA-256 为 `d0791e59d77f47ae9321836d9794d49b897f5274550868fc54399e2f39d7e20b`。
- 独立验证：`verify_resume_artifacts`、`verify_evaluation` 和 runner `--resume` verify-only 均通过；恢复没有重新准备、训练或计算指标。
- 边界：没有 `lambda_lre=0` 配对控制，且未计算 DPAM，因此这些数值不得进入性能排名。

真实 smoke 额外发现并修复三个流水线问题：生成器错误地把绝对 upstream root 改成相对路径；runner 未创建首次 evaluation 的父目录，导致 130 个样本计算后发布失败；verify-only 会覆盖首次运行的耗时/显存记录。修复后绝对 upstream 路径保持不变、evaluation root 在启动前创建、每次结果写入不可变 `result_history`。runner CLI 现在强制显式指定包含 FTGS++/`gsplat`/`tinycudann` 的生产 Python。

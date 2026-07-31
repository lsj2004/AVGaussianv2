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

精确暂停/恢复单测验证了：3 步暂停后从同一 checkpoint 恢复至 6 步，不重启、不改合同、最终状态与连续执行一致。Native 投影测试验证 seed/LRE 变化可复用合同，而模型配置变化必然改变投影。

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
- 全量单元/合同测试：375 项通过（37.13 秒）；包括任意 seed、精确暂停/恢复、Native 投影、30k/5k LRE 合同，以及 Source Audio 论文指标。
- 两份真实 scene1 配置（seed 42、73）均通过资产审计，且其 Native-affecting 投影与现有 AudioGS Native 合同一致。

未执行真实 30k GPU 训练。本报告证明代码与实验合同闭环，不把单元测试冒充实验结果；新结果必须按修复后的协议重新运行。两个短 GPU smoke 已完成配置与 Native 合同预检，但当前 Codex GPU 提权审批服务中断，尚未进入 CUDA runtime；因此不能把准备失败误报为 smoke 通过。

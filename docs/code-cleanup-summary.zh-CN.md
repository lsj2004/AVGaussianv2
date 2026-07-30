# 代码清理说明

本文记录 `clean/codebase-structure` 对仓库冗余代码的处理边界，避免后续重新引入
已退役工作流，或因为文件较长而误删严格 benchmark 的安全逻辑。

## 1. 已删除的主要冗余

### 1.1 Scene1 pilot 工作流

已从当前 clean 主线移除：

- `avgaussianv2/cli/pilot.py`
- `avgaussianv2/cli/pilot_worker.py`
- `avgaussianv2/cli/pilot_eval.py`
- pilot 专用的 checkpoint、selection、training、evaluation 和 report 状态机
- `scripts/pilot_scene1_opera.sh`
- pilot 专属测试

删除原因：

1. 它是正式 cam38 benchmark 之前的单场景诊断工作流；
2. 正式 benchmark 已覆盖固定预算训练、resume、评估、报告、证据和多 GPU 编排；
3. 仓库没有提交可作为当前结论的 pilot 正式结果；
4. pilot 生产代码约一万行，且大部分状态机不再被正式 benchmark 调用。

完整历史实现仍保留在：

```text
origin/agent/scene1-pilot
```

原始 spec/plan 也继续保留，但已经标记为 archived。

### 1.2 重复的 sample device wrapper

普通训练、benchmark worker 和 pilot worker 曾各自实现一份
`_DeviceSampleSequence`。现在统一为：

```text
avgaussianv2/data/tensor.py
```

该模块同时拥有唯一的 `move_sample()` 实现。正式 benchmark 不再为了一个 tensor
搬运函数依赖旧 experiment evaluator。

### 1.3 `TrainingBundle` 迁移兼容

删除了以下旧接口：

- `samples=` 参数别名；
- 三参数位置形式推断；
- `.samples` 属性别名；
- 自定义 `init=False` 构造逻辑。

当前只接受明确字段：

```text
model
train_samples
eval_samples
audio_loss_fn
```

### 1.4 AudioGS checkpoint 重复加载

FiLM backend 与 Gaussian-token Cross-Attention backend 曾复制 checkpoint
存在性、payload、model factory、静态 cache 对齐和 strict state restore。

现在统一由：

```text
avgaussianv2/backends/audio_audiogs.py::_load_audiogs_model
```

负责，两个 backend 共享相同加载语义。

### 1.5 Ablation preparation 重复工具

architecture 和 cross-attention preparation 曾复制：

- JSON object 读取；
- crash-safe atomic write；
- clean Git repository identity。

现在统一放在：

```text
avgaussianv2/benchmark/artifacts.py
```

evidence 字段仍保持 `root/commit/clean`，没有改变协议格式。

### 1.6 小型死代码

删除了未引用的 `FiLMConditionedAudioUNet._BLOCK_NAMES`。

## 2. 清理规模

清理前：

```text
avgaussianv2 Python: 29,458 行
tests Python:        17,670 行
```

清理后：

```text
avgaussianv2 Python: 18,389 行
tests Python:         9,739 行
```

生产代码减少 37.6%，测试代码减少 44.9%。主要减量来自退役 pilot，而不是删除
核心模型或 benchmark correctness checks。

## 3. 为什么没有继续删除 benchmark 大文件

以下文件仍然很长：

- `benchmark/orchestration.py`
- `benchmark/training.py`
- `benchmark/assets.py`
- `benchmark/native.py`
- `benchmark/production.py`
- `benchmark/evaluation.py`

它们包含的不是简单模型算法，而是：

- cam38 防泄漏；
- immutable input 和 source hash；
- exact resume；
- checkpoint transaction 和 rolling retention；
- subprocess/GPU failure propagation；
- atomic report publication；
- retained-FD 防目录替换；
- native contract 和 provenance 校验。

这些逻辑有表面相似的 atomic write、fsync 和 path validation，但失败恢复、锁和返回
语义不同。为了减少行数强行合并，会扩大安全关键抽象的耦合面。因此本轮只合并了
语义完全相同的 helper。

## 4. 后续可做的结构拆分

后续若继续优化可读性，优先做“移动而非删除”：

1. 将 `benchmark/orchestration.py` 按 preflight、native jobs、scene jobs、suite
   report 拆模块；
2. 将 `benchmark/training.py` 的 checkpoint store 与 step execution 分开；
3. 将 `benchmark/assets.py` 的 AudioGS、FTGS++ 审计分开；
4. 为 Linux-only retained-FD production runner 建立明确的平台边界；
5. 在 P1/Mask/Plain 合入统一模型 registry 后，再删除各分支兼容 runner。

每次拆分都应保持 schema、hash、resume 和 failure-injection 测试不变，不应把
“文件变短”作为唯一成功标准。

## 5. 验证

本轮清理完成后：

```text
核心/metrics/ablation/CLI 目标测试：112 passed
FiLM 随机敏感测试重复运行：       20/20 passed
全量测试：                         242 passed, 73 failed
```

73 个全量失败均来自当前 macOS 环境不提供 Linux `/proc/self/fd` retained-FD
路径，或本机不存在配置中的 `/mnt/sda` 服务器资产。没有出现已删除 pilot 模块的
残留 import，也没有清理引起的核心功能失败。

此外已通过：

- repository-wide Ruff；
- `git diff --check`；
- Python `compileall`；
- 所有保留 CLI 的 `--help` 导入检查；
- README 中脚本路径存在性检查。

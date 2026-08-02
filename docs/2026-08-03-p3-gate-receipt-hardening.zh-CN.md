# P3 gate 与 relay 成功凭证修复

## 1. 审计发现

30k 训练后的旧临时流水线存在两个 P0 完整性缺口：

1. `/tmp/avgf_p3_30k_gate.py` 只重算 evaluation `manifest.json` 的 SHA-256 并读取
   CSV，没有重新审计绑定 checkpoint、runtime contract 和严格训练证据；
2. Audio-only 与最终报告 relay 只等待上游 PID 消失。若上游失败但正式目录残留旧 gate
   或旧报告，下游可能把“进程结束”误判为“阶段成功”。

最终报告虽然会再次严格验证全部模型结果，但这发生在候选多 seed 与 Audio-only GPU
预算已经消耗之后，不能作为续跑决策的唯一兜底。

## 2. 修复

### 2.1 决策前严格验证

新增 `scripts/run_verified_p3_30k_gate.py`：

- 内容寻址并验证原 gate 实现 SHA-256；
- 用 `verify_evaluation` 替换原 gate 的弱 loader；
- 对 5k、10k、30k 主分支和 30k counterfactual 分支重新审计 evaluation、checkpoint、
  runtime contract 与 training evidence；
- 强制 scene、system、seed42、step、sample order、checkpoint step 和
  `main_update_matched` 一致；
- 强制所有 gate evaluation 使用完全相同的 metric protocol；
- CSV 行均值、不可变 evaluation summary 和 sample inventory 必须一致。

原 gate 仍负责预注册的数值门槛和 Pareto 规则；严格包装器只强化证据，不改变选择标准。

### 2.2 token 绑定的成功凭证

新增两个版本化 relay：

- `scripts/run_p3_post_30k_relay.zsh`；
- `scripts/run_audio_only_after_p3_relay.zsh`。

每次启动生成唯一 run token。P3 后处理只有在 8/8 30k 独立复核、causal 评测、严格 gate
以及必要的候选多 seed 全部成功后，才原子发布 `p3_pipeline_receipt.json`。Audio-only
relay 必须匹配该 token、gate SHA-256 和 finalist，完成对应 30k manifest 的独立复核后
才发布 `audio_only_pipeline_receipt.json`。

adaptive runner 的内容哈希不足以单独代表完整执行闭包，因此 relay 还逐个固定其传递
依赖：relocated Python runner、GPU PID watchdog、manifest shard generator、main verifier
和 causal verifier。watchdog Zsh 包装器继续调用的 descendant-watchdog Python 实现也
单独固定；任一 `/tmp` 传递依赖字节变化都会在领取 GPU 前失败。

最终报告 relay 现在必须匹配 Audio-only receipt 的 token、gate SHA-256、finalist 及
confirmation/robustness manifest SHA-256。上游失败、旧 receipt、旧 gate 或 PID 复用均
不能解锁下游。

候选 gate、causal manifest、seed manifest 和 multiseed report 均先写入本次 token 的
pending 路径，成功后才提升到正式路径，避免失败尝试覆盖权威产物。

## 3. 运行边界

本修复不修改冻结训练提交 `98d158e4`、模型、优化器、manifest、continuation 或当前
30k supervisor。只替换尚在等待的 P3 后处理、Audio-only 和最终报告 relay，因此不会
中断当前 GPU 训练，也不会改变任何已注册实验配置。

最终部署冻结于提交 `276626f50750325005b18f92571854794526103c`。旧的三个等待 relay
已停止，新会话依次为：

- `avgf-p3-after-30k-versioned-v6`；
- `avgf-audio-only-after-p3-versioned-v6`；
- `avgf-final-report-after-baseline-v9`。

训练 supervisor 仍为原 PID `1661222`；两次等待 relay 迁移期间，第二条 30k
continuation 从精确 17.5k 连续推进到 22k，证明训练未被迁移中断。最终 P3 与
Audio-only token 分别为 `3f966671-db47-4c2a-9c4a-ba81b361d9ff` 和
`5711f4a0-1cc7-43c1-9fdd-4059a113e7f3`，后续 receipt 必须逐字匹配。

## 4. 验证

- strict gate 与三段 receipt relay 定向测试：17/17 通过；
- Zsh 语法：三个 relay 全部通过；
- Ruff：通过；
- 完整测试集：446/446 通过。

此外，冻结工具对既有真实
`query_dependent_p1_no_rgbd / scene1_opera / seed42 / step5k` counterfactual smoke
执行 strict verify-only：130/130 样本通过；checkpoint SHA-256 为
`5ba28978b7076c708b20d9827588286c0c72d8597c7da76d4d5869176af3f6eb`，evaluation
content SHA-256 为
`8f4262e1bba61f44cf9658a6df3968f13f33a0abee5d44f7c2d20c8c20a95cff`。这验证了严格
loader 对实际 `no_rgbd` 分支的目录、身份和训练证据映射，而不只是 mock 接口。

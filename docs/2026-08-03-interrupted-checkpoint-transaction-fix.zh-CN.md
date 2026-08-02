# P3 checkpoint 中断事务修复报告

## 1. 结论

P3 首个 30k run 暴露了一个真实的 checkpoint 发布竞态：训练已到精确 30,000
updates，但 watchdog 在 `final.pt` 发布过程中终止 worker。rolling checkpoint 与 30k
milestone 已完整落盘，`checkpoint_io.json` 却仍只提交到 29.5k，旧 verifier 因 inventory
不一致而拒绝 resume。

修复后，流水线能够在 exclusive worker lock 内严格识别并恢复唯一的中断尾事务，且不会
放宽对多余文件、哈希、fingerprint、sidecar counter、inode 或 artifact journal 的校验。

## 2. 根因

旧发布顺序为：

1. progress 把 observed/exact step 写到当前 checkpoint 边界；
2. 写 rolling checkpoint；
3. 写 milestone 和 final；
4. 最后提交包含 rolling checkpoint SHA-256 的 I/O sidecar。

步骤 1 与步骤 4 之间被终止时，progress 会声明一个尚未被 sidecar 提交的 exact
checkpoint；同时原子写留下的临时文件会被严格 output inventory 拒绝。这不是训练数值
错误，而是跨多个文件无法天然原子的事务恢复缺口。

## 3. 修复

提交 `ca21d2ce38e800e8baf5664c105cbe7a857e714f` 包含：

- progress 在 checkpoint 提交前只引用上一个 durable exact step；
- `benchmark_worker --resume` 在构建 runtime 前、exclusive output lock 内执行事务恢复；
- 只允许一个安全的未提交尾 checkpoint；所有已提交 checkpoint 必须重新校验 SHA-256；
- 尾 checkpoint 必须可加载，并匹配 schema、fingerprint、compatibility、stage、step 和
  单调 I/O counters；
- 正常/最终 checkpoint 补齐相同字节的 milestone/final 后提升到 sidecar；
- 5k/10k `stop_after` 边界不直接提升，而是删除未提交尾部并从上一个 committed
  checkpoint 精确重放，保证 paused 结果按原路径发布；
- 多个临时文件、路径穿越、symlink/hardlink、journal 已提交冲突、哈希不一致均 fail
  closed；
- 提供版本化恢复入口 `scripts/repair_interrupted_checkpoint_transaction.py`，记录修复前后
  inventory、哈希、工具 commit 和 verifier 结果。

## 4. 验证

- 最终 checkpoint 发布中断故障注入：通过；恢复后从最终 step 直接完成，不重训；
- pause milestone 发布中断故障注入：通过；回滚后精确重放并得到 paused 结果；
- benchmark training/worker 回归：`63 passed`；
- 全项目：`435 passed`；
- Ruff：通过；
- 冻结训练代码 `98d158e4` 的原 verifier：恢复后独立通过。

## 5. 正式 run 恢复证据

对象：`cross_attention_masks / scene1_opera / seed42 / lambda_lre=0 / 30k`。

恢复前：

- rolling checkpoints：29k、29.5k、30k；
- sidecar committed：29k、29.5k；
- 30k milestone 已存在；
- `final.pt` 不存在；
- 残留 `.final.pt.3ic3ykig.tmp`。

恢复后：

- rolling checkpoints：29.5k、30k；
- sidecar committed：29.5k、30k；
- 原子临时文件：0；
- 30k rolling、milestone、final 的 SHA-256 均为
  `e43cb399075e392ceb644c373da66bcc687545458c0e82259fce80aecc3744af`；
- progress SHA-256 保持
  `ff68be9fdedc06c423ddfaf0489878af635578dc3d2f84d4e189cf48c5620ebb`；
- 恢复后 sidecar SHA-256 为
  `97dacc88e4ee04efda7dc0cfe228c5376c427e1a5d3b1925cdc53700a84186fb`；
- 恢复报告 SHA-256 为
  `00699e282b17852b65120e155c4d8fdff4700efbc2be42fd63c55e259a64fb2b`。

机器可读报告：

```text
results/lre_loss_ablation_visual_time_v3/recovery/cross_attention_masks__scene1_opera__seed42__lre0000__step030000.json
```

恢复后的 worker 已发布 `artifact_hashes.json`，并在 GPU1 完成 30k evaluation：130 个
held-out 样本，content SHA-256 为
`93f3f9e377d45df751ee0cc1709b2683ab688e77c17f3269332cae44be7a733d`。冻结
`98d158e4` verifier 独立通过，run result 状态为 `succeeded`，因此该 run 已正式计入
P3 30k 的 `1/8`。

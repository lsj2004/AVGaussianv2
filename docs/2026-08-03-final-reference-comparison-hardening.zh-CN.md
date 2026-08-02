# 最终基线与参考对比完整性修复

## 1. 问题

P2 已经对 `source_binaural`、`mono` 和 `native_audiogs` 计算了与模型一致的两场景
绝对指标，但阶段性 P2/P3 主表承担的是架构筛选，三项参考被放在独立表中。因此只阅读
候选主表时，会误以为没有进行 Mono 和 Source Audio 对比。

此外，最终报告生成器存在两处证据强度不一致：

1. 模型评估使用 `load_evaluation`，只校验不可变 evaluation generation，没有重新审计
   checkpoint、运行时契约和严格训练证据；
2. Source/Mono 参考值从 P2 汇总报告读取，但没有重新核对参考原始聚合产物及文件级
   verification manifest。

## 2. 修复

- 最终模型矩阵改用 `verify_evaluation`。最终报告生成前，每个 30k 模型结果均重新审计
  evaluation、checkpoint、runtime contract 和 training evidence；
- Source/Mono/native AudioGS 继续以“绝对参考榜”展示，并与最终候选给出逐指标差值；
- Source/Mono 参考输入增加双层校验：
  - P2 报告记录的 `aggregate.json` 与 `verification.json` SHA-256 必须匹配；
  - verification manifest 中列出的全部文件必须逐个重新计算 SHA-256；
  - P2 scene-macro 数值必须与已验证 `aggregate.json` 中的 scene-macro 完全一致；
- 增加 checkpoint/训练证据篡改拒绝测试，以及参考明细文件篡改拒绝测试。

## 3. 对比边界

最终报告分成两个层次：

- **严格公平主榜**：最终候选、同架构 `lambda_lre=0` control、update-matched
  Audio-only；统一为两场景、30k、seed `17/42/73`、相同样本集合，并做 paired
  hierarchical bootstrap；
- **绝对参考榜**：Source Audio、Mono、native AudioGS；沿用同一 held-out 样本协议、
  同一指标实现和 scene-macro 聚合，但不参与“最佳可训练架构”排名。

Source Audio 是输入参考而非 target oracle；Mono 是通道对称的退化诊断参考。Mono 的低
LRE/ILD/IPD 可能由左右声道相同机械地产生，不能单独解释为空间定位更好。因此三项必须
展示，但不能与可训练方法混为同一统计排名。

## 4. 验证

- `tests/test_final_fair_comparison.py`：9 项通过；
- 完整测试集：446 项通过；
- Ruff：通过。

完整测试复跑复现了同族随机初始化测试的不确定性：某些随机 UNet 权重会把测试中的微小
FiLM 扰动压到 `torch.allclose` 容差以内。所有依赖“FiLM 扰动必须可观测”的 GS-only
测试现统一通过共享构造器，在不污染调用者 RNG 的 `torch.random.fork_rng` 作用域内固定
初始化种子；三个相关测试并行重复 5 轮与完整测试均通过。

冻结最终生成器 `9fa1302` 已在正式 P2 目录执行真实 reference preflight：
`reference_aggregate_sha256=d10e68a5f90df29c538e66fdc889b4e3be46b01fd4b530af4effdce7dcf3e5e6`
与
`reference_verification_sha256=38aab786429fe56e4d73f163de9e7a1059f0c4d09fd8acd2385b5aeb54fd9958`
均通过，Source Binaural、Mono、native AudioGS 的 3×7 scene-macro 数值成功从严格路径
读取。最终报告不会在长实验结束后才首次接触这些历史参考产物。

最终 relay 还从冻结工具代码外部固定 P2 报告自身 SHA-256
`920364e134896d2e495ba8985936d5798048bc5b4b2ed763604569db3e722eb0`，并同时固定上述
aggregate/verification 哈希。这样不再依赖“P2 JSON 内部自证”，即使正式产物 worktree
的当前 HEAD 后续变化，也不能用一组彼此一致但非原始的替换文件生成最终报告。
该 relay 冻结于提交 `ab4ab11d5a103ddb34687873a75f6e45a1139987`，部署会话为
`avgf-final-report-after-baseline-v9`；上游 Audio-only receipt token 为
`5711f4a0-1cc7-43c1-9fdd-4059a113e7f3`，替换未
影响候选训练或基线队列。

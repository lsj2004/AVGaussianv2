# 最终架构、参数与资源证据加固

更新日期：2026-08-03

## 问题

此前最终报告生成器已严格输出 30k 公平模型榜和 Source/Mono/native AudioGS 绝对
参照榜，但没有把 P2 的四架构配置、CUDA checkpoint 参数审计和 5k 资源成本纳入同一
机器可读证据链。这样即使指标表正确，最终交付仍不能直接回答“比较了什么架构、实际
优化了多少参数、筛选成本是多少”。

## 修复

`scripts/build_final_fair_comparison.py` 现在要求三个新增输入：

- `architecture_config_report.json`；
- `p2_cuda_parameter_audit.json`；
- `p2_resource_report.json`。

构建器 fail closed 地复核以下合同：

1. 三份产物的 schema、version、repository 必须与 30k gate 完全一致；
2. 架构集合必须恰好为 `audio_only`、`query_dependent_p1`、
   `joint_conditioned`、`cross_attention_masks`；
3. 架构报告到参数审计、资源报告，以及参数审计到资源报告的 path/SHA-256 指针必须
   与实际输入一致；
4. 每个架构签名必须等于 canonical JSON 的 SHA-256；
5. 参数审计必须恰好覆盖四架构乘两个场景，并与架构报告内嵌审计逐字段一致；
6. 资源报告必须恰好覆盖四架构乘四个 lambda 乘两个场景的 32 runs，且峰值、均值、
   GPU-hour 聚合必须能从逐 run 记录重算；
7. finalist 和 no-finalist 两条报告路径都必须保留上述架构证据。

最终 JSON 保存完整 active hyperparameters、架构签名、worker/evaluation identity、两个
场景的参数量范围及逐场景值、P2 5k 资源聚合。Markdown 新增架构表和完整 active
hyperparameters；资源数字明确标注为 P2 5k screening，不能解释成 30k 最终成本。
`plain_unet` 明确标注为 P2 正式 32-run 矩阵前淘汰。

## 正式证据根

| 产物 | SHA-256 |
|---|---|
| architecture config | `dc4bb138457e300098709a7e08757f8c38a0a7261fb5632309faa4a8be6dad6c` |
| CUDA parameter audit | `85299ea4f04435f05125ed22a10134152fd854651b7bf31e4d962a5819848356` |
| P2 resource report | `4db3fbc9ff8dacc848e567fcea5c9462b52bc0c60c681c2fa1fc7f4a67c24cd9` |

正式产物预检已通过，得到以下 `(total params min, peak MiB)`：

| 架构 | total params min | P2 5k peak MiB |
|---|---:|---:|
| audio_only | 42,653,534 | 1,453 |
| query_dependent_p1 | 42,912,908 | 2,359 |
| joint_conditioned | 42,653,534 | 2,997 |
| cross_attention_masks | 35,559,708 | 2,261 |

其中 query P1 的 checkpoint 审计显示 7,865,794 个 orphan-trainable 参数；最终报告会
原样披露，不能把“requires_grad=True”误写为“optimizer 实际更新”。

## 流水线接入

最终报告 relay 同时固定上述三份产物的 SHA-256，并向构建器传入三个必需参数。等待
relay 只有在 Audio-only receipt、P3 gate、manifest、P2 reference 根和架构/资源根全部
匹配时才会发布最终 JSON/Markdown。

定向测试覆盖正常验证、架构篡改拒绝、Markdown 架构段、no-finalist 路径和 relay 静态
合同。全量回归为 `448 passed`，Ruff、格式、zsh 语法和正式产物预检均通过。报告构建器
冻结于 commit `912b08b12dfeaeda2fa9095c7e7ea0ab0e4cef36`，文件 SHA-256 为
`bdf2ed0e2838b9866b42194592828d3b7515da8d4919256bce507a031e65c31a`。部署只替换等待
会话，不影响正在运行的 P3 训练和既有上游 receipt token。

最终 relay 代码冻结于 `e6dde5659ef2de875f821637e0da5aa4a7458332`，当前等待会话为
`avgf-final-report-after-baseline-v10`。启动日志已复核：上游 Audio-only relay PID
`2090950`、token `5711f4a0-1cc7-43c1-9fdd-4059a113e7f3`、builder commit/SHA 和 clean
worktree 全部匹配。旧 v9 只包含等待逻辑，已被替换；P3 supervisor 与上游 relay 未重启。

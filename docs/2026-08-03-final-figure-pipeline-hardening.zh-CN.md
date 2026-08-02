# 最终公平对比图表流水线

更新日期：2026-08-03

## 缺口

最终报告生成器已经能输出严格公平模型榜、Source/Mono/native 绝对参照榜、架构配置、
参数审计和 P2 资源表，但实验计划还要求横向图和 Pareto 图。此前没有自动图表产物，
不能把手工截图或人工复制数字作为最终证据。

## 实现

新增纯标准库、确定性的 `scripts/render_final_fair_comparison_svg.py`。它只读取已经通过
严格构建器生成的 `final_fair_comparison.json`，验证 schema/status、四架构集合、参数与
资源字段、所有绘图数值的有限性，然后原子发布三张自包含 SVG：

1. `strict-model-metric-ratios.svg`：candidate、同架构 control、Audio-only 在 8 个主指标
   上相对 Audio-only 的比值；小于 1 表示误差更低；
2. `absolute-reference-ratios.svg`：最终候选、Audio-only、Source Binaural、Mono、native
   AudioGS 的 7 个公共指标比值；图内明确标注只是 metric-matched 描述性参照；
3. `p2-architecture-resource-pareto.svg`：四架构的 optimizer-active 参数、P2 5k 平均
   pipeline 秒数与峰值显存；左下更优，粗边框表示参数量/耗时二维非支配点。

图表不会构造任意加权总分。资源图严格标注为 P2 5k screening，不能解释为 30k 成本；
Source/Mono 图不把绝对参照伪装成 update-/seed-matched 排名。候选误差允许为 0，但作为
归一化分母的 Audio-only 指标必须严格大于 0，避免无定义比值。

finalist 和 no-finalist 两条路径都能生成图表；无 finalist 时，严格图显示 Audio-only 与
被 30k gate 淘汰的 seed42 候选，不伪造候选多 seed。

## 仍待完成

该流水线补齐最终横向与资源 Pareto 图的自动生成能力。5k→10k→30k 纵向曲线仍必须等
P3 与 Audio-only 的全部适用节点完成后，从逐 continuation 的已验证 evaluation 生成；
不能用当前部分结果提前封图。

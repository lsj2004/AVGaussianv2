# Query-dependent P1：cam38 统一评测协议

## 目标

`query_dependent_p1` 已作为 AVGaussianFusionv2 的一个音频后端接入原有
cam38 strict benchmark runner。它不使用单独的数据划分、训练预算或指标：

- cam00–cam37 训练，cam38 测试；
- seed 42、batch size 1；
- 2,000 step warmup、30,000 step main；
- 5k、10k、30k 三个统一评估点；
- 复用基准 A 的有序训练样本序列、FTGS++ checkpoint、AudioGS checkpoint
  和原生损失；
- Scene1 评估 130 个窗口，Scene7 评估 293 个窗口。

因此，P1 正常分支可与同项目已有的 native AudioGS、audio-only、FiLM、
direct conditioned U-Net、gated residual、aligned-mask cross-attention 和
Gaussian-token cross-attention 做同协议横向比较。

## P1 数据流

```text
source audio + listener pose
    -> AudioGS native Gaussian render
    -> complex STFT audio queries

FTGS++ RGB/depth/alpha + camera rays
    -> metric visual memory

audio query geometry x visual metric geometry
    -> listener/head-relative attention bias
    -> bounded complex residual
    -> native AudioGS render + residual
```

相机位姿不是作为一个全局条件向量简单拼接，而是参与每个音频 query 与每个
视觉 token 的对应关系计算。

## 训练约束

每个正样本配一个“同一 frame、不同训练 camera”的负条件。训练目标包含：

```text
max(0, margin + loss(correct_camera) - loss(wrong_camera))
```

该约束只改变条件，不改变 source/target audio，也不会使用 cam38 的目标音频。
warmup 与 main 的负样本序列分别由固定 seed offset `10000` 和 `20000`
确定，并写入可复核证据。

## 同 checkpoint 因果评估

- `query_dependent_p1`：正确 RGBD 与相机；
- `query_dependent_p1_no_rgbd`：移除 RGBD 条件；
- `query_dependent_p1_wrong_camera`：保持目标音频和目标视觉指标不变，改用同一
  frame 的训练相机条件。

后两者复用正常 P1 的同一个 checkpoint，不重新训练。报告会输出
correct-vs-noRGBD 和 correct-vs-wrong-camera 的逐样本 paired delta 与 win rate。

## 统一入口

以下命令中的第六个参数选择后端：

```bash
scripts/run_cross_attention_cam38.sh scene1_opera prepare 0 query_dependent_p1 30000 query_dependent_p1
scripts/run_cross_attention_cam38.sh scene1_opera diagnose 0 query_dependent_p1 30000 query_dependent_p1
scripts/run_cross_attention_cam38.sh scene1_opera train 0 query_dependent_p1 30000 query_dependent_p1

scripts/run_cross_attention_cam38.sh scene1_opera eval 0 query_dependent_p1 5000 query_dependent_p1
scripts/run_cross_attention_cam38.sh scene1_opera eval 0 query_dependent_p1_no_rgbd 5000 query_dependent_p1
scripts/run_cross_attention_cam38.sh scene1_opera eval 0 query_dependent_p1_wrong_camera 5000 query_dependent_p1
scripts/run_cross_attention_cam38.sh scene1_opera report 0 query_dependent_p1 30000 query_dependent_p1
```

将 `scene1_opera` 替换为 `Scene7playing` 即运行第二个场景。正式比较应完成两个
场景的 30k 训练和三个评估点；短程 diagnostic 只验证链路与梯度，不能作为性能结论。

输出统一位于：

```text
runs/query_dependent_p1_cam38/<scene>/
```

准备阶段会拒绝脏工作树、非基准 checkpoint、不同数据顺序及
`model.audio_backend` 之外的原始配置差异。

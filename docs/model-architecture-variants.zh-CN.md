# AVGaussianV2 各版本架构图

本文用统一图例对比 AVGaussianV2 已实现和正式评估过的模型版本。图中的模块和
连接尽量对应实际代码，而不是只表达概念。

需要先区分：

- **分支不等于模型版本**：`cross-attention-run` 与
  `cross-attention-benchmark` 的代码树相同。
- **同架构也可以是不同实验系统**：Native AudioGS 与 GS-only 30k 的推理
  架构相同，只是 checkpoint 和 continuation 状态不同。
- **训练目标可以变化而推理架构不变**：P1 与 Spatial P1 的推理数据流完全
  相同，只在训练 loss 上不同。

## 1. 统一图例

所有图使用相同语义：

```mermaid
flowchart LR
    V["视觉模块<br/>FreeTimeGS++ / RGBD encoder"]:::visual
    A["声学模块<br/>AudioGS / U-Net"]:::audio
    F["融合模块<br/>FiLM / Cross-Attention"]:::fusion
    T["仅训练时使用<br/>Loss / negative condition"]:::training
    O["双耳音频输出"]:::output

    V --> F --> A --> O
    T -.-> F

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef training fill:#f5def5,stroke:#8a3d8a,color:#301530
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

- 绿色：视觉场、RGBD 或几何视觉表示。
- 蓝色：AudioGS、声学 Gaussian、U-Net 和音频合成。
- 黄色：视觉和音频发生交互的位置。
- 紫色虚线：只在训练时使用，不改变推理架构。
- 灰色：最终输出。

## 2. 一张图看完所有路线

```mermaid
flowchart TB
    INPUT["Source binaural audio<br/>listener pose / target camera"]

    INPUT --> GS["AudioGS acoustic Gaussians"]
    GS --> NATIVE["Native / GS-only<br/>直接声学 Gaussian 渲染"]

    GS --> FEATURES["AudioGS mono/diff feature maps"]
    FEATURES --> PLAIN["Plain U-Net<br/>无视觉"]
    FEATURES --> FILM["FiLM U-Net<br/>全局 RGBD embedding"]
    FEATURES --> MASK["Mask Cross-Attention<br/>RGBD tokens"]

    INPUT --> STFT["STFT query family"]
    STFT --> PROTO["Early token prototype<br/>source STFT + RGBD + pose"]
    STFT --> NRGBD["AudioGS-native RGBD Cross-Attn<br/>native STFT + RGBD"]
    STFT --> GT["Gaussian-token Cross-Attention<br/>source STFT + RGBD + pose + acoustic tokens"]
    STFT --> P1["Query-dependent P1<br/>native STFT + 世界几何 bias"]

    P1 -. "只改训练 loss" .-> SP["Spatial P1"]

    classDef base fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef conditioned fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef training fill:#f5def5,stroke:#8a3d8a,color:#301530
    class GS,NATIVE,FEATURES,PLAIN,STFT base
    class FILM,MASK,PROTO,NRGBD,GT,P1 conditioned
    class SP training
```

从左到右，核心变化是：

1. 是否使用 AudioGS U-Net；
2. 视觉条件是一个全局向量、普通 token，还是带三维几何的 memory；
3. 融合发生在 U-Net feature、AudioGS mask，还是 complex STFT residual；
4. 输出是否保留 native AudioGS 作为残差锚点。

## 3. 共同的视觉输入

除 Native、GS-only 和 Plain U-Net 外，其他版本都从同一目标相机和时间渲染
视觉条件：

```mermaid
flowchart LR
    TIME["visual_time"] --> FTGS["FreeTimeGS++<br/>dynamic visual Gaussians"]:::visual
    CAMERA["target w2c + intrinsic"] --> RENDER["Differentiable rasterization"]:::visual
    FTGS --> RENDER
    RENDER --> RGB["RGB"]:::visual
    RENDER --> DEPTH["Expected depth"]:::visual
    RENDER --> ALPHA["Alpha / validity"]:::visual

    RGB --> CONDITION["Condition representation"]:::fusion
    DEPTH --> CONDITION
    ALPHA --> CONDITION

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
```

不同版本从 `Condition representation` 开始分叉：

| 版本 | 视觉表示 |
|---|---|
| FiLM / Direct / Gated | 一个全局 128 维 RGBD embedding |
| Early codex token | RGBD tokens，无显式二维位置编码 |
| AudioGS-native / Gaussian-token / Mask Cross-Attention | 带二维位置编码的 RGBD tokens |
| P1 / Spatial P1 | RGBD + camera/world XYZ + normal + alpha 的几何 memory |

## 4. Native AudioGS / GS-only 30k

Native AudioGS 与 GS-only 30k 使用相同推理结构。区别只是 Native 是原生
checkpoint，GS-only 是在统一协议下继续训练 30k updates。

```mermaid
flowchart LR
    SRC["Source binaural audio"]:::audio --> STFT["Source STFT / magnitude"]:::audio
    POSE["Listener pose"]:::audio --> FIELD["AudioGS acoustic Gaussians<br/>mono/diff SH field + distance"]:::audio
    STFT --> SYNTH["Native AudioGS synthesis"]:::audio
    FIELD --> SYNTH
    SYNTH --> OUT["Predicted binaural waveform"]:::output

    RGBD["Visual RGBD"]:::visual
    RGBD -. "未使用" .-> SYNTH

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

关键点：

- 没有视觉输入。
- GS-only checkpoint 的实际 forward 绕过继承的 U-Net renderer。
- 它是所有条件模型的 native 声学锚点和空间指标基线。

## 5. Plain U-Net

Plain U-Net 使用 AudioGS 原有 mono/diff U-Net，但不输入视觉条件。

```mermaid
flowchart LR
    SRC["Source binaural audio"]:::audio --> MAG["Source magnitude / phase"]:::audio
    POSE["Listener pose"]:::audio --> FIELD["AudioGS acoustic fields<br/>mono SH / diff SH / distance"]:::audio

    MAG --> MF["Mono features<br/>source mag + mono field + distance"]:::audio
    FIELD --> MF
    MAG --> DF["Diff features<br/>diff field + distance"]:::audio
    FIELD --> DF

    MF --> UNET["Dual-branch AudioGS U-Net<br/>encoder / decoder / skip connections"]:::audio
    DF --> UNET
    UNET --> MM["Mono mask<br/>softplus + 0.1"]:::audio
    UNET --> DM["Diff mask<br/>tanh"]:::audio
    MM --> SYNTH["Original AudioGS binaural synthesis<br/>reuse source phase"]:::audio
    DM --> SYNTH
    SYNTH --> OUT["Predicted binaural waveform"]:::output

    RGBD["Visual RGBD"]:::visual
    RGBD -. "未使用" .-> UNET

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

与 GS-only 的区别：它真正调用 U-Net 生成 mono/diff masks。与 FiLM 的区别：
U-Net 内没有视觉调制。

## 6. FiLM native residual

FiLM 是基础 AVGaussianV2 的主要结构。视觉 RGBD 被压缩为一个全局 embedding，
在 U-Net 的 8 个 encoder/decoder block 后执行 channel-wise scale/shift。

```mermaid
flowchart LR
    subgraph VIS["Visual condition"]
        RGBD["RGB + normalized depth + valid mask"]:::visual
        RGBD --> CNN["5-channel CNN + global pooling"]:::visual
        CNN --> EMB["Global view embedding<br/>B x 128"]:::fusion
    end

    subgraph AUDIO["AudioGS paths"]
        INPUT["Source audio + listener pose"]:::audio
        INPUT --> NATIVE["Native GS-only render"]:::audio
        INPUT --> PLAIN["Plain AudioGS U-Net"]:::audio
        INPUT --> COND["Same AudioGS U-Net<br/>8 zero-init FiLM adapters"]:::fusion
        EMB --> COND
    end

    COND --> DELTA["Condition delta<br/>conditioned U-Net - plain U-Net"]:::fusion
    PLAIN --> DELTA
    NATIVE --> SUM["native + condition delta"]:::fusion
    DELTA --> SUM
    SUM --> OUT["Predicted binaural waveform"]:::output

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

公式：

```text
output = native_AudioGS
       + conditioned_U-Net(RGBD)
       - plain_U-Net
```

这样 FiLM 初始化为零时，输出严格退化为 native AudioGS，不会在训练开始时让
随机条件分支接管完整音频。

## 7. Direct conditioned U-Net

Direct 版本保留相同 RGBD encoder 和 FiLM U-Net，但移除 native residual
组合，直接让 conditioned U-Net 负责完整输出。

```mermaid
flowchart LR
    RGBD["RGB / depth / alpha"]:::visual --> ENC["Global RGBD encoder"]:::visual
    ENC --> EMB["View embedding"]:::fusion

    INPUT["Source audio + listener pose"]:::audio --> FEATURES["AudioGS mono/diff features"]:::audio
    FEATURES --> UNET["FiLM-conditioned AudioGS U-Net"]:::fusion
    EMB --> UNET
    UNET --> SYNTH["Mono/diff masks + AudioGS synthesis"]:::audio
    SYNTH --> OUT["Predicted binaural waveform"]:::output

    NATIVE["Native GS-only render"]:::audio
    NATIVE -. "不参与输出" .-> OUT

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

公式：

```text
output = conditioned_U-Net(RGBD)
```

它同时改变了视觉条件和输出锚定方式，所以不能单独用它估计 RGBD 的因果贡献。

## 8. Gated native residual

Gated 版本与 FiLM 完全相同，只在 condition delta 外增加一个可学习的全局标量
gate，初始化为 1，范围限制在 `(0, 2)`。

```mermaid
flowchart LR
    RGBD["RGBD"]:::visual --> EMB["Global embedding"]:::fusion
    INPUT["Source audio + pose"]:::audio --> NATIVE["Native GS-only render"]:::audio
    INPUT --> PLAIN["Plain U-Net"]:::audio
    INPUT --> COND["FiLM U-Net"]:::fusion
    EMB --> COND

    COND --> SUB["conditioned - plain"]:::fusion
    PLAIN --> SUB
    SUB --> GATE["Learned scalar gate g<br/>0 < g < 2"]:::fusion
    NATIVE --> SUM["native + g x delta"]:::fusion
    GATE --> SUM
    SUM --> OUT["Predicted binaural waveform"]:::output

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

公式：

```text
output = native_AudioGS + g * (conditioned_U-Net - plain_U-Net)
```

一个全局 gate 只能整体控制视觉修正强度，不能区分不同频率、时间和
mono/diff 区域。

## 9. Early codex audio-token prototype

这是 `codex/cross-attention-audio-tokens` 最早实现的独立原型。它不加载
AudioGS checkpoint，也没有 native AudioGS anchor；AudioGS 在这里仅体现在项目
背景和 pose 定义中。

```mermaid
flowchart LR
    SRC["Source binaural audio"]:::audio --> STFT["Source complex STFT<br/>mid / side / ILD / IPD"]:::audio
    STFT --> AT["16 x 4 audio patch tokens"]:::audio

    RGBD["RGB / depth / alpha"]:::visual --> VT["RGBD tokens<br/>无显式 2D position"]:::visual
    POSE["Listener pose"]:::audio --> PT["2 pose tokens"]:::audio
    VT --> MEM["Visual + pose memory"]:::fusion
    PT --> MEM

    AT --> XATTN["Gated self + cross-attention"]:::fusion
    MEM --> XATTN
    XATTN --> HEAD["Complex STFT prediction head"]:::fusion
    STFT --> HEAD
    HEAD --> ISTFT["iSTFT"]:::audio
    ISTFT --> OUT["Predicted binaural waveform"]:::output

    A3DGS["AudioGS checkpoint / acoustic Gaussians"]:::audio
    A3DGS -. "未加载、未使用" .-> XATTN

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

它使用自定义 waveform + ILD/IPD/LRE loss，没有进入后续严格 AudioGS criterion
和 cam38 正式比较，因此应视为结构验证原型。

## 10. AudioGS-native RGBD Cross-Attention

正式 `cross-attention-benchmark` 的初版先解决“必须从真实 AudioGS Gaussian
预测出发”的问题。它移除 AudioGS U-Net，以 native AudioGS waveform 的 STFT
作为 query 和 residual anchor，只查询 RGBD tokens。

```mermaid
flowchart LR
    INPUT["Source audio + listener pose"]:::audio --> NATIVE["Native AudioGS Gaussian render"]:::audio
    NATIVE --> NSTFT["Native complex STFT"]:::audio
    NSTFT --> AT["16 x 4 native-audio tokens"]:::audio

    RGBD["RGB / depth / alpha"]:::visual --> VT["2D-positioned RGBD tokens"]:::visual
    AT --> XATTN["Gated self + RGBD cross-attention"]:::fusion
    VT --> XATTN
    XATTN --> DELTA["conditioned tokens - native tokens"]:::fusion
    DELTA --> HEAD["Bounded complex STFT residual"]:::fusion
    NSTFT --> SUM["native STFT + residual"]:::fusion
    HEAD --> SUM
    SUM --> ISTFT["iSTFT"]:::audio
    ISTFT --> OUT["Predicted binaural waveform"]:::output

    UNET["AudioGS U-Net"]:::audio
    UNET -. "移除" .-> XATTN

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

这一步已经使用真实 AudioGS 输出，但 cross-attention memory 仍只有 RGBD，没有
显式 pose token 或 acoustic-Gaussian attribute token。

## 11. Gaussian-token complex Cross-Attention

这个版本完全移除上游 AudioGS U-Net。它让 source-audio STFT patch token
查询三类 memory：RGBD tokens、listener pose tokens 和显式 AudioGS Gaussian
attribute tokens。

```mermaid
flowchart LR
    subgraph QUERY["Audio query"]
        SRC["Source binaural audio"]:::audio --> STFT["Complex STFT<br/>mid / side / ILD / IPD"]:::audio
        STFT --> AT["16 x 4 patch audio tokens"]:::audio
    end

    subgraph MEMORY["Cross-attention memory"]
        RGBD["RGB / depth / alpha"]:::visual --> VT["2D RGBD tokens"]:::visual
        POSE["Listener pose"]:::audio --> PT["Pose tokens"]:::audio
        AG["AudioGS attributes<br/>xyz / quaternion / mono-diff SH<br/>TF coordinate / pose response"]:::audio
        AG --> GT["16 x 16 Gaussian tokens"]:::audio
        VT --> MEM["Concatenated memory"]:::fusion
        PT --> MEM
        GT --> MEM
    end

    AT --> XATTN["Gated self + cross-attention blocks"]:::fusion
    MEM --> XATTN
    XATTN --> DELTA["Bounded complex STFT residual"]:::fusion

    SRC --> NATIVE["Native AudioGS Gaussian render"]:::audio
    NATIVE --> NSTFT["Native output STFT"]:::audio
    NSTFT --> SUM["native STFT + complex residual"]:::fusion
    DELTA --> SUM
    SUM --> ISTFT["iSTFT"]:::audio
    ISTFT --> OUT["Predicted binaural waveform"]:::output

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

关键区别：

- AudioGS U-Net 完全不参与。
- 显式看到声学 Gaussian 属性，但视觉只是一组普通二维 tokens。
- 输出是 complex spectrogram residual，不是 AudioGS mono/diff masks。

## 12. AudioGS mask-protocol Cross-Attention

Mask 版本把 cross-attention 放回 AudioGS renderer 的标准位置：输入是原始
mono/diff feature maps，输出仍是 mono/diff masks。

```mermaid
flowchart LR
    subgraph AUDIO["AudioGS feature protocol"]
        INPUT["Source audio + listener pose"]:::audio
        INPUT --> MF["Mono features<br/>3 x F x T"]:::audio
        INPUT --> DF["Diff features<br/>2 x F x T"]:::audio
        MF --> MP["Mono 16 x 4 patch tokens"]:::audio
        DF --> DP["Diff 16 x 4 patch tokens"]:::audio
        MP --> FUSE["Concatenate + linear fusion"]:::fusion
        DP --> FUSE
    end

    RGBD["RGB / depth / alpha"]:::visual --> VT["2D RGBD tokens"]:::visual
    FUSE --> XATTN["Self-attention + RGBD cross-attention"]:::fusion
    VT --> XATTN
    XATTN --> HEADS["Linear unpatchify heads"]:::fusion
    HEADS --> MM["Mono mask"]:::audio
    HEADS --> DM["Diff mask"]:::audio
    MM --> RENDER["Conditioned AudioGS synthesis"]:::audio
    DM --> RENDER

    INPUT --> NATIVE["Native GS-only render"]:::audio
    INPUT --> PLAIN["Same mask renderer without RGBD"]:::audio
    RENDER --> DELTA["conditioned - plain"]:::fusion
    PLAIN --> DELTA
    NATIVE --> SUM["native + mask-renderer delta"]:::fusion
    DELTA --> SUM
    SUM --> OUT["Predicted binaural waveform"]:::output

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

相对 Gaussian-token 版本，它恢复了 AudioGS 的输入输出协议和双耳合成算法；
相对 FiLM，它用从头训练的 patch transformer 替代带局部卷积和 skip
connection 的 U-Net。

## 13. Query-dependent P1

P1 的重点不是增加更多 token，而是显式建立“某个音频时频 query 应该关注哪个
三维视觉位置”的 listener-relative 几何关系。

```mermaid
flowchart LR
    subgraph VIS["Geometric visual memory"]
        RGBD["RGB / metric depth / alpha"]:::visual
        CAM["w2c + intrinsic"]:::visual
        RGBD --> GEO["Camera XYZ / world XYZ<br/>camera normal / world normal"]:::visual
        CAM --> GEO
        GEO --> VM["VisualMemory<br/>tokens + 3D positions + normals<br/>confidence + camera pose"]:::visual
    end

    subgraph AQ["Target-view audio queries"]
        INPUT["Source audio + listener pose"]:::audio
        INPUT --> NATIVE["Native AudioGS render"]:::audio
        INPUT --> FIELDS["Mono/diff fields<br/>source magnitude / distance"]:::audio
        NATIVE --> NSTFT["Native complex STFT"]:::audio
        NSTFT --> QF["Magnitude / phase features"]:::audio
        FIELDS --> QF
        QF --> QT["8 x 2 audio query tokens<br/>+ pose embedding"]:::audio
    end

    QT --> BIAS["Per-head geometry bias"]:::fusion
    VM --> BIAS
    BIAS --> XATTN["Geometry-biased cross-attention"]:::fusion
    QT --> XATTN
    VM --> XATTN
    XATTN --> DEC["ConvTranspose decoder"]:::fusion
    DEC --> CORR["Bounded log-magnitude<br/>phase + additive correction"]:::fusion
    NSTFT --> APPLY["Correct native STFT"]:::fusion
    CORR --> APPLY
    APPLY --> ISTFT["iSTFT"]:::audio
    ISTFT --> OUT["Predicted binaural waveform"]:::output

    WRONG["Same-frame wrong camera RGBD"]:::training
    WRONG -. "训练时再次前向" .-> VM
    LOSS["margin + audio_total(correct)<br/>vs audio_total(wrong)"]:::training
    OUT -.-> LOSS

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef training fill:#f5def5,stroke:#8a3d8a,color:#301530
    classDef output fill:#e8e8e8,stroke:#555,color:#111
```

几何 bias 同时使用：

- listener 到 visual point 的距离和方向；
- listener/head 坐标中的方向和法向；
- surface facing；
- condition camera 与 listener 的相对位置和朝向；
- 视觉相机 view direction 与 listener direction 的 parallax；
- 音频 query 的频率和时间坐标。

与前两种 cross-attention 最大的区别是：P1 的 attention score 显式依赖目标
listener、condition camera 和每个三维 visual point 的相对几何。

## 14. Spatial P1

Spatial P1 与 P1 的**推理图完全相同**。它只替换训练时的 camera-contrast
目标。

```mermaid
flowchart LR
    ARCH["与 Query-dependent P1<br/>完全相同的推理网络"]:::fusion
    POS["Correct-camera prediction"]:::audio
    NEG["Same-frame wrong-camera prediction"]:::audio
    TARGET["Target binaural audio"]:::audio

    ARCH --> POS
    ARCH --> NEG

    POS --> SP["Spatial error S(correct)"]:::training
    NEG --> SN["Spatial error S(wrong)"]:::training
    TARGET --> SP
    TARGET --> SN

    SP --> LOSS["0.10 x S(correct)<br/>+ 0.50 x max(0, 0.05 - [S(wrong)-S(correct)])"]:::training
    SN --> LOSS

    LRE["30% LRE"]:::training --> SP
    ILD["25% ILD"]:::training --> SP
    IPD["25% IPD"]:::training --> SP
    DIFF["20% binaural difference"]:::training --> SP

    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef fusion fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef training fill:#f5def5,stroke:#8a3d8a,color:#301530
```

原 P1 使用：

```text
max(0, margin + audio_total(correct) - audio_total(wrong))
```

Spatial P1 使用：

```text
S = 0.30 LRE + 0.25 ILD + 0.25 IPD + 0.20 binaural_difference

loss = 0.10 S(correct)
     + 0.50 max(0, 0.05 - [S(wrong) - S(correct)])
```

所以 Spatial P1 不是“新模型”，而是同一个 P1 模型在不同监督信号下得到的
checkpoint。

## 15. Visual-only

Visual-only 是 benchmark 对照，不产生音频。它只更新 FreeTimeGS++ 视觉场，
用于判断联合音频训练是否损害视觉质量。

```mermaid
flowchart LR
    TIME["visual_time"] --> FTGS["FreeTimeGS++ visual Gaussians"]:::visual
    CAMERA["w2c + intrinsic"] --> RENDER["Differentiable RGBD render"]:::visual
    FTGS --> RENDER
    RENDER --> RGB["Rendered RGB"]:::visual
    TARGET["Target RGB"]:::visual --> LOSS["RGB L1 + DSSIM<br/>+ visual anchor"]:::training
    RGB --> LOSS
    LOSS -. "update visual Gaussians only" .-> FTGS

    AUDIO["AudioGS"]:::audio
    AUDIO -. "不参与" .-> LOSS

    classDef visual fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef audio fill:#dcecff,stroke:#2563a6,color:#10243a
    classDef training fill:#f5def5,stroke:#8a3d8a,color:#301530
```

它与 Native FreeTimeGS++ 也是同架构、不同训练状态的关系。

## 16. 差异矩阵

| 版本 | 视觉条件 | AudioGS U-Net | 融合位置 | 输出形式 | Native anchor |
|---|---|---|---|---|---|
| Native / GS-only | 无 | 不使用 | 无 | Native waveform | 自身 |
| Plain U-Net | 无 | 使用 | 无 | U-Net masks -> waveform | 无 |
| FiLM | 全局 embedding | 使用 | 8 个 U-Net block | U-Net condition delta | 有 |
| Direct | 全局 embedding | 使用 | 8 个 U-Net block | Conditioned U-Net waveform | 无 |
| Gated | 全局 embedding | 使用 | 8 个 U-Net block | Scalar-gated condition delta | 有 |
| Early codex token | 2D RGBD tokens + pose | 不加载 | Source-STFT token transformer | Complex STFT prediction | 无 |
| AudioGS-native RGBD | 2D RGBD tokens | 移除 | Native-STFT token transformer | Complex STFT residual | 有 |
| Gaussian-token | 2D tokens + pose + acoustic tokens | 移除 | STFT token transformer | Complex STFT residual | 有 |
| Mask Cross-Attn | 2D RGBD tokens | 替换为 transformer | AudioGS mask renderer | Mono/diff mask delta | 有 |
| P1 | 3D geometric VisualMemory | 不使用 | Geometry-biased STFT attention | Magnitude/phase/additive residual | 有 |
| Spatial P1 | 与 P1 相同 | 与 P1 相同 | 与 P1 相同 | 与 P1 相同 | 有 |
| Visual-only | 不适用 | 不参与 | 无 | RGBD only | 不适用 |

## 17. 如何直观看这些差异

最关键的三条观察：

1. **FiLM、Direct、Gated 是同一个 U-Net family。**
   它们主要区别在输出怎么与 native AudioGS 合成，不是三个完全不同的 backbone。
2. **Cross-Attention 经历了三次关键修正。**
   早期原型没有 AudioGS；native RGBD 版加入 AudioGS anchor；Gaussian-token
   版再加入显式 pose 和 acoustic Gaussian memory。
3. **Gaussian-token 与 Mask Cross-Attention 不是同一种输出协议。**
   前者直接预测 complex STFT residual；后者预测 AudioGS mono/diff masks。
4. **P1 真正新增的是 query-dependent 三维对应。**
   Spatial P1 没有继续增加结构，只修改了 correct/wrong-camera 的训练目标。

结合实验结果，当前结构演进可以概括为：

```mermaid
flowchart LR
    GS["GS-only<br/>空间指标稳"] --> FILM["FiLM<br/>主损失显著改善<br/>但 LRE 退化"]
    FILM --> MASK["Mask Cross-Attn<br/>视觉有效<br/>空间排列不稳"]
    MASK --> P1["P1<br/>显式三维几何对应<br/>主损失与 LRE 较均衡"]
    P1 --> SP["Spatial P1<br/>只改 loss<br/>5k 好，30k 反转"]

    GT["Gaussian-token<br/>旁路视觉的负结果"] -. "说明普通 token 不够" .-> P1
    PLAIN["Plain U-Net<br/>说明 U-Net 本身有贡献"] -. "控制 backbone 因素" .-> FILM

    classDef good fill:#dff3e4,stroke:#287a43,color:#102a18
    classDef warn fill:#fff0c7,stroke:#a96800,color:#3a2600
    classDef bad fill:#f8dddd,stroke:#a64040,color:#351010
    class P1 good
    class GS,FILM,MASK,PLAIN warn
    class GT,SP bad
```

## 18. 对应代码

| 架构部分 | 代码位置 |
|---|---|
| 顶层视觉 -> 条件 -> 音频前向 | `avgaussianv2/models/fusion.py` |
| Native / FiLM / Direct / Gated 组合 | `avgaussianv2/backends/audio_audiogs.py` |
| FiLM U-Net | `avgaussianv2/models/film_unet.py` |
| RGBD global encoder | `avgaussianv2/models/rgbd.py` |
| 2D RGBD token encoder | `avgaussianv2/models/visual_tokens.py` |
| Early token prototype | `origin/codex/cross-attention-audio-tokens:avgaussianv2/models/cross_attention_audio.py` |
| AudioGS-native RGBD Cross-Attention | `origin/agent/cross-attention-benchmark:avgaussianv2/models/cross_attention_audio.py` |
| Gaussian-token Cross-Attention | `avgaussianv2/models/cross_attention_audio.py` |
| AudioGS Gaussian token schema | `avgaussianv2/models/acoustic_gaussian_tokens.py` |
| Mask Cross-Attention | `avgaussianv2/models/mask_cross_attention.py` |
| P1 audio queries / geometry bias | `avgaussianv2/models/p1_audio.py` |
| P1 geometric VisualMemory | `avgaussianv2/models/p1_visual.py` |
| P1 camera-contrast training objective | `avgaussianv2/train.py` |
| Spatial P1 historical objective | `origin/agent/p1-spatial-camera-contrast:avgaussianv2/train.py` |

本文只画已经在当前实验矩阵中实现或评估的版本。LRE-anchor 等后续诊断配置是
输出后处理消融，不作为一个新的基础架构重复绘制。

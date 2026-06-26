---
title: "Flamingo：如何给冻结的大语言模型LLM装上一双眼睛"
date: 2026-06-21T20:00:00+08:00
draft: false
tags: ["Flamingo", "VLM", "Multi-Modal", "Few-Shot Learning", "In-Context Learning", "Cross-Attention"]
categories: ["Paper Interpretation"]
showToc: true
TocOpen: true
math: true
---

> **论文基本信息**
>
> * **Title**: Flamingo: a Visual Language Model for Few-Shot Learning
> * **Authors**: Jean-Baptiste Alayrac et al.
> * **Institution**: DeepMind
> * **arXiv**: [2204.14198](https://arxiv.org/abs/2204.14198)
> * **Venue**: Advances in Neural Information Processing Systems 35 (NeurIPS 2022)


2022 年，DeepMind 发布了论文 《Flamingo: a Visual Language Model for Few-Shot Learning》。

今天再看 Flamingo，它的性能数字可能已经不再耀眼，但它所建立的架构范式依然影响深远：
不从头训练一个庞大的多模态模型，而是尽可能保留已经训练好的视觉模型和语言模型，
只在二者之间加入少量可训练的连接模块。

Flamingo 想解决的问题可以概括为：

能不能让一个已经具备强大语言能力的大模型，在不破坏原有能力的情况下学会看图片和视频，
并像语言模型一样通过上下文示例完成新任务？

它给出的答案主要由三个部分组成：
* **用 Perceiver Resampler 把大量视觉特征压缩成固定数量的视觉 Token**

* **用 Gated XATTN-DENSE 模块，把视觉信息逐层注入冻结的语言模型**

* **用交织图文数据和特殊注意力掩码，实现多模态 In-context Learning**

整套架构可以先概括成下面的数据流：
```text
图片 / 视频
     │
     ▼
冻结的 NFNet-F6 视觉编码器
     │
     │ 可变长度的视觉特征序列（图像为空间特征，视频为时空特征）
     ▼
Perceiver Resampler
     │
     │ 固定数量的视觉 Token，通常为 64 个
     ▼
Gated XATTN-DENSE 模块
     │
     │ 插入冻结语言模型的不同深度
     ▼
冻结的 Decoder-only Language Model
     │
     ▼
自回归生成文本
```


---

# 1.Perceiver Resampler：将可变大小的视觉特征重采样为固定数量的视觉 Token 



### 1.1 输入视觉特征

设一个图像或视频经过冻结的视觉编码器后得到视觉特征：

$$
X_f\in\mathbb{R}^{T\times S\times d}
$$

其中：

* (T) 表示视频的采样帧数；对于单张图片，可以令 (T=1)；
* (S) 表示每一帧经过视觉编码器后得到的视觉向量数量。例如，若视觉编码器输出一个 (14\times14) 的特征网格，则 (S=196)；
* (d) 表示每个视觉向量的特征维度。

对于视频，Flamingo 会为不同帧加入可学习的时间位置编码，使模型能够区分各视觉向量来自哪个时间位置。随后，模型将时间维和空间维展平：

$$
X_f\in\mathbb{R}^{T\times S\times d}
\longrightarrow
\widetilde{X}_f\in\mathbb{R}^{(TS)\times d}
$$

因此，单张图片最终形成 (S) 个视觉向量，而包含 (T) 帧的视频最终形成 (TS) 个视觉向量。

这些视觉向量的数量会随输入图像或视频而变化。接下来，Perceiver Resampler 会将它们重
采样为固定数量的视觉 Token。

为了简化后续表达，下面统一使用：

$$
\widetilde{X}_f\in\mathbb{R}^{N_v\times D}
$$

表示展平后的视觉特征序列。

### 1.2 为什么需要 Perceiver Resampler？

问题在于，视觉特征序列的长度$N_v$并不固定。
对于单张图片,$N_v$取决于视觉编码器输出的特征网格大小；对于视频，$N_v$ 还会随着
采样帧数 $T$ 增加：$N_v=TS$

如果让语言模型中的文本 Token 直接查询全部视觉向量，那么 Cross-Attention 的 
计算量和显存开销都会随着$N_v$增长。

Flamingo 因此在视觉编码器和语言模型之间加入了 Perceiver Resampler。它的目标
不是直接完成视觉理解，而是建立一个固定长度的视觉接口：

$$
\text{可变长度视觉序列}
\quad
[N_v,D]
\quad\longrightarrow\quad
\text{固定长度视觉序列}
\quad
[R,D]
$$

在 Flamingo 中：
$
R=64
$

也就是说，无论一个图像或视频最初产生多少个视觉向量，Perceiver Resampler 
最终都会为它输出 64 个视觉 Token。

### 1.3 64个可学习的Latent

为了得到固定数量的输出，Perceiver Resampler 内部维护了一组可学习的初始 Latent：

$$
X^{(0)}\in\mathbb{R}^{R\times D},
\qquad R=64
$$

可以把它们理解为 64 个可学习的“视觉信息槽位”或“查询向量”。

需要注意，初始 Latent 并不是由当前图像生成的，而是模型自身的可训练参数。
对于不同图像，它们使用相同的初始值；经过多层 Attention 更新后，
才会变成包含当前图像或视频内容的视觉 Token。

其基本过程可以概括为：

```text
视觉特征：[N_v, D] 
    │ 被查询和聚合      
    │
    ▼ 
64 个初始 Latent：[64, D]
    │ 经过多层更新
    │
    ▼
64 个内容相关的视觉 Token：[64, D]
```

### 1.4 数据流图

下面的图展示了Preceiver Resampler中较完整的张量变化

```text
说明：
B = Batch Size，图中主流程省略 B
T = 视频帧数，单张图片时 T = 1
S = 每一帧的空间特征位置数
N_v = T × S
R = Latent 数量，Flamingo 中 R = 64
D = Resampler 隐藏维度，Flamingo 中 D = 1536
H = 注意力头数，Flamingo 中 H = 16
d_h = D / H = 96


                  一个图像或视频
                         │
                         ▼
                冻结的 NFNet-F6
                         │
                         ▼
           视觉特征 X_f: [B, T, S, D]
                         │
                         │      时间位置编码
                         │   E_time: [1, T, 1, D]
                         │          │
                         └─────── Add
                                    │
                                    ▼
                   X_f: [B, T, S, D]
                                    │
                     展平时间维和空间维
                         N_v = T × S
                                    │
                                    ▼
                   X_f: [B, N_v, D]
                                    │
                                    │
                                    │       可学习的初始 Latents
                                    │       X_param⁽⁰⁾: [R, D]
                                    │                │
                                    │         沿 Batch 维广播
                                    │                │
                                    │                ▼
                                    │       X⁽⁰⁾: [B, R, D]
                                    │
                                    └──────────┬───────────┘
                                               │
                                               ▼
              ┌─────────────────────────────────────────────┐
              │        Perceiver Resampler 第 l 层           │
              │                                             │
              │ 当前 Latents：                               │
              │ X⁽ˡ⁾: [B, R, D]                             │
              │                                             │
              │ 拼接视觉特征与当前 Latents：                   │
              │ C⁽ˡ⁾ = Concat(X_f, X⁽ˡ⁾)                    │
              │ C⁽ˡ⁾: [B, N_v + R, D]                       │
              │                                             │
              │ Query 来源：X⁽ˡ⁾                             │
              │ Key/Value 来源：C⁽ˡ⁾                         │
              │                                             │
              │ Q = X⁽ˡ⁾W_Q                                 │
              │ K = C⁽ˡ⁾W_K                                 │
              │ V = C⁽ˡ⁾W_V                                 │
              │                                             │
              │ 多头拆分后：                                  │
              │ Q: [B, H, R, d_h]                           │
              │ K: [B, H, N_v + R, d_h]                     │
              │ V: [B, H, N_v + R, d_h]                     │
              │                                             │
              │ Attention Scores = QKᵀ                      │
              │ [B, H, R, N_v + R]                          │
              │                                             │
              │ Attention Output：                          │
              │ [B, H, R, d_h]                              │
              │        │                                    │
              │        ▼ 合并 H 个注意力头                    │
              │ [B, R, D]                                   │
              │        │                                    │
              │ Attention Output + Residual                 │
              │        │                                    │
              │        ▼                                    │
              │ FFN：                                       │
              │ [B,R,D] → [B,R,4D] → [B,R,D]                │
              │        │                                    │
              │ FFN Output + Residual                       │
              └────────────────────┬────────────────────────┘
                                   │
                                   ▼
                         X⁽ˡ⁺¹⁾: [B, R, D]
                                   │
                              重复 6 层
                                   │
                                   ▼
                     当前图像或视频的视觉 Token
                           [B, 64, 1536]
                                   │
                                   ▼
                      Gated Cross-Attention
```
这张图虽然包含较多维度，但真正需要关注的只有两条信息流：

```text
视觉特征 X_f ───────────────┐
                           ├──▶ 更新 64 个 Latent
当前 Latents X⁽ˡ⁾ ─────────┘
```

视觉特征提供当前图像或视频的内容，而当前 Latent 负责以固定数量的查询位置读取和组织这些内容。

### 1.5 单层Resampler的工作

设第 (l) 层输入的 Latent 为：

$$
X^{(l)}\in\mathbb{R}^{R\times D}
$$

首先，将展平后的视觉特征与当前 Latent 沿序列维拼接：

$$
\widetilde{X}_f;X^{(l)}
\in
\mathbb{R}^{(N_v+R)\times D}
$$

随后：
$$
Q = X^{(l)} W_Q,
\qquad
K = C^{(l)} W_K,
\qquad
V = C^{(l)} W_V
$$

因此：
Query 来自当前的$R$个 Latent；
Key 和 Value 同时包含 $N_v$ 个视觉向量和 (R) 个 Latent。

Attention 更新可以简写为：

$$
X^{(l+1)} = X^{(l)} + \operatorname{Attention}(Q = X^{(l)}, K = [\widetilde{X}_f ; X^{(l)}], V = [\widetilde{X}_f ; X^{(l)}])
$$

随后再经过前馈网络：

$$
\widehat{X}^{(l+1)} = X^{(l+1)} + \operatorname{FFN}(X^{(l+1)})
$$

经过多层迭代后，最初与图像内容无关的 Latent 会逐渐吸收视觉特征，
最终形成 64 个内容相关的视觉 Token。


### 1.6 Perceiver Resampler 的作用

Perceiver Resampler 的核心作用可以概括为：

$$
\underbrace{\widetilde{X}_f}{\text{长度随输入变化}}
\in
\mathbb{R}^{N_v\times D}
\quad
\longrightarrow
\quad
\underbrace{X_{\text{visual}}}_{\text{固定长度}}
\in
\mathbb{R}^{64\times D}
$$

它解决的首先是视觉接口问题，而不是直接完成语言推理：

无论上游输入是一张图片还是一个视频，下游语言模型看到的都是固定数量的视觉 Token。

这使得后续视觉—语言 Cross-Attention 的计算规模保持稳定。

当然，固定长度也意味着信息瓶颈：无论原始视觉序列包含多少信息，最终都必须被浓缩到 64 个向量中。对于高度依赖细粒度局部信息的任务，这种压缩可能损失部分视觉细节。

完成重采样后，这 64 个视觉 Token 会作为 Key 和 Value，被送入语言模型不同深度的 Gated Cross-Attention 模块。接下来要解决的问题便从：

如何把不定长视觉特征转换为固定长度表示？

转变为：

如何在不破坏预训练语言能力的情况下，将这些视觉表示注入冻结的语言模型？

---

# 2. Gated XATTN-DENSE：把视觉信息注入语言模型深处

经过 Perceiver Resampler 后，每个图像或视频都被表示为固定数量的视觉 Token：

$$
X_v\in\mathbb{R}^{R\times D_v},
\qquad R=64
$$

但此时，视觉信息仍然位于语言模型之外。

接下来需要解决的问题是：

如何让冻结的语言模型在生成文本时读取这些视觉 Token，同时尽量保留原有的语言能力？

一种直接的方法，是把视觉 Token 投影到语言模型的隐藏维度后，与文本 Token 拼接在输入端：

```text
[视觉 Token] + [文本 Token]
              │
              ▼
         Language Model
```

这种方法只在语言模型入口处注入一次视觉信息。随着隐藏状态逐层传播，模型后续
只能依赖已经混入文本表示中的视觉信息。

Flamingo 采用了另一种思路：在冻结语言模型的不同深度插入新的 Cross-Attention 模块，
让文本隐藏状态在推理过程中可以反复读取视觉 Token。

### 2.1 模块插入位置

设预训练语言模型由一系列 Transformer Block 构成：

```text
文本 Embedding
      │
      ▼
Frozen LM Block 1
      │
      ▼
Frozen LM Block 2
      │
      ▼
Frozen LM Block 3
      │
     ...
      │
      ▼
Language Model Head
```

Flamingo 在部分冻结的 LM Block 之前，插入可训练的 Gated XATTN-DENSE 模块：

```text
文本隐藏状态
      │
      ▼
Gated XATTN-DENSE      ← 读取视觉 Token
      │
      ▼
Frozen LM Block
      │
      ▼
下一层文本隐藏状态
```

因此，完整结构更接近：

```text
Text Embedding
      │
      ▼
Gated XATTN-DENSE
      │
      ▼
Frozen LM Block 1
      │
      ▼
Gated XATTN-DENSE
      │
      ▼
Frozen LM Block 2
      │
     ...
```

这里需要特别注意：

Gated XATTN-DENSE 不是把原有 Transformer Block 拆开，再插入 Self-Attention 和 FFN 之间；
它是一个新增模块，被放在部分原始语言模型层之前。

原始语言模型的参数保持冻结，训练时主要更新：
* Perceiver Resampler；
* Gated Cross-Attention；
* 新增的 Dense FFN；
* 对应的门控参数。


### 2.2 Cross-Attention 中如何查询

在 Gated XATTN-DENSE 的 Cross-Attention 中，信息流向是：**文本隐藏状态主动查询视觉 Token。**

设进入当前 Gated XATTN-DENSE 模块的文本隐藏状态为：

$$
H^{(l)}
\in
\mathbb{R}^{B\times L\times D_{\mathrm{LM}}}
$$

其中：

* $B$ 表示 Batch Size；
* $L$ 表示文本序列长度；
* $D_{\mathrm{LM}}$ 表示语言模型的隐藏维度。

Perceiver Resampler 输出的视觉 Token 记为：

$$
X_v
\in
\mathbb{R}^{B\times R\times D_v},
\qquad R=64
$$

在 Cross-Attention 中：

$$
Q=\operatorname{LN}(H^{(l)})W_Q
\qquad
K=X_vW_K,
\qquad
V=X_vW_V
$$

也就是说：

* **Query 来自当前文本隐藏状态；**
* **Key 和 Value 来自 64 个视觉 Token。**

可以将其画成：
```text
当前文本隐藏状态 H⁽ˡ⁾
     [B, L, D_LM]
             │
             │ 作为 Query
             ▼
      ┌─────────────────┐
      │ Cross-Attention │
      └────────-────────┘
               ▲
               │ 作为 Key 和 Value
               │
      64 个视觉 Token X_v
          [B, 64, D_v]
```

其核心计算可以简写为：

$$
\operatorname{XAttn}(H^{(l)},X_v)
=
\operatorname{Softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V
$$

对于单个注意力头，张量维度大致为：

$$
Q\in\mathbb{R}^{B\times L\times d_h}
$$

$$
K,V\in\mathbb{R}^{B\times R\times d_h}
$$

因此，注意力分数矩阵的形状是：

$$
QK^\top
\in
\mathbb{R}^{B\times L\times R}
$$

在 Flamingo 中 (R=64)，所以可以理解为：

> 每一个文本位置，都会对当前图像或视频的 64 个视觉 Token 计算一组注意力权重。

例如，某个文本位置可能主要关注第 3、17 和 42 个视觉 Token，而另一个文本位置可能关注完全不同的视觉信息。

```text
文本位置 h₁ ──关注──▶ 视觉 Token 3、17、42
文本位置 h₂ ──关注──▶ 视觉 Token 8、11、36
文本位置 h₃ ──关注──▶ 视觉 Token 5、29、61
```

经过注意力加权后，每个文本位置都会得到一个视觉信息向量：

$$
\Delta H_{\mathrm{vision}}
=
\operatorname{XAttn}(H^{(l)},X_v)
\in
\mathbb{R}^{B\times L\times D_{\mathrm{LM}}}
$$

因此，Cross-Attention 的输出长度仍然是 (L)，而不是 64。

原因在于：

> Attention 的输出位置数量由 Query 的数量决定。

这里一共有 (L) 个文本 Query，所以最终得到 (L) 个视觉信息增量，每个增量对应一个文本位置：

```text
L 个文本隐藏状态
        │
        │ 分别查询 64 个视觉 Token
        ▼
L 个视觉信息增量
```


### 2.3 将查询结果写回文本隐藏状态

经过 Cross-Attention 后，每个文本位置都会得到一个对应的视觉信息增量：

$$
\Delta H_{\mathrm{vision}}
=
\operatorname{XAttn}(H^{(l)},X_v)
\in
\mathbb{R}^{B\times L\times D_{\mathrm{LM}}}
$$

它与原始文本隐藏状态 (H^{(l)}) 具有完全相同的形状：

$$
H^{(l)}
\in
\mathbb{R}^{B\times L\times D_{\mathrm{LM}}}
$$

因此，可以逐位置地将视觉信息加回文本隐藏状态。

在 Flamingo 中，这个视觉增量还会先经过一个可学习门控：

$$\widetilde{H}^{(l)} = H^{(l)} + \tanh\!\left(\alpha_{\mathrm{xattn}}^{(l)}\right)\Delta H_{\mathrm{vision}}$$

其中：

- $H^{(l)}$ 是进入 Cross-Attention 之前的文本隐藏状态；
- $\Delta H_{\mathrm{vision}}$ 是文本查询视觉 Token 后得到的视觉信息增量；
- $\alpha_{\mathrm{xattn}}^{(l)}$ 是当前 Gated Cross-Attention 层的可学习门控参数；
- $\widetilde H^{(l)}$ 是融合视觉信息后的文本隐藏状态。

整个过程可以画成：

```text
                         64 个视觉 Token X_v
                              [B, 64, D_v]
                                      │
                                 Key / Value
                                      │
                                      ▼
原始文本隐藏状态 H⁽ˡ⁾ ───────▶ Cross-Attention
     [B, L, D_LM]                 Query = H⁽ˡ⁾
          │                             │
          │                             ▼
          │                 视觉信息增量 ΔH_vision
          │                       [B, L, D_LM]
          │                             │
          │                 × tanh(α_xattn⁽ˡ⁾)
          │                             │
          └─────────────── Residual Add
                                        │
                                        ▼
                         融合后的文本状态 H̃⁽ˡ⁾
                              [B, L, D_LM]
```

这里没有生成新的文本 Token，也没有改变文本序列长度 (L)。

Cross-Attention 所做的是：为原有的 (L) 个文本位置分别计算一个视觉补充向量，再将这些向量加回对应的文本隐藏状态：

```text
文本位置 h₁ + 对应的视觉增量 Δh₁ → 融合状态 h̃₁
文本位置 h₂ + 对应的视觉增量 Δh₂ → 融合状态 h̃₂
文本位置 h₃ + 对应的视觉增量 Δh₃ → 融合状态 h̃₃
...
文本位置 hL + 对应的视觉增量 ΔhL → 融合状态 h̃L
```

因此，更新前后的形状保持不变：

$$\left[B, L, D_{\mathrm{LM}}\right] + \left[B, L, D_{\mathrm{LM}}\right] \longrightarrow \left[B, L, D_{\mathrm{LM}}\right]$$

可以把这一过程理解为：

> 文本隐藏状态保留在残差主干上，Cross-Attention 只负责根据当前文本上下文，从视觉 Token 中提取一份额外证据，并将其作为增量补充到文本表示中。

更新后的 (\widetilde H^{(l)}) 已经同时包含两类信息：


$$\widetilde{H}^{(l)} = \underbrace{H^{(l)}}_{\text{原有文本上下文}} + \underbrace{\tanh\!\left(\alpha_{\mathrm{xattn}}^{(l)}\right)\Delta H_{\mathrm{vision}}}_{\text{按当前文本查询得到的视觉信息}}$$


随后，这个融合后的文本状态会继续进入 Gated XATTN-DENSE 中新增的 Dense FFN。





### 2.4 为什么是文本查询视觉，而不是视觉查询文本

Flamingo 的最终目标是自回归生成文本。

在生成下一个 Token 时，模型需要根据当前语言上下文判断：

> 为了继续生成，我现在需要从图像中读取什么信息？

因此，当前文本隐藏状态最适合作为 Query。

例如，当文本上下文分别是：

```text
“What color is the car?”
“How many people are visible?”
“What is the person holding?”
```

虽然它们读取的是同一组视觉 Token，但由于文本 Query 不同，Cross-Attention 可以分别关注：

* 汽车颜色相关的视觉信息；
* 人物数量相关的视觉信息；
* 手部及其附近物体相关的视觉信息。

所以，这一机制并不是把整张图像无差别地灌入每个文本位置，而是让文本状态根据当前上下文，有选择地读取视觉表示。



### 2.5 不同深度使用的 Query 也不同 

同一组 64 个视觉 Token 会被送入语言模型不同深度的 Gated XATTN-DENSE 模块，
但每一层使用的文本隐藏状态不同：

$$
H^{(1)},H^{(2)},\ldots,H^{(l)}
$$

随着文本状态经过更多语言模型层，它会逐渐融合更丰富的上下文信息。因此，即使视觉 Token 没有变化，
不同深度的 Cross-Attention 也可能读取不同的视觉内容：

```text
浅层文本状态 H¹ ──查询──▶ 视觉 Token X_v
中层文本状态 H² ──查询──▶ 视觉 Token X_v
深层文本状态 H³ ──查询──▶ 视觉 Token X_v
```

可以把这 64 个视觉 Token 看作一个固定的视觉记忆库，而语言模型在推理的不同阶段，
根据当前文本状态反复查询这个记忆库。

这一小节可以概括为：

$$
\boxed{
\text{Text Hidden States as Query}
\quad
\longrightarrow
\quad
\text{Visual Tokens as Key and Value}
}
$$

Cross-Attention 得到的视觉信息随后不会直接替换原文本状态，而是经过门控后，
以残差形式注入文本隐藏状态。

---


# 3. 交织图文输入与注意力掩码：实现多模态 In-context Learning

前两部分解决了两个架构问题：

* Perceiver Resampler 将可变长度的视觉特征转换成固定数量的视觉 Token；
* Gated XATTN-DENSE 让冻结的语言模型能够在不同深度读取这些视觉 Token。

但 Flamingo 面对的输入并不一定只有一张图片和一句问题。它还希望处理下面这种图文交替出现的上下文：

```text
[图片 1] [问题 1] [答案 1]
[图片 2] [问题 2] [答案 2]
[图片 3] [问题 3] [待生成答案]
```

这就产生了一个新的问题：

> 当上下文中同时存在多张图片时，当前文本应该读取哪一组视觉 Token？

Flamingo 通过交织图文训练数据和特殊的 Cross-Attention Mask，建立图片与后续文本之间的对应关系。

### 3.1 什么是交织图文输入？

介绍图像与文本交替出现的输入形式，并与单独的图文对进行对比。

### 3.2 文本能够看到哪些信息？

分别讨论两套注意力机制：

* Causal Self-Attention：当前文本 Token 可以看到此前所有文本 Token；
* Gated Cross-Attention：当前文本 Token 只直接读取最近一张前置图片的视觉 Token。

### 3.3 单图 Cross-Attention Mask

使用如下例子说明视觉可见范围：

```text
输入顺序：

[图片 1] → [文本 A] → [图片 2] → [文本 B] → [图片 3] → [文本 C]
```

对应关系为：

```text
文本 A ──直接查询──▶ 图片 1
文本 B ──直接查询──▶ 图片 2
文本 C ──直接查询──▶ 图片 3
```

并画出简化掩码矩阵：

```text
                 图片 1    图片 2    图片 3
文本 A              ✓         ×         ×
文本 B              ×         ✓         ×
文本 C              ×         ×         ✓
```

### 3.4 只直接读取最近图片，不等于更早图片完全失效

Flamingo 的视觉 Cross-Attention 规定：每个文本 Token 只直接读取它前面最近出现的那张图片。

例如，对于下面的交织序列：

```text
[图片 1] → [文本 A] → [图片 2] → [文本 B] → [图片 3] → [文本 C]
```

直接的视觉对应关系是：

```text
文本 A ──Cross-Attention──▶ 图片 1
文本 B ──Cross-Attention──▶ 图片 2
文本 C ──Cross-Attention──▶ 图片 3
```

因此，文本 C 不会在 Cross-Attention 中直接查询图片 1 和图片 2 的视觉 Token。

但这并不意味着更早的图片对文本 C 完全没有影响。

关键在于，文本 A 和文本 B 在前面的网络层中已经分别读取了图片 1 和图片 2。
它们的隐藏状态不再是纯文本表示，而是已经包含了与对应图片有关的信息：

```text
图片 1 ──Cross-Attention──▶ 文本 A 的隐藏状态
图片 2 ──Cross-Attention──▶ 文本 B 的隐藏状态
图片 3 ──Cross-Attention──▶ 文本 C 的隐藏状态
```

与此同时，语言模型中的因果 Self-Attention 允许文本 C 读取此前的文本 A 和文本 B。

因此，文本 C 接收到的信息可以分成两条路径：

```text
直接视觉路径：

图片 3 的视觉 Token
        │
        ▼
Cross-Attention
        │
        ▼
当前文本 C


间接上下文路径：

图片 1 → 文本 A 的隐藏状态 ─┐
                         ├─Self-Attention─▶ 当前文本 C
图片 2 → 文本 B 的隐藏状态 ─┘
```

换句话说：

* 文本 C 通过 Cross-Attention **直接读取图片 3**；
* 文本 C 通过 Self-Attention **读取文本 A 和文本 B**；
* 而文本 A、B 的隐藏状态中，已经包含了由图片 1、2 注入的部分视觉信息。

从一层计算过程来看，可以简化为：

$$
\widetilde H_C^{(l)}
=
H_C^{(l)}+
\operatorname{GatedXAttn}
\left(
H_C^{(l)},X_{\mathrm{image\ 3}}
\right)
$$

当前文本首先从最近的图片 3 中读取视觉信息。随后，进入冻结语言模型 Block 的因果 Self-Attention：

$$
H_C^{(l+1)}
=
\operatorname{SelfAttention}
\left(
\widetilde H_A^{(l)},
\widetilde H_B^{(l)},
\widetilde H_C^{(l)}
\right)
$$

其中：

* $\widetilde H_A^{(l)}$ 已经融合过图片 1 的信息；
* $\widetilde H_B^{(l)}$ 已经融合过图片 2 的信息；
* $\widetilde H_C^{(l)}$ 直接融合了图片 3 的信息。

因此，当前文本 C 的信息来源可以概括为：

$$\boxed{ \text{当前图片的直接视觉证据}+ \text{此前文本携带的历史上下文} }$$

这种设计的意义是：

> 当前文本不会把多张图片的视觉 Token 直接混在同一次 Cross-Attention 中，
> 但仍然可以通过此前文本理解前面的示例、任务规则和对话上下文。

需要注意，这种间接传播并不等于文本 C 可以完整恢复图片 1 和图片 2 的全部视觉细节。
更早图片的信息已经经过此前文本隐藏状态的筛选和压缩，因此它们主要用于保留任务规则、
语义线索和上下文关系，而当前问题所需的具体视觉证据仍主要来自最近一张图片。

### 3.5 Few-shot / In-context Learning 如何发生？

给出一个完整示例：

[图片 1]
Question: 图中有几只动物？
Answer: 2

[图片 2]
Question: 图中有几只动物？
Answer: 3

[图片 3]
Question: 图中有几只动物？
Answer:

模型通过此前文本学习任务形式和输出格式，同时通过 Cross-Attention 读取图片 3 的视觉内容，
从而生成新的答案。


### 3.6 本章总结

Flamingo 的多模态上下文学习可以概括为：

文本 Self-Attention 负责保留此前示例中的任务规则和语言上下文，视觉 Cross-Attention Mask 则
负责让当前文本读取与自己对应的图片。

这种“双轨注意力”使 Flamingo 能够在不更新模型参数的情况下，仅通过少量图文示例理解新任务。

---

# 总结：Flamingo 留下了什么？

回到最开始的问题：

> 如何让一个已经具备强大语言能力的大模型，在尽量不破坏原有能力的情况下学会看图和看视频？

Flamingo 给出的答案不是从头训练一个全新的多模态模型，而是将已经训练好的视觉模型和语言模型连接起来，
并且只训练中间的适配模块。

它的核心设计可以概括为三条主线。

### 第一，**Perceiver Resampler 解决视觉输入长度不固定的问题**。

图像和视频经过视觉编码器后，会产生数量不固定的视觉特征。单张图片的视觉向量数量取决于特征图大小，
视频还会随着采样帧数增加而变长。

如果直接把这些视觉特征全部交给语言模型，计算成本会随着视觉序列长度迅速增长。Flamingo 因此
使用 Perceiver Resampler， 通过 64 个可学习 Latent 将每个图像或视频重采样为
固定数量的视觉 Token。

这一步建立了一个稳定的视觉接口：

$$
\text{可变长度视觉特征}
\longrightarrow
\text{固定数量视觉 Token}
$$

也就是说，无论输入是一张图片还是一段视频，下游语言模型接收到的都是统一形式的视觉表示。

### 第二，**Gated XATTN-DENSE 解决视觉信息如何注入冻结语言模型的问题**。

得到视觉 Token 之后，Flamingo 并没有简单地把它们拼接到文本输入前面，而是在冻结语言模型的
不同深度插入 Gated XATTN-DENSE 模块。

在这个模块中，文本隐藏状态作为 Query，视觉 Token 作为 Key 和 Value。也就是说，
语言模型在生成过程中，会根据当前文本上下文主动查询视觉信息。

更关键的是，视觉信息不是直接强行加入文本状态，而是先经过零初始化的门控：

$$
H' =
H+
\tanh(\alpha)
\Delta H_{\text{vision}}
$$

训练开始时，$\alpha=0$，所以视觉分支对语言模型几乎没有影响。随着训练进行，门控逐渐打开，
视觉信息才被平滑地注入文本隐藏状态。

这使得 Flamingo 能够在冻结语言模型主体的前提下，学习视觉—语言融合，同时尽量保持原
有语言能力和训练稳定性。

### 第三，**交织图文输入和注意力掩码解决多图上下文学习的问题**。

Flamingo 不只处理“一张图 + 一句话”的简单图文对，而是支持图像和文本交替出现的输入形式：

```text
[图片 1] [问题 1] [答案 1]
[图片 2] [问题 2] [答案 2]
[图片 3] [问题 3] [待生成答案]
```

为了避免多张图片的视觉信息混在一起，Flamingo 在 Cross-Attention 中使用特殊掩码：每个文本 Token 只直接读取它前面最近一张图片的视觉 Token。

与此同时，语言模型的因果 Self-Attention 仍然允许当前文本读取此前所有文本。因此，模型既能从前面的文本示例中学习任务形式和输出格式，又能从当前对应图片中读取视觉证据。

可以概括为：

```text
Self-Attention：
读取此前文本，学习任务规则和上下文格式

Cross-Attention：
读取当前对应图片，获取视觉证据
```

这正是 Flamingo 实现多模态 In-context Learning 的关键。

所以，Flamingo 真正留下的不是某一个单独模块，而是一套非常清晰的多模态建模范式：

```text
冻结视觉编码器
      │
      ▼
Perceiver Resampler
      │
      ▼
固定数量视觉 Token
      │
      ▼
Gated Cross-Attention
      │
      ▼
冻结语言模型
      │
      ▼
自回归生成文本
```

它证明了一件非常重要的事情：

> 构建多模态大模型，并不一定要重新训练完整的视觉模型和语言模型。只要设计足够好的视觉接口和注入机制，就可以让强大的单模态模型协同工作。

当然，Flamingo 也有局限。

Perceiver Resampler 的 64 个视觉 Token 是一种高效压缩，但也可能成为信息瓶颈；固定的视觉 Token 在进入语言模型前已经生成，并不会根据具体问题动态重新采样图像细节；冻结语言模型降低了训练成本，但也限制了语言主干对多模态表示的适应能力。

即便如此，Flamingo 仍然是多模态大模型发展中的重要节点。它将视觉压缩、深层视觉注入、交织图文上下文学习这三件事系统地结合起来，奠定了后来许多视觉语言模型的设计基础。

用一句话概括 Flamingo：

> 它先把图像或视频压缩成固定数量的视觉 Token，再让冻结语言模型在不同深度通过带门控的 Cross-Attention 反复读取这些视觉 Token，从而把语言模型的上下文学习能力扩展到多模态场景。

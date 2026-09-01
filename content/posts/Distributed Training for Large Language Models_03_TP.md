---
title: "Distributed Training for Large Language Models_03_TP"
date: 2026-08-26T10:00:00+08:00
draft: false
tags: ["Distributed Training", "DeepSpeed", "ZeRO","TP", "Megatron", "3D Parallelism", "LLM"]
categories: ["System & Architecture"]
showToc: true
TocOpen: true
math: true
---





# 1.3 TP (Tensor Parallelism - 张量并行)：深入单层内部，切分巨大矩阵

上一节的 Pipeline Parallelism（PP）解决了这样一个问题：

> **整个模型太大，单张 GPU 放不下怎么办？**

PP 给出的答案是：

```text
按照 Layer 切模型
```

例如：

```text
GPU 0 → Layer  1 ~ 10
GPU 1 → Layer 11 ~ 20
GPU 2 → Layer 21 ~ 30
GPU 3 → Layer 31 ~ 40
```

这样，一个完整模型就可以跨多张 GPU 存储。

但是当模型规模继续膨胀时，又会遇到一个更加棘手的问题：

> **如果连一个 Transformer Layer 本身都大到单张 GPU 放不下呢？**

例如 Transformer 中的：

* Q / K / V Projection
* Attention Output Projection
* MLP Up Projection
* MLP Down Projection

本质上都包含巨大的矩阵乘法。

当 Hidden Size 不断增大时，一个 Weight Matrix 本身就可能拥有数亿甚至更多参数。

此时：

```text
PP
│
└── 只能沿 Layer 边界切
```

已经无法继续解决问题。

于是必须把“手术刀”进一步深入 Layer 内部：

> **把一个 Tensor / Weight Matrix 本身切到多张 GPU 上。**

这就是：

> **Tensor Parallelism（TP，张量并行）**

如果只用一句话概括：

> **PP 切 Layer，TP 切 Layer 内部的 Tensor。**

---

## 1. TP 到底在并行什么？

假设一个最普通的 Linear Layer：

$$
Y = XW
$$

其中：

* $X$：输入 Activation
* $W$：模型权重
* $Y$：输出 Activation

假设：

$$
X \in \mathbb{R}^{B\times H}
$$

$$
W \in \mathbb{R}^{H\times H'}
$$

那么：

$$
Y \in \mathbb{R}^{B\times H'}
$$

在普通单卡训练中：

```text
              X
              │
              ▼
        ┌───────────┐
        │    GPU    │
        │           │
        │ 完整矩阵 W │
        └─────┬─────┘
              │
              ▼
            Y=XW
```

整个矩阵 $W$ 都必须保存在同一张 GPU 上。

但 Tensor Parallelism 会把：

```text
                W
                │
          ┌─────┴─────┐
          ▼           ▼
         W₁           W₂
          │           │
          ▼           ▼
        GPU 0       GPU 1
```

也就是说：

> **一张 GPU 不再保存完整的 Weight Matrix，而只保存其中的一部分。**

多张 GPU 共同完成：

$$
Y=XW
$$

这一层计算。

---

## 2. 用“工厂造车”理解 TP

继续使用汽车工厂的比喻。

在 Pipeline Parallelism 中，我们做的是：

```text
车间 0 → 底盘
          ↓
车间 1 → 发动机
          ↓
车间 2 → 车身
          ↓
车间 3 → 喷漆
```

也就是：

> **不同车间负责不同生产阶段。**

但现在问题更加极端。

假设发动机本身已经巨大到：

> **一个车间根本无法独立完成。**

于是不能再说：

```text
车间 1 → 整台发动机
```

而必须改成：

```text
                 一台巨大 V12 发动机
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       工人 0          工人 1          工人 2
       左侧部分         中间部分         右侧部分
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    拼成完整发动机
```

映射到 GPU：

```text
                 一个巨大 Linear Layer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        GPU 0          GPU 1          GPU 2
       Weight 1       Weight 2       Weight 3
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    得到最终结果
```

所以：

> **PP 是几个车间依次完成不同工序，而 TP 是几个工人同时完成同一个工序。**

这是两者最本质的区别。

---

## 3. 一个矩阵到底应该怎么切？

对于：

$$
Y=XW
$$

假设：

$$
W=
\begin{bmatrix}
w_{11} & w_{12} & w_{13} & w_{14}\\
w_{21} & w_{22} & w_{23} & w_{24}\\
w_{31} & w_{32} & w_{33} & w_{34}\\
w_{41} & w_{42} & w_{43} & w_{44}
\end{bmatrix}
$$

最经典的 Tensor Parallel 切分方法主要有两种：

1. **Column Parallel**
2. **Row Parallel**

也就是：

```text
               Weight Matrix W
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Column Parallel          Row Parallel
        按列切                   按行切
```

这两个概念是理解 Megatron Tensor Parallelism 的核心。

---

## 4. Column Parallel：按输出维度切矩阵

首先看 Column Parallel。

对于：

$$
Y=XW
$$

我们把 $W$ 沿着列方向切开：

$$
W
= 
\begin{bmatrix} 
W_1 & W_2 
\end{bmatrix}
$$

于是：

$$
XW
=
X
\begin{bmatrix}
W_1 & W_2
\end{bmatrix}
$$

可以写成：

$$
Y
=
\begin{bmatrix}
XW_1 & XW_2
\end{bmatrix}
$$

也就是：

$$
Y=
\begin{bmatrix}
Y_1 & Y_2
\end{bmatrix}
$$

其中：

$$
Y_1=XW_1
$$

$$
Y_2=XW_2
$$

于是可以直接分给两张 GPU：

```text
                      输入 X
                 （两张 GPU 都有）
                   /           \
                  /             \
                 ▼               ▼
          ┌─────────────┐ ┌─────────────┐
          │    GPU 0    │ │    GPU 1    │
          │             │ │             │
          │     W₁      │ │     W₂      │
          └──────┬──────┘ └──────┬──────┘
                 │               │
                 ▼               ▼
              Y₁=XW₁          Y₂=XW₂
                 │               │
                 └───────┬───────┘
                         │
                         ▼
                     [Y₁ | Y₂]
```

最大的好处是：

> **GPU 0 和 GPU 1 可以同时完成自己负责的矩阵乘法。**

如果 TP Size 为 $N$，那么理想情况下，每张 GPU 只需要保存：

$$
\frac{1}{N}
$$

的 Weight Matrix。

例如 TP=4：

```text
完整 W：

┌─────────────────────────────────────────┐
│                  W                      │
└─────────────────────────────────────────┘

Column Parallel：

┌──────────┐
│   W₀     │ → GPU 0
└──────────┘

┌──────────┐
│   W₁     │ → GPU 1
└──────────┘

┌──────────┐
│   W₂     │ → GPU 2
└──────────┘

┌──────────┐
│   W₃     │ → GPU 3
└──────────┘
```

于是巨大的矩阵参数被真正拆散到了多张 GPU。

### 一个具体计算案例：两张 GPU 如何完成 Column Parallel？

假设输入：

$$
X=
\begin{bmatrix}
1 & 2 & 3 & 4
\end{bmatrix}
$$

权重矩阵：

$$
W=
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
$$

如果在单张 GPU 上直接计算：

$$
Y=XW
$$

那么：

$$
Y=
\begin{bmatrix}
1 & 2 & 3 & 4
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
$$

最终得到：

$$
Y=
\begin{bmatrix}
90 & 100 & 110 & 120
\end{bmatrix}
$$

---

现在假设使用 **2 张 GPU 做 Column Parallel**。

因为 Column Parallel 是按照权重矩阵 $W$ 的“列”切分，所以：

$$
W=
\begin{bmatrix}
W_1 & W_2
\end{bmatrix}
$$

其中：

$$
W_1=
\begin{bmatrix}
1 & 2 \\
5 & 6 \\
9 & 10 \\
13 & 14
\end{bmatrix}
$$

$$
W_2=
\begin{bmatrix}
3 & 4 \\
7 & 8 \\
11 & 12 \\
15 & 16
\end{bmatrix}
$$

也就是说：

```text
GPU 0:

W₁ =
[ 1   2
  5   6
  9  10
 13  14 ]


GPU 1:

W₂ =
[ 3   4
  7   8
 11  12
 15  16 ]
```

---

## 5. Column Parallel 为什么不一定立刻 All-Gather？

这里有一个很重要的细节。

很多对 TP 的介绍会直接写：

```text
GPU 0 → Y₀
GPU 1 → Y₁
       ↓
   All-Gather
       ↓
    完整 Y
```

逻辑上没有问题，但工程实现并不一定要马上这么做。

因为如果下一层也能够直接消费：

```text
Y₀
Y₁
```

这种分片状态，那么就没有必要急着把完整 $Y$ 拼回来。

也就是说：

> **Tensor Parallelism 的优化核心之一，就是让 Tensor 尽可能保持 Sharded 状态，避免不必要的通信。**

这一点在 Megatron-LM 中尤其重要。

---

## 6. Row Parallel：按输入维度切矩阵

再来看 Row Parallel。

仍然考虑：

$$
Y=XW
$$

这次把 $W$ 按行切：

$$
W=
\begin{bmatrix}
W_1\\
W_2
\end{bmatrix}
$$

那么输入 $X$ 也要沿着对应维度切分：

$$
X=
\begin{bmatrix}
X_1 & X_2
\end{bmatrix}
$$

于是：

$$
Y
=
\begin{bmatrix}
X_1 & X_2
\end{bmatrix}
\begin{bmatrix}
W_1\\
W_2
\end{bmatrix}
$$

展开得到：

$$
Y
=
X_1W_1+X_2W_2
$$

于是两张 GPU 可以分别计算：

$$
Y_1=X_1W_1
$$

$$
Y_2=X_2W_2
$$

然后：

$$
Y=Y_1+Y_2
$$

数据流可以表示成：

```text
                  输入 X
                     │
               Split Tensor
               /           \
              ▼             ▼
           X₁               X₂
            │               │
            ▼               ▼
     ┌─────────────┐ ┌─────────────┐
     │    GPU 0    │ │    GPU 1    │
     │     W₁      │ │     W₂      │
     └──────┬──────┘ └──────┬──────┘
            │               │
            ▼               ▼
        Y₁=X₁W₁         Y₂=X₂W₂
            │               │
            └───────┬───────┘
                    │
                    ▼
                All-Reduce
                    │
                    ▼
              Y = Y₁ + Y₂
```

这里的关键区别出现了：

> **Column Parallel 的局部输出可以通过拼接组成完整结果；Row Parallel 的局部输出则必须通过求和得到最终结果。**

因此 Row Parallel 天然需要一个 Reduce 操作。

在多 GPU 场景下通常表现为：

> **All-Reduce**

### 一个具体计算案例：两张 GPU 如何完成 Row Parallel？

假设输入：

$$
X=
\begin{bmatrix}
1 & 2 & 3 & 4
\end{bmatrix}
$$

权重矩阵：

$$
W=
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6\\
7 & 8
\end{bmatrix}
$$

如果在单张 GPU 上直接计算：

$$
Y=XW
$$

那么：

$$
Y=
\begin{bmatrix}
1 & 2 & 3 & 4
\end{bmatrix}
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6\\
7 & 8
\end{bmatrix}
$$

得到：

$$
Y=
\begin{bmatrix}
1\times1+2\times3+3\times5+4\times7,
\quad
1\times2+2\times4+3\times6+4\times8
\end{bmatrix}
$$

即：

$$
Y=
\begin{bmatrix}
50 & 60
\end{bmatrix}
$$

---

现在假设使用 **2 张 GPU 做 Row Parallel**。

因为 $W$ 是按照“行”切分的，所以：

$$
W=
\begin{bmatrix}
W_1\\
W_2
\end{bmatrix}
$$

其中：

$$
W_1=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

$$
W_2=
\begin{bmatrix}
5 & 6\\
7 & 8
\end{bmatrix}
$$

与之对应，输入 $X$ 也必须沿相同维度切开：

$$
X=
\begin{bmatrix}
X_1 & X_2
\end{bmatrix}
$$

其中：

$$
X_1=
\begin{bmatrix}
1 & 2
\end{bmatrix}
$$

$$
X_2=
\begin{bmatrix}
3 & 4
\end{bmatrix}
$$

于是：

```text
GPU 0:

X₁ = [1  2]

W₁ = [1  2
      3  4]


GPU 1:

X₂ = [3  4]

W₂ = [5  6
      7  8]
```

---

### GPU 0 计算自己的 Partial Result

GPU 0 计算：

$$
Y_1=X_1W_1
$$

即：

$$
Y_1=
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

得到：

$$
Y_1=
\begin{bmatrix}
1\times1+2\times3,
\quad
1\times2+2\times4
\end{bmatrix}
$$

所以：

$$
\boxed{
Y_1=
\begin{bmatrix}
7 & 10
\end{bmatrix}
}
$$

---

### GPU 1 计算自己的 Partial Result

GPU 1 计算：

$$
Y_2=X_2W_2
$$

即：

$$
Y_2=
\begin{bmatrix}
3 & 4
\end{bmatrix}
\begin{bmatrix}
5 & 6\\
7 & 8
\end{bmatrix}
$$

得到：

$$
Y_2=
\begin{bmatrix}
3\times5+4\times7,
\quad
3\times6+4\times8
\end{bmatrix}
$$

所以：

$$
\boxed{
Y_2=
\begin{bmatrix}
43 & 50
\end{bmatrix}
}
$$

---

### 为什么最后必须进行 Reduce？

现在两张 GPU 手里分别只有：

```text
GPU 0 → Y₁ = [ 7, 10 ]

GPU 1 → Y₂ = [43, 50]
```

注意：

> **这两个结果都不是完整的最终输出。**

真正的矩阵乘法结果应该是：

$$
Y
=
X_1W_1+X_2W_2
$$

因此必须把两个 Partial Result 相加：

$$
Y = Y_1 + Y_2
$$

即：

$$
Y =
\begin{bmatrix}
7 & 10
\end{bmatrix} +
\begin{bmatrix}
43 & 50
\end{bmatrix}
$$

最终得到：

$$
Y =
\begin{bmatrix}
50 & 60
\end{bmatrix}
$$


这正好与单卡直接计算：

$$
XW
$$

得到的结果完全一致。

整个过程可以画成：

```text
                     X = [1 2 3 4]
                           │
                     Split Tensor
                    /             \
                   ▼               ▼

             X₁=[1 2]          X₂=[3 4]
                 │                 │
                 ▼                 ▼

            ┌─────────┐       ┌─────────┐
            │  GPU 0  │       │  GPU 1  │
            │         │       │         │
            │ W₁      │       │ W₂      │
            │ [1 2]   │       │ [5 6]   │
            │ [3 4]   │       │ [7 8]   │
            └────┬────┘       └────┬────┘
                 │                 │
                 ▼                 ▼

             Y₁=[7,10]        Y₂=[43,50]
                 │                 │
                 └────────┬────────┘
                          │
                          ▼
                       Reduce
                     Element-wise
                         SUM
                          │
                          ▼

                    [7,10]+[43,50]
                          │
                          ▼

                    Y = [50,60]
```

所以 Row Parallel 最值得记住的并不是“按行切”这几个字，而是下面这个数学关系：

$$
\boxed{XW = X_1W_1 + X_2W_2 + \cdots + X_NW_N}
$$

也就是说，每张 GPU 只能算出最终输出的一个 **Partial Sum（部分和）**：

```text
GPU 0 → Partial Y₀
GPU 1 → Partial Y₁
GPU 2 → Partial Y₂
...
```

这些结果无法像 Column Parallel 那样简单进行拼接。

它们必须执行：

$$
\boxed{
Y=\sum_i Y_i
}
$$

因此在 Tensor Parallel Group 中就自然需要：

> **Reduce / All-Reduce**

如果所有 TP Rank 都需要继续持有完整的 $Y$，通常就可以通过 **All-Reduce** 在求和的同时把最终结果同步到所有 GPU。

这就是为什么 **Row Parallel 天然伴随着 Reduce 操作**。


---

### 7. Column Parallel 与 Row Parallel 为什么经常成对出现？

单独看 Column Parallel 或 Row Parallel 可能感觉比较抽象。

但放回 Transformer 的 MLP 中，就会非常漂亮。

一个简化的 Transformer MLP 可以写成：

$$
H
\xrightarrow{W_1}
Z
\xrightarrow{\text{Activation}}
A
\xrightarrow{W_2}
Y
$$

其中：

$$
W_1\in\mathbb{R}^{H\times 4H}
$$

$$
W_2\in\mathbb{R}^{4H\times H}
$$

第一层通常会把 Hidden Dimension 扩大：

```text
H → 4H
```

第二层再压回：

```text
4H → H
```

Megatron 风格的 Tensor Parallel 可以这样切：

```text
                  输入 X
                     │
                     ▼
            Column Parallel
               Linear W₁
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       GPU 0                  GPU 1
     [Z 的一半]             [Z 的一半]
          │                     │
          ▼                     ▼
       GELU / SiLU           GELU / SiLU
          │                     │
          ▼                     ▼
             Row Parallel W₂
          │                     │
          ▼                     ▼
       Partial Y₀           Partial Y₁
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
                 All-Reduce
                     │
                     ▼
                  完整 Y
```

注意这里有一个非常巧妙的地方。

Column Parallel 产生：

```text
GPU 0 → Z₀
GPU 1 → Z₁
```

我们并没有立刻：

```text
All-Gather → 完整 Z
```

而是直接让：

```text
Z₀ → GPU 0 的下一层计算
Z₁ → GPU 1 的下一层计算
```

也就是说：

> **前一层的 Column Parallel 输出，正好可以直接成为后一层 Row Parallel 的分片输入。**

因此：

```text
Column Parallel
      ↓
Sharded Activation
      ↓
Activation Function
      ↓
Row Parallel
      ↓
All-Reduce
```

这就是 Megatron Tensor Parallel 中非常核心的设计思想之一：

> **不是每做一次 GEMM 都通信，而是通过合理安排矩阵切分方式，把通信次数压缩到必要的位置。**

---

## 8. Transformer MLP 中的 TP

用一张完整图表示：

```text
                         Input X
                            │
                            ▼
                ┌─────────────────────┐
                │ Column Parallel W₁  │
                └──────────┬──────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
             GPU 0                   GPU 1
              │                        │
          XW₁_part0                XW₁_part1
              │                        │
              ▼                        ▼
          GELU / SiLU              GELU / SiLU
              │                        │
              ▼                        ▼
          Row Parallel             Row Parallel
             W₂_0                    W₂_1
              │                        │
              ▼                        ▼
         Partial Y₀               Partial Y₁
              │                        │
              └───────────┬────────────┘
                          │
                          ▼
                      All-Reduce
                          │
                          ▼
                       Output Y
```

因此，对于一个典型 MLP Block：

> **第一层 Column Parallel，第二层 Row Parallel。**

这样既切开了两个巨大 Weight Matrix，又尽量减少了中间 Activation 的额外通信。

---

## 9. Attention 中又是怎么切的？

Transformer 的 Self-Attention 同样非常适合 Tensor Parallel。

一个简化的 Multi-Head Attention 可以理解为：

```text
                 Hidden States X
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         Wq           Wk           Wv
          │            │            │
          ▼            ▼            ▼
          Q            K            V
          │            │            │
          └────────────┼────────────┘
                       ▼
              Multi-Head Attention
                       │
                       ▼
                      Wo
                       │
                       ▼
                    Output
```

假设模型有：

```text
32 Attention Heads
```

TP Size 为：

```text
4
```

那么非常自然地可以：

```text
GPU 0 → Head  0 ~  7
GPU 1 → Head  8 ~ 15
GPU 2 → Head 16 ~ 23
GPU 3 → Head 24 ~ 31
```

也就是说：

> **Attention Head 本身天然就是一个非常适合切分的维度。**

QKV Projection 可以使用 Column Parallel：

```text
                X
                │
                ▼
      QKV Column Parallel
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
   GPU 0      GPU 1      GPU 2 ...
  部分 Heads  部分 Heads  部分 Heads
```

各张 GPU 独立完成自己负责的 Attention Heads：

```text
GPU 0:
Q₀K₀ᵀ → Softmax → V₀

GPU 1:
Q₁K₁ᵀ → Softmax → V₁
```

最后 Attention Output Projection 再使用 Row Parallel：

```text
各 GPU 的 Head 输出
          │
          ▼
    Row Parallel Wo
          │
          ▼
      All-Reduce
          │
          ▼
     完整 Hidden State
```

因此一个典型 Transformer Layer 的 Tensor Parallel 可以抽象成：

```text
                  Transformer Layer
                         │
                         ▼
                QKV Column Parallel
                         │
                         ▼
                Attention Heads
                  分布在各 GPU
                         │
                         ▼
               Output Row Parallel
                         │
                         ▼
                     All-Reduce
                         │
                         ▼
                 MLP Column Parallel
                         │
                         ▼
                    GELU / SiLU
                         │
                         ▼
                  MLP Row Parallel
                         │
                         ▼
                     All-Reduce
```

这已经非常接近 Megatron-LM Tensor Parallel 的核心结构了。

---

## 10. TP 为什么通信如此频繁？

现在就可以理解为什么 TP 比 PP 更“吃网络”。

PP 的通信通常发生在：

```text
Stage 0
   │
Activation
   ▼
Stage 1
```

也就是说：

> **跨 Pipeline Stage 时通信。**

但是 TP 不一样。

它发生在：

> **一个 Transformer Layer 内部。**

例如一个 Layer 中：

```text
Attention
    │
    ├── QKV Projection
    │
    ├── Attention
    │
    └── Output Projection
            │
            ▼
        All-Reduce

MLP
    │
    ├── Up Projection
    │
    ├── Activation
    │
    └── Down Projection
            │
            ▼
        All-Reduce
```

如果模型有：

```text
80 Transformer Layers
```

那么一次 Forward / Backward 中会不断触发 TP Collective Communication。

所以 TP 的问题不是：

> **只传一次很大的数据。**

而更接近：

> **模型每推进若干计算阶段，就需要参与 TP 的 GPU 再次同步。**

这使得 TP 对：

* Bandwidth
* Latency
* GPU Interconnect Topology

都非常敏感。

---

## 11. 为什么 TP 对 Latency 特别敏感？

假设 GPU 0 和 GPU 1 正在共同计算同一个 Layer：

```text
GPU 0 → Partial Result 0
GPU 1 → Partial Result 1
```

下一步计算需要：

```text
Partial Result 0
       +
Partial Result 1
       ↓
   完整结果
```

那么通信没有完成之前：

```text
下一步计算
   ↓
无法开始
```

于是：

```text
GPU Compute
██████████

          Collective Communication
          █████

               GPU Compute
               ██████████
```

通信直接进入了模型的 Critical Path。

因此如果通信延迟增加：

```text
GPU Compute
██████████

          Communication
          ███████████████████

                             GPU Compute
                             ██████████
```

GPU 就只能等待。

这和 DP 有明显区别。

DP 中大量 Gradient Communication 可以尝试和 Backward Computation 重叠：

```text
Backward
████████████████████████████

     All-Reduce
     ████████
            ████████
```

而 TP 的一些通信直接决定下一段 Layer Computation 是否能够继续。

所以：

> **TP 通常比 DP 更加延迟敏感。**

---

## 12. TP 的通信模式

可以简单总结：

| 并行方式 | 主要通信                                                           |
| ---- | -------------------------------------------------------------- |
| DP   | Gradient All-Reduce / Reduce-Scatter                           |
| PP   | Activation / Activation Gradient Point-to-Point                |
| TP   | Layer 内部 All-Reduce / All-Gather / Reduce-Scatter 等 Collective |

TP 最常见的特点是：

> **高频 Collective Communication。**

这里“高频”是相对于 DP 和 PP 而言。

并不是说每一个矩阵乘法之后都一定需要 All-Reduce，而是：

> **因为 TP 深入到了 Layer 内部，所以一次完整 Forward / Backward 中会多次进入跨 GPU 同步点。**

---

## 13. 为什么 TP 通常优先放在 NVLink / NVSwitch 内？

这就决定了 TP 的物理部署方式。

假设一台服务器内部有 8 张 GPU：

```text
                 NVSwitch
          ┌─────────┼─────────┐
          │         │         │
        GPU 0     GPU 1     GPU 2
          │         │         │
          └─────────┼─────────┘
                 ...
```

这些 GPU 可以通过：

* NVLink
* NVSwitch

进行高带宽、低延迟通信。

对于 TP 来说，这是非常理想的环境。

因此训练系统往往优先采用：

```text
Node 0

GPU 0 ─┐
GPU 1 ─┤
GPU 2 ─┼── Tensor Parallel Group
GPU 3 ─┘

高速 NVLink / NVSwitch
```

而不是优先：

```text
Node 0 GPU 0
     │
     │ Network Switch
     │
Node 1 GPU 0
```

原因就在于 TP 的 Layer 内同步对网络非常敏感。

所以更准确的工程规律应该写成：

> **TP 通常优先限制在高速 GPU Interconnect 域内，例如单机 NVLink / NVSwitch。**

而不是绝对地说：

> “TP 永远不能跨机器。”

在超大规模训练中，TP 也可能跨节点部署，但通常需要非常高速的网络，而且性能代价会明显增大。

---

## 14. 为什么 TP Size 也不能无限增大？

既然 TP 可以切矩阵，那么一个自然的问题是：

> **能不能把一个矩阵切到 128 张、256 张 GPU 上？**

理论上很多 Tensor 都可以继续切。

但工程上 TP Size 并不是越大越好。

假设：

```text
TP = 2
```

每张 GPU 计算量大约减少到：

$$
\frac{1}{2}
$$

如果：

```text
TP = 8
```

每张 GPU 的局部矩阵更小。

继续增加：

```text
TP = 64
```

局部 GEMM 会越来越小。

于是会出现两个问题。

第一：

> **单张 GPU 上的矩阵太小，Tensor Core 计算效率可能下降。**

第二：

> **参与 Collective Communication 的 GPU 数量越来越多。**

最终可能出现：

```text
计算时间 ↓

但

通信时间 ↑
```

例如：

```text
TP Size
   ↑

Compute per GPU
   ↓↓↓

Communication Overhead
   ↑↑↑
```

所以 Tensor Parallelism 本质上存在一个平衡：

$$
\boxed{
\text{Compute Saving}
\quad\leftrightarrow\quad
\text{Communication Cost}
}
$$

这也是为什么实际系统不会无脑增加 TP Size。

---

## 15. TP 到底解决了什么？

现在重新看 Tensor Parallelism。

Pipeline Parallelism 能做到：

```text
GPU 0 → Layer 1 ~ 10
GPU 1 → Layer 11 ~ 20
GPU 2 → Layer 21 ~ 30
GPU 3 → Layer 31 ~ 40
```

但是一个 Layer 仍然完整存在于某张 GPU 上。

TP 则进一步把：

```text
一个 Layer
```

变成：

```text
                    One Layer
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        GPU 0         GPU 1         GPU 2
       Tensor 0      Tensor 1      Tensor 2
```

因此：

> **TP 解决的是单个 Layer / Tensor 本身过大的问题。**

它把模型并行的粒度从：

```text
Layer
```

继续推进到了：

```text
Tensor / Matrix
```

---

## 16. TP 的代价

当然，TP 也不是免费的午餐。

它解决了显存问题，却换来了极高的通信复杂度。

| 优势                          | 代价                                 |
| --------------------------- | ---------------------------------- |
| 巨大 Weight Matrix 可以跨 GPU 存储 | Layer 内出现 Collective Communication |
| 单层计算可以多 GPU 并行              | 对 Bandwidth 要求高                    |
| 降低单卡 Parameter Memory       | 对 Latency 非常敏感                     |
| 可以提高大型 GEMM 的并行度            | TP Size 过大可能降低计算效率                 |
| 解决单层无法装入单卡的问题               | 对 GPU 拓扑要求高                        |

所以可以把 TP 理解成：

> **用高速通信换取单层内部的模型切分。**

---

## 18. 三种并行方式的核心差异

| 并行模式   | 核心切分对象          | 每张 GPU 保存什么    | 主要通信内容                           | 通信特点                   | 常见物理部署                   |
| ------ | --------------- | -------------- | -------------------------------- | ---------------------- | ------------------------ |
| **DP** | Batch / Data    | 完整模型副本         | Gradient                         | 通信相对集中，可与 Backward 重叠  | 很适合跨 Node 扩展             |
| **PP** | Model Layers    | 一部分 Layer      | Activation / Activation Gradient | Stage 间 Point-to-Point | 可以跨 Node                 |
| **TP** | Tensor / Matrix | 单层 Tensor 的一部分 | Partial Result / Activation      | Layer 内高频 Collective   | 优先 NVLink / NVSwitch 高速域 |

如果继续使用工厂比喻：

```text
DP：

工厂 0 → 完整生产线 → 汽车 A
工厂 1 → 完整生产线 → 汽车 B
工厂 2 → 完整生产线 → 汽车 C

核心：
复制生产线，切订单
```

```text
PP：

车间 0 → 底盘
          ↓
车间 1 → 发动机
          ↓
车间 2 → 车身
          ↓
车间 3 → 喷漆

核心：
拆生产线
```

```text
TP：

                  巨大发动机
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       工人 0       工人 1       工人 2
       负责 1/3      负责 1/3      负责 1/3
          │           │           │
          └───────────┼───────────┘
                      ▼
                 完成同一个部件

核心：
拆一个工序内部的工作
```

---

## 19. 从 1D 并行走向真正的 3D Parallelism

到这里，我们已经有了三把不同的“手术刀”：

```text
DP
│
├── 切 Data
└── 提升训练吞吐量


PP
│
├── 切 Layer
└── 解决整个模型单卡放不下


TP
│
├── 切 Tensor
└── 解决单个 Layer 单卡放不下
```

但工业界真正训练数百亿、数千亿参数模型时，通常不会只选择其中一种。

因为它们解决的是三个不同维度的问题。

例如：

```text
TP = 8
PP = 4
DP = 16
```

意味着：

```text
每 8 张 GPU
      │
      └── 共同切一个 Layer
          Tensor Parallel

4 个 Pipeline Stage
      │
      └── 共同承载一个模型
          Pipeline Parallel

再复制 16 份这样的模型流水线
      │
      └── 处理不同数据
          Data Parallel
```

总 GPU 数量就是：

$$
8\times4\times16
=
512
$$

也就是说：

$$
\boxed{
N_{\text{GPU}}
=
N_{\text{TP}}
\times
N_{\text{PP}}
\times
N_{\text{DP}}
}
$$

这才是真正意义上的：

> **3D Parallelism**

它不是三个技术简单地并排放在一起，而是：

```text
                       3D Parallelism
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
         DP                  PP                  TP
      切 Data             切 Layer           切 Tensor
          │                   │                   │
          ▼                   ▼                   ▼
      扩展吞吐量          扩展模型深度         扩展单层宽度
```






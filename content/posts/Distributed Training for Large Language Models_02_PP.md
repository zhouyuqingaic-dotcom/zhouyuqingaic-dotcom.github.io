---
title: "Distributed Training for Large Language Models_02_PP"
date: 2026-08-26T10:00:00+08:00
draft: false
tags: ["Distributed Training", "DeepSpeed", "ZeRO","PP", "Megatron", "3D Parallelism", "LLM"]
categories: ["System & Architecture"]
showToc: true
TocOpen: true
math: true
---

# 1.2 PP (Pipeline Parallelism - 流水线并行)：数据流动，模型按层切分

上一节的 Data Parallelism 解决了一个问题：

> **模型放得进单张 GPU，但一张 GPU 算得太慢怎么办？**

DP 给出的答案是：

```text
复制模型
   +
切分数据
   +
All-Reduce 梯度
```

但当模型继续变大，一个更加根本的问题出现了：

> **如果模型本身已经无法完整装进一张 GPU 呢？**

例如，一个 70B 模型仅仅使用 BF16 保存参数，就需要大约：

$$
70\times10^9\times2
\approx
140\text{ GB}
$$

一张 80GB GPU 连参数本身都装不下。

这个时候继续增加 DP Rank 没有意义，因为纯 DP 要求：

```text
GPU 0 → 完整模型
GPU 1 → 完整模型
GPU 2 → 完整模型
...
```

于是工业界自然产生了一个新的想法：

> **既然一张 GPU 装不下整个模型，那就把模型拆开。**

而最直观的拆法，就是：

> **按照 Layer 切。**

这就是：

> **Pipeline Parallelism（PP，流水线并行）**

如果用一句话概括 PP：

> **数据不断流动，模型按照 Layer 分布在不同 GPU 上。**

---

## 1. PP 到底在切什么？

假设一个 Transformer 一共有 40 个 Layer。

如果整个模型无法放进一张 GPU，那么可以把它拆成：

```text
GPU 0 → Layer  1 ~ 10
GPU 1 → Layer 11 ~ 20
GPU 2 → Layer 21 ~ 30
GPU 3 → Layer 31 ~ 40
```

于是原来的模型：

```text
Input
  │
  ▼
Layer 1
  │
Layer 2
  │
Layer 3
  │
 ...
  │
Layer 40
  │
  ▼
Output
```

被物理映射成：

```text
Input
  │
  ▼
┌─────────────────────────┐
│ GPU 0                   │
│ Layer 1  ~ Layer 10     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ GPU 1                   │
│ Layer 11 ~ Layer 20     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ GPU 2                   │
│ Layer 21 ~ Layer 30     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ GPU 3                   │
│ Layer 31 ~ Layer 40     │
└────────────┬────────────┘
             │
             ▼
           Output
```

这里每一段模型通常称为：

> **Pipeline Stage**

例如：

```text
Stage 0 → GPU 0 → Layer  1 ~ 10
Stage 1 → GPU 1 → Layer 11 ~ 20
Stage 2 → GPU 2 → Layer 21 ~ 30
Stage 3 → GPU 3 → Layer 31 ~ 40
```

最重要的变化在于：

> **每张 GPU 不再保存完整模型，而只保存自己负责的那一部分 Layer。**

因此，PP 第一次真正实现了：

```text
多个 GPU 的显存
      ↓
共同承载一个模型
```

这与 DP 有本质区别。

---

## 2. 用“工厂造车”理解 PP

继续使用前面的汽车工厂比喻。

在 DP 中，每一个车间都拥有一套完整造车能力：

```text
车间 0：
底盘 → 发动机 → 车身 → 喷漆 → 完整汽车

车间 1：
底盘 → 发动机 → 车身 → 喷漆 → 完整汽车
```

所以 DP 相当于：

> **复制整条生产线，然后给每条生产线不同订单。**

而 PP 完全不一样。

如果完整生产线太巨大，一个车间根本装不下，那就把生产过程拆成多个阶段：

```text
车间 0
只负责底盘
   │
   ▼
车间 1
只负责发动机
   │
   ▼
车间 2
只负责车身
   │
   ▼
车间 3
只负责喷漆
```

汽车必须依次经过所有车间：

```text
订单
 │
 ▼
[底盘]
 │
 ▼
[发动机]
 │
 ▼
[车身]
 │
 ▼
[喷漆]
 │
 ▼
完整汽车
```

映射回神经网络：

```text
Input
 │
 ▼
GPU 0
Layer 1 ~ 10
 │
 │ Activation
 ▼
GPU 1
Layer 11 ~ 20
 │
 │ Activation
 ▼
GPU 2
Layer 21 ~ 30
 │
 │ Activation
 ▼
GPU 3
Layer 31 ~ 40
 │
 ▼
Loss
```

所以 PP 和 DP 最直观的区别就是：

```text
DP：
模型不动
数据分开跑

PP：
模型被拆开
数据依次流过模型
```

---

## 3. PP 的 Forward：Activation 向前流动

假设：

```text
GPU 0 → Layer 1  ~ 10
GPU 1 → Layer 11 ~ 20
GPU 2 → Layer 21 ~ 30
GPU 3 → Layer 31 ~ 40
```

首先，一个 Batch 被送入 GPU 0：

```text
Input
  │
  ▼
GPU 0
Layer 1 ~ 10
```

GPU 0 完成计算之后，并不会得到最终模型输出。

它只得到：

> **Layer 10 的 Activation。**

记作：

$$
h_{10}
$$

于是 GPU 0 把：

$$
h_{10}
$$

发送给 GPU 1。

GPU 1 将它作为 Layer 11 的输入：

$$
h_{20} = f_{11:20}(h_{10})
$$

然后再把 $h_{20}$ 发送给 GPU 2。

整个 Forward 可以表示成：

```text
Input
  │
  ▼
┌────────────────────┐
│ GPU 0              │
│ Layer 1 ~ 10       │
└─────────┬──────────┘
          │
          │ h₁₀
          │ Activation
          ▼
┌────────────────────┐
│ GPU 1              │
│ Layer 11 ~ 20      │
└─────────┬──────────┘
          │
          │ h₂₀
          │ Activation
          ▼
┌────────────────────┐
│ GPU 2              │
│ Layer 21 ~ 30      │
└─────────┬──────────┘
          │
          │ h₃₀
          │ Activation
          ▼
┌────────────────────┐
│ GPU 3              │
│ Layer 31 ~ 40      │
└─────────┬──────────┘
          │
          ▼
        Output
          │
          ▼
         Loss
```

因此：

> **Forward 阶段真正跨 GPU 传递的不是模型参数，而是 Activation。**

这一点非常重要。

---

## 4. PP 的 Backward：Activation Gradient 反向流动

Forward 最终在最后一个 Stage 上得到 Loss。

接下来开始反向传播。

反向传播的方向正好相反：

```text
Forward:

GPU 0
  ↓
GPU 1
  ↓
GPU 2
  ↓
GPU 3


Backward:

GPU 0
  ↑
GPU 1
  ↑
GPU 2
  ↑
GPU 3
```

例如 GPU 3 完成自己负责 Layer 的反向传播之后，会计算出：

> **对于输入 Activation 的梯度。**

记作：

$$
\frac{\partial L}{\partial h_{30}}
$$

然后把它发送回 GPU 2：

```text
             Backward

┌────────────────────┐
│ GPU 0              │
│ Layer 1 ~ 10       │
└─────────▲──────────┘
          │
          │ ∂L/∂h₁₀
          │
┌─────────┴──────────┐
│ GPU 1              │
│ Layer 11 ~ 20      │
└─────────▲──────────┘
          │
          │ ∂L/∂h₂₀
          │
┌─────────┴──────────┐
│ GPU 2              │
│ Layer 21 ~ 30      │
└─────────▲──────────┘
          │
          │ ∂L/∂h₃₀
          │
┌─────────┴──────────┐
│ GPU 3              │
│ Layer 31 ~ 40      │
└────────────────────┘
```

于是 PP 的完整数据流实际上是：

```text
Forward:

Input
  ↓
Stage 0
  ↓ Activation
Stage 1
  ↓ Activation
Stage 2
  ↓ Activation
Stage 3
  ↓
Loss


Backward:

Loss
  ↓
Stage 3
  ↑ Activation Gradient
Stage 2
  ↑ Activation Gradient
Stage 1
  ↑ Activation Gradient
Stage 0
```

所以 Pipeline Stage 之间最主要的通信内容可以概括为：

| 阶段       | 发送内容                |
| -------- | ------------------- |
| Forward  | Activation          |
| Backward | Activation Gradient |

而不是像 DP 那样主要同步完整模型的 Gradient。

---

## 5. 一个严重的问题：Pipeline Bubble

到这里 PP 看起来非常完美。

模型放不进一张卡？

那就拆成 4 段：

```text
GPU 0 → 1/4 Model
GPU 1 → 1/4 Model
GPU 2 → 1/4 Model
GPU 3 → 1/4 Model
```

显存问题解决了。

但是一个新的问题马上出现。

假设我们一次直接把整个 Batch 作为一个整体送入流水线。

最开始：

```text
时间 ───────────────────────────────►

GPU 0: [ Forward ]
GPU 1: [   Idle   ]
GPU 2: [   Idle   ]
GPU 3: [   Idle   ]
```

因为 GPU 1 必须等待 GPU 0 算完之后才能获得 Activation。

接下来：

```text
时间 ───────────────────────────────►

GPU 0: [ Forward ][   Idle   ]
GPU 1: [   Idle   ][ Forward ]
GPU 2: [   Idle   ][   Idle   ]
GPU 3: [   Idle   ][   Idle   ]
```

再往后：

```text
GPU 0: [ F ][   ][   ][   ]
GPU 1: [   ][ F ][   ][   ]
GPU 2: [   ][   ][ F ][   ]
GPU 3: [   ][   ][   ][ F ]
```

大量 GPU 在大量时间内都处于：

> **Idle**

也就是“干等”。

这部分由于流水线没有被填满而浪费掉的时间，被称为：

> **Pipeline Bubble（流水线气泡）**

---

## 6. 为什么会产生 Bubble？

原因非常简单：

> **后面的 Stage 必须等待前面的 Stage 产生 Activation。**

例如：

```text
GPU 3 想计算
     │
     ▼
必须先等 GPU 2
     │
     ▼
GPU 2 必须先等 GPU 1
     │
     ▼
GPU 1 必须先等 GPU 0
```

所以流水线刚启动的时候：

```text
Stage 0 → 工作
Stage 1 → 等待
Stage 2 → 等待
Stage 3 → 等待
```

等数据终于流到 Stage 3，流水线才算完全“填满”。

Backward 结束时同样会经历一个逐渐排空的过程。

因此 PP 天生存在：

```text
Pipeline Fill
     +
Steady State
     +
Pipeline Drain
```

其中 Fill 和 Drain 阶段就会产生 Bubble。

---

## 7. 解决 Bubble 的第一步：Micro-batching

既然一个完整 Batch 太“大”，导致 Pipeline 很难填满，那么最自然的解决办法就是：

> **把一个 Batch 再切成很多 Micro-batch。**

例如：

```text
Global / Training Batch
        │
        ▼
┌───────┬───────┬───────┬───────┐
│ MB 0  │ MB 1  │ MB 2  │ MB 3  │ ...
└───────┴───────┴───────┴───────┘
```

然后让这些 Micro-batch 连续进入 Pipeline。

假设有 4 个 Pipeline Stage：

```text
S0 → S1 → S2 → S3
```

那么调度过程可以变成：

```text
Time ───────────────────────────────────────►

Stage 0:  F0   F1   F2   F3   F4   F5
Stage 1:       F0   F1   F2   F3   F4   F5
Stage 2:            F0   F1   F2   F3   F4   F5
Stage 3:                 F0   F1   F2   F3   F4   F5
```

其中：

```text
F0 = Micro-batch 0 的 Forward
F1 = Micro-batch 1 的 Forward
F2 = Micro-batch 2 的 Forward
...
```

现在观察 GPU 0。

当它把 MB0 交给 GPU 1 之后：

> **它不需要等待 MB0 走完整个模型。**

而是马上开始计算：

```text
MB1
```

与此同时：

```text
GPU 0 → MB1
GPU 1 → MB0
```

下一时刻：

```text
GPU 0 → MB2
GPU 1 → MB1
GPU 2 → MB0
```

再下一时刻：

```text
GPU 0 → MB3
GPU 1 → MB2
GPU 2 → MB1
GPU 3 → MB0
```

于是流水线逐渐被填满：

```text
        MB3    MB2    MB1    MB0
         │      │      │      │
         ▼      ▼      ▼      ▼
       GPU 0  GPU 1  GPU 2  GPU 3
```

这才真正像一条工业流水线。

---

## 8. Micro-batch 越多，Bubble 越小

假设：

* Pipeline Stage 数量为 $p$
* Micro-batch 数量为 $m$

在一个简化的 Pipeline Forward 调度中，完成 $m$ 个 Micro-batch 需要大约：

$$
m+p-1
$$

个时间槽。

其中额外的：

$$
p-1
$$

就是 Pipeline Fill 带来的 Bubble。

因此可以近似理解：

$$
\text{Bubble Fraction}
\approx
\frac{p-1}{m+p-1}
$$

例如：

```text
Pipeline Stages = 4
Micro-batches   = 4
```

那么：

$$
\frac{4-1}{4+4-1}
=
\frac{3}{7}
$$

Bubble 非常明显。

如果增加到：

```text
Pipeline Stages = 4
Micro-batches   = 32
```

那么：

$$
\frac{3}{35}
\approx
8.6%
$$

可以看到：

> **Micro-batch 数量越多，Pipeline Bubble 占整个训练时间的比例通常越低。**

直觉上很好理解。

假设一家工厂启动一次生产线需要固定 3 分钟：

```text
启动成本 = 3 min
```

如果只生产 4 辆汽车，这 3 分钟非常显眼。

但如果连续生产 1000 辆汽车，那么 3 分钟启动成本几乎可以忽略。

Pipeline Parallelism 也是同样的道理。

---

## 9. 仅仅连续 Forward 还不够

不过问题还没有彻底解决。

假设我们简单地：

> **先让所有 Micro-batch 全部 Forward，再全部 Backward。**

这类思路可以抽象成：

```text
F0 F1 F2 F3 F4 ...
             ↓
B4 B3 B2 B1 B0 ...
```

那么每一个 Stage 在 Forward 时产生的 Activation 都必须一直保存：

```text
MB0 Activation
MB1 Activation
MB2 Activation
MB3 Activation
MB4 Activation
...
```

直到对应 Micro-batch 的 Backward 到来。

结果就是：

> **Activation Memory 会不断堆积。**

这对于大模型而言同样非常危险。

因此，更先进的 Pipeline Schedule 会尝试：

> **尽早执行 Backward，释放已经不再需要的 Activation。**

这就引出了一个非常经典的调度策略：

> **1F1B**

---

## 10. 1F1B：一个 Forward 紧接一个 Backward

1F1B 全称可以理解为：

> **One Forward, One Backward**

也就是当 Pipeline 进入稳定状态之后，每个 Stage 尽量交替执行：

```text
1 个 Forward
      ↓
1 个 Backward
      ↓
1 个 Forward
      ↓
1 个 Backward
```

而不是：

```text
Forward
Forward
Forward
Forward
Forward
...
Backward
Backward
Backward
Backward
...
```

简化来看：

```text
Time ─────────────────────────────────────────►

Stage 0: F0  F1  F2  F3  B0  F4  B1  F5  B2 ...
Stage 1:     F0  F1  F2  B0  F3  B1  F4  B2 ...
Stage 2:         F0  F1  B0  F2  B1  F3  B2 ...
Stage 3:             F0  B0  F1  B1  F2  B2 ...
```

实际调度会比这张简图更加精细，但核心思想非常简单：

> **不要让 Forward 无限向前堆积，而是尽早插入 Backward。**


> **注意：1F1B 中的 Backward 并不意味着立刻更新参数。**

> 某个 Micro-batch 完成 Backward 后，通常只是：
>
> ```text
> 计算 Gradient
>      ↓
> 累积到 Gradient Buffer
>      ↓
> 释放该 Micro-batch 的 Activation
> ```
>
> 当前 Training Step 中的所有 Micro-batch 都完成 Forward / Backward 后，才会统一执行：
>
> ```text
> Gradient Accumulation
>        ↓
> （如有 DP）Gradient Synchronization
>        ↓
> Optimizer Step
>        ↓
> Wₜ → Wₜ₊₁
> ```
>
> 因此，**1F1B 交错的是 Forward 和 Backward，而不是 Forward 和参数更新。**




这样做有两个重要好处。
---

### 好处一：减少 Activation Memory

某个 Micro-batch 完成 Backward 后，对应 Activation 就可以释放。

因此显存中同时存活的 Activation 数量可以显著减少。

可以把它想象成：

```text
纯 Forward 堆积：

F0 → 保存 A0
F1 → 保存 A1
F2 → 保存 A2
F3 → 保存 A3
F4 → 保存 A4

显存：
[A0][A1][A2][A3][A4]...


1F1B：

F0 → 保存 A0
...
B0 → 使用 A0 → 释放 A0

F1 → 保存 A1
...
B1 → 使用 A1 → 释放 A1
```

所以：

> **1F1B 不仅是性能优化，也是显存优化。**

---

### 好处二：让 Forward 与 Backward 在 Pipeline 中交错

进入稳定状态之后，不同 GPU 可能同时进行完全不同的工作：

```text
GPU 0 → Forward MB5
GPU 1 → Backward MB2
GPU 2 → Forward MB4
GPU 3 → Backward MB3
```

整个集群会更加接近持续工作的流水线状态。

这也是 Megatron 风格 Pipeline Parallelism 中非常重要的调度思想。

---

## 11. PP 的完整训练过程

现在我们可以把 PP 的逻辑完整串起来。

首先模型被切成多个 Stage：

```text
Model

Layer 1
Layer 2
...
Layer 40

        │ Partition
        ▼

Stage 0 → Layer  1 ~ 10
Stage 1 → Layer 11 ~ 20
Stage 2 → Layer 21 ~ 30
Stage 3 → Layer 31 ~ 40
```

然后 Batch 被切成多个 Micro-batch：

```text
Batch
  │
  ▼
MB0 MB1 MB2 MB3 MB4 ...
```

Forward 时：

```text
MB0:

Stage 0
   │ Activation
   ▼
Stage 1
   │ Activation
   ▼
Stage 2
   │ Activation
   ▼
Stage 3
```

Backward 时：

```text
Stage 0
   ▲ Activation Gradient
   │
Stage 1
   ▲ Activation Gradient
   │
Stage 2
   ▲ Activation Gradient
   │
Stage 3
```

多个 Micro-batch 再以流水线方式交错执行：

```text
             Pipeline

MB3 ──► Stage 0
MB2 ─────────► Stage 1
MB1 ───────────────► Stage 2
MB0 ─────────────────────► Stage 3
```

于是 PP 最核心的结构就是：

```text
模型切成 Stage
      +
Batch 切成 Micro-batch
      +
Stage 间传递 Activation
      +
Pipeline Schedule
```

---

## 12. PP 的通信：为什么通常只需要和“邻居”说话？

PP 有一个非常重要的通信特点：

> **每个 Stage 通常只需要与前一个 Stage 和后一个 Stage 通信。**

例如：

```text
GPU 0 ⇄ GPU 1 ⇄ GPU 2 ⇄ GPU 3
```

GPU 0 不需要直接把 Activation 发给 GPU 3。

它只需要发送给：

```text
GPU 1
```

GPU 1 再计算新的 Activation，继续发送给 GPU 2。

所以从通信拓扑来看：

```text
DP：

GPU 0 ─┐
GPU 1 ─┼── Collective Communication
GPU 2 ─┤
GPU 3 ─┘


PP：

GPU 0 ⇄ GPU 1 ⇄ GPU 2 ⇄ GPU 3

Point-to-Point
```

Forward 主要发送：

> **Activation**

Backward 主要发送：

> **Activation Gradient**

因此 PP 的主要通信是：

> **Point-to-Point Communication**

而不是 DP 那种：

> **Collective Communication**

---

## 13. PP 的带宽要求真的很低吗？

这里需要特别注意一个容易产生误解的说法：

> “PP 只传 Activation，所以通信量很小。”

这并不总是正确。

Activation Tensor 的大小可能与：

* Micro Batch Size
* Sequence Length
* Hidden Size

直接相关。

粗略来说，其规模与：

$$
B_{\text{micro}}
\times
S
\times
H
$$

处于同一数量级。

其中：

* $B_{\text{micro}}$：Micro Batch Size
* $S$：Sequence Length
* $H$：Hidden Size

在超长上下文、大 Hidden Size 场景下，Activation 本身同样可能非常大。

所以更准确的描述应该是：

> **PP 并不是“通信量一定很小”，而是通信主要发生在相邻 Stage 之间，而且不像 TP 那样需要在很多 Layer 内部反复执行 Collective Communication。**

因此：

```text
PP
↓
通信频率相对较低
↓
Point-to-Point
↓
对拓扑要求相对宽松
↓
更适合跨 Node
```

而：

```text
TP
↓
Layer 内频繁通信
↓
Collective Communication
↓
对延迟 / 带宽极其敏感
↓
通常优先部署在 NVLink / NVSwitch 域内
```

这也是后面理解 3D Parallelism 物理拓扑的重要基础。

---

## 14. PP 最大的性能敌人：Stage 不平衡

除了 Bubble 之外，PP 还有另外一个非常重要的问题：

> **不同 Pipeline Stage 的计算量必须尽量均衡。**

假设：

```text
GPU 0 → 10 ms
GPU 1 → 10 ms
GPU 2 → 10 ms
GPU 3 → 30 ms
```

那么即使前三个 Stage 很快：

```text
GPU 0 ██████████
GPU 1 ██████████
GPU 2 ██████████
GPU 3 ██████████████████████████████
```

整个 Pipeline 最终仍然会被 GPU 3 拖住。

因为下一轮数据必须等待这个 Stage。

这就像一条工厂流水线：

```text
Stage 0 → 每分钟生产 10 个
Stage 1 → 每分钟生产 10 个
Stage 2 → 每分钟生产 10 个
Stage 3 → 每分钟生产  3 个
```

整条流水线最终吞吐量不可能达到每分钟 10 个。

真正决定性能的是：

> **最慢的那个 Stage。**

因此 Pipeline Partition 并不能简单地理解成：

```text
40 Layer / 4 GPU
=
每卡 10 Layer
```

因为不同 Layer 的：

* Parameter Size
* FLOPs
* Activation Size
* Attention Cost

可能并不完全相同。

真正工程化的 Pipeline Parallelism 通常需要尽可能做到：

> **Load Balance（负载均衡）**

也就是让不同 Stage 的执行时间尽可能接近。

---

## 15. 为什么有时一个 GPU 不只负责一段连续 Layer？

如果 Pipeline Stage 数量继续增加，Bubble 会越来越明显。

一种进一步优化的方法是：

> **Interleaved Pipeline Parallelism**

也可以理解成：

> **Virtual Pipeline Stage**

例如原本：

```text
GPU 0 → Layer  1 ~ 10
GPU 1 → Layer 11 ~ 20
GPU 2 → Layer 21 ~ 30
GPU 3 → Layer 31 ~ 40
```

可以进一步拆得更细，让一张 GPU 在逻辑上承担多个较小的 Virtual Stage。

例如：

```text
GPU 0 → Chunk 0 + Chunk 4
GPU 1 → Chunk 1 + Chunk 5
GPU 2 → Chunk 2 + Chunk 6
GPU 3 → Chunk 3 + Chunk 7
```

通过更加细粒度地交错执行不同模型 Chunk，可以进一步改善 Pipeline 调度效率。

这也是现代超大模型训练系统经常使用的优化手段之一。

这里暂时不用深入具体调度算法，只需要记住：

> **Pipeline 切得越细，调度空间越大，但通信和调度复杂度也会随之上升。**

---

## 16. PP 到底解决了什么？

现在重新看 PP。

Data Parallelism 的问题是：

```text
GPU 0 → 100% Model
GPU 1 → 100% Model
GPU 2 → 100% Model
GPU 3 → 100% Model
```

所以无论增加多少 GPU：

> **单张 GPU 仍然必须装下整个模型。**

而 Pipeline Parallelism 将其变成：

```text
GPU 0 → 25% Model
GPU 1 → 25% Model
GPU 2 → 25% Model
GPU 3 → 25% Model
```

因此 PP 真正解决的是：

> **模型整体无法放入单张 GPU 的问题。**

例如一个模型理论上需要：

```text
200GB Model Parameters
```

单张：

```text
80GB GPU
```

显然放不下。

但如果合理切到 4 张 GPU：

```text
GPU 0 → ~50GB
GPU 1 → ~50GB
GPU 2 → ~50GB
GPU 3 → ~50GB
```

模型就有可能被共同承载。

所以：

> **PP 第一次真正把多张 GPU 的显存联合起来承载同一个模型。**

---

## 17. 但 PP 并不是免费的午餐

PP 虽然解决了模型容量问题，却引入了一系列新的成本：

| 问题                    | 原因                                    |
| --------------------- | ------------------------------------- |
| **Pipeline Bubble**   | Stage 必须等待前后依赖                        |
| **Activation 通信**     | Stage 之间需要发送中间结果                      |
| **Activation Memory** | Forward 结果需要保留给 Backward              |
| **Load Imbalance**    | 最慢 Stage 决定整个 Pipeline 吞吐量            |
| **复杂调度**              | Micro-batch、1F1B、Interleaving 等调度更加复杂 |

因此：

> **PP 本质是在用更加复杂的执行调度和 Stage 间通信，换取模型参数的跨 GPU 切分。**

这是一种典型的：

```text
显存容量 ↑
     ↕
调度复杂度 ↑
```

的工程权衡。

---

## 18. PP 和 DP 的本质区别

现在可以把 DP 与 PP 放在一起比较：

|                | DP               | PP                               |
| -------------- | ---------------- | -------------------------------- |
| 全称             | Data Parallelism | Pipeline Parallelism             |
| 切分对象           | Data             | Model Layers                     |
| 每张 GPU 是否有完整模型 | **是**            | **否**                            |
| 每张 GPU 处理的数据   | 不同               | 同一个 Micro-batch 的不同阶段            |
| 主要通信           | Gradient         | Activation / Activation Gradient |
| 通信模式           | Collective       | Point-to-Point                   |
| 核心目标           | 提升吞吐量            | 让大模型跨 GPU 存储                     |
| 主要问题           | 参数/状态重复          | Pipeline Bubble                  |

用工厂比喻：

```text
DP：

车间 0 → 完整生产汽车 A
车间 1 → 完整生产汽车 B
车间 2 → 完整生产汽车 C
车间 3 → 完整生产汽车 D


PP：

车间 0 → 底盘
           ↓
车间 1 → 发动机
           ↓
车间 2 → 车身
           ↓
车间 3 → 喷漆
```

所以可以记成：

> **DP 是复制生产线，PP 是拆分生产线。**

---

## 19. PP 的极限：如果一个 Layer 自己都装不下呢？

到这里，看起来 PP 已经可以不断增加 GPU：

```text
模型太大
   ↓
增加 Pipeline Stage
   ↓
每张 GPU 少放一些 Layer
```

但继续思考会发现一个新的物理极限。

PP 切分模型的最基本单位仍然是：

> **Layer。**

假设某一个 Transformer Layer 本身就已经非常巨大：

```text
                Transformer Layer
                       │
        ┌──────────────┴──────────────┐
        │                             │
   Attention                       MLP
        │                             │
        ▼                             ▼
巨大 Weight Matrix              巨大 Weight Matrix
```

甚至出现：

> **单独一个 Layer 就已经无法装入单张 GPU。**

那么：

```text
GPU 0 → Layer 1
```

本身就会：

```text
OOM
```

此时 PP 已经无法继续切了。

因为它只能做到：

```text
Layer 1 | Layer 2 | Layer 3 | Layer 4
    ↑
按 Layer 边界切
```

但我们现在真正需要的是：

```text
           一个 Layer
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
     GPU 0   GPU 1   GPU 2
```

也就是：

> **把同一个 Layer 内部的 Tensor / Matrix 再继续切开。**

这便进入了另外一个完全不同的并行维度：

> **Tensor Parallelism（TP，张量并行）**

因此从 DP 到 PP，再到 TP，实际上是一层一层向模型内部深入：

```text
DP
│
└── 不切模型
    只切 Data
        │
        ▼
PP
│
└── 切 Model
    按 Layer 切
        │
        ▼
TP
│
└── 继续深入 Layer
    切 Tensor / Matrix
```

可以把这个演化过程浓缩成三个问题：

```text
模型放得下，只是训练太慢？
        │
        └────► DP：切 Data

整个模型放不进一张 GPU？
        │
        └────► PP：切 Layer

连一个 Layer 都放不进一张 GPU？
        │
        └────► TP：切 Tensor
```

接下来，我们就继续把“手术刀”深入 Transformer Layer 内部，看看大模型训练中最核心、同时也是对 GPU 高速互联要求最高的并行技术之一：








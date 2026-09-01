---
title: "Distributed Training for Large Language Models_01_DP"
date: 2026-08-26T10:00:00+08:00
draft: false
tags: ["Distributed Training", "DeepSpeed", "ZeRO", "DP","Megatron", "3D Parallelism", "LLM"]
categories: ["System & Architecture"]
showToc: true
TocOpen: true
math: true
---

> **导读**
>
> 随着大语言模型（LLM）和多模态框架参数量呈指数级增长，单机单卡的物理显存早已无法承载千亿参数的
> 训练任务。为了打破“显存墙”与“通信墙”，工业界演化出了极其精妙的分布式训练架构。
>
> 本系列将从底层逻辑出发，用最直观的“工厂造车”比喻，剥丝抽茧地拆解大模型分布式训练的核心技术：
> 从 3D 并行（DP、PP、TP）的基础概念，到微软颠覆性的 ZeRO 显存魔术（ZeRO-1/2/3），
> 再到 DeepSpeed 与 Megatron 两大框架的工程哲学与生态位差异。

# 1. 3D 并行基石：如何把模型和数据“大卸八块”？

在大模型训练中，DP、PP、TP 被称为 **3D Parallelism（3D 并行）**的三个基础维度。

其中最经典、也最容易理解的一种并行方式，就是 **Data Parallelism（DP，数据并行）**。

# 1.1 DP (Data Parallelism - 数据并行)：模型不动，切分数据

如果只用一句话概括 DP，可以记住：

> **模型不动，切分数据。**

但如果只理解到这一层，其实还没有真正理解 Data Parallelism。

因为 DP 真正需要解决的核心问题并不是：

> “怎么把一个 Batch 分给多张 GPU？”

而是：

> **不同 GPU 明明处理的是不同数据，计算出来的梯度也不同，为什么最后还能训练出完全相同的模型？**

答案就是：

> **All-Reduce。**

因此，一个更完整的定义应该是：

> **DP = 完整模型副本 + 数据切分 + 梯度 All-Reduce。**

接下来，我们从最简单的数据切分开始，一步一步走到 DP 真正的核心机制——**All-Reduce**。

---

## 1. DP 到底在并行什么？

DP 并行的不是模型，而是：

> **训练数据。**

假设现在有 8 张 GPU。

在纯 Data Parallelism 中，每一张 GPU 都保存一份 **100% 完整且完全相同的模型参数**：

```text
GPU 0 → [完整模型 W]
GPU 1 → [完整模型 W]
GPU 2 → [完整模型 W]
GPU 3 → [完整模型 W]
...
GPU 7 → [完整模型 W]
```

真正被切开的，是训练数据。

比如：

```text
Global Batch Size = 32
Data Parallel Size = 8
```

那么这 32 个训练样本可以平均分给 8 张 GPU：

```text
GPU 0 → 样本  1 ~  4
GPU 1 → 样本  5 ~  8
GPU 2 → 样本  9 ~ 12
GPU 3 → 样本 13 ~ 16
...
GPU 7 → 样本 29 ~ 32
```

因此，在不考虑 Gradient Accumulation 的情况下：

$$
B_{\text{local}} = \frac{B_{\text{global}}}{N_{\text{DP}}}
$$

代入：

$$
B_{\text{local}}
=
\frac{B_{\text{global}}}{N_{\text{DP}}}
=
\frac{32}{8}
=
4
$$

也就是说，每张 GPU 只需要处理 4 个样本。

---

### 用“工厂造车”理解 DP

可以把这个过程想象成一家汽车工厂。

假设现在有 8 个完全相同的汽车生产车间：

```text
GPU 0 → 车间 0
GPU 1 → 车间 1
GPU 2 → 车间 2
...
GPU 7 → 车间 7
```

每个车间里面，都保存着一份：

> **完整且完全相同的造车图纸。**

这份“图纸”就是模型参数。

现在工厂一次收到了 32 个汽车订单。

如果只让一个车间生产全部 32 辆汽车，速度自然很慢。

于是工厂把订单平均分给 8 个车间：

```text
车间 0 → 生产 4 辆
车间 1 → 生产 4 辆
车间 2 → 生产 4 辆
...
车间 7 → 生产 4 辆
```

由于 8 个车间可以同时工作，因此整体吞吐量就得到了提升。

所以 DP 最直观的优势就是：

> **用更多 GPU 同时处理更多数据。**

但这里马上会出现一个问题。

---

## 2. 为什么仅仅切数据还不够？

假设训练开始时，所有 GPU 上的模型参数都是：

$$
W
$$

也就是：

```text
GPU 0 → W
GPU 1 → W
GPU 2 → W
...
GPU 7 → W
```

虽然它们的模型完全一样，但是处理的数据并不一样。

因此经过 Forward 和 Backward 之后，每张 GPU 得到的梯度也必然不同：

```text
GPU 0 → g₀
GPU 1 → g₁
GPU 2 → g₂
GPU 3 → g₃
...
GPU 7 → g₇
```

这是非常自然的。

因为：

```text
不同的数据
    ↓
不同的 Loss
    ↓
不同的 Gradient
```

那么问题来了。

假如每张 GPU 直接使用自己算出来的局部梯度更新参数：

$$
W_0'
=
W-\eta g_0
$$

$$
W_1'
=
W-\eta g_1
$$

$$
W_2'
=
W-\eta g_2
$$

那么由于：

$$
g_0 \neq g_1 \neq g_2 \neq \cdots
$$

自然会得到：

$$
W_0'
\neq
W_1'
\neq
W_2'
\neq
\cdots
$$

也就是说：

```text
训练开始：

GPU 0 → W
GPU 1 → W
GPU 2 → W
GPU 3 → W

        ↓ 各自处理不同数据

GPU 0 → g₀
GPU 1 → g₁
GPU 2 → g₂
GPU 3 → g₃

        ↓ 如果各自更新

GPU 0 → W₀'
GPU 1 → W₁'
GPU 2 → W₂'
GPU 3 → W₃'

结果：

W₀' ≠ W₁' ≠ W₂' ≠ W₃'
```

只需要进行一步训练，原本完全相同的多个模型副本就会开始分叉。

这样继续训练下去，相当于：

> **8 张 GPU 分别训练出了 8 个不同的模型。**

这显然不是我们想要的 Data Parallelism。

因此 DP 必须解决一个核心问题：

> **如何让处理不同数据的 GPU，在参数更新之前重新获得完全相同的梯度？**

于是，DP 中最重要的 Collective Communication 操作出现了：

> **All-Reduce。**

---

## 3. DP 的灵魂：All-Reduce 梯度同步

All-Reduce 的作用可以用一句话概括：

> **把所有 GPU 的局部梯度聚合成一个全局梯度，然后再让所有 GPU 都拿到这个相同的结果。**

假设 4 张 GPU 分别得到了：

```text
GPU 0 → g₀
GPU 1 → g₁
GPU 2 → g₂
GPU 3 → g₃
```

经过 All-Reduce：

```text
GPU 0: g₀ ──┐
GPU 1: g₁ ──┤
GPU 2: g₂ ──┼────► All-Reduce ────► g
GPU 3: g₃ ──┘
```

其中：

$$
g
=
\frac{1}{N}
\sum_{i=0}^{N-1}g_i
$$

如果 $N=4$：

$$
g
=
\frac{g_0+g_1+g_2+g_3}{4}
$$

于是 All-Reduce 完成之后：

```text
GPU 0 → g
GPU 1 → g
GPU 2 → g
GPU 3 → g
```

现在所有 GPU 都重新拿到了完全相同的梯度。

然后它们再分别执行：

$$
W_{t+1}
=
W_t-\eta g
$$

由于：

* 初始参数一样；
* 梯度一样；
* Optimizer 一样；
* Learning Rate 一样；

所以更新结果自然也一样：

```text
GPU 0 → W'
GPU 1 → W'
GPU 2 → W'
GPU 3 → W'
```

于是：

$$
W_0' = W_1' = W_2' = W_3'
$$

这就是同步 Data Parallelism 能成立的根本原因。

因此可以把 DP 最核心的逻辑总结为：

> **数据可以不同，局部梯度可以不同，但在参数更新之前，全局梯度必须重新变得一致。**

而完成这件事情的，就是：

> **All-Reduce。**

---

## 4. All-Reduce 到底是怎么工作的？

从逻辑上看，All-Reduce 非常简单：

```text
局部梯度
    │
    ▼
全局聚合
    │
    ▼
结果发给所有 GPU
```

但如果模型有数十亿甚至数百亿参数，就不可能简单粗暴地：

> “所有 GPU 把完整梯度全部发给某一张卡，然后再由它统一广播。”

因为这样很容易形成严重的通信热点。

工业界通常会使用更加高效的 Collective Communication 算法。

其中最经典的一种就是：

> **Ring All-Reduce（环形 All-Reduce）**

假设现在有 4 张 GPU：

```text
        ┌───────────────┐
        │               ▼
     GPU 0            GPU 1
        ▲               │
        │               ▼
     GPU 3 ◄───────── GPU 2
```

每张 GPU 只需要主要和相邻 GPU 交换数据。

Ring All-Reduce 从逻辑上可以拆成两个阶段：

$$
\boxed{\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}}
$$


也就是：

```text
              All-Reduce
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Reduce-Scatter         All-Gather
  Reduce + 分片          收集 + 拼齐
```

这两个操作非常重要。

不仅 DP 会用到它们，后面的 ZeRO 也会围绕它们进行重新设计。

---

## 5. 第一阶段：Reduce-Scatter

假设每张 GPU 都计算出了一份完整的局部梯度。

为了方便理解，我们把完整梯度切成 4 个 Chunk：

```text
GPU 0: [A₀] [B₀] [C₀] [D₀]
GPU 1: [A₁] [B₁] [C₁] [D₁]
GPU 2: [A₂] [B₂] [C₂] [D₂]
GPU 3: [A₃] [B₃] [C₃] [D₃]
```

其中：

```text
A₀
A₁
A₂
A₃
```

代表不同 GPU 对**同一段模型参数**计算出来的局部梯度。

同理：

```text
B₀ B₁ B₂ B₃
```

对应另一段参数。

最终我们需要计算：

```text
A = A₀ + A₁ + A₂ + A₃

B = B₀ + B₁ + B₂ + B₃

C = C₀ + C₁ + C₂ + C₃

D = D₀ + D₁ + D₂ + D₃
```

如果最终需要平均值，再除以参与 DP 的 GPU 数量即可。

---

### Reduce：先完成梯度规约

所谓 Reduce，本质就是：

> **把不同 GPU 上对应位置的数据进行 Sum。**

例如：

```text
A₀ ──┐
A₁ ──┤
A₂ ──┼────► SUM ────► A
A₃ ──┘
```

其中：

$$
A
=
A_0+A_1+A_2+A_3
$$

B、C、D 同理。

但 Reduce-Scatter 并不是把最终得到的：

```text
[A] [B] [C] [D]
```

全部交给每一张 GPU。

它还有第二个动作：

> **Scatter。**

---

### Scatter：最终结果分散保存

Reduce-Scatter 完成之后，结果可能变成：

```text
GPU 0 → [A]
GPU 1 → [B]
GPU 2 → [C]
GPU 3 → [D]
```

需要注意：

这里的 `[A]` 已经不是 GPU 0 自己的局部梯度 `A₀`。

而是：

$$
A=A_0+A_1+A_2+A_3
$$

也就是说：

```text
GPU 0 → [A]   ← 已经完成全局规约
GPU 1 → [B]   ← 已经完成全局规约
GPU 2 → [C]   ← 已经完成全局规约
GPU 3 → [D]   ← 已经完成全局规约
```

此时所有 Gradient Chunk 实际上都已经计算正确。

但是：

> **没有任何一张 GPU 拥有完整的 `[A B C D]`。**

每张 GPU 只拥有整个最终结果的一部分。

这就是：

```text
Reduce
  │
  └── 不同 GPU 的对应数据进行 Sum

Scatter
  │
  └── 最终结果分散到不同 GPU
```

所以叫：

> **Reduce-Scatter**

可以把它想象成 4 个人一起完成一幅拼图：

```text
                  Reduce-Scatter

GPU 0 ──┐
GPU 1 ──┼────► 全局计算 ────► GPU 0 持有 [A]
GPU 2 ──┤                     GPU 1 持有 [B]
GPU 3 ──┘                     GPU 2 持有 [C]
                             GPU 3 持有 [D]
```

大家共同完成了整幅拼图的计算。

但每个人手里只保留了其中一块。

---

### Ring 中的数据是怎么流动的？

在 Ring All-Reduce 中，GPU 被组织成一个逻辑环：

```text
        ┌───────────────┐
        │               ▼
     GPU 0            GPU 1
        ▲               │
        │               ▼
     GPU 3 ◄───────── GPU 2
```

梯度 Chunk 会不断沿着环传递。

每张 GPU：

1. 向下一张 GPU 发送一个 Chunk；
2. 从上一张 GPU 接收一个 Chunk；
3. 将收到的数据与本地对应结果执行 Sum；
4. 再继续向后传递。

可以把这个过程想象成：

> **一边传递，一边做加法。**

经过若干轮之后，对应的 Chunk 就完成了所有 GPU 数据的规约。

最终：

```text
GPU 0 → [A]
GPU 1 → [B]
GPU 2 → [C]
GPU 3 → [D]
```

此时第一阶段完成。

---

## 6. 第二阶段：All-Gather

现在每张 GPU 都拥有一块已经计算完成的最终梯度：

```text
GPU 0 → [A]
GPU 1 → [B]
GPU 2 → [C]
GPU 3 → [D]
```

但标准 Data Parallelism 接下来需要每张 GPU 都执行相同的参数更新。

因此所有 GPU 最终都必须获得：

```text
[A] [B] [C] [D]
```

于是进入第二阶段：

> **All-Gather**

All-Gather 的核心动作是：

> **把已经计算完成的分片互相交换，最终让每一个参与者都收集齐所有分片。**

开始时：

```text
GPU 0 → [A]
GPU 1 → [B]
GPU 2 → [C]
GPU 3 → [D]
```

随着 Chunk 在 Ring 中继续传递：

```text
        ┌───────────────┐
        │               ▼
     GPU 0            GPU 1
      [A]              [B]
        ▲               │
        │               ▼
     GPU 3 ◄───────── GPU 2
      [D]              [C]
```

每张 GPU 一边发送自己已经拥有的 Chunk，一边接收其他 GPU 的 Chunk。

经过若干轮之后：

```text
GPU 0 → [A] [B] [C] [D]

GPU 1 → [A] [B] [C] [D]

GPU 2 → [A] [B] [C] [D]

GPU 3 → [A] [B] [C] [D]
```

于是所有 GPU 最终都拥有了：

> **完整且完全一致的全局梯度。**

整个 All-Reduce 的数据状态变化，就可以表示成：

```text
每张 GPU 都拥有完整的局部梯度

GPU 0 → [A₀][B₀][C₀][D₀]
GPU 1 → [A₁][B₁][C₁][D₁]
GPU 2 → [A₂][B₂][C₂][D₂]
GPU 3 → [A₃][B₃][C₃][D₃]

               │
               ▼

      ┌─────────────────┐
      │ Reduce-Scatter  │
      │   Reduce + 分片  │
      └────────┬────────┘

               │
               ▼

GPU 0 → [A]
GPU 1 → [B]
GPU 2 → [C]
GPU 3 → [D]

               │
               ▼

      ┌─────────────────┐
      │   All-Gather    │
      │    收集所有分片   │
      └────────┬────────┘

               │
               ▼

GPU 0 → [A][B][C][D]
GPU 1 → [A][B][C][D]
GPU 2 → [A][B][C][D]
GPU 3 → [A][B][C][D]
```

因此再次得到：

$$
\boxed{ \text{All-Reduce} = \text{Reduce-Scatter}+ \text{All-Gather} }
$$

这条关系非常重要。

后面进入 ZeRO 时，我们还会重新回来审视这个过程。

> **伏笔：Reduce-Scatter 完成以后，每张 GPU 实际上已经只持有 $1/N$ 的最终规约结果。**
>
> 传统 DP 随后又通过 All-Gather，把完整结果重新复制给了所有 GPU。
>
> 这个看似理所当然的动作，后面会成为 ZeRO 消除显存冗余的重要突破口。

现在暂时记住这个现象即可。

---

## 7. 把整个 DP 训练流程重新串起来

现在理解了 All-Reduce，再回头看 Data Parallelism，整个训练过程就非常清晰了。

完整流程可以概括成：

```text
Data Split
    ↓
Forward
    ↓
Backward
    ↓
Local Gradient
    ↓
All-Reduce
    ↓
Global Gradient
    ↓
Optimizer Step
    ↓
参数保持一致
```

展开之后：

```text
                          Global Batch
                                │
                                ▼
                           Split Data
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
         GPU 0                GPU 1              ... GPU N
           │                    │                    │
           ▼                    ▼                    ▼
      [完整模型 W]         [完整模型 W]          [完整模型 W]
           │                    │                    │
           ▼                    ▼                    ▼
        Forward              Forward              Forward
           │                    │                    │
           ▼                    ▼                    ▼
        Backward             Backward             Backward
           │                    │                    │
           ▼                    ▼                    ▼
           g₀                   g₁                   gₙ
           │                    │                    │
           └────────────────────┼────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │     All-Reduce      │
                    │                     │
                    │  Reduce-Scatter     │
                    │         +           │
                    │    All-Gather       │
                    └──────────┬──────────┘
                               │
                               ▼
                       全局一致的梯度 g
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
         GPU 0               GPU 1              ... GPU N
           │                   │                   │
           ▼                   ▼                   ▼
     Optimizer Step      Optimizer Step       Optimizer Step
           │                   │                   │
           ▼                   ▼                   ▼
          W'                  W'                  W'
```

因此，Data Parallelism 可以浓缩成一个闭环：

```text
同一个模型
    ↓
不同的数据
    ↓
不同的局部梯度
    ↓
All-Reduce
    ↓
相同的全局梯度
    ↓
相同的参数更新
    ↓
重新得到同一个模型
```

这就是 DP 的本质。

---

## 8. 工程实现：DDP 如何隐藏 All-Reduce 开销？

到这里还有一个现实问题。

假设一个模型拥有数十亿参数。

那么它的 Gradient 同样非常巨大。

如果训练流程真的严格按照：

```text
Forward
    ↓
Backward 全部完成
    ↓
等待
    ↓
All-Reduce 所有梯度
    ↓
Optimizer Step
```

那么 GPU 会出现明显的空闲时间：

```text
计算完成
   ↓
GPU 等网络
   ↓
通信完成
   ↓
继续计算
```

这显然会影响训练效率。

因此现代 Distributed Data Parallel 实现通常不会等整个 Backward 完成之后，才一次性通信所有 Gradient。

一种常见做法就是：

> **Gradient Bucket。**

---

### Gradient Bucket

模型中可能拥有成千上万个参数 Tensor。

如果每计算出一个 Tensor 的梯度就立即发起一次 All-Reduce：

```text
Gradient 1 Ready → All-Reduce
Gradient 2 Ready → All-Reduce
Gradient 3 Ready → All-Reduce
...
```

会产生大量非常细碎的通信操作。

因此框架通常会把多个 Gradient 聚合成一个较大的：

> **Bucket**

例如：

```text
Gradient 1 ─┐
Gradient 2 ─┤
Gradient 3 ─┼────► Bucket 0
Gradient 4 ─┘

Gradient 5 ─┐
Gradient 6 ─┼────► Bucket 1
Gradient 7 ─┘
```

当一个 Bucket 中需要的梯度全部 Ready 之后，就可以立刻启动这个 Bucket 的 All-Reduce。

---

### Communication-Computation Overlap

Transformer 的反向传播并不是所有 Layer 同时完成。

而是从后面的 Layer 开始逐层向前计算：

```text
Layer N
   ↓
Layer N-1
   ↓
Layer N-2
   ↓
...
   ↓
Layer 1
```

因此，当后面某些 Layer 的梯度已经计算完成时，没有必要等待整个模型。

可以立刻启动 All-Reduce。

例如：

```text
Backward:

Layer N
Gradient Ready ─────────────► All-Reduce Bucket 0
        │
        ▼
Layer N-1
Gradient Ready ─────────────► All-Reduce Bucket 1
        │
        ▼
Layer N-2
Gradient Ready ─────────────► All-Reduce Bucket 2
        │
        ▼
       ...
```

于是 GPU 可以：

```text
计算 Layer N-1 的梯度
          │
          │ 同时
          ▼
通信 Layer N 已经算好的梯度
```

也就是：

> **一边计算，一边通信。**

这被称为：

> **Communication-Computation Overlap**

理想情况下，大量 All-Reduce 通信时间都可以被 Backward 的计算过程隐藏掉。

所以现实中的 DDP 更接近：

```text
Backward Computation
████████████████████████████████

      All-Reduce
      █████████
            █████████
                  █████████
                        ██████
```

而不是：

```text
Backward Computation
████████████████████████

                        All-Reduce
                        ████████████████
```

这对于大规模分布式训练的性能非常重要。

---

## 9. DP 的通信代价

DP 虽然需要进行 Gradient All-Reduce，但它的通信模式和后面要讲的 TP 有明显区别。

DP Rank 之间主要同步的是：

> **Gradient**

而不需要在每一个 Transformer Layer 的前向计算过程中不断交换 Activation 或矩阵中间结果。

因此可以简单对比：

| 并行方式   | 切分对象   | 每张 GPU 是否保存完整模型 | 主要通信内容                      | 通信特点                |
| ------ | ------ | --------------- | --------------------------- | ------------------- |
| **DP** | Data   | 是               | Gradient                    | 通信相对集中，单次数据量较大      |
| **PP** | Layer  | 否               | Activation / Gradient       | Pipeline Stage 之间通信 |
| **TP** | Tensor | 否               | Activation / Partial Result | 通信频繁，对带宽和延迟非常敏感     |

因此，大规模训练集群中经常会看到这样的物理映射：

```text
单机内部：

GPU ── NVLink/NVSwitch ── GPU
 │                         │
 └──────── TP Group ───────┘

不同服务器：

Node 0
  │
  │ InfiniBand / RoCE
  │
Node 1
  │
  │
Node 2

      DP Group
```

也就是说：

> **TP 往往优先放在高速 NVLink / NVSwitch 域内，而 DP 更适合向不同服务器节点扩展。**

原因就在于：

TP 的通信发生得更加频繁。

而 DP 主要围绕 Gradient All-Reduce 展开，因此在通信拓扑设计上相对更容易扩展到跨节点。

当然，模型越来越大之后，Gradient All-Reduce 本身同样会成为非常重要的性能瓶颈。

如果模型参数量为 $\Psi$，每个梯度元素占用 $b$ Bytes，那么单轮梯度同步涉及的数据规模与：

$$
\Psi \times b
$$

处于同一数量级。

所以：

> **DP 并不是“通信很少”，而是它的通信模式比 TP 更集中、更容易与 Backward 进行重叠。**

---

## 10. DP 真正解决了什么，又没有解决什么？

理解完 All-Reduce 之后，我们再重新审视 DP。

DP 真正解决的问题其实是：

> **吞吐量。**

假设一个模型可以完整放进单张 GPU。

单张 GPU 一次只能处理 4 个样本。

那么使用 8 张 GPU 后，就可以同时处理：

$$
4\times8=32
$$

个样本。

也就是：

```text
1 GPU
   ↓
4 Samples

8 GPUs
   ↓
32 Samples
```

因此 DP 的核心价值是：

> **增加并行计算能力，提高训练吞吐量。**

但它没有解决另外一个问题：

> **模型本身太大怎么办？**

因为无论有多少张 GPU，每张 GPU 都依然保存完整模型。

例如你拥有：

```text
8 × 80GB GPU
```

物理显存总量看起来是：

$$
8\times80
=
640\text{ GB}
$$

但纯 DP 并不会把这 640GB 显存聚合成一个统一的大显存池。

实际情况是：

```text
GPU 0 → 80GB → 完整模型
GPU 1 → 80GB → 完整模型
GPU 2 → 80GB → 完整模型
...
GPU 7 → 80GB → 完整模型
```

而不是：

```text
GPU 0 → 模型的 1/8
GPU 1 → 模型的 1/8
GPU 2 → 模型的 1/8
...
GPU 7 → 模型的 1/8
```

所以：

> **DP 聚合了算力，却没有聚合模型容量。**

对于纯 DP 来说，决定模型能不能启动的仍然是：

> **单张 GPU 的显存容量。**

---

## 11. 70B 模型为什么纯 DP 依然 OOM？

假设现在训练一个：

> **70B，也就是 700 亿参数的模型。**

如果仅使用 BF16 / FP16 保存模型参数，每个参数占用 2 Bytes。

那么仅参数本身需要：

$$
70\times10^9\times2
=
140\times10^9\text{ Bytes}
$$

约为：

$$
140\text{ GB}
$$

而一张常见的 A100 80GB GPU 只有：

$$
80\text{ GB}
$$

因此：

$$
140\text{ GB} > 80\text{ GB}
$$

甚至还没有开始 Forward，模型参数本身就已经无法完整装入单张 GPU。

如果使用纯 DP：

```text
GPU 0 → 尝试加载完整 140GB 模型 → OOM
GPU 1 → 尝试加载完整 140GB 模型 → OOM
GPU 2 → 尝试加载完整 140GB 模型 → OOM
GPU 3 → 尝试加载完整 140GB 模型 → OOM
...
GPU 7 → 尝试加载完整 140GB 模型 → OOM
```

即使你有 8 张 GPU，结果也没有任何区别。

因为纯 DP 要求：

> **每张 GPU 都必须拥有完整模型。**

更麻烦的是，真正训练时显存里保存的还远远不只有 Parameters。

还包括：

* Model Parameters
* Gradients
* Optimizer States
* Activations
* Temporary Buffers
* CUDA Workspace

尤其是在 Adam + Mixed Precision 训练中，Optimizer States 会占据非常巨大的显存空间。

所以一个 70B 模型训练时真正需要的显存，远远不止简单的：

```text
140GB
```

后面在：

> **「谁吃掉了显存？混合精度与 Adam 优化器的代价」**

这一节中，我们会专门把这些显存开销一项一项拆开。

现在这里只需要记住：

> **只要模型本身不能完整装进单张 GPU，纯 DP 就无能为力。**

---

## 12. 从 DP 的极限引出 PP / TP

于是问题自然出现了：

> **既然切数据不能解决模型太大的问题，那么还能切什么？**

答案是：

> **切模型。**

这就是大模型并行真正开始变得有意思的地方。

模型本身又可以从两个不同的维度进行切分。

---

### 第一种思路：按照 Layer 切

假设模型一共有 40 层。

可以这样分：

```text
GPU 0 → Layer  1 ~ 10
GPU 1 → Layer 11 ~ 20
GPU 2 → Layer 21 ~ 30
GPU 3 → Layer 31 ~ 40
```

这样，每张 GPU 就不再需要保存完整模型。

而只需要保存其中一部分 Layer。

这就是：

> **Pipeline Parallelism（PP，流水线并行）**

也就是：

```text
切模型的“层”
```

---

### 第二种思路：连一个 Layer 都太大怎么办？

如果模型继续变大。

甚至出现：

> **单独一个 Transformer Layer 都无法放进一张 GPU。**

那么仅仅按照 Layer 切已经不够了。

必须继续深入到 Layer 内部。

例如一个巨大的 Linear：

```text
                  Linear Layer
                       │
              巨大的 Weight Matrix
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        GPU 0        GPU 1        GPU 2
       一部分矩阵     一部分矩阵     一部分矩阵
          │            │            │
          └────────────┼────────────┘
                       ▼
                  合并计算结果
```

也就是说：

> **把同一层内部的 Tensor / Matrix 继续切开。**

这就是：

> **Tensor Parallelism（TP，张量并行）**

---

于是，大模型训练最经典的三个并行维度就完整出现了：

```text
                     3D Parallelism
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
          DP               PP               TP
    Data Parallel    Pipeline Parallel   Tensor Parallel
           │                │                │
           ▼                ▼                ▼
         切数据            切模型层          切层内 Tensor
```

可以先记住一句最简单的话：

> **DP 切数据，PP 切层，TP 切层内的矩阵。**

但它们解决的问题并不一样：

| 并行方式   | 最核心的问题                     |
| ------ | -------------------------- |
| **DP** | 模型放得下，但单卡训练吞吐量不够           |
| **PP** | 整个模型放不进单卡，需要按照 Layer 拆开    |
| **TP** | 单个 Layer 都太大，需要继续切分 Tensor |

因此，DP 的极限最终把我们带到了新的问题：

> **既然整个模型已经无法放进一张 GPU，那么应该如何让一个 Batch 流经分布在不同 GPU 上的模型层？**


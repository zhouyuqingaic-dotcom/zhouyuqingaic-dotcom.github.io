---
title: "深度解读：基于多模态大模型的 Medical VQA 域自适应研究"
date: 2026-03-28T14:00:00+08:00
draft: false  # 记得这里一定要改成 false
tags: ["Medical AI", "Multi-modal", "VQA", "Qwen3-VL"]
categories: ["Paper Interpretation"]
showToc: true
TocOpen: true
math: true  # 开启数学公式渲染
---

## 1. 核心挑战
在医疗场景下，通用多模态模型（如 Qwen3-VL）直接迁移到特定医学领域（如放射影像）时，往往面临显著的 **Domain Gap**。本文主要探讨如何通过两阶段域自适应（Two-Stage Domain Adaptation）来提升模型的感知能力。

## 2. 核心架构：CES-NET (或 RouterB_Plus)
模型的核心逻辑在于如何平衡通用知识与医学专业知识。

### 关键公式
我们定义的加权损失函数如下：
$$L_{total} = \alpha L_{vqa} + \beta L_{alignment} + \gamma L_{rag}$$

其中，$\alpha, \beta, \gamma$ 是调节各部分权重的超参数。

## 3. 我的深度思考 (Insight)
* **关于 QLoRA 的应用**：在 8x A40 服务器上，通过 QLoRA 微调 Qwen3-VL 展现出了极高的显存性价比，但在处理高分辨率医学图像时，Attention 机制的瓶颈依然存在。
* **RAG 的必要性**：单纯靠微调很难让模型记住所有的医学长尾知识，引入检索增强（RAG）结合 BioMedCLIP 的编码能力，能显著降低幻觉。

## 4. 实验结论
在特定数据集上的测试表明，该方案相较于 Baseline 在 BLEU-4 指标上提升了约 15%。

---
*本文由 Yuqing 撰写，转载请注明出处。*



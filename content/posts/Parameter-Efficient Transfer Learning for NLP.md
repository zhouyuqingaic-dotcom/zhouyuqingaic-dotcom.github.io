---
title: "Adapter 深度解读：从 2019 瓶颈架构看今日 LoRA 与 LLaMA-Adapter 的演进根基"
date: 2026-05-16T21:16:00+08:00
draft: false
tags: ["Adapter", "PEFT", "LoRA", "LLaMA-Adapter", "Bottleneck"]
categories: ["Paper Interpretation"]
showToc: true
TocOpen: true
math: true
---

> **论文基本信息**
> * **Title**: Parameter-Efficient Transfer Learning for NLP
> * **arXiv**: [1902.00751](https://arxiv.org/abs/1902.00751)
> * **Venue**: International Conference on Machine Learning (ICML) 2019

# Parameter-Efficient Transfer Learning for NLP


> 在深度学习的范式演进中，‘预训练 + 微调’早已成为解决各类下游任务的标准答案。然而，随着模型参数量呈指数级膨胀，全量微调的边际成本已变得不可接受。如何在保留模型原本通用知识的前提下，以最小的代价实现特定任务的适配？2019 年，Houlsby 等人提出的 Adapter 架构为这一历史性难题提供了极具开创性的解答。通过在 Transformer 层中巧妙地插入可训练的瓶颈模块，Adapter 彻底重塑了模型微调的路线图。本文将从论文的原始出发点开始，详细拆解其核心创新、特征维度转换逻辑，并探讨其作为 PEFT 范式源头，对当今多模态微调架构的深远影响。
>
> ---
> —— TL;DR


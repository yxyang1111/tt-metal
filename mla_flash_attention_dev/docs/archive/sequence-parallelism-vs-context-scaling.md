# Sequence Parallelism 和 Context Scaling 简明说明

更新时间：`2026-04-27`

这篇文档只回答一个问题：

> **Sequence Parallelism 和 context scaling 都跟“长序列”有关，但它们到底是不是一回事？**

先说结论：**不是一回事。`Sequence Parallelism` 主要是“怎么把序列维度上的计算分到多张卡上”，属于系统/并行执行问题；`context scaling` 更常指“怎么让模型支持更长上下文窗口”，属于模型能力和长上下文建模问题。** 这个区分可以从 Megatron 的并行文档和长上下文综述的分类中直接看出来。[S1][C1]

## 1. 什么是 Sequence Parallelism

Megatron/NVIDIA 的官方文档把 `Sequence Parallelism (SP)` 定义为：在 Transformer 层里，沿着序列维度把计算负载和 activation memory 分散到多张 GPU 上；它是对 tensor parallelism 的扩展，并且通常要求 `tensor_model_parallel_size > 1`。[S1]

提出 `Sequence Parallelism` 的原始论文也说明了它的目标：**减少 activation recomputation 带来的额外计算，并降低 activation memory**；论文把它和 selective activation recomputation 一起作为改进大模型训练效率的方法，而不是作为改变注意力语义的方法。[S2]

所以可以把 SP 记成一句话：**它主要解决“长序列或大 batch 训练时显存和并行效率怎么办”，而不是“模型本身能不能理解更长上下文”。** 这是对 [S1][S2] 的直接归纳。

## 2. 什么是 context scaling

长上下文综述把这类问题概括为“extend the context length in LLMs”，并把方法分成修改位置编码、修改 attention 机制，以及训练、微调、推理阶段的不同方案。[C1] 在工程讨论里，大家也常把这类工作统称成 `context scaling`；但论文标题里更常见的写法是 `context window extension` 或 `extend context length`。[C1][C3][C4][C5]

几个典型例子：

- `ALiBi` 用线性 bias 代替传统位置嵌入，让模型在推理时更容易外推到比训练时更长的序列。[C2]
- `Position Interpolation (PI)` 不是把位置直接外推，而是把输入位置线性缩放回原始上下文范围；论文报告它可以把 RoPE 模型扩到 `32K`，并尽量保持原窗口内的效果。[C3]
- `YaRN` 也是 RoPE 长上下文扩展方法，论文强调它比此前方法需要更少 token 和更少训练步数。[C4]
- `LongRoPE` 则把这一方向继续推进到超长上下文，论文报告可扩到 `2048K` token，同时尽量保住短上下文性能。[C5]

所以一句话记忆：**context scaling 主要解决“模型最多能看多长、位置编码怎么改、长上下文能力怎么保住”。** 这是对 [C1][C2][C3][C4][C5] 的直接归纳。

## 3. 它们的核心区别

| 维度 | Sequence Parallelism | context scaling |
| --- | --- | --- |
| 关注点 | 分布式执行和显存分摊。[S1][S2] | 模型可支持的上下文长度与长上下文效果。[C1] |
| 主要手段 | 沿序列维度切分 activation / 计算，常与 TP 配合。[S1] | 改位置编码、改 attention、做长上下文微调或推理策略。[C1][C2][C3][C4][C5] |
| 直接结果 | 更容易把训练或推理“跑起来、放得下”。[S1][S2] | 让模型“理论上和经验上能用更长上下文”。[C1][C3][C4][C5] |
| 是否直接改变模型的长上下文语义能力 | 通常不直接改变，重点是系统侧扩展。[S1][S2] | 会直接影响长上下文能力，因为它就是在解决窗口扩展问题。[C1][C3][C4][C5] |

## 4. 最容易混淆的一点

`context scaling` 不等于 `Context Parallelism (CP)`。Megatron 文档里的 `Context Parallelism` 也是一种**系统并行**方法：它把所有层的 activation 沿序列维度分到多张卡上，官方文档明确说它对 long-context training 很关键。[S1] 也就是说：

- `Sequence Parallelism` 和 `Context Parallelism` 都是“怎么并行算”的问题。[S1]
- `context scaling` 则更接近“怎么把模型窗口扩长”的问题。[C1][C3][C4][C5]

## 5. 如果只记一句话

**`Sequence Parallelism` 是系统侧的“切分算”；`context scaling` 是模型侧的“把窗口拉长”。长上下文系统里二者经常一起出现，但它们回答的不是同一个问题。** 这是结合 [S1][S2][C1][C3][C4][C5] 得出的总结。

## 参考来源

[S1] NVIDIA, *Parallelisms Guide — Megatron Bridge*.  
https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html

[S2] Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, arXiv:2205.05198.  
https://arxiv.org/abs/2205.05198

[C1] Wang et al., *Beyond the Limits: A Survey of Techniques to Extend the Context Length in Large Language Models*, arXiv:2402.02244.  
https://arxiv.org/abs/2402.02244

[C2] Press et al., *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, arXiv:2108.12409.  
https://arxiv.org/abs/2108.12409

[C3] Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation*, arXiv:2306.15595.  
https://arxiv.org/abs/2306.15595

[C4] Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, arXiv:2309.00071.  
https://arxiv.org/abs/2309.00071

[C5] Ding et al., *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*, arXiv:2402.13753.  
https://arxiv.org/abs/2402.13753

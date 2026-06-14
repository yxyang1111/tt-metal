# DeepSeek V4 Attention 与 TT 数据流建模笔记

更新时间：`2026-04-24`

## 1. 这份文档要回答什么

这份笔记聚焦三个问题：

1. DeepSeek V4 在 attention 上，公开确认了哪些新东西。
2. 这些变化为什么会给 TT 上的数据流/代价模型带来新的压力测试。
3. 如果我们希望当前建模方法是“相对通用”的，应该优先补哪些 synthetic test case。

## 2. 证据边界

本文只把下面几类材料当作一手或近一手依据：

- DeepSeek 官方 Hugging Face model card：
  - [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
  - [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- 对应公开 `config.json`
- 对照基线：
  - [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
  - [DeepSeek-V3.2 技术报告摘要](https://arxiv.org/abs/2512.02556)

需要明确一点：

> 截至目前，V4 官方公开材料已经明确给出 `CSA + HCA`、`1M context`、`27% single-token inference FLOPs`、`10% KV cache` 这些高层结论，但 **CSA / HCA 的完整数学定义与 kernel 级细节还没有像 V3/FlashMLA 那样公开完整展开**。因此下文会严格区分“公开确认”和“基于 config 的合理推断”。

## 3. 公开确认的信息

从官方 V4 model card，至少可以确定以下几点：

- V4 是 `1M` context 的 MoE 系列。
- attention 主升级被官方明确命名为 **Hybrid Attention Architecture**。
- 这个 hybrid attention 由 **Compressed Sparse Attention (CSA)** 和 **Heavily Compressed Attention (HCA)** 组合而成。
- 在 `1M` context 下，`DeepSeek-V4-Pro` 的单 token 推理只需要相对 `DeepSeek-V3.2` 的：
  - `27%` single-token inference FLOPs
  - `10%` KV cache
- V4 还引入了 `mHC` 和 `Muon`，但它们不是本文关注的 attention/dataflow 主体。

如果只用一句话概括，V4 在 attention 上的方向不是“把 MLA 再快一点”，而是：

> **把长上下文注意力拆成多个精度/压缩级别不同的子路径，让“逻辑看见的 1M token”与“物理真正读到的 KV 字节数”强烈脱钩。**

## 4. 从公开 config 能读出的结构线索

虽然官方还没把 CSA/HCA 全部讲透，但公开 config 已经暴露出不少重要信号。

### 4.1 明确存在局部精确路径

V4 Pro/Flash 的 config 都有：

- `sliding_window = 128`

这至少说明 V4 的 attention 里明确存在一个**近邻局部窗口**。不管 CSA/HCA 的完整实现是什么，这个字段都意味着：

- 最近 `128` token 有一条 locality 很强的精确访问路径；
- 对 TT 来说，这一段最像当前主线 decode 里“连续 paged K + reuse/mcast”最容易发挥的部分。

### 4.2 明确存在稀疏检索/候选筛选路径

V4 Pro/Flash 都公开了：

- `index_n_heads = 64`
- `index_head_dim = 128`
- `index_topk = 1024` (`Pro`) / `512` (`Flash`)
- `num_hash_layers = 3`

这组字段已经非常直接地指向“先建索引/先筛候选，再做主 attention”的结构。即使官方还没给完整算法，也可以稳妥地说：

- V4 不是对全部历史 token 做统一 dense sweep；
- attention 前面显式插入了一个 **index/select** 子过程；
- 这个子过程本身就会带来元数据访问、候选 gather、bank skew 和 top-k 选择开销。

对 TT 建模最重要的含义是：

> **未来模型不能只算 `QK/PV` 主体 FLOPs 和 KV bytes，还必须单独建模 indexer / selector / metadata path。**

### 4.3 明确存在分层压缩调度，而不是单一 attention 模式

V4 config 里最有价值的字段是：

- `compress_ratios`

它不是单个标量，而是一串 schedule。

公开配置里能看到的典型模式是：

- `DeepSeek-V4-Pro`: 近似 `[128, 128, 4, 128, 4, 128, ..., 4, 0]`
- `DeepSeek-V4-Flash`: 近似 `[0, 0, 4, 128, 4, 128, ..., 4, 0]`

即便官方还没解释这个数组的精确定义，下面这些判断已经相当稳：

- V4 不是“所有层一个 attention 配方”；
- 模型内部明确区分了 **不压缩 / 轻压缩 / 重压缩** 多种模式；
- `4` 和 `128` 这两个量级差异极大，说明 V4 不是只做小修小补，而是把不同层/阶段放在完全不同的访存 regime 下。

一个保守但很有用的推断是：

- `0` 更像“不压缩/原路径”
- `4` 更像“中等压缩、保留更多细节”
- `128` 更像“强压缩摘要路径”

这也可以直观理解为 attention 的几种“花活”：

1. 近处精确看
2. 远处挑着看
3. 更远处只看高压缩摘要
4. 不同层还会换着来

### 4.4 KV 仍然是高度压缩/共享的，而不是回到标准 MHA

V4 config 还有几个非常关键的点：

- `num_key_value_heads = 1`
- `q_lora_rank`
- `o_lora_rank`
- `qk_rope_head_dim = 64`
- `compress_rope_theta = 160000`

这说明 V4 在 attention state 上仍然保留了很强的“DeepSeek 风格”：

- KV 不是普通 MHA 那种“每头都存一份完整 K/V”
- query/output 侧仍然带有明显的低秩/latent 结构
- RoPE 也很可能已经和压缩路径发生耦合，而不是一套位置编码通吃所有路径

这点很重要，因为它说明：

> **V4 并没有背离“压缩 KV、让 attention 更经济”这个总方向，只是把 `单一路径 MLA` 推进成了 `局部精确 + 稀疏筛选 + 多级压缩` 的混合体。**

### 4.5 相对 V3.2，V4 的变化不是“上下文更长”这么简单

如果只看官方公开 config，V3.2 和 V4 的关系大致可以概括成：

- `V3 / FlashMLA`：更接近单一路径的经济 attention，重点是 latent KV、paged decode、`reuse_k`
- `V3.2`：已经出现明显的 sparse/index 信号，例如 `index_topk = 2048`
- `V4`：在 sparse 之外，再加入 `sliding_window = 128`、`compress_ratios`、`num_hash_layers = 3`、`1M context`

所以 V4 的真正新增难点不是“把 `S` 再拉长”，而是把长上下文 attention 变成了一个**分路径、分压缩率、分层调度**的问题。

## 5. 把 V4 放回当前 TT 主线里看

当前仓库里真正稳定、可讨论的数据流主线仍然是：

- `flash-mla-dataflow-first-principles.md`
- `flash-mla-current-dataflow-analysis.md`
- `experiments/perf_model/README.md`

对应的现有回归入口主要是：

- `tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py`
- `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill.py`
- `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill_v_embedding_space.py`

这些路径今天覆盖得最好的，是一种**单一 steady-state MLA decode/prefill**：

- `Q` 的组织和分发规则比较稳定
- `K` 大体还是连续 chunk / paged chunk 地流动
- decode 可以利用 `reuse_k`
- `column sender -> multicast -> worker -> lane/tree reduce` 这个骨架比较稳定

而 V4 真正带来的变化是：**这个骨架不再足以描述“所有层都在干什么”。**

当前模型更像是在回答：

> 给定一个统一 attention kernel，`Q/K/V` 怎么搬、怎么算、怎么算代价。

V4 要求模型开始回答：

> 给定一个 hybrid attention schedule，不同层究竟在搬哪一种状态、读哪一种历史、只读多少、metadata 多少、局部窗口和全局摘要如何合并。

## 6. V4 对 TT 数据流/代价模型提出的新要求

### 6.1 不能再把“逻辑序列长度”直接当成“物理读流量”

这是最核心的一条。

在当前 FlashMLA 语境下，我们常常默认：

- `S` 越长
- 读到的 `K/V` chunk 越多
- `reader_ms` 近似线性上涨

V4 之后，这个近似不再天然成立，因为真实代价会拆成至少三部分：

1. 近邻窗口的精确读
2. 稀疏索引选中的候选读
3. 远端压缩摘要读

也就是说，未来模型里必须同时区分：

- `logical_context_len`
- `exact_tokens_touched`
- `sparse_tokens_selected`
- `compressed_summary_tokens`
- `metadata_bytes`

### 6.2 不能假设每个 worker/column 读的是同一批连续页

当前 TT decode 非常依赖一个结构性事实：

- 同列 core 对同一轮 `K chunk` 有高复用；
- 所以 sender 从固定 bank 读一次，再沿列 multicast，很划算。

而 V4 的 sparse/indexed path 会天然冲击这个假设：

- 不同 query/head/layer 选中的候选页可能不同；
- 候选集合可能是非连续、跨 bank、甚至高度 skew 的；
- 这会直接削弱“统一 sender + 统一 multicast”的收益。

因此，通用模型必须能表达至少三种读模式：

- `contiguous_stream`
- `paged_gather_clustered`
- `paged_gather_random`

### 6.3 不能把 attention 只看成主计算 kernel，还要把 indexer/select path 独立出来

V4 config 里的 `index_*` 和 `num_hash_layers` 说明：

- 在真正进入主 attention 前，已经发生了一轮候选筛选；
- 这轮筛选本身就有算子开销、SRAM/L1 压力、以及额外同步点。

换句话说，V4 的真实 latency 不是：

- `reader + compute + writer`

而更像：

- `indexer/select + reader + compute + merge/reduce + writer`

### 6.4 不能再用“单一层平均行为”代替全模型行为

`compress_ratios` 暴露出的最大信号，就是**层间 attention mode 不同**。

这要求 perf model 至少具备：

- per-layer mode schedule
- per-layer bytes touched
- per-layer metadata overhead
- 最终按 layer 累加，而不是只拿一个“代表层”乘层数

如果没有这一层抽象，模型很可能只能拟合当前 MLA，却无法泛化到 V4 这种 hybrid schedule。

### 6.5 不要把 `V = K[..., :d_v]` 当成永恒公理

当前 TT MLA 路径里，一个很关键的优势是：

- `V` 可以复用 `K` 的一部分视图

这在 V3/FlashMLA 下很成立，也正是当前 decode 数据流漂亮的原因之一。

但 V4 公开 config 还没有明确保证：

- 所有 attention 子路径都保留这个关系；
- 强压缩摘要路径仍然可以零成本还原成“`V` 是 `K` 前缀视图”的形式。

因此，**通用建模**应该把这个性质降级为可选 workload flag，而不是硬编码前提。

## 7. 建议补的测试矩阵

如果我们要拿 V4 来检验“模型是不是通用的”，建议至少准备下面六类 synthetic case。

### T0. 现有 MLA 基线

目的：保留今天已经能解释清楚的主线。

- dense/paged MLA decode
- dense/chunked MLA prefill
- `reuse_k = true`

对应现有入口：

- `test_mla_decode.py`
- `test_mla_prefill.py`
- `test_mla_prefill_v_embedding_space.py`

### T1. Local-window only

目标形态：

- `sliding_window = 128`
- 只读最近连续页
- 无全局 sparse gather
- 无压缩摘要

作用：

- 验证模型能不能把“V4 的近邻精确路径”退化回当前最容易映射到 TT 的连续流模式。

### T2. Indexed sparse gather

目标形态：

- `index_topk in {512, 1024, 2048}`
- 只读索引选中的全局页
- 选择模式分别做：
  - clustered
  - bank-balanced
  - bank-skewed
  - random

作用：

- 专门压 reader / metadata path；
- 检验 sender+mcast 假设在稀疏 gather 下什么时候失效。

### T3. Local + sparse hybrid

目标形态：

- 最近 `128` token 精确访问
- 远端另加 `topk` 稀疏候选

作用：

- 这是最接近 V4 公开口径的第一层近似；
- 能直接测试“局部连续流 + 全局离散 gather”在 TT 上如何拼接。

### T4. Compression-ratio sweep

目标形态：

- 压缩率 `r in {4, 128}`
- 同一 `logical_context_len` 下，对比不同 `physical_bytes`

作用：

- 检验模型能否把“看到的历史长度”和“真的搬了多少字节”拆开；
- 对应 V4 `compress_ratios` 的核心精神。

### T5. Layer-schedule sweep

目标形态：

- early dense / late compressed
- alternating `4 / 128`
- dense tail

作用：

- 检验模型是不是必须假设“全层同构”；
- 这是从“能解释一个 kernel”走向“能解释一个 hybrid model”的关键一步。

### T6. 1M context admission test

目标形态：

- `S in {128K, 256K, 512K, 1M}`
- paged KV
- 变 page size / prefetch distance / bank mapping

作用：

- 不只是看算术量；
- 更看 page table、metadata、bank locality、NoC hop 和 admission boundary。

## 8. 对当前项目最值得先做的三件事

### 8.1 先扩 workload schema，不要急着猜完整 V4 kernel

建议在通用模型里先引入这些字段：

- `attention_mode`: `dense_mla | local_window | sparse_indexed | compressed | hybrid`
- `local_window`
- `index_topk`
- `selection_pattern`
- `compression_ratio`
- `compression_schedule`
- `metadata_bytes`
- `kv_relation`: `shared_view | separate | unknown`

这样即使官方还没把 V4 kernel 全开源，模型也能先吃下它最关键的结构变化。

### 8.2 把 smoke test 从“只扫 S/B”扩展到“扫访问模式”

当前 `perf_model` 已经证明解析模型对 `S/B` 的响应是健康的。下一步最有价值的不是继续只扫 `S/B`，而是补：

- contiguous vs gathered
- no-index vs indexed
- no-compress vs `r=4` vs `r=128`
- homogeneous-layer vs scheduled-layer

### 8.3 先把 V4 当成“方法学验证集”，不要急着把它写成完全确定的实现说明

目前最合适的定位不是：

- “我们已经完全知道 V4 kernel 怎么写”

而是：

- “V4 给了我们一个很好的外部压力测试：如果模型真的通用，就应该能容纳 local window、indexed sparse gather、multi-ratio compression 和 layer schedule 这四种变化。”

## 9. 一句话结论

DeepSeek V4 对当前 TT MLA 主线最重要的启发，不是又多了一个模型名，而是它把 attention 从“单一路径的 MLA 数据流问题”推进成了：

> **局部精确访问 + 稀疏候选访问 + 多级压缩摘要 + 分层调度** 的混合数据流问题。

如果我们的建模方法真想做到“相对通用”，最先要补的不是 V4 专用参数表，而是能把这四类访问模式统一表达出来的 workload/schema/test matrix。

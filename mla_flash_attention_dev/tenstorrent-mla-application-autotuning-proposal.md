# Tenstorrent 上 MLA 应用与自动寻优模型提案

## 1. 文档定位

这份文档用于把下面三条线收敛成一个统一提案：

- `MLA + Flash Attention` 在 Tenstorrent 上的应用与落地
- 围绕 `layout / KV forwarding / multicast / NoC / pipeline` 的系统优化
- 一个面向 MLA workload 的 `offline autotuning / policy selection` 模型

本文的目标不是继续发散想法，而是给出一份可以直接演化为下面任一产物的中间稿：

- 论文 proposal
- arXiv 初稿提纲
- 项目开题文档
- 后续实验与实现的总控文档

相关已有材料：

- `mla_flash_attention_dev/tenstorrent-mla-paper-ideas.md`
- `mla_flash_attention_dev/tenstorrent-mla-expanded-roadmap.md`
- `mla_flash_attention_dev/next-step-execution-plan.md`
- `mla_flash_attention_dev/ref/tt-metal_kv_forwarding_design_space_autotuner.md`

## 2. 一句话主张

本文建议的核心主张是：

**MLA 在 Tenstorrent 这类显式 dataflow 架构上的收益，不只来自 latent KV cache 压缩本身，还来自对数据布局、通信拓扑、流水调度和执行计划的联合优化；这些优化可以被组织成一个可建模、可搜索、可缓存的自动寻优框架。**

这条主张有三个层次：

- `应用层`：把 MLA 作为 TT 上长上下文推理的重要工作负载
- `系统层`：解释为什么 TT 的 `L1 / NoC / core layout / explicit dataflow` 会显著影响 MLA 收益
- `方法层`：提出一个 `hierarchical autotuning model` 来自动选择最优执行计划

## 3. 推荐论文命题

### 3.1 推荐题目方向

最推荐的题目方向是：

**`Efficient MLA Inference on Tenstorrent: Dataflow Mapping, Cache Layout, and Autotuned Execution Planning`**

可以接受的替代表述：

- `Mapping Multi-Head Latent Attention to Explicit-Dataflow Accelerators`
- `Cache Layout and Autotuned Scheduling for MLA on Tenstorrent`
- `Understanding and Optimizing MLA on Tenstorrent with Policy-Guided Dataflow Planning`

### 3.2 论文类型判断

这更像一篇：

- `systems` 论文
- `architecture-aware ML systems` 论文
- `kernel/dataflow co-design + autotuning` 论文

而不是：

- 新模型结构论文
- 新 attention 机制论文
- 训练算法论文

### 3.3 核心研究问题

建议把全文收敛成下面 4 个问题：

1. `MLA + Flash Attention` 在 Tenstorrent 上最自然的应用落点是什么？
2. MLA 的收益在 TT 上主要受哪些系统因素控制：`DRAM`、`NoC`、`L1`、`layout`、`multicast` 还是 `compute`？
3. 在不同 workload 和物理布局下，最优执行计划是否显著变化？
4. 能否用一个轻量但有效的寻优模型，自动选择接近 oracle 的执行配置？

## 4. 推荐 scope

## 4.1 论文 scope

从论文叙事上，建议主标题围绕：

- `MLA inference on Tenstorrent`
- `dataflow mapping + communication organization + autotuned execution planning`

不要把题目收窄成纯 kernel patch，也不要把题目扩大到完整多芯片全栈。

## 4.2 工程主战场

为了可做性，建议把工程主战场分成两层：

- `主优化战场`：`single-chip non-causal prefill`
- `应用验证战场`：`paged MLA decode`

这样设计的原因是：

- `non-causal prefill` 更容易干净地暴露 `KV forwarding / multicast / layout / dual NoC / pipeline` 等问题
- `paged decode` 更符合 MLA 的实际应用价值，可以作为论文里的 end application validation
- 两者共享一部分底层 `SDPA` 主栈，结果可以形成“受控优化 + 应用验证”的结构

## 4.3 第一篇不建议纳入主线的内容

- 不把 `training / backward` 放进第一篇
- 不把 `full model reproduction` 放进第一篇
- 不把 `multi-chip` 作为第一阶段必须项
- 不把 autotuner 默认写成 `online real-time search`
- 不把 quantization 作为第一篇必须项

## 5. 当前已有资产

从仓内现状看，这个方向已经有足够的起跑资产：

### 5.1 通用 Attention / Flash-style SDPA

- `ttnn.transformer.scaled_dot_product_attention`
- `ttnn.transformer.chunked_scaled_dot_product_attention`
- `ttnn.transformer.scaled_dot_product_attention_decode`
- `ttnn.transformer.paged_scaled_dot_product_attention_decode`
- `ttnn.transformer.windowed_scaled_dot_product_attention`
- `ttnn.transformer.joint_scaled_dot_product_attention`
- `ttnn.transformer.ring_joint_scaled_dot_product_attention`
- `ttnn.transformer.ring_distributed_scaled_dot_product_attention`

### 5.2 MLA 主路径

- `ttnn.transformer.flash_mla_prefill`
- `ttnn.transformer.chunked_flash_mla_prefill`
- `ttnn.transformer.flash_multi_latent_attention_decode`
- `ttnn.transformer.paged_flash_multi_latent_attention_decode`

### 5.3 模型接入

- `models/demos/deepseek_v3/tt/mla/mla1d.py`
- `models/demos/deepseek_v3/tt/mla/mla2d.py`

### 5.4 核心实现层

- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`

### 5.5 已有测试资产

- `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill.py`
- `tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py`
- `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill_v_embedding_space.py`
- `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_flash_mla_deepseek.py`

因此，这不是“从零发明”的题，而是“基于现有可运行实现，提炼成一个系统问题并加入自动寻优模型”的题。

## 6. 统一系统框架

建议把整篇文章组织成下面这个框架：

```mermaid
flowchart LR
    workload[MLA Workload Signature]
    candidate[Candidate Execution Plans]
    filter[Feasibility Filter]
    cost[Analytical Cost Model]
    topk[Top-K Measured Reranking]
    cache[Cached Policy]
    runtime[TTNN/TT-Metal Execution]
    profile[Profiler Feedback]

    workload --> candidate
    candidate --> filter
    filter --> cost
    cost --> topk
    topk --> cache
    cache --> runtime
    runtime --> profile
    profile --> cost
```

这里的核心思想是：

- 不是把 autotuner 写成一个黑盒搜索器
- 而是把它写成 `规则过滤 + cost model + 少量真实测量 rerank + policy cache` 的分层系统

这更符合 TT 这类平台的实际工程形态，也更容易站得住。

## 7. 自动寻优模型提案

## 7.1 模型定位

推荐把 autotuner 定位为：

**`hierarchical offline autotuner with cached policy selection`**

也就是：

- 训练和搜索主要离线完成
- 运行时只做轻量 workload signature 匹配
- 输出是一个可缓存的执行策略

不建议第一版直接把它写成在线实时搜索，因为那会引入额外复杂度，而且未必有必要。

## 7.2 输入特征

建议把输入特征分成三组。

### 7.2.1 Workload 特征

- `mode`: `prefill` / `decode`
- `causal`: `true` / `false`
- `paged`: `true` / `false`
- `batch_size`
- `seq_len_q`
- `seq_len_kv`
- `num_q_heads`
- `num_kv_heads`
- `head_dim_qk`
- `head_dim_v`
- `kv_lora_rank`
- `d_rope`
- `block_size`

### 7.2.2 Hardware 特征

- `compute_with_storage_grid_size`
- `available_sub_core_grids`
- `dram_bank_count`
- `dram_bank_to_worker locality`
- `noc_channel_count`
- `single_core_l1_budget`
- `arch`
- `single_chip / multi_chip`

### 7.2.3 中间结构特征

这些特征不是原始 workload 给的，而是由 host 侧快速估计出来：

- `chain_length_stats`
- `mcast_eligibility_ratio`
- `estimated_injector_hotspot`
- `estimated_active_core_ratio`
- `estimated_dram_bytes`
- `estimated_noc_bytes`
- `estimated_l1_usage`
- `tail_imbalance_score`

## 7.3 输出决策变量

autotuner 需要为每个 workload 选择一组执行计划参数。

### 7.3.1 并行与分块

- `batch_parallel_factor`
- `head_parallel_factor`
- `q_parallel_factor`
- `q_chunk_size`
- `k_chunk_size`

### 7.3.2 核布局策略

- `layout_policy = {default, row_packed_by_head, row_packed_by_batch_head, bandwidth_balanced, custom_sub_core_grids}`

### 7.3.3 通信拓扑策略

- `chain_mode_policy = {global_unicast, global_multicast, per_chain_hybrid}`
- `topology_mode = {star_unicast, row_multicast, tree2, tree4, hybrid_tree}`
- `tail_policy = {strict_uniform, dummy_iters, fallback_unicast}`

### 7.3.4 流水和缓存策略

- `pipeline_depth`
- `prefetch_distance`
- `k_cb_depth`
- `v_cb_depth`
- `kv_transfer_mode`

### 7.3.5 热点均衡策略

- `injector_policy`
- `rotating_period`
- `sub_injector_policy`
- `dual_noc_policy`

## 7.4 目标函数

建议定义一个多目标评分函数，而不是只看单一 latency：

```text
Score(a | x) =
    w_lat  * Latency_est(a, x)
  + w_dram * DRAM_est(a, x)
  + w_noc  * NOC_est(a, x)
  + w_stall * Stall_est(a, x)
  + Penalty_constraints(a, x)
```

其中：

- `x` 是 workload + hardware + intermediate features
- `a` 是候选执行计划
- `Penalty_constraints` 用于处理不可行或高风险配置

### 7.4.1 主目标

- 对 `decode`：最小化 `latency / token`
- 对 `prefill`：最小化总 latency，或在固定延迟预算下最大化吞吐

### 7.4.2 次目标

- `DRAM bytes`
- `NoC bytes`
- `injector stall`
- `receiver idle`
- `L1 overflow risk`
- `multicast coverage`

### 7.4.3 约束项

- 数值正确性
- chunk/page 对齐约束
- L1 容量约束
- TT kernel 当前可实现性
- 运行时开销不能抵消优化收益

## 7.5 模型结构

推荐用三层模型，而不是一步到位的黑盒机器学习。

### 7.5.1 第 1 层：Feasibility Filter

先过滤掉明显不可行的候选：

- L1 放不下
- chunk/block 不对齐
- 拓扑模式和物理布局不兼容
- chain 太短，不值得启用 tree
- paged case 下 page/block 组合不合适

这层的作用是减少搜索空间，避免 autotuner 浪费时间在无效配置上。

### 7.5.2 第 2 层：Analytical Cost Model

对可行候选做快速估分。

建议至少显式建模下面几项：

- `DRAM traffic model`
- `NoC traffic model`
- `injector pressure model`
- `multicast benefit estimate`
- `pipeline overlap estimate`
- `active core efficiency estimate`

最重要的是不要追求特别精确，而是追求：

- 能快速排序
- 能区分明显好坏
- 能给出稳定的 top-K

### 7.5.3 第 3 层：Measured Reranking

对 cost model 选出来的 top-K 候选做少量真实测量，然后重排：

- `K = 3` 或 `5`
- 测量真实 latency
- 记录 profiler 指标
- 更新 policy cache

这一步的目标是缩小 `heuristic/cost model` 和 `oracle` 的差距。

## 7.6 推荐的具体实现版本

如果你想把 autotuner 写得更像“模型”，建议分两个版本。

### V1：纯分析式 autotuner

结构：

- 规则过滤
- 手工设计 cost model
- top-K 测量
- policy cache

优点：

- 最稳
- 最容易先做出来
- 最适合第一篇

### V2：带学习残差的 ranker

在 V1 的基础上，再加一个轻量学习器，对 `CostModel` 的残差做修正。

输入：

- workload 特征
- 硬件特征
- 设计空间参数
- cost model 的中间输出

输出：

- `residual correction`
- 或直接做 pairwise ranking

建议模型形态：

- `XGBoost / LightGBM`
- 或一个很小的 `MLP ranker`

这样写的好处是：

- 不需要大量样本
- 容易解释
- 能把论文里的“模型”成分补强

如果时间有限，第一篇可以只做 V1，在 conclusion 里把 V2 作为扩展。

## 7.7 运行时部署方式

推荐把 autotuner 的最终产物做成 `policy cache`：

```text
signature(workload, hardware) -> best_plan
```

运行时流程：

1. 提取 workload signature
2. 查 cache
3. 命中则直接用 cached policy
4. 未命中则走一次离线候选评估或小规模 top-K 测量
5. 把结果写回 cache

这比“每次运行都搜索”更符合真实系统，也更容易让论文读者接受。

## 8. 推荐论文贡献写法

建议把贡献明确写成下面 4 点。

### Contribution 1

我们给出了 `MLA + Flash Attention` 在 Tenstorrent 显式 dataflow 架构上的统一映射，覆盖：

- 模型层
- 算子层
- program factory
- reader/compute/writer 数据流层

### Contribution 2

我们把 TT 上 MLA 的关键性能问题统一成一个系统设计空间，核心维度包括：

- `layout`
- `KV forwarding`
- `multicast / hybrid / tree`
- `pipeline`
- `dual NoC`
- `injector balancing`

### Contribution 3

我们提出了一个 `hierarchical autotuning model`，自动为不同 MLA workload 选择接近 oracle 的执行计划。

### Contribution 4

我们用真实 profiling 和 workload sweep 证明：

- MLA 在 TT 上的收益区间是可建模的
- 不同 workload 的最优计划差异显著
- autotuner 能显著缩小固定策略与 oracle 的差距

## 9. 实验设计

## 9.1 基线

论文中建议至少放下面几组基线：

- `MHA baseline`
- `GQA/MQA baseline`
- `MLA current baseline`
- `MLA optimized fixed-policy`
- `MLA autotuned`
- `oracle best-in-search-space`

## 9.2 工作负载

建议 workload 至少覆盖：

- `seq_len = 1k / 4k / 16k / 64k`
- `batch = 1 / 4 / 8`
- `prefill`
- `decode`
- `paged`
- `non-paged`

如果时间有限：

- 把 `single-chip non-causal prefill` 做成主消融
- 把 `paged decode` 做成应用验证

## 9.3 指标

- `latency`
- `latency / token`
- `throughput`
- `DRAM bytes`
- `NoC bytes`
- `injector stall`
- `receiver idle`
- `active core ratio`
- `oracle gap`
- `search overhead`
- `policy cache hit rate`
- `numerical correctness`

## 9.4 必须要有的图

建议至少准备下面这些图：

1. `MHA / GQA / MLA` 对比图
2. 不同 `layout_policy` 的 latency / traffic 图
3. `fixed heuristic vs autotuner vs oracle` 图
4. 不同 `seq_len` 下 bottleneck 切换图
5. autotuner 搜索开销与收益图
6. 一张 `dataflow / communication` 总图

## 10. 推荐论文结构

## 10.1 推荐章节

1. Introduction
2. Background: MLA, Flash-style attention, and Tenstorrent execution model
3. Problem formulation
4. Mapping MLA to Tenstorrent
5. Design space of dataflow and communication strategies
6. Hierarchical autotuning model
7. Experimental setup
8. Results and analysis
9. Limitations and future work

## 10.2 各章重点

### 第 4 章：Mapping

重点回答：

- MLA 的数据流在 TT 上怎么落地
- prefill / decode 哪些部分共享，哪些部分不同
- 为什么 TT 上值得做 layout / communication 级优化

### 第 5 章：Design Space

重点回答：

- 为什么固定策略不够
- 为什么 `layout / pipeline / multicast / tree / dual NoC` 是同一系统问题的不同面

### 第 6 章：Autotuning

重点回答：

- 为什么需要自动选计划
- 为什么该模型不是黑盒，而是可解释、可缓存、可部署

## 11. 摘要草案

可以先用下面这个摘要骨架：

> Multi-Head Latent Attention (MLA) reduces KV-cache footprint and is particularly attractive for long-context inference. However, on explicit-dataflow accelerators such as Tenstorrent, the realized benefit of MLA depends not only on cache compression, but also on cache layout, communication topology, and execution scheduling. In this work, we present a unified mapping of MLA and Flash-style attention to the Tenstorrent stack, covering model integration, operator design, and low-level dataflow organization. We identify a structured design space spanning core layout, KV forwarding, multicast and tree-based communication, pipeline depth, and NoC usage. Based on this design space, we propose a hierarchical autotuning framework that combines feasibility filtering, analytical cost modeling, and top-K measured reranking to select near-oracle execution plans for different MLA workloads. Our study shows that the optimal strategy varies substantially across sequence lengths, batching regimes, and paging settings, and that autotuned planning consistently outperforms fixed execution policies while remaining practical through cached policy selection.

## 12. 执行顺序建议

建议按下面顺序推进，而不是把所有内容并行展开：

1. 先固定论文主线和 workload family
2. 先拿到 `MLA current baseline`
3. 先做 `layout / forwarding / hybrid / pipeline` 的固定策略消融
4. 再实现 `autotuner V1`
5. 用 `oracle gap` 判断 autotuner 是否足够强
6. 最后决定是否引入 `V2 residual ranker`

## 13. 风险与降级路径

这条线最大的风险有三个：

- autotuner 效果不明显
- 搜索开销太高
- 优化点太多，故事失焦

因此建议预先准备两条降级路径。

### 降级路径 A

如果 autotuner 效果一般，就把它降级成：

- `offline policy selector`
- 论文中的次要章节

此时主故事仍然成立：

- MLA on TT
- layout / communication / pipeline optimization
- performance analysis

### 降级路径 B

如果 `paged decode` 来不及做强，就把第一版论文收敛成：

- `single-chip non-causal prefill` 主论文
- `paged decode` 作为 future work 或小节补充

## 14. 最终建议

如果你的目标是：

- 既要做 `MLA 在 TT 上的应用`
- 又要提出一个 `寻优模型`
- 同时还想尽量保留论文性

那么最稳妥的落点就是：

**把文章写成一篇“MLA 在 Tenstorrent 上的系统映射与自动执行计划选择”论文，其中 `single-chip non-causal prefill` 是主优化平台，`paged decode` 是应用验证，autotuner 采用 `规则过滤 + cost model + top-K measured reranking + policy cache` 的分层结构。**

这会比“直接做一个黑盒 autotuner”更稳，也比“只讲实现没有自动调度”更像一篇完整论文。

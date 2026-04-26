# [Archived] Tenstorrent 上 MLA + Flash Attention 的扩展研究框架

> 已归档（2026-04-13）。
> 这份文档主要保存早期扩展路线与系统故事草稿，仍可作为历史参考，但不再作为当前论文执行总纲。
> 当前请优先参考 `current-docs.md` 与 `sf-mla-paper-positioning.md`；若需要回看旧版总纲，再继续阅读本文。

## 1. 文档定位

这份文档是在 `tenstorrent-mla-paper-ideas.md` 的基础上做进一步扩展。

原文档更偏“论文方向筛选”和“最小可投稿版本”，而这份扩展稿希望把你的想法收敛成一个更完整的系统故事。

## 1.1 2026-04 统一口径

当前应把这份文档理解为 `SF-MLA` 的总纲，而不是“MLA + Flash Attention 优化点列表”。  
统一口径如下：

- **核心命题**：`MLA decode on spatial accelerators` 是新的 `spatial mapping problem`
- **文章类型**：`operator characterization + dataflow design + cost model + DSE + architecture implication`
- **具体硬件语境**：Tenstorrent / Tensix，使 `reader / compute / writer`、`CB`、`dual NoC`、`hardware multicast` 这些问题显性化

因此，这份文档后面提到的三条主线更适合被改写为：

1. `Characterization`：为什么 MLA decode 在 spatial accelerator 上是新的 mapping 对象
2. `Design`：怎样围绕 shared-latent reuse、projection-attention pipeline 与 multicast/reduction topology 设计数据流
3. `Model + DSE`：怎样从 first-order 走向 second-order，并搜索最优 mapping

如果后续要继续写成论文、技术报告，或者拆成多个实现文档，这份稿子可以作为总纲。

---

## 2. 核心主线

一句话概括整个方向：

**MLA 把注意力的主要瓶颈从“巨大的传统 KV cache”转成了“更紧凑但更依赖布局、通信拓扑和异步流水组织的 latent restoration”，而 Tenstorrent 恰好暴露了显式 dataflow、L1 放置、NoC 路径和多核调度，因此可以把 MLA decode 写成一个“spatial mapping + pipeline coupling + DSE”的系统研究问题。**

这条主线有三个层次：

- **Characterization 层**：解释 MLA 为什么不再是传统 MHA 的小变体
- **Design 层**：围绕 dataflow、layout、multicast、pipeline 和 reduction 定义参数化 mapping
- **Model + DSE 层**：让不同输入、不同拓扑、不同资源约束下的最优配置可以被解释和搜索出来，而不是只靠手工调

所以最终要讲的不是“我在 TT 上跑了 MLA”，而是：

**MLA 在显式 dataflow 架构上的收益，不只是 KV cache 压缩本身，还来自对数据布局、通信拓扑、流水耦合和 mapping 切换的联合优化。**

---

## 3. 统一问题定义

我们希望回答的核心问题可以整理成 4 个：

### 3.1 映射问题

如何把 `MLA + Flash Attention` 映射到 Tenstorrent 的显式 dataflow 执行模型中？

这里重点关心：

- prefill 和 decode 的路径是否应该分开设计
- latent K/V、RoPE 分支和输出投影在 TT 上分别落到哪里
- `reader / compute / writer` 三阶段如何围绕 MLA 张量组织协同

### 3.2 通信问题

当 MLA 在长上下文和多核场景下展开时，瓶颈究竟在：

- DRAM 读取
- NoC 转发
- multicast 资格受限
- injector 热点
- tree reduction / tree forwarding
- 还是 compute 本身

这里需要特别强调：对 `decode` 来说，问题已经不再只是“减少一次 kernel 的 IO”，而是如何把 `latent restoration + attention + reduction` 组织成适合 spatial accelerator 的跨核执行计划。

### 3.3 调度问题

给定一个 workload 和一个硬件拓扑，怎样选择下面这些配置才最优：

- batch / head / q_chunk 的分组方式
- chain / multicast / tree 的通信模式
- pipeline 深度和 buffer 深度
- injector / sub-injector / forwarding 路由
- 是否需要 dummy iteration、轮转 injector 或双 NoC 分流

### 3.4 泛化问题

同一套手工规则能否同时适用于：

- 小 batch decode
- 长上下文 prefill
- paged KV cache
- 多芯片部署
- 不同 grid 形状和不同物理布局

如果不能，就需要引入自动寻优器。

换句话说，这篇文章不应被包装为“一个固定 kernel 在所有场景下都最好”，而应强调：

- 最优 mapping 会切换
- 切换原因可被 profiling 与模型解释
- 这正是 `SF-MLA` 要解决的问题

---

## 4. 第一部分：MLA + Flash Attention 在 Tenstorrent 上的应用

这一部分回答“它到底落在哪里、如何跑起来、哪些路径最值得做”。

## 4.1 建议的应用落点

结合当前仓库代码，最自然的落点有三层：

### 4.1.1 模型接入层

- `models/demos/deepseek_v3/tt/mla/`

这里适合放：

- 端到端 MLA block 的拼接逻辑
- prefill / decode 的上层调度
- paged cache、RoPE、输出投影等模型侧行为

### 4.1.2 通用算子层

- `ttnn/cpp/ttnn/operations/transformer/sdpa/`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`

这里适合放：

- `flash_mla_prefill`
- `chunked_flash_mla_prefill`
- `flash_multi_latent_attention_decode`
- `paged_flash_multi_latent_attention_decode`

也就是把 MLA 视作标准 SDPA 家族中的一个变体，而不是孤立 demo。

### 4.1.3 核心系统层

这里真正决定性能：

- `sdpa_program_factory.cpp`
- `reader_interleaved.cpp`
- `writer_interleaved.cpp`
- `sdpa.cpp`
- `sdpa_flash_decode.cpp`

这几层共同决定：

- 如何分 core
- K/V 如何读
- 是否做 forwarding
- 是否能 multicast
- 在 decode 中如何做归约与通信

## 4.2 适合优先做的应用场景

按性价比排序，我建议优先看下面 4 个场景：

### 场景 A：单芯片 decode

这是最适合先建立 baseline 的地方。

原因：

- MLA 的主要收益在 decode
- per-token latency 对系统优化最敏感
- 更容易看到 DRAM / NoC / kernel launch 的真实瓶颈

### 场景 B：长上下文 prefill

这是最适合讲 KV forwarding、multicast 和流水调度价值的地方。

原因：

- chunked/paged prefill 直接涉及 K/V 的读写模式
- sequence 变长后更容易出现 DRAM 和 NoC 的瓶颈切换
- 很适合讲“通信拓扑是否自适应”

### 场景 C：paged KV cache

这是最适合连接 vLLM 风格部署和系统调度的场景。

可以讲：

- page_table 带来的地址不连续性
- paged MLA cache 对 forwarding / multicast 的影响
- 分页粒度和 chunk 粒度之间的耦合

### 场景 D：多芯片 MLA

这个场景更难，但后续可以自然扩展到：

- ring distributed attention
- 多设备 joint attention
- chip 间 cache 切分和通信层级

不建议一开始就把它作为第一阶段主线，但它很适合放在“未来扩展”或“第二阶段成果”里。

## 4.3 这一部分最值得强调的研究问题

### 4.3.1 MLA 和标准 MHA/GQA 在 TT 上的系统差异

应重点突出：

- `KV latent cache` 更小，但布局自由度更高
- `RoPE` 分支和 `NoPE` 分支分离后，数据布局可能不再对称
- prefill 与 decode 的最优路径可能完全不同
- 标准 SDPA 的最优 chunking，不一定是 MLA 的最优 chunking

### 4.3.2 Flash Attention 风格在线 softmax 与 MLA latent-space 的耦合

这里可以讲：

- MLA 并不是简单换个张量形状
- latent-space V 与输出投影会改变最优融合边界
- 某些中间结果是更适合保留在 L1，还是更适合尽快投影后再传递

### 4.3.3 应用层的论文论点

这一部分可以形成的核心论点是：

**在 Tenstorrent 上，MLA + Flash Attention 的最佳实现不是直接照搬 GPU 风格 kernel，而是重新围绕显式 dataflow、L1 生命周期和通信组织来设计。**

---

## 5. 第二部分：MLA + FA 在 Tenstorrent 上的系统优化

这一部分是全文的核心技术部分，重点不是“模型结构创新”，而是“系统协同优化”。

## 5.1 当前系统瓶颈可以怎样理解

结合现有 SDPA 和 NoC 文档，当前最值得持续跟踪的瓶颈有：

- injector 从 DRAM 读 K/V 时容易成为单点热点
- forwarding 偏保守，multicast 资格要求严格
- 同一条 chain 上不同 core 的 `q_chunk_count` 不均匀时容易退化
- DRAM 读与 NoC 转发重叠不足
- 同一条路径上没有充分利用双 NoC
- 长链情况下单点分发压力过大
- 一旦物理布局不理想，multicast 覆盖率明显下降

换句话说，真正的问题不是“TT 能不能做 MLA”，而是：

**怎样让 MLA 的 latent cache 缩减收益不被 NoC 热点和调度不匹配吞掉。**

## 5.2 这一部分可以拆成 6 个优化方向

### 5.2.1 数据流流水化

目标：

- 让 `DRAM read -> L1 staging -> NoC forward -> compute` 更强地重叠

关键想法：

- prefetch distance
- triple buffering
- K/V 打包传输
- forward 当前 chunk 时预取下一个 chunk

重点关注：

- L1 占用是否还能接受
- semaphore 协议是否需要从“阻塞式”改成更显式的 credit/pipeline 协议

### 5.2.2 核布局与分组优化

目标：

- 让同一个 `(batch, head)` 的 worker 更容易形成利于 multicast 的物理布局

关键想法：

- row-aware mapping
- row-packed by head
- row-packed by batch-head
- sub-core grid 感知的布局

重点关注：

- 是否能提高 multicast 资格覆盖率
- 是否能减少 gap 和尾部不均匀带来的退化

### 5.2.3 单播 / 多播 / 混合拓扑

目标：

- 不再要求全局统一策略，而是让每条 chain 独立决定最合适的模式

关键想法：

- per-chain hybrid
- 对尾部不均匀用 dummy iteration 保持协议一致
- 局部 fallback 到 unicast，而不是全局退化

重点关注：

- 什么时候 multicast 更划算
- 什么时候 multicast 的资格成本和同步成本超过收益

### 5.2.4 树形 forwarding

目标：

- 把“一个 injector 覆盖所有 receiver”的中心化压力改造成层级式转发

关键想法：

- `tree2`
- `tree4`
- 行内 multicast + 行间 unicast 的 hybrid tree
- 复用 decode 路径中树形归约的思想，但把它翻转成树形广播

重点关注：

- 长链场景下是否能减少 injector 热点
- O(N) 压力是否能变成更接近 O(log N) 的层级化代价

### 5.2.5 轮转 injector

目标：

- 把持续压在单个核心上的 DRAM + NoC 热点摊开

关键想法：

- 每个 chunk 轮转
- 每 N 个 chunk 轮转
- 只在 detect hotspot 后轮转
- 结合 DRAM bank locality 做偏置轮转

重点关注：

- 轮转是否会引入额外同步成本
- 轮转粒度过细是否反而破坏 steady-state pipeline

### 5.2.6 双 NoC 分流

目标：

- 避免所有 forwarding 都拥挤在同一个通道上

关键想法：

- DRAM read 和 NoC write 分到不同通道
- K 与 V 分通道
- 不同子树或不同 receiver group 分通道

重点关注：

- 双通道是否真的改善拥塞
- 地址与物理布局是否允许稳定获益

## 5.3 如何把优化问题写成一个系统论文故事

建议不要把这部分写成“很多小技巧”，而要写成一个分层优化框架：

### 层 1：基础映射

- 把 MLA + FA 正确映射到 TT 上

### 层 2：通信组织

- 改善 K/V 的传输方式和转发拓扑

### 层 3：资源均衡

- 减少 injector、NoC、DRAM bank 的热点

### 层 4：策略自适应

- 不同 workload 和拓扑选择不同的最优配置

这样，NoC、多播、树形转发、双通道这些点就不再是分散 patch，而是一个统一系统框架的一部分。

## 5.4 这一部分建议形成的核心论点

可以收敛成下面这句：

**在 Tenstorrent 上，MLA + Flash Attention 的关键优化不在单个 matmul，而在 latent cache 流的组织方式，包括如何读、如何转发、如何多播、如何做层级化通信以及如何根据物理布局自适应。**

---

## 6. 第三部分：自动寻优模型

这一部分是把“手工调度经验”升级为“可自动选择最优配置”的关键。

## 6.1 为什么必须做 autotuning

如果只做人工规则，很快会碰到下面的问题：

- 不同 batch、seq len、head 数下，最优拓扑不同
- 不同物理 grid 和不同布局下，multicast 成功率不同
- 同一个优化在 short context 下有利，在 long context 下可能反而拖慢
- 不同链长、不同 page/block 大小下，最佳 pipeline 深度不一样

也就是说，这不是一个单一“最佳 kernel”的问题，而是：

**给定 workload + hardware topology，怎样自动选出最优执行计划。**

## 6.2 autotuner 的输入特征

我建议把输入特征分成三类：

### 6.2.1 workload 特征

- batch size
- `seq_len_q`
- `seq_len_kv`
- `num_q_heads`
- `num_kv_heads`
- `kv_lora_rank`
- `d_rope`
- causal / non-causal
- prefill / decode
- paged / non-paged
- block size / chunk size 候选

### 6.2.2 硬件特征

- `compute_with_storage_grid_size`
- 可用 `sub_core_grids`
- 物理核心行列分布
- DRAM bank 数量与 bank 到 worker 的 locality
- NoC 通道数量和分布
- 单核 L1 可用容量
- 是否单芯片 / 多芯片

### 6.2.3 中间状态特征

- chain 长度分布
- multicast 资格比例
- 估计的 injector hotspot 程度
- 预计 L1 占用
- 预计 DRAM / NoC traffic
- 预计 active core ratio

## 6.3 autotuner 的搜索空间

建议把搜索空间设计成“可分层剪枝”的形式。

### 6.3.1 并行划分

- `batch_parallel_factor`
- `head_parallel_factor`
- `q_parallel_factor`
- `q_chunk_size`
- `k_chunk_size`

### 6.3.2 核布局

- `default`
- `row_packed_by_head`
- `row_packed_by_batch_head`
- `bandwidth_balanced`
- `custom_sub_core_grids`

### 6.3.3 通信模式

- `global_unicast`
- `global_multicast`
- `per_chain_hybrid`
- `tree2`
- `tree4`
- `hybrid_tree`

### 6.3.4 流水和缓存

- `pipeline_depth`
- `prefetch_distance`
- `k_cb_depth`
- `v_cb_depth`
- `kv_transfer_mode`

### 6.3.5 热点均衡

- `injector_policy`
- `rotating_period`
- `sub_injector_policy`
- `dual_noc_policy`

## 6.4 autotuner 的目标函数

建议不要只优化单一 latency，而是采用多目标排序：

### 主目标

- decode：`latency / token`
- prefill：总 latency 或吞吐

### 次目标

- DRAM traffic
- NoC traffic
- injector stall
- active core 利用率
- L1 溢出风险
- multicast 覆盖率

### 约束

- 数值正确性
- L1 容量
- page / chunk 对齐要求
- kernel 可实现性

## 6.5 autotuner 的实现建议

最稳妥的实现方式不是“直接上黑盒搜索”，而是三层结构：

### 第 1 层：规则过滤

用显式规则去掉明显不可行的配置：

- L1 放不下
- block / chunk 不对齐
- multicast 几何条件不满足
- chain 太短不值得 tree 化

### 第 2 层：轻量 cost model

用可解释模型估计：

- DRAM 读取量
- NoC hop 数和通信体积
- injector 压力
- compute / memory 重叠程度

### 第 3 层：小规模实测搜索

只在 top-K 候选上做真实 benchmark：

- greedy / beam search
- Bayesian optimization
- cost-model-guided local search

## 6.6 autotuner 的最终输出

最终不只是输出一个数字，而是一份“执行计划”：

- 选定的 `program_config`
- core 分组与布局
- chain / multicast / tree 的拓扑配置
- pipeline 深度与 buffer 深度
- injector / sub-injector 选择
- NoC 通道使用策略

换句话说，autotuner 输出的是：

**给定 workload 和 topology 的最优 MLA execution schedule。**

## 6.7 这一部分建议形成的核心论点

这一部分可以收敛成：

**MLA 在 Tenstorrent 上的最优执行方式高度依赖输入和物理拓扑，因此需要从固定 heuristic 升级为 topology-aware 的自动调度与寻优。**

---

## 7. 把三部分串成一个统一论文故事

建议全文的故事线不是并列三章，而是层层递进：

## 7.1 故事起点

`MLA + Flash Attention` 很适合长上下文推理，但在显式 dataflow 架构上，收益是否能真正兑现，取决于数据流和通信组织。

## 7.2 第一层贡献：映射与 baseline

先建立：

- MLA + FA 在 TT 上的 prefill / decode 基线
- 对比 MHA / GQA / MLA 的基本收益
- 明确瓶颈从 DRAM 逐步转向 NoC、布局和调度

## 7.3 第二层贡献：系统优化

再证明：

- forwarding / multicast / tree / dual-NoC / rotating injector 可以显著改变瓶颈
- 同样的 MLA 算法，不同通信组织会得到明显不同的性能曲线

## 7.4 第三层贡献：自动寻优

最后说明：

- 最优配置不是固定的
- 需要根据 workload 与 topology 自动选择
- autotuner 可以稳定优于单一人工策略

## 7.5 最终论文论点

最终可以落成下面这个更完整的命题：

**Tenstorrent 上的 MLA + Flash Attention 是一个“算子映射、NoC 协同优化和 topology-aware 自动调度”三者耦合的问题；只有同时优化这三层，MLA 的理论 cache 优势才能稳定转化为真实推理收益。**

---

## 8. 推荐的实验框架

为了让这份想法最终能落到论文或技术报告，我建议实验矩阵至少覆盖下面几个维度。

## 8.1 workload 维度

- decode：`B=1/2/4/8`
- prefill：`S=1K/4K/16K/32K/128K`
- `num_q_heads / num_kv_heads`
- `kv_lora_rank`
- `d_rope`
- paged / non-paged

## 8.2 系统配置维度

- 单芯片 vs 多芯片
- unicast vs multicast vs hybrid vs tree
- fixed injector vs rotating injector
- single NoC vs dual NoC
- baseline layout vs row-aware layout

## 8.3 对比基线

- 标准 MHA / GQA / MLA
- 无 forwarding 的 MLA
- 手工 tuned 版本
- autotuned 版本

## 8.4 指标

- latency / token
- prefill latency
- throughput
- DRAM bytes
- NoC bytes / hop
- multicast 覆盖率
- injector stall 占比
- L1 使用率
- 数值误差

---

## 9. 最小可落地版本与扩展版本

## 9.1 最小可落地版本

如果你想先快速推进，我建议先做：

1. MLA + Flash Attention baseline
2. NoC / multicast / KV forwarding 的一到两个核心优化
3. 一个轻量 cost model + 小规模搜索器

这已经足够形成一个很完整的技术故事。

## 9.2 扩展版本

后续再继续加：

- tree forwarding
- rotating injector
- dual NoC
- 多芯片 MLA
- 在线 runtime selector

这样可以从“可投稿版本”自然扩展到“更完整的系统项目”。

---

## 10. 与现有文档的关系

这份扩展稿建议和下面几份文档配合使用：

- `tenstorrent-mla-paper-ideas.md`
  - 负责论文方向筛选、投稿主线和最小版本判断
- `ref/tt-metal_SDPA算子梳理.md`
  - 负责建立 API、实现目录和代码入口全景图
- `ref/tt-metal-flash-attention-dataflow-parallelism.md`
  - 负责解释当前 Flash Attention 的数据流和并行切分
- `ref/NoC_与多播代码实现详解.md`
  - 负责说明 NoC 单播、多播和 runtime args 的代码细节
- `ref/tt-metal_kv_forwarding_design_space_autotuner.md`
  - 负责把 forwarding 的优化方向整理成设计空间和 autotuner 框架
- `ref/tt-metal_kv_forwarding_iccad_experiment_plan.md`
  - 负责实验矩阵、图表设计和最小投稿版本

换句话说，这份文档本身不代替这些细化文档，而是负责把它们组织成一个统一的总框架。

---

## 11. 下一步建议

如果要继续往下写，我建议按下面顺序拆成 3 份子文档：

1. `mla-application-and-mapping.md`
   - 专门写 MLA + FA 在 TT 上的应用场景、代码落点和基线路径
2. `mla-noc-multicast-optimization.md`
   - 专门写 NoC、KV forwarding、multicast、tree、dual NoC 和流水优化
3. `mla-autotuner-framework.md`
   - 专门写特征、搜索空间、cost model 和 runtime selection

这样后续不管是写技术报告、研究计划，还是落到具体实现，都更容易继续扩展。

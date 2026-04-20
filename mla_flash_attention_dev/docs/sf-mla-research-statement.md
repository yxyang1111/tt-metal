# SF-MLA Research Statement

本文关注的问题是：如何为 `MLA decode` 定义一个真正适合 `spatial accelerator` 的 dataflow abstraction，并在这个 abstraction 下定义和求解优化问题。`MLA` 很重要，spatial accelerator 也很重要，但与 `MHA` 相比，`MLA` 引入了更复杂的跨-head 数据重用、latent restoration 和数据搬运，因此性能不再只取决于单个 kernel 的实现，而更取决于整套 dataflow 如何构建、映射并与硬件协同。基于这一点，`SF-MLA` 的核心不是再讲一个更快的 kernel，而是至少要定义一个能够包含当前实验性 `FlashMLA` 的 parameterized dataflow abstraction；如果后续实验支持，也可以在同一 abstraction 下给出比当前 `FlashMLA` 更好的 design instance。若暂时做不到后者，论文至少要把 `FlashMLA` 这个 dataflow design 本身讲清楚，并通过详细 profiling 解释它在不同 shape 下的收益、瓶颈与支持边界。

## 1. 关键词

- `Multi-Latent Attention`
- `Spatial Accelerators`
- `Dataflow Design`
- `Design Space Exploration`
- `Pipeline Coupling`
- `Hardware Multicast`
- `Tenstorrent`

## 2. 背景

- **定位**：这篇文章的重点不是“再做一个更快的 FlashMLA kernel”，而是围绕一个能够容纳 `FlashMLA` 的 `MLA spatial dataflow abstraction` 讲清楚三件事：这个 abstraction / design 是什么，它为什么适合 `MLA + TT`，以及在这个 abstraction 下应该优化什么。它更像一篇围绕 `dataflow abstraction / design + optimization problem formulation + empirical explanation + architecture implication` 展开的系统/架构论文。
- **动机**：`MLA` 通过 latent `KV` 压缩降低 `KV cache` 开销，但 decode 时必须在线恢复 `per-head K/V`，因此它不再是传统 `MHA decode` 那种相对单纯的 memory-bound 问题，而是进入了 **compute-memory mixed regime**。相比 `MHA`，`MLA` 带来更复杂的跨-head 数据重用与数据搬运。
- **采用现状**：根据目前可明确查证的一手论文，`MLA` 的原生、大规模、公开采用主要集中在 `DeepSeek` 家族：`DeepSeek-V2` 将 `MLA` 作为核心架构引入，`DeepSeek-V3` 延续了这一设计。`DeepSeek` 之外，公开可见的进展目前更多是研究性迁移与验证，而不是主流模型家族的原生采用，例如 `MHA2MLA` 将 `Llama` 类模型迁移到 `MLA`，`TransMLA` 将 `GQA` 预训练模型转换为 `MLA`，另有工作在小语言模型上系统研究 `MLA` 的效率-质量权衡。因此，更准确的表述是：`MLA` 的公开原生采用目前主要由 `DeepSeek` 代表，而 `DeepSeek` 之外的公开进展仍以迁移、复现和中小规模实验为主。相关引用：`DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model` (`arXiv:2405.04434`), `DeepSeek-V3 Technical Report` (`arXiv:2412.19437`), Ji et al., `Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs` (`ACL 2025`), `TransMLA: Multi-Head Latent Attention Is All You Need` (`arXiv:2502.07864`), `Latent Multi-Head Attention for Small Language Models` (`arXiv:2506.09342`).
- **硬件语境**：spatial accelerator 的关键特征是分布式 `SRAM`、decoupled `reader / compute / writer`、以及 `NoC` 上的 multicast / reduction。在这种架构上，优化重点不仅是 tile 大小，还包括数据驻留、通信拓扑、流水深度和跨核协作方式。
- **当前 dataflow**：当前最值得论文化的对象，不是一个空泛的“大 mapping space”，而是把 `exp FlashMLA` 抽象成一个 dataflow family。当前的 concrete instance 可以概括为一个以 `S-block` 为基本空间单元的 decode dataflow：`Q` 由 output core 持有并向 worker 扩散，`K` 从 `ND-sharded DRAM` 以 page-level pipeline 读取并在 block 内 multicast，`V` 直接从 `K` buffer 派生而不是单独 staging，最后通过拓扑感知的 tree reduction 汇总输出。
- **dataflow 的好处**：这套 dataflow 的核心价值在于减少重复 `DRAM` 读取，显式利用 `K/V overlap` 和 on-chip multicast，把 `K read / multicast / compute / reduction` 串成更深的流水，并让 reduction 与 placement/topology 对齐。因此论文真正要说明的不是“它快了多少”，而是“它为什么在 `TT + MLA` 上是一个好的 dataflow”，或者更具体地说，为什么它比更弱的数据流组织更合理。
- **shape 支持边界**：如果当前 `FlashMLA` 在不同 `seq_len / batch / q_shard / head-group / core` 配置下支持不好、收益不稳定，或者瓶颈模式明显变化，这本身就是论文必须解释的现象。详细 profiling 不是附属工作，而是说明这套 dataflow 在哪些 shape 下成立、在哪些 shape 下暴露边界的核心证据。按当前 `Part II`，这类证据已经至少分成两层：一层是 `decode_256~32k` proxy sweep 给出的 phase transition 与 `K` / writer backpressure 迁移，另一层是 representative direct probe 给出的 corrected-runtime `A1/A2/A3` 算术利用率与 `B/C/D` 容量边界。
- **当前最强实证锚点**：当前最扎实的 direct 结果已经能支持下面几件事：`decode` 会从 compute-critical 迁移到 reader-writer coupling；corrected-runtime `A1/A2/A3` 的 `PM FPU util` 会从 `low-teens` 抬升到 `mid-20s`，但 `TT / DeepSeek-4c` 的差值始终很小；`4c/8c` 的一阶区别主要体现在 q-core admission boundary，而不是所有已 fit 点上的普适加速。因此论文更稳妥的写法是：`SF-MLA` 的优势首先来自 mapping / dataflow 对 front-side starvation 的缓解，而不是把 decode 算术利用率抬到一个完全不同的饱和区间。
- **优化问题**：借用 dataflow 文献中的表述，`dataflow` 可以理解为一组 `mapping rules`，它定义了某个硬件设计所支持的 `supported mappings` 子空间。本文不把优化问题写成“所有 attention 实现的通用搜索”，而是写成：在这套 dataflow abstraction 或具体 design 下，如何联合选择 `seq` chunk/page 切分、`S-block` 分配、`DRAM bank` 对齐、core 角色、buffer 深度、multicast / reduction 调度等参数，使吞吐最大化并尽量降低 reader / writer coupling。
- **研究空白**：`FlashAttention`、`FlashDecoding`、`FlashMLA` 主要面向 GPU；`FLAT` 和 `FlatAttention` 提供了 spatial attention 的启发，但没有系统把 `TT + MLA` 上这种具体 dataflow 讲清楚，也没有围绕这套 dataflow 明确定义 optimization problem、再通过 detailed profiling 解释其收益来源与 shape 边界。
- **TT 语境**：选择 `Tenstorrent Wormhole/Blackhole` 作为 concrete substrate 很合适，因为 Tensix 的 `NCRISC / BRISC / TRISC`、`CB`、dual `NoC` 和 hardware multicast，正好把这些 mapping rules 和耦合问题显性化。

## 3. 预期贡献

- **dataflow abstraction / design 贡献**：定义一个能够包含当前 `FlashMLA` 的 parameterized spatial dataflow abstraction，并把 `S-block` 空间分工、`Q` 扩散、paged `K` 读取、block 内 multicast、从 `K` 直接派生 `V`、以及 topology-aware tree reduction 抽象成可组合的 design elements。当前 `FlashMLA` 是这个 abstraction 的一个 concrete instance；如果后续实验支持，也可以在同一 abstraction 下给出更优的 design instance。
- **优化问题贡献**：在这套 dataflow abstraction 或 design 下，定义一个清晰的 optimization problem，而不是泛泛地谈大搜索空间。具体来说，就是给定 workload 与硬件资源，在受支持的 mapping rules 内联合优化 chunk/page 切分、`S-block` / core 分配、bank 对齐、buffer 深度、multicast / reduction 调度等参数。论文的重点是把这个问题定义清楚，而不是把求解器本身包装成主要创新。
- **profiling / 解释与架构含义贡献**：通过 detailed profiling、cost model、architecture simulation 和 evaluation，解释当前 `FlashMLA` 或其变体在不同 shape 下何时有效、收益来自哪里、哪些 shape 下支持不好或暴露明显边界，以及瓶颈为什么会从 compute-critical 转向 reader-writer coupling，并据此提炼对未来 spatial accelerator 组织的启发。按当前证据链，最强的 profiling claim 应至少覆盖三件事：`decode` 的 bottleneck phase transition、representative `A1/A2/A3` 上 corrected-runtime 算术利用率曲线，以及 `4c -> 8c` 主要解锁容量墙而非已 fit 点普适加速。

## 4. 方法

- **Characterization**：先分析 `MLA` 算子特征和当前 `TT` 路径，说明 `MLA` 不是 `MHA` 的小变体，而是需要显式 dataflow 设计的对象。这里最重要的是把 `decode` 的 phase transition、compute share 与真实算术利用率的区别、以及 `K` 供给和 writer/reduction backpressure 的迁移讲清楚。
- **Dataflow Abstraction / Design**：把当前实验性 `FlashMLA` 路径抽象成一个清晰的 dataflow abstraction，明确它的基本单元、数据路径、通信拓扑、流水组织，以及它相对于更弱 baseline 的关键优势；如果存在更好的变体，再把它写成同一 abstraction 下的新 design instance。
- **Optimization Problem**：在这个 dataflow abstraction 下形式化优化问题，明确目标函数、设计变量与约束条件；搜索/auto-tuning 只作为辅助求解手段，在受支持的 mappings 中找到代表性高质量配置。
- **Evaluation**：评测重点不是只报一个 speedup，而是回答四类问题：这个 dataflow 比弱基线好在哪里；当前 `FlashMLA` 在不同 shape 下哪里有效、哪里支持不好；在什么 workload / 资源条件下最优配置会切换；这些现象对未来硬件组织意味着什么。现阶段尤其要把 representative direct 证据用好，即 `A1/A2/A3` 的 corrected-runtime `PM/perf-counter`、`TT vs DeepSeek-4c` 的 starvation 差异，以及 `B/C/D` 上 `4c/8c` 的容量边界。

## 5. To-do

- 先决定最终 claim：是“定义一个能包含 `FlashMLA` 的 dataflow abstraction”，还是“在同一 abstraction 下给出一个比 `FlashMLA` 更好的 design instance”；这两个层次不能混着写。
- 如果当前没有更强 design，就把 `FlashMLA` 这个 dataflow design 本身讲清楚，画出正式图，并把它的优势和边界写扎实。
- 在已有 `A1/A2/A3` representative arithmetic-utilization curve 之上，继续 densify `seq_len` 维度，优先补 `8k/16k` 或容量边界点，回答当前 corrected-runtime `PM/perf-counter` 结论是不是只在 representative points 成立，还是能更接近 dense sweep。
- 把这套 dataflow 的“好处”写得更尖锐，明确对比弱基线时到底减少了哪些读写、解锁了哪些 overlap、缓解了哪些热点。
- 把优化问题写得更形式化，明确目标函数、变量、约束和求解方式；搜索/auto-tuning 只保留为必要的求解实现，不作为核心贡献展开。
- 更尖锐地写清楚与 `FLAT`、`FlatAttention`、`FlashDecoding`、`FlashMLA` 的差异，避免被看成已有工作的 MLA 变体。
- 补强 baseline 叙事，让读者更直观看到“无 multicast / 浅流水 / first-order-only / untuned mapping”分别损失多少。
- 把“当前硬件测量”到“未来架构推演”的证据链写得更严密，特别是 second-order terms 迁移到新架构时的适用边界。
- 尽快把 placeholder figures 换成正式图，并增加 limitation discussion，例如当前主要覆盖 decode、batch size 较小、模型参数较固定、Blackhole 结果以 projection / simulation 为主。

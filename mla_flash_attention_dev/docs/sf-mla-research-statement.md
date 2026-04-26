# SF-MLA Research Statement

本文关注的问题是：如何为 `MLA decode` 定义一个真正适合 `spatial accelerator` 的 dataflow abstraction，并在这个 abstraction 下定义和求解优化问题。`MLA decode` 重要，因为它是 `DeepSeek-V3 / R1` 这一档主流旗舰模型在线服务的主要计算路径，而 decode per-token latency 又直接决定服务成本与 SLA；`spatial accelerator` 重要，因为它用分布式 `SRAM`、显式 `NoC` 和硬件 multicast 换取了更高的 `perf / $` 与 `perf / W`，正在成为 LLM 推理新增算力供给的主力之一。但与 `MHA / GQA` 相比，`MLA` 在 decode 上把单路 latent 扩展回 `Hq` 个 Q head 的数据路径（即 latent restoration 和跨-head 数据重用）显著改变了算术强度、数据搬运拓扑与流水耦合方式，具体量级详见 §2；因此在 spatial accelerator 上，性能不再只取决于单个 kernel 的实现，而更取决于整套 dataflow 如何构建、映射并与硬件协同。基于这一点，`SF-MLA` 的核心不是再讲一个更快的 kernel，而是至少要定义一个能够包含当前实验性 `FlashMLA` 的 parameterized dataflow abstraction；如果后续实验支持，也可以在同一 abstraction 下给出比当前 `FlashMLA` 更好的 design instance。若暂时做不到后者，论文至少要把 `FlashMLA` 这个 dataflow design 本身讲清楚，并通过详细 profiling 解释它在不同 shape 下的收益、瓶颈与支持边界。

## 1. 关键词

- `Multi-Latent Attention`
- `Spatial Accelerators`
- `Dataflow Design`
- `Design Space Exploration`
- `Pipeline Coupling`
- `Hardware Multicast`
- `Tenstorrent`

## 2. 背景

**论文定位**。本文不是"再做一个更快的 `FlashMLA` kernel"，而是围绕一个能够容纳 `FlashMLA` 的 `MLA spatial dataflow abstraction` 讲清楚三件事：这个 abstraction / design 是什么、它为什么适合 `MLA + TT`、以及在它之下应该优化什么。整体属于 `dataflow abstraction / design + optimization problem formulation + empirical explanation + architecture implication` 类型的系统/架构论文。

**`MLA decode` 的重要性**。LLM 在线服务里，`prefill` 在请求入队时只跑一次、成本被整段回答分摊；真正被反复执行、主导 per-token latency 与单位 token 成本的是 `decode`——它在一次用户回答里会被连续重复几百到几千次。而 `DeepSeek-V3 / R1` 这一档当前主流的旗舰开源模型又把 `MLA` 当作 attention 主路径，每产出一个 token 都完整跑一次 `MLA decode`。也就是说，`MLA decode` 就是这一档模型在在线部署里被执行最频繁的那段计算，优化它直接决定它们的出字速度、吞吐和每 token 成本，而不是一个离实际部署较远的 benchmark。结构上，`MLA` 把每个 head 各自的 `K / V` 压成一份所有 head 共用的低维 latent——缓存体量因此小了大约一个数量级，但代价是这条 latent 必须在片上被所有 head 共享重用才能恢复各自的注意力，由此带来两条一阶变化：数据通路从"每个 head 各读各的独立流"变成"读一次、片上分发给所有 head 反复使用"的共享流；每从片外搬一份 latent 要配套做的计算比 `MHA / GQA` 多出好几倍，`decode` 不再只被带宽卡住，而要同时被算力约束。这两点合起来，使得 `MLA decode` 不再是被 `FlashAttention / FlashDecoding` 这类 kernel 写法基本"吃掉"的 `MHA decode` 问题，而是算术强度、数据重用拓扑与流水耦合方式都发生一阶变化的新对象。

**`spatial accelerator` 的重要性**。以 `Tenstorrent Wormhole / Blackhole` 为代表的一批 spatial accelerator（以及 `Cerebras`、`Groq`、若干国内同类架构）的设计卖点是 distributed `SRAM` + explicit `NoC` + hardware multicast，在 matmul-heavy、可被 mapping 精细控制的工作负载上能把 `perf / $` 和 `perf / W` 做到高于通用 GPU，是近年 LLM 推理新增算力供给的主力之一。对 `MLA decode` 这种"一条共享 latent 要分发给所有 Q head 反复使用"的工作负载，这些硬件特征恰好能被组织成"读一次 latent，on-chip reuse 到所有 Q head"，在架构级别上与 `MLA` 的数据重用模式对齐。更关键的是，与 GPU 的隐式 `L2` / `warp scheduler` / `cache coherence` 不同，spatial accelerator 把 tile 大小、数据驻留、通信拓扑、流水深度、跨核协作这些旋钮全部暴露到算子层——`MLA decode` 新增的那些一阶耦合正好在这里被显性化为可以被刻意设计和优化的变量，得到的 dataflow design 与 cost model 也能直接反哺未来 accelerator 在 `NoC` 拓扑、bank 数、core 角色预留等方面的组织方式。因此把 `MLA decode` 放在 spatial accelerator 上研究，不是"任选一个非 GPU 平台"，而是工作负载特征与硬件特征双向匹配的问题选择；具体到 substrate，本文选 `Tenstorrent Wormhole / Blackhole`，因为 Tensix 的 `NCRISC / BRISC / TRISC`、`CB`、dual `NoC` 和硬件 multicast 让上述耦合问题全部落到可控层面，是写清这套 dataflow 最合适的 concrete platform。

**采用现状与研究空白**。按目前可查证的一手文献，`MLA` 的原生、大规模、公开采用主要集中在 `DeepSeek` 家族：`DeepSeek-V2` 将其作为核心架构引入，`DeepSeek-V3` 延续了该设计；`DeepSeek` 之外的公开进展以迁移与复现为主——`MHA2MLA` 把 `Llama` 类模型迁移到 `MLA`，`TransMLA` 把 `GQA` 预训练模型转换为 `MLA`，另有若干工作在小语言模型上研究 `MLA` 的效率-质量权衡（`DeepSeek-V2`, `arXiv:2405.04434`；`DeepSeek-V3`, `arXiv:2412.19437`；Ji et al., ACL 2025；`TransMLA`, `arXiv:2502.07864`；`arXiv:2506.09342`）。算子侧，`FlashAttention`、`FlashDecoding`、`FlashMLA` 都以 GPU 为主要目标；spatial side 上 `FLAT` 和 `FlatAttention` 提供了启发，但没有系统写清 `TT + MLA` 这种具体 dataflow，也没有围绕它明确定义 optimization problem 并通过 detailed profiling 解释其收益来源与 shape 边界。这就是本文要填的位置。

**研究对象：当前实验性 `FlashMLA` dataflow 及其实证锚点**。本文最值得论文化的对象不是一个空泛的"大 mapping space"，而是把当前实验性 `FlashMLA` 抽象成一个 dataflow family。它的 concrete instance 是一个以 `S-block` 为空间基本单元的 decode dataflow：`Q` 由 output core 持有并向 worker 扩散，`K` 从 `ND-sharded DRAM` 按 page-level pipeline 读取并在 block 内 multicast，`V` 直接从 `K` buffer 派生而不单独 staging，最后通过 topology-aware tree reduction 汇总输出。其核心价值是减少重复 `DRAM` 读取、显式利用 `K / V overlap` 与 on-chip multicast、把 `K read / multicast / compute / reduction` 串成更深的流水，并让 reduction 与物理 placement 对齐。因此论文要说的不是"它快了多少"，而是它为什么在 `TT + MLA` 上是一个好的 dataflow，以及它在不同 `seq_len / batch / q_shard / head-group / core` 配置下的收益边界。按当前 `Part II` 的证据链，`decode` 瓶颈会从 compute-critical 迁移到 reader-writer coupling；corrected-runtime `A1 / A2 / A3` 的 `PM FPU util` 会从 low-teens 抬升到 mid-20s，但 `TT / DeepSeek-4c` 的差值始终很小；`4c → 8c` 的一阶区别主要体现在 q-core admission boundary，而不是所有已 fit 点上的普适加速。综合起来，更稳妥的论文立场是：`SF-MLA` 的优势首先来自 mapping / dataflow 对 front-side starvation 的缓解，而不是把 decode 算术利用率抬到完全不同的饱和区间。

**优化问题**。借用 dataflow 文献中的表述，`dataflow` 可以看作一组 `mapping rules`，它定义了某个硬件设计所支持的 `supported mappings` 子空间。本文不把优化问题写成"所有 attention 实现的通用搜索"，而是写成：在上述 dataflow abstraction 或 concrete design 下，如何联合选择 `seq` chunk / page 切分、`S-block` 分配、`DRAM bank` 对齐、core 角色、buffer 深度、multicast / reduction 调度等参数，使吞吐最大化并尽量降低 reader / writer coupling。

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

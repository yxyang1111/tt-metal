# [Archived] Tenstorrent 上做 MLA 的论文方向整理

> 已归档（2026-04-13）。
> 这份文档主要保存早期的论文方向 brainstorm 和候选打包方式，不再作为当前论文定位的主依据。
> 当前请优先参考 `current-docs.md` 与 `sf-mla-paper-positioning.md`；若需要回看早期选题演化，再继续阅读本文。

## 1. 先给结论

如果目标是 **按当前材料尽快收束成一篇完整论文**，最合适的路线不是去发明一个全新的 MLA 变体，也不是把故事写成“再做一个更快的 FlashMLA kernel”，而是做：

`MLA decode on spatial accelerators` 的问题定义、数据流设计、建模和设计空间探索。

当前最推荐的方向是：

**`MLA as a Spatial Mapping Problem: Characterization, Dataflow Design, and DSE on Tenstorrent`**

原因很简单：

- MLA 的主要收益点本来就在 `KV cache 压缩` 和 `decode 内存流量下降`
- Tenstorrent 的公开软件栈正好强调 `显式 dataflow、L1/DRAM 放置、NoC 访存、tile/shard`
- 这种组合最适合写成一篇 **operator characterization + dataflow design + cost model + DSE + architecture implication** 的系统/架构论文

一句话建议：

**第一篇只做 inference，优先做 decode-only，不碰训练，不碰 backward，不要一开始就追求全模型复现；prefill 保留为 supporting evidence，而不是唯一主战场。**

### 1.1 当前统一口径

本文是早期“论文方向池”文档。当前应以 `sf-mla-paper-positioning.md` 为统一口径，本文中的若干“方向 A-H”更适合理解成：

- 可被纳入 `SF-MLA` 的子问题或实验资产
- 而不是 8 条彼此并列、竞争主标题的路线

---

## 2. 为什么 MLA 和 Tenstorrent 很搭

从公开资料看，Tenstorrent 的几个关键点是：

- Tensix 采用显式的数据流模式，核心思路是 `reader / compute / writer`
- 片上存储以 `L1/SRAM` 为中心，每个 Tensix tile 有约 `1.5MB SRAM`
- 访存不是统一线性地址，而是显式的 `x, y, local_address`
- 数据布局上支持 `interleaved` 和 `sharded`
- 计算以 `32x32 tile` 为自然粒度
- TT-NN/TT-Metalium 已经提供 attention、profiling、kernel API 等基础设施

而 MLA 的关键点是：

- 用压缩后的 latent 表示替代传统大体积 KV cache
- decode 阶段通常更偏 `memory-bound`，尤其是长上下文时
- FlashMLA 的公开结果也说明 MLA decode 既可能是带宽瓶颈，也可能是算力瓶颈

所以对 Tenstorrent 来说，MLA 的研究价值非常自然：

- 能不能减少 DRAM 读取
- 能不能降低 NoC 流量
- 能不能把 latent cache 和 rotary 分支放到更好的 memory layout
- 能不能把 decode 路径重新表述成更适合 Tensix dataflow 的 spatial mapping 问题

这几点都很像一篇硬件/系统论文该回答的问题，但真正的论文主张应从“单点优化”升级为：

- 为什么 MLA decode 已经不是传统 MHA 的小变体
- 为什么它在 spatial accelerator 上会暴露新的 mapping、multicast 和 pipeline coupling 问题
- 为什么需要可解释模型和 DSE，而不是只做局部 kernel 优化

---

## 3. 适合做的方向总览

> 2026-04 更新：下面这些方向现在更适合作为 `SF-MLA` 的组成模块，而不是并列的论文题目候选。当前最优先的组合仍是 `decode characterization + dataflow design + model/DSE`。

| 方向 | 核心内容 | 出稿速度 | 工程难度 | 论文性 | 我的建议 |
| --- | --- | --- | --- | --- | --- |
| 方向 A | `MLA decode baseline + profiling` | 很快 | 低 | 中 | 最适合起步 |
| 方向 B | `latent KV cache 的 sharding / layout 优化` | 快 | 中 | 高 | 最推荐 |
| 方向 C | `fused MLA decode kernel` | 中 | 高 | 高 | 第二阶段做 |
| 方向 D | `MHA / GQA / MLA 在 Tenstorrent 上的系统对比` | 很快 | 低 | 中高 | 适合第一篇 |
| 方向 E | `低精度 MLA cache（BF16/FP8/INT8）` | 中 | 中高 | 高 | 有卖点但有风险 |
| 方向 F | `prefill / decode 分路径调度优化` | 快 | 中 | 中高 | 很实用 |
| 方向 G | `MLA 的性能模型 / roofline / traffic model` | 很快 | 中 | 高 | 非常适合快速投稿 |
| 方向 H | `多芯片长上下文 MLA` | 慢 | 高 | 高 | 不适合第一篇 |

---

## 4. 每个方向具体可以做什么

### 方向 A：MLA decode baseline + profiling

这是最稳、最快的入门方向。

你可以做的事情：

- 在 TT-NN 上先搭一个功能正确的 MLA decode 路径
- 至少把 `cache_kv` 和 `cache_rk` 两部分逻辑跑通
- 做单层 attention block 的 microbenchmark
- 用 profiling 工具定位瓶颈：DRAM、NoC、tile format conversion、kernel launch、L1 容量

可以写成的论文点：

- “Tenstorrent 上 MLA decode 的真实瓶颈是什么”
- “MLA 相比 MHA/GQA 在该架构上究竟受益在哪里”
- “长上下文下 latency/token 的主要决定因素是什么”

适合投稿：

- workshop
- short paper
- arXiv 先发

我的评价：

**这是最适合立刻开工的第一步。**

### 方向 B：latent KV cache 的 sharding / layout 优化

这是我最推荐的真正“可发论文”方向。

你可以做的事情：

- 比较 `interleaved DRAM`、`interleaved L1`、`sharded L1`、`hybrid layout`
- 分别测试 `cache_kv` 和 `cache_rk` 是否应采用不同布局
- 研究按 `sequence`、`head`、`latent dimension` 三种维度切分时的性能差异
- 测试 NoC contention 是否会成为长上下文 decode 的主瓶颈

为什么这个方向强：

- 很 Tenstorrent，别的平台不一定有同样明显的 data placement 问题
- 很 MLA，latent cache 正好给了你新的布局自由度
- 容易形成清晰实验：同一个算法，不同 memory mapping

可以形成的核心论点：

**MLA 在显式 dataflow 架构上的收益，不只来自 cache 压缩本身，还来自更好的缓存布局和 NoC 交通组织。**

### 方向 C：fused MLA decode kernel

这是更偏 kernel co-design 的方向，但在当前口径下它应被视为：

- 某个特定 mapping / dataflow point
- `SF-MLA Design` 部分的一个实现实例
- 而不是论文唯一卖点

你可以做的事情：

- 把 latent 读取、投影、attention score、online softmax、输出累积做成更深的融合
- 尽量减少中间 tensor 回写 DRAM/L1
- 用 reader/compute/writer overlap 隐藏访存延迟
- 针对 tile 粒度设计更合适的分块

为什么值得做：

- Tenstorrent 的编程模型天然适合讲 `dataflow fusion`
- decode 场景下每个 token 的启动开销很敏感，融合通常更容易看到效果

风险：

- 工程难度明显上升
- 如果 baseline 还没稳定，容易被 kernel debug 拖很久

我的建议：

**把它作为第二阶段，而不是第一阶段。**

### 方向 D：MHA / GQA / MLA 的系统对比

这是最容易包装成论文的一种写法。

你可以做的事情：

- 在同一套 Tenstorrent 平台上对比 `MHA`、`GQA/MQA`、`MLA`
- 比较不同上下文长度下的 `latency/token`、吞吐、显存/内存占用、带宽利用
- 看哪一种注意力机制更匹配 Tenstorrent 的 memory hierarchy

论文价值：

- 不是单纯做“我写了个 MLA”
- 而是回答“哪类 attention 更适合这种显式 dataflow 架构”

这个方向非常适合和方向 A/B 结合，形成：

**算法机制对比 + Tenstorrent 特化实现 + 性能分析**

### 方向 E：低精度 MLA cache

这个方向有潜在卖点，因为 MLA 本来就很适合做 cache 压缩。

你可以做的事情：

- 做 `BF16 cache` 对比 `FP8/INT8 cache`
- 保持 compute 为 BF16，只压缩存储
- 单独研究 `NoPE` 部分与 `RoPE` 部分是否需要不同精度
- 分析量化误差与带宽节省之间的平衡

为什么它适合 Tenstorrent：

- Tenstorrent 的 unpacker/packer 本身就强调数据格式转换
- 可以把“低精度存储 + 较高精度计算”作为一个平台特性点来讲

风险：

- 精度与 kernel 支持的细节需要较多验证
- 如果没有现成的 FP8/INT8 路径，工程周期可能变长

### 方向 F：prefill / decode 分路径调度

这个方向很实用，也比较容易写。

你可以做的事情：

- prefill 和 decode 使用不同 kernel 版本
- prefill 偏吞吐，decode 偏单 token latency
- 研究不同 batch、不同 context length 下最优 core allocation
- 观察什么时候 MLA 把瓶颈从 DRAM 变成 NoC 或 compute

适合写成：

- runtime scheduling
- kernel selection policy
- system optimization

### 方向 G：性能模型 / roofline / traffic model

如果你想 **最快形成论文故事**，这是很好的选项。

你可以做的事情：

- 建立 MHA/GQA/MLA 在 Tenstorrent 上的理论 traffic model
- 估计每 token 的 DRAM 读取、NoC 往返、L1 占用
- 用真实 profiling 数据校准模型
- 找到不同 sequence length 下的瓶颈切换点

为什么强：

- 不要求你第一天就写出最强 kernel
- 只要有一个可运行 baseline，就能做大量分析
- 很适合 case study、benchmark paper、workshop

适合形成的论点：

**MLA 在显式 dataflow 架构上的收益区间是可建模的，并且与 cache layout 和上下文长度强相关。**

### 方向 H：多芯片长上下文 MLA

这是能讲大故事的方向，但不适合“快速发”。

你可以做的事情：

- 把 latent cache 分散到多芯片
- 研究 chip 间通信与 cache 分配
- 长上下文下比较不同 partition 策略

问题是：

- 系统复杂度太高
- 第一篇很容易做不完

所以我的建议是：

**先不要碰。**

---

## 5. 我最推荐的 3 个“论文包”

### 论文包 1：最快成稿

题目方向：

**`Multi-Head Latent Attention on Tenstorrent: A Case Study of Decode Bottlenecks and Memory Traffic`**

内容组合：

- 先做 MLA decode baseline
- 再做 MHA/GQA/MLA 对比
- 最后补一个 profiling + traffic model

优点：

- 实现门槛最低
- 最容易快速写出一篇 workshop/arXiv

缺点：

- 创新性更偏分析，不是特别偏“新 kernel”

### 论文包 2：最均衡，最推荐

题目方向：

**`Efficient Mapping of MLA to Explicit-Dataflow Accelerators: Cache Layout and Kernel Design on Tenstorrent`**

内容组合：

- baseline MLA decode
- cache layout/sharding 优化
- 一个轻量 fused kernel
- profiling 证明 NoC/DRAM 改善

优点：

- 工程和论文性比较平衡
- 很像一篇标准的系统优化论文

缺点：

- 比论文包 1 多一些 kernel 工程

### 论文包 3：更有卖点，但风险更大

题目方向：

**`Low-Precision MLA Cache on Tenstorrent for Long-Context Decoding`**

内容组合：

- MLA baseline
- 低精度 latent cache
- 精度/速度/带宽三者权衡

优点：

- 标题更容易吸引人
- 更像“方法 + 系统”

缺点：

- 更依赖底层支持
- 容易卡在数值和实现细节上

---

## 6. 如果目标是“最快发”，我建议你这样做

### 推荐主线

当前建议直接选择：

**`方向 A + 方向 B + 方向 G`，并把它们统一到 `SF-MLA` 框架下。**

也就是：

- 做 `MLA decode characterization`
- 做 `latent cache layout/sharding + communication topology`
- 做 `performance model / profiling / DSE`

这是当前最快形成完整论文闭环的组合，因为它天然对应：

1. `Characterization`
2. `Design`
3. `Model + DSE`
4. `Architecture implication`

### 不建议一开始做的事情

- 不要先做 full model training
- 不要先做 backward
- 不要先做多芯片
- 不要一上来就做非常激进的 quantization
- 不要把目标定成“完整复现 DeepSeek 全栈”

### 推荐理由

这条路线最容易形成完整故事：

1. MLA 在这个架构上为什么是新的 mapping 问题
2. baseline 的瓶颈和 phase transition 在哪里
3. 哪种 mapping / cache layout / topology 最优
4. latency 和 traffic 为什么会随 mapping 切换
5. 这个结论对显式 dataflow accelerator 有什么普遍意义

---

## 7. 最小可做实验矩阵

### baseline

- `MHA`
- `GQA` 或 `MQA`
- `MLA baseline`
- `MLA optimized`

### workload

- `seq_len = 1k / 4k / 16k / 64k`
- `batch = 1 / 4 / 8`
- `decode-only`
- 如有精力再补 `prefill + decode`

### metrics

- `latency per token`
- `tokens/s`
- `DRAM bytes per token`
- `NoC traffic / stall`
- `L1 occupancy`
- `kernel time breakdown`
- `数值误差`

### ablation

- `interleaved` vs `sharded`
- `DRAM-resident cache` vs `L1-friendly cache`
- `separate kernels` vs `fused kernels`
- `BF16 cache` vs `low-precision cache`
- 不同 `latent dim` 和 `RoPE dim`

这套实验已经足够支撑一篇第一版论文。

---

## 8. 落地实施顺序

### 第 1 周：先把 baseline 跑通

- 先做单层 MLA attention block
- 只做 inference
- 先做 decode-only
- 用 synthetic workload 也可以，不一定一开始就接完整模型

### 第 2 周：拿到 profiling 和瓶颈

- 跑 TT 的 profiling 工具
- 分清是 DRAM、NoC、L1、kernel launch 还是 compute 在卡
- 先写一页实验日志，别急着优化

### 第 3 周：做 cache layout/sharding

- 这是最容易出结果的优化
- 先比较 2 到 3 种布局，不要一次做太多

### 第 4 周：再决定要不要做 fused kernel

- 如果 layout 已经能拉开差距，就足够写第一版
- 如果结果还不够漂亮，再补一个轻量 fusion

### 第 5 周：写论文

- 先写 motivation 和 bottleneck analysis
- 再写 optimization
- 最后写 generalized insight

---

## 9. 可以直接拿去用的论文标题

- `Mapping Multi-Head Latent Attention to Tenstorrent Accelerators`
- `Efficient MLA Decoding on Explicit-Dataflow Architectures`
- `Cache Layout Matters: Optimizing MLA on Tenstorrent`
- `Understanding MLA Decode Bottlenecks on Tenstorrent`
- `A Tenstorrent Case Study of Multi-Head Latent Attention`
- `Sharded Latent KV Cache for Efficient MLA Decoding on Tenstorrent`

---

## 10. 我对你最具体的建议

如果你现在就是想 **尽快开题并尽快出第一篇**，我建议你把目标收敛成下面这句话：

**在 Tenstorrent 上把 MLA decode 明确提出为新的 spatial mapping problem，并在此基础上研究 latent KV cache 的 sharding/layout、communication topology、pipeline coupling、cost model 与 DSE。**

这是最像论文、同时也最容易做完的一条线。

如果结果顺利，后续自然可以扩成第二篇：

- fused kernel
- 低精度 cache
- prefill/decode runtime
- 多芯片扩展

---

## 11. 可参考的公开资料

- Tenstorrent TT-NN 文档：<https://docs.tenstorrent.com/tt-metal/latest/ttnn/>
- Tenstorrent TT-Metalium memory 文档：<https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html>
- Tenstorrent Tensix compute/dataflow 文档：<https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html>
- Tenstorrent profiling 文档：<https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html>
- FlashMLA：<https://github.com/deepseek-ai/FlashMLA>

---

## 12. 最后一句话版本

**最快能发的一篇，不是“我在 Tenstorrent 上做了一个更快的 FlashMLA”，而是“我证明了当 attention 从 MHA 走向 MLA、硬件从 GPU 走向 spatial accelerator 时，问题已经上升为 mapping + cost model + DSE 的系统问题”。**

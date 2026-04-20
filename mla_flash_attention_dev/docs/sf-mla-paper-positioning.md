# SF-MLA 论文定位与统一口径

## 1. 一句话定位

这篇文章最合适的定位不是“再做一个更快的 FlashMLA kernel”，而是把 **MLA decode 明确提出为一个新的 spatial mapping problem**。

更准确地说，本文是一篇围绕 **operator characterization + dataflow design + cost model + design space exploration (DSE) + architecture implication** 组织起来的系统/架构论文。

## 2. 关键词

- `Multi-Latent Attention`
- `Spatial Accelerators`
- `Dataflow Design`
- `Design Space Exploration`
- `Pipeline Coupling`
- `Hardware Multicast`
- `Tenstorrent`

## 3. 为什么这个问题成立

### 3.1 算法侧

MLA 通过 latent KV 压缩显著降低 KV cache 开销，但 decode 时必须在线恢复 per-head K/V。  
因此，它不再是传统 MHA decode 那种相对单纯的 memory-bound 问题，而会进入 **compute-memory mixed regime**。

### 3.2 硬件侧

Spatial accelerator 的关键特征是：

- 分布式 SRAM / L1
- decoupled `reader / compute / writer`
- 片上 NoC 的 multicast / reduction
- 显式 placement、buffering 和跨核协作

因此，优化重点不只在 tile 大小，而在：

- 数据驻留与搬运路径
- 通信拓扑
- 流水深度
- 跨核协同方式

### 3.3 研究空白

现有 `FlashAttention`、`FlashDecoding`、`FlashMLA` 主要围绕 GPU 的 IO-aware execution 展开。  
`FLAT` 和 `FlatAttention` 虽然对 spatial attention 有启发，但没有覆盖 MLA 的 latent restoration、异步 pipeline coupling 和自动化 DSE。

## 4. 文章核心主张

本文的核心主张是：

1. **当 attention 从 MHA 走向 MLA 时，decode 已经不再只是 kernel 内部 IO 优化问题，而是跨核 mapping 问题。**
2. **当硬件从 GPU 走向 Tenstorrent 风格的 spatial accelerator 时，关键优化对象会从“单 kernel 的 shared memory/register tiling”扩展为“mapping + communication topology + asynchronous pipeline”。**
3. **因此需要一套新的 challenge-driven methodology。**

本文将这套方法统一称为 **SF-MLA 框架**。

## 5. SF-MLA 框架

建议全文按下面五段组织：

### 5.1 Characterization

先用 operational intensity、profiling 和微架构利用率说明：

- MLA 不是 MHA 的小改动
- decode bottleneck 会随 `seq_len`、mapping 和硬件资源变化而切换
- 在 spatial accelerator 上，瓶颈来自 compute、reader、writer 之间的 **pipeline coupling**
- `compute share` 很高不等于 decode 的真实算术利用率已经很高；representative corrected-runtime `A1/A2/A3` 说明 `PM FPU util` 只会从 `low-teens` 升到 `mid-20s`
- `4c` 与 `8c` 的一阶区别主要体现在 q-core admission boundary，而不是所有已 fit 点上的普适加速

### 5.2 Concept

从三个高层概念提出 dataflow 设计：

- shared-latent reuse
- projection-attention pipeline
- reduction / multicast topology

### 5.3 Design

把高层概念具体化为可实现的数据流与 mapping：

- four-level tiling
- S-block sender / receiver 组织
- shared-latent multicast
- bank affinity
- page-level overlap
- `NoC0 / NoC1` 分工
- 三处理器 decoupled pipeline

### 5.4 Model + DSE

先建立 first-order analytical model，给出理想 steady-state。  
再加入 second-order empirical correction，吸收真实机器上的非理想效应：

- reader back-pressure
- writer stall
- multicast hotspot
- pipeline coupling loss

最后由 auto-tuner / search framework 做 exhaustive-but-fast 的配置搜索。

### 5.5 Evaluation

评测部分建议分成四类：

1. 当前硬件性能与优化空间
2. 微架构利用率与瓶颈迁移，包括 proxy sweep 的 phase transition 和 representative direct `PM/perf-counter`
3. cost model 与 tuner 的有效性
4. 容量边界与未来硬件资源变化下的架构含义

## 6. 三个核心挑战

当前论文口径下，MLA on spatial accelerators 主要回答三个挑战：

1. **复杂数据重用**：latent restoration 与 per-head reuse 使数据驻留和共享方式不再显然。
2. **decoupled execution 下的 pipeline coupling**：reader / compute / writer 的局部饱和会相互传递，形成 back-pressure。
3. **large / non-obvious design space**：最优 mapping 会随 `seq_len`、batch、head、硬件拓扑和 buffer 资源变化而切换。

## 7. 与已有工作的区分

### 7.1 与 GPU attention 优化的区别

在 GPU 语境下，attention 优化通常围绕：

- shared memory / register tiling
- HBM traffic reduction
- 单 kernel 内部的 IO-aware execution

在 spatial accelerator 上，问题会扩展为：

- 如何定义跨核 mapping
- 如何设计通信拓扑
- 如何做异步流水和 buffering
- 如何利用 multicast / reduction / placement

### 7.2 与 `FlashAttention` / `FlashDecoding` / `FlashMLA` 的区别

本文不是再提出一个更快的 attention kernel，而是解释：

- 为什么 MLA decode 在 spatial accelerator 上会暴露新的 mapping 问题
- 这些问题如何被系统化建模
- 为什么需要 DSE 与 cost model，而不是只做单点 kernel 优化

### 7.3 与 `FLAT` / `FlatAttention` 的区别

本文不仅讨论 spatial attention 的执行，还覆盖：

- MLA 特有的 latent restoration
- decoupled pipeline coupling
- hardware multicast / reduction
- second-order empirical model
- workload-aware 的 DSE

## 8. 预期贡献

建议把贡献写成下面五类：

1. **问题定义贡献**：把 MLA decode on spatial accelerators 明确提出为新的 spatial mapping problem。
2. **框架贡献**：提出 `SF-MLA`，把全文组织成 challenge-driven framework。
3. **设计贡献**：给出参数化 dataflow，包括 S-block sender/receiver、shared-latent multicast、projection-attention pipeline、NoC tree reduction。
4. **模型贡献**：从 first-order analytical model 推进到 second-order empirical model，把 coupling 与 hotspot 纳入预测。
5. **实证贡献**：用 Wormhole profiling、auto-tuning 与架构推演证明 bottleneck phase transition、representative arithmetic-utilization trend、模型有效性和 mapping 切换，并把 `4c/8c` 的容量墙与 pipeline coupling 现象直接钉住。

## 9. 当前最适合强调的实证亮点

现阶段最有说服力的结果是：

- profiling 支撑下的 bottleneck phase transition
- corrected-runtime representative `A1/A2/A3` 说明 decode `PM FPU util` 会从 `low-teens` 抬升到 `mid-20s`，但 `TT / DeepSeek-4c` 差值始终很小
- `TT vs DeepSeek-4c` 的 direct 结果更支持“front-side starvation 被缓解”，而不是“算术利用率进入完全不同区间”
- `4c vs 8c` 结果说明 `8c` 的一阶价值主要是抬高 q-core admission boundary，而不是已 fit 点上的普适加速
- second-order model 对现实执行行为的解释能力
- tuner 接近 empirical best 的能力
- 32K 等长序列点上 bottleneck 从 compute-critical 向 reader-writer coupling 的转移

像 `4c vs 8c`、不同 dataflow 变体、不同 baseline 的结果，都应优先被解释为：

- design-space 中不同 mapping point 的对比
- 容量边界和 pipeline coupling 的证据
- multicast / reduction / lane capacity 的结构性影响

而不应单独写成“某个 kernel 比另一个 kernel 更快”的 benchmark 叙事。

## 10. 当前的限制讨论

目前应诚实保留以下边界：

- 重点覆盖 `decode`
- batch size 与模型参数覆盖仍有限
- decode corrected-runtime `PM/perf-counter` 目前只补到了 representative `A1/A2/A3`，还不是 `decode_1k~32k` dense direct curve
- `NoC` / `DRAM` 的 direct 利用率百分比当前仍不可稳定获得
- 某些 Blackhole 结果更多依赖 projection / simulation
- `8 q_shards` / 更激进 head-group 扩展尚未完全打通
- second-order terms 在未来架构上的迁移边界需要单独讨论

## 11. 旧文档修订原则

修订旧文档时统一遵守下面几条：

1. 不再把文章叙事写成“更快的 FlashMLA kernel”。
2. 不再把 `prefill` 写成当前论文的唯一主战场；`prefill` 可以保留为 characterization、对照或 supporting evidence。
3. 所有 `baseline`、`8-core`、`fairness`、`profiling`、`simulation` 文档，都要明确它们在论文中的角色是：
   - `Characterization`
   - `Design`
   - `Model + DSE`
   - `Evaluation`
4. 所有“更快 / 最快 / speedup”表述，都尽量改写成：
   - 哪个 mapping point 被解锁
   - 哪个 bottleneck 被缓解
   - 哪个设计维度发生切换
5. 对纯实验归档、原始输出和 sweep 表，只补解释口径，不大改数字本体。

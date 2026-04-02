# 论文重构大纲：SF-MLA / Star-MLA

## 0. 命名讨论

| 候选名 | 展开 | 优势 | 劣势 |
|---|---|---|---|
| **SF-MLA** | Spatial-Flow MLA | 简洁；直击核心（spatial accelerator + dataflow + MLA） | 缩写不够直觉 |
| **Star-MLA** | Spatial Tiled Accelerator Research for MLA | 好记、好发音；"Star"暗示多核辐射式拓扑 | "Star"可能被理解为星形拓扑而非通用空间加速器 |
| **SpatialFlow-MLA** | — | 最显式 | 偏长 |
| **MESA-MLA** | Mapping & Exploration on Spatial Accelerators for MLA | 有"探索"含义 | 和 OpenGL MESA 重名 |

建议：**SF-MLA** 或 **Star-MLA** 都可以，如果偏学术简洁选 SF-MLA，如果偏好记好传播选 Star-MLA。

---

## 1. 核心立意（一句话）

> 在 spatial accelerator（以 Tenstorrent Tensix 为代表）上，**MLA 的数据重用复杂度远高于 MHA/GQA**，使得 **数据流设计空间（dataflow design space）** 成为性能的决定性因素；本文构建了一套从 **数据流建模 → 解析代价模型 → 自动化 DSE → 架构仿真 → 硬件实测** 的完整方法论，在 Wormhole 上实证验证，并进一步利用该框架探索了 chiplet 场景下 MLA 的最优架构配置。

---

## 2. 四大贡献（重构后）

### Contribution 1: MLA 在 Spatial Accelerator 上的数据流分析与设计

**动机：**
- MLA（Multi-Latent Attention，DeepSeek-V2/V3）将 KV 压缩到低秩 latent space，decode 时需要在线还原 K/V，因此数据流中的 **数据重用模式** 比 MHA/GQA 复杂得多
- 在 spatial accelerator（众核 + 片上网络 + per-core SRAM）上，数据重用不再只是"是否 cache-friendly"，而是涉及：
  - **空间维度的数据重用**（哪些 core 共享同一份 K/V latent？multicast / unicast / broadcast？）
  - **时间维度的数据重用**（CB depth、double buffering、pipeline overlap）
  - **拓扑维度**（independent cores vs. S-Block multicast vs. tree reduction）
- 因此需要一套 **spatial accelerator-aware 的数据流设计框架**

**具体贡献：**
- 对 MLA decode/prefill 在 Tensix 上的数据流进行了系统建模：
  - Producer-Consumer multicast graph
  - Reader (NCRISC) → Compute (TRISC) → Writer (BRISC) 的解耦流水
  - K/V 数据通过 NoC multicast 实现空间重用
  - Online softmax + tree reduction 实现跨核归约
- 深入分析了当前 TT-MLA 数据流的设计逻辑：
  - 为什么 decode 用 `4 lanes × 4 cores/lane` 的 independent 模式
  - 为什么 prefill 用 chunked flash 路径
  - Reader 内部从 `issue` 主导迁移到 `reserve/block`（downstream backpressure）主导的阶段变化
  - Writer 内部 `cb_wait_front` 占绝对主导的含义
- 与 MHA 的数据流进行对比分析：
  - MHA decode: Q 小、KV 大但结构简单，每 head 独立
  - MLA decode: latent 需要在线展开，数据量不同、复用模式不同
  - 空间加速器上 MLA 比 MHA 更需要精细的 multicast/buffering 设计

### Contribution 2: 数据流设计空间探索（DSE）方法论

**动机：**
- 数据流中有大量可调参数（head grouping, tile grouping, block size, pipeline depth, reduction strategy, math fidelity, page size...）
- 穷举不现实，但参数之间有强耦合（例如 block size 影响 SRAM 占用 → 影响可用 core 数 → 影响 multicast 收益）
- 需要一套系统的 DSE 方法

**具体贡献：**

**(a) 数据流构建与参数化**
- 将数据流参数化为 `(G_head, T_group, B_kv, depth, overlap, reduce_strategy, topology_mode, math_fidelity, page_size_strategy)` 的紧凑设计空间
- 支持 `independent` / `sblock_multicast` / `tree` 三种拓扑模式
- Workload-aware heuristic pruning 减少冗余搜索

**(b) 解析代价模型（Analytical Cost Model）**
- DRAM traffic model（baseline per-core-read vs. multicast ideal）
- Per-block stage latency: `t_dram`, `t_mcast`, `t_compute`
- Pipeline steady-state formula
- 新增三类运行时 stall 指标：
  - `reader_backpressure_ms`（对应 WH 上观测到的 `cb_reserve_back` 阻塞）
  - `writer_backpressure_ms`（对应 `cb_wait_front` 等待）
  - `sender_hotspot_ms`（multicast 注入热点）
- SRAM feasibility constraints
- 线性校准模型（用少量硬件实测数据回归修正系数）

**(c) 自动寻优器（Auto-Tuner）**
- Enumerate → Score → Rank
- 单目标（latency）和多目标（Pareto: latency vs. DRAM traffic）
- Bucketed policy cache（sequence_pow2 bucketing）
- Top-K measured reranking：profile JSON → MeasurementDB → bridge → rerank

**(d) 架构仿真**
- 一阶模型：替换硬件常数（DRAM BW, compute FLOPS, NoC BW），评估 dataflow 在不同架构上的理论下界
- 二阶经验模型：把 WH 上观测到的 pipeline coupling 项（reader reserve, writer cb_wait）回灌到目标架构
  - 得到的结论：A-BH 二阶比一阶慢 1.6x~1.8x
  - Compute 在 7~10 cores 就退出 critical path
  - 真正的 floor 是 reader active + writer wait←reader
- Active-core sensitivity sweep：找到 crossover 点和 plateau 点
- 方法 B（原生 Flash-MLA 拓扑）在 BH 上的对比：32k 下快 3.1x~5.6x

### Contribution 3: Wormhole 硬件实测验证

**具体内容：**
- 在 Tenstorrent Wormhole N300 上对 Flash-MLA decode + prefill 做了完整 profile sweep
- Decode: 256 → 32k 的阶段迁移：
  - `256~1k`: compute-critical
  - `2k~8k`: writer 贴近 critical path
  - `16k`: reader 也开始贴近
  - `32k`: reader + writer + compute 全部饱和
- Prefill: 从 256 开始就是 reader_writer_saturated
- Reader 内部分解：`page_table / reserve / issue / wait / push`
  - 1k: issue 76.8%, reserve 0.7%
  - 32k: reserve 61.6%, issue 33.3%
- Writer 内部分解：`cb_wait / issue / barrier / pop`
  - 几乎所有 case: cb_wait > 94%
- 关键洞察：
  - decode 长序列不是"纯 compute-bound"，而是深度耦合的流水线饱和
  - Reader 瓶颈的本质是 downstream backpressure (L1 buffer turnover)
  - Writer 瓶颈的本质是等待上游 compute 产出（pipeline-coupled）
- Profile → MeasurementDB → Autotuner rerank 的数据闭环

### Contribution 4: Chiplet 场景下 MLA 的架构探索

**动机：**
- 利用我们的 DSE + 仿真框架，回答 MLA 在 chiplet 架构设计中的关键问题

**关键研究问题（用我们的框架来回答）：**

1. **SRAM 大小对 FlashMLA 是最合适的吗？**
   - 当前 Tensix: 1MB L1 per core
   - BH: 1.5MB L1 per core
   - 更大的 SRAM → 更深的 CB → 更少的 backpressure，但面积/功耗代价
   - 用 cost model + 二阶经验模型扫描 SRAM 大小 vs. 性能的 Pareto

2. **TOPS 是最合适的吗？**
   - 当前 Tensix compute: ~1 GFLOPS per core (BF16, HiFi4)
   - 二阶模型已经显示 compute 在 7~10 cores 就退出 critical path
   - → 对当前 dataflow，单核 TOPS 提升的边际收益有限
   - 更好的策略可能是降低单核 TOPS + 增加 SRAM / NoC BW

3. **SRAM 1.5MB + 1GFLOPS 的组合是否最优？**
   - 用 DSE 框架做 (SRAM_size, TOPS_per_core, NoC_BW) 的联合扫描
   - 找到 FlashMLA 特定 workload 下的最优配置点

4. **是否四个核合成一个大核会更好？**
   - 4× SRAM (4MB) + 4× compute → 但 multicast 并行度降低
   - Trade-off: 更少的 backpressure vs. 更少的空间重用
   - 用 cost model 对比：`64 small cores` vs. `16 big cores` 在 MLA decode 下的性能

5. **与 MHA 对比：MLA 是否改变了最优架构选择？**
   - MHA decode: 纯 memory-bound，DRAM BW 主导
   - MLA decode: 多了 latent projection compute，pipeline coupling 更重
   - 同一个 spatial accelerator 上，MHA 和 MLA 的最优 core 配置可能不同

---

## 3. 论文结构（建议）

```
Title: SF-MLA: Dataflow Design Space Exploration for Multi-Latent Attention
       on Spatial Accelerators

1. Introduction
   - LLM 推理的 decode 瓶颈
   - MLA 的兴起（DeepSeek-V2/V3）及其独特的数据流挑战
   - Spatial accelerator 的机遇与数据流复杂性
   - 本文方法论概述 + 贡献列表

2. Background
   2.1 Efficient Attention Mechanisms
       - MHA → GQA/MQA → MLA
       - FlashAttention 系列
       - PagedAttention + KV cache management
   2.2 MLA (Multi-Latent Attention) 详解
       - 低秩压缩、latent projection
       - MLA 的 decode 数据流特殊性
       - MLA + Paged KV cache
   2.3 Tenstorrent 架构
       - Tensix core 微架构（5 个 baby RISC-V）
       - 解耦 data mover + circular buffer
       - 双 NoC + hardware multicast
       - Wormhole vs. Blackhole 参数对比
   2.4 Related Work
       - FlatAttention, FLAT, Timeloop/MAESTRO
       - FlashMLA (GPU)
       - 众核/non-GPU attention 实现

3. MLA Dataflow on Spatial Accelerators
   3.1 MLA 数据流分析
       - Decode: Q(latent) × K(latent→full) × V(latent→full) 的 IO pattern
       - 空间重用 vs. 时间重用 vs. 拓扑选择
   3.2 当前 TT-MLA 数据流设计剖析
       - Producer-Consumer multicast + tree reduction
       - Reader/Compute/Writer 解耦流水
       - 为什么这么设计 + 与 MHA 实现的对比
   3.3 数据流参数化
       - 设计空间定义
       - 约束与可行性

4. Analytical Cost Model & DSE Framework
   4.1 解析代价模型
       - DRAM traffic, NoC latency, compute, backpressure
   4.2 Auto-Tuner
       - 搜索空间、启发式剪枝、评分排序
   4.3 架构仿真
       - 一阶/二阶经验模型
       - Pipeline coupling 回灌

5. Evaluation on Wormhole
   5.1 实验设置
   5.2 Profile 阶段分解
       - Decode 阶段迁移
       - Prefill 饱和流水线
   5.3 Reader/Writer 内部分解
   5.4 Cost Model 精度验证
   5.5 Auto-Tuner 质量

6. Architecture Exploration for MLA on Chiplets
   6.1 SRAM Size Sensitivity
   6.2 Compute (TOPS) Sensitivity
   6.3 Core Granularity: Many Small vs. Few Large
   6.4 MLA vs. MHA: 最优架构是否不同？
   6.5 Blackhole 仿真与验证

7. Discussion
   - 方法论的可迁移性
   - 局限性
   - 多卡扩展展望

8. Conclusion
```

---

## 4. 与旧论文的主要差异

| 维度 | 旧论文 (McastAttn) | 新论文 (SF-MLA / Star-MLA) |
|---|---|---|
| 注意力类型 | MHA/GQA | **MLA**（核心）+ MHA 对比 |
| 核心主题 | Multicast-aware dataflow | **数据流 DSE 方法论** |
| 数据流复杂性 | 相对固定（Producer-Consumer-Reducer） | 参数化设计空间 + 自动寻优 |
| Cost Model | 一阶解析 | 一阶 + **二阶经验（pipeline coupling）** |
| 评估 | 预计 latency/DRAM 改善 | **真实 WH profile + 阶段分解** |
| 架构探索 | 无 | **chiplet 架构设计空间探索** |
| 仿真 | 无 | **WH→BH 一阶/二阶仿真** |
| 实用性 | 固定数据流 | **Profile→MeasurementDB→Autotuner 闭环** |

---

## 5. 关键 Figure 规划

1. **MLA vs MHA 数据流对比图** — 显示 MLA 为什么数据重用更复杂
2. **Tensix 微架构 + 数据流映射图** — Reader/Compute/Writer + CB + dual NoC
3. **DSE 框架总览图** — Workload → Cost Model → Auto-Tuner → Profile → Calibrate 闭环
4. **WH Decode 阶段迁移图** — NCRISC/BRISC/TRISC share 随 seq_len 变化
5. **Reader/Writer 内部分解堆叠图** — reserve/issue/wait/push 和 cb_wait/issue/barrier/pop
6. **BH 一阶 vs 二阶 vs 方法 B 性能对比图**
7. **架构探索 Pareto 图** — SRAM size / TOPS / core count vs. MLA latency
8. **MLA vs MHA 最优架构配置对比图**

---

## 6. 需要补充的工作

- [ ] MLA 和 MHA 在同一框架下的定量对比（数据流差异 + 性能差异）
- [ ] SRAM/TOPS/core granularity 的参数化仿真扫描
- [ ] 4-core-merge 场景的 cost model 扩展
- [ ] FlashMLA (GPU) baseline 对比数据
- [ ] DeepSeek-V2/V3 workload 参数的标准化
- [ ] MLA + PagedAttention 的数据流细节补充

# [Archived] SF-MLA 下一阶段执行计划

> 已归档（2026-04-13）。
> 这份文档保留的是一版过渡期执行计划，不再作为当前项目的直接行动指南。
> 当前请优先参考 `current-docs.md` 与 `sf-mla-paper-positioning.md`；若需要看历史阶段收敛过程，再继续阅读本文。

## 1. 文档目的

这份文档用于把当前的研究想法收敛成一条可执行的项目路线，目标不是继续扩展方向，而是明确：

- 接下来优先做什么
- 哪些内容先不做
- 每个阶段的产出是什么
- 什么时候进入下一阶段

本文默认基于以下已有材料继续推进：

- `tenstorrent-mla-expanded-roadmap.md`
- `ref/tt-metal_kv_forwarding_iccad_experiment_plan.md`
- `ref/tt-metal_kv_forwarding_design_space_autotuner.md`
- `ref/NoC_与多播代码实现详解.md`
- `dev_log.md`
- `sf-mla-paper-positioning.md`

---

## 2. 2026-04 定位更新

这份执行计划早期是按 `non-causal prefill + KV forwarding` 的工程路线写的；当前论文口径已经统一更新为：

- **主问题**：`MLA decode on spatial accelerators` 是新的 `spatial mapping problem`
- **文章形态**：`operator characterization + dataflow design + cost model + DSE + architecture implication`
- **主框架**：`SF-MLA`
- **prefill`/`KV forwarding`/`8-core`/`profiling`/`simulation` 的角色**：作为 `Characterization`、`Design` 或 `Evaluation` 证据，而不是论文本身的唯一主线

因此，本文后续凡是提到早期的 `Experiment D`、`KV forwarding`、`prefill` 或 `speedup`，都应按新的论文结构来解读，而不再被视为“整篇论文的最终标题与主战场”。

---

## 3. 总体策略

当前不适合把 `decode characterization`、`dataflow implementation`、`cost model`、`DSE`、`multi-chip` 全部无约束并行推进。

接下来应采用一条更稳妥的主线：

1. 先固定论文主线与问题定义
2. 再把现有 profiling、8-core、实验路径整理成统一的 mapping 证据
3. 用可解释模型和 DSE 把“经验调参”升级成方法论
4. 最后把工程结果收束成 challenge-driven 论文故事

一句话版本：

**先把 `MLA decode as a spatial mapping problem` 讲清楚，再用 `prefill / KV forwarding / 8-core / simulation` 去支撑 `SF-MLA` 的设计、建模与 DSE。**

---

## 4. 默认决策

为避免后续反复摇摆，当前先默认采用下面这些决策：

### 3.1 主线场景

- 第一阶段主线：`single-chip MLA decode`
- 第一阶段问题定义：`MLA decode on spatial accelerators = mapping + communication + pipeline coupling`
- 第一阶段重点：`characterization / dataflow / model / DSE`

### 3.2 次要场景

- `prefill`：作为 supporting evidence，用于解释 forwarding / multicast / overlap 的结构问题
- `multi-chip`：先不作为主目标，只保留为架构扩展方向
- `Blackhole projection / simulation`：作为架构含义与设计空间扩展证据

### 3.3 autotuner 定位

- 第一阶段把 autotuner 视为 `SF-MLA` 中的 `offline DSE / cached policy selector`
- 不默认把它写成在线实时搜索
- 只有在 `oracle gap`、`search overhead`、`top-k quality` 足够有说服力时，才升级为标题级贡献

### 3.4 评估重点

第一阶段优先关注下面这些指标：

- `latency / token`
- bottleneck phase transition
- `reader / writer / compute` stall breakup
- `NoC bytes` / `multicast hotspot` / `bank affinity`
- first-order vs second-order model 误差
- tuner `top-1 gap` / search cost
- correctness / stability / fairness boundary

---

## 4. 项目路线图

```mermaid
flowchart TD
    scopeFreeze[ScopeFreeze]
    baselineMap[BaselineAndCodeMap]
    ablationD[ExperimentD_Ablation]
    autoGate{AutotunerWorthIt}
    autotunerE[ExperimentE_Autotuner]
    writeupPack[WriteupAndReportPack]

    scopeFreeze --> baselineMap
    baselineMap --> ablationD
    ablationD --> autoGate
    autoGate -->|yes| autotunerE
    autoGate -->|no| writeupPack
    autotunerE --> writeupPack
```

---

## 5. 阶段计划

| 阶段 | 时间建议 | 主要目标 | 核心输出 |
| --- | --- | --- | --- |
| Phase 0 | `2-3` 天 | 冻结论文问题与 workload | `scope note`、主假设、evaluation map |
| Phase 1 | `1` 周 | 建代码地图与 characterization 资产 | baseline 说明、代码路径图、profile 入口 |
| Phase 2 | `2-3` 周 | 做 dataflow / mapping 设计与证据收敛 | 设计变量表、关键消融、阶段结论 |
| Phase 3 | `1-2` 周 | 建 cost model 与 DSE | `oracle gap`、MAPE、tuner 质量 |
| Phase 4 | `1` 周 | 整理成 `SF-MLA` 论文包 | 图表清单、主结论、报告草稿 |

---

## 6. Phase 0：冻结范围

### 6.1 目标

把“想做很多事”收敛成“先做一条主线”。

### 6.2 要完成的事

- 明确第一阶段以 `single-chip MLA decode` 为论文主战场
- 明确 `prefill` 作为 supporting evidence，而不是论文唯一主线
- 明确 `multi-chip` 暂不纳入第一阶段必须项
- 把 workload 按 `characterization / design / model / implication` 分类
- 固定第一阶段的主指标、对照基线和公平性边界

### 6.3 建议的 workload family

- `F1`：当前主形状
- `F2`：更长序列
- `F3`：`GQA / MQA` 变化
- `F4`：不利布局、尾部不均匀、`mcast` 不可用的 stress case

### 6.4 退出标准

满足下面三条即可结束本阶段：

- 能用一句话说清第一阶段主线
- 能列出第一批代表 workload
- 能说明哪些内容暂时不做

### 6.5 当前产物

本阶段对应的范围冻结文档：

- `phase-0-scope-note.md`

---

## 7. Phase 1：建立 baseline 和代码地图

### 7.1 目标

建立一张统一的“从模型到 kernel”的路径图，并固定基线入口。

### 7.2 重点代码入口

#### 模型层

- `models/demos/deepseek_v3/tt/mla/mla1d.py`
- `models/demos/deepseek_v3/tt/mla/mla2d.py`

#### 算子层

- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp`

#### program factory / kernel 层

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`

### 7.3 第一批测试入口

- `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill.py`
- `tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py`
- `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill_v_embedding_space.py`

### 7.4 本阶段输出

- 一张代码路径图：`model -> op -> program factory -> reader/compute/writer`
- 一份 baseline 说明表
- 一份第一批 workload 清单

### 7.5 退出标准

- 已明确哪些结果算“应用 baseline”
- 已明确哪些结果算“优化主证据”
- 已固定第一批可复现实验入口

### 7.6 当前产物

本阶段目前已完成的代码路径图文档：

- `phase-1-code-path-map.md`

---

## 8. Phase 2：执行优化项消融 D

### 8.1 目标

证明每个优化点都在解决明确瓶颈，而不是随机调参。

### 8.2 推荐消融顺序

按“从保守到激进”的顺序推进：

1. `B0 = NC-current-auto`
2. `B1 = B0 + per-chain hybrid`
3. `B2 = B1 + layout-aware mapping`
4. `B3 = B2 + pipelined read/forward`
5. `B4 = B3 + dual NoC`
6. `B5 = B4 + rotating injector`
7. `B6 = B5 + tree forwarding`

### 8.3 第一阶段必须完成项

如果时间有限，优先完成下面四项：

- `per-chain hybrid`
- `layout-aware mapping`
- `pipelined read/forward`
- `dual NoC`

### 8.4 第一阶段可暂缓项

- `rotating injector`
- `tree forwarding`

这两项更适合做成强化版结果或第二阶段工作。

### 8.5 主要指标

- speedup vs `NC-current-auto`
- speedup vs `NC-naive`
- NoC bytes
- DRAM utilization
- injector stall
- receiver idle
- overlap ratio
- FPU utilization

### 8.6 workload 组织建议

- 短链：保留 `1` 个 sanity case
- 中链：重点看 `chain_len = 3~4`
- 长链：重点看 `chain_len >= 6`
- 增加至少 `1` 个不利布局 stress case

### 8.7 本阶段输出

- 累积消融图
- 按链长分组的 speedup 图
- 一个 profiler timeline case study
- 一页“每个优化项解决什么问题”的总结

### 8.8 退出标准

- 至少有 `1-2` 个优化项在中长链场景上稳定优于 `NC-current-auto`
- 短链 sanity case 没有明显回退
- profiler 能支撑性能变化的解释

### 8.9 当前产物

本阶段目前已完成的 `Experiment D` 执行文档：

- `experiment-D-ablation-plan.md`
- `experiment-D-b0-baseline.md`（B0：`NC-current-auto` 基线说明与结果表模板）

---

## 9. Phase 3：决定 autotuner 的级别

### 9.1 目标

决定 autotuner 是：

- 论文或报告中的主贡献
- 次要章节
- 未来工作

### 9.2 建议对比

1. `expert default`
2. `single-rule heuristic`
3. `oracle / exhaustive on reduced space`
4. `your autotuner`

### 9.3 必须回答的问题

- 不同 workload 的最优配置是否真的不同
- reduced space 上是否能拿到 oracle
- autotuner 与 oracle 的差距是否足够小
- 冷启动和热启动开销是否可接受

### 9.4 决策门

只有满足下面条件，autotuner 才升级为主贡献：

- `oracle gap` 数据完整
- 搜索开销可解释
- 不同 workload 的最优配置差异明显
- autotuner 稳定优于固定 expert 规则

否则默认降级为：

- 离线策略选择
- 附录或次要章节
- future work

### 9.5 本阶段输出

- `oracle vs heuristic vs autotuner` 对比图
- 搜索成本与收益 trade-off 图
- 一页“autotuner 是否值得放到标题里”的结论

---

## 10. Phase 4：整理成报告或论文包

### 10.1 目标

把当前工作组织成一套完整而收敛的叙事，而不是零散的实验结果。

### 10.2 推荐结构

#### 第一层：映射与 baseline

- MLA + Flash Attention 在 TT 上的应用落点
- baseline 行为和瓶颈画像

#### 第二层：通信与系统优化

- KV forwarding
- multicast / hybrid
- layout-aware mapping
- pipelined read/forward
- dual NoC

#### 第三层：自动配置

- autotuner 或 offline policy selection

### 10.3 应形成的主要图表

- baseline 问题陈述图
- 消融累计增益图
- 按链长分组的 speedup 图
- profiler 时间线图
- autotuner vs expert/oracle 图

### 10.4 本阶段输出

- 主报告草稿
- 图表清单
- 实验矩阵附录
- 关键结论摘要

---

## 11. 本周任务清单

如果只看最近一周，建议按下面顺序推进：

1. 固定论文主线：`MLA decode as a spatial mapping problem`
2. 把现有文档按 `Characterization / Design / Model+DSE / Evaluation` 重新归类
3. 整理一张 `model -> op -> kernel -> profiling` 的代码路径图
4. 把 `8-core`、`profile`、实验性 FlashMLA、主线 FlashMLA 统一到一个设计空间表述里
5. 先把 `shared-latent reuse / pipeline coupling / multicast-reduction topology` 定为第一批核心设计点
6. 给模型和 DSE 写一个明确的“升级条件”，避免后面继续停留在经验调参

---

## 12. 成功标准

如果后续执行顺利，第一阶段应至少达到下面几个结果：

- 能明确解释 MLA decode 的主要瓶颈及其 phase transition
- 能说明每个设计点缓解了哪一类 pipeline coupling 或 mapping 约束
- 能证明最优 mapping 会随 workload 和硬件资源变化而切换
- 能量化 first-order 与 second-order model 的误差差距
- 能把整个故事收敛成 `Characterization -> Design -> Model+DSE -> Architecture Implication`

---

## 13. 当前不建议做的事

- 不要再把论文写成“更快的 FlashMLA kernel”故事
- 不要把 `prefill`、`decode`、`multi-chip` 三条线并列成三个主标题
- 不要默认把 autotuner 写成在线实时机制
- 不要让 profiling、8-core、simulation、baseline 对比彼此脱节

---

## 14. 文档使用方式

建议后续这样使用这份文档：

- 每完成一个阶段，在 `dev_log.md` 记录结论
- 每开始一个优化项，在对应实验或实现文档里补充细节
- 每周回看一次本文件，判断是否需要调整范围或降级某些目标

如果后续继续细化，建议再拆出两份子文档：

1. `weekly-task-board.md`
2. `experiment-D-ablation-plan.md`

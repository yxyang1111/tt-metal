# [Archived] Phase 0 Scope Note

> 已归档（2026-04-13）。
> 这份文档记录的是旧阶段的范围冻结结论，带有明显的阶段性工程口径。
> 当前请优先参考 `current-docs.md` 与 `sf-mla-paper-positioning.md`；若需要回看历史决策，再继续阅读本文。

## 1. 文档目的

这份文档用于完成 `next-step-execution-plan.md` 里的第一步，也就是把第一阶段主线正式冻结下来。

本阶段的目标不是继续扩展方向，而是回答下面四个实际问题：

1. 第一阶段到底做哪条主线
2. 哪些方向暂时不做
3. 第一批 workload 应该如何组织
4. 后续实验应以什么 baseline 和指标为准

---

## 2. Phase 0 结论

### 2.1 当前统一主线

第一阶段正式固定为：

**把 `MLA decode on spatial accelerators` 明确写成新的 `spatial mapping problem`，并用 `SF-MLA` 作为统一框架来组织 characterization、dataflow design、cost model、DSE 和 architecture implication。**

### 2.2 decode、prefill 与实验资产的定位

当前各条线的角色固定如下：

- `decode`：论文主战场，负责问题定义、mapping 设计、建模与 DSE
- `prefill`：supporting evidence，主要用于解释 forwarding / multicast / overlap 等结构性行为
- `8-core`、`WH/BH`、实验性 FlashMLA：设计空间中的不同 mapping point
- `simulation / projection`：未来架构含义与资源变化分析

### 2.3 autotuner 的定位

当前把 autotuner 固定为：

- `offline DSE / cached policy selector`
- `SF-MLA` 框架的一部分，而不是独立的在线系统

只有在后续满足下面条件时，才升级为标题级贡献：

- reduced-space `oracle` 可获得
- `oracle gap` 与 `top-1 gap` 数据完整
- 搜索开销可解释
- 相比固定 expert 规则稳定更优

### 2.4 当前不把什么当成主标题

当前不把下面这些内容单独当成论文主标题：

- “更快的 FlashMLA kernel”
- 单一 `prefill` 优化故事
- 仅以 `8-core` 为核心的 benchmark 结论
- 在线 request-time autotuning

---

## 3. 为什么这样冻结范围

### 3.1 现在最有论文闭环的是 decode mapping

现有材料已经覆盖了：

- MLA decode 的 baseline / fairness / capability probe
- 主线 FlashMLA 与实验性 FlashMLA 的数据流差异
- `4c vs 8c` 的容量边界与 mapping 切换
- Wormhole profiling 与 Blackhole projection / simulation

这些资产天然适合收束为：

- 问题定义
- 设计空间
- cost model
- DSE
- 架构含义

### 3.2 prefill 仍然重要，但不再是唯一主战场

prefill 不是要被删除，而是要被重新解释为：

- shared-latent reuse 与 forwarding 机制的 supporting evidence
- `pipeline coupling` 和 `multicast` 行为的额外观测窗口
- decode 论文主线之外的补充对照

### 3.3 这更符合当前研究空白

当前真正有论文价值的，不是“又做了一个更快 kernel”，而是：

- 为什么 MLA 在 spatial accelerator 上暴露出新的 mapping 问题
- 为什么最优 dataflow 会随 workload 和硬件资源变化而切换
- 为什么需要 `first-order + second-order model + DSE`

---

## 4. 第一阶段固定研究问题

在当前冻结范围下，第一阶段优先回答下面四类问题：

### 4.1 Characterization：MLA decode 到底是什么问题

要回答清楚：

- MLA decode 何时处于 `compute-memory mixed regime`
- bottleneck 如何随 `seq_len`、batch、head、硬件资源变化而切换
- `reader / compute / writer` 的回压如何形成 `pipeline coupling`

### 4.2 Design：哪些 dataflow 设计最关键

重点关注：

- shared-latent reuse
- projection-attention pipeline
- multicast / reduction topology
- S-block / lane / bank-affinity mapping

### 4.3 Model + DSE：为什么不能只靠经验调参

要证明：

- first-order model 不足以解释真实机器行为
- second-order correction 能显著降低误差
- 最优 mapping 会在不同 workload 上切换

### 4.4 Architecture implication：这些结论对未来硬件意味着什么

重点关注：

- lane 容量与 active-core 覆盖
- multicast 扇出和热点
- reader / writer 饱和的 phase transition
- 新架构资源变化下最优 mapping 的迁移

---

## 5. 第一批 workload family

第一批 workload 按 `D1-D4` 四类固定下来。

| Family | 目的 | 建议参数 | 当前状态 |
| --- | --- | --- | --- |
| `D1: Decode baseline` | 建立标准 MLA decode 行为画像 | `B=1/2`, `H=32`, `seq_len=256/1k/4k` | 已覆盖 |
| `D2: Long-context phase transition` | 观察 bottleneck 切换 | `seq_len=8k/16k/32k/128k` | 已部分覆盖 |
| `D3: Mapping stress` | 放大 lane / q_shards / 容量边界 | `B=6/8/12`, `H=24/32`, `dqhpc=8` | 已覆盖较多 |
| `D4: Architecture sensitivity` | 观察不同 mapping / 架构资源下的最优点迁移 | `4c vs 8c`、WH vs BH projection | 已部分覆盖 |

### 5.1 第一批代表 workload

为了避免一开始把矩阵铺得过大，第一批先固定为下面 8 个代表 case：

1. `W1_D1_decode_1k`: `B=1, H=32, seq_len=1k`
2. `W2_D1_decode_4k`: `B=1, H=32, seq_len=4k`
3. `W3_D2_decode_8k`: `B=1, H=32, seq_len=8k`
4. `W4_D2_decode_32k`: `B=1, H=32, seq_len=32k`
5. `W5_D3_qshard3`: `B=8, H=24, seq_len=8k`
6. `W6_D3_qshard4`: `B=6, H=32, seq_len=8k`
7. `W7_D3_capacity_boundary`: `B=8/12, H=32, seq_len=8k`
8. `W8_D4_arch_switch`: `WH 4c vs 8c` 或 `WH vs BH projection`

说明：

- `W1-W4` 用于 characterization 与模型校准
- `W5-W7` 用于 design / mapping / 容量边界
- `W8` 用于 architecture implication

---

## 6. 第一阶段 baseline 冻结

### 6.1 功能对照层

第一阶段主表默认使用下面四类 baseline：

- `Reference Attention`
- `Flash Attention`
- `FlashMLA (TT Mainline)`
- `DeepSeek / experimental FlashMLA`

### 6.2 设计空间层

如果需要做设计空间对照，按下面层次组织：

- `mainline vs experimental`
- `4c vs 8c`
- `with vs without multicast`
- `first-order vs second-order`

### 6.3 配置策略层

如果进入 DSE 阶段，再增加：

- `expert default`
- `heuristic`
- `autotuner`
- `oracle / exhaustive on reduced space`

---

## 7. 第一阶段指标冻结

第一阶段统一使用下面这些指标：

### 主指标

- `latency / token`
- throughput
- supported coverage / capacity boundary

### 微架构与资源指标

- `reader / writer / compute` stall breakup
- `NoC bytes`
- `multicast hotspot`
- `DRAM utilization`
- `active core ratio`
- `bank affinity`

### 模型与搜索指标

- first-order MAPE
- second-order MAPE
- tuner `top-1 gap`
- search cost

### 兜底指标

- correctness
- seed-to-seed stability
- fairness boundary / measurement scope

---

## 8. 当前明确不做的内容

为了保证第一阶段闭环，下面这些内容先明确不作为必须项：

- 把论文写成“谁比谁更快”的 benchmark 故事
- 把 `multi-chip` 纳入第一阶段主实验
- 把在线 autotuner 写成主系统
- 把 `8 q_shards` 打通当成第一阶段是否成立的前提
- 在证据链还没闭环时继续扩写大而全的总纲文档

---

## 9. Phase 0 完成标准

这份 scope note 生效后，`Phase 0` 可视为完成，条件如下：

1. 已明确一句话主线
2. 已明确第一阶段不做什么
3. 已固定 `D1-D4` workload family
4. 已固定第一批代表 workload
5. 已固定 baseline、指标与 DSE 口径

---

## 10. 下一步

在这份文档基础上，下一步直接进入：

- `Phase 1：建立代码地图与 characterization 资产`

最具体的动作是：

1. 整理 `model -> op -> program factory -> kernel -> profiler` 的代码路径图
2. 统一 `mainline / experimental / 8-core / simulation` 的设计空间表述
3. 准备 first-order / second-order model 所需的最小实验集

# Phase 0 Scope Note

## 1. 文档目的

这份文档用于完成 `next-step-execution-plan.md` 里的第一步，也就是把第一阶段主线正式冻结下来。

本阶段的目标不是继续扩展方向，而是回答下面四个实际问题：

1. 第一阶段到底做哪条主线
2. 哪些方向暂时不做
3. 第一批 workload 应该如何组织
4. 后续实验应以什么 baseline 和指标为准

---

## 2. Phase 0 结论

### 2.1 第一阶段主线

第一阶段正式固定为：

**`single-chip non-causal prefill` 场景下，围绕 `MLA + Flash Attention` 的 `KV forwarding / multicast / layout / pipeline / NoC` 系统优化。**

这里的重点不是把所有 MLA 路径都同时做完，而是先围绕最容易形成闭环的场景，把：

- 基线建立清楚
- 通信瓶颈解释清楚
- 优化项收益证明清楚

### 2.2 MLA + Flash Attention 在第一阶段中的定位

第一阶段中，`MLA + Flash Attention` 的定位是：

- 作为应用背景和算子映射对象
- 作为解释为什么 latent cache、布局和 NoC 组织重要的核心案例
- 作为后续扩展到 decode 与 paged 场景的统一入口

但第一阶段真正的工程主战场是：

- `non-causal prefill`
- `KV forwarding`
- `multicast / hybrid`
- `layout-aware mapping`
- `pipelined read/forward`
- `dual NoC`

### 2.3 decode 的定位

`decode` 在第一阶段不作为核心优化主线，而是：

- 用于补充 MLA 应用背景
- 用于解释后续为什么可能扩展到 `flash decode` / tree-style 通信
- 作为第二阶段或对照场景保留

第一阶段不把下面这些问题作为必须完成项：

- decode 主路径性能优化
- decode 树形归约的系统性扩展
- decode 作为论文主贡献

### 2.4 multi-chip 的定位

`multi-chip` 暂不纳入第一阶段必须项。

它保留为：

- 后续自然扩展方向
- 未来讨论 ring distributed / joint attention / chip 间 cache 切分时的接口

但在当前阶段，不要求把它纳入主实验矩阵或主叙事。

### 2.5 autotuner 的定位

第一阶段先把 autotuner 定位为：

- `offline / cached policy selector`
- 设计空间搜索和配置选择机制

第一阶段明确不把 autotuner 默认写成：

- 在线实时搜索
- request-time tuning

只有在后续满足下面几个条件时，才考虑把 autotuner 升级为主贡献：

- reduced-space `oracle` 可获得
- `oracle gap` 数据完整
- 搜索开销可解释
- 相比固定 expert 规则确实稳定更优

---

## 3. 为什么这样冻结范围

当前做这个范围冻结，主要基于下面几条判断：

### 3.1 当前最容易形成闭环的是 non-causal prefill

现有材料已经说明：

- `A/B/C` 类实验资产已经比较完整
- 真正最值得继续投入的是 `D：优化项消融`
- `E：autotuner` 是否值得升级，取决于后续数据

所以最稳妥的推进方式不是再扩方向，而是先把 `non-causal prefill` 做深。

### 3.2 这个场景最适合讲通信优化

相比标准应用映射问题，`single-chip non-causal prefill` 更容易直接暴露：

- injector 热点
- NoC forwarding 开销
- multicast 资格受限
- layout 对覆盖率的影响
- overlap 不足造成的瓶颈

这正好对应第一阶段最想解决的系统问题。

### 3.3 decode 和 multi-chip 都更适合第二阶段

`decode` 更偏单 token 路径与归约组织。  
`multi-chip` 更偏系统扩展与跨设备通信。

这两条线当然重要，但它们会显著增加：

- 实现复杂度
- 叙事分叉
- 实验矩阵规模

因此不适合和第一阶段主线并行展开。

### 3.4 autotuner 必须有条件地推进

如果没有：

- `oracle gap`
- search overhead
- workload 差异性

那么 autotuner 很容易退化成“包装层”，而不是一个真正站得住的贡献点。

所以第一阶段的正确姿势是：

- 先做设计空间和优化项
- 再决定 autotuner 是否升级

---

## 4. 第一阶段固定研究问题

在当前冻结范围下，第一阶段只回答下面四类问题：

### 4.1 baseline 瓶颈是什么

要回答清楚：

- `NC-current-auto` 的主要瓶颈到底在哪里
- 瓶颈如何随着链长、序列长度和布局变化而切换

### 4.2 每个优化项解决了什么问题

重点关注：

- `per-chain hybrid`
- `layout-aware mapping`
- `pipelined read/forward`
- `dual NoC`

### 4.3 收益是否跨 workload 稳定

要证明收益不是只在单一 shape 或单一布局上成立，而是至少覆盖：

- 主形状
- 长序列
- `GQA / MQA`
- stress case

### 4.4 autotuner 是否值得进入主标题

第一阶段不默认回答“autotuner 一定是主贡献”，而是要通过后续结果决定：

- 升级为主贡献
- 保留为次要章节
- 降级为 future work

---

## 5. 第一批 workload family

第一批 workload 按 `F1-F4` 四类固定下来。

| Family | 目的 | 建议参数 | 当前状态 |
| --- | --- | --- | --- |
| `F1: 当前主形状` | 复用已有结果，快速迭代 | `B=1, NH=8, NKV=1, S=1024, D=128, q_chunk=128, k_chunk=128` | 已覆盖 |
| `F2: 更长序列` | 放大 `chain / sync / overlap` 问题 | `S=2048/4096/8192, D=128`，其余尽量保持接近 `F1` | 已部分覆盖 |
| `F3: GQA/MQA 变化` | 观察 `NKV` 变化对 sharing 价值的影响 | `NH=8/16, NKV=1/4, S=1024 or 2048, D=128` | 已部分覆盖 |
| `F4: 不利布局/尾部` | 验证 `hybrid/layout/tail policy` 的价值 | 非同行、非完美矩形、`q_chunk_count` 不均匀、`mcast` 不可用 | 当前最缺 |

### 5.1 核数 sweep 约定

所有 family 默认至少 sweep 一组核心数：

- `4`
- `8`
- `16`
- `24`
- `32`
- `48`
- `56/64`

### 5.2 第一批代表 workload

为了避免一开始就把实验矩阵铺得过大，第一批先固定为下面 8 个代表 case：

1. `W1_F1_base`: `B=1, NH=8, NKV=1, S=1024, D=128`
2. `W2_F2_s2048`: `B=1, NH=8, NKV=1, S=2048, D=128`
3. `W3_F2_s4096`: `B=1, NH=8, NKV=1, S=4096, D=128`
4. `W4_F2_s8192`: `B=1, NH=8, NKV=1, S=8192, D=128`
5. `W5_F3_gqa`: `B=1, NH=8, NKV=4, S=1024, D=128`
6. `W6_F3_heads16`: `B=1, NH=16, NKV=4, S=1024, D=128`
7. `W7_F4_badLayout`: `mcast` 不可用、几何布局不利的 stress case
8. `W8_F4_tailUneven`: `q_chunk_count` 不均匀的尾部 stress case

说明：

- `W1-W6` 主要用于复用已有基线和做 feature ablation
- `W7-W8` 是第一阶段必须补齐的 stress case

---

## 6. 第一阶段 baseline 冻结

### 6.1 功能对照层

第一阶段主表默认使用下面四类 baseline：

- `CP`
- `NC-naive`
- `NC-current-auto`
- `NC-proposed`

### 6.2 通信策略层

如果需要做策略消融，按下面层次组织：

- `unicast`
- `multicast`
- `per-chain hybrid`
- `tree`（仅当实现后再加入）

说明：

- `unicast` 作为弱基线和消融项保留
- 主 baseline 默认是 `NC-current-auto`

### 6.3 配置策略层

如果进入 autotuner 阶段，再增加：

- `expert default`
- `heuristic`
- `autotuner`

---

## 7. 第一阶段指标冻结

第一阶段统一使用下面这些指标：

### 主指标

- latency
- speedup vs `NC-current-auto`
- speedup vs `NC-naive`

### 通信与资源指标

- NoC bytes
- DRAM utilization
- injector stall
- receiver idle
- overlap ratio
- FPU utilization

### 兜底指标

- correctness
- seed-to-seed stability
- metadata overhead

---

## 8. 当前明确不做的内容

为了保证第一阶段闭环，下面这些内容先明确不作为必须项：

- 把 `decode` 做成第一阶段主优化主线
- 把 `multi-chip` 纳入第一阶段主实验
- 把 `tree forwarding` 作为第一批必须实现项
- 把 `rotating injector` 作为第一批必须实现项
- 把 autotuner 写成在线实时搜索
- 在没有 stress case 之前继续扩写大而全的总纲文档

---

## 9. Phase 0 完成标准

这份 scope note 生效后，`Phase 0` 可视为完成，条件如下：

1. 已明确一句话主线
2. 已明确第一阶段不做什么
3. 已固定 `F1-F4` workload family
4. 已固定第一批代表 workload
5. 已固定 baseline 与指标层次

---

## 10. 下一步

在这份文档基础上，下一步直接进入：

- `Phase 1：建立 baseline 和代码地图`

最具体的动作是：

1. 整理 `model -> op -> program factory -> kernel` 的代码路径图
2. 固定第一批实验入口和测试入口
3. 准备 `Experiment D` 的 `B0 -> B4` 消融执行顺序

# [Archived] TT-Metal SDPA KV Forwarding 优化设想、设计空间与自动寻优方案

> 已归档（2026-04-13）。
> 这份文档聚焦旧的 `prefill KV forwarding` 设计空间与 autotuner 设想，技术细节可参考，但不再代表当前论文题目与主线。
> 当前请优先参考 `current-docs.md` 与 `sf-mla-paper-positioning.md`。

## 1. 文档目标

本文基于 `docs/` 中已有的 SDPA、NoC、单播/多播分析，整理我们当前对 **SDPA 非因果 KV forwarding 路径** 的统一想法，重点覆盖以下 5 个方向：

1. 流水线化 DRAM 读取与 NoC 转发
2. 拓扑层优化
3. 树形转发替代星形拓扑
4. 轮转 Injector
5. 利用 NoC 双通道

同时，本文希望把这些想法组织成一个可执行的框架：

- 明确当前基线与瓶颈
- 把各个优化方向统一到一个**设计空间**
- 给出一个可落地的**自动寻优器**方案
- 形成后续实现与实验的路线图

本文聚焦的不是 decode 树形归约，而是 **prefill 非因果路径中的 KV chain forwarding / multicast**。不过，decode 中已经存在的树形归约思想，对我们设计树形转发和层级调度有很强的借鉴价值。

## 2. 现有基线

从 `tt-metal-sdpa-analysis.md`、NoC 文档和相关说明可以把当前基线概括为：

### 2.1 当前 forwarding 的工作方式

- 只在 **非因果、非 chunked** 场景下构建 KV forwarding。
- 一个 `injector` 核从 DRAM 读取 K/V。
- 其他处理同一 `(batch, head)` 的核心作为 `receiver`，从 injector 接收 K/V，而不是各自去 DRAM 重读。
- 支持两种基本模式：
  - `unicast`
  - `multicast`

### 2.2 当前已经具备的优点

- 已经证明“**共享 K/V 流**”可以有效降低 DRAM 重复读取。
- K/V 本身就有双缓冲基础，系统已经具备流水化雏形。
- NoC API、semaphore 协议、multicast 地址编码、CB 同步框架都已成熟。
- Host 端已经能做 chain 构建、核心映射、runtime args 注入。

### 2.3 当前主要限制

当前版本的 forwarding 仍然偏“刚性”：

- DRAM 读取与 NoC 转发偏串行，重叠不足。
- `multicast` 的判定是强约束，且整体偏保守。
- 当前更接近“injector 中心化”的转发模型，injector 很容易成为热点。
- 当前 forwarding 默认只使用 `NOC0`，没有显式利用双 NoC。
- `mcast_enabled` 的组织方式偏编译期，一旦某些 chain 不满足条件，容易出现整体退化。
- `q_chunk_count` 不均匀、物理布局不理想、跨行、长链等情况下，性能不稳定。

换句话说，当前实现已经有了“能工作、能提速”的基础版本，但还没有把 **拓扑自适应、资源均衡、通道并行、自动配置** 这几层做完整。

## 3. 我们的总体目标

我们想做的不是单点优化，而是把 forwarding 路径从“固定协议”升级为“**可调度、可建模、可搜索**”的系统。

总体目标可以概括成四件事：

1. 让 DRAM、NoC、Compute 三者的重叠更充分。
2. 让 forwarding 拓扑根据 workload 和核布局自适应。
3. 让热点从单个 injector 扩散到多个核心和多个 NoC 通道。
4. 让最终配置不靠手工拍脑袋，而是能通过设计空间搜索自动给出。

## 4. 五个核心优化方向

## 4.1 流水线化 DRAM 读取与 NoC 转发

### 4.1.1 当前问题

当前 injector 更像串行状态机：

```text
从 DRAM 读 K_0 -> 转发 K_0 -> 从 DRAM 读 V_0 -> 转发 V_0 -> 从 DRAM 读 K_1 -> ...
```

虽然底层 NoC API 是异步的，但从系统视角看，DRAM 读和 L1->L1 转发的重叠仍然不够强。

### 4.1.2 我们的想法

把 injector 改造成真正的 **read/forward pipeline**：

- 当前 chunk 正在向 receiver 转发时，下一 chunk 已经从 DRAM 开始预取。
- 如果 K/V 仍然分开发送，则让：
  - `forward(K_i)` 与 `read(V_i)` 重叠
  - `forward(V_i)` 与 `read(K_{i+1})` 重叠
- 如果后续支持 K+V 打包，则进一步变成“读 bundle / 发 bundle”的流水线。

可以把它理解成把 injector 从“blocking sender”改成“有 prefetch distance 的 producer”。

### 4.1.3 直接收益

- 隐藏 DRAM 读延迟
- 降低 injector 的空转时间
- 让 forwarding 更接近 steady-state streaming

### 4.1.4 代价与约束

- K/V CB 深度需要从 `2` 提高到 `3` 或更高，至少要支持 triple buffering
- L1 占用会上升
- 协议从“收齐 ready 再发”变成更显式的阶段化/信用化，状态机会复杂一些

### 4.1.5 对应设计空间旋钮

| 维度 | 选项 |
| --- | --- |
| `pipeline_depth` | `1`, `2`, `3`, `4` |
| `prefetch_distance` | `0`, `1`, `2` |
| `kv_transfer_mode` | `separate`, `fused_bundle` |
| `k_cb_depth` | `2`, `3`, `4` |
| `v_cb_depth` | `2`, `3`, `4` |

## 4.2 拓扑层优化

这里的“拓扑层”不是单指 NoC 物理拓扑，而是 **head 到核的映射、chain 的构造方式、以及每条 chain 选择什么 forwarding 模式**。

### 4.2.1 当前问题

当前实现里，multicast 能否启用，受到很多硬条件影响：

- 是否同行
- 是否无 gap
- `q_chunk_count` 是否一致
- `mcast_enabled` 组织方式偏全局

这会导致一个现实问题：  
只要少数 chain 不满足条件，整体就容易退回更保守的模式。

### 4.2.2 我们的想法

拓扑层优化分成三部分：

1. **Per-Chain 混合模式**
   - 每条 chain 独立决定用 `unicast`、`multicast`，还是后续的 `tree`
   - 不再要求全局“一刀切”

2. **核布局感知的 chain 构建**
   - 利用 `sub_core_grids` 或更显式的布局策略，让同一 head 的 worker 尽量落在同一物理行
   - 从“被动检查能不能 mcast”升级为“主动排布让它更可能 mcast”

3. **放宽不均匀尾部约束**
   - 对尾部 `q_chunk_count` 少 1 的核心，允许用 dummy iteration 保持协议一致
   - 用少量空转换取更多 chain 保持 multicast 资格

### 4.2.3 直接收益

- 提高 multicast 覆盖率
- 降低“最差 chain 拖累全局”的问题
- 为后续 tree / rotating injector 提供更好的布局基础

### 4.2.4 对应设计空间旋钮

| 维度 | 选项 |
| --- | --- |
| `chain_mode_policy` | `global_unicast`, `global_multicast`, `per_chain_hybrid` |
| `layout_policy` | `default`, `row_packed_by_head`, `row_packed_by_batch_head`, `custom_sub_core_grids` |
| `tail_policy` | `strict_uniform`, `dummy_iters`, `fallback_unicast` |
| `chain_partition_policy` | `contiguous`, `row_localized`, `bandwidth_balanced` |

## 4.3 树形转发替代星形拓扑

### 4.3.1 当前问题

当前更接近 injector 中心化：

- 在 unicast 下，injector 要向多个 receiver 逐个发送
- 在 multicast 下，injector 虽然一次发出，但数据源压力仍然集中在一个核心

当 chain 很长时，这会出现两个问题：

- injector 的 NoC 发射压力过大
- injector 读 DRAM 与发 NoC 的复合负担越来越重

### 4.3.2 我们的想法

当 chain 足够长时，不再坚持“一个 injector 覆盖所有 receiver”，而是采用 **层级树形转发**：

```text
injector
  -> sub-injector A -> receiver set A
  -> sub-injector B -> receiver set B
```

本质上，这是一种 forwarding fanout 的分层化：

- 根节点负责从 DRAM 读一次
- 中间节点负责二次分发
- 叶子节点只接收

树形拓扑可以是：

- 二叉树
- 四叉树
- 行内 multicast + 行间 unicast 的混合树

### 4.3.3 为什么这和 decode 的 tree reduction 有呼应

decode 路径已经证明：

- 当单点聚合/分发成本过高时
- 用树形层次化组织通信是有效的

区别在于：

- decode 是“树形归约”
- 我们这里是“树形广播/转发”

但它们共享相同的系统思想：

- 用层级结构把单点压力摊薄
- 用 `O(log N)` 级别的层次替代 `O(N)` 的中心化压力

### 4.3.4 适用区间

树形 forwarding 不一定总是最优。  
它更像是长链场景的增强策略，特别适合：

- `chain_length > 8`
- `chain_length > 16`
- injector 本身已经是 DRAM 和 NoC 双热点

### 4.3.5 对应设计空间旋钮

| 维度 | 选项 |
| --- | --- |
| `topology_mode` | `star_unicast`, `row_multicast`, `tree2`, `tree4`, `hybrid_tree` |
| `tree_arity` | `2`, `4` |
| `tree_threshold` | `4`, `8`, `12`, `16` |
| `sub_injector_policy` | `nearest`, `load_balanced`, `row_local` |

## 4.4 轮转 Injector

### 4.4.1 当前问题

当前一个 chain 在整个 K 迭代期间通常由同一个 injector 负责：

- 同一个核心读所有 K chunk / V chunk
- 同一个核心承担所有发送责任

这会导致：

- DRAM channel 压力集中
- 单核心 NoC 压力集中
- 某些布局下产生长期热点

### 4.4.2 我们的想法

让 injector 随时间轮转，而不是固定不变。

轮转粒度可以有几种：

- 每个 `k_chunk` 轮转一次
- 每 `N` 个 chunk 轮转一次
- 按 phase 轮转
- 按 head-group 轮转

也可以做更聪明的策略：

- round-robin
- 按 DRAM bank 亲和性轮转
- 按最近几轮 sender stall 轮转
- 按当前核负载加权轮转

### 4.4.3 直接收益

- 分散 DRAM 读取热点
- 降低某一个 injector 长时间成为瓶颈的概率
- 为 dual NoC 和 tree forwarding 提供更均匀的流量来源

### 4.4.4 代价与约束

- runtime args 需要支持更动态的 sender 身份
- semaphore 协议要允许 sender 身份切换
- 如果与 tree forwarding 叠加，角色管理会更复杂

### 4.4.5 对应设计空间旋钮

| 维度 | 选项 |
| --- | --- |
| `injector_policy` | `fixed`, `round_robin`, `weighted_rr`, `bank_aware`, `stall_aware` |
| `rotation_period` | `1`, `2`, `4`, `8` chunks |
| `eligible_injector_set` | `all_workers`, `row_leaders`, `sub_injectors_only` |

## 4.5 利用 NoC 双通道

### 4.5.1 当前问题

Wormhole 有 `NOC0` 和 `NOC1` 两条独立通道，但当前 forwarding 路径基本可视为主要使用单通道。

这意味着：

- K/V forwarding 没有显式利用两条通道并行
- 控制流和数据流也没有明确做通道分工

### 4.5.2 我们的想法

把双 NoC 当成真正的并行资源来建模，而不是底层细节：

1. **K / V 分通道**
   - `K -> NOC0`
   - `V -> NOC1`

2. **控制 / 数据分通道**
   - 数据流走一条 NoC
   - semaphore / control traffic 走另一条 NoC

3. **树形 forwarding 中的分层分通道**
   - 根到中间层走一个通道
   - 中间层到叶子走另一个通道

### 4.5.3 直接收益

- K/V 发送可以并行
- 控制消息不再和数据消息争同一路径
- forwarding 的 steady-state 吞吐更高

### 4.5.4 实现注意点

`NOC1` 不是简单复制 `NOC0`：

- 坐标方向相反
- multicast 范围编码也要按 `NOC1` 规则翻转

因此 dual NoC 不是只改一个枚举值，而是要把 **地址编码、矩形方向、通道配对策略** 一起纳入设计空间。

### 4.5.5 对应设计空间旋钮

| 维度 | 选项 |
| --- | --- |
| `noc_assignment_k` | `NOC0`, `NOC1` |
| `noc_assignment_v` | `NOC0`, `NOC1` |
| `noc_assignment_ctrl` | `NOC0`, `NOC1`, `follow_data` |
| `noc_split_policy` | `kv_split`, `data_ctrl_split`, `tree_level_split` |

## 5. 一个统一的设计空间

如果把上面五个方向放到一起，我们真正要搜索的是一个统一 design space，而不是 5 个孤立开关。

## 5.1 设计空间分层

可以把设计空间分成 6 层：

1. **工作负载层**
2. **映射层**
3. **拓扑层**
4. **协议层**
5. **缓冲/流水线层**
6. **通道与调度层**

## 5.2 工作负载特征

这些特征不是 tuner 的可调参数，而是它的输入特征：

| 特征 | 含义 |
| --- | --- |
| `B` | batch size |
| `NQH/NKH` | query / kv heads |
| `Sq/Sk` | Q / K 序列长度 |
| `DH` | head dim |
| `q_chunk_size / k_chunk_size` | 分块大小 |
| `q_num_chunks` | 每个 head 上的 Q 分块数 |
| `is_causal` | 是否因果 |
| `is_chunked` | 是否 paged / chunked |
| `available_cores` | 可用核心数 |
| `physical_layout` | 核的物理行列布局 |
| `L1_budget` | 当前可用 L1 预算 |

## 5.3 设计空间主表

| 类别 | 变量 | 典型取值 | 说明 |
| --- | --- | --- | --- |
| 映射 | `sub_core_grids` | default / row packed / custom | 控制 head 与核心的物理排布 |
| 映射 | `head_mapping_policy` | default / row_local / balance_dram | Head 如何映射到核心 |
| 拓扑 | `chain_mode_policy` | unicast / multicast / per_chain_hybrid | 每条 chain 用什么 forwarding |
| 拓扑 | `topology_mode` | star / tree2 / tree4 / hybrid_tree | sender fanout 结构 |
| 拓扑 | `tree_threshold` | 4 / 8 / 12 / 16 | 多长的链切换到树形 |
| 协议 | `sync_policy` | blocking / credit | sender 与 receiver 的同步风格 |
| 协议 | `tail_policy` | strict / dummy_iters / fallback | q_chunk 不均匀怎么处理 |
| 流水线 | `pipeline_depth` | 1 / 2 / 3 / 4 | DRAM 读与转发的重叠深度 |
| 流水线 | `prefetch_distance` | 0 / 1 / 2 | 提前预取距离 |
| 缓冲 | `k_cb_depth` | 2 / 3 / 4 | K buffer 深度 |
| 缓冲 | `v_cb_depth` | 2 / 3 / 4 | V buffer 深度 |
| 传输 | `kv_transfer_mode` | separate / fused_bundle | K/V 是否打包 |
| 调度 | `injector_policy` | fixed / round_robin / weighted / bank_aware | Injector 如何选 |
| 调度 | `rotation_period` | 1 / 2 / 4 / 8 | 多久轮转一次 |
| 通道 | `noc_assignment_k` | 0 / 1 | K 走哪条 NoC |
| 通道 | `noc_assignment_v` | 0 / 1 | V 走哪条 NoC |
| 通道 | `noc_assignment_ctrl` | 0 / 1 / follow_data | semaphore/control 走哪条 NoC |

## 5.4 可行性约束

不是所有组合都合法。至少要有一层 feasibility pruning：

| 约束 | 含义 |
| --- | --- |
| `multicast_requires_rectangle` | 目标必须可表达成矩形 |
| `same_l1_addr_required` | 所有目标核必须使用一致的 L1 地址布局 |
| `l1_budget_ok` | `pipeline_depth` 与 `cb_depth` 不能超过 L1 预算 |
| `noc1_addr_transform` | 选择 `NOC1` 时必须做坐标翻转和 mcast 方向变换 |
| `causal_restriction` | 当前基线只对 non-causal forwarding 生效 |
| `chunked_restriction` | paged/chunked 路径暂时不直接套用当前 forwarding |
| `tree_requires_length` | 链太短时，tree 反而不划算 |
| `rotation_requires_runtime_schedule` | 轮转 injector 需要运行时 sender 身份调度 |

## 5.5 编译期与运行时空间要分开

这是实现自动寻优器时非常关键的一点。

有些旋钮适合做 **compile-time knobs**：

- 是否支持 tree 模式
- 是否支持 credit 协议
- CB 深度
- 是否支持 dual NoC
- kernel 中是否包含 per-chain runtime branch

有些旋钮更适合做 **runtime knobs**：

- 某条 chain 用 unicast 还是 multicast
- 当前 chain 的矩形范围
- 当前轮次谁做 injector
- tree 某层的 sender / receiver 映射

这一区分决定了：

- program cache 是否能有效复用
- autotuner 搜索一次是重新编译还是只换 runtime args

我们的原则应该是：

- **把高频决策尽量做成运行时**
- **把影响 kernel 结构和 CB 布局的决策留在编译期**

## 6. 自动寻优器：我们真正希望它做什么

自动寻优器的目标不是“盲搜所有配置”，而是：

1. 从 workload 中提取关键特征
2. 生成可行候选
3. 用代价模型快速筛掉明显差的配置
4. 对少量高价值候选做实测
5. 输出最终配置，并缓存结果

## 6.1 输入与输出

### 输入

- workload 描述
  - `B, NQH, NKH, Sq, Sk, DH`
  - `q_chunk_size, k_chunk_size`
  - `is_causal, is_chunked`
- 硬件描述
  - 芯片类型
  - 可用核心网格
  - NoC 通道信息
  - DRAM channel 信息
  - L1 可用预算
- 当前程序能力
  - 是否编译了 tree / dual NOC / credit 等功能

### 输出

- `SDPAProgramConfig` 的推荐值
- 核布局建议
- 每条 chain 的 forwarding 模式
- injector 调度策略
- NoC 通道分配
- 需要的 CB 深度
- 对应的 runtime plan

一个更工程化的输出对象可以抽象成：

```text
ForwardingPlan = {
  compile_config: {
    supports_tree,
    supports_dual_noc,
    sync_policy,
    k_cb_depth,
    v_cb_depth,
    pipeline_depth,
  },
  mapping_config: {
    sub_core_grids,
    head_mapping_policy,
  },
  runtime_config: {
    per_chain_mode,
    injector_policy,
    rotation_period,
    tree_threshold,
    noc_assignment_k,
    noc_assignment_v,
    noc_assignment_ctrl,
  }
}
```

这样后续不管是落到 C++ host 侧、runtime args 生成器，还是配置缓存，都有一个统一承载体。

## 6.2 代价模型：以现有 perf model 为起点，但不能止步于它

当前代码里已经有 `create_op_performance_model(...)`，但它本质上主要估算：

- `QK` matmul FLOPs
- `PV` matmul FLOPs
- 数学精度影响
- 因果约减半

它提供的是一个 **compute floor**，但没有建模：

- DRAM 读取
- NoC 转发
- semaphore 同步
- injector 热点
- tree / multicast / dual NoC 的通信差异
- softmax/mask/协议开销

因此，自动寻优器不能只复用这个 model，而应在它上面叠加通信项。

一个更完整的目标函数可以写成：

```text
T_total ~= max(T_compute, T_dram_overlap, T_noc_overlap) + T_sync + T_tail
```

其中：

- `T_compute`
  - 来自现有 compute perf model，可作为下界
- `T_dram_overlap`
  - DRAM 读取在流水线中的有效暴露时间
- `T_noc_overlap`
  - K/V/control 在 NoC 上的关键路径时间
- `T_sync`
  - semaphore / credit / linked multicast 的同步成本
- `T_tail`
  - 由 q_chunk 不均匀、热点 injector、尾部 idle 引起的尾延迟

### 6.2.1 建议的通信子模型

可以先做一个足够简单但实用的分解：

```text
T_dram = bytes_dram / bw_dram_eff * bank_balance_factor
T_noc  = max(bytes_k / bw_noc_k, bytes_v / bw_noc_v, bytes_ctrl / bw_ctrl)
T_sync = num_handshakes * rtt_sem + num_barriers * t_barrier
T_tail = skew(q_chunk_count, injector_load, tree_depth)
```

这样哪怕一开始不完美，也已经能支持 search ranking。

## 6.3 候选生成与剪枝

推荐先做 **规则驱动的候选生成**，再做模型排序。

### 第一阶段：规则剪枝

先排除明显不合法的方案：

- 非矩形 receiver 集合，不允许单次 multicast
- L1 预算不足，不允许 `pipeline_depth=3`
- 链太短，不考虑 tree
- `NOC1` 没处理坐标翻转，不允许 dual NoC
- 当前 program 没编译进对应能力，不允许该模式

### 第二阶段：启发式生成

针对 workload 特征给出有限候选，例如：

- `chain_length <= 4`：
  - `star_unicast`
  - `row_multicast`
- `4 < chain_length <= 16`：
  - `per_chain_hybrid`
  - `multicast + pipeline_depth=2`
- `chain_length > 16`：
  - `tree2`
  - `tree4`
  - `rotating_injector + dual_noc`

## 6.4 搜索策略

不建议一开始就做完全黑盒贝叶斯优化。  
更适合我们的做法是 **分阶段层次搜索**：

### 阶段 A：选布局与大拓扑

先决定：

- `sub_core_grids`
- head 的排布
- 每条 chain 用 unicast / multicast / tree

### 阶段 B：选协议与流水线

再决定：

- blocking 还是 credit
- `pipeline_depth`
- `prefetch_distance`
- dummy iteration 是否开启

### 阶段 C：选调度与通道

最后决定：

- injector 是否轮转
- rotation 周期
- `K/V/control` 的 NoC 分配

### 阶段 D：局部微调

在 top-K 候选周围微调：

- `q_chunk_size`
- `k_chunk_size`
- `cb_depth`

这会比全局平铺暴力搜索高效很多。

## 6.5 经验评测环节

自动寻优器不能完全靠解析模型，至少要保留一个小规模 empirical phase。

建议把评测分成两类：

### 6.5.1 微基准

专门测以下 primitive：

- DRAM -> L1 带宽
- L1 -> L1 unicast 延迟 / 吞吐
- L1 -> L1 multicast 延迟 / 吞吐
- semaphore RTT
- dual NOC 并行发送收益
- tree fanout 的层次开销

### 6.5.2 Operator 级 benchmark

直接在 SDPA forwarding 场景下测：

- 端到端 latency
- DRAM 带宽利用率
- NoC 利用率
- injector stall
- receiver idle

现有的 `test/profile_sdpa_multi_core.py` 可以作为最早期的 profiling harness 种子，用来扩展出这些 forwarding 实验。

## 6.6 结果缓存

自动寻优器输出不应只存在于一次运行中，而应缓存成“workload signature -> config”映射：

```text
signature = {
  arch, B, NQH, NKH, Sq_bucket, Sk_bucket, DH,
  is_causal, is_chunked, num_cores, layout_signature
}
```

命中缓存时：

- 直接复用已有 forwarding plan
- 避免重复搜索
- 对相近 workload 也可做 warm start

## 7. 推荐的实施路线图

为了降低实现风险，建议分 3 个阶段推进。

## 7.1 Phase 0：先把基础设施立住

目标：

- 把 forwarding 设计空间显式化
- 把 compile-time / runtime knobs 分清楚
- 引入 per-chain hybrid 模式
- 引入 layout-aware mapping
- 搭建 autotuner skeleton

建议优先做：

1. `per_chain_hybrid`
2. `sub_core_grids` 驱动的 row-aware 布局
3. 设计空间对象与可行性剪枝
4. 代价模型 v0

## 7.2 Phase 1：做最稳的性能增量

目标：

- 流水线化 DRAM read + forward
- dual NoC
- rotating injector

这些方向的共同特点是：

- 收益比较直接
- 不必先改成层级拓扑
- 适合作为 autotuner 的第一批关键 knobs

## 7.3 Phase 2：上更激进的拓扑

目标：

- tree forwarding
- credit-based protocol
- paged/chunked 扩展

这是更高风险但也更有系统价值的一阶段，尤其适合长链和极限带宽场景。

## 7.4 Phase 3：更远期的扩展

目标：

- common-prefix causal forwarding
- KV 压缩转发
- 更通用的 runtime adaptive protocol

这一阶段更偏研究性质，不建议和 P0/P1 混在一起推进。

## 8. 建议的优先级重排

结合现有文档分析和实现风险，我建议的优先级如下：

| 优先级 | 方向 | 原因 |
| --- | --- | --- |
| `P0` | Per-Chain 混合模式 | 投入低，马上能减少“整体退化” |
| `P0` | 核布局感知 | 很多后续优化都依赖这个基础 |
| `P0` | 设计空间对象 + autotuner 骨架 | 不先建这个，后续优化会继续散 |
| `P1` | 流水线化 DRAM 读与转发 | 直接提升 injector steady-state 吞吐 |
| `P1` | NoC 双通道 | 与流水线配合后收益更明显 |
| `P1` | 轮转 Injector | 用于摊薄热点，提升鲁棒性 |
| `P2` | 树形 forwarding | 长链收益大，但协议复杂 |
| `P2` | credit 协议 | 和流水线一起做更合理 |
| `P3` | paged/chunked 扩展 | 价值高，但耦合更多模块 |
| `P3` | common-prefix causal | 收益巨大，但最复杂 |

## 9. 我们最终想得到什么

如果把这套方案做完整，我们最终应该能得到的不是“又多了几个 if 分支”，而是一套更系统的机制：

1. forwarding 不再是固定实现，而是可调拓扑
2. 核布局、协议、流水线、injector、NoC 分配都能一起建模
3. compile-time 与 runtime 决策被明确拆开
4. autotuner 能根据 workload 自动选出 forwarding plan
5. 新优化不会继续散落成难以维护的 patch，而是都收敛到统一 design space

## 10. 一句话总结

我们真正想做的，是把当前 SDPA 非因果路径里的 KV forwarding，从“一个能工作的 chain/multicast 优化”，升级成“一个可配置、可分层、可自动搜索的通信子系统”。

这 5 个方向里：

- **流水线化** 解决重叠不足
- **拓扑层优化** 解决全局刚性
- **树形转发** 解决长链中心化瓶颈
- **轮转 injector** 解决热点不均
- **双 NoC** 解决单通道利用不足

而设计空间和自动寻优器，则是把这些想法真正变成工程能力的关键。

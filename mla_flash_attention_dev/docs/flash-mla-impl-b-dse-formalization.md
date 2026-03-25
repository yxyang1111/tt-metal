# FlashMLA 实现 B 的参数化与 DSE 设计文档

## 1. 目的

本文基于以下四份材料，把 `实现 B / FlashMLA` 形式化为一个可以做 DSE（Design Space Exploration，设计空间探索）的参数系统：

- `mla_flash_attention_dev/docs/mla-two-implementations-deep-dive.md`
- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`
- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/rt_args_common.hpp`
- `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`

目标不是再解释一次实现流程，而是回答下面三个问题：

1. 实现 B 到底有哪些“可变参数”？
2. 哪些参数是真正适合做 DSE 的主变量，哪些只是派生量或环境量？
3. 这些参数之间的约束关系是什么，应该按什么顺序去 sweep？

本文默认“当前代码”为真值。特别注意：Wormhole 当前实现已经不是简单的“6 个物理 DRAM controller”，而是通过 `get_optimal_dram_bank_to_logical_worker_assignment()` 使用 bank view / bank endpoint 级别的映射，因此 DSE 应优先围绕 `bank endpoint` 而不是只围绕 `physical controller id` 建模。

---

## 2. 核心抽象：Block-Lane 模型

实现 B 最适合被抽象成一个二维结构，而不是简单地理解成“一批核”：

- `N_S`: S Block 数量
- `C_S`: 每个 S Block 内的 core 数

把第 `i` 个 S Block 的第 `b` 个 core 记为 `core(i, b)`，则：

- `i in [0, N_S)` 是 **序列并行 / DRAM stream 维度**
- `b in [0, C_S)` 是 **Q shard lane 维度**

定义：

```text
Lane b = { core(0,b), core(1,b), ..., core(N_S-1,b) }
```

这条抽象很重要，因为它把两个最核心的结构参数分开了：

- `N_S` 决定每个 Q shard 有多少个序列并行参与者，也基本决定可同时打开多少个 DRAM 读流
- `C_S` 决定一个 launch 最多能同时承载多少个 Q shard lane，也决定一次 K multicast 的扇出规模

这也解释了当前代码里的两个关键约束：

```text
B = total_q_heads / num_q_heads_per_core
B <= C_S
device_chunk_size = C_S * k_chunk_size
```

其中：

- `B` 是当前 launch 中的 Q shard 数
- `C_S` 不是 sequence parallel 宽度，而是 lane 容量
- `N_S` 才是每个 lane 的 sequence parallel 宽度

---

## 3. 形式化配置向量

把一个 FlashMLA 配置记为：

```text
Theta = (A, W, G, M, P, C, X)
```

其中：

- `A`: Architecture，硬件架构与 NoC/DRAM 拓扑
- `W`: Workload，模型与输入规模
- `G`: Grid / Topology，S Block 几何与 lane 映射
- `M`: Memory，sharding / chunk / page / tile
- `P`: Pipeline，reader-writer overlap、multicast、VC、同步
- `C`: Compute，数值精度与 compute kernel 配置
- `X`: Scale-out，多设备 SP 参数

后文所有参数都放进这 7 类里。

---

## 4. 一级设计变量：真正建议做 sweep 的参数

### 4.1 架构与拓扑参数 `A + G`

| 参数 | 符号 | 当前 BH | 当前 WH | 作用 | DSE 地位 |
|---|---|---|---|---|---|
| 架构族 | `arch` | `blackhole` | `wormhole_b0` | 决定 grid 尺寸、NoC 包大小、DRAM 端点分布 | 一级变量 |
| NoC 最大软件包大小 | `P_noc_max` | 16 KB | 8 KB | 限制 `P_k` 的上界 | 架构常量 |
| NoC 最大 transaction id | `TRID_max` | `0xF` | `0xF` | 限制 `N_trid`，当前可用窗口通常是 14 | 架构常量 |
| NOC 选择 | `noc_id` | 当前仅优化 `NOC0` | 当前仅优化 `NOC0` | S Block 坐标依赖 NOC 镜像关系 | 一级变量，当前未暴露 |
| S Block 数量 | `N_S` | 8 | 6 | 决定每个 lane 的 sequence parallel 宽度 | 一级变量 |
| 每块 core 数 | `C_S` | 8 | 4 | 决定 lane 容量、K fanout 扇出、device chunk 大小 | 一级变量 |
| S Block 形状 | `shape(S_i)` | `4 col x 2 row` | `2 col x 2 row` | 决定 multicast 矩形、DRAM 邻近性、空闲核分布 | 一级变量 |
| S Block 物理坐标 | `coords(S_i)` | 硬编码 | 硬编码 | 决定 hop 距离与矩形是否合法 | 一级变量 |
| Bank endpoint 映射 | `bank_map(i)` | `(1,3,2,0,5,7,6,4)` | 当前代码实际为 `(1,2,0,4,9,8)` | 决定 DRAM 邻近性与 ND sharding 顺序 | 一级变量 |
| Tree reduction 拓扑 | `tree_order` | 8 -> 4 -> 2 -> 1 | 6 -> 3 -> 2 -> 1 | 决定 reduction 路径长度与并行度 | 一级变量 |
| Sender 选择策略 | `sender(i)` | 每个 block 的第 0 个 core | 每个 block 的第 0 个 core | 决定谁负责读 DRAM 和做 K multicast | 一级变量，当前固定 |

### 4.2 存储与流式参数 `M`

| 参数 | 符号 | 当前状态 | 作用 | DSE 地位 |
|---|---|---|---|---|
| K chunk 大小 | `K_c = k_chunk_size` | `ProgramConfig` 直接暴露，默认 128 | 决定 chunk 数、页数、L1 占用、mask 粒度、读写重叠效率 | 一级变量 |
| Q heads / core | `H_c = num_q_heads_per_core` | 通过 Q shard spec 和 tiny tile 间接决定 | 决定 lane 数 `B`、Q tile 高度、Q fanout 规模 | 一级变量 |
| Q tile 高度 | `Tq_h` | 当前通常等于 `H_c` | 决定 `PNHt = H_c / Tq_h`，影响 tiny-tile 利用率 | 一级变量 |
| KV ND shard 顺序 | `shard_order` | 当前等于 `OPTIMAL_DRAM_BANK_ORDER` | 决定 chunk -> bank endpoint 映射 | 一级变量 |
| Page size 策略 | `P_k` | 当前自动选“最大可整除 page” | 决定每 chunk 的 page 数与 overlap 粒度 | 一级变量，但当前未显式暴露 |
| DRAM page 数 | `N_page` | 由 `P_k` 派生 | 决定 trid/BRISC pipeline 深度 | 派生量 |
| K 双缓冲深度 | `buf_k` | 当前固定为 2 | 支撑 DRAM read 与 compute overlap | 二级变量，当前固定 |
| 输出/中间结果 CB 大小 | `buf_o`, `buf_ms` | 由 topology 和 tile 派生 | 决定 L1 占用与 tree reduction 缓冲 | 派生量 |

### 4.3 Pipeline 与同步参数 `P`

| 参数 | 符号 | 当前状态 | 作用 | DSE 地位 |
|---|---|---|---|---|
| trid 窗口深度 | `N_trid` | 当前固定为 `NOC_MAX_TRANSACTION_ID - 1` | 决定 DRAM 读流水线并发度 | 一级变量，但当前未暴露 |
| VC 分配策略 | `vc_policy` | 当前按 block index 硬编码到 4 个 VC | 影响 NoC 冲突 | 一级变量，但当前未暴露 |
| K fanout 策略 | `k_fanout` | 当前为 sender `noc_async_write_multicast` | 决定 K 分发成本 | 一级变量，当前固定 |
| Q fanout 策略 | `q_fanout` | 当前为“ready 信号 multicast + receiver 从 output core 读 Q” | 决定 Q 分发成本 | 一级变量，当前固定 |
| KV ready 门槛 | `kv_ready_value` | 当前固定 3 | 与上游 fused op 耦合，影响启动时序 | 系统参数 |
| Receiver-ready 协议 | `recv_ready_protocol` | 当前固定 | 保证双缓冲地址一致性 | 二级变量 |
| Full-grid 信号范围 | `signal_scope` | 当前是 full-device grid | 影响 ready semaphore 广播成本 | 一级变量，当前未暴露 |

### 4.4 Compute 与数值参数 `C`

| 参数 | 符号 | 当前状态 | 作用 | DSE 地位 |
|---|---|---|---|---|
| Math fidelity | `mf` | `compute_kernel_config` 直接暴露 | 影响精度与吞吐 | 一级变量 |
| Math approx mode | `approx_math` | 直接暴露 | 影响 SFPU/算子近似 | 一级变量 |
| FP32 dst accumulator | `fp32_dst` | 直接暴露 | 影响数值稳定性与 `dst_size` | 一级变量 |
| Dst full sync | `dst_sync` | 直接暴露 | 影响 `dst_size` 与调度 | 一级变量 |
| `dst_size` | `D_dst` | 由 `fp32_dst` 和 `dst_sync` 派生 | 决定 tail reduction 的 block 划分 | 派生量 |
| Scale | `alpha` | 调用时输入 | 影响 softmax 数值范围 | 环境量 |
| `head_dim_v` | `D_v` | 调用时输入 | 决定 AV 输出 tile 数 | 环境量 |
| `exp_approx_mode` | `exp_mode` | API 暴露，但 kernel 内当前硬编码为 `false` | 名义上影响 `exp` 近似方式 | “假旋钮” |

### 4.5 多设备参数 `X`

| 参数 | 符号 | 当前状态 | 作用 | DSE 地位 |
|---|---|---|---|---|
| SP 设备数 | `N_sp` | 在 fused path 中使用 | 决定 round-robin device chunk 切分 | 一级变量 |
| 当前设备索引 | `sp_idx` | fused path 运行时输入 | 决定 local cur pos 和写 cache 归属 | 环境量 |
| 每设备 chunk 大小 | `D_chunk` | `ProgramConfig` 暴露，但 standalone `FlashMLA.op()` 不直接用 | 决定全局位置到本地位置映射 | 一级变量 |
| 本地 cur pos | `pos_local` | 由 `get_device_mla_work_assignment()` 派生 | 决定本设备需要处理的有效序列长度 | 派生量 |

---

## 5. 环境参数：不属于实现 knob，但决定最优点

这些参数不一定由 FlashMLA 内核本身控制，但 DSE 时必须一起记录，否则“最优参数”没有可比性：

| 参数 | 符号 | 含义 |
|---|---|---|
| 总 Q head 数 | `H_q` | 决定 lane 数 `B = H_q / H_c` |
| KV head 数 | `H_kv` | 当前实现要求 `H_kv = 1` |
| KV batch 维度 | `B_kv` | 当前实现要求 `B_kv = 1` |
| `kvpe_dim` | `D_k` | K/Q 参与 attention 的总维度 |
| `head_dim_v` | `D_v` | V 从 K 中前 `D_v` 列读取 |
| 当前 decode 位置 | `pos` | 决定活跃 chunk 数 |
| `max_seq_len` | `S_max` | 决定总 shard 数和总 cache 容量 |
| Q/K 数据类型 | `dtype_q`, `dtype_k` | 当前常见为 `bf16` / `bf8_b` |
| 设备 grid 尺寸 | `G_dev` | 决定是否容纳特定 S Block 布局 |

---

## 6. 派生变量与关键公式

### 6.1 Lane 与 active core 数

```text
B = H_q / H_c
B <= C_S

num_cores_per_batch = N_S
num_active_cores = B * N_S
```

解释：

- `B` 是活跃 Q shard lane 数
- 每个 lane 用一个 “跨 block 的竖切片”
- 所以总 active core 数是 `lane 数 x 每个 lane 的 block 数`

### 6.2 Chunk 分解

```text
valid_seq_len = align_up(pos + 1, K_c)
num_chunks = valid_seq_len / K_c
active_s_blocks = min(num_chunks, N_S)     # NKV=1 的当前实现语义
```

含义：

- 短序列时，不是所有 S Block 都真的干活
- `N_S` 过大时，长序列可能受益，短序列则会有更多 idle block

### 6.3 K page 粒度

设：

```text
K_TILE_H = 32
TILE_W   = 32
DHt      = D_k / TILE_W
Sk_t     = K_c / K_TILE_H
k_chunk_tiles = Sk_t * DHt
k_chunk_bytes = k_chunk_tiles * k_tile_size
```

当前 page 选择策略为：

```text
P_k = max { p | p <= noc_max_page_size, p % k_tile_size = 0, k_chunk_bytes % p = 0 }
N_page = k_chunk_bytes / P_k
```

这意味着：

- `K_c` 不只影响 chunk 数，也会影响 `P_k` 与 `N_page`
- `K_c` 越大，不一定越好
- 太大时可能降低 overlap 细粒度、提高 L1 压力
- 太小时会增加 chunk 数和 tree reduction 次数

### 6.4 Device chunk

当前 `ProgramConfig` 中有：

```text
D_chunk = C_S * K_c
```

这不是偶然关系，而是当前实现的结构性假设：

- `C_S` 决定一个 block 内的 lane 容量
- `K_c` 决定每个 chunk 的 token 数
- 多设备 SP 时，本地可连续承载的 token 区间长度被建模为二者乘积

### 6.5 Tail reduction block 划分

当前代码中：

```text
dst_size = f(fp32_dst, dst_sync)
vDHt = D_v / 32
要求: vDHt % dst_size == 0
```

这说明数值配置会反向约束 tail reduction 的合法性，而不是只影响精度。

---

## 7. 当前实现的硬约束

下面这些约束在当前代码里是“必须满足”，否则不是编译失败，就是 runtime assert，或者虽然能跑但语义不完整。

| 约束 | 形式 | 说明 |
|---|---|---|
| Q 前两维固定 | `q_shape[0] = 1`, `q_shape[1] = 1` | 当前 standalone FlashMLA 只覆盖简化 decode 形态 |
| Q heads 可整除 | `H_q % H_c = 0` | lane 数必须是整数 |
| Q heads / core 上限 | `H_c < 32` | 当前 tiny tile 路径要求 |
| Q shard 必须装进 block | `B <= C_S` | lane 数不能超过 block 容量 |
| KV 必须是 ND sharded DRAM | `kv_mem_config.is_sharded()` 且有 `nd_shard_spec` | 当前 reader 依赖 ND shard 读地址 |
| KV shard 高度等于 chunk 大小 | `kv_shard_height = K_c` | 每个 shard 必须对应恰好一个 K chunk |
| 当前只支持 `PNHt = 1` | `H_c / Tq_h = 1` | 也就是 Q tile 高度需要等于 heads/core |
| 当前只支持 `H_kv = 1` | `num_kv_heads = 1` | 现实现是 MLA/NKV=1 特化 |
| 当前只支持 `B_kv = 1` | `Bkv = 1` | KV batch 维度未一般化 |
| 输出维度约束 | `vDHt % dst_size = 0` | tail reduction 的静态要求 |
| 设备 grid 下界 | BH: `x >= 11`, `y = 10`; WH: `x >= 8`, `y >= 7` | 必须装得下 block 布局 |
| 物理 multicast 矩形合法 | `coords(S_i)` 在物理坐标上必须是合法矩形或 torus 矩形 | K multicast 依赖 |

还需要特别标出一个**事实上的隐含约束**：

```text
当前测试覆盖主要使用 B = C_S
```

原因是：

- `physical_multicast_coords()` 目前按“整块”构造矩形
- `num_mcast_dests` 也是按“整块”计算
- 因此若 `B < C_S`，虽然显式断言允许，但实际 fanout / semaphore / inactive core 行为还没有被系统性验证

换句话说，现阶段做 DSE 时，建议把 `B = C_S` 当成第一阶段的保守搜索规则。

---

## 8. 哪些是“真旋钮”，哪些不是

### 8.1 当前已经生效，且值得 sweep

| 参数 | 说明 |
|---|---|
| `arch / grid family` | 不同芯片的最优解很可能不同 |
| `N_S`, `C_S`, `shape(S_i)`, `coords(S_i)` | 这是实现 B 的核心结构参数 |
| `bank_map(i)` / `shard_order` | 直接影响 DRAM 局部性 |
| `k_chunk_size` | 影响 chunk 数、page 数、L1 占用、mask 粒度 |
| `H_c` / `Tq_h` | 影响 lane 数、Q tiny-tile 利用率 |
| `mf`, `approx_math`, `fp32_dst`, `dst_sync` | 影响吞吐与精度 |
| `N_sp`, `D_chunk` | 多设备路径下必须 sweep |

### 8.2 当前生效，但还没被正式暴露出来

| 参数 | 说明 |
|---|---|
| `noc_id` | 当前布局只为 `NOC0` 定制 |
| `P_k` | 现在是自动策略，未作为独立 knob |
| `N_trid` | 现在直接取硬件上限，未做软件 sweep |
| `vc_policy` | 现在按 block index 硬编码 |
| `signal_scope` | 当前 Q ready 信号是 full-device grid 范围 |
| `sender(i)` | 当前固定为 block 第 0 个 core |
| `q_fanout` 策略 | 当前是“信号 multicast + Q peer read”，未做替代策略比较 |

### 8.3 API 上存在，但当前并没有真正影响 kernel 行为

| 参数 | 现状 |
|---|---|
| `ProgramConfig.exp_approx_mode` | Python API 有这个字段，但 `flash_mla.hpp` 当前把 `exp_approx_mode` 直接写死为 `false` |

### 8.4 只适合记录，不适合单独 sweep

| 参数 | 原因 |
|---|---|
| `N_page` | 完全由 `K_c`、dtype、`P_k` 决定 |
| `num_active_cores` | 完全由 `B` 和 `N_S` 决定 |
| `num_mcast_dests` | 当前等于 `C_S - 1` |
| `pos_local` | 由 `pos`、`sp_idx`、`D_chunk`、`N_sp` 决定 |
| `dst_size` | 由 `fp32_dst` 和 `dst_sync` 决定 |

---

## 9. 推荐的 DSE 搜索顺序

不要一开始就把所有参数一起 sweep。推荐按“结构 -> 流式 -> 数值 -> 多设备”的顺序逐层展开。

### 9.1 Phase 1：先扫拓扑

固定：

- `K_c = 128`
- `H_c = Tq_h`
- `mf = LoFi`
- `fp32_dst = false`
- `dst_sync = false`

优先 sweep：

- `arch`
- `N_S`
- `C_S`
- `shape(S_i)`
- `coords(S_i)`
- `bank_map(i)`
- `tree_order`

观察指标：

- 单 token latency
- active core ratio
- DRAM bank 冲突
- tree reduction 开销
- multicast 合法性与稳定性

### 9.2 Phase 2：在固定拓扑上扫 memory / streaming

优先 sweep：

- `K_c`
- `P_k` 或 `page_size_policy`
- `N_trid`
- `vc_policy`

观察指标：

- DRAM 吞吐
- reader/BRISC overlap 程度
- `cb_k_in` 的 L1 占用
- page 级同步开销

### 9.3 Phase 3：扫 Q lane 与 tiny tile

优先 sweep：

- `H_c`
- `Tq_h`

观察指标：

- `B = H_q / H_c` 是否能充分填满 `C_S`
- Q fanout 开销
- tiny tile 的 padding 浪费

### 9.4 Phase 4：扫 compute / numeric

优先 sweep：

- `mf`
- `approx_math`
- `fp32_dst`
- `dst_sync`

观察指标：

- PCC / max error
- latency
- tail reduction 是否仍满足静态约束

### 9.5 Phase 5：最后再扫多设备

优先 sweep：

- `N_sp`
- `D_chunk`

观察指标：

- local_cur_pos 分布
- device ownership 切换开销
- skip_attention / skip_kv_cache_update 比例

---

## 10. 建议的搜索裁剪规则

为了避免设计空间爆炸，建议先用下面这些 pruning rules：

| 规则 | 原因 |
|---|---|
| 先限制 `B = C_S` | 当前 fanout / rectangle / semaphore 路径主要在这个条件下验证过 |
| 只考虑 `K_c` 为 32 的整数倍 | K 使用标准 `32 x 32` tile |
| 只考虑物理上是合法 multicast 矩形的 block | 否则 K fanout 不成立 |
| 若保持 1 reader / bank 亲和，则 `N_S` 不要超过可用 bank endpoint 数 | 否则会退化为 bank sharing |
| `P_k` 先从“最大可整除值”开始，再向更小 page 试探 | 大 page 吞吐好，小 page overlap 好 |
| 先固定 `q_fanout` 和 `sender(i)` | 避免同时改变太多通信路径 |
| 多设备 DSE 前先把单设备拓扑收敛 | 否则难以分辨瓶颈来自 chip 内还是 chip 间 |

---

## 11. 建议记录的 DSE 结果表

建议把每个配置记录成如下字段，便于后续做 Pareto 分析：

| 字段 | 含义 |
|---|---|
| `config_id` | 配置编号 |
| `arch` | `blackhole` / `wormhole_b0` |
| `noc_id` | `0` / `1` |
| `N_S`, `C_S` | block 结构参数 |
| `shape(S_i)` / `coords(S_i)` | 拓扑实例 |
| `bank_map` | bank endpoint 顺序 |
| `tree_order` | reduction 拓扑 |
| `K_c` | K chunk size |
| `H_c`, `Tq_h` | Q lane / tiny tile 参数 |
| `P_k`, `N_page`, `N_trid` | reader pipeline 参数 |
| `vc_policy` | NoC 策略 |
| `mf`, `fp32_dst`, `dst_sync`, `approx_math` | 数值配置 |
| `N_sp`, `D_chunk` | 多设备参数 |
| `latency_us` | 单 token 延迟 |
| `dram_gbps` | DRAM 有效带宽 |
| `active_core_ratio` | 活跃核比例 |
| `noc_stall_ratio` | NoC 侧 stall 比例 |
| `pcc` / `max_abs_err` | 正确性指标 |
| `compile_ok` / `run_ok` | 可用性指标 |

---

## 12. 为正式 DSE 还需要补的代码改造

如果要把当前实现真正升级成“可系统 sweep”的 DSE 平台，建议优先做下面 6 件事：

1. 把 `grid` 从“硬编码类”升级成“可生成的 topology config”
2. 显式暴露 `page_size_policy` 或 `k_page_size_override`
3. 显式暴露 `trid_window`
4. 显式暴露 `vc_policy`
5. 把 `exp_approx_mode` 真正从 Python 传到 kernel
6. 把 `B < C_S` 的 partial-block 路径补齐并单测

建议的配置接口可以长成这样：

```python
FlashMLADSEConfig(
    arch="wormhole_b0",
    noc_id=0,
    num_s_blocks=6,
    cores_per_block=4,
    block_coords=...,
    bank_map=(1, 2, 0, 4, 9, 8),
    tree_order=(((0, 1), (2, 3), (4, 5)), ((0, 2),), ((0, 4),)),
    k_chunk_size=128,
    num_q_heads_per_core=8,
    q_tile_height=8,
    page_size_policy="auto_max_divisor",
    trid_window=14,
    vc_policy="4vc_split",
    q_fanout="peer_read_after_signal",
    k_fanout="sender_multicast",
    math_fidelity="LoFi",
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    dst_full_sync_en=False,
    exp_approx_mode=False,
    num_sp_devices=1,
    device_chunk_size="auto",
)
```

---

## 13. 推荐的首轮 DSE 空间

如果现在就要开始做第一轮 sweep，我建议不要从“所有参数”开始，而是先扫下面这个最小可用空间：

### 13.1 Blackhole

```text
N_S          in {6, 8}
C_S          in {4, 8}
K_c          in {64, 128, 256}
H_c = Tq_h   in {4, 8, 16}
mf           in {LoFi, HiFi2}
fp32_dst     in {false, true}
dst_sync     in {false, true}
bank_map     in {proximity-based baseline, one or two hand-tuned permutations}
```

### 13.2 Wormhole

```text
N_S          in {4, 6}
C_S          in {4, 6}
K_c          in {64, 128, 256}
H_c = Tq_h   in {4, 8}
mf           in {LoFi, HiFi2}
fp32_dst     in {false, true}
dst_sync     in {false, true}
bank_map     in {current optimal-worker order, 1-2 个局部扰动版本}
```

注意：WH 第一轮 DSE 建议继续围绕“矩形 block + bank endpoint 邻近”这一原则展开，不要一上来就尝试很激进的非矩形布局。

---

## 14. 总结

对实现 B 来说，真正应该被形式化的不是单个孤立参数，而是下面 4 组强耦合变量：

1. `N_S x C_S`：决定 sequence parallel 宽度与 lane 容量
2. `coords(S_i) + bank_map(i) + tree_order`：决定拓扑、局部性和归约代价
3. `K_c + P_k + N_trid + vc_policy`：决定 DRAM streaming 的效率
4. `H_c + Tq_h + mf + fp32_dst + dst_sync`：决定 tiny-tile 利用率、数值与 compute 性能

因此，最合理的 DSE 组织方式不是“把所有参数平铺 sweep”，而是：

```text
先拓扑 -> 再 chunk/page -> 再 tiny-tile/lane -> 再 numeric -> 最后多设备
```

按这个结构推进，后面无论是做 BH 深度优化，还是把实现 B 系统性迁移到 WH / 多设备场景，都会更清晰，也更容易形成可复用的 autotuning 框架。

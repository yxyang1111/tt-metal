# TT FlashMLA 理论性能模型与四种组合评估

## 1. 目标

本文为 TT 当前两套 MLA decode attention 实现建立一个统一的理论性能模型，并给出下列四个组合的细粒度评估：

- 实现 A on Wormhole
- 实现 A on Blackhole
- 实现 B on Wormhole
- 实现 B on Blackhole

这里的“性能”拆成 3 个层次：

1. `DRAM`：KV cache 从片外 DRAM 进入芯片的读取压力
2. `NoC`：Q fanout、K multicast、tree reduction 等片上网络传播
3. `Tensix`：TRISC 上的 `QK matmul + online softmax + AV matmul`

本文只覆盖 **single-token decode 的 attention core**，也就是从 `Q_attn + KV cache` 到 `attn_out` 这一段。`wkv_b1 / wkv_b2 / wo` 等外围 matmul 不并入主表，但会在结论里单独讨论它们对端到端时延的影响。

## 2. 范围与事实边界

- 实现 A 指 `models/demos/deepseek_v3/tt/mla/mla1d.py` 中调用的 `paged_flash_multi_latent_attention_decode`
- 实现 B 指 `models/demos/deepseek_v3_b1/micro_ops/flash_mla/` + `unified_kernels/flash_mla.hpp`
- `A-WH` 是当前生产路径的直接建模
- `B-BH` 是当前实验性 FlashMLA 的直接建模
- `A-BH` 是把 A 的现有 decode 拓扑原样搬到 BH，仅替换硬件常数后的理论估计
- `B-WH` 是把 B 的 Wormhole 6-block 拓扑代入后的理论估计；当前 `op.py` 在 WH 上仍走 `_wh_reference_fallback()`，所以这一列是 **结构性预测，不是当前可执行内核的实测值**

主要依据：

- `mla_flash_attention_dev/docs/mla-two-implementations-deep-dive.md`
- `mla_flash_attention_dev/docs/flash-mla-impl-b-dse-formalization.md`
- `mla_flash_attention_dev/docs/tenstorrent-wormhole-blackhole-architecture.md`
- `mla_flash_attention_dev/docs/mla-performance-analysis.md`
- `models/demos/deepseek_v3/tt/mla/mla1d.py`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`
- `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`
- `ttnn/core/operation.cpp`

## 3. 建模输入

### 3.1 共同数据形状

用当前 MLA decode 的主形状建模：

- `Sq = 1`
- `D_qk = 576`
- `D_v = 512`
- `k_chunk_size = 128`
- `Q/O = bf16 = 2 B`
- `KV cache = bf8 = 1 B`

因此：

- 单个 `K chunk` 大小 = `128 x 576 x 1 B = 73,728 B = 72 KB`
- 单个 lane 的 `Q` 大小若 `8 heads/core` = `8 x 576 x 2 B = 9,216 B`
- 单个 lane 的部分 `O` 大小若 `8 heads/core` = `8 x 512 x 2 B = 8,192 B`

### 3.2 两个架构采用的硬件常数

| 项 | Wormhole | Blackhole | 说明 |
|---|---:|---:|---|
| AI 时钟 | 1.0 GHz | 1.35 GHz | 架构文档 / UMD 常量 |
| DRAM 总带宽 | 258 GB/s | 512 GB/s | WH 这里故意采用 `ttnn/core/operation.cpp` 的 repo-side 建模值，而不是 288 GB/s 的 marketing/spec 值 |
| DRAM bank 数 | 6 | 8 | 用来估算 ND-sharding 后的每 bank 压力 |
| Tensix L1 可见容量 | 1,499,136 B | 1,572,864 B | 影响 CB 装载余量 |
| NoC payload 宽度 | 256 bit | 512 bit | 来自 `noc_parameters.h` |
| 单 NoC 理论链路带宽 | 32 GB/s | 86.4 GB/s | `payload_width x f_clk` |
| NOC 最大软件包 | 8 KB | 16 KB | B 的分页流水线直接受此约束 |
| DRAM 读对齐 | 32 B | 64 B | B 的 page 选择要满足它 |
| `NOC_MAX_TRANSACTION_ID` | 0xF | 0xF | B 使用 `NUM_TRIDS = 14` 的 page pipeline |

### 3.3 两种实现采用的“当前拓扑”

#### 实现 A

根据 `mla-two-implementations-deep-dive.md` 中的 decode 说明，A 的当前代表性布局可简化为：

- `4` 个 Q-shard lane
- 每个 lane 用 `4` 个核做 sequence parallel
- 总活跃核数 `16`
- compute fidelity 采用 `HiFi4`
- decode 中没有 K multicast；同一份 K 会被不同 lane 独立从 DRAM 读取

#### 实现 B on WH

根据 `FlashMLAOptimalGridNOC0_WH` 与 `test_flash_mla_wh.py`：

- 本地 heads：`32`
- `4` 个 lane，每 lane `8` heads
- `6` 个 S block，每 block `4` 核
- 总活跃核数 `24`
- 每 block 的第 0 个核负责读 DRAM + K multicast
- compute fidelity 采用 `LoFi`

#### 实现 B on BH

根据 `FlashMLAOptimalGridNOC0` 与 `test_flash_mla.py`：

- 本地 heads：`64`
- `8` 个 lane，每 lane `8` heads
- `8` 个 S block，每 block `8` 核
- 总活跃核数 `64`
- 每 block 的第 0 个核负责读 DRAM + K multicast
- compute fidelity 采用 `LoFi`

#### 实现 A on BH

当前仓库没有 A 的 BH 专门拓扑，因此本文采用保守假设：

- 保持 A 的 `4 lane x 4 cores/lane = 16 active cores`
- 只替换 BH 的 DRAM / NoC / clock 常数
- 这样得到的是“架构替换后的保守下界”，不是 A 在 BH 上 fully retuned 的上界

## 4. 建模过程

### 4.1 分阶段时间模型

统一用下面的临界路径表达式：

```text
T_core ~= T_q_preamble + max(T_reader, T_compute) + T_reduce
```

其中：

- `T_q_preamble`：Q fanout 或 Q 预备阶段
- `T_reader`：K 的 DRAM 读取，以及实现 B 中 sender->receivers 的 K 分发
- `T_compute`：Tensix 上的 `QK + softmax + AV`
- `T_reduce`：多核 / 多 block 的 tail reduction

### 4.2 实现 A 的公式

实现 A 的 decode 路径没有显式 Q multicast，因此：

```text
T_A_core ~= max(T_A_dram, T_A_compute) + T_A_reduce
```

其关键项为：

```text
K_A_bytes = L_A * S * D_qk * b_k
```

这里 `L_A = 4`，因为 4 个 lane 会各自把整条序列的 K 再读一遍。于是：

```text
T_A_dram = K_A_bytes / BW_dram
```

计算项：

```text
F_decode = 2 * H_local * S * (D_qk + D_v)
T_A_compute = F_decode / (N_active_A * Phi_A_core * f_clk)
```

本文取：

- `N_active_A = 16`
- `Phi_A_core = 1024 flop/cycle`（HiFi4）

归约项近似为：

```text
T_A_reduce ~= R_A * O_lane_bytes * h_reduce / BW_noc
```

其中 `R_A = 2`，因为 `4 cores/lane` 对应 `log2(4)=2` 轮归约。

### 4.3 实现 B 的公式

实现 B 的关键变化是把 K 的片外流量从“每 lane 一遍”改成“每 launch 一遍”，随后用 NoC 做 block-local 扩散：

```text
K_B_bytes = S * D_qk * b_k
T_B_dram = K_B_bytes / BW_dram
```

Q fanout 采用“output core 发信号，其他 block 对 output core 做 peer read”，因此把它单独记成前导项：

```text
T_B_q ~= ((N_S - 1) * Q_lane_bytes * h_q) / BW_noc
```

K multicast 的关键点是区分：

- `injected bytes`：sender 真正注入 NoC 的字节数
- `delivered bytes`：网络最终送达所有 receivers 的总字节数

临界路径更接近 sender 端注入，因此本文用：

```text
T_B_kmcast ~= ((K_B_bytes / N_S) * h_k) / BW_noc
```

而总 delivered traffic 另外记录为：

```text
K_B_delivered = (C_S - 1) * S * D_qk * b_k
```

计算项：

```text
T_B_compute = F_decode / (N_active_B * Phi_B_core * f_clk)
```

本文取：

- `Phi_B_core = 4096 flop/cycle`（LoFi）
- `N_active_B = 24` on WH
- `N_active_B = 64` on BH

最终：

```text
T_B_core ~= T_B_q + max(T_B_dram, T_B_kmcast, T_B_compute) + T_B_reduce
```

### 4.4 B 的 page pipeline 细节

对 `72 KB` 的一个 `K chunk`：

- WH：`noc_max_page_size = 8 KB`，刚好切成 `9` 个 page
- BH：最大 16 KB 不能整除 72 KB，按 `get_max_page_size_and_num_pages()` 会退到 `12 KB`，切成 `6` 个 page
- 两边都使用 `NUM_TRIDS = 14`，所以 page-level pipeline 深度足够覆盖单 chunk 的 page 数

这意味着：

- `B-WH` 的 sender/BRISC 同步更细碎，NoC 固定成本稍高
- `B-BH` 每 chunk page 更少，流水线更容易接近 steady-state

### 4.5 用 A-WH 现有实测做 sanity check

`mla-performance-analysis.md` 给出 A-WH decode 的两条实测：

- `D-32h-1k`: `0.223 ms`
- `D-32h-4k`: `0.275 ms`

把上面的 `T_A_core` 再加一个单 op 的 `~0.15 ms` dispatch 下界，可得到：

| Case | 模型值 (`T_core + 0.15`) | 实测 | 偏差 |
|---|---:|---:|---:|
| A-WH, 32h, 1k | 0.160 ms | 0.223 ms | 1.39x |
| A-WH, 32h, 4k | 0.188 ms | 0.275 ms | 1.47x |

这个误差带是合理的：page-table 访问、barrier、CB backpressure、softmax 控制开销都没有被 fully materialize。后文其他三列也应理解为 **结构正确的理论下界**，真实值通常会落在它上方约 `1.2x ~ 1.5x`。

## 5. Wormhole：A vs B

### 5.1 Wormhole 下的关键中间量

固定：

- 本地 heads：`32`
- `A-WH`: `4 lanes x 4 cores/lane = 16 active cores`
- `B-WH`: `4 lanes x 6 blocks = 24 active cores`

由此得到：

- A 的 K DRAM 总字节：`4 x S x 576`
- B 的 K DRAM 总字节：`1 x S x 576`
- 因而 B 把 **片外 K 流量直接压到 A 的 1/4**

按 6 个 DRAM bank 均匀分摊时，每 bank 平均 K 负载为：

| `S` | A-WH 每 bank | B-WH 每 bank |
|---|---:|---:|
| 1,024 | 0.375 MB | 0.094 MB |
| 4,096 | 1.500 MB | 0.375 MB |
| 32,768 | 12.000 MB | 3.000 MB |

### 5.2 A-WH 结果

| `S` | K DRAM (MB) | `T_dram` (ms) | `T_compute` (ms) | `T_reduce` (ms) | `T_core` (ms) | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---|
| 1024 | 2.25 | 0.0091 | 0.0044 | 0.0010 | 0.0102 | DRAM |
| 4096 | 9.00 | 0.0366 | 0.0174 | 0.0010 | 0.0376 | DRAM |
| 32768 | 72.00 | 0.2926 | 0.1393 | 0.0010 | 0.2936 | DRAM |

解读：

- 在 WH 上，A 的 decode core 基本一直被 **重复的 K DRAM 读取**主导
- `T_reduce` 只有 KB 级，不是主要问题
- 当 `S` 从 `4k` 到 `32k`，时延几乎按 DRAM 字节线性增长

### 5.3 B-WH 结果

这里采用 `FlashMLAOptimalGridNOC0_WH` 的 `6 block x 4 cores` 拓扑。

| `S` | K DRAM (MB) | `T_dram` (ms) | `T_q` (ms) | `T_kmcast` (ms) | `T_compute` (ms) | `T_reduce` (ms) | `T_core` (ms) | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 0.56 | 0.0023 | 0.0043 | 0.0041 | 0.0007 | 0.0023 | 0.0107 | K mcast |
| 4096 | 2.25 | 0.0091 | 0.0043 | 0.0164 | 0.0029 | 0.0023 | 0.0230 | K mcast |
| 32768 | 18.00 | 0.0732 | 0.0043 | 0.1311 | 0.0232 | 0.0023 | 0.1377 | K mcast |

补充的 NoC 容量信息：

- Q fanout delivered bytes：约 `180 KB`，基本是固定成本
- K multicast delivered bytes：
  - `1k`: `1.69 MB`
  - `4k`: `6.75 MB`
  - `32k`: `54.00 MB`

解读：

- B-WH 把片外 DRAM 压力成功降下来了，但瓶颈转移到了 **sender -> receivers 的 K multicast**
- 在 `32k` 时，`T_kmcast` 已明显大于 `T_dram`
- 换句话说：B 在 WH 上不再是“外存带宽问题”，而是“片上 NoC 分发问题”

### 5.4 Wormhole 结论

按 core-only 下界比较：

- `1k`: A-WH `0.0102 ms`，B-WH `0.0107 ms`，几乎持平
- `4k`: A-WH `0.0376 ms`，B-WH `0.0230 ms`，B 快 `1.63x`
- `32k`: A-WH `0.2936 ms`，B-WH `0.1377 ms`，B 快 `2.13x`

如果两者都再加一个单 op 的 `0.15 ms` dispatch，下界会变成：

- `1k`: `0.1602 ms` vs `0.1607 ms`
- `4k`: `0.1876 ms` vs `0.1730 ms`
- `32k`: `0.4436 ms` vs `0.2877 ms`

所以在 Wormhole 上，B 的优势主要出现在 **中长序列**，而且优势来自“把瓶颈从 DRAM 搬到 NoC”。短序列时，dispatch 和固定同步成本会吃掉大部分收益。

## 6. Blackhole：A vs B

### 6.1 Blackhole 下的关键中间量

固定：

- 本地 heads：`64`
- `A-BH`: 仍保守地用 `4 lanes x 4 cores/lane = 16 active cores`
- `B-BH`: 原生 `8 lanes x 8 blocks = 64 active cores`

按 8 个 DRAM bank 均匀分摊时，每 bank 平均 K 负载为：

| `S` | A-BH 每 bank | B-BH 每 bank |
|---|---:|---:|
| 1,024 | 0.281 MB | 0.070 MB |
| 4,096 | 1.125 MB | 0.281 MB |
| 32,768 | 9.000 MB | 2.250 MB |

### 6.2 A-BH 结果

| `S` | K DRAM (MB) | `T_dram` (ms) | `T_compute` (ms) | `T_reduce` (ms) | `T_core` (ms) | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---|
| 1024 | 2.25 | 0.0046 | 0.0064 | 0.0011 | 0.0076 | Compute |
| 4096 | 9.00 | 0.0184 | 0.0258 | 0.0011 | 0.0269 | Compute |
| 32768 | 72.00 | 0.1475 | 0.2063 | 0.0011 | 0.2075 | Compute |

解读：

- 只把 A 从 WH 搬到 BH，而不改它的拓扑，BH 更高的 DRAM 带宽会先把 `T_dram` 压低
- 结果是 A-BH 的瓶颈从 WH 上的 DRAM 变成 **HiFi4 + 16 active cores 下的 compute**
- 这说明 A 不是简单“换更强芯片就自然更快”，它还需要配套改变 lane / core 拓扑

### 6.3 B-BH 结果

这里采用 `FlashMLAOptimalGridNOC0` 的原生 `8 block x 8 cores` 布局。

| `S` | K DRAM (MB) | `T_dram` (ms) | `T_q` (ms) | `T_kmcast` (ms) | `T_compute` (ms) | `T_reduce` (ms) | `T_core` (ms) | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 0.56 | 0.0012 | 0.0037 | 0.0019 | 0.0004 | 0.0011 | 0.0068 | K mcast |
| 4096 | 2.25 | 0.0046 | 0.0037 | 0.0077 | 0.0016 | 0.0011 | 0.0126 | K mcast |
| 32768 | 18.00 | 0.0369 | 0.0037 | 0.0614 | 0.0129 | 0.0011 | 0.0663 | K mcast |

补充的 NoC 容量信息：

- Q fanout delivered bytes：约 `504 KB`
- K multicast delivered bytes：
  - `1k`: `3.94 MB`
  - `4k`: `15.75 MB`
  - `32k`: `126.00 MB`

解读：

- BH 的 DRAM 足够强，使得 B-BH 很早就不再受 DRAM 限制
- 真正压住 B-BH 的是 **K multicast 的片上传播**
- 但 BH 的 NoC 宽度和时钟也足够高，所以即便进入 NoC-bound，绝对时延仍然很低

### 6.4 Blackhole 结论

按 core-only 下界比较：

- `1k`: A-BH `0.0076 ms`，B-BH `0.0068 ms`
- `4k`: A-BH `0.0269 ms`，B-BH `0.0126 ms`，B 快 `2.14x`
- `32k`: A-BH `0.2075 ms`，B-BH `0.0663 ms`，B 快 `3.13x`

若都加一个单 op 的 `0.15 ms` dispatch，下界变成：

- `1k`: `0.1576 ms` vs `0.1568 ms`
- `4k`: `0.1769 ms` vs `0.1626 ms`
- `32k`: `0.3575 ms` vs `0.2163 ms`

所以在 BH 上，B 的优势更大，原因不是单纯 DRAM 更大，而是：

1. B 的 `one-pass DRAM + block-local multicast` 结构更适合 8-bank BH
2. BH 的 512-bit NoC 能把 multicast 的代价压住
3. A 如果不改变拓扑，只会把瓶颈从 DRAM 换成 compute，并不会自然接近 B

## 7. 四种组合放到一起看

### 7.1 瓶颈迁移图

| 组合 | 主瓶颈迁移 |
|---|---|
| A-WH | 明确的 DRAM-bound |
| B-WH | 从 DRAM-bound 迁移为 NoC-bound（主要是 K multicast） |
| A-BH | 从 DRAM-bound 迁移为 compute-bound（因为外存更快，但拓扑未变） |
| B-BH | 仍是 NoC-bound，但绝对值最低 |

### 7.2 真正重要的结构差异

不是所有优势都来自“BH 比 WH 强”，更关键的是实现边界：

| 结构项 | 实现 A | 实现 B | 直接后果 |
|---|---|---|---|
| K 片外读取 | 每个 lane 读一遍 | 每个 launch 只读一遍 | B 把 K DRAM 压到 A 的 `1 / L` |
| Q 分发 | 本地 / 各核自取 | output core fanout | B 增加了小的固定 NoC 成本 |
| K 分发 | 无 | sender -> receivers multicast | B 把瓶颈从 DRAM 换到 NoC |
| compute fidelity | HiFi4 | LoFi | B 在 decode 上更偏吞吐 |
| 活跃核数 | 16 | 24 (WH) / 64 (BH) | B 更能摊薄 compute |

### 7.3 对端到端 decode 链路的含义

上面的表只看 attention core。本仓库里真正的端到端 decode 还要加上外围 op：

- A 的完整链路还有 `wq_kv_a / q_norm / kv_norm / wq_b / wkv_b1 / RoPE / concat / cache update / a2a / wkv_b2 / wo`
- `mla-two-implementations-deep-dive.md` 给出的经验值是：A 的完整 decode 链路有 `~12-18` 次独立 dispatch
- B 虽然把 SDPA core 做成了一个 `generic_op`，但 `wkv_b1 / wkv_b2 / wo` 仍未融合到同一个 kernel

因此：

- 如果只替换 attention core，B 的优势主要在中长序列显现
- 如果未来继续把 `wkv_b1 / wkv_b2 / wo` 并入 B，端到端收益会远大于本文表里的 core-only 倍数

## 8. 结论

### 8.1 对 Wormhole

- A-WH 的核心问题是 **4 个 lane 对同一份 K 的重复 DRAM 读取**
- B-WH 已经能把这个问题转化成片上 K multicast；在 `4k` 和 `32k` 上，core-only 理论下界分别比 A-WH 快 `1.63x` 和 `2.13x`
- WH 上若继续演进 B，优先级已经不再是 DRAM，而是 **K multicast / Q preamble / reduction 的 NoC 调度**

### 8.2 对 Blackhole

- 单纯把 A 搬到 BH，只会把瓶颈从 DRAM 换成 compute，不会自动变成“像 B 一样快”
- B-BH 是四种组合里最合理的结构：DRAM 一次读完、NoC 宽、活跃核多
- 在 `32k` 下，B-BH 的 core-only 下界是 `0.0663 ms`，相对 A-BH 的 `0.2075 ms` 有 `3.13x` 的结构优势

### 8.3 最关键的一句话

**实现 A 和实现 B 的本质差异，不是“一个在 WH、一个在 BH”，而是 A 让共享 K 的代价留在了 DRAM，B 则把这部分代价前移到了片上 NoC。**

对 decode 这种 `Sq=1`、`Sk` 很长的工作负载，这个结构差异比单纯提升芯片峰值带宽更重要。

## 9. 后续建议

如果接下来要把这个理论模型继续推进成更接近真实 runtime 的工程模型，我建议按下面顺序补：

1. 对 B 补一个更真实的 `NoC efficiency` 项，把 VC 冲突、semaphore、page barrier 的气泡计进去
2. 对 A-BH 做一版“8 lane / 更高 active-core”的敏感性分析，验证它是否会重新回到 DRAM-bound
3. 把 `wkv_b1 / wkv_b2 / wo` 纳入端到端模型，给出 “core-only” 与 “full decode” 两套时延表
4. 若 WH kernel bring-up 稳定后，用 `B-WH` 实测把本文的 `T_q` / `T_kmcast` 系数再校准一轮

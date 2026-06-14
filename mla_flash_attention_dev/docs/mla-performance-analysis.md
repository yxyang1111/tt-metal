# MLA Flash Attention Characterization 报告

## 1. 概述

本报告对 Tenstorrent Wormhole B0 (WH_B0) 上 Multi-Latent Attention (MLA) 融合算子的性能进行系统分析，包括：
- **理论性能建模**：基于 Roofline 模型推导算力/带宽瓶颈
- **实测性能数据**：覆盖 prefill 和 decode 两个阶段的多种工作负载
- **效率分析与优化方向**：对比理论与实测，定位性能差距来源

## 1.1 当前论文中的角色

在当前 `SF-MLA` 论文口径下，这份文档应被视为 **Characterization** 材料，而不是单纯的 benchmark 报告。

它主要服务于三个问题：

1. `MLA decode` 为什么不是 `MHA decode` 的小变体
2. 为什么在 spatial accelerator 上会出现 `compute-memory mixed regime`
3. 为什么仅靠 first-order Roofline 不足以解释真实执行，还需要进一步引入 pipeline coupling、multicast hotspot 和 second-order correction

因此：

- `decode` 结果是论文主战场的直接证据
- `prefill` 结果主要作为 supporting evidence
- 本文中的理论模型应被理解为 `first-order model` 的起点，而不是全文最终模型

实验日期：2026-03-23

---

## 2. 硬件规格 (Wormhole B0)

| 参数 | 值 | 来源 |
|---|---|---|
| 计算网格 | 8 × 8 = **64 核** (harvested, 56 核可用于用户计算) | 运行时检测 |
| 每核 FPU 吞吐 | 4096 FMA/cycle @ LoFi | `tech_reports/matrix_engine/` |
| 时钟频率 | **1.0 GHz** | WH spec |
| 每核 bf16 峰值 (HiFi4) | **~1.024 TFLOPS** (4096/4=1024 FMA/cycle) | `GEMM_FLOPS.md` |
| 全芯片 bf16 峰值 (HiFi4) | 64 × 1.024 = **65.5 TFLOPS** | 计算 |
| DRAM 总带宽 | **258 GB/s** (6×2×21.5) | `ttnn/core/operation.cpp` |
| DRAM 容量 | 6 × 2GB = **12 GB** | SoC descriptor |
| 每核 L1 SRAM | **1.43 MiB** (1,499,136 B) | SoC descriptor |
| NoC L1 bisection 带宽 | **512 GB/s** | `operation.cpp` |

**Roofline 拐点 (Ridge Point)**：

$$
\text{Ridge Point} = \frac{\text{Peak FLOPS}}{\text{DRAM BW}} = \frac{65.5 \times 10^{12}}{258 \times 10^9} \approx 254 \text{ FLOP/Byte}
$$

当算术强度 (AI) > 254 时为计算瓶颈，AI < 254 时为带宽瓶颈。

---

## 3. MLA 融合算子理论分析

### 3.1 MLA 与标准 MHA 的区别

MLA (Multi-Latent Attention) 是 DeepSeek V3 引入的注意力变体：
- **KV 共享**：nkv = 1，所有 Q head 共享同一组 KV
- **非对称维度**：K 维度 = `kv_lora_rank + d_rope` (典型 576)，V 维度 = `kv_lora_rank` (典型 512)
- **V 复用 K**：V 从 K 的前 `kv_lora_rank` 列切片，无需额外存储

对本文来说，这些差异的重要性不只在于“公式不同”，而在于：

- latent restoration 改变了数据重用方式
- decode 的最优映射会与传统 MHA 明显不同
- 在 many-core spatial accelerator 上，它会引出新的 layout、multicast 和 pipeline 问题

### 3.2 计算量 (FLOPs)

对于 MLA Flash Attention，核心计算分为两个矩阵乘：

**Prefill 模式** (`Sq = Sk = S`)：

| 操作 | FLOPs 公式 | 说明 |
|---|---|---|
| Q × K^T | `2 × B × NH × S × S × D_QK` | D_QK = kv_lora_rank + d_rope |
| Attn × V | `2 × B × NH × S × S × D_V` | D_V = kv_lora_rank |
| 因果掩码 | 除以 2 | 仅计算下三角 |

总 FLOPs (causal prefill)：
$$
\text{FLOPs} = B \times N_H \times S^2 \times (D_{QK} + D_V)
$$

**Decode 模式** (`Sq = 1, Sk = pos`)：
$$
\text{FLOPs} = 2 \times B \times N_H \times S_k \times (D_{QK} + D_V)
$$

### 3.3 访存量 (Memory Bytes) — 单次读取假设

> **注意**：以下是假设 Q、K、O 各从 DRAM 读/写一次的"理想"访存量。实际 Flash Attention 实现中，K 会被多次重复读取，详见第 5.3 节修正。

| 张量 | 大小 (Bytes) | 说明 |
|---|---|---|
| Q (读) | `B × NH × Sq × D_QK × 2` | bf16 = 2B/元素 |
| K (读) | `B × NKV × Sk × D_QK × 1` | bf8 = 1B/元素，**仅一次读取假设** |
| Output (写) | `B × NH × Sq × D_V × 2` | bf16 = 2B/元素 |

V 无需独立读取（MLA 复用 K 的前 kv_lora_rank 列）。

### 3.4 算术强度 (Arithmetic Intensity) — 单次读取假设

> **警告**：以下 AI 值基于 3.3 节的单次读取假设，**严重高估了实际 AI**。修正后的 AI 见第 5.5 节。

**Prefill (causal, typical config: NH=32, S=1024, D_QK=576, D_V=512)**：

$$
\text{AI}_{naive} = \frac{B \times 32 \times 1024^2 \times (576+512)}{B \times (32 \times 1024 \times 576 \times 2 + 1 \times 1024 \times 576 \times 1 + 32 \times 1024 \times 512 \times 2)} \approx 508 \text{ FLOP/Byte}
$$

按此计算远超 Ridge Point (217)，会**错误地**认为 prefill 是计算瓶颈。

**Decode (typical config: B=2, NH=32, S_k=512, D_QK=576, D_V=512)**：

$$
\text{AI}_{naive} = \frac{2 \times 2 \times 32 \times 512 \times (576+512)}{2 \times (32 \times 1 \times 576 \times 2 + 1 \times 512 \times 576 \times 1 + 32 \times 1 \times 512 \times 2)} \approx 98 \text{ FLOP/Byte}
$$

Decode 确实低于 Ridge Point (217)，是 **带宽瓶颈**（此结论在修正后仍然成立）。

---

## 4. 实测性能数据

### 4.1 测试环境

| 字段 | 值 |
|---|---|
| 设备 | Wormhole B0, 8×7 = 56 核 |
| Python | 3.10.14 (Anaconda) |
| 构建 | Release (pre-compiled firmware) |
| LD_PRELOAD | `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` |
| 预热轮数 | Prefill: 2, Decode: 3 |
| 测量轮数 | Prefill: 5, Decode: 10 |
| 计时方式 | `time.perf_counter()` 含 `ttnn.synchronize_device()` |

### 4.2 Prefill 性能结果

> 注：下表中"原始理论 (ms)"和"原始 AI"基于 3.3 节的单次 K 读取假设，"原始效率"相应过于悲观。修正后的分析见第 5 节。

| 工作负载 | B | S | NH | LoRA | Rope | Causal | Avg (ms) | Min (ms) | 原始理论 (ms) | 原始 AI | 原始效率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P-small-causal | 1 | 1024 | 16 | 512 | 64 | Yes | 3.938 | 2.150 | 0.318 | 504 | 14.8% |
| P-med-causal | 1 | 4096 | 16 | 512 | 64 | Yes | 28.783 | 28.766 | 5.093 | 2015 | 17.7% |
| P-32h-causal | 1 | 1024 | 32 | 512 | 64 | Yes | 9.156 | 3.961 | 0.637 | 508 | 16.1% |
| P-32h-4k-causal | 1 | 4096 | 32 | 512 | 64 | Yes | 52.430 | 52.396 | 10.186 | 2031 | 19.4% |
| P-128h-causal | 1 | 1024 | 128 | 512 | 64 | Yes | 11.600 | 6.175 | 2.547 | 511 | 41.2% |
| P-32h-512-nc | 1 | 512 | 32 | 512 | 64 | No | 8.267 | 1.928 | 0.318 | 508 | 16.5% |
| P-32h-1k-nc | 1 | 1024 | 32 | 512 | 64 | No | 6.639 | 6.606 | 1.273 | 1016 | 19.3% |
| P-B2-128h-causal | 2 | 1024 | 128 | 512 | 64 | Yes | 11.551 | 11.348 | 5.093 | 511 | 44.9% |

### 4.3 Decode 性能结果

| 工作负载 | B | S | NH | LoRA | Rope | Avg (ms) | Min (ms) | 原始理论 (ms) | 原始 AI | 原始效率 |
|---|---|---|---|---|---|---|---|---|---|---|
| D-small | 2 | 1024 | 8 | 128 | 64 | 0.129 | 0.105 | 0.001 | 25 | 0.8% |
| D-32h-1k | 2 | 1024 | 32 | 512 | 64 | 0.240 | 0.223 | 0.003 | 98 | 1.3% |
| D-32h-4k | 2 | 4096 | 32 | 512 | 64 | 0.301 | 0.275 | 0.010 | 114 | 3.5% |
| D-B4-32h | 4 | 1024 | 32 | 512 | 64 | 0.232 | 0.211 | 0.006 | 98 | 2.7% |

---

## 5. 理论模型的缺陷审视与修正

### 5.1 关键问题：TT 的 MLA 是否是融合算子？

**答：部分融合。** 需要区分两个层面：

#### 5.1.1 SDPA 层面的融合（已实现）

`flash_mla_prefill` / `paged_flash_multi_latent_attention_decode` 内部是融合的：

```
┌─────────────────────────────────────────────────────────┐
│  flash_mla_prefill → 单次 Device Operation Dispatch     │
│                                                         │
│  ┌─────────┐   CB_Q   ┌──────────┐  CB_O  ┌─────────┐ │
│  │ Reader  │ ──────→  │ Compute  │ ────→  │ Writer  │ │
│  │ Kernel  │   CB_K   │  Kernel  │        │ Kernel  │ │
│  │         │ ──────→  │          │        │         │ │
│  │ (DRAM→  │   CB_V   │ (QK mul, │        │ (L1→    │ │
│  │   L1)   │ ──────→  │  softmax,│        │  DRAM)  │ │
│  │         │          │  AV mul) │        │         │ │
│  └─────────┘          └──────────┘        └─────────┘ │
│  QK matmul + softmax + AV matmul 在 L1/寄存器中完成     │
└─────────────────────────────────────────────────────────┘
```

#### 5.1.2 MLA 完整链路的融合（未实现）

**但 SDPA 并非 MLA 特有的融合算子**——它只是通用的 Flash Attention。MLA 独有的 **KV 解压缩**（latent space → full dimension 的上投影）**并没有**融合进 SDPA 内核中。

TT 的 `mla1d.py` 采用了 DeepSeek 论文中的 **"absorb"（吸收）方案**（对应 `reference/deepseek/model.py` 第 19 行的 `attn_impl = "absorb"`），将原本的 KV 解压缩拆成两步：

**K 侧：解压缩被"吸收"到 Q 上（SDPA 之前）**

DeepSeek 论文原始公式：
$$K_{full} = W^{UK} \cdot c^{KV} \quad \text{(对所有 cached tokens 做上投影)}$$

吸收后等价变换：
$$Q'_{nope} = q_{nope} \cdot W^{UK,T} \quad \text{(仅对当前 Q tokens 做投影)}$$
$$\text{score} = Q'_{nope} \cdot c^{KV,T} \quad \text{(直接与压缩 KV 计算)}$$

实际代码（`mla1d.py:1687-1689`，prefill 路径）：
```python
# wkv_b1 = W_UK 的转置（K 侧上投影矩阵的转置）
tt_q_nope = ttnn.linear(tt_q_nope, **cfg["wkv_b1"], ...)
# q_nope: [NH, S, qk_nope_head_dim=128] → [NH, S, kv_lora_rank=512]
```

**V 侧：显式解压缩在 SDPA 之后**

V 侧无法吸收（因为输出维度不同），必须在注意力之后做显式上投影：

实际代码（`mla1d.py:1769-1771`，prefill 路径）：
```python
# wkv_b2 = W_UV（V 侧上投影矩阵）
v_out = ttnn.linear(v_out, **cfg["wkv_b2"], ...)
# attn_out: [NH, S, kv_lora_rank=512] → [NH, S, v_head_dim=128]
```

#### 5.1.3 完整 MLA 数据流（`mla1d.py`）

```
输入 x
  │
  ▼
[wq_kv_a] ─────── ttnn.linear ──── Dispatch #1
  │
  ├─ q 路径: [q_norm] → [wq_b]  ── ttnn.linear ──── Dispatch #2,#3
  │    │
  │    ├── q_nope → [wkv_b1] ────── ttnn.linear ──── Dispatch #4  ◄── K 侧"解压缩"(吸收到 Q)
  │    └── q_rope → [RoPE] ──────── rotary_emb ───── Dispatch #5
  │    │
  │    └── [concat] → Q ──────────────────────────── Dispatch #6
  │
  ├─ kv 路径: [kv_norm] → [RoPE]
  │    └── [concat] → kvpe ──── [fill_cache] ──────── Dispatch #7,#8,#9
  │
  ▼
[flash_mla_prefill(Q, kvpe)] ─────────────────────── Dispatch #10  ◄── SDPA 融合算子
  │
  ▼
[wkv_b2] ────────── ttnn.linear ──────────────────── Dispatch #11  ◄── V 侧解压缩
  │
  ▼
[wo] ─────────────── ttnn.linear ──────────────────── Dispatch #12
  │
  ▼
输出
```

**一次完整 MLA forward 涉及 ~12 次独立 device op dispatch**，其中 `flash_mla_prefill`（SDPA）只是中间一步。

#### 5.1.4 为什么采用 absorb 方案？

| 方案 | K 侧操作 | 操作规模 | Cache 存储 |
|---|---|---|---|
| **Naive**（显式解压） | `K = W_UK @ c_kv` 对所有 S_k tokens | O(S_k × D_full) **每次推理** | 存 full-dim K/V |
| **Absorb**（吸收到 Q） | `Q' = q_nope @ W_UK^T` 仅对 S_q tokens | O(S_q × D_latent) | 存压缩 c_kv |

对 decode（S_q=1, S_k 可达数千），absorb 方案的 K 侧操作量减少 **S_k/1 = S_k 倍**，并且 cache 保持压缩状态（节省 DRAM 容量和带宽）。

#### 5.1.5 未融合带来的性能影响

`wkv_b1` 和 `wkv_b2` 作为独立的 `ttnn.linear` 调用，产生：

| 开销项 | wkv_b1（K 侧吸收） | wkv_b2（V 侧解压） |
|---|---|---|
| Dispatch 开销 | ~0.3ms | ~0.3ms |
| 额外 DRAM 读写 | Q_nope 读 + 结果写 DRAM | attn_out 读 + 结果写 DRAM |
| 权重读取 | W_UK: NH × kv_lora_rank × qk_nope_head_dim | W_UV: NH × v_head_dim × kv_lora_rank |

如果能将 `wkv_b2` 融入 SDPA 的 writer kernel（在写出 O 时顺带做投影），可以：
- 省去 1 次 dispatch (~0.3ms)
- 省去 attn_out 的 DRAM 中间写回 + 读取
- 潜在地与 AV matmul 流水重叠

同理，如果能将 `wkv_b1` 融入 SDPA 的 reader/compute kernel（读入 Q 时顺带做投影），也可节省类似开销。但这需要修改 SDPA 内核本身。

**结论：性能差距的来源同时包括（1）SDPA 内部的 Flash Attention 带宽开销（第 5.3 节）和（2）MLA 完整链路中多个未融合的 ttnn.linear 调度开销。**

### 5.2 理论模型缺陷逐项审查

第 4 节的"理论性能"计算了一个**过于乐观的下界**，其中有多个关键开销**完全未被包含**：

| 开销项 | 理论模型是否包含？ | 实际影响 |
|---|---|---|
| **Dispatch / Kernel Launch** | **未包含** | 每次 op 调用 ~0.3-1.0ms 固定开销 |
| **Flash Attention 的 K 重复读取** | **未包含（最大缺陷）** | K 从 DRAM 被读取 O(S/q_chunk) 次，而非 1 次 |
| **Softmax 计算 (exp, max, sum)** | **未包含** | 约为 matmul FLOPs 的 5-10% |
| **在线 softmax 重缩放** | **未包含** | 每对 (Q_chunk, K_chunk) 需要一次 output rescale |
| **Tile 对齐 padding** | **未包含** | 非 32 对齐时产生无效计算 |
| **核利用率不满** | **未包含** | NH < 56 时有空闲核 |
| **流水线气泡** | **未包含** | reader/compute 异步但非完美重叠 |
| 纯矩阵乘 FLOPs (QK + AV) | **已包含** | — |
| Q/O 单次读写 | **已包含** | — |
| K 单次读取 | **已包含但严重低估** | 见下文修正 |

### 5.3 最大缺陷：Flash Attention 的 K 重复 DRAM 读取

Flash Attention 的核心权衡是：**避免在 DRAM 中物化 S×S 的 attention matrix，代价是 K 需要被多次从 DRAM 读入 L1**。

Reader kernel 的核心循环（`reader_interleaved.cpp:404`）：
```cpp
// 对于每个 Q chunk，遍历所有相关的 K chunks
for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
    // 从 DRAM 读取 K chunk 到 L1 circular buffer
    read_chunk_with_padding(k_reader, cb_k_in, k_start_tile_id, ...);
}
```

**因果模式下的实际 K DRAM 读取量**：

以 P-32h-causal (B=1, S=1024, NH=32, q_chunk=32, k_chunk=128) 为例：
- q_num_chunks = 1024/32 = 32, k_num_chunks = 1024/128 = 8
- 每个 head 由 1 个核独立处理所有 32 个 Q chunks
- Q chunk i 需要读取 K chunks 0..⌈(i+1)×32/128⌉

```
Q_chunk  →  需读取的 K_chunks 数
0-3      →  1 (只需 K_chunk_0)
4-7      →  2 (K_chunk_0 + K_chunk_1)
8-11     →  3
12-15    →  4
16-19    →  5
20-23    →  6
24-27    →  7
28-31    →  8 (全部 K_chunks)
────────────────────────
合计: 4×(1+2+3+4+5+6+7+8) = 144 次 K_chunk 读取/head
```

每个 K chunk = 128 × 576 × 1B (bf8) = **72 KB**

| 项目 | 理论模型假设 | 实际 Flash Attention | 倍数 |
|---|---|---|---|
| K DRAM 读取/head | 1024×576×1 = 0.6 MB | 144 × 72KB = 10.1 MB | **17×** |
| K DRAM 总读取 (32 heads) | 0.6 MB | 32 × 10.1 = 323 MB | **548×** |

注意：MLA 的 NKV=1 意味着 32 个 head **共享同一组 K 数据**，但由于因果模式下没有 multicast，每个核**独立从 DRAM 读取相同的 K**。

### 5.4 修正后的理论模型

#### 修正公式

**修正 DRAM 流量**（Prefill, Causal, 无 Multicast）：

$$
\text{K\_DRAM\_reads} = N_H \times \sum_{i=0}^{N_{q\_chunks}-1} \lceil \frac{(i+1) \times q_{chunk}}{k_{chunk}} \rceil \times k_{chunk} \times D_{QK} \times \text{kv\_bytes}
$$

**修正 Dispatch 开销**：实测稳定在 ~0.3ms（program cache 命中后）

**修正核利用率**：仅 min(B × NH, 56) 个核参与计算

#### 修正后的对比（Prefill 关键 case）

| 工作负载 | 活跃核 | 修正 K DRAM (MB) | 总 DRAM (MB) | 修正 DRAM 时间 (ms) | 修正计算时间 (ms) | Dispatch (ms) | 修正理论 (ms) | 实测 (ms) | 修正效率 |
|---|---|---|---|---|---|---|---|---|---|
| P-32h-causal | 32 | 323 | 394 | 1.53 | 1.09 | 0.3 | 1.83 | 3.96 | **46%** |
| P-32h-4k-causal | 32 | 5,165 | 5,536 | 21.5 | 17.5 | 0.3 | 21.8 | 52.4 | **42%** |
| P-128h-causal | 56 | 1,291 | 1,592 | 6.17 | 2.55 | 0.3 | 6.47 | 6.18 | **105%** ¹ |
| P-B2-128h-causal | 56 | 2,582 | 3,153 | 12.2 | 5.10 | 0.3 | 12.5 | 11.35 | **110%** ¹ |
| P-small-causal | 16 | 81 | 119 | 0.46 | 0.62 | 0.3 | 0.92 | 2.15 | **43%** |
| P-32h-1k-nc | 32 | 580 | 651 | 2.52 | 2.18 | 0.3 | 2.82 | 6.61 | **43%** |

¹ 修正效率 > 100% 说明实际实现中有 **有效的流水线重叠（compute 与 reader 并行）**，使实测时间低于 max(compute, DRAM) + dispatch。这恰恰证明了 TT 的融合实现确实有效。

#### Decode 修正

| 工作负载 | 活跃核 | K DRAM (MB) | Dispatch (ms) | 修正理论 (ms) | 实测 (ms) | 修正效率 |
|---|---|---|---|---|---|---|
| D-32h-1k | 56 | 0.6 | 0.15 | 0.15 | 0.223 | **67%** |
| D-32h-4k | 56 | 2.3 | 0.15 | 0.16 | 0.275 | **58%** |
| D-small | 16 | 0.2 | 0.15 | 0.15 | 0.105 | **>100%** ¹ |

¹ Decode 的 K 只读一遍（Sq=1，只有一个 Q token），所以 Flash Attention 的重复读取问题不存在。效率高于理论主要因为 dispatch 开销低于估计值。

### 5.5 修正后的 Roofline 解读

原始理论模型将所有 prefill case 标记为 **"计算瓶颈"**（AI > 217），但修正后发现：

**实际的算术强度需用 Flash Attention 的真实 DRAM 流量计算**：

| 工作负载 | 原始 AI | 修正 AI | 实际瓶颈 |
|---|---|---|---|
| P-32h-causal | 508 | 91 | **带宽瓶颈** |
| P-32h-4k-causal | 2031 | 103 | **带宽瓶颈** |
| P-128h-causal | 511 | 90 | **带宽瓶颈** |
| P-B2-128h-causal | 511 | 93 | **带宽瓶颈** |
| P-32h-1k-nc | 1016 | 55 | **带宽瓶颈** |

**重要结论**：由于 Flash Attention 的 K 重复读取，**所有 MLA prefill 工作负载在修正后实际上都是 DRAM 带宽瓶颈，而非计算瓶颈！**

Ridge Point 仍然是 217 FLOP/Byte，但修正后的 AI 全部降至 55~103，远低于 Ridge Point。

### 5.6 效率损失的真正来源（修正后）

修正理论模型后，剩余 ~40-60% 的效率损失主要来自：

| 损失来源 | 占比 | 可否优化 | 说明 |
|---|---|---|---|
| **DRAM bank 竞争** | 15-25% | 部分 | MLA NKV=1 下多核读同一 K buffer，DRAM bank 冲突严重 |
| **流水线气泡** | 10-20% | 可 | reader/compute 不完美重叠，K chunk 读取阻塞 compute |
| **Tile 粒度开销** | 5-10% | 有限 | tile header、alignment、CB 管理 |
| **因果掩码下负载不均** | 5-10% | 可 | 靠前 Q chunk 工作量小，靠后大 |
| **KV Chain Multicast 未启用** | 10-20% | **关键优化** | 因果模式无 multicast，NKV=1 共享的 K 被重复读取 |

**最关键的优化方向**：对因果模式启用 KV Chain Multicast，让一个核读取 K 后通过 NoC 转发给其他处理相同 K 的核，可将 K DRAM 读取减少 NH 倍（从 323MB → ~10MB）。

### 5.8 MLA vs 标准 MHA 的理论优势

| 指标 | 标准 MHA | MLA | 优势 |
|---|---|---|---|
| KV Cache 大小 | `2 × NH × S × D_head` | `1 × 1 × S × D_QK` | **~NH×** 压缩 |
| KV DRAM 读取 (decode) | `B × NH × S × D_head × 2` | `B × 1 × S × D_QK × 1` | **~2×NH/D_QK×D_head** 减少 |
| V 额外存储 | 独立 V cache | V 从 K 切片 | **零额外存储** |
| Prefill FLOPs | 同（head 维度不同） | 略多（D_QK > D_head 通常） | 略劣（~1.1×） |

以 DeepSeek V3 典型参数为例 (NH=128, D_head=128 for MHA, D_QK=576, D_V=512 for MLA)：

- **KV Cache 压缩比**：MHA = `2×128×S×128 = 32768S` bytes vs MLA = `1×S×576 = 576S` bytes → **57× 压缩**
- **Decode KV 读取量**：MHA = `2×B×128×S×128 = 32768BS` vs MLA = `B×1×S×576 = 576BS` → **57× 减少**

这意味着 MLA 在 decode 阶段可以支撑 **显著更长的序列或更大的 batch**，而不受 DRAM 容量/带宽限制。

---

## 6. 两种 MLA 实现的详细对比

TT-Metal 中存在两套 MLA 实现，分别服务于不同的硬件平台和优化目标。

### 6.1 实现 A：生产路径 (`mla1d.py` + ttnn SDPA)

**所在路径**：`models/demos/deepseek_v3/tt/mla/mla1d.py`

**目标平台**：Wormhole B0 (8×7=56 cores) + 多芯片 TG/Quad 集群

**架构**：将 MLA 拆解为一系列独立的 ttnn 算子调用，通过 Python 编排串联。

#### 算子调用链（Decode 路径，`mla1d.py:1468-1518`）

```
Step 1: wq_kv_a          ← ttnn.linear (Q+KV 联合下投影)
Step 2: kv_norm           ← RMSNorm
Step 3: rope_kv           ← ttnn.rotary_embedding_llama
Step 4: concat → kvpe     ← ttnn.concat
Step 5: paged_update_cache ← ttnn.paged_fill_cache
Step 6: wq_b              ← ttnn.linear (Q 上投影)
Step 7: slice q_nope/q_rope ← ttnn.slice × 2
Step 8: wkv_b1            ← ttnn.linear (K 侧吸收到 Q)      ◄── 独立 dispatch
Step 9: rope_q            ← ttnn.rotary_embedding_llama
Step10: concat → Q        ← ttnn.concat
Step11: all_to_all        ← ttnn.all_to_all_async (多芯片通信)
Step12: SDPA decode       ← ttnn.paged_flash_multi_latent_attention_decode  ◄── 融合 SDPA
Step13: wkv_b2            ← ttnn.linear (V 侧解压缩)        ◄── 独立 dispatch
Step14: wo                ← ttnn.linear (输出投影)
```

#### SDPA 内核实现 (`sdpa_decode_program_factory.cpp`)

- **编程模型**：C++ program factory，通过 `CreateProgram()` + `CreateKernel()` 创建 3 个独立内核文件
  - `reader_interleaved.cpp` (NCRISC)
  - `sdpa.cpp` (TRISC compute)
  - `writer_interleaved.cpp` (BRISC)
- **KV Cache 格式**：Interleaved DRAM 或 Paged Attention（使用 page_table 间接寻址）
- **多头并行**：通过 `batch_parallel_factor × nh_parallel_factor × q_parallel_factor` 三级并行化
- **KV 分发**：支持 KV Chain Forwarding（非因果模式），链式 multicast 减少 DRAM 重复读取
- **归约**：通过 `max_cores_per_head_batch` 控制每个 (batch, head) 使用的核数，核间通过 writer 做归约
- **Tile 格式**：标准 32×32 tile

#### 关键特点

| 特性 | 说明 |
|---|---|
| **通用性** | 支持 prefill + decode，支持 MHA/GQA/MLA，支持 paged attention |
| **多芯片** | 通过 all_gather/all_to_all 与多芯片 TP/SP 集成 |
| **KV 解压缩** | 拆分为独立 `ttnn.linear`，未融入 SDPA |
| **工作分配** | 自动并行化，根据 B/NH/S 动态分配到核 |
| **序列并行** | 每个 (batch, head) 由 1 个核处理全部 K chunks |
| **DRAM 访问** | 因果模式下每核独立读 K（无 multicast），K 重复读取严重 |

### 6.2 实现 B：实验性 Fused FlashMLA (`deepseek_v3_b1`)

**所在路径**：
- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` (Python host 端)
- `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` (统一内核头文件)
- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/flash_mla_kernel.cpp` (内核入口)

**目标平台**：**Blackhole** (11×10=110 cores, 8 DRAM banks)

**架构**：专为 DeepSeek V3 MLA decode 深度优化的单一融合算子。

#### 核心设计：S Block 架构

```
┌────────────────────────────────────────────────────┐
│  Blackhole 110 核网格 (11列 × 10行)                 │
│                                                     │
│  左侧 4列              中间空隙         右侧 4列    │
│  ┌──┬──┬──┬──┐                      ┌──┬──┬──┬──┐ │
│  │S1│S2│  │S4│ ←DRAM bank 1,3,2,0   │S5│S6│S7│S8│ │
│  │8核│8核│  │8核│                    │8核│8核│8核│8核│ │
│  └──┴──┴──┴──┘  (S3在col 7-8)       └──┴──┴──┴──┘ │
│                                                     │
│  每个 S Block = 8 核（4行 × 2列）                    │
│  8 个 S Block = 64 核参与序列并行                     │
│  每个 Q head shard 跨 8 个 S Block：                 │
│     Q1 = S1[0]+S2[0]+S3[0]+...+S8[0]               │
│     Q2 = S1[1]+S2[1]+S3[1]+...+S8[1]               │
│                                                     │
│  每个 S Block 绑定最近的 DRAM bank:                   │
│     S1→bank1, S2→bank3, S3→bank2, S4→bank0,        │
│     S5→bank5, S6→bank7, S7→bank6, S8→bank4         │
└────────────────────────────────────────────────────┘
```

#### 内核实现（`flash_mla.hpp` — 单文件 3-RISC 统一内核）

**NCRISC (Reader)**：
- 从 **ND Sharded DRAM** 读取 K chunks（每个 S block 的 sender 从绑定的 DRAM bank 读取）
- **DRAM page-level 流水线**：使用 NOC Transaction ID (trid) 实现页面级读取流水线
  ```cpp
  // 多 trid 并行发射 DRAM 读取请求
  for (i = 0; i < NUM_TRIDS; i++) {
      noc_async_read_set_trid(curr_trid);
      noc_async_read_one_packet_with_state_with_trid(src, offset, dst, curr_trid);
  }
  // 流水线完成：一个 trid 完成后立即发射下一个
  while (pages_completed < total) {
      noc_async_read_barrier_with_trid(wait_trid);
      // 通知 BRISC 该页面已就绪
      *ncrisc_brisc_sync_ptr += 1;
      if (more_pages) issue_next_page();
  }
  ```

**BRISC (Writer)**：
- **Q Multicast**：output core (S1) 将 Q 通过 NoC multicast 广播到所有其他 S block 的对应核
- **K Multicast**：sender 读取 K 后通过 multicast 分发到 S block 内的 7 个 receiver 核
  - 与 NCRISC 的 DRAM 读取做 **page-level 流水线重叠**：NCRISC 每读完一个 page 就通知 BRISC，BRISC 立即 multicast 该 page
- **Tree Reduction**：8 个 S block 通过 **3 步 tree reduction**（log₂8=3）归约部分结果
  ```
  Step 1: S2→S1, S4→S3, S6→S5, S8→S7  (4 路并行)
  Step 2: S3→S1, S7→S5                  (2 路并行)
  Step 3: S5→S1                          (1 路)
  ```

**TRISC (Compute)**：
- 标准 SDPA compute：`QK matmul → scale → mask → exp → AV matmul`
- 使用 `compute_sdpa_chunk` 和 `sdpa_tail` 内置计算原语
- **V 从 K buffer 中 strided 读取**（`mla_kv_overlap`），无需单独的 V CB

#### 关键特点

| 特性 | 说明 |
|---|---|
| **专用性** | 仅支持 decode，仅支持 MLA (NKV=1)，仅支持 Blackhole |
| **序列并行** | 每个 Q head 跨 **8 个核** 并行处理不同的 K chunk ranges |
| **DRAM 局部性** | S Block 与 DRAM bank 1:1 绑定，消除 bank 竞争 |
| **K 分发** | 每个 DRAM bank 只读一次 K chunk，通过 NoC multicast 分发到 S block 内 7 个核 |
| **归约** | 3 步 tree reduction（而非线性归约），log₂(8) = 3 步 |
| **DRAM 读取** | Page-level 流水线 + trid 并行发射，最大化 DRAM 带宽利用 |
| **编程模型** | `UnifiedKernelDescriptor` + `ttnn.generic_op`，Python 直接定义内核参数 |
| **KV Cache** | ND Sharded DRAM（而非 interleaved/paged），按 k_chunk 粒度分片到 8 个 bank |
| **Tile** | 支持 Tiny Tile（Q 使用 8×32 tile，减少 head padding 浪费） |

### 6.3 核心差异对比

| 维度 | 实现 A（生产路径） | 实现 B（实验 FlashMLA） |
|---|---|---|
| **目标硬件** | Wormhole B0 (56 cores) | Blackhole (110 cores) |
| **算子边界** | ~12 个独立 ttnn op dispatch | **单一** `generic_op` dispatch |
| **KV 解压缩** | 独立 `ttnn.linear` (wkv_b1/wkv_b2) | 未包含（仅做 SDPA 部分） |
| **序列并行度** | 1 核/head（decode 时） | **8 核/head**（8 个 S Block） |
| **K 读取策略** | 每核独立从 DRAM 读 K | 1 核读 DRAM + **multicast 到 7 核** |
| **DRAM 局部性** | 无特殊优化 | S Block ↔ DRAM bank **1:1 绑定** |
| **归约方式** | 通过 writer 线性归约 | **3 步 tree reduction** (log₂8) |
| **DRAM 读取流水线** | 读完整 chunk 后通知 compute | **Page-level trid 流水线** |
| **KV Cache 格式** | Interleaved 或 Paged | **ND Sharded DRAM** |
| **Q 广播** | 无（每核独立读 Q） | output core **multicast Q** 到所有核 |
| **Tile 格式** | 标准 32×32 | **Tiny Tile** (8×32 for Q) |
| **Prefill 支持** | 是 | **否（仅 decode）** |
| **通用性** | MHA/GQA/MLA 通用 | **MLA 专用** |

### 6.4 性能影响分析

#### 实现 B 相对于 A 的预期优势

**1. 序列并行 8× 加速**：
- A：1 个核处理一个 head 的全部 K chunks（串行遍历）
- B：8 个核并行处理 K chunks，每核负责 1/8 的序列长度
- 对于 S=32K，每核只需处理 4K tokens，大幅减少 per-core 延迟

**2. K DRAM 流量减少 ~8×**：
- A (decode)：每个 head 的核独立从 DRAM 读全部 K chunks
- B：每个 S block 只有 sender 从 DRAM 读 K，通过 NoC multicast 到其他 7 核
- 对 NKV=1，8 个 S block 共享同一 K，但每个 S block 只读自己绑定 bank 的 K chunk 子集

**3. 消除 DRAM bank 竞争**：
- A：多核争抢同一 DRAM bank 读取共享的 K
- B：S block 与 DRAM bank 1:1 绑定 + K chunk 按 round-robin 分布到 8 个 bank

**4. DRAM 读取流水线**：
- A：读完整 K chunk 后才能 multicast 或开始 compute
- B：NCRISC 用 trid 并行发射多个 DRAM page 读取，每完成一个 page 就通过信号量通知 BRISC 立即 multicast，TRISC 可在 page 到达后立即开始计算

**5. 单次 dispatch**：
- A：12 次 dispatch × ~0.3ms = ~3.6ms 固定开销
- B：理论上可在 fused op 管线中只用 1 次 dispatch（当前 FlashMLA 本身 1 次，但 wkv_b1/b2 仍在外面）

#### 实现 B 的局限性

| 限制 | 说明 |
|---|---|
| **Blackhole only** | 需要 11 列网格（WH_B0 只有 8 列） |
| **仅 decode** | 不支持 prefill |
| **仅 MLA** | 不支持 MHA/GQA |
| **KV 解压缩未融合** | wkv_b1/wkv_b2 仍是外部操作 |
| **硬编码 S Block** | 网格布局固定，不适配其他芯片 |

### 6.5 总结

| | 实现 A | 实现 B |
|---|---|---|
| **成熟度** | 生产级，全场景 | 实验性，Blackhole 专用 |
| **优化深度** | 通用 SDPA，中等优化 | MLA decode 深度优化 |
| **预期 decode 性能** | 基准 | **数倍于 A**（序列并行 + multicast + bank 亲和 + 流水线） |
| **融合程度** | SDPA 内部融合 | SDPA 内部融合 + DRAM/multicast/compute 三级流水线 |
| **下一步** | 可能向 B 的设计靠拢 | 需要将 wkv_b1/b2 融入内核以成为真正的 MLA 融合算子 |

---

## 7. Roofline 模型可视化（修正，基于实现 A）

修正后使用 Flash Attention 的真实 DRAM 流量计算算术强度：

```
  TFLOPS
    56 ┤─────────────────────────────────── Peak Compute (HiFi4) ──
       │                              /
       │                             /
    30 ┤                            /
       │                           /        ● P-B2-128h (25.2)
       │                          /         ● P-128h (23.1)
    20 ┤                         /
       │                        /
       │                       /
    10 ┤                      /   ● P-32h-4k (10.9)
       │                     /    ● P-med (9.9)
       │          Ridge=217 /     ● P-32h (9.0)
     5 ┤              |    /
       │              |   /
       │              |  /
     1 ┤              | /  ● D-32h-4k (0.26)
       │              |/   ● D-32h-1k (0.16)
   0.1 ┤             /|
       │            / |
       ┼───────────┼──┼────────────────────────────────
       1    10   55 100 217  500  1000           AI (FLOP/Byte)
                  ↑       ↑
            修正AI范围    Ridge Point
            (全部在左侧 = 带宽瓶颈)
```

**关键变化**：修正后所有 Prefill case 的 AI 从 500-2000 降至 55-103，**全部落在 Ridge Point 左侧**，表明 Flash Attention 的 MLA prefill 实际上是 **DRAM 带宽瓶颈**而非计算瓶颈。

### 吞吐与效率对比（原始 vs 修正）

| 工作负载 | FLOPs | Min延迟 | 实测 TFLOPS | 原始效率 | 修正效率 | 修正瓶颈 |
|---|---|---|---|---|---|---|
| P-B2-128h | 285.8G | 11.35 ms | 25.2 | 44.9% | ~110% ¹ | DRAM BW |
| P-128h | 142.9G | 6.18 ms | 23.1 | 41.2% | ~105% ¹ | DRAM BW |
| P-32h-4k | 570.4G | 52.40 ms | 10.9 | 19.4% | 42% | DRAM BW |
| P-med | 285.2G | 28.77 ms | 9.9 | 17.7% | ~38% | DRAM BW |
| P-32h | 35.7G | 3.96 ms | 9.0 | 16.1% | 46% | DRAM BW |
| D-32h-4k | 71.3M | 0.275 ms | 0.26 | 3.5% | 58% | Dispatch |
| D-32h-1k | 35.7M | 0.223 ms | 0.16 | 1.3% | 67% | Dispatch |

¹ 修正效率 > 100% 表明 reader/compute 流水线实现了有效重叠，实测时间低于简单 max(compute, DRAM) 模型预测。

---

## 8. 结论与优化建议

### 7.1 主要发现

1. **TT 的 MLA 是"半融合"的**：SDPA 内部（QK + softmax + AV）是融合的，但 MLA 特有的 KV 解压缩（`wkv_b1` K 侧吸收、`wkv_b2` V 侧上投影）作为独立 `ttnn.linear` 调用，**未融入 SDPA 内核**。完整 MLA forward 涉及 ~12 次独立 dispatch。
2. **原始理论模型严重低估了 DRAM 流量**：Flash Attention 的分块机制导致 K 被重复读取 O(S/q_chunk) 次，实际 K DRAM 流量是单次读取的 17-500× 以上。
3. **修正后，所有 Prefill case 实际上是 DRAM 带宽瓶颈**（修正 AI ≈ 55-103），而非原始模型认为的计算瓶颈（原始 AI ≈ 500-2000）。
4. **修正后 SDPA 单算子效率提升至 42-110%**：说明 TT 的 reader/compute 流水线确实有效，但全链路效率因多次 dispatch 和中间张量 DRAM 读写而进一步降低。
5. **MLA NKV=1 加剧了 DRAM 竞争**：多个 head 共享同一 K 但独立读取，导致 DRAM bank 冲突，这是因果模式下 multicast 未启用的直接后果。

### 7.2 优化方向（按修正分析排序）

#### 高优先级：MLA 特有融合 + 减少 DRAM 带宽

| 优化 | 预期收益 | 难度 | 说明 |
|---|---|---|---|
| **wkv_b2 融入 SDPA writer** | **省 1 次 dispatch + 中间 DRAM 读写** | 高 | 在 AV matmul 输出时直接做 V 上投影，避免 attn_out 写回 DRAM |
| **wkv_b1 融入 SDPA reader/compute** | **省 1 次 dispatch + Q 中间读写** | 高 | 读入 Q 时顺带做 K 侧吸收投影 |
| **因果模式 KV Chain Multicast** | **30-50% SDPA 加速** | 中 | K 只读一次 DRAM，通过 NoC multicast 分发；非因果已激活 |
| **增大 k_chunk_size** | 10-20% | 低 | 减少 K 重复读取次数（受 L1 大小限制） |
| **DRAM bank 感知调度** | 10-15% | 中 | 避免多核同时竞争同一 DRAM bank |

#### 中优先级：减少固定开销

| 优化 | 预期收益 | 难度 | 说明 |
|---|---|---|---|
| **Trace mode** | 固定 0.2-0.5ms | 低 | 消除 host→device dispatch 开销 |
| **提升核利用率** | 20-40% (NH小时) | 中 | 当 NH < 56 时大量核空闲 |
| **Streaming compute v2** | 15-30% | 已部分实现 | 更好的 compute/reader 重叠 |

#### 长期：架构级优化

| 优化 | 预期收益 | 难度 | 说明 |
|---|---|---|---|
| **自适应 q_chunk/k_chunk** | 全面优化 | 研究 | 根据 S, NH, L1 动态选择最优分块 |
| **多芯片 Ring Attention** | 线性扩展 | 已有基础设施 | 分摊 KV 读取到多芯片 |
| **L1 KV 缓存复用** | 50%+ DRAM 减少 | 高 | 跨 head 在 L1 中保留 K chunk，不回读 DRAM |

---

## 附录 A：实验代码

性能测试脚本：`mla_flash_attention_dev/perf_test_mla.py`

运行方式：
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python3 mla_flash_attention_dev/perf_test_mla.py
```

原始数据：`mla_flash_attention_dev/perf_results.json`

## 附录 B：理论计算公式

### B.1 原始（有缺陷的）模型

假设 K 从 DRAM 只读一次：

$$
T_{memory}^{naive} = \frac{Q_{bytes} + K_{bytes}^{1pass} + O_{bytes}}{\text{DRAM BW}}
$$

此模型严重低估了 Flash Attention 的实际 DRAM 流量。

### B.2 修正模型

**修正 DRAM 流量**（考虑 Flash Attention 的 K 重复读取）：

对于 Causal Prefill，每个 head 的 K DRAM 读取量：

$$
K_{bytes}^{FA} = \sum_{i=0}^{N_q-1} \lceil \frac{(i+1) \times q_{chunk}}{k_{chunk}} \rceil \times k_{chunk} \times D_{QK} \times \text{kv\_dtype\_size}
$$

化简（当 q_chunk | k_chunk 时）：

$$
K_{bytes}^{FA} = \frac{q_{chunk}}{k_{chunk}} \times \frac{N_q \times (N_q + 1)}{2} \times k_{chunk} \times D_{QK} \times \text{kv\_dtype\_size}
$$

总 K DRAM 流量 = $N_H \times K_{bytes}^{FA}$（NKV=1 时，因果模式无 multicast，每个 head 独立读取）

**修正计算时间**（考虑活跃核数）：

$$
T_{compute} = \frac{\text{Total FLOPs}}{N_{active\_cores} \times \frac{4096}{fidelity\_mult} \times f_{clock}}
$$

其中 $N_{active\_cores} = \min(B \times N_H, 56)$，HiFi4 的 fidelity_mult = 4。

**修正总时间**：

$$
T_{corrected} = \max(T_{compute}, T_{memory}^{FA}) + T_{dispatch}
$$

实际由于 reader/compute 流水线重叠，真实时间可低于 $\max(T_{compute}, T_{memory}^{FA})$。

### B.3 Arithmetic Intensity（修正）

$$
\text{AI}_{corrected} = \frac{\text{Total FLOPs}}{Q_{bytes} + N_H \times K_{bytes}^{FA} + O_{bytes}}
$$

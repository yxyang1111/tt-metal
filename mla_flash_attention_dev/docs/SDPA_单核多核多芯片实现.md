# TensTorrent SDPA：单核、多核与多芯片实现说明

本文档说明 TensTorrent SDPA 在单核、多核（单芯片）和多芯片三个层级上的实现方式。

---

## 一、单核 SDPA — 一个 Tensix 核心内部发生了什么

每个 Tensix 核心有 5 个 RISC-V，分别运行 Reader / Writer / Compute 三个内核，它们通过 L1 中的 **Circular Buffer** 异步协作。

### 1.1 单核计算流程

假设某个核心被分配了若干 Q chunk，以及完整的 K/V，它执行以下伪代码：

```
for each (batch, head) assigned to this core:
    for each Q_chunk assigned to this core:
        Reader: 从 DRAM 读 Q_chunk → L1 cb_q
        初始化 prev_max = -inf, prev_sum = 0, OUT_ACC = 0
        
        for k_chunk = 0 to k_chunk_end:    // causal 时截断到 q 位置
            Reader: 预取 K_chunk, V_chunk → L1 cb_k, cb_v (双缓冲)
            Reader: 若需要 mask，读 mask → L1 cb_mask
            
            Compute:
              1. QK = Q_chunk @ K_chunk^T            // tile matmul
              2. QK += mask (仅对角块)
              3. cur_max = max(prev_max, rowmax(QK))  // 在线 softmax
              4. P = exp((QK - cur_max) * scale)
              5. cur_sum = rowsum(P)
              6. OUT_IM = P @ V_chunk                 // tile matmul
              7. if 非首个 chunk:
                   correction = exp((prev_max - cur_max) * scale)
                   prev_sum *= correction
                   OUT_ACC *= correction              // 修正历史累积
                   cur_sum += prev_sum
              8. OUT_ACC += OUT_IM
              9. swap(prev ↔ cur)                    // ping-pong 缓冲区
            
        Compute: OUT = OUT_ACC / cur_sum              // 最终归一化
        Writer: 从 L1 cb_out 写回 DRAM
```

### 1.2 流水线重叠

关键在于 Reader 和 Compute **并行执行**：

```
时间 →
Reader(RISC0):  [读Q]  [读K0,V0]  [读K1,V1]  [读K2,V2] ...
Compute(RISC2): ........[Q@K0→P@V0] [Q@K1→P@V1] [Q@K2→P@V2] ...
Writer(RISC1):  .................................[写OUT0] ...
```

K 和 V 的 Circular Buffer 大小为 **2 倍 chunk**（双缓冲），当 Compute 处理 chunk_i 时，Reader 已经在预取 chunk_i+1，从而隐藏 DRAM 延迟。

### 1.3 Circular Buffer 布局

单核 L1 内的关键缓冲区：

| 缓冲区 | 用途 | 备注 |
|--------|------|------|
| cb_q (cb_0) | Q chunk | 单/双缓冲 |
| cb_k (cb_1) | K chunk | 双缓冲 (×2) |
| cb_v (cb_2) | V chunk | 双缓冲 (×2) |
| cb_mask (cb_3) | Mask | 双缓冲 (×2) |
| cb_qk_im (cb_24) | QK 中间结果 | — |
| cb_out_25/26 | 输出累积 | ping-pong |
| cb_max_27/28 | 行最大值 | ping-pong |
| cb_sum_29/30 | 行求和 | ping-pong |
| cb_exp_diff (cb_31) | exp 修正因子 | — |
| cb_out (cb_16) | 最终输出 | — |

所有中间状态（max、sum、累积输出）**全程留在 L1**，只有最终结果写回 DRAM。

---

## 二、多核 SDPA — 如何在一块芯片上并行

### 2.1 三维并行划分

`sdpa_program_factory.cpp` 按优先级依次在 **batch → heads → Q chunks** 三个维度上划分：

```cpp
batch_parallel_factor = min(B, num_cores)
nh_parallel_factor    = min(num_cores / batch_pf, NQH)
q_parallel_factor     = min(num_cores / (batch_pf * nh_pf), q_num_chunks)
```

每个核心分到：

- `batch_per_core` 个 batch
- `nh_per_core` 个 head
- `q_per_core` 个 Q chunk

核心 `i` 的工作范围通过三维索引计算：

```cpp
local_batch_start = (i / (nh_pf * q_pf)) * batch_per_core
local_nh_start    = ((i / q_pf) % nh_pf) * nh_per_core
local_q_start     = (i % q_pf) * q_per_core
```

### 2.2 Causal 模式：各核完全独立

在 causal attention 中，**各核之间不需要任何通信**。每个核心独立地：

1. 读自己负责的 Q chunk
2. 遍历所有 causal 约束内的 K/V chunk（`k < q_position`）
3. 将输出写回 DRAM

这是最简单也最高效的并行模式。

### 2.3 Causal 负载均衡 — BALANCED_Q_PARALLEL

问题：Q0 只需处理 1 个 K chunk，但 Qn-1 需要处理 n 个 K chunk，直接顺序分配会严重不均衡。

解决方案 — **low + high 配对**：

```
Core 0: Q0 (最少工作) + Q(n-1) (最多工作)
Core 1: Q1           + Q(n-2)
Core 2: Q2           + Q(n-3)
...
```

代码实现：

```cpp
if (q_iter < q_chunks_per_core / 2) {
    q_chunk = local_q_start + q_iter;           // 低 chunk
} else {
    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);  // 高 chunk
}
```

这样每个核心的总工作量几乎相同，带来约 **1.6× 加速**。

### 2.4 非 Causal 模式：KV Chain 转发

当同一个 head 的多个 Q chunk 分布在不同核上时，这些核需要读**完全相同的 K/V**。为避免每个核都从 DRAM 独立读取，采用 **KV chain 转发**：

```
DRAM → [Core0: Injector] --NoC-→ [Core1: Receiver] --NoC-→ [Core2: Sink]
            读K/V并转发           接收K/V并转发             接收K/V，不再转发
```

角色定义：

- **Injector**：从 DRAM 读 K/V，同时 NoC 写给下一个核
- **Receiver**：通过 semaphore 等待上一核的 K/V 数据
- **Sink**：链尾，只接收不转发

如果链上的核恰好在同一物理行且满足条件，还会升级为 **Multicast**，一次 NoC 操作写给多个目标核。

### 2.5 FlashDecode 的多核 — 序列维度分片 + 树形归约

在解码阶段（query 长度为 1），多核之间不是各自处理不同的 Q，而是**分担同一个 KV cache 的不同片段**，最后做归约。

**第一步：KV cache 分片**

```
KV cache 总长度 = valid_seq_len
每核处理: [k_chunk_start, k_chunk_end)
```

**第二步：各核独立计算局部注意力**

每个核对自己负责的 KV 片段执行标准 FlashAttention，得到局部结果 (O_local, M_local, L_local)：

- O_local：局部加权输出
- M_local：局部行最大值
- L_local：局部行求和

**第三步：树形归约**

不是简单的全规约，而是构建一棵**二叉树**（支持最多 64 核，6 轮归约）：

```
         Core0 (root)
        /            \
    Core1            Core2
   /     \          /     \
 Core3  Core4    Core5   Core6
```

每轮归约中：

1. 叶子核通过 **NoC 写** 把 (O, M, L) 发到父核的 L1
2. 用**信号量**通知父核数据就绪（单个 32 位信号量，每轮占 4 位）
3. 父核执行 **correction_block**，合并子核与本地的结果：

```
new_max = max(local_max, child_max)
local_O = local_O × exp(local_max - new_max) + child_O × exp(child_max - new_max)
local_L = local_L × exp(local_max - new_max) + child_L × exp(child_max - new_max)
```

4. 根核做完所有轮次后，执行 `OUT = O / L` 最终归一化

整个归约在 **O(log N) 轮** 内完成。

---

## 三、多芯片 SDPA — 跨设备的分布式注意力

TensTorrent 提供了三种多芯片 SDPA 实现，适用于不同场景。

### 3.1 Ring Distributed SDPA（推理，无运行时通信）

**思路**：每个设备处理全局序列中的特定 Q chunk，K/V 在本地完整可用。

```
设备 0: 处理 Q_chunk_0 和 Q_chunk_(2N-1)
设备 1: 处理 Q_chunk_1 和 Q_chunk_(2N-2)
...
设备 N-1: 处理 Q_chunk_(N-1) 和 Q_chunk_N
```

```cpp
const uint32_t chunk_1 = ring_id;
const uint32_t chunk_2 = (2 * ring_size) - ring_id - 1;
```

特点：

- **零运行时跨设备通信**：每个设备独立计算
- 输出按设备分片，由 host 端 gather 重排
- 本质上是把多核的 BALANCED_Q_PARALLEL 策略扩展到多设备

### 3.2 Ring Joint SDPA（推理，带 All-Gather）

**思路**：Q/K/V 按序列维度在 ring 上分片，通过 All-Gather 收集远端 K/V。

```
步骤 1: 每设备处理 local K/V
步骤 2: Ring All-Gather 获取邻居的 K/V
步骤 3: 对 gathered K/V 做 attention
步骤 4: 处理 joint K/V (如 CLS token)
步骤 5: LSE 在线更新合并各步结果
```

通信通过 `ring_attention_all_gather_async` 实现，底层使用 **TT Fabric + Ethernet**（Wormhole 有 16 个以太网核心，3200 Gbps 双向带宽）。

用 `RingSDPAFusedOpSignaler` 做信号量协调，使 All-Gather 与 attention 计算重叠。

### 3.3 Ring Attention SDPA（训练，前向+反向）

**思路**：经典的 Ring Attention 算法，每步在 ring 上轮换 K/V。

```python
for step in range(ring_size):
    # 本地 attention
    out_local, inter = ring_sdpa_fw(Q, K_current, V_current)
    
    # 在线 softmax 合并
    global_max = max(global_max, local_max)
    global_out = rescale(global_out) + rescale(out_local)
    global_sum = rescale(global_sum) + local_sum
    
    # Ring shift: K/V 传给下一个设备
    if step < ring_size - 1:
        K_current = ring_shift(K_current)
        V_current = ring_shift(V_current)
```

`ring_shift` 通过 **Socket 连接** 实现，采用两阶段避免死锁：

1. 偶数设备发、奇数设备收
2. 奇数设备发、偶数设备收

反向传播时按相反顺序重算，梯度同样通过 `ring_shift` 回传。

---

## 四、三级架构对比总结

| 维度 | 单核 | 多核（单芯片） | 多芯片 |
|------|------|----------------|--------|
| **并行粒度** | K/V chunk 串行 | batch × heads × Q chunks | 设备间分序列长度 |
| **核间通信** | 无 | Causal: 无; 非Causal: KV chain/multicast; Decode: 树形归约 | All-Gather / Ring Shift / 无 |
| **通信介质** | — | L1 NoC (核内) | Ethernet (3200 Gbps) + Fabric |
| **中间状态** | L1 ping-pong | 各核 L1 独立 | 各设备 L1 独立，LSE 在线合并 |
| **softmax** | 在线 softmax (单核内) | Decode: 树形归约 + correction_block | Ring: 每步在线合并 |
| **负载均衡** | — | BALANCED_Q_PARALLEL (low+high) | Ring Distributed (chunk配对) |
| **典型加速** | 基准 | ~20× vs naive | 随设备数近线性 |

核心设计哲学是：**每一级都尽量让中间结果留在最近的存储层（L1 > DRAM > 跨设备），通过在线 softmax 避免全局通信**。

---

## 五、参考

- `tt-metal/tech_reports/FlashAttention/FlashAttention.md` — FlashAttention 技术报告
- `tt-metal/tech_reports/FlashAttention/FlashDecode.md` — FlashDecode 技术报告
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` — 多核划分与单核参数
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/` — FlashDecode 树形归约
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_*` — 多芯片 Ring SDPA

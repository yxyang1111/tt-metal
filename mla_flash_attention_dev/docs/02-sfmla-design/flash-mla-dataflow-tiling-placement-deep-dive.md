# TT 实验性 FlashMLA：数据流、Tiling 与 Placement

> 本文基于 `models/demos/deepseek_v3_b1/` 下的实验性 FlashMLA decode 实现。
> 核心源码：`unified_kernels/flash_mla.hpp`、`micro_ops/flash_mla/op.py`、`kernels/rt_args_common.hpp`

---

## 第一部分：数据流是怎么设计的，为什么这样设计

### 1.1 问题背景

DeepSeek V3 使用的 MLA（Multi-Latent Attention）有一个特别之处：**K 和 V 共享同一张 tensor**。V 不是一个独立的矩阵，而是 KV cache 的前 `head_dim_v` 列。换句话说，你只要把 K 从 DRAM 搬进 L1，V 就已经在那里了——不需要额外的搬运。

在 decode 场景下，Q 也很特殊：每个 token 只有一个 Q 向量（或很少几个 head 的 Q 向量），数据量极小。真正的数据量大头是 KV cache——序列长度可以到 32K 甚至更长，每个 chunk 128 个 token × 576 维 = 一块不小的矩阵。

所以 **整个数据流的设计出发点**是：

> K 的搬运是唯一的大流量瓶颈。V 搭 K 的便车。Q 小到可以广播。计算能力充足但会被数据搬运卡住。

### 1.2 整体架构：三个处理器各司其职

Tenstorrent 的 Tensix 核心里有多个 RISC 处理器。实验性 FlashMLA 用了其中三个，各做一件事：

- **NCRISC**——专门从 DRAM 往 L1 搬 K 数据。它只做这一件事，但做得很精细：用多个 transaction ID 做流水线读取，一页读完就通知 BRISC 可以往外转发了，不用等整块 chunk 全部读完。

- **BRISC**——充当片上网络控制器。它做三件事：（1）把 Q 从 output core 扩散到所有 worker core；（2）把 NCRISC 读到的 K 数据通过 multicast 发给同组的其它 core；（3）在所有 chunk 算完后，执行 tree reduction 协议，把各组的部分结果合并成最终输出。

- **TRISC**——做纯计算。拿到 Q 和 K 之后，执行标准的 flash attention 分块算法（QK^T → online softmax → P·V），处理完所有 chunk 后，如果是 reduction receiver 就继续做 tail reduction 合并来自其它组的结果。

**为什么要这样分工？** 因为 NCRISC 的 DRAM 读和 BRISC 的 NOC multicast 可以在硬件层面真正并行。NCRISC 每读完一个 page（比如 8KB），BRISC 就可以立刻把这个 page multicast 出去，同时 NCRISC 继续读下一个 page。这形成了一条 **page 级别的流水线**，而不是"读完整块 → 发完整块 → 再读下一块"的串行模式。

### 1.3 S block：空间布局的基本单元

实验性 FlashMLA 把芯片上的核心组织成若干个 **S block**（Sequence block）。以 Wormhole B0 为例，有 6 个 S block，每个 block 4 个核心。以 Blackhole 为例，有 8 个 S block，每个 block 8 个核心。

每个 S block 是一组在物理上相邻的核心。它们绑定在一起做三件事：

1. **共享同一个 DRAM bank 的数据**——block 内第一个核心（sender）从绑定的 DRAM bank 读 K，然后 multicast 到 block 内其它核心。这样 block 内只有一次 DRAM 读，其余核心靠片上通信拿到 K。

2. **并行处理不同的 K chunk**——同一个 batch 的序列被切成多个 chunk，分给不同 S block 的同一位置的核心。比如 batch 0 的工作会分配给 S1[0]、S2[0]、S3[0]、...、S6[0]，这 6 个核心各自负责不同的 chunk 段，实现序列长度方向的并行。

3. **参与 tree reduction**——每个 S block 算完自己负责的 chunk 后，产出一组部分结果（部分 output + 对应的 online softmax 统计量 m/s）。这些部分结果通过 tree reduction 的方式汇聚到 S1 上的 output core，形成最终结果。

**为什么不让每个核心自己去 DRAM 读？** 因为 DRAM 带宽是稀缺资源。如果 24 个核心（WH）或 64 个核心（BH）同时去读同一份 KV cache，DRAM 会成为严重瓶颈。用 S block 的 multicast 模式，DRAM 读请求只有 6 次（WH）或 8 次（BH），剩下的全在片上完成。

### 1.4 Chunk 分配策略：strided 分配与 DRAM bank 对齐

KV cache 在 DRAM 上按 ND shard 存储，使用 ROUND_ROBIN_1D 策略：chunk 0 存在 bank 0，chunk 1 存在 bank 1，……，chunk 5 回到 bank 0（WH 6 个 bank 的情况）。

每个 S block 绑定了一个 DRAM bank。chunk 分配使用 strided 方式：S block i 的 sender 核心负责 chunk i、chunk i+N、chunk i+2N、……（N 是 S block 总数）。

这样 **每个 sender 读到的所有 chunk 都落在同一个 DRAM bank 上**。没有跨 bank 竞争，没有冗余读。这不是巧合，是刻意对齐的结果。

### 1.5 V 的处理：搭 K 的便车

这是实验性 FlashMLA 和主线 SDPA 最大的区别之一。

主线 SDPA decode 即使有 `reuse_k` 选项，也通常保留一个独立的 `cb_v_in` buffer。实验性 FlashMLA **完全没有 V 的专用 buffer**。TRISC 在做 P·V 矩阵乘时，直接从 `cb_k_in`（K buffer）里用 strided matmul 读取前 `vDHt` 列作为 V。

这省掉了：一整套 V 的 DRAM 读取、一整套 V 的 multicast、一个 CB 和相关的同步逻辑。

### 1.6 Q 的处理：一次广播到全组

Q tensor 预先 height-sharded 到 S1 的各个 output core 上。每个 output core 拿到的 Q 就是自己那个 batch 需要的 Q head。

计算开始时，output core 把自己 L1 里的 Q 广播出去——具体做法是：output core 等所有 worker 报到（semaphore），然后发一个全芯片 multicast 的"Q 已就绪"信号。各 worker core 收到信号后，各自从 output core 的 L1 把 Q 读到自己的 `cb_q_in`。

Q 总共只有几个 tiny tile（8×32 = 512 字节级别），广播一次的代价远小于每个核心自己去 DRAM 读。

### 1.7 Tree Reduction：从多组部分结果到一个最终结果

每个 S block 的核心在处理完自己的 K chunk 之后，TRISC 会产出一组部分结果：部分 output O（`vDHt` 个 tiny tile）和对应的 online softmax 统计量 m/s（1 个 tiny tile）。

6 个 S block（WH）的部分结果需要合并。如果串行做需要 5 步，但用 tree reduction 可以 3 步完成（log₂(6)≈3）：

```
Step 1（3 对并行）：S2 → S1, S4 → S3, S6 → S5
Step 2（1 对）    ：S3 → S1
Step 3（1 对）    ：S5 → S1
```

每一步中，sender 把自己的 O 和 m/s 写到 partner 的指定 buffer slot，然后用 semaphore 通知对方。receiver 在 TRISC 里调用 `sdpa_tail` 做合并——这本质上就是 online softmax 的跨块合并：先用两边的 max 做 rescale，再合并 sum 和 weighted output。

**为什么每个 step 用独立的 buffer slot？** 因为不同 step 的 sender 可能乱序到达。比如 S5 可能在 step 3 先发完（因为 S5 的 chunk 少），但 S3 在 step 2 还没发完。如果共用同一段 buffer，后到的数据会覆盖先到的。所以 `cb_ms_in` 和 `cb_out_in` 的大小是 `per_step_tiles × NUM_TREE_REDUCTION_STEPS`，每步写不同的 slot。

### 1.8 双缓冲与 NCRISC-BRISC 同步

K buffer（`cb_k_in`）是双缓冲的，大小为 `Sk_chunk_t × DHt × 2`。当前 chunk 在 TRISC 计算的同时，NCRISC 可以把下一个 chunk 读进另一半 buffer。

NCRISC 和 BRISC 之间通过一段共享的 semaphore 区域同步。这段区域 16 字节，包含 4 个量：

- `ncrisc_brisc_sync_curr`：当前 buffer 的 page 完成计数
- `ncrisc_brisc_sync_next`：下一 buffer 的 page 完成计数
- `k_write_curr_ptr_shared`：当前 buffer 的 L1 起始地址
- `k_write_next_ptr_shared`：下一 buffer 的 L1 起始地址

每完成一个 chunk，两对指针 swap，实现乒乓切换。

---

## 第二部分：完整数据流伪代码

### 符号表

**张量与维度**

| 符号 | 含义 |
|------|------|
| Q | 查询向量，decode 下为单 token，shape [B, H_q, d] |
| K | KV cache 的全部列，shape [S, d]，按 chunk 分片存储在 DRAM |
| V | KV cache 的前 d_v 列，V = K[:, :d_v]，V ⊂ K，不单独存储 |
| O | attention 输出，shape [B, H_q, d_v] |
| m | online softmax 运行统计量：行最大值（标量/向量） |
| l | online softmax 运行统计量：指数和（标量/向量） |
| S | 有效序列长度（= pos + 1，向上对齐到 c_k） |
| d | KV 的 head 维度（如 576） |
| d_v | V 的 head 维度（d_v ≤ d，如 512） |
| B | batch 数（= Q head 总数 / 每核 head 数） |
| H_q | Q head 总数 |

**分块与分页**

| 符号 | 含义 |
|------|------|
| c_k | chunk 大小，每 chunk 包含的 token 数（如 128） |
| C | 总 chunk 数 = ⌈S / c_k⌉ |
| K_j | 第 j 个 chunk 的 K 数据，shape [c_k, d] |
| V_j | 第 j 个 chunk 的 V 数据 = K_j[:, :d_v]，shape [c_k, d_v] |
| p_k | 单个 page 的字节大小（≤ NOC 单包上限，WH ≤ 8KB，BH ≤ 16KB） |
| P | 每个 chunk 的 page 数 = ⌈sizeof(K_j) / p_k⌉ |

**芯片拓扑与放置**

| 符号 | 含义 |
|------|------|
| N | S block 总数（WH=6, BH=8），也是序列维并行度 |
| M | 每个 S block 内的核心数（WH=4, BH=8） |
| S_i | 第 i 个 S block（i = 1..N） |
| S_i[b] | S block i 中服务 batch b 的那个核心 |
| b_i | S block i 绑定的 DRAM bank |
| G | 全芯片 worker grid 的物理矩形 (x_0, y_0) → (x_1, y_1) |
| G_i | S block i 的物理 multicast 矩形 |

**核心角色与标记**

| 符号 | 含义 |
|------|------|
| r | 核心编号（core_id），在同 batch 的 N 个核心中 r ∈ [0, N) |
| c_start | 该核心负责的第一个 chunk 编号，= r |
| c_end | 该核心负责的最后一个 chunk 的边界 |
| is_sender | 是否为 S block 内的 K multicast sender（每 block 1 个） |
| is_output | 是否为 output core（S_1 中的核心） |

**Tree Reduction**

| 符号 | 含义 |
|------|------|
| T | tree reduction 步数 = ⌈log₂(N)⌉ |
| R[t] | 第 t 步的角色表：每个核心为 sender / receiver / idle |
| (O_i, m_i, l_i) | S block i 产出的部分结果 |
| slot_t | reduction 中间 buffer 的第 t 个 slot（防乱序覆盖） |
| α | 合并时的旧值修正系数 = exp(m_old - m_new) |

**硬件资源**

| 符号 | 含义 |
|------|------|
| L1 | 核心本地 SRAM（~1.5MB），存放 circular buffer |
| CB_K | K 的 circular buffer，双缓冲，容量 2 × c_k × d |
| CB_Q | Q 的 circular buffer |
| DEST | TRISC 的目标寄存器文件，flash attention 中间态常驻于此 |
| NOC0, NOC1 | 两条独立的片上网络，可同时承载不同方向的流量 |
| W | 多 TRID 滑动窗口大小（≤ 14），控制 DRAM 读流水线深度 |
| σ | semaphore，片上同步原语，核心间通过原子加减通信 |

### 整体流程

```
Input:  Q[B, H_q, d],  K[S, d] (在 DRAM, ND-sharded),  pos
Output: O[B, H_q, d_v]

1. 划分 N 个 S block {S_1, ..., S_N}，每个含 M 个核心构成矩形 G_i，
   绑定 DRAM bank b_i。

2. 将序列切成 C = ⌈S/c_k⌉ 个 chunk，stride=N 分配：
   S_i 的核心负责 chunk {i, i+N, i+2N, ...}，全部落在 b_i 上。

3. 每个 S_i 内，sender 从 b_i 读 K_j → L1，multicast 到 G_i 内 M-1 个 receiver。
   每 chunk 仅 N 次 DRAM 读（不是 N×M 次）。

4. Q 从 S_1 的 output core 广播到全部 B×N 个 worker（数据极小）。

5. 各核心独立做 flash attention 分块计算：
   (O_i, m_i, l_i) = FlashAttn(Q, {K_j})，V_j = K_j[:, :d_v]。

6. {(O_i, m_i, l_i)}_{i=1..N} 通过 T 步 tree reduction 汇聚到 S_1 的 output core，
   得到 O_final = O / l。
```

### 阵列级：Host 如何组织全局数据流

```
Input:  芯片型号, Q.shape, K.shape, c_k
Output: 每个核心的 (is_sender, is_output, r, c_start, c_end, R[0..T-1])

1. 确定拓扑
   选 N, M；为每个 S_i 选定核心矩形 G_i，绑定 bank b_i。
   G_i 中核心在 NOC mesh 上邻近 b_i 且构成物理矩形。

2. 映射 batch
   对每个 b ∈ [0, B)：
     分配核心 S_1[b], S_2[b], ..., S_N[b]。
     S_1[b].is_output = true（持有 Q_b，写最终 O_b）。
     各 S_i 的第一批核心 is_sender = true。
     核心编号 r = i - 1（在 N 个核心中的序号）。

3. Chunk 分配
   C = ⌈(pos+1) / c_k⌉
   对核心编号 r：c_start = r,  stride = N
     c_end = r + (⌈C/N⌉ − 1)·N + 1  （含余数修正）
   KV shard round-robin：chunk j 在 bank b_{j mod N}
   → 核心 r 的所有 chunk 全在 b_r 上。

4. Reduction 拓扑
   T = ⌈log₂(N)⌉
   构造角色表 R[t] = {(src_i, dst_i)}, t = 0..T-1
   写入每个核心的 runtime args：每步查 R[t] 知自己是 sender/receiver/idle
     及 partner 的物理坐标 (x_partner, y_partner)。
```

### S block 级：一组核心如何协作搬运和计算

```
Input:  S_i 内 M 个核心（1 sender + M-1 receiver），
        bank b_i 上的 K chunks {K_{r}, K_{r+N}, K_{r+2N}, ...}
Output: (O_i, m_i, l_i)

1. DRAM → L1 读取（sender 执行，NOC0）
   对每个 K_j (j = r, r+N, ...):
     K_j 被切成 P 个 page，每 page p_k 字节。
     NCRISC 用 W-深度 TRID 滑动窗口异步读：
       发出读 page_0 (trid_1), page_1 (trid_2), ...
       完成 page_p 后：共享计数器 σ_sync += 1（通知 BRISC）

2. L1 → L1 multicast（sender BRISC 执行，NOC0）
   for p = 0..P-1:
     等 σ_sync ≥ p+1   // NCRISC 已读完 page_p
     multicast(CB_K + p·p_k, G_i, p_k, M-1 个 dest)
   发 σ_mcast = VALID 到 G_i 内所有 receiver

3. 计算（所有 M 个核心的 TRISC 并行执行）
   V_j = K_j[:, :d_v]    // 无额外搬运
   O, m, l ← flash_attn_chunk_update(O, m, l, Q, K_j, V_j)

4. 双缓冲
   CB_K 容量 = 2 × sizeof(K_j)。
   chunk j 在 CB_K[0] 计算时，chunk j+N 的读取写入 CB_K[1]。
   每完成一个 chunk，swap(CB_K[0], CB_K[1])。

5. 输出 (O_i, m_i, l_i)
```

### 核心级：三个处理器如何并行工作

```
Input:  三个处理器 (NCRISC, BRISC, TRISC) 共享 L1、CB_K、CB_Q、σ
Output: (O, m, l) 或 O_final

─── NCRISC ───

  Input:  addr_K (DRAM 基地址), r, N, C, P, p_k, W
  Output: K_j → CB_K (双缓冲)

  1. c_start = r, 计算 c_end
  2. for j = c_start; j < c_end; j += N:
       reserve(CB_K)                // 等 TRISC 消费完旧数据
       if is_sender:
         ptr ← write_ptr(CB_K)
         publish(ptr → σ_ptr)       // 告诉 BRISC buffer 地址
         trid ← 1
         for p = 0..P-1:            // 滑动窗口读
           async_read(addr_K + shard(j) + p·p_k, ptr + p·p_k, trid)
           trid = trid mod W + 1
           if p ≥ W: wait(trid_oldest); σ_sync += 1
         drain remaining trids; σ_sync += 1 each
         swap(σ_sync_curr, σ_sync_next)
       if receiver:
         inc(σ_recv_ready → sender)  // 告诉 sender 已 reserve
         wait(σ_mcast = VALID)       // 等 multicast 到达
         reset(σ_mcast)
       push(CB_K)                    // 通知 TRISC: K_j 就绪

─── BRISC ───

  Input:  CB_Q(output core L1), CB_K(sender L1), G_i, G, R[0..T-1]
  Output: Q → 全部核心, K → G_i 内核心, reduction 传输

  1. Q 广播
     if is_output:
       wait_front(CB_Q)
       if is_sender:
         wait(σ_q_ready = B×N−1)          // 所有 worker 报到
         mcast_signal(σ_q_ready, G)         // 全芯片 multicast 通知
       else:
         inc(σ_q_ready → sender_of_S_1)
         wait(σ_q_ready = 1)
     else:
       wait(σ_q_ready = 1)
       reserve(CB_Q)
       read(output_core.L1 → CB_Q, sizeof(Q_b))  // 从 output core 拉 Q
       push(CB_Q)

  2. Mask
     if (pos+1) mod c_k ≠ 0:
       构造 mask[c_k]: 前 (pos mod c_k + 1) 个 0, 其余 -∞

  3. K multicast (仅 is_sender):
     for j = c_start; j < c_end; j += N:
       wait(σ_recv_ready = M−1)     // 等 receiver 就绪
       reset(σ_recv_ready)
       for p = 0..P-1:
         wait(σ_sync ≥ p+1)         // 等 NCRISC 完成 page_p
         mcast(CB_K + p·p_k → G_i, p_k, M−1)
       mcast_signal(σ_mcast = VALID, G_i)
       swap buffers

  4. Tree reduction
     for t = 0..T-1:
       role, partner ← R[t]
       if role = sender:
         wait_front(O, m, l)         // 等 TRISC 产出
         write(m, l → partner.slot_t)
         write(O   → partner.slot_t)
         inc(σ_reduce → partner, bit_t)
         break                        // sender 发完即退
       if role = receiver:
         wait(σ_reduce.bit_t = 1)     // 等 partner 数据到达
         push(O', m', l' → TRISC)     // 交给 TRISC 合并

─── TRISC ───

  Input:  CB_Q, CB_K, mask, R[0..T-1]
  Output: (O, m, l) → CB_out, 或 O_final → output shard

  1. wait_front(CB_Q)            // 等 Q
  2. O ← 0,  m ← −∞,  l ← 0    // 初始化 DEST 寄存器中的状态
  3. for j = c_start; j < c_end; j += N:
       wait_front(CB_K)          // 等 K_j
       // ─── flash attention chunk update (全部在 DEST 中) ───
       A_j ← Q · K_j^T                         // mm1: QK^T
       if j = c_end−N and mask: A_j += mask     // 尾块遮蔽
       m' ← max(m, rowmax(A_j))                 // 新行最大
       α ← exp(m − m')                          // 修正系数
       O ← α·O + exp(A_j − m') · V_j            // V_j = K_j[:,:d_v], mm2
       l ← α·l + rowsum(exp(A_j − m'))          // 更新指数和
       m ← m'
       pop(CB_K)                 // 释放 K buffer → NCRISC 可写下一块

  4. pack (O, m, l) → CB_out

  5. Tail reduction (若 receiver in R):
     // 计算本核心需要接收合并的 partner 数
     n_recv ← |{t : R[t].role = receiver and partner active}|
     for k = 1..n_recv:
       wait_front(O'_k, m'_k, l'_k)         // 来自 BRISC push
       m'' ← max(m, m'_k)
       O ← exp(m−m'')·O + exp(m'_k−m'')·O'_k
       l ← exp(m−m'')·l + exp(m'_k−m'')·l'_k
       m ← m''
     O_final ← O / l                         // 最终归一化
     pack O_final → output shard
```

---

## 第三部分：Placement、单播/多播与参数-硬件匹配

### 3.1 核心布局：如何把核心放到正确的位置

#### 放置的核心思路

实验性 FlashMLA 的 placement 是 **"先选拓扑，再映射数据"**，而不是通常的"先看数据形状，再自动分配核心"。

具体来说，host 侧直接硬编码了每个 S block 的逻辑坐标，以及每个 S block 对应哪个 DRAM bank。硬编码的依据是：

1. **DRAM bank 亲和性**：通过 `device.get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)` 查询每个 DRAM bank 附近最近的 worker 核心，然后围绕这些核心构造 S block。

2. **Multicast 矩形约束**：NOC multicast 要求目标核心形成一个在物理坐标系下的矩形。每个 S block 的 4 个核心（WH）或 8 个核心（BH）被选在一个 2×2 或 4×2 的矩形里，这样一次 multicast 就能覆盖整组。

3. **不重叠**：不同 S block 的核心集合不能重叠。不同 batch 使用不同 S block 内的核心位置（block[batch_idx]），也不重叠。

#### WH B0 的具体 placement

6 个 S block 分两组放在芯片的左右两侧：

| S block | 逻辑核心坐标 | 绑定 DRAM bank | 物理位置 |
|---------|-------------|---------------|---------|
| S1 | (0,0)(1,0)(0,1)(1,1) | Bank 1 | 左上 |
| S2 | (0,3)(1,3)(0,4)(1,4) | Bank 2 | 左中 |
| S3 | (2,5)(3,5)(2,6)(3,6) | Bank 0 | 左下 |
| S4 | (4,0)(5,0)(4,1)(5,1) | Bank 4 | 右上 |
| S5 | (4,2)(5,2)(4,3)(5,3) | Bank 9 | 右中 |
| S6 | (4,5)(5,5)(4,6)(5,6) | Bank 8 | 右下 |

#### BH 的具体 placement

8 个 S block 分左右两组（col 0-3 和 col 7-10），每组 4 个 block，每 block 8 个核心：

| S block | 绑定 DRAM bank | 物理位置 |
|---------|---------------|---------|
| S1 | Bank 1 | 左侧 col 0-3, row 1-2 |
| S2 | Bank 3 | 左侧 col 0-3, row 3-4 |
| S3 | Bank 2 | 左侧 col 0-3, row 7-8 |
| S4 | Bank 0 | 左侧 col 0-3, row 9-0（环绕） |
| S5 | Bank 5 | 右侧 col 7-10, row 1-2 |
| S6 | Bank 7 | 右侧 col 7-10, row 4-5 |
| S7 | Bank 6 | 右侧 col 7-10, row 6-7 |
| S8 | Bank 4 | 右侧 col 7-10, row 9-0（环绕） |

S4 和 S8 的核心跨越了 row 0 和 row 9，利用了 NOC 的 **torus 环绕** 特性——multicast 地址从 row 9 "绕回"到 row 0，在 torus 拓扑上仍然形成连续矩形。

### 3.2 四种通信模式的详细对比

#### 模式一：K DRAM 读（NCRISC → DRAM bank）

| 属性 | 值 |
|------|---|
| 方向 | DRAM → L1（pull） |
| 使用的 NOC | NOC0 (READ_NOC_INDEX = 0) |
| 发起者 | 每个 S block 的 sender core 的 NCRISC |
| 数据量 | 每 chunk: k_page_size × k_num_pages（典型 ~144KB） |
| 流水线 | 多 TRID 滑动窗口（最多 14 个 in-flight 请求） |
| 每 chunk 的 DRAM 读次数 | num_pages（WH 典型 18 次，BH 典型 9 次） |
| 双缓冲 | 有。当前 chunk 读进 buffer A，上一 chunk 的 buffer B 正在被 TRISC 消费 |

NCRISC 的 DRAM 读是整个系统中唯一的 DRAM 访问点。一个核心一次读一个 shard（一个 chunk），且该 shard 保证落在绑定的 DRAM bank 上。

#### 模式二：K Multicast（BRISC sender → S block 内 receivers）

| 属性 | 值 |
|------|---|
| 方向 | L1 → L1（push，multicast） |
| 使用的 NOC | NOC0 (MCAST_NOC_INDEX = 0) |
| 发起者 | sender core 的 BRISC |
| 目标 | 同 S block 内的所有 receiver core |
| 寻址方式 | 物理矩形多播：(start_x, start_y) → (end_x, end_y) |
| 数据量 | 逐 page 多播，每 page = k_page_size（≤8KB WH / ≤16KB BH） |
| 接收方数量 | WH: 3 (4-1), BH: 7 (8-1) |
| 同步方式 | 三阶段握手（见下） |

**K multicast 的三阶段握手**：

1. **Receiver ready**：每个 receiver 的 NCRISC 在 `cb_reserve_back` 完成后（buffer 空间已就绪），向 sender 的 `receiver_ready_semaphore` 发一个 +1。sender 等这个 semaphore 达到 num_receivers 才开始发。

2. **Page-level pipeline**：sender 的 BRISC 不需要等整块 chunk 读完。它看 NCRISC 的共享完成计数器：NCRISC 读完 page 0 → 计数 +1 → BRISC 立即 multicast page 0，同时 NCRISC 继续读 page 1。

3. **Chunk complete**：所有 page 多播完成后，sender 用 `noc_semaphore_set_multicast` 往矩形内所有 receiver 的 `mcast_semaphore` 写入 "VALID"，receiver 收到后 `cb_push_back` 通知 TRISC。

**为什么用 multicast 而不是 unicast？** 因为 S block 内的核心构成物理矩形。一次 multicast 写操作，NOC 硬件会自动复制数据到矩形内的所有核心，带宽消耗只算一次写。如果用 unicast 逐个发，sender 的出口带宽会成为瓶颈（4 个 receiver 需要发 4 次）。

#### 模式三：Q 广播（output core → 全部 worker）

| 属性 | 值 |
|------|---|
| 信号方向 | output core → 全芯片（multicast semaphore） |
| 数据方向 | worker → output core（各自 pull） |
| 信号使用的 NOC | NOC0（multicast semaphore inc） |
| 数据使用的 NOC | NOC1 (READ_NOC_INDEX = 1) |
| 数据量 | q_chunk_size_bytes（典型 ~256B - 几KB） |
| 寻址方式 | 信号：全芯片逻辑矩形 (0,0)→(max_x, max_y) 多播; 数据：点对点 unicast read |

**具体流程**：

1. 各 non-output core 的 BRISC 向 output core 的 `q_input_mcast_semaphore` 发 +1（unicast，NOC1）。
2. Output core 等 semaphore 达到 `full_grid_num_dests - 1`（所有人都准备好了）。
3. Output core 往全芯片矩形发 multicast semaphore +1：**"Q 已就绪"**。
4. 每个 non-output core 收到信号后，自己用 `noc_async_read`（NOC1）从 output core 的 L1 拉 Q。

**为什么不直接 multicast 数据？** 因为 non-output core 分散在不同 S block 的不同位置，不构成一个矩形。如果强行做多次矩形 multicast，开销更大。而 Q 数据很小（通常 < 1KB），每个核心各自 pull 一次的总代价很低。信号用全芯片矩形 multicast 是因为 semaphore 只有几字节，矩形越大效率越高。

#### 模式四：Tree Reduction（sender → partner，点对点 unicast）

| 属性 | 值 |
|------|---|
| 方向 | L1 → L1（push，unicast） |
| 使用的 NOC | NOC1 (WRITE_NOC_INDEX = 1) |
| 发起者 | sender S block 对应核心的 BRISC |
| 目标 | partner S block 对应核心（同一 batch_idx 位置） |
| 数据量 | o_write_size + ms_write_size（典型 ~几KB） |
| 同步方式 | 位域编码 semaphore（每步一位） |
| 写模式 | posted write（发完即走，不等确认） |

**为什么 reduction 用 unicast 而不是 multicast？** 因为 tree reduction 的每一步都是点对点的：一个特定 sender 写给一个特定 receiver。没有"同时发给多个"的需求。

**Semaphore 的位域编码**：

reduction semaphore 是一个 uint32_t，每一步占 1 位。当 step 0 的 sender 到达时，它往 partner 的 semaphore 的 bit 0 加 1。当 step 1 的 sender 到达时，往 bit 1 加 1。receiver 通过检查对应 bit 是否被 set 来判断哪一步的数据到了。

这样 **不同 step 的 sender 可以乱序到达**，receiver 可以按任意顺序处理已到达的数据。

### 3.3 四种 NOC 的使用分配

Tenstorrent 有两条独立的 NOC（NOC0 和 NOC1），可以同时传输不同数据而互不干扰。实验性 FlashMLA 的分配是：

| NOC | NCRISC 使用 | BRISC 使用 |
|-----|------------|-----------|
| NOC0 | DRAM 读（READ_NOC_INDEX=0） | K multicast + Q 信号 multicast (MCAST_NOC_INDEX=0) |
| NOC1 | receiver ready 信号 (ATOMIC_NOC_INDEX=1) | Q 数据 pull + tree reduction write (READ/WRITE_NOC_INDEX=1) |

关键点：**NCRISC 的 DRAM 读走 NOC0，BRISC 的 multicast 也走 NOC0**。但它们不冲突——NCRISC 读的是 DRAM → local L1，BRISC 写的是 local L1 → remote L1。NOC0 可以同时承载读和写方向的流量。

而 NOC1 留给了信号传输和 tree reduction 数据传输，这些都是小量的、低频的操作。

### 3.4 Virtual Channel 分配

NOC 有多个 virtual channel（VC），用来避免同一条物理链路上不同流量相互阻塞。实验性 FlashMLA 对 DRAM 读使用的 VC 按如下规则分配：

```
S block 0-3（左侧）: VC = s_block_idx & 1    → 0, 1, 0, 1
S block 4-7（右侧）: VC = 2 + (idx-4) & 1    → 2, 3, 2, 3
```

左右两侧用不同的 VC 范围（0-1 vs 2-3），避免跨芯片半区的 NOC 拥塞。同侧相邻 S block 交替使用 VC，避免物理相邻的读请求在同一 VC 上排队。

### 3.5 参数与硬件约束的对应关系

| 参数 | 含义 | WH B0 典型值 | BH 典型值 | 硬件约束 |
|------|------|-----------|---------|---------|
| `k_chunk_size` | 每 chunk token 数 | 128 | 128 | 必须等于 KV shard 高度 |
| `k_page_size` | 单次 NOC 传输大小 | ≤ 8192 B | ≤ 16384 B | NOC 单包上限 |
| `k_num_pages` | 一个 chunk 拆成的 page 数 | ~18 | ~9 | chunk_bytes / page_size |
| `NUM_TRIDS` | DRAM 读流水线深度 | 14 | 14 | 硬件 transaction ID 数量 - 1 |
| `num_mcast_dests` | multicast 目标数 | 3 | 7 | S block 内核数 - 1 |
| `dst_size` | DEST 寄存器容量（标准 tile 数） | 16 | 8 或 16 | fp32 模式减半，half-sync 减半 |
| `fp32_dest_acc_en` | DEST 用 FP32 累加 | 必须 False | 可选 | FP32 时 DEST 行数减半 |
| `dst_full_sync_en` | DEST 全同步模式 | 必须 True | 可选 | 全同步时 DEST 可用行翻倍 |
| S block 数 | 序列并行度 | 6 | 8 | 可用 DRAM bank 数量 |
| cores_per_block | 每 block 核心数 | 4 | 8 | 受 grid 大小和矩形约束限制 |
| `Q_TILE_HEIGHT` | Q 的 tiny tile 高度 | 8 | 8 | 每核 Q head 数决定 |
| `K_TILE_HEIGHT` | K 的标准 tile 高度 | 32 | 32 | Tensix 原生 tile 尺寸 |

**DEST 寄存器容量如何决定 tiling**：

TRISC 做 SDPA 计算时，需要在 DEST 寄存器中同时保持多个区域：
- mm2 输出区（V matmul 结果）：vDHt 个 tiny tile
- max / sum 统计区：2 个 tiny tile
- mm1 输出区（QK^T 结果）：Sk_chunk_t 个 tiny tile

总需求 = (vDHt + 2 + Sk_chunk_t) × packed_tile_size 行。

在 WH 上，`fp32_dest_acc_en=False` + `dst_full_sync_en=True` 给出 16 个标准 tile = 512 行。典型需求 (16+2+4)×16 = 352 行，够用。但如果开了 FP32（行数减半到 256）就不够了——这就是 WH 上必须关 FP32 的硬件原因。

### 3.6 完整通信流示意

把上面所有通信模式放在一起，一个 batch 处理过程中的完整通信流是：

```
阶段 1：Q 广播
  S1[0] output core ──信号 multicast (NOC0)──→ 全芯片所有核心
  所有 worker ──各自 unicast read (NOC1)──→ S1[0] output core

阶段 2：K 搬运与 multicast（对每个 chunk 重复）
  S1 sender ←─ DRAM bank 1 (NOC0 read)
  S2 sender ←─ DRAM bank 2 (NOC0 read)
  ...（6路 DRAM 读并行）
  S1 sender ──矩形 multicast (NOC0)──→ S1 block 内 3 个 receiver
  S2 sender ──矩形 multicast (NOC0)──→ S2 block 内 3 个 receiver
  ...（6路 multicast 并行）

阶段 3：计算（各核心独立，无通信）
  每个核心的 TRISC 独立做 flash attention 分块计算

阶段 4：Tree reduction
  Step 1: S2[0] ──unicast write (NOC1)──→ S1[0]
          S4[0] ──unicast write (NOC1)──→ S3[0]    （3对并行）
          S6[0] ──unicast write (NOC1)──→ S5[0]
  Step 2: S3[0] ──unicast write (NOC1)──→ S1[0]
  Step 3: S5[0] ──unicast write (NOC1)──→ S1[0]

最终：S1[0] 的 output core 拿到完整结果，已在 cb_out_final / output tensor shard 中。
```

# TT MLA 两种实现的内核级数据流与计算详解

## 1. 概述

TT-Metal 中存在两套 MLA 实现路径：

| | 实现 A：生产路径 | 实现 B：实验 FlashMLA |
|---|---|---|
| 位置 | `mla1d.py` + `sdpa_decode_program_factory.cpp` | `deepseek_v3_b1/micro_ops/flash_mla/` |
| 硬件 | Wormhole B0 (8×7=56 cores) | Blackhole (11×10=110 cores) |
| 阶段 | Prefill + Decode | 仅 Decode |
| 编程模型 | C++ Program Factory + 3 独立内核文件 | Python `generic_op` + 统一内核头文件 |

本文档深入到内核级别，详细说明每种实现中数据如何流动、计算如何执行。

---

## 2. 实现 A：生产路径的完整数据流

### 2.1 Decode 全链路（Python 层，`mla1d.py:1430-1520`）

以 DeepSeek V3 典型参数为例：
- `B=32`（batch），`NH_local=16`（本设备 Q 头数），`NKV=1`
- `kv_lora_rank=512`，`qk_nope_head_dim=128`，`qk_rope_head_dim=64`
- `kvpe_dim=576`（=512+64），`qk_head_dim=192`（=128+64）
- `q_lora_rank=1536`，`v_head_dim=128`

```
输入 x: [1, 1, B=32, dim=7168]  (WIDTH sharded, L1)
  │
  │ ① wq_kv_a — ttnn.linear（融合 Q 下投影 + KV 下投影）
  │   权重: [dim, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
  │        = [7168, 1536+512+64] = [7168, 2112]
  │   输出: [1, 1, 32, 2112]  (WIDTH sharded)
  │
  │ ② all_gather + fast_reduce_nc（多设备 AllReduce）
  │   输出: [1, 1, 32, 2112]  (L1)
  │
  │ ③ slice 分三路:
  │   ├─ tt_q:       [1, 1, 32, 1536]  (q_lora_rank)
  │   ├─ tt_kv_nope: [1, 1, 32, 512]   (kv_lora_rank)
  │   └─ tt_kv_rope: [1, 1, 32, 64]    (qk_rope_head_dim)
  │
  ├── Q 路径 ──────────────────────────────────────────
  │   ④ q_norm: RMSNorm [1,1,32,1536]
  │   ⑤ wq_b: ttnn.linear [1,1,32,1536] → [1,1,32, NH_local × qk_head_dim]
  │           = [1,1,32,16×192] = [1,1,32,3072]
  │   ⑥ reshape → [1,32,16,192], slice 为:
  │      q_nope: [1,32,16,128]   q_rope: [1,32,16,64]
  │   ⑦ wkv_b1: ttnn.linear 对 q_nope  ◄── K 侧"解压缩"吸收到 Q
  │      [1,NH_local,32,128] × W_UK^T → [1,NH_local,32,512]
  │   ⑧ RoPE: rotary_embedding_llama 对 q_rope
  │      [1,32,16,64] → [1,32,16,64]
  │   ⑨ concat: q_nope(512) + q_rope(64) → Q_attn [1,32,16,576]
  │   ⑩ all_to_all: TP 通信，重分布 Q heads
  │      → [1,4,128,576]  (height sharded)
  │
  ├── KV 路径 ─────────────────────────────────────────
  │   ④' kv_norm: RMSNorm [1,1,32,512]
  │   ⑤' KV RoPE: rotary_embedding_llama 对 kv_rope
  │       [1,32,1,64] → [1,32,1,64]
  │   ⑥' concat: kv_nope(512) + kv_rope(64) → kvpe [1,32,1,576]
  │   ⑦' paged_update_cache: 写入 kvpe_cache
  │       kvpe_cache shape: [max_pages, 1, block_size, 576]  (Paged DRAM)
  │
  ├── SDPA Decode ─────────────────────────────────────
  │   ⑪ paged_flash_multi_latent_attention_decode(Q_attn, kvpe_cache)
  │      Q:  [1, 4, 128, 576]  (height sharded L1)
  │      KV: [max_pages, 1, block_size, 576]  (Paged DRAM)
  │      → attn_out: [1, 4, 128, 512]  (height sharded L1)
  │          (输出 head_dim_v = kv_lora_rank = 512)
  │
  ├── V 侧解压缩 ──────────────────────────────────────
  │   ⑫ wkv_b2: ttnn.linear  ◄── 独立 dispatch
  │      [1,128,4,512] × W_UV → [1,128,4,128]
  │      (transpose 后做 matmul，将 latent 512 → v_head_dim 128)
  │
  └── 输出投影 ─────────────────────────────────────────
      ⑬ all_gather + reshape
      ⑭ wo: ttnn.linear [1,1,32,128×128] → [1,1,32,7168]
      → 输出: [1, 1, 32, 7168]
```

### 2.2 SDPA Decode 内核内部数据流

`paged_flash_multi_latent_attention_decode` 的底层是 `sdpa_decode_program_factory.cpp`，创建 3 个独立的内核文件：

```
sdpa_decode/device/kernels/
├── dataflow/reader_decode_all.cpp   (NCRISC - 数据读取)
├── compute/sdpa_flash_decode.cpp    (TRISC - 矩阵计算)
└── dataflow/writer_decode_all.cpp   (BRISC - 数据写出/归约)
```

#### 2.2.1 工作分配（Host 端，`sdpa_decode_program_factory.cpp:162-173`）

```
num_active_cores = num_cores_per_head × num_kv_heads × B / num_heads_per_core

典型 MLA decode 配置 (WH_B0):
  B=4 (batch×head_parallel), NKV=1, max_cores_per_head_batch=4
  → num_cores_per_head = 4, num_active_cores = 4×1×4/1 = 16
  → 每个 batch 由 4 个核并行处理 K 的不同 chunk 范围
```

#### 2.2.2 Reader (NCRISC) — `reader_decode_all.cpp`

每个核的 Reader 执行如下流程：

```
1. 读取 cur_pos_tensor 获得当前序列位置
2. 计算本核负责的 K chunk 范围:
     valid_seq_len = nearest_32(cur_pos + 1)
     k_num_chunks = valid_seq_len / k_chunk_size
     本核负责: chunks [core_id, core_id + stride, core_id + 2*stride, ...]
              其中 stride = num_cores_per_head

3. 读 Q:
     从 Q sharded memory 读入 cb_q_in
     Q shape per core: [PNHt tiles × DHt tiles]

4. 对于每个分配的 K chunk:
     a. 从 DRAM 读取 K chunk:
        - Paged: 通过 page_table 查找物理页 → noc_async_read_tile
        - Non-paged: 直接计算 DRAM 地址 → noc_async_read_tile
        K chunk: [Sk_chunk_t × DHt] tiles
     b. 将 K chunk 推入 cb_k_in circular buffer
     c. Compute 内核从 cb_k_in 消费 K chunk 并计算

5. 注意: 每个 K chunk 被独立从 DRAM 读取到 L1
   NKV=1 时多核可能读取相同的 K 数据（无 multicast）
```

#### 2.2.3 Compute (TRISC) — `sdpa_flash_decode.cpp`

对于分配到的每个 K chunk，Compute 执行 Online Flash Attention：

```
初始化:
  O = 0           // 累积输出，shape: [PNHt × vDHt]
  m = -inf        // 每行最大值，shape: [PNHt]
  l = 0           // 每行 exp 之和，shape: [PNHt]

For each K_chunk:
  ┌──────────────────────────────────────────────┐
  │ 1. QK matmul:                                │
  │    S = Q × K_chunk^T                         │
  │    [PNHt, DHt] × [DHt, Sk_chunk_t]           │
  │    → S: [PNHt, Sk_chunk_t]                   │
  │                                              │
  │ 2. Scale:                                    │
  │    S = S × scale                             │
  │                                              │
  │ 3. Mask (最后一个 chunk):                     │
  │    S[i][j] = -inf  where j > cur_pos         │
  │                                              │
  │ 4. Online softmax update:                    │
  │    m_new = max(m_old, row_max(S))            │
  │    correction = exp(m_old - m_new)           │
  │    P = exp(S - m_new)                        │
  │    l_new = l_old * correction + row_sum(P)   │
  │                                              │
  │ 5. Output rescale:                           │
  │    O = O * correction                        │
  │                                              │
  │ 6. AV matmul:                                │
  │    O += P × V_chunk                          │
  │    [PNHt, Sk_chunk_t] × [Sk_chunk_t, vDHt]  │
  │    V_chunk 从 K buffer 的前 vDHt 列读取       │
  │    (MLA V-from-K: skip_src_cols = DHt-vDHt)  │
  │                                              │
  │ 7. Update state:                             │
  │    m = m_new, l = l_new                      │
  └──────────────────────────────────────────────┘

最终:
  O = O / l   // 归一化
  将 O 和统计量 (m, l) 写入输出 CB
```

#### 2.2.4 Writer (BRISC) — `writer_decode_all.cpp`

```
1. 如果 num_cores_per_head > 1:
     接收其他核的部分结果进行 Tree Reduction:
     For each reduction round (log2(num_cores_per_head) rounds):
       if 本核是 receiver:
         等待 sender 通过 NoC 写入部分 O 和 (m, l)
         执行 sdpa_tail: 合并两组 (O, m, l) → 更新后的 (O, m, l)
       elif 本核是 sender:
         通过 NoC 将本核的 (O, m, l) 发送给 receiver
         完成后退出

2. 最终 reducer 核(通常是 core_0):
     将归约后的 O 写入输出 tensor (DRAM 或 L1 sharded)
```

#### 2.2.5 实现 A 的 DRAM 访问模式图

```
            ┌──────────────────────────────────┐
            │           DRAM (6 banks)          │
            │                                  │
            │  KV Cache (Paged/Interleaved)    │
            │  ┌────┬────┬────┬────┬────┬────┐ │
            │  │Pg 0│Pg 1│Pg 2│... │    │    │ │
            │  └──┬─┴──┬─┴──┬─┴────┴────┴────┘ │
            └─────┼────┼────┼──────────────────┘
                  │    │    │
      ┌───────────┼────┼────┼───────────────────┐
      │     各核独立通过 NoC 读取 K chunks        │
      │                                         │
      │  Core 0    Core 1    Core 2    Core 3   │
      │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │
      │  │Read │  │Read │  │Read │  │Read │   │
      │  │K[0] │  │K[1] │  │K[2] │  │K[3] │   │
      │  │K[4] │  │K[5] │  │K[6] │  │K[7] │   │
      │  │ ... │  │ ... │  │ ... │  │ ... │   │
      │  │Comp │  │Comp │  │Comp │  │Comp │   │
      │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘   │
      │     │        │        │        │       │
      │     └──── Tree Reduction ──────┘       │
      │              │                          │
      │           Output                        │
      └─────────────────────────────────────────┘

问题: NKV=1 时，4 个核读取相同的 K 数据
     → 同一 DRAM bank 被 4 个核同时争抢
```

---

## 3. 实现 B：实验性 FlashMLA 的完整数据流

### 3.1 整体架构：S Block

Blackhole 的 110 个工作核 (11列 × 10行) 中，FlashMLA 使用 **8 个 S Block**（共 64 核）：

```
逻辑列:  0    1    2    3    4    5    6    7    8    9   10
       ┌───────────────────────────────────────────────────────┐
行 0   │ S4   S4   S4   S4    ·    ·    ·   S8   S8   S8   S8 │  ← S4/S8 跨 row9→row0
行 1   │ S1   S1   S1   S1    ·    ·    ·   S5   S5   S5   S5 │     (torus 环绕, 实际相邻)
行 2   │ S1   S1   S1   S1    ·    ·    ·   S5   S5   S5   S5 │
行 3   │ S2   S2   S2   S2    ·    ·    ·    ·    ·    ·    ·  │
行 4   │ S2   S2   S2   S2    ·    ·    ·   S6   S6   S6   S6 │
行 5   │  ·    ·    ·    ·    ·    ·    ·   S6   S6   S6   S6 │
行 6   │  ·    ·    ·    ·    ·    ·    ·   S7   S7   S7   S7 │
行 7   │ S3   S3   S3   S3    ·    ·    ·   S7   S7   S7   S7 │
行 8   │ S3   S3   S3   S3    ·    ·    ·    ·    ·    ·    ·  │
行 9   │ S4   S4   S4   S4    ·    ·    ·   S8   S8   S8   S8 │  ← S4/S8 跨 row9→row0
       └───────────────────────────────────────────────────────┘
                  ·  = 未被 FlashMLA 使用的核心 (共 46 个)

每个 S Block 均为 4列 × 2行 = 8 核, 大小完全相同。
```

**为什么这样分布？** 核心设计原则是 **DRAM 邻近性**：

Blackhole 的 8 个 DRAM Bank 分布在芯片左右两侧边缘:
  - Bank 0-3 在物理列 0 (紧邻 worker 左侧 cols 0-3)
  - Bank 4-7 在物理列 9 (紧邻 worker 右侧 cols 7-10)

S Block 到 DRAM Bank 的映射基于物理距离最小化 (bank 子通道 Y 坐标 vs S Block 物理行):

```
  S Block → DRAM Bank │ Bank 子通道 phys-Y │ S Block phys-Y   │ 距离
  ─────────────────────┼────────────────────┼──────────────────┼──────
  S1 → Bank 1          │ 2, 3, 10           │ 3, 4             │ ≤1 hop
  S2 → Bank 3          │ 5, 6, 7            │ 5, 6             │ ≤1 hop
  S3 → Bank 2          │ 4, 8, 9            │ 9, 10            │ ≤1 hop
  S4 → Bank 0          │ 0, 1, 11           │ 11, 2 (环绕)     │ ≤1 hop
  S5 → Bank 5          │ 2, 3, 10           │ 3, 4             │ ≤1 hop
  S6 → Bank 7          │ 5, 6, 7            │ 6, 7             │ ≤1 hop
  S7 → Bank 6          │ 4, 8, 9            │ 8, 9             │ ≤1 hop
  S8 → Bank 4          │ 0, 1, 11           │ 11, 2 (环绕)     │ ≤1 hop
```

中间列 4-6 (物理列 5-7) 距左右 DRAM 都最远, 不适合 DRAM 带宽密集的 FlashMLA。
未使用的 46 核可被模型中其他 fused op (MoE, down_proj 等) 占用, 不会浪费。

S Block 定义 (op.py FlashMLAOptimalGridNOC0.BLOCKS, 坐标 (x=列, y=行)):
  S1 = [(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)] → DRAM bank 1
  S2 = [(0,3),(1,3),(2,3),(3,3),(0,4),(1,4),(2,4),(3,4)] → DRAM bank 3
  S3 = [(0,7),(1,7),(2,7),(3,7),(0,8),(1,8),(2,8),(3,8)] → DRAM bank 2
  S4 = [(0,9),(1,9),(2,9),(3,9),(0,0),(1,0),(2,0),(3,0)] → DRAM bank 0
  S5 = [(7,1),(8,1),(9,1),(10,1),(7,2),(8,2),(9,2),(10,2)] → DRAM bank 5
  S6 = [(7,4),(8,4),(9,4),(10,4),(7,5),(8,5),(9,5),(10,5)] → DRAM bank 7
  S7 = [(7,6),(8,6),(9,6),(10,6),(7,7),(8,7),(9,7),(10,7)] → DRAM bank 6
  S8 = [(7,9),(8,9),(9,9),(10,9),(7,0),(8,0),(9,0),(10,0)] → DRAM bank 4

### 3.2 KV Cache 布局：ND Sharded DRAM

```
KV Cache shape: [batch=1, nkv=1, max_seq_len=32768, kvpe_dim=576]

ND Sharding: 按 k_chunk_size=128 切分，round-robin 分布到 8 个 DRAM bank

  DRAM bank 1 (S1): chunk 0, chunk 8, chunk 16, ...
  DRAM bank 3 (S2): chunk 1, chunk 9, chunk 17, ...
  DRAM bank 2 (S3): chunk 2, chunk 10, chunk 18, ...
  DRAM bank 0 (S4): chunk 3, chunk 11, chunk 19, ...
  DRAM bank 5 (S5): chunk 4, chunk 12, chunk 20, ...
  DRAM bank 7 (S6): chunk 5, chunk 13, chunk 21, ...
  DRAM bank 6 (S7): chunk 6, chunk 14, chunk 22, ...
  DRAM bank 4 (S8): chunk 7, chunk 15, chunk 23, ...

每个 shard = [1, 1, 128, 576] = 128 × 576 × 1B(bf8) = 72KB
```

### 3.3 Q Head 到核的映射

```
batch=1, num_heads=64, num_q_heads_per_core=8
→ 8 个 Q shards，每个 shard = 8 个 head
→ 每个 Q shard 映射到 8 个 S Block 中对应索引的核

Q shard 0: S1[0], S2[0], S3[0], S4[0], S5[0], S6[0], S7[0], S8[0]
Q shard 1: S1[1], S2[1], S3[1], S4[1], S5[1], S6[1], S7[1], S8[1]
...
Q shard 7: S1[7], S2[7], S3[7], S4[7], S5[7], S6[7], S7[7], S8[7]

每个 Q shard 由 8 个核并行处理：
  - S1 的核: output core (存储最终输出 + 参与归约)
  - S2-S8 的核: worker cores (序列并行 + 参与归约)

Q Tensor: [1, 1, 64, 576]
  height sharded 到 S1 的 8 个核
  每核: 8 个 head × 576 dim = [8, 576] (使用 Tiny Tile 8×32)
```

### 3.4 三个 RISC 的详细数据流

以一个 batch 的 8 个核（跨 S1-S8）为例，假设 `cur_pos = 1023`：

```
valid_seq_len = nearest_128(1024) = 1024
k_num_chunks = 1024 / 128 = 8
每核负责: chunk[core_num], stride=8
→ core 0 (S1): chunk 0
  core 1 (S2): chunk 1
  core 2 (S3): chunk 2
  ...
  core 7 (S8): chunk 7

每个核只处理 1 个 K chunk（128 tokens × 576 dim）
```

如果 `cur_pos = 4095`：

```
k_num_chunks = 4096 / 128 = 32
每核负责: 32/8 = 4 个 chunks (strided)
→ core 0 (S1): chunks 0, 8, 16, 24
  core 1 (S2): chunks 1, 9, 17, 25
  ...
  core 7 (S8): chunks 7, 15, 23, 31
```

#### 3.4.1 NCRISC (Reader) — DRAM 读取 + 页级流水线

```
时间线 (S1 核 = mcast sender):
──────────────────────────────────────────────────────────
                    DRAM 读取                NoC 写入
──────────────────────────────────────────────────────────
Phase 1: 等待 kv_cache_cur_pos_ready 信号量
         (确保前序 fused op 已完成 KV cache 写入)

Phase 2: 对于每个 K chunk:

  使用 NOC Transaction ID (trid) 实现页级流水线:

  trid 1 ──► 发射 page 0 读取请求 ──────────────────►
  trid 2 ──► 发射 page 1 读取请求 ──────────────────►
  trid 3 ──► 发射 page 2 读取请求 ──────────────────►
  ...
             │                                        │
             ▼                                        ▼
  trid 1 完成 ► *ncrisc_brisc_sync_ptr += 1 ──► BRISC 收到通知
                                                  ├► 立即 multicast page 0
  trid 2 完成 ► *ncrisc_brisc_sync_ptr += 1 ──► BRISC 收到通知
                                                  ├► 立即 multicast page 1
  ...

  K chunk 全部到达 L1 后:
  cb_push_back(cb_k_in, k_chunk_tiles)
  ──► TRISC 可以开始 compute

关键: DRAM 读取和 multicast 是 page 级重叠的!
     不需要等整个 chunk 读完再 multicast
──────────────────────────────────────────────────────────

非 sender 核 (S2-S8):
  发送 receiver_ready 信号给 sender
  等待 mcast_semaphore → BRISC 的 multicast 完成后自动设置
  cb_push_back(cb_k_in, k_chunk_tiles)
```

#### 3.4.2 BRISC (Writer) — Q 广播 + K Multicast + Tree Reduction

BRISC 承担三个职责：

```
═══════════════════════════════════════════════════════
Phase 1: Q 输入广播
═══════════════════════════════════════════════════════

Output core (S1 的核):
  Q 已在 L1 sharded memory 中 (cb_q_in)
  等待所有其他核发送 q_input_mcast_semaphore
  然后 multicast Q 到全网格所有核:
    noc_semaphore_inc_multicast(q_input_mcast_semaphore_addr,
                                1, full_grid_num_dests)

Non-output core:
  如果是 S1 的核 (has Q shard):
    等待 multicast 完成信号
  其他核 (S2-S8):
    从 output core 的 L1 地址读取 Q:
      noc_async_read(output_core_noc_addr, local_cb_q_addr, q_size)
    推入 cb_q_in

═══════════════════════════════════════════════════════
Phase 2: K Multicast (仅 sender 核, 每个 S Block 的第一个核)
═══════════════════════════════════════════════════════

For each K chunk:
  1. 等待 NCRISC 完成至少 1 个 page 的 DRAM 读取:
       noc_semaphore_wait_min(ncrisc_brisc_sync_ptr, 1)
       invalidate_l1_cache()  // 确保看到最新数据
       page_addr = *k_write_ptr_shared  // NCRISC 写入的地址

  2. 等待 S Block 内所有 receiver 核就绪:
       noc_semaphore_wait(receiver_ready_semaphore, num_mcast_dests=7)

  3. 逐页 multicast (与 DRAM 读取流水线重叠):
       For page 0:
         noc_async_write_multicast(local_addr, mcast_addr, page_size, 7)
       For page 1..N:
         等待该 page DRAM 读取完成
         noc_async_write_multicast(...)

  4. 设置 mcast_semaphore 通知所有 receiver K 已就绪:
       noc_semaphore_set_multicast(mcast_sem, mcast_addr, 7)

═══════════════════════════════════════════════════════
Phase 3: Tree Reduction
═══════════════════════════════════════════════════════

8 个 S Block 的部分 (O, m, l) 结果通过 3 步归约:

Step 1 (4 路并行):
  S2 ──send──► S1    S4 ──send──► S3
  S6 ──send──► S5    S8 ──send──► S7

Step 2 (2 路并行):
  S3 ──send──► S1    S7 ──send──► S5

Step 3 (1 路):
  S5 ──send──► S1

Sender 的执行:
  cb_wait_front(cb_out_o, out_tiles)   // 等待 compute 完成
  cb_wait_front(cb_out_ms, 1)          // 等待统计量
  noc_async_write(local_ms, partner_ms_addr)   // 发送 (m,l)
  noc_async_write(local_o, partner_o_addr)     // 发送 O
  noc_semaphore_inc(partner_sem, step_bit)     // 通知 receiver
  break  // sender 在发送后立即退出

Receiver 的执行:
  cb_reserve_back(cb_ms_in, 1)
  cb_reserve_back(cb_out_in, out_tiles)
  while (*sem & step_mask == 0): spin  // 等待 sender
  cb_push_back(cb_ms_in, 1)
  cb_push_back(cb_out_in, out_tiles)
  // → TRISC 的 sdpa_tail 执行归约

信号量编码: 使用位编码避免 3 步之间的干扰
  step 0: bit 0   step 1: bit 1   step 2: bit 2
```

#### 3.4.3 TRISC (Compute) — SDPA Flash Attention + Tree Reduction

```
═══════════════════════════════════════════════════════
Phase 1: Flash Attention Compute (与实现 A 类似但更精细)
═══════════════════════════════════════════════════════

初始化:
  sdpa_custom_mm_block_init_short<transpose_k>(cb_q_in, cb_k_in)
  cb_wait_front(cb_q_in, q_chunk_tiles)  // Q 已由 BRISC 广播就绪

寄存器布局 (DST accumulator):
  ┌─────────────────────────────────────────────────┐
  │ offset 0:    mm2_dst (AV matmul 累积输出 O)     │
  │              vDHt 个 tile                       │
  │ offset X:    max_dst (每行最大值 m)              │
  │ offset X+2:  sum_dst (每行 exp 之和 l)           │
  │ offset X+16: corr_exp_dst (correction factor)   │
  │ offset Y:    mm1_dst (QK matmul 临时结果 S)      │
  └─────────────────────────────────────────────────┘

For each K chunk (num_chunks 个):
  compute_sdpa_chunk():
    1. 等待 K chunk: cb_wait_front(cb_k_in, k_chunk_tiles)

    2. QK matmul (transpose K):
         S = Q[PNHt, DHt] × K^T[DHt, Sk_chunk_t]
         → S: [PNHt, Sk_chunk_t]  存入 mm1_dst

    3. Scale: S *= scale_bf16

    4. Mask (如果是最后一个 chunk):
         从 cb_mask 读取 mask tile
         S += mask  (将无效位置设为 -inf)

    5. Online softmax:
         if first_chunk:
           m = row_max(S)
         else:
           m_new = max(m_old, row_max(S))
           correction = exp(m_old - m_new)
           O *= correction        // rescale 之前的输出
           l *= correction        // rescale 之前的 sum

         P = exp(S - m)
         l += row_sum(P)

    6. AV matmul:
         O += P[PNHt, Sk_chunk_t] × V[Sk_chunk_t, vDHt]
         V 从 K buffer 的前 vDHt 列 strided 读取

    7. 释放 K chunk: cb_pop_front(cb_k_in, k_chunk_tiles)

所有 chunks 完成后:
  if 直接输出 (无归约):
    O = O / l  (reciprocal scale)
    pack → cb_out_final (sharded output)
  else:
    pack m,l → cb_out_ms
    pack O → cb_out_o 或 cb_interm_out
    → 等待 Tree Reduction

═══════════════════════════════════════════════════════
Phase 2: Tree Reduction Compute (接收方执行)
═══════════════════════════════════════════════════════

sdpa_tail() 将两组 (O_local, m_local, l_local) 和
          (O_remote, m_remote, l_remote) 合并:

For each reduction step where this core is receiver:
  1. 从 cb_ms_in 读取 remote 的 (m, l)
  2. 从 cb_out_in 读取 remote 的 O

  3. 合并:
     m_new = max(m_local, m_remote)
     corr_local = exp(m_local - m_new)
     corr_remote = exp(m_remote - m_new)
     O_new = O_local * corr_local + O_remote * corr_remote
     l_new = l_local * corr_local + l_remote * corr_remote

  4. 如果是最后一步:
     O_final = O_new / l_new
     pack → cb_out_final

  5. 否则继续下一步归约
```

### 3.5 完整时间线图

```
时间 ────────────────────────────────────────────────────────►

S1 sender 核 (output core + mcast sender):
BRISC:  [Q mcast]─────[等NCRISC]─[K mcast pg0]─[K mcast pg1]─...─[Tree recv]─[Tree recv]─[Tree recv]
NCRISC: ────────────[DRAM read pg0]─[pg1]─[pg2]─...──────────────────────────────────────────────────
TRISC:  ──────────────────────[wait Q]──[wait K]──[QK+softmax+AV]──[wait K]──...──[reduce]──[reduce]──[output]

S2 receiver 核:
BRISC:  [Q read from S1]─[signal ready]─[wait K mcast]─...──────[Tree send]───────────────────────
NCRISC: [signal ready]─[wait K mcast]─...────────────────────────────────────────────────────────
TRISC:  ────────────────[wait Q]──[wait K]──[QK+softmax+AV]──[wait K]──...──[send O,m,l to S1]────

S5 receiver 核:
BRISC:  [Q read from S1]─[signal ready]─[wait K mcast]─...──────[Tree recv from S6]─[Tree send to S1]
NCRISC: ...
TRISC:  ──────────────────[wait K]──[QK+softmax+AV]──...──[reduce S6]──[send to S1]─────────────────

关键重叠:
  ✓ DRAM 读取 (NCRISC) 与 K multicast (BRISC) 页级重叠
  ✓ K multicast 与 compute 可部分重叠 (double-buffered cb_k_in)
  ✓ Tree reduction 的不同步可以并行 (step 1 的 4 组同时进行)
```

---

## 4. 两种实现的关键差异总结

### 4.1 K 数据流对比

```
实现 A (每个 batch+head 对由 4 个核处理):
  DRAM bank X ←──read──── Core 0 (chunk 0, 4, 8, ...)
  DRAM bank X ←──read──── Core 1 (chunk 1, 5, 9, ...)   ← 同一 bank 被 4 核争抢
  DRAM bank X ←──read──── Core 2 (chunk 2, 6, 10, ...)
  DRAM bank X ←──read──── Core 3 (chunk 3, 7, 11, ...)

实现 B (每个 Q shard 由 8 个核处理):
  DRAM bank 1 ←──read──── S1 sender ──mcast──► S1 的 7 个 receiver
  DRAM bank 3 ←──read──── S2 sender ──mcast──► S2 的 7 个 receiver  ← 每 bank 只 1 核读
  DRAM bank 2 ←──read──── S3 sender ──mcast──► S3 的 7 个 receiver
  ...
```

### 4.2 计算并行度对比

```
实现 A:
  Core 0: chunk0 → chunk4 → chunk8 → chunk12 → ...  (串行遍历)

实现 B:
  S1[i]: chunk0 ──────►     ← 每核只处理 1/8 的 chunks
  S2[i]: chunk1 ──────►
  S3[i]: chunk2 ──────►     全部并行
  ...
  S8[i]: chunk7 ──────►
         └── Tree Reduce → Output
```

### 4.3 dispatch 开销对比

```
实现 A (完整 MLA decode):
  wq_kv_a(0.3ms) → AG(0.3ms) → reduce(0.3ms) → slice×3(0.9ms)
  → q_norm(0.3ms) → kv_norm(0.3ms) → wq_b(0.3ms) → slice×2(0.6ms)
  → wkv_b1(0.3ms) → RoPE×2(0.6ms) → concat×2(0.6ms)
  → cache_update(0.3ms) → a2a(0.3ms)
  → SDPA_decode(0.3ms)     ◄── 核心计算
  → wkv_b2(0.3ms) → AG(0.3ms) → wo(0.3ms)
  ≈ ~18 次 dispatch × ~0.15-0.3ms = 3-5ms 固定开销

实现 B (FlashMLA decode micro-op 本身):
  1 次 generic_op dispatch
  ≈ ~0.15-0.3ms 固定开销
  (但 wkv_b1/wkv_b2/wo 等仍在外部)
```

---

## 5. 实现 B 的独特优化技术详解

### 5.1 NOC Transaction ID (trid) 流水线

传统 DRAM 读取是阻塞的（读完才能读下一个）。实现 B 使用 trid 实现非阻塞流水线：

```cpp
// NCRISC 中的流水线读取 (flash_mla.hpp:292-333)
constexpr uint32_t NUM_TRIDS = NOC_MAX_TRANSACTION_ID - 1;

// 一次性发射 NUM_TRIDS 个并行读取请求
for (uint32_t i = 0; i < NUM_TRIDS && pages_issued < total; ++i) {
    noc_async_read_set_trid(curr_trid);
    noc_async_read_one_packet_with_state_with_trid(src, offset, dst, curr_trid);
    curr_trid = (curr_trid % NUM_TRIDS) + 1;
    pages_issued++;
}

// 按序等待完成，每完成一个就通知 BRISC + 发射新请求
while (pages_completed < total) {
    noc_async_read_barrier_with_trid(wait_trid);  // 等待特定 trid
    *ncrisc_brisc_sync_ptr += 1;  // 通知 BRISC 该页就绪
    if (more_pages) issue_next_page();  // 立即发射下一个
}
```

效果：DRAM 带宽利用率接近理论峰值，延迟被流水线隐藏。

### 5.2 NCRISC-BRISC 同步机制

两个 RISC 核在同一物理核上通过 L1 共享内存协调：

```
L1 共享区域 (从 ncrisc_brisc_sync_semaphore_addr 开始):
  offset 0:  sync_curr_ptr     — 当前 chunk 的 page 完成计数
  offset 4:  sync_next_ptr     — 下一个 chunk 的 page 完成计数 (double buffer)
  offset 8:  k_write_curr_ptr  — 当前 K chunk 在 L1 中的写入地址
  offset 12: k_write_next_ptr  — 下一个 chunk 的写入地址 (double buffer)

流程:
  NCRISC: 写入 K page → 递增 sync_counter → 继续下一个 page
  BRISC:  等待 sync_counter ≥ 1 → 读取 K 地址 → multicast 该 page
         等待 sync_counter ≥ 2 → multicast page 2
         ...

双缓冲: 处理 chunk N 的 multicast 与 chunk N+1 的 DRAM 读取重叠
  通过 std::swap(curr_ptr, next_ptr) 在两组指针之间切换
```

### 5.3 Tiny Tile 优化

实现 B 对 Q tensor 使用 8×32 的 Tiny Tile（而非标准 32×32）：

```
标准 Tile (32×32):
  如果 num_q_heads_per_core = 8:
    Q tile = [32, 32]，但实际只有前 8 行有效
    → 24 行是 padding → 75% 计算浪费

Tiny Tile (8×32):
  Q tile = [8, 32]，8 行全部有效
  → 0% padding 浪费
  → QK matmul 和 AV matmul 的计算量减少 4×

stats_tile 和 im_tile 也使用 Tiny Tile，减少中间结果的存储和计算开销
```

---

## 6. 实现对比总表

| 维度 | 实现 A (生产路径) | 实现 B (实验 FlashMLA) |
|---|---|---|
| **Python 入口** | `mla1d.py` → 12+ ttnn ops | `op.py` → 1 `generic_op` |
| **C++ 内核** | 3 个独立 .cpp 文件 | 1 个统一 .hpp (3 RISC 段) |
| **DRAM 读取** | 各核独立 noc_async_read | trid 流水线 + 页级发射 |
| **K 分发** | 无 multicast (decode) | S Block sender → 7 receivers |
| **NCRISC-BRISC 协调** | CB push/pop 隐式同步 | 显式信号量 + 共享 L1 指针 |
| **归约** | C++ 中定义 tree reduction | Python 定义 3 步 tree reduction |
| **Q 广播** | 每核独立从 sharded memory 读 | output core multicast 到全网格 |
| **Tile** | 标准 32×32 | **Tiny Tile** 8×32 (Q/stats) |
| **KV Cache** | Interleaved/Paged DRAM | **ND Sharded DRAM** |
| **DRAM 局部性** | 无保证 | **S Block ↔ DRAM bank 1:1** |
| **两种 MLA 的共同点** | V-from-K (mla_kv_overlap) | V-from-K (vDHt < DHt) |
| **compute 算法** | Online Flash Attention | Online Flash Attention (相同) |
| **wkv_b1/b2 融合** | 未融合 | 未融合 |

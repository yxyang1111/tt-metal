# SDPA 中的片上网络 (NoC)、多播 (Multicast) 与片间通信 — 代码实现详解

本文档从代码层面完整剖析 SDPA 实现中 **NoC 单播**、**NoC 多播** 和 **片间 (Ring Distributed)** 三种数据传输模式的具体实现。

---

## 一、总体架构：从 Host 到 Device 的完整链路

```
┌───────────────────── Host (C++) ──────────────────────┐
│                                                        │
│  sdpa_program_factory.cpp                              │
│    ├─ 计算并行度（batch × head × q_chunk）              │
│    ├─ 为每个核心分配工作                                │
│    ├─ 构建 KV Chain（核间转发链）                       │
│    ├─ 判断是否启用 Multicast                            │
│    ├─ 创建 CB / Semaphore / Kernel                     │
│    └─ SetRuntimeArgs (物理坐标、地址...)                │
│                                                        │
└────────────────────────┬───────────────────────────────┘
                         │ EnqueueProgram
                         ▼
┌───────────────────── Device ──────────────────────────┐
│                                                        │
│  Core 0: reader_interleaved.cpp (dataflow)             │
│          sdpa.cpp (compute)                            │
│          writer_interleaved.cpp (dataflow)              │
│                                                        │
│  Core 1: 同上                                          │
│  Core 2: 同上                                          │
│  ...                                                   │
└────────────────────────────────────────────────────────┘
```

---

## 二、片上 NoC 单播：KV Chain 转发

### 2.1 为什么需要 KV Chain？

当多个核心处理同一 batch+head 的不同 Q chunk 时，它们都需要完整的 K/V 数据。
如果每个核心都从 DRAM 独立读取，DRAM 带宽成为瓶颈。

**KV Chain 解决方案**：只有链头 (injector) 从 DRAM 读取 K/V，然后通过 NoC **转发** 给链上的下一个核心。

```
DRAM ──读取──→ Core 0 (injector) ──NoC 写──→ Core 1 ──NoC 写──→ Core 2 (sink)
                  │                            │                     │
              处理 Q[0:2]                  处理 Q[2:4]           处理 Q[4:6]
              遍历全部K                    遍历全部K             遍历全部K
```

### 2.2 Host 端：构建 Chain 并传递物理坐标

**`sdpa_program_factory.cpp` 中的关键结构体：**

```cpp
// 第 43~57 行
struct CoreChainInfo {
    bool participates = false;     // 是否参与 chain
    bool is_injector = false;      // 是否是链头（负责从 DRAM 读 K/V）
    bool is_sink = false;          // 是否是链尾（只接收，不转发）
    CoreCoord prev_physical;       // 上游核心的 **物理坐标**
    CoreCoord next_physical;       // 下游核心的 **物理坐标**
    uint32_t mcast_num_dests = 0;  // multicast 目标数（单播时为 0）
    uint32_t mcast_sender_wait = 0;
};
```

**构建 chain 时获取物理坐标：**

```cpp
// 第 895~898 行 — 设置上游核心的物理坐标
const uint32_t prev_core_idx = segments[chain_order[pos - 1]].core_idx;
chain.prev_physical = core_work[prev_core_idx].physical_core;
//                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                    physical_core 在更早的地方通过
//                    device->worker_core_from_logical_core(logical_core) 获取
```

**通过 SetRuntimeArgs 传给 kernel：**

```cpp
// 第 1225~1231 行
reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.x));  // → kernel: prev_physical_x
reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.y));  // → kernel: prev_physical_y
reader_args.push_back(static_cast<uint32_t>(chain.next_physical.x));  // → kernel: next_physical_x
reader_args.push_back(static_cast<uint32_t>(chain.next_physical.y));  // → kernel: next_physical_y
reader_args.push_back(chain.mcast_num_dests);
reader_args.push_back(chain.mcast_sender_wait);
```

### 2.3 Kernel 端：单播 NoC 读写的具体实现

**`reader_interleaved.cpp` — 初始化 NoC 地址（第 190~193 行）：**

```c
// 单播模式（mcast_enabled == false）
sender_semaphore_noc_addr =
    get_noc_addr(prev_physical_x, prev_physical_y, sender_semaphore_addr);
//  ^^^^^^^^^^^^
//  构造一个 64-bit NoC 地址 = [y坐标 | x坐标 | L1地址]
//  prev_physical_x/y 是 Host 通过 runtime args 传来的物理坐标

receiver_semaphore_noc_addr =
    get_noc_addr(next_physical_x, next_physical_y, receiver_semaphore_addr);
```

**接收方 — 等待并接收数据（第 411~418 行）：**

```c
if (should_receive) {
    // 1. 在本地 CB 中预留空间
    cb_reserve_back(cb_k_in, k_chunk_tiles);
    cb_k_start_address = get_write_ptr(cb_k_in);  // 获取 CB 的 L1 写地址

    // 2. 标记自己"尚未收到数据"
    noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);

    // 3. 通知上游核心"我已准备好接收" — 远程原子加操作
    noc_semaphore_inc(sender_semaphore_noc_addr, 1);
    //                 ^^^^^^^^^^^^^^^^^^^^^^^^
    //                 这是上游核心的信号量的 NoC 地址
    //                 noc_semaphore_inc 通过 NoC 将远程核心 L1 中的信号量 +1

    // 4. 阻塞等待，直到上游核心写完数据后把 receiver_semaphore 设为 VALID
    noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);

    // 5. 数据已到达 CB，通知 compute 核心
    cb_push_back(cb_k_in, k_chunk_tiles);
}
```

**发送方 — 单播写数据（第 463~479 行）：**

```c
if (should_forward) {
    // 1. 等待下游核心准备好
    noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);  // 重置信号量

    // --- 单播模式 ---
    // 2. 构造目标 NoC 地址 = 下游核心的物理坐标 + 本地 CB 地址
    //    （关键：因为所有核心上同一 CB 的 L1 地址布局相同，
    //     所以直接用本地 cb_k_start_address 作为目标地址）
    uint64_t k_unicast_data_addr =
        get_noc_addr(next_physical_x, next_physical_y, cb_k_start_address);
    //  结果: [next_y | next_x | cb_k_start_address]

    // 3. 从本地 L1 写到远程核心的 L1
    noc_async_write(
        cb_k_start_address,          // 源：本地 L1 地址
        k_unicast_data_addr,         // 目标：远程核心 L1 的 NoC 地址
        k_chunk_tiles * k_tile_bytes // 传输大小
    );

    // 4. 等待写操作进入 NoC（不等完成）
    noc_async_writes_flushed();

    // 5. 通知下游核心数据已写完
    noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
    //                       ^本地值              ^远程核心信号量的 NoC 地址
    //   把本地的 VALID 值写到远程核心的 receiver_semaphore
}
```

**单播时序图：**

```
时间 →

Core 0 (injector)           Core 1 (中间)            Core 2 (sink)
─────────────────           ──────────────           ──────────────
从 DRAM 读 K chunk
  noc_async_read(...)
  noc_async_read_barrier()
                             cb_reserve_back()
                             set(receiver_sem, INVALID)
                             inc(Core0.sender_sem, 1) ←─── NoC 远程原子加
wait(sender_sem, 1) ◄────
set(sender_sem, 0)
noc_async_write(本地L1 → Core1.L1) ────→ 数据到达 Core1 CB
noc_async_writes_flushed()
set_remote(VALID → Core1.receiver_sem) ─→
                             wait(receiver_sem, VALID) ◄── 解除阻塞
                             cb_push_back()
                             用 K 做计算...
                             同时转发给 Core 2:
                             noc_async_write(本地L1 → Core2.L1) ────→
                             set_remote(VALID → Core2.receiver_sem) ─→
                                                      wait(receiver_sem, VALID) ◄──
                                                      用 K 做计算...
```

---

## 三、片上 NoC 多播：一次写入所有核心

### 3.1 多播的优势与前提条件

单播 chain 是串行的：injector → Core 1 → Core 2 → ... 延迟 = N × 单次延迟。
多播可以 **一次性** 把数据写给所有核心：injector → (Core 1, Core 2, ...) 延迟 ≈ 1 × 单次延迟。

**Host 端判断 multicast 资格（第 933~1055 行）的三个条件：**

```cpp
// 条件 1：chain 中所有核心必须在同一行（物理 Y 坐标相同）
//         因为多播地址是矩形区域，跨行会把无关核心包含在内
const uint32_t ref_y = core_work[chain_core_indices[0]].physical_core.y;
bool same_row = true;
for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
    if (core_work[chain_core_indices[ci]].physical_core.y != ref_y) {
        same_row = false;
        break;
    }
}

// 条件 2：矩形区域内不能有非 chain 的工作核心
//         否则多播会覆盖无关核心的 CB 和信号量
bool has_gap = false;  // 检查 [min_x, max_x] 范围内是否有非 chain 核心

// 条件 3：chain 中所有核心的 q_chunk_count 必须相同
//         否则某些核心提前结束不会发送信号量回复，导致 injector 死锁
```

### 3.2 Host 端：构造多播矩形

```cpp
// 第 1075~1098 行
// 计算矩形边界
uint32_t min_x = core_work[cand.core_indices[0]].physical_core.x;
uint32_t max_x = min_x;
for (size_t ci = 1; ci < cand.core_indices.size(); ++ci) {
    uint32_t x = core_work[cand.core_indices[ci]].physical_core.x;
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
}
const CoreCoord rect_start = CoreCoord{min_x, injector_y};
const CoreCoord rect_end = CoreCoord{max_x, injector_y};

// 如果 injector 在矩形内部（不在边界上），硬件会把它也算一个目标
const bool injector_inside_rect = (injector_x > min_x && injector_x < max_x);
const uint32_t mcast_num_dests = injector_inside_rect ? chain_size : num_receivers;

// 配置 injector：prev/next 不再是相邻核心，而是矩形的起止坐标
injector_chain.prev_physical = rect_start;  // 多播矩形左端
injector_chain.next_physical = rect_end;    // 多播矩形右端
injector_chain.mcast_num_dests = mcast_num_dests;
injector_chain.mcast_sender_wait = num_receivers;  // 等待 N-1 个接收方回复
```

### 3.3 Kernel 端：多播 NoC 操作

**初始化多播地址（第 176~189 行）：**

```c
if constexpr (mcast_enabled) {
    // 仅 injector 核心配置多播基地址
    if (is_injector) {
        // 构造多播基地址（addr=0，稍后用 OR 拼入实际 L1 偏移）
        // prev_physical 和 next_physical 现在是矩形的起止坐标
        mcast_base_noc_addr = get_noc_multicast_addr(
            prev_physical_x,   // rect 起始 x
            prev_physical_y,   // rect 起始 y
            next_physical_x,   // rect 结束 x
            next_physical_y,   // rect 结束 y
            0);                 // L1 地址=0

        // 64-bit 布局:
        // [y_start(6) | x_start(6) | y_end(6) | x_end(6) | 0...0(36)]

        // 信号量的多播地址 = base | 信号量的 L1 地址
        mcast_sem_noc_addr = mcast_base_noc_addr | receiver_semaphore_l1_addr;

        // 需要等待的接收方数量
        sender_wait_count = mcast_sender_wait;  // = chain_size - 1
    }
}
```

**多播写数据（第 466~474 行）：**

```c
if constexpr (mcast_enabled) {
    // 拼入 CB 的 L1 地址（因为 base 的低 36 位全 0，OR 就是拼接）
    uint64_t k_mcast_addr = mcast_base_noc_addr | cb_k_start_address;

    // 一次性写给矩形内所有核心
    noc_async_write_multicast(
        cb_k_start_address,          // 源：本地 L1
        k_mcast_addr,                // 目标：多播 NoC 地址
        k_chunk_tiles * k_tile_bytes,// 大小
        mcast_num_dests,             // 目标核心数
        true                          // ⚠️ linked=true
    );

    // linked=true 的含义：
    // NoC 硬件保证本操作和紧接的下一个操作背靠背执行，
    // 中间不会被其他 NoC 事务插入。
    // 这样"数据"和"信号量"对接收方是原子到达的。

    // 紧跟信号量多播（这是 linked 的 companion）
    noc_semaphore_set_multicast(
        valid_semaphore_addr,        // 源：本地信号量值（VALID）
        mcast_sem_noc_addr,          // 目标：多播信号量地址
        mcast_num_dests
    );
    // ⚠️ 这两行之间绝对不能插入 noc_async_read_barrier()，否则死锁！
}
```

**多播 vs 单播的数据流对比：**

```
单播 Chain（串行）：
  DRAM → Core 0 ──写──→ Core 1 ──写──→ Core 2 ──写──→ Core 3
                  ~5μs          ~5μs          ~5μs
  总延迟 ≈ 15μs

多播（并行）：
  DRAM → Core 0 ═══╦══写══╦══写══╦══写══╗
                    ▼      ▼      ▼      ▼
                  Core 1  Core 2  Core 3
  总延迟 ≈ 5μs (一次多播)
```

---

## 四、片间通信：Ring Distributed SDPA

### 4.1 Ring Distributed 的核心思想

当序列长度超过单芯片的处理能力时，将 Q 序列分配到多个芯片上。每个芯片拥有完整的 K/V，
但只处理 Q 的一部分。

**关键设计：不需要运行时跨芯片通信。**

```
全局序列 Q[0 : S]，ring_size=2，分成 2*2=4 个 chunk：

  chunk_0 | chunk_1 | chunk_2 | chunk_3
  ────────────────────────────────────→ 全局序列方向

  芯片 0 (ring_id=0): 处理 chunk_0 (低) + chunk_3 (高)
  芯片 1 (ring_id=1): 处理 chunk_1 (低) + chunk_2 (高)
```

**为什么低+高配对？** 因为 causal mask 的特性。chunk_0 只看到少量 K (因果约束)，
chunk_3 看到几乎全部 K，两者工作量互补，实现负载均衡。

### 4.2 Host 端：片间分配的代码实现

**`ring_distributed_sdpa_program_factory.cpp` 第 75~80 行：**

```cpp
// Q 的完整序列长度 = q_shape[2]
// 每设备的本地序列长度 = 完整序列 / (2 * ring_size)
const uint32_t Sq = q_shape[2] / (2 * ring_size);

// 该设备处理哪两个 chunk
const uint32_t chunk_1 = ring_id;                           // 低位 chunk
const uint32_t chunk_2 = (2 * ring_size) - ring_id - 1;    // 高位 chunk
```

**例子：ring_size=2，全局序列长度=8192**

| ring_id | chunk_1 | chunk_2 | 处理的 Q 范围 |
|---------|---------|---------|--------------|
| 0 | chunk_0 (token 0~2047) | chunk_3 (token 6144~8191) | 低+高 |
| 1 | chunk_1 (token 2048~4095) | chunk_2 (token 4096~6143) | 中间两段 |

### 4.3 两阶段执行

每个芯片运行 **两个 phase**，分别处理自己的低位和高位 chunk：

```cpp
// 第 131~138 行 — 计算两个 phase 的 Q 偏移
uint32_t chunked_q_chunk_offset_phase_1 = (chunk_1 * Sq) / q_chunk_size;
uint32_t chunked_q_chunk_offset_phase_2 = (chunk_2 * Sq) / q_chunk_size;

// 第 455~458 行 — 读写偏移
uint32_t read_offset_phase_1 = chunk_1 * Sqt;   // 从全局 Q tensor 的哪里读
uint32_t read_offset_phase_2 = chunk_2 * Sqt;
uint32_t write_offset_phase_1 = 0;               // 写到本地输出 tensor 的哪里
uint32_t write_offset_phase_2 = Sqt;             // 第二个 chunk 紧接在第一个后面
```

**在 reader kernel 中的两阶段循环（reader_interleaved.cpp 内部逻辑）：**

```c
for (uint32_t phase = 0; phase < num_phases; ++phase) {
    // phase 0: 处理 chunk_1（低位），Q 从 read_offset_phase_1 开始读
    // phase 1: 处理 chunk_2（高位），Q 从 read_offset_phase_2 开始读
    //
    // 两个 phase 都遍历完整的 K/V（因果模式下通过 mask 裁剪）
    for (q_chunk = ...) {
        读 Q chunk
        for (k_chunk = ...) {
            读 K chunk (从本地 DRAM，不涉及跨芯片通信！)
            读 V chunk
            → 交给 compute 核心做 QK^T, softmax, V
        }
    }
}
```

### 4.4 数据如何到达各芯片？

**关键事实：各芯片独立拥有完整的 Q, K, V 张量。**

在 Python 测试代码中：

```python
# test_sdpa_two_chips.py 第 323~325 行
# 同一份 Q/K/V 数据被发送到每个设备
tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
```

每个芯片拿到完整数据后，通过 `ring_id` 参数只处理自己负责的 Q 部分，但对 K/V 做完整遍历。

**Host 到 Device 的数据传输路径：**

```
Host CPU Memory
     │
     │ ttnn.from_torch(..., device=dev0)     ← PCIe DMA
     │ ttnn.from_torch(..., device=dev1)     ← Ethernet → PCIe DMA
     ▼
┌──────────┐    Ethernet    ┌──────────┐
│  Chip 0  │◄══════════════►│  Chip 1  │
│  DRAM    │   (3200Gbps)   │  DRAM    │
│  Q,K,V   │                │  Q,K,V   │
│  (完整)  │                │  (完整)  │
└──────────┘                └──────────┘
```

N300 上两个芯片通过 PCIe 卡上的 Ethernet 互联：
- Chip 0 直接连 PCIe，Host 通过 PCIe DMA 传输
- Chip 1 通过 Chip 0 的 Ethernet 中转，或由 UMD 驱动直接路由

### 4.5 输出重组（Host 端）

各芯片的输出只包含自己处理的 Q chunk 对应的结果，Host 需要重新排列：

```python
# test_sdpa_two_chips.py 第 38~54 行
def gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, global_seq_len):
    chunk_size = global_seq_len // (2 * ring_size)

    for device_id, device_output in enumerate(ring_outputs):
        first_chunk_id = device_id                           # chunk_1
        second_chunk_id = (2 * ring_size - 1) - device_id   # chunk_2

        # 设备输出的前半是 chunk_1 的结果，后半是 chunk_2 的结果
        first_chunk_output = device_output[:, :, :chunk_size, :]
        second_chunk_output = device_output[:, :, chunk_size:2*chunk_size, :]

        # 放回全局序列的正确位置
        final_output[:, :, first_chunk_id*chunk_size:(first_chunk_id+1)*chunk_size, :] = first_chunk_output
        final_output[:, :, second_chunk_id*chunk_size:(second_chunk_id+1)*chunk_size, :] = second_chunk_output
```

### 4.6 Ring Distributed 与片上多核的组合

片间和片上并行是 **正交的**：

```
Chip 0 (ring_id=0, 处理 chunk_0 + chunk_3)
  ├─ Core 0: batch=0, head=0~3, q_chunks=0~1
  ├─ Core 1: batch=0, head=0~3, q_chunks=2~3  ──KV chain──→ 接收 Core 0 转发的 K/V
  ├─ Core 2: batch=0, head=4~7, q_chunks=0~1
  └─ Core 3: batch=0, head=4~7, q_chunks=2~3

Chip 1 (ring_id=1, 处理 chunk_1 + chunk_2)
  ├─ Core 0: batch=0, head=0~3, q_chunks=0~1
  ├─ Core 1: batch=0, head=0~3, q_chunks=2~3
  ├─ Core 2: batch=0, head=4~7, q_chunks=0~1
  └─ Core 3: batch=0, head=4~7, q_chunks=2~3
```

片上的多核调度（KV chain + multicast）和片间的 Q 分配完全独立。
Ring Distributed 的 program factory 中 `mcast_enabled=0`（第 236 行），
说明当前版本的 ring distributed 不在片间使用 multicast，
而是每个芯片内部按普通 SDPA 的方式管理多核。

---

## 五、三种模式的完整对比

| 特性 | 单播 (Unicast) | 多播 (Multicast) | 片间 (Ring Distributed) |
|------|---------------|-----------------|----------------------|
| **数据传输** | Core A → Core B (一对一) | Core A → {Core B,C,D,...} (一对多) | Host → 各芯片的 DRAM |
| **传输介质** | 片上 NoC | 片上 NoC | PCIe + Ethernet |
| **传输内容** | K/V chunk (L1 → L1) | K/V chunk (L1 → 多个 L1) | 完整 Q/K/V tensor |
| **运行时通信** | 每个 K chunk 都需要 | 每个 K chunk 都需要 | **无** |
| **延迟** | O(chain_length) | O(1) | 仅初始数据传输 |
| **地址格式** | `[y\|x\|addr]` (48-bit) | `[y_s\|x_s\|y_e\|x_e\|addr]` (60-bit) | DRAM 地址 (32-bit) |
| **同步方式** | semaphore inc/wait | semaphore multicast | Host 端 gather |
| **使用场景** | 核心不在同一行 | 核心在同一行且无空洞 | 序列太长，单芯片放不下 |

---

## 六、关键代码文件索引

| 文件 | 作用 | 关键行号 |
|------|------|---------|
| `sdpa_program_factory.cpp` | Host 端 SDPA 配置 | 43~57: CoreChainInfo; 880~922: chain 构建; 933~1055: multicast 资格检查; 1058~1126: 多播矩形配置; 1196~1234: runtime args |
| `reader_interleaved.cpp` | Device 端 reader kernel | 80~84: semaphore ID; 176~194: NoC 地址初始化; 411~418: 接收方逻辑; 463~479: 发送方逻辑（单播+多播） |
| `ring_distributed_sdpa_program_factory.cpp` | 片间 Ring SDPA | 75~80: chunk 分配; 131~138: phase 偏移; 202~236: compile args（mcast_enabled=0） |
| `dataflow_api_addrgen.h` | NoC 地址构造 | 174~187: `get_noc_multicast_addr`; 203~206: `get_noc_addr` |
| `noc_parameters.h` | NoC 地址宏和常量 | 256~264: `NOC_XY_ADDR`, `NOC_MULTICAST_ADDR` |

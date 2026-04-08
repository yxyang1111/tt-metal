# Tenstorrent 片上网络 (NoC) 与多播 (Multicast) 详解

本文档详细解释 Wormhole B0 芯片上 **Network-on-Chip (NoC)** 的工作原理、地址编码方式、API 用法，
并以 SDPA 内核中的实际代码为例说明单播 (unicast)、多播 (multicast) 和核间通信的完整流程。

---

## 目录

1. [NoC 架构概览](#1-noc-架构概览)
2. [坐标系统与拓扑](#2-坐标系统与拓扑)
3. [NoC 地址编码](#3-noc-地址编码)
4. [核心 API 函数](#4-核心-api-函数)
5. [信号量同步机制](#5-信号量同步机制)
6. [Circular Buffer 与 NoC 协作](#6-circular-buffer-与-noc-协作)
7. [Host 端配置流程](#7-host-端配置流程)
8. [实战：SDPA 中的 NoC 使用](#8-实战sdpa-中的-noc-使用)
9. [常见错误与注意事项](#9-常见错误与注意事项)

---

## 1. NoC 架构概览

Wormhole B0 芯片内置 **两条独立的 NoC 网络**（NOC0 和 NOC1），每个核心通过 NoC 路由器
与所有其他核心、DRAM bank、PCIe 端点相连。NoC 是所有数据移动的基础设施：

```
┌─────────────────────────────────────────────────┐
│                  Wormhole B0 芯片                 │
│                                                   │
│   ┌──────┐  ┌──────┐  ┌──────┐  ... ┌──────┐    │
│   │Core  │──│Core  │──│Core  │──────│Core  │    │
│   │(0,0) │  │(1,0) │  │(2,0) │      │(7,0) │    │
│   └──┬───┘  └──┬───┘  └──┬───┘      └──┬───┘    │
│      │         │         │              │        │
│   ┌──┴───┐  ┌──┴───┐  ┌──┴───┐      ┌──┴───┐    │
│   │Core  │──│Core  │──│Core  │──────│Core  │    │
│   │(0,1) │  │(1,1) │  │(2,1) │      │(7,1) │    │
│   └──┬───┘  └──┬───┘  └──┬───┘      └──┴───┘    │
│      │         │         │              ...      │
│      ⋮         ⋮         ⋮                       │
│                                                   │
│   ═══════════ DRAM Banks ═══════════════         │
│   ═══════════ PCIe / Ethernet ══════════         │
└─────────────────────────────────────────────────┘
```

**关键参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| NoC 数量 | 2 (NOC0, NOC1) | 双 NoC 可并行传输 |
| NoC Payload 宽度 | 256 bit (32 字节) | 单拍有效数据 |
| 单包最大传输 | 8 KB (256 × 32B) | `NOC_MAX_BURST_SIZE` |
| L1 读对齐 | 16 字节 | `NOC_L1_READ_ALIGNMENT_BYTES` |
| L1 写对齐 | 16 字节 | `NOC_L1_WRITE_ALIGNMENT_BYTES` |
| DRAM 读对齐 | 32 字节 | `NOC_DRAM_READ_ALIGNMENT_BYTES` |
| DRAM 写对齐 | 16 字节 | `NOC_DRAM_WRITE_ALIGNMENT_BYTES` |
| NoC 网格大小 | 10 × 12 | Wormhole 物理网格 |
| 地址位宽 | 36 bit 本地 + 6+6 bit 坐标 | 共 48 bit |

---

## 2. 坐标系统与拓扑

### 2.1 三层坐标

Wormhole 使用三层坐标系统：

| 坐标类型 | 用途 | 范围（Tensix） |
|----------|------|----------------|
| **Logical** | Host 编程用，连续编号 | x: 0~7, y: 0~7（取决于 harvest） |
| **Virtual/Translated** | 硬件翻译后，Kernel/NoC 实际使用 | 从 (18, 18) 开始偏移 |
| **Physical (NOC0)** | 硬件原始坐标 | x: 0~9, y: 0~11 |

**坐标转换链：**

```
Logical (0,0)  ──device->worker_core_from_logical_core()──>  Virtual (18,18)
                                                                    │
                                                        用于 NoC 地址构造
```

### 2.2 NOC0 与 NOC1 的坐标关系

两条 NoC 的坐标方向相反：

```
NOC0:  坐标原点在左上角        NOC1:  坐标原点在右下角
       (0,0) → (9,0)                 (9,11) → (0,11)
         ↓                               ↓
       (0,11)                          (9,0)
```

转换规则：
- **NOC0**: `physical(x, y) = virtual(x, y)` （直通）
- **NOC1**: `physical(x, y) = (noc_size_x - 1 - x, noc_size_y - 1 - y)` （翻转）

```c
// noc_nonblocking_api.h
#define NOC_0_X(noc_index, noc_size_x, x) x
#define NOC_0_Y(noc_index, noc_size_y, y) y
// NOC1 翻转
#define NOC_0_X_PHYS_COORD(noc_index, noc_size_x, x) \
    (noc_index == 0 ? (x) : (noc_size_x - 1 - (x)))
#define NOC_0_Y_PHYS_COORD(noc_index, noc_size_y, y) \
    (noc_index == 0 ? (y) : (noc_size_y - 1 - (y)))
```

### 2.3 Harvesting 对坐标的影响

Wormhole 支持行级 harvesting（屏蔽有缺陷的 Tensix 行）：
- Logical 坐标始终从 0 连续编号，但总行数减少
- Harvested 行的 Physical 坐标被跳过
- Virtual/Translated 坐标重新映射到可用行

---

## 3. NoC 地址编码

### 3.1 单播地址（64 bit）

单播地址把 **目标核心坐标** 和 **本地 L1 地址** 编码在一个 64 bit 整数中：

```
 63                48 47      42 41      36 35                              0
┌────────────────────┬─────────┬──────────┬─────────────────────────────────┐
│     (unused)       │  noc_y  │  noc_x   │        L1 local address        │
│                    │ (6 bit) │ (6 bit)  │           (36 bit)             │
└────────────────────┴─────────┴──────────┴─────────────────────────────────┘
```

对应的宏：

```c
// noc_parameters.h
#define NOC_ADDR_LOCAL_BITS   36
#define NOC_ADDR_NODE_ID_BITS  6

#define NOC_XY_ADDR(x, y, addr)              \
    ((uint64_t)(y) << (36 + 6))  |           \
    ((uint64_t)(x) << 36)        |           \
    ((uint64_t)(addr))
```

### 3.2 多播地址（64 bit）

多播地址编码一个 **矩形区域** 的起始和结束坐标：

```
 63       54 53      48 47      42 41      36 35                              0
┌──────────┬──────────┬──────────┬──────────┬─────────────────────────────────┐
│ y_start  │ x_start  │  y_end   │  x_end   │        L1 local address        │
│ (6 bit)  │ (6 bit)  │ (6 bit)  │ (6 bit)  │           (36 bit)             │
└──────────┴──────────┴──────────┴──────────┴─────────────────────────────────┘
```

对应的宏：

```c
// noc_parameters.h
#define NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, addr)             \
    ((uint64_t)(x_start) << (36 + 2*6))  |  /* bit 48~53 */                 \
    ((uint64_t)(y_start) << (36 + 3*6))  |  /* bit 54~59 */                 \
    ((uint64_t)(x_end)   << 36)          |  /* bit 36~41 */                 \
    ((uint64_t)(y_end)   << (36 + 6))    |  /* bit 42~47 */                 \
    ((uint64_t)(addr))                       /* bit 0~35  */
```

**多播目标区域是一个矩形**，所有坐标在 `(x_start, y_start)` 到 `(x_end, y_end)` 之间（含边界）的核心
都会收到数据。

```
     x_start         x_end
       ↓               ↓
      ┌───┬───┬───┬───┐  ← y_start
      │ ✓ │ ✓ │ ✓ │ ✓ │
      ├───┼───┼───┼───┤
      │ ✓ │ ✓ │ ✓ │ ✓ │
      ├───┼───┼───┼───┤
      │ ✓ │ ✓ │ ✓ │ ✓ │  ← y_end
      └───┴───┴───┴───┘

   ✓ = 都会收到多播数据
```

### 3.3 地址拼接技巧

实际代码中常用 **先构造 base 地址（addr=0），再用 OR 拼入 L1 偏移** 的技巧：

```c
// 先构造不含 L1 地址的 multicast base
uint64_t mcast_base = get_noc_multicast_addr(x_start, y_start, x_end, y_end, 0);

// 需要时 OR 上实际的 L1 地址
uint64_t k_mcast_addr = mcast_base | cb_k_start_address;
uint64_t sem_mcast_addr = mcast_base | semaphore_l1_addr;
```

因为 L1 地址占低 36 bit，坐标占高位，两者不重叠，所以 OR 操作等价于加法。

---

## 4. 核心 API 函数

所有 NoC API 定义在 `tt_metal/hw/inc/api/dataflow/dataflow_api.h`，在 dataflow kernel 中直接调用。

### 4.1 地址构造

```c
// 单播地址：指定核心 (noc_x, noc_y) 上的 L1 地址 addr
uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr);

// 多播地址：矩形区域 (x_start,y_start) 到 (x_end,y_end) 上的 L1 地址 addr
uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start, uint32_t noc_y_start,
    uint32_t noc_x_end,   uint32_t noc_y_end,
    uint32_t addr);

// 本地核心的 L1 地址（不含坐标）
uint64_t get_noc_addr(uint32_t addr);  // 等价于 get_noc_addr(my_x, my_y, addr)
```

### 4.2 异步读（DRAM/远程 L1 → 本地 L1）

```c
// 从 NoC 地址 src_noc_addr 读取 size 字节到本地 L1 的 dst_local_l1_addr
void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size);

// 读取单个 tile（内部通过 TensorAccessor 计算 NoC 地址）
void noc_async_read_tile(uint32_t tile_id, const auto& reader, uint32_t dst_l1_addr);

// 等待所有读操作完成
void noc_async_read_barrier();
```

### 4.3 异步写 — 单播（本地 L1 → 远程 L1/DRAM）

```c
// 将本地 L1 的 src_local_l1_addr 写 size 字节到远程 NoC 地址 dst_noc_addr
void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size);

// 写 tile
void noc_async_write_tile(uint32_t tile_id, const auto& writer, uint32_t src_l1_addr);

// 等待写完成（不阻塞后续写发射）
void noc_async_writes_flushed();

// 等待所有写操作完成（完全阻塞）
void noc_async_write_barrier();
```

### 4.4 异步写 — 多播（本地 L1 → 多个远程核心的 L1）

```c
// 将本地 L1 数据多播到矩形区域内的所有核心
void noc_async_write_multicast(
    uint32_t src_local_l1_addr,       // 源：本地 L1 地址
    uint64_t dst_noc_addr_multicast,  // 目标：多播 NoC 地址
    uint32_t size,                    // 传输字节数
    uint32_t num_dests,               // 目标核心数量
    bool linked = false               // 是否与下一个操作链接
);

// 多播信号量（通常紧跟 multicast 写之后）
void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t num_dests,
    bool linked = false
);
```

**`linked` 参数**：当 `linked=true` 时，NoC 硬件保证此操作和紧接的下一个操作
**原子地** 背靠背执行（不会被其他 NoC 事务插入）。
这对"先写数据、再写信号量"的模式至关重要。

---

## 5. 信号量同步机制

核间通过 NoC 通信时，需要信号量来协调"数据已就绪"和"可以开始写入"。

### 5.1 Host 端创建信号量

```cpp
// Host 端 (C++)
uint32_t sem_id = CreateSemaphore(program, core_range, initial_value);
```

在 `core_range` 内的每个核心上分配一个 4 字节信号量，初始值为 `initial_value`。

### 5.2 Kernel 端获取地址

```c
// Device 端 (Kernel)
uint32_t sem_addr = get_semaphore(sem_id);  // 获取该信号量在本核心 L1 中的地址
volatile tt_l1_ptr uint32_t* sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
```

### 5.3 信号量操作

| 函数 | 作用 | 范围 |
|------|------|------|
| `noc_semaphore_set(ptr, val)` | 将本地信号量设为 `val` | 本地 |
| `noc_semaphore_wait(ptr, val)` | 阻塞直到信号量等于 `val` | 本地 |
| `noc_semaphore_inc(noc_addr, val)` | 远程信号量原子加 `val` | 远程单播 |
| `noc_semaphore_set_remote(l1_addr, noc_addr)` | 把本地值写到远程信号量 | 远程单播 |
| `noc_semaphore_set_multicast(l1_addr, mcast_addr, n)` | 把本地值多播到多个远程信号量 | 远程多播 |

### 5.4 核间握手协议示例

```
发送方 (Core A)                         接收方 (Core B)
─────────────────                       ─────────────────
  等待接收方就绪:                         初始化：
  noc_semaphore_wait(                     *valid_sem = VALID   (表示空间已就绪)
    sender_sem, 1)
                                          通知发送方可以写：
                                          noc_semaphore_inc(
                                            CoreA.sender_sem, 1)
  清除信号：
  noc_semaphore_set(
    sender_sem, 0)                        等待数据到达：
                                          noc_semaphore_set(receiver_sem, INVALID)
  写数据到 Core B：                        ... (等待中)
  noc_async_write(
    local_l1, CoreB.l1, size)

  通知数据已写完：
  noc_semaphore_set_remote(
    valid_sem, CoreB.receiver_sem)         noc_semaphore_wait(receiver_sem, VALID)
                                           → 数据可用！
```

---

## 6. Circular Buffer 与 NoC 协作

Circular Buffer (CB) 是核心的 L1 数据 FIFO，与 NoC 配合完成数据搬运。

### 6.1 典型数据流：DRAM → L1（读）

```c
// 1. 在 CB 中预留空间
cb_reserve_back(cb_id, num_tiles);

// 2. 获取 CB 的写指针（L1 地址）
uint32_t write_ptr = get_write_ptr(cb_id);

// 3. 通过 NoC 从 DRAM 读数据到 CB 的 L1 空间
noc_async_read(dram_noc_addr, write_ptr, size);
noc_async_read_barrier();  // 等待读完成

// 4. 通知 compute 核心数据已就绪
cb_push_back(cb_id, num_tiles);
```

### 6.2 典型数据流：L1 → DRAM（写）

```c
// 1. 等待 compute 核心产出数据
cb_wait_front(cb_id, num_tiles);

// 2. 获取 CB 的读指针
uint32_t read_ptr = get_read_ptr(cb_id);

// 3. 通过 NoC 写到 DRAM
noc_async_write(read_ptr, dram_noc_addr, size);
noc_async_write_barrier();

// 4. 释放 CB 空间
cb_pop_front(cb_id, num_tiles);
```

### 6.3 核间转发：L1 → 远程 L1

```c
// 发送方：从自己的 CB 写到目标核心的 CB（L1 地址相同）
uint32_t cb_start_address = get_write_ptr(cb_k_in);
uint64_t remote_addr = get_noc_addr(target_x, target_y, cb_start_address);
noc_async_write(cb_start_address, remote_addr, size);
```

**核心要点**：因为所有核心上同一个 CB 的 L1 布局相同，所以发送方可以直接用自己的 CB 地址作为
目标核心的 CB 地址。

---

## 7. Host 端配置流程

从 Host 到 Device 的完整 NoC 配置链：

```
Host 程序
    │
    ├─ 1. CreateProgram()
    │
    ├─ 2. 计算核心网格
    │      CoreRange core_grid({0,0}, {grid_x-1, grid_y-1});
    │
    ├─ 3. 创建 Circular Buffer
    │      CreateCircularBuffer(program, core_grid, cb_config);
    │
    ├─ 4. 创建 Semaphore
    │      sender_sem = CreateSemaphore(program, core_grid, INVALID);
    │      receiver_sem = CreateSemaphore(program, core_grid, INVALID);
    │
    ├─ 5. 创建 Kernel
    │      CreateKernel(program, "reader.cpp", core_grid, DataMovementConfig);
    │      CreateKernel(program, "compute.cpp", core_grid, ComputeConfig);
    │      CreateKernel(program, "writer.cpp", core_grid, DataMovementConfig);
    │
    ├─ 6. 逐核心设置 RuntimeArgs
    │      for each core:
    │          physical = device->worker_core_from_logical_core(logical);
    │          SetRuntimeArgs(program, kernel_id, core, {
    │              dram_addr,
    │              physical.x, physical.y,  // 目标核心的 NoC 坐标
    │              mcast_num_dests,
    │              ...
    │          });
    │
    └─ 7. EnqueueProgram(cq, program)
           → 将 runtime args DMA 到各核心 L1
           → 启动 kernel 执行
```

### 7.1 坐标转换

Host 使用 Logical 坐标，Kernel 需要 Virtual (Translated) 坐标：

```cpp
// Host 端
CoreCoord logical_core(x, y);
CoreCoord physical_core = device->worker_core_from_logical_core(logical_core);

// 传给 kernel
runtime_args.push_back(physical_core.x);  // NoC x 坐标
runtime_args.push_back(physical_core.y);  // NoC y 坐标
```

### 7.2 Multicast 编码

Host 端也可以预计算 multicast 编码：

```cpp
// Host 端
uint32_t mcast_encoding = device->get_noc_multicast_encoding(noc_index, core_range);
```

---

## 8. 实战：SDPA 中的 NoC 使用

以 `reader_interleaved.cpp`（SDPA reader kernel）为例，展示三种 NoC 模式。

### 8.1 模式一：DRAM → L1（读取 Q/K/V）

最基础的模式——从 DRAM 读取 tile 到 Circular Buffer：

```c
// dataflow_common.hpp — read_chunk_with_padding
for (uint32_t col = 0; col < src_cols; ++col) {
    noc_async_read_tile(start_tile_id, reader, write_ptr);
    //                  ↑ tile ID    ↑ TensorAccessor（内部计算 DRAM bank 的 NoC 地址）
    start_tile_id += 1;
    write_ptr += inner_ptr_stride;
}
noc_async_read_barrier();
```

`TensorAccessor` 封装了 interleaved tensor 的地址计算逻辑：
- 给定 tile_id，算出它在哪个 DRAM bank
- 返回 `NOC_XY_ADDR(bank_noc_x, bank_noc_y, bank_offset)` 格式的 64-bit 地址

### 8.2 模式二：单播 L1 → L1（KV Chain 转发）

当多个核心处理同一组 K/V 时，由第一个核心读 DRAM，然后通过 NoC 转发给链上的下一个核心：

```c
// reader_interleaved.cpp — 单播转发
if (should_forward) {
    // 等待下一核心准备好接收
    noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);

    // 构造目标核心的 NoC 地址
    uint64_t k_unicast_data_addr =
        get_noc_addr(next_physical_x, next_physical_y, cb_k_start_address);

    // 从本地 L1 写到目标核心的 L1
    noc_async_write(cb_k_start_address, k_unicast_data_addr, k_chunk_tiles * k_tile_bytes);

    // 刷新写通道，通知接收方
    noc_async_writes_flushed();
    noc_semaphore_set_remote(valid_semaphore_addr, receiver_semaphore_noc_addr);
}
```

**接收方的对应代码：**

```c
// 接收转发的 K chunk
cb_reserve_back(cb_k_in, k_chunk_tiles);
cb_k_start_address = get_write_ptr(cb_k_in);

// 设置"未就绪"，通知发送方可以写
noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
noc_semaphore_inc(sender_semaphore_noc_addr, 1);

// 等待数据到达
noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
cb_push_back(cb_k_in, k_chunk_tiles);
```

### 8.3 模式三：多播 L1 → 多个 L1（Multicast KV 分发）

当 chain 中有多个接收者时，使用多播一次性发送给所有核心：

```c
// reader_interleaved.cpp — 初始化阶段

// 构造多播基地址（addr=0，后面用 OR 拼入实际 L1 地址）
mcast_base_noc_addr = get_noc_multicast_addr(
    prev_physical_x,   // 接收者矩形区域的起始 x
    prev_physical_y,   // 接收者矩形区域的起始 y
    next_physical_x,   // 接收者矩形区域的结束 x
    next_physical_y,   // 接收者矩形区域的结束 y
    0);                 // L1 地址=0，稍后 OR 拼入

// 信号量多播地址
mcast_sem_noc_addr = mcast_base_noc_addr | receiver_semaphore_l1_addr;
```

```c
// reader_interleaved.cpp — 转发阶段

if (should_forward) {
    noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
    noc_semaphore_set(sender_semaphore_addr_ptr, 0);

    // 拼入 CB 的 L1 地址
    uint64_t k_mcast_addr = mcast_base_noc_addr | cb_k_start_address;

    // 多播写数据（linked=true：与后续信号量操作原子链接）
    noc_async_write_multicast(
        cb_k_start_address,         // 源：本地 L1
        k_mcast_addr,               // 目标：多播地址
        k_chunk_tiles * k_tile_bytes, // 大小
        mcast_num_dests,            // 目标核心数
        true);                       // linked=true ⚠️

    // 紧跟多播信号量（必须在 linked write 之后立即发射！）
    noc_semaphore_set_multicast(
        valid_semaphore_addr,
        mcast_sem_noc_addr,
        mcast_num_dests);
}
```

### 8.4 Host 端如何配置 Multicast

在 `sdpa_program_factory.cpp` 中，Host 端为每个核心计算 chain 信息：

```cpp
struct CoreChainInfo {
    bool is_injector;          // 是否为链头（负责从 DRAM 读取并转发）
    bool is_sink;              // 是否为链尾（只接收，不转发）
    CoreCoord prev_physical;   // 上游核心物理坐标（或 mcast 起始核心）
    CoreCoord next_physical;   // 下游核心物理坐标（或 mcast 结束核心）
    uint32_t mcast_num_dests;  // multicast 目标数量
    uint32_t mcast_sender_wait; // 发送方需等待的信号量计数
};
```

这些参数通过 `SetRuntimeArgs` 传入 Kernel：

```cpp
// Host 端
reader_args.push_back(chain.prev_physical.x);    // → kernel 中的 prev_physical_x
reader_args.push_back(chain.prev_physical.y);    // → kernel 中的 prev_physical_y
reader_args.push_back(chain.next_physical.x);    // → kernel 中的 next_physical_x
reader_args.push_back(chain.next_physical.y);    // → kernel 中的 next_physical_y
reader_args.push_back(chain.mcast_num_dests);    // → kernel 中的 mcast_num_dests
reader_args.push_back(chain.mcast_sender_wait);  // → kernel 中的 sender_wait_count
```

### 8.5 完整数据流图

```
                     DRAM
                       │
                 noc_async_read
                       │
                       ▼
  ┌─────────────────────────────────────────┐
  │  Core 0 (injector)                       │
  │  ┌──────────┐                            │
  │  │  CB K/V  │── noc_async_write_multicast ──┐
  │  └──────────┘                            │  │
  │  noc_semaphore_set_multicast ────────────│──│──┐
  └──────────────────────────────────────────┘  │  │
                                                │  │
  ┌─────────────────────────────────────────┐   │  │
  │  Core 1 (receiver)                       │   │  │
  │  ┌──────────┐                            │   │  │
  │  │  CB K/V  │ ← 数据通过多播直接写入     │◀──┘  │
  │  └──────────┘                            │     │
  │  noc_semaphore_wait(VALID) ◀─────────────│─────┘
  └──────────────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │  Core 2 (receiver)                       │
  │  ┌──────────┐                            │
  │  │  CB K/V  │ ← 同时收到相同数据         │◀──┘
  │  └──────────┘                            │
  │  noc_semaphore_wait(VALID) ◀─────────────│
  └──────────────────────────────────────────┘
```

---

## 9. 常见错误与注意事项

### 9.1 对齐要求

| 操作类型 | 最小对齐 |
|----------|---------|
| L1 读 | 16 字节 |
| L1 写 | 16 字节 |
| DRAM 读 | 32 字节 |
| DRAM 写 | 16 字节 |

违反对齐会导致 **硬件挂起**（无错误提示，只是永远不完成）。

### 9.2 Linked Write 规则

`noc_async_write_multicast(..., linked=true)` 的 **严格规则**：

1. linked write 后 **必须** 紧跟一个 companion 操作（通常是 `noc_semaphore_set_multicast`）
2. 两者之间 **不能** 插入 `noc_async_read_barrier()`，否则会 **死锁**
3. Linked write 的 companion 必须发往 **相同的目标**

```c
// ✅ 正确
noc_async_write_multicast(src, dst, size, n, true);   // linked=true
noc_semaphore_set_multicast(sem, mcast_sem, n);        // companion

// ❌ 错误 — 死锁！
noc_async_write_multicast(src, dst, size, n, true);   // linked=true
noc_async_read_barrier();                               // 💀 阻塞了 linked write
noc_semaphore_set_multicast(sem, mcast_sem, n);        // 永远不会执行
```

### 9.3 Multicast 不含发送者自身

`noc_async_write_multicast` 的 `num_dests` 不包含发送者自身（如果发送者不在矩形内），
但如果发送者在矩形内，它也会收到数据，且 `num_dests` 应包含发送者。

### 9.4 大传输分块

单次 NoC 传输最大 **8KB**。超过 8KB 的传输需要分多次发射，或使用 tile 级 API（内部自动分块）。

### 9.5 NOC0 vs NOC1 的多播方向

- **NOC0**：多播从 `(start_x, start_y)` 到 `(end_x, end_y)`，其中 start <= end
- **NOC1**：多播方向相反，需要交换 start 和 end

Host 端的 `get_noc_multicast_encoding` 已经自动处理了这个转换：

```cpp
// device.cpp
if (noc_index == 0) {
    return hal.noc_multicast_encoding(start.x, start.y, end.x, end.y);
} else {
    return hal.noc_multicast_encoding(end.x, end.y, start.x, start.y);  // 交换！
}
```

---

## 附录：关键文件索引

| 文件 | 内容 |
|------|------|
| `hw/inc/api/dataflow/dataflow_api.h` | 高层 Dataflow API（kernel 中 `#include` 这个） |
| `hw/inc/internal/dataflow/dataflow_api_addrgen.h` | 地址生成函数 `get_noc_addr`, `get_noc_multicast_addr` |
| `hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` | NoC 常量、地址宏定义 |
| `hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h` | 非阻塞 NoC 底层 API |
| `hw/inc/internal/tt-1xx/wormhole/noc/noc.h` | 阻塞式 NoC 底层 API |
| `ttnn/.../sdpa/device/kernels/dataflow/reader_interleaved.cpp` | SDPA reader：完整的多播/单播/DRAM 读 |
| `ttnn/.../sdpa/device/sdpa_program_factory.cpp` | SDPA Host 端：核心网格、chain 配置、runtime args |
| `impl/device/device.cpp` | `get_noc_multicast_encoding`, 坐标转换 |

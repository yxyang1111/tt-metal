# TT-Metal SDPA (Scaled Dot-Product Attention) 实现详解

## 1. 概述

TT-Metal 中的 SDPA 实现是一套完整的 FlashAttention 算法在 Tenstorrent Wormhole/Blackhole 架构上的适配，基于 FlashAttention-2 的核心思想（Q 并行 + 在线 softmax on KV chunks），并结合了 FlashAttention-3 的部分优化（流水线数据搬运、异步执行）。相比基线实现可达 **20x 加速**。

### 1.1 代码组织结构

```
ttnn/cpp/ttnn/operations/transformer/
├── sdpa_config.hpp                    # 共享配置结构体 SDPAProgramConfig
├── sdpa/                              # 主 SDPA 实现 (Prefill)
│   ├── sdpa.hpp / sdpa.cpp            # 顶层 API
│   ├── sdpa_nanobind.cpp              # Python 绑定
│   └── device/
│       ├── sdpa_device_operation.hpp/cpp        # 设备操作定义与验证
│       ├── sdpa_device_operation_types.hpp      # 参数/输入类型定义
│       ├── sdpa_program_factory.hpp/cpp         # Program 构建工厂
│       ├── sdpa_perf_model.hpp/cpp              # 性能模型
│       ├── sdpa_subblock_utils.hpp              # Subblock 大小计算工具
│       ├── joint_sdpa_*                         # Joint SDPA 变体
│       ├── ring_joint_sdpa_*                    # Ring Joint SDPA 变体
│       ├── ring_distributed_sdpa_*              # Ring Distributed SDPA 变体
│       ├── ring_fusion.hpp/cpp                  # Ring Fusion 工具
│       └── kernels/
│           ├── compute/
│           │   ├── sdpa.cpp                     # 标准 SDPA 计算核
│           │   ├── compute_common.hpp           # 核心计算原语 (matmul, softmax, etc.)
│           │   ├── compute_streaming.hpp        # Streaming 计算路径
│           │   ├── joint_sdpa.cpp               # Joint SDPA 计算核
│           │   └── ring_joint_sdpa.cpp          # Ring Joint SDPA 计算核
│           └── dataflow/
│               ├── reader_interleaved.cpp       # Reader 数据搬运核
│               ├── writer_interleaved.cpp       # Writer 数据搬运核
│               ├── dataflow_common.hpp          # 共享数据流工具
│               └── ring_*.cpp                   # Ring 变体数据搬运核
├── sdpa_decode/                       # Decode 阶段 SDPA (Flash Decode)
│   ├── sdpa_decode.hpp/cpp
│   └── device/
│       ├── sdpa_decode_device_operation.hpp/cpp
│       ├── sdpa_decode_program_factory.hpp/cpp
│       └── kernels/
│           ├── compute/sdpa_flash_decode.cpp
│           └── dataflow/reader_decode_all.cpp, writer_decode_all.cpp
└── sdpa_windowed/                     # 窗口化 SDPA
    ├── sdpa_windowed.hpp/cpp
    └── device/
        ├── sdpa_windowed_device_operation.hpp/cpp
        ├── sdpa_windowed_program_factory.hpp/cpp
        └── kernels/
            ├── compute/sdpa_windowed.cpp
            └── dataflow/reader_windowed.cpp, writer_windowed.cpp
```

---

## 2. 核心配置与数据类型

### 2.1 SDPAProgramConfig

```cpp
struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;  // 计算核网格大小
    std::optional<CoreRangeSet> sub_core_grids; // 子核范围
    std::size_t q_chunk_size;                   // Q 分块大小（序列维度）
    std::size_t k_chunk_size;                   // K 分块大小（序列维度）
    std::optional<bool> exp_approx_mode;        // 指数近似模式
    uint32_t max_cores_per_head_batch = 16;     // 每个 head-batch 最大核数
};
```

- `q_chunk_size` 和 `k_chunk_size` 必须是 `TILE_WIDTH (32)` 的倍数
- 这两个超参数决定了 FlashAttention 内外循环中 Q/K/V 块的大小，影响 L1 使用量和性能

### 2.2 SDPAParams — 操作参数

```cpp
struct SDPAParams {
    std::optional<float> scale;                      // 缩放因子, 默认 1/sqrt(d_head)
    MemoryConfig output_mem_config;                  // 输出内存配置
    std::optional<SDPAProgramConfig> program_config;  // 程序配置
    bool is_causal = false;                          // 因果注意力
    std::optional<int64_t> chunk_start_idx;          // 分块起始索引（标量）
    std::optional<Tensor> chunk_start_idx_tensor;    // 分块起始索引（设备张量）
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_mla = false;                            // Multi-Latent Attention 模式
    std::optional<uint32_t> head_dim_v;              // V 的 head 维度
    std::optional<uint32_t> sliding_window_size;     // 滑动窗口大小
};
```

### 2.3 SDPAInputs — 输入张量

```cpp
struct SDPAInputs {
    Tensor q;                              // [B, NQH, Sq, DH] 查询
    Tensor k;                              // [B, NKH, Sk, DH] 键
    std::optional<Tensor> v;               // [B, NKH, Sk, DH] 值（MLA 模式下可空）
    std::optional<Tensor> attn_mask;       // 注意力掩码
    std::optional<Tensor> page_table;      // 分页 KV 缓存页表
    std::optional<Tensor> attention_sink;  // 注意力汇（StreamingLLM）
};
```

### 2.4 支持的数据类型

- **BFloat16**: 标准精度
- **BFP8_B** (Block FP8): Tenstorrent 特有格式，16 个数据共享一个指数，每个数据 7 位尾数。尺寸约为 BF16 的一半
- **BFP4_B**: 更低精度的块浮点格式

---

## 3. SDPA API 变体详解

### 3.1 标准 SDPA（Prefill）

```cpp
Tensor scaled_dot_product_attention(
    Q, K, V,
    attn_mask,         // 可选注意力掩码
    is_causal,         // 因果模式
    scale,             // 缩放因子
    sliding_window_size, // 滑动窗口
    memory_config,
    program_config,
    compute_kernel_config,
    attention_sink     // 注意力汇
);
```

**功能**：标准 FlashAttention Prefill 路径，支持以下模式：
- **因果注意力**（`is_causal=true`）：自动生成下三角掩码
- **自定义掩码**（`attn_mask` 提供）：支持 batch/head 广播
- **滑动窗口注意力**：通过 `sliding_window_size` 限制注意力范围
- **注意力汇**（Attention Sink）：StreamingLLM 风格，部分注意力概率被 sink 吸收

### 3.2 分块 SDPA（Chunked Prefill）

```cpp
// 标量版本
Tensor chunked_scaled_dot_product_attention(
    Q, K, V, page_table, chunk_start_idx, ...);

// 设备张量版本（支持 Trace）
Tensor chunked_scaled_dot_product_attention(
    Q, K, V, page_table, chunk_start_idx_tensor, ...);
```

**功能**：用于 **分页 KV 缓存** 场景，每次只处理一个 Q 块对应的 SDPA 计算：
- `page_table`: 页表张量 `[B, num_pages]`, INT32 类型，ROW_MAJOR 布局
- `chunk_start_idx`: Q 块在全局序列中的起始位置，必须是 `q_chunk_size` 的整数倍
- 设备张量版本允许在 Trace 模式下动态改变起始索引

### 3.3 Joint SDPA（多模态注意力）

```cpp
std::tuple<Tensor, Tensor> joint_scaled_dot_product_attention(
    Q, K, V,                    // 主输入
    joint_Q, joint_K, joint_V,  // 联合输入（如图像 token）
    joint_strategy,             // 连接策略
    program_config, scale, compute_kernel_config
);
```

**功能**：用于多模态模型（如 Stable Diffusion），将两组 KV 序列拼接后做联合注意力计算。

### 3.4 Ring Joint SDPA（多设备分布式注意力）

```cpp
std::tuple<Tensor, Tensor, Tensor> ring_joint_scaled_dot_product_attention(
    Q, K, V,
    joint_Q, joint_K, joint_V,
    persistent_output_buffer_k/v,  // AllGather 的持久输出缓冲
    joint_strategy, logical_n,
    program_config, dim,
    multi_device_global_semaphore,
    num_links, cluster_axis,
    mesh_device, topology,
    subdevice_id, ccl_core_grid_offset,
    scale, compute_kernel_config,
    core_allocation_strategy
);
```

**功能**：将 Joint SDPA 与 AllGather 融合为一个操作，在多设备 Mesh 上实现分布式注意力。

### 3.5 Ring Distributed SDPA

```cpp
Tensor ring_distributed_scaled_dot_product_attention(
    Q, K, V,
    ring_size,    // Ring 中设备数量
    ring_id,      // 当前设备 ID（可选，自动推断）
    scale, memory_config, program_config,
    compute_kernel_config,
    page_table,   // 可选分页 KV 缓存
    chunk_start_idx
);
```

**功能**：纯分布式 Ring Attention，每个设备持有一部分 K/V，通过 Ring 拓扑在设备间传递。

### 3.6 Flash MLA Prefill（Multi-Latent Attention）

```cpp
Tensor flash_mla_prefill(Q, K, head_dim_v, V, attn_mask, is_causal, ...);
Tensor chunked_flash_mla_prefill(Q, K, head_dim_v, page_table, chunk_start_idx, ...);
```

**功能**：支持 DeepSeek 风格的 Multi-Latent Attention，K 和 V 存在于同一潜在空间中，V 可从 K 导出（`head_dim_v <= head_dim_k`）。

### 3.7 SDPA Decode（解码阶段）

```cpp
Tensor scaled_dot_product_attention_decode(Q, K, V, is_causal, attn_mask, cur_pos, ...);
Tensor paged_scaled_dot_product_attention_decode(Q, K, V, page_table, ...);
Tensor flash_multi_latent_attention_decode(Q, K, V, head_dim_v, ...);
Tensor paged_flash_multi_latent_attention_decode(Q, K, V, head_dim_v, page_table, ...);
```

**功能**：Flash Decode 路径，针对自回归生成的单 token 解码优化：
- Q 只有 1 个 token 长度
- `cur_pos` / `cur_pos_tensor`: 每个 batch 元素当前序列位置
- `k_chunk_size` 自动选择为序列长度最大的 2 幂因子（上限 512）
- 支持分页 KV 缓存和 MLA

### 3.8 Windowed SDPA（窗口化注意力）

```cpp
Tensor windowed_scaled_dot_product_attention(
    Q, K, V,
    cu_window_seqlens,   // 累积窗口长度
    scale, memory_config, program_config, compute_kernel_config
);
```

**功能**：块对角注意力模式，常用于视觉 Transformer（如 Qwen2.5-VL）：
- 输入 `cu_window_seqlens = [0, 10, 25, 45]` 定义三个窗口
  - 窗口 1: token 0-9 互相注意
  - 窗口 2: token 10-24 互相注意
  - 窗口 3: token 25-44 互相注意
- 注意力掩码在 kernel 内部动态生成，避免传输大掩码张量

---

## 4. 硬件架构适配

### 4.1 Wormhole 架构

- **8×10 Tensix 核网格**，通过 2D 环形 NoC 互连
- 每个 Tensix 核包含：
  - **1.5 MB L1 SRAM**（共 120 MB 片上存储）
  - **5 个 RISC-V 核**：
    - RISC0 (Reader): 发起 NoC 传输，L1 ↔ DRAM / L1 ↔ L1
    - RISC1 (Writer): 发起 NoC 传输
    - RISC2/3/4 (Compute): 驱动矩阵处理引擎（32×32 tile 操作）
  - 这些 RISC 并行运行，原生支持**流水线和数据搬运/计算重叠**
- **12 通道 GDDR6 DRAM**：总 288 GB/s 带宽
- **16 个以太网核**：3200 Gbps 双向带宽用于 scaleout

### 4.2 TT-Metal 编程模型

- 一个 **Program** 映射到 Tensix 核的 2D 网格
- 每个核执行三个 **Kernel**：
  - **Reader kernel** → RISC0：从 DRAM 读取数据到 L1
  - **Writer kernel** → RISC1：从 L1 写数据到 DRAM
  - **Compute kernel** → RISC2/3/4：执行矩阵乘法、softmax 等
- 核间通过 **Circular Buffer** 同步（线程安全的生产者/消费者队列）

---

## 5. FlashAttention 算法实现细节

### 5.1 并行化策略

输出形状为 `[B, NQH, Sq, DH]`，沿三个维度并行化：

```
batch_parallel_factor = min(B, num_cores)
nh_parallel_factor    = min(num_cores / batch_factor, NQH)
q_parallel_factor     = min(num_cores / (batch * nh), q_num_chunks)
```

每个核负责计算 `batch_per_core × nh_per_core × q_per_core` 个 Q 块。

### 5.2 因果负载均衡（BALANCED_Q_PARALLEL）

对于因果注意力：
- Q0 只需要关注 K0，但 Q_{n-1} 需要关注 K0...K_{n-1}
- **朴素分配**（连续 Q 块分配给同一核）会导致负载不均

**优化方案**：每个核分配一个"低"Q 和一个"高"Q：
```
核 0: Q0 和 Q_{n-1}
核 1: Q1 和 Q_{n-2}
...
```

这种方式完美平衡了各核的工作量，带来 **1.6x 加速**。

### 5.3 核心计算循环（sdpa_inner_loop）

这是所有 SDPA 变体共享的核心计算模板：

```
对于每个 Q 块 q_iter:
  初始化 ping-pong 缓冲区 (cur/prev max, sum, output)

  对于每个 K 块 k_chunk:
    1. QK = Q_chunk @ K_chunk^T              # matmul_blocks (转置 K)
    2. 可选: QK += mask                       # 因果/自定义掩码
    3. cur_max = max(prev_max, row_max(QK))   # reduce_c<MAX>
    4. QK = exp((QK - cur_max) * scale)       # sub_exp_block_bcast_cols_inplace
       同时: cur_sum = row_sum(exp_result)     # L1 累积优化
    5. OUT_IM = QK @ V_chunk                  # matmul_blocks
    6. 如果不是第一个 K 块:
       a. exp_diff = exp((prev_max - cur_max) * scale)
       b. prev_sum *= exp_diff                # 重缩放
       c. cur_sum += prev_sum
       d. OUT_ACC += OUT_PREV * exp_diff      # L1 累积 matmul
    7. 交换 ping-pong 缓冲区

  最终归一化:
    cur_sum = matmul_reduce(cur_sum)  # 行内 reduce
    cur_sum = 1 / cur_sum             # recip
    output  = OUT_ACC * cur_sum       # 最终归一化
```

### 5.4 关键优化

#### 5.4.1 缩放因子融合

传统实现中 `QK *= 1/sqrt(d_head)` 是独立的乘法操作。本实现将缩放融合到 `exp()` 中：
```
exp((QK - max) * scale)     ← scale 融合到 exp 的参数中
exp((prev_max - cur_max) * scale)  ← 同样融合
```
性能关键路径上的 exp 计算免费获得了缩放。

#### 5.4.2 L1 累积优化

`sub_exp_block_bcast_cols_inplace` 同时执行：
1. `QK = exp((QK - cur_max) * scale)` — 原地修改
2. `cur_sum = row_sum(exp_result)` — 利用 L1 累积在 DST 寄存器中直接规约

避免了额外的规约 pass。

#### 5.4.3 双缓冲流水线

Q、K、V 的 Circular Buffer 容量均为 **2 个 chunk**，实现双缓冲：
```
Reader:  [读 K0] [读 K1] [读 K2] ...
Compute:        [计算 K0] [计算 K1] ...
Writer:                          [写 out0] ...
```
Reader、Writer、Compute 通过 Circular Buffer 同步，数据搬运延迟被计算完全隐藏。

#### 5.4.4 稀疏因果掩码读取

因果注意力中，只有对角线上的 score 矩阵需要应用掩码。对于 Q_i 只关注 K_0...K_i 的情况，直接限制 K 循环范围：
```
k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t
```
跳过不需要计算的 K 块，减少 DRAM 压力。

#### 5.4.5 Streaming Compute V2

当满足特定条件（非因果、无自定义掩码、无 attention sink、非分块、fp32_dest_acc 关闭、subblock_h ≤ 2）时，启用流式计算路径：
- 使用 `cb_push_back_hold_wr_ptr` 直接写入 `cb_qkt_im`，无需行缓冲
- 归一化使用 1-tile recip scratch CB
- 减少内存使用和同步开销

#### 5.4.6 KV Chain Forwarding（非因果路径）

对于非因果注意力，多个核处理同一个 head 时，启用 **KV 链式转发**：
- **Injector 核**：从 DRAM 读取 KV 并转发给链中其他核
- **Receiver 核**：从 injector 接收 KV 数据而非独立从 DRAM 读取
- 支持 **Unicast**（点对点传输）和 **Multicast**（一对多广播）
- Multicast 要求所有链核在同一物理行，且 q_chunk_count 一致

#### 5.4.7 指数函数多项式近似

`compute_common.hpp` 实现了自定义的高性能 exp 函数：
- **范围缩减**：`exp(x) = exp(r) * 2^k`，其中 `r = x - k*ln(2)`
- **多项式逼近**：支持 1-4 阶多项式（通过 Horner 方法求值）
- **Blackhole 特有优化**：使用 `SFPARECIP` 硬件指令
- **ReLU 零值保护**：对 exp 的负输出使用 packer ReLU 清零，避免数值问题
- **VectorMode::C 优化**：只计算 tile 中第一列的 exp（统计量只需要列向量）

---

## 6. 单卡执行全流程详解

本节以标准 SDPA Prefill 为例，详细说明单张 Wormhole 卡上从 Host 调度到三个 Kernel 协同完成 FlashAttention 的完整流程。

### 6.1 整体架构：一个 Program 三种 Kernel

SDPA 被编译为一个 TT-Metal **Program**，部署到 Wormhole 芯片的 2D Tensix 核网格上（最大 8×10 = 80 核）。每个 Tensix 核内部 5 个 RISC-V 核同时运行三段独立的 C++ kernel 代码：

```
┌────────────────────────────────────────────────────────────────────┐
│                  单张 Wormhole 卡 (8×10 Tensix 网格)                │
│                                                                    │
│  每个 Tensix 核 (1.5 MB L1 SRAM):                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
│  │    RISC0      │  │    RISC1      │  │   RISC2 / 3 / 4     │     │
│  │  (Reader)     │  │  (Writer)     │  │    (Compute)        │     │
│  │               │  │               │  │                     │     │
│  │  从 DRAM 读   │  │  生成掩码     │  │  QK = Q @ K^T       │     │
│  │  Q, K, V      │  │  + 缩放常量   │  │  softmax(QK)        │     │
│  │  到 L1 CB     │  │  写回 output  │  │  O = softmax @ V    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘     │
│         │                 │                      │                 │
│         └─────────────────┴──────────────────────┘                 │
│                  通过 Circular Buffer (CB) 同步                      │
│                  Reader 生产 → Compute 消费/生产 → Writer 消费       │
└────────────────────────────────────────────────────────────────────┘
```

三个 kernel 的分工：
- **Reader** (RISC0)：从 DRAM 读取 Q/K/V/mask/page_table/attention_sink 到 L1 的 Circular Buffer
- **Writer** (RISC1)：生成缩放常量和掩码 tile，等待 Compute 产出结果后写回 DRAM
- **Compute** (RISC2/3/4)：执行 FlashAttention 的核心算法——矩阵乘法、在线 softmax、累积归一化

### 6.2 Host 端：并行化与工作分配

Host 端 `SDPAProgramFactory::create` 在编译 Program 时确定并行策略。输出张量形状为 `[B, NQH, Sq, DH]`，沿前三个维度依次分配并行度：

```
batch_parallel_factor = min(B, num_cores)
nh_parallel_factor    = min(num_cores / batch_factor, NQH)
q_parallel_factor     = min(num_cores / (batch × nh), q_num_chunks)
```

每个核 `i` 被分配的工作范围通过整除/取模计算：

```
batch 范围:   [(i / (nh_par × q_par)) × batch_per_core,  +batch_per_core)
head  范围:   [((i / q_par) % nh_par) × nh_per_core,     +nh_per_core)
q_chunk 范围: [(i % q_par) × q_per_core,                 +q_per_core)
```

**示例**：`B=2, NQH=8, Sq=2048, q_chunk_size=128` 时，`q_num_chunks=16`。使用 64 个核：
- `batch_parallel=2, nh_parallel=8, q_parallel=4`
- 每个核处理 1 batch × 1 head × 4 个 Q chunk

**因果负载均衡**（`BALANCED_Q_PARALLEL`）：对于因果注意力，Q_0 只需看 1 个 K 块而 Q_{n-1} 需看 n 个 K 块。如果把连续 Q 块分给同一核，负载会严重不均。优化方案是每个核从两头各取一半：

```cpp
if (q_iter < q_chunks_per_core / 2) {
    q_chunk = local_q_start + q_iter;                           // 前半: Q_0, Q_1, ...
} else {
    q_chunk = q_num_chunks - 1 - (local_q_start + back_iter);   // 后半: Q_{n-1}, Q_{n-2}, ...
}
```

Q_0（计算量最小）和 Q_{n-1}（计算量最大）配对后，每个核的总工作量基本一致，实测带来 **1.6x 加速**。

### 6.3 Reader Kernel（RISC0）：数据搬运到 L1

Reader 是整个流水线的"源头"，负责将 Q、K、V 数据从 DRAM 搬运到 L1 的 Circular Buffer 中供 Compute 消费。

#### 6.3.1 主循环结构

```
for each phase:                              // 通常 1 个 phase
 for each batch in [local_batch_start, local_batch_end):
   if chunked: 读取该 batch 的 page_table → cb_page_table (c6)
   for each head in [local_nh_start, local_nh_end):
     if attention_sink: 读取 sink 值 → cb_attention_sink (c4)
     for each q_iter in [0, q_chunks_per_core):
       ① 读 Q chunk → cb_q_in (c0)
       ② 确定 K 循环范围:
          因果:   k_end = ceil(q_high_idx / Sk_chunk_t)
          非因果: k_end = Skt / Sk_chunk_t (全部 K)
       for each k_chunk in [0, k_end):
         ③ 读 K chunk → cb_k_in (c1)，以转置方式存放
         ④ if 有用户掩码: 读 mask → cb_mask_in (c3)
         ⑤ if subblock push 且 k_chunk==0: 分 subblock 读 Q
         ⑥ 读 V chunk → cb_v_in (c2)
```

#### 6.3.2 K 的转置读取

QK^T 矩阵乘法要求 K 被转置。Reader 在从 DRAM 读取 K 时直接按转置布局写入 L1，通过调整写指针的步幅实现：

```cpp
// 正常: 按行写, stride = dst_cols × tile_bytes
// 转置: 按列写, stride = tile_bytes; 行间 stride = tile_bytes × dst_rows
uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;
```

这样 Compute 拿到的 K 数据已经是转置后的布局，matmul 时直接使用 `transpose=true` 标志即可。

#### 6.3.3 Padding 处理

当有效序列长度不是 chunk_size 的整数倍时，最后一个 chunk 会出现 padding。Reader 读取有效 tile 后，将多余位置填零：

```cpp
// 读有效数据
for (row < src_rows, col < src_cols):
    noc_async_read_tile(tile_id, reader, write_ptr);

// 零填充 padding 区域
for (row, col) 超出 (src_rows, src_cols) 范围:
    fill_tile_zeros<tile_bytes>(cb_id, tile_idx);  // 通过 NoC 从芯片零地址区读取
```

#### 6.3.4 分页 KV 缓存读取（Chunked 模式）

在 chunked prefill 模式下，K/V 存储在分页缓存中。Reader 先读取页表，再通过虚拟→物理地址映射读取实际 tile：

```cpp
// 虚拟序列 tile 索引 → 物理 tile ID
physical_tile_id = page_table[virtual_block] × (num_heads × block_size_t × Wt)
                 + head × block_size_t × Wt
                 + block_row × Wt;
```

#### 6.3.5 双缓冲

K 和 V 的 CB 大小均为 **2 个 chunk**（`k_tiles = Sk_chunk_t × DHt × 2`），允许 Reader 读下一个 K/V chunk 的同时 Compute 仍在处理当前 chunk。Q 的 CB 大小在 `q_per_core > 1` 时也设为 2 倍以支持双缓冲。

#### 6.3.6 Q Subblock Push 优化

当 `Sq_chunk_t / qk_subblock_h > 1` 时，Q 不是一次性全部读完再 push，而是在第一个 K chunk 到达后分 subblock 逐步 push。这让 Compute 能在第一个 Q subblock 就绪时立即开始 QK matmul 的第一个 subblock 计算，而 Reader 同时继续读取后续 Q subblock：

```cpp
if (k_chunk == 0) {
    for (q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
        read_q_subblock(reader, cb_q_in, tile_id, q_sub * subblock_h, subblock_h, ...);
        // 每个 subblock 读完后立即 cb_push_back，Compute 可以逐步开始计算
    }
}
```

#### 6.3.7 读取阈值与 Barrier

为避免 NoC 拥塞，Reader 设置了读取 barrier 阈值。每读取 `barrier_threshold` 个 tile 后执行一次 `noc_async_read_barrier()`：

```cpp
constexpr uint32_t barrier_threshold = ((512 / num_readers) * (1024 + 128)) / tile_bytes;
```

### 6.4 Writer Kernel（RISC1）：掩码生成与输出写回

Writer 承担两个职责：在计算开始前生成所需常量，以及在计算完成后写回结果。

#### 6.4.1 初始化：生成常量 Tile

在进入主循环之前，Writer 一次性生成两个恒等 tile：

```cpp
generate_reduce_scaler(cb_identity_scale_in, 1.0f_packed);  // 全 1.0 标量 tile → c5
generate_bcast_col_scalar(cb_col_identity, 1.0f_packed);     // 全 1.0 列向量 tile → c7
```

这两个 tile 在整个 kernel 生命周期中保持 fronted，被 Compute 反复使用：
- `cb_identity_scale_in`：用于 `reduce_c<MAX>` 和 `reduce_c<SUM>` 中的 scale 参数
- `cb_col_identity`：用于 `matmul_reduce` 实现行内求和

如果启用轻量级掩码（streaming v2 路径），Writer 还生成一个全 `-inf` tile：

```cpp
for (i = 0; i < mask_tile_size / 4; i++)
    ptr[i] = 0xFF80FF80;  // bfloat16 -inf
```

#### 6.4.2 因果掩码动态生成

当用户未提供 mask 且为因果/滑动窗口模式时，Writer 在主循环中为每个 Q chunk 动态生成掩码（使用 BFP4_b 格式以最小化内存占用）。对每个 `(q_tile, k_tile)` 组合，判断属于三种情况之一：

| 类型 | 条件 | 填充 |
|------|------|------|
| FULLY_ALLOWED | k_tile 完全在因果三角内 | 全 0 |
| FULLY_MASKED | k_tile 完全在因果三角外 | 全 -inf |
| PARTIAL_MASK | k_tile 在对角线上 | 逐元素判断 |

对角线 tile 的生成通过 `fill_custom_diagonal_tile_bfp4` 实现，按 BFP4_b 的 4 个 face 布局（16×16 子块）逐行判断每个元素是否需要 mask。

**滑动窗口掩码**额外引入 `trailing_diagonal_offset`，形成带状注意力模式而非简单下三角。

#### 6.4.3 输出写回

Writer 等待 Compute 产出一个 output chunk 后，将有效 tile 写回 DRAM：

```cpp
cb_wait_front(cb_out, out_chunk_tiles);       // 阻塞等待 Compute 完成
for each valid (row, col):
    noc_async_write_tile(tile_id, out_writer, l1_read_addr);
noc_async_write_barrier();                     // 等待所有写完成
cb_pop_front(cb_out, out_chunk_tiles);         // 释放 CB 空间
```

只写回 `out_row_tile_count × vDHt` 个有效 tile，padding 区域跳过不写。

### 6.5 Compute Kernel（RISC2/3/4）：FlashAttention 核心算法

Compute 实现了 FlashAttention-2 的在线 softmax 算法，是计算密度最高的部分。它有两条执行路径。

#### 6.5.1 标准路径（sdpa_standard）

用于因果/掩码/分块/attention_sink/滑动窗口等复杂场景。核心是 `sdpa_inner_loop` 函数，使用 **ping-pong buffer** 交替存储中间统计量：

```
对于每个 Q chunk (q_iter):
  建立 ping-pong 别名:
    alias_prev_sum ←→ alias_cur_sum     (cb_sum_A=c29 / cb_sum_B=c30)
    alias_prev_max ←→ alias_cur_max     (cb_max_A=c27 / cb_max_B=c28)
    alias_mm2_prev ←→ alias_mm2_cur     (cb_out_im_A=c25 / cb_out_im_B=c26)

  对于每个 K chunk (k_chunk):
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 1: QK = Q_chunk @ K_chunk^T                           │
    │   matmul_blocks(cb_q_in, cb_k_in, cb_qk_im,               │
    │                 transpose=true)                             │
    │   ▸ Q 保留在 CB 中（多次使用），K 被消费                      │
    │   ▸ 结果写入 cb_qk_im (c24), 大小 Sq_chunk_t × Sk_chunk_t  │
    ├─────────────────────────────────────────────────────────────┤
    │ Step 2: 可选 QK += mask                                    │
    │   add_block_inplace(cb_qk_im, cb_mask_in, tiles)           │
    │   ▸ 因果/自定义/padding 掩码加到 score 上                    │
    ├─────────────────────────────────────────────────────────────┤
    │ Step 3: cur_max = max(prev_max, row_max(QK))               │
    │   reduce_c<MAX>(cb_qk_im → alias_cur_max)                  │
    │   ▸ 对 QK 每行求最大值（列方向规约）                         │
    │   ▸ 如果不是第一个 K chunk，与 prev_max 做 eltwise_max       │
    │   ▸ 结果是 Sq_chunk_t 个 tile 的列向量                      │
    ├─────────────────────────────────────────────────────────────┤
    │ Step 4: QK = exp((QK - cur_max) * scale)                   │
    │         cur_sum = row_sum(exp_result)                       │
    │   sub_exp_block_bcast_cols_inplace(cb_qk_im, cur_max,      │
    │                                    → cur_sum)              │
    │   ▸ 关键融合优化:                                           │
    │     - scale 融合到 exp 参数中，免费获得缩放                   │
    │     - L1 累积: exp 结果在 DST 寄存器中直接 reduce_sum        │
    │     - Packer ReLU: 清零 exp 的负输出（数值保护）             │
    │   ▸ QK 原地修改为 softmax 分子                              │
    ├─────────────────────────────────────────────────────────────┤
    │ Step 5: OUT_IM = softmax(QK) @ V_chunk                     │
    │   matmul_blocks(cb_qk_im, cb_v_in, alias_mm2_cur,          │
    │                 transpose=false)                            │
    │   ▸ V 被消费，结果写入当前 output 中间缓冲                   │
    ├─────────────────────────────────────────────────────────────┤
    │ Step 6: 如果不是第一个 K chunk，修正累积值:                   │
    │   exp_diff = exp((prev_max - cur_max) * scale)             │
    │   prev_sum *= exp_diff        // 重缩放旧的 sum              │
    │   cur_sum  += prev_sum        // 合并 sum                   │
    │   OUT_ACC  += OUT_PREV × exp_diff  // L1 累积 matmul        │
    │   ▸ mul_block_bcast_cols 使用 pack_accumulate=true           │
    │     将 prev_out × exp_diff 累积到 cur_out 上                │
    ├─────────────────────────────────────────────────────────────┤
    │ Step 7: 交换 ping-pong 指针                                 │
    │   swap(alias_prev_sum, alias_cur_sum)                      │
    │   swap(alias_mm2_prev, alias_mm2_cur)                      │
    │   swap(alias_prev_max, alias_cur_max)                      │
    └─────────────────────────────────────────────────────────────┘

  最终归一化 (所有 K chunk 处理完毕):
    ① matmul_reduce<Sq_chunk_t>(cb_col_identity, prev_sum)
       ▸ 行内最终 reduce: 将 partial sum 变为真正的 row sum
       ▸ 实现为 Mx1 matmul（每行乘以全 1 向量）
    ② recip_block_inplace(prev_sum)
       ▸ prev_sum = 1.0 / prev_sum
    ③ mul_block_bcast_cols(OUT_ACC, prev_sum, cb_out)
       ▸ output = OUT_ACC × (1/sum)，写入 cb_out (c16)
       ▸ Writer 随后将 cb_out 写回 DRAM
```

#### 6.5.2 Streaming 路径（sdpa_standard_v2）

当满足以下所有条件时启用：
- 非因果、无用户掩码、无 attention_sink、非分块模式
- `fp32_dest_acc_en` 关闭
- `qk_subblock_h ≤ 2` 且 `Sk_chunk_t % (8/subblock_h) == 0`
- `Sq_chunk_t / subblock_h > 1`（至少两个 Q subblock）

此路径使用 `cb_push_back_hold_wr_ptr` 直接写入 `cb_qkt_im`，不需要独立的行缓冲 CB。归一化使用 1-tile recip scratch CB (`c4`)。总体减少 CB 使用量和同步开销。

### 6.6 三个 Kernel 的流水线协同

三个 kernel 通过 Circular Buffer 实现生产者-消费者同步。CB 是一个环形队列，关键 API：

| API | 角色 | 含义 |
|-----|------|------|
| `cb_reserve_back(cb, n)` | 生产者 | 预留 n 个 tile 的写空间（满时阻塞） |
| `cb_push_back(cb, n)` | 生产者 | 提交 n 个 tile 为已生产 |
| `cb_wait_front(cb, n)` | 消费者 | 等待至少 n 个 tile 可消费（空时阻塞） |
| `cb_pop_front(cb, n)` | 消费者 | 释放 n 个 tile 的空间 |

完整时序示意（以一个 Q chunk 处理两个 K chunk 为例）：

```
时间 →

Reader:   [读Q→c0] [读K0→c1] [读V0→c2]  [读K1→c1]  [读V1→c2]
               │         │         │          │          │
           push_back  push_back push_back  push_back push_back

Compute:         [wait c0,c1]  [QK0=Q@K0^T]  [wait c2]  [O0=QK0@V0]
                                                    │
                               [wait c1] [QK1=Q@K1^T] [wait c2] [O1=QK1@V1]
                                                                      │
                                                    [归一化: O/sum → c16]
                                                                   push_back

Writer:  [生成scale,mask]                       [wait c16] [写回DRAM]
```

**双缓冲效果**：由于 K 和 V 的 CB 容量为 2 个 chunk，Reader 可以在 Compute 处理 K_0 时同时读取 K_1 到另一个 slot：

```
Reader:  [读 K0 → slot A] [读 K1 → slot B] [读 K2 → slot A] ...
Compute:                  [处理 K0 from A] [处理 K1 from B] [处理 K2 from A] ...
```

数据搬运延迟被计算完全隐藏。

### 6.7 KV Chain Forwarding（非因果路径的核间 KV 共享）

在非因果注意力中，多个核可能处理同一个 (batch, head) 的不同 Q chunk，但都需要遍历相同的全部 K/V。朴素做法是每个核独立从 DRAM 读取，浪费带宽。

**优化**：Host 在构建 Program 时分析哪些核共享同一 head，建立 **Chain 拓扑**。选择一个 **Injector 核** 从 DRAM 读 KV，然后通过 NoC L1→L1 传输转发给链中的 **Receiver 核**。

```
                    DRAM
                     │
                     ↓ 只读一次
              ┌──────────────┐
              │  Injector 核  │
              │  (从DRAM读KV) │
              └──────┬───────┘
                     │ NoC L1→L1 转发
            ┌────────┼────────┐
            ↓        ↓        ↓
      ┌─────────┐ ┌─────────┐ ┌─────────┐
      │Receiver │ │Receiver │ │Receiver │
      │  核 1   │ │  核 2   │ │  核 3   │
      └─────────┘ └─────────┘ └─────────┘
```

**同步协议（Unicast 模式）**：

```
Injector:                              Receiver:
  从 DRAM 读 K chunk                     cb_reserve_back(cb_k_in)
  noc_semaphore_wait(sender_sem)  ←────  noc_semaphore_inc(sender_sem, 1)
  noc_async_write(K → Receiver L1)       noc_semaphore_wait(receiver_sem)
  noc_semaphore_set_remote(valid)  ────→ (数据到达，继续计算)
  (对 V 重复同样过程)
```

**Multicast 模式**：当所有链核在同一物理行且 q_chunk_count 一致时，使用 NoC multicast 一对多广播，用 `noc_async_write_multicast` + `noc_semaphore_set_multicast` 替代逐核 unicast。

**Chain 构建策略**（Host 端）：
1. 遍历所有 head，找到跨多核的 head
2. 选择 Injector：优先选择单 head 的核，且物理 X 坐标远离已有 injector（分散 DRAM channel 负载）
3. 非均匀 q_chunk_count 时按降序排列（重的核先发送，轻的核少接收）
4. Multicast 检查：所有链核同行、无间隙核、q_chunk_count 一致

### 6.8 端到端数据流图

```
                              DRAM
                               │
              ┌────────────────┼────────────────┐
              ↓                ↓                ↓
          ┌───────┐       ┌───────┐        ┌───────┐
          │Q tiles│       │K tiles│        │V tiles│
          │ (行序) │       │(转置序)│        │ (行序) │
          └───┬───┘       └───┬───┘        └───┬───┘
              ↓               ↓                ↓
          cb_q_in         cb_k_in           cb_v_in          Reader 写入
           (c0)            (c1)              (c2)
              │               │                │
              │    ┌──────────┘                │
              │    │   QK = Q @ K^T            │
              │    │      ↓                    │
              │    │  cb_qk_im (c24)           │
              │    │      │                    │
              │    │  [+ mask from c3]         │
              │    │  [row_max → c27/c28]      │
              │    │  [exp((QK-max)*scale)]     │
              │    │  [row_sum → c29/c30]       │
              │    │      │                    │
              │    │  softmax(QK) @ V ─────────┘
              │    │      ↓
              │    │  cb_out_im_A/B (c25/c26)     Compute 中间结果
              │    │      │
              │    │  [exp(prev_max - cur_max) → c31]
              │    │  [OUT_ACC += prev × exp_diff]
              │    │      │
              │    │  [归一化: OUT / sum]
              │    │      ↓
              │    │  cb_out (c16)                 Compute 写入
              │    │      │
              └────┘      ↓
                      Writer → DRAM               Writer 读取并写回
```

### 6.9 关键性能数据

根据 Tenstorrent 官方 FlashAttention 技术报告（在 n150 单卡上测试）：

| 指标 | 数值 |
|------|------|
| 相比基线（中间结果写 DRAM）的加速 | **平均 20x**（范围 9x–44x） |
| 因果负载均衡带来的额外加速 | **1.6x** |
| BFP8 vs BF16 的提升 | 有提升但 < 2x（部分 compute-bound） |
| 测试 head_dim | {64, 128, 256} |
| 测试 seq_len | {512, ..., 16384} |
| 测试平台 | Wormhole n150 (单芯片, ≤160W) |

加速来源：
1. 中间结果保持在 L1 而非写回 DRAM
2. 因果注意力跳过不必要的计算（上三角区域）
3. Reader/Compute/Writer 三级流水线隐藏数据搬运延迟

---

## 7. Circular Buffer 完整布局

| CB 索引 | 名称 | 用途 | 大小 |
|---------|------|------|------|
| c_0 | cb_q_in | Q 输入 | Sq_chunk_t × DHt × q_buffer_factor |
| c_1 | cb_k_in | K 输入（双缓冲） | Sk_chunk_t × DHt × 2 |
| c_2 | cb_v_in | V 输入（双缓冲） | Sk_chunk_t × vDHt × 2 |
| c_3 | cb_mask_in | 注意力掩码 | Sq_chunk_t × Sk_chunk_t × 2 或 1（轻量级） |
| c_4 | cb_attention_sink / cb_recip_scratch | 注意力汇 或 流式 recip 临时 | Sq_chunk_t / 1 |
| c_5 | cb_identity_scale_in | 恒等缩放标量 | 1 tile |
| c_6 | cb_page_table | 页表 | page_table_stick_size |
| c_7 | cb_col_identity | 列恒等向量 | 1 tile |
| c_8 | cb_chunk_start_idx | 分块起始索引（compute） | 32 bytes |
| c_9 | cb_chunk_start_idx_writer | 分块起始索引（writer） | 32 bytes |
| c_16 | cb_out | 最终输出 | Sq_chunk_t × vDHt |
| c_24 | cb_qk_im | QK 中间结果 | Sq_chunk_t × Sk_chunk_t |
| c_25 | cb_out_im_A | 输出中间结果 A（ping-pong） | Sq_chunk_t × vDHt |
| c_26 | cb_out_im_B | 输出中间结果 B（ping-pong） | Sq_chunk_t × vDHt |
| c_27 | cb_max_A | 当前最大值 A（ping-pong） | Sq_chunk_t |
| c_28 | cb_max_B | 当前最大值 B（ping-pong） | Sq_chunk_t |
| c_29 | cb_sum_A | 当前求和 A（ping-pong） | Sq_chunk_t |
| c_30 | cb_sum_B | 当前求和 B（ping-pong） | Sq_chunk_t |
| c_31 | cb_exp_max_diff | exp(prev_max - cur_max) | Sq_chunk_t |

中间结果使用 **Float16_b** 格式（非 Float32），以减少 L1 占用。

---

## 8. Subblock 大小选择

`determine_largest_subblock_size` 从预定义的候选列表中选择最大的 subblock：

```cpp
constexpr std::array<std::pair<uint32_t, uint32_t>, 20> subblocks = {{
    {2,4}, {4,2}, {1,8}, {8,1}, {1,7}, {7,1}, {2,3}, {3,2},
    {1,6}, {6,1}, {1,5}, {5,1}, {2,2}, {1,4}, {4,1}, {1,3},
    {3,1}, {1,2}, {2,1}, {1,1},
}};
```

约束条件：
1. `subblock_h × subblock_w ≤ dst_size`（fp32 时 dst_size=4，fp16 时 dst_size=8）
2. `block_height % subblock_h == 0` 且 `block_width % subblock_w == 0`

目标是最大化 DST 寄存器利用率。

---

## 9. Program 构建流程（SDPAProgramFactory::create）

### 9.1 维度计算

```
B = batch_size
NQH = num_query_heads
NKH = num_kv_heads
Sq = Q sequence length
Sk = K sequence length (计算方式取决于是否 chunked)
DH = head_dim
padded_Sq = ceil(Sq / q_chunk_size) * q_chunk_size
padded_Sk = ceil(Sk / k_chunk_size) * k_chunk_size
```

### 9.2 并行因子计算

```
batch_parallel_factor = min(B, num_cores)
nh_parallel_factor = min(num_cores / batch, NQH)
q_parallel_factor = min(num_cores / (batch * nh), q_num_chunks)
```

### 9.3 Matmul 参数配置

两次 matmul 的参数独立配置：
- **QK matmul**: `[Sq_chunk_t, DHt] @ [DHt, Sk_chunk_t]` → `[Sq_chunk_t, Sk_chunk_t]`
- **Out matmul**: `[Sq_chunk_t, Sk_chunk_t] @ [Sk_chunk_t, vDHt]` → `[Sq_chunk_t, vDHt]`

### 9.4 Granularity 参数

各类 granularity 确保 tile 处理的正确性（必须整除 tile 总数）：
- `stats_granularity`: 统计计算的 DST 粒度
- `sub_exp_granularity`: sub-exp 操作的粒度
- `mul_bcast_granularity`: 广播乘法的粒度
- `dht_granularity`: head_dim 方向的粒度
- `reduce_granularity`: 规约操作的粒度

### 9.5 Kernel 编译参数

每个 kernel 接收大量编译时参数：
- **Reader**: 维度参数 + 模式标志 + Tensor Accessor Args + 信号量 ID
- **Writer**: 维度参数 + 缩放因子 + 模式标志 + Output Accessor Args
- **Compute**: Matmul 配置 + 维度 + 模式标志 + Granularity defines

### 9.6 运行时参数

每个核接收不同的运行时参数：
- 该核负责的 batch/head/q_chunk 范围
- 缓冲区地址
- Chain 转发元数据（非因果路径）

---

## 10. 验证逻辑

`validate_on_program_cache_miss` 执行详细验证：

1. **存储检查**：所有输入必须在设备上，分配了 buffer
2. **布局检查**：所有输入必须是 TILE 布局
3. **数据类型检查**：支持 BF16、BFP8_B、BFP4_B
4. **交织检查**：操作数必须是 DRAM/L1 interleaved（非 sharded）
5. **padding 检查**：仅序列维度允许 padding
6. **GQA 验证**：`NQH >= NKH` 且 `NQH % NKH == 0`
7. **因果/掩码互斥**：`is_causal` 和 `attn_mask` 不能同时提供
8. **分块模式验证**：页表、chunk_start_idx 的一致性
9. **MLA 验证**：`head_dim_v <= head_dim_q`

---

## 11. 性能模型

`create_op_performance_model` 计算理想执行周期数：

```
FLOPS(causal) = 2 × B × Sq² × NQH × DH
```

考虑因素：
- 计算网格维度
- 数学精度（HiFi2/HiFi4/LoFi）
- 仅支持 Wormhole B0 和 Blackhole 架构

---

## 12. Ring Distributed SDPA 特殊处理

Ring SDPA 在标准 FlashAttention 基础上增加了跨设备的统计量合并：

### 12.1 Log-Sum-Exp (LSE) 累积

每个 ring 迭代后，通过 LSE 合并两个设备的结果：

```
sig = sigmoid(cur_lse - prev_lse)
out = prev_out - sig * (prev_out - cur_out)
lse = prev_lse - logsigmoid(prev_lse - cur_lse)
```

### 12.2 Padding 处理

每个设备持有的 KV 分片可能有 padding：
- `logical_nt`: 全局逻辑序列长度
- `local_padded_Nt`: 每个设备上的 padded 长度
- 超出逻辑长度的 KV 块被跳过

---

## 13. TT-Train 中的 SDPA

tt-train 模块包含完整的前向/后向 SDPA 实现：

### 13.1 前向（sdpa_fw）
- 标准 FlashAttention 前向，保存 LSE 统计量用于反向传播

### 13.2 后向（sdpa_bw）
- 分为 Q 梯度和 KV 梯度两个独立操作
- Q 梯度：`dQ = softmax(QK^T) @ dO @ V^T`
- KV 梯度：`dK = Q^T @ (dO ⊙ P^T)`, `dV = P^T @ dO`

### 13.3 Ring 分布式训练变体
- `ring_sdpa_fw` / `ring_sdpa_bw`: 在多设备上的分布式训练支持

---

## 14. 测试覆盖

测试分布在多个目录：

| 路径 | 内容 |
|------|------|
| `tests/ttnn/unit_tests/operations/sdpa/` | SDPA prefill/decode 单元测试 |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/` | 夜间测试（chunked, joint, ring, MLA） |
| `tests/tt_metal/tt_metal/test_sdpa_reduce_c.cpp` | 底层 reduce_c 操作测试 |
| `tests/didt/test_sdpa_op.py` | DIDT 测试 |
| `tests/sweep_framework/sweeps/model_traced/` | 模型追踪的 sweep 测试 |
| `tt-train/tests/ops/` | 训练 SDPA 前向/后向测试 |

---

## 15. 因果 vs 非因果注意力与 Multicast 适用性

### 15.1 因果注意力（Causal Attention）

因果注意力的核心约束：**每个 token 只能看到它自己和它之前的 token，不能看到未来的 token。** 注意力分数矩阵是一个下三角：

```
Q\K   t0   t1   t2   t3
t0   [✓    ✗    ✗    ✗ ]     t0 只能看 t0
t1   [✓    ✓    ✗    ✗ ]     t1 能看 t0, t1
t2   [✓    ✓    ✓    ✗ ]     t2 能看 t0, t1, t2
t3   [✓    ✓    ✓    ✓ ]     t3 能看所有
```

所有自回归语言模型（GPT、LLaMA、Qwen、DeepSeek、Mistral 等）的 attention 都是 causal 的。

### 15.2 非因果注意力（Non-causal Attention）

非因果注意力中，每个 token 可以看到序列中的所有 token（或由自定义 mask 决定）。典型场景：

| 场景 | 模型示例 | 原因 |
|------|---------|------|
| 编码器模型 | BERT、RoBERTa | 目标是理解整个句子，需要全局上下文 |
| 视觉 Transformer | ViT、DINOv2、Swin | 图像 patch 之间没有时序关系 |
| 多模态模型图像部分 | Qwen2.5-VL、LLaVA | 图像 token 之间互相 attend |
| 扩散模型 | Stable Diffusion (DiT)、Flux | 去噪过程需要全局信息 |
| 蛋白质结构预测 | AlphaFold2/3、Boltz | 氨基酸残基之间双向 attend |
| 窗口化注意力 | Qwen2.5-VL windowed SDPA | 图像 token 在窗口内互相 attend |

### 15.3 为什么因果注意力无法使用 Multicast

KV chain forwarding / multicast 的核心前提是：**链中所有核需要完全相同的 K/V 数据流，以相同的节奏消费。**

非因果模式下每个 Q chunk 需遍历全部 K chunk，所有核的 K 循环完全一致，可以共享 K/V 流。但因果模式下，不同 Q chunk 需要的 K 范围不同：

```
Core 0 (处理 Q_0):  K 循环 [K_0]                      → 1 次迭代
Core 1 (处理 Q_1):  K 循环 [K_0, K_1]                  → 2 次迭代
Core 2 (处理 Q_2):  K 循环 [K_0, K_1, K_2]             → 3 次迭代
Core 3 (处理 Q_3):  K 循环 [K_0, K_1, K_2, K_3]        → 4 次迭代
```

这会导致**信号量死锁**：Multicast 协议中 injector 要等所有 N 个 receiver 发出 ready 信号后才发送下一个 K chunk。如果 Core 0 处理完 K_0 后退出了 K 循环，进入归一化阶段，不再发 ready 信号，injector 会永久阻塞等待 Core 0 的信号。

即使强行让所有核接收全部 K chunk（不需要的丢弃），也会：
- 浪费 NoC 带宽（Core 0 接收 n 个 K chunk 但只用 1 个）
- 浪费时间（Core 0 被迫等 n 个迭代才能结束）
- 抵消因果优化的核心收益（跳过上三角区域减少计算量）

因此，代码中直接排除了因果路径：

```cpp
if (!is_causal && !is_chunked) {
    // 只在非因果、非分块时构建 chain 拓扑
}
```

---

## 16. 非因果 Multicast 优化空间分析

以下分析基于对 `reader_interleaved.cpp` 转发协议和 `sdpa_program_factory.cpp` chain 构建逻辑的详细审计。

### 16.1 协议层优化

#### 16.1.1 K+V 合并传输，减少信号量往返

**现状**：每个 K chunk 和 V chunk 各需要一次完整的信号量握手（4 次信号量操作/k_chunk）。

**优化**：将 K 和 V 合并为一次传输：

```
当前: 对每个 k_chunk:
  [recv: ready→] [inj: wait→read_K→send_K→signal]    ← K 握手
  [recv: ready→] [inj: wait→read_V→send_V→signal]    ← V 握手
  总计: 4 次信号量操作

优化后:
  [recv: ready→] [inj: wait→read_K+V→send_K+V→signal] ← 1 次握手
  总计: 2 次信号量操作，减少 50%
```

**收益**：信号量操作涉及 NoC 往返延迟（~100ns 级别），小 chunk_size 时同步开销占比显著。

#### 16.1.2 流水线化 DRAM 读取与 NoC 转发

**现状**：Injector 串行执行：从 DRAM 读 → 等 receiver ready → 发送。

**优化**：利用 NoC 读（DRAM→L1）和写（L1→L1）可以并行，重叠下一个 chunk 的 DRAM 读取与当前 chunk 的转发：

```
当前: [从DRAM读K_0] → [发K_0] → [从DRAM读V_0] → [发V_0] → [从DRAM读K_1] → ...
优化: [从DRAM读K_0] → [发K_0 + 同时从DRAM读V_0] → [发V_0 + 同时从DRAM读K_1] → ...
```

**收益**：DRAM 读取延迟被转发操作隐藏，injector 有效吞吐接近翻倍。需要 K/V CB 各 3 倍缓冲。

#### 16.1.3 信用（Credit）制替代阻塞式信号量

**现状**：Injector 每次发送前必须等所有 N 个 receiver 都发出 ready 信号。任何一个 receiver 的 compute 稍慢都会阻塞全链。

**优化**：改用信用制——receiver 初始发出 2 个 credit（对应双缓冲的 2 个 CB slot），injector 有 credit 就发送，不阻塞等全部。消除"木桶效应"。

**限制**：此优化更适合 unicast 路径——multicast 无法按核粒度做信用控制。

### 16.2 拓扑层优化

#### 16.2.1 放弃 All-or-Nothing，支持 Per-Chain 混合模式

**现状**：`mcast_enabled` 是编译时常量。只要有一个 chain 不满足 multicast 条件，所有 chain 退化为 unicast：

```cpp
constexpr bool mcast_enabled = get_compile_time_arg_val(30) == 1;
```

**优化**：将 `mcast_enabled` 改为运行时 per-core 参数，每个 chain 独立决定用 multicast 还是 unicast。实现难度低（仅改一个 constexpr if 为 runtime branch），但收益高——部分满足条件的 chain 不再被拖累。

#### 16.2.2 核布局感知的 Chain 构建

**现状**：逻辑核到物理核的映射由 device 固定（行优先线性映射），chain 构建只能被动检查是否同行。

**优化**：利用 `SDPAProgramConfig::sub_core_grids` 显式安排核布局，使同一 head 的核落在同一物理行：

```
当前随机映射:              优化映射:
  Row 0: H0 H1 H2 H3       Row 0: H0 H0 H0 H0 H0 H0 H0 H0
  Row 1: H0 H1 H2 H3       Row 1: H1 H1 H1 H1 H1 H1 H1 H1
  Row 2: H0 H1 H2 H3       Row 2: H2 H2 H2 H2 H2 H2 H2 H2
  跨行 → 无法 mcast        同行 → 全部可 mcast
```

**收益**：绝大多数场景都能满足 multicast 条件。需要同时考虑 DRAM channel 均衡。

#### 16.2.3 树形转发替代星形拓扑

**现状**：一个 injector 核向所有 N 个 receiver 发送，injector 的 NoC 写带宽成为瓶颈。

**优化**：使用 2 级树形拓扑——injector 发给 2-3 个 sub-injector，sub-injector 再各自 multicast 给子集 receiver：

```
当前 (星形):                          优化 (树形):
  Injector → Recv 0,1,...,N           Injector → Sub-A → Recv 0,1,2,3
                                               → Sub-B → Recv 4,5,6,7
```

**收益**：链长 > 8 时显著降低 injector 的 NoC 压力。

#### 16.2.4 轮转 Injector

**现状**：整个 K 迭代过程中同一个 injector 从 DRAM 读所有 K chunk。

**优化**：不同 K chunk 由不同核担任 injector（K_0 由 Core A 读并广播，K_1 由 Core B），DRAM 读取负载分散到多个核的对应 DRAM channel。

### 16.3 适用范围扩展

#### 16.3.1 扩展到 Paged KV 缓存（Chunked 模式）

**现状**：Chunked 模式直接禁用 chain forwarding。但同一 head 的不同 Q chunk 仍遍历相同 paged K/V。

**优化**：Injector 在做完 page table 翻译后，将物理 tile 数据转发给 receiver——receiver 不需要知道 page table，直接接收数据。扩展后覆盖 LLM cross-attention 等更多工作负载。

#### 16.3.2 部分因果支持（Common-Prefix Multicast）

**这是最有价值但也最复杂的优化。**

虽然因果注意力整体不能 chain forward，但对角线以下的 K chunk 是所有 Q chunk 共需的：

```
Q_0: 需要 K_0                           ← K_0 被所有 Q 需要
Q_1: 需要 K_0, K_1                      ← K_0 可以 multicast!
Q_2: 需要 K_0, K_1, K_2                 ← K_1 可以 multicast 给 3 个核!
Q_3: 需要 K_0, K_1, K_2, K_3
```

**方案**：构建递减式 chain——K_0 multicast 给所有核，K_1 给 n-1 个核，...，越靠近对角线的 K 越少核需要。

```
K_0:  Injector → mcast 给 7 个 receiver
K_1:  Injector → mcast 给 6 个 receiver
...
K_6:  Injector → unicast 给 1 个 receiver
K_7:  Core 7 自己从 DRAM 读
```

**DRAM 带宽收益**（n=8 个 K chunk）：

```
朴素因果: 总 DRAM 读取 = 1+2+3+...+8 = 36 次
优化后:   Injector 读 7 次 + Core 7 自读 1 次 = 8 次
节省: 36 → 8 = 4.5 倍
```

**挑战**：multicast 矩形随 K 迭代缩小（receiver 逐步退出），需要动态调整 mcast 地址范围和信号量等待数，协议复杂度高。

#### 16.3.3 放宽 Uniform q_chunk_count 约束

**现状**：Multicast 要求所有链核有相同的 q_chunk_count。`q_num_chunks % q_parallel_factor != 0` 时尾部核分到更少 Q chunk，导致 multicast 退化。

**优化**：让尾部核做 dummy Q 迭代——不执行实际计算，但参与信号量协议（发 ready、接收、丢弃）。在大多数实际场景中，不均匀差距仅为 1，overhead 很小。

### 16.4 数据层优化

#### 16.4.1 利用 NoC 双通道

Wormhole 有 NOC0 和 NOC1 两个独立通道。当前所有 chain forwarding 传输使用 NOC0。优化方案：K 用 NOC0 发，V 用 NOC1 发，实现 K/V 并行传输，理论传输时间减半。

#### 16.4.2 KV 传输压缩

Injector 从 DRAM 读取后将 K/V 压缩为 BFP4（尺寸减为原来的 1/4 相对 BF16）再转发，receiver 端解压。NoC 传输量大幅减少，但引入压缩/解压延迟和精度损失。

### 16.5 优化优先级排序

| 优先级 | 优化项 | 收益 | 实现难度 | 适用场景 |
|--------|--------|------|----------|---------|
| **P0** | Per-Chain 混合模式 | 高 | 低 | 通用 |
| **P0** | 核布局感知 | 高 | 中 | 通用 |
| **P1** | 流水线化 DRAM 读+转发 | 高 | 中 | DRAM 带宽受限 |
| **P1** | 放宽 q_chunk_count 约束 | 中 | 低 | 非整除序列长度 |
| **P1** | K+V 合并传输 | 中 | 中 | 小 chunk_size |
| **P2** | 扩展到 paged KV | 高 | 高 | LLM cross-attention |
| **P2** | 部分因果支持 | 很高 | 很高 | **LLM prefill（最大场景）** |
| **P2** | 双 NoC 通道 | 中 | 中 | NoC 带宽受限 |
| **P3** | 树形转发 | 中 | 高 | 超长链（>16 核） |
| **P3** | 轮转 injector | 中 | 高 | DRAM channel 不均衡 |
| **P3** | 信用制 | 低-中 | 中 | compute 延迟差异大 |
| **P3** | KV 传输压缩 | 中 | 高 | NoC 带宽极度受限 |

---

## 17. 总结

TT-Metal 的 SDPA 实现是一个工程精度很高的 FlashAttention 移植，主要特点：

1. **全面的变体支持**：Prefill、Decode、Chunked、Joint、Ring、Windowed、MLA 共 7 大类 10+ 个 API
2. **深度硬件适配**：充分利用 Tensix 的 5 RISC-V 核实现 Reader/Writer/Compute 流水线
3. **内存效率**：中间结果保持在 L1（120 MB）中，避免 DRAM round-trip
4. **计算优化**：缩放融合到 exp、L1 累积规约、自定义多项式 exp 近似、稀疏掩码读取
5. **分布式支持**：Ring Attention 支持多设备 scaleout，与 AllGather 融合减少通信开销
6. **训练支持**：tt-train 中包含完整的前向/后向实现及分布式变体
7. **程序缓存**：通过 `compute_program_hash` 实现程序缓存，避免重复编译
8. **KV Chain Forwarding**：非因果路径支持 Unicast/Multicast 两种核间 KV 共享模式，减少 DRAM 带宽压力

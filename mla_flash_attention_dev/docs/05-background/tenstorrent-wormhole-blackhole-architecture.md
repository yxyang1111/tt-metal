# Tenstorrent Wormhole 与 Blackhole 架构深度解析

## 目录

1. [概述](#1-概述)
2. [Tensix 核心架构](#2-tensix-核心架构)
3. [Wormhole 架构详解](#3-wormhole-架构详解)
4. [Blackhole 架构详解](#4-blackhole-架构详解)
5. [Wormhole 与 Blackhole 对比](#5-wormhole-与-blackhole-对比)
6. [Network-on-Chip (NoC) 网络](#6-network-on-chip-noc-网络)
7. [内存层次结构](#7-内存层次结构)
8. [Tensix 指令与计算管线](#8-tensix-指令与计算管线)
9. [以太网互连与可扩展性](#9-以太网互连与可扩展性)
10. [产品形态](#10-产品形态)

---

## 1. 概述

Tenstorrent 是一家以 **RISC-V** 为核心架构的 AI 芯片公司，由 Jim Keller 担任 CEO。其芯片采用独特的 **Tensix 核心** 阵列设计，通过片上网络（NoC）互连，强调 **显式数据搬运**（无硬件缓存一致性）和 **确定性延迟**。

Tenstorrent 的芯片演进路线为：

| 代际 | 芯片名称 | 制程 | 状态 |
|------|---------|------|------|
| 第1代 | Grayskull | 12nm | 已量产（已停产） |
| 第2代 | **Wormhole** | 12nm | 量产中 |
| 第3代 | **Blackhole** | 6nm（TSMC） | 量产中 |

本文重点介绍 **Wormhole（B0 版本）** 和 **Blackhole** 两代架构。

---

## 2. Tensix 核心架构

Tensix 核心是 Tenstorrent 芯片的计算基本单元。每个 Tensix 核心是一个高度集成的 **近存计算单元**，包含 RISC-V 处理器、矩阵引擎、向量引擎和本地 SRAM。

### 2.1 内部结构

每个 Tensix 核心（T Tile）内部包含以下组件：

```
┌─────────────────────────────────────────────────────┐
│                   L1 SRAM (1464 KiB)                │
│            (地址空间从 0x00000000 开始)                │
├─────────────────────────────────────────────────────┤
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │BRISC │ │TRISC0│ │TRISC1│ │TRISC2│ │NCRISC│     │
│  │ (B)  │ │ (T0) │ │ (T1) │ │ (T2) │ │ (NC) │     │
│  │RV32IM│ │RV32IM│ │RV32IM│ │RV32IM│ │RV32IM│     │
│  │2KB   │ │4KB   │ │4KB   │ │4KB   │ │4KB   │     │
│  │ RAM  │ │ RAM  │ │ RAM  │ │ RAM  │ │ RAM  │     │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ │16KB  │     │
│     │     ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ │IRAM  │     │
│     │     │Tensix│ │Tensix│ │Tensix│ └──────┘     │
│     │     │Pipe 0│ │Pipe 1│ │Pipe 2│              │
│     │     └──┬───┘ └──┬───┘ └──┬───┘              │
│     │        └────┬────┘────┘                      │
│     │        ┌────┴────┐                           │
│     │        │Tensix   │ (8 互斥锁 + 8 信号量)      │
│     │        │  Sync   │                           │
│     │        └────┬────┘                           │
│  ┌──┴──┬────┬────┼────┬─────┬─────┬─────┬────┐    │
│  │Scala│ThCf│Unpa│Matr│ Pack│Vecto│ TDMA│Xmov│    │
│  │ r   │ g  │ ck │ ix │    │ r   │     │    │    │
│  │(ThCo│    │    │(FPU│    │(SFPU│     │    │    │
│  │ n)  │    │    │)   │    │)    │     │    │    │
│  └─────┴────┴────┴────┴─────┴─────┴─────┴────┘    │
│  ┌──────────┐ ┌──────┐ ┌──────┐ ┌────────┐        │
│  │  SrcA    │ │ SrcB │ │ Dst  │ │ Lreg   │        │
│  │  (4 KiB) │ │(4KiB)│ │(32Ki)│ │ (1KiB) │        │
│  └──────────┘ └──────┘ └──────┘ └────────┘        │
├─────────────────────────────────────────────────────┤
│                  NoC 接口 (双 NoC)                   │
└─────────────────────────────────────────────────────┘
```

### 2.2 五个 RISC-V 核心

每个 Tensix 核心包含 **5 个 RV32IM RISC-V 核心**（通常称为 "Baby RISC-V"）：

| 核心 | 别名 | 职责 | 核心本地 RAM |
|------|------|------|-------------|
| **B** | BRISC | 数据搬运核心 0（DM0），负责初始化、NoC 数据传输 | 2 KiB |
| **T0** | TRISC0 | 计算核心 MATH0，驱动 Unpack 单元 | 4 KiB |
| **T1** | TRISC1 | 计算核心 MATH1，驱动 Matrix/Vector 单元 | 4 KiB |
| **T2** | TRISC2 | 计算核心 MATH2，驱动 Pack 单元 | 4 KiB |
| **NC** | NCRISC | 数据搬运核心 1（DM1），NoC 数据传输 | 4 KiB + 16 KiB IRAM |

这 5 个核心完全**裸机运行**——没有中断、没有用户态/内核态分离、没有虚拟化。它们各自独立执行，但可通过硬件信号量和互斥锁进行同步。

### 2.3 Tensix 计算后端（8 个执行单元）

| 执行单元 | 功能 |
|---------|------|
| **Scalar (ThCon)** | 3×64 个 32-bit GPR，标量 ALU 运算，L1 读写 |
| **ThCfg** | 配置寄存器管理 |
| **Unpack** | 从 L1 解包数据到 SrcA/SrcB（支持多种数据格式） |
| **Matrix (FPU)** | 矩阵乘法引擎：**每周期执行 Dst[8,16] = SrcB[8,16] @ SrcA[16,16]**，包含 **2048 个乘法器**（7b × 5b） |
| **Pack** | 将 Dst 寄存器中的数据打包写回 L1 |
| **Vector (SFPU)** | **32 路 SIMD**，用于非线性激活函数等逐元素运算 |
| **TDMA** | Tile DMA 操作 |
| **Xmov** | 数据移动操作 |

### 2.4 寄存器文件

| 寄存器 | 大小 | 用途 |
|--------|------|------|
| **SrcA** | 4 KiB | Unpack 输出，Matrix 输入操作数 A |
| **SrcB** | 4 KiB | Unpack 输出，Matrix 输入操作数 B |
| **Dst** | 32 KiB | Matrix/Vector 输出，Pack 输入 |
| **Lreg** | 1 KiB (32 元素 × 32-bit) | SFPU 工作寄存器 |

### 2.5 支持的数据格式

Tensix 核心支持丰富的数据格式：

- **FP32** / **TF32** (TensorFloat-32)
- **FP16** / **BFloat16**
- **FP8** (E4M3 / E5M2)
- **Block Floating Point**: BLOCKFP2, BLOCKFP4, BLOCKFP8
- **INT8** / **UINT8**
- **INT32**

### 2.6 向量宽度

Wormhole 和 Blackhole 的向量引擎均为 **32 × 32-bit** 宽度（即每次处理 32 个 32-bit 元素）。

---

## 3. Wormhole 架构详解

### 3.1 芯片概览

| 参数 | 值 |
|------|-----|
| 制程 | 12nm GlobalFoundries |
| Die 面积 | ~670 mm² |
| AI 时钟 | 最高 1.0 GHz |
| Tensix 核心总数 | 80（8 × 10 阵列） |
| DRAM | 12 GB GDDR6（6 bank × 2 GB） |
| 片上 SRAM | ~120 MB（80 × 1.5 MB） |
| 以太网通道 | 16 × 100 Gbps |
| 峰值 FP8 算力 | 262 TFLOPS（单芯 N150） |

### 3.2 SoC 网格布局

Wormhole 的 SoC 采用 **10 × 12** 的 NoC 网格，包含以下 Tile 类型：

```
     x=0  x=1  x=2  x=3  x=4  x=5  x=6  x=7  x=8  x=9
y=0  [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ]
y=1  [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ]  (简化示意)
y=2  [Rtr] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ]
y=3  [PCIe][ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ] [ T ]
...
y=10 [ARC] [ E ] [ E ] [ E ] [ E ] [ E ] [ E ] [ E ] [ E ] [ E ]
y=11 [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ] [ D ]
```

| Tile 类型 | 数量 | 功能说明 |
|-----------|------|---------|
| **T (Tensix)** | 80 | 计算核心（8×10 阵列） |
| **D (DRAM)** | 18 | DRAM 控制器 NoC 端点（6 bank × 3 端口/bank） |
| **E (Ethernet)** | 16 | 以太网桥接核心，各含 1 个 RISC-V + 256 KB L1 |
| **ARC** | 1 | Argonaut RISC 核心，芯片管理 |
| **PCIe** | 1 | 主机 PCIe 桥接 |
| **Router** | 4 | NoC 路由节点 |

### 3.3 Tensix 核心细节

| 参数 | 值 |
|------|-----|
| L1 SRAM（硬件） | 1464 KiB（含分布式寄存器约 1.5 MB） |
| L1 SRAM（软件可见） | 1,499,136 B |
| 核心本地 RAM（5 核总计） | ~30 KiB（2+4+4+4+4 KiB） |
| NCRISC IRAM | 16 KiB |
| Matrix 乘法器 | 2048 个（7b × 5b） |
| Vector (SFPU) 宽度 | 32 lanes × 32-bit |
| 每核 RISC-V 核心 | 5 个 RV32IM |

### 3.4 DRAM 子系统

| 参数 | 值 |
|------|-----|
| DRAM 类型 | GDDR6 |
| DRAM 控制器数 | 6 |
| 每 bank 容量 | 2 GB（2 通道 × 1 GB） |
| 总容量 | 12 GB |
| 每 bank NoC 端口 | 3 |
| NoC 对齐（读） | 32 Bytes |
| NoC 对齐（写） | 16 Bytes |
| 带宽 | ~288 GB/s（单芯） |

### 3.5 Harvesting（良率收割）

Wormhole 仅支持 **按行收割 Tensix 核心**：
- 列数固定为 8
- 可收割 1～2 行（常见配置为 8×8 = 64 核用于计算）
- DRAM、PCIe、ARC、Ethernet、Router 核心**不支持收割**

### 3.6 NoC 参数

| 参数 | 值 |
|------|-----|
| NoC 数量 | 2（NOC0 / NOC1，坐标镜像映射） |
| Payload 宽度 | 256-bit (32 Bytes) |
| 最大突发字数 | 256 words |
| 虚通道 (VC) 数 | 16 |
| 广播 VC 起始 | VC 4 |
| 路由器端口 | 3（NIU / X / Y） |
| 最大软件包大小 | 8 KB |
| 多播形状 | 矩形 |

---

## 4. Blackhole 架构详解

### 4.1 芯片概览

| 参数 | 值 |
|------|-----|
| 制程 | **6nm TSMC** |
| AI 时钟（Busy） | **1.35 GHz** |
| AI 时钟（Idle） | 800 MHz |
| Tensix 核心总数 | **140**（14 × 10 阵列） |
| 可用计算核心 | **130**（13 × 10，1 列用于 dispatch） |
| DRAM | **32 GB GDDR6**（8 bank × ~4 GB） |
| 片上 SRAM | **~210 MB**（140 × 1.5 MB） |
| 以太网通道 | 14（通常固定收割 2 个，实际可用 12） |
| 峰值 FP8 算力 | **664–745 TFLOPS** |
| 板卡功耗 | 最高 300W |

### 4.2 SoC 网格布局

Blackhole 的 SoC 采用 **17 × 12** 的 NoC 网格，比 Wormhole 大幅扩展。以下布局基于
`blackhole_implementation.hpp` 中的实际 NOC0 坐标：

```
       x=0     x=1    x=2   ..  x=7    x=8     x=9     x=10  ..  x=16
y=0   [DRAM]  [Rtr]  [PCIe] .. [Rtr]  [ ARC ] [DRAM]   [Rtr]  .. [Rtr]
y=1   [DRAM]  [ETH]  [ETH]  .. [ETH]  [ Rtr ] [DRAM]   [ETH]  .. [ETH]
y=2   [DRAM]  [ T ]  [ T ]  .. [ T ]  [ SEC ] [DRAM]   [ T ]  .. [ T ]
y=3   [DRAM]  [ T ]  [ T ]  .. [ T ]  [L2CPU] [DRAM]   [ T ]  .. [ T ]
y=4   [DRAM]  [ T ]  [ T ]  .. [ T ]  [ Rtr ] [DRAM]   [ T ]  .. [ T ]
y=5   [DRAM]  [ T ]  [ T ]  .. [ T ]  [L2CPU] [DRAM]   [ T ]  .. [ T ]
y=6   [DRAM]  [ T ]  [ T ]  .. [ T ]  [ Rtr ] [DRAM]   [ T ]  .. [ T ]
y=7   [DRAM]  [ T ]  [ T ]  .. [ T ]  [L2CPU] [DRAM]   [ T ]  .. [ T ]
y=8   [DRAM]  [ T ]  [ T ]  .. [ T ]  [ Rtr ] [DRAM]   [ T ]  .. [ T ]
y=9   [DRAM]  [ T ]  [ T ]  .. [ T ]  [L2CPU] [DRAM]   [ T ]  .. [ T ]
y=10  [DRAM]  [ T ]  [ T ]  .. [ T ]  [ Rtr ] [DRAM]   [ T ]  .. [ T ]
y=11  [DRAM]  [ T ]  [ T ]  .. [ T ]  [ Rtr ] [DRAM]   [ T ]  .. [ T ]
```

- **x=0 列**：全部为 **DRAM bank 0-3** 的 NoC 端点（4 bank × 3 端口 = 12 个）
- **x=9 列**：全部为 **DRAM bank 4-7** 的 NoC 端点（4 bank × 3 端口 = 12 个）
- **x=8 列**：中央基础设施"脊柱"（ARC + Security + L2CPU + Router）
- **x=1-7, x=10-16**：Tensix 计算核心（y=2-11）和以太网核心（y=1）
- **y=0 行**：ARC (x=8) + PCIe (x=2, x=11) + Router (其余位置)

| Tile 类型 | 数量 | 位置（NOC0 坐标） |
|-----------|------|------------------|
| **Tensix** | 140 (14×10) | x∈{1-7, 10-16}, y∈{2-11} |
| **DRAM** | 24 端点 (8 bank × 3 端口) | x=0（bank 0-3）和 x=9（bank 4-7）整列 |
| **Ethernet** | 14 | y=1 行，x∈{1-7, 10-16}（不含 x=8） |
| **ARC** | 1 | **(8, 0)** |
| **PCIe** | 2 | **(2, 0)** 和 **(11, 0)** |
| **Security** | 1 | **(8, 2)** |
| **L2CPU** | 4 | **(8, 3)**, **(8, 5)**, **(8, 7)**, **(8, 9)** |
| **Router** | 18 | y=0 行（x∈{1,3-7,10,12-16}）+ x=8 列（y∈{1,4,6,8,10,11}） |

### 4.3 重大架构改进（相对 Wormhole）

#### 4.3.1 "Big" RISC-V 核心

Blackhole 新增 **16 个 SiFive X280 "Big" RISC-V 核心**（位于 L2CPU Tile），用于：
- 控制面处理
- 卸载主机 CPU 职责
- 通用计算任务

这是区别于 Wormhole 的关键创新——Wormhole 完全依赖主机 CPU 做控制面工作。

#### 4.3.2 L1 数据缓存

Blackhole 在 Tensix 核心的 L1 上新增了一个小型 **写穿数据缓存**：
- **4 × 16B cacheline**
- 默认禁用，可通过环境变量或 API 调用启用
- 跨核访问需要手动 invalidate
- 需要注意缓存一致性问题

#### 4.3.3 DRAM 核心可编程性

每个 DRAM bank 包含 **1 个 RISC-V 核心** + **128 KB L1**（runtime 尚未完全开放编程）。

#### 4.3.4 增强的以太网核心

每个以太网核心从 Wormhole 的 **1 个 RISC-V + 256 KB L1** 升级为 **2 个 RISC-V + 512 KB L1**。

### 4.4 Tensix 核心细节

| 参数 | 值 |
|------|-----|
| L1 SRAM（硬件） | 1464 KiB（与 Wormhole 相同） |
| L1 SRAM（软件可见） | 1,572,864 B (**1.5 MiB**) |
| L1 数据缓存 | 4 × 16B cacheline（写穿，默认禁用） |
| Matrix/Vector 宽度 | 32 × 32-bit（与 Wormhole 相同） |
| 每核 RISC-V 核心 | 5 个 RV32IM |

> 注意：L1 软件可见大小从 Wormhole 的 1,499,136 B 增加到 Blackhole 的 1,572,864 B（多出约 72 KiB）。

### 4.5 DRAM 子系统

| 参数 | 值 |
|------|-----|
| DRAM 类型 | GDDR6 |
| DRAM 控制器数 | **8** |
| 每 bank 容量 | **~4 GB**（SoC 可用 ~4,278,190,080 B，部分保留给 barrier） |
| 总容量 | **~32 GB** |
| 每 bank NoC 端口 | 3 |
| NoC 对齐（读） | **64 Bytes**（比 Wormhole 的 32B 翻倍） |
| NoC 对齐（写） | 16 Bytes |
| DRAM 核心 | 每 bank 1 个 RISC-V + 128 KB L1 |
| 带宽 | **~512 GB/s** |

### 4.6 Harvesting（良率收割）

Blackhole 的收割策略与 Wormhole 不同：
- Tensix 核心：**按列收割**（而非 Wormhole 的按行收割）
- DRAM：**最多收割约 1 个 bank**
- Ethernet：**固定收割 2 个核心**
- PCIe、ARC、Router：不支持收割

### 4.7 NoC 参数

| 参数 | 值 |
|------|-----|
| NoC 数量 | 2（NOC0 / NOC1） |
| 虚通道 (VC) 数 | 16 |
| 广播 VC 起始 | VC 4 |
| NoC 地址位宽 | 本地 36-bit + 节点 ID 6-bit |
| 最大软件包大小 | **16 KB**（Wormhole 为 8 KB） |
| L1 对齐（读/写） | 16 Bytes |
| DRAM/PCIe 对齐（读） | **64 Bytes** |
| DRAM/PCIe 对齐（写） | 16 Bytes |
| 多播形状 | **矩形 + 步进 (Strided) + L 形** |

> 关键差异：Blackhole 的 NoC 增加了**步进多播**和 **L 形多播**，比 Wormhole 仅支持矩形多播更灵活。

---

## 5. Wormhole 与 Blackhole 对比

### 5.1 核心规格对比

| 参数 | Wormhole B0 | Blackhole |
|------|------------|-----------|
| **制程** | 12nm GF | **6nm TSMC** |
| **AI 时钟** | ~1.0 GHz | **1.35 GHz** |
| **SoC NoC 网格** | 10 × 12 | **17 × 12** |
| **Tensix 总数** | 80 (8×10) | **140 (14×10)** |
| **可用计算核心** | 64 (8×8, N150) | **130 (13×10)** |
| **单核 L1 SRAM** | ~1464 KiB | ~1464 KiB + L1 Cache |
| **总片上 SRAM** | ~120 MB | **~210 MB** |
| **DRAM** | 12 GB (6×2GB) | **32 GB (8×~4GB)** |
| **DRAM 带宽** | ~288 GB/s | **~512 GB/s** |
| **Ethernet** | 16 × 100G | 14 × (更高速率) |
| **Big RISC-V** | 无 | **16 × SiFive X280** |
| **FP8 峰值** | 262 TFLOPS (N150) | **664-745 TFLOPS** |
| **功耗** | 160W (N150) | **300W** |

### 5.2 架构差异总结

| 维度 | Wormhole | Blackhole |
|------|----------|-----------|
| Tensix 收割方式 | 按行 | 按列 |
| DRAM 读对齐 | 32B | 64B |
| NoC 最大包 | 8 KB | 16 KB |
| 多播形状 | 仅矩形 | 矩形 + 步进 + L 形 |
| 以太网 RISC-V | 1 个/核 | 2 个/核 |
| 以太网 L1 | 256 KB | 512 KB |
| DRAM 可编程 | 不可 | 有 RISC-V（待开放） |
| L1 数据缓存 | 无 | 有（写穿，默认禁用） |
| TLB 大小 | 1 MB（静态） | 2 MB（静态） |

### 5.3 编程差异注意事项

从 Wormhole 迁移到 Blackhole 时需要注意：

1. **NoC Flush**：Blackhole 上 RISC-V 到 L1 的延迟比 NoC 延迟低，可能导致数据竞争。之前在 Wormhole 上省略的 NoC flush 在 Blackhole 上**必须显式执行**。

2. **对齐要求**：DRAM/PCIe 读取的对齐从 32B 变为 64B，现有 kernel 需要检查对齐是否满足。

3. **坐标系统**：网格从 10×12 扩展到 17×12，且 Tensix 核心在中间有"脊柱"间断（x=8,9），坐标映射逻辑不同。

4. **L1 缓存一致性**：如果启用了 L1 数据缓存，跨核通信时需手动 invalidate 缓存。

---

## 6. Network-on-Chip (NoC) 网络

### 6.1 双 NoC 拓扑

两代芯片均采用 **双 NoC** 设计（NOC0 和 NOC1），两个网络的坐标系互为镜像映射。双 NoC 的设计意图是：
- 提供双倍的总带宽
- 避免读/写方向冲突
- 常见模式：NOC0 用于读，NOC1 用于写（或反之）

### 6.2 NoC 操作类型

| 操作 | 说明 |
|------|------|
| **Unicast Read** | 从远端 Tile 读取数据到本地 L1 |
| **Unicast Write** | 将本地 L1 数据写入远端 Tile |
| **Multicast Write** | 将数据广播到一组 Tile（矩形区域） |
| **Atomic Increment** | 远端原子信号量操作（用于同步） |

### 6.3 NoC 地址编码

NoC 地址由以下部分组成：
- **节点 ID**（坐标 x, y）
- **本地地址**（Wormhole: 36-bit, Blackhole: 36-bit）

地址编码格式：`[NOC_Y][NOC_X][LOCAL_ADDR]`

### 6.4 对齐要求汇总

| 操作目标 | Wormhole 读/写 | Blackhole 读/写 |
|---------|---------------|-----------------|
| L1 ↔ L1 | 16B / 16B | 16B / 16B |
| DRAM | 32B / 16B | **64B** / 16B |
| PCIe | 32B / 16B | **64B** / 16B |

---

## 7. 内存层次结构

### 7.1 整体内存层次

```
┌───────────────────────────────────────────────┐
│              主机内存 (Host DRAM)               │
│                  (通过 PCIe)                    │
└───────────────────┬───────────────────────────┘
                    │ PCIe
┌───────────────────┴───────────────────────────┐
│           片外 DRAM (GDDR6)                    │
│   WH: 6 bank × 2GB = 12GB                     │
│   BH: 8 bank × ~4GB = ~32GB                   │
│         (通过 NoC 访问, 带宽 ~288-512 GB/s)     │
└───────────────────┬───────────────────────────┘
                    │ NoC
┌───────────────────┴───────────────────────────┐
│           Tensix L1 SRAM (片上)                │
│   每核: ~1464 KiB                              │
│   WH 总计: ~120 MB | BH 总计: ~210 MB          │
│   (本地访问延迟极低, 确定性)                     │
└───────────────────┬───────────────────────────┘
                    │
┌───────────────────┴───────────────────────────┐
│      Tensix 寄存器文件 (SrcA/SrcB/Dst/Lreg)    │
│   SrcA: 4KiB, SrcB: 4KiB, Dst: 32KiB         │
│   (与 FPU/SFPU 直连, 单周期访问)                │
└───────────────────────────────────────────────┘
```

### 7.2 L1 SRAM 内存映射（Tensix 核心）

```
0x00000000  ┌─────────────────────┐
            │   Kernel 代码与数据   │
            │  (~1432 KB 可用)      │
            │                     │
            ├─────────────────────┤
            │   CB (Circular      │
            │    Buffers)         │
            ├─────────────────────┤
            │   Runtime 保留区域   │
0x0016DFFF  └─────────────────────┘  (1464 KiB = 0x16E000)

0xFFB00000  ┌─────────────────────┐  配置/状态寄存器区域 (1 MiB)
            │  核心本地 RAM        │
            │  NoC 寄存器         │
            │  Tensix 配置寄存器   │
0xFFBFFFFF  └─────────────────────┘

0xFFC00000  ┌─────────────────────┐  NCRISC IRAM (16 KiB)
0xFFC03FFF  └─────────────────────┘

0xFFE00000  ┌─────────────────────┐  Tensix MMIO 区域
            │  ThCon GPR (0x00)   │
            │  Pipe MMIO (0x40000)│
            │  Semaphore (0x80020)│
            │  Config Regs (0xF0) │
0xFFEFFFFF  └─────────────────────┘
```

### 7.3 Circular Buffer（循环缓冲区）

tt-metal 编程模型的核心是 **Circular Buffer (CB)**。CB 位于 L1 SRAM 中，是生产者-消费者之间的数据通道：

- **Reader kernel** (BRISC/NCRISC)：从 DRAM/远端 L1 读取数据，写入 CB
- **Compute kernel** (TRISC0/1/2)：从 CB 读取数据，计算后写入输出 CB
- **Writer kernel** (BRISC/NCRISC)：从输出 CB 读取数据，写回 DRAM/远端 L1

这种模式实现了**数据搬运与计算的流水线并行**。

---

## 8. Tensix 指令与计算管线

### 8.1 指令管线架构

```
RISC-V 代码 / MMIO 写入
        │
        ▼
┌─────────────────┐
│ Tensix 指令管道   │ ← 32-bit 指令，与 RISC-V 指令集完全不同
│ (每 T 核一个)     │
├─────────────────┤
│ Macro-Op 展开器  │ ← MOP 指令展开为指令序列
├─────────────────┤
│ Replay 展开器    │ ← REPLAY 指令重放缓冲的指令
└────────┬────────┘
         ▼
┌─────────────────┐
│   Tensix Sync   │ ← 互斥锁/信号量同步
└────────┬────────┘
         ▼
  分发到 8 个后端执行单元
```

### 8.2 Macro-Op Expander（宏操作展开器）

`MOP` 指令是一种**指令压缩机制**，一条 MOP 指令可展开为一系列预编程的 Tensix 操作。通过 `MOP_CFG` 配置 9 个模板寄存器 (`mop_cfg[0..8]`)，MOP 支持两种模板：

- **Template 0**：条件分支 + 循环，基于 zmask 位选择不同的操作路径
- **Template 1**：嵌套双重循环，适用于矩阵乘法等规则计算

### 8.3 Replay Expander（重放展开器）

`REPLAY` 指令支持三种模式：
- **Record**：录制接下来的 N 条指令到缓冲区
- **Tee**：录制的同时也执行
- **Playback**：重放缓冲区中的指令

这允许将常用的指令序列录制一次后反复重放，减少指令带宽需求。

### 8.4 典型矩阵乘法流程

```
1. Unpack: 从 L1 → SrcA[16,16], SrcB[8,16]   (TRISC0 驱动)
2. Matrix: Dst[8,16] = SrcB[8,16] @ SrcA[16,16] (TRISC1 驱动, 2048 MAC/cycle)
3. (可选) Vector: 对 Dst 做非线性变换          (TRISC1 驱动)
4. Pack: Dst → L1                              (TRISC2 驱动)
```

三个 TRISC 核心可以**流水线重叠**执行——当 T1 在做矩阵计算时，T0 可以 Unpack 下一批数据，T2 可以 Pack 上一批结果。同步通过硬件信号量实现。

---

## 9. 以太网互连与可扩展性

### 9.1 芯片间互连

Tenstorrent 架构的一个关键特性是芯片通过 **以太网直连** 实现多芯片扩展，无需专用互连交换芯片。

| 参数 | Wormhole | Blackhole |
|------|----------|-----------|
| 以太网核心数 | 16 | 14（常固定收割 2 个） |
| 每通道速率 | 100 Gbps | 更高（4×800G QSFP-DD 端口） |
| 互连拓扑 | 点对点以太网 | 点对点以太网 |

### 9.2 多芯片系统

#### Wormhole 产品形态

| 产品 | 芯片数 | 互连 |
|------|--------|------|
| **N150** | 1 | PCIe 接入 |
| **N300** | 2 | 200G 片间互连 + PCIe |
| **T3K (TG Galaxy)** | 8 (4×N300) | 以太网 mesh |
| **TGG Galaxy** | 32 | 4×8 mesh 拓扑 |

#### Blackhole 产品形态

| 产品 | 芯片数 | 说明 |
|------|--------|------|
| **P100** | 1 | 基础单芯片卡 |
| **P150a/P150b** | 1 | 高配单芯片卡，120-140 Tensix |
| **P300** | 2 | 双芯片卡 |
| **Galaxy** | 32 | 4×8 mesh, 理论 ~23.8 PFLOPS FP8 |

### 9.3 以太网核心编程

以太网核心也是可编程的 RISC-V 核心，分为两种模式：
- **Active Ethernet**：正在参与芯片间通信
- **Idle Ethernet**：未被通信使用，可用于其他计算任务（如 dispatch）

在 Blackhole 上，idle 以太网核心的**第二个 RISC-V** 也已可用。

---

## 10. 产品形态

### 10.1 Wormhole 板卡规格

| 规格 | n150d/n150s | n300d/n300s |
|------|-------------|-------------|
| Wormhole ASIC 数量 | 1 | 2 |
| Tensix 核心 | 72 | 128 (64/芯片) |
| 片上 SRAM | 108 MB | 192 MB |
| GDDR6 | 12 GB | 24 GB |
| 内存带宽 | 288 GB/s | 576 GB/s |
| FP8 算力 | 262 TFLOPS | 466 TFLOPS |
| FP16 算力 | 74 TFLOPS | 131 TFLOPS |
| 主机接口 | PCIe Gen4 x16 | PCIe Gen4 x16 |
| 板卡功耗 | 160W | 300W |
| 外形 | d=双槽, s=单槽 | d=双槽, s=单槽 |

### 10.2 Blackhole 板卡规格

| 规格 | p150a | p150b |
|------|-------|-------|
| Blackhole ASIC | 1 | 1 |
| Tensix 核心 | 120 | 120 |
| 片上 SRAM | 180 MB | 180 MB |
| GDDR6 | 32 GB | 32 GB |
| 内存带宽 | 512 GB/s | 512 GB/s |
| BLOCKFP8 算力 | 664 TFLOPS | 664 TFLOPS |
| AI 时钟 | 1.35 GHz | 1.35 GHz |
| 板卡功耗 | 300W | 300W |

---

## 附录 A：关键源码文件索引

以下为 tt-metal 仓库中与架构相关的关键文件：

### Wormhole

| 用途 | 路径 |
|------|------|
| SoC 拓扑 YAML | `tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml` |
| 核心配置 | `tt_metal/hw/inc/internal/tt-1xx/wormhole/core_config.h` |
| NoC 参数 | `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` |
| 内存映射 | `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` |
| UMD 实现 | `tt_metal/third_party/umd/device/api/umd/device/arch/wormhole_implementation.hpp` |

### Blackhole

| 用途 | 路径 |
|------|------|
| SoC 拓扑 YAML | `tt_metal/soc_descriptors/blackhole_140_arch.yaml` |
| 核心配置 | `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h` |
| NoC 参数 | `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h` |
| UMD 实现 | `tt_metal/third_party/umd/device/api/umd/device/arch/blackhole_implementation.hpp` |
| Bring-Up 指南 | `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md` |

### 通用文档

| 用途 | 路径 |
|------|------|
| 坐标系统说明 | `tt_metal/third_party/umd/docs/COORDINATE_SYSTEMS.md` |
| 内存开发者指南 | `docs/source/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.rst` |
| DRAM 分配器 | `tech_reports/memory/allocator.md` |
| Galaxy 指南 | `docs/source/tech_reports/WH_Galaxy/Galaxy_WH_6U_SW_Guide.md` |

---

## 附录 B：参考资料

- [Tenstorrent Blackhole 官方页面](https://tenstorrent.com/hardware/blackhole)
- [Tenstorrent Wormhole 规格](https://docs.tenstorrent.com/aibs/wormhole/specifications.html)
- [Tenstorrent Blackhole 规格](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [Corsix: Wormhole 系列拆解](https://www.corsix.org/content/tt-wh-part5) — Tensix T Tile 深度分析
- [Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf) — ASPLOS 论文
- [tt-isa-documentation (GitHub)](https://github.com/tenstorrent/tt-isa-documentation) — 官方 ISA 文档

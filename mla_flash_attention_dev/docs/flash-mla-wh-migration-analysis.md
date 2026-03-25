# 实验性 FlashMLA (实现 B) 从 Blackhole 迁移到 Wormhole 的可行性分析与方案

## 1. 执行摘要

**结论：迁移可行，但需要重大的架构适配**。核心计算逻辑（Online Flash Attention、Tree Reduction、V-from-K）完全通用，无架构依赖。主要挑战在于 **S Block 网格重新设计**（WH 只有 8 列 / 6 DRAM bank vs BH 的 11 列 / 8 DRAM bank）以及若干 NoC/硬件参数调整。

预估工作量：中等偏大，主要集中在 `op.py` 的 S Block 定义、核心映射和参数调整，内核代码 (`flash_mla.hpp`) 改动较小。

---

## 2. 现有实现 B 对 Blackhole 的具体依赖分析

### 2.1 硬编码的 BH 依赖项

| # | 依赖项 | 位置 | BH 值 | WH 值 | 影响程度 |
|---|--------|------|-------|-------|---------|
| 1 | **网格尺寸断言** `device_grid.x >= 11, y == 10` | `op.py:390-391` | 11×10 | **8×10** (满配) 或 8×8 (收割) | **阻断性** — 必须修改 |
| 2 | **S Block 布局** 8 个 4col×2row 块 | `op.py:73-82` (FlashMLAOptimalGridNOC0.BLOCKS) | 11 列空间，cols {0-3, 7-10} | 仅 8 列 {0-7} | **阻断性** — 必须重新设计 |
| 3 | **DRAM bank 数** 8 bank | S Block↔bank 映射, ND sharding | 8 bank | **6 bank** | **阻断性** — S Block 从 8 降到 6 |
| 4 | **NoC 最大包大小** | `op.py:26` `get_noc_max_page_size()` | 16384 (16KB) | **8192 (8KB)** | 中 — 需修改函数 |
| 5 | **DRAM 读对齐** | 内核隐含 | 64B | **32B** | 低 — 方向兼容（32B 对齐是 64B 的子集） |
| 6 | **L1 数据缓存** | `flash_mla.hpp:477` `invalidate_l1_cache()` | 有 L1 cache，需 invalidate | **无 L1 cache** | 低 — WH 上为 no-op |
| 7 | **NOC_MAX_TRANSACTION_ID** | `flash_mla.hpp:295` trid 流水线 | 需验证 BH 值 | 需验证 WH 值 | 中 — 可能影响流水深度 |
| 8 | **物理坐标映射** | `physical_multicast_coords()` 等 | BH 逻辑→物理映射 | WH 映射不同 | 低 — 已用 `worker_core_from_logical_core` 抽象 |

### 2.2 与架构无关的部分（无需修改）

| 组件 | 说明 |
|------|------|
| Online Flash Attention 计算 | TRISC 上的 QK matmul → scale → softmax → AV matmul 完全通用 |
| Tree Reduction 合并逻辑 | `sdpa_tail` 合并 (O, m, l) 的算法与硬件无关 |
| V-from-K 策略 | 从 K buffer 中 strided 读取 V 列，无架构依赖 |
| CB (Circular Buffer) 机制 | WH/BH 的 CB API 完全一致 |
| 信号量同步机制 | `noc_semaphore_*` 系列 API 两平台一致 |
| Tiny Tile (8×32) | WH 也支持自定义 tile 尺寸 |
| 双缓冲 K chunk | 通过 CB 实现，无硬件依赖 |
| NCRISC-BRISC 共享 L1 同步 | 两平台核内 L1 共享机制相同 |
| DM_DYNAMIC_NOC 模式 | WH 也支持 |

---

## 3. 核心挑战：WH DRAM 拓扑 vs BH DRAM 拓扑

### 3.1 BH DRAM 布局（当前实现）

```
BH: 8 DRAM bank，对称分布在芯片左右两侧

     DRAM 列              Worker 区域 (11 col)              DRAM 列
     bank 0-3                                              bank 4-7
     (phys x=0)      cols 0  1  2  3  4  5  6  7  8  9 10  (phys x=9)
                      ← S1-S4 →        gap        ← S5-S8 →
```

- **优点**: 左右各 4 个 bank，每侧 4 列 worker 核，天然对称
- 每个 S Block: 4 col × 2 row = 8 cores，离对应 DRAM bank ≤ 1 hop

### 3.2 WH DRAM 布局

```
WH: 6 DRAM bank，不对称分布

     DRAM 列         Worker 区域 (8 col)           DRAM 列
     bank 0-1                                     bank 2-5
     (phys x=0)    cols 0  1  2  3  4  5  6  7    (phys x=5)
                   ← 4 cols →        ← 4 cols →
```

WH 的 6 个 DRAM bank 子通道在 NOC0 坐标系中的分布：

| Bank | 子通道 (phys x,y) | 最近 Worker 列 (logical) | 最近 Worker 行 (logical) |
|------|------------------|------------------------|------------------------|
| 0 | (0,0), (0,1), (0,11) | x=0 (phys 1) | y=0 (phys 1), y=9 (phys 11) |
| 1 | (0,5), (0,6), (0,7) | x=0 (phys 1) | y=4 (phys 5), y=5 (phys 7) |
| 2 | (5,0), (5,1), (5,11) | x=3,4 (phys 4,6) | y=0 (phys 1), y=9 (phys 11) |
| 3 | (5,2), (5,9), (5,10) | x=3,4 (phys 4,6) | y=1 (phys 2), y=7 (phys 9), y=8 (phys 10) |
| 4 | (5,3), (5,4), (5,8) | x=3,4 (phys 4,6) | y=2 (phys 3), y=3 (phys 4), y=6 (phys 8) |
| 5 | (5,5), (5,6), (5,7) | x=3,4 (phys 4,6) | y=4 (phys 5), y=5 (phys 7) |

**关键差异**：
1. Bank 数从 8 降到 6 → S Block 数量从 8 降到 6
2. Bank 0-1 仅在左边缘 (phys x=0)，最近 worker 列仅 1 列 (logical x=0)
3. Bank 2-5 **全部**在同一列 (phys x=5)，需要共享 logical cols {3, 4}

### 3.3 WH Worker 核心完整网格 (逻辑坐标 → 物理坐标)

```
逻辑 y:    0    1    2    3    4    5    6    7    8    9
物理 y:    1    2    3    4    5    7    8    9   10   11
           ──────────────────────────────────────────────
逻辑 x=0 (phys 1): T    T    T    T    T    T    T    T    T    T
逻辑 x=1 (phys 2): T    T    T    T    T    T    T    T    T    T
逻辑 x=2 (phys 3): T    T    T    T    T    T    T    T    T    T
逻辑 x=3 (phys 4): T    T    T    T    T    T    T    T    T    T
       (phys 5 = DRAM 列，无 worker)
逻辑 x=4 (phys 6): T    T    T    T    T    T    T    T    T    T
逻辑 x=5 (phys 7): T    T    T    T    T    T    T    T    T    T
逻辑 x=6 (phys 8): T    T    T    T    T    T    T    T    T    T
逻辑 x=7 (phys 9): T    T    T    T    T    T    T    T    T    T

共 80 核 (8 × 10)
```

---

## 4. 迁移方案

### 4.1 WH S Block 设计（6 Block, 每块 4 核）

基于 DRAM 邻近性原则，设计 6 个 S Block，每块 2 列 × 2 行 = 4 核：

```
逻辑坐标网格:

y:    0    1    2    3    4    5    6    7    8    9
    ┌────────────────────────────────────────────────┐
x=0 │ S1   ·    ·    ·   S2   S2    ·    ·    ·   S1 │  ← Bank 0/1
x=1 │ S1   ·    ·    ·   S2   S2    ·    ·    ·   S1 │
x=2 │  ·    ·    ·    ·    ·    ·    ·    ·    ·    ·  │  (空闲)
x=3 │ S3   S4   S5   S5   S6   S6   S5   S4   S4   S3 │  ← Bank 2-5
x=4 │ S3   S4   S5   S5   S6   S6   S5   S4   S4   S3 │  (注: 待调整)
x=5 │  ·    ·    ·    ·    ·    ·    ·    ·    ·    ·  │  (空闲)
x=6 │  ·    ·    ·    ·    ·    ·    ·    ·    ·    ·  │  (空闲)
x=7 │  ·    ·    ·    ·    ·    ·    ·    ·    ·    ·  │  (空闲)
    └────────────────────────────────────────────────┘
```

精确定义：

```python
class FlashMLAOptimalGridNOC0_WH:
    """WH B0 S Block 布局 — 6 Block × 4 cores = 24 核"""

    BLOCKS = (
        # S1 → Bank 0 (phys x=0, subchannels at phys y={0,1,11})
        # Logical: cols {0,1}, rows {9,0} (torus 环绕, phys y=11→1)
        (((0, 9), (1, 9), (0, 0), (1, 0)), 0),

        # S2 → Bank 1 (phys x=0, subchannels at phys y={5,6,7})
        # Logical: cols {0,1}, rows {4,5} (phys y=5→7)
        (((0, 4), (1, 4), (0, 5), (1, 5)), 1),

        # S3 → Bank 2 (phys x=5, subchannels at phys y={0,1,11})
        # Logical: cols {3,4}, rows {9,0} (torus 环绕)
        (((3, 9), (4, 9), (3, 0), (4, 0)), 2),

        # S4 → Bank 3 (phys x=5, subchannels at phys y={2,9,10})
        # Logical: cols {3,4}, rows {7,8} (phys y=9→10)
        (((3, 7), (4, 7), (3, 8), (4, 8)), 3),

        # S5 → Bank 4 (phys x=5, subchannels at phys y={3,4,8})
        # Logical: cols {3,4}, rows {2,3} (phys y=3→4)
        (((3, 2), (4, 2), (3, 3), (4, 3)), 4),

        # S6 → Bank 5 (phys x=5, subchannels at phys y={5,6,7})
        # Logical: cols {3,4}, rows {4,5} (phys y=5→7)
        (((3, 4), (4, 4), (3, 5), (4, 5)), 5),
    )

    NUM_BLOCKS = 6
    CORES_PER_BLOCK = 4

    OPTIMAL_DRAM_BANK_ORDER = (0, 1, 2, 3, 4, 5)

    # 3-步 Tree Reduction (6 → 3 → 2 → 1)
    TREE_REDUCTION_ORDER = (
        ((0, 1), (2, 3), (4, 5)),  # Step 1: S2→S1, S4→S3, S6→S5
        ((0, 2),),                  # Step 2: S3→S1
        ((0, 4),),                  # Step 3: S5→S1
    )
    NUM_TREE_REDUCTION_STEPS = 3
```

每个 S Block 的 DRAM 距离验证：

| S Block | 逻辑列 | 物理列 | DRAM Bank phys-x | X 距离 (hops) | Y 距离 (hops) |
|---------|--------|--------|-------------------|---------------|---------------|
| S1 | 0,1 | 1,2 | 0 | 1-2 | ≤1 |
| S2 | 0,1 | 1,2 | 0 | 1-2 | ≤1 |
| S3 | 3,4 | 4,6 | 5 | 1 | ≤1 |
| S4 | 3,4 | 4,6 | 5 | 1 | ≤1 |
| S5 | 3,4 | 4,6 | 5 | 1 | ≤1 |
| S6 | 3,4 | 4,6 | 5 | 1 | ≤1 |

### 4.2 Multicast 矩形验证

WH 仅支持矩形 multicast。需验证每个 S Block 在**物理坐标系**中构成有效矩形：

| S Block | 物理核坐标 | 矩形范围 | 是否有效矩形 | 中间非 worker 节点 |
|---------|-----------|---------|-------------|-------------------|
| S1 | (1,11),(2,11),(1,1),(2,1) | x=[1,2], y=11→1 (torus wrap) | ✅ Torus 矩形 | 包含 y=0 (DRAM) |
| S2 | (1,5),(2,5),(1,7),(2,7) | x=[1,2], y=[5,7] | ✅ 矩形 | 包含 y=6 (Ethernet) |
| S3 | (4,11),(6,11),(4,1),(6,1) | x=[4,6], y=11→1 (torus wrap) | ✅ Torus 矩形 | 包含 x=5 (DRAM), y=0 (DRAM) |
| S4 | (4,9),(6,9),(4,10),(6,10) | x=[4,6], y=[9,10] | ✅ 矩形 | 包含 x=5 (DRAM) |
| S5 | (4,3),(6,3),(4,4),(6,4) | x=[4,6], y=[3,4] | ✅ 矩形 | 包含 x=5 (DRAM) |
| S6 | (4,5),(6,5),(4,7),(6,7) | x=[4,6], y=[5,7] | ✅ 矩形 | 包含 x=5 (DRAM), y=6 (Eth) |

> **注意**：multicast 矩形中包含非 worker 节点 (DRAM、Ethernet、Router) 是安全的——这些节点会收到数据但不会处理，仅略增 NoC 流量。这在 BH 上同理。

### 4.3 Torus 环绕 Multicast 风险分析

S1 和 S3 使用 torus 环绕（y 从 11 到 1）。这在 WH 上需要特别验证：

**WH NoC 是 2D Torus**，multicast 矩形规范是 `(start_x, start_y)` 到 `(end_x, end_y)`。当 `start_y > end_y` 时，NoC 硬件会环绕传播。这在 BH 上已被 S4/S8 验证可行，WH 理论上应支持相同机制。

**但需要实测验证**：建议编写一个小的 multicast 单元测试，从 `(1,11)` 发送到 `(2,1)` 的矩形区域，确认 torus wrap multicast 在 WH 上正常工作。如果不可行，需要将 S1/S3 改为不跨越边界的布局，例如：
- S1 替代方案: cols {0,1}, rows {0,1} → (0,0),(1,0),(0,1),(1,1)
- S3 替代方案: cols {3,4}, rows {0,1} → (3,0),(4,0),(3,1),(4,1)

### 4.4 备选方案：增大 S Block（6 Block × 6 核）

如果需要更高的序列并行度，可以将 S Block 扩展为 3 列 × 2 行 = 6 核：

```python
# 左侧 S Block 扩展到 3 列
S1 = ((0,9),(1,9),(2,9),(0,0),(1,0),(2,0)), bank=0  # 第 3 列距 DRAM 2 hop
S2 = ((0,4),(1,4),(2,4),(0,5),(1,5),(2,5)), bank=1

# 右侧 S Block 仍用 2 列 (cols {3,4} 是距中央 DRAM 最近的)
# 但为匹配 6 核需增加行数: 2 列 × 3 行
S3 = ((3,9),(4,9),(3,0),(4,0),(3,1),(4,1)), bank=2
...
```

这种方案提供 36 个活跃核，但存在两个问题：
1. **非矩形 S Block**（如 2 列 × 3 行会包含更多中间节点）
2. **S Block 间行复用风险**（需仔细排列避免重叠）

**推荐先用 4 核版本验证功能正确性**，再根据性能需求扩展。

---

## 5. 详细代码修改清单

### 5.1 `op.py` 修改

| 修改项 | 当前代码 | 目标代码 | 优先级 |
|--------|---------|---------|--------|
| 网格断言 | `device_grid.x >= 11, y == 10` | 参数化: `x >= 8, y >= 8` | P0 |
| S Block 定义 | `FlashMLAOptimalGridNOC0.BLOCKS` (8 块) | 新增 `FlashMLAOptimalGridNOC0_WH.BLOCKS` (6 块) | P0 |
| DRAM bank 映射 | `OPTIMAL_DRAM_BANK_ORDER = (1,3,2,0,5,7,6,4)` | `(0,1,2,3,4,5)` | P0 |
| Tree Reduction | 8→1 的 3-step tree | 6→1 的 3-step tree | P0 |
| `get_noc_max_page_size()` | `return 16384` | WH: `return 8192`, BH: `return 16384` | P0 |
| Grid 选择 | 硬编码 `FlashMLAOptimalGridNOC0` | 根据 `device.arch()` 选择 WH/BH grid | P0 |
| `full_grid_mcast` 范围 | 基于 11×10 全网格 | 基于 8×10 (或 8×8) 全网格 | P0 |
| Q shard 数验证 | `B <= cores_per_s_block` (≤8) | `B <= cores_per_s_block` (≤4) | P1 |
| VC (虚通道) 分配 | 基于 8 个 S Block | 基于 6 个 S Block 重新分配 | P1 |

### 5.2 `flash_mla.hpp` 内核修改

| 修改项 | 当前代码 | 目标代码 | 优先级 |
|--------|---------|---------|--------|
| `invalidate_l1_cache()` | 无条件调用 | 用 `#ifdef ARCH_BLACKHOLE` 条件编译，或保留为 no-op | P1 |
| trid 流水线深度 | `NOC_MAX_TRANSACTION_ID - 1` | 验证 WH 上 `NOC_MAX_TRANSACTION_ID` 的值，可能需要调整窗口大小 | P1 |
| NoC 索引分配 | `READ_NOC_INDEX=0, MCAST_NOC_INDEX=0, WRITE_NOC_INDEX=1` | 验证 WH 上是否需要调整 NoC 0/1 分配 | P1 |

### 5.3 `rt_args_common.hpp` — 无需修改

`get_runtime_args()` 和 `get_device_mla_work_assignment()` 完全通用，无架构依赖。

### 5.4 `flash_mla_kernel.cpp` — 无需修改

仅作为 args 解包入口，无架构依赖。

---

## 6. 性能影响预估

### 6.1 定量对比

| 指标 | BH (当前) | WH (迁移后, 4核/block) | WH (迁移后, 6核/block) | 比率 |
|------|----------|----------------------|----------------------|------|
| S Block 数 | 8 | 6 | 6 | 0.75× |
| 每块核数 | 8 | 4 | 6 | 0.5-0.75× |
| 总活跃核 | 64 | 24 | 36 | 0.37-0.56× |
| DRAM 并行读取流 | 8 | 6 | 6 | 0.75× |
| 单 bank DRAM 带宽 | ~64 GB/s | ~48 GB/s | ~48 GB/s | 0.75× |
| 总 DRAM 带宽 | ~512 GB/s | ~288 GB/s | ~288 GB/s | 0.56× |
| Tree Reduction 步数 | 3 | 3 | 3 | 1.0× |
| 每核处理 K chunk 数 | S/8 | S/6 | S/6 | 1.33× |
| 最大 Q shard 数 | 8 | 4 | 6 | 0.5-0.75× |

### 6.2 瓶颈分析

**DRAM 带宽受限场景**（长序列 decode）：
- WH 总 DRAM 带宽为 BH 的 56%
- 但 WH 的计算峰值也更低 (262 vs 664+ TFLOPS)
- DRAM 带宽 / 计算能力 比值: WH ≈ 288/262 ≈ 1.1 B/FLOP, BH ≈ 512/664 ≈ 0.77 B/FLOP
- **WH 的算力/带宽比实际更偏带宽**，说明 DRAM 密集型的 FlashMLA 在 WH 上的"相对效率"可能反而更好

**计算受限场景**（短序列）：
- 每个 S Block 只有 4-6 核，序列并行度下降
- 但短序列本身 chunk 数少，4-6 核可能已足够

**Dispatch 开销**：
- 实现 B 的优势（单次 generic_op dispatch）在 WH 上同样有效
- 相比实现 A 的 ~18 次 dispatch, 减少 3-5ms 固定开销

### 6.3 与实现 A (生产路径) 在 WH 上的对比

| 维度 | 实现 A (WH 生产) | 实现 B 迁移 (WH) |
|------|------------------|------------------|
| DRAM bank 争用 | 多核读同一 bank（NKV=1 时严重） | 每 bank 仅 1 核读 + multicast |
| Dispatch 开销 | ~18 次 × 0.15-0.3ms | 1 次 × 0.15-0.3ms |
| 序列并行核数 | 4 (典型 MLA decode) | 6 (一个 S Block 来自每个 bank) |
| K multicast | 仅 MLA 列主模式下列方向 multicast | S Block 内矩形 multicast |
| 页级流水线 | 无 (chunk 级) | 有 trid 流水线 |
| V 解压缩 | 独立 dispatch (wkv_b2) | 可内部处理 |

**结论**：即使在 WH 上，实现 B 的 DRAM 效率和 dispatch 开销优势依然显著。

---

## 7. 实施路线图

### Phase 0: 验证基础设施 (1-2 天)

1. **Torus wrap multicast 测试**
   - 在 WH 上编写单元测试，验证 multicast 矩形 `(1,11)→(2,1)` 的 torus 环绕是否正常
   - 如果失败，使用不跨边界的 S1/S3 替代布局

2. **NOC_MAX_TRANSACTION_ID 确认**
   - 在 WH 的 `noc_parameters.h` 中查找或通过测试确认 trid 最大值
   - 如果与 BH 不同，调整 `NUM_TRIDS` 计算

3. **ND Sharding on WH 验证**
   - 确认 `ttnn.NDShardSpec` 在 WH 上可用且支持 6-bank round-robin
   - 参考现有的 WH ND sharding 测试用例

### Phase 1: Core 迁移 (3-5 天)

1. **创建 `FlashMLAOptimalGridNOC0_WH` 类**
   - 定义 6 个 S Block（4 核/块），含 DRAM bank 映射和 tree reduction 拓扑

2. **参数化 `op.py`**
   - 根据 `device.arch()` 选择 WH/BH grid 类
   - 调整 `get_noc_max_page_size()` 为架构感知
   - 更新网格断言为参数化的

3. **调整内核编译参数**
   - `invalidate_l1_cache()` 条件化
   - 验证 trid 流水线在 WH 上工作

4. **编写 WH 单元测试**
   - 移植 `test_flash_mla.py` 为 WH 版本
   - 从小规模 (batch=1, seq_len=128, 64 heads) 开始

### Phase 2: 功能验证 (2-3 天)

1. 逐步增加序列长度测试: 128 → 1024 → 4096 → 32768
2. 验证数值精度 (PCC > 0.999 vs golden reference)
3. 多 batch 测试 (batch=1,2,4)

### Phase 3: 性能调优 (3-5 天)

1. 尝试 6 核/块版本，对比 4 核/块性能
2. 测量实际 DRAM 带宽利用率
3. 优化 VC 分配避免 NoC 拥塞
4. 与实现 A 在 WH 上进行端到端性能对比

### Phase 4: 集成 (2-3 天)

1. 将 WH 版本集成到 DeepSeek V3 推理流水线
2. 端到端延迟验证
3. 收割场景 (8×8) 适配

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| WH Torus wrap multicast 不工作 | 低 | 高 | 使用不跨边界的 S Block 布局替代 |
| ND Sharding 在 WH 上不可用 | 低 | 高 | 回退到 Interleaved DRAM + 手动 bank 映射 |
| NOC_MAX_TRANSACTION_ID 差异导致 trid 不工作 | 中 | 中 | 减少 trid 窗口大小或回退到同步读取 |
| 4 核/块性能不足 | 中 | 中 | 扩展到 6 核/块（增加 2 列工作核） |
| WH 收割后仅 8×8=64 核 | 确定 | 低 | 24 核方案仍在 64 核范围内，无影响 |
| 某些 CB/信号量 API 在 WH 上行为不同 | 低 | 中 | 通过单元测试逐步验证 |

---

## 9. 结论

实验性 FlashMLA (实现 B) 的核心算法和编程模式在 WH 上完全可行。主要工作集中在：

1. **S Block 重新设计**（6 Block × 4 核 → 24 核，适配 WH 的 6-bank 不对称 DRAM 拓扑）
2. **NoC 参数调整**（最大包大小、对齐等）
3. **架构选择逻辑**（在 Python 层根据设备类型选择对应的 grid 配置）

其核心优势——DRAM 单流读取 + multicast 扩散、trid 流水线、Tiny Tile、单 dispatch——在 WH 上均可保留，预期能显著改善 WH 上 MLA decode 的性能（尤其是消除 DRAM bank 争用和减少 dispatch 开销）。

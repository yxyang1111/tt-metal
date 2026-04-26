# [Archived] TT-Metal Flash Attention 当前数据传输模式与并行划分

> 已归档（2026-04-23）。
> 这份文档在最近一个月内未继续更新，保留作历史背景参考。
> 当前默认不作为 `docs/` 根目录主入口；如需当前主线，请优先参考 `current-docs.md`。

本文基于当前仓库里的 `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/` 实现整理，重点回答两个问题：

1. Flash Attention 在 TT 上到底怎样切分并行度。
2. Q / K / V / mask / output 分别怎样在 DRAM、L1、NoC 之间流动。

本文主要聚焦标准 prefill 路径 `ttnn.transformer.scaled_dot_product_attention`，并补充说明 `chunked/paged prefill` 和 `FlashDecode` 与主线实现的差异。

## 1. 先给结论

当前 TT-Metal Flash Attention 可以先用一句话概括：

- 一个 SDPA program 会跑在一块 `compute_with_storage_grid_size` 指定的 2D core grid 上。
- 每个 core 同时运行三个 kernel：`reader`、`compute`、`writer`。
- host 侧先按 `batch -> q_head -> Q chunks` 的优先顺序切分并行度。
- `Q` 基本总是每个 core 自己读。
- `K/V` 在 `causal` 或 `chunked/paged` 场景下也基本是每个 core 自己读。
- 只有在 `non-causal` 且 `!is_chunked` 时，才会尝试启用 `KV forwarding`，把同一份 `K/V chunk` 在多个 core 之间复用。
- 这条 forwarding 路径又分成 `unicast`、`multicast`、`auto`、`hybrid` 四类行为。

可以先把当前实现记成下面这张表：

| 路径 | Q 的来源 | K/V 的来源 | 核间通信 | 备注 |
| --- | --- | --- | --- | --- |
| `causal prefill` | 每核本地读取 | 每核本地读取，只读到当前 Q chunk 需要的 KV 范围 | 无 | 主线、最简单 |
| `non-causal prefill`，无 forwarding | 每核本地读取 | 每核本地读取全部需要的 K/V | 无 | 同一 head 落在单核时就是这样 |
| `non-causal prefill` + `unicast` | 每核本地读取 | injector 从 DRAM 读，沿链逐跳单播给后继 core | 有，点对点 | 允许链上 `q_chunk_count` 不一致 |
| `non-causal prefill` + `multicast` | 每核本地读取 | injector 从 DRAM 读，一次多播给一组 receiver | 有，一对多 | 需要严格满足拓扑条件 |
| `chunked/paged prefill` | 每核本地读取当前 Q chunk | 每核经 `page_table` 读 paged K/V | 无 forwarding | 代码里明确不支持 paged forwarding |
| `FlashDecode` | Q 很小，通常整块处理 | 每核处理一段 KV cache | 有，但不是 chain forwarding，而是树形归约 | 独立实现，不走 prefill 这套链路 |

## 2. 代码入口与角色划分

主线代码位置：

- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
- `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`

这套实现的分工很清晰：

- `program_factory`
  - 决定 grid 大小、chunk 大小、并行划分
  - 创建 circular buffers
  - 配置 compile-time args 和 runtime args
  - 在 `non-causal` 情况下构建 KV forwarding 的 chain / mcast 拓扑
- `reader`
  - 负责把 `Q/K/V/mask/page_table/attention_sink` 搬到 L1
  - 在需要时执行 `K/V` 的单播或多播转发
- `compute`
  - 执行 FlashAttention-2 风格的在线 softmax 主循环
  - 在 L1 中维护 `prev_max / prev_sum / out_accumulator`
- `writer`
  - 生成隐式 mask
  - 把最终输出从 L1 写回 output tensor

## 3. 一个 core 内部的数据流

### 3.1 三个 kernel 的协作方式

每个 Tensix core 内部不是单线程串行做完读、算、写，而是三条并行流水：

```text
Reader  (RISC0): 读 Q / K / V / mask / page_table / attention_sink
Compute (RISC2-4): 做 QK、在线 softmax、P@V、归一化
Writer  (RISC1): 生成隐式 mask，并把 OUT 写回 DRAM
```

它们通过 L1 里的 circular buffer 解耦，因此可以形成读算写重叠。

### 3.2 关键 circular buffer

当前主线最值得记住的 CB 有这些：

| CB | 作用 |
| --- | --- |
| `c_0` | `Q chunk` |
| `c_1` | `K chunk` |
| `c_2` | `V chunk` |
| `c_3` | `mask` |
| `c_4` | `attention_sink` |
| `c_5` | 标量 identity / scale 辅助 |
| `c_6` | `page_table`，仅 chunked/paged prefill 使用 |
| `c_7` | 列方向 identity，供归约和广播辅助 |
| `c_8` | `chunk_start_idx` 给 compute 读 |
| `c_9` | `chunk_start_idx` 给 writer 读 |
| `c_16` | 最终输出 |
| `c_24` | `QK` 中间结果 |
| `c_25/c_26` | 输出累积的 ping-pong buffer |
| `c_27/c_28` | `max` 统计的 ping-pong buffer |
| `c_29/c_30` | `sum` 统计的 ping-pong buffer |
| `c_31` | `exp(max_diff)` 修正因子 |

其中最核心的设计点是：

- `K` 和 `V` 是双缓冲的，reader 可以在 compute 消费当前 chunk 时预取下一个 chunk。
- `max/sum/out_acc` 始终留在 L1，不落回 DRAM。
- 只有最终 `O = softmax(QK) @ V` 的结果会写回输出张量。

### 3.3 一个 Q chunk 的典型执行过程

对一个 core 来说，标准 prefill 的逻辑可以粗略写成：

```text
for each (batch, head) assigned to this core:
  for each Q chunk assigned to this core:
    Reader: 把 Q chunk 放到 c_0
    Compute: 初始化在线 softmax 状态

    for each needed K/V chunk:
      Reader: 获取 K/V chunk
      Reader: 如有需要，转发 K/V 给其他 core
      Reader/Writer: 准备 mask
      Compute: 做 QK、mask、exp、sum、P@V、在线累积

    Compute: 最终归一化，把输出放到 c_16
    Writer: 把 c_16 写回输出 tensor
```

## 4. 并行度是怎样切的

### 4.1 张量形状与 chunk

主线路径的输入约定是：

- `Q`: `[B, NQH, Sq, DH]`
- `K`: `[B, NKH, Sk, DH]`
- `V`: `[B, NVH, Sk, DH]`

host 侧先根据 `q_chunk_size` 和 `k_chunk_size` 把序列维切成 chunk：

- `Sq_chunk_t = q_chunk_size / 32`
- `Sk_chunk_t = k_chunk_size / 32`
- `q_num_chunks = padded_Sq / q_chunk_size`
- `k_num_chunks = padded_Sk / k_chunk_size`

这里 `32` 是 tile 高宽，TT 的 reader / compute 都围绕 tile 工作。

### 4.2 并行优先级

当前 prefill 的并行划分顺序是固定的：

1. 先切 `batch`
2. 再切 `q heads`
3. 最后切 `Q chunks`

也就是：

```text
parallelism = batch_parallel_factor
            * nh_parallel_factor
            * q_parallel_factor
```

公式直接来自 `sdpa_program_factory.cpp`：

```text
batch_parallel_factor = min(B, num_cores)
nh_parallel_factor    = min(num_cores / batch_parallel_factor, NQH)
q_parallel_factor     = min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks)
```

然后做 ceiling divide，得到每个 core 的工作块：

- `batch_per_core = ceil(B / batch_parallel_factor)`
- `nh_per_core = ceil(NQH / nh_parallel_factor)`
- `q_per_core = ceil(q_num_chunks / q_parallel_factor)`

### 4.3 core i 的三维坐标映射

对于逻辑编号为 `i` 的 core，host 侧按下面公式给它派工：

```text
local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core
local_nh_start    = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core
local_q_start     = (i % q_parallel_factor) * q_per_core
```

再分别 clamp 到真实范围，避免非整除时越界。

这意味着：

- 如果 `B` 很大，核心会先被 batch 填满。
- 如果 `B` 不大但 `NQH` 很大，会继续沿 head 维扩张。
- 如果 `batch/head` 还填不满 core，才继续沿 `Q chunks` 扩张。

### 4.4 causal 下的负载均衡：`BALANCED_Q_PARALLEL`

`causal` 路径有一个很关键的特殊处理。

问题在于：

- 越靠前的 Q chunk，能看到的 K/V chunk 越少
- 越靠后的 Q chunk，能看到的 K/V chunk 越多

如果直接顺序分块，就会让后面的 core 明显更忙。

因此当前实现会在满足以下条件时启用 `BALANCED_Q_PARALLEL`：

- `is_causal == true`
- `q_per_core * q_parallel_factor == q_num_chunks`
- `q_per_core` 是偶数

启用后，每个 core 不再只拿一段连续 Q chunk，而是拿一对对称位置的 chunk：

```text
Core 0: Q0 + Q(n-1)
Core 1: Q1 + Q(n-2)
Core 2: Q2 + Q(n-3)
...
```

这样可以把“前面轻、后面重”的工作更均匀地摊开。

## 5. 当前主线有哪些数据传输模式

## 5.1 模式 A：本地 DRAM/L1 流水，不做核间复用

这是最基础也最常见的模式。适用于：

- `causal prefill`
- `non-causal` 但某个 `(batch, head)` 只落在单个 core 上
- `non-causal` 但用户显式关闭 forwarding
- `chunked/paged prefill`

这时数据流非常直接：

```text
Q: DRAM -> 本地 L1(c_0)
K: DRAM -> 本地 L1(c_1)
V: DRAM -> 本地 L1(c_2)
OUT: 本地 L1(c_16) -> DRAM
```

需要强调两点：

- `Q` 当前实现基本不在 core 间共享，也不参与 forwarding。
- `K/V` 是否能复用，完全取决于是否进入 `non-causal` 的 KV forwarding 路径。

### 5.2 mask 是怎样来的

mask 有两种来源：

1. 用户显式提供 `attn_mask`
   - reader 从外部 tensor 读入 `c_3`
   - 支持 batch/head 维广播
2. 用户不提供显式 mask
   - writer kernel 在 device 上生成 mask
   - 包括 `causal` mask、sliding window mask、padding mask

这点很容易忽略：当前实现里，显式 mask 的读取和隐式 mask 的生成不在同一个 kernel 里。

### 5.3 causal 为什么天然没有核间通信

`causal` 模式虽然同一 head 可能也被拆到多个 core，但每个 core 只需要算自己那组 Q chunk 对应的因果可见区间，没必要把 `K/V` 在 core 之间共享。

这里真正的优化点不是核间通信，而是：

- 只遍历需要的 `K/V chunk`
- 用 `BALANCED_Q_PARALLEL` 把轻重 chunk 配平
- 用双缓冲把 reader 和 compute 重叠起来

## 5.4 模式 B：non-causal 下的 KV chain forwarding

当 `is_causal == false` 时，同一个 `(batch, head)` 的不同 Q chunk 往往都要看完整的 `K/V`。如果这些 Q chunk 分布在多个 core 上，让每个 core 各自去 DRAM 读一遍 `K/V` 会很浪费。

于是当前实现引入了 `KV forwarding`：

```text
injector 从 DRAM 读 K/V
  -> 发给下一个 core
     -> 下一个 core 继续转发
        -> ...
```

这条路径只在以下条件下才会考虑启用：

- `is_causal == false`
- `TT_SDPA_KV_FORWARDING_MODE != disabled`
- `!is_chunked`

也就是说：

- `non-causal` 才会走
- `paged/chunked` 明确不走

### 5.5 chain 是怎么建出来的

host 侧会先统计每个 core 负责哪些 `(batch, head, q_chunk_range)`，然后把同一个 `(batch, head)` 但落在多个 core 上的 segment 组织成链。

链上有三种角色：

- `injector`
  - 从 DRAM 读入共享的 `K/V chunk`
  - 把这份数据转发给后继 core
- `receiver`
  - 不从 DRAM 读这份共享 `K/V`
  - 先等上游 core 把数据写到自己 L1
  - 如果自己不是链尾，还要继续转发
- `sink`
  - 只接收，不再转发

为了让这套机制工作，program_factory 会额外创建三个 semaphore：

- `sender`
- `receiver`
- `valid`

reader kernel 用这些 semaphore 做“我准备好了 / 你可以发了 / 数据已到达”的握手。

## 5.6 模式 C：`unicast`

### 什么时候会用到

会进入 `unicast` 的典型情况有：

- 用户显式设置 `TT_SDPA_KV_FORWARDING_MODE=unicast`
- `auto` 模式下，有链不满足多播条件，整次运行退回单播
- `hybrid` 模式下，某条链不满足多播条件，这条链单独退回单播

### 数据是怎么传的

`unicast` 是典型的逐跳链式转发：

```text
DRAM -> Core A(injector)
Core A --unicast--> Core B
Core B --unicast--> Core C
Core C --unicast--> Core D
```

reader kernel 的关键动作是：

1. sender 从 DRAM 读入 `K/V chunk`
2. 如果需要转发，就对下一个 core 执行 `noc_async_write`
3. flush 写通道后，用 `noc_semaphore_set_remote` 通知对端数据可用
4. receiver 侧先 `wait`，等数据到了再把该 chunk 推入自己的 CB

### `unicast` 的一个重要优点

`unicast` 可以容忍链上不同 core 的 `q_chunk_count` 不一致。

当前实现里，如果链上各 core 的 `q_chunk_count` 不一样，会按 `q_chunk_count` 从大到小稳定排序，再用：

```text
should_forward = q_iter < next_core_q_chunks
```

来保证重工作量的 sender 只在对方还需要时继续转发。

这也是为什么：

- 尾部不均匀
- 某些 core 比别的 core 少几个 Q chunk

这种 case 往往还可以安全走 `unicast`，但不能走 `multicast`。

## 5.7 模式 D：`multicast`

### 什么时候会用到

`multicast` 只会出现在 `non-causal` forwarding 链路里，而且必须满足严格资格检查。

当前实现的关键约束有三条：

1. 链上所有 physical core 必须在同一行
2. `mcast` 覆盖的 `[min_x, max_x]` 矩形里不能夹着非链上的 active worker core
3. 链上所有 core 的 `q_chunk_count` 必须一致

只有都满足，链才会升级成 `multicast`。

### 数据是怎么传的

一旦某条链被标记为 `mcast`，就不再是逐跳传递，而是：

```text
DRAM -> Injector
Injector --multicast--> 同一行矩形区域内的所有 receiver
```

这时：

- injector 保存 `mcast` 矩形起止坐标
- receiver 不再继续往下转发
- sender 使用 `noc_async_write_multicast`
- 完成后再用 `noc_semaphore_set_multicast` 一次性通知多个 receiver

### `auto`、`hybrid`、`multicast_strict` 的区别

当前实现里这几个模式的语义分别是：

- `auto`
  - all-or-nothing
  - 只有所有多核链都满足 mcast 条件时才整体启用 mcast
  - 只要有一条不满足，就整次运行退回 unicast
- `hybrid`
  - per-chain hybrid
  - 满足条件的链用 mcast，不满足的链保留 unicast
- `multicast` / `mcast` / `multicast_strict`
  - 当前代码里等价于 strict 模式
  - 只要不是所有多核链都可 mcast，就直接报错

### 为什么 mcast 不是“永远更快”

从当前仓库的实验看，mcast 的收益来自“减少重复 DRAM 读、放大 fanout”，但它本身也有设置和同步成本。

经验上可以这样理解：

- 单 sink 微基准里，`multicast` 延迟比 `unicast` 更高，说明 mcast 本身有额外 setup 成本
- 当 fanout 很小，比如链长只有 2，mcast 的收益不明显
- 当 fanout 变大，比如链长到 4 甚至更长，减少重复 DRAM 读的收益开始压过 setup 成本，mcast 往往更优

结合当前实验结论，可以记成：

- `chain_len <= 2`：forwarding 模式差异通常很小
- `chain_len >= 4` 且拓扑可 mcast：`multicast / auto` 往往明显优于 `unicast`

## 5.8 模式 E：chunked / paged prefill

这条路径的目标不是在 core 间复用 `K/V`，而是处理长上下文和 paged KV cache。

它和标准 prefill 的最大区别有三点：

1. 一次只处理一个 Q chunk
2. `K/V` 不是逻辑连续序列，而是通过 `page_table` 间接定位
3. 代码里明确写了：`forwarding not supported for paged mode`

因此它的数据流是：

```text
Q chunk: DRAM -> 本地 L1
page_table: DRAM -> 本地 L1(c_6)
K/V block: 根据 page_table 从 paged cache -> 本地 L1
OUT: 本地 L1 -> DRAM
```

如果用的是 `chunk_start_idx_tensor`，还会额外有：

- compute 从 `c_8` 读动态 chunk 起点
- writer 从 `c_9` 读动态 chunk 起点

这条路径更像“按页取数”，而不是“多核之间共享同一份 K/V”。

## 6. 从 reader / compute / writer 三个角度看数据流

### 6.1 reader 在搬什么

reader 的职责可以分成四类：

1. 读 `Q`
2. 读或接收 `K/V`
3. 读显式 `mask`
4. 在需要时执行 `K/V` 的 `unicast` 或 `multicast`

其中一个不太显眼但很重要的优化是：

- 如果 `Q chunk` 还能继续被拆成多个 subblock，reader 会把 Q 的 push 延后到 `K` 转发之后，再按 subblock 逐步推给 compute
- 这样可以减少 reader 端的阻塞，更好地和 K/V 转发、compute 重叠

### 6.2 compute 在干什么

compute 核本质上是标准的 FlashAttention-2 在线 softmax 内环：

1. `Q_chunk @ K_chunk^T`
2. 加 mask
3. 求每行新的 `cur_max`
4. 做 `exp((QK - cur_max) * scale)`
5. 累加 `cur_sum`
6. 做 `P @ V_chunk`
7. 用 `exp(prev_max - cur_max)` 修正历史的 `sum` 和 `out_acc`
8. 继续累计
9. 最终做归一化

它的关键不是“把大矩阵显式存下来再 softmax”，而是：

- 一边扫 `K/V chunk`
- 一边维护 `prev_max / prev_sum / out_acc`

这就是为什么中间结果可以一直留在 L1，而不必把完整 score matrix 落回 DRAM。

### 6.3 writer 在干什么

writer 不是单纯的“收尾写回”。

它还承担了两件工作：

1. 当用户没有提供显式 mask 时，在 device 上生成隐式 mask
2. 把 compute 放到 `c_16` 的输出块写回 output tensor

所以 writer 也参与了主流水，而不是纯粹的最后一步。

## 7. 一张“什么时候走哪种模式”的判定图

可以把当前 prefill 的选择逻辑近似写成：

```text
standard prefill?
  |
  +-- is_causal = true
  |     -> 不做 KV forwarding
  |     -> 每核本地读所需 K/V
  |     -> 启用 balanced Q 并行（若条件满足）
  |
  +-- is_causal = false
        |
        +-- is_chunked/paged = true
        |     -> 不做 KV forwarding
        |     -> 每核经 page_table 读 paged K/V
        |
        +-- is_chunked/paged = false
              |
              +-- 某个 (batch, head) 只落在单核
              |     -> 每核本地读 K/V
              |
              +-- 某个 (batch, head) 落在多核
                    -> 进入 KV forwarding
                    -> 依据 mode 和拓扑条件选择：
                       - unicast
                       - multicast
                       - auto
                       - hybrid
```

## 8. 与 FlashDecode 的区别

FlashDecode 不是“把 prefill 的 Q 长度设成 1”这么简单，它在并行度和通信模式上都不同。

### 8.1 decode 的并行度

decode 场景里：

- `Q` 很小，通常就是单 token
- 不能再像 prefill 一样主要靠切 `Q chunks` 来扩 core

因此 decode 的主思路是：

- 先按 `batch`
- 再按 `kv_head`
- 如果 core 还很多，再让多个 core 分摊同一个 head-batch 的 KV 片段

官方 LLM 文档也明确说明了：

- decode 里 `q_chunk_size` 基本不起作用
- `max_cores_per_head_batch` 用来限制同一个 head-batch 最多用多少核
- 实际经验上超过 16 核/ head-batch 收益会很差，甚至变慢

### 8.2 decode 的数据传输模式

decode 不走 prefill 的 `KV chain forwarding`。

它的典型流程是：

1. 每个 core 只读自己负责的一段 KV cache
2. 各自算出局部的在线 softmax 结果 `(O_local, M_local, L_local)`
3. 再通过树形归约在 core 间合并这些局部结果

所以 decode 的通信模式是：

- 不是“把同一份 K/V 发给别的核”
- 而是“把局部统计量和局部输出向父节点归并”

这和 prefill 的链式 `unicast / multicast` 是两种完全不同的思路。

## 9. 对当前实现的理解建议

如果你要读代码，推荐按下面顺序：

1. `sdpa_nanobind.cpp`
   - 看对外 API 和张量约定
2. `sdpa_program_factory.cpp`
   - 看并行划分
   - 看 CB 分配
   - 看 forwarding mode 和 chain/mcast 构图
3. `reader_interleaved.cpp`
   - 看 Q/K/V/mask 的真实搬运
   - 看 unicast / multicast 的真正发包位置
4. `compute/sdpa.cpp` + `compute_common.hpp`
   - 看在线 softmax 的内核主循环
5. `writer_interleaved.cpp`
   - 看隐式 mask 生成和输出写回

如果你要先建立直觉，再回头看代码，可以把本文和下面几篇配合着读：

- `docs/tt-metal-sdpa-analysis.md`
- `docs/tt-metal_SDPA算子梳理.md`
- `docs/SDPA_单核多核多芯片实现.md`
- `docs/tt-metal_unicast_multicast.md`

## 10. 最后压缩成三句话

1. 当前 TT Flash Attention prefill 的主并行轴是 `batch -> q_head -> Q chunks`，不是先切 K/V，也不是先切输出列。
2. `Q` 基本总是每核自己读；`K/V` 只有在 `non-causal && !is_chunked` 时才会尝试跨核复用。
3. `multicast` 不是默认总能用的“更强单播”，它只适用于很规整的链；一旦拓扑不规则、尾部不均匀或走 paged cache，就会退回 `unicast` 或本地读取。

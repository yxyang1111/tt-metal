# Flash MLA 在 TT 上的数据流笔记

可以先把 FlashMLA 在 TT 上想成一条"Q 常驻、K 流过、Track 内归并"的二维片内流水。这里先约定两个术语：`Track` 指 Q 方向的工作带，`Lane` 指 K 方向的工作带。`Q shard` 指单个 `Track` 在当前计算阶段常驻的查询分片；在 decode 中，它直接来自按 head 维切开的 Q，在 prefill 中，它则来自当前 Q chunk 再按 head 维切开的结果。`K chunk` 指 `K_latent` 沿序列维切出的流式块。逻辑上，整个算子先为当前计算阶段准备 Q：若 Q 很短（如 decode），就直接把 Q 按 head 方向切成 `N_track` 份；若 Q 较长（如 prefill），则先沿 sequence 维切成 Q chunk，再在每个 Q chunk 内按 head 方向切成 `N_track` 份。之后每个 `Track` 持有并复用自己对应的 `Q shard`，各 `Lane` 则依次处理分配到本 `Lane` 的 `K chunk`；于是每个 worker 始终拿着固定的 `Q shard`、不断接收本 `Lane` 流过的 `K chunk`，对每个 chunk 做一次局部 online softmax 更新。等所有 `Lane` 都跑完后，再沿每个 `Track` 把不同 `Lane` 上的局部 `(m, l, o)` 树形归并到 `Tsnk`，最后得到该 `Track` 对应的 output shard。后面的细节，本质上都是在展开这条片内数据流主线。

从实现视角看，这条逻辑上的片内数据流还需要进一步落到具体硬件与片外数据组织上。需要先把 `N_track × N_lane` 的逻辑网格映射到实际 Tensix core，决定每个 `Track` 由哪颗 core 负责把 Q 读入片上并沿 `Track` 分发、每个 `Lane` 由哪颗 core 负责从 DRAM 读取 `K chunk` 并沿 `Lane` 分发，以及每个 `Track` 最终由哪颗 core 作为 `Tsnk` 完成归并和写回；同时还要安排片外 I/O 与数据排布，使 Q、`K_latent` 和 output 在 DRAM bank view 上的布局与这种二维工作分工对齐。后文关于逻辑-物理映射、DRAM bank view、Q/K 的读取与广播，其实都是在把这条逻辑数据流具体化到 TT 的 NoC 与片外存储体系中。

## 1. 虚拟二维工作网格

```text
                        Track 1          Track 2          ...      Track N_track
                    ┌────────────────┬────────────────┬─────────┬──────────────────┐
Lane 1              │ Tsrc(1)/Lsrc(1)│    Tsrc(2)     │   ...   │ Tsrc(N_track)    │
                    ├────────────────┼────────────────┼─────────┼──────────────────┤
Lane 2              │    Lsrc(2)     │                │   ...   │                  │
                    ├────────────────┼────────────────┼─────────┼──────────────────┤
...                 │      ...       │      ...       │   ...   │       ...        │
                    ├────────────────┼────────────────┼─────────┼──────────────────┤
Lane N_lane         │ Lsrc(N_lane)/  │    Tsnk(2)     │   ...   │ Tsnk(N_track)    │
                    │    Tsnk(1)     │                │         │                  │
                    └────────────────┴────────────────┴─────────┴──────────────────┘
                         ▼                ▼                         ▼
                    Track 1 内        Track 2 内                Track N_track 内
                      树形归并          树形归并                    树形归并
```

- `Tsrc(b)` 表示第 `b` 个 `Track` 的 `track source`：它负责从 DRAM 读取该 `Track` 的 `Q shard` 到本地 L1，并作为该 `Track` 内其它 worker 的 `Q` 拉取源。实现上它更接近 `source L1 + signal/pull`，而不是一次硬件 multicast。
- `Tsnk(b)` 表示第 `b` 个 `Track` 的 `track sink`：它在 tail 阶段负责等待该 `Track` 内树形归并的子结果就绪，把子结果读入本地 merge 输入缓冲，再由本地计算核完成归并、归一化与最终写回。
- `Lsrc(j)` 表示第 `j` 个 `Lane` 的 `lane source`：它负责从 DRAM 读取当前 `K chunk`，并沿该 `Lane` 把 `K chunk` 分发给其它 core。更准确地说，同一颗 core 上的 `NCRISC` 负责读取当前 `K chunk`，同一颗 core 上的 `BRISC` 负责执行 `lane-wise broadcast` / multicast。
- 图中已将网格转置成"`Lane` 按行、`Track` 按列"的画法，因此 `Tsrc(b)` 固定画在第 `b` 列顶端，`Tsnk(b)` 固定画在第 `b` 列底端，`Lsrc(j)` 固定画在第 `j` 行最左格。一个物理 core 可以同时承担多个逻辑角色，因此左上角 `Tsrc(1)/Lsrc(1)` 表示同一颗 core 同时是 `Track 1` 的 `track source` 和 `Lane 1` 的 `lane source`；左下角 `Lsrc(N_lane)/Tsnk(1)` 表示同一颗 core 同时是最后一条 `Lane` 的 `lane source` 和 `Track 1` 的 `track sink`。

Flash MLA 在 TT 上可视为映射到一张 `N_track × N_lane` 的虚拟二维工作网格。

- `Track` 表示 Q 方向的并行划分。每个 `Track` 对应一份固定的 `Q shard`。
- `Lane` 表示 K 方向的并行划分。每个 `Lane` 负责一组按序列维切出的 `K chunk`。
- 单个 `worker` 对应一个 `(b, j)` 位置，处理本 `Track` 的 `Q shard` 与本 `Lane` 当前的 `K chunk` 的局部 attention 更新。

本文涉及的片上分发不是泛指任意 multicast，而仅指两类与这张虚拟网格对齐的规则数据流：

- `track-wise Q distribution`：沿同一 `Track` 由 `Tsrc` 在本地 L1 先物化 `Q shard`，再由该 `Track` 其它 worker 通过 `signal + pull` 拿到自己的本地副本。
- `lane-wise broadcast`：沿同一 `Lane` 由 `Lsrc` 把当前 `K chunk` 推送给该 `Lane` 其它 core。

后文统一使用这两个更具体的术语，而不再泛泛写 multicast。前者强调 `Q` 路的 source/pull 语义，后者强调 `K` 路的 source/multicast 语义。

## 2. 运行前的静态准备

在算子启动前，需要完成两个准备。

### 2.1 逻辑网格到物理 core 的映射

先确定虚拟网格的 `Track` 数与 `Lane` 数，再将每个逻辑坐标映射到具体的 Tensix core。

这里：

- 对 `Q` 路，`Tsrc(b)` 指负责从 DRAM 读取 `Q shard` 并服务 `Track b` 内 pull 的源 core。
- 对 `K` 路，`Lsrc(j)` 指负责从 DRAM 读取 `K chunk` 并执行 `lane-wise broadcast` 的源 core。
- `Tsnk(b)` 指负责接收 `Track b` 内树形归并最终结果并执行最终 merge / 写回的目标 core。

这样，后续对 `Track`、`Lane`、`Tsrc`、`Lsrc`、`Tsnk` 的描述既保持算法上的规则性，又与实际硬件位置一一对应。通常来说，`Lsrc` 对应的 Tensix core 会更接近 DRAM NoC 接口，`Tsnk` 对应的 Tensix core 会更接近 DRAM output NoC 接口。

### 2.2 DRAM 数据布局

更准确地说，这里应叫 TT 运行时 / allocator 可见的 `DRAM bank view`，而不只是物理 `DRAM controller`。

以 Wormhole 为例，底层 `dram` 拓扑里有 6 个 physical channel / controller，每个 channel 挂 3 个 `DRAM NoC subchannel / endpoint`；`tt-metal` 的 `dram_views` 再从中为 worker 侧选定 endpoint，并配合 `address_offset` 把同一 physical channel 暴露成 2 个 software-visible views，因此 allocator 实际看到的是 12 个 views，而不是仅仅 6 个 physical controllers。

为了避免和教科书里更底层的 `DRAM die` 内部 `bank` 概念混淆，论文里更严谨的叫法宜为 `DRAM bank view`；下文若为简洁继续写 `bank`，均指这种 `bank view`。

`K_latent` 沿序列维切成多个 chunk，并按 `Lane` 编号对齐到多个 `DRAM bank view`，使得第 `j` 个 `Lane` 长期对应第 `j` 个 `bank view`。

下面示意一次 `iter` 内 K 在网格里的流向：

```text
DRAM (N_lane 个 bank view，lane j 独占 b[j])                片上虚拟网格 (N_track 个 Track × N_lane 个 Lane)
时间轴：iter 1 → iter 2 → ...                              图示一次 iter 内 K 在网格里的流向

       lane 1    lane 2    lane 3   ...    lane N_lane        lane 1   lane 2   lane 3  ...  lane N_lane
       b[1]      b[2]      b[3]            b[N_lane]        ┌────────┬────────┬────────┬─────┬──────────┐
      ┌──────┐  ┌──────┐  ┌──────┐        ┌──────┐  Track 1 │  Lsrc  │  Lsrc  │  Lsrc  │ ... │   Lsrc   │
iter 1│ ch 1 │  │ ch 2 │  │ ch 3 │  ...   │ ch N │          ├────────┼────────┼────────┼─────┼──────────┤
      ├──────┤  ├──────┤  ├──────┤        ├──────┤  Track 2 │   ▼    │   ▼    │   ▼    │ ... │    ▼     │
iter 2│chN+1 │  │chN+2 │  │chN+3 │  ...   │ch 2N │ ─read─▶  ├────────┼────────┼────────┼─────┼──────────┤
      ├──────┤  ├──────┤  ├──────┤        ├──────┤  Track 3 │   ▼    │   ▼    │   ▼    │ ... │    ▼     │
iter 3│ch2N+1│  │ch2N+2│  │ch2N+3│  ...   │ch 3N │          ├────────┼────────┼────────┼─────┼──────────┤
      ├──────┤  ├──────┤  ├──────┤        ├──────┤   ...    │  ...   │  ...   │  ...   │ ... │   ...    │
      │ ...  │  │ ...  │  │ ...  │        │ ...  │          ├────────┼────────┼────────┼─────┼──────────┤
      └──────┘  └──────┘  └──────┘        └──────┘  Track N_track │ ▼  │   ▼    │   ▼    │  ▼  │    ▼     │
                                                           └────────┴────────┴────────┴─────┴──────────┘
```

图中，`lane-wise broadcast` 会把当前 `K chunk` 推给本 `Lane` 其它 core。

- `N = N_lane`
- `ch X` = `K_latent` 沿 `S` 轴的第 `X` 个 `chunk`，大小为 `Lk × d_k` 元素。
- 一次 `iter t`：`Lane j` 的 `Lsrc(j)`（一颗具体 core）从 `b[j]` 读 `ch ((t-1)·N + j)`，写入本地 `cb_k_in`，再做一次 `lane-wise broadcast`，把当前 `K chunk` 推给本 `Lane` 其它 `N_track - 1` 颗 core。

这里还需要补一层实现语义：在 `K` 路上，receiver 并不会对称地"自己再读一遍 `K` payload"。更准确地说，`Lsrc` 所在 core 的 `BRISC` 直接把 payload 组播写入 receiver 预留好的本地目标区；receiver 侧的 reader / `NCRISC` 主要负责等待 `mcast-ready` semaphore，并在 ready 后把本地 `cb_k_in` 发布给 compute 使用。

## 3. 运行时数据流

### 3.1 Q 的一次性分发（Q-stationary）

如果沿用 DNN 数据流术语，这一部分可称为 `Q-stationary`。Q 的作用是为每个 `Track` 提供一份稳定的查询分片。为此，每个 `Track` 会指定一颗 `Tsrc`。该 core 的 reader / `NCRISC` 先从 DRAM 读取该 `Track` 对应的 `Q shard`，写入本地 `cb_q_in`。随后，同一 `Track` 的其它 worker 并不是被动接收一次硬件 multicast；更准确地说，它们在轻量 ready / signal 协调后，从 `Tsrc` 的本地 L1 逐个发起 pull，把这份 `Q shard` 拉到各自本地。

之所以不把 `Q` 路实现成和 `K` 路完全对称的矩形 multicast，是因为同一 `Track` 的 worker 在物理坐标上通常并不构成硬件组播要求的矩形；而 `Q` 本身体积较小，因此采用 `source L1 + signal/pull` 的实现更自然，也能把 NoC0 留给主导性的 `K` 路。

完成这一步后，同一 `Track` 内所有 core 持有逻辑上相同的 `Q shard`，并且这份数据在整个算子执行期间常驻于 L1。后续遍历所有 `K chunk` 时，不再需要重新读取或重新分发 Q。

换言之，`Q shard` 是常驻并被反复复用的，而 K 仍按 `chunk` 流式经过网格，因此整个算子更准确地说是以 `Q-stationary` 为核心、同时叠加 `K-streaming` 的混合数据流。

```text
DRAM 侧 Q [Hq × d_k]                          片上虚拟网格 (N_track 个 Track × N_lane 个 Lane)

┌────────────────────────────┐               ┌─────┬─────┬─────┬───────────┐
│ heads [1, Hq/N_track]      │ ── read ──▶   │Tsrc │  ▶  │  ▶  │    ▶      │  Track 1
│                            │               │     │     │     │           │
├────────────────────────────┤               ├─────┼─────┼─────┼───────────┤
│ heads [Hq/N_track+1, 2·Hq/N_track] │─read─▶│Tsrc │  ▶  │  ▶  │    ▶      │  Track 2
│                            │               │     │     │     │           │
├────────────────────────────┤               ├─────┼─────┼─────┼───────────┤
│ heads [2·Hq/N_track+1, 3·Hq/N_track]│─read─▶│Tsrc│  ▶  │  ▶  │    ▶      │  Track 3
│                            │               │     │     │     │           │
├────────────────────────────┤               ├─────┼─────┼─────┼───────────┤
│          ...               │     ...       │ ... │ ... │ ... │    ...    │  ...
├────────────────────────────┤               ├─────┼─────┼─────┼───────────┤
│ heads [(N_track-1)·Hq/N_track+1, Hq]│─read─▶│Tsrc│  ▶  │  ▶  │    ▶      │  Track N_track
│                            │               │     │     │     │           │
└────────────────────────────┘               └─────┴─────┴─────┴───────────┘
        │                                     Lane1  Lane2  ... Lane(N_lane)
        └────────────────────── track-wise signal + pull ────────────────────┘
```

- `Q` 水平带 `b` = `Track b` 的 `Q shard`
- 大小为 `(Hq / N_track) × d_k`
- `tile` 形状为 `⌈Hq / N_track / 32⌉ × (d_k / 32)`

### 3.2 K 的流式读取与 Lane 内分发

1. `K_latent` 沿序列维按 `Lk` 切成 `T = ⌈S / Lk⌉` 个 `K chunk`，再按 `Lane` 编号分派到 `N_lane` 个 `Lane`。对某一 `Lane j` 而言，它负责的是集合 `{j, j + N_lane, j + 2N_lane, ...}` 中的那些 `chunk`。
2. 每个 `Lane` 会指定一颗 `Lsrc`。在每一轮迭代中，该 core 上的 `NCRISC` 从该 `Lane` 对应的 `DRAM bank view` 读取当前 `K chunk`，写入本地 `cb_k_in`。
3. 当前 `K chunk` 就绪后，同一颗 core 上的 `BRISC` 再沿该 `Lane` 对其它 core 做一次 `lane-wise broadcast` / multicast。这样，同一 `Lane` 内所有 core 在同一轮都获得相同的当前 `K chunk`，并且每个 `K chunk` 只会从 DRAM 读取一次，其余副本都通过片上分发获得。
4. receiver 侧不会主动对 `K` payload 再发起一次对称读取；更准确地说，`Lsrc` 的 `BRISC` 直接把 payload 写入 receiver 预留好的本地目标区，receiver 侧的 reader / `NCRISC` 通过等待 `mcast-ready` semaphore 知道当前 `K chunk` 已经到齐，再把本地 `cb_k_in` 暴露给 `TRISC`。
5. 这条 `K` 路通常依赖双缓冲 `cb_k_in` 形成 ping-pong 流水：`Lsrc` 可以在 `compute K_t` 的同时提前 `read K_{t+1}`，但不能无限超前；更早的 `K_{t+2}` 仍要等到前一个 buffer slot 被释放。

### 3.3 worker 上的计算和局部更新

1. 当某个 `worker` 同时具备本 `Track` 的 `Q shard` 与本 `Lane` 当前已就绪的 `K chunk` 后，就可以执行一次完整的局部 attention 更新。该更新包含四个逻辑步骤：计算 `QKᵀ`、加入 `mask`、执行 `online softmax` 递推、执行 `PV` 并更新局部输出。
2. 在 `Lsrc` core 上，本地 `TRISC` 可以在当前 `K_t` 的本地读取完成后立刻开始计算，因此通常会比同条 `Lane` 其它 worker 更早进入 `compute K_t`。对其它 worker（包括同时承担 `Tsrc` 或 `Tsnk` 角色的 worker）而言，它们进入 `compute K_t` 的时刻取决于同一个 `mcast-ready` handoff：一旦 ready，便立刻开始本轮计算。
3. 对任一 `worker` 而言，其执行模式在所有迭代中保持不变：`Q shard` 固定不变，输入的 `K chunk` 按该 `Lane` 的分派顺序依次到达；每处理一个 `K chunk`，本地 `(m, l, o)` 就更新一次。当该 `Lane` 负责的全部 `K chunk` 都被处理完成后，这个 `worker` 持有的是一个局部结果：它对应于本 `Track` 的 `Q shard` 对该 `Lane` 所覆盖那部分 K 序列区间的 attention 累积结果。

### 3.4 Track 内归并与写回

1. 当所有 `Lane` 都完成各自负责的 `K chunk` 处理后，每个 `Track` 内会存在 `N_lane` 份局部 `(m, l, o)`。这些局部结果共享同一份 `Q shard`，差别仅在于它们覆盖的是不同的 K 子区间。因此，接下来的任务是沿 `Track` 将这些局部结果归并为一份完整结果。
2. 归并通过 `Track` 内树形拓扑完成。每一次归并都使用 `online softmax` 对应的 `merge` 规则，因此可以逐层把多个局部三元组合成为一个最终三元组。经过 `⌈log₂ N_lane⌉` 层后，结果落在该 `Track` 预先指定的 `Tsnk` 上。
3. 这里的 `wait child-ready` 更准确地说发生在 writer / reduction 路，而不是 `K` 输入 reader 路。child 先把自己的局部 `(l, m, o)` 写到 parent / `Tsnk` 预留的 round-specific slot，并递增对应 round 的 reducer semaphore；parent / `Tsnk` 侧的 `BRISC` 轮询这个 semaphore，在 child 就绪后把远端的 `l / m / o` 读入本地的 merge 输入缓冲，再交给本地 `TRISC` 做 online-softmax merge。所以虽然它和 `K` 的 ready/wait 一样都涉及片上 `L1 -> L1` 通信，但它属于 tail 阶段的 writer / reduction 协议，而不属于前半段的 reader 输入协议。
4. 当前实现还对 reduction 的消费顺序做了固定约束：writer kernel 与 compute kernel 都按 `round = 1, 2, 3, ...` 的顺序处理 child 结果。因此，即使 round 2 比 round 1 更早 ready，当前 kernel 也不会先 merge round 2；它会继续等待 round 1，再按顺序把各 round 的输入送入 merge 流水。
5. `Tsnk` 在完成所有 round 的 merge 与最终归一化后，再把该 `Track` 的 `output shard` 写回 DRAM 中预先分配好的位置。所有 `Tsnk` 写回的切片彼此不重叠，最终在 DRAM 中拼接成完整的 `Output` 张量。

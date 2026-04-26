# Flash MLA 数据流：第一性原理定义

本文分三部分，可独立阅读：

- **Part A — 少公式的严谨重述**：在尽量少用公式的前提下，用书面化、定义清晰的语言重述 Flash MLA 在 TT 上的数据流。
- **Part B — 数据流怎么走**：结合 Tenstorrent 硬件讲清楚数据从哪来、怎么走、去哪里。
- **Part C — 形式化版**：把输入、所有可变参数、约束、代价模型用符号写清楚（硬件无关）。

---

# Part A — 少公式的严谨重述

这一部分不是新的形式化定义，而是在尽量减少公式的前提下，对 Part B / Part C 的核心内容做一次书面化重述。目标是用严格术语说明 Flash MLA 在 TT 上的数据放置、传输、局部计算与最终归并之间的关系。

## A.1 整体组织方式

从执行结构看，Flash MLA 在 TT 上可视为映射到一张 `N_lane × N_col` 的虚拟二维工作网格。
   
- **Lane（行）**表示 Q 方向的并行划分。每一行对应一份固定的 `Q shard`。
- **Column（列）**表示 K 方向的并行划分。每一列负责一组按序列维切出的 `K chunk`。
- **单个 worker** 对应一个 `(Lane, Column)` 位置，处理"本行的 `Q shard`"与"本列当前 `K chunk`"之间的局部 attention 更新。
- **Lane root** 是每一行的归并终点，负责汇总该行所有局部结果，并将最终输出写回 DRAM。

因此，整个算子的稳态结构可以概括为：**Q 在行内复制并常驻，K 在列内按 chunk 流动，局部计算发生在 worker 上，最终结果沿行归并。**

## A.2 运行前的静态准备

算子启动前，host 需要完成两类静态配置。

第一类是**逻辑网格到物理 core 的映射**。算法先确定虚拟网格的行数与列数，再将每个逻辑坐标映射到具体的 Tensix core。这样，后续对 Lane、Column、sender、root 的描述既保持算法上的规则性，又与实际硬件位置一一对应。

第二类是**DRAM 数据布局**。其中最关键的是 `K_latent`：它沿序列维切成多个 chunk，并按列号对齐到多个 DRAM bank，使得第 `i` 列长期对应第 `i` 个 bank。`Q` 的布局相对简单，启动时由每个 Lane 的 `Q reader` 读取一次即可。`output` tensor 也会预先分配好对应的写回位置。需要强调的是，**V 不以独立张量存在于该数据流中**，而是始终视为 `K_latent` 在最后一维上的前 `d_v` 列。

这两类准备共同保证了后续 steady state 的基本结构：`Q` 的分发路径在启动时确定，`K` 的读取来源在整个执行期间保持稳定。

## A.3 运行时数据流

### A.3.1 Q 的一次性分发

`Q` 的作用是为每个 Lane 提供一份稳定的查询分片。为此，每一行会指定一颗 `Q reader`。该 core 从 DRAM 读取本行对应的 `Q shard`，写入本地 `cb_q_in`，随后沿该行对其余 core 做一次 NoC multicast。

完成这一步后，同一 Lane 内所有 core 持有完全相同的 `Q shard`，并且这份数据在整个算子执行期间常驻于 L1。后续遍历所有 `K chunk` 时，不再需要重新读取或重新广播 `Q`。因此，`Q` 在时间上是一次性初始化的，在空间上是行内复制的。

### A.3.2 K 的流式读取与列内分发

`K` 的处理方式与 `Q` 不同。`K_latent` 沿序列维按 `Lk` 切成 `T = ⌈S / Lk⌉` 个 `K chunk`，再按列号分派到 `N_col` 个 Column。对某一列 `i` 而言，它负责的是集合 `{i, i + N_col, i + 2N_col, ...}` 中的那些 chunk。

每一列会指定一颗 `K sender`。在每一轮迭代中，该 core 从本列对应的 DRAM bank 读取当前 `K chunk`，写入本地 `cb_k_in`，随后沿该列对其它 core 做一次 NoC multicast。这样，同一列内所有 core 在同一时刻都获得相同的当前 `K chunk`。

由此形成的结构是明确的：**同一列共享当前 K，同一行共享 Q，不同行列的交点负责不同的局部子问题。** 另外，每个 `K chunk` 只会从 DRAM 读取一次，其余副本都通过片上分发获得。

### A.3.3 worker 上的局部更新

当某个 worker 同时具备本行的 `Q shard` 与本列当前的 `K chunk` 后，就可以执行一次完整的局部 attention 更新。该更新包含四个逻辑步骤：计算 `QKᵀ`、加入 mask、执行 online softmax 递推、执行 `PV` 并更新局部输出。

这一阶段有两条实现特征需要明确指出。

第一，**V 不独立搬运**。在 MLA 中，`V` 直接取自当前 `K chunk` 的前 `d_v` 列，因此 `PV` 所需的 `V` 是 `K` 的列视图，而不是另一份单独加载的数据。

第二，**logits 与概率矩阵不在 L1 中完整具象保存**。片上长期保留的是 online softmax 的 running state，即 `(m, l, o)`；而 `logits` 与中间概率仅在寄存器与计算流水中短暂存在。

因此，从存储与带宽角度看，该局部更新阶段只保留对后续递推必需的状态，而不保留完整的中间矩阵。

### A.3.4 对整列负责的 K 范围完成递推

对任一 worker 而言，其执行模式在所有迭代中保持不变：`Q shard` 固定不变，输入的 `K chunk` 按列的分派顺序依次到达；每处理一个 `K chunk`，本地 `(m, l, o)` 就更新一次。

当该列负责的全部 `K chunk` 都被处理完成后，这个 worker 持有的是一个**局部结果**：它对应于"本行 `Q shard`"对"本列所覆盖那部分 K 序列区间"的 attention 累积结果。换言之，单个 worker 并不直接得到最终输出，而是得到最终输出中的一个可归并部分。

## A.4 行内归并与写回

当所有列都完成各自负责的 `K chunk` 处理后，每个 Lane 内会存在 `N_col` 份局部 `(m, l, o)`。这些局部结果共享同一份 `Q shard`，差别仅在于它们覆盖的是不同的 K 子区间。因此，接下来的任务是沿 Lane 将这些局部结果归并为一份完整结果。

归并通过行内树形拓扑完成。每一次归并都使用 online softmax 对应的 merge 规则，因此可以逐层把多个局部三元组合成为一个最终三元组。经过 `⌈log₂ N_col⌉` 层后，结果落在该行预先指定的 `Lane root` 上。

`Lane root` 随后对最终的 `(m, l, o)` 执行归一化，得到本行对应的 output shard，并将其写回 DRAM 中预先分配好的位置。所有 Lane root 写回的切片彼此不重叠，最终在 DRAM 中拼接成完整的 `Output` 张量。

## A.5 这种数据流为何适配 TT

这种组织方式与 TT 的硬件特征高度一致，原因主要有四点。

**第一，K 的 bank 级并行容易建立。** `K chunk` 沿列分派后，不同列可以并行访问不同 DRAM bank，从而降低读取侧的集中争用。

**第二，NoC multicast 可以直接减少重复搬运。** `Q` 适合按行读取一次后在行内复制，`K` 适合按列读取一次后在列内复制，因此大量重复的 DRAM 访问被替换为片上分发。

**第三，读取、分发和计算可以形成流水。** 在 page 级切分与多 slot `cb_k_in` 的支持下，NCRISC、BRISC 与 TRISC 可以分别承担读取、广播与计算，从而在时间轴上实现重叠。

**第四，L1 的使用保持聚焦。** `Q` 只需常驻一份；`K` 以环形 buffer 方式流经；`V` 复用 `K` 的存储；`logits` 不完整落地；最终使 L1 主要承载 steady state 所必需的状态与缓冲。

因此，从工程实现角度看，Flash MLA 在 TT 上并不是将标准 attention 直接搬到硬件上执行，而是将其重写为一种适合二维网格、DRAM bank 并行与 NoC 多播的流式归并过程。

## A.6 decode 与 prefill 的差别

decode 与 prefill 的主要差别在于 `Q` 的长度，而不在于数据流骨架本身。

在 decode 场景中，`Q` 很短，因此运行形态更接近于用一个较小的 `Q shard` 去遍历一条较长的 `K` 序列；此时主要开销通常集中在 `K` 的读取、列内分发以及最终的 Lane 归并。

在 prefill 场景中，`Q` 也会沿序列维继续切块，因此同样的数据流会对多个 `Q chunk` 重复执行。但对每个 `Q chunk` 而言，结构保持不变：仍然是行内准备 `Q`、列内流动 `K`、worker 做局部递推、Lane root 做最终归并与写回。

## A.7 总结

如果用一句尽量简洁且保持严格性的表述来概括，那么可以写成：

**Flash MLA 在 TT 上的数据流，本质上是：以 Lane 为单位一次性分发并常驻 `Q shard`，以 Column 为单位持续流式分发 `K chunk`，由每个 worker 对其负责的 `(Q shard, K chunk)` 对执行局部 online softmax 累加，最后沿 Lane 归并并写回输出。**

---

# Part B — 数据流怎么走（Tenstorrent 视角）

## B.1 背景

Flash MLA = Flash（把长 K 切成 chunk，用 online softmax 流式累加，不一次算完整 softmax）+ MLA（V 不是独立张量，V 就是 K 的前若干列）。所以整个算子**只流动 Q 和 K**；用到 V 时直接在 K 的缓冲里按偏移取，V 从不单独搬、不单独存。

## B.2 硬件设置

上机前只需要搞清楚两件事：用到哪些 core，数据放在哪几个 DRAM bank。

**逻辑网格（算法视角）。** Flash MLA 先在算法层画一张 `N_lane × N_col` 的**虚拟二维网格**。行（Lane）负责不同 Q shard，列（Column）负责不同 K 段；每列里挑一个 logical 位置做 **sender**（唯一回 DRAM 读 K 的位置），每行里挑一个 logical 位置做 **root**（唯一写 output 的位置）。这张网格的形状只由算法参数决定，和物理 core 坐标无关。

```
                 Column 0     Column 1     ...    Column N_col-1
              ┌────────────┬────────────┬───────┬──────────────┐
    Lane 0    │     *      │     *      │  ...  │      *       │ ──┐
              ├────────────┼────────────┼───────┼──────────────┤   │ Lane 内
    Lane 1    │            │            │       │              │ ──┤ 树形归并
              ├────────────┼────────────┼───────┼──────────────┤   │
    Lane N_lane-1│  R       │            │       │              │ ──┘  (R=root)
              └────────────┴────────────┴───────┴──────────────┘
                  ▲            ▲            ▲        ▲
             DRAM bank 0   bank 1       ...    bank N_col-1
                         (* = Column sender)
```

**逻辑 → 物理映射。** host 侧再用一张常量表把每个逻辑坐标 `(i, b)` 映射到一颗具体的 Tensix core 物理坐标（WH 常用 6×4=24 core，BH 常用 8×8=64 core）。这张映射要一起考虑三件事：(a) sender 所在的物理 core 要离它负责读的 DRAM bank 近；(b) 同列的 logical core 在物理上应连成规则矩形，让 `noc_async_write_multicast` 最省跳数；(c) NOC0 与 NOC1 的物理坐标互为镜像，换 NOC 时要换映射表。换句话说，逻辑网格可以按算法自由描述，物理映射则由硬件拓扑约束。

**DRAM 布局。** K_latent 按 `k_chunk_size` ND-sharded 到 `N_col` 个 bank，bank 顺序与 Column 编号对齐，这样第 i 列的 sender（映射之后的那颗物理 core）永远只读第 i 个 bank。page_table、cur_pos、output tensor 也在 DRAM 分配好。**V 不需要任何准备**——整个算子不需要 DRAM 里有 V。

## B.3 数据流

下面按"存→读→传→算→归并→写回"的顺序讲。本小节先交代**数据怎么存在 DRAM 里**，后面几小节再讲读取、传递、计算、写回。

### B.3.1 数据怎么存在 DRAM 里

算子开跑前，DRAM 里就这几样东西：

| 张量 | 形状 | 存储方式 |
|---|---|---|
| `K_latent` | `[B, S, Hkv, d_k]` | 沿 S 轴切 chunk，chunk 轮转分配到 `N_col` 个 DRAM bank |
| `Q` | `[B, Sq, Hq, d_k]` | 普通 interleaved tensor，按 tile 轮转放各 bank |
| `output` | `[B, Sq, Hq, d_v]` | host 预分配的空壳，等算子结束时写入 |
| `page_table` | `[B, max_blocks_per_seq]` | int32 索引表，**paged KV 时才有** |
| `cur_pos` | `[B]` | int32 位置向量，**decode 时才有** |

**一句话：DRAM 里只有 K 和 Q，加一块空的 output，外加两张可选小表。** 下面只对最关键的 K_latent 多说几句，其余几张没什么特别。

> Q 和 K 都还会被进一步切块（K 沿 `S` 切 chunk，Q 在 prefill 场景下沿 `Sq` 切 chunk），但"切多细、怎么流水"这件事留到后面讲流水线的小节再说。此处只说静态布局。

#### 什么**不在** DRAM

- **V**：不存在。用到时就是 `K_latent[..., :d_v]` 的列视图。
- **Mask**：片上就地生成，不进 DRAM。
- **`(m, l, o)` running stats**：只活在 L1。
- **Lane 划分**：算法概念，DRAM 侧不可见。

### B.3.2 Q：从 DRAM 取进来，按 Lane 分发

Q 的目标是：**让虚拟网格里每一颗 core 的 `cb_q_in` 里都有本 Lane 对应的那份 Q shard**。做法很直接——每 Lane 派一颗 core 回 DRAM 读一次，然后沿本 Lane 做一次 NoC 行广播。全程 Q 只穿过 DRAM 一次。

#### Q 怎么切 + Qr 怎么读怎么传（一张图）

decode 场景下 Q 的全貌是 `[Hq, d_k]`（暂时忽略 batch）。按 `N_lane` 把 `Hq` 切成多段，每段一条水平带，作为对应 Lane 的 Q shard；每条水平带由该 Lane 的 **Q reader** 从 DRAM 读进本地 `cb_q_in`，再沿本 Lane 行多播给其它 `N_col − 1` 颗 core。

```
   DRAM 侧 Q  [Hq × d_k]                         片上虚拟网格 (N_lane 行 × N_col 列)
 ┌────────────────────────────┐                ┌─────┬─────┬─────┬───────────┐
 │ heads [0, Hq/N_lane)             │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane 0
 │                            │                │     │     │     │           │
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │ heads [Hq/N_lane, 2·Hq/N_lane)   │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane 1
 │                            │                │     │     │     │           │
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │ heads [2·Hq/N_lane, 3·Hq/N_lane) │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane 2
 │                            │                │     │     │     │           │
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │          ...               │     ...        │ ... │ ... │ ... │    ...    │  ...
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │ heads [(N_lane-1)·Hq/N_lane, Hq) │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane N_lane-1
 │                            │                │     │     │     │           │
 └────────────────────────────┘                └─────┴─────┴─────┴───────────┘
           │                                    Col0   Col1   ...  Col(N_col-1)
           │                                      └──────── 行多播 ────────┘
           │
    Q 水平带 j = Lane j 的 Q shard
     大小 = (Hq/N_lane) × d_k
     tile = ⌈Hq/N_lane/32⌉ × (d_k/32)
```

图里几件事一次到位：

- **按 Lane 切头**。`Hq` 维度被按 `N_lane` 切成多段（每段 `Hq/N_lane` 个头 × `d_k` 列），左边的 DRAM Q 图已经按 Lane 画出横条。
- **每 Lane 一个 Qr**。虚拟网格每一行第一个 cell 标 `Qr`，是该 Lane 的 Q reader；其余 cell 标 `▶` 代表行内接收者。
- **Read 一次 + 行多播一次**。水平箭头 `── read ──▶` 是 DRAM → Qr 的 `noc_async_read_tile`；右侧虚拟网格里的横向 `▶` 链是 Qr 沿行做的 `noc_async_write_multicast`。
- **规模**。单 Lane Q shard = `(Hq/N_lane) × d_k` 个元素 = `⌈Hq/N_lane/32⌉ × (d_k/32)` 个 tile。

举几个典型尺寸（DeepSeek-V3：`Hq=128, d_k=576 = 18 tile 列`）：

| 配置 | Lane 数 | 每 Lane 头数 | 每 Lane tile | 说明 |
|---|---|---|---|---|
| WH, 小网格 | 4 | 32 | `1 × 18 = 18` tile | 恰好占满一个 tile 行 |
| WH | 8 | 16 | `1 × 18 = 18` tile (padded) | 16 行在 32 行 tile 里半空，靠 padding |
| BH | 16 | 8 | `1 × 18` tile (padded) | padding 更明显 |


#### 怎么走、谁动、之后怎么用

三步动作，一次性做完就结束：

1. **NCRISC 读 DRAM**：Qr 的 NCRISC 发 `noc_async_read_tile`，把本 Lane 的 Q shard 搬进本地 `cb_q_in`，`noc_async_read_barrier` 后本地 Q 就绪。
2. **BRISC 行多播**：Qr 的 BRISC 对本 Lane 其它 `N_col − 1` 颗 core 发一次 `noc_async_write_multicast`，把 `cb_q_in` 内容直接推到对方同名 `cb_q_in` 地址。
3. **信号量对齐**：多播落地后每个接收 core 的 `q_input_mcast_semaphore` +1；接收侧用 `noc_semaphore_wait` 等到 1 即视为可用。

做完这三步后，Q 就**不再从 DRAM 取、也不再走 NoC**。之后每颗 core 在遍历自己那一列所有 K chunk 的过程中，反复用这份 `cb_q_in` 做 QKᵀ。所以 `cb_q_in` 只需装得下一份 Q shard，但占用周期 = 整个算子。

一句话带走：**每 Lane 一次 DRAM 读 + 一次行多播，Q 就全网格到位；之后 Q 全程躺在 L1，不动。**

### B.3.3 K：从 DRAM 读进来，按列内分发

Q 已经在全网格到位，接下来是 K。目标：**让同一列（Column）的所有 core 的 `cb_k_in` 里都有当前迭代要算的那个 K chunk**。

K_latent 在单 batch 下就是一个 `[S, d_k]` 的矩阵（MLA 下 `Hkv = 1`）。沿 `S` 每 `Lk` 行切一个 chunk，整条序列变成 `T = ⌈S/Lk⌉` 个 chunk，每个 chunk 大小 `Lk × d_k`（DeepSeek-V3 `d_k=576`：`Lk=32` 时 18 tile，`Lk=64` 时 36 tile）。这 `T` 个 chunk 按 B.3.1 的 round-robin 分给各列对应的 DRAM bank：chunk `t` 落在 `bank_map[t mod N_col]`。**每列 i 独占 `bank_map[i]`，领走从 i 开始、按 `N_col` 为步长分配到该列的那一串 chunk**，约 `⌈T/N_col⌉` 个；所有列拼起来正好覆盖整条 S，彼此不相交、每个 K chunk 从 DRAM 被读**恰好一次**。

与 Q 一次性搬完不同，K 是**迭代**的：外层循环 `⌈T/N_col⌉` 次，每次每列搬一个 chunk，走一遍"读 → 列多播 → 被 TRISC 算 → 腾 slot"。所以 `cb_k_in` 必须是多 slot 循环 buffer（具体几 slot、怎么切 page 放到 B.4）。

```
   DRAM (N_col 个 bank，col i 独占 b[i])                                片上虚拟网格 (N_lane 行 × N_col 列)
   时间轴：iter 0 → iter 1 → ...                                       图示一次 iter 内 K 在网格里的流向

           col 0     col 1     col 2    ...    col N_col-1                  col 0    col 1    col 2  ...   col N_col-1
           b[0]      b[1]      b[2]            b[N_col-1]                 ┌────────┬────────┬────────┬─────┬──────────┐
          ┌──────┐  ┌──────┐  ┌──────┐        ┌──────┐           Lane 0    │   Ks   │   Ks   │   Ks   │ ... │    Ks    │
   iter 0 │ ch 0 │  │ ch 1 │  │ ch 2 │  ...   │ chN-1│                     ├────────┼────────┼────────┼─────┼──────────┤
          ├──────┤  ├──────┤  ├──────┤        ├──────┤           Lane 1    │   ▼    │   ▼    │   ▼    │ ... │    ▼     │
   iter 1 │ ch N │  │chN+1 │  │chN+2 │  ...   │ch2N-1│ ── read ──▶         ├────────┼────────┼────────┼─────┼──────────┤  列多播
          ├──────┤  ├──────┤  ├──────┤        ├──────┤           Lane 2    │   ▼    │   ▼    │   ▼    │ ... │    ▼     │  推给本列
   iter 2 │ ch2N │  │ch2N+1│  │ch2N+2│  ...   │ch3N-1│                     ├────────┼────────┼────────┼─────┼──────────┤  其它 N_lane-1
          ├──────┤  ├──────┤  ├──────┤        ├──────┤            ...      │  ...   │  ...   │  ...   │ ... │   ...    │  颗 core
          │ ...  │  │ ...  │  │ ...  │        │ ...  │                     ├────────┼────────┼────────┼─────┼──────────┤
          └──────┘  └──────┘  └──────┘        └──────┘           Lane N_lane-1│ ▼    │   ▼    │   ▼    │  ▼  │    ▼     │
                                                                            └────────┴────────┴────────┴─────┴──────────┘

   N = N_col；ch X = K_latent 沿 S 轴的第 X 个 chunk (= Lk × d_k 元素)
   一次 iter t：col i 的 Ks（一颗具体 core）从 b[i] 读 ch (t·N + i) → 入本地 cb_k_in → 沿列向本列其它 N_lane-1 颗 core 多播
```

每列挑一颗物理 core 作 **K sender (Ks)**（通常挑离对应 bank 最近那颗）。一次迭代内它做三件事：NCRISC 发 `noc_async_read_tile` 把一个 chunk 搬进本地 `cb_k_in` 的下一个可用 slot → BRISC 发 `noc_async_write_multicast` 推给同列其它 `N_lane − 1` 颗 core 的同位 slot → 接收侧 `k_input_mcast_semaphore` +1，看到值到位即可用。之后 TRISC 消费：`cb_wait_front` 拿到 slot 做 QKᵀ / softmax update / PV，算完 `cb_pop_front` 腾位置——腾出来 NCRISC 才能装下一 chunk，这条把 K 的 DRAM 读节奏和 TRISC 的计算节奏直接绑在一起。

从 core 的视角看关系是清楚的：**同列的 `N_lane` 颗 core 共享同一个 K chunk（列多播），但各自用不同的 Q shard（Lane 切分）——所以"同列 = 同 K、不同 Q；同行 = 同 Q、不同 K"**。

一句话带走：**`T = ⌈S/Lk⌉` 个 K chunk 被均匀切给 `N_col` 列，每列一颗 Ks 按序搬运——每迭代读一个 `Lk × d_k` 的 chunk + 一次列多播，直到整条 S 被消费完。**

### B.3.4 TRISC 把 (Q, K) 算成本地 (m, l, o)

到这一步，每颗 core 手上齐了两份数据：`cb_q_in` 里一份**常驻**的 Q shard（本 Lane 的 `Hq/N_lane` 头 × `d_k`），`cb_k_in` 里本列当前 iter 的 K chunk（`Lk × d_k`）。接下来是纯片内计算——TRISC 每 iter 走一轮 **QKᵀ → mask → online softmax 更新 → PV**，把结果累积到本地 running state `(m, l, o)`。

这一步有三个关键事实贯穿始终：**V 不单独搬**——做 PV 时 V 就是 `cb_k_in` 里这个 K chunk 的前 `d_v` 列切片，零拷贝、不额外占 CB、不走 NoC（性质 #1）；**mask 片上就地生成**——因果 / 尾块 padding / sliding window 全部由 BRISC 按 `(cur_pos, chunk_t)` 现算，不占 DRAM 也不占 CB；**整条 logits / P 从不具象**——它们只在 TRISC 的 dst register 里过一趟就消失，永远在 L1 具象保存的只有 `(m, l, o)` 三份 running stat（`m`、`l` 各是 `Hq/N_lane` 个标量，`o` 是 `Hq/N_lane × d_v` 的矩阵）。这条就是 Flash 的精髓（性质 #3）。

```
   单 iter 内一颗 core 的 TRISC 计算

       cb_q_in                                          cb_k_in
    (常驻 Q shard)                              (本列当前 iter 的 K chunk)
    [Hq/N_lane × d_k]                                 [Lk × d_k]
           │                                                │
           └────────────────  ① QKᵀ matmul  ────────────────┘
                                    │
                                    ▼
                         logits [Hq/N_lane × Lk]
                                    │
                           ② + mask (BRISC 现算)
                                    │
                                    ▼
                 ┌───────── ③ online softmax 更新 ─────────┐
                 │   m_new = max(m_old, rowmax(logits))    │
                 │   α     = exp(m_old − m_new)            │
                 │   P     = exp(logits − m_new)           │
                 │   l_new = α · l_old + rowsum(P)         │
                 └──────────────────────────────────────────┘
                                    │
                                    ▼
                         P [Hq/N_lane × Lk]     V = cb_k_in[:, :d_v]
                                                [Lk × d_v]   ← 零拷贝列视图
                                    │                 │
                                    └──── ④ PV mm ────┘
                                             │
                                             ▼
                         o_new = α · o_old + P · V       [Hq/N_lane × d_v]
                         (m, l, o) 原地更新；cb_pop_front 腾掉 K slot → NCRISC 才能装下一 chunk
```

整个算子对一颗 core 来说，就是把本列所有 chunk（`{i, i+N_col, i+2N_col, ...}`，约 `⌈T/N_col⌉` 个）依次过一遍上面这四步。跑完之后，本地 `(m, l, o)` 就是 **本 (Lane, Column) cell 的局部结果**——`m`、`l` 记录本 cell 覆盖的那段 K 上的 softmax 统计量，`o` 是对应的部分 output。这些局部结果接下来要沿 Lane 做归并树（B.3.5）才能合成完整 attention。

最后一条要点：`cb_k_in` 某 slot **必须等 TRISC 完成 ④ 步的 `o` 更新才能 `cb_pop_front`**——这条直接把 K 的 DRAM 读节奏和 TRISC 的计算节奏绑死（性质 #4），也是 B.4 讨论多 slot / page 切分流水的根本原因。

一句话带走：**每 iter 每 core 做 `QKᵀ + mask + online softmax + PV`，把结果攒到本地 `(m, l, o)`；本列 chunk 全跑完后，core 手上就是本 (Lane, Column) cell 的局部 output，等着 Lane 内归并。**

### B.3.5 Lane 内归并 + 写回 DRAM

B.3.4 跑完后，每 Lane 的 `N_col` 颗 core 各自攥着一份针对**同一 Q shard、不同 K chunk 段**的局部 `(m_i, l_i, o_i)`。B.3.5 做两件事：沿 Lane 行把这 `N_col` 份合成一份完整结果，然后归一化并写回 DRAM。

**怎么合并**。两份局部结果 `(m_A, l_A, o_A)`、`(m_B, l_B, o_B)` 的合并用的就是 online softmax 公式（和 B.3.4 step ③ 同一个，只是两个输入都是已经算好的 partial）：

```
   m_new = max(m_A, m_B);   α_A = exp(m_A − m_new),   α_B = exp(m_B − m_new)
   l_new = α_A · l_A + α_B · l_B;   o_new = α_A · o_A + α_B · o_B
```

操作可结合可交换（性质 #6），所以 `N_col` 份怎么配对都正确——做成二叉树最省时间：每层一半 core 出局，`⌈log₂ N_col⌉` 层后只剩一颗。硬件上一条 merge 边 = 一次 NoC 点对点搬运（发送侧把 `(m, l, o)` 写到接收侧的 `cb_ms_in / cb_out_o`，打 `reducer_semaphore`）+ 一次 TRISC 按公式合并（接收侧 `cb_wait_front` → 合入本地 `(m, l, o)` → `cb_pop_front`）。发送侧从此退出，接收侧活到下一 stage。

**归并到哪颗 core**。最后剩下的那颗就是 **Lane root**——host 侧在逻辑→物理映射表里为每个 Lane 指定的一颗固定 core，通常挑离 DRAM `output` bank 最近的位置。WH 用 `6→3→2→1`、BH 用 `8→4→2→1`，kernel 数学不变。

```
   Lane j (以 BH、8 列配置为例)

    c0   c1   c2   c3   c4   c5   c6   c7        ← 各持局部 (m_i, l_i, o_i)
     │    │    │    │    │    │    │    │
     └merge┘   └merge┘   └merge┘   └merge┘        ← stage 1: 8 → 4
        │         │         │         │
        └─ merge ─┘         └─ merge ─┘           ← stage 2: 4 → 2
              │                   │
              └──────── merge ────┘               ← stage 3: 2 → 1
                        │
                        ▼
                    Lane root (一颗固定 core)
                  拿到 (m_final, l_final, o_final)
```

**写回 DRAM**。Lane root 在树顶拿到 `(m_final, l_final, o_final)` 后做一次归一化 `out = o_final / l_final`，得本 Lane Q shard 的输出 `[Hq/N_lane × d_v]`，放进 `cb_out_final`。然后 Lane root 的 NCRISC 按本 Lane 在 `output` 里的行偏移（第 `Lane_id · Hq/N_lane` 行起的那几行）发 `noc_async_write_tile`，把 `cb_out_final` 的内容写到 DRAM `output` tensor。`N_lane` 个 Lane root 各写各那一段、互不重叠，拼起来正好是完整 `[B, Sq, Hq, d_v]`；`noc_async_write_barrier` + `output_semaphore` 一打，算子结束。

一句话带走：**`N_col` 份局部 `(m, l, o)` 沿二叉树 `⌈log₂ N_col⌉` 跳归并到 Lane root → 归一化 → root 的 NCRISC 写回 DRAM `output` 对应行；`N_lane` 个 Lane root 并行写完整个 output tensor。**

## B.4 chunk 内再切 page（流水线）

B.3.3 / B.3.4 里有一条约束：`cb_k_in` 的某个 slot 被 `cb_pop_front` 之前，NCRISC 装不进下一份 K；TRISC 又必须等 K slot 到位才能算。这把 **"DRAM 读 → NoC 多播 → TRISC 算"** 三段绑在了同一条时间轴上。如果 `cb_k_in` 只开 1 个 slot，这三段纯串行，总耗时 ≈ `T_read + T_mcast + T_compute`，哪段慢就拖住整体。B.4 要讲的就是怎么把它们叠起来。

**做法**：把一个 K chunk 再沿 tile 方向切成 `P` 个 **page**（每 page 若干 tile），`cb_k_in` 开 `P_slot ≥ 2` 个 slot。NCRISC 在读 page `n+1` 的时候，BRISC 正在多播 page `n`，TRISC 正在算 page `n-1`——三段在时间轴上互相掩盖。`P_slot = 2 ~ 4` 通常就够，再大会被 L1 容量挡住。chunk 之间也能接力：本 chunk 最后一个 page 还在算时，NCRISC 可以开始读下一 chunk 的第一个 page。

```
time →
NCRISC  read   │ p0 │ p1 │ p2 │ p3 │
BRISC   mcast  │    │ p0 │ p1 │ p2 │ p3 │
TRISC   compute│    │    │ p0 │ p1 │ p2 │ p3 │
                    └─── 三段叠起来的流水 ───┘
```

**两个旋钮**。
- **`P_slot`（slot 深度）**：决定流水深度。受限于每核 L1（≈ 1.5 MB），需要同时装下 `cb_q_in`、多 slot 的 `cb_k_in`、`cb_out_o`、`cb_ms_in`、`cb_mask`、`cb_out_final` 等所有 CB。
- **page size**：决定每段 stage 的粒度。page 太小，NCRISC 每 tile 的 setup 开销占比高；page 太大，三段叠不起来。经验值是让 `T_read(page) ≈ T_compute(page)`，两段就能恰好掩盖。

**对 Q 和归并的同理做法**。
- **Q**（B.3.2）：Q 只读一次且常驻，Qr 读进本地 `cb_q_in` 的那一次 DRAM read 也可以切 page 和后续第一批 K 的读并发，让 Q 的准备时间被 K 流水掩盖。
- **归并**（B.3.5）：多个 Q chunk（prefill）或多个 batch 时，本 chunk 的 Lane 归并可以和下一 chunk 的局部计算在时间上重叠。

一句话带走：**把 chunk 再切 page + 多开 `cb_k_in` slot，就能把 DRAM 读 / NoC 播 / TRISC 算三段叠起来跑；深度和 page 尺寸由 L1 容量与"带宽 ↔ 计算"平衡决定。**

## B.5 几点补充说明

下面几条不是可调旋钮，而是数据流在形状上必须满足的前后一致性；B.3 / B.4 各节用到的地方已经用 `(性质 #X)` 标了出处。

1. **V 是 K 的视图**。V 永远 = `cb_k_in` 里当前 K chunk 的前 `d_v` 列切片——没有 `cb_v_in`，也没有对 V 的独立 DRAM / NoC 搬运。
2. **K 从 DRAM 只读一次**。每个 K chunk 仅由它所在列的 sender core 从 DRAM 读一次，其余 `N_lane − 1` 颗 core 通过 NoC multicast 拿到。
3. **Logits 不具象**。全程用 online softmax 递推，`logits` 和 `P` 只在 TRISC 的 dst register 里过一趟；L1 里唯一具象保存的中间量是 `(m, l, o)` running stats。
4. **`cb_k_in` slot 的回收约束**。某个 slot 只有在 TRISC 完成它对应 K 的 QKᵀ + PV + `(m, l, o)` 更新之后才能 `cb_pop_front`，这是 B.4 流水必须尊重的一条同步规则。
5. **Q 进 `cb_q_in` 一次**。每 Lane 只做一次 DRAM 读 + 一次行多播，`cb_q_in` 的内容在整个算子期间不再变动。
6. **归并算子可结合可交换**。`(m, l, o)` 的 merge 是 online softmax 的对称版本，所以 Lane 归并树的拓扑可以自由选——WH 常用 `6→3→2→1`，BH 常用 `8→4→2→1`，kernel 数学一字不改。

> **decode vs prefill 的差别**：只差在 Q 的长度。decode 时 `Sq = 1`，`cb_q_in` 很小，主要开销在 K 列广播 + Lane 归并；prefill 时 Q 沿 `Sq` 再切 `Lq` 成多个 Q chunk，每个 Q chunk 独立走一遍 B.3 的完整流程，chunk 之间无归并。形状差别不影响以上 6 条。

---

# Part C — 形式化定义

Part C 是 Part B 的硬件无关版本，用符号把算子定义清楚。本次只先写 **输入输出**（C.1），后续章节（递推、并行结构、参数空间、代价模型、正确性不变量等）等 C.1 稳定后再逐步补齐。

## C.1 输入输出

Flash MLA 作为一个函数，其输入输出由一个参数向量 `W` 完全决定：

```text
W = ( B, Hq, Hkv, Sq, S, d_k, d_v, causal, mask_type )
```

| 符号 | 含义 | 值域 / 约束 |
|---|---|---|
| `B` | batch 大小 | `≥ 1` |
| `Hq` | Q 头数 | `≥ 1` |
| `Hkv` | KV 头数 | `≥ 1`；`Hq % Hkv == 0`（MLA 常取 `Hkv = 1`，令 `g = Hq / Hkv`）|
| `Sq` | Q 序列长度 | decode `= 1`；prefill `≥ 1` |
| `S` | KV 序列长度 | `≥ Sq` |
| `d_k` | K 每头 latent 维 | `> 0` |
| `d_v` | V 每头 output 维 | `0 < d_v ≤ d_k` |
| `causal` | 是否因果 | `{true, false}` |
| `mask_type` | mask 族 | `{none, causal, sliding_window(w), padding, sink}` |

给定 `W`，一次调用的 I/O 签名完全参数化如下——两个"真"输入 `Q`、`K_latent` 必选，`page_table`、`cur_pos` 按场景可选，`Output` 是算子唯一输出；`V` 和 `Mask` 不出现在签名里：`V` 恒等于 `K_latent[..., :d_v]`（K 在 `d_k` 维上的前 `d_v` 列零拷贝视图），`Mask` 由 `mask_type` 结合 `cur_pos` 和当前 chunk 位置在片上就地生成。

```text
输入：
  Q          : [B,  Sq, Hq,  d_k]                            — 必选
  K_latent   : [B,  S,  Hkv, d_k]                            — 必选
  page_table : [B,  ⌈S / page_block_size⌉]  (int32)          — 可选，paged KV 时提供
  cur_pos    : [B]                          (int32)          — 可选，decode 时提供

输出：
  Output     : [B,  Sq, Hq,  d_v]

不出现在签名中（定义性）：
  V          ≡  K_latent[..., :d_v]    ∈ R^{B × S × Hkv × d_v}     — 视图，非张量
  Mask       ≡  mask_fn(mask_type, cur_pos, chunk_pos)              — 片上就地生成
```

## C.2 参数全集：H + Θ

一次 Flash MLA 调用由三类参数共同决定：问题参数 `W`（§C.1，决定"算什么"）、硬件常量 `H`（平台决定，**不可调**，是 `Θ` 合法性的硬上限）、实现旋钮 `Θ`（用户 / 编译器决定"怎么算"）。结果只由 `W` 决定，延迟 / 吞吐 / 能耗由 `H` 与 `Θ` 的匹配决定。下表把 `H` 和 `Θ` 的所有分量一次列清楚；`*` 标一级旋钮，是 autotuner 优先 sweep 的对象。

```text
H = ( N_core, L1_cap, N_risc, N_NoC, BW_NoC, N_bank, BW_bank, BW_DRAM,
      FLOPS_peak, tile, DST )                                       硬件常量
Θ = ( G, T, Π, C, X )                                               实现旋钮
     网格   切块  流水  计算  扩机
```

| 组 | 符号 | 含义 | 值域 / 典型值（WH / BH）|
|---|---|---|---|
| **H 硬件** | `N_core` | 单设备可用 Tensix worker core 数 | WH 8×8 = 64；BH 8×10 = 80（worker 子网）|
|  | `L1_cap` | 单核 L1 SRAM 容量 | WH ≈ 1.5 MB；BH ≈ 1.5 MB（含所有 CB + 程序栈 + 半屏）|
|  | `N_risc` | 单核可编程 RISC 数 | 5：NCRISC、BRISC、TRISC0 / 1 / 2（各自异步流水）|
|  | `N_NoC` | NoC plane 数 | 2：NOC0、NOC1（互为镜像 torus）|
|  | `BW_NoC` | 单链路 NoC 带宽 | ~32 B/cycle（与频率相关）|
|  | `N_bank` | DRAM bank 数 | WH 12；BH 8 |
|  | `BW_bank` | 单 bank DRAM 带宽 | 架构 / 频率相关（≈ 25 GB/s 级）|
|  | `BW_DRAM` | 聚合 DRAM 带宽 | `≈ N_bank · BW_bank` |
|  | `FLOPS_peak` | 单核峰值吞吐（BF16）| 架构相关 |
|  | `tile` | 架构强制的 matmul tile | `32 × 32` 元素（所有 CB / matmul / 搬运粒度的基本单位）|
|  | `DST` | TRISC 目的寄存器容量 | 以 tile 计（WH 半 dst 4 tile / full 8；BH 更大）|
| **G 网格** | `N_col` * | Column 数（虚拟网格列数，K 方向并行度）| `≥ 1`；约束 `N_col · N_lane ≤ N_core` |
|  | `N_lane` * | Lane 数（虚拟网格行数，Q 方向并行度）| `≥ 1`；需满足 Q shard `(Hq / N_lane) · d_k · sizeof(dtype(Q))` 能装进 `L1_cap` |
|  | `loc(K)` * | `K_latent` 在 DRAM 的布局 | `{interleaved, ND-sharded(N_col), paged(page_table)}`；与 `N_col / assign / sender` 必须一致对齐（见 C3）|
|  | `loc(Q)` | `Q` 在 DRAM 的布局 | 通常 `interleaved`；不进入稳态流水，只影响 `qreader(b)` 一次性读的延迟 |
|  | `assign` | K chunk → Column 的分派函数 | 默认 round-robin：`assign(t) = t mod N_col` |
|  | `sender(i)` | 第 `i` 列从 DRAM 读 K 的那一行 | `∈ [0, N_lane)`；通常挑离 K bank `i` 物理最近的一行 |
|  | `qreader(b)` | 第 `b` 行 Lane 从 DRAM 读 Q 的那一列（Qr）| `∈ [0, N_col)`；通常挑离 Q bank 物理最近的一列，读完沿本行多播 `N_col − 1` 份 |
|  | `root(b)` | 第 `b` 行 Lane 的归并终点所在列 | `∈ [0, N_col)`；通常挑离 output bank 物理最近的一列 |
|  | `tree(b)` * | 第 `b` 行 Lane 的归并树拓扑 | 任意结合 / 交换二叉树（WH `6→3→2→1`；BH `8→4→2→1`）|
|  | `φ: (i, b) → core` * | 虚拟网格 → 物理 core 的映射 | 受 NoC 拓扑、DRAM bank 位置、multicast 矩形三者共同约束 |
| **T 切块** | `Lk` * | K chunk 长度（沿 `S` 切块的颗粒）| `Lk ∈ {32, 64, …, S}`，必须是 `tile` 的整数倍；派生 `T = ⌈S / Lk⌉` 段 |
|  | `Pk` * | K chunk 内 page 大小（流水颗粒）| `Pk ∣ Lk` 且 `Pk ≥ tile`；派生 `Np = Lk / Pk` 个 page |
|  | `Lq` * | Q chunk 长度（prefill 才用）| decode 固定 `= 1`；prefill `∈ {tile, 2·tile, …, Sq}` |
|  | `τ` | 每 Lane 负责的 Q head 数 | 由 `N_lane` 派生：`τ = Hq / N_lane` |
| **Π 流水** | `slots(cb_k_in)` * | K 侧环形缓冲 slot 数 = page 级流水深度 | `≥ 1`；典型 `2 ~ 4`，受 `L1_cap` 封顶，决定 "read ↔ mcast ↔ compute" 三段叠深度 |
|  | `depth_chunk` | chunk 级流水深度 | `≥ 1`；允许相邻 chunk 首尾接力 |
|  | `fanout_mcast` | K 多播扇出策略 | `{single-issue, forwarding-chain}` |
|  | `sync(K_t)` | K 到位的同步粒度 | `{chunk, page}` |
|  | `sync(reduce)` | Lane 归并的同步粒度 | `{step, stream}` |
| **C 计算** | `dtype(Q)` * | Q 精度 | `{bf16, fp16, fp8_e4m3, …}` |
|  | `dtype(K)` * | K 精度 | 同上 |
|  | `dtype(acc)` * | QKᵀ / PV 累加器精度 | `{fp32, fp16}`（受 `DST` 寄存器影响）|
|  | `dtype(m, l, o)` | 递推状态精度 | 通常 `fp32` |
|  | `kernel(QK)`, `kernel(PV)` | matmul 变体 | 实现相关（tile 尺寸、unpack 变体）|
|  | `kernel(softmax)` | online softmax 变体 | 实现相关 |
| **X 扩机** | `D` | 设备数 | `≥ 1` |
|  | `split(S)` | KV 在设备间切分 | `{contiguous, round_robin(D), hash}` |
|  | `reduce_D` | 跨设备 `(m, l, o)` 归并拓扑 | 结构同 `tree(b)` |

**派生量与一致性约束**。给定 `(W, H, Θ)` 就能直接算出下面这些量，它们不是旋钮、只是"账"——代价模型（C.6）和合法性判据（C1–C5）都在这些派生量上面建立：

```text
T              = ⌈S / Lk⌉                                           K chunk 总数
Np             = Lk / Pk                                            chunk 内 page 数
τ              = Hq / N_lane                                        每 Lane 的 Q head 数
|𝒞(i)|         = ⌈T / N_col⌉                                       每列（Column）要处理的 chunk 数
Bytes_L3(i)    ≈ (T / N_col) · Lk · d_k · sizeof(dtype(K))         每列（Column）的 DRAM K 读字节
size(cb_q_in)  = τ · d_k · sizeof(dtype(Q))                   单 slot，常驻整个算子
size(cb_out_o) = τ · d_v · sizeof(dtype(acc))                 单 slot，常驻整个算子
size(cb_k_in)  = slots(cb_k_in) · Pk · d_k · sizeof(dtype(K)) K 流水缓冲，多 slot 轮转
```

任何合法 `Θ` 必须同时满足：

```text
(C1)  size(cb_q_in) + size(cb_out_o) + size(cb_k_in) + 其它 CB + 程序栈  <  L1_cap
(C2)  N_col · N_lane                   ≤  N_core
(C3)  loc(K) 的 bank 条数              =  N_col，且 bank 布局与 assign / sender 对齐
                                           （§B.5 性质 #2：每 chunk 从 DRAM 只读一次）
(C4)  size(cb_q_in) 单 slot 常驻         （§B.5 性质 #5：Q 对每 Lane 只读一次）
(C5)  tile | Lk,   tile | Pk,   tile | Lq     （所有 matmul 颗粒是 tile 的整数倍）
```

换句话说，`H` 里每一项都可能在某个旋钮上形成硬上限：`L1_cap` 决定 `slots(cb_k_in)` 和 `N_lane` 的下限；`N_core` 决定 `N_col · N_lane` 的上限；`N_bank` 决定 `N_col` 的"最划算值"；`tile` 决定 `Lk / Pk / Lq` 必须对齐 32；`DST` 决定 `dtype(acc)` 和 `Pk` 的组合。所有代价模型与 autotuner 的目标函数（见 C.6）都建立在 `(H, Θ)` 这对符号上。

---

## C.3 数据流的形式化

把 §B.3 的六步数据流用 C.1 / C.2 的符号一次写清楚。所有符号沿用前面定义，这里只补三个派生集合：

```text
𝒲     = { w(i, b) : i ∈ [0, N_col), b ∈ [0, N_lane) }      所有 worker
𝒞(i)  = { t : assign(t) = i }                         第 i 列（Column）要处理的 K chunk 集合
τ     = Hq / N_lane                                     每 Lane 的 head 数
```

每个 worker 看到的切片（`t ∈ 𝒞(i)`，`T = ⌈S / Lk⌉`）：

```text
Q_shard(b) = Q[·, ·, b·τ : (b+1)·τ, :]             ∈ R^{Sq × τ × d_k}
K_t        = K_latent[·, t·Lk : (t+1)·Lk, ·, :]    ∈ R^{Lk × d_k}
V_t        ≡ K_t[:, :d_v]                           ∈ R^{Lk × d_v}  零拷贝视图
M_t        = mask_fn(mask_type, cur_pos, t·Lk)     ∈ R^{Sq × Lk}    片上生成
```

---

**① 读 + 播（数据进片）**。每条 K chunk 只进片一次、每份 Q shard 只进片一次：

```text
K：  ∀ i, ∀ t ∈ 𝒞(i) :   w(sender(i), ·) ←DRAM─ K_t        ─NoC mcast─▶ cb_k_in[w(i, b)], ∀ b
Q：  ∀ b              :   w(·, qreader(b)) ←DRAM─ Q_shard(b) ─NoC mcast─▶ cb_q_in[w(i, b)], ∀ i
```

`K` 每 chunk 触发一次（§B.5 性质 #2）；`Q` 整算子触发一次、之后常驻 `cb_q_in`（§B.5 性质 #5）。Mask 由 BRISC 就地算，不进 DRAM / NoC。

---

**② 本地 online softmax 递推**。每个 `w(i, b)` 维护一份 `(m, l, o)`，初始 `(−∞, 0, 0)`。按 `t ∈ 𝒞(i)` 从小到大：

```text
(F1)  S_t   = (Q_shard(b) · K_tᵀ) · scale + M_t        ∈ R^{(Sq·τ) × Lk}
(F2)  m'    = max( m, rowmax(S_t) )
(F3)  P_t   = exp( S_t − m' )
(F4)  α     = exp( m − m' )
(F5)  l'    = α · l + rowsum(P_t)
(F6)  o'    = α · o + P_t · V_t                         ← V_t = K_t[:, :d_v] 零拷贝
(F7)  (m, l, o) ← (m', l', o')
```

`Logits` / `P_t` 不落 L1（§B.5 性质 #3）。跑完 `𝒞(i)` 得到本 (Lane, Column) cell 的局部三元组：

```text
Λ(i, b) = (m, l, o)_i^b
```

---

**③ Lane 归并 + 写回**：

两份局部三元组的合并算子

```text
(m_a, l_a, o_a) ⊕ (m_b, l_b, o_b) ≜
    m*  = max(m_a, m_b),   α_a = exp(m_a − m*),   α_b = exp(m_b − m*)
    l*  = α_a · l_a + α_b · l_b
    o*  = α_a · o_a + α_b · o_b
```

`⊕` 结合 + 交换（§B.5 性质 #6），所以沿 `tree(b)` 做 `⌈log₂ N_col⌉` 层归并的结果唯一，落在 `w(root(b), b)`；随后归一化写回 DRAM：

```text
(m, l, o)*_b = ⊕_{i ∈ [0, N_col)} Λ(i, b)                  落在 w(root(b), b)
Output[·, ·, b·τ : (b+1)·τ, :]  =  o*_b / l*_b             w(root(b), b) ──DRAM──▶
```

`N_lane` 个 root 各写一段，互不重叠，拼起来就是完整的 `Output ∈ R^{B × Sq × Hq × d_v}`。

---

**等价性**。把 ①②③ 合起来：

```text
Output[·, ·, b·τ : (b+1)·τ, :]  =  o*_b / l*_b
                                    其中 (m, l, o)*_b = ⊕_{i} Λ(i, b)
                                         Λ(i, b)      = fold_{t ∈ 𝒞(i)} (F1)–(F7)
```

对 `|𝒞(i)|` 归纳 + `⊕` 结合律可证：结果与 C.1 一次性算 `softmax(Q · Kᵀ) · V` 完全一致，与 `𝒞(i)` 的内部顺序、`tree(b)` 的形状均无关——这正是 §B.3.5 里说"归并树拓扑随便选"、§B.3.4 里说"逐 chunk 过一遍 online softmax"都合法的形式化根据。

Π 里 `slots(cb_k_in) ≥ 2` 允许把 ①②③ 的读 / 播 / 算沿时间轴错开（§B.4），只要遵守"某 slot 被 `cb_pop_front` 必须等 (F1)–(F7) 对该 chunk 完成"（§B.5 性质 #4），上式仍然成立。


---

## 附录：本定义与既有 TT 文档的关系

本定义**取代**此前多份 TT 特定数据流描述（`flash-mla-current-dataflow-analysis.md`、`experimental-flash-mla-dataflow-analysis.md`、`flash-mla-impl-b-dse-formalization.md`）中关于"Flash MLA 数据流是什么"的定义性陈述。原文档继续作为"某个具体实现在本抽象下的投影"保留。论文 formalism、profile 结论叙述、tuner 设计一律以本文 Part C 的符号为基准语言。

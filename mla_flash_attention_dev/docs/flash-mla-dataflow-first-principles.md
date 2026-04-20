# Flash MLA 数据流：第一性原理定义

本文分两部分，可独立阅读：

- **Part A — 数据流怎么走**：结合 Tenstorrent 硬件讲清楚数据从哪来、怎么走、去哪里。
- **Part B — 形式化版**：把输入、所有可变参数、约束、代价模型用符号写清楚（硬件无关）。

---

# Part A — 数据流怎么走（Tenstorrent 视角）

## A.1 背景

Flash MLA = Flash（把长 K 切成 chunk，用 online softmax 流式累加，不一次算完整 softmax）+ MLA（V 不是独立张量，V 就是 K 的前若干列）。所以整个算子**只流动 Q 和 K**；用到 V 时直接在 K 的缓冲里按偏移取，V 从不单独搬、不单独存。

## A.2 硬件设置

上机前只需要搞清楚两件事：用到哪些 core，数据放在哪几个 DRAM bank。

**逻辑网格（算法视角）。** Flash MLA 先在算法层画一张 `N_S × C_S` 的**虚拟二维网格**。行（Lane）负责不同 Q shard，列（S-block）负责不同 K 段；每列里挑一个 logical 位置做 **sender**（唯一回 DRAM 读 K 的位置），每行里挑一个 logical 位置做 **root**（唯一写 output 的位置）。这张网格的形状只由算法参数决定，和物理 core 坐标无关。

```
                 S-block 0    S-block 1    ...    S-block N_S-1
              ┌────────────┬────────────┬───────┬──────────────┐
    Lane 0    │     *      │     *      │  ...  │      *       │ ──┐
              ├────────────┼────────────┼───────┼──────────────┤   │ Lane 内
    Lane 1    │            │            │       │              │ ──┤ 树形归并
              ├────────────┼────────────┼───────┼──────────────┤   │
    Lane C_S-1│    R       │            │       │              │ ──┘  (R=root)
              └────────────┴────────────┴───────┴──────────────┘
                  ▲            ▲            ▲        ▲
             DRAM bank 0   bank 1       ...    bank N_S-1
                          (* = S-block sender)
```

**逻辑 → 物理映射。** host 侧再用一张常量表把每个逻辑坐标 `(i, b)` 映射到一颗具体的 Tensix core 物理坐标（WH 常用 6×4=24 core，BH 常用 8×8=64 core）。这张映射要一起考虑三件事：(a) sender 所在的物理 core 要离它负责读的 DRAM bank 近；(b) 同列的 logical core 在物理上应连成规则矩形，让 `noc_async_write_multicast` 最省跳数；(c) NOC0 与 NOC1 的物理坐标互为镜像，换 NOC 时要换映射表。换句话说，逻辑网格可以按算法自由描述，物理映射则由硬件拓扑约束。

**DRAM 布局。** K_latent 按 `k_chunk_size` ND-sharded 到 `N_S` 个 bank，bank 顺序与 S-block 列号对齐，这样第 i 列的 sender（映射之后的那颗物理 core）永远只读第 i 个 bank。page_table、cur_pos、output tensor 也在 DRAM 分配好。**V 不需要任何准备**——整个算子不需要 DRAM 里有 V。

## A.3 数据流

下面按"存→读→传→算→归并→写回"的顺序讲。本小节先交代**数据怎么存在 DRAM 里**，后面几小节再讲读取、传递、计算、写回。

### A.3.1 数据怎么存在 DRAM 里

算子开跑前，DRAM 里就这几样东西：

| 张量 | 形状 | 存储方式 |
|---|---|---|
| `K_latent` | `[B, S, Hkv, d_k]` | 沿 S 轴切 chunk，chunk 轮转分配到 `N_S` 个 bank |
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

### A.3.2 Q：从 DRAM 取进来，按 Lane 分发

Q 的目标是：**让虚拟网格里每一颗 core 的 `cb_q_in` 里都有本 Lane 对应的那份 Q shard**。做法很直接——每 Lane 派一颗 core 回 DRAM 读一次，然后沿本 Lane 做一次 NoC 行广播。全程 Q 只穿过 DRAM 一次。

#### Q 怎么切 + Qr 怎么读怎么传（一张图）

decode 场景下 Q 的全貌是 `[Hq, d_k]`（暂时忽略 batch）。按 `Hq` 切 `C_S` 段，每段一条水平带，作为对应 Lane 的 Q shard；每条水平带由该 Lane 的 **Q reader** 从 DRAM 读进本地 `cb_q_in`，再沿本 Lane 行多播给其它 `N_S − 1` 颗 core。

```
   DRAM 侧 Q  [Hq × d_k]                         片上虚拟网格 (C_S 行 × N_S 列)
 ┌────────────────────────────┐                ┌─────┬─────┬─────┬───────────┐
 │ heads [0, Hq/C_S)          │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane 0
 │                            │                │     │     │     │           │
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │ heads [Hq/C_S, 2·Hq/C_S)   │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane 1
 │                            │                │     │     │     │           │
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │ heads [2·Hq/C_S, 3·Hq/C_S) │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane 2
 │                            │                │     │     │     │           │
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │          ...               │     ...        │ ... │ ... │ ... │    ...    │  ...
 ├────────────────────────────┤                ├─────┼─────┼─────┼───────────┤
 │ heads [(C_S-1)·Hq/C_S, Hq) │ ── read ──▶   │ Qr  │ ▶   │  ▶  │    ▶      │  Lane C_S-1
 │                            │                │     │     │     │           │
 └────────────────────────────┘                └─────┴─────┴─────┴───────────┘
           │                                    S-blk0 S-blk1 ...  S-blk(N_S-1)
           │                                      └──────── 行多播 ────────┘
           │
    Q 水平带 j = Lane j 的 Q shard
     大小 = (Hq/C_S) × d_k
     tile = ⌈Hq/C_S/32⌉ × (d_k/32)
```

图里几件事一次到位：

- **按 Lane 切头**。`Hq` 维度被切成 `C_S` 段（每段 `Hq/C_S` 个头 × `d_k` 列），左边的 DRAM Q 图已经按 Lane 画出横条。
- **每 Lane 一个 Qr**。虚拟网格每一行第一个 cell 标 `Qr`，是该 Lane 的 Q reader；其余 cell 标 `▶` 代表行内接收者。
- **Read 一次 + 行多播一次**。水平箭头 `── read ──▶` 是 DRAM → Qr 的 `noc_async_read_tile`；右侧虚拟网格里的横向 `▶` 链是 Qr 沿行做的 `noc_async_write_multicast`。
- **规模**。单 Lane Q shard = `(Hq/C_S) × d_k` 个元素 = `⌈Hq/C_S/32⌉ × (d_k/32)` 个 tile。

举几个典型尺寸（DeepSeek-V3：`Hq=128, d_k=576 = 18 tile 列`）：

| 配置 | C_S | 每 Lane 头数 | 每 Lane tile | 说明 |
|---|---|---|---|---|
| WH, 小网格 | 4 | 32 | `1 × 18 = 18` tile | 恰好占满一个 tile 行 |
| WH | 8 | 16 | `1 × 18 = 18` tile (padded) | 16 行在 32 行 tile 里半空，靠 padding |
| BH | 16 | 8 | `1 × 18` tile (padded) | padding 更明显 |


#### 怎么走、谁动、之后怎么用

三步动作，一次性做完就结束：

1. **NCRISC 读 DRAM**：Qr 的 NCRISC 发 `noc_async_read_tile`，把本 Lane 的 Q shard 搬进本地 `cb_q_in`，`noc_async_read_barrier` 后本地 Q 就绪。
2. **BRISC 行多播**：Qr 的 BRISC 对本 Lane 其它 `N_S − 1` 颗 core 发一次 `noc_async_write_multicast`，把 `cb_q_in` 内容直接推到对方同名 `cb_q_in` 地址。
3. **信号量对齐**：多播落地后每个接收 core 的 `q_input_mcast_semaphore` +1；接收侧用 `noc_semaphore_wait` 等到 1 即视为可用。

做完这三步后，Q 就**不再从 DRAM 取、也不再走 NoC**。之后每颗 core 在遍历自己那一列所有 K chunk 的过程中，反复用这份 `cb_q_in` 做 QKᵀ。所以 `cb_q_in` 只需装得下一份 Q shard，但占用周期 = 整个算子。

一句话带走：**每 Lane 一次 DRAM 读 + 一次行多播，Q 就全网格到位；之后 Q 全程躺在 L1，不动。**

### A.3.3 K：从 DRAM 读进来，按 S-block 列内分发

Q 已经在全网格到位，接下来是 K。目标：**让同一 S-block 列的所有 core 的 `cb_k_in` 里都有当前迭代要算的那个 K chunk**。

K_latent 在单 batch 下就是一个 `[S, d_k]` 的矩阵（MLA 下 `Hkv = 1`）。沿 `S` 每 `Lk` 行切一个 chunk，整条序列变成 `T = ⌈S/Lk⌉` 个 chunk，每个 chunk 大小 `Lk × d_k`（DeepSeek-V3 `d_k=576`：`Lk=32` 时 18 tile，`Lk=64` 时 36 tile）。这 `T` 个 chunk 按 A.3.1 的 round-robin 分给 `N_S` 个 DRAM bank：chunk `t` 落在 `bank_map[t mod N_S]`。**每列 i 独占 `bank_map[i]`，领走 chunk `{i, i+N_S, i+2·N_S, ...}` 这一串**，约 `⌈T/N_S⌉` 个；`N_S` 列拼起来正好覆盖整条 S，彼此不相交、每个 K chunk 从 DRAM 被读**恰好一次**。

与 Q 一次性搬完不同，K 是**迭代**的：外层循环 `⌈T/N_S⌉` 次，每次每列搬一个 chunk，走一遍"读 → 列多播 → 被 TRISC 算 → 腾 slot"。所以 `cb_k_in` 必须是多 slot 循环 buffer（具体几 slot、怎么切 page 放到 A.4）。

```
   DRAM (N_S 个 bank，col i 独占 b[i])                                 片上虚拟网格 (C_S 行 × N_S 列)
   时间轴：iter 0 → iter 1 → ...                                       图示一次 iter 内 K 在网格里的流向

           col 0     col 1     col 2    ...    col N_S-1                       col 0    col 1    col 2  ...   col N_S-1
           b[0]      b[1]      b[2]            b[N_S-1]                     ┌────────┬────────┬────────┬─────┬──────────┐
          ┌──────┐  ┌──────┐  ┌──────┐        ┌──────┐           Lane 0    │   Ks   │   Ks   │   Ks   │ ... │    Ks    │
   iter 0 │ ch 0 │  │ ch 1 │  │ ch 2 │  ...   │ chN-1│                     ├────────┼────────┼────────┼─────┼──────────┤
          ├──────┤  ├──────┤  ├──────┤        ├──────┤           Lane 1    │   ▼    │   ▼    │   ▼    │ ... │    ▼     │
   iter 1 │ ch N │  │chN+1 │  │chN+2 │  ...   │ch2N-1│ ── read ──▶         ├────────┼────────┼────────┼─────┼──────────┤  列多播
          ├──────┤  ├──────┤  ├──────┤        ├──────┤           Lane 2    │   ▼    │   ▼    │   ▼    │ ... │    ▼     │  推给本列
   iter 2 │ ch2N │  │ch2N+1│  │ch2N+2│  ...   │ch3N-1│                     ├────────┼────────┼────────┼─────┼──────────┤  其它 C_S-1
          ├──────┤  ├──────┤  ├──────┤        ├──────┤            ...      │  ...   │  ...   │  ...   │ ... │   ...    │  颗 core
          │ ...  │  │ ...  │  │ ...  │        │ ...  │                     ├────────┼────────┼────────┼─────┼──────────┤
          └──────┘  └──────┘  └──────┘        └──────┘           Lane C_S-1│   ▼    │   ▼    │   ▼    │  ▼  │    ▼     │
                                                                            └────────┴────────┴────────┴─────┴──────────┘

   N = N_S；ch X = K_latent 沿 S 轴的第 X 个 chunk (= Lk × d_k 元素)
   一次 iter t：col i 的 Ks（一颗具体 core）从 b[i] 读 ch (t·N + i) → 入本地 cb_k_in → 沿列向本列其它 C_S-1 颗 core 多播
```

每列挑一颗物理 core 作 **K sender (Ks)**（通常挑离对应 bank 最近那颗）。一次迭代内它做三件事：NCRISC 发 `noc_async_read_tile` 把一个 chunk 搬进本地 `cb_k_in` 的下一个可用 slot → BRISC 发 `noc_async_write_multicast` 推给同列其它 `C_S − 1` 颗 core 的同位 slot → 接收侧 `k_input_mcast_semaphore` +1，看到值到位即可用。之后 TRISC 消费：`cb_wait_front` 拿到 slot 做 QKᵀ / softmax update / PV，算完 `cb_pop_front` 腾位置——腾出来 NCRISC 才能装下一 chunk，这条把 K 的 DRAM 读节奏和 TRISC 的计算节奏直接绑在一起。

从 core 的视角看关系是清楚的：**同列的 C_S 颗 core 共享同一个 K chunk（列多播），但各自用不同的 Q shard（Lane 切分）——所以"同列 = 同 K、不同 Q；同行 = 同 Q、不同 K"**。

一句话带走：**`T = ⌈S/Lk⌉` 个 K chunk 被均匀切给 N_S 列，每列一颗 Ks 按序搬运——每迭代读一个 `Lk × d_k` 的 chunk + 一次列多播，直到整条 S 被消费完。**

### A.3.4 TRISC 把 (Q, K) 算成本地 (m, l, o)

到这一步，每颗 core 手上齐了两份数据：`cb_q_in` 里一份**常驻**的 Q shard（本 Lane 的 `Hq/C_S` 头 × `d_k`），`cb_k_in` 里本列当前 iter 的 K chunk（`Lk × d_k`）。接下来是纯片内计算——TRISC 每 iter 走一轮 **QKᵀ → mask → online softmax 更新 → PV**，把结果累积到本地 running state `(m, l, o)`。

这一步有三个关键事实贯穿始终：**V 不单独搬**——做 PV 时 V 就是 `cb_k_in` 里这个 K chunk 的前 `d_v` 列切片，零拷贝、不额外占 CB、不走 NoC（红线 #1）；**mask 片上就地生成**——因果 / 尾块 padding / sliding window 全部由 BRISC 按 `(cur_pos, chunk_t)` 现算，不占 DRAM 也不占 CB；**整条 logits / P 从不具象**——它们只在 TRISC 的 dst register 里过一趟就消失，永远在 L1 具象保存的只有 `(m, l, o)` 三份 running stat（`m`、`l` 各是 `Hq/C_S` 个标量，`o` 是 `Hq/C_S × d_v` 的矩阵）。这条就是 Flash 的精髓（红线 #3）。

```
   单 iter 内一颗 core 的 TRISC 计算

       cb_q_in                                          cb_k_in
    (常驻 Q shard)                              (本列当前 iter 的 K chunk)
    [Hq/C_S × d_k]                                    [Lk × d_k]
           │                                                │
           └────────────────  ① QKᵀ matmul  ────────────────┘
                                    │
                                    ▼
                         logits [Hq/C_S × Lk]
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
                         P [Hq/C_S × Lk]     V = cb_k_in[:, :d_v]
                                                [Lk × d_v]   ← 零拷贝列视图
                                    │                 │
                                    └──── ④ PV mm ────┘
                                             │
                                             ▼
                         o_new = α · o_old + P · V       [Hq/C_S × d_v]
                         (m, l, o) 原地更新；cb_pop_front 腾掉 K slot → NCRISC 才能装下一 chunk
```

整个算子对一颗 core 来说，就是把本列所有 chunk（`{i, i+N_S, i+2N_S, ...}`，约 `⌈T/N_S⌉` 个）依次过一遍上面这四步。跑完之后，本地 `(m, l, o)` 就是 **本 (Lane, S-block) cell 的局部结果**——`m`、`l` 记录本 cell 覆盖的那段 K 上的 softmax 统计量，`o` 是对应的部分 output。这些局部结果接下来要沿 Lane 做归并树（A.3.5）才能合成完整 attention。

最后一条要点：`cb_k_in` 某 slot **必须等 TRISC 完成 ④ 步的 `o` 更新才能 `cb_pop_front`**——这条直接把 K 的 DRAM 读节奏和 TRISC 的计算节奏绑死（红线 #4），也是 A.4 讨论多 slot / page 切分流水的根本原因。

一句话带走：**每 iter 每 core 做 `QKᵀ + mask + online softmax + PV`，把结果攒到本地 `(m, l, o)`；本列 chunk 全跑完后，core 手上就是本 (Lane, S-block) cell 的局部 output，等着 Lane 内归并。**

### A.3.5 Lane 内归并 + 写回 DRAM

A.3.4 跑完后，每 Lane 的 `N_S` 颗 core 各自攥着一份针对**同一 Q shard、不同 K chunk 段**的局部 `(m_i, l_i, o_i)`。A.3.5 做两件事：沿 Lane 行把这 `N_S` 份合成一份完整结果，然后归一化并写回 DRAM。

**怎么合并**。两份局部结果 `(m_A, l_A, o_A)`、`(m_B, l_B, o_B)` 的合并用的就是 online softmax 公式（和 A.3.4 step ③ 同一个，只是两个输入都是已经算好的 partial）：

```
   m_new = max(m_A, m_B);   α_A = exp(m_A − m_new),   α_B = exp(m_B − m_new)
   l_new = α_A · l_A + α_B · l_B;   o_new = α_A · o_A + α_B · o_B
```

操作可结合可交换（红线 #6），所以 `N_S` 份怎么配对都正确——做成二叉树最省时间：每层一半 core 出局，`⌈log₂ N_S⌉` 层后只剩一颗。硬件上一条 merge 边 = 一次 NoC 点对点搬运（发送侧把 `(m, l, o)` 写到接收侧的 `cb_ms_in / cb_out_o`，打 `reducer_semaphore`）+ 一次 TRISC 按公式合并（接收侧 `cb_wait_front` → 合入本地 `(m, l, o)` → `cb_pop_front`）。发送侧从此退出，接收侧活到下一 stage。

**归并到哪颗 core**。最后剩下的那颗就是 **Lane root**——host 侧在逻辑→物理映射表里为每个 Lane 指定的一颗固定 core，通常挑离 DRAM `output` bank 最近的位置。WH 用 `6→3→2→1`、BH 用 `8→4→2→1`，kernel 数学不变。

```
   Lane j (以 BH, N_S = 8 为例)

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

**写回 DRAM**。Lane root 在树顶拿到 `(m_final, l_final, o_final)` 后做一次归一化 `out = o_final / l_final`，得本 Lane Q shard 的输出 `[Hq/C_S × d_v]`，放进 `cb_out_final`。然后 Lane root 的 NCRISC 按本 Lane 在 `output` 里的行偏移（第 `Lane_id · Hq/C_S` 行起的那几行）发 `noc_async_write_tile`，把 `cb_out_final` 的内容写到 DRAM `output` tensor。`C_S` 个 Lane root 各写各那一段、互不重叠，拼起来正好是完整 `[B, Sq, Hq, d_v]`；`noc_async_write_barrier` + `output_semaphore` 一打，算子结束。

一句话带走：**`N_S` 份局部 `(m, l, o)` 沿二叉树 `⌈log₂ N_S⌉` 跳归并到 Lane root → 归一化 → root 的 NCRISC 写回 DRAM `output` 对应行；`C_S` 个 Lane root 并行写完整个 output tensor。**

## A.4 chunk 内再切 page（流水线）

*（占位：待 A.3 写完后重写这一节与 A.3 的流水部分合并）*

## A.5 decode vs prefill

只差在 Q 的长度与形状：

- **decode**：Q 长度 = 1，Q 在 Lane owner core 的 L1 上是 tiny-tile，拷贝代价可以忽略。压力全在 K 列（DRAM 读 + 列内 multicast）和 Lane 归并树上。
- **prefill**：Q 是一大段，需要沿 Q 再切 `Lq`。多了一条"Q chunk 并行"轴，每个 Q chunk 独立完成上面的完整流程，chunk 之间无归并。非因果 prefill 还会在每列内部再做一轮"K forwarding chain"（`noc_async_write_multicast` 链路化），本质仍是 K 列广播的放大版。

## A.6 不能碰的六条红线

数据流的形状由这六条固定；可调的只是"切多大、摆多少、流多深"这些旋钮（详见 Part B）。

1. V 永远是 K 的 L1 缓冲视图——不存在 cb_v_in、不存在对 V 的独立 NoC/DRAM 搬运。
2. 每段 K 从 DRAM → 片上**只发生一次**（每列的 sender core），其余 core 只通过 NoC multicast 获得。
3. 全程保持 online softmax 递推，任何时刻都不得把整条 Logits 具象化。
4. `cb_k_in` 的某个 slot 只有在 TRISC 完成它对应 chunk 的 QKᵀ + PV + `(m, l, o)` 更新之后才能被 `cb_pop_front`。
5. Q 对每个 Lane 只进入 cb_q_in 一次，生命周期覆盖整条 K 序列。
6. 归并算子可结合可交换，所以 Lane 归并树的形状可以自由选；这也是为什么 WH 可以用 6→3→2→1、BH 可以用 8→4→2→1 而无需改 kernel 数学。

---

# Part B — 形式化版

## B.1 输入规格

一次 Flash MLA 调用的输入由**问题张量**和**问题参数**两部分构成。

### B.1.1 问题参数 `W`（workload）

```text
W = ( B, Hq, Hkv, Sq, S, d_k, d_v, causal, mask_type )
```

| 符号 | 含义 | 约束 |
|---|---|---|
| `B` | batch size | `B ≥ 1` |
| `Hq` | Q 头数 | `Hq ≥ 1` |
| `Hkv` | KV 头数 | `Hkv ≥ 1`，且 `Hq % Hkv == 0`（MLA 中常取 `Hkv = 1`）|
| `Sq` | Q 序列长度 | decode: `Sq = 1`；prefill: `Sq ≥ 1` |
| `S` | KV 序列长度 | `S ≥ Sq` |
| `d_k` | K 每头的 latent 维度 | `d_k > 0` |
| `d_v` | V 每头的 output 维度 | `0 < d_v ≤ d_k` |
| `causal` | 是否因果 | `{true, false}` |
| `mask_type` | mask 族 | `{none, causal, sliding_window(w), padding, sink}` |

### B.1.2 问题张量

```text
Q         : [B, Sq, Hq,  d_k]
K_latent  : [B, S,  Hkv, d_k]
V         ≡  K_latent[..., :d_v]         (这是一条恒等，不是一次赋值)
Mask      : 由 mask_type 决定；若就地生成，则不作为输入张量
Output    : [B, Sq, Hq, d_v]
```

**输入侧不变量**（必须在 API 层保证）：

```text
I1.   V 不作为独立张量出现在输入/输出签名中。
I2.   若提供 page table（paged KV），则 K_latent 的存储布局由 page table 间接决定，但 I1 仍然成立。
```

### B.1.3 数学定义（单 `(b, h)` 对）

```text
scale     = 1 / sqrt(d_k)
Logits    = Q[b, :, h, :] · K_latent[b, :, h//g, :]ᵀ · scale   ∈ R^{Sq × S}
P         = softmax( Logits + Mask )                            ∈ R^{Sq × S}
V_view    = K_latent[b, :, h//g, :d_v]                          ∈ R^{S × d_v}
Output[b, :, h, :] = P · V_view                                  ∈ R^{Sq × d_v}

其中 g = Hq / Hkv 是 GQA group。
```

## B.2 流式递推定义

将 `S` 沿序列方向切成 `T = ⌈S / Lk⌉` 段。对每个 `(b, h)`，初始化：

```text
m⁽⁰⁾ = -∞     ∈ R^{Sq}
l⁽⁰⁾ = 0      ∈ R^{Sq}
o⁽⁰⁾ = 0      ∈ R^{Sq × d_v}
```

对 `t = 0, 1, …, T-1`，令 `K_t = K_latent[b, t·Lk:(t+1)·Lk, h//g, :]`，`V_t = K_t[:, :d_v]`，`M_t` 为对应的 mask 段：

```text
(F1)  S_t   = (Q[b, :, h, :] · K_tᵀ) · scale + M_t
(F2)  m⁽ᵗ⁺¹⁾ = max( m⁽ᵗ⁾, rowmax(S_t) )
(F3)  P_t   = exp( S_t - m⁽ᵗ⁺¹⁾ )
(F4)  α_t   = exp( m⁽ᵗ⁾ - m⁽ᵗ⁺¹⁾ )
(F5)  l⁽ᵗ⁺¹⁾ = α_t · l⁽ᵗ⁾ + rowsum(P_t)
(F6)  o⁽ᵗ⁺¹⁾ = α_t · o⁽ᵗ⁾ + P_t · V_t
```

最终：

```text
(F7)  Output[b, :, h, :] = o⁽ᵀ⁾ / l⁽ᵀ⁾
```

`V_t` 出现在 (F6) 中，但它始终是 `K_t` 的视图，不引入独立读写。

## B.3 并行结构

### B.3.1 Worker 网格

定义 worker 集合为二维网格：

```text
𝒲 = { w(i, b) : i ∈ [0, N_S),  b ∈ [0, C_S) }
```

- `i`：**S-block index**（Seq-K 分区）
- `b`：**Lane index**（Q 分区）

派生结构：

```text
S-block(i) = { w(i, b) : b ∈ [0, C_S) }
Lane(b)    = { w(i, b) : i ∈ [0, N_S) }
```

### B.3.2 Chunk 分派函数

```text
assign : [0, T) → [0, N_S)        (chunk → S-block)
```

`assign(t) = t mod N_S`（stride 分派）是默认选择。每个 `i` 分到的 chunk 集合：

```text
𝒞(i) = { t : assign(t) = i }
```

### B.3.3 跨 worker 归并

`Lane(b)` 内的 `N_S` 个 worker 各自持有局部统计 `(m_i, l_i, o_i)`，需合并为单一 `(m*, l*, o*)`。合并算子 `⊕`：

```text
(m_a, l_a, o_a) ⊕ (m_b, l_b, o_b) =
    m*  = max(m_a, m_b)
    α_a = exp(m_a - m*)
    α_b = exp(m_b - m*)
    l*  = α_a · l_a + α_b · l_b
    o*  = α_a · o_a + α_b · o_b
```

`⊕` 满足结合律与交换律，因此归并树拓扑可自由选择。

## B.4 存储层级抽象

```text
L3   共享大容量存储
L2   块内共享快存储（S-block 内所有 worker 可共同访问）
L1   每个 worker 的私有快存储
```

**放置函数** `π`：

| 对象 | `π(·)` | 备注 |
|---|---|---|
| `K_latent` | `L3` | 唯一来源 |
| `K_t`（当前活跃 chunk）| `L2(S-block(assign(t)))` | 每 chunk 在每个 S-block 内只有一份 |
| `V_t` | ⊥（无） | 视图，不占用任何 `π` |
| `Q` shard | `L1(w(i_b, b))` 初始，`L1(w(i, b))  ∀ i` 广播后 | `i_b` 为 Lane owner |
| `(m_i, l_i, o_i)` | `L1(w(i, b))` | 每 worker 一份 |
| `Mask_t`（边缘 chunk）| `L1(∪ 生成者)` | 由规则就地生成 |

## B.5 可变参数向量 Θ

一个 Flash MLA 实现的具体配置：

```text
Θ = ( T, G, M, Π, C, X )
```

以下表格逐项列出所有可变参数。`*` 标记的是一级旋钮（autotuner 应当优先 sweep）。

### B.5.1 T — Tiling

| 符号 | 含义 | 值域 | 派生关系 |
|---|---|---|---|
| `Lk` * | K chunk 长度 | `{16, 32, 64, …, S}` | `T = ⌈S / Lk⌉` |
| `Lq` * | Q chunk 长度 | `{1, …, Sq}` | prefill 才有意义；decode 固定为 `1` |
| `Pk` * | K page 大小 | `Pk ∣ Lk` | — |
| `Np`   | 每 chunk page 数 | — | `Np = Lk / Pk` |
| `τ`  * | 每 worker 一次处理的 Q head 数 | `τ ∣ Hq` | — |

### B.5.2 G — Grid / Topology

| 符号 | 含义 | 值域 |
|---|---|---|
| `N_S` * | S-block 数 | `N_S ≥ 1` |
| `C_S` * | 每 S-block 的 worker 数 | `C_S ≥ 1` |
| `shape(S_i)` | 第 i 个 S-block 的几何布局 | 实现相关（1D 链 / 2D 矩形等）|
| `tree(b)` * | Lane b 的归并树拓扑 | 任意满足 §B.3.3 `⊕` 结合律的树 |
| `sender(i)` | S-block i 的 L3→L2 读取者 | `∈ S-block(i)` |
| `assign` | chunk → S-block 分派 | 默认 `t mod N_S` |

约束：

```text
C1.  总 worker 数 = N_S · C_S ≤ 设备可用 worker 数
C2.  Q shard 数   = Hq / τ   ≤ C_S · (并行 launch 数)
C3.  每 Lane chunk 数 = |𝒞(i)| = ⌈T / N_S⌉
```

### B.5.3 M — Memory / Placement

| 符号 | 含义 | 值域 |
|---|---|---|
| `loc(K)`   * | K 在 L3 的布局 | `{contiguous, round_robin(bank_map), paged(page_table)}` |
| `loc(Q)`     | Q 初始位置 | `{L3, L1(w(·, b))}` |
| `buf_K`    * | L2 中并发 K chunk 缓冲数 | `≥ 1` |
| `buf_page` * | L1/L2 page 级缓冲深度 | `≥ 1` |
| `view_V`     | V 从 K 派生的偏移/步长 | 默认 `[0, d_v)` |

放置层不变量：

```text
I3.  V ∉ dom(π)       (V 不是可放置对象)
I4.  ∀ t : π(K_t) 在 L3 侧被引用的次数 = 1（对每个 S-block）
```

### B.5.4 Π — Pipeline

| 符号 | 含义 | 值域 |
|---|---|---|
| `ppl_chunk` * | chunk 级流水深度 | `≥ 1` |
| `ppl_page`  * | chunk 内 page 级流水深度 | `≥ 1` |
| `fanout_bcast` * | K 块内广播扇出 | `∈ [1, C_S - 1]` |
| `sync(K_t)` | K_t 到达的同步粒度 | `{chunk, page}` |
| `sync(reduce)` | 归并同步粒度 | `{step, stream}` |

### B.5.5 C — Compute

| 符号 | 含义 | 值域 |
|---|---|---|
| `dtype(Q)`, `dtype(K)` | 输入精度 | `{bf16, fp16, fp8_e4m3, …}` |
| `dtype(acc)` | 累加器精度 | `{fp32, fp16}` |
| `dtype(m, l, o)` | running stats 精度 | 通常 `fp32` |
| `kernel(QK)`, `kernel(PV)` | matmul 变体 | 实现相关 |
| `kernel(softmax)` | online softmax 变体 | 实现相关 |

### B.5.6 X — Scale-out

| 符号 | 含义 | 值域 |
|---|---|---|
| `D` | 设备数 | `≥ 1` |
| `split(S)` | S 在设备间切分 | `{contiguous, round_robin, hash}` |
| `reduce_D` | 设备间 `(m, l, o)` 归并拓扑 | 类 `tree(b)` |

## B.6 正确性不变量（必须由任意 Θ 满足）

以下六条对 `∀ Θ` 成立，否则该实现不是 Flash MLA。

```text
(R1)  MLA 不变量：     ∀ t. V_t ≡ view(K_t, [0, d_v))。  V 不出现在 π 中。

(R2)  单次远程读：    ∀ t ∀ S-block i = assign(t).
                      |{ L3 → L2 搬运事件 (K_t, i) }| = 1。

(R3)  Online softmax 单调性：
                      每个 worker 的状态转移必须与 (F2)–(F6) 一致；
                      不得在任意时刻具象化完整 softmax(Logits)。

(R4)  Chunk 封闭：    π(K_t) 的缓冲释放 ⇒ 该 worker 已完成 (F3)–(F6) 对 K_t 的消费。

(R5)  Q 局部性：      ∀ Lane(b) ∀ w ∈ Lane(b).
                      Q 进入 L1(w) 的次数 = 1（覆盖该 Lane 的全部 T_i chunk）。

(R6)  归并可结合：     ⊕ 满足结合律与交换律；
                      故 tree(b) 的形状只影响延迟，不影响结果。
```

## B.7 一阶代价模型

用 Θ 可以直接给出硬件无关的量级代价。令 `s = sizeof(dtype(K))`，`s_a = sizeof(dtype(acc))`。

**L3 读字节数（每 Lane 共享）**：

```text
Bytes_L3 = T · Lk · d_k · s
```

**片上广播字节数（每 S-block）**：

```text
Bytes_bcast = T · Lk · d_k · s · (C_S - 1) / fanout_bcast
```

**Lane 归并字节数**：

```text
Bytes_reduce = depth(tree(b)) · Sq · (d_v + 2) · s_a
```

**计算量（每 Lane）**：

```text
FLOPs = 2 · Sq · S · d_k    (QK^T)
      + 2 · Sq · S · d_v    (PV)
      + Θ(Sq · S)           (softmax 及辅助)
```

**总 latency 的 first-order 估计**：

```text
T_latency(Θ) ≈ max(
    Bytes_L3   / BW_L3,
    Bytes_bcast/ BW_L2,
    FLOPs      / FLOPS_peak,
    depth(tree(b)) · t_reduce_step
)
```

autotuner 目标函数形式为：

```text
minimize   α · T_latency(Θ) + β / Throughput(Θ) + γ · Energy(Θ)
subject to (R1) ∧ (R2) ∧ (R3) ∧ (R4) ∧ (R5) ∧ (R6)
           ∧ C1 ∧ C2 ∧ C3
           ∧ I1 ∧ I2 ∧ I3 ∧ I4
```

## B.8 与具体实现对齐的检查清单

给定一个声称是 Flash MLA 的实现，填出以下五项即可判定它在本定义下的位置：

| 项 | 内容 |
|---|---|
| **1. 网格** | `(N_S, C_S, shape(S_i), tree(b), sender(i), assign)` |
| **2. 切块** | `(Lk, Pk, Np, Lq, τ)` |
| **3. 放置** | `(loc(K), loc(Q), view_V, buf_K, buf_page)` |
| **4. 流水** | `(ppl_chunk, ppl_page, fanout_bcast, sync(K_t), sync(reduce))` |
| **5. 不变量验证** | 逐条检查 R1–R6；任一条被破坏即非 Flash MLA |

---

## 附录：本定义与既有 TT 文档的关系

本定义**取代**此前多份 TT 特定数据流描述（`flash-mla-current-dataflow-analysis.md`、`experimental-flash-mla-dataflow-analysis.md`、`flash-mla-impl-b-dse-formalization.md`）中关于"Flash MLA 数据流是什么"的定义性陈述。原文档继续作为"某个具体实现在本抽象下的投影"保留。论文 formalism、profile 结论叙述、tuner 设计一律以本文 Part B 的符号为基准语言。

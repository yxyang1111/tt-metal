# Flash MLA 数据流：Q流 / KV流 双视角形式化

> 作者注：这份文档从"两条正交的数据流"角度重新定义 Flash MLA 的数据流。它与 `flash-mla-dataflow-first-principles.md` 互补——前者回答"算的是什么 / 在几条并行轴上展开"，本文回答"数据怎么流、在哪里交汇"。
>
> 本文不使用任何 Tenstorrent 特定术语（RISC 角色、CB、NOC、semaphore 等）。TT 上的具体实现是本抽象的一个投影，不是定义本身。

---

## 1. 为什么换成"流"的视角

Attention 计算本身是一个二维问题：一边是 Q（行），一边是 K/V（列）。任何 Flash 实现都必须对这张 `Nq × T` 的 tile 网格做**块级遍历**，并把遍历产生的中间量沿一条方向累加（Flash 的 online softmax 把这条方向锁死为 K 方向）。

过去的描述（含本项目之前的文档）通常只讲"chunk 级循环 + 并行轴"，这套描述在一维上是准确的，但对**数据的物理流向**留白过多：

- K chunk 在片上"广播"——广播是沿哪个方向？
- `(m, l, o)` 归并——归并是沿哪个方向？
- Q 为什么能只搬一次？

把整套数据流拆成**两条互相垂直、各自单调的流**后，这些问题都有统一答案：

- **Q流 Φ_Q**：固定 Q-tile、沿 K 方向前进的流。它承载 Q 常驻 + 运行状态 `(m, l, o)` 的演化。
- **KV流 Φ_K**：固定 KV-tile、沿 Q 方向分发的流。它承载 K/V 数据的一次性远程读 + 片上分发。

两条流在每一个**计算 cell** 上交汇。**交汇点**就是所有真正的算术发生的地方，也是所有同步原语存在的理由。

这份文档把这个 picture 形式化。

---

## 2. 基本对象：Tile 网格与 Compute Cell

### 2.1 记号

沿用 §`flash-mla-dataflow-first-principles.md` 第 2 节的原始记号：`B, Hq, Hkv, Sq, S, d_k, d_v`。本文以"单 head 视角"展开，`(b, h)` 维度是 embarrassingly parallel 的外循环，不在双流讨论范围内。

### 2.2 Tile 切分

沿 Q 切 `Lq`，沿 K 切 `Lk`：

```text
Nq = ⌈Sq / Lq⌉             # Q 方向 tile 数
T  = ⌈S  / Lk⌉             # KV 方向 tile 数

Qtile[q] ∈ R^{Lq × d_k}     q ∈ [0, Nq)
Ktile[k] ∈ R^{Lk × d_k}     k ∈ [0, T)
Vtile[k] ≡ Ktile[k][:, 0:d_v]   # MLA 不变量：视图，不独立存储
```

decode 特例：`Lq = 1`，`Nq = 1`，整张网格退化为一条 `1 × T` 的单行。

### 2.3 Compute Cell

定义 `Cell(q, k)` 为处于网格位置 `(q, k)` 的一个原子计算单元：

```text
输入：Qtile[q],  Ktile[k],  Vtile[k] (视图)
输出：m_qk, l_qk, o_qk —— 合并进 Q-tile q 的 running stats
```

Cell 内部做的事就是 `flash-mla-dataflow-first-principles.md` 的 (1)–(6)：

```text
S_qk = Qtile[q] · Ktile[k]ᵀ / sqrt(d_k) + Mask_qk
m_qk = rowmax(S_qk)
P_qk = exp(S_qk - m_qk)
o_qk = P_qk · Vtile[k]
l_qk = rowsum(P_qk)
```

最后用 online softmax 合并律把 `(m_qk, l_qk, o_qk)` 并入 `(m, l, o)[q]`。

关键：**Cell 是原子、无状态的算子**；所有状态都在"流"里。

---

## 3. 两条流的定义

### 3.1 Q流 Φ_Q(q)

固定 `q`、沿 `k` 从 0 扫到 `T-1`：

```text
Φ_Q(q) := ( Qtile[q],  (m, l, o)[q],  k = 0 → 1 → ... → T-1 )
```

特征：
- **Q-tile 在流上常驻**：整条 Φ_Q(q) 只需一份 Qtile[q]。
- **运行状态沿流单调演化**：`(m, l, o)[q]` 必须严格按 online softmax 合并律更新，但各 Cell 输出 `(m_qk, l_qk, o_qk)` 可以乱序到达再按结合律归并。
- **流的终点**是归一化：`O[q] = o[q] / l[q]`。

Q流数目 = `Nq`。decode 时只有 1 条。

### 3.2 KV流 Φ_K(k)

固定 `k`、沿 `q` 从 0 扫到 `Nq-1`：

```text
Φ_K(k) := ( Ktile[k],  Vtile[k]-as-view,  q = 0 → 1 → ... → Nq-1 )
```

特征：
- **KV-tile 在流上常驻**：整条 Φ_K(k) 共享一份 Ktile[k]；V 不占用额外流。
- **流上每一站都消费**：每个 Cell 把 Ktile[k] 与对应的 Qtile[q] 撞到一起。
- **流的生命期**是一次远程读到所有消费者都用完。

KV流数目 = `T`。prefill 和 decode 没有结构差异，只差 `T` 值大小。

### 3.3 两条流的不对称性

| 性质 | Q流 Φ_Q | KV流 Φ_K |
|---|---|---|
| 携带数据 | Qtile + `(m, l, o)` | Ktile（含 V 视图） |
| 沿流方向的算子 | **归并**（online softmax 合并） | **分发**（同一份 K 到多个 q） |
| 沿流方向是否有数据依赖 | 有（合并前后） | 无（各 q 独立消费） |
| 结合律 | 满足（见 R6） | N/A（分发无归约） |
| 片外 I/O | 仅 Q 初始读 + O 写出 | K 每 tile 一次远程读 |

这张表是整套 Flash MLA 数据流的骨架。后面一切通信、同步、缓冲规则都从这里自然推出。

---

## 4. 交汇：Cell 的双重归属

每个 `Cell(q, k)` 同时属于 `Φ_Q(q)` 和 `Φ_K(k)`：

```text
Cell(q, k) ∈ Φ_Q(q) ∩ Φ_K(k)
```

换句话说，整张网格是两族流的**笛卡尔交**：

```text
网格 = { Cell(q, k) : (q, k) ∈ [0, Nq) × [0, T) }
     = ⋃_{q} Φ_Q(q)                           # 横切
     = ⋃_{k} Φ_K(k)                           # 纵切
```

两种切法不是选择题，而是对**同一份计算**的两种观察角度：

- 站在 Q 侧看，数据是 KV 一段一段涌入、Q 原地不动、`(m, l, o)` 不断更新。
- 站在 KV 侧看，数据是 Q 一个一个路过、KV 原地不动、扇出到所有 q。
- 站在 Cell 侧看，这里是两条流的交汇点、原子计算的发生地。

**设计空间的本质**：一条物理流（worker 发出的 load/store 序列）究竟"乘坐"哪条流，是由 placement 决定的——这是 §6 要回答的问题。

---

## 5. 三类数据的生命周期（流语义）

Flash MLA 里实际在动的数据只有三类，按所属流分类：

### 5.1 归属 Q流 的数据

| 对象 | 初始化 | 生命期 | 终止 |
|---|---|---|---|
| `Qtile[q]` | 流开始时从上游（或片外）加载一次 | 覆盖整条 Φ_Q(q) | 流末尾即可丢弃 |
| `(m, l, o)[q]` | 流开始时置 `(-∞, 0, 0)` | 覆盖整条 Φ_Q(q) | 归一化后输出 `O[q]` |
| `Mask_qk`（因果/尾块/sliding window）| 每个 Cell 进入前按 `(q, k)` 公式即时构造 | 单 Cell | 无 |

### 5.2 归属 KV流 的数据

| 对象 | 初始化 | 生命期 | 终止 |
|---|---|---|---|
| `Ktile[k]` | 流开始时从片外 L3 读一次 | 覆盖整条 Φ_K(k) | 所有 Cell 完成后释放 |
| `Vtile[k]` (view) | 与 Ktile[k] 同生 | 与 Ktile[k] 同生 | 与 Ktile[k] 同死 |
| `page_table[k]` (paged KV) | 读 `Ktile[k]` 前拿到 | 等同 Ktile[k] 读取过程 | 用完即丢 |

### 5.3 归属 Cell 的纯临时量

| 对象 | 生命期 |
|---|---|
| `S_qk`, `P_qk`, `m_qk`, `l_qk`, `o_qk` | 仅在 Cell 内部，出 Cell 即并入 Q流 |

**不变量（强条件，R1 的流语义版本）**：

> V 永远是 KV流的"免费第二视图"，在任何时刻、任何存储层级都不独立存在；Vtile 的读取 = 对已驻留的 Ktile 同一缓冲的切片访问。

---

## 6. 物理 Mapping：两条流铺在 worker 网格上

### 6.1 Worker 网格

沿用 `flash-mla-dataflow-first-principles.md` 的二维 worker 网格 `G[i, b]`：

```text
G[i, b]      i ∈ [0, N_S),  b ∈ [0, C_S)
```

把 Q-tile 和 KV-tile 分别指派给两个轴：

```text
K-partition:  k ∈ K_i    iff  k 属于 i 号 S-block 的一段 K（|K_i| = T / N_S）
Q-partition:  q ∈ Q_b    iff  q 属于 b 号 Lane 的 Q-shard（|Q_b| = Nq / C_S）
```

Worker `G[i, b]` 的工作集 = 它负责的 Cell 集：

```text
Cells(i, b) = { Cell(q, k) : q ∈ Q_b, k ∈ K_i }
```

### 6.2 流在网格上的方向

| 流 | 在网格上的方向 | 每条流在多少 worker 上流过 |
|---|---|---|
| Φ_Q(q), q ∈ Q_b | 沿 **i 轴** 方向（"纵列"方向，固定 b）| 所有 `N_S` 个 worker（这就是一个 Lane） |
| Φ_K(k), k ∈ K_i | 沿 **b 轴** 方向（"横行"方向，固定 i）| 所有 `C_S` 个 worker（这就是一个 S-block） |

这同时定义了两个集合：

- **Lane(b)** = `{G[i, b] : i ∈ [0, N_S)}` 是 Φ_Q 的物理承载。
- **S-block(i)** = `{G[i, b] : b ∈ [0, C_S)}` 是 Φ_K 的物理承载。

两者的交 `G[i, b]` 就是 Cell 的承载点。**每个 worker 同时在一条 Q流和一条 KV流上。** 这是"交汇"在物理层的精确对应。

### 6.3 每条流的角色

**Φ_K(k) 在一个 S-block 内的执行**：

1. 选定一个 worker 为 `sender(i)`，由它完成 L3 → 片上的唯一一次读（R2）。
2. 其余 `C_S - 1` 个 worker 在本 S-block 内**接收**（多播 / 逐播 / 视图），构成 KV流的中段。
3. S-block 内所有 Cell `Cell(q, k)`, `q ∈ Q_b, b ∈ [0, C_S)` 完成后，流的本段生命期结束，缓冲可回收。

**Φ_Q(q) 在一个 Lane 内的执行**：

1. Qtile[q] 放在 Lane 内某个 owner worker 的 L1（通常是 `i=0` 那一个）。
2. Lane 内其他成员在需要时从 owner 拉取 Qtile[q]（R5），**不回 L3**。
3. 每个 `G[i, b]` 完成它负责的 `Cells(i, b)` 后，得到部分 `(m, l, o)[q] for q ∈ Q_b`。
4. 沿 Lane 按 `tree(b)` 归并，最终归一化得到 `O[q] for q ∈ Q_b`。

---

## 7. 通信需求从流语言自然推出

### 7.1 KV流 → 需要"分发"

Φ_K(k) 的流内操作是分发，对应 S-block 内的广播：

```text
bytes(Φ_K broadcast, per chunk) = Lk · d_k · sizeof(dtype) · (C_S - 1)
```

`C_S` 越大，KV流内广播总流量越大，但每条流只读一次 L3。

### 7.2 Q流 → 需要"归并"

Φ_Q(q) 的流内操作是归并，对应 Lane 内的 reduction tree：

```text
bytes(Φ_Q reduce, per tile, per stage) = Lq · (d_v + 2) · sizeof(acc)
number of stages = depth(tree(b))
```

`N_S` 越大，每条 Q流经过的 stage 越多，归并代价越重；但 KV 并行度越高，带宽瓶颈越软。

### 7.3 L3 流量与 Q流 / KV流的非对称性

由于 V 是 Vtile 的视图、且 R1–R2 成立：

```text
bytes(L3 read, total)  =  ΣΣ 每条 Φ_K 一次 Ktile 读
                        =  T · Lk · d_k · sizeof(dtype)           # 仅 K，完全没 V
```

Q流引起的 L3 流量：

```text
bytes(L3 Q read) =  Nq · Lq · d_k · sizeof(dtype)    # 每条 Φ_Q 仅初始 1 次
bytes(L3 O write)=  Nq · Lq · d_v · sizeof(dtype)    # 每条 Φ_Q 仅末尾 1 次
```

对长序列 `S ≫ Sq` 而言，KV流主导 L3 带宽；对 decode（`Sq=1`）而言更是如此。**这是"长序列 decode 是 KV 带宽 bound"的形式化表述。**

### 7.4 片上流量总和

```text
bytes(on-chip)
  ≈ 分发（KV流内）:   T · Lk · d_k · sizeof(dtype) · (C_S - 1)
  + 归并（Q流内）:    Nq · depth(tree) · Lq · (d_v+2) · sizeof(acc)
  + Q 流内 Q 拉取:   Nq · Lq · d_k · (C_S - 1)           # R5，相对小
```

三项的相对大小直接决定"加大 N_S 还是加大 C_S"这类旋钮的走向，是 autotuner 的 first-order 目标函数。

---

## 8. 参数向量 Θ（双流视角重写）

把 `flash-mla-dataflow-first-principles.md` §7 的 Θ 重新按"流"维度组织，就得到下面更紧凑的形式：

### 8.1 切片参数（Tile 颗粒）

| 参数 | 含义 | 影响的流 |
|---|---|---|
| `Lq` | Q-tile 长 | Φ_Q(q) 每 Cell 的计算密度 |
| `Lk` | KV-tile 长 | Φ_K(k) 每 Cell 的计算密度、一次远程读的块大小 |
| `Pk` | page 大小（`Lk = Np · Pk`）| Φ_K 内部分发/同步的最小单位 |

### 8.2 流密度参数

| 参数 | 定义 | 含义 |
|---|---|---|
| `Nq` | `⌈Sq / Lq⌉` | Q流总数 |
| `T`  | `⌈S / Lk⌉`  | KV流总数 |

### 8.3 流物理承载参数

| 参数 | 含义 |
|---|---|
| `N_S` | Φ_K 的 i 维并行度（S-block 数）。每条 Φ_Q 穿过 N_S 个 stage |
| `C_S` | Φ_Q 的 b 维并行度（Lane 容量）。每条 Φ_K 扇出到 C_S 个成员 |
| `Q_b` 划分 | Q流到 b 轴的映射 |
| `K_i` 划分 | KV流到 i 轴的映射 |
| `sender(i)` | Φ_K(k), k∈K_i 的远程读 owner |
| `tree(b)` | Φ_Q(q), q∈Q_b 的归并拓扑 |

### 8.4 流内缓冲参数

| 参数 | 含义 |
|---|---|
| `buf_K` | 每条 Φ_K 的 staging 深度（预取多少 KV-tile）|
| `buf_page` | Φ_K 内 page 级缓冲深度 |
| `buf_stats` | Φ_Q 内部分 `(m, l, o)` 的合并缓冲深度 |
| `buf_Q` | Φ_Q 内 Qtile 的备份数（通常 1，R5 下 Lane 共用 owner 的那一份）|

### 8.5 流内同步参数

| 参数 | 含义 |
|---|---|
| `π_K` | Φ_K 内"到达"的同步粒度（page / chunk） |
| `π_Q` | Φ_Q 内"合并"的同步粒度（每 Cell 后 / 每批 Cell 后） |

### 8.6 数值与 scale-out

`dtype`、`kernel(QK/softmax/PV)`、多设备 `split(S)` 与 `flash-mla-dataflow-first-principles.md` 一致，不再重复。

**总结**：Θ 在双流语言下有更清晰的分层——**流密度 (Nq, T)** 是输入派生的结构量，**流物理承载 (N_S, C_S, sender, tree)** 是 placement 旋钮，**流内缓冲/同步 (buf_\*, π_\*)** 是 pipeline 旋钮。这三层正好对应 autotuner 的三级搜索空间。

---

## 9. 不变量 R1–R6（流语言重述）

这六条与 `flash-mla-dataflow-first-principles.md` §8 的 R1–R6 同源、同义，但用"流"的话更直观：

**R1（MLA · V-免流）**
Φ_V(k) 不作为独立流存在；任何对 Vtile[k] 的消费都解析为对 Φ_K(k) 携带的 Ktile[k] 缓冲的偏移视图。

**R2（KV流远程读唯一性）**
每条 Φ_K(k) 从 L3 到片上的搬运恰好发生一次；该流在 S-block 内的扩散完全由 sender → 其余成员的片上路径完成。

**R3（Q流 online softmax 单调）**
每条 Φ_Q(q) 内，`(m, l, o)` 仅通过 online softmax 合并律演化；禁止在 Φ_Q 中途构造完整 `softmax(QK)`。

**R4（KV流封闭）**
Φ_K(k) 承载的 Ktile 缓冲，只有在所有 `{Cell(q, k) : q ∈ Q_b, b ∈ [0, C_S)}` 完成并合并进各自 Q流之后才能释放；违反则出现"还在读就被下一条 Φ_K 覆盖"。

**R5（Q流局部性）**
每条 Φ_Q(q) 的 Qtile[q] 只从片外加载一次、生命周期覆盖整条流；流内其他 worker 从 Lane owner 就近拉取，不回 L3。

**R6（Q流归并可结合）**
Φ_Q(q) 上的 `(m, l, o)` 合并算子满足结合律与交换律，因此 `tree(b)` 的具体形状可自由选择；KV流 `k` 的处理顺序在 R3 之下也可乱序（只要并入 Q流时仍用合并律）。

任何一条被破坏，都可以用这份语言明确指出："某某实现打破了 R_i"，比如：
- 把 V 单独 readback → 违反 R1；
- 把同一条 Φ_K 的 Ktile 读两次 → 违反 R2；
- 把部分 Cell 的 softmax 先"完整"再合并 → 违反 R3；
- Ktile 缓冲提前回收 → 违反 R4；
- Qtile 回 L3 → 违反 R5；
- reduce 必须串行某种顺序才能结果正确 → 违反 R6。

---

## 10. decode / prefill 在双流视角下的退化形态

| 维度 | decode | prefill |
|---|---|---|
| `Lq, Nq` | `Lq = 1, Nq = 1` | `Lq = tile 长, Nq > 1` |
| Q流条数 | 1 | `Nq` |
| Q流结构 | 一条 Φ_Q 横跨所有 Φ_K；所有 worker 同属一条 Q流 | `Nq` 条独立 Q流；不同 Q流互不通信 |
| KV流条数 | `T = ⌈S / Lk⌉` | 与 decode 相同 |
| KV流结构 | 每条 Φ_K 在 S-block 内广播 | 每条 Φ_K 在 S-block 内广播（相同） |
| 瓶颈来源 | Φ_K 的 L3 读 + Φ_Q 唯一归并 | Φ_K 的 L3 读 + 多条 Φ_Q 互相挤片上广播 |
| 额外并行轴 | 无 | A_Q：不同 Q流间完全独立（embarrassingly parallel） |

结论：**decode 是双流抽象的极端情形**——Q流退化为一条线，整套算力问题塌缩到 KV 流带宽与 Lane 归并延迟上。而 prefill 的新并行度来源（A_Q）就是"多条 Φ_Q 可以完全并行地跑"，不需要新规则。

---

## 11. 代价模型（供 autotuner 用）

基于 §7 的流量推导，first-order cost 有三项：

### 11.1 KV流组分

```text
cost_K(Θ) = T · [ α · Lk · d_k / BW_L3                     (远程读延迟)
                + β · Lk · d_k · (C_S - 1) / BW_onchip      (片上分发)
                + γ_Cell(Lq, Lk) · (C_S · |Q_b|)            (算术成本) ]
```

其中 `γ_Cell(Lq, Lk)` 是单 Cell 的 FLOPs 时间。

### 11.2 Q流组分

```text
cost_Q(Θ) = Nq · [ δ_init · Lq · d_k / BW_L3                (Q 初始读)
                 + ε_reduce · depth(tree(b)) · Lq·(d_v+2) / BW_onchip
                 + ζ_norm · Lq · d_v                        (最终归一化 + 写出) ]
```

### 11.3 总成本

```text
cost(Θ) = parallelise( cost_K, cost_Q  on  N_S · C_S workers )
        + overhead( π_K, π_Q, buf_* )
```

`parallelise(·)` 用硬件 calibration 做具体近似（roofline / pipeline 解析 / 仿真）。

**Autotuner 的目标是在 R1–R6 约束下，旋转 Θ 的"流物理承载 + 流内缓冲/同步"参数，使 `cost(Θ)` 最小。** 由于 R1–R6 是硬约束，搜索空间被显著收紧——大量"看似合理"的 placement 会直接被 R2 或 R5 否决。

---

## 12. 本定义与其他文档的关系

- 与 `flash-mla-dataflow-first-principles.md`：**同一套 ground truth 的两个等价投影**。前者按"沿 K 切 chunk + 三条并行轴"组织，本文按"两条流交汇"组织。两者的 R1–R6 和 Θ 可以 1:1 对应。使用时：
  - 讨论"算的是什么 / 在几条轴上展开" → 引用 `flash-mla-dataflow-first-principles.md`
  - 讨论"数据怎么流 / 哪里分发 / 哪里归并 / placement 怎么选" → 引用本文
- 与 `flash-mla-current-dataflow-analysis.md`、`experimental-flash-mla-dataflow-analysis.md`、`flash-mla-impl-b-dse-formalization.md`：这些是"某个具体实现在本抽象下的投影"。具体 mapping 方法：
  1. 指出它的 `(N_S, C_S, Q_b, K_i, sender, tree)` —— 物理流如何铺。
  2. 指出它的 `(Lq, Lk, Pk, buf_K, buf_page, buf_stats, π_K, π_Q)` —— 流内深度与同步。
  3. 指出它如何具现化 R1（V 视图）、R2（唯一远程读）、R5（Q 不回 L3）—— 这三条决定了"这是 Flash MLA 还是退化 SDPA"。
- 与 autotuner / paper formalism：搜索空间与成本函数的基底定义直接取自本文 §8 与 §11。

---

## 13. 一页速查

```text
对象：     Cell(q, k)  ← 两条流在此交汇
Q流 Φ_Q(q)： 固定 q, 沿 k 前进；载运 (Qtile, (m,l,o))；流内算子 = 归并
KV流 Φ_K(k)：固定 k, 沿 q 扩散；载运 (Ktile, V-view)；流内算子 = 分发
映射：     Φ_Q ↔ Lane(b) = {G[i, b] : i}
          Φ_K ↔ S-block(i) = {G[i, b] : b}
通信：     KV流产出"广播"；Q流产出"归并"
V 对策：   永远是 KV流里的同缓冲偏移视图，不是独立流
六条铁律： R1 V-免流 · R2 KV流唯一远程读 · R3 Q流 online-softmax 单调
          R4 KV流封闭 · R5 Q流局部性 · R6 Q流归并可结合
旋钮分层： (Lq, Lk, Pk)          ← tile 颗粒
          (N_S, C_S, sender, tree) ← 流物理承载
          (buf_*, π_*)          ← 流内深度/同步
```

这十三行就是整套 Flash MLA 数据流的压缩表达；其余章节都是它的展开。

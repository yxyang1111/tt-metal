# Flash MLA Dataflow on Tenstorrent — Paper Section (Bilingual Draft)

> 本文是基于 `flash-mla-dataflow-first-principles.md` Part A + Part B 改写的论文级 section 草稿。每小节中文在前、英文在后；公式用 LaTeX 块排版；图位留为 `Figure N` 占位，可后续替换为真实插图。适合直接嵌入论文的 "System Design" 或 "Method" 一节。
>
> This file is a paper-section draft derived from `flash-mla-dataflow-first-principles.md` (Parts A + B). Each subsection is bilingual — Chinese first, English second. Equations use displayed LaTeX. Figure slots are placeholders. Intended as drop-in text for the "System Design" or "Method" section of a conference submission.

---

## 3.1 Overview / 概览

### 中文

现代大语言模型在 decode 阶段对注意力算子的约束主要来自两方面：其一，`Sq = 1` 时 query 张量极小而 key/value 缓存随上下文线性增长，系统长期处于带宽受限状态；其二，传统实现会为每个 token 具象化完整的 `Sq × S` logits 矩阵，在长序列下 L1 / register 压力成为二次瓶颈。Multi-head Latent Attention (MLA) 通过把 V 表达为 K 的前 `d_v` 列视图，消去了一条独立的片外数据流；FlashAttention 通过沿 K 序列轴分块并维护 online softmax 三元组 `(m, l, o)`，消去了对完整 logits 的具象化。我们在 Tenstorrent 多核芯片上给出 Flash MLA 的第一性数据流定义，并配合一套硬件无关的形式化，使得同一算子的不同实现可以在统一参数空间内严格比较。本节的核心观察可概括为：**算子期间只有 Q 与 K 在 DRAM ↔ 片上 SRAM 之间流动，V 永不独立存在，L1 上唯一的具象中间态是 `(m, l, o)`**。

### English

The attention operator is a persistent bottleneck during LLM decode, for two reasons. First, when `Sq = 1` the query tensor is tiny while the key/value cache grows linearly with context length, leaving the system bandwidth-bound. Second, naive implementations materialise the full `Sq × S` logits matrix, which makes L1/register pressure a secondary quadratic bottleneck at long sequence lengths. Multi-head Latent Attention (MLA) eliminates one of the two off-chip data streams by expressing V as the leading `d_v` columns of a single latent key tensor K. FlashAttention eliminates the materialisation of logits by splitting K along the sequence axis and maintaining running softmax statistics `(m, l, o)`. In this section we give a first-principles description of Flash MLA on the Tenstorrent many-core architecture and a hardware-independent formalisation of its parameter space, which enables different implementations of the same operator to be compared on a common ground. The central observation of this section is that **only Q and K are ever transferred between DRAM and on-chip SRAM; V is accessed exclusively as a zero-copy column view of K, and the running triple `(m, l, o)` is the only intermediate state materialised in L1**.

---

## 3.2 Preliminaries / 预备知识

### 中文

给定 query 张量 $Q \in \mathbb{R}^{B \times S_q \times H_q \times d_k}$ 和潜在 key/value 张量 $K_\text{latent} \in \mathbb{R}^{B \times S \times H_{kv} \times d_k}$，Flash MLA 算子输出 $\text{Output} \in \mathbb{R}^{B \times S_q \times H_q \times d_v}$，满足对每个 $(b, h)$ 对，令 $g = H_q / H_{kv}$、$V_\text{view} = K_\text{latent}[b, :, h/g, :d_v]$、$\text{scale} = 1/\sqrt{d_k}$，有

$$
\text{Output}[b, :, h, :] = \text{softmax}\!\Big( \text{scale}\cdot Q[b, :, h, :]\,K_\text{latent}[b, :, h/g, :]^\top + M \Big)\,V_\text{view}.
$$

mask $M$ 由枚举 `mask_type`（`none`, `causal`, `sliding_window(w)`, `padding`, `sink`）配合位置信息在片上就地生成，不作为张量输入。**这是算子的语义定义**，下文所有实现都须在"不改变该输出"的前提下展开。

Tenstorrent device 的相关结构可以用五个量概括：一张由若干 Tensix cores 构成的二维网格，每核含约 $1.5\,$MB 私有 L1 SRAM 和五颗可编程 RISC（NCRISC 负责 DRAM 访存、BRISC 负责 NoC 与 mask 生成、三颗 TRISC 负责矩阵乘与 softmax）；两张互为镜像的 torus 网上片（NOC0 与 NOC1）；片外 GDDR 被划分为若干独立寻址的 DRAM bank（Wormhole 12 bank、Blackhole 8 bank）；所有 matmul 与片上搬运以 $32\times 32$ 的 tile 为粒度。

### English

Given a query tensor $Q \in \mathbb{R}^{B \times S_q \times H_q \times d_k}$ and a latent key/value tensor $K_\text{latent} \in \mathbb{R}^{B \times S \times H_{kv} \times d_k}$, Flash MLA produces $\text{Output} \in \mathbb{R}^{B \times S_q \times H_q \times d_v}$. Writing $g = H_q / H_{kv}$, $V_\text{view} = K_\text{latent}[b, :, h/g, :d_v]$ and $\text{scale} = 1/\sqrt{d_k}$, the operator is defined by

$$
\text{Output}[b, :, h, :] = \text{softmax}\!\Big( \text{scale}\cdot Q[b, :, h, :]\,K_\text{latent}[b, :, h/g, :]^\top + M \Big)\,V_\text{view}
$$

for every $(b, h)$. The mask $M$ is generated on-chip from the categorical `mask_type` (`none`, `causal`, `sliding_window(w)`, `padding`, `sink`) combined with position metadata; it is never a tensor argument. This is the operator's semantic definition, and every implementation we describe below must reproduce this output exactly.

The Tenstorrent device is summarised by five features: a two-dimensional grid of Tensix cores, each holding roughly $1.5\,$MB of private L1 SRAM and five programmable RISCs (NCRISC for DRAM, BRISC for NoC and on-the-fly mask generation, three TRISCs for matmul and softmax); two mirrored torus Networks-on-Chip (NOC0 and NOC1); a set of independently addressable off-chip DRAM banks (twelve on Wormhole, eight on Blackhole); and a hardware-enforced $32\times 32$ tile as the quantum of matmul and on-chip transfer.

---

## 3.3 Virtual Grid Abstraction / 虚拟网格抽象

### 中文

Flash MLA 的并行结构首先定义在一张与物理核坐标解耦的**虚拟网格**上。令 $N_S$ 为沿 K 序列的切分数、$C_S$ 为沿 Q 头的切分数，虚拟网格即

$$
\mathcal{W} = \{\,w(i, b)\;:\;i \in [0, N_S),\ b \in [0, C_S)\,\}.
$$

第 $i$ 列称为 S-block $i$，负责 K 序列的一段；第 $b$ 行称为 Lane $b$，持有 Q 头的一段 shard。每列内由一个逻辑位置 $\text{sender}(i)$ 唯一承担该列的 DRAM K 读取；每行内由 $\text{qreader}(b)$ 唯一承担 Q 读取，由 $\text{root}(b)$ 作为 Lane 归并的终点并把输出写回 DRAM。三个"角色变量"彼此独立：它们分别对齐 K、Q、Output 在 DRAM 中的 bank 位置。图 1 给出该网格和三类角色的示意。

虚拟网格随后通过一张常量映射 $\varphi: (i, b) \mapsto \text{core}$ 投影到 Tensix 物理坐标。$\varphi$ 的选择受三条约束：(i) $\text{sender}(i)$ 所映射的物理核须紧邻它读的 DRAM bank；(ii) 同列（同行）的核在物理上构成规则矩形，使 `noc_async_write_multicast` 的跳数最小；(iii) 当切换 NOC0 / NOC1 时镜像 $\varphi$，以保持与 NoC 地址空间方向一致。这一层分离使得算法层的并行度（$N_S, C_S$）可以由设计空间搜索决定，物理映射则由硬件拓扑独立约束。

### English

The parallel structure of Flash MLA is defined on a **virtual grid** decoupled from physical core coordinates. With $N_S$ the K-sequence partitioning and $C_S$ the Q-head partitioning, the virtual grid is

$$
\mathcal{W} = \{\,w(i, b)\;:\;i \in [0, N_S),\ b \in [0, C_S)\,\}.
$$

Column $i$, called S-block $i$, handles a contiguous K-sequence range; row $b$, called Lane $b$, owns a shard of the query heads. Within each column a single logical position $\text{sender}(i)$ is the sole DRAM reader for K; within each row, $\text{qreader}(b)$ is the sole DRAM reader for Q and $\text{root}(b)$ serves both as the terminus of the Lane reduction and as the DRAM writer for the output. The three role variables are mutually independent; each is chosen to align with the DRAM bank hosting K, Q, or Output, respectively. Figure 1 depicts the grid and the three roles.

The virtual grid is then projected onto physical Tensix coordinates by a constant mapping $\varphi: (i, b) \mapsto \text{core}$. Three constraints shape $\varphi$: (i) the physical core carrying $\text{sender}(i)$ must be adjacent to the DRAM bank it reads; (ii) physical cores within a column (respectively a row) must form a rectangular region so that `noc_async_write_multicast` incurs minimum hop count; (iii) when switching between NOC0 and NOC1, $\varphi$ is mirrored to stay consistent with the NoC address orientation. This layering lets the algorithmic parallelism $(N_S, C_S)$ be chosen by design-space search while the physical mapping is constrained independently by hardware topology.

> **Figure 1.** The virtual grid $\mathcal{W}$. Rows are Lanes (one Q shard each), columns are S-blocks (one K-chunk stream each). Highlighted positions mark the three roles $\text{sender}(i)$, $\text{qreader}(b)$, $\text{root}(b)$, which may be chosen independently.

---

## 3.4 Dataflow / 数据流

### 中文

整条数据流按五个语义阶段展开，逻辑上严格有序；§3.6 再说明如何在 page 粒度上沿时间轴流水重叠。记 K chunk 总数 $T = \lceil S / L_k \rceil$，分派函数 $\mathcal{C}(i) = \{\,t : \text{assign}(t) = i\,\}$（默认 round-robin $\text{assign}(t) = t \bmod N_S$），每 Lane 的 head 数 $\tau = H_q / C_S$。

**(i) Read — 从 DRAM 读。** 对每个 $t \in \mathcal{C}(i)$，$\text{sender}(i)$ 所映射的物理核发起一次 DRAM 读，把

$$
K_t = K_\text{latent}[\cdot,\ t L_k\!:\!(t{+}1)L_k,\ \cdot,\ :] \in \mathbb{R}^{L_k \times d_k}
$$

写进本地环形缓冲 `cb_k_in` 的一个 slot。由于 DRAM bank 数 $N_\text{bank}$ 与 $N_S$ 对齐，每个 chunk 从片外到片上**恰好触发一次**传输。同时，每个 $\text{qreader}(b)$ 在算子入口处一次性读入 Q shard $Q_\text{shard}(b) = Q[\cdot, \cdot, b\tau\!:\!(b{+}1)\tau, :]$，此后该 shard 在本 Lane 所有 worker 的 `cb_q_in` 中常驻整个算子生命周期。

**(ii) Broadcast — 沿 NoC 放大。** $\text{sender}(i)$ 通过 NoC multicast 把 $K_t$ 送到同列的 $C_S{-}1$ 颗接收者；$\text{qreader}(b)$ 沿 Lane 把 $Q_\text{shard}(b)$ 送到同行的 $N_S{-}1$ 颗接收者。Mask $M_t$ 不进入 NoC，由每颗接收者的 BRISC 按 `mask_type` 与 `cur_pos` 就地生成。

**(iii) Recurrence — 本地 online softmax 递推。** 每个 $w(i, b)$ 维护运行态 $(m, l, o)$，初始 $(-\infty, 0, 0)$。对 $t \in \mathcal{C}(i)$ 顺次更新：

$$
\begin{aligned}
S_t &= \text{scale}\cdot Q_\text{shard}(b)\,K_t^\top + M_t, &\\
m' &= \max(m,\ \text{rowmax}(S_t)), &\alpha = e^{m - m'},\\
P_t &= e^{S_t - m'}, &l' = \alpha\,l + \text{rowsum}(P_t),\\
o' &= \alpha\,o + P_t\,V_t, &V_t \equiv K_t[:, :d_v],\\
(m, l, o) &\leftarrow (m', l', o'). &
\end{aligned}
$$

$V$ 的访问在 TRISC 内部沿 $d_k$ 维做零拷贝列视图，$S_t$ 与 $P_t$ 只驻留 TRISC 目的寄存器。跑完 $\mathcal{C}(i)$ 得到本 (Lane, S-block) cell 的局部结果 $\Lambda(i, b) = (m, l, o)_i^b$。

**(iv) Reduce — 沿 Lane 归并。** 两份局部三元组的合并算子 $\oplus$ 与 (iii) 中的 online softmax 同形：令 $m^* = \max(m_a, m_b)$、$\alpha_a = e^{m_a - m^*}$、$\alpha_b = e^{m_b - m^*}$，则 $l^* = \alpha_a l_a + \alpha_b l_b$、$o^* = \alpha_a o_a + \alpha_b o_b$。$\oplus$ 满足结合律与交换律，因此沿 Lane 的二叉树 $\text{tree}(b)$ 做 $\lceil \log_2 N_S \rceil$ 层归并的结果唯一，停留在 $w(\text{root}(b), b)$：

$$
(m, l, o)^*_b = \bigoplus_{i \in [0, N_S)} \Lambda(i, b).
$$

**(v) Write-back — 归一化并写回 DRAM。** 在 $\text{root}(b)$ 处做归一化 $o^*_b / l^*_b$，得到本 Lane 的输出行段 $\text{Output}[\cdot, \cdot, b\tau\!:\!(b{+}1)\tau, :]$，由其 NCRISC 一次 DRAM 写落盘。$C_S$ 个 root 的写区间互不重叠，并集即完整的 $\text{Output}$。

### English

The dataflow unfolds as five semantic stages that are logically ordered; §3.6 explains how they are overlapped along a page-granular time axis. Let $T = \lceil S / L_k \rceil$ be the total number of K chunks, $\mathcal{C}(i) = \{\,t : \text{assign}(t) = i\,\}$ the chunks assigned to S-block $i$ (with default round-robin $\text{assign}(t) = t \bmod N_S$), and $\tau = H_q / C_S$.

**(i) Read.** For each $t \in \mathcal{C}(i)$ the core carrying $\text{sender}(i)$ issues a single DRAM read that places

$$
K_t = K_\text{latent}[\cdot,\ t L_k\!:\!(t{+}1)L_k,\ \cdot,\ :] \in \mathbb{R}^{L_k \times d_k}
$$

into a slot of the local circular buffer `cb_k_in`. Because the DRAM bank count $N_\text{bank}$ is aligned with $N_S$, each chunk triggers exactly one off-chip transfer. In parallel, each $\text{qreader}(b)$ issues a one-shot DRAM read for the query shard $Q_\text{shard}(b) = Q[\cdot, \cdot, b\tau\!:\!(b{+}1)\tau, :]$, which remains resident in the `cb_q_in` of every worker in Lane $b$ for the entire operator.

**(ii) Broadcast.** $\text{sender}(i)$ multicasts $K_t$ over the NoC to the $C_S{-}1$ other cores within its column, and $\text{qreader}(b)$ multicasts $Q_\text{shard}(b)$ to the $N_S{-}1$ other cores within its row. Masks $M_t$ are not transmitted on the NoC; each receiver's BRISC generates them on the fly from `mask_type` and `cur_pos`.

**(iii) Recurrence.** Every $w(i, b)$ maintains running softmax statistics $(m, l, o)$, initialised to $(-\infty, 0, 0)$. Sweeping $t \in \mathcal{C}(i)$ in order, the worker applies

$$
\begin{aligned}
S_t &= \text{scale}\cdot Q_\text{shard}(b)\,K_t^\top + M_t, &\\
m' &= \max(m,\ \text{rowmax}(S_t)), &\alpha = e^{m - m'},\\
P_t &= e^{S_t - m'}, &l' = \alpha\,l + \text{rowsum}(P_t),\\
o' &= \alpha\,o + P_t\,V_t, &V_t \equiv K_t[:, :d_v],\\
(m, l, o) &\leftarrow (m', l', o'). &
\end{aligned}
$$

V is consumed in-place by TRISC as a zero-copy column view of K along the $d_k$ axis, and neither $S_t$ nor $P_t$ leaves the TRISC destination registers — only the running triple lives in L1. Upon finishing $\mathcal{C}(i)$, the worker retains the local triple $\Lambda(i, b) = (m, l, o)_i^b$.

**(iv) Reduce.** Two local triples merge via the operator $\oplus$, which has the same form as stage (iii): with pivot $m^* = \max(m_a, m_b)$, $\alpha_a = e^{m_a - m^*}$ and $\alpha_b = e^{m_b - m^*}$, we have $l^* = \alpha_a l_a + \alpha_b l_b$ and $o^* = \alpha_a o_a + \alpha_b o_b$. Since $\oplus$ is associative and commutative, the $\lceil \log_2 N_S \rceil$-depth binary tree $\text{tree}(b)$ along Lane $b$ produces a unique reduced triple at $w(\text{root}(b), b)$:

$$
(m, l, o)^*_b = \bigoplus_{i \in [0, N_S)} \Lambda(i, b).
$$

**(v) Write-back.** At $\text{root}(b)$ the normalised output $o^*_b / l^*_b$ constitutes Lane $b$'s output slice $\text{Output}[\cdot, \cdot, b\tau\!:\!(b{+}1)\tau, :]$, which is written to DRAM by the same core's NCRISC. The $C_S$ output slices are mutually disjoint and their union is the complete output tensor.

> **Figure 2.** Swimlane schematic of the five stages. Top: DRAM reads by $\text{sender}(i)$ and $\text{qreader}(b)$. Middle: NoC multicast along columns (K) and rows (Q). Bottom: per-worker recurrence, Lane reduction tree, and DRAM write-back at $\text{root}(b)$.

---

## 3.5 Formalisation / 形式化

### 中文

我们将算子的所有参数划分为三组互不相交的集合，以支持跨实现比较与自动调优。**问题参数** $W$ 完全刻画"算什么"：

$$
W = (B,\ H_q,\ H_{kv},\ S_q,\ S,\ d_k,\ d_v,\ \text{causal},\ \text{mask\_type}).
$$

**硬件常量** $H$ 由 Tenstorrent 芯片给出、不可调：

$$
H = (N_\text{core},\ L_1^\text{cap},\ N_\text{NoC},\ \text{BW}_\text{NoC},\ N_\text{bank},\ \text{BW}_\text{bank},\ \text{FLOPS}_\text{peak},\ \text{tile},\ \text{DST}).
$$

**实现旋钮** $\Theta$ 由用户或编译器选：

$$
\Theta = (G,\ T,\ \Pi,\ C,\ X).
$$

分组 $G$ 汇总所有"并行结构"相关量：网格 $(N_S, C_S)$、分派 $\text{assign}$、角色函数 $\text{sender} / \text{qreader} / \text{root}$、每 Lane 的归并拓扑 $\text{tree}$、逻辑-物理映射 $\varphi$、以及 K 的 DRAM 布局 $\text{loc}(K) \in \{\text{interleaved}, \text{ND-sharded}, \text{paged}\}$。$T$ 汇总 tile 粒度的切分：K chunk 长度 $L_k$、chunk 内 page 大小 $P_k$、prefill 场景下的 Q chunk 长度 $L_q$。$\Pi$ 由 `cb_k_in` 的 slot 数刻画，决定读 / 播 / 算三段沿时间轴的叠加深度。$C$ 汇总数值精度与 kernel 变体；$X$ 描述跨设备扩展。

整条流水的输出可写为如下闭式：

$$
\text{Output}[\cdot, \cdot, b\tau\!:\!(b{+}1)\tau, :] = \frac{o^*_b}{l^*_b},\qquad (m, l, o)^*_b = \bigoplus_{i \in [0, N_S)} \Lambda(i, b),
$$

其中 $\Lambda(i, b)$ 是对 $\mathcal{C}(i)$ 内全部 $t$ 的递推 fold。对 $|\mathcal{C}(i)|$ 归纳再结合 $\oplus$ 的结合律与交换律，可证：该表达式与 §3.2 的原始算子定义完全等价，且**对 $\mathcal{C}(i)$ 内部顺序以及 $\text{tree}(b)$ 形状均不敏感**。这一结果是我们可以把 $\text{assign}$ 与 $\text{tree}$ 当作独立旋钮、从而搜索大量实现拓扑的形式化根据。

### English

We partition all parameters of the operator into three disjoint groups, to enable comparison across implementations and to support automatic tuning. **Problem parameters** $W$ fully specify "what is being computed":

$$
W = (B,\ H_q,\ H_{kv},\ S_q,\ S,\ d_k,\ d_v,\ \text{causal},\ \text{mask\_type}).
$$

**Hardware constants** $H$ are fixed by the Tenstorrent device and are not tunable:

$$
H = (N_\text{core},\ L_1^\text{cap},\ N_\text{NoC},\ \text{BW}_\text{NoC},\ N_\text{bank},\ \text{BW}_\text{bank},\ \text{FLOPS}_\text{peak},\ \text{tile},\ \text{DST}).
$$

**Implementation knobs** $\Theta$ are chosen by the user or the compiler:

$$
\Theta = (G,\ T,\ \Pi,\ C,\ X).
$$

Group $G$ collects all parallel-structure knobs: the grid shape $(N_S, C_S)$; the chunk assignment $\text{assign}$; the role functions $\text{sender}, \text{qreader}, \text{root}$; the per-Lane reduction topology $\text{tree}$; the logical-to-physical mapping $\varphi$; and the DRAM layout $\text{loc}(K) \in \{\text{interleaved}, \text{ND-sharded}, \text{paged}\}$. Group $T$ collects tile-granularity knobs: the K chunk length $L_k$, the intra-chunk page size $P_k$, and, for prefill, the Q chunk length $L_q$. Group $\Pi$ is parameterised by the slot count of `cb_k_in`, which determines the temporal overlap among read, multicast, and compute. Group $C$ collects numerical precisions and kernel variants, and group $X$ describes multi-device extension.

The output of the full pipeline admits the closed form

$$
\text{Output}[\cdot, \cdot, b\tau\!:\!(b{+}1)\tau, :] = \frac{o^*_b}{l^*_b},\qquad (m, l, o)^*_b = \bigoplus_{i \in [0, N_S)} \Lambda(i, b),
$$

where $\Lambda(i, b)$ is the fold of the per-chunk recurrence over $\mathcal{C}(i)$. Induction on $|\mathcal{C}(i)|$ together with the associativity and commutativity of $\oplus$ shows that this expression coincides with the definition in §3.2, and is **invariant under any permutation of $\mathcal{C}(i)$ or any shape of $\text{tree}(b)$**. This is the formal justification for treating $\text{assign}$ and $\text{tree}$ as independent knobs when exploring the implementation space.

---

## 3.6 Invariants, Feasibility, and Pipelining / 不变量、可行性与流水

### 中文

$\Theta$ 的合法域由两层约束共同刻画。第一层是**算法层不变量**——任何被称作"Flash MLA"的实现必须同时满足：

1. **(R1, V 视图性)** $V$ 永不以独立张量形式出现在任何存储层，仅作 K 的零拷贝列视图访问。
2. **(R2, K 单次远程读)** 对每个 $(i, t)$，$K_t$ 从 DRAM 到片上的传输恰好发生一次。
3. **(R3, Q 单次远程读且常驻)** 对每个 Lane $b$，$Q_\text{shard}(b)$ 的 DRAM 传输恰好发生一次，内容在算子结束前一直驻留 `cb_q_in`。
4. **(R4, Logits 非具象)** 任何时刻都不得在 L1 构造完整的 logits 或概率矩阵；L1 上的具象中间态只有 $(m, l, o)$。
5. **(R5, 合法回收)** `cb_k_in` 中某 slot 被回收，当且仅当对应 chunk 的 (F1)–(F7) 已完成。
6. **(R6, 合并律)** $\oplus$ 满足结合律与交换律。

第二层是**硬件可行性约束**，源自 $H$：$N_S \cdot C_S \le N_\text{core}$；每核 L1 占用之和须小于 $L_1^\text{cap}$；$\text{tile}$ 整除 $L_k, P_k, L_q$。在 (R1)–(R6) 与可行性约束共同刻画的可行域中，剩余的 $\Theta$ 自由度即为 autotuner 的搜索空间。

阶段 (i)–(iii) 在纯语义上是串行的，但 `cb_k_in` 的多 slot 设计允许沿 page 粒度把它们沿时间轴错开。当 $\text{slots}(\text{cb\_k\_in}) \ge 3$ 且每 page 的 read、multicast、compute 三段时长相近时，稳态吞吐由最慢一段决定。这一流水重排不改变 (R1)–(R6)，因此 §3.5 的等价性定理保证输出严格不变——这是在硬件实际延迟下挖掘设备利用率的根本方法。

### English

The admissible region of $\Theta$ is characterised by two layers of constraints. The first layer consists of **algorithmic invariants** that every implementation claiming to be Flash MLA must satisfy:

1. **(R1, V as view)** $V$ never appears as an independent tensor at any level of the memory hierarchy; it is always accessed as a zero-copy column view of K.
2. **(R2, K single remote read)** For every $(i, t)$, the transfer of $K_t$ from DRAM to on-chip SRAM occurs exactly once.
3. **(R3, Q single remote read and residency)** For every Lane $b$, $Q_\text{shard}(b)$ is transferred from DRAM exactly once, and remains resident in `cb_q_in` for the entire operator.
4. **(R4, Logits non-materialisation)** No full logits or probability matrix is ever present in L1; the only materialised intermediate state is the running triple $(m, l, o)$.
5. **(R5, Legal buffer release)** A slot of `cb_k_in` may be released if and only if the recurrence (F1)–(F7) for the corresponding chunk has completed.
6. **(R6, Merge associativity)** $\oplus$ is associative and commutative.

The second layer consists of **hardware feasibility constraints** drawn from $H$: $N_S \cdot C_S \le N_\text{core}$; the sum of all per-core L1 residents must be smaller than $L_1^\text{cap}$; and $\text{tile}$ must divide $L_k, P_k$, and $L_q$. Inside the feasible region jointly defined by (R1)–(R6) and these constraints, the remaining degrees of freedom in $\Theta$ form the search space of an autotuner.

Stages (i)–(iii) are semantically sequential, but the multi-slot design of `cb_k_in` permits them to be overlapped along a page-granular time axis. When $\text{slots}(\text{cb\_k\_in}) \ge 3$ and the per-page read, multicast, and compute durations are comparable, steady-state throughput is bounded by the slowest of the three. This pipelining preserves invariants (R1)–(R6), so the equivalence theorem of §3.5 guarantees that the output is unchanged — this is the principal lever by which Flash MLA reclaims utilisation in the face of the real latencies of a multi-core device.

> **Figure 3.** Time-axis schematic of the page-level pipeline. Three swim lanes — NCRISC (read), BRISC (multicast), and TRISC (compute) — process successive slots of `cb_k_in`. Steady-state is reached once three pages are in flight simultaneously.

---

## 3.7 Summary / 小结

### 中文

本节将 Flash MLA 从算法层到硬件层的数据流分为三层抽象：**算子语义**（§3.2）、**虚拟网格**（§3.3）与**五段数据流**（§3.4）；再用参数三元组 $(W, H, \Theta)$ 把"算什么 / 在什么硬件上 / 怎么算"分开形式化（§3.5），并给出六条算法不变量与一组硬件可行性约束共同刻画的合法域（§3.6）。这套框架一方面为我们随后给出的具体实现提供可验证的正确性基线，另一方面把"同一算子的不同实现"放进同一坐标系，使跨实现比较与设计空间搜索成为可能。

### English

This section has presented the Flash MLA dataflow at three layers of abstraction: the operator semantics (§3.2), the virtual grid that carries the algorithmic parallelism (§3.3), and the five-stage dataflow that maps the grid onto the Tenstorrent hardware (§3.4). We then formalised the parameters by the triple $(W, H, \Theta)$ — "what is computed / on what hardware / in what way" (§3.5) — and characterised the admissible region of $\Theta$ via six algorithmic invariants together with a small set of hardware feasibility constraints (§3.6). The resulting framework provides a verifiable correctness baseline for the concrete implementation we describe next, and, by placing different implementations of the same operator in a shared coordinate system, enables cross-implementation comparison and systematic design-space search.

# Decode Lane/Column Roles SVG 脚本规格

## 目标产物
- `decode_lane_column_roles_theoretical.svg`
- `decode_lane_column_roles_actual_decode32k.svg`

这两张图必须是**同构**的：同一套角色顺序、同一套处理器轨顺序、同一套颜色语义、同一套三段时间分区。读者应该能直接把“理论稳态”与“实际运行”逐行对照。

## 1. 通用脚本约定

### 1.1 画布与布局

| 项目 | 建议值 |
| --- | --- |
| `width` | `1600` |
| `height` | `1180` |
| `left_margin` | `250` |
| `right_margin` | `40` |
| `top_margin` | `92` |
| `footer_height` | `44` |
| `role_row_height` | `116` |
| `role_gap` | `18` |
| `track_height` | `24` |
| `track_gap` | `6` |
| `role_label_width` | 包含在 `left_margin` 内，左侧固定文本区 |

### 1.2 时间轴

这两张图都使用**归一化横轴** `x in [0, 1]`，不直接画真实 `us` 长度。  
实际图在脚注里补充 `decode_32k` 的统计锚点。

共享三段时间分区：

| 分区 | 区间 | 含义 |
| --- | --- | --- |
| `q_prep` | `[0.00, 0.12]` | `Q` 前置准备，只发生一次 |
| `k_loop` | `[0.12, 0.74]` | `K chunk` 稳态主循环 |
| `tail` | `[0.74, 1.00]` | `tree reduction + final output` |

### 1.3 角色与轨道顺序

角色从上到下固定：

1. `Q source（兼 worker）`
2. `Column sender（兼 worker）`
3. `Ordinary worker`
4. `Lane root（兼 worker）`

每个角色内部轨道顺序固定：

1. `NCRISC`
2. `TRISC`
3. `BRISC`

### 1.4 颜色语义

| 语义 | 颜色 |
| --- | --- |
| `Q local / pull` | `#2ec4b6` |
| `K read / NCRISC issue` | `#1982c4` |
| `K wait / DRAM wait` | `#ffca3a` |
| `reserve / backpressure` | `#ff595e` |
| `K multicast` | `#3d5a80` |
| `tree / sender wait` | `#9b5de5` |
| `QK^T` | `#1b998b` |
| `softmax` | `#ff9f1c` |
| `P@V` | `#5c7cfa` |
| `output write / gather` | `#00bbf9` |
| `idle / ghost / inactive` | `#2a3340` |

### 1.5 共享绘制规则
- 所有 box 使用圆角矩形；工作 box 不透明，`idle / ghost` 使用半透明。
- 两张图都要在顶部明确写：`Q source / Column sender / Lane root` 是逻辑角色，不一定总是三个不同的物理 core。
- `Q` 统一画在前置阶段，语义固定为 `source L1 + per-worker pull`。
- `K` 统一画在中段主循环，语义固定为 `sender NCRISC read -> sender BRISC multicast -> all-worker TRISC compute`。
- `tail` 统一画成 `worker BRISC send -> root BRISC wait/push -> root TRISC merge -> root BRISC output`。
- receiver 侧不要画成“自己主动接收 `K` payload”。更准确地说，payload 由 sender `BRISC` 直接组播写入对端预留的本地目标区；对端本地 reader/NCRISC 负责做 `CB_K` 的 reserve/push bookkeeping 并等待 `mcast-ready semaphore`，再把 `CB_K` 暴露给 `TRISC`。理论图里这一步可以不单独画出来。
- `TRISC` 的可视化 chunk 边界用细竖线；不要在 `NCRISC` / `BRISC` 上重复画满屏竖线。
- 这是纯泳道图，不画跨 lane 的 `Q pull` / `child send` 箭头；若需要解释同步，只用短注释或脚注说明。

## 2. 图 A：理论版泳道图

### 2.1 标题与副标题
- 标题：`理想化 Decode 稳态流水（最小空泡 / 最小同步）`
- 副标题：`Q preparation once; read K_(t+1) || multicast K_t || compute K_t; tail reduction kept compact.`

### 2.2 中段可视化 chunk 槽位

理论图不需要画 256 个 `K chunk`，只画 3 个可见槽位，末尾加省略提示。

| 槽位 | 在 `k_loop` 内的相对区间 |
| --- | --- |
| `K0` | `[0.10, 0.28]` |
| `K1` | `[0.38, 0.56]` |
| `K2` | `[0.66, 0.84]` |

对 `Column sender` 需要体现三处理器错位重叠：
- `NCRISC read` 比对应 `TRISC compute` 提前一个小偏移
- `BRISC multicast` 位于二者之间
- 整体读感是 `read K_(t+1) || multicast K_t || compute K_t`

### 2.3 逐行脚本规格

#### `Q source（兼 worker）`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `Q local / ready` -> `mcast-ready K0/K1/K2` | `q_prep [0.20, 0.85]`; `mcast-ready K0 [0.14,0.18]`; `K1 [0.28,0.32]`; `K2 [0.50,0.54]` | `Q source` 在完成 `Q` 分发后应与普通 worker 一致；理论图省略提前做好的 `CB_K reserve`，只保留 ready 信号 |
| `TRISC` | `compute K0` -> `compute K1` -> `compute K2` | 直接对齐三段 `K0/K1/K2` | 不画明显空泡，只保留很窄的 chunk 间缝隙 |
| `BRISC` | `serve Q pulls` -> `send local (m,l,O)` | `q_prep [0.25, 0.95]`; `tail [0.08, 0.28]` | 前段服务 lane 内 pull，尾段作为普通 worker 发送局部结果 |

#### `Column sender（兼 worker）`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `read K0` -> `read K1` -> `read K2` | `read K0 [0.00,0.14]`; `read K1 [0.14,0.28]` 连续贴合；`read K2 [0.36,0.50]` 需要在 `K0` 计算释放 buffer slot 之后再开始 | `CB_K` 是双缓冲，理论图最多只表达“当前块计算 + 下一块读取”重叠 |
| `TRISC` | `compute K0` -> `compute K1` -> `compute K2` | sender 的 `compute K0` 应在 `read full K0` 后立刻开始，并早于其它 worker；后续 `K1/K2` 同理 | 用来体现 sender 本地先拿到 `K_t` |
| `BRISC` | `pull Q once` -> `mcast K0` -> `mcast K1` -> `mcast K2` -> `send local (m,l,O)` | `q_prep [0.30, 0.80]`; `mcast K0 [0.14,0.18]`; `mcast K1 [0.28,0.32]`; `mcast K2 [0.50,0.54]`; `tail [0.08, 0.28]` | `mcast K1` 应在 `read K1` 完成后立刻开始；理论图里 `mcast-ready` 与 `mcast` 对齐，作为同一个逻辑 handoff 窗口来画 |

#### `Ordinary worker`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `mcast-ready K0/K1/K2` | `mcast-ready K0 [0.14,0.18]`; `K1 [0.28,0.32]`; `K2 [0.50,0.54]` | 不把 receiver 画成主动收 payload 的对称 reader；理论图省略可提前完成的本地缓冲准备，只强调 semaphore ready |
| `TRISC` | `compute K0` -> `compute K1` -> `compute K2` | `compute K0 [0.18,0.36]`; `K1 [0.36,0.54]`; `K2 [0.54,0.72]` | 每段都应在对应 `mcast-ready` 之后立刻起算，主要视觉是连续计算 |
| `BRISC` | `pull Q once` -> `send local (m,l,O)` | `q_prep [0.25, 0.80]`; `tail [0.10, 0.36]` | `Q pull` 放在 `BRISC`，尾段统一写成发送局部部分状态 |

#### `Lane root（兼 worker）`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `mcast-ready K0/K1/K2` -> `ghost` | `mcast-ready K0 [0.14,0.18]`; `K1 [0.28,0.32]`; `K2 [0.50,0.54]`; 尾段 ghost | root 作为 worker 也要参与 `K` ready，但不承担 reduction 接收主链；理论图省略可提前完成的本地缓冲准备 |
| `TRISC` | `compute K0` -> `compute K1` -> `compute K2` -> `merge r0` -> `merge r1` -> `tail merge` | 中段对齐 `K0/K1/K2`，即 `[0.18,0.36]`, `[0.36,0.54]`, `[0.54,0.72]`；尾段 `[0.30,0.46]`, `[0.52,0.68]`, `[0.70,0.82]` | 尾段体现 merge 是 `TRISC` 算术 |
| `BRISC` | `pull Q once` -> `wait child-ready r0` -> `wait child-ready r1` -> `final output write / gather` | `q_prep [0.25, 0.75]`; `tail [0.10, 0.28]`, `[0.34, 0.52]`, `[0.82, 0.95]` | 这里的 `wait child-ready` 指等待 reducer semaphore，表示 child 的局部 `(m,l,O)` 已写到 parent/root 可读位置 |

### 2.4 理论图的附加标记
- 在 `Column sender` 中段上方加一条注释：`ping-pong CB_K: compute K0 overlaps read K1; read K2 waits until K0 slot is freed`
- 在 `Ordinary worker / NCRISC` 或脚注中注明：`after Q distribution, Q source / workers / lane root all wait the same mcast-ready semaphore`
- 在脚注中注明：theoretical 图不单画 receiver 侧 `CB_K reserve/push`，`mcast-ready` 与 `mcast` 对齐，表示同一个抽象 handoff 窗口
- `tail` 区只保留必要同步，不画长等待气泡

## 3. 图 B：实际版泳道图（`decode_32k`）

### 3.1 标题与副标题
- 标题：`实际 Decode_32k 运行图（bubble / reserve / sender-tree wait）`
- 副标题：`Anchored by decode_32k: reader reserve 57.6%, K reserve 54.8%, writer cb_wait 99.8%, sender/tree 49.95/50.05.`

### 3.2 实际图必须显式写出的锚点
- `reader reserve ≈ 57.58%`
- `K reserve ≈ 54.82%`
- `writer cb_wait ≈ 99.78%`
- `wait-front ≈ 70.91%`
- `reserve-back ≈ 29.09%`
- `sender_cb_wait / tree_child_wait ≈ 49.95 / 50.05`
- `root_cb_wait / output_gather_wait ≈ 0 / 0`

### 3.3 实际图的骨架约束

实际图应与理论图保持**同一套阶段语义**，即仍然是：

- `read K_t -> mcast K_t -> mcast-ready -> compute K_t`
- `send local (m,l,O) -> wait child-ready -> merge -> final output`

区别只在于：`decode_32k` 会把这些阶段之间的等待和背压明显拉长，所以实际图要表现为“**同骨架，但气泡更多、可见工作更晚、更短**”。

建议使用下面这组可见窗口来画 3 个代表性 chunk：

| 语义窗口 | 在 `k_loop` 内的相对区间 |
| --- | --- |
| sender `mcast K0` / receiver `mcast-ready K0` | `[0.48, 0.52]` |
| sender `mcast K1` / receiver `mcast-ready K1` | `[0.62, 0.66]` |
| sender `mcast K2` / receiver `mcast-ready K2` | `[0.76, 0.80]` |
| worker `compute K0` | `[0.52, 0.62]` |
| worker `compute K1` | `[0.66, 0.76]` |
| worker `compute K2` | `[0.80, 0.90]` |
| sender `compute K0/K1/K2` | 分别比 worker 早一小格：`[0.48,0.58]`, `[0.62,0.72]`, `[0.76,0.86]` |

这些窗口不是逐核 timestamp trace，而是为了把 aggregate counter 造成的“前重后挤”效果稳定地画出来。

### 3.4 逐行脚本规格

#### `Q source（兼 worker）`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `Q local / light` -> `wait K0 ready` -> `mcast-ready K0` -> `wait K1 ready` -> `mcast-ready K1` -> `wait K2 ready` -> `mcast-ready K2` | `q_prep [0.20, 0.85]`; `k_loop` 用与普通 worker 相同的 ready/wait 窗口 | `Q source` 作为 worker 也会承受接收侧 ready 压力；这里只是示意性复用 shared receive pressure |
| `TRISC` | `wait_front` -> `compute K0` -> `compute K1` -> `compute K2` -> `reserve_back` | `wait_front [0.00,0.52]`; 三段 compute 对齐 worker | 与普通 worker 同步受压，不画成特殊“先行核” |
| `BRISC` | `serve Q pulls` -> `wait local-ready` -> `send local (m,l,O)` | `q_prep [0.25, 0.95]`; `tail [0.06,0.74]`, `[0.74,0.86]` | 尾段明显拉长，体现 writer backpressure |

#### `Column sender（兼 worker）`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `K reserve` -> `read K0` -> `reserve gap` -> `read K1` -> `reserve gap` -> `read K2` -> `K wait` | `K reserve [0.00,0.18]`; `read K0 [0.18,0.48]`; `read K1 [0.52,0.62]`; `read K2 [0.66,0.76]`; `K wait [0.76,1.00]` | 这是实际图里最重要的 reader 侧压力展示：可见读阶段被 reserve/backpressure 拉得很散 |
| `TRISC` | `wait_front` -> `compute K0` -> `compute K1` -> `compute K2` -> `reserve_back` | `wait_front [0.00,0.48]`; `compute K0 [0.48,0.58]`; `K1 [0.62,0.72]`; `K2 [0.76,0.86]`; `reserve_back [0.86,1.00]` | sender 的 `TRISC` 仍略早于其它 worker，但被前后两端严重挤压 |
| `BRISC` | `pull Q once` -> `mcast K0` -> `mcast K1` -> `mcast K2` -> `wait local-ready` -> `send local (m,l,O)` | `q_prep [0.30,0.80]`; `mcast [0.48,0.52]`, `[0.62,0.66]`, `[0.76,0.80]`; `tail [0.00,0.70]`, `[0.70,0.88]` | `mcast` 仍然是短脉冲，尾段主瓶颈是 sender 侧 wait |

#### `Ordinary worker`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `wait K0 ready` -> `mcast-ready K0` -> `wait K1 ready` -> `mcast-ready K1` -> `wait K2 ready` -> `mcast-ready K2` | `wait [0.00,0.48]`, `[0.52,0.62]`, `[0.66,0.76]`; `ready [0.48,0.52]`, `[0.62,0.66]`, `[0.76,0.80]` | 保持与理论图一致：不把 receiver 画成主动读取 payload，而是把等待 ready 的时间显式拉长 |
| `TRISC` | `wait_front` -> `compute K0` -> `compute K1` -> `compute K2` -> `reserve_back` | `wait_front [0.00,0.52]`; `compute [0.52,0.62]`, `[0.66,0.76]`, `[0.80,0.90]`; `reserve_back [0.90,1.00]` | 这是实际图最核心的 `TRISC` 气泡形状 |
| `BRISC` | `pull Q once` -> `wait local-ready` -> `send local (m,l,O)` | `q_prep [0.25,0.80]`; `tail [0.10,0.82]`, `[0.82,0.94]` | `wait local-ready` 要明显比 `send local` 长 |

#### `Lane root（兼 worker）`

| Track | Segment order | Placement | 说明 |
| --- | --- | --- | --- |
| `NCRISC` | `light / non-critical` -> `wait K0 ready` -> `mcast-ready K0` -> `wait K1 ready` -> `mcast-ready K1` -> `wait K2 ready` -> `mcast-ready K2` | `q_prep` 仅画短 ghost；`k_loop` 用与普通 worker 相同的 ready/wait 窗口 | root 在 `K` 主循环里仍然是 worker，只是在 `Q` 前置阶段和 reduction 主链里不是 reader 关键路径 |
| `TRISC` | `wait_front` -> `compute K0` -> `compute K1` -> `compute K2` -> `merge r0` -> `merge r1` -> `tail merge` | `k_loop` 同普通 worker；尾段 `merge r0 [0.38,0.56]`; `merge r1 [0.78,0.90]`; `tail merge [0.90,0.96]` | merge 仍然是 `TRISC` 算术，但现在被 child-ready 明显推迟 |
| `BRISC` | `pull Q once` -> `wait child-ready r0` -> `read child r0` -> `wait child-ready r1` -> `read child r1` -> `final output` | `q_prep [0.30,0.75]`; `tail [0.00,0.30]`, `[0.30,0.38]`, `[0.38,0.72]`, `[0.72,0.78]`, `[0.96,1.00]` | 这里和理论图保持同骨架，只是 `wait child-ready` 被显著拉长，而 `read child` / `final output` 仍然很短 |

### 3.5 实际图的附加标记
- 在 `Column sender / NCRISC` 上方加注：`reader reserve stretches gaps before visible read K_t`
- 在 `TRISC` 轨道上加注：`same K-stage skeleton as theory, but front bubbles dominate`
- 在 `Lane root / BRISC` 尾段上加注：`tree_child_wait dominates; child reads and final output stay short`
- 在脚注里补一句：`Actual view reuses the theoretical-stage skeleton; decode_32k counters appear as stretched wait/reserve gaps and delayed visible work islands`
- 在脚注里补一句：`receiver-side wait K_t / mcast-ready windows are schematic shared receive pressure, not direct per-role traces`
- 在脚注里补一句：`This is a calibrated schematic anchored by aggregate counters, not a per-core timestamp trace.`

## 4. 生成脚本的最小数据模型

后续真正写 Python / SVG 生成脚本时，建议脚本先构造下面这类数据对象：

```text
DiagramSpec
  - title
  - subtitle
  - canvas(width, height, margins)
  - sections[{id, label, start, end}]
  - roles[
      {
        role_label,
        tracks[
          {
            track_label,
            segments[
              {label, section_id, start_frac, end_frac, color, style, note}
            ]
          }
        ]
      }
    ]
  - arrows[{from_role, from_track, to_role, to_track, label, phase}]
  - footnotes[]
```

脚本实现时，理论图与实际图应该共用：
- 同一套 `canvas`
- 同一套 `sections`
- 同一套 `roles / tracks` 顺序
- 同一套 `palette`

只替换每条 `track.segments` 的内容与长度。

# Part II 当前可支持的实验结论（逐条说明）

## 0. 口径与边界

这份文档只整理 **当前 `Part II` 已经被现有实验稳定支撑** 的结论，并把每条结论拆成：

- 结论是什么
- 为什么可以这样解释
- 直接支持它的实验数据是什么
- 现在还不能外推出什么

当前证据分成两类：

- **dense proxy sweep**：`TT` 主线 FlashMLA 的 `decode_256~32k` 与 `prefill_256~4k`，用于回答机制问题，比如 bubble 在哪里、reader/writer 谁更接近 critical path、`K/V` 哪一路更差。
- **direct representative probe**：`1k/4k/8k/16k/32k` representative seq，与固定 `8k` 的 `required_q_cores=24/28/32/40/48` 边界轴上的 corrected-runtime PM/perf-counter、以及 `24/32/48` 点上的 device-only/source-level probe，用于回答 representative point 的直接硬件利用率、`TT vs DeepSeek-4c`、`4c vs 8c` 和容量边界问题。

因此，这里要先明确两条边界：

- **可以直接下结论**：`decode` 阶段迁移、bubble 组成与迁移、`K` 路供给问题、writer sender/tree backpressure、`1k/4k/8k/16k/32k` representative seq 上的算术利用率、`4c/8c` 的容量边界。
- **还不能直接下结论**：`decode` full sweep 上可信的 `NOC UTIL (%)`、`MULTICAST NOC UTIL (%)`、`DRAM BW UTIL (%)` 百分比。

## 0.1 指标速查

先给一个总原则：**文档里的百分比经常不是同一个分母**。只有明确属于同一组拆分的列，才适合横向比较或相加；不同线程、不同 stage、不同 source 的百分比，通常不能直接加到 `100%`。

### 基本字段

- `case`：实验点名字，比如 `decode_4k`、`repr_seq_32k`、`cross48q_b12_qs4_s8k`。
- `seq_len`：序列长度。
- `batch`：batch size。
- `classification`：这一点当前更接近哪种瓶颈阶段；它是一个“阶段标签”，不是单独硬件计数器。
- `kernel us` / `kernel ms` / `kernel wall`：这次 kernel 的总时长。`us` 是微秒，`ms` 是毫秒。

### 线程离 critical path 有多近

- `compute share`：compute 线程窗口相对 kernel window 的占比。越高，表示 compute 越贴近 critical path；**不等于**算术利用率高。
- `BRISC share`：`BRISC` 这条数据搬运线程离 critical path 有多近。越高，表示这条搬运路径越来越影响总时长。
- `NCRISC share`：`NCRISC` 这条数据搬运线程离 critical path 有多近。读法和 `BRISC share` 一样。

这三类指标更适合回答“最后哪条线程在决定总时延”，不适合直接回答“FPU 有没有算满”。

### compute bubble 是怎么组成的

- `bubble / kernel`：compute bubble counter 相对 kernel 时长的密度。它可以大于 `1x`，因为这是累计 stall counter，不是 wall-time 切片。
- `wait-front share`：在 compute bubble 内部，前端供给不足造成的那部分占比。白话就是“前面没喂上来，我在等输入”。
- `reserve-back share`：在 compute bubble 内部，后端回压造成的那部分占比。白话就是“后面没腾出地方，我算完也塞不出去”。
- `wait-front / kernel`：`wait-front` counter 再除以 kernel 时长，所以也可能超过 `100%`。
- `reserve_back / kernel`：`reserve-back` counter 再除以 kernel 时长，所以也可能超过 `100%`。

其中：

- `wait-front share + reserve-back share = 100%`，因为它们在切同一个 compute bubble；
- 但它们**不能**和 `reader reserve`、`writer cb_wait` 横向相加，因为后者不是同一个分母。

### reader 侧指标

- `reader reserve`：reader 这边卡在 reserve/backpressure 等待上的 stage-level share。越高，说明 reader 越像供给瓶颈。
- `K issue`：reader 这边真正把时间花在发 `K` 路读请求上的占比。越高，说明 `K` 路更像在“干活”而不是“等待”。
- `K reserve`：reader 这边卡在 `K` 路 reserve/slot 可用性等待上的占比。越高，越说明 `K` 路供给不顺。
- `V issue`：reader 这边真正把时间花在发 `V` 路读请求上的占比。
- `V reserve`：reader 这边卡在 `V` 路 reserve/slot 等待上的占比。
- `K read GB/s`：`K` 路有效读带宽 proxy。它不是 direct `NoC` 利用率，而是一个“`K` 路到底读得顺不顺”的间接指标。

### writer / reduction 侧指标

- `writer cb_wait`：writer 这边卡在 circular buffer / output path 等待上的 stage-level share。越高，说明后端越排不空。
- `sender wait` / `sender`：writer source breakdown 里，等待停在 sender/local-ready 这一步的占比。
- `tree wait` / `tree`：writer source breakdown 里，等待停在 tree-reduction child arrival 这一步的占比。
- `root wait` / `root`：writer source breakdown 里，等待停在 root reduction 这一步的占比。
- `output wait` / `output`：writer source breakdown 里，等待停在 final gather / output 这一步的占比。

这组指标主要回答：“writer 的等待到底堆在 reduction 主链路，还是最后 gather 收尾。”

### PM / perf-counter / 更接近硬件利用率的指标

- `PM IDEAL`：performance model 给出的 ideal baseline。若它在一整列里都是 `1.0`，通常说明这一轮还是 placeholder，不能信。
- `PM COMPUTE`：performance model 里 compute-limited 的时间分量。
- `PM BANDWIDTH`：performance model 里 bandwidth-limited 的时间分量。
- `PM FPU util` / `PM FPU UTIL`：更接近真实算术利用率的 direct 指标。它比 `compute share` 更适合回答“FPU 有没有算满”。
- `Avg FPU util`：全 grid 平均 FPU 利用率，读法上和 `PM FPU util` 接近，但更偏 hardware-average 口径。
- `Avg MATH util`：全 grid 平均 math pipe 利用率，和 `Avg FPU util` 类似，也是硬件平均口径。

这些指标适合回答“算术利用率到底高不高”；如果它们缺失或 placeholder，就不要拿 `compute share` 去替代。

### 跨实现/跨配置对比指标

- `TT kernel us`、`DeepSeek-4c kernel us`、`4c kernel us`、`8c kernel us`：本质上都是同一个 kernel 时长指标，只是实现或配置不同。
- `TT PM FPU util`、`DeepSeek-4c PM FPU util`：同一个 `PM FPU util` 指标在两条实现上的直接对照。
- `TT Avg FPU util`、`DeepSeek-4c Avg FPU util`：同一个 `Avg FPU util` 指标在两条实现上的直接对照。
- `TT K reserve`、`DeepSeek-4c K reserve`、`8c K reserve`：同一个 `K reserve` 指标在不同实现/配置上的直接对照。
- `TT V reserve`、`DeepSeek-4c V reserve`：同一个 `V reserve` 指标在不同实现上的直接对照。
- `TT wait-front / kernel`、`DeepSeek-4c wait-front / kernel`：同一个 compute front-stall density 在不同实现上的直接对照。
- `DeepSeek / TT`、`8c / 4c`、`8c / TT`：时延比值。`< 1x` 表示前者更快，`> 1x` 表示前者更慢。

### 容量/可运行性指标

- `required q-cores`：这个 workload 需要多少 q-core，当前口径下等于 `batch * q_shards`。
- `4c status`：`4c` 配置能不能跑。`unsupported_precheck` 表示还没真正执行，就在 admission/preflight 阶段因为容量不够被拦下。

### 当前还不能可信使用的 direct 带宽指标

- `NOC UTIL`：direct `NoC` 利用率百分比。
- `MULTICAST NOC UTIL`：direct multicast `NoC` 利用率百分比。
- `DRAM BW UTIL`：direct `DRAM` 带宽利用率百分比。

这些列如果是 `missing` 或明显 placeholder，就只能说明“当前没采到可信值”，不能拿来下结论。

---

## 1. `decode` 的阶段迁移已经固定

**结论**

`decode` 随序列增长会稳定经历四个阶段：

1. `256~1k`：`compute_on_critical_path`
2. `2k~8k`：`writer_close_to_critical_path`
3. `16k`：`reader_close_to_critical_path`
4. `32k`：`reader_writer_saturated`

这里的 `critical path` 可以简单理解成：“这一轮 kernel 总时间最终被哪条最慢链路决定”。

如果用更白话的方式解释这四个阶段，可以这样理解：

- **`compute_on_critical_path`（`256~1k`）**：短序列时，表面上看是 compute 最“挂在最慢路径上”。这不等于算术单元已经打满，而是说这时 kernel 总时间更多投影在 compute 线程上；前后级虽然已经有等待，但 reader 还没有明显堆积，系统更像“compute 一边算、一边偶尔等前面喂数据”。
- **`writer_close_to_critical_path`（`2k~8k`）**：序列变长以后，writer 开始明显变成主压力源之一。更直白地说，就是后端写出和 reduction 这条链越来越难排空，`writer cb_wait` 明显抬高，后端堵塞开始更强地把压力反推回 compute。
- **`reader_close_to_critical_path`（`16k`）**：再往长序列走，reader 侧也开始明显跟不上了，尤其是 `K` 路供给问题会变得非常显性。可以把这个阶段理解成：前端喂数已经不再只是“小拖慢”，而是足以直接决定 kernel 总时间的主问题之一。
- **`reader_writer_saturated`（`32k`）**：到最长序列点时，前端 reader 和后端 writer 都已经进入高压区，形成前后同时卡住的强耦合流水线。此时不能再把问题看成单点瓶颈，而要理解成“前端供给不足 + 后端回压过强”一起决定了总时延。

**说明**

这说明当前 `decode` 不是单一瓶颈模型。短序列更像 compute thread 挂在 critical path 上，中段开始 writer 更显性，长序列再继续演化成 reader 与 writer 同时饱和。

**证据数据**

| case | classification | kernel us | reader reserve | writer cb_wait | wait-front share | reserve-back share |
|---|---|---:|---:|---:|---:|---:|
| `decode_256` | `compute_on_critical_path` | 42.198 | 1.1% | 90.1% | 87.4% | 12.6% |
| `decode_1k` | `compute_on_critical_path` | 76.223 | 1.1% | 93.9% | 86.1% | 13.9% |
| `decode_2k` | `writer_close_to_critical_path` | 109.765 | 1.2% | 96.2% | 82.0% | 18.0% |
| `decode_4k` | `writer_close_to_critical_path` | 177.332 | 32.0% | 98.0% | 78.1% | 21.9% |
| `decode_8k` | `writer_close_to_critical_path` | 311.103 | 46.7% | 98.9% | 74.8% | 25.2% |
| `decode_16k` | `reader_close_to_critical_path` | 575.166 | 55.4% | 99.6% | 72.0% | 28.0% |
| `decode_32k` | `reader_writer_saturated` | 1112.087 | 57.6% | 99.8% | 70.9% | 29.1% |

表中指标含义：`case` 是实验点；`classification` 是阶段标签；`kernel us` 是 kernel 总时长；`reader reserve` 是 reader 侧卡在 reserve 的占比；`writer cb_wait` 是 writer 侧卡在 cb wait 的占比；`wait-front share` 和 `reserve-back share` 是 compute bubble 内部前端等待与后端回压的占比。

这些列**不能横向相加到 `100%`**，因为它们不是在切同一个“总蛋糕”，分母并不一样：

- `reader reserve`：是 **reader 这一侧** 的 stage-level stall share，表示 reader 有多少比例卡在 reserve。
- `writer cb_wait`：是 **writer 这一侧** 的 stage-level stall share，表示 writer 有多少比例卡在 cb wait。
- `wait-front share` 和 `reserve-back share`：是 **compute bubble 内部** 的组成拆分，所以这两列彼此是同一个分母，才会两列加起来等于 `100%`。
- `kernel us`：是绝对时长，不是百分比。

所以例如 `decode_4k` 这一行里：

- `reader reserve = 32.0%` 说的是 reader 自己这一侧有约三分之一时间卡在 reserve；
- `writer cb_wait = 98.0%` 说的是 writer 自己这一侧几乎一直在等；
- `wait-front = 78.1%`、`reserve-back = 21.9%` 说的是 compute bubble 里面，前端等待和后端回压各占多少。


**边界**

这里的阶段结论来自 proxy sweep，回答的是“瓶颈结构怎么迁移”，不是 `DeepSeek` full sweep 的 direct 硬件曲线。

---

## 2. 当前最可信的“硬件/线程利用率”结论：compute 始终贴着关键路径，但真实算术利用率并不高

**结论**

如果按当前能可信解释的“硬件/线程利用率”来读，最稳妥的结论是：

- compute thread 几乎一直贴着 kernel critical path；
- reader/writer 线程也会随序列增长越来越接近 critical path；
- 但 decode 的**真实算术利用率**并没有接近饱和，当前只在 `low-teens -> mid-20s` 区间。

**说明**

`compute share`、`BRISC share`、`NCRISC share` 更接近“线程离 critical path 有多近”；真正更接近硬件算术利用率的是 corrected-runtime `PM FPU util`。两者不能混为一谈。

### 2.1 线程级 critical-path attachment

| case | compute share | BRISC share | NCRISC share |
|---|---:|---:|---:|
| `decode_1k` | 99.68% | 83.8% | 39.5% |
| `decode_4k` | 99.86% | 93.0% | 62.7% |
| `decode_16k` | 99.96% | 97.9% | 88.8% |
| `decode_32k` | 99.98% | 98.9% | 94.3% |

表中指标含义：`compute share` 表示 compute 线程离 critical path 有多近；`BRISC share` 表示 `BRISC` 搬运线程离 critical path 有多近；`NCRISC share` 表示 `NCRISC` 搬运线程离 critical path 有多近。这三列更像“谁在决定总时延”，不是“谁的算术利用率更高”。

这说明随着序列增长，reader/writer 相关线程越来越贴边，kernel 越来越像一条强耦合流水线。

### 2.2 corrected-runtime 直接算术利用率

| case | seq_len | TT PM FPU util | DeepSeek-4c PM FPU util | TT Avg FPU util | DeepSeek-4c Avg FPU util |
|---|---:|---:|---:|---:|---:|
| `repr_seq_1k` | 1k | 12.17% | 12.63% | 10.36% | 10.75% |
| `repr_seq_4k` | 4k | 20.57% | 20.84% | 17.13% | 17.35% |
| `repr_seq_8k` | 8k | 23.11% | 23.34% | 19.18% | 19.37% |
| `repr_seq_16k` | 16k | 24.73% | 24.85% | 20.48% | 20.58% |
| `repr_seq_32k` | 32k | 25.61% | 25.67% | 21.19% | 21.24% |

表中指标含义：`seq_len` 是代表点序列长度；`TT PM FPU util` 和 `DeepSeek-4c PM FPU util` 是两条实现各自的直接算术利用率；`TT Avg FPU util` 和 `DeepSeek-4c Avg FPU util` 是对应实现在全 grid 上的平均 FPU 利用率。

**可以直接说什么**

- decode 的算术利用率会随序列增长单调抬升，而且在 `8k/16k` 上继续上升，但增量已经开始收敛；
- 即使到 `32k`，direct `PM FPU util` 也只是 `~25.6%`，还远不能说“已经算术饱和”；
- `TT` 与 `DeepSeek-4c` 的 `PM FPU util` 差值始终很小，最大只有 `0.46 pp`。

**边界**

当前 direct `PM FPU util` 已补到 representative seq `1k/4k/8k/16k/32k`，但还不是 `decode_1k~32k` dense curve。

---

## 3. Bubble 的主位置始终在 compute 的 `wait-front`，但长序列越来越带有 `reserve-back` 回压成分

**结论**

当前 `decode` bubble 的主位置始终在 compute 线程的 `wait-front`，但随着序列变长，`reserve-back` 占比持续上升，说明 long-seq 已经不是纯前端 starvation，而是前端供给不足和后端回压并存。

**证据数据**

| case | bubble / kernel | wait-front share | reserve-back share |
|---|---:|---:|---:|
| `decode_256` | 1.66x | 87.4% | 12.6% |
| `decode_1k` | 2.86x | 86.1% | 13.9% |
| `decode_4k` | 2.40x | 78.1% | 21.9% |
| `decode_8k` | 2.26x | 74.8% | 25.2% |
| `decode_16k` | 1.07x | 72.0% | 28.0% |
| `decode_32k` | 1.05x | 70.9% | 29.1% |

表中指标含义：`bubble / kernel` 是 compute bubble 相对 kernel 时长的密度；`wait-front share` 是 compute bubble 内前端供给不足的占比；`reserve-back share` 是 compute bubble 内后端回压的占比。

**说明**

- 短序列时，bubble 主要由 `wait-front` 主导，典型表现是 compute 在等前面喂数据。
- 长序列时，`reserve-back` 从 `12.6%` 抬到 `29.1%`，说明后端排空困难开始持续反馈到 compute。

**边界**

这里的 `wait-front/reserve-back` 是 counter density，不应直接当作 wall-time share。

---

## 4. `decode_4k` 是最关键的迁移点

**结论**

`decode_4k` 是从“主要是前端 starvation”转向“reader/writer 同时开始显性卡住”的关键转折点。

**证据数据**

| case | reader reserve | writer cb_wait | K reserve | wait-front share | reserve-back share |
|---|---:|---:|---:|---:|---:|
| `decode_1k` | 1.13% | 93.87% | 0.37% | 86.1% | 13.9% |
| `decode_4k` | 31.98% | 97.98% | 31.64% | 78.1% | 21.9% |

表中指标含义：`reader reserve` 是 reader 侧总等待里卡在 reserve 的占比；`writer cb_wait` 是 writer 侧总等待强度；`K reserve` 是 reader 卡在 `K` 路 reserve 的占比；`wait-front share` 和 `reserve-back share` 仍是 compute bubble 内部前端等待与后端回压的拆分。

**说明**

到 `4k` 时，reader 的 `reserve` 和 writer 的 `cb_wait` 同时进入高位，说明系统已经不再只是“compute 在等前面数据”，而是流水线前后两侧都开始持续施压。

更具体地说，`4k` 之所以被当作关键迁移点，不是因为它先验上有什么特殊名字，而是因为它是当前 sweep 里第一个把前端 reader 供给问题和后端 writer 回压**同时拉到显性区间**的点。和 `decode_1k` 相比，`decode_4k` 上 `reader reserve` 从 `1.13%` 跳到 `31.98%`，`K reserve` 从 `0.37%` 跳到 `31.64%`，而 `writer cb_wait` 又同时保持在 `97.98%` 的高位；因此从这个点开始，最自然的读法不再是“单边压力变大了”，而是“前端供给不足和后端回压一起开始主导流水线行为”。

**边界**

`4k` 仍是 proxy 上观察到的关键迁移点；direct corrected-runtime PM 现在已经补出 `8k/16k` 中间点，但还没有 densify 到完整 sweep。

同时，当前还**不能**把 `4k` 严格写成“`SRAM/L1` 在这里刚好占满”。现有结果并没有直接给出 `SRAM/L1 occupancy`、某个 buffer 的精确满载点，或类似 `unsupported_precheck` 那样的硬容量边界信号。如果这里真是一条非常明确的 hard wall，通常会期待看到 admission/precheck 失败、明确的 occupancy counter，或者很清晰的资源越界信号；而 `decode_4k` 目前没有这类证据。

因此，现阶段更稳妥的写法是：`decode_4k` 更像一个 **soft knee / 软拐点**，而不是已经被钉死的 hard wall。也就是说，工作集规模、`K` 路供给周转和 downstream backpressure 在这里第一次耦合到足够强，因而把 reader/writer 的双向压力清楚地暴露出来；但这还不等于我们已经证明“某块 `SRAM/L1` 在 `4k` 恰好满了”。

---

## 5. Reader 侧的主问题明确在 `K` 路，不在 `V` 路

**结论**

当前 `decode` 的 reader 供给瓶颈几乎全部集中在 `K` 路，而不是 `V` 路。

**证据数据**

| case | K issue | K reserve | V issue | V reserve |
|---|---:|---:|---:|---:|
| `decode_1k` | 56.87% | 0.37% | 25.33% | 0.81% |
| `decode_4k` | 38.70% | 31.64% | 17.51% | 0.56% |
| `decode_16k` | 27.39% | 52.79% | 12.53% | 2.77% |
| `decode_32k` | 26.10% | 54.82% | 11.96% | 2.80% |

表中指标含义：`K issue` 和 `V issue` 表示 reader 真正把时间花在发 `K/V` 路读请求上的占比；`K reserve` 和 `V reserve` 表示 reader 卡在 `K/V` 路 reserve 或 slot 等待上的占比。

这里这四列**不会直接加到 `100%`**，因为这张表是简化版，只保留了 `issue` 和 `reserve` 两类最关键指标；完整的 reader source attribution 还包含 `K wait` 和 `V wait` 两列。也就是说，reader 侧完整分解实际上是：`K issue + K reserve + K wait + V issue + V reserve + V wait ≈ 100%`，剩下的微小差值主要来自四舍五入。比如 `decode_4k` 的完整 reader 侧六列是 `38.70 + 31.64 + 10.07 + 17.51 + 0.56 + 0.53 = 99.01%`。同时，writer 侧的 `sender/tree/root/output` 又是另一组单独归一化的分解，不能和 reader 侧这些列横向相加。

**说明**

- `K issue` 从 `56.87%` 降到 `26.10%`；
- `K reserve` 从 `0.37%` 升到 `54.82%`；
- 与之相比，`V reserve` 到 `32k` 仍只有 `2.80%`。

这说明 long-seq 下 reader 不是“KV 一起变差”，而是 **`K` 路越来越多地卡在 reserve / slot 可用性等待上**。

---

## 6. 带宽相关的间接证据也指向 `K` 路供给恶化，但这还不是 direct NoC 利用率

**结论**

当前能够可靠支持的“带宽相关”结论是：随着序列变长，`K` 路的有效供给能力在下降；但这还不能被写成“已经拿到了可信的 NoC 带宽利用率百分比”。

**证据数据**

### 6.1 `K` 路有效读带宽 proxy

| case | K read GB/s | K reserve |
|---|---:|---:|
| `decode_1k` | 39.52 | 0.37% |
| `decode_4k` | 42.52 | 31.64% |
| `decode_16k` | 18.47 | 52.79% |
| `decode_32k` | 18.00 | 54.82% |

表中指标含义：`K read GB/s` 是 `K` 路有效读带宽 proxy，用来看 `K` 路读得是否顺畅；`K reserve` 是 `K` 路 reserve 等待占比，用来看 reader 是否越来越卡在 `K` 路供给上。

### 6.2 corrected-runtime 的 representative `PM BANDWIDTH`

| case | seq_len | PM IDEAL | PM BANDWIDTH |
|---|---:|---:|---:|
| `repr_seq_1k` | 1k | 8704 | 2129 |
| `repr_seq_4k` | 4k | 34816 | 8516 |
| `repr_seq_8k` | 8k | 69632 | 17033 |
| `repr_seq_16k` | 16k | 139264 | 34066 |
| `repr_seq_32k` | 32k | 278528 | 68132 |

表中指标含义：`seq_len` 是代表点序列长度；`PM IDEAL` 是 performance model 的 ideal baseline；`PM BANDWIDTH` 是 performance model 里的 bandwidth-limited 时间分量。

**说明**

- 从 proxy 看，`K read GB/s` 到 long-seq 下降明显，而 `K reserve` 同时升高；
- 从 corrected-runtime representative seq rerun（`1k/4k/8k/16k/32k`）看，`PM BANDWIDTH` 也随序列增长而增长。

这两件事都说明：**带宽压力在增加，而且 reader 的 `K` 供给越来越跟不上流水线节奏**。

**边界**

这些数据仍然不能替代 direct `NOC UTIL (%)`。它们只能支持“带宽相关压力在增大”，不能支持“当前 NoC 已达到 xx% 利用率”。

---

## 7. Writer 的主等待不在 final gather，而在 `sender/tree reduction` 链路

**结论**

writer 侧的主 backpressure 不是 `root` 或 `output gather`，而是 `sender` 与 `tree` 两段 reduction 链路。

**证据数据**

### 7.1 proxy 代表点 source-level attribution

| case | sender wait | tree wait | root wait | output wait |
|---|---:|---:|---:|---:|
| `decode_1k` | 48.69% | 51.31% | 0.00% | 0.00% |
| `decode_4k` | 49.59% | 50.41% | 0.00% | 0.00% |
| `decode_16k` | 49.89% | 50.11% | 0.00% | 0.00% |
| `decode_32k` | 49.95% | 50.05% | 0.00% | 0.00% |

表中指标含义：`sender wait` 表示 writer 等 sender/local-ready 的占比；`tree wait` 表示 writer 等 tree-child arrival 的占比；`root wait` 表示等待 root reduction 的占比；`output wait` 表示等待 final gather/output 的占比。

### 7.2 writer 总等待强度

| case | writer cb_wait |
|---|---:|
| `decode_1k` | 93.87% |
| `decode_4k` | 97.98% |
| `decode_16k` | 99.57% |
| `decode_32k` | 99.78% |

表中指标含义：`writer cb_wait` 是 writer 侧总等待强度，越高表示 writer 越像被后端路径长期卡住。

**说明**

这说明当前 writer 的等待不是尾部 gather 的偶发问题，而是 reduction 主链路本身就是结构性同步成本。

---

## 8. 当前 `TT-MLA` 慢的根因已经可以直接写成“前端 `K` 供给不足 + 后端 reduction backpressure”

**结论**

当前 `TT-MLA` 的 slowdown 本质上是双向挤压：

- 前端：`K` 路供给跟不上，compute 更容易 `wait-front`
- 后端：writer/reduction 排不空，compute 的 `reserve-back` 越来越重

**证据数据**

| 指标 | 短序列代表值 | 长序列代表值 |
|---|---|---|
| reader reserve | `decode_1k = 1.13%` | `decode_32k = 57.58%` |
| K reserve | `decode_1k = 0.37%` | `decode_32k = 54.82%` |
| writer cb_wait | `decode_1k = 93.87%` | `decode_32k = 99.78%` |
| wait-front share | `decode_1k = 86.1%` | `decode_32k = 70.9%` |
| reserve-back share | `decode_1k = 13.9%` | `decode_32k = 29.1%` |

表中指标含义：`指标` 是要对比的现象；`短序列代表值` 和 `长序列代表值` 分别用 `decode_1k` 与 `decode_32k` 的值来表示这一指标随序列增长的迁移方向。

**说明**

这组数据共同说明，表面上虽然 `compute share` 很高，但它反映的不是“算得很满”，而是 compute 被前后级 stall 一起夹住了，所以总是挂在 critical path 上。

---

## 9. `SF-MLA` / `DeepSeek-4c` 的优势首先来自减少 front-side starvation，不是把算术利用率拉到完全不同的区间

**结论**

现在已经可以比较稳妥地写：`DeepSeek-4c` 的优势首先来自 mapping/dataflow 对前端 starvation 的缓解，而不是“用了更多核”或者“算术利用率完全不同”。

**证据数据**

| case | TT kernel us | DeepSeek-4c kernel us | TT wait-front / kernel | DeepSeek-4c wait-front / kernel | TT PM FPU util | DeepSeek-4c PM FPU util |
|---|---:|---:|---:|---:|---:|---:|
| `repr_seq_1k` | 73.86 | 70.36 | 108.1% | 94.9% | 12.17% | 12.63% |
| `repr_seq_4k` | 174.67 | 171.26 | 87.2% | 81.3% | 20.57% | 20.84% |
| `repr_seq_32k` | 1114.36 | 1110.67 | 74.5% | 73.2% | 25.61% | 25.67% |

表中指标含义：`TT kernel us` 和 `DeepSeek-4c kernel us` 是两条实现的 kernel 时长；`TT wait-front / kernel` 和 `DeepSeek-4c wait-front / kernel` 是两条实现的前端等待密度；`TT PM FPU util` 和 `DeepSeek-4c PM FPU util` 是两条实现的直接算术利用率。

补充 source-level 证据：

| case | TT K reserve | DeepSeek-4c K reserve | TT V reserve | DeepSeek-4c V reserve |
|---|---:|---:|---:|---:|
| `repr_seq_4k` | 31.48% | 28.59% | 2.25% | 0.56% |
| `repr_seq_32k` | 51.40% | 51.06% | 2.62% | 0.39% |

表中指标含义：`TT K reserve` 与 `DeepSeek-4c K reserve` 是两条实现的 `K` 路 reserve 等待占比；`TT V reserve` 与 `DeepSeek-4c V reserve` 是两条实现的 `V` 路 reserve 等待占比。

**说明**

- `DeepSeek-4c` 在 `repr_seq_1k/repr_seq_4k/repr_seq_32k` 三个代表点上 kernel 都略低；
- `wait-front / kernel` 也 consistently 更低；
- 但 `PM FPU util` 与 `TT` 的差值始终很小。

所以更合理的写法不是“DeepSeek 算得更满”，而是“DeepSeek 更少卡在前端 starvation 上”。

**边界**

这条结论目前建立在 representative direct points 与固定 `8k` 的 densified boundary points（`24/28/32/40/48 q-cores`）上，而不是 `DeepSeek` 的 dense seq sweep。

---

## 10. 在已经 fit 进 `4c` 容量的点上，`8c` 几乎不会自动更快

**结论**

当 workload 已经 fit 进 `4c` 的 q-core 容量时，`8c` 并不会自动带来额外性能收益。

**证据数据**

| case | required q-cores | 4c kernel us | 8c kernel us | 8c / 4c |
|---|---:|---:|---:|---|
| `fit24q_b6_h32_qs4_s4k` | 24 | 172.47 | 172.11 | 0.998x |
| `fit24q_b6_h32_qs4_s8k` | 24 | 256.62 | 257.24 | 1.002x |
| `fit24q_b6_h32_qs4_s32k` | 24 | 692.26 | 688.83 | 0.995x |
| `fit24q_b8_h24_qs3_s4k` | 24 | 182.97 | 183.00 | 1.000x |
| `fit24q_b8_h24_qs3_s8k` | 24 | 289.99 | 291.02 | 1.004x |
| `fit24q_b8_h24_qs3_s32k` | 24 | 851.90 | 859.46 | 1.009x |

表中指标含义：`required q-cores` 是该 workload 需要的 q-core 数量；`4c kernel us` 和 `8c kernel us` 是 `4c/8c` 配置下的 kernel 时长；`8c / 4c` 是两者时长比，小于 `1x` 表示 `8c` 更快。

同时，source-level 结构也几乎不变：

| 指标 | 4c | 8c |
|---|---|---|
| `K reserve` | `0.11%~0.13%` | `0.11%~0.13%` |
| `sender` | `41.73%~43.83%` | `41.71%~43.82%` |
| `tree` | `55.96%~57.56%` | `55.97%~57.58%` |

表中指标含义：`指标` 是要对比的 source-level 现象；`4c` 和 `8c` 是该现象在两种配置下的取值范围，用来看扩到 `8c` 后结构有没有明显变化。

同一条 `fit24q_b6_h32_qs4_s8k` corrected-runtime `perf-fpu` anchor 上，算术利用率也没有出现 `8c` 跳变：

| case | TT PM FPU util | DeepSeek-4c PM FPU util | DeepSeek-8c PM FPU util |
|---|---:|---:|---:|
| `fit24q_b6_h32_qs4_s8k` | 21.86% | 12.08% | 12.08% |

表中指标含义：`TT PM FPU util`、`DeepSeek-4c PM FPU util`、`DeepSeek-8c PM FPU util` 是同一条已 fit 配置点在三种实现/配置下的直接算术利用率，用来看 `8c` 是否在已 fit 点上带来算术利用率跳变。

**说明**

`8c≈4c` 不只是 kernel 时间接近，reader/writer 的主等待结构也没有被改写；同一条 `fit24q_b6_h32_qs4_s8k` anchor 上，`DeepSeek-4c/8c` 的 `PM FPU util` 也几乎完全相同。因此 `8c` 的价值不在“所有点都更快”。

---

## 11. `8c` 的一阶价值是跨越容量墙，而不是已 fit 点上的普适加速

**结论**

`8c` 最直接的价值是把 q-core 容量上限从 `24` 抬到 `48`，从而让 `4c` 跑不动的点重新变成可运行。

**证据数据**

| case | required q-cores | 4c status | 8c kernel us | TT kernel us | 8c / TT |
|---|---:|---|---:|---:|---|
| `cross32q_b8_qs4_s8k` | 32 | `unsupported_precheck` | 290.74 | 336.05 | 0.865x |
| `cross48q_b12_qs4_s8k` | 48 | `unsupported_precheck` | 376.06 | 382.63 | 0.983x |

表中指标含义：`required q-cores` 是 workload 需要的 q-core 数量；`4c status` 表示 `4c` 是否能通过 admission/precheck；`8c kernel us` 和 `TT kernel us` 是 `8c` 与 `TT` 的 kernel 时长；`8c / TT` 是两者时长比，小于 `1x` 表示 `8c` 更快。

`4c` 的直接报错信息是：

- `batch * deepseek_num_q_shards must be <= 24, got 32`
- `batch * deepseek_num_q_shards must be <= 24, got 48`

同一条 boundary corrected-runtime `perf-fpu` rerun 还把这条轴 densify 成了 `24/28/32/40/48 q-cores` 的直接算术利用率：

| case | required q-cores | DeepSeek-4c status | TT PM FPU util | DeepSeek-8c PM FPU util |
|---|---:|---|---:|---:|
| `fit24q_b6_h32_qs4_s8k` | 24 | `completed` | 21.86% | 12.08% |
| `cross28q_b7_qs4_s8k` | 28 | `unsupported_precheck` | 21.60% | 13.78% |
| `cross32q_b8_qs4_s8k` | 32 | `unsupported_precheck` | 21.27% | 13.74% |
| `cross40q_b10_qs4_s8k` | 40 | `unsupported_precheck` | 20.36% | 16.40% |
| `cross48q_b12_qs4_s8k` | 48 | `unsupported_precheck` | 18.21% | 18.53% |

表中指标含义：`required q-cores` 是该 boundary 点需要的 q-core 数量；`DeepSeek-4c status` 表示 `4c` 是否能通过 admission；`TT PM FPU util` 和 `DeepSeek-8c PM FPU util` 是这条 `24/28/32/40/48` q-core 轴上的直接算术利用率。

**说明**

这张 densified 表说明两件事：第一，`4c` 真正的 hard wall 就在 `24 -> 28 q-cores` 之间，`D28` 已经是“刚越墙就直接 unsupported”的第一手证据；第二，`8c` 的一阶作用仍然是抬高 admission boundary，而不是在所有 workload 上都天然加速，也不是先把算术利用率抬到另一个区间。

---

## 12. 即使跨过了容量墙，残余主瓶颈仍然在 sender/tree backpressure

**结论**

容量墙不是终点。即使 `8c` 已经把 case 跑通，剩余主矛盾仍然是 downstream backpressure，尤其是 sender/tree reduction 链路。

**证据数据**

| case | 8c K reserve | 8c sender | 8c tree | 8c root | 8c output | 8c reserve_back / kernel |
|---|---:|---:|---:|---:|---:|---|
| `cross32q_b8_qs4_s8k` | 0.12% | 41.74% | 57.55% | 0.71% | 0.00% | 257.1% |
| `cross48q_b12_qs4_s8k` | 0.14% | 49.41% | 49.72% | 0.88% | 0.00% | 276.3% |

表中指标含义：`8c K reserve` 是 `8c` 下 `K` 路 reserve 等待占比；`8c sender/tree/root/output` 是 `8c` 下 writer 各 reduction/source 阶段的等待占比；`8c reserve_back / kernel` 是 `8c` 下后端回压密度相对 kernel 时长的比值。

**说明**

`cross48q_b12_qs4_s8k` 这个点非常关键：

- `K reserve` 已经很低，不再像 `TT` long-seq 那样是 reader 供给主问题；
- 但 `reserve_back / kernel` 仍高达 `276.3%`；
- writer source 仍几乎全部落在 `sender/tree`，不是 `root/output`。

这说明跨过容量墙后，主瓶颈会继续转到 reduction/backpressure，而不是自然消失。

---

## 13. `prefill` 可以作为控制组：算术利用率会上升，但主 wall-time 仍然主要由流水线等待决定

**结论**

`prefill` 从一开始就是 reader/writer 强耦合的饱和流水线。即使 direct PM util 随序列上升，主 wall-time 仍主要由等待决定。

**证据数据**

| case | PM FPU util | writer cb_wait | wait-front share | reserve-back share |
|---|---:|---:|---:|---:|
| `prefill_256` | 10.20% | 91.50% | 95.98% | 4.02% |
| `prefill_512` | 13.90% | 95.71% | 94.62% | 5.38% |
| `prefill_1k` | 16.76% | 98.25% | 93.58% | 6.42% |
| `prefill_2k` | 18.48% | 99.19% | 92.93% | 7.07% |
| `prefill_4k` | 19.43% | 99.62% | 92.58% | 7.42% |

表中指标含义：`PM FPU util` 是 prefill 的直接算术利用率；`writer cb_wait` 是 writer 侧总等待强度；`wait-front share` 和 `reserve-back share` 是 compute bubble 内前端等待与后端回压的占比。

额外的 `prefill_4k` PM triplet：

- `kernel wall = 55.516 ms`
- `PM COMPUTE = 10.785 ms`
- `PM BANDWIDTH = 0.545 ms`
- `PM FPU UTIL = 19.427%`

**说明**

`prefill` 的意义主要是提供一个控制组：即使 direct PM util 变高，wall-time 仍可以主要由流水线等待决定。

---

## 14. 关于 `NoC` / `DRAM` 带宽利用率：当前最可靠的结论是“还不能可信地给出百分比”

**结论**

如果问题是“当前 `Part II` 能不能回答 `NoC` 带宽利用率到底是多少”，最准确的答案是：**现在还不能**。

**证据数据**

### 14.1 decode full sweep 的 direct `NOC/DRAM` 列仍不可用

| case | PM IDEAL | PM COMPUTE | PM BANDWIDTH | PM FPU UTIL | NOC UTIL | MULTICAST NOC UTIL | DRAM BW UTIL |
|---|---:|---:|---:|---:|---|---|---|
| `decode_256` | 1.0 | 1.0 | 1.0 | 0.002 | `missing` | `missing` | `missing` |
| `decode_512` | 1.0 | 1.0 | 1.0 | 0.002 | `missing` | `missing` | `missing` |
| `decode_1k` | 1.0 | 1.0 | 1.0 | 0.001 | `missing` | `missing` | `missing` |
| `decode_2k` | 1.0 | 1.0 | 1.0 | 0.001 | `missing` | `missing` | `missing` |
| `decode_4k` | 1.0 | 1.0 | 1.0 | 0.001 | `missing` | `missing` | `missing` |
| `decode_8k` | 1.0 | 1.0 | 1.0 | 0.000 | `missing` | `missing` | `missing` |
| `decode_16k` | 1.0 | 1.0 | 1.0 | 0.000 | `missing` | `missing` | `missing` |
| `decode_32k` | 1.0 | 1.0 | 1.0 | 0.000 | `missing` | `missing` | `missing` |

表中指标含义：`PM IDEAL`、`PM COMPUTE`、`PM BANDWIDTH` 是 performance model 三项；`PM FPU UTIL` 是 direct FPU 利用率；`NOC UTIL`、`MULTICAST NOC UTIL`、`DRAM BW UTIL` 是 direct `NoC`/multicast/`DRAM` 带宽利用率。这里这些列目前是 `missing` 或 placeholder，所以不能当作真实硬件利用率。

配套状态总结是：

- `NOC UTIL` 可用：`0/8`
- `DRAM BW UTIL` 可用：`0/8`

### 14.2 `--collect-noc-traces` 当前也不能稳定放大

现有 source report 明确记录了：

- `--collect-noc-traces` 会触发 `Invalid NoC transfer type`

**说明**

因此现在最多只能写成：

- 有带宽压力增大的**间接证据**，比如 `K read GB/s` 下降、`PM BANDWIDTH` 上升、`K reserve` 升高；
- 但**没有**可信的 direct `NOC UTIL (%)` / `DRAM BW UTIL (%)` 百分比曲线。

---

## 15. 最终可直接写入正文的总括

把当前 `Part II` 所有可支持结论压缩成一句话，就是：

> 现阶段 `decode` 的主问题不是算术单元已经饱和，而是 long-seq 下前端 `K` 供给不足与后端 `sender/tree` reduction backpressure 同时放大了 compute bubble；`DeepSeek/SF-MLA` 的现有优势主要来自对 front-side starvation 的缓解，而 `8c` 的一阶价值主要体现在跨越 q-core 容量墙，而不是已 fit 点上的普适加速。

如果压缩成三条最重要的工程结论，则是：

1. **先打 `K` 路供给**：`K reserve` 是 decode long-seq 最稳定、最强的 reader 侧问题。
2. **再打 sender/tree reduction**：writer 主等待不在 final gather，而在 reduction 主链路。
3. **`8c` 主要解决 admission，不自动解决 backpressure**：跨过容量墙以后，残余瓶颈仍在 downstream backpressure。

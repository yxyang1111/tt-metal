# FlashMLA WH 组件利用率与空泡详细分析

## 1. 文档目的

这份文档的目标不是再重复一遍“谁慢谁快”，而是把当前 Wormhole 上 FlashMLA 的 profiling 结果按**组件利用率**和**空泡来源**重新整理成一份可直接指导优化的分析文档，重点回答四个问题：

- `decode` / `prefill` 分别是谁在逼近 critical path。
- reader、writer、compute 三侧各自的“忙”到底在忙什么。
- 空泡主要出现在什么位置，规模有多大，随序列长度如何迁移。
- 现阶段哪些结论可以直接下，哪些结论还不能下。

## 2. 数据来源与解释口径

### 2.1 数据来源

本文混合使用了三类数据源：

- full sweep 组件窗口与 stage-level 数据：
  - `mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_results.json`
- decode compute bubble 汇总：
  - `mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_decode_summary.csv`
- 同步代表点 source-level / direct PM 补充：
  - `mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe_sync_smoke/`
  - `mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe_repr_sync/`
  - `mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe_sync_extra/`

需要注意：不同表格有时来自不同 rerun，所以同一 case 的绝对时延可能会有很小的 run-to-run 漂移；但本文关注的是**趋势**与**归因**，这部分结论是稳定的。

### 2.2 “利用率”在本文里的准确含义

本文里“利用率”分三层，必须区分开读：

- `BRISC/NCRISC/TRISC share`：
  - 这是**线程窗口占比**，表示某条线程在 kernel window 里占了多大比例。
  - 它能回答“谁在逼近 critical path”，但不能直接回答“FPU 真正用了多少”。
- reader / writer stage share：
  - 这是**组件内部阶段占比**，例如 reader 的 `issue / reserve / wait`，writer 的 `cb_wait / issue / barrier / pop`。
  - 它最适合回答“空泡主要积在哪个组件的哪个阶段”。
- `PM FPU UTIL (%)` / `PM COMPUTE`：
  - 这是更接近**真正算术利用率**的 direct PM 指标。
  - 目前只在 `prefill` 上可靠，在 `decode` 上仍是 placeholder。

### 2.3 “空泡有多大”在本文里的准确含义

`DEVICE COMPUTE CB WAIT FRONT` 和 `DEVICE COMPUTE CB RESERVE BACK` 在本文统一按：

- **accumulated stall density**

来解释，而不是 wall-time 占比。也就是说：

- 当 `bubble density vs kernel > 1x` 时，不代表“kernel 有超过 100% 的时间在空泡”。
- 它代表的是：跨核心、跨轮次累计起来，compute 在这些位置上看到了大量 stall。

所以：

- `wait-front share in bubble` 更适合读成“compute 主要在等输入/前端数据到齐”。
- `reserve-back share in bubble` 更适合读成“compute 逐步更多地感受到输出端/下游回压”。

### 2.4 当前可直接用、不能直接用的指标

当前可信度分层如下：

- 可直接用：
  - decode / prefill 的 `BRISC/NCRISC/TRISC share`
  - reader / writer stage-level breakdown
  - decode compute bubble counter
  - decode 代表点 source-level breakdown
  - prefill 的 direct PM util
- 不能直接用：
  - decode 的 `PM IDEAL / PM COMPUTE / PM BANDWIDTH / PM FPU UTIL`
  - decode 的 `NOC / MULTICAST NOC / DRAM BW util`
  - 当前 `--collect-noc-traces` 路线上的 direct NOC attribution

## 3. 一页结论

- `decode_256 ~ 1k` 仍然是 compute 最后退休，但 writer 已经明显不轻，reader 仍以 `issue` 为主。
- `decode_2k ~ 8k` 的主变化不是 compute 变慢，而是 writer 几乎整段被 `cb_wait` 占住，同时 reader 从 `issue` 迅速转向 `reserve`。
- `decode_16k ~ 32k` 时，reader / writer / compute 三方都贴近 kernel window，reader 内部主导空泡已经明确落在 `K reserve`，而 writer 的主等待稳定地均分在 `sender_cb_wait` 和 `tree_child_wait`。
- `prefill_256 ~ 4k` 从起点开始就是 `reader_writer_saturated`，reader 始终以 `wait` 主导，writer 始终以 `cb_wait` 主导。
- `prefill` 的 direct PM util 虽然会从 `10.2%` 升到 `19.4%`，但仍远低于真正算术饱和。这说明 wall time 的主矛盾不是算力不足，而是耦合流水线里的等待。

### 3.1 代表性泳道图

静态泳道图资源已经导出到：

- `mla_flash_attention_dev/docs/assets/flash-mla-wh-profile-swimlanes/`

其中：

- `SVG` 适合文档内联预览
- `PNG` 适合拿去汇报或贴到外部文档
- 原始 Mermaid 源版见 `mla_flash_attention_dev/docs/flash-mla-wh-profile-swimlane-diagrams.md`
- 如果需要看同一个 `S block` 内部、不同 `S block` 之间、多 core、`Host CPU`、`DRAM`、`CB/semaphore` 信号的详细版，请看：
  - `mla_flash_attention_dev/docs/flash-mla-wh-detailed-swimlane-diagrams.md`
  - `mla_flash_attention_dev/docs/assets/flash-mla-wh-profile-swimlanes-detailed/`

#### `decode_1k`：短序列，仍接近 compute-critical

![decode_1k 泳道图](assets/flash-mla-wh-profile-swimlanes/decode_1k.svg)

#### `decode_4k`：关键迁移点，reader 开始显著转向 `K reserve`

![decode_4k 泳道图](assets/flash-mla-wh-profile-swimlanes/decode_4k.svg)

#### `decode_16k`：长序列形态基本固定，`K reserve + sender/tree wait` 成为主轴

![decode_16k 泳道图](assets/flash-mla-wh-profile-swimlanes/decode_16k.svg)

#### `decode_32k`：`decode_16k` 形态继续固化

![decode_32k 泳道图](assets/flash-mla-wh-profile-swimlanes/decode_32k.svg)

#### `prefill_4k`：direct PM util 可读，但主 wall-time 仍由流水线等待决定

![prefill_4k 泳道图](assets/flash-mla-wh-profile-swimlanes/prefill_4k.svg)

## 4. Decode：各组件利用率与空泡详细分析

### 4.1 线程窗口层：谁在逼近 critical path

| case | 分类 | kernel us | max TRISC share | BRISC share | NCRISC share |
|---|---|---:|---:|---:|---:|
| `decode_256` | `compute_on_critical_path` | 40.917 | 99.37% | 70.34% | 33.35% |
| `decode_512` | `compute_on_critical_path` | 58.109 | 99.55% | 79.14% | 31.42% |
| `decode_1k` | `compute_on_critical_path` | 73.531 | 99.64% | 83.47% | 37.06% |
| `decode_2k` | `writer_close_to_critical_path` | 106.069 | 99.75% | 88.54% | 42.56% |
| `decode_4k` | `writer_close_to_critical_path` | 171.855 | 99.85% | 92.93% | 62.64% |
| `decode_8k` | `writer_close_to_critical_path` | 303.168 | 99.91% | 95.99% | 78.70% |
| `decode_16k` | `reader_close_to_critical_path` | 560.899 | 99.95% | 97.84% | 88.74% |
| `decode_32k` | `reader_writer_saturated` | 1084.760 | 99.98% | 98.88% | 94.34% |

这张表最重要的点不是 `TRISC` 始终接近 `100%`，而是：

- `TRISC share` 在全段都接近满窗，说明 compute 线程始终没有“早早退休”。
- 但 `BRISC/NCRISC` 随序列增长持续逼近 kernel window，说明真正的变化来自 reader / writer 不断贴边。
- 也就是说，decode 的瓶颈迁移不是“compute 不重要了”，而是从“compute 末退休”逐步过渡到“reader / writer / compute 共同压在 critical window 上”。

可以把 decode 分成三段来读：

- `256 ~ 1k`：
  - 仍然最接近 `compute_on_critical_path`
  - writer 已经不轻，但 reader 还没进入明显反压区
- `2k ~ 8k`：
  - writer 先贴边
  - reader 开始明显向 `reserve` 漂移
- `16k ~ 32k`：
  - NCRISC 几乎贴满 kernel window
  - 这时说“纯 compute-bound”已经不准确，更准确的是强耦合饱和

### 4.2 Reader / Writer 内部阶段占比

| case | reader issue | reader reserve | reader wait | writer cb_wait | writer issue | writer barrier | writer pop |
|---|---:|---:|---:|---:|---:|---:|---:|
| `decode_256` | 75.45% | 0.67% | 15.22% | 89.42% | 5.40% | 4.89% | 0.29% |
| `decode_512` | 74.12% | 0.66% | 15.44% | 90.25% | 5.06% | 4.38% | 0.31% |
| `decode_1k` | 77.27% | 0.70% | 16.49% | 94.15% | 3.01% | 2.64% | 0.21% |
| `decode_2k` | 78.81% | 0.72% | 16.95% | 96.28% | 1.97% | 1.63% | 0.13% |
| `decode_4k` | 47.94% | 39.78% | 10.65% | 98.10% | 0.96% | 0.86% | 0.07% |
| `decode_8k` | 38.47% | 51.96% | 8.58% | 98.90% | 0.57% | 0.49% | 0.04% |
| `decode_16k` | 34.90% | 59.71% | 4.59% | 99.56% | 0.22% | 0.20% | 0.02% |
| `decode_32k` | 33.38% | 61.54% | 4.37% | 99.77% | 0.11% | 0.11% | 0.01% |

从 reader 端看，迁移非常清楚：

- `decode_256 ~ 2k`：
  - reader 主要时间都花在 `issue`
  - `reserve` 只有 `0.67% ~ 0.72%`
  - 这意味着 reader 主要还是在主动发请求，而不是被下游堵住
- `decode_4k`：
  - 这是最关键的迁移点
  - `issue` 从 `78.81%` 突降到 `47.94%`
  - `reserve` 从 `0.72%` 突升到 `39.78%`
- `decode_8k ~ 32k`：
  - `reserve` 上升到 `51.96% -> 61.54%`
  - `issue` 下降到 `38.47% -> 33.38%`
  - reader 已从“主动发请求”转成“主要在等下游释放空间”

从 writer 端看，结论同样稳定：

- `cb_wait` 从 `89.42%` 一路升到 `99.77%`
- `issue / barrier / pop` 全部被压成边角料

这说明 decode writer 不是 issue-bound，也不是 barrier-bound，而是：

- 大部分时间都在等上游结果 ready 或下游 reduction 链路继续推进

### 4.3 Compute 看到的空泡到底有多大

| case | kernel us | wait-front us | reserve-back us | bubble density vs kernel | wait-front share in bubble | reserve-back share in bubble |
|---|---:|---:|---:|---:|---:|---:|
| `decode_256` | 42.198 | 61.200 | 8.789 | 1.659x | 87.44% | 12.56% |
| `decode_512` | 59.574 | 162.581 | 19.845 | 3.062x | 89.12% | 10.88% |
| `decode_1k` | 76.223 | 187.483 | 30.225 | 2.856x | 86.12% | 13.88% |
| `decode_2k` | 109.765 | 234.231 | 51.309 | 2.601x | 82.03% | 17.97% |
| `decode_4k` | 177.332 | 332.069 | 93.027 | 2.397x | 78.12% | 21.88% |
| `decode_8k` | 311.103 | 525.598 | 177.100 | 2.259x | 74.80% | 25.20% |
| `decode_16k` | 575.166 | 442.915 | 172.588 | 1.070x | 71.96% | 28.04% |
| `decode_32k` | 1112.087 | 829.716 | 340.300 | 1.052x | 70.91% | 29.09% |

这张表要分两个维度来读。

第一，空泡组成怎么变：

- `wait-front share` 从约 `87% ~ 89%` 下降到 `71%`
- `reserve-back share` 从约 `11% ~ 13%` 上升到 `29%`

这说明：

- 短序列 decode 时，compute 主要在等“前面没有数据喂过来”
- 序列变长后，compute 逐步更多地感受到输出端 / 回压侧问题

第二，空泡绝对量有多大：

- `wait-front` 从 `61.2 us` 增长到 `829.7 us`
- `reserve-back` 从 `8.8 us` 增长到 `340.3 us`

也就是说，长序列 decode 的 compute 不仅还在等输入，而且越来越明显地感受到了下游回压。

### 4.4 Reader 的空泡到底来自 K 还是 V

| case | K total | V total | k_issue | k_reserve | k_wait | v_issue | v_reserve | v_wait | sender wait | tree-child wait | root wait | output-gather wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `decode_1k` | 72.51% | 27.49% | 56.92% | 0.36% | 14.44% | 25.36% | 0.81% | 0.67% | 48.58% | 51.42% | 0.00% | 0.00% |
| `decode_4k` | 80.95% | 19.05% | 38.70% | 31.64% | 10.07% | 17.51% | 0.56% | 0.53% | 49.59% | 50.41% | 0.00% | 0.00% |
| `decode_16k` | 84.05% | 15.95% | 27.39% | 52.79% | 3.49% | 12.53% | 2.77% | 0.33% | 49.89% | 50.11% | 0.00% | 0.00% |
| `decode_32k` | 84.61% | 15.39% | 26.10% | 54.82% | 3.33% | 11.96% | 2.80% | 0.32% | 49.95% | 50.05% | 0.00% | 0.00% |

reader 的 source-level 迁移可以直接写成：

- `decode_1k`：
  - `K/V issue` 主导
  - `k_reserve` 只有 `0.36%`
- `decode_4k`：
  - `k_reserve` 直接升到 `31.64%`
  - 这是第一次明确看到 reader backpressure 落到 `K` 路
- `decode_16k`：
  - `k_reserve = 52.79%`
  - reader 内部已经明确由 `K reserve` 主导
- `decode_32k`：
  - `k_reserve = 54.82%`
  - 这个模式继续固化

这里最重要的工程结论是：

- 长序列 decode 的 reader 空泡不是笼统的“reserve 变大了”
- 而是更具体地说：**`K` 路 buffering / turnover / consumption path 正在制造主导性 backpressure**

`V reserve` 始终很小：

- `decode_4k = 0.56%`
- `decode_16k = 2.77%`
- `decode_32k = 2.80%`

因此不能把长序列 reader backpressure 归因到 `V` 路。

### 4.5 Writer 的空泡到底卡在哪一段

writer source-level 结果比 stage-level 更进一步，说明 `writer cb_wait` 主要被哪一段吃掉：

- `sender_cb_wait` 基本稳定在 `48.58% ~ 49.95%`
- `tree_child_wait` 基本稳定在 `50.05% ~ 51.42%`
- `root_cb_wait = 0%`
- `output_gather_wait = 0%`

这意味着：

- writer 的主等待**不在 final gather**
- 也**不在 root 最后写出**
- 真正的主等待发生在：
  - sender 在等本地 partial output ready
  - tree / root reduction 在等 child partial result 到齐

因此 decode 的 writer 问题，不应简单描述成“写带宽不够”，而应描述成：

- 输出结果在 sender 到 tree reduction 这条内部链路上没有持续 ready

### 4.6 Decode 的空泡传导链

把上面的 reader / writer / compute 三层数据放到一起，decode 的空泡传导链可以更具体地写成：

1. 短序列时，reader 主要在发 K/V 请求，compute 主要看到 `wait-front`。
2. 到 `decode_4k`，reader 的 `reserve` 突然抬升，尤其是 `k_reserve` 抬升，表示 `K` 路 turnover 开始被下游堵住。
3. writer 端则持续被 `cb_wait` 占满，而且 source-level 证明主等待在 `sender/tree` 内部链路。
4. compute 端因此一方面继续看到大量 `wait-front`，另一方面开始越来越多地看到 `reserve-back`。
5. 到 `decode_16k ~ 32k`，reader / writer / compute 已经形成强耦合回压回路。

## 5. Prefill：各组件利用率与空泡详细分析

### 5.1 线程窗口与 stage-level 占比

| case | 分类 | kernel ms | max TRISC share | BRISC share | NCRISC share | reader issue | reader reserve | reader wait | writer cb_wait | writer issue | writer barrier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefill_256` | `reader_writer_saturated` | 0.413 | 99.30% | 99.89% | 97.08% | 28.90% | 0.28% | 70.30% | 91.26% | 4.18% | 4.51% |
| `prefill_512` | `reader_writer_saturated` | 1.212 | 99.75% | 99.96% | 99.00% | 30.62% | 0.30% | 68.65% | 95.63% | 2.27% | 2.07% |
| `prefill_1k` | `reader_writer_saturated` | 4.009 | 99.93% | 99.99% | 99.70% | 32.08% | 0.32% | 67.19% | 98.30% | 0.81% | 0.87% |
| `prefill_2k` | `reader_writer_saturated` | 14.557 | 99.98% | 100.00% | 99.92% | 32.74% | 0.33% | 66.53% | 99.20% | 0.39% | 0.40% |
| `prefill_4k` | `reader_writer_saturated` | 55.339 | 99.99% | 100.00% | 99.98% | 33.15% | 0.33% | 66.10% | 99.62% | 0.19% | 0.19% |

这张表说明 prefill 和 decode 最大的区别是：

- prefill **从最短点开始**就是 `reader_writer_saturated`
- 不存在 decode 那种“先 compute-critical，再 writer-close，再 reader-close”的迁移过程

reader 端的特征非常稳定：

- `issue` 只是在 `28.90% -> 33.15%` 间小幅提升
- `reserve` 始终接近零，只在 `0.28% -> 0.33%`
- `wait` 始终占主导，而且只是缓慢从 `70.30%` 降到 `66.10%`

这说明 prefill reader 的主问题不是像长 decode 那样“被 reserve/backpressure 卡住”，而是：

- 大部分时间都在等 in-flight read completion

writer 端同样很稳定：

- `cb_wait` 从 `91.26%` 升到 `99.62%`
- `issue / barrier` 很快被压到个位数甚至接近零

这意味着 prefill writer 几乎一直都在等上游产出，而不是在忙真正的 issue / barrier 工作。

### 5.2 Direct PM 与 compute bubble

| case | kernel ms | PM COMPUTE ms | PM BANDWIDTH ms | PM FPU util | wait-front us | reserve-back us | wait-front share in bubble | reserve-back share in bubble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefill_256` | 0.413 | 0.042 | 0.034 | 10.196% | 8673.901 | 362.881 | 96.0% | 4.0% |
| `prefill_512` | 1.212 | 0.169 | 0.068 | 13.904% | 24165.691 | 1372.770 | 94.6% | 5.4% |
| `prefill_1k` | 4.021 | 0.674 | 0.136 | 16.762% | 77121.683 | 5287.725 | 93.6% | 6.4% |
| `prefill_2k` | 14.595 | 2.696 | 0.273 | 18.475% | 273708.894 | 20821.675 | 92.9% | 7.1% |
| `prefill_4k` | 55.516 | 10.785 | 0.545 | 19.427% | 1029320.475 | 82474.895 | 92.6% | 7.4% |

这张表可以直接回答“prefill 的真实算术利用率到底怎样”：

- `PM FPU UTIL` 会从 `10.196%` 升到 `19.427%`
- 这说明序列越长，算术管线确实越“热”
- 但即使到 `prefill_4k`，direct compute 也只解释了约 `19.4%` 的 kernel wall time

也就是说：

- compute 确实在工作
- 但它远没有被真正打满
- wall time 里更大的部分仍然来自 pipeline waiting

compute bubble 组成也非常稳定：

- `wait-front share` 始终在 `92.6% ~ 96.0%`
- `reserve-back share` 只是在 `4.0% -> 7.4%` 间缓慢上升

因此 prefill compute 看到的主空泡，不是“输出端突然塌了”，而是：

- 整个过程都主要在等前端数据和前序流水段把输入喂齐

### 5.3 为什么 TRISC 几乎满窗，但 PM FPU util 只有 10% ~ 19%

这是 prefill 里最容易误读的地方。

从 `TRISC share` 看：

- `prefill_256 ~ 4k` 全部接近 `100%`

如果只看这个指标，很容易误判成：

- “compute 已经打满”

但 direct PM 恰好给了一个相反的结论：

- `PM FPU util = 10.2% -> 19.4%`

这两个指标并不矛盾，原因是：

- `TRISC share` 只是说明 compute 线程还在 kernel window 里活着
- `PM FPU util` 才更接近“其中有多少时间真正在做算术”

因此 prefill 的正确读法是：

- compute 线程窗口被整个流水线占住了
- 但 compute 中真正执行高效算术的比例仍然不高
- 其余时间被 `wait-front` 等待吞掉了

### 5.4 Prefill 的空泡传导链

prefill 的链路比 decode 更稳定，也更“从一开始就耦合”：

1. reader 主要时间花在 `wait`，说明它一直在等读完成，而不是在等 CB 空间。
2. writer 几乎整个窗口都是 `cb_wait`，说明它始终在等 compute / reduction 产出。
3. compute 端则持续以 `wait-front` 为主，说明输入侧等待从起点开始就是主导项。

这三件事放在一起，形成的图像非常一致：

- prefill 不是单点瓶颈，而是 reader / writer / compute 三条流水线从一开始就强耦合

## 6. 各组件逐项回答：利用率如何，空泡在哪里，空泡有多大

### 6.1 Reader

decode：

- 短序列：
  - reader 利用率以 `issue` 为主
  - 主任务是主动发 K/V 读请求
- 中长序列：
  - `reserve` 成为 reader 主导项
  - 而且 source-level 证明主导项是 `K reserve`，不是 `V reserve`
- 空泡位置：
  - 从 `decode_4k` 开始，空泡主要在 `K` 路 reserve / downstream turnover
- 空泡规模：
  - `reader reserve share` 从 `0.67%` 一路增到 `61.54%`

prefill：

- reader 利用率一直以 `wait` 为主
- 空泡位置不在 `reserve`，而在等待 in-flight read completion
- 空泡规模：
  - `reader wait share` 始终维持在 `66% ~ 70%`

### 6.2 Writer

decode：

- writer 从短序列开始就很“忙”，但忙的不是 issue，而是 `cb_wait`
- 空泡位置：
  - source-level 证明主要卡在 `sender_cb_wait` 和 `tree_child_wait`
  - 不在 `root`，也不在 `output_gather`
- 空泡规模：
  - `writer cb_wait share` 从 `89.42%` 升到 `99.77%`

prefill：

- writer 几乎全程都是 `cb_wait` 主导
- 空泡位置：
  - 主要是等待 compute / reduction 结果产出
- 空泡规模：
  - `writer cb_wait share` 从 `91.26%` 升到 `99.62%`

### 6.3 Compute

decode：

- `TRISC share` 始终接近满窗，说明 compute 没有早退休
- 但 compute bubble 里：
  - `wait-front` 始终最大
  - `reserve-back` 随序列变长不断抬升
- 空泡规模：
  - `wait-front` 从 `61.2 us` 增到 `829.7 us`
  - `reserve-back` 从 `8.8 us` 增到 `340.3 us`

prefill：

- compute thread 也几乎满窗
- 但 `PM FPU util` 只有 `10% ~ 19%`
- 空泡位置：
  - 主要是 `wait-front`
- 空泡规模：
  - `wait-front` 从 `8.67 ms` 累积到 `1029.32 ms`
  - `reserve-back` 也增长，但始终是次要项

## 7. 可以直接指导优化的结论

### 7.1 Decode 的优化优先级

- 第一优先级不是继续单独榨 compute，而是解决 reader 的 `K reserve`
- 第二优先级是拆解 writer 内部 `sender/tree` 两段等待的真正来源
- `decode_4k` 是最值得盯的迁移点，因为它是 reader 从 `issue` 主导切到 `reserve` 主导的起点

### 7.2 Prefill 的优化优先级

- 不应再把 prefill 简化成“compute-bound”
- 更合适的工程表述是：
  - 这是一个 reader / writer / compute 强耦合饱和流水线
- 优先级应该放在：
  - reader 的 read completion overlap
  - writer 对 compute / reduction 产出的等待
  - compute 的 `wait-front` 来源

### 7.3 最容易误判的点

- `TRISC share` 高，不等于 FPU util 高
- `bubble density vs kernel > 1x`，不等于 wall time 上空泡超过 `100%`
- writer `cb_wait` 高，不等于“最后写出带宽不够”；在 decode 上它主要是 sender / tree reduction 链路等待

## 8. 现在还不能直接下的结论

- decode 的 direct PM util 仍不可用，因此不能把 decode 说成“FPU 利用率是 xx%”
- decode 的 NOC / DRAM direct util 仍不可用，因此不能把长序列 backpressure 精确分账到 NoC 或 DRAM
- prefill 的 source-level K/V attribution 目前还没有像 decode 那样跑通，因此 prefill 现阶段仍主要依赖 stage-level 与 direct PM 联合解读

## 9. 最终结论

如果把当前结果压缩成一句最准确的话，应该是：

- `decode`：从短序列的 compute-critical，迁移到中段 writer close-to-critical，再迁移到长序列的 `K reserve + sender/tree wait + compute bubble` 强耦合饱和。
- `prefill`：从起点开始就是 `reader wait + writer cb_wait + compute wait-front` 的强耦合流水线；direct PM util 虽然随序列增长上升，但主 wall time 仍不是由算术饱和主导。

因此，当前 FlashMLA 在 WH 上最值得优先优化的，并不是“再多榨一点纯算力”，而是：

- decode 的 `K` 路 turnover / downstream backpressure
- decode writer 的 sender/tree reduction 等待
- prefill 的 reader wait / writer cb_wait / compute wait-front 这条耦合链

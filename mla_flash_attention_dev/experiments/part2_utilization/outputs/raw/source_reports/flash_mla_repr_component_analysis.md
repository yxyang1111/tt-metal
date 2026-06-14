# FlashMLA 代表点利用率与空泡补充分析

## 1. 这轮补充到底补到了什么

这轮补充的关键不是重新跑全 sweep，而是把先前卡住的 host-side Tracy capture race 压住，然后用一组代表点把更细的 component-level 结论补齐。

当前实际补到的点是：

- `decode_1k`：来自 `flash_mla_pm_bubble_probe_sync_smoke`
- `decode_4k`：来自 `flash_mla_pm_bubble_probe_repr_sync`
- `decode_32k`：来自 `flash_mla_pm_bubble_probe_repr_sync`
- `prefill_4k`：来自 `flash_mla_pm_bubble_probe_repr_sync`

这轮运行统一采用：

- `--sync-host-device`
- `--enable-sum-profiling`
- `device_compute_cb_wait_front`
- `device_compute_cb_reserve_back`

结果上有三点最重要：

1. `--sync-host-device` 已经证明能稳定解决之前的 host trace 落地问题。
2. `decode` 的 `compute bubble` 和 `source marker` 现在已经能稳定读出来。
3. `decode` 的 direct `PM/NOC/DRAM` 利用率仍不可用，但 `prefill_4k` 已经拿到了**非 placeholder** 的 PM triplet / `PM FPU UTIL (%)`。

## 2. 代表点总表

| case | 分类 | kernel us | BRISC share | NCRISC share | max TRISC share | wait-front us | reserve-back us | bubble density vs kernel | PM FPU util | 判读 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `decode_1k` | `compute_starved_by_input` | 76.203 | 83.8% | 39.6% | 99.7% | 186.672 | 30.264 | 2.847x | `0.001%` | bubble 可用，PM 仍 placeholder |
| `decode_4k` | `writer_close_to_critical_path` | 176.475 | 93.0% | 62.9% | 99.9% | 326.990 | 93.082 | 2.380x | `0.001%` | bubble 可用，PM 仍 placeholder |
| `decode_32k` | `reader_writer_saturated` | 1111.539 | 98.9% | 94.3% | 100.0% | 828.256 | 340.447 | 1.051x | `missing/0` | bubble 可用，PM 仍 placeholder |
| `prefill_4k` | `reader_writer_saturated` | 55516.068 | 100.0% | 100.0% | 100.0% | 1029320.475 | 82474.895 | 20.027x | `19.427%` | PM 可读，bubble 可读 |

说明：

- `bubble density vs kernel` 仍然必须按 `accumulated stall density` 解读，不能当 wall-time 占比。
- `decode` 的 `PM IDEAL / PM COMPUTE / PM BANDWIDTH` 仍固定在 `1.0 ns`，所以 `decode` 这层 direct PM 还是不能用于真实利用率结论。
- `prefill_4k` 则不同，它的 `PM IDEAL / PM COMPUTE / PM BANDWIDTH` 已经是有效数值。

## 3. Decode：各组件利用率与空泡来源

### 3.1 线程窗口层

- `decode_1k`：`TRISC` 已接近整个 kernel window，`BRISC` 约 `83.8%`，`NCRISC` 约 `39.6%`。这说明短序列点仍然是 compute 最后退休，但 writer 已经不轻。
- `decode_4k`：`BRISC share` 提升到 `93.0%`，`NCRISC share` 也到 `62.9%`。此时 writer 已明显贴边，reader 也开始进入高压区。
- `decode_32k`：`BRISC/NCRISC/TRISC` 基本同时贴满窗口，说明这是典型的 reader/writer/compute 三方强耦合饱和点。

### 3.2 compute 空泡有多大

| case | wait-front share in bubble | reserve-back share in bubble | 解读 |
|---|---:|---:|---|
| `decode_1k` | 86.0% | 14.0% | 仍以 input-side wait 为主 |
| `decode_4k` | 77.8% | 22.2% | output/backpressure 成分开始抬升 |
| `decode_32k` | 70.9% | 29.1% | reserve-back 已经变成不可忽略的大项 |

这张表和已有 full sweep 结论是一致的：

- 短 decode 时，compute bubble 主要还是 `wait-front`
- 随着序列变长，`reserve-back` 会持续抬升
- 到长序列时，compute 看到的 stall 已经明显带有 output/backpressure 性质

### 3.3 reader：到底是 K 还是 V 在制造空泡

#### `decode_1k`

- reader stage 仍是 `issue` 主导：`issue=78.93%`
- source-level 上 `K` 路总计约 `72.51%`，`V` 路约 `27.48%`
- 主导项是 `k_issue=56.92%`，其次是 `v_issue=25.36%`

含义：

- 短序列点 reader 主要还在主动发 K/V 读请求
- 此时的 reader stall 还不是 reserve/backpressure 主导

#### `decode_4k`

- reader stage 已明显转向 reserve/issue 混合：`reserve=31.96%`，`issue=55.80%`
- source-level 上 `K` 路总计约 `80.95%`，`V` 路约 `19.05%`
- 其中最关键的新量是 `k_reserve=31.64%`

含义：

- `decode_4k` 的 reader 反压主要来自 `K` 路，而不是 `V` 路
- 这说明长序列下最先卡住 reader turnover 的是 `K` 相关 buffering / consumption path

#### `decode_32k`

- reader stage 已变成 `reserve` 主导：`reserve=57.58%`
- source-level 上 `K` 路总计约 `84.62%`，`V` 路约 `15.38%`
- 主导项已经是 `k_reserve=54.82%`

含义：

- 长序列 decode 的 reader 空泡已经非常明确地集中在 `K reserve`
- `V reserve=2.80%` 明显更小，说明真正主导 long-seq backpressure 的不是 `V` 路

把三点连起来看，reader 端的迁移可以直接写成：

- `decode_1k`：`K/V issue` 主导
- `decode_4k`：`K issue + K reserve` 并存
- `decode_32k`：`K reserve` 成为 reader 内部第一大项

## 4. Decode：writer 的等待到底卡在哪一段

writer source-level 的结论比预期更清楚，而且在三个代表点上非常稳定：

| case | sender_cb_wait | tree_child_wait | root_cb_wait | output_gather_wait |
|---|---:|---:|---:|---:|
| `decode_1k` | 48.58% | 51.42% | 0.00% | 0.00% |
| `decode_4k` | 49.59% | 50.41% | 0.00% | 0.00% |
| `decode_32k` | 49.95% | 50.05% | 0.00% | 0.00% |

这说明：

- writer 的主等待并不在 final output gather
- 也不在 root 端等待最终 `cb_out`
- 真正占满 writer wait 的，是：
  - `sender_cb_wait`
  - `tree_child_wait`

换句话说，writer 这条链路的关键空泡不是“最后写出没带宽”，而是：

- sender 在等本地 partial output ready
- tree/root reduction 在等 child partial result 到齐

这和已有 `writer cb_wait` 结论高度一致，只是现在第一次从 source-level 上把它直接拆开了。

## 5. Prefill：第一次拿到可直接解释的 PM 利用率

`prefill_4k` 是这轮最有价值的新点，因为它第一次让 PM triplet 和 `PM FPU UTIL (%)` 变成了可直接解读的数值：

- `kernel wall = 55.516 ms`
- `PM IDEAL = 10.785 ms`
- `PM COMPUTE = 10.785 ms`
- `PM BANDWIDTH = 0.545 ms`
- `PM FPU UTIL = 19.427%`

这里最重要的不是 `19.4%` 这个数字本身，而是它和 wall time 的差：

- `PM COMPUTE / kernel wall` 只有约 `19.4%`
- 仍有约 `44.731 ms`
- 也就是约 `80.6%` 的 kernel wall time 不在 direct PM compute 里

再把它和已有 stage-level / bubble 数据放在一起看：

- `reader wait = 66.25%`
- `writer cb_wait = 99.62%`
- `wait-front bubble share = 92.6%`
- `reserve-back bubble share = 7.4%`
- `bubble density vs kernel = 20.027x`

因此 `prefill_4k` 的最合理结论是：

- 算术管线本身没有被真正打满
- 真正把 wall time 拉长的，仍然是强耦合流水线里的等待
- 而这份等待主要体现为：
  - reader 等 in-flight read completion
  - writer 等 compute/reduction 产出
  - compute 端看到的大量 `wait-front`

## 6. 现在哪些实验完成了，哪些还没完成

### 已完成

- `decode` representative point 的 host capture 已通过 `--sync-host-device` 跑通
- `decode` representative point 的 `compute bubble` 已补齐
- `decode` representative point 的 reader / writer source-level breakdown 已拿到
- `decode_16k` 的 source-level 迁移点已补到
- `prefill_256 -> prefill_4k` 的 PM triplet / `PM FPU UTIL` 已补齐

### 仍未完成

- `decode` direct `PM IDEAL / PM COMPUTE / PM BANDWIDTH / PM FPU UTIL` 仍是 placeholder
- `decode` direct `NOC / MULTICAST NOC / DRAM BW util` 仍不可用
- `--collect-noc-traces` 路线当前会触发 `Invalid NoC transfer type`，暂不适合放大
- `prefill` 目前 direct PM util 已补到 `4k`，但更长序列或 NOC/DRAM direct util 仍未补齐

## 7. 当前最稳妥的工程结论

- `decode` 的 direct PM 利用率依然不能作为主判断依据，但 component-level stall attribution 现在已经足够清楚。
- `decode` 长序列的 reader 空泡主要来自 `K reserve`，不是 `V reserve`。
- `decode` writer 的等待主要均分在 `sender local ready` 和 `tree child arrival` 两段，不是 final gather。
- `prefill_256 -> prefill_4k` 都显示出：即使 direct PM util 随序列上升，主 wall-time 仍然主要由流水线等待决定，而不是单纯算力不足。

## 8. 继续补跑后的新增结果

后面又继续补了一轮同步 sweep：

- `decode_16k`
- `prefill_256`
- `prefill_512`
- `prefill_1k`
- `prefill_2k`

对应输出目录在：

- `mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_pm_bubble_probe_sync_extra/`

### 8.1 `decode_16k` 把 `4k -> 32k` 的迁移点补清楚了

`decode_16k` 的代表值如下：

- `kernel = 575.215 us`
- `BRISC share = 97.9%`
- `NCRISC share = 88.8%`
- `wait-front share in bubble = 71.9%`
- `reserve-back share in bubble = 28.1%`

reader stage-level：

- `reserve = 55.48%`
- `issue = 39.86%`

reader source-level：

- `k_reserve = 52.79%`
- `k_issue = 27.39%`
- `v_issue = 12.53%`
- `v_reserve = 2.77%`

writer source-level：

- `sender_cb_wait = 49.89%`
- `tree_child_wait = 50.11%`
- `root_cb_wait = 0%`
- `output_gather_wait = 0%`

这说明：

- `decode_16k` 已经非常接近 `decode_32k` 的 long-seq 形态，而不是介于 `4k` 和 `32k` 中间的模糊状态。
- reader 端已经明确由 `K reserve` 主导，`V reserve` 仍然很小。
- writer 端仍然是 `sender/tree` 两段几乎五五开，主等待仍不在 final gather。

因此 decode 的 source-level 迁移现在可以更完整地写成：

- `decode_1k`：`K/V issue` 主导
- `decode_4k`：`K issue + K reserve` 并存
- `decode_16k`：`K reserve` 已成为 reader 内部主导项
- `decode_32k`：`K reserve` 继续保持主导并进一步固化

### 8.2 `prefill` 的 direct PM 利用率已经从 `256 -> 4k` 连起来了

这轮最大的新增价值是把 `prefill_256 -> prefill_4k` 的 PM 曲线补齐了：

| case | kernel ms | PM COMPUTE ms | PM BANDWIDTH ms | PM FPU util | wait-front share in bubble | reserve-back share in bubble |
|---|---:|---:|---:|---:|---:|---:|
| `prefill_256` | 0.413 | 0.042 | 0.034 | 10.20% | 96.0% | 4.0% |
| `prefill_512` | 1.212 | 0.169 | 0.068 | 13.90% | 94.6% | 5.4% |
| `prefill_1k` | 4.021 | 0.674 | 0.136 | 16.76% | 93.6% | 6.4% |
| `prefill_2k` | 14.595 | 2.696 | 0.273 | 18.48% | 92.9% | 7.1% |
| `prefill_4k` | 55.516 | 10.785 | 0.545 | 19.43% | 92.6% | 7.4% |

这张表说明两件事：

- `PM FPU UTIL` 会随着序列长度增大从 `10.2%` 升到 `19.4%`，但始终远低于真正意义上的算术饱和。
- compute bubble 内部始终以 `wait-front` 为主，且 share 只从 `96.0%` 缓慢降到 `92.6%`；`reserve-back` 虽然会上升，但仍是次要项。

把它再和 stage-level 结果放在一起看：

- reader 的 `wait share` 只是在 `70.31% -> 68.65% -> 67.31% -> 66.63% -> 66.25%` 间缓慢下降
- writer 的 `cb_wait share` 则从 `91.50%` 持续升到 `99.62%`
- `bubble density vs kernel` 在全段都维持在大约 `20x ~ 22x`

所以 `prefill` 这条路径现在可以更精确地写成：

- direct PM 利用率是**可测且在上升**的
- 但即使到 `prefill_4k`，`PM COMPUTE / kernel wall` 也只有约 `19.4%`
- 主 wall time 依然被 `reader wait + writer cb_wait + compute wait-front` 主导

### 8.3 对“利用率”和“空泡”的最新统一口径

到这一步，结论可以收敛成：

- `decode`：stage-level、compute bubble、source-level stall attribution 都已经够强；唯独 direct PM/NOC/DRAM util 仍然不可信。
- `prefill`：不仅 stage-level/bubble 可信，direct PM util 也已经在 `256 -> 4k` 这整段上跑通了。
- 这意味着后续如果还要继续“详细分析各个组件的利用率如何”，优先级应该分开：
  - `decode`：继续深挖 source-level / pipeline-coupling
  - `prefill`：可以正式把 PM util 纳入主分析，而不只看 stage proxy

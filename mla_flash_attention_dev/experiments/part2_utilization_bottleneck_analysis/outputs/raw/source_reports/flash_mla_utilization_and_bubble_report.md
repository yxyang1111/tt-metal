# FlashMLA 现有结果利用率与空泡分析报告

## 1. 结论摘要

- 现有结果里，**最可信的“利用率”结论不是 PM 百分比**，而是 `BRISC/NCRISC/TRISC` 相对 `kernel window` 的线程窗口占比，以及 `reader reserve` / `writer cb_wait` 这类阶段级 stall share。
- 对 `decode` 来说，阶段迁移非常清楚：`256~1k` 仍是 compute 最后退休，`2k~8k` 转成 writer 贴边，`16k` 开始 reader 也贴边，`32k` 进入 reader/writer 一起饱和。
- 对 `decode` 来说，真正的主空泡不是“compute 自己空转很多”，而是 **reader downstream backpressure + writer output-availability wait**。reader 的 `reserve share` 会从 `0.7%` 抬升到 `61.5%`，writer 的 `cb_wait share` 会从 `89.4%` 抬升到 `99.8%`。
- guarded `decode_safe` rerun 已经把 `DEVICE COMPUTE CB WAIT FRONT / RESERVE BACK` 的 `8/8` 个点补齐。bubble 内部始终以 `wait-front` 为主，但 `reserve-back` 占比会从 `12.6%` 升到 `29.1%`。
- 对 `prefill` 来说，结果和 decode 明显不同：它从最短点起就已经是 `reader + writer + compute` 强耦合饱和流水线，而不是短 decode 那种 compute-critical 形态。
- 截至当前，**不能**从现有结果里直接得出可信的 `PM FPU UTIL (%)`、`NOC UTIL (%)`、`MULTICAST NOC UTIL (%)`、`DRAM BW UTIL (%)`。当前 guarded rerun 里的这些列仍是 placeholder 或 missing。

## 2. 数据来源与解释口径

本报告只基于当前已经稳定拿到的结果：

- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_results.json`
- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_decode_summary.csv`
- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_summary.md`

为避免误读，这里明确区分三类指标：

1. 线程窗口利用率代理量
   指 `BRISC/NCRISC/TRISC` 相对 `DEVICE KERNEL DURATION` 的占比。它能说明哪条线程正在逼近 critical path，但**不是**直接的 FPU/NOC/DRAM 百分比利用率。
2. 阶段级空泡/反压代理量
   指 `reader reserve share`、`reader wait share`、`writer cb_wait share` 这类 stage-level share。它们能说明 stall 在流水线的哪一段累积。
3. compute bubble counter
   指 `DEVICE COMPUTE CB WAIT FRONT [ns]`、`DEVICE COMPUTE CB RESERVE BACK [ns]`。当前应按 **accumulated stall counter density** 解读，而不是按 wall-time slice 解读，因为短序列点上它们的和会超过单次 `kernel wall time`。

## 3. Decode：利用率代理量与空泡联合分析

### 3.1 总表

| case | 分类 | BRISC share | NCRISC share | reader reserve share | writer cb_wait share | bubble density vs kernel | wait-front share in bubble | reserve-back share in bubble |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `decode_256` | `compute_on_critical_path` | 70.3% | 33.4% | 0.7% | 89.4% | 1.66x | 87.4% | 12.6% |
| `decode_512` | `compute_on_critical_path` | 79.1% | 31.4% | 0.7% | 90.2% | 3.06x | 89.1% | 10.9% |
| `decode_1k` | `compute_on_critical_path` | 83.5% | 37.1% | 0.7% | 94.1% | 2.86x | 86.1% | 13.9% |
| `decode_2k` | `writer_close_to_critical_path` | 88.5% | 42.6% | 0.7% | 96.3% | 2.60x | 82.0% | 18.0% |
| `decode_4k` | `writer_close_to_critical_path` | 92.9% | 62.6% | 39.8% | 98.1% | 2.40x | 78.1% | 21.9% |
| `decode_8k` | `writer_close_to_critical_path` | 96.0% | 78.7% | 52.0% | 98.9% | 2.26x | 74.8% | 25.2% |
| `decode_16k` | `reader_close_to_critical_path` | 97.8% | 88.7% | 59.7% | 99.6% | 1.07x | 72.0% | 28.0% |
| `decode_32k` | `reader_writer_saturated` | 98.9% | 94.3% | 61.5% | 99.8% | 1.05x | 70.9% | 29.1% |

### 3.2 Decode 的阶段迁移

- `256~1k`：compute 仍然是最后退休的线程，但 `BRISC` 已经不轻。writer 的 `cb_wait share` 从最短点就已经是 `89.4%`，说明 writer 很早就不是 issue-bound，而是在等上游结果 ready。
- `2k~8k`：writer 进入贴边区。这里最重要的现象不是“writer 带宽打满”，而是 `BRISC share` 已经到 `88.5% -> 96.0%`，且 writer 内部几乎完全被 `cb_wait` 主导。
- `4k`：这是 decode 最关键的迁移点。reader 的 `reserve share` 从 `0.7%` 直接跳到 `39.8%`，表明 reader 已从 issue-dominant 转向 backpressure-dominant。
- `16k~32k`：reader 也进入贴边区，`NCRISC share` 达到 `88.7% -> 94.3%`。到 `32k`，reader 和 writer 都几乎把窗口填满，已经不能再叫“纯 compute-bound”。

### 3.3 Decode 的空泡到底在哪里

- writer 侧：`cb_wait share` 从 `89.4%` 一路升到 `99.8%`，说明 writer 长时间是在等 compute/reduction 把结果推到输出 CB，而不是在等写 issue 自身完成。
- reader 侧：`reserve share` 在 `decode_4k` 之后快速抬升，到 `decode_32k` 达 `61.5%`。这表示 reader 端越来越多时间耗在 `cb_reserve_back()`，也就是典型 downstream backpressure。
- compute 侧：bubble counter 表明 `wait-front` 始终是 bubble 内部主项，但 `reserve-back` 占比从 `12.6%` 升到 `29.1%`。这和 reader 端 `reserve share` 的升高是同向的，说明长序列下 compute 侧看到的 stall 也越来越带有 output/backpressure 成分。

### 3.4 如何正确解释 `bubble density vs kernel`

`bubble density vs kernel` 在短序列上会高于 `1x`，例如：

- `decode_512 = 3.06x`
- `decode_1k = 2.86x`
- `decode_4k = 2.40x`

这**不**表示 compute 真正 wall-time 上有 `306%` 的空泡，而是说明：

- `wait-front` / `reserve-back` 是跨 compute 线程累计的 stall counter
- 它们更适合用来比较 bubble 的**内部组成变化**和**相对密度变化**
- 不适合直接拿来做 “kernel 时间里有多少百分比在空泡” 这样的 wall-time 解释

因此，当前 decode 的最稳妥表述是：

- 短序列时 compute 仍然最后退休
- 中长序列时 writer 和 reader 都在贴边
- 真正锁住长序列曲线的，是 reader reserve 与 writer cb_wait 所代表的传播式反压，而不是单一 compute 算力不够

## 4. Prefill：现有结果里的利用率代理量

Prefill 当前没有像 decode 那样补齐 compute bubble counter，但 stage-level 利用率代理量已经足够清楚：

| case | 分类 | NCRISC share | BRISC share | reader 主导项 | writer 主导项 |
|---|---|---:|---:|---|---|
| `prefill_256` | `reader_writer_saturated` | 97.1% | 99.9% | `wait 70.3%` | `cb_wait 91.3%` |
| `prefill_512` | `reader_writer_saturated` | 99.0% | 100.0% | `wait 68.7%` | `cb_wait 95.6%` |
| `prefill_1k` | `reader_writer_saturated` | 99.7% | 100.0% | `wait 67.2%` | `cb_wait 98.3%` |
| `prefill_2k` | `reader_writer_saturated` | 99.9% | 100.0% | `wait 66.5%` | `cb_wait 99.2%` |
| `prefill_4k` | `reader_writer_saturated` | 100.0% | 100.0% | `wait 66.1%` | `cb_wait 99.6%` |

这说明：

- prefill 从最短点起就已经不是 “compute 最后退休” 的形态。
- `NCRISC` 基本贴满整个窗口，但 reader 内部主导项始终是 `wait`，而不是 decode 那种后期转成 `reserve`。
- `BRISC` 几乎从一开始就贴满窗口，且 writer 内部几乎始终是 `cb_wait`。

所以 prefill 的最准确表述是：

- 它从起点开始就是一个 `reader + writer + compute` 强耦合的饱和流水线
- reader 更像在等 in-flight read completion
- writer 更像在等 compute/reduction 产出
- 它和 decode 的差异，不是“谁更吃算力”，而是 stall 的主导位置不同

## 5. 当前不能从现有结果直接得出的结论

下面这些结论，目前**不能**直接从现有结果里严肃给出：

- `PM FPU UTIL (%)` 的真实百分比利用率
- `NOC UTIL (%)`、`MULTICAST NOC UTIL (%)`、`DRAM BW UTIL (%)` 的真实利用率
- compute 自己在 wall-time 上有多少比例是在等输入、多少比例是在等输出
- `K/V reserve`、`sender/root/tree/output wait` 这些 source-level attribution 的直接归因

原因很简单：

- 当前 guarded rerun 中，`PM IDEAL / PM COMPUTE / PM BANDWIDTH` 固定为 `1.0 ns`
- `PM FPU UTIL (%)` 只有 `0.000 ~ 0.002`
- `NOC/MULTICAST/DRAM BW UTIL` 全部缺失

因此，现阶段应统一采用如下口径：

- `decode compute bubble`：可分析
- `stage-level utilization/stall proxy`：可分析
- `direct PM/NOC/DRAM utilization`：不可分析

## 6. 对当前 WH 实现的工程含义

- 对 decode，不应该再把优化重点放在“单纯再榨一点 compute”上。更值得优先处理的是 reader 的 buffer turnover / downstream backpressure，以及 writer 的 output-availability wait。
- `decode_4k` 是最值得盯的迁移点，因为它同时出现了 reader reserve 的显著抬升和 writer 的持续贴边。
- `decode_16k ~ 32k` 已经进入 reader/writer 双贴边区，后续如果还要继续优化长序列，单纯扩大 active-core 并不会解决根因。
- 对 prefill，优先级也不应放在“先证明 compute 不够”，而应直接把它视作强耦合饱和流水线来处理。

## 7. 一句话总结

基于当前**可用且可信**的结果，FlashMLA 在 WH 上的现状不是“纯 compute-bound”，而是：

- `decode` 短序列偏 compute-critical
- `decode` 中长序列是 reader/writer/compute 强耦合贴边流水线
- `prefill` 从起点开始就是 reader/writer/compute 同步饱和
- 现阶段能可信分析的是阶段级利用率代理量和 decode compute bubble counter，不能可信分析的是 direct PM/NOC/DRAM 百分比利用率

## 8. 最新同步 Rerun 补充

后面又补了一轮带 `--sync-host-device` 的代表点 rerun，用来解决之前的 host trace 落地问题，并补更细的 component-level attribution。

这轮补到的代表点是：

- `decode_1k`
- `decode_4k`
- `decode_32k`
- `prefill_4k`

补充结论如下：

- `--sync-host-device` 已经证明能稳定解决当前机器上的 Tracy host capture race；`decode_4k / decode_32k / prefill_4k` 都成功生成了 `ops_perf_results_*.csv`。
- `decode` 的 direct PM 仍然不可用：`PM IDEAL / PM COMPUTE / PM BANDWIDTH` 仍是 `1.0 ns`，`PM FPU UTIL` 仍是 `0.001` 或缺失，`NOC/DRAM util` 仍为空。
- 但 `decode` 的 source-level stall attribution 已经补出来了：
  - `decode_1k` 时 reader 仍以 `K/V issue` 为主，`K` 路约占 `72.5%`
  - `decode_4k` 时 `k_reserve` 已升到 `31.64%`
  - `decode_32k` 时 `k_reserve` 已升到 `54.82%`
- 这说明长序列 decode 的 reader backpressure 现在可以更直接地归因到 `K` 路，而不是笼统地说成“reader reserve”。
- writer source-level 也很稳定：`sender_cb_wait` 和 `tree_child_wait` 基本各占一半，`root_cb_wait / output_gather_wait` 约为 `0%`。也就是说，writer 主等待不在 final gather，而在 sender 与 tree reduction 链路内部。
- `prefill_4k` 则第一次拿到了可直接解释的 PM 值：
  - `kernel wall = 55.516 ms`
  - `PM COMPUTE = 10.785 ms`
  - `PM BANDWIDTH = 0.545 ms`
  - `PM FPU UTIL = 19.427%`
- 这说明 `prefill_4k` 的算术管线本身只解释了约 `19.4%` 的 kernel wall time，剩余约 `80.6%` 仍是强耦合流水线里的等待，与 `reader wait + writer cb_wait + compute wait-front` 的结论一致。
- 这一轮也单点试过 `--collect-noc-traces`，但仍会触发 `Invalid NoC transfer type`，所以当前不适合直接放大到 NOC full sweep。

更完整的代表点补充分析已经单独放到：

- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_repr_sync/flash_mla_repr_component_analysis.md`

## 9. 继续补跑后的新增结论

后面又继续补了一轮同步 sweep：

- `decode_16k`
- `prefill_256`
- `prefill_512`
- `prefill_1k`
- `prefill_2k`

### 9.1 `decode_16k` 说明 long-seq decode 的转折已经固定

`decode_16k` 的结果是：

- `BRISC share = 97.9%`
- `NCRISC share = 88.8%`
- `wait-front share in bubble = 71.9%`
- `reserve-back share in bubble = 28.1%`
- reader `reserve = 55.48%`
- `k_reserve = 52.79%`
- `v_reserve = 2.77%`

writer source-level 仍然几乎完全固定在：

- `sender_cb_wait = 49.89%`
- `tree_child_wait = 50.11%`

这意味着 `decode_16k` 已经和 `decode_32k` 同型：

- reader 空泡主导项已经稳定落在 `K reserve`
- writer 主等待仍不在 final gather，而在 sender/tree reduction 内部

### 9.2 `prefill` 的 direct PM util 已经从 `256 -> 4k` 跑通

| case | kernel ms | PM COMPUTE ms | PM BANDWIDTH ms | PM FPU util |
|---|---:|---:|---:|---:|
| `prefill_256` | 0.413 | 0.042 | 0.034 | 10.20% |
| `prefill_512` | 1.212 | 0.169 | 0.068 | 13.90% |
| `prefill_1k` | 4.021 | 0.674 | 0.136 | 16.76% |
| `prefill_2k` | 14.595 | 2.696 | 0.273 | 18.48% |
| `prefill_4k` | 55.516 | 10.785 | 0.545 | 19.43% |

同时，bubble 的主成分也保持稳定：

- `wait-front share` 约从 `96.0%` 缓慢降到 `92.6%`
- `reserve-back share` 约从 `4.0%` 升到 `7.4%`
- reader `wait share` 约从 `70.31%` 降到 `66.25%`
- writer `cb_wait share` 约从 `91.50%` 升到 `99.62%`

因此现在可以更明确地下结论：

- `prefill` 的 PM 利用率是可测的，而且确实会随序列增长而上升
- 但即使到 `prefill_4k`，direct compute 也只解释了约 `19.4%` 的 kernel wall time
- 主 wall-time 依然由 `reader wait + writer cb_wait + compute wait-front` 主导

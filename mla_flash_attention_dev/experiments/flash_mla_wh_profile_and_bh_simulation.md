# Flash-MLA WH Profiling And BH Simulation

## 0. 执行摘要

- `decode` 在 WH 上呈现很清楚的阶段迁移：`256~1k` 还是 compute-critical，`2k~8k` 先变成 writer 贴边，`16k` 开始 reader 也贴边，`32k` 进入 reader/writer 一起饱和
- `decode` 的真正新结论不是“writer 很忙”，而是 writer 几乎一直主要卡在 `cb_wait_front`；与此同时 reader 会从短序列的 `issue` 主导，迁移到长序列的 `cb_reserve_back` 主导
- `prefill` 和 decode 不一样，它几乎从最短测点开始就是 `reader + writer + compute` 强耦合饱和流水线：reader 主要在 `wait`，writer 主要在 `cb_wait`
- `A-BH` 的一阶模型给出 `compute/dram crossover ~= 23 active cores`；但把 WH detailed 的 `reader reserve + writer cb_wait` 回灌进去后，二阶经验模型显示它会比一阶理想值慢约 `1.62x ~ 2.53x`，并且 `compute` 大约在 `4~10 cores` 就退出 critical path，随后很快进入 reader/writer floor 主导的平台区
- `B-BH` 相对经验二阶 `A-BH` 在 full sweep 上都更快，优势约 `1.28x -> 5.60x`；相对一阶 `A-BH` 则大约从 `1k` 开始反超，到 `32k` 扩大到约 `3.1x`，主瓶颈转成 `K multicast`
- guarded `decode_safe` rerun 已经把 `DEVICE COMPUTE CB WAIT FRONT / RESERVE BACK` 的 `8` 个序列点全部补齐；bubble 内部一直以 `wait_front` 为主，但 `reserve_back` 占比会从 `12.6%` 抬升到 `29.1%`。不过这轮 `PM IDEAL/COMPUTE/BANDWIDTH` 仍固定在 `1.0 ns`，`NOC/DRAM util` 仍为空，因此应解读成 `bubble-complete / PM-incomplete`

## 1. 目标

本次工作现在分四部分：

- 在现有 Wormhole 硬件上，对当前可运行的 Flash-MLA decode 路径做真实 profile，并用 `prefill` detailed sweep 作为对照
- 对 Blackhole 做一版一阶模拟分析，评估当前实现 A 直接搬到 BH 的结果
- 对 Blackhole 做一版一阶模拟分析，评估原生 Flash-MLA 实现 B 在 BH 上的理论下界
- 对当前实现 A 再补一版经验二阶模型，把 WH detailed 观测到的 pipeline coupling 项回灌到 BH

这里的 WH 主实测对象仍然是当前仓库里可直接运行的 `paged_flash_multi_latent_attention_decode`，也就是当前 production decode 路径；
而 `profile_flash_mla_wh_detailed.py` 又额外补了 `chunked_flash_mla_prefill` 的 full sweep，对照 decode 的阶段迁移。

## 2. 产物位置

- 主用 full-sweep detailed profiling 脚本：`mla_flash_attention_dev/experiments/profile_flash_mla_wh_detailed.py`
- 旧版 3-point profiling / 一阶 A-BH/B-BH 对照脚本：`mla_flash_attention_dev/experiments/profile_flash_mla_wh.py`
- 可视化脚本：`mla_flash_attention_dev/experiments/render_flash_mla_profile_visuals.py`
- 旧版 3-point 原始结果 JSON：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh/flash_mla_wh_profile_results.json`
- 旧版 3-point 自动生成摘要：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh/flash_mla_wh_profile_report.md`
- full-sweep detailed 原始结果 JSON：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_results.json`
- full-sweep detailed 自动生成摘要：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_report.md`（现已包含经验校准的 `A-BH` 二阶模型）
- full-sweep detailed measurement DB：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_measurement_db.json`
- PM 利用率 / compute 空泡最小 rerun 方案：`mla_flash_attention_dev/experiments/flash_mla_pm_and_bubble_min_rerun_plan.md`
- PM 利用率 / compute 空泡 runner：`mla_flash_attention_dev/experiments/run_flash_mla_pm_bubble_probe.py`
- guarded decode PM/bubble 可视化脚本：`mla_flash_attention_dev/experiments/render_flash_mla_pm_bubble_visuals.py`
- guarded decode PM/bubble 汇总 CSV：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_decode_summary.csv`
- guarded decode PM/bubble 摘要：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_summary.md`
- guarded decode PM/bubble dashboard：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_dashboard.html`
- guarded decode PM/bubble SVG 图目录：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/visuals/`
- 同步代表点 PM/bubble 输出目录：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_repr_sync/`
- 同步代表点 component-level 补充分析：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_repr_sync/flash_mla_repr_component_analysis.md`
- 同步补充 sweep 输出目录：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_sync_extra/`
- 可视化摘要：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_visual_summary.md`
- 完整性与利用率说明：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_profile_completeness_and_utilization.md`
- 现有结果利用率与空泡分析报告：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_utilization_and_bubble_report.md`
- 可视化 dashboard：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_visual_dashboard.html`
- SVG 图目录：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/visuals/`

## 3. 运行方式

如果要重现本文里现在采用的 full sweep 结果，直接运行：

```bash
python3 mla_flash_attention_dev/experiments/profile_flash_mla_wh_detailed.py
```

这个 detailed 脚本现在会做两件事：

1. 在 WH 上跑真实 Tracy/device-profiler，抓 `BRISC/NCRISC/TRISC*` 的 device kernel duration
2. 自动整理 detailed summary，并额外生成一版由 WH 观测量校准的 `A-BH` 二阶经验模型

如果你只想做旧版的 3-point quick smoke / 对照抽样，可以继续运行：

```bash
python3 mla_flash_attention_dev/experiments/profile_flash_mla_wh.py --cases decode_1k decode_4k decode_32k
```

旧版 simple script 里的 3-point 一阶 `A-BH / B-BH` 对照，仍然由 `profile_flash_mla_wh.py` 提供；
而本文现在采用的 full-sweep `A-BH / B-BH` 表，则是沿用同一组一阶公式按 `decode_256 -> decode_32k` 扩展整理。

如果你只想基于现有 JSON 重新生成整理后的图和可视化摘要，可以直接运行：

```bash
python3 mla_flash_attention_dev/experiments/render_flash_mla_profile_visuals.py
```

如果你只想重生成这次 guarded `decode_safe` rerun 的 PM/bubble 表格和 dashboard，可以直接运行：

```bash
python3 mla_flash_attention_dev/experiments/render_flash_mla_pm_bubble_visuals.py
```

说明：

- 这版脚本最终采用 `cpp_device_perf_report.csv` 作为稳定数据源
- 原因是本机环境下 Tracy host-side `ops_perf_results_*.csv` 后处理会卡住，但 device-side C++ report 是稳定可得的
- 因此本次瓶颈判定主要依据 `DEVICE KERNEL / BRISC / NCRISC / TRISC*` 的窗口占比
- 为了继续细分 `NCRISC`，这次又在真正的 decode reader kernel 上加了低开销 `TS_DATA` 计数，累计 `page_table / reserve / issue / wait / push`
- 这里的 `reserve` 不是简单函数调用开销，而是 `cb_reserve_back()` 等待下游释放 CB 空间的阻塞时间，因此它可以被视为 reader 的 backpressure stall
- 你后续又把这套埋点扩展到了更全面的 detailed harness：除了 decode，还覆盖了 `chunked_flash_mla_prefill`，并且新增了 writer 侧 `cb_wait / issue / barrier / pop` 的分解
- 新增的 guarded decode PM/bubble 整理脚本不会重新跑硬件，只会读取 `flash_mla_pm_bubble_probe_decode_safe` 里的 manifest 和 CSV，把 bubble 表格、SVG 和 dashboard 重生成出来

## 4. WH 实测结果

### 4.1 核心结果表

`4.1 ~ 4.4` 现在都已经改成和 `profile_flash_mla_wh_detailed.py` 对齐的全量 decode sweep；
`4.9` 再补 `prefill_256 -> prefill_4k` 的全量结果。

| case | batch | seq_len | kernel us | us/item | ncrisc us | brisc us | compute us(max trisc) | lower-bound reader GB/s | 4-lane equivalent GB/s | est out-write GB/s | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode_256 | 2 | 256 | 40.917 | 20.458583 | 13.647 | 28.782 | 40.659 | 21.610 | 86.440 | 2.277 | compute_on_critical_path |
| decode_512 | 2 | 512 | 58.109 | 29.054417 | 18.256 | 45.985 | 57.845 | 32.308 | 129.232 | 1.425 | compute_on_critical_path |
| decode_1k | 2 | 1024 | 73.531 | 36.765417 | 27.253 | 61.376 | 73.267 | 43.284 | 173.136 | 1.068 | compute_on_critical_path |
| decode_2k | 2 | 2048 | 106.069 | 53.034417 | 45.138 | 93.910 | 105.809 | 52.268 | 209.072 | 0.698 | writer_close_to_critical_path |
| decode_4k | 2 | 4096 | 171.855 | 85.927333 | 107.650 | 159.711 | 171.596 | 43.833 | 175.332 | 0.410 | writer_close_to_critical_path |
| decode_8k | 2 | 8192 | 303.168 | 151.584250 | 238.594 | 291.010 | 302.909 | 39.553 | 158.212 | 0.225 | writer_close_to_critical_path |
| decode_16k | 1 | 16384 | 560.899 | 560.899500 | 497.738 | 548.789 | 560.631 | 18.960 | 75.840 | 0.060 | reader_close_to_critical_path |
| decode_32k | 1 | 32768 | 1084.760 | 1084.760250 | 1023.405 | 1072.649 | 1084.496 | 18.443 | 73.772 | 0.031 | reader_writer_saturated |

解释：

- `lower-bound reader GB/s` 只按“唯一 K 字节”计算，是保守下界
- 当前实现 A 会发生多 lane 重读 K，因此更接近真实外存压力的是 `4-lane equivalent GB/s`
- `est out-write GB/s` 是按输出张量字节估算的 write-side 有效带宽
- `compute us(max trisc)` 来自 detailed sweep 的 `max trisc` 列，因此这里不再固定写成 `TRISC1`

### 4.2 WH reader 内部分解

新增的低开销 reader 分解结果如下，已经按 `decode_256 -> decode_32k` 全量整理：

| case | batch | seq_len | ncrisc us | page_table us/core(avg) | reserve/block us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 2 | 256 | 13.647 | 0.380 | 0.035 | 3.929 | 0.793 | 0.071 | 5.317 | 39.0% | 0.7% | 75.4% | 15.2% | 1.4% |
| decode_512 | 2 | 512 | 18.256 | 0.448 | 0.035 | 3.933 | 0.819 | 0.071 | 5.652 | 31.0% | 0.7% | 74.1% | 15.4% | 1.3% |
| decode_1k | 2 | 1024 | 27.253 | 0.420 | 0.071 | 7.871 | 1.680 | 0.145 | 10.735 | 39.4% | 0.7% | 77.3% | 16.5% | 1.4% |
| decode_2k | 2 | 2048 | 45.138 | 0.410 | 0.143 | 15.697 | 3.376 | 0.293 | 20.240 | 44.8% | 0.7% | 78.8% | 16.9% | 1.5% |
| decode_4k | 2 | 4096 | 107.650 | 0.485 | 26.069 | 31.420 | 6.977 | 0.588 | 66.180 | 61.5% | 39.8% | 47.9% | 10.6% | 0.9% |
| decode_8k | 2 | 8192 | 238.594 | 0.432 | 84.893 | 62.856 | 14.012 | 1.177 | 163.871 | 68.7% | 52.0% | 38.5% | 8.6% | 0.7% |
| decode_16k | 1 | 16384 | 497.738 | 0.484 | 214.921 | 125.626 | 16.528 | 2.362 | 360.923 | 72.5% | 59.7% | 34.9% | 4.6% | 0.7% |
| decode_32k | 1 | 32768 | 1023.405 | 0.525 | 463.280 | 251.306 | 32.916 | 4.727 | 754.885 | 73.8% | 61.5% | 33.4% | 4.4% | 0.6% |

这里有两个重要解释：

- 这些计数聚焦的是 paged KV reader 热路径，不覆盖全部 `NCRISC` 工作，所以它们的和不会精确等于 `DEVICE NCRISC KERNEL DURATION`
- 但它们已经足够说明 `NCRISC` 内部的主导阶段如何迁移

### 4.3 Decode reader 连续 sweep 解读

- `decode_256 -> decode_2k` 这一段里，`reserve share` 一直只有 `0.7%`，`issue share` 稳定在 `74.1% ~ 78.8%`，说明 reader 仍然主要在主动发起读请求，还没有被下游明显反压住。
- 真正的转折点是 `decode_4k`：`reserve share` 从 `0.7%` 直接跳到 `39.8%`，已经和 `issue` 的 `47.9%` 接近，reader 从 issue-dominant 开始转向 backpressure-dominant。
- `decode_8k -> decode_32k` 时，`reserve share` 继续从 `52.0% -> 59.7% -> 61.5%`，而 `issue share` 继续从 `38.5% -> 34.9% -> 33.4%`，说明长序列下 `cb_reserve_back()` 已经稳定成为 reader 内部的主导 stall。
- `page_table` 始终只有 `0.380 ~ 0.525 us/core(avg)`，`push` 也始终只有 `0.071 ~ 4.727 us/core(avg)`，这两项都不是主要矛盾。
- `coverage vs thread` 从短序列的 `31.0% ~ 44.8%` 提升到长序列的 `61.5% ~ 73.8%`，说明长序列时这些 marker 对 hot path 的解释力更强。

### 4.4 关键趋势

1. `NCRISC share` 按 full sweep 依次是 `33.4% -> 31.4% -> 37.1% -> 42.6% -> 62.6% -> 78.7% -> 88.7% -> 94.3%`。
2. `BRISC share` 按 full sweep 依次是 `70.3% -> 79.1% -> 83.5% -> 88.5% -> 92.9% -> 96.0% -> 97.8% -> 98.9%`。
3. reader 内部的 `reserve share` 按 full sweep 依次是 `0.7% -> 0.7% -> 0.7% -> 0.7% -> 39.8% -> 52.0% -> 59.7% -> 61.5%`。
4. writer 内部的 `cb_wait share` 按 full sweep 依次是 `89.4% -> 90.2% -> 94.1% -> 96.3% -> 98.1% -> 98.9% -> 99.6% -> 99.8%`。
5. 按 4-lane 重读折算的 reader 吞吐依次是 `86.4 -> 129.2 -> 173.1 -> 209.1 -> 175.3 -> 158.2 -> 75.8 -> 73.8 GB/s`，也就是在 `decode_2k` 左右达到峰值后开始明显下滑。

这五点放在一起，结论比较明确：

- `decode_256 -> decode_2k` 还可以近似看成 “compute 最后退休，但 writer 已经不轻，reader 仍然是 issue-dominant”。
- `decode_4k` 是最明确的迁移点：writer 已经贴边，同时 reader 内部第一次出现显性的 downstream backpressure。
- `decode_8k -> decode_32k` 已经很难再叫“纯 compute-bound”；更准确的表述是 **reader + writer + compute 共同压在 critical window 上的深流水 kernel**。
- 从 reader 内部看，长序列的主导 stall 不是单一的 `noc_async_read_barrier()`，而是更偏向 `cb_reserve_back()` 代表的下游反压。
- 换句话说，瓶颈不是单一 stage，而是 `K repeated reads + downstream buffering/backpressure + on-chip reduction/write-side` 共同把流水线填满。

### 4.5 关于 `OP TO OP LATENCY`

结果 JSON 里也有 `OP TO OP LATENCY [ns]`，但这项在 Tracy/profile 模式下会混入：

- host-side profiling 开销
- `synchronize_device()` 带来的人工间隙
- 调度和进程调度噪声

所以本次没有把它作为主要瓶颈依据。真正可信的仍然是 device kernel 内部的 `BRISC/NCRISC/TRISC` 分解。

### 4.6 更全面的 Detailed Sweep

你后面补的 `profile_flash_mla_wh_detailed.py` 把这次 profiling 扩展成了两条路径一起看：

- `prefill`：`ttnn.transformer.chunked_flash_mla_prefill`
- `decode`：`ttnn.transformer.paged_flash_multi_latent_attention_decode`
- `reader`：`page_table / reserve / issue / wait / push`
- `writer`：`cb_wait / issue / barrier / pop`

它的覆盖范围也更完整：

- `prefill_256 -> prefill_4k`
- `decode_256 -> decode_32k`

因此下面这几条结论，不再只是补 `1k / 4k / 32k` 的代表点，而是直接基于 `256 -> 32k` 的逐点 decode sweep。

### 4.7 Decode 的阶段迁移更清楚了

更全面的 decode sweep 逐点结果如下：

| case | batch | seq_len | 分类 | 含义 |
|---|---:|---:|---|---|
| `decode_256` | 2 | 256 | `compute_on_critical_path` | 仍然是 compute 最后退休 |
| `decode_512` | 2 | 512 | `compute_on_critical_path` | 仍然是 compute 最后退休 |
| `decode_1k` | 2 | 1024 | `compute_on_critical_path` | 仍然是 compute 最后退休 |
| `decode_2k` | 2 | 2048 | `writer_close_to_critical_path` | BRISC 已经贴近 kernel 窗口 |
| `decode_4k` | 2 | 4096 | `writer_close_to_critical_path` | BRISC 已经贴近 kernel 窗口 |
| `decode_8k` | 2 | 8192 | `writer_close_to_critical_path` | BRISC 已经贴近 kernel 窗口 |
| `decode_16k` | 1 | 16384 | `reader_close_to_critical_path` | NCRISC 开始明显贴近 kernel 窗口 |
| `decode_32k` | 1 | 32768 | `reader_writer_saturated` | reader 和 writer 都几乎把窗口填满 |

把逐点结果压缩成区间，就是：

| 区间 | 分类 | 含义 |
|---|---|---|
| `256 ~ 1k` | `compute_on_critical_path` | 仍然是 compute 最后退休 |
| `2k ~ 8k` | `writer_close_to_critical_path` | BRISC 已经贴近 kernel 窗口 |
| `16k` | `reader_close_to_critical_path` | NCRISC 开始明显贴近 kernel 窗口 |
| `32k` | `reader_writer_saturated` | reader 和 writer 都几乎把窗口填满 |

也就是说，之前只能从少数点位看出大趋势，而现在 `256 -> 32k` 的连续 sweep 已经把每个迁移点直接落到了具体 case 上：

- 真正从 compute 主导转向 writer 贴边，发生在 `decode_2k`
- `decode_2k -> decode_8k` 都还属于 writer-close 阶段
- 真正从 writer 贴边转向 reader 也贴边，发生在 `decode_16k`
- 到 `decode_32k` 才进入 reader/writer 一起饱和

### 4.8 Decode 的 Writer 实际上主要在等

这次新增的 writer 分解非常关键，因为它纠正了一个容易误读的点。

按 `BRISC total` 看，`decode_2k -> 32k` 的 writer 确实越来越接近 critical path；但看内部细分，full sweep 的结论更清楚：

| case | batch | seq_len | brisc us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 2 | 256 | 28.782 | 10.860 | 0.656 | 0.594 | 0.035 | 24.498 | 85.1% | 89.4% | 5.4% | 4.9% | 0.3% |
| decode_512 | 2 | 512 | 45.985 | 15.274 | 0.856 | 0.742 | 0.052 | 29.092 | 63.3% | 90.2% | 5.1% | 4.4% | 0.3% |
| decode_1k | 2 | 1024 | 61.376 | 23.299 | 0.744 | 0.652 | 0.052 | 44.962 | 73.3% | 94.1% | 3.0% | 2.6% | 0.2% |
| decode_2k | 2 | 2048 | 93.910 | 39.546 | 0.809 | 0.668 | 0.052 | 77.529 | 82.6% | 96.3% | 2.0% | 1.6% | 0.1% |
| decode_4k | 2 | 4096 | 159.711 | 72.297 | 0.709 | 0.637 | 0.052 | 143.123 | 89.6% | 98.1% | 1.0% | 0.9% | 0.1% |
| decode_8k | 2 | 8192 | 291.010 | 137.924 | 0.800 | 0.685 | 0.052 | 274.007 | 94.2% | 98.9% | 0.6% | 0.5% | 0.0% |
| decode_16k | 1 | 16384 | 548.789 | 267.461 | 0.584 | 0.548 | 0.052 | 532.093 | 97.0% | 99.6% | 0.2% | 0.2% | 0.0% |
| decode_32k | 1 | 32768 | 1072.649 | 529.359 | 0.600 | 0.557 | 0.052 | 1055.945 | 98.4% | 99.8% | 0.1% | 0.1% | 0.0% |

这说明：

- `cb_wait share` 从最短点 `decode_256` 就已经有 `89.4%`，到 `decode_2k` 之后进一步拉到 `96.3% -> 98.1% -> 98.9% -> 99.6% -> 99.8%`。
- 真正的 `issue / barrier / pop` 在全 sweep 上都只占很小一部分，没有哪一个 case 表现出“writer issue 自身成为主导瓶颈”的迹象。
- `writer_close_to_critical_path` 不等于 “输出写带宽本身已经打满”
- 更准确地说，BRISC 大部分时间都在 `cb_wait_front()` 上等待 compute/reduction 把结果推到输出 CB
- 所以 decode 长序列时，writer 是 **pipeline-coupled / output-availability bound**，而不是纯粹的 NoC 写出带宽瓶颈

这和 reader 侧 `reserve/block` 的结论是互补的：

- reader 长序列越来越多时间花在等下游腾空间
- writer 长序列几乎整个线程窗口都在等上游产出结果

换句话说，decode 后半段已经是一个强耦合的深流水系统，而不是几个独立 stage 分别吃满各自带宽。

### 4.9 Prefill 路径的特征和 Decode 很不一样

更全面的 detailed sweep 也给出了 `prefill_256 -> prefill_4k` 的全量实测：

| case | batch | seq_len | kernel us | us/item | ncrisc us | brisc us | compute us(max trisc) | est k-read GB/s | est out-write GB/s | reader 主导项 | writer 主导项 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `prefill_256` | 1 | 256 | 412.770 | 1.612383 | 400.724 | 412.329 | 409.874 | 0.368 | 20.344 | `wait 70.3%` | `cb_wait 91.3%` | `reader_writer_saturated` |
| `prefill_512` | 1 | 512 | 1211.550 | 2.366309 | 1199.414 | 1211.104 | 1208.574 | 0.246 | 13.853 | `wait 68.7%` | `cb_wait 95.6%` | `reader_writer_saturated` |
| `prefill_1k` | 1 | 1024 | 4009.476 | 3.915504 | 3997.468 | 4009.029 | 4006.625 | 0.148 | 8.370 | `wait 67.2%` | `cb_wait 98.3%` | `reader_writer_saturated` |
| `prefill_2k` | 1 | 2048 | 14556.958 | 7.107889 | 14544.947 | 14556.502 | 14554.106 | 0.081 | 4.610 | `wait 66.5%` | `cb_wait 99.2%` | `reader_writer_saturated` |
| `prefill_4k` | 1 | 4096 | 55339.322 | 13.510577 | 55327.319 | 55338.878 | 55336.467 | 0.043 | 2.425 | `wait 66.1%` | `cb_wait 99.6%` | `reader_writer_saturated` |

Prefill reader 分解全量表：

| case | batch | seq_len | ncrisc us | page_table us/core(avg) | reserve/block us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_256 | 1 | 256 | 400.724 | 0.443 | 0.806 | 82.123 | 199.745 | 0.999 | 291.505 | 72.7% | 0.3% | 28.9% | 70.3% | 0.4% |
| prefill_512 | 1 | 512 | 1199.414 | 0.475 | 2.712 | 273.738 | 613.855 | 3.337 | 906.107 | 75.5% | 0.3% | 30.6% | 68.7% | 0.4% |
| prefill_1k | 1 | 1024 | 3997.468 | 0.468 | 9.853 | 985.467 | 2063.975 | 12.038 | 3096.468 | 77.5% | 0.3% | 32.1% | 67.2% | 0.4% |
| prefill_2k | 1 | 2048 | 14544.947 | 0.514 | 37.443 | 3722.899 | 7565.107 | 45.536 | 11461.041 | 78.8% | 0.3% | 32.7% | 66.5% | 0.4% |
| prefill_4k | 1 | 4096 | 55327.319 | 0.545 | 145.882 | 14453.581 | 28818.300 | 176.968 | 43936.937 | 79.4% | 0.3% | 33.2% | 66.1% | 0.4% |

Prefill writer 分解全量表：

| case | batch | seq_len | brisc us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_256 | 1 | 256 | 412.329 | 322.773 | 14.778 | 15.938 | 0.196 | 358.132 | 86.9% | 91.3% | 4.2% | 4.5% | 0.1% |
| prefill_512 | 1 | 512 | 1211.104 | 1050.039 | 24.873 | 22.683 | 0.436 | 1107.971 | 91.5% | 95.6% | 2.3% | 2.1% | 0.0% |
| prefill_1k | 1 | 1024 | 4009.029 | 3702.518 | 30.547 | 32.644 | 0.804 | 3796.350 | 94.7% | 98.3% | 0.8% | 0.9% | 0.0% |
| prefill_2k | 1 | 2048 | 14556.502 | 13928.075 | 54.591 | 56.065 | 1.610 | 14131.300 | 97.1% | 99.2% | 0.4% | 0.4% | 0.0% |
| prefill_4k | 1 | 4096 | 55338.878 | 53926.021 | 102.627 | 100.210 | 3.136 | 54483.307 | 98.5% | 99.6% | 0.2% | 0.2% | 0.0% |

这里的含义也很明确：

- prefill 的所有测点都已经是 `reader_writer_saturated`，从 `256` 开始就不是短 decode 那种 “compute 最后退休” 的形态。
- `NCRISC` 里主导项一直是 `wait`，而且 share 只是在 `70.3% -> 68.7% -> 67.2% -> 66.5% -> 66.1%` 之间缓慢下降，不会像 decode 那样从 `issue` 切换到 `reserve`。
- `BRISC` 里主导项几乎一直是 `cb_wait`，而且 share 从 `91.3%` 继续升到 `99.6%`。
- `page_table`、`reserve`、`push/pop` 都非常小，不是主要矛盾。

所以 prefill 更像是：

- reader 在等待 in-flight read completion
- writer 在等待 compute/reduction 产出
- 三条主线程从较短序列开始就已经是强耦合的饱和流水线

这和 decode 的差异在于：

- decode 是先 `compute -> writer -> reader+writer`
- prefill 则几乎从起点开始就是 `reader + writer + compute` 一起贴着窗口走

### 4.10 这轮数据现在完整到哪一层

如果把 full detailed profile 和这次 guarded `decode_safe` rerun 一起看，当前状态是：

- `base measured profile`：`13/13`，已经覆盖 `kernel / BRISC / NCRISC / TRISC* / classification`。
- `reader stage breakdown`：`13/13`，已经覆盖 `page_table / reserve / issue / wait / push`。
- `writer stage breakdown`：`13/13`，已经覆盖 `cb_wait / issue / barrier / pop`。
- `decode reader source breakdown`：`0/8`，当前还没有真实 `K/V` source 数据。
- `decode writer source breakdown`：`0/8`，当前还没有真实 `sender/root/tree/output` source 数据。
- `decode compute bubble counters`：`8/8`，guarded rerun 已经补出 `DEVICE COMPUTE CB WAIT FRONT / RESERVE BACK`。
- `decode PM triplet raw columns`：`8/8` 列存在，但当前值固定是 `PM IDEAL=1.0 ns`、`PM COMPUTE=1.0 ns`、`PM BANDWIDTH=1.0 ns`。
- `decode PM FPU util raw column`：`8/8` 列存在，但当前只有 `0.000 ~ 0.002`。
- `decode NOC/MULTICAST/DRAM util`：`0/8` usable，当前仍然为空。

所以，本文现在已经完整到 **stage-level + decode compute-bubble counter** 这一层；
但还没有完整到 **usable PM utilization / NOC-DRAM util / source-level attribution** 这三层。

这次 guarded rerun 的 decode 状态矩阵如下：

| case | stage profile | compute bubble | PM triplet | PM FPU util | NOC/MCAST/DRAM util |
|---|---|---|---|---|---|
| decode_256 | ok | ok | placeholder | placeholder | missing |
| decode_512 | ok | ok | placeholder | placeholder | missing |
| decode_1k | ok | ok | placeholder | placeholder | missing |
| decode_2k | ok | ok | placeholder | placeholder | missing |
| decode_4k | ok | ok | placeholder | placeholder | missing |
| decode_8k | ok | ok | placeholder | placeholder | missing |
| decode_16k | ok | ok | placeholder | placeholder | missing |
| decode_32k | ok | ok | placeholder | placeholder | missing |

### 4.11 这次 Guarded Decode Rerun 真正补齐了什么

这轮 guarded rerun 先用 `decode_256` 做 preflight，`status=passed`、`attempts=1`，然后完整跑完了 `decode_256 -> decode_32k` 的 `8` 个点。
它真正补齐的是 `compute-side bubble counter`，而不是可直接解释的 `PM/NOC/DRAM utilization`。

`8` 个 decode 点的整理表如下：

| case | seq_len | kernel us | compute us(max trisc) | wait-front counter us | reserve-back counter us | bubble density vs kernel | wait-front share in bubble | reserve-back share in bubble | reader reserve share | writer cb_wait share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 256 | 42.198 | 41.947 | 61.200 | 8.789 | 1.66x | 87.4% | 12.6% | 0.7% | 89.4% |
| decode_512 | 512 | 59.574 | 59.319 | 162.581 | 19.845 | 3.06x | 89.1% | 10.9% | 0.7% | 90.2% |
| decode_1k | 1024 | 76.223 | 75.977 | 187.483 | 30.225 | 2.86x | 86.1% | 13.9% | 0.7% | 94.1% |
| decode_2k | 2048 | 109.765 | 109.517 | 234.231 | 51.309 | 2.60x | 82.0% | 18.0% | 0.7% | 96.3% |
| decode_4k | 4096 | 177.332 | 177.080 | 332.069 | 93.027 | 2.40x | 78.1% | 21.9% | 39.8% | 98.1% |
| decode_8k | 8192 | 311.103 | 310.853 | 525.598 | 177.100 | 2.26x | 74.8% | 25.2% | 52.0% | 98.9% |
| decode_16k | 16384 | 575.166 | 574.917 | 442.915 | 172.588 | 1.07x | 72.0% | 28.0% | 59.7% | 99.6% |
| decode_32k | 32768 | 1112.087 | 1111.839 | 829.716 | 340.300 | 1.05x | 70.9% | 29.1% | 61.5% | 99.8% |

这里有三个关键点：

- `wait-front` 在 bubble 组成里始终是主导项，但 share 会从 `87.4%` 逐步降到 `70.9%`；与此同时 `reserve-back share` 会从 `12.6%` 升到 `29.1%`。
- `bubble density vs kernel` 在短序列上会明显超过 `1x`，因此这些值必须按 **stall-density / accumulated counter** 来解读，而不能直接当作 wall-time share。
- 如果把这组 bubble counter 和 stage-level 结果放在一起看，趋势是一致的：`reader reserve share` 会从 `0.7%` 抬升到 `61.5%`，而 `writer cb_wait share` 会从 `89.4%` 继续抬到 `99.8%`；所以长序列 decode 依然更像 **reader backpressure + writer output-availability wait** 共同主导的强耦合流水线。

### 4.12 现在还能不能直接看 PM / NOC / DRAM util

还不能。

这轮 guarded rerun 的原始捕获状态是：

- `PM IDEAL / PM COMPUTE / PM BANDWIDTH`：`8/8` 个点都固定在 `1.0 ns`。
- `PM FPU UTIL (%)`：`8/8` 个点都只有 `0.000 ~ 0.002`。
- `NOC UTIL (%)`、`MULTICAST NOC UTIL (%)`、`DRAM BW UTIL (%)`：`8/8` 个点仍然缺失。

因此，这轮结果应该被明确解读成：

- `bubble-complete`
- `PM-incomplete`

相关整理产物已经单独放到：

- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_decode_summary.csv`
- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_summary.md`
- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/flash_mla_pm_bubble_dashboard.html`
- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_decode_safe/visuals/`

后续又额外做了两次更保守的验证：

- `perf-fpu` smoke：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_perf_fpu_smoke/run_manifest.json`
- 独立健康探测：`mla_flash_attention_dev/experiments/profile_outputs/flash_mla_pm_bubble_probe_healthcheck_after_perf_fpu/run_manifest.json`

这两次都停在 `preflight failed`，错误签名一致是：

- `Timeout waiting for Ethernet core service remote IO request`
- `Read unexpected run_mailbox value`

所以截至目前，最稳妥的结论是：

- `compute bubble counter` 已拿到
- `perf-fpu / NOC util` 还没有拿到一次成功进入 profiling 的 smoke-run
- 后续应先跑独立 `--preflight-only` 健康探测，再决定是否继续尝试更激进的参数组合

### 4.13 把这些结果统一翻译成“利用率与空泡”口径

如果只问“现有结果能不能分析利用率和空泡”，最稳妥的回答是：**能，但要分层**。

#### 4.13.1 现在可以直接解读的

- `BRISC/NCRISC/TRISC` 相对 `DEVICE KERNEL DURATION` 的占比。这一层更接近“线程窗口利用率代理量”，它能回答“哪条线程在逼近 critical path”，但不能直接回答“FPU 利用率是不是 80%”。
- `reader reserve`、`reader wait`、`writer cb_wait` 这类 stage-level share。这一层能回答 stall/反压主要堆在 reader 还是 writer，属于现阶段最可信的空泡代理量。
- `DEVICE COMPUTE CB WAIT FRONT / RESERVE BACK`。这批 guarded decode rerun 已经把 `8/8` 个点补齐，能直接用来看 compute-side bubble 的组成变化。

#### 4.13.2 现在还不能直接解读的

- `PM FPU UTIL (%)`、`NOC UTIL (%)`、`MULTICAST NOC UTIL (%)`、`DRAM BW UTIL (%)` 的真实硬件利用率百分比。
- compute 自己按 wall-time 计，到底有多少比例在等输入、多少比例在等输出。
- `K/V reserve`、`sender/root/tree/output wait` 这些 source-level stall attribution 的直接归因。

原因是当前 guarded rerun 里：

- `PM IDEAL / PM COMPUTE / PM BANDWIDTH` 仍固定在 `1.0 ns`
- `PM FPU UTIL (%)` 仍只有 `0.000 ~ 0.002`
- `NOC/MULTICAST/DRAM BW UTIL` 仍全部缺失

所以这批结果应继续统一解读成：

- `stage-level utilization/stall proxy = usable`
- `decode compute bubble counter = usable`
- `direct PM/NOC/DRAM utilization = not yet usable`

#### 4.13.3 如果用“利用率与空泡”语言重述 Decode

- `decode_256 ~ 1k` 仍可视为 compute 最后退休，但 writer 已经很早就不是 issue-bound，而是主要在等上游结果 ready。
- `decode_2k ~ 8k` 的主要变化不是“writer 自己写带宽打满”，而是 writer 几乎整个线程窗口都被 `cb_wait` 占住；与此同时 reader 在 `decode_4k` 开始从 issue-dominant 转成 reserve-dominant。
- `decode_16k ~ 32k` 时，reader 和 writer 都几乎贴满 kernel window；因此最准确的表述不再是“pure compute-bound”，而是 **reader downstream backpressure + writer output-availability wait + compute** 共同压在 critical window 上。
- compute bubble counter 也支持这一点：bubble 内部的 `wait-front share` 会从 `87.4%` 下降到 `70.9%`，而 `reserve-back share` 会从 `12.6%` 升到 `29.1%`，说明长序列下 output/backpressure 成分在增强。

这里还要再强调一次口径：

- `bubble density vs kernel` 在短序列上会超过 `1x`
- 因此 `wait-front / reserve-back` 必须按 `accumulated stall density` 来读
- 不能把它们直接解释成 wall-time 上的“空泡百分比”

#### 4.13.4 如果用同一口径重述 Prefill

- `prefill_256 -> prefill_4k` 从起点开始就是 `reader_writer_saturated`
- `NCRISC share` 基本全程在 `97.1% -> 100.0%`
- `BRISC share` 也几乎全程贴满
- reader 内部主导项始终是 `wait`
- writer 内部主导项几乎始终是 `cb_wait`

所以 prefill 更准确的工程表述是：

- 它从最短点开始就已经是 `reader + writer + compute` 的强耦合饱和流水线
- reader 更像在等 in-flight read completion
- writer 更像在等 compute/reduction 产出
- 它和 decode 的区别不在于“是不是算力不够”，而在于 stall 的主导位置不同

#### 4.13.5 工程含义

- 对 decode，后续优化优先级不应只盯 compute，而应优先盯 reader 的 buffer turnover / downstream backpressure，以及 writer 的 output-availability wait。
- `decode_4k` 是最值得盯的迁移点，因为它第一次同时表现出 reader reserve 的大幅抬升和 writer 的持续贴边。
- 对 prefill，应直接把它视为饱和流水线问题，而不是先把它归类成“compute-bound”再看。

如果只想单独阅读这一部分，也可以直接看抽出的独立整理版：

- `mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_utilization_and_bubble_report.md`

### 4.14 最新同步代表点 rerun 又补出了什么

后面又补了一轮更保守的同步 rerun，统一加了：

- `--sync-host-device`
- `--enable-sum-profiling`

这轮代表点包括：

- `decode_1k`（先前 sync smoke）
- `decode_4k`
- `decode_32k`
- `prefill_4k`

它带来了三个新的结论。

#### 4.14.1 `--sync-host-device` 已经证明能解决 host capture race

之前 after-device-restart 的失败，本质上不是 `child-run` 打不开设备，而是 Tracy host trace 没落地。

这轮加上 `--sync-host-device` 之后：

- `decode_1k` sync smoke 成功生成了 `ops_perf_results_*.csv`
- `decode_4k / decode_32k / prefill_4k` 代表点也都成功生成了 `ops_perf_results_*.csv`

所以到这一步可以比较明确地说：

- 当前机器上 `sum` 路线并不是根本不可用
- 真正的关键 guardrail 是先做 plain preflight，再给 Tracy 加 `--sync-host-device`

#### 4.14.2 Decode 的 direct PM 仍然不可用，但 source-level stall attribution 已经补出来了

`decode_1k / 4k / 32k` 这三个点上，原始 PM 列的状态依然是：

- `PM IDEAL / PM COMPUTE / PM BANDWIDTH = 1.0 ns`
- `PM FPU UTIL (%) = 0.001` 或缺失
- `NOC / MULTICAST NOC / DRAM BW util = missing`

也就是说，**decode direct PM utilization 这层仍然没有补出来**。

但这轮真正新补出来的是 source-level stall attribution：

- `decode_1k`：reader 里仍然是 `K/V issue` 主导；`K` 路总占比约 `72.5%`，`V` 路约 `27.5%`
- `decode_4k`：reader 开始明显出现 `K reserve`；`k_reserve=31.64%`
- `decode_32k`：reader 已经变成 `K reserve` 主导；`k_reserve=54.82%`

这意味着：

- decode 长序列 reader 端真正主导 backpressure 的是 `K` 路，而不是 `V` 路
- `reader reserve` 这项现在可以从“经验解释”推进成更直接的 source-level 结论

writer 侧的 source-level 分解也很稳定：

- `sender_cb_wait` 约 `48.6% -> 49.6% -> 50.0%`
- `tree_child_wait` 约 `51.4% -> 50.4% -> 50.0%`
- `root_cb_wait` 约 `0%`
- `output_gather_wait` 约 `0%`

这说明 decode writer 的主等待并不在 final output gather，而是：

- sender 在等本地 partial output ready
- tree/root reduction 在等 child partial result 到齐

#### 4.14.3 Prefill 第一次拿到了可直接解释的 PM 利用率

这轮最重要的新点其实是 `prefill_4k`。

它不再是 placeholder，而是给出了可直接解释的 PM triplet：

- `kernel wall = 55.516 ms`
- `PM IDEAL = 10.785 ms`
- `PM COMPUTE = 10.785 ms`
- `PM BANDWIDTH = 0.545 ms`
- `PM FPU UTIL = 19.427%`

这组数和 stage/bubble 数据放在一起，含义很明确：

- `prefill_4k` 并不是算术管线打满
- `PM COMPUTE / kernel wall` 只有约 `19.4%`
- 仍有约 `80.6%` 的 wall time 不在 direct PM compute 里

而已有的 stage/bubble 数据恰好解释了这部分 gap：

- reader 仍以 `wait` 主导
- writer 仍以 `cb_wait` 主导
- compute bubble 里 `wait-front share` 高达约 `92.6%`

所以 `prefill_4k` 现在可以更明确地写成：

- 算术管线没有真正饱和
- 主 wall-time 仍由强耦合流水线等待决定
- 也就是 `reader wait + writer cb_wait + compute wait-front` 共同把 kernel 拉长

#### 4.14.4 当前还能不能继续追 NOC util

这轮也单点试过 `--collect-noc-traces`，但在 `decode_1k` 上直接触发了：

- `TT_FATAL: Invalid NoC transfer type on device: 0`

因此当前更稳妥的结论是：

- `sum + --sync-host-device`：可用，适合继续补 `bubble` 和 source-level 归因
- `perf-fpu`：单点 smoke 能跑，但 decode 上仍没有带来可解释 PM 值
- `--collect-noc-traces`：当前机器/当前路径下仍不稳定，不宜直接放大全 sweep

### 4.15 同步补充 sweep 把 `decode_16k` 和 `prefill PM curve` 补齐了

之后又继续补了一轮同步 sweep：

- `decode_16k`
- `prefill_256`
- `prefill_512`
- `prefill_1k`
- `prefill_2k`

#### 4.15.1 `decode_16k` 现在也能直接落到 source-level

`decode_16k` 的关键值是：

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

这说明 `decode_16k` 已经非常接近 `decode_32k` 的 long-seq 形态：

- reader 端已经明确由 `K reserve` 主导，而不是 `V reserve`
- writer 端依然几乎完全由 `sender/tree` 两段等待构成

所以 decode source-level 的迁移现在可以更完整地写成：

- `1k`：`K/V issue` 主导
- `4k`：`K issue + K reserve` 并存
- `16k/32k`：`K reserve` 主导

#### 4.15.2 `prefill` 的 PM 利用率曲线已经从 `256 -> 4k` 连起来了

这轮最重要的新补充，是把 prefill 的 direct PM 利用率从 `256` 一直补到了 `4k`：

| case | kernel ms | PM COMPUTE ms | PM BANDWIDTH ms | PM FPU util | wait-front share in bubble | reserve-back share in bubble |
|---|---:|---:|---:|---:|---:|---:|
| `prefill_256` | 0.413 | 0.042 | 0.034 | 10.20% | 96.0% | 4.0% |
| `prefill_512` | 1.212 | 0.169 | 0.068 | 13.90% | 94.6% | 5.4% |
| `prefill_1k` | 4.021 | 0.674 | 0.136 | 16.76% | 93.6% | 6.4% |
| `prefill_2k` | 14.595 | 2.696 | 0.273 | 18.48% | 92.9% | 7.1% |
| `prefill_4k` | 55.516 | 10.785 | 0.545 | 19.43% | 92.6% | 7.4% |

这张表的含义很清楚：

- `PM FPU UTIL` 会从 `10.2%` 持续升到 `19.4%`
- 但即使到 `prefill_4k`，direct compute 也仍然只解释了约 `19.4%` 的 kernel wall time
- compute bubble 里始终以 `wait-front` 为主，`reserve-back` 只是次要项

再结合已有 stage-level：

- reader 的 `wait share` 只是在 `70.31% -> 66.25%` 间缓慢下降
- writer 的 `cb_wait share` 则从 `91.50%` 升到 `99.62%`

因此 prefill 现在可以更明确地写成：

- direct PM util 是**可测且随序列增长而上升**的
- 但主 wall-time 依然不是由算术管线饱和决定
- 真正主导的仍然是 `reader wait + writer cb_wait + compute wait-front`

#### 4.15.3 到这一步的实验完成口径

到现在可以把“完成情况”进一步收敛成：

- `decode`：`stage-level + compute bubble + source-level attribution` 已经足够完整；`direct PM/NOC/DRAM util` 仍未补齐
- `prefill`：`stage-level + compute bubble + direct PM util` 已经在 `256 -> 4k` 上补齐

也就是说，后续如果继续做 component-level utilization 分析：

- 对 `decode`，主轴应该继续放在 source-level / pipeline-coupling
- 对 `prefill`，则可以正式把 PM util 纳入主分析，而不只停留在 stage proxy

## 5. BH 模拟分析

### 5.1 当前实现 A 直接搬到 BH（一阶模型）

保持 A 的 decode 拓扑不变，只替换为 BH 常数后，full decode sweep 的一阶模型如下：

| case | seq_len | K DRAM (MB) | dram ms | compute ms | reduce ms | core ms | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---|
| decode_256 | 256 | 0.5625 | 0.0012 | 0.0016 | 0.0011 | 0.0027 | compute |
| decode_512 | 512 | 1.1250 | 0.0023 | 0.0032 | 0.0011 | 0.0043 | compute |
| decode_1k | 1024 | 2.2500 | 0.0046 | 0.0064 | 0.0011 | 0.0075 | compute |
| decode_2k | 2048 | 4.5000 | 0.0092 | 0.0129 | 0.0011 | 0.0140 | compute |
| decode_4k | 4096 | 9.0000 | 0.0184 | 0.0258 | 0.0011 | 0.0269 | compute |
| decode_8k | 8192 | 18.0000 | 0.0369 | 0.0516 | 0.0011 | 0.0527 | compute |
| decode_16k | 16384 | 36.0000 | 0.0737 | 0.1032 | 0.0011 | 0.1043 | compute |
| decode_32k | 32768 | 72.0000 | 0.1475 | 0.2063 | 0.0011 | 0.2074 | compute |

结论：

- BH 更高的 DRAM 带宽会先把 A 的外存读取时间压下去。
- 但在当前保守假设的 `16 active cores` 下，`decode_256 -> decode_32k` 全 sweep 都会先被 compute 压住。
- 所以 “把现在的 A 直接搬去 BH” 当然会更快，但结构问题并没有被真正解决。
- 不过这一节仍然只是 `compute vs dram` 的一阶 lower-bound，它还没有把 WH 上已经观察到的 pipeline 耦合项带进来。

继续做 active-core 敏感性扫描后，可以把这个结论再往前推进一步：

| case | seq_len | crossover active cores | 16c ms | 20c ms | 24c ms | 32c ms | 48c ms | 64c ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 256 | 23 | 0.0027 | 0.0024 | 0.0023 | 0.0023 | 0.0023 | 0.0023 |
| decode_512 | 512 | 23 | 0.0043 | 0.0037 | 0.0034 | 0.0034 | 0.0034 | 0.0034 |
| decode_1k | 1024 | 23 | 0.0075 | 0.0063 | 0.0057 | 0.0057 | 0.0057 | 0.0057 |
| decode_2k | 2048 | 23 | 0.0140 | 0.0114 | 0.0103 | 0.0103 | 0.0103 | 0.0103 |
| decode_4k | 4096 | 23 | 0.0269 | 0.0217 | 0.0195 | 0.0195 | 0.0195 | 0.0195 |
| decode_8k | 8192 | 23 | 0.0527 | 0.0424 | 0.0380 | 0.0380 | 0.0380 | 0.0380 |
| decode_16k | 16384 | 23 | 0.1043 | 0.0836 | 0.0748 | 0.0748 | 0.0748 | 0.0748 |
| decode_32k | 32768 | 23 | 0.2074 | 0.1662 | 0.1486 | 0.1486 | 0.1486 | 0.1486 |

解读：

- 对这组 workload，A-BH 的 `compute/dram crossover` 在全 sweep 上都稳定在 `23 active cores` 左右。
- `16c` 和 `20c` 时仍是 compute-bound；到 `24c` 以后就重新回到 DRAM-bound，继续增加 active cores 基本只剩平台期。
- 这说明“一阶 A-BH 是 compute-bound”的判断更精确地说是：**在当前保守假设的 `16 active cores` 下，全 decode sweep 都会被判成 compute-bound；但只要 active cores 提到约 `24`，一阶模型里瓶颈就又回到 DRAM。**

### 5.2 把 WH detailed 观测量回灌到 A-BH（二阶经验模型）

上面的 `5.1` 只回答了一个问题：如果把 A 的拓扑原样搬去 BH，仅替换 DRAM / compute 常数，理想下界会先被谁压住。

但这次更全面的 WH detailed sweep 已经说明，A 的 decode 里还有两个不能忽略的强耦合项：

- reader 的 `reserve/block`，本质是等下游腾出 CB 空间
- writer 的 `cb_wait_front`，本质是等上游 compute/reduction 把结果推到输出 CB

因此我又补了一版经验二阶模型，并已经直接写进 `profile_flash_mla_wh_detailed.py` 自动产出。它的做法是：

- 用 WH decode detailed profile 里的 `issue / wait / push / reserve` 和 `cb_wait / issue / barrier / pop` 作为校准源
- 把 active 子项按 BH 的 DRAM / NoC / 时钟常数缩放
- 把 `reader reserve` 和 `writer cb_wait` 当作 pipeline coupling 项迭代回灌到 BH kernel window
- 因而它不再是纯粹的 `compute vs dram`，而是一个带 backpressure 的经验模型

得到的结果如下：

| case | seq_len | ideal 1st-order ms | empirical 2nd-order ms | compute ms | reader act/reserve ms | writer act/wait ms | 主导 | uplift vs ideal |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| decode_256 | 256 | 0.0027 | 0.0069 | 0.0016 | 0.0068 / 0.0000 | 0.0021 / 0.0018 | reader | 2.53x |
| decode_512 | 512 | 0.0043 | 0.0095 | 0.0032 | 0.0091 / 0.0000 | 0.0071 / 0.0024 | writer | 2.19x |
| decode_1k | 1024 | 0.0075 | 0.0135 | 0.0064 | 0.0135 / 0.0000 | 0.0069 / 0.0043 | reader | 1.79x |
| decode_2k | 2048 | 0.0140 | 0.0227 | 0.0129 | 0.0227 / 0.0000 | 0.0069 / 0.0085 | reader | 1.62x |
| decode_4k | 4096 | 0.0269 | 0.0450 | 0.0258 | 0.0411 / 0.0039 | 0.0069 / 0.0190 | reader | 1.67x |
| decode_8k | 8192 | 0.0527 | 0.0922 | 0.0516 | 0.0777 / 0.0145 | 0.0071 / 0.0420 | reader | 1.75x |
| decode_16k | 16384 | 0.1043 | 0.1826 | 0.1032 | 0.1431 / 0.0395 | 0.0069 / 0.0871 | reader | 1.75x |
| decode_32k | 32768 | 0.2074 | 0.3715 | 0.2063 | 0.2833 / 0.0881 | 0.0069 / 0.1813 | reader | 1.79x |

这组数的含义非常直接：

- `A-BH` 不再只是“一阶模型下的 compute-bound”。
- 一旦把 WH 上已经观察到的 `reserve` 和 `cb_wait` 带进去，full sweep 的 uplift 会变成 `2.53x -> 2.19x -> 1.79x -> 1.62x -> 1.67x -> 1.75x -> 1.75x -> 1.79x`。
- `decode_256` 的二阶主导项已经是 reader，`decode_512` 是 writer，`decode_1k -> decode_32k` 又回到 reader；真正先顶到上限的已经不是 pure compute，而是 reader / writer 耦合出来的 floor。

active-core sweep 也因此发生了明显变化：

| case | seq_len | 16c | 20c | 24c | 32c | 48c | 64c |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 256 | 0.0069 (reader) | 0.0069 (reader) | 0.0069 (reader) | 0.0069 (reader) | 0.0069 (reader) | 0.0069 (reader) |
| decode_512 | 512 | 0.0095 (writer) | 0.0095 (writer) | 0.0095 (writer) | 0.0095 (writer) | 0.0095 (writer) | 0.0095 (writer) |
| decode_1k | 1024 | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) |
| decode_2k | 2048 | 0.0227 (reader) | 0.0227 (reader) | 0.0227 (reader) | 0.0227 (reader) | 0.0227 (reader) | 0.0227 (reader) |
| decode_4k | 4096 | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) |
| decode_8k | 8192 | 0.0922 (reader) | 0.0914 (reader) | 0.0914 (reader) | 0.0914 (reader) | 0.0914 (reader) | 0.0914 (reader) |
| decode_16k | 16384 | 0.1826 (reader) | 0.1784 (reader) | 0.1784 (reader) | 0.1784 (reader) | 0.1784 (reader) | 0.1784 (reader) |
| decode_32k | 32768 | 0.3715 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) |

解读：

- 一阶模型里的 `23 active cores` crossover 依然有价值，但它更像“理想化 compute/dram 边界”。
- 二阶经验模型显示，`decode_256` 会直接落到 reader floor，`decode_512` 会直接落到 writer floor，而 `decode_1k -> decode_32k` 则几乎全部落到 reader floor。
- 换句话说，如果把 WH 上已经显现的 pipeline backpressure 原样带到 BH，A-BH 几乎不会因为多加 active cores 而继续明显缩短；真正缺的不是再多一些算力，而是把当前 `reader reserve` / `writer cb_wait` 所代表的流水线耦合和拓扑问题一起改掉。

### 5.2.1 二阶模型里的两个新阈值

如果继续把 active-core 扫描往下看到比 `16c` 更低，会发现二阶模型和一阶模型的阈值并不是一回事。

这里区分两个点：

- `一阶 dram crossover`：理想 `compute vs dram` 模型里，什么时候从 compute-bound 变成 dram-bound
- `二阶 non-compute crossover`：带上 `reader reserve + writer cb_wait` 之后，什么时候 compute 不再是 kernel window 的主导项
- `二阶 plateau start`：继续加 active cores 时，时延已经进入距 `64c` floor 约 `2%` 以内的平台区

| case | seq_len | 一阶 dram crossover | 二阶 non-compute crossover | 二阶 plateau start | `16c -> 64c` 二阶收益 | 64c floor ms |
|---|---:|---:|---:|---:|---:|---:|
| decode_256 | 256 | `23c` | `4c` | `4c` | `1.00x` | `0.0069` |
| decode_512 | 512 | `23c` | `6c` | `6c` | `1.00x` | `0.0095` |
| decode_1k | 1024 | `23c` | `8c` | `8c` | `1.00x` | `0.0135` |
| decode_2k | 2048 | `23c` | `10c` | `9c` | `1.00x` | `0.0227` |
| decode_4k | 4096 | `23c` | `9c` | `13c` | `1.00x` | `0.0450` |
| decode_8k | 8192 | `23c` | `8c` | `15c` | `1.01x` | `0.0914` |
| decode_16k | 16384 | `23c` | `8c` | `17c` | `1.02x` | `0.1784` |
| decode_32k | 32768 | `23c` | `7c` | `17c` | `1.03x` | `0.3617` |

这张表非常关键，因为它说明：

- 一阶模型里看到的 `23c`，只是“理想 compute/dram 边界”。
- 但在二阶经验模型里，`compute` 实际上在 `4c ~ 10c` 之间就已经退出 critical path，远低于 `23c`。
- 真正更接近 runtime 的结论是：**A-BH 会很早撞上非 compute 的 floor，平台区会在 `4c ~ 17c` 之间逐步形成，因此 `16c -> 64c` 基本没有继续扩展的空间。**

### 5.2.2 谁在制造这个 floor

再往前拆一层，可以对二阶模型里的耦合项做一个解释性归因。

这里没有新增硬件埋点，而是把：

- `reader reserve` 按 `compute / writer_total` 的相对窗口大小做软归因
- `writer cb_wait` 按 `compute / reader_total` 的相对窗口大小做软归因

得到的 full sweep 归因如下：

| case | seq_len | reader active | reserve<-compute | reserve<-writer | writer active | wait<-compute | wait<-reader | 最大耦合项 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| decode_256 | 256 | 0.0068 | 0.0000 | 0.0000 | 0.0021 | 0.0003 | 0.0015 | `wait<-reader` |
| decode_512 | 512 | 0.0091 | 0.0000 | 0.0000 | 0.0071 | 0.0006 | 0.0018 | `wait<-reader` |
| decode_1k | 1024 | 0.0135 | 0.0000 | 0.0000 | 0.0069 | 0.0014 | 0.0029 | `wait<-reader` |
| decode_2k | 2048 | 0.0227 | 0.0000 | 0.0000 | 0.0069 | 0.0031 | 0.0054 | `wait<-reader` |
| decode_4k | 4096 | 0.0411 | 0.0020 | 0.0020 | 0.0069 | 0.0069 | 0.0121 | `wait<-reader` |
| decode_8k | 8192 | 0.0777 | 0.0074 | 0.0070 | 0.0071 | 0.0151 | 0.0269 | `wait<-reader` |
| decode_16k | 16384 | 0.1431 | 0.0207 | 0.0189 | 0.0069 | 0.0315 | 0.0557 | `wait<-reader` |
| decode_32k | 32768 | 0.2833 | 0.0461 | 0.0420 | 0.0069 | 0.0647 | 0.1166 | `wait<-reader` |

这一步带来的理解比“reader/writer 很忙”更具体：

- 绝对值最大的 floor 一直是 `reader active`。
- 最大的**耦合项**在 full sweep 的所有 case 上都是 `writer wait <- reader`。
- `reserve<-compute` 和 `reserve<-writer` 在 `decode_256 -> decode_2k` 基本还是零，但从 `decode_4k` 开始变成不可忽略，并在长序列下持续抬升。
- 也就是说，BH 上当前 A 路径更像是：reader 先形成一个很厚的 active floor，然后这个 floor 继续传导成 writer 的等待；`compute` 仍然没有消失，但它已经不是最先决定 scaling 形状的因素。

这也解释了为什么二阶 active-core sweep 很早就进入平台：

- 你继续加 cores，主要改善的是 `compute`
- 但真正锁住曲线的却是 `reader active + writer wait<-reader`
- 所以 `A-BH` 的下一步优化重点，已经不是“再给更多 active cores”，而是要改变 reader / writer 之间的传播式 backpressure

### 5.2.3 代码层下一轮 marker 设计已经落好

上面 `5.2.2` 里的归因仍然是经验拆分，它的价值在于帮助解释曲线形状；但从 decode kernel 的实际代码路径看，下一轮已经可以把一部分“软归因”替换成更直接的 source marker。

这轮我已经把 profiling 代码往这个方向补了一层，下一次重跑 detailed sweep 时，重点会多出两组 marker：

reader source marker：

- `SDPA-K-RESERVE-SUM`
- `SDPA-K-ISSUE-SUM`
- `SDPA-K-WAIT-SUM`
- `SDPA-K-PUSH-SUM`
- `SDPA-V-RESERVE-SUM`
- `SDPA-V-ISSUE-SUM`
- `SDPA-V-WAIT-SUM`
- `SDPA-V-PUSH-SUM`

writer source marker：

- `SDPA-WRITER-SENDER-CB-WAIT-SUM`
- `SDPA-WRITER-ROOT-CB-WAIT-SUM`
- `SDPA-WRITER-TREE-CHILD-WAIT-SUM`
- `SDPA-WRITER-OUTPUT-GATHER-WAIT-SUM`

这组 marker 的意义，比上一版 purely statistical 的二阶归因更接近真实代码来源：

- `reader reserve` 不再只是抽象的 “reader / writer 耦合项”
- 对当前 decode 路径来说，它更直接对应 `cb_k_in / cb_v_in` 等待 compute 消费腾出空间
- 换句话说，reader 的 `reserve/block` 更像是 **compute-consumption bound / L1 buffer turnover**，而不是直接等待 writer drain

writer 侧也终于能把几类等待拆开：

- `sender_cb_wait`：sender 在等本地 partial output ready
- `root_cb_wait`：root 在等最终可写出的 `cb_out`
- `tree_child_wait`：root/parent 在等 child 把 reduction 结果送到中间 buffer
- `output_gather_wait`：sharded GQA 输出核在等 reducer 核都到齐

因此下一轮 rerun 之后，下面这两个判断就能从“经验解释”推进成“直接计数”：

1. `reader reserve` 到底主要卡在 `K` 还是 `V`
2. writer 的等待里，到底是 `sender local ready`、`tree child arrival`，还是 `final output gather` 在主导

### 5.3 原生 Flash-MLA 实现 B on BH

采用原生 `8 block x 8 cores` 的 B 布局：

| case | seq_len | K DRAM (MB) | dram ms | q fanout ms | k mcast ms | compute ms | reduce ms | core ms | 主瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode_256 | 256 | 0.1406 | 0.0003 | 0.0037 | 0.0005 | 0.0001 | 0.0011 | 0.0054 | k_mcast |
| decode_512 | 512 | 0.2812 | 0.0006 | 0.0037 | 0.0010 | 0.0002 | 0.0011 | 0.0058 | k_mcast |
| decode_1k | 1024 | 0.5625 | 0.0012 | 0.0037 | 0.0019 | 0.0004 | 0.0011 | 0.0068 | k_mcast |
| decode_2k | 2048 | 1.1250 | 0.0023 | 0.0037 | 0.0038 | 0.0008 | 0.0011 | 0.0087 | k_mcast |
| decode_4k | 4096 | 2.2500 | 0.0046 | 0.0037 | 0.0077 | 0.0016 | 0.0011 | 0.0126 | k_mcast |
| decode_8k | 8192 | 4.5000 | 0.0092 | 0.0037 | 0.0154 | 0.0032 | 0.0011 | 0.0202 | k_mcast |
| decode_16k | 16384 | 9.0000 | 0.0184 | 0.0037 | 0.0307 | 0.0064 | 0.0011 | 0.0356 | k_mcast |
| decode_32k | 32768 | 18.0000 | 0.0369 | 0.0037 | 0.0614 | 0.0129 | 0.0011 | 0.0663 | k_mcast |

把它和 `A-BH` 的一阶/二阶模型放到一起对比，full sweep 如下（`>1x` 表示 `B-BH` 更快）：

| case | seq_len | A-BH ideal ms | A-BH empirical ms | B-BH ms | `B vs A ideal` | `B vs A empirical` |
|---|---:|---:|---:|---:|---:|---:|
| decode_256 | 256 | 0.0027 | 0.0069 | 0.0054 | 0.51x | 1.28x |
| decode_512 | 512 | 0.0043 | 0.0095 | 0.0058 | 0.74x | 1.63x |
| decode_1k | 1024 | 0.0075 | 0.0135 | 0.0068 | 1.11x | 1.99x |
| decode_2k | 2048 | 0.0140 | 0.0227 | 0.0087 | 1.61x | 2.60x |
| decode_4k | 4096 | 0.0269 | 0.0450 | 0.0126 | 2.14x | 3.59x |
| decode_8k | 8192 | 0.0527 | 0.0922 | 0.0202 | 2.60x | 4.56x |
| decode_16k | 16384 | 0.1043 | 0.1826 | 0.0356 | 2.93x | 5.13x |
| decode_32k | 32768 | 0.2074 | 0.3715 | 0.0663 | 3.13x | 5.60x |

结论：

- B-BH 的主瓶颈在 full sweep 上都不再是 DRAM，而是 `K multicast`。
- 但如果和**一阶** `A-BH` 比，`decode_256 / 512` 这两个短序列点上 B-BH 还没有占优，因为固定的 `q fanout + reduce` 成本更明显；大约从 `decode_1k` 开始才反超。
- 如果和**经验二阶** `A-BH` 比，B-BH 在 full sweep 上都更快，而且优势会从 `decode_256` 的 `1.28x` 持续扩大到 `decode_32k` 的 `5.60x`。
- `32k` 下，`A-BH` 一阶是 `0.2074 ms`，经验二阶是 `0.3715 ms`，而 `B-BH = 0.0663 ms`；这也是为什么长序列下结构差距会越来越大。

## 6. 总结

### 6.1 对当前 WH 实现

- `decode_256 / 512 / 1k`：compute 仍在 critical path。
- `decode_2k / 4k / 8k`：writer 已经贴近 critical path。
- `decode_16k`：reader 也开始贴近 critical path。
- `decode_32k`：reader + writer + compute 三条主路径几乎全部饱和。
- decode reader 分解表明，`decode_256 -> decode_2k` 都还是 `issue` 主导，`reserve share` 维持在 `0.7%`。
- `decode_4k` 是 reader 内部最明确的迁移点：`reserve share` 直接跳到 `39.8%`，从这里开始 reader 明显被 `cb_reserve_back` 反压。
- `decode_8k -> decode_32k` 时，reader 内部最大的 stall 已经稳定是 `reserve/block`，`reserve share` 持续抬升到 `61.5%`。
- decode writer 分解表明，BRISC 的“贴近 critical path”主要体现为 `cb_wait_front`，而不是写 issue/bandwidth 本身；`cb_wait share` 会从 `89.4%` 一路升到 `99.8%`。
- guarded decode rerun 还进一步把 compute-side bubble counter 补齐了：bubble 内部的 `wait-front share` 会从 `87.4%` 降到 `70.9%`，`reserve-back share` 会从 `12.6%` 升到 `29.1%`；但这组值需要按 `stall density` 而不是 wall-time share 来解读。
- prefill 从 `prefill_256` 开始就已经是 `reader_writer_saturated`。
- prefill reader 始终主要在 `wait`，share 约 `70.3% -> 66.1%`；writer 则几乎始终主要在 `cb_wait`，share 约 `91.3% -> 99.6%`。

因此，对当前 WH Flash-MLA 的最准确表述不是“纯 compute-bound”，而是：

- **decode 短序列偏 compute-critical**
- **decode 中长序列进入 deeply pipelined、强耦合的 reader/writer/compute 贴边状态**
- **decode reader 内部的主导 stall 从 issue 迁移到 downstream backpressure**
- **decode writer 内部则几乎始终以 `cb_wait_front` 为主，说明它主要在等上游产出**
- **prefill 更早进入 reader/writer/compute 同步饱和状态**

### 6.2 对 BH

- 如果只看一阶 `compute vs dram` 模型，那么当前实现 A 搬去 BH 之后，在 `16 active cores` 假设下，`decode_256 -> decode_32k` 全 sweep 都会先被 compute 压住。
- 这个一阶 `compute/dram crossover` 在 full sweep 上基本稳定在 `23 ~ 24 active cores`。
- 但把 WH detailed profile 里已经观测到的 `reader reserve + writer cb_wait` 回灌进去后，A-BH 的二阶经验值会比一阶理想值高出约 `1.62x ~ 2.53x`。
- 更关键的是，二阶模型里 `compute` 大约在 `4 ~ 10 active cores` 就退出 critical path，而平台区会在 `4 ~ 17 cores` 之间逐步形成。
- 因而 A-BH 很快就不再体现出明显的 active-core scaling，而是转成 reader / writer floor 主导。
- 这个 floor 里最厚的绝对项是 `reader active`，而最大的耦合项在 full sweep 上一直都是 `writer wait <- reader`。
- 从代码路径上看，reader 的 `reserve/block` 更像 `cb_k_in / cb_v_in` 被 compute 消费不够快，而不是直接等待 writer drain。
- 如果使用原生实现 B，则 BH 上的主瓶颈会变成 K multicast；相对经验二阶 `A-BH`，它在 full sweep 上都更快，优势约 `1.28x -> 5.60x`。
- 相对一阶 `A-BH`，B-BH 则是在 `decode_256 / 512` 这两个短序列点上还不占优，大约从 `decode_1k` 开始反超，并在 `decode_32k` 扩大到约 `3.13x`。
- 所以 BH 上真正值得做的不是“单纯移植 A”，而是推进更接近 B 的原生 Flash-MLA 拓扑。
- 现在这版 BH 分析已经不再只是 decode 的一阶比较，而是开始把 `writer cb_wait` 和 `reader reserve` 这类流水线耦合项显式纳入 A-BH 模型。

## 7. 下一步建议

1. 继续保留现在这版 detailed stage-level profile 作为主基线，因为 `kernel / reader stages / writer stages` 这一层已经完整。
2. 用这轮已经补好的 source marker 重跑 `profile_flash_mla_wh_detailed.py`，把 `K/V reserve` 和 `sender/root/tree/output wait` 的直接计数拉出来，补齐 `source-level attribution` 这一层。
3. 再补一轮真正可解释的 `PM/NOC/DRAM util` rerun，把当前 `PM IDEAL/COMPUTE/BANDWIDTH=1.0 ns` 和 `NOC/DRAM missing` 这层 placeholder/missing 状态替换成稳定值。
4. 用新的 source marker、稳定的 PM util 和这轮已经拿到的 compute bubble counter 回写 BH 二阶模型，把当前的软归因 `wait<-reader / reserve<-compute` 替换成更接近 direct counter 的项，并校准这版模型里 `1.62x ~ 2.53x` 的惩罚系数是否稳定。
5. 如果要推进 B-WH 或 B-BH 的工程实现，下一步最值得优先验证的是 `K multicast` 的 NoC 利用率和 sender core 的饱和点。

## 8. 与 Autotuner 的衔接

这轮 profiling 工作现在已经不再只是分析文档输入，而是开始直接反哺 `mla_flash_attention_dev/autotuner/basic_autotuner.py`。

### 8.1 已经落进 autotuner 的观测

根据这份 WH profile 里看到的阶段迁移，autotuner 里已经新增了三类更贴近运行时 stall 的指标：

- `reader_backpressure_ms`
- `writer_backpressure_ms`
- `sender_hotspot_ms`

它们分别对应：

- reader 从 `issue` 向 `reserve/block` 迁移的趋势
- writer 长时间卡在 `cb_wait_front` 的趋势
- sender/injector 在更强 multicast / reduction 拓扑下的热点风险

下一轮如果用带新 marker 的 detailed JSON 重新 bridge，measurement metadata 里还会继续多出更细的 source share，例如：

- `profile_reader_k_reserve_share / profile_reader_v_reserve_share`
- `profile_writer_sender_cb_wait_share / profile_writer_root_cb_wait_share`
- `profile_writer_tree_child_wait_share / profile_writer_output_gather_wait_share`

也就是说，profile 里的结论现在已经开始变成 plan ranking 的一部分，而不再只是文字判断。

### 8.2 现在可以直接把 WH profile 转成 measurement DB

autotuner 新增了一个桥接入口，可以直接把这份 profile JSON 转成 `MeasurementDB`：

```bash
python -m mla_flash_attention_dev.autotuner \
  --build-wh-profile-measurement-db \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_results.json \
  --measurement-db-output \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_measurement_db.json
```

如果你想桥接本文现在主用的 full-sweep 数据，优先用 detailed 版：

- `flash_mla_wh_detailed_profile_results.json`
- 它同时包含 `prefill_256 -> prefill_4k` 和 `decode_256 -> decode_32k`

如果只是想桥接旧版 3-point quick smoke，simple 版也仍然支持：

- `flash_mla_wh_profile_results.json`

对应的 legacy/simple bridge 例子可以写成：

```bash
python -m mla_flash_attention_dev.autotuner \
  --build-wh-profile-measurement-db \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh/flash_mla_wh_profile_results.json \
  --measurement-db-output \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh/flash_mla_wh_measurement_db.json
```

默认桥接模式是：

```text
reference_current_a
```

它的含义是：

- profile latency 优先绑定到一个更接近当前 production A 路径的候选 key
- 不会默认把 WH 上测到的 A 路径真值，直接覆盖到 B-like analytical top-1
- 并且现在会优先贴合 profile harness 里已经明确给出的 A 路径特征，例如 `HiFi4`、`independent`、`k_chunk_size=128`，以及 decode 下的 `4 lanes x 4 cores/lane` 风格

实现上，这一步也已经被收敛成显式的 reference 配置对象；因此 bridge 生成的 measurement DB 不只会存 `latency_ms`，还会带上 reference 目标值和 analytical 映射信息，后面回查时更容易判断映射有没有跑偏。

如果你只是想做一种更激进的“把 profile latency 直接挂到当前 analytical 最优候选”的实验，也可以显式切到：

```bash
--profile-measurement-selection-mode analytical_best
```

如果你想控制 reference A-path 是更贴近 profiling harness，还是更贴近 `mla1d.py` 的 production 默认配置，也可以进一步指定：

```bash
--profile-measurement-reference-source profile_harness
--profile-measurement-reference-source mla1d_defaults
--profile-measurement-reference-source hybrid_auto
```

其中 `mla1d_defaults` 当前对应的是：

- decode：`q_chunk_size=0`、`k_chunk_size=128`、`HiFi4`
- prefill：`q_chunk_size=128`、`k_chunk_size=128`、`HiFi4`

### 8.3 这样接入 autotuner rerank

生成 measurement DB 后，就可以对 exact profiled workload 开 rerank：

```bash
python -m mla_flash_attention_dev.autotuner \
  --workload-json workload.json \
  --hardware-json hardware.json \
  --measurement-db mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_measurement_db.json \
  --rerank-top-k 64
```

这里要注意一个边界：

- `reference_current_a` 更适合做 “把当前 production path 的真实 latency 绑定到对应候选”
- 它不是在声称 analytical top-1 已经被真实测过
- 当前 rerank 会把 analytical top-K 和当前 exact workload 已测 candidate 一起纳入 rerank pool，但只有当 measured latency 本身更优时，它才会真的改写 best plan
- 所以 `measurement_db` 更像“把真实 A-path 基线带回候选池”，而不是“只要命中 measurement 就一定压过 analytical 最优”
- 如果后面你要做真正的 top-K measured reranking，最理想的形态仍然是 benchmark harness 直接按 candidate 去实跑

### 8.4 这轮衔接工作的意义

这意味着当前已经形成了一条更完整的闭环：

1. `profile_flash_mla_wh*.py` 产出实测阶段分解
2. 实测结论进入 cost model，影响 analytical ranking
3. profile JSON 还能继续桥接成 `measurement_db`
4. profile JSON 现在还能直接拟合 calibration model
5. autotuner 可以在 exact workload 上利用这些测量值做 rerank，或者在更一般 workload 上利用 calibration 调整 ranking

换句话说，这轮之后，profile 和 autotuner 之间已经不再是“文档上的弱连接”，而是开始具备真正的数据闭环。

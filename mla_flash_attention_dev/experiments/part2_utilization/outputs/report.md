# Part II 利用率与瓶颈归因

## 1. Part I 对齐口径

- Part I 现在把 `DeepSeek FlashMLA` 固定为 `decode` 主方法；本目录中的 dense stage/source/bubble characterization 仍主要来自 `TT 主线 FlashMLA` 的 profiler 路径，因此它服务的是 `mechanism proxy`，不是 direct DeepSeek profile。
- 代理对齐检查显示：`decode_1k` 上 `DeepSeek / TT` latency ratio = `0.894x`，说明短序列阶段形态仍可能不完全一致。
- 但在 Part I 真正最需要承接机理解释的 long-seq 区间，`decode_4k / 16k / 32k` 的 `DeepSeek / TT` ratio 已分别收敛到 `0.956x / 0.973x / 0.994x`，最大偏差约 `4.4%`。
- 因此 Part II 的推荐读法是：`DeepSeek FlashMLA` 负责解释 Part I 主图里的性能现象，`TT 主线 FlashMLA` 提供 dense long-seq 机理 proxy；这两者需要同时保留，不能互相替代。

## 2. 数据范围

- `decode` full sweep：来自 `flash_mla_pm_bubble_probe_decode_safe` 的 `256 / 512 / 1k / 2k / 4k / 8k / 16k / 32k`。
- `decode` source-level representative points：`decode_1k` 来自 `sync_minimal`，`decode_4k / decode_32k` 来自 `repr_sync`，`decode_16k` 来自 `sync_extra`。
- `prefill` 控制对照：`prefill_256 / 512 / 1k / 2k` 来自 `sync_extra`，`prefill_4k` 来自 `repr_sync`。
- `A1/A2/A3` paired direct device-only representative points：来自 `deepseek_flash_mla_part2_probe` 的 `TT vs DeepSeek-4c`，当前可直接读取 `DEVICE KERNEL DURATION / WAIT FRONT / RESERVE BACK`，同一批 raw log 现在也能离线补出 `reader/writer` source marker。
- `B/C/D` 的 `4c/8c` paired direct device-only points：来自同一条 `deepseek_flash_mla_part2_probe` `device_only` 链路；`B/C` 用来回答“已 fit 点上 `8c` 会不会更快”，`D1/D2` 用来把 `4c unsupported -> 8c supported` 的容量边界正式钉住。
- 同一批 `B/C/D` artifacts 现在还能离线补出 reader / writer / reduction 三侧 source-level direct evidence：`TT` 与 `DeepSeek-4c/8c` 都能稳定读到 reader `K/V` 与 writer `sender / root / tree / output` marker；`D1/D2` 上 `DeepSeek-4c` 的 `n/a` 则来自 unsupported_precheck，而不是 marker 缺失。
- `24q/28q/32q/40q/48q` 的 corrected-runtime `perf-fpu` boundary points：来自 `deepseek_flash_mla_part2_probe_pm_fix_verify_boundary_perf_fpu_rerun` 与 `deepseek_flash_mla_part2_probe_pm_fix_verify_boundary_mid_perf_fpu_rerun` 的 `TT / DeepSeek-4c / DeepSeek-8c`；当前 densified q-core 轴为 `24q/28q/32q/40q/48q`，其中 `24q` 是已 fit anchor，其余点用于观察跨墙后 `8c` 的直接算术利用率变化。
- 为了收口 decode 侧 `PM/perf-counter` 直采，先额外补了跨 `DeepSeek-4c / TT` 的六条 `A1` smoke：初始现象是 `tracy_report + sum / no-runtime-analysis / perf-fpu + sync` 仍只给出 `PM IDEAL / COMPUTE / BANDWIDTH = 1 / 1 / 1` 或直接缺列。后续定位到两层根因：`SdpaDecodeDeviceOperation` 缺少 `create_op_performance_model`，以及 runtime 实际仍加载旧的 `ttnn/ttnn/_ttnn.so -> build_Release/lib/_ttnncpp.so`。在补齐 decode perf model 并重新 `cmake --install build_Release --component tt_pybinds` / `tar` 后，修正 runtime 的 `A1__tt` / `A1__deepseek4c` rerun 已稳定给出 `PM IDEAL / COMPUTE / BANDWIDTH = 8704 / 8704 / 2129`；对应的 `perf-fpu` rerun 还会同时产出 `SFPU/FPU/MATH` 利用率列。 同一条 corrected-runtime `perf-fpu` 链路现已扩到 `1k/4k/8k/16k/32k`，`PM FPU util (TT / DeepSeek-4c)` = `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` = `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%。
- 口径说明：`decode_256~8k` 为 `batch=2`，`decode_16k/32k` 为 `batch=1`；这与实验计划中 Part II 的 mixed-batch characterization 用途一致。

## 3. 核心结论

- `decode` 的阶段迁移已经固定：`256~1k` 为 `compute_on_critical_path`，`2k~8k` 为 `writer_close_to_critical_path`，`16k` 为 `reader_close_to_critical_path`，`32k` 为 `reader_writer_saturated`。
- 更直观地看，代表点 `decode_1k / 4k / 16k / 32k` 的 `compute share` 已接近 `99.68% / 99.86% / 99.96% / 99.98%`；此外，修正 runtime 后的 representative seq direct rerun（`1k/4k/8k/16k/32k`）现在已经把 decode 侧算术利用率直接量化出来：`PM FPU util (TT / DeepSeek-4c)` 为 `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` 为 `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%。这说明 decode 算术利用率会随序列拉长从 low-teens 抬升到 mid-20s，但两条实现的 `PM FPU util` 差值始终不超过 0.46 pp，不支持“已经接近算术饱和”或“SF-MLA 主要靠算术利用率跃升取胜”的解读。
- `A1` smoke + corrected-runtime representative rerun 已把 decode PM 路径状态钉住：初始六条 smoke 用来暴露问题，表现为 `DeepSeek-4c` 的 `tracy_report + sum / no-runtime-analysis / perf-fpu + sync` 与 `TT` 的 `tracy_report + sum / perf-fpu + sync` 都会回落到 placeholder / 缺列；在补齐 `SdpaDecodeDeviceOperation::create_op_performance_model` 并同步 runtime 之后，`TT` 与 `DeepSeek-4c` 的 `tracy_report + sum + sync` / `tracy_report + perf-fpu + sync` 都已经恢复到非-placeholder PM，后者还能直接产出硬件 `FPU / SFPU / MATH` 列。 同一条 `perf-fpu` 链路扩到 `1k/4k/8k/16k/32k` 后，`PM FPU util (TT / DeepSeek-4c)` 也已形成 representative trend：`1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%。
- `decode_4k` 是关键迁移点：reader `reserve share` 从 `decode_1k` 的 `1.1%` 抬升到 `32.0%`，writer `cb_wait` 同时保持在 `98.0%`。
- `decode` 的 compute bubble 始终以 `wait-front` 为主，但 `reserve-back share` 会从 `decode_1k` 的 `13.9%` 升到 `decode_32k` 的 `29.1%`，说明 long-seq stall 越来越带有 downstream backpressure 成分。
- 新补的 `A1/A2/A3` paired direct device-only probe 已经给出一条更直接的 supporting evidence：`DeepSeek-4c` 在 `1k / 4k / 32k` 的 device kernel duration 为 `70.36 / 171.26 / 1110.67 us`，对应 `TT` 为 `73.86 / 174.67 / 1114.36 us`；同时 `wait-front / kernel` 从 `TT` 的 `108.1% / 87.2% / 74.5%` 降到 `DeepSeek-4c` 的 `94.9% / 81.3% / 73.2%`。
- `A2/A3` 同一批 raw log 现在也把 `SF-MLA` 的 direct source-level supporting evidence 补出来了：`DeepSeek-4c` 的 `K reserve` 在 `4k / 32k` 上为 `28.59% / 51.06%`，对应 `TT` 为 `31.48% / 51.40%`；`V reserve` 则从 `TT` 的 `2.25% / 2.62%` 降到 `DeepSeek-4c` 的 `0.56% / 0.39%`，而 `sender/tree` 基本维持同一结构。
- 新补的 `B/C` direct `4c vs 8c` 结果又把一件事钉得很清楚：`B1~C3` 六个点的 `8c / 4c kernel` 只落在 `0.995x~1.009x`；这些点都满足 `batch * q_shards = 24`，说明当 workload 已经 fit 进 `4c` 容量时，单纯扩到 `8c` 并不会自动更快。
- 新补的 `B/C` reader+writer source-level direct 又把“为什么 `8c≈4c`”往前推了一步：`TT` 的 `K reserve` 落在 `0.23%~30.87%`，但 `DeepSeek-4c/8c` 都只在 `0.11%~0.13% / 0.11%~0.13%`；`8c-4c` 的 `K reserve` 与 `tree_child_wait` 差值最大也只有 `0.00 pp / 0.09 pp`。这说明在已 fit 点上，`8c` 并没有改写 reader 供给或 writer / reduction 的主等待结构。
- `D1/D2` 已把容量边界正式钉住：`required_q_cores = 32 / 48` 时，`4c` 会直接 preflight unsupported，而 `8c` 可以跑通，对应的 `8c / TT` kernel ratio 为 `0.865x / 0.983x`。
- 新补的 boundary corrected-runtime `perf-fpu` 又把这条边界接到了同一条 direct arithmetic-utilization 轴上：`TT PM FPU util @ 24q/28q/32q/40q/48q = 24q 21.86%; 28q 21.60%; 32q 21.27%; 40q 20.36%; 48q 18.21%`；`DeepSeek-8c PM FPU util @ 24q/28q/32q/40q/48q = 24q 12.08%; 28q 13.78%; 32q 13.74%; 40q 16.40%; 48q 18.53%；`DeepSeek-4c` 则只在 `<= 24q` 仍可运行，`28q` 起直接 `unsupported_precheck`。因此 `8c` 的第一作用是把 `4c` 无法 admission 的点重新变成可运行，而不是先触发一个算术利用率跳变。
- 但容量墙不是终点：`D2` 的 `DeepSeek-8c` 虽然已经可跑，`reserve-back / kernel` 仍有 `276.3%`，说明清掉 q-core admission 之后，主矛盾会继续转到 downstream backpressure。
- `D2` 的 writer source-level direct 也支持这一点：`DeepSeek-8c` 上 `K reserve/sender/tree/root/output = 0.14%/49.41%/49.72%/0.88%/0.00%`。也就是说，跨过容量墙之后 reader 侧已经不是主问题，剩余压力仍主要堆在 sender / tree reduction 链路，而不是 final gather。
- `decode` 的 source-level attribution 已经够强：reader 的 `K reserve` 从 `decode_1k` 的 `0.37%` 升到 `decode_32k` 的 `54.82%`，而 `V reserve` 在 `decode_32k` 仍只有 `2.80%`。
- 代表点总表还能看到 `K` 供给瓶颈的增强：`effective single-pass K read bandwidth` 从 `decode_1k` 的 `39.52` 下降到 `decode_32k` 的 `18.00`，但与此同时 `K reserve` 也从 `0.37%` 升到 `54.82%`，说明问题不只是“有没有在读”，而是“读侧供给是否能持续跟上整个流水线”。
- `writer` 的主等待不在 final gather：四个代表点上 `sender_cb_wait` 与 `tree_child_wait` 基本五五开，`decode_32k` 分别为 `49.95% / 50.05%`。
- `prefill` 从起点就是强耦合饱和流水线：`PM FPU util` 从 `prefill_256` 的 `10.20%` 升到 `prefill_4k` 的 `19.43%`，但同期 `writer cb_wait` 仍从 `91.50%` 升到 `99.62%`。

## 4. 利用率 / Bubble / 供给与回压分析

- **如何读当前的“利用率”**：`decode_1k / 4k / 16k / 32k` 的 `compute share` 已达到 `99.68% / 99.86% / 99.96% / 99.98%`。现在，修正 runtime 后的 representative seq rerun（`1k/4k/8k/16k/32k`）也已经给出 decode 侧 direct arithmetic-utilization curve：`PM FPU util (TT / DeepSeek-4c)` = `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` = `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%。因此这里的“高利用率”不能写成“算术吞吐已经接近峰值”；更准确的说法是：full-sweep 的 `compute share` 表示 compute thread 始终贴着 kernel critical path，而 representative-point direct PM/counter 则说明 decode 的算术利用率会从 low-teens 迁移到 mid-20s，但仍远未接近峰值。
- **Bubble 是怎么迁移的**：`wait-front share` 在 `decode_1k -> 32k` 上从 `86.1%` 下降到 `70.9%`，而 `reserve-back share` 从 `13.9%` 升到 `29.1%`；与此同时，`decode_4k` 已出现 `reader reserve=32.0%` 与 `writer cb_wait=98.0%` 的同步抬升。说明短序列主要是 front-side starvation，长序列则逐步演化成 front-side starvation 与 downstream backpressure 并存的 stalled pipeline。
- **供给哪里不好**：reader 内部最坏的不是 `V`，而是 `K`。`K issue` 从 `decode_1k` 的 `56.87%` 下降到 `decode_32k` 的 `26.10%`，`K reserve` 从 `0.37%` 升到 `54.82%`；相比之下，`V reserve` 到 `decode_32k` 仍只有 `2.80%`。再结合 `effective single-pass K read bandwidth` 从 `39.52` 下降到 `18.00`，可以更稳妥地写成：long-seq 的主问题不是“KV 都在一起变慢”，而是 `K` 路持续供给能力不足，reader 越来越多地卡在 reserve / slot 可用性等待上，而不是忙于真正 issue 新请求。
- **回压哪里不好**：writer 侧 `cb_wait` 几乎全程贴边，从 `decode_1k` 的 `93.87%` 升到 `decode_32k` 的 `99.78%`；source-level 上 `sender/tree` 在代表点上始终接近五五开，`decode_32k` 为 `49.95%/50.05%`，`root/output` 近乎为 `0`。再结合 `D2` 的 `DeepSeek-8c reserve-back / kernel=276.3%` 与 `sender/tree/root/output=49.41%/49.72%/0.88%/0.00%`，可以看出跨过 q-core 容量墙之后，残余主瓶颈仍然堆在 sender / tree reduction 链路，而不是 final gather。
- **为什么会慢**：把供给与回压合在一起看，当前 `TT-MLA` 的 slowdown 本质上是双向挤压。前端 `K` 供给跟不上，compute 更容易出现 `wait-front`；后端 reduction / output 路径排不空，compute 又承担越来越重的 `reserve-back`。因此表面上看到的是 compute 始终贴着 critical path，但根因其实是供给与回压共同放大了 bubble，这也是阶段会从 `compute_on_critical_path` 依次迁移到 `writer_close_to_critical_path`、`reader_close_to_critical_path`，最终走到 `reader_writer_saturated` 的原因。

## 5. 问题导向摘要

- **为什么当前 `TT-MLA` 不好**：现有 direct proxy 已经足够强。`decode_1k -> 32k` 上，`reader reserve` 从 `1.13%` 升到 `57.58%`，`writer cb_wait` 从 `93.87%` 升到 `99.78%`，`K reserve` 从 `0.37%` 升到 `54.82%`，因此当前实现的主要问题是供给与回压，而不是算子数学本身。
- **为什么 `SF-MLA` 更好**：现在已经有 `A1/A2/A3` 的 paired direct device-only evidence。`DeepSeek-4c` 在三个代表点上的 kernel duration 分别为 `70.36 / 171.26 / 1110.67 us`，对应 `TT` 为 `73.86 / 174.67 / 1114.36 us`；`wait-front / kernel` 也从 `TT` 的 `108.1% / 87.2% / 74.5%` 降到 `DeepSeek-4c` 的 `94.9% / 81.3% / 73.2%`。`A2/A3` 的 raw log 还显示 `DeepSeek-4c` 的 `K reserve` 为 `28.59% / 51.06%`，对应 `TT` 为 `31.48% / 51.40%`。与此同时，`B/C` 六个 `4c vs 8c` 点的 `8c / 4c kernel` 只在 `0.995x~1.009x`，`B/C` reader `K reserve` 与 writer `sender/tree` 也保持在近乎不变的结构：`4c` 为 `K reserve=0.11%~0.13%`、`sender/tree=41.73%~43.83% / 55.96%~57.56%`，`8c` 为 `K reserve=0.11%~0.13%`、`sender/tree=41.71%~43.82% / 55.97%~57.58%`。说明这件事不能简单解释为“用了更多核”；当 workload 已经 fit 进 `4c` 容量时，多出来的 `8c` headroom 不会自动变成速度。新的 corrected-runtime `perf-fpu` rerun（`1k/4k/8k/16k/32k`）又补上了 decode 算术利用率的 direct curve：`PM FPU util (TT / DeepSeek-4c)` = `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` = `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%，且两条实现的 `PM FPU util` 差值始终不超过 0.46 pp。因此当前更合理的 direct 写法是：`SF-MLA` 的优势首先来自 mapping / dataflow 对 front-side starvation 的缓解，而不是把 decode 算术利用率抬到一个截然不同的饱和区间；更多核心主要在跨越容量墙时才成为必要条件，而不是所有点上的普适速度来源。
- **为什么 `SF-MLA` 仍有改进空间**：direct device-only 结果已经说明“更好”不等于“问题消失”。`DeepSeek-4c` 在 `A2/A3` 上的 `reserve-back / kernel` 仍有 `26.8% / 30.1%`，`D2` 的 `DeepSeek-8c reserve-back / kernel` 也仍有 `276.3%`，`D2` 的 writer source-level 也显示 `sender/tree = 49.41%/49.72%`，而 proxy 侧的 `decode_32k` 仍显示 `K reserve=54.82%`、`sender/tree=49.95%/50.05%`。这说明即便换更优 mapping，也不会自动消除 `K` 供给与 reduction backpressure。
- **`4c / 8c` 的边界是什么**：现在已经可以 direct 地写。`B/C` 六个点都满足 `batch * q_shards = 24`，`8c / 4c kernel` 只在 `0.995x~1.009x`；一旦进入 `D1/D2` 的 `32 / 48`，`4c` 就会直接 unsupported，`8c` 则恢复可运行。同时 `B/C` 的 reader `K reserve` 与 writer `sender/tree` 也都显示 `4c/8c` 仍停留在几乎相同的结构，新的 boundary corrected-runtime `perf-fpu` rerun 又补上了同一条轴上的算术利用率说明：`TT PM FPU util @ 24q/28q/32q/40q/48q = 24q 21.86%; 28q 21.60%; 32q 21.27%; 40q 20.36%; 48q 18.21%`，`DeepSeek-8c PM FPU util @ 24q/28q/32q/40q/48q = 24q 12.08%; 28q 13.78%; 32q 13.74%; 40q 16.40%; 48q 18.53%；`DeepSeek-4c` 则只在 `<= 24q` 仍可运行，`28q` 起直接停在 unsupported_precheck。因此 `8c` 的一阶价值是抬高 q-core 容量上限，而不是在所有点上都天然更快，也不是先把算术利用率抬到另一个区间。

## 6. 论文图片

- Part I proxy alignment：`visuals/decode_deepseek_proxy_alignment.svg`
- Decode 阶段迁移：`visuals/decode_phase_transition.svg`
- Decode reader/writer 反压信号：`visuals/decode_backpressure_signals.svg`
- Decode bubble density：`visuals/decode_bubble_density.svg`
- Decode bubble 总时长与组成：`visuals/decode_bubble_composition.svg`
- Decode reader source attribution：`visuals/decode_reader_source_breakdown.svg`
- Decode writer source attribution：`visuals/decode_writer_source_breakdown.svg`
- 核内泳道图（Reader / Compute / Writer）：`visuals/decode_intra_core_swimlane.svg`
- 核间泳道图（sender / worker / tree / root）：`visuals/decode_inter_core_swimlane.svg`
- Prefill 控制对照：`visuals/prefill_control.svg`

## 7. 论文表格

- Part I 代理对齐表：`tables/part1_proxy_alignment.md`
- Decode 阶段分类总表：`tables/decode_phase_classification.md`
- Decode source-level attribution：`tables/decode_source_attribution.md`
- Decode 利用率 / Bubble / 归因总表：`tables/decode_utilization_bubble_summary.md`
- Decode PM / perf-counter 路径状态：`tables/decode_pm_path_status.md`
- Decode PM / perf-counter 容量边界补证：`tables/decode_pm_boundary_perf_summary.md`
- A1/A2/A3 direct device-only 对照表：`tables/direct_device_only_a123_summary.md`
- 4c / 8c / capacity-boundary 对照表：`tables/direct_device_only_4c_8c_boundary_summary.md`
- 4c / 8c source-level 对照表：`tables/direct_device_only_4c_8c_source_summary.md`
- 问题导向结论边界表：`tables/problem_driven_evidence_boundary.md`
- Prefill 控制对照表：`tables/prefill_control.md`
- 正文段落草稿：`paper_ready_paragraphs.md`

## 8. 原始快照

- 汇总 JSON：`raw/part2_utilization.json`
- Part I 代理对齐 CSV：`raw/part1_proxy_alignment.csv`
- Decode 阶段分类 CSV：`raw/decode_phase_classification.csv`
- Decode source attribution CSV：`raw/decode_source_attribution.csv`
- Decode 利用率 / Bubble / 归因总表 CSV：`raw/decode_utilization_bubble_summary.csv`
- Decode PM / perf-counter 路径状态 CSV：`raw/decode_pm_path_status.csv`
- Decode PM / perf-counter 容量边界补证 CSV：`raw/decode_pm_boundary_perf_summary.csv`
- A1/A2/A3 direct device-only 摘要 CSV：`raw/direct_device_only_a123_summary.csv`
- A1/A2/A3 direct device-only pairwise CSV：`raw/direct_device_only_a123_pairwise.csv`
- 4c / 8c / capacity-boundary 摘要 CSV：`raw/direct_device_only_4c_8c_boundary_summary.csv`
- 4c / 8c / capacity-boundary pairwise CSV：`raw/direct_device_only_4c_8c_boundary_pairwise.csv`
- 4c / 8c source-level 摘要 CSV：`raw/direct_device_only_4c_8c_source_summary.csv`
- 4c / 8c source-level pairwise CSV：`raw/direct_device_only_4c_8c_source_pairwise.csv`
- Prefill 控制对照 CSV：`raw/prefill_control.csv`
- 运行 manifest 快照：`raw/manifests/`
- 上游分析报告快照：`raw/source_reports/`


# Part II 利用率与瓶颈归因

## 1. Part I 对齐口径

- Part I 现在把 `DeepSeek FlashMLA` 固定为 `decode` 主方法；本目录中的 dense stage/source/bubble characterization 仍主要来自 `TT 主线 FlashMLA` 的 profiler 路径，因此它服务的是 `mechanism proxy`，不是 direct DeepSeek profile。
- 代理对齐检查显示：`decode_1k` 上 `DeepSeek / TT` latency ratio = `0.864x`，说明短序列阶段形态仍可能不完全一致。
- 但在 Part I 真正最需要承接机理解释的 long-seq 区间，`decode_4k / 16k / 32k` 的 `DeepSeek / TT` ratio 已分别收敛到 `1.025x / 0.976x / 0.997x`，最大偏差约 `2.5%`。
- 因此 Part II 的推荐读法是：`DeepSeek FlashMLA` 负责解释 Part I 主图里的性能现象，`TT 主线 FlashMLA` 提供 dense long-seq 机理 proxy；这两者需要同时保留，不能互相替代。

## 2. 数据范围

- `decode` full sweep：来自 `flash_mla_pm_bubble_probe_decode_safe` 的 `256 / 512 / 1k / 2k / 4k / 8k / 16k / 32k`。
- `decode` source-level representative points：`decode_1k` 来自 `sync_minimal`，`decode_4k / decode_32k` 来自 `repr_sync`，`decode_16k` 来自 `sync_extra`。
- `prefill` 控制对照：`prefill_256 / 512 / 1k / 2k` 来自 `sync_extra`，`prefill_4k` 来自 `repr_sync`。
- 口径说明：`decode_256~8k` 为 `batch=2`，`decode_16k/32k` 为 `batch=1`；这与实验计划中 Part II 的 mixed-batch characterization 用途一致。

## 3. 核心结论

- `decode` 的阶段迁移已经固定：`256~1k` 为 `compute_on_critical_path`，`2k~8k` 为 `writer_close_to_critical_path`，`16k` 为 `reader_close_to_critical_path`，`32k` 为 `reader_writer_saturated`。
- `decode_4k` 是关键迁移点：reader `reserve share` 从 `decode_1k` 的 `1.1%` 抬升到 `32.0%`，writer `cb_wait` 同时保持在 `98.0%`。
- `decode` 的 compute bubble 始终以 `wait-front` 为主，但 `reserve-back share` 会从 `decode_1k` 的 `13.9%` 升到 `decode_32k` 的 `29.1%`，说明 long-seq stall 越来越带有 downstream backpressure 成分。
- `decode` 的 source-level attribution 已经够强：reader 的 `K reserve` 从 `decode_1k` 的 `0.37%` 升到 `decode_32k` 的 `54.82%`，而 `V reserve` 在 `decode_32k` 仍只有 `2.80%`。
- `writer` 的主等待不在 final gather：四个代表点上 `sender_cb_wait` 与 `tree_child_wait` 基本五五开，`decode_32k` 分别为 `49.95% / 50.05%`。
- `prefill` 从起点就是强耦合饱和流水线：`PM FPU util` 从 `prefill_256` 的 `10.20%` 升到 `prefill_4k` 的 `19.43%`，但同期 `writer cb_wait` 仍从 `91.50%` 升到 `99.62%`。

## 4. 论文图片

- Part I proxy alignment：`visuals/decode_deepseek_proxy_alignment.svg`
- Decode 阶段迁移：`visuals/decode_phase_transition.svg`
- Decode reader/writer 反压信号：`visuals/decode_backpressure_signals.svg`
- Decode bubble density：`visuals/decode_bubble_density.svg`
- Decode bubble 组成：`visuals/decode_bubble_composition.svg`
- Decode reader source attribution：`visuals/decode_reader_source_breakdown.svg`
- Decode writer source attribution：`visuals/decode_writer_source_breakdown.svg`
- Prefill 控制对照：`visuals/prefill_control.svg`

## 5. 论文表格

- Part I 代理对齐表：`tables/part1_proxy_alignment.md`
- Decode 阶段分类总表：`tables/decode_phase_classification.md`
- Decode source-level attribution：`tables/decode_source_attribution.md`
- Prefill 控制对照表：`tables/prefill_control.md`

## 6. 原始快照

- 汇总 JSON：`raw/part2_utilization_bottleneck_analysis.json`
- Part I 代理对齐 CSV：`raw/part1_proxy_alignment.csv`
- Decode 阶段分类 CSV：`raw/decode_phase_classification.csv`
- Decode source attribution CSV：`raw/decode_source_attribution.csv`
- Prefill 控制对照 CSV：`raw/prefill_control.csv`
- 运行 manifest 快照：`raw/manifests/`
- 上游分析报告快照：`raw/source_reports/`


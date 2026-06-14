# Decode 代表点利用率 / Bubble / 归因总表（TT 主线 proxy）

说明：
- 这张表只汇总当前 `Part II` 已经有足够证据支撑的 `TT 主线 FlashMLA` representative points，用来回答“为什么当前 `TT-MLA` 方法不好”。
- `DeepSeek / TT` 只表示该代表点作为 `Part I` long-seq mechanism proxy 的贴近程度，不等于 `DeepSeek` 的 direct profiler。
- `compute share` 表示 compute thread 贴近 kernel critical path 的程度；`PM FPU util` 更接近实际算术利用率，两者应结合解读。
- 当前这张 proxy 总表里的 decode `PM FPU util` 仍统一记为 `n/a`，因为它对应的上游 dense TT proxy CSV 依然是 placeholder 或缺失；但 `tables/decode_pm_path_status.md` 中的 corrected-runtime representative seq rerun（`1k/4k/8k/16k/32k`） 已补出 representative-point direct arithmetic-utilization trend：`PM FPU util (TT / DeepSeek-4c)` = `1k` 12.17% / 12.63%；`4k` 20.57% / 20.84%；`8k` 23.11% / 23.34%；`16k` 24.73% / 24.85%；`32k` 25.61% / 25.67%，`Avg FPU util on full grid` = `1k` 10.36% / 10.75%；`4k` 17.13% / 17.35%；`8k` 19.18% / 19.37%；`16k` 20.48% / 20.58%；`32k` 21.19% / 21.24%；两条实现的 `PM FPU util` 差值始终不超过 0.46 pp。

| case | seq_len | batch | classification | DeepSeek / TT | PM FPU util | compute share | reader reserve | writer cb_wait | bubble / kernel | K reserve | sender wait | tree wait | K read GB/s |
|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| decode_1k | 1024 | 2 | compute_on_critical_path | 0.894x | n/a | 99.68% | 1.13% | 93.87% | 2.86x | 0.37% | 48.69% | 51.31% | 39.52 |
| decode_4k | 4096 | 2 | writer_close_to_critical_path | 0.956x | n/a | 99.86% | 31.98% | 97.98% | 2.40x | 31.64% | 49.59% | 50.41% | 42.52 |
| decode_16k | 16384 | 1 | reader_close_to_critical_path | 0.973x | n/a | 99.96% | 55.45% | 99.57% | 1.07x | 52.79% | 49.89% | 50.11% | 18.47 |
| decode_32k | 32768 | 1 | reader_writer_saturated | 0.994x | n/a | 99.98% | 57.58% | 99.78% | 1.05x | 54.82% | 49.95% | 50.05% | 18.00 |

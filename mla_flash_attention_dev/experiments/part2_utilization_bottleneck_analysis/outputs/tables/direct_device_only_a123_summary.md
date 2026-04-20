# A1/A2/A3 Direct Device-Only Summary

说明：

- 这里汇总的是当前已补到的 `A1/A2/A3` direct representative-point probe。
- 数据来源：`mla_flash_attention_dev/experiments/profile_outputs/deepseek_flash_mla_part2_probe/*/reports/ops_perf_results.csv`
- 当前使用的是稳定的 `device_only` 链路，因此本表的主指标仍是 device-side `kernel / wait_front / reserve_back`；但同一批 raw log 现在也会离线补出 `reader/writer` source marker。
- decode 侧 `PM FPU util` 仍不在这张表里；代表点级的 arithmetic-utilization 结论见 `tables/decode_pm_path_status.md` 中 corrected-runtime representative seq rerun（`1k/4k/8k/16k/32k`）部分。
- 表中数值为每个 `ops_perf_results.csv` 内样本的均值。

| case | seq_len | method | kernel_us | wait_front_us | reserve_back_us | wait_front_vs_kernel | reserve_back_vs_kernel | reader src | K reserve | V reserve | sender | tree |
|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| A1 | 1k | DeepSeek-4c | 70.36 | 66.79 | 14.97 | 94.9% | 21.3% | yes | 0.32% | 0.79% | 45.54% | 47.97% |
| A1 | 1k | TT | 73.86 | 79.84 | 15.13 | 108.1% | 20.5% | yes | 0.35% | 0.79% | 45.80% | 48.13% |
| A2 | 4k | DeepSeek-4c | 171.26 | 139.23 | 45.83 | 81.3% | 26.8% | yes | 28.59% | 0.56% | 48.58% | 49.38% |
| A2 | 4k | TT | 174.67 | 152.40 | 46.58 | 87.2% | 26.7% | yes | 31.48% | 2.25% | 48.65% | 49.35% |
| A3 | 32k | DeepSeek-4c | 1110.67 | 813.28 | 334.42 | 73.2% | 30.1% | yes | 51.06% | 0.39% | 49.81% | 49.91% |
| A3 | 32k | TT | 1114.36 | 829.96 | 340.38 | 74.5% | 30.5% | yes | 51.40% | 2.62% | 49.81% | 49.91% |

当前可直接读出的最小结论：

- `A1/A2/A3` 三个代表点上，`DeepSeek-4c` 的 device kernel duration 都略低于 `TT`：`70.36/73.86 us`、`171.26/174.67 us`、`1110.67/1114.36 us`。
- `wait_front` 在三点上都占主导，且 `DeepSeek-4c` 的 `wait_front_vs_kernel` 低于 `TT`：`94.9% vs 108.1%`、`81.3% vs 87.2%`、`73.2% vs 74.5%`。
- `reserve_back` 在 `A2/A3` 进入更明显区间，且 `DeepSeek-4c` 并没有把它消除：`A2=26.8%`，`A3=30.1%`；这说明长序列下 backpressure 仍然是后续优化目标。
- 同一批 `A1/A2/A3` raw log 现在也能直读 `reader/writer` source marker：`A2/A3` 上 `DeepSeek-4c` 的 `K reserve` 为 `28.59% / 51.06%`，对应 `TT` 为 `31.48% / 51.40%`；`V reserve` 则从 `TT` 的 `2.25% / 2.62%` 降到 `DeepSeek-4c` 的 `0.56% / 0.39%`。
- 但 `A2/A3` 上 `sender/tree` 基本没有被改写：`TT` 为 `48.65%/49.35%`、`49.81%/49.91%`；`DeepSeek-4c` 为 `48.58%/49.38%`、`49.81%/49.91%`。这更像是 front-side 供给被改善，而不是 writer/reduction 结构被彻底重塑。

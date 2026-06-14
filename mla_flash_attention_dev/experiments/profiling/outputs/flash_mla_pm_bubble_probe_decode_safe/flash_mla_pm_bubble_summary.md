# FlashMLA Decode PM/Bubble Summary

## 产物位置

- 汇总 CSV：`flash_mla_pm_bubble_decode_summary.csv`
- 图表目录：`visuals/`
- Dashboard：`flash_mla_pm_bubble_dashboard.html`

## 执行状态

- preflight：`decode_256`，状态 `passed`，尝试次数 `1`。
- decode guarded rerun：`8/8` 个 case 完成，覆盖 `256, 512, 1k, 2k, 4k, 8k, 16k, 32k`。
- compute bubble 计数：`8/8` 可用。
- `PM IDEAL/COMPUTE/BANDWIDTH` placeholder：`8/8`。
- `PM FPU UTIL` placeholder：`8/8`。
- `NOC UTIL` 可用：`0/8`；`DRAM BW UTIL` 可用：`0/8`。

## Bubble 总表

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

说明：`wait-front/reserve-back` 这里按 counter 密度来解读，而不是直接当作 wall-time share。
因为短序列点上 `wait-front + reserve-back` 会超过单次 kernel wall time，说明它们是跨 compute 线程累计后的 stall counter。

## PM / NOC / DRAM 原始捕获状态

| case | PM IDEAL [ns] | PM COMPUTE [ns] | PM BANDWIDTH [ns] | PM FPU UTIL (%) | NOC UTIL (%) | MULTICAST NOC UTIL (%) | DRAM BW UTIL (%) | 判读 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| decode_256 | 1.0 | 1.0 | 1.0 | 0.002 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |
| decode_512 | 1.0 | 1.0 | 1.0 | 0.002 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |
| decode_1k | 1.0 | 1.0 | 1.0 | 0.001 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |
| decode_2k | 1.0 | 1.0 | 1.0 | 0.001 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |
| decode_4k | 1.0 | 1.0 | 1.0 | 0.001 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |
| decode_8k | 1.0 | 1.0 | 1.0 | 0.000 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |
| decode_16k | 1.0 | 1.0 | 1.0 | 0.000 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |
| decode_32k | 1.0 | 1.0 | 1.0 | 0.000 | missing | missing | missing | bubble-ok / PM-placeholder / NOC-DRAM-missing |

## 关键结论

- compute bubble 计数已经在 `decode_256 -> decode_32k` 的 `8/8` 个点上补齐；`wait-front share in bubble` 从 `87.4%` 下降到 `70.9%`，但始终保持主导。
- `reserve-back share in bubble` 会从 `12.6%` 抬升到 `29.1%`；这和 detailed profile 里 `reader reserve share` 从 `0.7%` 抬升到 `61.5%` 的趋势是一致的。
- writer 侧的 `cb_wait` 仍然几乎一路饱和，从 `89.4%` 上升到 `99.8%`，说明长序列 decode 仍然是强耦合流水线，而不是单纯的 PM/FPU 问题。
- 这批 guarded rerun 目前应当被解读成 `bubble-complete / PM-incomplete`：`PM IDEAL/COMPUTE/BANDWIDTH` 固定在 `1.0 ns`，`PM FPU UTIL` 只有 `0.000 ~ 0.002`，`NOC/MULTICAST/DRAM BW UTIL` 仍为空，不宜当作真实利用率。

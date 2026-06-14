# 四方法共同支持的 Probe 结果：按 Seq Len 聚合

说明：这里只保留四方法共同支持的 workloads。
说明：每个单元格是该轴下全部 supported configs 的聚合结果，格式为 `avg / min / max`，括号内为参与聚合的配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `256` | avg 18.077 / min 3.533 / max 74.554 ms (80 cfgs) | avg 1706.208 / min 215.069 / max 6531.647 ms (80 cfgs) | avg 1706.038 / min 321.530 / max 8695.569 ms (80 cfgs) | avg 2077.145 / min 316.121 / max 9484.476 ms (80 cfgs) |
| `512` | avg 27.073 / min 2.599 / max 112.985 ms (80 cfgs) | avg 2043.011 / min 348.667 / max 14979.037 ms (80 cfgs) | avg 1925.942 / min 291.070 / max 9179.607 ms (80 cfgs) | avg 2050.148 / min 292.556 / max 10451.892 ms (80 cfgs) |
| `1k` | avg 48.146 / min 5.451 / max 207.180 ms (80 cfgs) | avg 1809.932 / min 276.130 / max 9545.724 ms (80 cfgs) | avg 1854.346 / min 274.454 / max 9407.935 ms (80 cfgs) | avg 2172.313 / min 340.023 / max 9531.958 ms (80 cfgs) |
| `2k` | avg 85.650 / min 8.340 / max 389.030 ms (80 cfgs) | avg 1787.174 / min 302.562 / max 6684.907 ms (80 cfgs) | avg 1799.488 / min 335.497 / max 8718.960 ms (80 cfgs) | avg 1887.066 / min 333.461 / max 7590.373 ms (80 cfgs) |
| `4k` | avg 156.463 / min 14.712 / max 745.423 ms (80 cfgs) | avg 1796.381 / min 212.556 / max 6756.698 ms (80 cfgs) | avg 1930.732 / min 258.277 / max 9434.106 ms (80 cfgs) | avg 1918.743 / min 312.506 / max 9724.290 ms (80 cfgs) |
| `8k` | avg 292.746 / min 30.657 / max 1421.436 ms (80 cfgs) | avg 2778.380 / min 253.504 / max 72593.024 ms (80 cfgs) | avg 1769.754 / min 249.186 / max 7458.477 ms (80 cfgs) | avg 2019.899 / min 321.840 / max 9116.947 ms (80 cfgs) |
| `16k` | avg 553.506 / min 48.342 / max 2768.763 ms (80 cfgs) | avg 1646.407 / min 232.685 / max 7955.134 ms (80 cfgs) | avg 1664.224 / min 220.843 / max 6650.839 ms (80 cfgs) | avg 1768.187 / min 279.179 / max 7591.726 ms (80 cfgs) |
| `32k` | avg 1069.117 / min 84.907 / max 5371.038 ms (80 cfgs) | avg 1825.890 / min 227.972 / max 15377.234 ms (80 cfgs) | avg 1730.804 / min 266.459 / max 7864.614 ms (80 cfgs) | avg 2043.804 / min 237.370 / max 14139.350 ms (80 cfgs) |
| `64k` | avg 2095.938 / min 150.870 / max 10760.952 ms (80 cfgs) | avg 1773.773 / min 265.980 / max 9723.812 ms (80 cfgs) | avg 1777.073 / min 216.669 / max 10463.092 ms (80 cfgs) | avg 1974.812 / min 253.031 / max 8995.742 ms (80 cfgs) |
| `128k` | avg 4239.252 / min 262.905 / max 22020.212 ms (80 cfgs) | avg 1911.449 / min 262.363 / max 11261.233 ms (80 cfgs) | avg 1997.422 / min 336.789 / max 8557.682 ms (80 cfgs) | avg 2469.061 / min 287.176 / max 41914.491 ms (80 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `256` | avg 293.3 / min 19.1 / max 1073.9 tok/s (80 cfgs) | avg 3.3 / min 0.5 / max 18.6 tok/s (80 cfgs) | avg 3.1 / min 0.5 / max 12.8 tok/s (80 cfgs) | avg 2.6 / min 0.4 / max 10.7 tok/s (80 cfgs) |
| `512` | avg 177.5 / min 47.2 / max 568.3 tok/s (80 cfgs) | avg 2.7 / min 0.3 / max 9.4 tok/s (80 cfgs) | avg 2.8 / min 0.4 / max 13.5 tok/s (80 cfgs) | avg 2.6 / min 0.4 / max 12.5 tok/s (80 cfgs) |
| `1k` | avg 101.8 / min 24.7 / max 332.3 tok/s (80 cfgs) | avg 3.2 / min 0.6 / max 26.1 tok/s (80 cfgs) | avg 2.8 / min 0.4 / max 12.8 tok/s (80 cfgs) | avg 2.6 / min 0.4 / max 14.3 tok/s (80 cfgs) |
| `2k` | avg 58.3 / min 13.6 / max 199.2 tok/s (80 cfgs) | avg 2.9 / min 0.7 / max 15.6 tok/s (80 cfgs) | avg 3.1 / min 0.4 / max 23.0 tok/s (80 cfgs) | avg 3.1 / min 0.5 / max 24.0 tok/s (80 cfgs) |
| `4k` | avg 32.5 / min 7.7 / max 112.5 tok/s (80 cfgs) | avg 3.1 / min 0.4 / max 18.8 tok/s (80 cfgs) | avg 2.9 / min 0.4 / max 13.8 tok/s (80 cfgs) | avg 3.2 / min 0.4 / max 14.6 tok/s (80 cfgs) |
| `8k` | avg 17.8 / min 4.1 / max 60.3 tok/s (80 cfgs) | avg 3.1 / min 0.0 / max 13.5 tok/s (80 cfgs) | avg 3.2 / min 0.3 / max 13.9 tok/s (80 cfgs) | avg 2.6 / min 0.4 / max 8.6 tok/s (80 cfgs) |
| `16k` | avg 9.7 / min 2.1 / max 32.5 tok/s (80 cfgs) | avg 3.6 / min 0.5 / max 20.6 tok/s (80 cfgs) | avg 3.4 / min 0.6 / max 19.3 tok/s (80 cfgs) | avg 3.2 / min 0.4 / max 16.6 tok/s (80 cfgs) |
| `32k` | avg 5.1 / min 1.1 / max 17.3 tok/s (80 cfgs) | avg 3.4 / min 0.5 / max 16.0 tok/s (80 cfgs) | avg 3.1 / min 0.5 / max 9.7 tok/s (80 cfgs) | avg 3.0 / min 0.5 / max 14.0 tok/s (80 cfgs) |
| `64k` | avg 2.7 / min 0.6 / max 9.2 tok/s (80 cfgs) | avg 3.0 / min 0.6 / max 14.0 tok/s (80 cfgs) | avg 3.2 / min 0.4 / max 15.5 tok/s (80 cfgs) | avg 2.8 / min 0.5 / max 14.0 tok/s (80 cfgs) |
| `128k` | avg 1.4 / min 0.3 / max 4.7 tok/s (80 cfgs) | avg 2.7 / min 0.5 / max 8.8 tok/s (80 cfgs) | avg 2.6 / min 0.6 / max 12.9 tok/s (80 cfgs) | avg 2.8 / min 0.2 / max 12.0 tok/s (80 cfgs) |

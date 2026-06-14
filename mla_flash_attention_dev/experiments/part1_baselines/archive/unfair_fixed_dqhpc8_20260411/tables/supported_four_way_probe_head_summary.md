# 四方法共同支持的 Probe 结果：按 Heads 聚合

说明：这里只保留四方法共同支持的 workloads。
说明：固定 `seq_len=8k`，对该轴下全部共同支持配置做聚合；格式为 `avg / min / max`，括号内为配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `H=8` | avg 220.968 / min 30.657 / max 874.182 ms (25 cfgs) | avg 2188.586 / min 378.014 / max 7617.647 ms (25 cfgs) | avg 2157.313 / min 249.186 / max 7458.477 ms (25 cfgs) | avg 2330.983 / min 364.723 / max 8709.958 ms (25 cfgs) |
| `H=16` | avg 263.603 / min 43.111 / max 864.860 ms (20 cfgs) | avg 1723.977 / min 312.153 / max 5839.676 ms (20 cfgs) | avg 1437.223 / min 274.481 / max 3635.065 ms (20 cfgs) | avg 1564.692 / min 321.840 / max 4098.781 ms (20 cfgs) |
| `H=24` | avg 391.432 / min 64.497 / max 1421.436 ms (20 cfgs) | avg 2122.838 / min 253.504 / max 7630.336 ms (20 cfgs) | avg 1987.373 / min 284.049 / max 6214.042 ms (20 cfgs) | avg 2469.888 / min 362.287 / max 9116.947 ms (20 cfgs) |
| `H=32` | avg 319.655 / min 73.334 / max 904.641 ms (15 cfgs) | avg 6041.296 / min 309.640 / max 72593.024 ms (15 cfgs) | avg 1277.036 / min 370.592 / max 3074.499 ms (15 cfgs) | avg 1508.381 / min 618.286 / max 2825.360 ms (15 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `H=8` | avg 29.9 / min 11.3 / max 60.3 tok/s (25 cfgs) | avg 3.2 / min 1.4 / max 10.0 tok/s (25 cfgs) | avg 3.3 / min 1.0 / max 13.9 tok/s (25 cfgs) | avg 3.0 / min 1.1 / max 7.9 tok/s (25 cfgs) |
| `H=16` | avg 16.2 / min 6.3 / max 31.4 tok/s (20 cfgs) | avg 3.0 / min 0.9 / max 13.5 tok/s (20 cfgs) | avg 2.9 / min 1.1 / max 7.7 tok/s (20 cfgs) | avg 2.8 / min 0.9 / max 8.6 tok/s (20 cfgs) |
| `H=24` | avg 11.2 / min 5.4 / max 22.1 tok/s (20 cfgs) | avg 3.0 / min 0.7 / max 7.9 tok/s (20 cfgs) | avg 3.5 / min 0.6 / max 8.8 tok/s (20 cfgs) | avg 2.4 / min 0.5 / max 5.5 tok/s (20 cfgs) |
| `H=32` | avg 8.6 / min 4.1 / max 15.1 tok/s (15 cfgs) | avg 2.9 / min 0.0 / max 12.9 tok/s (15 cfgs) | avg 2.9 / min 0.3 / max 10.8 tok/s (15 cfgs) | avg 2.0 / min 0.4 / max 6.5 tok/s (15 cfgs) |

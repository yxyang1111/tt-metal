# 四方法共同支持的 Probe 结果：按 Batch 聚合

说明：这里只保留四方法共同支持的 workloads。
说明：固定 `seq_len=8k`，对该轴下全部共同支持配置做聚合；格式为 `avg / min / max`，括号内为配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `B=1` | avg 102.769 / min 30.657 / max 245.233 ms (20 cfgs) | avg 4264.406 / min 312.153 / max 72593.024 ms (20 cfgs) | avg 795.504 / min 249.186 / max 3074.499 ms (20 cfgs) | avg 979.292 / min 321.840 / max 2825.360 ms (20 cfgs) |
| `B=2` | avg 186.072 / min 42.025 / max 456.340 ms (20 cfgs) | avg 1207.892 / min 253.504 / max 2931.832 ms (20 cfgs) | avg 1133.925 / min 284.049 / max 2247.861 ms (20 cfgs) | avg 1194.581 / min 362.287 / max 2309.170 ms (20 cfgs) |
| `B=4` | avg 345.909 / min 74.738 / max 904.641 ms (20 cfgs) | avg 1600.908 / min 309.640 / max 4669.629 ms (20 cfgs) | avg 1328.188 / min 370.592 / max 4570.772 ms (20 cfgs) | avg 1736.653 / min 599.974 / max 4098.781 ms (20 cfgs) |
| `B=8` | avg 539.003 / min 140.499 / max 1421.436 ms (15 cfgs) | avg 3226.333 / min 590.951 / max 7630.336 ms (15 cfgs) | avg 3131.784 / min 574.505 / max 6214.042 ms (15 cfgs) | avg 3496.694 / min 933.737 / max 9116.947 ms (15 cfgs) |
| `B=16` | avg 527.932 / min 265.372 / max 874.182 ms (5 cfgs) | avg 6482.261 / min 4434.133 / max 7617.647 ms (5 cfgs) | avg 5890.232 / min 4066.623 / max 7458.477 ms (5 cfgs) | avg 6186.192 / min 4040.879 / max 8709.958 ms (5 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `B=1` | avg 13.0 / min 4.1 / max 32.6 tok/s (20 cfgs) | avg 1.8 / min 0.0 / max 3.2 tok/s (20 cfgs) | avg 2.0 / min 0.3 / max 4.0 tok/s (20 cfgs) | avg 1.6 / min 0.4 / max 3.1 tok/s (20 cfgs) |
| `B=2` | avg 15.6 / min 4.4 / max 47.6 tok/s (20 cfgs) | avg 2.6 / min 0.7 / max 7.9 tok/s (20 cfgs) | avg 2.6 / min 0.9 / max 7.0 tok/s (20 cfgs) | avg 2.3 / min 0.9 / max 5.5 tok/s (20 cfgs) |
| `B=4` | avg 17.7 / min 4.4 / max 53.5 tok/s (20 cfgs) | avg 3.8 / min 0.9 / max 12.9 tok/s (20 cfgs) | avg 4.4 / min 0.9 / max 10.8 tok/s (20 cfgs) | avg 3.0 / min 1.0 / max 6.7 tok/s (20 cfgs) |
| `B=8` | avg 21.4 / min 5.6 / max 56.9 tok/s (15 cfgs) | avg 4.5 / min 1.0 / max 13.5 tok/s (15 cfgs) | avg 3.9 / min 1.3 / max 13.9 tok/s (15 cfgs) | avg 3.6 / min 0.9 / max 8.6 tok/s (15 cfgs) |
| `B=16` | avg 36.2 / min 18.3 / max 60.3 tok/s (5 cfgs) | avg 2.6 / min 2.1 / max 3.6 tok/s (5 cfgs) | avg 2.9 / min 2.1 / max 3.9 tok/s (5 cfgs) | avg 2.7 / min 1.8 / max 4.0 tok/s (5 cfgs) |

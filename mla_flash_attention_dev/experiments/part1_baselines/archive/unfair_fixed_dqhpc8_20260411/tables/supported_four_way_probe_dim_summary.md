# 四方法共同支持的 Probe 结果：按 Value Dim 聚合

说明：这里只保留四方法共同支持的 workloads。
说明：固定 `seq_len=8k`，对该轴下全部共同支持配置做聚合；格式为 `avg / min / max`，括号内为配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=128` | avg 140.762 / min 30.657 / max 362.565 ms (16 cfgs) | avg 1799.735 / min 286.970 / max 6974.761 ms (16 cfgs) | avg 1899.972 / min 366.423 / max 7324.544 ms (16 cfgs) | avg 1897.383 / min 427.098 / max 8709.958 ms (16 cfgs) |
| `value=192` | avg 202.740 / min 39.299 / max 561.561 ms (16 cfgs) | avg 1832.862 / min 343.347 / max 6584.644 ms (16 cfgs) | avg 1609.312 / min 399.551 / max 5940.342 ms (16 cfgs) | avg 1729.705 / min 371.843 / max 4892.071 ms (16 cfgs) |
| `value=256` | avg 254.635 / min 50.305 / max 660.572 ms (16 cfgs) | avg 1398.231 / min 313.135 / max 5833.324 ms (16 cfgs) | avg 1533.492 / min 274.481 / max 5458.842 ms (16 cfgs) | avg 1933.478 / min 362.287 / max 7097.048 ms (16 cfgs) |
| `value=384` | avg 376.257 / min 65.913 / max 1000.792 ms (16 cfgs) | avg 2189.361 / min 253.504 / max 7630.336 ms (16 cfgs) | avg 1837.310 / min 284.049 / max 7458.477 ms (16 cfgs) | avg 2130.129 / min 360.241 / max 9116.947 ms (16 cfgs) |
| `value=512` | avg 489.338 / min 88.377 / max 1421.436 ms (16 cfgs) | avg 6671.711 / min 312.153 / max 72593.024 ms (16 cfgs) | avg 1968.682 / min 249.186 / max 5142.676 ms (16 cfgs) | avg 2408.800 / min 321.840 / max 6254.000 ms (16 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=128` | avg 30.3 / min 13.6 / max 60.3 tok/s (16 cfgs) | avg 3.3 / min 1.3 / max 12.9 tok/s (16 cfgs) | avg 3.0 / min 0.6 / max 10.8 tok/s (16 cfgs) | avg 3.0 / min 0.5 / max 7.9 tok/s (16 cfgs) |
| `value=192` | avg 21.2 / min 9.5 / max 44.1 tok/s (16 cfgs) | avg 2.8 / min 0.5 / max 8.3 tok/s (16 cfgs) | avg 3.2 / min 1.0 / max 8.8 tok/s (16 cfgs) | avg 2.8 / min 0.4 / max 8.6 tok/s (16 cfgs) |
| `value=256` | avg 16.9 / min 7.6 / max 34.7 tok/s (16 cfgs) | avg 4.0 / min 1.4 / max 13.5 tok/s (16 cfgs) | avg 3.3 / min 0.5 / max 8.4 tok/s (16 cfgs) | avg 2.7 / min 0.6 / max 6.0 tok/s (16 cfgs) |
| `value=384` | avg 11.6 / min 5.5 / max 23.6 tok/s (16 cfgs) | avg 3.1 / min 0.7 / max 8.7 tok/s (16 cfgs) | avg 3.5 / min 0.8 / max 13.9 tok/s (16 cfgs) | avg 2.5 / min 0.8 / max 6.5 tok/s (16 cfgs) |
| `value=512` | avg 9.1 / min 4.1 / max 18.3 tok/s (16 cfgs) | avg 2.1 / min 0.0 / max 4.4 tok/s (16 cfgs) | avg 2.8 / min 0.3 / max 7.5 tok/s (16 cfgs) | avg 2.0 / min 0.4 / max 5.4 tok/s (16 cfgs) |

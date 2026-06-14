# 公平四方法结果：按 Seq Len 聚合

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：每个单元格是该轴下全部公平 benchmark configs 的聚合结果，格式为 `avg / min / max`，括号内为参与聚合的配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `256` | avg 1.435 / min 1.014 / max 2.113 ms (4 cfgs) | avg 0.175 / min 0.162 / max 0.197 ms (4 cfgs) | avg 0.174 / min 0.162 / max 0.202 ms (4 cfgs) | avg 0.180 / min 0.160 / max 0.188 ms (4 cfgs) |
| `512` | avg 8.344 / min 1.335 / max 20.648 ms (4 cfgs) | avg 0.185 / min 0.175 / max 0.196 ms (4 cfgs) | avg 0.189 / min 0.178 / max 0.204 ms (4 cfgs) | avg 0.182 / min 0.151 / max 0.211 ms (4 cfgs) |
| `1k` | avg 22.024 / min 5.313 / max 39.754 ms (4 cfgs) | avg 0.194 / min 0.179 / max 0.215 ms (4 cfgs) | avg 0.194 / min 0.176 / max 0.208 ms (4 cfgs) | avg 0.189 / min 0.171 / max 0.216 ms (4 cfgs) |
| `2k` | avg 43.465 / min 15.240 / max 74.994 ms (4 cfgs) | avg 0.227 / min 0.205 / max 0.239 ms (4 cfgs) | avg 0.224 / min 0.203 / max 0.234 ms (4 cfgs) | avg 0.218 / min 0.192 / max 0.255 ms (4 cfgs) |
| `4k` | avg 85.989 / min 27.867 / max 146.124 ms (4 cfgs) | avg 0.276 / min 0.255 / max 0.287 ms (4 cfgs) | avg 0.285 / min 0.255 / max 0.312 ms (4 cfgs) | avg 0.285 / min 0.251 / max 0.335 ms (4 cfgs) |
| `8k` | avg 160.153 / min 57.758 / max 264.559 ms (4 cfgs) | avg 0.380 / min 0.355 / max 0.412 ms (4 cfgs) | avg 0.407 / min 0.362 / max 0.473 ms (4 cfgs) | avg 0.427 / min 0.358 / max 0.523 ms (4 cfgs) |
| `16k` | avg 310.293 / min 124.215 / max 507.768 ms (4 cfgs) | avg 0.618 / min 0.561 / max 0.675 ms (4 cfgs) | avg 0.641 / min 0.578 / max 0.702 ms (4 cfgs) | avg 0.678 / min 0.570 / max 0.873 ms (4 cfgs) |
| `32k` | avg 574.959 / min 226.829 / max 916.238 ms (4 cfgs) | avg 1.094 / min 0.973 / max 1.207 ms (4 cfgs) | avg 1.115 / min 0.997 / max 1.230 ms (4 cfgs) | avg 1.200 / min 0.998 / max 1.587 ms (4 cfgs) |
| `64k` | avg 1086.137 / min 470.148 / max 1671.564 ms (4 cfgs) | avg 5.137 / min 1.810 / max 14.687 ms (4 cfgs) | avg 2.093 / min 1.877 / max 2.303 ms (4 cfgs) | avg 2.254 / min 1.867 / max 3.001 ms (4 cfgs) |
| `128k` | avg 2170.260 / min 860.125 / max 3425.909 ms (4 cfgs) | avg 22.068 / min 3.448 / max 29.940 ms (4 cfgs) | avg 15.595 / min 3.594 / max 35.604 ms (4 cfgs) | avg 12.282 / min 3.555 / max 37.599 ms (4 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `256` | avg 749.8 / min 473.3 / max 986.3 tok/s (4 cfgs) | avg 5752.6 / min 5077.3 / max 6181.6 tok/s (4 cfgs) | avg 5776.4 / min 4959.9 / max 6168.9 tok/s (4 cfgs) | avg 5589.9 / min 5313.0 / max 6239.1 tok/s (4 cfgs) |
| `512` | avg 310.5 / min 48.4 / max 748.9 tok/s (4 cfgs) | avg 5408.6 / min 5095.1 / max 5705.8 tok/s (4 cfgs) | avg 5314.5 / min 4891.5 / max 5623.3 tok/s (4 cfgs) | avg 5593.0 / min 4738.9 / max 6614.9 tok/s (4 cfgs) |
| `1k` | avg 78.7 / min 25.2 / max 188.2 tok/s (4 cfgs) | avg 5191.5 / min 4640.6 / max 5584.5 tok/s (4 cfgs) | avg 5178.7 / min 4811.7 / max 5681.0 tok/s (4 cfgs) | avg 5334.3 / min 4619.7 / max 5848.0 tok/s (4 cfgs) |
| `2k` | avg 32.9 / min 13.3 / max 65.6 tok/s (4 cfgs) | avg 4421.0 / min 4184.5 / max 4869.9 tok/s (4 cfgs) | avg 4482.1 / min 4276.2 / max 4918.0 tok/s (4 cfgs) | avg 4633.0 / min 3919.2 / max 5209.6 tok/s (4 cfgs) |
| `4k` | avg 17.2 / min 6.8 / max 35.9 tok/s (4 cfgs) | avg 3629.2 / min 3479.8 / max 3921.6 tok/s (4 cfgs) | avg 3533.3 / min 3204.5 / max 3916.4 tok/s (4 cfgs) | avg 3563.6 / min 2988.3 / max 3987.8 tok/s (4 cfgs) |
| `8k` | avg 8.7 / min 3.8 / max 17.3 tok/s (4 cfgs) | avg 2641.2 / min 2424.8 / max 2819.2 tok/s (4 cfgs) | avg 2488.5 / min 2112.0 / max 2762.1 tok/s (4 cfgs) | avg 2404.1 / min 1912.1 / max 2789.9 tok/s (4 cfgs) |
| `16k` | avg 4.2 / min 2.0 / max 8.1 tok/s (4 cfgs) | avg 1630.2 / min 1481.1 / max 1783.8 tok/s (4 cfgs) | avg 1573.9 / min 1423.8 / max 1730.9 tok/s (4 cfgs) | avg 1515.9 / min 1145.8 / max 1754.8 tok/s (4 cfgs) |
| `32k` | avg 2.3 / min 1.1 / max 4.4 tok/s (4 cfgs) | avg 923.4 / min 828.3 / max 1027.3 tok/s (4 cfgs) | avg 905.6 / min 813.3 / max 1002.9 tok/s (4 cfgs) | avg 863.8 / min 630.2 / max 1001.9 tok/s (4 cfgs) |
| `64k` | avg 1.2 / min 0.6 / max 2.1 tok/s (4 cfgs) | avg 404.4 / min 68.1 / max 552.5 tok/s (4 cfgs) | avg 482.6 / min 434.2 / max 532.8 tok/s (4 cfgs) | avg 460.6 / min 333.2 / max 535.6 tok/s (4 cfgs) |
| `128k` | avg 0.6 / min 0.3 / max 1.2 tok/s (4 cfgs) | avg 99.1 / min 33.4 / max 290.1 tok/s (4 cfgs) | avg 145.8 / min 28.1 / max 278.2 tok/s (4 cfgs) | avg 203.8 / min 26.6 / max 281.3 tok/s (4 cfgs) |

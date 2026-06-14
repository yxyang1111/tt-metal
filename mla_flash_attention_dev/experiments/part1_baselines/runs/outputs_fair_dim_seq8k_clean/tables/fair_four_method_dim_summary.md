# 公平四方法结果：按 Value Dim 切片

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：按多组固定 `seq_len` 切片展示 `value_dim` 轴；每个单元格格式为 `avg / min / max`，括号内为参与聚合的配置数。

## 固定 `seq_len=8k`

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `seq_len=8k`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=128` | avg 40.107 / min 18.959 / max 70.697 ms (4 cfgs) | avg 0.250 / min 0.236 / max 0.271 ms (4 cfgs) | avg 0.255 / min 0.235 / max 0.276 ms (4 cfgs) | avg 0.259 / min 0.236 / max 0.300 ms (4 cfgs) |
| `value=192` | avg 62.895 / min 26.174 / max 112.980 ms (4 cfgs) | avg 0.279 / min 0.262 / max 0.300 ms (4 cfgs) | avg 0.275 / min 0.253 / max 0.292 ms (4 cfgs) | avg 0.292 / min 0.267 / max 0.344 ms (4 cfgs) |
| `value=256` | avg 82.565 / min 34.516 / max 147.245 ms (4 cfgs) | avg 0.308 / min 0.288 / max 0.318 ms (4 cfgs) | avg 0.310 / min 0.289 / max 0.324 ms (4 cfgs) | avg 0.315 / min 0.283 / max 0.378 ms (4 cfgs) |
| `value=384` | avg 128.055 / min 47.198 / max 210.914 ms (4 cfgs) | avg 0.325 / min 0.304 / max 0.349 ms (4 cfgs) | avg 0.333 / min 0.304 / max 0.358 ms (4 cfgs) | avg 0.347 / min 0.305 / max 0.417 ms (4 cfgs) |
| `value=512` | avg 166.458 / min 66.321 / max 266.574 ms (4 cfgs) | avg 0.390 / min 0.355 / max 0.417 ms (4 cfgs) | avg 0.398 / min 0.361 / max 0.431 ms (4 cfgs) | avg 0.420 / min 0.362 / max 0.516 ms (4 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=128` | avg 31.4 / min 14.1 / max 52.7 tok/s (4 cfgs) | avg 4017.8 / min 3693.8 / max 4232.7 tok/s (4 cfgs) | avg 3938.4 / min 3626.4 / max 4252.1 tok/s (4 cfgs) | avg 3897.7 / min 3334.9 / max 4242.8 tok/s (4 cfgs) |
| `value=192` | avg 21.0 / min 8.9 / max 38.2 tok/s (4 cfgs) | avg 3592.2 / min 3330.7 / max 3816.9 tok/s (4 cfgs) | avg 3644.4 / min 3427.6 / max 3946.2 tok/s (4 cfgs) | avg 3456.1 / min 2904.4 / max 3749.0 tok/s (4 cfgs) |
| `value=256` | avg 16.0 / min 6.8 / max 29.0 tok/s (4 cfgs) | avg 3250.6 / min 3145.3 / max 3474.8 tok/s (4 cfgs) | avg 3231.4 / min 3083.5 / max 3460.9 tok/s (4 cfgs) | avg 3218.3 / min 2646.4 / max 3532.6 tok/s (4 cfgs) |
| `value=384` | avg 10.7 / min 4.7 / max 21.2 tok/s (4 cfgs) | avg 3087.6 / min 2869.1 / max 3290.8 tok/s (4 cfgs) | avg 3020.0 / min 2795.3 / max 3284.1 tok/s (4 cfgs) | avg 2924.9 / min 2398.0 / max 3275.9 tok/s (4 cfgs) |
| `value=512` | avg 7.9 / min 3.8 / max 15.1 tok/s (4 cfgs) | avg 2577.7 / min 2398.8 / max 2819.6 tok/s (4 cfgs) | avg 2530.6 / min 2321.6 / max 2773.1 tok/s (4 cfgs) | avg 2424.7 / min 1936.6 / max 2762.0 tok/s (4 cfgs) |


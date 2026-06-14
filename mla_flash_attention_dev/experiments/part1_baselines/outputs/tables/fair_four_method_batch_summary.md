# 公平四方法结果：按 Batch 切片

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：按多组固定 `seq_len` 切片展示 `batch` 轴；每个单元格格式为 `avg / min / max`，括号内为参与聚合的配置数。

## 固定 `seq_len=8k`

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `seq_len=8k`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `B=1` | avg 148.824 / min 62.169 / max 233.072 ms (4 cfgs) | avg 0.398 / min 0.382 / max 0.414 ms (4 cfgs) | avg 0.406 / min 0.382 / max 0.432 ms (4 cfgs) | avg 0.426 / min 0.364 / max 0.520 ms (4 cfgs) |
| `B=2` | avg 287.203 / min 130.898 / max 436.862 ms (4 cfgs) | avg 0.392 / min 0.358 / max 0.428 ms (4 cfgs) | avg 0.400 / min 0.368 / max 0.439 ms (4 cfgs) | avg 0.304 / min 0.268 / max 0.355 ms (4 cfgs) |
| `B=4` | avg 552.142 / min 244.543 / max 853.795 ms (4 cfgs) | avg 0.416 / min 0.382 / max 0.456 ms (4 cfgs) | avg 0.415 / min 0.373 / max 0.447 ms (4 cfgs) | avg 0.338 / min 0.318 / max 0.350 ms (4 cfgs) |
| `B=8` | avg 1064.141 / min 469.655 / max 1675.706 ms (4 cfgs) | avg 0.547 / min 0.528 / max 0.564 ms (4 cfgs) | avg 0.442 / min 0.406 / max 0.477 ms (4 cfgs) | avg 0.464 / min 0.464 / max 0.464 ms (1 cfgs) |
| `B=16` | avg 2135.229 / min 903.111 / max 3430.652 ms (4 cfgs) | avg 0.908 / min 0.875 / max 0.939 ms (4 cfgs) | avg 0.641 / min 0.599 / max 0.679 ms (4 cfgs) | - |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `B=1` | avg 8.5 / min 4.3 / max 16.1 tok/s (4 cfgs) | avg 2513.9 / min 2415.3 / max 2617.0 tok/s (4 cfgs) | avg 2470.2 / min 2317.2 / max 2619.5 tok/s (4 cfgs) | avg 2391.0 / min 1922.3 / max 2748.6 tok/s (4 cfgs) |
| `B=2` | avg 8.5 / min 4.6 / max 15.3 tok/s (4 cfgs) | avg 5142.6 / min 4673.3 / max 5585.4 tok/s (4 cfgs) | avg 5028.9 / min 4556.0 / max 5432.3 tok/s (4 cfgs) | avg 6657.4 / min 5635.9 / max 7474.8 tok/s (4 cfgs) |
| `B=4` | avg 9.0 / min 4.7 / max 16.4 tok/s (4 cfgs) | avg 9668.7 / min 8779.5 / max 10459.5 tok/s (4 cfgs) | avg 9695.1 / min 8945.8 / max 10715.1 tok/s (4 cfgs) | avg 11841.4 / min 11426.7 / max 12595.3 tok/s (4 cfgs) |
| `B=8` | avg 9.4 / min 4.8 / max 17.0 tok/s (4 cfgs) | avg 14647.5 / min 14187.4 / max 15154.8 tok/s (4 cfgs) | avg 18228.4 / min 16759.2 / max 19690.1 tok/s (4 cfgs) | avg 17225.8 / min 17225.8 / max 17225.8 tok/s (1 cfgs) |
| `B=16` | avg 9.5 / min 4.7 / max 17.7 tok/s (4 cfgs) | avg 17627.9 / min 17037.2 / max 18296.2 tok/s (4 cfgs) | avg 25020.9 / min 23554.1 / max 26720.3 tok/s (4 cfgs) | - |


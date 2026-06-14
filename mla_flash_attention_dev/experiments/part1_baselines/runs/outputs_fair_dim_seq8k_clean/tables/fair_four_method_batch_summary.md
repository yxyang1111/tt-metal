# 公平四方法结果：按 Batch 切片

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：按多组固定 `seq_len` 切片展示 `batch` 轴；每个单元格格式为 `avg / min / max`，括号内为参与聚合的配置数。

## 固定 `seq_len=8k`

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `seq_len=8k`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。
说明：当前公平版结果里 `batch` 只有一个取值，所以该表主要用于查看该单值在当前切片下的聚合表现。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `B=1` | avg 96.016 / min 18.959 / max 266.574 ms (20 cfgs) | avg 0.310 / min 0.236 / max 0.417 ms (20 cfgs) | avg 0.314 / min 0.235 / max 0.431 ms (20 cfgs) | avg 0.327 / min 0.236 / max 0.516 ms (20 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `B=1` | avg 17.4 / min 3.8 / max 52.7 tok/s (20 cfgs) | avg 3305.2 / min 2398.8 / max 4232.7 tok/s (20 cfgs) | avg 3273.0 / min 2321.6 / max 4252.1 tok/s (20 cfgs) | avg 3184.3 / min 1936.6 / max 4242.8 tok/s (20 cfgs) |


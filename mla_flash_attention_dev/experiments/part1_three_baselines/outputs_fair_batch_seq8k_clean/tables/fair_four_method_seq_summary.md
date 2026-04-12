# 公平四方法结果：按 Seq Len 聚合

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：每个单元格是该轴下全部公平 benchmark configs 的聚合结果，格式为 `avg / min / max`，括号内为参与聚合的配置数。
说明：当前公平版结果里 `seq_len` 只有一个取值，所以该表主要用于查看该单值在当前切片下的聚合表现。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `8k` | avg 837.508 / min 62.169 / max 3430.652 ms (20 cfgs) | avg 0.532 / min 0.358 / max 0.939 ms (20 cfgs) | avg 0.461 / min 0.368 / max 0.679 ms (20 cfgs) | avg 0.365 / min 0.268 / max 0.520 ms (13 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `8k` | avg 9.0 / min 4.3 / max 17.7 tok/s (20 cfgs) | avg 9920.1 / min 2415.3 / max 18296.2 tok/s (20 cfgs) | avg 12088.7 / min 2317.2 / max 26720.3 tok/s (20 cfgs) | avg 7752.7 / min 1922.3 / max 17225.8 tok/s (13 cfgs) |

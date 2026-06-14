# 公平四方法结果：按 Value Dim 切片

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：按多组固定 `seq_len` 切片展示 `value_dim` 轴；每个单元格格式为 `avg / min / max`，括号内为参与聚合的配置数。

## 固定 `seq_len=1k`

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `seq_len=1k`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。
说明：当前公平版结果里 `value_dim` 只有一个取值，所以该表主要用于查看该单值在当前切片下的聚合表现。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 168.478 / min 168.478 / max 168.478 ms (1 cfgs) | avg 0.232 / min 0.232 / max 0.232 ms (1 cfgs) | avg 0.233 / min 0.233 / max 0.233 ms (1 cfgs) | avg 0.257 / min 0.257 / max 0.257 ms (1 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 35.6 / min 35.6 / max 35.6 tok/s (1 cfgs) | avg 25897.8 / min 25897.8 / max 25897.8 tok/s (1 cfgs) | avg 25761.8 / min 25761.8 / max 25761.8 tok/s (1 cfgs) | avg 23350.3 / min 23350.3 / max 23350.3 tok/s (1 cfgs) |

## 固定 `seq_len=8k`

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `seq_len=8k`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。
说明：当前公平版结果里 `value_dim` 只有一个取值，所以该表主要用于查看该单值在当前切片下的聚合表现。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 1301.623 / min 1301.623 / max 1301.623 ms (1 cfgs) | avg 0.478 / min 0.478 / max 0.478 ms (1 cfgs) | avg 0.465 / min 0.465 / max 0.465 ms (1 cfgs) | avg 0.440 / min 0.440 / max 0.440 ms (1 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 4.6 / min 4.6 / max 4.6 tok/s (1 cfgs) | avg 12553.6 / min 12553.6 / max 12553.6 tok/s (1 cfgs) | avg 12899.2 / min 12899.2 / max 12899.2 tok/s (1 cfgs) | avg 13645.8 / min 13645.8 / max 13645.8 tok/s (1 cfgs) |

## 固定 `seq_len=32k`

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `seq_len=32k`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。
说明：当前公平版结果里 `value_dim` 只有一个取值，所以该表主要用于查看该单值在当前切片下的聚合表现。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 5156.778 / min 5156.778 / max 5156.778 ms (1 cfgs) | avg 7.140 / min 7.140 / max 7.140 ms (1 cfgs) | avg 8.467 / min 8.467 / max 8.467 ms (1 cfgs) | avg 4.271 / min 4.271 / max 4.271 ms (1 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 1.2 / min 1.2 / max 1.2 tok/s (1 cfgs) | avg 840.3 / min 840.3 / max 840.3 tok/s (1 cfgs) | avg 708.6 / min 708.6 / max 708.6 tok/s (1 cfgs) | avg 1404.9 / min 1404.9 / max 1404.9 tok/s (1 cfgs) |

## 固定 `seq_len=128k`

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `seq_len=128k`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。
说明：当前公平版结果里 `value_dim` 只有一个取值，所以该表主要用于查看该单值在当前切片下的聚合表现。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 20592.962 / min 20592.962 / max 20592.962 ms (1 cfgs) | avg 12.977 / min 12.977 / max 12.977 ms (1 cfgs) | avg 9.811 / min 9.811 / max 9.811 ms (1 cfgs) | avg 8.866 / min 8.866 / max 8.866 ms (1 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `value=512` | avg 0.3 / min 0.3 / max 0.3 tok/s (1 cfgs) | avg 462.4 / min 462.4 / max 462.4 tok/s (1 cfgs) | avg 611.6 / min 611.6 / max 611.6 tok/s (1 cfgs) | avg 676.7 / min 676.7 / max 676.7 tok/s (1 cfgs) |


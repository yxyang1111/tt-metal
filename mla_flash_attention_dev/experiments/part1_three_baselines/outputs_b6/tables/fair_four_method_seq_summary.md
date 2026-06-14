# 公平四方法结果：按 Seq Len 聚合

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：每个单元格是该轴下全部公平 benchmark configs 的聚合结果，格式为 `avg / min / max`，括号内为参与聚合的配置数。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `256` | avg 43.347 / min 43.347 / max 43.347 ms (1 cfgs) | avg 0.195 / min 0.195 / max 0.195 ms (1 cfgs) | avg 0.191 / min 0.191 / max 0.191 ms (1 cfgs) | avg 0.194 / min 0.194 / max 0.194 ms (1 cfgs) |
| `512` | avg 81.890 / min 81.890 / max 81.890 ms (1 cfgs) | avg 0.197 / min 0.197 / max 0.197 ms (1 cfgs) | avg 0.198 / min 0.198 / max 0.198 ms (1 cfgs) | avg 0.216 / min 0.216 / max 0.216 ms (1 cfgs) |
| `1k` | avg 168.478 / min 168.478 / max 168.478 ms (1 cfgs) | avg 0.232 / min 0.232 / max 0.232 ms (1 cfgs) | avg 0.233 / min 0.233 / max 0.233 ms (1 cfgs) | avg 0.257 / min 0.257 / max 0.257 ms (1 cfgs) |
| `2k` | avg 326.317 / min 326.317 / max 326.317 ms (1 cfgs) | avg 0.268 / min 0.268 / max 0.268 ms (1 cfgs) | avg 0.266 / min 0.266 / max 0.266 ms (1 cfgs) | avg 0.285 / min 0.285 / max 0.285 ms (1 cfgs) |
| `4k` | avg 652.267 / min 652.267 / max 652.267 ms (1 cfgs) | avg 0.323 / min 0.323 / max 0.323 ms (1 cfgs) | avg 0.313 / min 0.313 / max 0.313 ms (1 cfgs) | avg 0.308 / min 0.308 / max 0.308 ms (1 cfgs) |
| `8k` | avg 1301.623 / min 1301.623 / max 1301.623 ms (1 cfgs) | avg 0.478 / min 0.478 / max 0.478 ms (1 cfgs) | avg 0.465 / min 0.465 / max 0.465 ms (1 cfgs) | avg 0.440 / min 0.440 / max 0.440 ms (1 cfgs) |
| `16k` | avg 2617.145 / min 2617.145 / max 2617.145 ms (1 cfgs) | avg 5.503 / min 5.503 / max 5.503 ms (1 cfgs) | avg 0.736 / min 0.736 / max 0.736 ms (1 cfgs) | avg 11.139 / min 11.139 / max 11.139 ms (1 cfgs) |
| `32k` | avg 5156.778 / min 5156.778 / max 5156.778 ms (1 cfgs) | avg 7.140 / min 7.140 / max 7.140 ms (1 cfgs) | avg 8.467 / min 8.467 / max 8.467 ms (1 cfgs) | avg 4.271 / min 4.271 / max 4.271 ms (1 cfgs) |
| `64k` | avg 10177.482 / min 10177.482 / max 10177.482 ms (1 cfgs) | avg 9.500 / min 9.500 / max 9.500 ms (1 cfgs) | avg 8.915 / min 8.915 / max 8.915 ms (1 cfgs) | avg 7.704 / min 7.704 / max 7.704 ms (1 cfgs) |
| `128k` | avg 20592.962 / min 20592.962 / max 20592.962 ms (1 cfgs) | avg 12.977 / min 12.977 / max 12.977 ms (1 cfgs) | avg 9.811 / min 9.811 / max 9.811 ms (1 cfgs) | avg 8.866 / min 8.866 / max 8.866 ms (1 cfgs) |

## 吞吐量统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| `256` | avg 138.4 / min 138.4 / max 138.4 tok/s (1 cfgs) | avg 30801.0 / min 30801.0 / max 30801.0 tok/s (1 cfgs) | avg 31429.0 / min 31429.0 / max 31429.0 tok/s (1 cfgs) | avg 30913.4 / min 30913.4 / max 30913.4 tok/s (1 cfgs) |
| `512` | avg 73.3 / min 73.3 / max 73.3 tok/s (1 cfgs) | avg 30439.1 / min 30439.1 / max 30439.1 tok/s (1 cfgs) | avg 30237.9 / min 30237.9 / max 30237.9 tok/s (1 cfgs) | avg 27789.8 / min 27789.8 / max 27789.8 tok/s (1 cfgs) |
| `1k` | avg 35.6 / min 35.6 / max 35.6 tok/s (1 cfgs) | avg 25897.8 / min 25897.8 / max 25897.8 tok/s (1 cfgs) | avg 25761.8 / min 25761.8 / max 25761.8 tok/s (1 cfgs) | avg 23350.3 / min 23350.3 / max 23350.3 tok/s (1 cfgs) |
| `2k` | avg 18.4 / min 18.4 / max 18.4 tok/s (1 cfgs) | avg 22391.7 / min 22391.7 / max 22391.7 tok/s (1 cfgs) | avg 22542.3 / min 22542.3 / max 22542.3 tok/s (1 cfgs) | avg 21074.5 / min 21074.5 / max 21074.5 tok/s (1 cfgs) |
| `4k` | avg 9.2 / min 9.2 / max 9.2 tok/s (1 cfgs) | avg 18604.1 / min 18604.1 / max 18604.1 tok/s (1 cfgs) | avg 19153.1 / min 19153.1 / max 19153.1 tok/s (1 cfgs) | avg 19454.8 / min 19454.8 / max 19454.8 tok/s (1 cfgs) |
| `8k` | avg 4.6 / min 4.6 / max 4.6 tok/s (1 cfgs) | avg 12553.6 / min 12553.6 / max 12553.6 tok/s (1 cfgs) | avg 12899.2 / min 12899.2 / max 12899.2 tok/s (1 cfgs) | avg 13645.8 / min 13645.8 / max 13645.8 tok/s (1 cfgs) |
| `16k` | avg 2.3 / min 2.3 / max 2.3 tok/s (1 cfgs) | avg 1090.4 / min 1090.4 / max 1090.4 tok/s (1 cfgs) | avg 8150.3 / min 8150.3 / max 8150.3 tok/s (1 cfgs) | avg 538.7 / min 538.7 / max 538.7 tok/s (1 cfgs) |
| `32k` | avg 1.2 / min 1.2 / max 1.2 tok/s (1 cfgs) | avg 840.3 / min 840.3 / max 840.3 tok/s (1 cfgs) | avg 708.6 / min 708.6 / max 708.6 tok/s (1 cfgs) | avg 1404.9 / min 1404.9 / max 1404.9 tok/s (1 cfgs) |
| `64k` | avg 0.6 / min 0.6 / max 0.6 tok/s (1 cfgs) | avg 631.6 / min 631.6 / max 631.6 tok/s (1 cfgs) | avg 673.0 / min 673.0 / max 673.0 tok/s (1 cfgs) | avg 778.8 / min 778.8 / max 778.8 tok/s (1 cfgs) |
| `128k` | avg 0.3 / min 0.3 / max 0.3 tok/s (1 cfgs) | avg 462.4 / min 462.4 / max 462.4 tok/s (1 cfgs) | avg 611.6 / min 611.6 / max 611.6 tok/s (1 cfgs) | avg 676.7 / min 676.7 / max 676.7 tok/s (1 cfgs) |

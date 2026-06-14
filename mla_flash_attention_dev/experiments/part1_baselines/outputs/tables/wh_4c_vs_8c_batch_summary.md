# Wormhole DeepSeek FlashMLA：4核/8核 Batch 对照

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `H=32`、`H_kv=1`、`value_dim=512`、`rope_dim=64`、`seq_len=8k`、`deepseek_num_q_heads_per_core=8`，对 `batch=1/2/4/6/8/12` 做对照。
说明：此处 `q_shards=4`，所以所需 Q cores 为 `batch * 4`。
说明：4 核和 8 核来自两次独立 rerun；非 DeepSeek baseline 的轻微波动主要看作 run-to-run noise，这张表的重点是 DeepSeek 的支持边界和相对性能。

## DeepSeek 4核 vs 8核

| batch | 所需 Q cores | 4c | 8c | 8c 相对 4c |
|---|---:|---|---|---|
| `1` | `4` | `ok`, `0.464268 ms`, `2153.930 tok/s` | `ok`, `0.476736 ms`, `2097.598 tok/s` | `-2.6%`（略慢） |
| `2` | `8` | `ok`, `0.418401 ms`, `4780.103 tok/s` | `ok`, `0.364708 ms`, `5483.844 tok/s` | `+14.7%` |
| `4` | `16` | `ok`, `0.416083 ms`, `9613.459 tok/s` | `ok`, `0.436574 ms`, `9162.257 tok/s` | `-4.7%`（略慢） |
| `6` | `24` | `ok`, `0.517223 ms`, `11600.405 tok/s` | `ok`, `0.444241 ms`, `13506.173 tok/s` | `+16.4%` |
| `8` | `32` | `unsupported` | `ok`, `0.500324 ms`, `15989.649 tok/s` | 新解锁 |
| `12` | `48` | `unsupported` | `ok`, `0.598540 ms`, `20048.774 tok/s` | 新解锁 |

## 新解锁点与主线方法对照

说明：下面只看 8 核 sweep 内部的同轮结果，便于比较新解锁 batch 点的相对位置。

| batch | DeepSeek FlashMLA (8c) | Flash Attention | FlashMLA (TT Mainline) | 结论 |
|---|---|---|---|---|
| `8` | `0.500324 ms`, `15989.649 tok/s` | `0.639117 ms`, `12517.270 tok/s` | `0.524437 ms`, `15254.444 tok/s` | DeepSeek 8c 最快 |
| `12` | `0.598540 ms`, `20048.774 tok/s` | `0.779554 ms`, `15393.417 tok/s` | `0.582850 ms`, `20588.488 tok/s` | DeepSeek 8c 接近 TT 主线，明显快于 Flash |

## 小结

- 4 核版在这条 `q_shards=4` 路线上，容量上限是 `24 Q cores`，对应 `B<=6`。
- 8 核版把容量上限抬到 `48 Q cores`，对应 `B<=12`，因此 `B=8/12` 从“不支持”变成“可正式 benchmark”。
- 在重叠支持区间里，8 核版不是单调更快，但已经表现出真实性能收益，而不只是功能性扩容。
- 当前最值得继续观察的是：这种收益会不会在更大 `batch`、更多 `heads` 或更长 `seq_len` 下变得更稳定。

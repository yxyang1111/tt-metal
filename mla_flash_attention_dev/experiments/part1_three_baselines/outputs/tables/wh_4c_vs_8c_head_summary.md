# Wormhole DeepSeek FlashMLA：4核/8核 Head 对照

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：固定 `batch=8`、`H_kv=1`、`value_dim=512`、`rope_dim=64`、`seq_len=8k`、`deepseek_num_q_heads_per_core=8`，对 `H=8/16/24/32` 做对照。
说明：此处 `q_shards = H / 8`，所以 `H=8/16/24/32` 分别对应 `q_shards=1/2/3/4`。
说明：4 核和 8 核来自两次独立 rerun；非 DeepSeek baseline 的轻微波动主要看作 run-to-run noise，这张表的重点是 DeepSeek 在不同 `q_shards` 下的支持边界和相对性能。

## DeepSeek 4核 vs 8核

| num_heads | q_shards | 4c | 8c | 8c 相对 4c |
|---|---:|---|---|---|
| `8` | `1` | `ok`, `0.503277 ms`, `15895.808 tok/s` | `ok`, `0.510464 ms`, `15672.026 tok/s` | `-1.4%`（基本持平） |
| `16` | `2` | `ok`, `0.476272 ms`, `16797.136 tok/s` | `ok`, `0.470366 ms`, `17008.044 tok/s` | `+1.3%`（基本持平） |
| `24` | `3` | `ok`, `0.495132 ms`, `16157.319 tok/s` | `ok`, `0.421641 ms`, `18973.501 tok/s` | `+17.4%` |
| `32` | `4` | `unsupported` | `ok`, `0.495343 ms`, `16150.425 tok/s` | 新解锁 |

## 新解锁点与主线方法对照

说明：下面只看 8 核 sweep 内部的同轮结果，便于比较 `H=32` 新解锁点的相对位置。

| num_heads | DeepSeek FlashMLA (8c) | Flash Attention | FlashMLA (TT Mainline) | 结论 |
|---|---|---|---|---|
| `32` | `0.495343 ms`, `16150.425 tok/s` | `0.615567 ms`, `12996.148 tok/s` | `0.529264 ms`, `15115.330 tok/s` | DeepSeek 8c 最快 |

## 小结

- 当 `q_shards=1/2`（`H=8/16`）时，8 核版和 4 核版基本持平，说明这时容量还不是瓶颈。
- 当 `q_shards=3`（`H=24`）时，8 核版已经出现明显收益，说明更宽的 S-block 几何不只是在扩容量，也在改善这一档的并行映射。
- 当 `q_shards=4`（`H=32`）时，4 核版在 `B=8` 下直接超出 `24 Q cores` 上限，而 8 核版不仅解锁，还跑到了三种 device baseline 里最快。
- 这组结果和前面的 batch sweep 能互相印证：8 核版当前最稳定的收益，首先来自把 `q_shards=4` 相关场景从“不支持”推进到“可跑且有竞争力”。

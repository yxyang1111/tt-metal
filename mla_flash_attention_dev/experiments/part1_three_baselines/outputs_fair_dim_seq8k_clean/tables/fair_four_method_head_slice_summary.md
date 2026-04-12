# 公平四方法切片小表

说明：这份小表选取几组代表性的固定 `seq_len`，比较不同 `H` 下四方法的正式 benchmark 结果。
说明：每个单元格格式为 `avg / best / worst ms`；吞吐量单独列在对应小节下。
说明：这里的 `dqhpc / q_shards` 是当前公平策略下 DeepSeek 实际使用的并行度配置。

## 固定 `seq_len=8k`

### 延迟统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---|---|---|---|
| `8` | `2` | `4` | avg 18.959 / best 18.325 / worst 19.545 ms (10/10 kept) | avg 0.240 / best 0.234 / worst 0.250 ms (10/10 kept) | avg 0.237 / best 0.231 / worst 0.242 ms (8/10 kept) | avg 0.238 / best 0.234 / worst 0.242 ms (9/10 kept) |
| `16` | `4` | `4` | avg 28.434 / best 28.027 / worst 28.982 ms (10/10 kept) | avg 0.236 / best 0.228 / worst 0.249 ms (10/10 kept) | avg 0.235 / best 0.227 / worst 0.247 ms (9/10 kept) | avg 0.236 / best 0.225 / worst 0.252 ms (9/10 kept) |
| `24` | `8` | `3` | avg 42.337 / best 41.504 / worst 45.801 ms (9/10 kept) | avg 0.271 / best 0.263 / worst 0.282 ms (9/10 kept) | avg 0.276 / best 0.266 / worst 0.288 ms (10/10 kept) | avg 0.300 / best 0.287 / worst 0.316 ms (9/10 kept) |
| `32` | `8` | `4` | avg 70.697 / best 69.017 / worst 72.553 ms (10/10 kept) | avg 0.251 / best 0.239 / worst 0.268 ms (10/10 kept) | avg 0.273 / best 0.257 / worst 0.286 ms (10/10 kept) | avg 0.262 / best 0.250 / worst 0.269 ms (10/10 kept) |

### 吞吐量统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---:|---:|---:|---:|
| `8` | `2` | `4` | 52.7 tok/s | 4166.5 tok/s | 4213.7 tok/s | 4199.7 tok/s |
| `16` | `4` | `4` | 35.2 tok/s | 4232.7 tok/s | 4252.1 tok/s | 4242.8 tok/s |
| `24` | `8` | `3` | 23.6 tok/s | 3693.8 tok/s | 3626.4 tok/s | 3334.9 tok/s |
| `32` | `8` | `4` | 14.1 tok/s | 3978.1 tok/s | 3661.5 tok/s | 3813.3 tok/s |


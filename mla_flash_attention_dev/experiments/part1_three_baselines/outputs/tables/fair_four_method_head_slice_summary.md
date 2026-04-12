# 公平四方法切片小表

说明：这份小表选取几组代表性的固定 `seq_len`，比较不同 `H` 下四方法的正式 benchmark 结果。
说明：每个单元格格式为 `avg / best / worst ms`；吞吐量单独列在对应小节下。
说明：这里的 `dqhpc / q_shards` 是当前公平策略下 DeepSeek 实际使用的并行度配置。

## 固定 `seq_len=1k`

### 延迟统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---|---|---|---|
| `8` | `2` | `4` | avg 5.313 / best 5.037 / worst 5.921 ms (9/10 kept) | avg 0.183 / best 0.171 / worst 0.203 ms (10/10 kept) | avg 0.189 / best 0.176 / worst 0.214 ms (9/10 kept) | avg 0.187 / best 0.180 / worst 0.202 ms (10/10 kept) |
| `16` | `4` | `4` | avg 15.344 / best 14.844 / worst 16.335 ms (10/10 kept) | avg 0.179 / best 0.165 / worst 0.192 ms (9/10 kept) | avg 0.176 / best 0.164 / worst 0.185 ms (10/10 kept) | avg 0.171 / best 0.159 / worst 0.182 ms (9/10 kept) |
| `24` | `8` | `3` | avg 27.687 / best 27.065 / worst 28.357 ms (10/10 kept) | avg 0.215 / best 0.194 / worst 0.273 ms (10/10 kept) | avg 0.208 / best 0.189 / worst 0.248 ms (10/10 kept) | avg 0.216 / best 0.199 / worst 0.239 ms (9/10 kept) |
| `32` | `8` | `4` | avg 39.754 / best 39.288 / worst 40.445 ms (10/10 kept) | avg 0.196 / best 0.181 / worst 0.210 ms (10/10 kept) | avg 0.203 / best 0.181 / worst 0.212 ms (9/10 kept) | avg 0.181 / best 0.171 / worst 0.190 ms (9/10 kept) |

### 吞吐量统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---:|---:|---:|---:|
| `8` | `2` | `4` | 188.2 tok/s | 5450.9 tok/s | 5285.5 tok/s | 5347.0 tok/s |
| `16` | `4` | `4` | 65.2 tok/s | 5584.5 tok/s | 5681.0 tok/s | 5848.0 tok/s |
| `24` | `8` | `3` | 36.1 tok/s | 4640.6 tok/s | 4811.7 tok/s | 4619.7 tok/s |
| `32` | `8` | `4` | 25.2 tok/s | 5089.9 tok/s | 4936.7 tok/s | 5522.6 tok/s |

## 固定 `seq_len=8k`

### 延迟统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---|---|---|---|
| `8` | `2` | `4` | avg 57.758 / best 55.941 / worst 59.858 ms (10/10 kept) | avg 0.370 / best 0.354 / worst 0.409 ms (9/10 kept) | avg 0.364 / best 0.353 / worst 0.375 ms (9/10 kept) | avg 0.364 / best 0.356 / worst 0.379 ms (10/10 kept) |
| `16` | `4` | `4` | avg 115.062 / best 111.995 / worst 117.407 ms (10/10 kept) | avg 0.355 / best 0.343 / worst 0.371 ms (9/10 kept) | avg 0.362 / best 0.351 / worst 0.373 ms (9/10 kept) | avg 0.358 / best 0.350 / worst 0.375 ms (10/10 kept) |
| `24` | `8` | `3` | avg 203.234 / best 199.574 / worst 208.438 ms (10/10 kept) | avg 0.412 / best 0.407 / worst 0.418 ms (9/10 kept) | avg 0.428 / best 0.412 / worst 0.440 ms (9/10 kept) | avg 0.523 / best 0.509 / worst 0.568 ms (9/10 kept) |
| `32` | `8` | `4` | avg 264.559 / best 262.147 / worst 268.467 ms (10/10 kept) | avg 0.382 / best 0.378 / worst 0.395 ms (9/10 kept) | avg 0.473 / best 0.463 / worst 0.484 ms (10/10 kept) | avg 0.462 / best 0.449 / worst 0.469 ms (10/10 kept) |

### 吞吐量统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---:|---:|---:|---:|
| `8` | `2` | `4` | 17.3 tok/s | 2705.2 tok/s | 2743.5 tok/s | 2750.5 tok/s |
| `16` | `4` | `4` | 8.7 tok/s | 2819.2 tok/s | 2762.1 tok/s | 2789.9 tok/s |
| `24` | `8` | `3` | 4.9 tok/s | 2424.8 tok/s | 2336.5 tok/s | 1912.1 tok/s |
| `32` | `8` | `4` | 3.8 tok/s | 2615.7 tok/s | 2112.0 tok/s | 2163.7 tok/s |

## 固定 `seq_len=32k`

### 延迟统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---|---|---|---|
| `8` | `2` | `4` | avg 226.829 / best 223.891 / worst 230.961 ms (10/10 kept) | avg 0.998 / best 0.981 / worst 1.082 ms (10/10 kept) | avg 1.011 / best 0.992 / worst 1.081 ms (10/10 kept) | avg 0.999 / best 0.984 / worst 1.018 ms (10/10 kept) |
| `16` | `4` | `4` | avg 455.366 / best 426.969 / worst 474.442 ms (10/10 kept) | avg 0.973 / best 0.962 / worst 0.987 ms (9/10 kept) | avg 0.997 / best 0.985 / worst 1.010 ms (10/10 kept) | avg 0.998 / best 0.991 / worst 1.010 ms (10/10 kept) |
| `24` | `8` | `3` | avg 701.401 / best 687.705 / worst 721.839 ms (10/10 kept) | avg 1.207 / best 1.180 / worst 1.279 ms (10/10 kept) | avg 1.230 / best 1.216 / worst 1.243 ms (10/10 kept) | avg 1.587 / best 1.577 / worst 1.600 ms (10/10 kept) |
| `32` | `8` | `4` | avg 916.238 / best 900.841 / worst 952.396 ms (10/10 kept) | avg 1.196 / best 1.178 / worst 1.259 ms (10/10 kept) | avg 1.224 / best 1.214 / worst 1.236 ms (10/10 kept) | avg 1.216 / best 1.211 / worst 1.230 ms (9/10 kept) |

### 吞吐量统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---:|---:|---:|---:|
| `8` | `2` | `4` | 4.4 tok/s | 1002.1 tok/s | 989.1 tok/s | 1000.9 tok/s |
| `16` | `4` | `4` | 2.2 tok/s | 1027.3 tok/s | 1002.9 tok/s | 1001.9 tok/s |
| `24` | `8` | `3` | 1.4 tok/s | 828.3 tok/s | 813.3 tok/s | 630.2 tok/s |
| `32` | `8` | `4` | 1.1 tok/s | 835.9 tok/s | 817.2 tok/s | 822.1 tok/s |

## 固定 `seq_len=128k`

### 延迟统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---|---|---|---|
| `8` | `2` | `4` | avg 860.125 / best 837.438 / worst 919.300 ms (9/10 kept) | avg 28.081 / best 3.487 / worst 49.048 ms (10/10 kept) | avg 3.594 / best 3.579 / worst 3.676 ms (10/10 kept) | avg 3.555 / best 3.540 / worst 3.566 ms (8/10 kept) |
| `16` | `4` | `4` | avg 1825.631 / best 1731.216 / worst 1977.409 ms (10/10 kept) | avg 3.448 / best 3.436 / worst 3.460 ms (8/10 kept) | avg 18.704 / best 3.552 / worst 41.116 ms (10/10 kept) | avg 3.569 / best 3.540 / worst 3.667 ms (10/10 kept) |
| `24` | `8` | `3` | avg 2569.375 / best 2540.818 / worst 2621.853 ms (10/10 kept) | avg 26.805 / best 4.310 / worst 53.253 ms (10/10 kept) | avg 4.477 / best 4.451 / worst 4.522 ms (8/10 kept) | avg 37.599 / best 31.962 / worst 46.847 ms (8/10 kept) |
| `32` | `8` | `4` | avg 3425.909 / best 3383.744 / worst 3507.020 ms (10/10 kept) | avg 29.940 / best 4.300 / worst 92.673 ms (10/10 kept) | avg 35.604 / best 4.428 / worst 74.150 ms (10/10 kept) | avg 4.406 / best 4.381 / worst 4.433 ms (6/10 kept) |

### 吞吐量统计

| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---:|---:|---:|---:|---:|---:|---:|
| `8` | `2` | `4` | 1.2 tok/s | 35.6 tok/s | 278.2 tok/s | 281.3 tok/s |
| `16` | `4` | `4` | 0.5 tok/s | 290.1 tok/s | 53.5 tok/s | 280.2 tok/s |
| `24` | `8` | `3` | 0.4 tok/s | 37.3 tok/s | 223.3 tok/s | 26.6 tok/s |
| `32` | `8` | `4` | 0.3 tok/s | 33.4 tok/s | 28.1 tok/s | 227.0 tok/s |


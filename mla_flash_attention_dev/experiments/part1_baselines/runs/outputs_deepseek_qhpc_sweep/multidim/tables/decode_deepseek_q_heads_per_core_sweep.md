# Decode DeepSeek Q-Heads-Per-Core Sweep

说明：固定 `seq_len=8k`，其余轴保持默认 config：`B=1, H=16, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| dqhpc=1, q_shards=16 | avg 111.120 / best 108.168 / worst 114.654 ms (10/10 kept) | avg 0.364 / best 0.353 / worst 0.380 ms (10/10 kept) | avg 0.363 / best 0.351 / worst 0.384 ms (8/10 kept) | avg 0.165 / best 0.157 / worst 0.176 ms (9/10 kept) |
| dqhpc=2, q_shards=8 | avg 143.760 / best 140.440 / worst 154.861 ms (10/10 kept) | avg 0.358 / best 0.321 / worst 0.371 ms (8/10 kept) | avg 7.600 / best 0.339 / worst 11.320 ms (10/10 kept) | avg 0.275 / best 0.266 / worst 0.302 ms (9/10 kept) |
| dqhpc=4, q_shards=4 | avg 150.660 / best 146.075 / worst 155.332 ms (10/10 kept) | avg 5.287 / best 0.310 / worst 11.922 ms (10/10 kept) | avg 3.172 / best 0.319 / worst 8.346 ms (10/10 kept) | avg 0.362 / best 0.349 / worst 0.394 ms (10/10 kept) |
| dqhpc=8, q_shards=2 | avg 146.510 / best 142.622 / worst 151.559 ms (9/10 kept) | avg 1.857 / best 0.290 / worst 8.088 ms (8/10 kept) | avg 6.904 / best 0.307 / worst 25.371 ms (10/10 kept) | avg 0.575 / best 0.554 / worst 0.635 ms (10/10 kept) |

## 平均吞吐量

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| dqhpc=1, q_shards=16 | 9.0 tok/s | 2749.5 tok/s | 2753.4 tok/s | 6074.5 tok/s |
| dqhpc=2, q_shards=8 | 7.0 tok/s | 2794.5 tok/s | 131.6 tok/s | 3632.3 tok/s |
| dqhpc=4, q_shards=4 | 6.6 tok/s | 189.1 tok/s | 315.2 tok/s | 2759.5 tok/s |
| dqhpc=8, q_shards=2 | 6.8 tok/s | 538.5 tok/s | 144.8 tok/s | 1738.3 tok/s |

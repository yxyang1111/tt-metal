# Decode Head Sweep

说明：固定 `seq_len=8k`，其余轴保持默认 config：`B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| H=8 (dqhpc=2, q_shards=4) | avg 62.169 / best 60.705 / worst 65.494 ms (9/10 kept) | avg 0.382 / best 0.370 / worst 0.417 ms (8/10 kept) | avg 0.382 / best 0.370 / worst 0.400 ms (9/10 kept) | avg 0.364 / best 0.350 / worst 0.388 ms (10/10 kept) |
| H=16 (dqhpc=4, q_shards=4) | avg 123.699 / best 121.842 / worst 131.133 ms (8/10 kept) | avg 0.389 / best 0.369 / worst 0.437 ms (9/10 kept) | avg 0.385 / best 0.379 / worst 0.402 ms (9/10 kept) | avg 0.383 / best 0.372 / worst 0.410 ms (10/10 kept) |
| H=24 (dqhpc=8, q_shards=3) | avg 176.354 / best 173.959 / worst 178.615 ms (10/10 kept) | avg 0.414 / best 0.401 / worst 0.430 ms (10/10 kept) | avg 0.432 / best 0.424 / worst 0.439 ms (9/10 kept) | avg 0.520 / best 0.510 / worst 0.536 ms (10/10 kept) |
| H=32 (dqhpc=8, q_shards=4) | avg 233.072 / best 227.949 / worst 236.284 ms (10/10 kept) | avg 0.408 / best 0.396 / worst 0.428 ms (10/10 kept) | avg 0.426 / best 0.412 / worst 0.438 ms (10/10 kept) | avg 0.438 / best 0.407 / worst 0.521 ms (10/10 kept) |

## 平均吞吐量

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| H=8 (dqhpc=2, q_shards=4) | 16.1 tok/s | 2617.0 tok/s | 2619.5 tok/s | 2748.6 tok/s |
| H=16 (dqhpc=4, q_shards=4) | 8.1 tok/s | 2571.6 tok/s | 2594.2 tok/s | 2610.3 tok/s |
| H=24 (dqhpc=8, q_shards=3) | 5.7 tok/s | 2415.3 tok/s | 2317.2 tok/s | 1922.3 tok/s |
| H=32 (dqhpc=8, q_shards=4) | 4.3 tok/s | 2451.6 tok/s | 2349.8 tok/s | 2282.7 tok/s |

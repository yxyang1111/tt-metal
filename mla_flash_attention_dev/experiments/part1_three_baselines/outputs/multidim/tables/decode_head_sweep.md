# Decode Head Sweep

说明：固定 `seq_len=8k`，其余轴保持默认 config：`B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| H=8 (dqhpc=2, q_shards=4) | avg 57.758 / best 55.941 / worst 59.858 ms (10/10 kept) | avg 0.370 / best 0.354 / worst 0.409 ms (9/10 kept) | avg 0.364 / best 0.353 / worst 0.375 ms (9/10 kept) | avg 0.364 / best 0.356 / worst 0.379 ms (10/10 kept) |
| H=16 (dqhpc=4, q_shards=4) | avg 115.062 / best 111.995 / worst 117.407 ms (10/10 kept) | avg 0.355 / best 0.343 / worst 0.371 ms (9/10 kept) | avg 0.362 / best 0.351 / worst 0.373 ms (9/10 kept) | avg 0.358 / best 0.350 / worst 0.375 ms (10/10 kept) |
| H=24 (dqhpc=8, q_shards=3) | avg 203.234 / best 199.574 / worst 208.438 ms (10/10 kept) | avg 0.412 / best 0.407 / worst 0.418 ms (9/10 kept) | avg 0.428 / best 0.412 / worst 0.440 ms (9/10 kept) | avg 0.523 / best 0.509 / worst 0.568 ms (9/10 kept) |
| H=32 (dqhpc=8, q_shards=4) | avg 264.559 / best 262.147 / worst 268.467 ms (10/10 kept) | avg 0.382 / best 0.378 / worst 0.395 ms (9/10 kept) | avg 0.473 / best 0.463 / worst 0.484 ms (10/10 kept) | avg 0.462 / best 0.449 / worst 0.469 ms (10/10 kept) |

## 平均吞吐量

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| H=8 (dqhpc=2, q_shards=4) | 17.3 tok/s | 2705.2 tok/s | 2743.5 tok/s | 2750.5 tok/s |
| H=16 (dqhpc=4, q_shards=4) | 8.7 tok/s | 2819.2 tok/s | 2762.1 tok/s | 2789.9 tok/s |
| H=24 (dqhpc=8, q_shards=3) | 4.9 tok/s | 2424.8 tok/s | 2336.5 tok/s | 1912.1 tok/s |
| H=32 (dqhpc=8, q_shards=4) | 3.8 tok/s | 2615.7 tok/s | 2112.0 tok/s | 2163.7 tok/s |

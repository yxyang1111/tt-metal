# Decode Head Sweep

说明：固定 `seq_len=8k`，其余轴保持默认 config：`B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| H=8 (dqhpc=2, q_shards=4) | avg 66.321 / best 65.159 / worst 67.817 ms (10/10 kept) | avg 0.355 / best 0.344 / worst 0.368 ms (10/10 kept) | avg 0.361 / best 0.349 / worst 0.372 ms (10/10 kept) | avg 0.362 / best 0.346 / worst 0.373 ms (10/10 kept) |
| H=16 (dqhpc=4, q_shards=4) | avg 122.849 / best 119.776 / worst 126.231 ms (10/10 kept) | avg 0.372 / best 0.366 / worst 0.383 ms (8/10 kept) | avg 0.372 / best 0.365 / worst 0.387 ms (9/10 kept) | avg 0.379 / best 0.365 / worst 0.396 ms (9/10 kept) |
| H=24 (dqhpc=8, q_shards=3) | avg 210.090 / best 206.710 / worst 212.662 ms (10/10 kept) | avg 0.417 / best 0.409 / worst 0.427 ms (10/10 kept) | avg 0.431 / best 0.421 / worst 0.455 ms (8/10 kept) | avg 0.516 / best 0.504 / worst 0.522 ms (10/10 kept) |
| H=32 (dqhpc=8, q_shards=4) | avg 266.574 / best 263.813 / worst 271.783 ms (10/10 kept) | avg 0.417 / best 0.405 / worst 0.424 ms (9/10 kept) | avg 0.427 / best 0.421 / worst 0.435 ms (9/10 kept) | avg 0.424 / best 0.409 / worst 0.434 ms (9/10 kept) |

## 平均吞吐量

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| H=8 (dqhpc=2, q_shards=4) | 15.1 tok/s | 2819.6 tok/s | 2773.1 tok/s | 2762.0 tok/s |
| H=16 (dqhpc=4, q_shards=4) | 8.1 tok/s | 2691.4 tok/s | 2685.3 tok/s | 2639.1 tok/s |
| H=24 (dqhpc=8, q_shards=3) | 4.8 tok/s | 2400.8 tok/s | 2321.6 tok/s | 1936.6 tok/s |
| H=32 (dqhpc=8, q_shards=4) | 3.8 tok/s | 2398.8 tok/s | 2342.6 tok/s | 2361.2 tok/s |

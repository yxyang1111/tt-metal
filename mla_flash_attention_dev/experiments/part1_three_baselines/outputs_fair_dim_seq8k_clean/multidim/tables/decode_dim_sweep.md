# Decode Dim Sweep

说明：固定 `seq_len=8k`，其余轴保持默认 config：`B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| value=128, rope=64 | avg 70.697 / best 69.017 / worst 72.553 ms (10/10 kept) | avg 0.251 / best 0.239 / worst 0.268 ms (10/10 kept) | avg 0.273 / best 0.257 / worst 0.286 ms (10/10 kept) | avg 0.262 / best 0.250 / worst 0.269 ms (10/10 kept) |
| value=192, rope=64 | avg 112.980 / best 107.385 / worst 115.786 ms (10/10 kept) | avg 0.281 / best 0.268 / worst 0.296 ms (9/10 kept) | avg 0.292 / best 0.282 / worst 0.305 ms (9/10 kept) | avg 0.285 / best 0.276 / worst 0.292 ms (9/10 kept) |
| value=256, rope=64 | avg 147.245 / best 145.888 / worst 148.347 ms (10/10 kept) | avg 0.311 / best 0.298 / worst 0.323 ms (10/10 kept) | avg 0.321 / best 0.314 / worst 0.335 ms (10/10 kept) | avg 0.304 / best 0.294 / worst 0.314 ms (9/10 kept) |
| value=384, rope=64 | avg 210.914 / best 208.705 / worst 214.525 ms (10/10 kept) | avg 0.339 / best 0.331 / worst 0.349 ms (10/10 kept) | avg 0.356 / best 0.348 / worst 0.372 ms (9/10 kept) | avg 0.346 / best 0.338 / worst 0.360 ms (9/10 kept) |
| value=512, rope=64 | avg 266.574 / best 263.813 / worst 271.783 ms (10/10 kept) | avg 0.417 / best 0.405 / worst 0.424 ms (9/10 kept) | avg 0.427 / best 0.421 / worst 0.435 ms (9/10 kept) | avg 0.424 / best 0.409 / worst 0.434 ms (9/10 kept) |

## 平均吞吐量

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| value=128, rope=64 | 14.1 tok/s | 3978.1 tok/s | 3661.5 tok/s | 3813.3 tok/s |
| value=192, rope=64 | 8.9 tok/s | 3563.1 tok/s | 3427.6 tok/s | 3511.2 tok/s |
| value=256, rope=64 | 6.8 tok/s | 3210.7 tok/s | 3115.1 tok/s | 3284.2 tok/s |
| value=384, rope=64 | 4.7 tok/s | 2949.0 tok/s | 2808.2 tok/s | 2890.3 tok/s |
| value=512, rope=64 | 3.8 tok/s | 2398.8 tok/s | 2342.6 tok/s | 2361.2 tok/s |

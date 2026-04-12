# Decode Batch Sweep

说明：固定 `seq_len=8k`，其余轴保持默认 config：`B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。

## 延迟统计

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| B=1 | avg 233.072 / best 227.949 / worst 236.284 ms (10/10 kept) | avg 0.408 / best 0.396 / worst 0.428 ms (10/10 kept) | avg 0.426 / best 0.412 / worst 0.438 ms (10/10 kept) | avg 0.438 / best 0.407 / worst 0.521 ms (10/10 kept) |
| B=2 | avg 436.862 / best 434.643 / worst 442.518 ms (10/10 kept) | avg 0.428 / best 0.416 / worst 0.442 ms (10/10 kept) | avg 0.422 / best 0.413 / worst 0.434 ms (10/10 kept) | avg 0.312 / best 0.298 / worst 0.324 ms (10/10 kept) |
| B=4 | avg 853.795 / best 849.462 / worst 861.650 ms (10/10 kept) | avg 0.456 / best 0.433 / worst 0.490 ms (9/10 kept) | avg 0.447 / best 0.436 / worst 0.459 ms (9/10 kept) | avg 0.350 / best 0.344 / worst 0.357 ms (9/10 kept) |
| B=8 | avg 1675.706 / best 1641.566 / worst 1742.885 ms (10/10 kept) | avg 0.564 / best 0.550 / worst 0.611 ms (10/10 kept) | avg 0.477 / best 0.464 / worst 0.488 ms (9/10 kept) | - |
| B=16 | avg 3430.652 / best 3312.555 / worst 3503.231 ms (10/10 kept) | avg 0.921 / best 0.905 / worst 0.945 ms (10/10 kept) | avg 0.669 / best 0.649 / worst 0.695 ms (10/10 kept) | - |

## 平均吞吐量

| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| B=1 | 4.3 tok/s | 2451.6 tok/s | 2349.8 tok/s | 2282.7 tok/s |
| B=2 | 4.6 tok/s | 4673.3 tok/s | 4737.3 tok/s | 6400.6 tok/s |
| B=4 | 4.7 tok/s | 8779.5 tok/s | 8945.8 tok/s | 11426.7 tok/s |
| B=8 | 4.8 tok/s | 14187.4 tok/s | 16759.2 tok/s | - |
| B=16 | 4.7 tok/s | 17372.6 tok/s | 23922.8 tok/s | - |

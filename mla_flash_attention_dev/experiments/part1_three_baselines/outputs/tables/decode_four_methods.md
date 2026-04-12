# Decode 四方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。
说明：这张默认主表固定在 `config=b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，即 `B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`；更多 B/H/dims 对比见 `multidim/` 目录。
说明：`DeepSeek FlashMLA` 采用“预先完成一次张量适配后，只计时 backend device op”的口径，因此可与其它 TT device baseline 更公平比较。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| decode_256 | avg 1.410 / best 1.255 / worst 1.776 ms (10/10 kept) | avg 0.172 / best 0.162 / worst 0.186 ms (10/10 kept) | avg 0.165 / best 0.156 / worst 0.182 ms (9/10 kept) | avg 0.185 / best 0.169 / worst 0.198 ms (9/10 kept) |
| decode_512 | avg 20.648 / best 20.197 / worst 20.949 ms (10/10 kept) | avg 0.179 / best 0.170 / worst 0.198 ms (8/10 kept) | avg 0.178 / best 0.168 / worst 0.188 ms (10/10 kept) | avg 0.171 / best 0.157 / worst 0.198 ms (10/10 kept) |
| decode_1k | avg 39.754 / best 39.288 / worst 40.445 ms (10/10 kept) | avg 0.196 / best 0.181 / worst 0.210 ms (10/10 kept) | avg 0.203 / best 0.181 / worst 0.212 ms (9/10 kept) | avg 0.181 / best 0.171 / worst 0.190 ms (9/10 kept) |
| decode_2k | avg 74.994 / best 72.296 / worst 81.065 ms (10/10 kept) | avg 0.229 / best 0.218 / worst 0.240 ms (10/10 kept) | avg 0.234 / best 0.224 / worst 0.244 ms (9/10 kept) | avg 0.215 / best 0.206 / worst 0.224 ms (9/10 kept) |
| decode_4k | avg 146.124 / best 143.216 / worst 148.202 ms (10/10 kept) | avg 0.285 / best 0.276 / worst 0.297 ms (10/10 kept) | avg 0.312 / best 0.290 / worst 0.352 ms (9/10 kept) | avg 0.298 / best 0.283 / worst 0.312 ms (9/10 kept) |
| decode_8k | avg 264.559 / best 262.147 / worst 268.467 ms (10/10 kept) | avg 0.382 / best 0.378 / worst 0.395 ms (9/10 kept) | avg 0.473 / best 0.463 / worst 0.484 ms (10/10 kept) | avg 0.462 / best 0.449 / worst 0.469 ms (10/10 kept) |
| decode_16k | avg 507.768 / best 495.588 / worst 522.738 ms (10/10 kept) | avg 0.672 / best 0.663 / worst 0.692 ms (10/10 kept) | avg 0.699 / best 0.687 / worst 0.713 ms (9/10 kept) | avg 0.681 / best 0.673 / worst 0.699 ms (9/10 kept) |
| decode_32k | avg 916.238 / best 900.841 / worst 952.396 ms (10/10 kept) | avg 1.196 / best 1.178 / worst 1.259 ms (10/10 kept) | avg 1.224 / best 1.214 / worst 1.236 ms (10/10 kept) | avg 1.216 / best 1.211 / worst 1.230 ms (9/10 kept) |
| decode_64k | avg 1671.564 / best 1618.243 / worst 1725.478 ms (10/10 kept) | avg 2.224 / best 2.192 / worst 2.275 ms (9/10 kept) | avg 2.303 / best 2.272 / worst 2.351 ms (9/10 kept) | avg 2.279 / best 2.253 / worst 2.314 ms (10/10 kept) |
| decode_128k | avg 3425.909 / best 3383.744 / worst 3507.020 ms (10/10 kept) | avg 29.940 / best 4.300 / worst 92.673 ms (10/10 kept) | avg 35.604 / best 4.428 / worst 74.150 ms (10/10 kept) | avg 4.406 / best 4.381 / worst 4.433 ms (6/10 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| decode_256 | 709.4 tok/s | 5829.4 tok/s | 6079.0 tok/s | 5409.8 tok/s |
| decode_512 | 48.4 tok/s | 5581.9 tok/s | 5623.3 tok/s | 5847.9 tok/s |
| decode_1k | 25.2 tok/s | 5089.9 tok/s | 4936.7 tok/s | 5522.6 tok/s |
| decode_2k | 13.3 tok/s | 4361.7 tok/s | 4276.2 tok/s | 4660.3 tok/s |
| decode_4k | 6.8 tok/s | 3502.7 tok/s | 3204.5 tok/s | 3351.4 tok/s |
| decode_8k | 3.8 tok/s | 2615.7 tok/s | 2112.0 tok/s | 2163.7 tok/s |
| decode_16k | 2.0 tok/s | 1487.7 tok/s | 1429.8 tok/s | 1469.1 tok/s |
| decode_32k | 1.1 tok/s | 835.9 tok/s | 817.2 tok/s | 822.1 tok/s |
| decode_64k | 0.6 tok/s | 449.6 tok/s | 434.2 tok/s | 438.8 tok/s |
| decode_128k | 0.3 tok/s | 33.4 tok/s | 28.1 tok/s | 227.0 tok/s |

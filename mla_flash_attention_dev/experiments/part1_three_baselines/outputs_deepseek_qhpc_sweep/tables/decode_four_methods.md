# Decode 四方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。
说明：这张默认主表固定在 `config=b1_h16_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，即 `B=1, H=16, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`；更多 B/H/dims 对比见 `multidim/` 目录。
说明：`DeepSeek FlashMLA` 采用“预先完成一次张量适配后，只计时 backend device op”的口径，因此可与其它 TT device baseline 更公平比较。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| decode_256 | avg 1.135 / best 1.055 / worst 1.389 ms (8/10 kept) | avg 13.577 / best 0.179 / worst 25.535 ms (10/10 kept) | avg 15.306 / best 0.097 / worst 23.528 ms (10/10 kept) | avg 0.173 / best 0.157 / worst 0.198 ms (9/10 kept) |
| decode_512 | avg 1.741 / best 1.639 / worst 2.041 ms (10/10 kept) | avg 3.905 / best 0.104 / worst 10.958 ms (10/10 kept) | avg 3.626 / best 0.113 / worst 9.200 ms (10/10 kept) | avg 0.181 / best 0.161 / worst 0.193 ms (9/10 kept) |
| decode_1k | avg 11.630 / best 11.307 / worst 12.458 ms (10/10 kept) | avg 2.494 / best 0.126 / worst 8.976 ms (9/10 kept) | avg 6.780 / best 0.129 / worst 15.644 ms (10/10 kept) | avg 0.196 / best 0.183 / worst 0.208 ms (9/10 kept) |
| decode_2k | avg 37.048 / best 33.868 / worst 45.164 ms (10/10 kept) | avg 4.245 / best 0.142 / worst 11.537 ms (10/10 kept) | avg 3.292 / best 0.145 / worst 9.101 ms (10/10 kept) | avg 0.247 / best 0.235 / worst 0.269 ms (9/10 kept) |
| decode_4k | avg 71.107 / best 69.910 / worst 74.335 ms (9/10 kept) | avg 2.247 / best 0.191 / worst 7.918 ms (10/10 kept) | avg 3.136 / best 0.211 / worst 10.660 ms (10/10 kept) | avg 0.352 / best 0.335 / worst 0.363 ms (10/10 kept) |
| decode_8k | avg 146.510 / best 142.622 / worst 151.559 ms (9/10 kept) | avg 1.857 / best 0.290 / worst 8.088 ms (8/10 kept) | avg 6.904 / best 0.307 / worst 25.371 ms (10/10 kept) | avg 0.575 / best 0.554 / worst 0.635 ms (10/10 kept) |
| decode_16k | avg 271.443 / best 270.204 / worst 273.772 ms (10/10 kept) | avg 7.522 / best 0.503 / worst 16.449 ms (10/10 kept) | avg 8.979 / best 0.523 / worst 27.619 ms (10/10 kept) | avg 1.004 / best 0.984 / worst 1.071 ms (10/10 kept) |
| decode_32k | avg 508.424 / best 500.885 / worst 529.386 ms (10/10 kept) | avg 1.336 / best 0.920 / worst 2.964 ms (8/10 kept) | avg 16.633 / best 0.957 / worst 27.319 ms (10/10 kept) | avg 5.780 / best 1.856 / worst 10.346 ms (10/10 kept) |
| decode_64k | avg 941.778 / best 933.464 / worst 954.178 ms (10/10 kept) | avg 12.233 / best 1.769 / worst 28.603 ms (10/10 kept) | avg 1.838 / best 1.804 / worst 1.981 ms (7/10 kept) | avg 10.493 / best 3.549 / worst 18.779 ms (9/10 kept) |
| decode_128k | avg 1804.216 / best 1787.439 / worst 1850.707 ms (10/10 kept) | avg 3.430 / best 3.392 / worst 3.560 ms (8/10 kept) | avg 6.740 / best 3.514 / worst 11.169 ms (10/10 kept) | avg 7.013 / best 6.952 / worst 7.051 ms (9/10 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| decode_256 | 880.8 tok/s | 73.7 tok/s | 65.3 tok/s | 5779.5 tok/s |
| decode_512 | 574.5 tok/s | 256.1 tok/s | 275.7 tok/s | 5531.9 tok/s |
| decode_1k | 86.0 tok/s | 400.9 tok/s | 147.5 tok/s | 5096.1 tok/s |
| decode_2k | 27.0 tok/s | 235.6 tok/s | 303.8 tok/s | 4053.2 tok/s |
| decode_4k | 14.1 tok/s | 445.1 tok/s | 318.8 tok/s | 2844.4 tok/s |
| decode_8k | 6.8 tok/s | 538.5 tok/s | 144.8 tok/s | 1738.3 tok/s |
| decode_16k | 3.7 tok/s | 133.0 tok/s | 111.4 tok/s | 995.9 tok/s |
| decode_32k | 2.0 tok/s | 748.7 tok/s | 60.1 tok/s | 173.0 tok/s |
| decode_64k | 1.1 tok/s | 81.7 tok/s | 544.2 tok/s | 95.3 tok/s |
| decode_128k | 0.6 tok/s | 291.5 tok/s | 148.4 tok/s | 142.6 tok/s |

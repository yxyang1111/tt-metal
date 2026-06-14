# Decode 四方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。
说明：`DeepSeek FlashMLA` 采用“预先完成一次张量适配后，只计时 backend device op”的口径，因此可与其它 TT device baseline 更公平比较。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| decode_256 | avg 1.257 / best 1.091 / worst 1.455 ms (8/10 kept) | avg 0.169 / best 0.162 / worst 0.185 ms (9/10 kept) | avg 0.196 / best 0.172 / worst 0.265 ms (9/10 kept) | avg 0.177 / best 0.160 / worst 0.206 ms (9/10 kept) |
| decode_512 | avg 16.585 / best 15.973 / worst 17.141 ms (10/10 kept) | avg 0.177 / best 0.166 / worst 0.197 ms (10/10 kept) | avg 0.108 / best 0.103 / worst 0.115 ms (9/10 kept) | avg 0.210 / best 0.189 / worst 0.244 ms (10/10 kept) |
| decode_1k | avg 30.467 / best 30.095 / worst 31.577 ms (10/10 kept) | avg 0.204 / best 0.196 / worst 0.213 ms (9/10 kept) | avg 0.209 / best 0.197 / worst 0.226 ms (10/10 kept) | avg 0.180 / best 0.173 / worst 0.192 ms (9/10 kept) |
| decode_2k | avg 58.991 / best 57.609 / worst 60.943 ms (10/10 kept) | avg 0.229 / best 0.216 / worst 0.262 ms (10/10 kept) | avg 0.217 / best 0.206 / worst 0.233 ms (9/10 kept) | avg 0.208 / best 0.199 / worst 0.223 ms (10/10 kept) |
| decode_4k | avg 123.463 / best 120.992 / worst 128.236 ms (10/10 kept) | avg 0.277 / best 0.267 / worst 0.290 ms (9/10 kept) | avg 0.291 / best 0.279 / worst 0.317 ms (9/10 kept) | avg 0.298 / best 0.284 / worst 0.333 ms (9/10 kept) |
| decode_8k | avg 245.915 / best 222.086 / worst 288.117 ms (10/10 kept) | avg 0.416 / best 0.400 / worst 0.442 ms (10/10 kept) | avg 0.426 / best 0.418 / worst 0.440 ms (9/10 kept) | avg 0.432 / best 0.413 / worst 0.462 ms (10/10 kept) |
| decode_16k | avg 502.884 / best 493.871 / worst 535.022 ms (10/10 kept) | avg 0.686 / best 0.677 / worst 0.708 ms (9/10 kept) | avg 0.708 / best 0.699 / worst 0.724 ms (10/10 kept) | avg 0.690 / best 0.676 / worst 0.730 ms (10/10 kept) |
| decode_32k | avg 980.514 / best 952.253 / worst 1039.562 ms (10/10 kept) | avg 1.198 / best 1.178 / worst 1.249 ms (10/10 kept) | avg 1.240 / best 1.223 / worst 1.267 ms (10/10 kept) | avg 1.237 / best 1.228 / worst 1.254 ms (10/10 kept) |
| decode_64k | avg 1941.513 / best 1905.613 / worst 2004.374 ms (10/10 kept) | avg 2.232 / best 2.213 / worst 2.258 ms (10/10 kept) | avg 2.320 / best 2.301 / worst 2.386 ms (9/10 kept) | avg 2.331 / best 2.292 / worst 2.501 ms (10/10 kept) |
| decode_128k | avg 3820.783 / best 3649.933 / worst 3945.519 ms (10/10 kept) | avg 4.302 / best 4.287 / worst 4.326 ms (10/10 kept) | avg 4.430 / best 4.420 / worst 4.454 ms (10/10 kept) | avg 4.435 / best 4.418 / worst 4.459 ms (7/10 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| decode_256 | 795.7 tok/s | 5908.6 tok/s | 5103.7 tok/s | 5645.2 tok/s |
| decode_512 | 60.3 tok/s | 5640.4 tok/s | 9256.1 tok/s | 4764.3 tok/s |
| decode_1k | 32.8 tok/s | 4897.8 tok/s | 4795.8 tok/s | 5553.4 tok/s |
| decode_2k | 17.0 tok/s | 4370.3 tok/s | 4615.6 tok/s | 4797.2 tok/s |
| decode_4k | 8.1 tok/s | 3607.8 tok/s | 3437.8 tok/s | 3353.7 tok/s |
| decode_8k | 4.1 tok/s | 2404.1 tok/s | 2348.4 tok/s | 2313.0 tok/s |
| decode_16k | 2.0 tok/s | 1457.9 tok/s | 1412.9 tok/s | 1448.3 tok/s |
| decode_32k | 1.0 tok/s | 835.1 tok/s | 806.6 tok/s | 808.7 tok/s |
| decode_64k | 0.5 tok/s | 448.0 tok/s | 431.1 tok/s | 429.0 tok/s |
| decode_128k | 0.3 tok/s | 232.4 tok/s | 225.8 tok/s | 225.5 tok/s |

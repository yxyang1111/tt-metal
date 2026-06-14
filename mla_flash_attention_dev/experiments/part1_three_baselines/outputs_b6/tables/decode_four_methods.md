# Decode 四方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。
说明：`DeepSeek FlashMLA` 采用“预先完成一次张量适配后，只计时 backend device op”的口径，因此可与其它 TT device baseline 更公平比较。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| decode_256 | avg 43.347 / best 42.340 / worst 46.349 ms (9/10 kept) | avg 0.195 / best 0.178 / worst 0.211 ms (16/20 kept) | avg 0.191 / best 0.178 / worst 0.209 ms (19/20 kept) | avg 0.194 / best 0.184 / worst 0.208 ms (18/20 kept) |
| decode_512 | avg 81.890 / best 80.460 / worst 83.826 ms (10/10 kept) | avg 0.197 / best 0.187 / worst 0.208 ms (18/20 kept) | avg 0.198 / best 0.188 / worst 0.218 ms (20/20 kept) | avg 0.216 / best 0.200 / worst 0.233 ms (15/20 kept) |
| decode_1k | avg 168.478 / best 164.533 / worst 174.503 ms (10/10 kept) | avg 0.232 / best 0.219 / worst 0.253 ms (18/20 kept) | avg 0.233 / best 0.217 / worst 0.250 ms (19/20 kept) | avg 0.257 / best 0.250 / worst 0.274 ms (14/20 kept) |
| decode_2k | avg 326.317 / best 324.494 / worst 328.742 ms (10/10 kept) | avg 0.268 / best 0.255 / worst 0.286 ms (20/20 kept) | avg 0.266 / best 0.251 / worst 0.293 ms (19/20 kept) | avg 0.285 / best 0.278 / worst 0.299 ms (17/20 kept) |
| decode_4k | avg 652.267 / best 644.637 / worst 661.987 ms (10/10 kept) | avg 0.323 / best 0.313 / worst 0.337 ms (18/20 kept) | avg 0.313 / best 0.302 / worst 0.342 ms (19/20 kept) | avg 0.308 / best 0.303 / worst 0.322 ms (17/20 kept) |
| decode_8k | avg 1301.623 / best 1287.660 / worst 1321.381 ms (10/10 kept) | avg 0.478 / best 0.468 / worst 0.493 ms (18/20 kept) | avg 0.465 / best 0.453 / worst 0.492 ms (19/20 kept) | avg 0.440 / best 0.410 / worst 0.548 ms (12/20 kept) |
| decode_16k | avg 2617.145 / best 2596.292 / worst 2693.561 ms (10/10 kept) | avg 5.503 / best 0.675 / worst 20.552 ms (19/20 kept) | avg 0.736 / best 0.727 / worst 0.748 ms (16/20 kept) | avg 11.139 / best 0.554 / worst 27.193 ms (20/20 kept) |
| decode_32k | avg 5156.778 / best 5053.716 / worst 5244.840 ms (10/10 kept) | avg 7.140 / best 1.232 / worst 19.368 ms (20/20 kept) | avg 8.467 / best 1.236 / worst 17.692 ms (20/20 kept) | avg 4.271 / best 0.788 / worst 11.106 ms (20/20 kept) |
| decode_64k | avg 10177.482 / best 9997.757 / worst 10378.087 ms (10/10 kept) | avg 9.500 / best 2.326 / worst 27.359 ms (20/20 kept) | avg 8.915 / best 2.320 / worst 17.027 ms (20/20 kept) | avg 7.704 / best 1.364 / worst 14.772 ms (20/20 kept) |
| decode_128k | avg 20592.962 / best 20256.906 / worst 21162.229 ms (10/10 kept) | avg 12.977 / best 4.576 / worst 18.752 ms (20/20 kept) | avg 9.811 / best 4.429 / worst 25.959 ms (20/20 kept) | avg 8.866 / best 2.549 / worst 18.800 ms (20/20 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| decode_256 | 138.4 tok/s | 30801.0 tok/s | 31429.0 tok/s | 30913.4 tok/s |
| decode_512 | 73.3 tok/s | 30439.1 tok/s | 30237.9 tok/s | 27789.8 tok/s |
| decode_1k | 35.6 tok/s | 25897.8 tok/s | 25761.8 tok/s | 23350.3 tok/s |
| decode_2k | 18.4 tok/s | 22391.7 tok/s | 22542.3 tok/s | 21074.5 tok/s |
| decode_4k | 9.2 tok/s | 18604.1 tok/s | 19153.1 tok/s | 19454.8 tok/s |
| decode_8k | 4.6 tok/s | 12553.6 tok/s | 12899.2 tok/s | 13645.8 tok/s |
| decode_16k | 2.3 tok/s | 1090.4 tok/s | 8150.3 tok/s | 538.7 tok/s |
| decode_32k | 1.2 tok/s | 840.3 tok/s | 708.6 tok/s | 1404.9 tok/s |
| decode_64k | 0.6 tok/s | 631.6 tok/s | 673.0 tok/s | 778.8 tok/s |
| decode_128k | 0.3 tok/s | 462.4 tok/s | 611.6 tok/s | 676.7 tok/s |

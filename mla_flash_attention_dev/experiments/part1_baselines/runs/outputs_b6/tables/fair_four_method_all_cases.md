# 公平四方法全量结果

说明：这张表来自正式 decode benchmark，不是 capability probe。
说明：单元格格式为 `avg / best / worst ms; throughput`，统计前已经按主结果规则过滤慢尾异常点。
说明：当前 DeepSeek 并行度策略为 `fixed`。

| seq_len | B | H | value_dim | dqhpc | q_shards | config | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| `256` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 43.347 / best 42.340 / worst 46.349 ms; 138.4 tok/s | avg 0.195 / best 0.178 / worst 0.211 ms; 30801.0 tok/s | avg 0.191 / best 0.178 / worst 0.209 ms; 31429.0 tok/s | avg 0.194 / best 0.184 / worst 0.208 ms; 30913.4 tok/s |
| `512` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 81.890 / best 80.460 / worst 83.826 ms; 73.3 tok/s | avg 0.197 / best 0.187 / worst 0.208 ms; 30439.1 tok/s | avg 0.198 / best 0.188 / worst 0.218 ms; 30237.9 tok/s | avg 0.216 / best 0.200 / worst 0.233 ms; 27789.8 tok/s |
| `1k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 168.478 / best 164.533 / worst 174.503 ms; 35.6 tok/s | avg 0.232 / best 0.219 / worst 0.253 ms; 25897.8 tok/s | avg 0.233 / best 0.217 / worst 0.250 ms; 25761.8 tok/s | avg 0.257 / best 0.250 / worst 0.274 ms; 23350.3 tok/s |
| `2k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 326.317 / best 324.494 / worst 328.742 ms; 18.4 tok/s | avg 0.268 / best 0.255 / worst 0.286 ms; 22391.7 tok/s | avg 0.266 / best 0.251 / worst 0.293 ms; 22542.3 tok/s | avg 0.285 / best 0.278 / worst 0.299 ms; 21074.5 tok/s |
| `4k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 652.267 / best 644.637 / worst 661.987 ms; 9.2 tok/s | avg 0.323 / best 0.313 / worst 0.337 ms; 18604.1 tok/s | avg 0.313 / best 0.302 / worst 0.342 ms; 19153.1 tok/s | avg 0.308 / best 0.303 / worst 0.322 ms; 19454.8 tok/s |
| `8k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 1301.623 / best 1287.660 / worst 1321.381 ms; 4.6 tok/s | avg 0.478 / best 0.468 / worst 0.493 ms; 12553.6 tok/s | avg 0.465 / best 0.453 / worst 0.492 ms; 12899.2 tok/s | avg 0.440 / best 0.410 / worst 0.548 ms; 13645.8 tok/s |
| `16k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 2617.145 / best 2596.292 / worst 2693.561 ms; 2.3 tok/s | avg 5.503 / best 0.675 / worst 20.552 ms; 1090.4 tok/s | avg 0.736 / best 0.727 / worst 0.748 ms; 8150.3 tok/s | avg 11.139 / best 0.554 / worst 27.193 ms; 538.7 tok/s |
| `32k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 5156.778 / best 5053.716 / worst 5244.840 ms; 1.2 tok/s | avg 7.140 / best 1.232 / worst 19.368 ms; 840.3 tok/s | avg 8.467 / best 1.236 / worst 17.692 ms; 708.6 tok/s | avg 4.271 / best 0.788 / worst 11.106 ms; 1404.9 tok/s |
| `64k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 10177.482 / best 9997.757 / worst 10378.087 ms; 0.6 tok/s | avg 9.500 / best 2.326 / worst 27.359 ms; 631.6 tok/s | avg 8.915 / best 2.320 / worst 17.027 ms; 673.0 tok/s | avg 7.704 / best 1.364 / worst 14.772 ms; 778.8 tok/s |
| `128k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 20592.962 / best 20256.906 / worst 21162.229 ms; 0.3 tok/s | avg 12.977 / best 4.576 / worst 18.752 ms; 462.4 tok/s | avg 9.811 / best 4.429 / worst 25.959 ms; 611.6 tok/s | avg 8.866 / best 2.549 / worst 18.800 ms; 676.7 tok/s |

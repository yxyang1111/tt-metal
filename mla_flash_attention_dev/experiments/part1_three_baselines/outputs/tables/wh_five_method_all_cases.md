# 五方法全量结果（Wormhole 4c/8c 专项）

说明：这张表按 `fair_four_method_all_cases.md` 的样式整理，但数据来自 Wormhole 4c/8c 专项 rerun，不是原始公平四方法主表。
说明：单元格格式为 `avg / best / worst ms; throughput`；这里直接记录各专项 rerun 的结果，不额外做慢尾异常点过滤。
说明：同一配置若在多个专项 sweep 中重复出现，只保留一行；优先保留该配置所属的主扫描维度。`B=6,H=32,seq_len=4k` 的 4c/8c 对比对 run-to-run 较敏感，补充复测见 `wh_4c_vs_8c_qshard4_seq_summary.md`。
说明：如果想看五种方法各自的实现入口、数据形态、reader/compute/writer 数据流、4c/8c 几何差异与中间结果流向，见 `mla_flash_attention_dev/experiments/part1_three_baselines/five-method-dataflow-analysis.md`；快速版先看该文档的 `第 0 节`。

| seq_len | B | H | value_dim | dqhpc | q_shards | config | Reference Attention | Flash Attention | TT-MLA | FlashMLA-4c | FlashMLA-8c |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| `1k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 184.851 / best 184.851 / worst 184.851 ms; 32.5 tok/s | avg 0.262 / best 0.221 / worst 0.324 ms; 22914.7 tok/s | avg 0.250 / best 0.214 / worst 0.304 ms; 23969.7 tok/s | avg 0.304 / best 0.256 / worst 0.349 ms; 19748.7 tok/s | avg 0.299 / best 0.261 / worst 0.320 ms; 20044.2 tok/s |
| `1k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 207.415 / best 207.415 / worst 207.415 ms; 38.6 tok/s | avg 0.310 / best 0.250 / worst 0.407 ms; 25836.9 tok/s | avg 0.268 / best 0.233 / worst 0.323 ms; 29850.5 tok/s | avg 0.327 / best 0.262 / worst 0.451 ms; 24464.5 tok/s | avg 0.336 / best 0.256 / worst 0.473 ms; 23840.5 tok/s |
| `4k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 613.309 / best 613.309 / worst 613.309 ms; 9.8 tok/s | avg 0.397 / best 0.352 / worst 0.474 ms; 15131.0 tok/s | avg 0.354 / best 0.326 / worst 0.401 ms; 16944.1 tok/s | avg 0.285 / best 0.263 / worst 0.329 ms; 21018.4 tok/s | avg 0.352 / best 0.324 / worst 0.404 ms; 17046.8 tok/s |
| `4k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 670.436 / best 670.436 / worst 670.436 ms; 11.9 tok/s | avg 0.421 / best 0.391 / worst 0.477 ms; 19000.9 tok/s | avg 0.377 / best 0.346 / worst 0.432 ms; 21196.2 tok/s | avg 0.396 / best 0.332 / worst 0.515 ms; 20225.2 tok/s | avg 0.358 / best 0.326 / worst 0.403 ms; 22360.3 tok/s |
| `8k` | `1` | `32` | `512` | `8` | `4` | `b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 216.350 / best 216.350 / worst 216.350 ms; 4.6 tok/s | avg 0.479 / best 0.428 / worst 0.563 ms; 2085.7 tok/s | avg 0.547 / best 0.502 / worst 0.594 ms; 1827.4 tok/s | avg 0.464 / best 0.427 / worst 0.531 ms; 2153.9 tok/s | avg 0.477 / best 0.445 / worst 0.527 ms; 2097.6 tok/s |
| `8k` | `2` | `32` | `512` | `8` | `4` | `b2_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 412.497 / best 412.497 / worst 412.497 ms; 4.8 tok/s | avg 0.463 / best 0.425 / worst 0.519 ms; 4320.8 tok/s | avg 0.504 / best 0.473 / worst 0.521 ms; 3970.4 tok/s | avg 0.418 / best 0.330 / worst 0.579 ms; 4780.1 tok/s | avg 0.365 / best 0.337 / worst 0.419 ms; 5483.8 tok/s |
| `8k` | `4` | `32` | `512` | `8` | `4` | `b4_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 778.834 / best 778.834 / worst 778.834 ms; 5.1 tok/s | avg 0.482 / best 0.443 / worst 0.545 ms; 8296.3 tok/s | avg 0.482 / best 0.458 / worst 0.530 ms; 8302.3 tok/s | avg 0.416 / best 0.359 / worst 0.512 ms; 9613.5 tok/s | avg 0.437 / best 0.351 / worst 0.491 ms; 9162.3 tok/s |
| `8k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 1167.308 / best 1167.308 / worst 1167.308 ms; 5.1 tok/s | avg 0.492 / best 0.465 / worst 0.541 ms; 12192.4 tok/s | avg 0.494 / best 0.456 / worst 0.543 ms; 12155.2 tok/s | avg 0.447 / best 0.421 / worst 0.467 ms; 13413.3 tok/s | avg 0.440 / best 0.410 / worst 0.495 ms; 13647.0 tok/s |
| `8k` | `8` | `8` | `512` | `8` | `1` | `b8_h8_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 452.851 / best 452.851 / worst 452.851 ms; 17.7 tok/s | avg 0.621 / best 0.555 / worst 0.654 ms; 12883.2 tok/s | avg 0.486 / best 0.424 / worst 0.594 ms; 16471.2 tok/s | avg 0.503 / best 0.431 / worst 0.542 ms; 15895.8 tok/s | avg 0.510 / best 0.442 / worst 0.629 ms; 15672.0 tok/s |
| `8k` | `8` | `16` | `512` | `8` | `2` | `b8_h16_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 863.885 / best 863.885 / worst 863.885 ms; 9.3 tok/s | avg 0.631 / best 0.563 / worst 0.732 ms; 12685.5 tok/s | avg 0.488 / best 0.446 / worst 0.519 ms; 16392.0 tok/s | avg 0.476 / best 0.421 / worst 0.520 ms; 16797.1 tok/s | avg 0.470 / best 0.454 / worst 0.503 ms; 17008.0 tok/s |
| `8k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 1233.803 / best 1233.803 / worst 1233.803 ms; 6.5 tok/s | avg 0.583 / best 0.552 / worst 0.632 ms; 13723.7 tok/s | avg 0.509 / best 0.486 / worst 0.552 ms; 15731.4 tok/s | avg 0.474 / best 0.440 / worst 0.519 ms; 16892.1 tok/s | avg 0.466 / best 0.442 / worst 0.501 ms; 17164.4 tok/s |
| `8k` | `8` | `32` | `512` | `8` | `4` | `b8_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 1551.229 / best 1551.229 / worst 1551.229 ms; 5.2 tok/s | avg 0.618 / best 0.580 / worst 0.672 ms; 12942.9 tok/s | avg 0.510 / best 0.484 / worst 0.562 ms; 15673.8 tok/s | unsupported (batch * deepseek_num_q_shards must be <= 24, got 32) | avg 0.500 / best 0.465 / worst 0.532 ms; 15989.6 tok/s |
| `8k` | `12` | `32` | `512` | `8` | `4` | `b12_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 2342.810 / best 2342.810 / worst 2342.810 ms; 5.1 tok/s | avg 0.761 / best 0.723 / worst 0.833 ms; 15763.2 tok/s | avg 0.585 / best 0.553 / worst 0.639 ms; 20499.7 tok/s | unsupported (batch * deepseek_num_q_shards must be <= 24, got 48) | avg 0.599 / best 0.553 / worst 0.668 ms; 20048.8 tok/s |
| `16k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 2306.870 / best 2306.870 / worst 2306.870 ms; 2.6 tok/s | avg 0.761 / best 0.734 / worst 0.807 ms; 7884.2 tok/s | avg 0.775 / best 0.746 / worst 0.829 ms; 7737.2 tok/s | avg 0.637 / best 0.580 / worst 0.683 ms; 9414.9 tok/s | avg 0.631 / best 0.589 / worst 0.675 ms; 9508.4 tok/s |
| `16k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 2351.581 / best 2351.581 / worst 2351.581 ms; 3.4 tok/s | avg 0.941 / best 0.908 / worst 1.002 ms; 8498.1 tok/s | avg 0.782 / best 0.738 / worst 0.834 ms; 10233.9 tok/s | avg 0.668 / best 0.634 / worst 0.727 ms; 11977.6 tok/s | avg 0.664 / best 0.628 / worst 0.727 ms; 12042.2 tok/s |
| `32k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 4501.536 / best 4501.536 / worst 4501.536 ms; 1.3 tok/s | avg 1.358 / best 1.293 / worst 1.464 ms; 4418.3 tok/s | avg 1.319 / best 1.289 / worst 1.351 ms; 4550.1 tok/s | avg 0.967 / best 0.913 / worst 1.072 ms; 6203.2 tok/s | avg 0.883 / best 0.864 / worst 0.920 ms; 6795.1 tok/s |
| `32k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | avg 4561.688 / best 4561.688 / worst 4561.688 ms; 1.8 tok/s | avg 1.637 / best 1.600 / worst 1.698 ms; 4888.2 tok/s | avg 1.372 / best 1.321 / worst 1.434 ms; 5829.4 tok/s | avg 1.053 / best 1.005 / worst 1.128 ms; 7596.6 tok/s | avg 1.053 / best 1.042 / worst 1.062 ms; 7595.3 tok/s |

## 平均延迟（ms）

说明：这里只保留上表每个单元格中的 `avg` 延迟；最后一列给出按当前表中显示值比较得到的最低延迟方法。

| seq_len | B | H | value_dim | dqhpc | q_shards | config | Reference Attention | Flash Attention | TT-MLA | FlashMLA-4c | FlashMLA-8c | 最低延迟方法 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `1k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 184.851 | 0.262 | 0.250 | 0.304 | 0.299 | `TT-MLA (0.250 ms)` |
| `1k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 207.415 | 0.310 | 0.268 | 0.327 | 0.336 | `TT-MLA (0.268 ms)` |
| `4k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 613.309 | 0.397 | 0.354 | 0.285 | 0.352 | `FlashMLA-4c (0.285 ms)` |
| `4k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 670.436 | 0.421 | 0.377 | 0.396 | 0.358 | `FlashMLA-8c (0.358 ms)` |
| `8k` | `1` | `32` | `512` | `8` | `4` | `b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 216.350 | 0.479 | 0.547 | 0.464 | 0.477 | `FlashMLA-4c (0.464 ms)` |
| `8k` | `2` | `32` | `512` | `8` | `4` | `b2_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 412.497 | 0.463 | 0.504 | 0.418 | 0.365 | `FlashMLA-8c (0.365 ms)` |
| `8k` | `4` | `32` | `512` | `8` | `4` | `b4_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 778.834 | 0.482 | 0.482 | 0.416 | 0.437 | `FlashMLA-4c (0.416 ms)` |
| `8k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 1167.308 | 0.492 | 0.494 | 0.447 | 0.440 | `FlashMLA-8c (0.440 ms)` |
| `8k` | `8` | `8` | `512` | `8` | `1` | `b8_h8_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 452.851 | 0.621 | 0.486 | 0.503 | 0.510 | `TT-MLA (0.486 ms)` |
| `8k` | `8` | `16` | `512` | `8` | `2` | `b8_h16_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 863.885 | 0.631 | 0.488 | 0.476 | 0.470 | `FlashMLA-8c (0.470 ms)` |
| `8k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 1233.803 | 0.583 | 0.509 | 0.474 | 0.466 | `FlashMLA-8c (0.466 ms)` |
| `8k` | `8` | `32` | `512` | `8` | `4` | `b8_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 1551.229 | 0.618 | 0.510 | unsupported | 0.500 | `FlashMLA-8c (0.500 ms)` |
| `8k` | `12` | `32` | `512` | `8` | `4` | `b12_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 2342.810 | 0.761 | 0.585 | unsupported | 0.599 | `TT-MLA (0.585 ms)` |
| `16k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 2306.870 | 0.761 | 0.775 | 0.637 | 0.631 | `FlashMLA-8c (0.631 ms)` |
| `16k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 2351.581 | 0.941 | 0.782 | 0.668 | 0.664 | `FlashMLA-8c (0.664 ms)` |
| `32k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 4501.536 | 1.358 | 1.319 | 0.967 | 0.883 | `FlashMLA-8c (0.883 ms)` |
| `32k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 4561.688 | 1.637 | 1.372 | 1.053 | 1.053 | `FlashMLA-4c = FlashMLA-8c (1.053 ms)` |

## 平均吞吐（tok/s）

说明：这里只保留上表每个单元格中的吞吐；最后一列给出按当前表中显示值比较得到的最高吞吐方法。

| seq_len | B | H | value_dim | dqhpc | q_shards | config | Reference Attention | Flash Attention | TT-MLA | FlashMLA-4c | FlashMLA-8c | 最高吞吐方法 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `1k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 32.5 | 22914.7 | 23969.7 | 19748.7 | 20044.2 | `TT-MLA (23969.7 tok/s)` |
| `1k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 38.6 | 25836.9 | 29850.5 | 24464.5 | 23840.5 | `TT-MLA (29850.5 tok/s)` |
| `4k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 9.8 | 15131.0 | 16944.1 | 21018.4 | 17046.8 | `FlashMLA-4c (21018.4 tok/s)` |
| `4k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 11.9 | 19000.9 | 21196.2 | 20225.2 | 22360.3 | `FlashMLA-8c (22360.3 tok/s)` |
| `8k` | `1` | `32` | `512` | `8` | `4` | `b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 4.6 | 2085.7 | 1827.4 | 2153.9 | 2097.6 | `FlashMLA-4c (2153.9 tok/s)` |
| `8k` | `2` | `32` | `512` | `8` | `4` | `b2_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 4.8 | 4320.8 | 3970.4 | 4780.1 | 5483.8 | `FlashMLA-8c (5483.8 tok/s)` |
| `8k` | `4` | `32` | `512` | `8` | `4` | `b4_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 5.1 | 8296.3 | 8302.3 | 9613.5 | 9162.3 | `FlashMLA-4c (9613.5 tok/s)` |
| `8k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 5.1 | 12192.4 | 12155.2 | 13413.3 | 13647.0 | `FlashMLA-8c (13647.0 tok/s)` |
| `8k` | `8` | `8` | `512` | `8` | `1` | `b8_h8_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 17.7 | 12883.2 | 16471.2 | 15895.8 | 15672.0 | `TT-MLA (16471.2 tok/s)` |
| `8k` | `8` | `16` | `512` | `8` | `2` | `b8_h16_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 9.3 | 12685.5 | 16392.0 | 16797.1 | 17008.0 | `FlashMLA-8c (17008.0 tok/s)` |
| `8k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 6.5 | 13723.7 | 15731.4 | 16892.1 | 17164.4 | `FlashMLA-8c (17164.4 tok/s)` |
| `8k` | `8` | `32` | `512` | `8` | `4` | `b8_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 5.2 | 12942.9 | 15673.8 | unsupported | 15989.6 | `FlashMLA-8c (15989.6 tok/s)` |
| `8k` | `12` | `32` | `512` | `8` | `4` | `b12_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 5.1 | 15763.2 | 20499.7 | unsupported | 20048.8 | `TT-MLA (20499.7 tok/s)` |
| `16k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 2.6 | 7884.2 | 7737.2 | 9414.9 | 9508.4 | `FlashMLA-8c (9508.4 tok/s)` |
| `16k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 3.4 | 8498.1 | 10233.9 | 11977.6 | 12042.2 | `FlashMLA-8c (12042.2 tok/s)` |
| `32k` | `6` | `32` | `512` | `8` | `4` | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 1.3 | 4418.3 | 4550.1 | 6203.2 | 6795.1 | `FlashMLA-8c (6795.1 tok/s)` |
| `32k` | `8` | `24` | `512` | `8` | `3` | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | 1.8 | 4888.2 | 5829.4 | 7596.6 | 7595.3 | `FlashMLA-4c (7596.6 tok/s)` |

## 最优方法分析

说明：这张表不是完整全因子 sweep，而是 `batch`、`head`、`seq_len(q_shards=3)`、`seq_len(q_shards=4)` 四组专项结果的汇总。因此“谁最优”的结论更适合按局部参数压力来理解，而不是机械地跨所有行做全局排序。

### 1. `TT-MLA` 为什么会在低压力区间最优

- 代表点：`1k, B=6, H=32`，`1k, B=8, H=24`，以及 `8k, B=8, H=8`。
- 这类点的共同特点是：`seq_len` 短，或者 `q_shards` 低，固定开销在总时间里占比更大。
- `TT-MLA` 的优势在于主线实现更成熟、全设备网格利用更直接，因此在真正的带宽/容量瓶颈出现之前，它更容易靠较低的固定成本拿到最低延迟和最高吞吐。
- `8k, B=12, H=32` 这类极高吞吐压力点上，`FlashMLA-8c` 已经非常接近 `TT-MLA`，但 `TT-MLA` 仍略胜，说明主线实现的整体 pipeline 和全网格调度在极限区间仍有余量。

### 2. `FlashMLA-4c` 为什么会在中等压力区间最优

- 代表点：`8k, B=1, H=32`，`8k, B=4, H=32`，以及表中 `4k, B=6, H=32` 的主 sweep 结果。
- 这类点的共同特点是：问题规模已经不算小，但还没有大到必须依赖更宽的 S-block 才能维持良好映射。
- `FlashMLA-4c` 的优势在于布局更紧凑，协调域更小，额外通信和调度成本更低；当容量还够用时，这种“轻量”几何反而可能比 8c 更占优。
- 换句话说，4c 的强项不是绝对并行度更高，而是在“不需要更多核也能装下”的区间里，能用更低的组织成本完成同样工作。

### 3. `FlashMLA-8c` 为什么会在高压力区间最优

- 代表点：`8k, B=2/6, H=32`，`8k, B=8, H=16/24/32`，以及 `16k`、`32k` 两组长序列点。
- 这类点的共同特点是：`batch` 更大、`H/q_shards` 更大、或者 `seq_len` 更长，系统开始更明显地受到容量边界、并行映射和 NOC/带宽压力影响。
- `FlashMLA-8c` 的第一层优势是扩容。对于 `q_shards=4` 路线，4c 的上限是 `24 Q cores`，而 8c 把上限抬到 `48 Q cores`，因此 `B=8/12`、`H=32,B=8` 这类点从“不支持”变成“可跑且有竞争力”。
- 第二层优势是高压力下的映射更舒展。即使在 4c 本来就能跑的 `q_shards=3/4` 路线上，8c 也常常能把负载分摊得更均匀，减少高压力下的回压和拥塞，所以在 `16k/32k` 这类长序列端优势更明显。
- 从现有结果看，8c 更像“高压力赢家”：当问题规模继续增大时，它的优势通常比 4c 更稳定，也更容易同时拿到低延迟和高吞吐。

### 4. 为什么这批点里 `Flash Attention` 没有拿到最优

- 这不意味着 `Flash Attention` 普遍不如 MLA，而是因为这张表刻意聚焦在 `MLA decode`、`H_kv=1`、`value_dim=512`、`q_shards=3/4` 这些更偏 MLA 专用实现优势的区域。
- 在这些点上，`TT-MLA` 和两种 `FlashMLA` 的实现更贴近目标数据流和并行组织方式，因此 `Flash Attention` 通常能保持不错表现，但很难成为这一组专项里的最优者。

### 5. 两个需要谨慎解读的点

- `4k, B=6, H=32, q_shards=4` 这一行在主表里是 `FlashMLA-4c` 最优，但这个点对 run-to-run 状态比较敏感；补充复测更接近 `8c >= 4c`，因此更稳妥的理解是“4c 在这点上可能略优，也可能只是和 8c 接近”。
- `32k, B=8, H=24, q_shards=3` 这一行里 4c 和 8c 的延迟基本打平，吞吐也只差很小一截，更像统计波动内的近似持平，而不是存在明确机制性差异。

### 6. 一句话总结

- 低压力区间，`TT-MLA` 更容易靠成熟实现和较低固定成本取胜。
- 中等压力且容量仍够用时，`FlashMLA-4c` 可能因为几何更紧凑而占优。
- 高压力区间，尤其是更大 `batch`、更大 `q_shards`、更长 `seq_len` 时，`FlashMLA-8c` 最能发挥扩容和负载均衡优势。

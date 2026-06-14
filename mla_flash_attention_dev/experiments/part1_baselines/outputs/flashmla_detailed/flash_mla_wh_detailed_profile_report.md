# FlashMLA WH detailed profile summary

本报告覆盖 paged/chunked MLA prefill 与 paged MLA decode 两条路径。

- prefill 实测路径：`ttnn.transformer.chunked_flash_mla_prefill`（`chunk_start_idx=0`）
- decode 实测路径：`ttnn.transformer.paged_flash_multi_latent_attention_decode`
- 自定义 reader marker：SDPA-PAGE-TABLE-SUM, SDPA-PAGED-RESERVE-SUM, SDPA-PAGED-ISSUE-SUM, SDPA-PAGED-WAIT-SUM, SDPA-PAGED-PUSH-SUM
- 自定义 writer marker：SDPA-WRITER-CB-WAIT-SUM, SDPA-WRITER-ISSUE-SUM, SDPA-WRITER-BARRIER-SUM, SDPA-WRITER-POP-SUM
- 可选 reader source marker：SDPA-K-RESERVE-SUM, SDPA-K-ISSUE-SUM, SDPA-K-WAIT-SUM, SDPA-K-PUSH-SUM, SDPA-V-RESERVE-SUM, SDPA-V-ISSUE-SUM, SDPA-V-WAIT-SUM, SDPA-V-PUSH-SUM
- 可选 writer source marker：SDPA-WRITER-SENDER-CB-WAIT-SUM, SDPA-WRITER-ROOT-CB-WAIT-SUM, SDPA-WRITER-TREE-CHILD-WAIT-SUM, SDPA-WRITER-OUTPUT-GATHER-WAIT-SUM
- cycle -> us 换算基于 WH AICLK `1000` MHz

## Prefill measured profile

| case | batch | seq_len | kernel us | us/item | ncrisc us | brisc us | compute us(max trisc) | bottleneck | est k-read GB/s | est out-write GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|

## Decode measured profile

| case | batch | seq_len | kernel us | us/item | ncrisc us | brisc us | compute us(max trisc) | bottleneck | est k-read GB/s | est out-write GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| decode_1k | 2 | 1024 | 73.531 | 36.765417 | 27.253 | 61.376 | 73.267 | compute_on_critical_path | 43.284 | 1.068 |
| decode_4k | 2 | 4096 | 173.101 | 86.550750 | 109.347 | 161.070 | 172.839 | writer_close_to_critical_path | 43.152 | 0.407 |
| decode_8k | 2 | 8192 | 303.168 | 151.584250 | 238.594 | 291.010 | 302.909 | writer_close_to_critical_path | 39.553 | 0.225 |
| decode_16k | 1 | 16384 | 560.899 | 560.899500 | 497.738 | 548.789 | 560.631 | reader_close_to_critical_path | 18.960 | 0.060 |
| decode_32k | 1 | 32768 | 1084.760 | 1084.760250 | 1023.405 | 1072.649 | 1084.496 | reader_writer_saturated | 18.443 | 0.031 |

## Prefill custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Decode custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 27.253 | 0.420 | 0.071 | 7.871 | 1.680 | 0.145 | 10.735 | 39.4% | 0.7% | 77.3% | 16.5% | 1.4% |
| decode_4k | 109.347 | 0.489 | 17.545 | 33.689 | 6.394 | 0.595 | 59.406 | 54.3% | 29.9% | 57.4% | 10.9% | 1.0% |
| decode_8k | 238.594 | 0.432 | 84.893 | 62.856 | 14.012 | 1.177 | 163.871 | 68.7% | 52.0% | 38.5% | 8.6% | 0.7% |
| decode_16k | 497.738 | 0.484 | 214.921 | 125.626 | 16.528 | 2.362 | 360.923 | 72.5% | 59.7% | 34.9% | 4.6% | 0.7% |
| decode_32k | 1023.405 | 0.525 | 463.280 | 251.306 | 32.916 | 4.727 | 754.885 | 73.8% | 61.5% | 33.4% | 4.4% | 0.6% |

## Prefill custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Decode custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 61.376 | 23.299 | 0.744 | 0.652 | 0.052 | 44.962 | 73.3% | 94.1% | 3.0% | 2.6% | 0.2% |
| decode_4k | 161.070 | 73.038 | 0.745 | 0.668 | 0.053 | 144.362 | 89.6% | 98.0% | 1.0% | 0.9% | 0.1% |
| decode_8k | 291.010 | 137.924 | 0.800 | 0.685 | 0.052 | 274.007 | 94.2% | 98.9% | 0.6% | 0.5% | 0.0% |
| decode_16k | 548.789 | 267.461 | 0.584 | 0.548 | 0.052 | 532.093 | 97.0% | 99.6% | 0.2% | 0.2% | 0.0% |
| decode_32k | 1072.649 | 529.359 | 0.600 | 0.557 | 0.052 | 1055.945 | 98.4% | 99.8% | 0.1% | 0.1% | 0.0% |

## Decode custom reader source breakdown

| case | thread us | k_reserve us/core(avg) | k_issue us/core(avg) | k_wait us/core(avg) | k_push us/core(avg) | v_reserve us/core(avg) | v_issue us/core(avg) | v_wait us/core(avg) | v_push us/core(avg) | profiled total us/core(max) | coverage vs thread | k_reserve share | k_issue share | k_wait share | k_push share | v_reserve share | v_issue share | v_wait share | v_push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_4k | 109.347 | 17.212 | 23.219 | 6.049 | 0.326 | 0.333 | 10.470 | 0.345 | 0.269 | 58.402 | 53.4% | 29.6% | 39.9% | 10.4% | 0.6% | 0.6% | 18.0% | 0.6% | 0.5% |

## Decode custom writer source breakdown

| case | thread us | sender_cb_wait us/core(avg) | root_cb_wait us/core(avg) | tree_child_wait us/core(avg) | output_gather_wait us/core(avg) | profiled total us/core(max) | coverage vs thread | sender_cb_wait share | root_cb_wait share | tree_child_wait share | output_gather_wait share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_4k | 161.070 | 73.038 | 0.000 | 74.389 | 0.000 | 154.947 | 96.2% | 49.5% | 0.0% | 50.5% | 0.0% |

## Decode A-BH empirical coupled model

这一节把 WH decode detailed profile 的 `reader reserve` / `writer cb_wait` 回灌到当前 A 路径的 BH 模型里，
不再只看理想化的 `compute vs dram`，而是把 pipeline backpressure 也显式带上。
下面的 source attribution 是解释性拆分，不是新的硬件计数器。

| case | seq_len | ideal 1st-order ms | empirical 2nd-order ms | compute ms | reader act/reserve ms | writer act/wait ms | dominant | uplift vs ideal |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| decode_1k | 1024 | 0.0075 | 0.0135 | 0.0064 | 0.0135/0.0000 | 0.0069/0.0043 | reader | 1.79x |
| decode_4k | 4096 | 0.0269 | 0.0490 | 0.0258 | 0.0462/0.0028 | 0.0070/0.0207 | reader | 1.82x |
| decode_8k | 8192 | 0.0527 | 0.0922 | 0.0516 | 0.0777/0.0145 | 0.0071/0.0420 | reader | 1.75x |
| decode_16k | 16384 | 0.1043 | 0.1826 | 0.1032 | 0.1431/0.0395 | 0.0069/0.0871 | reader | 1.75x |
| decode_32k | 32768 | 0.2074 | 0.3715 | 0.2063 | 0.2833/0.0881 | 0.0069/0.1813 | reader | 1.79x |

## Decode A-BH empirical scaling summary

| case | seq_len | 1st-order dram crossover | 2nd-order non-compute crossover | 2nd-order plateau start (<=2% of 64c) | 16c->64c speedup | 64c floor ms |
|---|---:|---:|---:|---:|---:|---:|
| decode_1k | 1024 | 23 | 8 | 8 | 1.00x | 0.0135 |
| decode_4k | 4096 | 23 | 9 | 12 | 1.00x | 0.0490 |
| decode_8k | 8192 | 23 | 8 | 15 | 1.01x | 0.0914 |
| decode_16k | 16384 | 23 | 8 | 17 | 1.02x | 0.1784 |
| decode_32k | 32768 | 23 | 7 | 17 | 1.03x | 0.3617 |

## Decode A-BH empirical coupling attribution

| case | seq_len | reader active | reserve<-compute | reserve<-writer | writer active | wait<-compute | wait<-reader | dominant coupling |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| decode_1k | 1024 | 0.0135 | 0.0000 | 0.0000 | 0.0069 | 0.0014 | 0.0029 | wait<-reader |
| decode_4k | 4096 | 0.0462 | 0.0014 | 0.0015 | 0.0070 | 0.0071 | 0.0136 | wait<-reader |
| decode_8k | 8192 | 0.0777 | 0.0074 | 0.0070 | 0.0071 | 0.0151 | 0.0269 | wait<-reader |
| decode_16k | 16384 | 0.1431 | 0.0207 | 0.0189 | 0.0069 | 0.0315 | 0.0557 | wait<-reader |
| decode_32k | 32768 | 0.2833 | 0.0461 | 0.0420 | 0.0069 | 0.0647 | 0.1166 | wait<-reader |

## Decode A-BH empirical coupled active-core sweep

| case | seq_len | 16c | 20c | 24c | 32c | 48c | 64c |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 1024 | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) |
| decode_4k | 4096 | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) |
| decode_32k | 32768 | 0.3715 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) |

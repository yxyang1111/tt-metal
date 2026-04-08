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
| decode_4k | 2 | 4096 | 173.101 | 86.550750 | 109.347 | 161.070 | 172.839 | writer_close_to_critical_path | 43.152 | 0.407 |

## Prefill custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Decode custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_4k | 109.347 | 0.489 | 17.545 | 33.689 | 6.394 | 0.595 | 59.406 | 54.3% | 29.9% | 57.4% | 10.9% | 1.0% |

## Prefill custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Decode custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_4k | 161.070 | 73.038 | 0.745 | 0.668 | 0.053 | 144.362 | 89.6% | 98.0% | 1.0% | 0.9% | 0.1% |

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
| decode_4k | 4096 | 0.0269 | 0.0490 | 0.0258 | 0.0462/0.0028 | 0.0070/0.0207 | reader | 1.82x |

## Decode A-BH empirical scaling summary

| case | seq_len | 1st-order dram crossover | 2nd-order non-compute crossover | 2nd-order plateau start (<=2% of 64c) | 16c->64c speedup | 64c floor ms |
|---|---:|---:|---:|---:|---:|---:|
| decode_4k | 4096 | 23 | 9 | 12 | 1.00x | 0.0490 |

## Decode A-BH empirical coupling attribution

| case | seq_len | reader active | reserve<-compute | reserve<-writer | writer active | wait<-compute | wait<-reader | dominant coupling |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| decode_4k | 4096 | 0.0462 | 0.0014 | 0.0015 | 0.0070 | 0.0071 | 0.0136 | wait<-reader |

## Decode A-BH empirical coupled active-core sweep

| case | seq_len | 16c | 20c | 24c | 32c | 48c | 64c |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_4k | 4096 | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) | 0.0490 (reader) |

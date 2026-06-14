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
| prefill_1k | 1 | 1024 | 4037.870 | 3.943233 | 4025.983 | 4037.291 | 4034.985 | reader_writer_saturated | 0.147 | 8.311 |
| prefill_4k | 1 | 4096 | 55904.924 | 13.648663 | 55893.003 | 55904.351 | 55902.014 | reader_writer_saturated | 0.042 | 2.401 |
| prefill_8k | 1 | 8192 | 218143.240 | 26.628813 | 218131.383 | 218142.667 | 218140.391 | reader_writer_saturated | 0.022 | 1.231 |
| prefill_16k | 1 | 16384 | 862221.635 | 52.625832 | 862209.601 | 862221.073 | 862218.602 | reader_writer_saturated | 0.011 | 0.623 |
| prefill_32k | 1 | 32768 | 3430012.966 | 104.675689 | 3430001.100 | 3430012.371 | 3430010.108 | reader_writer_saturated | 0.006 | 0.313 |

## Decode measured profile

| case | batch | seq_len | kernel us | us/item | ncrisc us | brisc us | compute us(max trisc) | bottleneck | est k-read GB/s | est out-write GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|

## Prefill custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_1k | 4025.983 | 0.451 | 9.834 | 985.479 | 2070.291 | 12.042 | 3106.070 | 77.2% | 0.3% | 32.0% | 67.3% | 0.4% |
| prefill_4k | 55893.003 | 0.542 | 145.738 | 14453.985 | 29028.846 | 177.023 | 44172.921 | 79.0% | 0.3% | 33.0% | 66.3% | 0.4% |
| prefill_8k | 218131.383 | 0.710 | 575.403 | 56939.896 | 113444.590 | 697.672 | 173097.754 | 79.4% | 0.3% | 33.2% | 66.1% | 0.4% |
| prefill_16k | 862209.601 | 1.065 | 2286.000 | 226006.441 | 448784.807 | 2770.044 | 685572.255 | 79.5% | 0.3% | 33.2% | 66.0% | 0.4% |
| prefill_32k | 3430001.100 | 1.944 | 9115.360 | 900527.842 | 1784964.770 | 11036.151 | 2728845.585 | 79.6% | 0.3% | 33.3% | 66.0% | 0.4% |

## Decode custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Prefill custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_1k | 4037.291 | 3725.845 | 31.594 | 33.255 | 0.804 | 3824.665 | 94.7% | 98.3% | 0.8% | 0.9% | 0.0% |
| prefill_4k | 55904.351 | 54454.938 | 102.446 | 100.852 | 3.137 | 55049.085 | 98.5% | 99.6% | 0.2% | 0.2% | 0.0% |
| prefill_8k | 218142.667 | 214583.227 | 192.280 | 187.178 | 6.269 | 216417.591 | 99.2% | 99.8% | 0.1% | 0.1% | 0.0% |
| prefill_16k | 862221.073 | 852205.273 | 380.528 | 367.129 | 12.876 | 858705.817 | 99.6% | 99.9% | 0.0% | 0.0% | 0.0% |
| prefill_32k | 3430012.371 | 3397977.260 | 742.663 | 708.800 | 25.757 | 3422714.541 | 99.8% | 100.0% | 0.0% | 0.0% | 0.0% |

## Decode custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

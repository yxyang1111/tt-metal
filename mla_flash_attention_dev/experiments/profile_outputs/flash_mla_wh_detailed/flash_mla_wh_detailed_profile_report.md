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
| prefill_256 | 1 | 256 | 412.770 | 1.612383 | 400.724 | 412.329 | 409.874 | reader_writer_saturated | 0.368 | 20.344 |
| prefill_512 | 1 | 512 | 1211.550 | 2.366309 | 1199.414 | 1211.104 | 1208.574 | reader_writer_saturated | 0.246 | 13.853 |
| prefill_1k | 1 | 1024 | 4009.476 | 3.915504 | 3997.468 | 4009.029 | 4006.625 | reader_writer_saturated | 0.148 | 8.370 |
| prefill_2k | 1 | 2048 | 14556.958 | 7.107889 | 14544.947 | 14556.502 | 14554.106 | reader_writer_saturated | 0.081 | 4.610 |
| prefill_4k | 1 | 4096 | 55339.322 | 13.510577 | 55327.319 | 55338.878 | 55336.467 | reader_writer_saturated | 0.043 | 2.425 |

## Decode measured profile

| case | batch | seq_len | kernel us | us/item | ncrisc us | brisc us | compute us(max trisc) | bottleneck | est k-read GB/s | est out-write GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| decode_256 | 2 | 256 | 40.917 | 20.458583 | 13.647 | 28.782 | 40.659 | compute_on_critical_path | 21.610 | 2.277 |
| decode_512 | 2 | 512 | 58.109 | 29.054417 | 18.256 | 45.985 | 57.845 | compute_on_critical_path | 32.308 | 1.425 |
| decode_1k | 2 | 1024 | 73.531 | 36.765417 | 27.253 | 61.376 | 73.267 | compute_on_critical_path | 43.284 | 1.068 |
| decode_2k | 2 | 2048 | 106.069 | 53.034417 | 45.138 | 93.910 | 105.809 | writer_close_to_critical_path | 52.268 | 0.698 |
| decode_4k | 2 | 4096 | 171.855 | 85.927333 | 107.650 | 159.711 | 171.596 | writer_close_to_critical_path | 43.833 | 0.410 |
| decode_8k | 2 | 8192 | 303.168 | 151.584250 | 238.594 | 291.010 | 302.909 | writer_close_to_critical_path | 39.553 | 0.225 |
| decode_16k | 1 | 16384 | 560.899 | 560.899500 | 497.738 | 548.789 | 560.631 | reader_close_to_critical_path | 18.960 | 0.060 |
| decode_32k | 1 | 32768 | 1084.760 | 1084.760250 | 1023.405 | 1072.649 | 1084.496 | reader_writer_saturated | 18.443 | 0.031 |

## Prefill custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_256 | 400.724 | 0.443 | 0.806 | 82.123 | 199.745 | 0.999 | 291.505 | 72.7% | 0.3% | 28.9% | 70.3% | 0.4% |
| prefill_512 | 1199.414 | 0.475 | 2.712 | 273.738 | 613.855 | 3.337 | 906.107 | 75.5% | 0.3% | 30.6% | 68.7% | 0.4% |
| prefill_1k | 3997.468 | 0.468 | 9.853 | 985.467 | 2063.975 | 12.038 | 3096.468 | 77.5% | 0.3% | 32.1% | 67.2% | 0.4% |
| prefill_2k | 14544.947 | 0.514 | 37.443 | 3722.899 | 7565.107 | 45.536 | 11461.041 | 78.8% | 0.3% | 32.7% | 66.5% | 0.4% |
| prefill_4k | 55327.319 | 0.545 | 145.882 | 14453.581 | 28818.300 | 176.968 | 43936.937 | 79.4% | 0.3% | 33.2% | 66.1% | 0.4% |

## Decode custom NCRISC stage breakdown

| case | thread us | page_table us/core(avg) | reserve us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | coverage vs thread | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 13.647 | 0.380 | 0.035 | 3.929 | 0.793 | 0.071 | 5.317 | 39.0% | 0.7% | 75.4% | 15.2% | 1.4% |
| decode_512 | 18.256 | 0.448 | 0.035 | 3.933 | 0.819 | 0.071 | 5.652 | 31.0% | 0.7% | 74.1% | 15.4% | 1.3% |
| decode_1k | 27.253 | 0.420 | 0.071 | 7.871 | 1.680 | 0.145 | 10.735 | 39.4% | 0.7% | 77.3% | 16.5% | 1.4% |
| decode_2k | 45.138 | 0.410 | 0.143 | 15.697 | 3.376 | 0.293 | 20.240 | 44.8% | 0.7% | 78.8% | 16.9% | 1.5% |
| decode_4k | 107.650 | 0.485 | 26.069 | 31.420 | 6.977 | 0.588 | 66.180 | 61.5% | 39.8% | 47.9% | 10.6% | 0.9% |
| decode_8k | 238.594 | 0.432 | 84.893 | 62.856 | 14.012 | 1.177 | 163.871 | 68.7% | 52.0% | 38.5% | 8.6% | 0.7% |
| decode_16k | 497.738 | 0.484 | 214.921 | 125.626 | 16.528 | 2.362 | 360.923 | 72.5% | 59.7% | 34.9% | 4.6% | 0.7% |
| decode_32k | 1023.405 | 0.525 | 463.280 | 251.306 | 32.916 | 4.727 | 754.885 | 73.8% | 61.5% | 33.4% | 4.4% | 0.6% |

## Prefill custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_256 | 412.329 | 322.773 | 14.778 | 15.938 | 0.196 | 358.132 | 86.9% | 91.3% | 4.2% | 4.5% | 0.1% |
| prefill_512 | 1211.104 | 1050.039 | 24.873 | 22.683 | 0.436 | 1107.971 | 91.5% | 95.6% | 2.3% | 2.1% | 0.0% |
| prefill_1k | 4009.029 | 3702.518 | 30.547 | 32.644 | 0.804 | 3796.350 | 94.7% | 98.3% | 0.8% | 0.9% | 0.0% |
| prefill_2k | 14556.502 | 13928.075 | 54.591 | 56.065 | 1.610 | 14131.300 | 97.1% | 99.2% | 0.4% | 0.4% | 0.0% |
| prefill_4k | 55338.878 | 53926.021 | 102.627 | 100.210 | 3.136 | 54483.307 | 98.5% | 99.6% | 0.2% | 0.2% | 0.0% |

## Decode custom BRISC stage breakdown

| case | thread us | cb_wait us/core(avg) | issue us/core(avg) | barrier us/core(avg) | pop us/core(avg) | profiled total us/core(max) | coverage vs thread | cb_wait share | issue share | barrier share | pop share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 28.782 | 10.860 | 0.656 | 0.594 | 0.035 | 24.498 | 85.1% | 89.4% | 5.4% | 4.9% | 0.3% |
| decode_512 | 45.985 | 15.274 | 0.856 | 0.742 | 0.052 | 29.092 | 63.3% | 90.2% | 5.1% | 4.4% | 0.3% |
| decode_1k | 61.376 | 23.299 | 0.744 | 0.652 | 0.052 | 44.962 | 73.3% | 94.1% | 3.0% | 2.6% | 0.2% |
| decode_2k | 93.910 | 39.546 | 0.809 | 0.668 | 0.052 | 77.529 | 82.6% | 96.3% | 2.0% | 1.6% | 0.1% |
| decode_4k | 159.711 | 72.297 | 0.709 | 0.637 | 0.052 | 143.123 | 89.6% | 98.1% | 1.0% | 0.9% | 0.1% |
| decode_8k | 291.010 | 137.924 | 0.800 | 0.685 | 0.052 | 274.007 | 94.2% | 98.9% | 0.6% | 0.5% | 0.0% |
| decode_16k | 548.789 | 267.461 | 0.584 | 0.548 | 0.052 | 532.093 | 97.0% | 99.6% | 0.2% | 0.2% | 0.0% |
| decode_32k | 1072.649 | 529.359 | 0.600 | 0.557 | 0.052 | 1055.945 | 98.4% | 99.8% | 0.1% | 0.1% | 0.0% |

## Decode A-BH empirical coupled model

这一节把 WH decode detailed profile 的 `reader reserve` / `writer cb_wait` 回灌到当前 A 路径的 BH 模型里，
不再只看理想化的 `compute vs dram`，而是把 pipeline backpressure 也显式带上。
下面的 source attribution 是解释性拆分，不是新的硬件计数器。

| case | seq_len | ideal 1st-order ms | empirical 2nd-order ms | compute ms | reader act/reserve ms | writer act/wait ms | dominant | uplift vs ideal |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| decode_256 | 256 | 0.0027 | 0.0069 | 0.0016 | 0.0068/0.0000 | 0.0021/0.0018 | reader | 2.53x |
| decode_512 | 512 | 0.0043 | 0.0095 | 0.0032 | 0.0091/0.0000 | 0.0071/0.0024 | writer | 2.19x |
| decode_1k | 1024 | 0.0075 | 0.0135 | 0.0064 | 0.0135/0.0000 | 0.0069/0.0043 | reader | 1.79x |
| decode_2k | 2048 | 0.0140 | 0.0227 | 0.0129 | 0.0227/0.0000 | 0.0069/0.0085 | reader | 1.62x |
| decode_4k | 4096 | 0.0269 | 0.0450 | 0.0258 | 0.0411/0.0039 | 0.0069/0.0190 | reader | 1.67x |
| decode_8k | 8192 | 0.0527 | 0.0922 | 0.0516 | 0.0777/0.0145 | 0.0071/0.0420 | reader | 1.75x |
| decode_16k | 16384 | 0.1043 | 0.1826 | 0.1032 | 0.1431/0.0395 | 0.0069/0.0871 | reader | 1.75x |
| decode_32k | 32768 | 0.2074 | 0.3715 | 0.2063 | 0.2833/0.0881 | 0.0069/0.1813 | reader | 1.79x |

## Decode A-BH empirical scaling summary

| case | seq_len | 1st-order dram crossover | 2nd-order non-compute crossover | 2nd-order plateau start (<=2% of 64c) | 16c->64c speedup | 64c floor ms |
|---|---:|---:|---:|---:|---:|---:|
| decode_256 | 256 | 23 | 4 | 4 | 1.00x | 0.0069 |
| decode_512 | 512 | 23 | 6 | 6 | 1.00x | 0.0095 |
| decode_1k | 1024 | 23 | 8 | 8 | 1.00x | 0.0135 |
| decode_2k | 2048 | 23 | 10 | 9 | 1.00x | 0.0227 |
| decode_4k | 4096 | 23 | 9 | 13 | 1.00x | 0.0450 |
| decode_8k | 8192 | 23 | 8 | 15 | 1.01x | 0.0914 |
| decode_16k | 16384 | 23 | 8 | 17 | 1.02x | 0.1784 |
| decode_32k | 32768 | 23 | 7 | 17 | 1.03x | 0.3617 |

## Decode A-BH empirical coupling attribution

| case | seq_len | reader active | reserve<-compute | reserve<-writer | writer active | wait<-compute | wait<-reader | dominant coupling |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| decode_256 | 256 | 0.0068 | 0.0000 | 0.0000 | 0.0021 | 0.0003 | 0.0015 | wait<-reader |
| decode_512 | 512 | 0.0091 | 0.0000 | 0.0000 | 0.0071 | 0.0006 | 0.0018 | wait<-reader |
| decode_1k | 1024 | 0.0135 | 0.0000 | 0.0000 | 0.0069 | 0.0014 | 0.0029 | wait<-reader |
| decode_2k | 2048 | 0.0227 | 0.0000 | 0.0000 | 0.0069 | 0.0031 | 0.0054 | wait<-reader |
| decode_4k | 4096 | 0.0411 | 0.0020 | 0.0020 | 0.0069 | 0.0069 | 0.0121 | wait<-reader |
| decode_8k | 8192 | 0.0777 | 0.0074 | 0.0070 | 0.0071 | 0.0151 | 0.0269 | wait<-reader |
| decode_16k | 16384 | 0.1431 | 0.0207 | 0.0189 | 0.0069 | 0.0315 | 0.0557 | wait<-reader |
| decode_32k | 32768 | 0.2833 | 0.0461 | 0.0420 | 0.0069 | 0.0647 | 0.1166 | wait<-reader |

## Decode A-BH empirical coupled active-core sweep

| case | seq_len | 16c | 20c | 24c | 32c | 48c | 64c |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 1024 | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) | 0.0135 (reader) |
| decode_4k | 4096 | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) | 0.0450 (reader) |
| decode_32k | 32768 | 0.3715 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) | 0.3617 (reader) |

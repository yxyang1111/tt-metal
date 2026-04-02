# FlashMLA WH profile summary

## WH measured profile

| case | kernel us | ncrisc us | brisc us | trisc1 us | wait_front us | bottleneck | est reader GB/s |
|---|---:|---:|---:|---:|---:|---|---:|
| decode_1k | 74.212 | 27.357 | 61.906 | 73.954 | 0.000 | compute_on_critical_path | 43.121 |
| decode_4k | 171.602 | 107.564 | 159.297 | 171.343 | 0.000 | balanced_with_reader_noc_pressure | 43.868 |
| decode_32k | 1084.909 | 1023.540 | 1072.610 | 1084.653 | 0.000 | reader_noc_saturated | 18.440 |

## WH custom NCRISC stage breakdown

| case | ncrisc us | page_table us/core(avg) | reserve/block us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | reserve share | issue share | wait share | push share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 27.357 | 0.467 | 0.071 | 7.846 | 1.686 | 0.146 | 10.769 | 0.7% | 76.8% | 16.5% | 1.4% |
| decode_4k | 107.564 | 0.471 | 26.179 | 31.408 | 6.998 | 0.588 | 66.228 | 39.9% | 47.8% | 10.7% | 0.9% |
| decode_32k | 1023.540 | 0.511 | 463.597 | 251.165 | 33.128 | 4.730 | 755.128 | 61.6% | 33.3% | 4.4% | 0.6% |

## BH simulated current path (A)

| case | seq_len | dram ms | compute ms | reduce ms | predicted core ms | dominant stage |
|---|---:|---:|---:|---:|---:|---|
| decode_1k | 1024 | 0.0046 | 0.0064 | 0.0011 | 0.0075 | compute |
| decode_4k | 4096 | 0.0184 | 0.0258 | 0.0011 | 0.0269 | compute |
| decode_32k | 32768 | 0.1475 | 0.2063 | 0.0011 | 0.2074 | compute |

## BH current path active-core sweep

| case | crossover active cores | 16c ms | 20c ms | 24c ms | 32c ms | 48c ms | 64c ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 23 | 0.0075 | 0.0063 | 0.0057 | 0.0057 | 0.0057 | 0.0057 |
| decode_4k | 23 | 0.0269 | 0.0217 | 0.0195 | 0.0195 | 0.0195 | 0.0195 |
| decode_32k | 23 | 0.2074 | 0.1662 | 0.1486 | 0.1486 | 0.1486 | 0.1486 |

## BH simulated native FlashMLA (B)

| case | seq_len | dram ms | q fanout ms | k mcast ms | compute ms | reduce ms | predicted core ms | dominant stage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| decode_1k | 1024 | 0.0012 | 0.0037 | 0.0019 | 0.0004 | 0.0011 | 0.0068 | k_mcast |
| decode_4k | 4096 | 0.0046 | 0.0037 | 0.0077 | 0.0016 | 0.0011 | 0.0126 | k_mcast |
| decode_32k | 32768 | 0.0369 | 0.0037 | 0.0614 | 0.0129 | 0.0011 | 0.0663 | k_mcast |

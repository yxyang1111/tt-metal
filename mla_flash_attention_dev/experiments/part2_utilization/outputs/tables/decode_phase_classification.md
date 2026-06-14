# Decode 阶段分类与关键指标

| case | seq_len | batch | classification | kernel us | BRISC share | NCRISC share | reader reserve | writer cb_wait | wait-front share | reserve-back share |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| decode_256 | 256 | 2 | compute_on_critical_path | 42.198 | 70.8% | 35.5% | 1.1% | 90.1% | 87.4% | 12.6% |
| decode_512 | 512 | 2 | compute_on_critical_path | 59.574 | 79.3% | 33.0% | 1.1% | 91.1% | 89.1% | 10.9% |
| decode_1k | 1024 | 2 | compute_on_critical_path | 76.223 | 83.8% | 39.5% | 1.1% | 93.9% | 86.1% | 13.9% |
| decode_2k | 2048 | 2 | writer_close_to_critical_path | 109.765 | 88.8% | 46.2% | 1.2% | 96.2% | 82.0% | 18.0% |
| decode_4k | 4096 | 2 | writer_close_to_critical_path | 177.332 | 93.0% | 62.7% | 32.0% | 98.0% | 78.1% | 21.9% |
| decode_8k | 8192 | 2 | writer_close_to_critical_path | 311.103 | 96.0% | 78.9% | 46.7% | 98.9% | 74.8% | 25.2% |
| decode_16k | 16384 | 1 | reader_close_to_critical_path | 575.166 | 97.9% | 88.8% | 55.4% | 99.6% | 72.0% | 28.0% |
| decode_32k | 32768 | 1 | reader_writer_saturated | 1112.087 | 98.9% | 94.3% | 57.6% | 99.8% | 70.9% | 29.1% |

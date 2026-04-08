# A-BH 二阶经验模型阈值

| case | seq_len | A-BH ideal | A-BH empirical | uplift | 1st dram crossover | 2nd noncompute crossover | plateau start | 64c floor |
|---|---|---:|---:|---|---:|---:|---:|---:|
| decode_1k | 1024 | 0.0075 ms | 0.0135 ms | 1.79x | 23c | 8c | 8c | 0.0135 ms |
| decode_4k | 4096 | 0.0269 ms | 0.0490 ms | 1.82x | 23c | 9c | 12c | 0.0490 ms |
| decode_8k | 8192 | 0.0527 ms | 0.0922 ms | 1.75x | 23c | 8c | 15c | 0.0914 ms |
| decode_16k | 16384 | 0.1043 ms | 0.1826 ms | 1.75x | 23c | 8c | 17c | 0.1784 ms |
| decode_32k | 32768 | 0.2074 ms | 0.3715 ms | 1.79x | 23c | 7c | 17c | 0.3617 ms |

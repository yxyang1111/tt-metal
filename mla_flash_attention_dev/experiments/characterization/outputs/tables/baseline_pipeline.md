# Baseline Pipeline Stage Characterization

| Case | Seq Len | Kernel (us) | NCRISC % | BRISC % | Compute % | Classification | Eff K BW (GB/s) |
|---|---:|---:|---:|---:|---:|---|---:|
| decode_1k | 1024 | 73.53 | 37.1% | 83.5% | 99.6% | compute_on_critical_path | 43.28 |
| decode_4k | 4096 | 173.1 | 63.2% | 93.0% | 99.8% | writer_close_to_critical_path | 43.15 |
| decode_8k | 8192 | 303.17 | 78.7% | 96.0% | 99.9% | writer_close_to_critical_path | 39.55 |
| decode_16k | 16384 | 560.9 | 88.7% | 97.8% | 100.0% | reader_close_to_critical_path | 18.96 |
| decode_32k | 32768 | 1084.76 | 94.3% | 98.9% | 100.0% | reader_writer_saturated | 18.44 |
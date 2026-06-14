# Baseline Bandwidth and Compute Characterization

| Case | Seq Len | Kernel (ms) | DRAM Floor w/ mcast (ms) | Kernel / Floor | OI (actual) | OI (ideal mcast) | Achieved (TFLOP/s) | PM FPU Util (TT) | Eff K BW/core (GB/s) | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode_1k | 1024 | 0.0735 | 0.0223 | 3.3x | 8.0 | 32.0 | 1.9394 | 12.17% | 43.28 | compute_on_critical_path |
| decode_4k | 4096 | 0.1731 | 0.0891 | 1.9x | 8.0 | 32.0 | 3.2953 | 20.57% | 43.15 | writer_close_to_critical_path |
| decode_8k | 8192 | 0.3032 | 0.1783 | 1.7x | 8.0 | 32.0 | 3.7631 | 23.11% | 39.55 | writer_close_to_critical_path |
| decode_16k | 16384 | 0.5609 | 0.1783 | 3.1x | 8.0 | 32.0 | 2.034 | 24.73% | 18.96 | reader_close_to_critical_path |
| decode_32k | 32768 | 1.0848 | 0.3565 | 3.0x | 8.0 | 32.0 | 2.1034 | 25.61% | 18.44 | reader_writer_saturated |
# Baseline Data Reuse Characterization

| Case | Seq Len | Batch | Workers | Head Groups | K Cache (MB) | K Redundancy | Total DRAM Read (MB) | Ideal DRAM Read (MB) | Waste % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 1024 | 2 | 32 | 4 | 1.18 | 4x | 17.83 | 4.46 | 75.0% |
| decode_4k | 4096 | 2 | 32 | 4 | 4.719 | 4x | 71.3 | 17.83 | 75.0% |
| decode_8k | 8192 | 2 | 32 | 4 | 9.437 | 4x | 142.61 | 35.65 | 75.0% |
| decode_16k | 16384 | 1 | 16 | 4 | 18.874 | 4x | 142.61 | 35.65 | 75.0% |
| decode_32k | 32768 | 1 | 16 | 4 | 37.749 | 4x | 285.21 | 71.3 | 75.0% |
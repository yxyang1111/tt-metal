# Native B-BH vs Current A-BH

| case | seq_len | A ideal | A empirical | B-BH | B / A ideal | B / A empirical | B dominant |
|---|---|---:|---:|---:|---:|---:|---|
| decode_256 | 256 | 0.0027 ms | 0.0069 ms | 0.0054 ms | 0.51x | 1.28x | k_mcast |
| decode_512 | 512 | 0.0043 ms | 0.0095 ms | 0.0058 ms | 0.74x | 1.63x | k_mcast |
| decode_1k | 1024 | 0.0075 ms | 0.0135 ms | 0.0068 ms | 1.11x | 1.99x | k_mcast |
| decode_2k | 2048 | 0.0140 ms | 0.0227 ms | 0.0087 ms | 1.61x | 2.60x | k_mcast |
| decode_4k | 4096 | 0.0269 ms | 0.0450 ms | 0.0126 ms | 2.14x | 3.59x | k_mcast |
| decode_8k | 8192 | 0.0527 ms | 0.0922 ms | 0.0202 ms | 2.60x | 4.56x | k_mcast |
| decode_16k | 16384 | 0.1043 ms | 0.1826 ms | 0.0356 ms | 2.93x | 5.13x | k_mcast |
| decode_32k | 32768 | 0.2074 ms | 0.3715 ms | 0.0663 ms | 3.13x | 5.60x | k_mcast |

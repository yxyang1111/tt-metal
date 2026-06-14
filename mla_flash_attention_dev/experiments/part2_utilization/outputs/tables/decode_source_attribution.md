# Decode 代表点 source-level attribution

| case | classification | k_issue | k_reserve | k_wait | v_issue | v_reserve | v_wait | sender_wait | tree_wait | root_wait | output_wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | compute_on_critical_path | 56.87% | 0.37% | 14.51% | 25.33% | 0.81% | 0.67% | 48.69% | 51.31% | 0.00% | 0.00% |
| decode_4k | writer_close_to_critical_path | 38.70% | 31.64% | 10.07% | 17.51% | 0.56% | 0.53% | 49.59% | 50.41% | 0.00% | 0.00% |
| decode_16k | reader_close_to_critical_path | 27.39% | 52.79% | 3.49% | 12.53% | 2.77% | 0.33% | 49.89% | 50.11% | 0.00% | 0.00% |
| decode_32k | reader_writer_saturated | 26.10% | 54.82% | 3.33% | 11.96% | 2.80% | 0.32% | 49.95% | 50.05% | 0.00% | 0.00% |

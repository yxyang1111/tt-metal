# Prefill 控制对照与 PM 利用率

| case | seq_len | classification | kernel ms | PM COMPUTE ms | PM BANDWIDTH ms | PM FPU util | reader wait | writer cb_wait | wait-front share | reserve-back share |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| prefill_256 | 256 | reader_writer_saturated | 0.413 | 0.042 | 0.034 | 10.20% | 70.31% | 91.50% | 95.98% | 4.02% |
| prefill_512 | 512 | reader_writer_saturated | 1.212 | 0.169 | 0.068 | 13.90% | 68.65% | 95.71% | 94.62% | 5.38% |
| prefill_1k | 1024 | reader_writer_saturated | 4.021 | 0.674 | 0.136 | 16.76% | 67.31% | 98.25% | 93.58% | 6.42% |
| prefill_2k | 2048 | reader_writer_saturated | 14.595 | 2.696 | 0.273 | 18.48% | 66.63% | 99.19% | 92.93% | 7.07% |
| prefill_4k | 4096 | reader_writer_saturated | 55.516 | 10.785 | 0.545 | 19.43% | 66.25% | 99.62% | 92.58% | 7.42% |

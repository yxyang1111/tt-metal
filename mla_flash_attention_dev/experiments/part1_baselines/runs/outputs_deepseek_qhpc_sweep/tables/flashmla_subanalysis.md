# FlashMLA 补充分析

## 说明

- 当前补充分析复用了已有 detailed C++ device profiler 结果：`/rshome/yuxin.yang/dev/tt-metal/mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_wh_detailed`
- 原因：本机环境下 `python3 -m tracy` 的 host-side 后处理容易卡住，因此这里直接复用稳定可得的 `cpp_device_perf_report.csv / profile_log_device.csv`。
- 口径：`decode_1k/4k/8k` 来自历史 `batch=2` detailed sweep，`decode_16k/32k` 为 `batch=1`；它们只作为补充内部归因，不混入 Part I 主图。

## Decode 阶段迁移

| case | seq_len | kernel us | classification | ncrisc share | brisc share | compute share |
|---|---:|---:|---|---:|---:|---:|
| decode_1k | 1024 | 73.531 | compute_on_critical_path | 37.1% | 83.5% | 99.6% |
| decode_4k | 4096 | 173.101 | writer_close_to_critical_path | 63.2% | 93.0% | 99.8% |
| decode_8k | 8192 | 303.168 | writer_close_to_critical_path | 78.7% | 96.0% | 99.9% |
| decode_16k | 16384 | 560.899 | reader_close_to_critical_path | 88.7% | 97.8% | 100.0% |
| decode_32k | 32768 | 1084.760 | reader_writer_saturated | 94.3% | 98.9% | 100.0% |

## 剩余优化空间

| case | seq_len | measured kernel ms | ideal 1st-order ms | empirical 2nd-order ms | uplift vs ideal | dominant |
|---|---:|---:|---:|---:|---:|---|
| decode_1k | 1024 | 0.0735 | 0.0075 | 0.0135 | 1.79x | reader |
| decode_4k | 4096 | 0.1731 | 0.0269 | 0.0490 | 1.82x | reader |
| decode_8k | 8192 | 0.3032 | 0.0527 | 0.0922 | 1.75x | reader |
| decode_16k | 16384 | 0.5609 | 0.1043 | 0.1826 | 1.75x | reader |
| decode_32k | 32768 | 1.0848 | 0.2074 | 0.3715 | 1.79x | reader |

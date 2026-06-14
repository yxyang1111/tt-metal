# Prefill 三方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) |
|---|---|---|---|
| prefill_1k | avg 58.324 / best 53.678 / worst 65.570 ms (10/10 kept) | avg 3.715 / best 3.693 / worst 3.732 ms (9/10 kept) | avg 3.951 / best 3.945 / worst 3.965 ms (10/10 kept) |
| prefill_4k | avg 305.807 / best 288.683 / worst 330.261 ms (10/10 kept) | avg 49.902 / best 49.245 / worst 53.748 ms (7/10 kept) | avg 52.546 / best 52.520 / worst 52.589 ms (10/10 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) |
|---|---:|---:|---:|
| prefill_1k | 17557.2 tok/s | 275618.8 tok/s | 259175.1 tok/s |
| prefill_4k | 13394.1 tok/s | 82081.6 tok/s | 77951.1 tok/s |

# Prefill 三方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) |
|---|---|---|---|
| prefill_1k | avg 295.972 / best 287.221 / worst 303.331 ms (10/10 kept) | avg 19.596 / best 19.567 / worst 19.735 ms (20/20 kept) | avg 21.066 / best 21.048 / worst 21.087 ms (20/20 kept) |
| prefill_4k | avg 1587.460 / best 1470.621 / worst 1654.760 ms (10/10 kept) | avg 266.852 / best 266.753 / worst 267.045 ms (20/20 kept) | avg 286.334 / best 286.220 / worst 286.487 ms (20/20 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) |
|---|---:|---:|---:|
| prefill_1k | 20758.7 tok/s | 313535.3 tok/s | 291655.9 tok/s |
| prefill_4k | 15481.3 tok/s | 92096.0 tok/s | 85829.9 tok/s |

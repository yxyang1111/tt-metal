# Prefill 三方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。
说明：这张默认主表固定在 `config=b1_h16_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，即 `B=1, H=16, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`；更多 B/H/dims 对比见 `multidim/` 目录。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) |
|---|---|---|---|
| prefill_1k | avg 35.327 / best 31.658 / worst 40.808 ms (10/10 kept) | avg 5.177 / best 1.923 / worst 12.814 ms (10/10 kept) | avg 6.907 / best 2.081 / worst 16.457 ms (10/10 kept) |
| prefill_4k | avg 193.165 / best 191.657 / worst 194.182 ms (10/10 kept) | avg 26.845 / best 26.812 / worst 26.879 ms (10/10 kept) | avg 28.754 / best 28.738 / worst 28.777 ms (10/10 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) |
|---|---:|---:|---:|
| prefill_1k | 28986.3 tok/s | 197805.5 tok/s | 148264.0 tok/s |
| prefill_4k | 21204.6 tok/s | 152581.3 tok/s | 142451.9 tok/s |

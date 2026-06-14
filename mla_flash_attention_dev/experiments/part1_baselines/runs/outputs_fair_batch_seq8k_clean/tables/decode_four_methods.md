# Decode 四方法 10 次统计（已去除慢尾异常点）

说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。
说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。
说明：这张默认主表固定在 `config=b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，即 `B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`；更多 B/H/dims 对比见 `multidim/` 目录。
说明：`DeepSeek FlashMLA` 采用“预先完成一次张量适配后，只计时 backend device op”的口径，因此可与其它 TT device baseline 更公平比较。

## 延迟统计

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---|---|---|---|
| decode_8k | avg 233.072 / best 227.949 / worst 236.284 ms (10/10 kept) | avg 0.408 / best 0.396 / worst 0.428 ms (10/10 kept) | avg 0.426 / best 0.412 / worst 0.438 ms (10/10 kept) | avg 0.438 / best 0.407 / worst 0.521 ms (10/10 kept) |

## 平均吞吐量

| case | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |
|---|---:|---:|---:|---:|
| decode_8k | 4.3 tok/s | 2451.6 tok/s | 2349.8 tok/s | 2282.7 tok/s |

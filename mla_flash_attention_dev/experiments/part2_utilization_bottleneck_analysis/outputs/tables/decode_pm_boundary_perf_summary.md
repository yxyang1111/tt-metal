# Decode PM / Perf-Counter Boundary Summary

说明：

- 这里汇总的是新补的 corrected-runtime `tracy_report + perf-fpu + sync` 容量边界点，专门用来把 decode 侧直接算术利用率和 `4c/8c` q-core admission boundary 接到同一条证据链上。
- 选点固定在 `seq=8k`，并优先沿着 `H=32 / q_shards=4` 这条主线观察：当前 densified q-core 轴为 `24q/28q/32q/40q/48q`；`24q` 是已 fit anchor，其余点用于观察跨墙后 `8c` 的直接算术利用率变化。
- `28q/32q/40q/48q` 上 `DeepSeek-4c` 若显示 `unsupported_precheck`，表示它在 preflight 阶段就被容量约束拦下；这正是本节想保留的边界证据，而不是 probe 异常。

| case | focus | method | status | seq_len | B | q_shards | required q-cores | PM FPU util | PM BANDWIDTH | 4c note |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | fit_24q | TT | completed | 8k | 6 | 4 | 24 | 21.86% | 102198 | n/a |
| B2 | fit_24q | DeepSeek-4c | completed | 8k | 6 | 4 | 24 | 12.08% | 102198 | n/a |
| B2 | fit_24q | DeepSeek-8c | completed | 8k | 6 | 4 | 24 | 12.08% | 102198 | n/a |
| D28 | cross_28q | TT | completed | 8k | 7 | 4 | 28 | 21.60% | 119231 | n/a |
| D28 | cross_28q | DeepSeek-4c | unsupported_precheck | 8k | 7 | 4 | 28 | n/a | n/a | batch * deepseek_num_q_shards must be <= 24, got 28 |
| D28 | cross_28q | DeepSeek-8c | completed | 8k | 7 | 4 | 28 | 13.78% | 119231 | n/a |
| D1 | cross_32q | TT | completed | 8k | 8 | 4 | 32 | 21.27% | 136264 | n/a |
| D1 | cross_32q | DeepSeek-4c | unsupported_precheck | 8k | 8 | 4 | 32 | n/a | n/a | batch * deepseek_num_q_shards must be <= 24, got 32 |
| D1 | cross_32q | DeepSeek-8c | completed | 8k | 8 | 4 | 32 | 13.74% | 136264 | n/a |
| D40 | cross_40q | TT | completed | 8k | 10 | 4 | 40 | 20.36% | 170330 | n/a |
| D40 | cross_40q | DeepSeek-4c | unsupported_precheck | 8k | 10 | 4 | 40 | n/a | n/a | batch * deepseek_num_q_shards must be <= 24, got 40 |
| D40 | cross_40q | DeepSeek-8c | completed | 8k | 10 | 4 | 40 | 16.40% | 170330 | n/a |
| D2 | cross_48q | TT | completed | 8k | 12 | 4 | 48 | 18.21% | 204396 | n/a |
| D2 | cross_48q | DeepSeek-4c | unsupported_precheck | 8k | 12 | 4 | 48 | n/a | n/a | batch * deepseek_num_q_shards must be <= 24, got 48 |
| D2 | cross_48q | DeepSeek-8c | completed | 8k | 12 | 4 | 48 | 18.53% | 204396 | n/a |

- 在已 fit 的 `B2` anchor 上，`PM FPU util (TT / 4c / 8c) = 21.86% / 12.08% / 12.08%`；`8c-4c` 的 `PM FPU util` 差值只有 `0.00 pp`。
- 沿 densified q-core 轴 `24q/28q/32q/40q/48q`，`TT PM FPU util` = `24q` 21.86%；`28q` 21.60%；`32q` 21.27%；`40q` 20.36%；`48q` 18.21%，`DeepSeek-8c PM FPU util` = `24q` 12.08%；`28q` 13.78%；`32q` 13.74%；`40q` 16.40%；`48q` 18.53%；`DeepSeek-4c` 只在 `<=24q` 仍可运行，`28q` 起就会直接 `unsupported_precheck`。
- 这说明 `8c` 的第一作用是跨越 q-core admission boundary，而不是让已 fit 点的算术利用率发生跳变；`B2` 上 `4c≈8c` 时，算术利用率本身也没有被明显改写，而 `D28` 已经直接给出了“刚越墙就被 admission 拦下”的第一手证据。

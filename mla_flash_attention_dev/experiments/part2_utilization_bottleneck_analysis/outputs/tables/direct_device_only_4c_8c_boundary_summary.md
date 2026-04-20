# 4c / 8c Direct Device-Only Summary

说明：

- 这里汇总的是当前已补到的 `B/C/D` direct representative-point probe，用来回答“`4c` 与 `8c` 的差别到底来自哪里”。
- `required_q_cores = batch * q_shards`；对 `4c` 可理解为 `24` 个 q-core 容量，对 `8c` 为 `48` 个 q-core 容量。
- `wait_front / reserve_back` 仍按 accumulated stall density / kernel 解读，因此数值可能大于 `100%`，不应直接当作 wall-time share。

## B/C：已 fit 点上的 4c vs 8c

| case | seq_len | B | H | q_shards | required q-cores | 4c kernel_us | 8c kernel_us | 8c / 4c | 4c / TT | 8c / TT |
|---|---|---|---|---|---|---:|---:|---|---|---|
| B1 | 4k | 6 | 32 | 4 | 24 | 172.47 | 172.11 | 0.998x | 0.902x | 0.900x |
| B2 | 8k | 6 | 32 | 4 | 24 | 256.62 | 257.24 | 1.002x | 0.788x | 0.790x |
| B3 | 32k | 6 | 32 | 4 | 24 | 692.26 | 688.83 | 0.995x | 0.610x | 0.607x |
| C1 | 4k | 8 | 24 | 3 | 24 | 182.97 | 183.00 | 1.000x | 0.907x | 0.907x |
| C2 | 8k | 8 | 24 | 3 | 24 | 289.99 | 291.02 | 1.004x | 0.863x | 0.866x |
| C3 | 32k | 8 | 24 | 3 | 24 | 851.90 | 859.46 | 1.009x | 0.747x | 0.753x |

- `B/C` 六个点的 `8c / 4c kernel` 只落在 `0.995x~1.009x`；这些点都满足 `required_q_cores=24`，说明当 workload 已经 fit 进 `4c` 容量时，单纯把 block 扩到 `8c` 并不会自动更快。
- `B` 组对应 `H=32 -> q_shards=4`，`C` 组对应 `H=24 -> q_shards=3`；两条主线虽然分片不同，但只要 `batch * q_shards = 24`，`8c` 都几乎不带来额外 kernel 收益。

## D：capacity boundary

| case | seq_len | B | q_shards | required q-cores | 4c status | 4c note | 8c kernel_us | TT kernel_us | 8c / TT | 8c reserve_back / kernel |
|---|---|---|---|---|---|---|---:|---:|---|---|
| D1 | 8k | 8 | 4 | 32 | unsupported_precheck | batch * deepseek_num_q_shards must be <= 24, got 32 | 290.74 | 336.05 | 0.865x | 257.1% |
| D2 | 8k | 12 | 4 | 48 | unsupported_precheck | batch * deepseek_num_q_shards must be <= 24, got 48 | 376.06 | 382.63 | 0.983x | 276.3% |

- `D1/D2` 把 capacity wall 钉得很清楚：`required_q_cores=32 / 48` 时，`4c` 会在 preflight 阶段被拦下，而 `8c` 可以正常运行，对应的 `8c / TT` kernel ratio 为 `0.865x / 0.983x`。
- `D2` 虽然已经被 `8c` 跑通，但 `reserve_back / kernel` 仍有 `276.3%`，说明 `8c` 先解决的是容量墙，接下来主矛盾会转向 downstream backpressure。

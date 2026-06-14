# 4c / 8c Direct Source-Level Summary

说明：

- 这张表直接复用现有 `deepseek_flash_mla_part2_probe` 的 `device_only` artifacts，从 `profile_log_device.csv` 离线解析 stage marker，不需要重跑 probe。
- 现在 `TT` 与 `DeepSeek-4c/8c` 都能稳定读到 reader `K/V` 与 writer `sender/tree/root/output` source marker；若 `D1/D2` 上 `DeepSeek-4c` 显示 `no / n/a`，原因是它在 preflight 阶段就 unsupported，而不是 marker 丢失。

## B/C：已 fit 点上的 reader / writer / reduction source

| case | seq_len | B | H | TT K reserve | 4c K reserve | 8c K reserve | 8c-4c K | 4c sender | 4c tree | 8c sender | 8c tree | 8c-4c tree | 8c / 4c kernel |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| B1 | 4k | 6 | 32 | 6.26% | 0.11% | 0.11% | 0.00 pp | 41.76% | 57.15% | 41.85% | 57.07% | -0.09 pp | 0.998x |
| B2 | 8k | 6 | 32 | 21.94% | 0.12% | 0.12% | -0.00 pp | 42.08% | 57.25% | 42.13% | 57.21% | -0.04 pp | 1.002x |
| B3 | 32k | 6 | 32 | 30.87% | 0.13% | 0.13% | 0.00 pp | 43.83% | 55.96% | 43.82% | 55.97% | 0.01 pp | 0.995x |
| C1 | 4k | 8 | 24 | 0.23% | 0.11% | 0.11% | -0.00 pp | 41.90% | 56.87% | 41.93% | 56.85% | -0.03 pp | 1.000x |
| C2 | 8k | 8 | 24 | 8.12% | 0.12% | 0.12% | -0.00 pp | 41.73% | 57.56% | 41.71% | 57.58% | 0.03 pp | 1.004x |
| C3 | 32k | 8 | 24 | 15.48% | 0.13% | 0.13% | -0.00 pp | 42.66% | 57.12% | 42.65% | 57.14% | 0.02 pp | 1.009x |

- `B/C` 六个已 fit 点上，`TT` 的 `K reserve` 落在 `0.23%~30.87%`；`DeepSeek-4c` 只在 `0.11%~0.13%`，`DeepSeek-8c` 也只是 `0.11%~0.13%`。
- `8c-4c` 的 `K reserve` 差值最大只有 `0.00 pp`，说明这些已 fit 点上 `8c≈4c` 不只是 writer 结构没变，reader 侧供给形态也几乎没有变。
- `B/C` 六个已 fit 点上，`DeepSeek-4c` 的 writer source 一直落在 `sender=41.73%~43.83%`、`tree=55.96%~57.56%`；`DeepSeek-8c` 也仍是 `sender=41.71%~43.82%`、`tree=55.97%~57.58%`。
- `8c-4c` 的 `tree_child_wait` 差值最大也只有 `0.09 pp`，说明这些已 fit 点上 `8c≈4c` 的直接原因之一，是 `8c` 并没有重塑 writer / reduction 的主等待结构。

## D：容量墙之后的 source-level 残余瓶颈

| case | seq_len | B | required q-cores | 8c K reserve | 8c sender | 8c tree | 8c root | 8c output | 8c reserve_back / kernel |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| D1 | 8k | 8 | 32 | 0.12% | 41.74% | 57.55% | 0.71% | 0.00% | 257.1% |
| D2 | 8k | 12 | 48 | 0.14% | 49.41% | 49.72% | 0.88% | 0.00% | 276.3% |

- `D2` 的 `DeepSeek-8c` 已经不是 final gather 问题：`K reserve/sender/tree/root/output = 0.14%/49.41%/49.72%/0.88%/0.00%`。
这说明跨过 q-core 容量墙之后，reader `K reserve` 已经很低，剩余压力仍主要堆在 sender / tree reduction 链路，而不是落到 final gather 或 reader 供给。

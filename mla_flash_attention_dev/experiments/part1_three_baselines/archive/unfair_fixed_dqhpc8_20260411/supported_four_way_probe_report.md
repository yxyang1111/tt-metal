# 四方法共同支持的 Probe 性能结果

- 这里只保留四种 decode 方法都成功执行的 workloads，不再展示 supported / unsupported 状态本身。
- 数据源是 `capability probe` 的单次 smoke measurement，因此适合做大空间粗粒度对比，不替代主 benchmark 的 10 次统计。
- 当前结果覆盖 `800` 个 supported cases、`80` 组 configs。
- 覆盖轴：`seq_len=256, 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k`，`B=1, 2, 4, 8, 16`，`H=8, 16, 24, 32`，`value_dim=128, 192, 256, 384, 512`。
- `B/H/value_dim` 维度图表固定在 `seq_len=8k`。

## 图表

- 四方法共同支持结果按 seq_len 的平均时延图：`visuals/supported_four_way_probe_seq_latency.svg`
- 四方法共同支持结果按 seq_len 的平均吞吐量图：`visuals/supported_four_way_probe_seq_throughput.svg`
- 四方法共同支持结果按 batch 的平均时延图：`visuals/supported_four_way_probe_batch_latency.svg`
- 四方法共同支持结果按 heads 的平均时延图：`visuals/supported_four_way_probe_head_latency.svg`
- 四方法共同支持结果按 value_dim 的平均时延图：`visuals/supported_four_way_probe_dim_latency.svg`

## 表格

- 全量结果表：`tables/supported_four_way_probe_all_cases.md`
- 按 `seq_len` 汇总：`tables/supported_four_way_probe_seq_summary.md`
- 按 `batch` 汇总：`tables/supported_four_way_probe_batch_summary.md`
- 按 `heads` 汇总：`tables/supported_four_way_probe_head_summary.md`
- 按 `value_dim` 汇总：`tables/supported_four_way_probe_dim_summary.md`

## 原始数据

- Supported-only JSON：`raw/capability_probe_supported_four_way_results.json`
- Supported-only CSV：`raw/capability_probe_supported_four_way_results.csv`


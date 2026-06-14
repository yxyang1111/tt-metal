# Part I 四方法 Decode 实验结果

## 1. 实验范围

- `decode` 四方法：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`DeepSeek FlashMLA`。
- `prefill` 三方法控制对照：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`。
- 默认主表固定在当前 payload 的 `default_config` 上；如果结果包含多维 sweep，`multidim/` 目录会按 `B / H / dims` 分轴展示。
- 当前 DeepSeek 并行度策略：`align_with_tt_mainline`；如为 `align_with_tt_mainline`，则 `H` 轴图会同时展示派生得到的 `dqhpc / num_q_shards`。
- 每种方法默认测量多次；主表统计前会剔除慢尾异常点，规则为 `modified z-score > 5.0` 且 `latency > median x 1.10`。
- 每个延迟单元格报告过滤后的平均值 / 最好值 / 最坏值；`8/10 kept` 表示 10 次中保留了 8 次样本。
- `reference attention` 使用 host 侧 torch reference SDPA 作为非 TT 控制组；TT 设备上的主比较应优先看 `Flash Attention / FlashMLA（TT 主线） / DeepSeek FlashMLA`。
- `DeepSeek FlashMLA` 当前主表采用“先做一次 DeepSeek -> builtin tensor adaptation、计时时只测 backend device op”的口径，因此不会被现有 Python 包装层里的 `to_torch/from_torch` 开销放大。
- 默认主表 config：`b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，即 `B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。
- `FlashMLA（TT 主线）` 的补充 detailed characterization 复用了已有稳定的 C++ device profiler 结果，仅作为内部归因补充，不混入主图口径。
- `tables/fair_four_method_batch_summary.md` 与 `tables/fair_four_method_dim_summary.md` 已用补跑的正式 `decode` benchmark 覆盖，不再是旧版 fair payload 中只有单值的占位切片。

## 2. 核心结论

- `decode_256`: DeepSeek = `avg 0.185 / best 0.169 / worst 0.198 ms (9/10 kept)`，平均吞吐量 = `~5409.8 tok/s`，相对 Flash = `~0.93x`，相对 TT 主线 FlashMLA = `~0.89x`。
- `decode_1k`: DeepSeek = `avg 0.181 / best 0.171 / worst 0.190 ms (9/10 kept)`，平均吞吐量 = `~5522.6 tok/s`，相对 Flash = `~1.08x`，相对 TT 主线 FlashMLA = `~1.12x`。
- `decode_8k`: DeepSeek = `avg 0.462 / best 0.449 / worst 0.469 ms (10/10 kept)`，平均吞吐量 = `~2163.7 tok/s`，相对 Flash = `~0.83x`，相对 TT 主线 FlashMLA = `~1.02x`。
- `decode_32k`: DeepSeek = `avg 1.216 / best 1.211 / worst 1.230 ms (9/10 kept)`，平均吞吐量 = `~822.1 tok/s`，相对 Flash = `~0.98x`，相对 TT 主线 FlashMLA = `~1.01x`。
- `decode_128k`: DeepSeek = `avg 4.406 / best 4.381 / worst 4.433 ms (6/10 kept)`，平均吞吐量 = `~227.0 tok/s`，相对 Flash = `~6.80x`，相对 TT 主线 FlashMLA = `~8.08x`。
- `batch` 补跑固定 `seq_len=8k`、聚合 `H=8/16/24/32` 后可以看到：`B=1~4` 时三种 TT 方法都维持在 `0.3~0.4 ms` 量级；其中 DeepSeek 在 `B=2` / `B=4` 的聚合时延分别约为 Flash 的 `1.29x` / `1.23x`、约为 TT 主线 FlashMLA 的 `1.32x` / `1.23x`。
- 同一组 `batch` 补跑里，TT 主线 FlashMLA 在更大 batch 开始明显拉开与 Flash 的差距：聚合口径下 `B=8` 约快 `1.24x`，`B=16` 约快 `1.42x`；但 DeepSeek 在这两个点覆盖不完整，`B=8` 仅 `1 cfg`，`B=16` 无成功记录，因此这里不能把它当作完整四方法结论。
- `value_dim` 补跑固定 `seq_len=8k`、`B=1`、聚合 `H=8/16/24/32` 后显示：当 `value_dim` 从 `128 -> 512` 时，reference latency 放大约 `4.15x`，而 Flash / TT 主线 FlashMLA / DeepSeek 仅放大约 `1.56x / 1.56x / 1.62x`，说明 TT decode 路径对 `value_dim` 增长更平滑。
- 在同一组 `value_dim` 补跑里，三种 TT 方法整体仍处于近似同一量级；到 `value_dim=512` 时，聚合延迟分别约为 Flash `0.390 ms`、TT 主线 FlashMLA `0.398 ms`、DeepSeek `0.420 ms`，DeepSeek 仍略慢，但差距已经缩小到个位数百分比量级。

## 3. 图表

- Decode 四方法时延图：`visuals/decode_latency_four_methods.svg`
- Decode 四方法吞吐量图：`visuals/decode_throughput_four_methods.svg`
- Decode 相对 reference 速度提升：`visuals/decode_speedup_vs_reference.svg`
- DeepSeek 相对其它 TT 方法的速度提升：`visuals/decode_deepseek_vs_tt_speedup.svg`
- `FlashMLA（TT 主线）` 阶段迁移补充图：`visuals/flashmla_phase_transition.svg`
- `FlashMLA（TT 主线）` 剩余优化空间补充图：`visuals/flashmla_remaining_optimization_gap.svg`
- Decode head sweep 时延图：`multidim/visuals/decode_head_sweep_latency.svg`
- 公平补跑 Batch 切片时延图（`seq_len=8k`）：`visuals/fair_batch_seq8k_slice_latency.svg`
- 公平补跑 Batch 切片吞吐量图（`seq_len=8k`）：`visuals/fair_batch_seq8k_slice_throughput.svg`
- 公平补跑 Value Dim 切片时延图（`seq_len=8k`）：`visuals/fair_dim_seq8k_slice_latency.svg`
- 公平补跑 Value Dim 切片吞吐量图（`seq_len=8k`）：`visuals/fair_dim_seq8k_slice_throughput.svg`

## 4. 表格

- 实验参数总表：`tables/experiment_parameters.md`
- Decode 默认主表：`tables/decode_four_methods.md`
- Decode 统计口径说明：`tables/decode_best_worst_ties.md`
- 公平四方法全量结果大表：`tables/fair_four_method_all_cases.md`
- 公平四方法切片小表：`tables/fair_four_method_head_slice_summary.md`
- 公平四方法按 Seq Len 聚合：`tables/fair_four_method_seq_summary.md`
- 公平四方法按 Batch 切片：`tables/fair_four_method_batch_summary.md`
- 公平四方法按 Value Dim 切片：`tables/fair_four_method_dim_summary.md`
- Prefill 默认主表：`tables/prefill_control.md`
- `FlashMLA（TT 主线）` 补充分析表：`tables/flashmla_subanalysis.md`
- 四方法共同支持的 probe 性能报告：`supported_four_way_probe_report.md`
- Capability probe 汇总表：`tables/capability_probe_summary.md`
- Capability probe 明细表：`tables/capability_probe.md`
- Decode batch sweep：`multidim/tables/decode_batch_sweep.md`
- Decode head sweep：`multidim/tables/decode_head_sweep.md`
- Decode dim sweep：`multidim/tables/decode_dim_sweep.md`
- Decode DeepSeek q_heads_per_core sweep：`multidim/tables/decode_deepseek_q_heads_per_core_sweep.md`

补跑切片说明：
`tables/fair_four_method_batch_summary.md` 当前固定 `seq_len=8k`，展示 `B=1/2/4/8/16`，并对 `H=8/16/24/32` 做聚合。
`tables/fair_four_method_dim_summary.md` 当前固定 `seq_len=8k`，展示 `value_dim=128/192/256/384/512`，并对 `B=1, H=8/16/24/32` 做聚合。
`batch` 补跑里 DeepSeek FlashMLA 在大 batch 下未全量成功：`B=8` 仅保留 `1 cfg`，`B=16` 无成功记录，因此对应单元格里的 `cfgs` 数和 `-` 需要结合覆盖率一起解读。
由于 `reference attention` 的时延量级远高于 TT device baselines，补跑 latency 图更适合看整体趋势；TT 方法之间的细粒度差别请优先结合对应 summary 表解读。

## 5. 原始与过滤后结果

- 原始 JSON：`raw/part1_four_method_results.json`
- 原始 CSV：`raw/part1_four_method_results.csv`
- 过滤后 JSON：`raw/part1_four_method_results_filtered.json`
- 过滤后 CSV：`raw/part1_four_method_results_filtered.csv`
- `FlashMLA（TT 主线）` detailed profile：`flashmla_detailed/flash_mla_wh_detailed_profile_results.json`


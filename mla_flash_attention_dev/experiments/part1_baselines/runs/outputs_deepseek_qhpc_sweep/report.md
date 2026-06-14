# Part I 四方法 Decode 实验结果

## 1. 实验范围

- `decode` 四方法：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`DeepSeek FlashMLA`。
- `prefill` 三方法控制对照：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`。
- 默认主表固定在当前 payload 的 `default_config` 上；如果结果包含多维 sweep，`multidim/` 目录会按 `B / H / dims` 分轴展示。
- 当前 DeepSeek 并行度策略：`fixed`；如为 `align_with_tt_mainline`，则 `H` 轴图会同时展示派生得到的 `dqhpc / num_q_shards`。
- 每种方法默认测量多次；主表统计前会剔除慢尾异常点，规则为 `modified z-score > 5.0` 且 `latency > median x 1.10`。
- 每个延迟单元格报告过滤后的平均值 / 最好值 / 最坏值；`8/10 kept` 表示 10 次中保留了 8 次样本。
- `reference attention` 使用 host 侧 torch reference SDPA 作为非 TT 控制组；TT 设备上的主比较应优先看 `Flash Attention / FlashMLA（TT 主线） / DeepSeek FlashMLA`。
- `DeepSeek FlashMLA` 当前主表采用“先做一次 DeepSeek -> builtin tensor adaptation、计时时只测 backend device op”的口径，因此不会被现有 Python 包装层里的 `to_torch/from_torch` 开销放大。
- 默认主表 config：`b1_h16_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，即 `B=1, H=16, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。
- `FlashMLA（TT 主线）` 的补充 detailed characterization 复用了已有稳定的 C++ device profiler 结果，仅作为内部归因补充，不混入主图口径。
- 当前目录用于回答“不同 `deepseek_num_q_heads_per_core` 会如何影响四方法对比”。
- 公平对齐并行度后的主结果见 `../outputs`；旧的固定 `dqhpc=8` 不公平对比归档见 `../archive/unfair_fixed_dqhpc8_20260411`。

## 2. 核心结论

- `decode_256`: DeepSeek = `avg 0.173 / best 0.157 / worst 0.198 ms (9/10 kept)`，平均吞吐量 = `~5779.5 tok/s`，相对 Flash = `~78.47x`，相对 TT 主线 FlashMLA = `~88.46x`。
- `decode_1k`: DeepSeek = `avg 0.196 / best 0.183 / worst 0.208 ms (9/10 kept)`，平均吞吐量 = `~5096.1 tok/s`，相对 Flash = `~12.71x`，相对 TT 主线 FlashMLA = `~34.55x`。
- `decode_8k`: DeepSeek = `avg 0.575 / best 0.554 / worst 0.635 ms (10/10 kept)`，平均吞吐量 = `~1738.3 tok/s`，相对 Flash = `~3.23x`，相对 TT 主线 FlashMLA = `~12.00x`。
- `decode_32k`: DeepSeek = `avg 5.780 / best 1.856 / worst 10.346 ms (10/10 kept)`，平均吞吐量 = `~173.0 tok/s`，相对 Flash = `~0.23x`，相对 TT 主线 FlashMLA = `~2.88x`。
- `decode_128k`: DeepSeek = `avg 7.013 / best 6.952 / worst 7.051 ms (9/10 kept)`，平均吞吐量 = `~142.6 tok/s`，相对 Flash = `~0.49x`，相对 TT 主线 FlashMLA = `~0.96x`。

## 3. 图表

- Decode 四方法时延图：`visuals/decode_latency_four_methods.svg`
- Decode 四方法吞吐量图：`visuals/decode_throughput_four_methods.svg`
- Decode 相对 reference 速度提升：`visuals/decode_speedup_vs_reference.svg`
- DeepSeek 相对其它 TT 方法的速度提升：`visuals/decode_deepseek_vs_tt_speedup.svg`
- `FlashMLA（TT 主线）` 阶段迁移补充图：`visuals/flashmla_phase_transition.svg`
- `FlashMLA（TT 主线）` 剩余优化空间补充图：`visuals/flashmla_remaining_optimization_gap.svg`
- Decode DeepSeek q_heads_per_core sweep 时延图：`multidim/visuals/decode_deepseek_q_heads_per_core_sweep_latency.svg`

## 4. 表格

- 实验参数总表：`tables/experiment_parameters.md`
- Decode 默认主表：`tables/decode_four_methods.md`
- Decode 统计口径说明：`tables/decode_best_worst_ties.md`
- Prefill 默认主表：`tables/prefill_control.md`
- `FlashMLA（TT 主线）` 补充分析表：`tables/flashmla_subanalysis.md`
- 四方法共同支持的 probe 性能报告：`supported_four_way_probe_report.md`
- Capability probe 汇总表：`tables/capability_probe_summary.md`
- Capability probe 明细表：`tables/capability_probe.md`
- Decode batch sweep：`multidim/tables/decode_batch_sweep.md`
- Decode head sweep：`multidim/tables/decode_head_sweep.md`
- Decode dim sweep：`multidim/tables/decode_dim_sweep.md`
- Decode DeepSeek q_heads_per_core sweep：`multidim/tables/decode_deepseek_q_heads_per_core_sweep.md`

## 5. 原始与过滤后结果

- 原始 JSON：`raw/part1_four_method_results.json`
- 原始 CSV：`raw/part1_four_method_results.csv`
- 过滤后 JSON：`raw/part1_four_method_results_filtered.json`
- 过滤后 CSV：`raw/part1_four_method_results_filtered.csv`
- `FlashMLA（TT 主线）` detailed profile：`flashmla_detailed/flash_mla_wh_detailed_profile_results.json`


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
- 默认主表 config：`b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，即 `B=6, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128, deepseek_q_heads_per_core=8`。

## 2. 核心结论

- `decode_256`: DeepSeek = `avg 0.194 / best 0.184 / worst 0.208 ms (18/20 kept)`，平均吞吐量 = `~30913.4 tok/s`，相对 Flash = `~1.00x`，相对 TT 主线 FlashMLA = `~0.98x`。
- `decode_1k`: DeepSeek = `avg 0.257 / best 0.250 / worst 0.274 ms (14/20 kept)`，平均吞吐量 = `~23350.3 tok/s`，相对 Flash = `~0.90x`，相对 TT 主线 FlashMLA = `~0.91x`。
- `decode_8k`: DeepSeek = `avg 0.440 / best 0.410 / worst 0.548 ms (12/20 kept)`，平均吞吐量 = `~13645.8 tok/s`，相对 Flash = `~1.09x`，相对 TT 主线 FlashMLA = `~1.06x`。
- `decode_32k`: DeepSeek = `avg 4.271 / best 0.788 / worst 11.106 ms (20/20 kept)`，平均吞吐量 = `~1404.9 tok/s`，相对 Flash = `~1.67x`，相对 TT 主线 FlashMLA = `~1.98x`。
- `decode_128k`: DeepSeek = `avg 8.866 / best 2.549 / worst 18.800 ms (20/20 kept)`，平均吞吐量 = `~676.7 tok/s`，相对 Flash = `~1.46x`，相对 TT 主线 FlashMLA = `~1.11x`。

## 3. 图表

- Decode 四方法时延图：`visuals/decode_latency_four_methods.svg`
- Decode 四方法吞吐量图：`visuals/decode_throughput_four_methods.svg`
- Decode 相对 reference 速度提升：`visuals/decode_speedup_vs_reference.svg`
- DeepSeek 相对其它 TT 方法的速度提升：`visuals/decode_deepseek_vs_tt_speedup.svg`

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

## 5. 原始与过滤后结果

- 原始 JSON：`raw/part1_four_method_results.json`
- 原始 CSV：`raw/part1_four_method_results.csv`
- 过滤后 JSON：`raw/part1_four_method_results_filtered.json`
- 过滤后 CSV：`raw/part1_four_method_results_filtered.csv`


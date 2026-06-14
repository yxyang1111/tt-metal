# Part IV 未来架构支持与展望

## 1. Part I 对齐口径

- Part I 已把 `DeepSeek FlashMLA` 固定为主方法；但当前 Part IV 的 WH→BH 推断仍建立在 `TT 主线 FlashMLA` 的 A-path detailed characterization，以及文档中的 native B-BH analytical table 上。
- long-seq 代理边界与 Part II / III 一致：`decode_4k / 16k / 32k` 的 `DeepSeek / TT` ratio 为 `0.956x / 0.973x / 0.994x`，最大偏差约 `4.4%`。
- 但 `decode_1k` 上 `DeepSeek / TT = 0.894x`，因此 short-seq 架构投影不能直接解释成 DeepSeek 的 direct future-hardware prediction。

## 2. 数据来源

- A-BH 二阶经验模型：`part1_baselines/outputs/flashmla_detailed/flash_mla_wh_detailed_profile_results.json` 中的 `bh_empirical_decode`。
- A-BH / B-BH 对照与文字结论：`flash_mla_wh_profile_and_bh_simulation.md`。
- 当前目录以 `decode_1k / 4k / 8k / 16k / 32k` 这五个代表点的 A-BH empirical threshold / plateau / coupling 结果为主；与此同时，也保留 summary markdown 中 `decode_256 -> 32k` 的 native B-BH full comparison table。

## 3. 核心结论

- current A-BH 的一阶 `dram crossover` 仍稳定在 `23c`，但二阶 `non-compute crossover` 已提前到 `decode_32k` 的 `7c`，平台区从 `17c` 开始。
- `decode_32k` 下，A-BH empirical latency = `0.3715 ms`，比 ideal first-order 仍高 `1.79x`；此时 `reader_total=0.3715 ms`，已经明显高于 `compute=0.2063 ms`。
- floor 的主要传播方向已经固定：`decode_4k` 到 `decode_32k` 都由 `writer_cb_wait_from_reader_ms` / `writer_cb_wait_from_reader_ms` 这类 reader→writer 传播式耦合主导，而不是再多给 compute cores 就能解决。
- native `B-BH` 的主瓶颈在当前五个代表点上都已经转成 `k_mcast`，并且相对 A-BH empirical 的优势会从 `decode_1k` 的 `1.99x` 持续扩大到 `decode_32k` 的 `5.60x`。
- 换句话说，Part IV 更稳妥的论文口径不是“给 A-path 多堆算力”，而是“承认 A-path 很早撞上 non-compute floor，并把未来优化重点转向更接近 native B 的拓扑与 multicast 设计”。

## 4. 论文图片

- BH latency comparison：`visuals/bh_latency_comparison.svg`
- BH crossover thresholds：`visuals/bh_crossover_thresholds.svg`
- A-BH active-core plateau：`visuals/bh_active_core_plateau.svg`
- A-BH coupling components：`visuals/bh_coupling_components.svg`

## 5. 论文表格

- A-BH 二阶经验阈值：`tables/bh_empirical_thresholds.md`
- Native B-BH vs current A-BH：`tables/bh_native_b_vs_a.md`
- A-BH coupling breakdown：`tables/bh_coupling_breakdown.md`
- A-BH active-core sweep：`tables/bh_active_core_sweep.md`

## 6. 原始快照

- 汇总 JSON：`raw/part4_architecture.json`
- Source inputs：`raw/source_inputs/`
- A-BH threshold CSV：`raw/bh_empirical_thresholds.csv`
- B-BH comparison CSV：`raw/bh_native_b_vs_a.csv`
- Active-core sweep CSV：`raw/bh_active_core_sweep.csv`


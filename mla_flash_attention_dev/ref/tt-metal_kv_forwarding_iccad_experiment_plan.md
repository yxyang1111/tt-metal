# TT-Metal KV Forwarding ICCAD 投稿实验规划

## 1. 文档目的

这份文档不是继续讨论实现细节，而是回答一个更直接的问题：

**基于当前项目的完成度，如果想围绕 TT-Metal 的 non-causal prefill KV forwarding / design space / autotuner 投稿一篇 ICCAD，应该优先做哪些实验。**

本文默认配套阅读：

- `docs/tt-metal_kv_forwarding_design_space_autotuner.md`
- `test/profile_results/sdpa_multi_core_detailed_report.md`
- `test/profile_sdpa_multi_core.py`

## 2. 先收敛论文主线

从当前仓库的资产和已有结果看，最适合的主线不是“泛化 SDPA 全景论文”，而是更聚焦的一条：

**面向 Wormhole 上 non-causal prefill 的 KV forwarding 通信优化与自动配置。**

更具体地说，论文主角应该是：

1. `non-causal prefill` 场景下的 KV forwarding
2. forwarding 的设计空间：拓扑、布局、流水线、NoC 分配、injector 调度
3. workload-aware 的配置选择方法（heuristic 或 autotuner）

不建议把下面几条同时当成主贡献：

- `causal BALANCED_Q_PARALLEL`
- `FlashDecode tree reduction`
- `多芯片 ring attention`

原因很简单：这些方向在上游技术报告或当前仓库里已经有较成熟的叙事与结果。  
真正更有新意、也更符合你当前项目积累的是：

- `NC KV forwarding` 的结构性瓶颈分析
- `forwarding topology / scheduling / pipeline` 优化
- `design-space + autotuner` 的系统化组织

## 3. 当前项目已经具备的实验基础

| 资产 | 当前状态 | 对投稿的意义 |
| --- | --- | --- |
| `test/profile_sdpa_multi_core.py` | 已有 `cp_nc_sweep`、`causal_balanced`、`noncausal_kv_chain`、`flash_decode` | 已经能支撑问题陈述、部分 baseline、部分瓶颈分析 |
| `test/profile_results/sdpa_multi_core_detailed_report.md` | 已有 4/8/16/24/32/48/56 核的 CP vs NC 结果 | 已经证明 `16` 核后 NC 瓶颈切到 `NoC forwarding / Sync` |
| `test/profile_results/README.md` | 已汇总单核 / 多核 / 双芯片结果 | 可以直接提炼论文摘要数字和初始图表 |
| `test/test_sdpa_single_core.py`、`test/test_sdpa_multi_core.py`、`test/test_sdpa_two_chips.py` | 已有 `cosine similarity`、`max abs diff` | 可复用为 correctness 验证入口 |
| `docs/tt-metal_kv_forwarding_design_space_autotuner.md` | 已明确设计空间 knob | 可以直接转成论文中的方法表和 ablation 维度 |
| `tt-metal/tests/tt_metal/tt_metal/perf_microbenchmark/` | 已有 DRAM / NoC / remote CB sync / fabric 微基准 | 适合给通信 cost model 和论文的微架构分析做支撑 |
| `TT_METAL_DEVICE_PROFILER` / Tracy / op report 能力 | 仓库内已具备工具链 | 适合做时间线和等待分解，而不只是端到端 latency |
| `experiments/iccad_exp_5_1/` | 已完成 `CP vs NC` 在 `S=1024/2048/4096`、多核数 sweep 与基础图表 | 已足够支撑问题陈述主图、chain onset 和 bottleneck 转折叙事 |
| `experiments/iccad_exp_5_2/` | 已完成 `naive / unicast / multicast / auto` 对照与基础图表 | 已验证 forwarding 收益主要来自 `multicast / auto`，`unicast` 更适合作为消融项 |

结论：  
**截至目前，`5.1 + 5.2 + 5.3` 首版已经完成，当前仓库已经足够支撑“问题陈述 + 强 baseline + primitive-level 证据”，但离一篇 ICCAD 还差 `2` 类主实验和 `1` 类兜底验证：**

1. `feature ablation`
2. `autotuner` 评估（如果要放进主标题）
3. `correctness / overhead` 兜底

### 3.1 截至目前已经可以写进论文的结论

1. `5.1` 已经证明：在 `S=1024/2048/4096` 上，`NC` 的瓶颈都在 `16` 核、`chain_len=2` 时从“compute / 调度开销”切到 `NoC forwarding / Sync`。
2. `5.2` 已经证明：`chain_len <= 2` 时 forwarding 模式收益很小，而 `chain_len >= 4` 时 `multicast / auto` 明显优于 `unicast`。
3. 因此论文里的“当前强基线”应该是 `NC-current-auto`，而不是把 `unicast` 当成最主要对照。
4. `5.3` 已经给出第一版 primitive ladder：`DRAM -> remote L1` 在 matched `QKV 2048x1536, 4-bank, BFP8` probe 上保留了约 `68.5%` 的本地带宽，而 `remote CB sync` / `CB sync + matmul` 会把 steady-state 吞吐进一步压到 `17.901 / 5.210 GB/s`。
5. `5.3` 的 `NoC latency` 也已经补齐：在当前 harvested Wormhole 上，单 sink `multicast` 延迟约为 `399 cycles`，是 `unicast` (`236 cycles`) 的 `1.69x`。
6. 因此下一阶段最值得投入时间的，不再是继续扩写 `A/B/C`，而是 `D + F`。

## 4. 投稿必须回答的 4 个问题

如果论文想成立，实验部分至少要系统回答下面 4 个问题：

### Q1. 当前 KV forwarding 的瓶颈到底是什么

不是只说“NC 比 CP 慢”，而是要证明：

- 低核数时，差距主要来自 full attention 本身 FLOPs 更高
- 中高核数时，额外损失主要来自 forwarding / sync / overlap 不足
- 真正的问题不只是 raw DRAM 带宽，而是通信协议与 injector 热点

### Q2. 你的优化为什么有效

要证明你提出的优化确实针对了核心瓶颈，而不是随机调参。

### Q3. 优化是否跨 workload 稳定

要证明收益不是只出现在一个 shape 上，而是能覆盖：

- 不同链长
- 不同序列长度
- 不同 `NQH/NKV`
- 不同布局条件

### Q4. 自动配置是否真的有价值

如果论文标题或方法里想强调 autotuner，就必须证明：

- 手工默认配置不是最优
- 不同 workload 最优配置不同
- autotuner 接近 oracle
- autotuner 成本可接受

如果做不到这点，就不要把 `autotuner` 放在标题主位置。

## 5. 必做实验矩阵

下面这 6 组实验是最核心的。  
如果**不**把 `autotuner` 放到论文主标题，那么 `A / B / C / D / F` 是**必做**，`E` 是**条件必做**；  
如果要把 `autotuner` 放到标题主位置，那么 `E` 也升级为**必做**。

### 5.1 实验 A：问题陈述与基线画像 `[已完成首版]`

**目标**：把“为什么当前 NC forwarding 值得研究”讲清楚。

**当前已完成**：

- 结果目录：`experiments/iccad_exp_5_1/results/`
- workload：`B=1, NH=8, NKV=1, D=128, q_chunk=128, k_chunk=128`
- 序列长度：`S=1024 / 2048 / 4096`
- 核数 sweep：`4 / 8 / 16 / 24 / 32 / 48 / 56`

**主要指标**：

- latency (`min/median ms`)
- achieved TFLOPS
- FPU utilization
- imbalance ratio
- `kv_chain_len`
- `total_noc_bytes`
- `injector_serial_lb_ms`
- `likely_bottleneck`

**已验证结论**：

1. 三个序列长度上，`NC` 都是在 `16` 核、`chain_len=2` 开始出现真正的 forwarding，且瓶颈同步切到 `NoC forwarding / Sync`。
2. `4 / 8` 核时 `NC > CP` 主要还是 full attention 本身更重，因此低核数差距不能直接拿来证明 forwarding 协议开销。
3. `CP` 的最优点在三个序列长度上都接近 `32` 核；而 `NC` 的最优点随着 `S` 增大向更高核数移动，符合通信开销随链长与序列长度放大的预期。

**当前结果摘要**：

| 序列长度 | 最佳 CP | 最佳 NC | forwarding 开始出现 | 结论摘要 |
| --- | --- | --- | --- | --- |
| `S=1024` | `32 cores / 0.224 ms` | `32 cores / 0.360 ms` | `16 cores / chain_len=2` | `NC/CP = 1.34~1.76x`，中高核数已明显受 forwarding 影响 |
| `S=2048` | `32 cores / 0.540 ms` | `48 cores / 0.956 ms` | `16 cores / chain_len=2` | `NC/CP = 1.52~2.05x`，随着序列变长，最优 NC 点开始右移 |
| `S=4096` | `32 cores / 1.747 ms` | `56 cores / 3.071 ms` | `16 cores / chain_len=2` | `NC/CP = 1.59~2.22x`，长序列下通信路径更值得优化 |

**论文里建议保留的图**：

1. `CP vs NC latency` 随核数变化曲线
2. `NC/CP` 比值随核数变化曲线
3. `chain length` 与 `bottleneck tag` 随核数变化图
4. `best CP / best NC latency by sequence length` 汇总图

**这组实验对 plan 的意义**：

- `A` 组已经足以支撑论文的问题陈述主线。
- 后续只建议小补一个 `S=8192` 或一个不利布局 stress case，不建议继续在 `A` 组上做大规模扩张。

### 5.2 实验 B：forwarding 对照收益 `[已完成首版]`

**目标**：定量回答“KV forwarding 本身到底值多少钱”。

这组实验原本是当前最缺、也是最关键的部分。  
现在它已经完成首版，并且已经把 baseline 体系基本定型。

**当前已完成的 baseline**：

1. `NC-naive`  
   每个核心都从 DRAM 独立读完整 K/V，不做 sharing
2. `NC-current-unicast`
3. `NC-current-multicast`
4. `NC-current-auto`  
   当前实现默认策略
5. `NC-current-hybrid`  
   暂未实现；等后续做 per-chain hybrid 时再加入

**当前已完成的 workload**：

- `chain_len = 1 / 2 / 4 / 6 / 8`
- 覆盖一个 `NKV=1` family
- 覆盖一个 `NKV=4` 的 GQA case

**主要指标**：

- latency
- NoC bytes
- speedup vs `NC-naive`
- 节省的 DRAM 重读比例

**当前结果摘要**：

| Workload | 链长 | 最优模式 | 关键观察 |
| --- | ---: | --- | --- |
| `chain1_nh8_nkv1` | `1` | `auto = 0.976 ms` | 四种模式几乎一致，`auto` 也没有启用 mcast |
| `chain2_nh8_nkv1` | `2` | `unicast = 0.544 ms` | 四种模式几乎等价，说明短链 forwarding 模式收益很小 |
| `chain4_nh8_nkv1` | `4` | `multicast = 0.348 ms` | `unicast = 0.433 ms`，比 `multicast` 慢约 `1.25x` |
| `chain6_nh6_nkv1` | `6` | `auto = 0.338 ms` | `multicast / auto` 已开始稳定优于 `naive`，收益约 `1.07x` |
| `chain8_nh7_nkv1` | `8` | `auto = 0.235 ms` | `auto` 相对 `naive` 约 `1.28x`，`multicast` 相对 `unicast` 约 `1.29x` |
| `gqa_chain4_nh8_nkv4` | `4` | `multicast = 0.332 ms` | GQA 下依然明显偏向 mcast，`unicast` 比 `multicast` 慢约 `1.27x` |

**已验证结论**：

1. `chain_len <= 2` 时 forwarding 模式收益很小，这类 case 不应该主导论文的实验叙事。
2. `unicast` 在 `chain_len >= 4` 时已经不是强基线，更适合作为通信策略消融，而不是论文主表里的“current best baseline”。
3. `multicast / auto` 才是当前 forwarding 路径的主要收益来源；在本轮 `chain > 1` 的测试中，`auto` 都启用了 mcast，并且通常接近或达到最优。
4. 因此后续 `hybrid` 的价值应被表述为：在 `mcast` 不可用、尾部不均匀或布局不利时，尽量逼近或超过 `multicast / auto`，而不是只拿来打 `naive`。

**论文里建议保留的图**：

1. `naive vs unicast vs multicast vs auto` 延迟柱状图
2. speedup vs `NC-naive` 图
3. 一张 DRAM 节省与 NoC 增长的 trade-off 示意或表格

**这组实验对 plan 的意义**：

- `B` 组已经把论文的 baseline hierarchy 讲清楚了。
- 后续主表应以 `NC-naive vs NC-current-auto vs proposed` 为主，`unicast` 进入消融表。

### 5.3 实验 C：primitive-level 通信微基准 `[已完成首版]`

**目标**：让你的 cost model 和瓶颈归因有“硬证据”，而不是纯猜测。

**当前已完成**：

- 结果目录：`experiments/iccad_exp_5_3/results/`
- 汇总文件：`experiments/iccad_exp_5_3/results/exp_5_3_primitive_microbenchmarks.json` 与 `experiments/iccad_exp_5_3/results/exp_5_3_primitive_microbenchmarks.md`
- 运行脚本：`experiments/iccad_exp_5_3/run_exp_5_3_primitive_microbenchmarks.py`
- 画图脚本：`experiments/iccad_exp_5_3/plot_exp_5_3_simple.py`
- 当前图表：`experiments/iccad_exp_5_3/results/figures/exp_5_3_primitive_throughput.png`、`experiments/iccad_exp_5_3/results/figures/exp_5_3_throughput_retention.png` 与 `experiments/iccad_exp_5_3/results/figures/exp_5_3_noc_latency.png`
- 当前设备是 harvested Wormhole；其中 `3` 条 legacy throughput microbenchmark 仍能产出 steady-state timing sample，但不会报告 `Test Passed`，因此统一标记为 `probe_only`
- `NoC latency` benchmark 原始版本在 `TT_METAL_DEVICE_PROFILER_DISPATCH=1` 下会触发 `dispatch_core_exhaustion`；当前版本改为自动选择最近的 worker-to-worker pair，并仅开启 device profiler，这也更贴近 forwarding 数据路径
- 当前机器上实际测得的 latency 路径为：worker logical `(0, 1) -> (0, 0)`，physical `(18, 19) -> (18, 18)`

**当前结果摘要**：

| Primitive | 配置 | 状态 | 稳态带宽/延迟 | 关键观察 |
| --- | --- | --- | --- | --- |
| `DRAM -> local L1` | `QKV 2048x1536, 4-bank, BFP8` | `probe_only (test_failed)` | `44.700 GB/s` | 作为本轮 local baseline |
| `DRAM -> remote L1` | `QKV 2048x1536, 4-bank, BFP8` | `probe_only (validation_failed)` | `30.638 GB/s` | 相对 local 保留 `68.5%` 吞吐 |
| `DRAM -> remote CB sync` | `single receiver, 32768x128, BFP8` | `probe_only (validation_failed)` | `17.901 GB/s` | 相对 local 保留 `40.0%` |
| `remote CB sync + matmul` | `single receiver, 32x2048x128, BFP8` | `passed` | `5.210 GB/s` | 为 sync-only 路径的 `29.1%` |
| `NoC unicast vs multicast latency` | `single sink` | `passed` | `unicast = 236 cycles`, `multicast = 399 cycles` | `multicast / unicast = 1.69x` |

**已验证结论**：

1. 在 matched `QKV 2048x1536, 4-bank, BFP8` probe 上，`DRAM -> remote L1` 相对 `DRAM -> local L1` 仍会损失约 `31.5%`，说明 forwarding 的第一跳远端写入并不“免费”，但它依然不是最重的后续阶段。
2. 真正明显掉速出现在 `remote CB sync` 以及 consumer-coupled 路径：steady-state 吞吐从 `44.700` 下降到 `30.638 -> 17.901 -> 5.210 GB/s`，和 `5.1` 里观察到的 `NoC forwarding / Sync` 瓶颈转折是一致的。
3. 单 sink worker-to-worker `NoC` latency 上，`multicast` 为 `unicast` 的 `1.69x`；连续 `3` 次单独复跑时 `unicast = 236~240 cycles`、`multicast = 397~399 cycles`，说明这条 latency probe 在修复后已经较稳定。
4. `dram_to_l1_qkv_bfp8_4banks`、`dram_to_remote_l1_qkv_bfp8_4banks`、`dram_to_remote_cb_sync_single_receiver_bfp8` 这三条结果应该被当成 `cost-model probe`，而不是 correctness claim；论文文字里需要显式说明这一点。

**论文里应形成的图**：

1. `primitive throughput ladder`：`exp_5_3_primitive_throughput.png`
2. `throughput retention vs local DRAM -> L1`：`exp_5_3_throughput_retention.png`
3. `unicast / multicast latency` 对比：`exp_5_3_noc_latency.png`

这组实验的作用不是单独拿高分，而是支撑：

- 为什么 pipeline 有用
- 为什么 multicast / tree / dual NoC 值得做
- 为什么 injector hotspot 与 sync stall 不是拍脑袋猜的

### 5.4 实验 D：优化项消融

**目标**：证明你的每个设计点都在解决明确问题。

建议按“从保守到激进”的顺序做增量式 ablation，而不是所有 feature 一次性打开。

**推荐 ablation 梯子**：

1. `B0 = NC-current-auto`  
   也就是当前默认策略，实测大多数有 sharing 价值的 case 会走到 mcast
2. `B1 = B0 + per-chain hybrid`
3. `B2 = B1 + layout-aware mapping`
4. `B3 = B2 + pipelined read/forward`
5. `B4 = B3 + dual NoC` 
6. `B5 = B4 + rotating injector`
7. `B6 = B5 + tree forwarding`

如果时间有限，我建议优先完成：

- `per-chain hybrid`
- `layout-aware mapping`
- `pipelined read/forward`
- `dual NoC`

`rotating injector` 和 `tree forwarding` 可以作为强化版或后续工作。

**主要指标**：

- speedup vs `NC-current`
- speedup vs `NC-naive`
- FPU utilization
- DRAM utilization
- NoC bytes
- `injector stall`
- `receiver idle`
- overlap ratio（如果 profiler 能给出）

**建议 workload**：

- 短链：`chain_len <= 2`  
  只保留 `1` 个 sanity case 即可
- 中链：`chain_len = 3~4`
- 长链：`chain_len >= 6`
- 再加一个“不利布局 / mcast 不可用”的 stress case

**论文里应形成的图**：

1. 累积消融柱状图
2. 按链长分组的 speedup 图
3. 一个 case study 的 timeline/profiler 图

### 5.5 实验 E：autotuner 评估

**目标**：证明“设计空间搜索”不是可有可无的包装，而是真的带来接近最优的配置。

如果你打算把 autotuner 作为论文卖点，这组实验必须做完整。

**建议对比**：

1. `expert default`  
   当前手工默认配置；首轮可以直接把 `current auto` 当成 expert baseline
2. `single-rule heuristic`  
   例如“`chain_len <= 2` 沿用 current auto；`chain_len >= 4` 且可 mcast 时优先 multicast / hybrid，否则 fallback unicast”
3. `oracle / exhaustive on reduced space`
4. `your autotuner`

**主要指标**：

- latency speedup vs `expert default`
- `oracle gap`
- 搜索时间
- 编译次数 / runtime-only 切换次数
- cache hit ratio

**建议 workload 集合**：

- 至少 `20~50` 个不同 workload
- 覆盖 `S`、`D`、`NQH/NKV`、core grid、layout 条件

**实验重点**：

- 一个 reduced design space 上做 exhaustive，给出真实最优作 oracle
- autotuner 在 full space 上运行，报告与 oracle 的差距
- 冷启动与热启动都要测
- 基于 `5.2` 先把搜索空间收窄：在 `mcast`-eligible 且 `chain_len >= 4` 的 case 里，`unicast` 只保留为对照，不必作为高优先候选

**论文里应形成的图**：

1. autotuner 与 expert/oracle 的 CDF 或 scatter 图
2. 搜索时间与收益的 trade-off 图

结论很明确：  
**如果没有 oracle gap 或 search overhead 数据，就不要把 autotuner 放成论文核心贡献。**

### 5.6 实验 F：正确性、稳定性与鲁棒性 `[已完成]`

**目标**：验证 `expert_default` 与 `5.5 full_space_best / proposed` 在代表 workload 上的数值正确性、多 seed 稳定性，以及 compile/runtime/autotuner overhead 是否足够作为论文里的 correctness/robustness 兜底证据。

**当前已完成**：

- 结果目录：`experiments/iccad_exp_5_6/results/`
- 运行脚本：`experiments/iccad_exp_5_6/run_exp_5_6_correctness_robustness.py`
- 画图脚本：`experiments/iccad_exp_5_6/plot_exp_5_6_simple.py`
- 结果摘要：`experiments/iccad_exp_5_6/results/exp_5_6_correctness_robustness.json` 与 `.md`
- 图表：`experiments/iccad_exp_5_6/results/figures/exp_5_6_min_pcc.png`、`exp_5_6_max_abs_diff.png`、`exp_5_6_runtime_stability.png` 与 `exp_5_6_overheads.png`
- 代表 workload 直接取自 `5.5` summary：`short_s1024_nh8_nkv1`、`mid_wrap_s1536_nh4_nkv1`、`long_s1536_nh6_nkv1` 与 `stress_s3072_nh8_nkv1`
- 对比策略：`expert_default` 与 `proposed`（即每个 workload 在 `5.5` 中的 `full_space_best`）
- 随机种子：`0 / 7 / 17 / 29 / 53`
- dtype：`BF16 / BFP8`
- 共完成 `16` 个 `workload × policy × dtype` 组合、`80` 个 seed 级验证

**当前结果摘要**：

| Workload | `5.5 proposed` | `BF16 hot ms (expert/proposed)` | `BFP8 hot ms (expert/proposed)` | worst PCC | worst max abs diff | standalone autotuner cold-start |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `short_s1024_nh8_nkv1` | `hybrid_baseline` | `0.566 / 0.535` | `0.523 / 0.534` | `0.999518` | `0.715080` | `124233.125 ms` |
| `mid_wrap_s1536_nh4_nkv1` | `hybrid_baseline` | `0.506 / 0.444` | `0.427 / 0.433` | `0.999529` | `0.471070` | `46946.067 ms` |
| `long_s1536_nh6_nkv1` | `unicast_baseline` | `0.525 / 0.527` | `0.436 / 0.453` | `0.999423` | `0.763535` | `49455.358 ms` |
| `stress_s3072_nh8_nkv1` | `hybrid_baseline` | `2.249 / 2.269` | `1.421 / 1.430` | `0.999934` | `1.379429` | `14768.208 ms` |

**已验证结论**：

1. 所有 `16` 个组合、共 `80` 个 seed run 都通过了 `PCC` 与 relaxed `allclose`；全局最差 `cosine similarity = 0.999346`、`PCC = 0.999423`。
2. `max abs diff` 的最坏点出现在 `stress_s3072_nh8_nkv1, BFP8`，为 `1.379429`；但该点的 `PCC` 仍为 `0.999934`，说明输出仍高度一致。
3. 以每个 seed 的 `min_ms` 作为 steady-state runtime 指标时，seed-to-seed `CV` 仅为 `0.44% ~ 3.90%`；说明 kernel steady state 基本稳定，少量 `median_ms` outlier 更像 host-side 抖动而不是算法不稳。
4. `runtime metadata overhead` 仅 `4.6 ~ 6.0 us`，相对 `0.427 ~ 2.269 ms` 的 hot runtime 可以视为可忽略。
5. `compile-time overhead` 对 `BF16` 大致为 `243.295 ~ 439.236 ms`，对 `BFP8` 为 `223.759 ms ~ 2860.195 ms`；这更支持“compile cost 需要摊销”的论文表述，而不支持 per-request compile 的在线叙事。
6. 基于 `5.5` candidate 表重放得到的 standalone `autotuner cold-start overhead` 为 `14.8 ~ 124.2 s`（`1 ~ 2` eval），远高于 steady-state kernel latency；因此 autotuner 只能被表述为 offline/cached 机制，而不应被表述为在线请求时实时搜索。
7. 这组实验里的 hot runtime 对比说明：`proposed` 在 `BF16` 的 `short / mid-wrap` 上能复现 `1.06x / 1.14x` 的改善，但在 `long / stress` 上基本持平；`BFP8` 下则整体更接近持平甚至轻微回退（4 个代表 workload 上 geomean `0.981x`）。因此 `5.6` 不应被当成额外的性能主证据，而应作为 correctness/overhead 兜底表。

**作为论文里的 correctness / robustness 兜底实验是否足够**：

我认为**足够**，理由是：

- 已覆盖 `short / mid-wrap / long / stress` 四类代表 workload
- 已覆盖 `BF16 / BFP8` 两种 dtype
- 已覆盖多 seed，并同时比较 `expert_default` 与 `5.5 full_space_best / proposed`
- 已同时给出 correctness、compile overhead、runtime metadata overhead 与 autotuner cold-start overhead

但它的边界也要写清楚：

- 它足以支撑“优化没有破坏数值结果，而且 steady-state 行为稳定”
- 它不足以支撑“autotuner 可在线低开销运行”
- 它也不足以额外强化“`BFP8` 下 proposed 持续带来速度收益”这一点

## 6. 建议的 workload 矩阵

为了避免实验只在一个 shape 上成立，建议至少准备下面 4 组 workload family。

| Family | 目的 | 建议参数 |
| --- | --- | --- |
| `F1: 当前主形状` | 复用现有结果、快速迭代 | `B=1, NH=8, NKV=1, S=1024, D=128` |
| `F2: 更长序列` | 放大 chain / sync / overlap 问题 | `S=2048/4096/8192`, `D=128` |
| `F3: GQA/MQA 变化` | 观察 `NKV` 变化对 sharing 价值的影响 | `NKV=1/4`, `NH=8/16` |
| `F4: 不利布局/尾部` | 验证 hybrid/layout/tail policy 的价值 | 非同行、非完美矩形、`q_chunk_count` 不均匀 |

截至目前：

- `5.1` 已覆盖 `F1 + F2`
- `5.2` 已覆盖 `F1 + F3`
- 真正还缺的是 `F4`

建议所有 family 都至少 sweep 一组核数：

- `4, 8, 16, 24, 32, 48, 56/64`

## 7. 建议的 baseline 体系

论文中的 baseline 最好分成三层，而不是只做 `CP vs NC`。

### 7.1 功能对照层

- `CP`
- `NC-naive`
- `NC-current-auto`
- `NC-proposed`

### 7.2 通信策略层

- `unicast`（弱基线 / 消融）
- `multicast`
- `per-chain hybrid`
- `tree`（如果实现）

根据 `5.2` 的结果，`unicast` 不应作为论文主表里的 strongest baseline；  
论文主 baseline 应该是 `NC-current-auto`。

### 7.3 配置策略层

- `expert default`
- `heuristic`
- `autotuner`

这样论文结构会更清楚：

- 第一层说明“为什么需要 forwarding”
- 第二层说明“怎样的 forwarding 更好”
- 第三层说明“为什么需要自动配置”

## 8. 我建议你最后在论文里放出的图表

如果实验完整，最终论文里的主图表建议控制在下面这几类：

1. `5.1`: `CP vs NC` 的问题陈述图
2. `5.2`: `NC-naive vs unicast vs multicast vs auto` 的收益图
3. primitive 微基准图
4. feature ablation 图（起点为 `NC-current-auto`）
5. autotuner vs expert/oracle 图（如果保留 autotuner）
6. correctness + overhead 表

可选加分项：

1. 一个 device profiler / NoC timeline case study
2. 一个 stress-case 的布局敏感性图
3. 一个 decode 或 multi-chip 的“迁移启发”讨论图

## 9. 当前项目下的优先级建议

如果目标是**尽快做出一版有投稿可能性的结果**，我建议按下面顺序推进：

1. 先把 `5.1 + 5.2 + 5.3` 固化成论文版图表和文字  
   当前已经具备问题陈述、baseline 体系和 primitive-level 证据。
2. 直接做 `D`  
   以 `NC-current-auto` 为起点做 feature ablation，重点打 `chain_len >= 4` 和 `mcast` 不可用 stress case。
3. 补 `F`  
   把 correctness、compile/runtime overhead 和多 seed 稳定性补齐。
4. 最后决定论文是否要把 `autotuner` 放进主标题  
   如果 `oracle gap` 和 `search cost` 还做不扎实，就把 autotuner 降级成 future work 或次要章节。

## 10. 最低可投稿版本

如果时间紧，我认为在 `5.1 + 5.2 + 5.3` 已完成首版的前提下，最低可投稿闭环还应满足：

1. 有 `NC-naive`、`NC-current-auto`、`proposed` 三层对照
2. 有至少 `2~3` 个真正实现的优化点，而不是只有设计文档
3. 有跨多个 workload family 的稳定收益  
   这一点现在已经有初步基础，但还需要 proposed 版本复现
4. 有 primitive-level 证据支撑瓶颈归因
5. 有 correctness 与 overhead 结果

如果还想把 `autotuner` 写成主贡献，则还必须额外满足：

1. reduced-space oracle
2. autotuner vs expert 对比
3. search overhead / cache hit / compile-time 结果

## 11. 一句话结论

基于当前项目，**最值得投稿 ICCAD 的方向是：把 non-causal prefill 的 KV forwarding 从“一个已有的链式/multicast 优化”提升成“一个可分析、可消融、可自动配置的通信子系统”。**

对应到实验上，最应该优先补的是：

1. feature ablation
2. correctness / overhead
3. autotuner 的 oracle-gap 与 overhead 评估（如果要把 autotuner 放进主标题）

其中 `5.1 + 5.2 + 5.3` 已经完成首版，所以当前最有价值的下一步不再是继续补 baseline，而是把“怎么进一步提升”和“能否稳定复现”这两层证据补扎实。

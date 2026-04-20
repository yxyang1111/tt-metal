# SF-MLA 论文实验计划

## 1. 文档目标

本文档基于当前 `latex/main.tex` 中已经收敛出的实验章节结构，以及
`mla_flash_attention_dev/docs/` 和 `mla_flash_attention_dev/experiments/`
目录下现有的实验文档、runner、profile 结果与 autotuner 代码，整理出一份新的论文实验计划。

这份计划有两个目的：

1. 让论文实验主线与当前章节结构完全对齐。
2. 明确区分“现在就能直接支撑论文的资产”和“需要继续补实现后才能进入主文的增强实验”。

核心判断如下：

- 以当前资产来看，论文最稳妥的 Part I 不再是单层“`decode` 四方法 + `prefill` 三方法”的待建设状态，而应拆成两个层次：
  1. `strict four-way` 公共支持域 benchmark：`decode` 四方法公平对比 + `prefill` 三方法控制对照；
  2. Wormhole `4c/8c` 专项：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`FlashMLA-4c`、`FlashMLA-8c` 的五方法 `decode` 对比。
- 公共支持域的四方法路径继续回答“同一 attention family 内 flash-style 实现带来了多少收益”和“generic MLA 与 DeepSeek 路径在公平公共空间里差多少”。
- Wormhole `4c/8c` 五方法路径则进一步回答“同一 MLA 语义下，TT 主线、4-core S-block 和 8-core S-block 在不同 `B/H/seq_len` 压力下分别谁更优、为什么更优”。
- 当前 `DeepSeek FlashMLA` 在 Part I 中已经不只是单个方法名，而是需要区分成：
  公平四方法里的 `DeepSeek FlashMLA` 公共支持域口径，以及 Wormhole 专项里的 `FlashMLA-4c / FlashMLA-8c` 两种几何。
- `DeepSeek FlashMLA` 的阶段迁移，以及 generic MLA 的利用率归因、cost model / tuner 和未来架构含义，继续分别放到 Part II / III / IV 讲扎实。
- 但需要明确：当前 `flash_mla_wh_detailed` 一类 detailed assets 主要覆盖 `FlashMLA（TT 主线）`；因此 Part II / III / IV 当前应统一按“`DeepSeek FlashMLA` 主叙事 + `TT 主线 FlashMLA` long-seq proxy”来写。更理想的增强版结果仍然是补一组 `DeepSeek FlashMLA` representative-point decode profile，用来进一步收紧 proxy 边界。
- 如果后续 `B0 / B1 / B2 / B3` 这条 FlashMLA 内部优化链真正落地并有稳定 benchmark harness，再把它作为增强版结果加入附录或扩展章节，而不是当前版本成立的前提。


## 2. 论文实验主线

当前论文实验部分已经收敛为四个问题，新的实验计划也完全按这四条主线组织：

| Part | 论文问题 | 当前章节位置 | 主文要回答的核心问题 |
|---|---|---|---|
| Part I | 公平四方法 decode / prefill 控制基线，以及 Wormhole `4c/8c` 五方法 decode 专项在当前硬件上能跑多快 | `Evaluation on Current Hardware` -> `Performance Characterization on Wormhole` | 公共支持域里四方法的性能边界如何；Wormhole 上 `FlashMLA-4c / FlashMLA-8c` 相对 `FlashMLA（TT 主线）` 的收益、代价和适用区间是什么；`prefill` 三方法控制对照给出什么结论 |
| Part II | 利用率与瓶颈归因 | `Microarchitectural  Utilization and Bottleneck Analysis` | reader / writer / compute 谁在贴 critical path，stall 到底来自哪里 |
| Part III | tuner 与 cost model 是否有效 | `Cost Model Fidelity and Auto-Tuner Effectiveness` | 二阶模型是否足够准确，autotuner 能否稳定选到高质量 mapping |
| Part IV | 对未来架构意味着什么 | `Architectural Implications Beyond Wormhole` | SRAM、TOPS、core granularity、MLA vs MHA、Blackhole projection 的结论是什么 |


## 3. 当前可直接复用的资产

### 3.1 代码 / runner

| 路径 / 组件 | 作用 | 可直接支撑的 Part | 当前状态 |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp` | 标准 attention / Flash Attention prefill 主 API，覆盖 `scaled_dot_product_attention` 与 chunked 变体 | Part I | 已可直接复用 |
| `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp` | 标准 attention / Flash Attention / `FlashMLA（TT 主线）` decode 主 API，覆盖 decode 与 paged decode 变体 | Part I | 已可直接复用 |
| `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` | `DeepSeek FlashMLA` decode 主实现，包含 `FlashMLADecode.op`、`FlashMLAProgramConfig` 与 WH `4c/8c` grid 定义；当前 Part I benchmark 通过一次性 tensor adaptation 后计时 backend device op | Part I / II | 已可直接复用，是当前公平四方法与 Wormhole `4c/8c` 专项的核心实现入口 |
| `models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla_wh.py` | `DeepSeek FlashMLA` 在 Wormhole 上的张量 shape、grid、DRAM shard 与 `cur_pos_tensor` 样例 | Part I / II | 已可直接复用，可改造成 benchmark / profiler 输入模板 |
| `experiments/profile_flash_mla_wh_detailed.py` | `FlashMLA（TT 主线）` 在 Wormhole 上的 decode / prefill detailed characterization，并附带 A-path Blackhole empirical coupled simulation | Part I / II / IV | 已可直接复用，但当前主要覆盖 TT 主线 MLA，不是 DeepSeek 主方法的直接证据 |
| `experiments/profile_flash_mla_wh.py` | 更轻量的 `FlashMLA（TT 主线）` profile runner | Part I | 可作为 quick rerun |
| `experiments/run_flash_mla_pm_bubble_probe.py` | `FlashMLA（TT 主线）` decode / prefill PM util 与 compute bubble probe rerun | Part II | 已可直接复用，但 decode direct PM 仍有口径限制 |
| `experiments/run_flash_mla_wh_smoke.py` | `FlashMLA（TT 主线）` / `DeepSeek FlashMLA` 的 Wormhole smoke 与最小输入验证入口 | 全部 | 已可直接复用，且已扩展支持 WH `4c/8c` smoke |
| `experiments/part1_three_baselines/run_part1_benchmarks.py` | 当前统一 Part I runner，支持 `strict_four_way` decode、公平切片、optional capability probe、三方法 prefill control，以及通过 `--deepseek-wh-cores-per-block=4/8` 生成 Wormhole `4c/8c` 专项 rerun | Part I | 已落地，是当前 Part I 主入口 |
| `experiments/part1_three_baselines/render_part1_results.py` | 把 Part I raw JSON/CSV 渲染成 report、fair 四方法表格、slice summary 与 capability probe 汇总 | Part I | 已落地 |
| `DeepSeek FlashMLA decode profiler runner（待补齐）` | 对主方法做 representative-point decode profiling，服务 Part II 归因 | Part II | 当前需要补充 |
| `autotuner/basic_autotuner.py` | offline analytical autotuner，支持 preset、measurement DB、rerank、calibration | Part III / IV | 已可直接复用 |
| `autotuner/profile_measurement_bridge.py` | 把 WH profile JSON 映射成 autotuner measurement DB | Part III | 已可直接复用 |

### 3.2 已有结果 / 说明文档

| 路径 | 作用 | 备注 |
|---|---|---|
| `docs/tt_attention_implementation_summary.md` | 梳理 `reference attention`、`Flash Attention`、`FlashMLA（TT 主线）` 的 API 与能力边界 | Part I 的公共基线定义来源 |
| `experiments/part1_three_baselines/outputs/report.md` | 当前公平四方法 + 三方法 prefill 的总报告 | Part I 公共支持域 benchmark 的主入口 |
| `experiments/part1_three_baselines/outputs/tables/fair_four_method_all_cases.md` | 公共支持域 decode 四方法全量结果表 | Part I 公平四方法主表 |
| `experiments/part1_three_baselines/outputs/tables/prefill_control.md` | 三方法 prefill 控制表 | Part I prefill 控制对照 |
| `experiments/part1_three_baselines/outputs/tables/wh_five_method_all_cases.md` | Wormhole `reference / Flash / FlashMLA（TT 主线） / FlashMLA-4c / FlashMLA-8c` 五方法 decode 汇总表 | Part I 五方法专项主表；当前是多次专项 rerun 合并后的 selected-case 汇总，不是 full-factorial sweep |
| `experiments/part1_three_baselines/outputs/tables/wh_4c_vs_8c_batch_summary.md`、`wh_4c_vs_8c_head_summary.md`、`wh_4c_vs_8c_seq_summary.md`、`wh_4c_vs_8c_qshard4_seq_summary.md` | Wormhole `4c/8c` 在 `batch / head / seq` 轴上的专项对照表 | Part I `4c/8c` 局部规律与解释依据 |
| `experiments/part1_three_baselines/five-method-dataflow-analysis.md` | 五方法数据形态、reader/compute/writer 路径、4c/8c 几何差异与中间结果流向分析 | Part I 五方法解释边界与附录材料 |
| `experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_report.md` | 当前最完整的 Wormhole `FlashMLA（TT 主线）` decode/prefill sweep 结果 | Part I 中 TT 主线 MLA 子分析 / Part II 的机理数据源 |
| `experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_results.json` | detailed profile 原始结构化结果 | tuner bridge / calibration 的输入 |
| `experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_measurement_db.json` | 现成 measurement DB | 可直接给 autotuner rerank |
| `experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_calibration.json` | 现成 calibration | 可直接给 autotuner ranking |
| `experiments/flash_mla_wh_profile_and_bh_simulation.md` | 现有 `FlashMLA（TT 主线）` WH + BH 实验与结论总整理 | 主要服务于 TT 主线 MLA 子分析与 Part IV |
| `experiments/flash_mla_pm_and_bubble_min_rerun_plan.md` | decode PM/bubble 最小 rerun 方案 | Part II 的操作手册 |
| `docs/experimental-flash-mla-optimization-implementation-plan.md` | 后续 FlashMLA 内部优化链与实现建议 | 不再是 Part I 主对比的前提，更适合作为增强版输入 |
| `docs/flash-mla-wh-component-utilization-and-bubble-analysis.md` | decode / prefill 利用率与 bubble 口径整理 | Part II 的解释口径来源 |
| `experiments/part2_utilization_bottleneck_analysis/` | 已固化的 Part II 汇总目录，含 proxy 对齐表/图、阶段迁移、source attribution、dashboard | 已可直接引用；当前口径是 `DeepSeek` 主叙事 + `TT 主线` long-seq mechanism proxy |
| `experiments/part3_cost_model_autotuner/` | 已固化的 Part III 汇总目录，含 fidelity、top-1、exact-case、search runtime | 已可直接引用；当前口径是 `generic MLA backend planning proxy` |
| `experiments/part4_architecture_implications/` | 已固化的 Part IV 汇总目录，含 A-BH empirical、active-core plateau、native B-BH 对照 | 已可直接引用；当前口径是 `generic MLA backend architecture proxy` |


## 4. 实验口径冻结

### 4.1 主设备与主工作负载

- 当前硬件主平台：`Tenstorrent Wormhole N300`
- 未来架构投影：`Blackhole`
- Part I 当前分成两层方法集合：
  - 公共支持域 `strict four-way`：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`DeepSeek FlashMLA`
  - Wormhole 专项 `five-method decode`：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`FlashMLA-4c`、`FlashMLA-8c`
- 公共支持域默认 config 已经落地为：
  `b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8`，对应 `decode=256..128k`、`prefill=1k/4k`，当前 fair 四方法结果使用 `align_with_tt_mainline` 作为 DeepSeek 并行度策略。
- Wormhole `4c/8c` 专项当前主要冻结：
  `H_kv=1`、`value_dim=512`、`rope_dim=64`、`block_size=64`、`k_chunk_size=128`，
  并通过 `--deepseek-wh-cores-per-block=4/8` 对比两种 S-block 宽度；现有五方法表的专项 rerun 以 `dqhpc=8` 为主。
- `reference attention` 与 `Flash Attention` 使用其主线 API 的 canonical attention shape。
- `FlashMLA（TT 主线）` 与 `DeepSeek/4c/8c` 应冻结相同的逻辑 MLA 问题规模（`batch`、`seq_len`、`current position`、`latent rank`、`RoPE dim`），但允许各自使用不同的 memory layout / output buffer / core-grid。
- 当前 Part I 里 benchmarked 的 `DeepSeek FlashMLA` 路径采用“先做一次 DeepSeek -> builtin tensor adaptation，再只计 backend device op”的口径；论文里应把它明确写成 `backend-op-only` 对比，而不是 Python end-to-end 包装层时间。
- `DeepSeek FlashMLA / FlashMLA-4c / FlashMLA-8c` 当前仅纳入 `decode` 主对比，不进入 `prefill` control 表，除非后续补出独立 prefill 路径。

### 4.2 主 workload 矩阵

Part I 的主对比现在应按“公共支持域 fair benchmark + Wormhole `4c/8c` 专项 rerun”组织为：

- 公共支持域 `decode` 主 sweep（四方法）：
  `256 / 512 / 1k / 2k / 4k / 8k / 16k / 32k / 64k / 128k`
- `prefill` control sweep（三方法，仅 `reference attention / Flash Attention / FlashMLA（TT 主线）`）：
  `1k / 4k`
- 公共支持域补充切片：
  固定 `seq_len=8k` 的 `batch`、`head`、`value_dim` 切片，以及可选的 `strict four-way` capability probe。
- Wormhole `4c/8c` 五方法专项：
  当前已形成 `batch` sweep（`seq_len=8k,H=32,q_shards=4`）、`head` sweep（`seq_len=8k,B=8,dqhpc=8`）、
  `seq` sweep（`B=8,H=24,q_shards=3` 和 `B=6,H=32,q_shards=4`）四组专项结果；五方法总表只保留这些 rerun 中的 selected cases，不应误读成 full-factorial 主 sweep。
- `flash_mla_wh_detailed` 这套 `FlashMLA（TT 主线）` dense sweep 可以继续作为机理 proxy，但不应直接替代主方法或 `4c/8c` 专项证据。

### 4.3 当前 batch 口径

- 公共支持域的 `decode` 主曲线仍可固定 `batch=1`，让 `sequence length` 成为主要自变量；这也是当前 fair 四方法默认主表的口径。
- 但当前 Part I 已经不仅是固定 `batch=1` 的单轴实验：
  还包括固定 `seq_len=8k` 的 `batch / head / value_dim` 公平切片，以及 Wormhole `4c/8c` 的 `batch / head / seq` 专项 rerun。
- 因而新的计划里不应再把“统一固定 `batch=1`”写成整个 Part I 的唯一口径，而应写成“公共支持域主曲线的默认口径”。
- 对 `FlashMLA-4c / FlashMLA-8c` 而言，是否支持某个点不只取决于外层 `batch`，而取决于 `required_q_cores = batch * q_shards`；因此 `B` 轴和 `q_shards` 轴需要联合解释。
- 现有 `profile_flash_mla_wh_detailed.py` 的 mixed-batch canonical case 仍然更适合服务 `FlashMLA（TT 主线）` 的内部 characterization 和 Part II 机理 proxy，不应直接替代当前 Part I 的 fair benchmark 或五方法专项表。
- 如果要让 Part II 真正解释主方法，则仍建议补一套 `DeepSeek FlashMLA` 的 representative-point direct profiler。

### 4.4 固定不变的配置

为了保证不同实验和不同 baseline 之间可比，以下项应冻结：

- 随机种子
- `decode` / `prefill` 的 `seq_len` 与 `batch`
- page table
- tensor layout / memory placement
- `math_fidelity`
- `exp_approx_mode`
- `block_size`
- `k_chunk_size`
- 对同一 attention family，复用完全相同的 `Q/K/V` 或 `Q/KV cache` 输入

### 4.5 当前可直接用、不可直接用的指标

当前可以直接进入 Part I 主图 / 主表的：

- 公共支持域四方法表中的过滤后 `avg / best / worst latency`
- Wormhole 五方法专项表中的 `avg / best / worst latency`
- throughput（`tok/s`）
- 相对 `reference attention` 的 normalized speedup（公共支持域）
- `DeepSeek FlashMLA` 相对 `Flash Attention`、`FlashMLA（TT 主线）` 的 latency delta 或 ratio（公共支持域）
- `FlashMLA-4c / FlashMLA-8c` 相对 `FlashMLA（TT 主线）` 的 latency / throughput 胜负与 `unsupported` 边界（Wormhole 专项）

当前更适合作为 `FlashMLA（主要是 TT 主线）` 子分析或 Part II / III 证据的：

- NCRISC / BRISC / TRISC 线程窗口占比
- decode reader stage breakdown
- decode writer stage breakdown
- decode compute bubble counters
- prefill direct PM FPU util
- first-order vs second-order model 误差
- autotuner top-1 vs empirically best

当前不建议作为主文核心证据的：

- decode direct `PM IDEAL / PM COMPUTE / PM BANDWIDTH / PM FPU UTIL`
- decode `NOC / MULTICAST NOC / DRAM util`
- 当前不稳定的 direct NoC attribution

解释口径上还需要明确一点：

- `reference attention` vs `Flash Attention` 可以解释为同一 attention family 内的实现收益。
- `FlashMLA（TT 主线）` / `DeepSeek FlashMLA` vs 前两者属于跨 attention family 的 system-level 对比，不应写成严格 algorithm-equivalent speedup。
- `DeepSeek FlashMLA` vs `FlashMLA（TT 主线）` 更接近同一 MLA 语义下的实现比较，但由于 grid、layout、output buffer 和 sharding 假设不同，仍应写成 system-level implementation comparison，而不是纯 kernel-only speedup。
- `FlashMLA-4c` vs `FlashMLA-8c` 属于同一路径不同 S-block 几何的对比；当前 benchmark 口径仍是一次性 adapter 后的 backend-op-only 测量，因此解释重点应放在 mapping / capacity / topology 影响，而不是 Python 包装层开销。


## 5. Part I：公平四方法基线 + Wormhole 五方法 4c/8c decode 专项

### 5.1 目标

回答五个问题：

1. 在公共支持域里，`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`DeepSeek FlashMLA` 的 `decode` 性能边界分别是什么；`prefill` 三方法控制对照呈现什么关系。
2. 同一 attention family 内，`Flash Attention` 相比 reference / 非 fused attention 带来了多少收益。
3. 同为 MLA 语义时，`DeepSeek FlashMLA` 相比 `FlashMLA（TT 主线）` 在公平公共空间里带来了多少系统级收益与代价。
4. 在 Wormhole 上把 `DeepSeek` 路线拆成 `FlashMLA-4c / FlashMLA-8c` 后，不同 `B/H/seq_len` 压力下谁更优、谁的支持边界更宽。
5. 哪些 `4c/8c` 现象应被解释为 mapping / capacity / topology 效应，哪些现象需要在 Part II 中继续用 dataflow / utilization 证据解释。

### 5.2 对比基线定义与最小可发表版本

Part I 的方法集合建议分成两层：

- 公共支持域 `strict four-way`：
  - `reference attention`：reference / 非 fused `matmul + softmax + matmul`，作为实现下界和控制组。
  - `Flash Attention`：TT 主线 `SDPA / Flash-Decode`，作为标准 attention 家族的高性能基线。
  - `FlashMLA（TT 主线）`：TT 主线 MLA prefill / decode，作为 generic MLA 基线。
  - `DeepSeek FlashMLA`：`models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` 路线；当前 Part I 使用一次性 adapter 后的 backend-op-only 计时口径。
- Wormhole `five-method decode` 专项：
  - 在上述前三条基线基础上，把 `DeepSeek FlashMLA` 明确拆成 `FlashMLA-4c` 与 `FlashMLA-8c` 两种 S-block 几何。

最小可发表版本应优先完成并固定：

- 公共支持域 `decode` 四方法主对比（主表 / 主图）。
- `prefill_1k / 4k` 的三方法对照表，`DeepSeek` 显式记为不适用而不是强行补点。
- Wormhole `wh_five_method_all_cases.md` 五方法总表，以及其中抽出的平均延迟 / 平均吞吐 summary。
- `wh_4c_vs_8c_batch/head/seq/qshard4_seq` 四组专项对照表，用来解释 `4c -> 8c` 的局部规律。
- `five-method-dataflow-analysis.md` 中的数据流说明，至少以摘要或附录形式进入论文 supporting material。
- `DeepSeek FlashMLA` 的 representative-point stage-level profile 仍然是增强项；如果 dense sweep 仍只能来自 `FlashMLA（TT 主线）`，则必须显式标成 proxy evidence。

需要明确的解释边界：

- `reference attention -> Flash Attention` 是同一 attention family 内的实现演进。
- `FlashMLA（TT 主线）` / `DeepSeek FlashMLA` 与前两者的比值可以报告，但应解释为跨 family 的 system-level comparison，而不是 drop-in replacement 的严格 speedup。
- `DeepSeek FlashMLA -> FlashMLA（TT 主线）` 更适合写成同一 MLA 语义下的实现级比较。

### 5.3 图表与表格设计

建议主文为 Part I 保留以下产物：

- 公共支持域主图：
  `decode` 长度 sweep 上四条 latency / throughput 曲线（reference / Flash Attention / `FlashMLA（TT 主线）` / `DeepSeek FlashMLA`）。
- 公共支持域辅图：
  相对 `reference attention` 的 normalized speedup，以及 `DeepSeek FlashMLA vs FlashMLA（TT 主线）` 的 latency delta / ratio 曲线。
- 控制表：
  `prefill_1k / 4k` 的三方法对照表。
- Wormhole 专项主表：
  `wh_five_method_all_cases.md`，以及其中抽出的平均延迟 / 平均吞吐小表。
- Wormhole 专项补充表：
  `wh_4c_vs_8c_batch_summary.md`、`wh_4c_vs_8c_head_summary.md`、`wh_4c_vs_8c_seq_summary.md`、`wh_4c_vs_8c_qshard4_seq_summary.md`。
- 补充说明材料：
  `five-method-dataflow-analysis.md`，以及 `DeepSeek FlashMLA` representative-point phase transition figure，或显式标注为 proxy 的 `FlashMLA（TT 主线）` phase transition / measured latency vs ideal-gap 图；这些材料主要用于承接 Part II 的机理分析。

### 5.4 建议执行接口与 runner 组织

当前仓库已经有一条明确的 Part I 执行链：

- `run_part1_benchmarks.py`：
  作为当前统一 benchmark harness，显式驱动 `reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`DeepSeek FlashMLA` 的公共支持域 benchmark，并保留三方法 prefill control、capability probe 和 Wormhole `4c/8c` rerun 入口。
- `render_part1_results.py`：
  负责把公共支持域结果渲染成 report、fair 四方法表格、切片 summary 与 capability probe 汇总。
- `reference attention`：
  当前已由统一 harness 调用 torch reference SDPA，作为控制组。
- `Flash Attention`：
  直接使用 `ttnn.transformer.scaled_dot_product_attention`、
  `chunked_scaled_dot_product_attention`、
  `scaled_dot_product_attention_decode`、
  `paged_scaled_dot_product_attention_decode`。
- `FlashMLA（TT 主线）`：
  直接使用 `ttnn.transformer.flash_mla_prefill`、
  `chunked_flash_mla_prefill`、
  `flash_multi_latent_attention_decode`、
  `paged_flash_multi_latent_attention_decode`。
- `DeepSeek FlashMLA`：
  构造 `FlashMLADecode.op` 所需的 `FlashMLAProgramConfig`、WH `4c/8c` grid、`cur_pos_tensor` 与专用 `Q / KV cache` shard 布局，再在一次性 adapter 后计时 backend device op。

需要注意：

- 公共支持域当前实现固定在 `comparison_mode=strict_four_way`；已有 fair 四方法主表和 slice summary 可直接复用。
- Wormhole `4c/8c` 五方法大表当前不是 renderer 自动产物，而是由多次 `--deepseek-wh-cores-per-block=4/8` 专项 rerun 汇总得到。
- 当前 `experiments/` 下现成的 detailed profile runner 仍主要覆盖 `FlashMLA（TT 主线）` characterization；`DeepSeek FlashMLA` 的 dedicated profiler runner 仍然缺失。
- 因此 Part I 的公共支持域主图 / 主表可以直接复用现有统一 harness，而 `4c/8c` 五方法专项则应继续沿用当前 rerun + 汇总表的组织方式。

### 5.5 必报指标

- 公共支持域四方法的过滤后 `avg / best / worst latency`
- 公共支持域的 throughput、相对 `reference attention` 的 normalized speedup
- `DeepSeek FlashMLA` 相对 `Flash Attention`、`FlashMLA（TT 主线）` 的 latency delta 或 ratio
- `prefill_1k / 4k` 的三方法控制表
- Wormhole 五方法专项的 `avg / best / worst latency; throughput`
- `FlashMLA-4c / FlashMLA-8c` 的 `unsupported` 边界，以及按 `B/H/seq_len` 统计出的 winner summary
- 如果版面允许，可补一个 logical KV footprint / bytes-read 说明项，用来解释跨 attention family 的差异


## 6. Part II：利用率与瓶颈归因

### 6.1 目标

把 Part I 中作为主方法的 `DeepSeek FlashMLA` 那条性能曲线背后的“为什么变慢 / 为什么还能继续优化”讲清楚，并且把 decode 与 prefill 区分开。当前已经有 `experiments/part2_utilization_bottleneck_analysis/` 收拢目录：它把 `FlashMLA（TT 主线）` 的 dense stage/source/bubble characterization 固定为 long-seq mechanism proxy，并通过 Part I 代理对齐表显式约束解释边界。`DeepSeek FlashMLA` 的 direct representative-point profiler 仍是增强项，而不是当前目录成立的前提。

### 6.2 主线实验

- 已落地目录：`experiments/part2_utilization_bottleneck_analysis/`，统一产出 `report.md`、`dashboard.html`、`visuals/`、`tables/` 与 `raw/`。
- 当前冻结口径：`decode_4k / 16k / 32k` 的 `DeepSeek / TT` latency ratio 已收敛到 `1.025x / 0.976x / 0.997x`，最大偏差约 `2.5%`；因此 dense `TT 主线 FlashMLA` profile 可作为 long-seq mechanism proxy，而 `decode_1k = 0.864x` 仅作为 short-seq 边界提醒。

`E2-A` `DeepSeek FlashMLA` representative-point decode profiling（增强项）

- 数据源：
  待新增的 `DeepSeek FlashMLA` decode profiler runner（基于 `FlashMLADecode.op` / `test_flash_mla_wh.py`）
- 目标：
  如果后续要把 proxy 进一步替换成主方法 direct evidence，则在 `decode_1k / 4k / 32k` 上确认主方法是否也经历
  `compute-critical -> writer-close -> reader-close -> reader_writer_saturated`
  的阶段迁移

`E2-B` `FlashMLA（TT 主线）` full sweep stage-level 分析

- 数据源：
  `flash_mla_wh_detailed_profile_results.json`
- 目标：
  给出 dense sweep 的
  `compute-critical -> writer-close -> reader-close -> reader_writer_saturated`
  阶段迁移，作为 generic MLA 机理证据和 `DeepSeek FlashMLA` 路径的解释 proxy

`E2-C` decode reader / writer 分解

- reader：`page_table / reserve / issue / wait / push`
- writer：`cb_wait / issue / barrier / pop`
- 目标：
  证明长序列 decode 的主矛盾不是单纯带宽不够，而是强耦合流水线中的 backpressure
- 备注：
  如果当前只能拿到 `FlashMLA（TT 主线）` 的分解数据，正文里要明确标成 proxy evidence

`E2-D` decode compute bubble rerun

- 使用 `run_flash_mla_pm_bubble_probe.py`，或补一条 `DeepSeek FlashMLA` decode bubble / wait counter 路径
- 目标：
  补齐 `wait-front / reserve-back` 趋势，证明 compute 从“只等输入”逐渐转向“两头受挤压”

`E2-E` prefill 对照

- 目标：
  证明 prefill 从最短点开始就是 `reader_writer_saturated`
- 作用：
  防止把 decode 的阶段迁移错误地推广为“所有 MLA 都这样”

### 6.3 可选 rerun 集

如果后续要进一步收紧 proxy 边界，建议组织为：

- `DeepSeek FlashMLA`：`decode_1k`
- `DeepSeek FlashMLA`：`decode_4k`
- `DeepSeek FlashMLA`：`decode_32k`
- `FlashMLA（TT 主线）`：`prefill_4k`

对应动作：

- `FlashMLA（TT 主线）` 路径可沿用已有命令：

```bash
python3 mla_flash_attention_dev/experiments/run_flash_mla_pm_bubble_probe.py \
  --minimal
```

- `DeepSeek FlashMLA` 路径则需要新增一个最小 decode profiler runner，至少覆盖 `1k / 4k / 32k` 三个点。

### 6.4 必报指标

- phase classification
- reader `reserve share` / `issue share`
- writer `cb_wait share`
- compute `wait-front share` / `reserve-back share`
- prefill `PM FPU util`

### 6.5 本部分的关键口径

- decode 的 direct PM util 目前仍不可靠，不作为主文主要证据。
- decode 的主文归因应建立在：
  stage-level breakdown + compute bubble counters + representative swimlane
  这三层证据上。
- `FlashMLA（TT 主线）` 的 dense sweep 可以继续支撑机理叙事，但如果用来解释 `DeepSeek FlashMLA`，至少要配套 Part I proxy alignment 表 / 图来显式约束解释边界；更理想的增强版做法仍是再补一个 `DeepSeek FlashMLA` representative-point measurement。


## 7. Part III：Cost Model 与 Autotuner

### 7.1 目标

证明三件事：

1. 二阶 cost model 相对一阶模型确实更贴近硬件。
2. autotuner 在当前搜索空间中能稳定找到高质量配置，且开销可接受。
3. 但由于当前 calibration / measurement-db 都建立在 `TT 主线 FlashMLA` 候选空间上，这部分结论应明确读作 `generic MLA backend planning proxy`。

### 7.2 当前可直接复用的链路

当前这条链路已经基本齐全：

1. `experiments/part3_cost_model_autotuner/`
   已把 report / table / figure / dashboard / raw snapshot 固化成单独目录
2. `profile_flash_mla_wh_detailed.py`
   产出 `flash_mla_wh_detailed_profile_results.json`
3. `profile_measurement_bridge.py`
   可把 profile JSON 转成 `measurement_db`
4. `basic_autotuner.py`
   支持：
   - analytical top-K
   - measured reranking
   - 直接从 WH profile 拟合 calibration

### 7.3 推荐执行顺序

第一步，确认现有 measurement DB：

```bash
python3 -m mla_flash_attention_dev.autotuner \
  --build-wh-profile-measurement-db \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_results.json \
  --measurement-db-output \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_measurement_db.json
```

第二步，从 detailed WH profile 直接拟合 calibration：

```bash
python3 -m mla_flash_attention_dev.autotuner \
  --wh-profile-json \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_profile_results.json \
  --fit-calibration-output \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_calibration.json
```

第三步，在 decode preset 上跑 calibrated autotuner：

```bash
python3 -m mla_flash_attention_dev.autotuner \
  --preset flash_decode_wh \
  --calibration-json \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_calibration.json \
  --top-k 10
```

第四步，可选地检查 measured rerank：

```bash
python3 -m mla_flash_attention_dev.autotuner \
  --preset flash_decode_wh \
  --measurement-db \
  mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed/flash_mla_wh_detailed_measurement_db.json \
  --rerank-top-k 8
```

### 7.4 主文要汇报什么

- first-order vs second-order MAPE
- leave-one-out cross-validation
- top-1 recommendation 与 empirically best latency 的差距
- search overhead

当前章节口径已经基本固定为：

- analytical -> calibrated fidelity：`MAPE 38.62% -> 19.58%`
- leave-one-out：`MAPE 24.65%`
- calibrated top-1：`tree + row_packed_by_head + k_chunk=256 + dual_noc=true`，selected latency `0.055 ms`
- rerank runtime：`2.660 s`，但当前 `measured_candidate_count = 0`
- exact-case 边界：`decode_4k` 的 profiled A-path candidate 在候选池中排第 `65`，measured latency 相对 best selected 仍慢 `1.43x`

### 7.5 需要额外注意的限制

- 当前 Part III 应读作 `generic MLA backend planning proxy`，而不是 `DeepSeek FlashMLA` 的 direct tuning result。
- autotuner 仍是 offline analytical planner，不会主动发 kernel。
- measured reranking 依赖已有 measurement DB，而不是现场 benchmark；当前 preset 级别 rerank 甚至还没有真正命中可改写 top-1 的 measured candidate。
- 如果后面要把 `B1 / B2 / B3` 也接进 tuner，最好增加统一 benchmark harness，而不是只靠静态 JSON bridge。


## 8. Part IV：未来架构支持与展望

### 8.1 目标

把 Wormhole 上得到的 characterization 转化成对 future architecture 的 proxy 结论，但必须明确当前 `experiments/part4_architecture_implications/` 仍建立在 `TT 主线 FlashMLA` 的 A-path detailed characterization 与 native `B-BH` analytical table 上，因此它回答的是 `generic MLA backend architecture proxy`，不是 `DeepSeek FlashMLA` 的 direct BH benchmark。

### 8.2 主线实验

`E4-A` Part I proxy alignment context

- 目标：
  用 `decode_4k / 16k / 32k` 的 `DeepSeek / TT` ratio 约束 long-seq proxy 的可解释边界

`E4-B` current A-BH ideal vs empirical projection

- 目标：
  固化 `A-BH ideal`、`A-BH empirical` 与 `decode_1k / 4k / 8k / 16k / 32k` 的 empirical uplift

`E4-C` active-core crossover / plateau sweep

- 目标：
  量化一阶 `dram crossover`、二阶 `non-compute crossover` 与 plateau start，说明 compute 会很早退出 critical path

`E4-D` reader / writer coupling breakdown

- 目标：
  证明 future hardware 上真正锁住 scaling 的仍是 reader / writer floor，而不是单纯继续堆 compute cores

`E4-E` native B-BH vs current A-BH

- 目标：
  比较 native `B-BH` 相对 current `A-BH` 的 topology / multicast 优势，并确认 long-seq 下优势会持续放大

### 8.3 当前可直接复用的实现

- `experiments/part4_architecture_implications/`
  已把 report / table / figure / dashboard / raw snapshot 固化成单独目录
- `profile_flash_mla_wh_detailed.py`
  已经内置 `A-BH empirical coupled` 路线
- `experiments/flash_mla_wh_profile_and_bh_simulation.md`
  提供 native `B-BH` 与 current `A-BH` 的 analytical 对照表

### 8.4 主文要汇报什么

- long-seq proxy 对齐：`decode_4k / 16k / 32k` 上 `DeepSeek / TT = 1.025x / 0.976x / 0.997x`
- current A-BH 的一阶 `dram crossover` 基本稳定在 `23c`
- 二阶 `non-compute crossover` 已提前到 `7~9c`，plateau start 提前到 `8~17c`
- `decode_32k` 上 A-BH empirical 仍比 ideal first-order 高 `1.79x`，且 `reader_total > compute`
- native `B-BH` 相对 A-BH empirical 的优势会从 `decode_1k` 的 `1.28x` 扩大到 `decode_32k` 的 `5.60x`
- 因而更稳妥的未来架构口径是：优先优化 topology / multicast / reader-writer coupling，而不是只继续堆 compute


## 9. 最小可发表版本 vs 增强版本

### 9.1 最小可发表版本

如果目标是尽快形成一版能自洽的论文实验，建议只要求以下交付：

- 公共支持域 `reference attention / Flash Attention / FlashMLA（TT 主线） / DeepSeek FlashMLA` 的 `decode` 四方法主对比图
- `prefill_1k / 4k` 的三方法控制对照表
- Wormhole `reference / Flash / FlashMLA（TT 主线） / FlashMLA-4c / FlashMLA-8c` 的五方法 decode 汇总表
- `DeepSeek FlashMLA vs FlashMLA（TT 主线）` 的 decode delta / ratio 曲线
- `FlashMLA-4c / FlashMLA-8c` 的 batch/head/seq 对照表与数据流说明摘要
- Part II 收拢目录中的 proxy 对齐 + reader / writer / compute utilization and bubble analysis
- second-order model fidelity
- autotuner top-1 quality + search overhead
- Blackhole projection / active-core plateau / native B-BH topology 对照

对应的收拢目录已经分别存在于：

- `experiments/part2_utilization_bottleneck_analysis/`
- `experiments/part3_cost_model_autotuner/`
- `experiments/part4_architecture_implications/`

也就是说，当前就已经可以完成一篇
**fair four-way baseline + Wormhole five-method decode specialization + model + architecture implications**
导向的实验部分。

### 9.2 增强版本

如果后续实现进度允许，再把下面几项加入主文：

- 更完整的五方法 full-factorial sweep
- 把 `wh_five_method_all_cases.md` 这类专项汇总纳入统一 render pipeline
- 更完整的 `prefill` / `batch` sweep
- standalone 与 fused 的联合结果
- 更完整的 `DeepSeek FlashMLA` source-level representative points
- 直接 benchmark 驱动的 measured reranking
- `B0 / B1 / B2 / B3` 这条 FlashMLA 内部优化链


## 10. 推荐执行顺序

当前仓库里，Phase C / D / E 已分别有 `part2_utilization_bottleneck_analysis/`、`part3_cost_model_autotuner/`、`part4_architecture_implications/` 三个收拢目录；如果上游 source assets 没有继续变化，这三阶段通常只需要 rerun 对应 `build_part*_results.py` 脚本即可刷新图表、表格和报告。

### Phase A：冻结公共支持域 fair benchmark 口径

- 明确 `reference attention / Flash Attention / FlashMLA（TT 主线） / DeepSeek FlashMLA` 的接口、输入张量语义和公平比较边界
- 冻结 `decode_256..128k` 与 `prefill_1k / 4k` 的主 workload 矩阵
- 冻结 `align_with_tt_mainline`、随机种子、page table、layout 和 `backend-op-only` 计时说明

### Phase B：冻结 Wormhole `4c/8c` 五方法专项

- 跑 / 更新 `--deepseek-wh-cores-per-block=4/8` 的 batch/head/seq/qshard4 专项 rerun
- 刷新 `wh_five_method_all_cases.md` 及其中的平均延迟 / 平均吞吐摘要
- 刷新 `wh_4c_vs_8c_*` 对照表与 `five-method-dataflow-analysis.md`

### Phase C：冻结 `DeepSeek FlashMLA` 与 `FlashMLA（TT 主线）` 的补充 characterization 资产

- 新增并跑 `DeepSeek FlashMLA` representative-point decode profiler
- 复核 `flash_mla_wh_detailed_profile_results.json`
- 复核 `flash_mla_wh_detailed_profile_report.md`
- 复核 `docs/flash-mla-wh-component-utilization-and-bubble-analysis.md`
- 对齐 `DeepSeek FlashMLA` representative points 与 `FlashMLA（TT 主线）` proxy 之间的解释口径

### Phase D：冻结 tuner 结果

- 重新生成 / 校验 measurement DB
- 重新生成 / 校验 calibration JSON
- 产出 `decode` preset 的 top-K 与 rerank summary

### Phase E：冻结 architecture projection

- 统一 Wormhole -> Blackhole 的 coupled projection 口径
- 把 SRAM / TOPS / core granularity / MLA-vs-MHA 四个结论固定下来

### Phase F：如果实现进度允许，再扩展 FlashMLA 内部优化结果

- 评估是否加入 `B0 -> B1 -> B2 -> B3` 的 FlashMLA 内部 speedup 链
- 再决定是否把这条优化链放进主文、附录或下一阶段论文版本


## 11. 一句话版本

基于当前章节与现有代码资产，Part I 最合理的改法不再是只保留单层四方法叙事，而是先用公共支持域的
`reference attention / Flash Attention / FlashMLA（TT 主线） / DeepSeek FlashMLA`
把公平基线立住，再用 Wormhole 专项的
`reference attention / Flash Attention / FlashMLA（TT 主线） / FlashMLA-4c / FlashMLA-8c`
把 `4c/8c` 几何差异、支持边界和数据流差异讲清楚；`prefill` 三方法控制表继续保留为控制结论。在此基础上，再把 `DeepSeek FlashMLA` 的 representative-point 归因、`FlashMLA（TT 主线）` 的 dense characterization、cost model / autotuner 和 future architecture implications 分别放到后续部分讲扎实。

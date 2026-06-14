# Part II 实验计划（基于当前 Part I）

## 0. 一页摘要

当前 `Part II` 不需要推倒重来，但需要从“单一 `TT 主线 FlashMLA` dense proxy 叙事”升级成“两层证据”结构：

1. **保留现有 `TT 主线 FlashMLA` dense proxy 资产**  
   用它继续回答完整 `decode` 阶段迁移、reader/writer/source attribution 和 `prefill` 控制对照。
2. **补一组最小但足够强的 direct evidence**  
   重点补 `DeepSeek FlashMLA` 与 Wormhole `FlashMLA-4c / FlashMLA-8c` 的 representative-point profiler / attribution，用来解释当前 `Part I` 里已经出现的五方法结论。

如果只追求最小可发表版本，建议把 `Part II` 的新增工作压缩到三类：

- `DeepSeek-4c` 对 `FlashMLA（TT 主线）` 的 direct 长序列对齐点
- `FlashMLA-4c / FlashMLA-8c` 在重叠区间的 direct 归因点
- `4c unsupported -> 8c supported` 的容量边界点

当前 `prefill` 不建议扩张；现有 `prefill` 结果足够保留为控制组。

如果进一步按**问题导向**收束，这一轮 `Part II` 实际上只需要围绕三件事组织：

1. **为什么当前 `TT-MLA` 方法不好**  
   重点不是证明它“数学上不如人”，而是证明当前实现里更多 wall-time 落在 `reader / writer / backpressure / bubble`，而不是有效 compute。
2. **为什么 `SF-MLA` 更好**  
   重点不是只看 latency 结果，而是证明它在 representative points 上把更多时间转成了有效计算，表现为更高的空间利用率、算术利用率和更低的 bubble / reserve。
3. **为什么 `SF-MLA` 仍然有改进空间**  
   重点是证明它在 long-seq 和高压力区间仍会被 `K reserve`、`reader reserve`、`tree reduction`、`writer cb_wait` 卡住，只是比当前 `TT-MLA` 更晚触发、程度更轻。

---

## 1. 为什么现在要改 Part II

当前 `Part I` 已经不是旧的单层四方法结构，而是两层：

1. **公共支持域 fair benchmark**
   - `reference attention`
   - `Flash Attention`
   - `FlashMLA（TT 主线）`
   - `DeepSeek FlashMLA`
2. **Wormhole 五方法专项**
   - `reference attention`
   - `Flash Attention`
   - `FlashMLA（TT 主线）`
   - `FlashMLA-4c`
   - `FlashMLA-8c`

而当前 `Part II` 的核心机理证据几乎都来自：

- `TT 主线 FlashMLA` 的 dense stage/source/bubble profiling
- `Part I` 的 `DeepSeek / TT` latency ratio 对齐表

这带来三个问题：

1. **能解释 long-seq 主趋势，但不能直接解释 4c/8c 几何差异**
2. **能解释 TT MLA 的阶段迁移，但不是 DeepSeek/4c/8c 的 direct profiler**
3. **能承接旧的四方法主图，但无法完整承接新的五方法专项结论**

因此，新的 `Part II` 目标不应是替换现有 proxy，而应是：

- **保留 proxy 的密集覆盖能力**
- **用最小 direct profiling 集合，把当前 `Part I` 最关键的新现象钉死**

---

## 2. 当前可复用资产盘点

### 2.1 可直接复用的 Part II 资产

这些资产已经可以继续保留，不需要重跑：

| 资产 | 路径 | 在新 Part II 中的角色 |
|---|---|---|
| Part II 总报告 | `mla_flash_attention_dev/experiments/part2_utilization/outputs/report.md` | 现有 proxy 叙事的总入口 |
| Part II build 脚本 | `mla_flash_attention_dev/experiments/part2_utilization/build_part2_results.py` | 当前 proxy pipeline 的产表/产图入口 |
| Part II README | `mla_flash_attention_dev/experiments/part2_utilization/README.md` | 当前目录用途说明 |
| decode 阶段迁移 / bubble / source 图表 | `mla_flash_attention_dev/experiments/part2_utilization/outputs/visuals/` | 保留为 `TT 主线 FlashMLA long-seq proxy` 图组 |
| decode 阶段分类 / source attribution / prefill control 表 | `mla_flash_attention_dev/experiments/part2_utilization/outputs/tables/` | 保留为 proxy 表组 |
| manifest / source report 快照 | `mla_flash_attention_dev/experiments/part2_utilization/outputs/raw/` | 保留为可复现实验快照 |

### 2.2 可直接复用的 Part I 资产

这些资产应直接作为新 `Part II` 的上游输入或选点依据：

| 资产 | 路径 | 用途 |
|---|---|---|
| fair 四方法总报告 | `mla_flash_attention_dev/experiments/part1_baselines/outputs/report.md` | 给出当前四方法主叙事和已有 slice 结论 |
| fair 四方法全量结果 | `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/fair_four_method_all_cases.md` | 给出公共支持域主表 |
| 五方法总表 | `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_five_method_all_cases.md` | 给出 `4c/8c/TT` 的代表性胜负和边界 |
| 4c vs 8c batch/head/seq 对照表 | `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_batch_summary.md`、`wh_4c_vs_8c_head_summary.md`、`wh_4c_vs_8c_seq_summary.md`、`wh_4c_vs_8c_qshard4_seq_summary.md` | 给出新 `Part II` 的最小补实验候选点 |
| 五方法数据流分析 | `mla_flash_attention_dev/experiments/part1_baselines/five-method-dataflow-analysis.md` | 给出 reader/compute/writer 和 `4c/8c` 几何差异的解释框架 |

### 2.3 可复用但必须明确标注为 proxy 的资产

以下结果仍然有价值，但**不能**写成 `DeepSeek FlashMLA` 或 `FlashMLA-4c/8c` 的 direct evidence：

| 资产 | 路径 | 必须如何标注 |
|---|---|---|
| decode 阶段迁移图 | `.../outputs/visuals/decode_phase_transition.svg` | 标成 `TT 主线 FlashMLA dense mechanism proxy` |
| decode bubble 图 | `.../outputs/visuals/decode_bubble_density.svg`、`decode_bubble_composition.svg` | 标成 proxy，不作为 DeepSeek direct profiling |
| decode reader/writer attribution 图 | `.../outputs/visuals/decode_reader_source_breakdown.svg`、`decode_writer_source_breakdown.svg` | 标成 proxy |
| decode 阶段分类表 | `.../outputs/tables/decode_phase_classification.md` | 标成 proxy |
| decode source attribution 表 | `.../outputs/tables/decode_source_attribution.md` | 标成 proxy |
| 当前 Part II 报告中的大部分核心结论 | `.../outputs/report.md` | 说明其建立在 `TT 主线 FlashMLA` profiling 上 |

### 2.4 当前已经足够、无需优先补跑的资产

- `prefill_control.svg`
- `prefill_control.md`
- `flash_mla_pm_bubble_probe_*` 的现有 prefill control 路线

原因：

- 当前 `Part I` 对 `prefill` 的要求仍然只是三方法控制对照
- 新增的 `4c/8c` 专项全部发生在 `decode`
- `Part II` 当前的最大缺口在 `decode` direct evidence，不在 `prefill`

---

## 3. 当前 Part II 的主要缺口

### 3.1 缺少 DeepSeek 路径的 direct profiler

当前只有：

- `DeepSeek / TT` 的 latency 对齐

当前没有：

- `DeepSeek` 本身的 stage-level profiler
- `DeepSeek` 本身的 source-level attribution
- 与 `TT 主线 FlashMLA` 同配置、同 profiling 口径下的 paired comparison

### 3.2 缺少 4c vs 8c 的 direct bottleneck comparison

当前 `Part I` 已经说明：

- `FlashMLA-4c` 在中等压力下可能最优
- `FlashMLA-8c` 在高压力下更稳定
- `4c` 有明确的 `batch * q_shards <= 24` 边界
- `8c` 把上限抬到 `48`

但当前 `Part II` 没有任何一张图或表直接回答：

- `4c` 为什么在某些点更快
- `8c` 为什么在长序列和大压力下更强
- `unsupported -> supported` 的边界究竟是 capacity 问题、mapping 问题，还是别的 bottleneck

### 3.3 现有 decode proxy 与当前 Part I 主口径不完全同构

当前 `Part II` 的 decode full sweep 是：

- `decode_256~8k`：`batch=2`
- `decode_16k/32k`：`batch=1`

而当前 `Part I` 新主叙事包括：

- `B=1,H=32` 的 fair 四方法主曲线
- `B=6,H=32,q_shards=4` 的 4c/8c q-shards=4 路线
- `B=8,H=24,q_shards=3` 的 4c/8c q-shards=3 路线
- `B=8/12,H=32` 的 4c unsupported / 8c supported 容量边界

所以 `Part II` 不能再只靠 mixed-batch proxy 去解释全部现象。

### 3.4 缺少对 run-to-run 敏感点的正式稳定性设计

当前 `Part I` 已经识别出：

- `4k, B=6, H=32, q_shards=4` 对 `4c/8c` 胜负方向比较敏感

这类点在进入 `Part II` 之前，必须先做：

- benchmark 复测
- profiler 复测
- 结果稳定性标注

否则很容易把波动点误写成机制性结论。

---

## 4. 优化后的 Part II 结构

建议把新的 `Part II` 明确拆成三层。

### 4.1 第 1 层：保留现有 TT 主线 dense proxy

回答的问题：

- decode 的完整阶段迁移是什么
- reader/writer/source attribution 大趋势是什么
- prefill 控制组体现了什么

证据来源：

- 现有 `part2_utilization/outputs/*`

角色定位：

- **完整 coverage 的 proxy 层**
- 负责提供 dense sweep、完整图组和已有 report

### 4.2 第 2 层：补 DeepSeek representative-point direct evidence

回答的问题：

- `DeepSeek FlashMLA` 在当前 Part I 主曲线上的关键点，是否真的呈现出与 proxy 一致的阶段迁移方向
- `1k` 的短序列偏差为什么更大
- `4k~32k` 的 long-seq 对齐是否在 direct profiler 上也成立
- `128k` 的长上下文优势对应的是哪一类 bottleneck

证据来源：

- 新增的 `DeepSeek` representative-point profiler

角色定位：

- **把 proxy 从“可信猜测”升级为“有 direct anchor 支撑的解释”**

### 4.3 第 3 层：补 Wormhole 4c/8c geometry direct evidence

回答的问题：

- 为什么 `4c` 在中等压力下可能更快
- 为什么 `8c` 在高压力下更强
- 为什么 `8c` 能解锁 `4c` 不支持的点
- 哪些点属于 mapping / capacity 优势，哪些点属于 long-seq bandwidth 优势

证据来源：

- 新增的 `4c / 8c / TT` paired representative-point profiling

角色定位：

- **直接承接当前 Part I 五方法专项**

### 4.4 本轮要直接回答的四个问题

| 问题 | 代表实验 | 主要看什么 | 希望得到的结论 |
|---|---|---|---|
| 为什么当前 `TT-MLA` 不好 | `A1/A2/A3`（必要时加 `A4`） | `compute_share`、`PM_FPU util`、`reader reserve`、`writer cb_wait`、`bubble density`、`K reserve` | 证明当前 `TT-MLA` 主要差在实现路径的数据供给与回压，而不是算子数学本身 |
| 为什么 `SF-MLA` 更好 | `A2/A3` + `B1/B3` + `C1/C3` | `lane_utilization`、`PM_FPU util`、`compute_share`、`bubble density`、`effective K read bandwidth` | 证明 `SF-MLA` 的收益来自更贴合的数据流、几何映射与更高有效计算占比 |
| `SF-MLA` 还差在哪 | `A3/A4` + `B3` + `C3` | `reader reserve`、`K reserve`、`reserve-back bubble`、`sender/tree wait` | 证明 `SF-MLA` 在 long-seq / 高压力下仍受 K 侧供给、归约与 writer 回压限制 |
| `4c` 和 `8c` 的边界分别是什么 | `B* / C* / D1/D2` | `required_q_cores`、`lane_utilization`、`supported/unsupported`、`4c/8c` 胜负方向 | 证明 `4c` 的主要问题是 capacity wall，`8c` 的主要价值是解锁高压力映射，但低压力下不一定处处更优 |

---

## 5. 建议保留与建议新增的结论边界

### 5.1 继续保留的结论

- `TT 主线 FlashMLA` 的 dense decode phase transition
- `K reserve` 相比 `V reserve` 更主导 long-seq decode
- `writer` 的等待主要不在 final gather
- `prefill` 是强耦合饱和流水线

### 5.2 需要改写的结论

当前凡是写成：

- “DeepSeek 在 long-seq 为什么慢 / 快”
- “4c 和 8c 分别为什么最优”

都不应只由现有 proxy 图直接支撑。

新计划里应改成：

- 先给出 `DeepSeek / TT` 或 `4c / 8c / TT` 的 direct representative evidence
- 再用现有 proxy 图去补密集趋势和机制细节

### 5.3 不建议再扩张的方向

- 现在不建议优先扩张 `prefill`
- 现在不建议优先追求 full-factorial `4c/8c` profiler
- 现在不建议把 `NOC/DRAM util` 当作默认必跑项

原因：

- 当前论文主矛盾在 `decode`
- full-factorial profiler 成本高且难稳定
- `NOC/DRAM util` 路线在当前机器上仍属于高风险扩展项

### 5.4 建议新增的结论模板

在新的 `Part II` 里，建议把结论固定写成下面这种问题导向模板，而不是泛泛地说“方法 A 快于方法 B”：

1. **关于 `TT-MLA`**
   - “当前 `TT-MLA` 的劣势主要来自实现路径中的 `reader reserve / writer cb_wait / bubble`，而不是算术计算本身。”
2. **关于 `SF-MLA`**
   - “`SF-MLA` 的优势主要来自更高的空间利用率、算术利用率，以及更低的 bubble / reserve share；它把更多 wall-time 转成了有效 compute。”
3. **关于 `SF-MLA` 的改进空间**
   - “`SF-MLA` 虽然更优，但在 long-seq 与高压力点仍会被 `K reserve`、`reader reserve`、`tree reduction`、`writer cb_wait` 卡住，因此仍有优化余地。”
4. **关于 `4c / 8c`**
   - “`4c` 的主要限制是 capacity wall，`8c` 的主要收益是解锁更高 `batch * q_shards` 与更稳的高压力映射；但 `8c` 并不自动意味着所有点都更快。”

---

## 6. 最小补实验集

下面这组实验是新的 `Part II` **最小 direct evidence 集合**。

### 6.1 A 组：DeepSeek vs TT 主线的 direct 对齐点

目标：

- 给当前 fair 四方法主曲线补 direct profiler anchor

配置：

| 组别 | config | seq_len | baseline | 目的 |
|---|---|---:|---|---|
| A1 | `b1_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | `1k` | `FlashMLA（TT 主线）` vs `DeepSeek FlashMLA(4c)` | 解释短序列 `1k` 对齐最差点 |
| A2 | 同上 | `4k` | 同上 | 解释从 short-seq 向 long-seq 过渡点 |
| A3 | 同上 | `32k` | 同上 | 解释 long-seq 已贴近 proxy 的代表点 |
| A4 | 同上 | `128k` | 同上 | 解释当前 Part I 中 DeepSeek 长上下文显著收益 |

说明：

- `A1/A2/A3` 是最小必跑
- `A4` 成本更高，但如果 Part I 的 `128k` 是正文主卖点，建议纳入

### 6.2 B 组：q_shards=4 路线的 4c/8c 直接比较

目标：

- 解释 `FlashMLA-4c` 与 `FlashMLA-8c` 在重叠可跑区间的胜负翻转

配置：

| 组别 | config | seq_len | baseline | 目的 |
|---|---|---:|---|---|
| B1 | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | `4k` | `TT` vs `4c` vs `8c` | 敏感点，必须做稳定性验证 |
| B2 | 同上 | `8k` | `TT` vs `4c` vs `8c` | 对齐 batch sweep 的中高压力区间 |
| B3 | 同上 | `32k` | `TT` vs `4c` vs `8c` | 解释长序列下 8c 稳定优势 |

说明：

- `B1` 是最关键点
- `B3` 是最稳的 long-seq geometry 对比点

### 6.3 C 组：q_shards=3 路线的 4c/8c 直接比较

目标：

- 解释 `B=8,H=24` 路线下 8c 为什么在部分点明显更优

配置：

| 组别 | config | seq_len | baseline | 目的 |
|---|---|---:|---|---|
| C1 | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | `4k` | `TT` vs `4c` vs `8c` | 对应 8c 增益明显的中序列点 |
| C2 | 同上 | `8k` | `TT` vs `4c` vs `8c` | 对齐当前 head sweep 主结论 |
| C3 | 同上 | `32k` | `TT` vs `4c` vs `8c` | 解释“长序列端接近平手”的边界现象 |

### 6.4 D 组：容量边界 / 解锁点

目标：

- 直接解释 `4c unsupported -> 8c supported`

配置：

| 组别 | config | seq_len | baseline | 目的 |
|---|---|---:|---|---|
| D1 | `b8_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | `8k` | `TT` vs `8c`，并记录 `4c unsupported` | 解释 `required_q_cores=32` 的边界 |
| D2 | `b12_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8` | `8k` | `TT` vs `8c`，并记录 `4c unsupported` | 解释 `required_q_cores=48` 的边界上限 |

说明：

- 这两点不要求 `4c` profiler，因为它本来就不支持
- 但需要把 `unsupported` 作为正式结果写进 `Part II`

### 6.5 E 组：稳定性守门实验

目标：

- 防止把波动点误读成机制结论

配置：

| 组别 | config | 重复策略 | 目的 |
|---|---|---|---|
| E1 | `b6_h32_hkv1_dv512_ro64_blk64_kc128_dqhpc8`, `seq=4k` | `4c/8c` 各做 `3` 次 profiler + `10` 次 benchmark | 校验敏感点 |
| E2 | `b8_h24_hkv1_dv512_ro64_blk64_kc128_dqhpc8`, `seq=32k` | `4c/8c` 各做 `3` 次 profiler | 校验“近似持平”点 |

### 6.6 每个实验点默认采集的指标

建议把 `Part II` 的默认指标分成六组，其中前四组属于**正文主指标**，后两组属于**增强指标**。

#### 6.6.1 Benchmark 主指标

| 指标 | 含义 | 用途 |
|---|---|---|
| `device_core_only_after_one_time_adapter_conversion` | 与 `Part I` 对齐的主 latency 口径 | 作为正文主性能口径 |
| `backend-op-only time` | 纯 backend device op 时间 | 判断 adapter 以外的纯 device 成本 |
| `adapter wall-time` | 一次性 adapter 额外成本 | 附录 / 端到端补充说明 |
| `throughput_tokens_per_s` | 吞吐量 | 与 latency 对照，避免只看单一指标 |
| `avg / best / worst` | benchmark 统计值 | 校验敏感点是否稳定 |
| `supported / unsupported` | 是否可运行 | 用于 `4c unsupported -> 8c supported` 边界结论 |

#### 6.6.2 空间利用率指标

| 指标 | 定义 | 解释问题 |
|---|---|---|
| `device_total_compute_cores` | 当前 Wormhole 口径下固定记为 `56` | 整个设备的 compute 核上限 |
| `layout_provisioned_cores` | `TT≈56`、`4c=24`、`8c=48` | 当前实现布局最多能用多少核 |
| `required_q_cores` | `batch * q_shards` | 当前 workload 实际需要多少 Q 槽位 |
| `lane_utilization` | `required_q_cores / layout_provisioned_cores` | 当前布局是否被吃满 |
| `chip_coverage` | `layout_provisioned_cores / 56` | 当前方法覆盖了整张芯片多少 compute 核 |

#### 6.6.3 时间利用率 / 阶段占比指标

| 指标 | 含义 | 解释问题 |
|---|---|---|
| `compute_share_pct` | compute thread 对 kernel critical path 的逼近程度 | 真正在算的时间占比高不高 |
| `brisc_share_pct` | BRISC 对 critical path 的逼近程度 | writer / reduction 压力大不大 |
| `ncrisc_share_pct` | NCRISC 对 critical path 的逼近程度 | reader / 取数压力大不大 |
| `reader_issue_share_pct` | reader issue 阶段 share | 读请求发射本身是不是主问题 |
| `reader_reserve_share_pct` | reader reserve 阶段 share | long-seq 供给与回压是否上升 |
| `writer_cb_wait_share_pct` | writer 等待 share | 归约 / 输出侧回压是否主导 |

#### 6.6.4 Bubble 指标

| 指标 | 含义 | 解释问题 |
|---|---|---|
| `bubble_density_vs_kernel` | `(wait_front + reserve_back) / kernel_time` | 空转在 kernel 时间里占多少 |
| `wait_front_share_in_bubble_pct` | bubble 中前向等待占比 | 上游喂不饱是否是主问题 |
| `reserve_back_share_in_bubble_pct` | bubble 中后向回压占比 | downstream backpressure 是否在增强 |

#### 6.6.5 Source-level attribution 指标

| 指标 | 含义 | 解释问题 |
|---|---|---|
| `k_issue_share_pct` | K issue share | K 请求发射成本 |
| `k_reserve_share_pct` | K reserve share | K 侧供给 / reserve 是否成为主矛盾 |
| `v_issue_share_pct` | V issue share | V issue 成本 |
| `v_reserve_share_pct` | V reserve share | 能否证明“不是 V 在卡” |
| `sender_cb_wait_share_pct` | writer sender local ready 等待 | sender 侧同步代价 |
| `tree_child_wait_share_pct` | tree child arrival 等待 | reduction tree 是否成为主问题 |
| `root_cb_wait_share_pct` | root local wait | root 合并压力 |
| `output_gather_wait_share_pct` | final gather wait | 最终 gather 是否真是瓶颈 |

#### 6.6.6 算术 / 带宽增强指标

| 指标 | 含义 | 建议用途 |
|---|---|---|
| `PM_FPU_UTIL (%)` | 实际计算 op 与理论峰值 op 的硬件近似指标 | 最直观的算术利用率指标，建议进入正文 |
| `pm_compute_ms` | PM 口径下 compute 时间 | 与 stage share 交叉验证 |
| `effective_single_pass_k_read_gbps` | 单次 K 读取的有效带宽 | 判断是不是 K 读带宽在限制 |
| `effective_output_write_gbps` | 输出写回的有效带宽 | 判断 writer 是否真的卡在带宽 |
| `PM BANDWIDTH [ns]` | PM 带宽阶段时间 | 增强解释，不必每点都进正文 |
| `DRAM BW UTIL (%)` | DRAM 带宽利用率 | 若采集稳定，可用于附录说明 |

#### 6.6.7 正文建议固定展示的“利用率 + bubble + 归因”六件套

如果正文只保留一组最直观指标，建议固定展示：

- `lane_utilization`
- `PM_FPU_UTIL (%)`
- `compute_share_pct`
- `reader_reserve_share_pct`
- `writer_cb_wait_share_pct`
- `bubble_density_vs_kernel`

如果正文还能多放两项，优先加：

- `k_reserve_share_pct`
- `sender_cb_wait_share_pct / tree_child_wait_share_pct`

---

## 7. 实验设计原则

### 7.1 计时口径必须和 Part I 对齐

对于 `DeepSeek / 4c / 8c`：

- 主口径继续使用：
  `device_core_only_after_one_time_adapter_conversion`
- 即：
  - 一次性 adapter 不计入主 latency
  - 计时主体是 backend device op

但为了让 `Part II` 更完整，建议**额外记录**：

- adapter wall-time
- backend-op-only time

这样正文仍沿用 `Part I` 主口径，附录里则可以解释：

- adapter 是否会在 end-to-end 使用中成为额外成本

### 7.2 profiler 与 benchmark 要成对出现

每个 representative point 建议产出两份结果：

1. **benchmark 验证**
   - 复用 `run_part1_benchmarks.py`
   - 用当前 `Part I` 的统计规则确认该点仍与主表一致
2. **profiler 采集**
   - 采集 stage/source/bubble
   - 用于 `Part II` 机理解释

这样可以避免出现：

- profiler 的 case 与 Part I 正式结果脱节

### 7.3 direct 与 proxy 的分工要写死

新的报告里应强制采用以下规则：

- **direct evidence 优先解释 representative points**
- **proxy evidence 只负责补连续趋势**
- **不能用 proxy 替代 direct 结论**

建议写成固定模板：

- `代表点 A/B/C 的 direct profiler 表明 ...`
- `dense proxy 进一步显示这一趋势在 2k~32k 区间延续 ...`

### 7.4 对敏感点必须加稳定性标签

对于 `4k, B=6, H=32, q_shards=4` 一类点：

- 如果 benchmark 胜负翻转
- 或 profiler 的关键 share 漂移明显

则正文写法必须改成：

- “该点对运行状态敏感，机制性结论仅限于 `4c≈8c` 或 `8c>=4c` 的保守判断”

而不是写成：

- “4c 必然优于 8c”

### 7.5 指标展示必须服务于“为什么 TT-MLA 差、为什么 SF-MLA 好、还差在哪”

新的 `Part II` 不建议再把指标展示成纯 profiler 指标堆砌，而应强制按下面三问组织：

1. **为什么 `TT-MLA` 差**
   - 重点看：`PM_FPU util`、`compute_share`、`reader reserve`、`writer cb_wait`、`bubble density`、`K reserve`
2. **为什么 `SF-MLA` 好**
   - 重点看：`lane_utilization`、`PM_FPU util`、`compute_share`、`bubble density`、`effective_single_pass_k_read_gbps`
3. **为什么 `SF-MLA` 还没到头**
   - 重点看：`reader reserve`、`K reserve`、`reserve-back bubble`、`sender/tree wait`

因此在 build 脚本和最终表格里，推荐优先新增一张统一摘要表，例如：

- `decode_utilization_bubble_summary.md`

用同一张表把 representative points 上的：

- 空间利用率
- 算术利用率
- bubble
- source-level 归因

放到一起，而不是把利用率、bubble、source attribution 完全拆散在三张互不相连的表里。

---

## 8. 建议新增的采集与构建入口

### 8.1 采集入口

建议新增一条 `DeepSeek / 4c / 8c` 的 Part II direct probe 入口，推荐二选一：

1. **扩展现有 `run_flash_mla_pm_bubble_probe.py`**
   - 加 `baseline`
   - 加 `--deepseek-wh-cores-per-block`
   - 加 `Part I case name -> profiler config` 映射
2. **新增独立 runner**
   - 例如：
     `run_deepseek_flash_mla_part2_probe.py`

推荐优先方案：

- **新增独立 runner**

原因：

- 当前 `run_flash_mla_pm_bubble_probe.py` 是为 `TT 主线 FlashMLA` 设计的
- 强行混入 `DeepSeek / 4c / 8c` 会把 manifest 结构和 case 语义搅乱
- 独立 runner 更容易明确记录：
  - `baseline`
  - `wh_cores_per_block`
  - `measurement_scope`
  - `adapter_included=false`

### 8.2 输出目录建议

建议新增：

- `mla_flash_attention_dev/experiments/profiling/outputs/deepseek_flash_mla_part2_probe_4c/`
- `mla_flash_attention_dev/experiments/profiling/outputs/deepseek_flash_mla_part2_probe_8c/`
- `mla_flash_attention_dev/experiments/profiling/outputs/flash_mla_tt_part2_paired_probe/`

每个目录至少包含：

- `run_manifest.json`
- 每个 case 独立子目录
- `reports/ops_perf_results_*.csv`
- 如可行，再附解析后的 `summary.md`

### 8.3 构建入口

建议后续扩展 `build_part2_results.py`，新增三组输出：

1. `decode_direct_alignment.md`
   - DeepSeek direct 点 vs TT proxy 点
2. `decode_4c_vs_8c_bottleneck_summary.md`
   - `4c/8c/TT` 的 representative-point phase/source 比较
3. `decode_capacity_boundary.md`
   - `unsupported -> supported` 的边界说明

对应图建议补：

- `decode_direct_alignment.svg`
- `decode_4c_vs_8c_phase_compare.svg`
- `decode_capacity_boundary.svg`

---

## 9. 推荐执行顺序

### Phase A：冻结现有可复用资产

- 不重跑当前 `Part II` 现有 proxy 资产
- 把现有 `outputs/*` 视为 `TT 主线 FlashMLA dense proxy package`
- 保留当前 `part1_proxy_alignment` 作为 long-seq proxy 对齐表

### Phase B：先打通 direct runner smoke

- 先跑一个最小 smoke point
- 推荐：
  - `A1` 的 `1k`
  - 或 `B1` 的 `4k`

通过标准：

- 能成功出 manifest
- 能成功出 `ops_perf_results`
- 能稳定复现 `backend-op-only` 口径

### Phase C：跑最小补实验集

优先顺序建议：

1. `A1/A2/A3`
2. `B1/B3`
3. `C1/C3`
4. `D1/D2`
5. `E1/E2`
6. `A4` 与其它扩展点

### Phase D：扩展 build_part2_results

- 新增 direct 与 `4c/8c` 的读入逻辑
- 生成新的表格与 SVG
- 更新 `outputs/report.md` 的叙事结构

### Phase E：可选扩展

仅在设备稳定时再考虑：

- `--collect-noc-traces`
- `perf-fpu`
- 更完整的 head/full-factorial profiler

---

## 10. 这份计划落地后的 Part II 叙事

如果按这份计划推进，新的 `Part II` 可以写成：

1. **先用 direct representative points 说明**
   - `DeepSeek` 在当前 `Part I` 主曲线上的关键点到底卡在哪里
   - `4c` 和 `8c` 为什么会在不同压力区间分胜负
2. **再用现有 TT 主线 dense proxy 扩展成连续趋势**
   - 阶段迁移如何随 `seq_len` 继续演化
   - reader/writer/source attribution 如何进入饱和区
3. **最后把 `prefill` 保留为控制组**
   - 不与 `4c/8c` 主叙事混在一起

一句话概括：

- **当前 `Part II` 最值得做的不是“再多跑一些 TT proxy”，而是“给当前 Part I 的 DeepSeek/4c/8c 关键结论补上最小 direct evidence”。**


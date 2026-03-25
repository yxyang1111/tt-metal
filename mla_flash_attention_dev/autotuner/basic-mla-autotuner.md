# Refined MLA Autotuner 说明

## 1. 目标

这份 autotuner 的目标不是直接替代真实 kernel benchmark，而是提供一个后续可以持续演进的离线骨架：

- 有统一的 workload / hardware 输入对象
- 有统一的五维并行输出对象
- 有 method-B 风格的 `S Block` 拓扑对象
- 有显式的 `compile-time` / `runtime` 配置分层
- 有显式的 group / topology / pipeline 决策
- 有可行性剪枝
- 有解析式 cost model
- 有 top-K 排序
- 有 bucketed policy cache
- 有可选 top-K measured reranking

对应代码位于：

- `mla_flash_attention_dev/autotuner/basic_autotuner.py`
- `mla_flash_attention_dev/autotuner/__main__.py`

---

## 2. 当前定位

这版 autotuner 是：

- `offline`
- `analytical`
- `rule-filtered`
- `topology-aware`
- `cacheable`

这版 autotuner 还不是：

- 在线实时搜索器
- 完整 oracle
- 直接对接真实 kernel compile args / runtime args 的最终版

更准确地说，它是一个 **V1 policy selector skeleton**，已经把下面这些此前只存在于设计文档里的概念落成了代码对象：

- `SBlockTopologySpec`
- `CompileTimeConfig`
- `RuntimeConfig`
- `PolicyBucketConfig`
- `MeasurementDB`

---

## 3. 基本用法

### 3.1 直接跑 preset

```bash
python -m mla_flash_attention_dev.autotuner --preset flash_decode_wh --top-k 5
python -m mla_flash_attention_dev.autotuner --preset mla_prefill_wh --top-k 5
```

### 3.2 用 JSON 输入

```bash
python -m mla_flash_attention_dev.autotuner \
  --workload-json workload.json \
  --hardware-json hardware.json \
  --top-k 5 \
  --output-json result.json
```

### 3.3 开启 bucketed cache

```bash
python -m mla_flash_attention_dev.autotuner \
  --preset flash_decode_wh \
  --cache-file policy_cache.json \
  --cache-key-mode sequence_pow2
```

### 3.4 开启 measured reranking

```bash
python -m mla_flash_attention_dev.autotuner \
  --preset flash_decode_wh \
  --measurement-db measurements.json \
  --rerank-top-k 8
```

---

## 4. 输入是什么

输入现在被拆成 5 层：

1. `MLAWorkloadSpec`
2. `HardwareTopologySpec`
3. `SearchSpace + CostWeights`
4. `PolicyBucketConfig`
5. `MeasurementDB`（可选）

其中：

- 前两层是被调优对象
- 后三层是 tuner 本身的控制参数

---

## 5. Workload 输入

主输入对象是 `MLAWorkloadSpec`。

### 5.1 字段列表

| 字段 | 含义 |
|---|---|
| `name` | workload 名称 |
| `mode` | `prefill` 或 `decode` |
| `causal` | 是否 causal |
| `paged` | 是否 paged |
| `batch_size` | batch 大小 |
| `seq_len_q` | Q 序列长度 |
| `seq_len_kv` | KV 序列长度 |
| `num_q_heads` | Q head 数 |
| `num_kv_heads` | KV head 数 |
| `kv_lora_rank` | MLA latent/value rank |
| `d_rope` | rope 维度 |
| `head_dim_qk` | QK 维度；若不填，默认 `kv_lora_rank + d_rope` |
| `head_dim_v` | V 维度；若不填，默认 `kv_lora_rank` |
| `q_dtype_bytes` | Q 元素字节数 |
| `kv_dtype_bytes` | KV 元素字节数 |
| `output_dtype_bytes` | 输出元素字节数 |
| `block_size` | paged/block 模式的 block size |

### 5.2 从输入推导出的中间量

给定某个候选 `q_chunk_size` / `k_chunk_size`，tuner 会自动推导：

```text
q_num_chunks  = ceil(seq_len_q  / q_chunk_size)
kv_num_chunks = ceil(seq_len_kv / k_chunk_size)
```

它们会直接影响：

- `q_parallel_factor`
- `kv_parallel_factor`
- group 切分方式
- page 粒度
- L1 预算
- DRAM / NoC / compute 成本

---

## 6. Hardware 输入

主输入对象是 `HardwareTopologySpec`。

### 6.1 字段列表

| 字段 | 含义 |
|---|---|
| `name` | 硬件配置名称 |
| `arch` | 架构名，如 `wormhole_b0` / `blackhole` |
| `compute_grid_x` | compute grid 列数 |
| `compute_grid_y` | compute grid 行数 |
| `dram_bank_endpoints` | 可建模的 DRAM bank endpoint 数 |
| `dram_bandwidth_GBs` | DRAM 理论带宽 |
| `noc_bandwidth_GBs` | 单 NoC 理论带宽 |
| `l1_bytes_per_core` | 每核 L1 可用容量 |
| `clock_GHz` | 时钟 |
| `max_noc_page_size` | NoC 最大软件 page size |
| `max_trid_window` | 当前可用 trid 窗口深度 |
| `noc_count` | NoC 通道数 |
| `num_devices` | 设备数 |
| `supports_multicast` | 是否支持 multicast |
| `supports_tree` | 是否支持 tree 模式 |
| `supports_dual_noc` | 是否支持 dual NoC |

### 6.2 当前内置 hardware preset

当前内置：

- `wormhole_b0`
- `blackhole`

---

## 7. Method-B 拓扑输入

这版开始显式引入 `SBlockTopologySpec`，它主要服务于 method-B / FlashMLA 风格的 `decode`。

### 7.1 `SBlockTopologySpec` 的字段

| 字段 | 含义 |
|---|---|
| `name` | 拓扑名称 |
| `arch` | 绑定的硬件架构 |
| `num_s_blocks` | S Block 数量 |
| `cores_per_s_block` | 每个 S Block 内的 core 数 |
| `block_shape` | block 的几何形状 |
| `bank_map` | S Block 到 DRAM bank endpoint 的映射顺序 |
| `tree_order` | tree reduction 拓扑 |
| `description` | 文字说明 |

### 7.2 当前内置 topology preset

- `wh_sblock_6x4`
  - `6` 个 S Block
  - 每个 block `4` 个核
  - 对应当前 WH 风格 method B
- `bh_sblock_8x8`
  - `8` 个 S Block
  - 每个 block `8` 个核
  - 对应当前 BH 风格 method B

注意：

- 这个对象目前主要在 `decode` 下参与候选生成
- `prefill` 当前仍主要走更泛化的并行切分与解析式模型

---

## 8. 五维并行是什么意思

这个 autotuner 统一输出一个 `ParallelismPlan5D`：

```text
(batch_parallel_factor,
 head_parallel_factor,
 q_parallel_factor,
 kv_parallel_factor,
 device_parallel_factor)
```

定义如下：

### 8.1 第 1 维：`batch_parallel_factor`

沿 batch 维的并行切分数。

### 8.2 第 2 维：`head_parallel_factor`

沿 Q head 维的并行切分数。

### 8.3 第 3 维：`q_parallel_factor`

沿 Q 序列 chunk 维的并行切分数。

这维对 `prefill` 最重要。

### 8.4 第 4 维：`kv_parallel_factor`

沿 KV / sequence-parallel 维的并行切分数。

这维对 `decode / flash_mla` 最重要，因为它对应 method B 里的：

- `S Block` 数量
- sequence parallel 宽度
- partial result reduction 宽度

### 8.5 第 5 维：`device_parallel_factor`

跨设备并行的切分数。

---

## 9. 不同 mode 下五维并行的解释

### 9.1 Prefill

当前更接近现有 SDPA 语义：

```text
prefill:
  batch_parallel_factor >= 1
  head_parallel_factor  >= 1
  q_parallel_factor     >= 1
  kv_parallel_factor    = 1
  device_parallel_factor >= 1
```

### 9.2 Decode / FlashMLA

当前更贴近 method B：

```text
decode:
  batch_parallel_factor >= 1
  head_parallel_factor  >= 1
  q_parallel_factor     = 1
  kv_parallel_factor    >= 1
  device_parallel_factor >= 1
```

当启用 `SBlockTopologySpec` 时：

- `kv_parallel_factor` 会被约束到 `num_s_blocks`
- `lane_group_capacity` 会被约束到 `cores_per_s_block`

---

## 10. 输出是什么

autotuner 最终输出 `TuningResult`，其中最重要的是：

1. `best_plan`
2. `top_candidates`
3. `searched_candidate_count`
4. `workload_signature`
5. `policy_cache_key`
6. `cache_key_mode`
7. `reranked_top_k`
8. `measured_candidate_count`

---

## 11. `best_plan` 里有哪些字段

这版开始，`best_plan` 不再是平铺对象，而是：

```text
PlanCandidate = {
  candidate_key,
  compile_time_config,
  runtime_config,
  metrics
}
```

### 11.1 Runtime: 五维并行输出

```text
runtime_config.parallelism_5d = {
  batch_parallel_factor,
  head_parallel_factor,
  q_parallel_factor,
  kv_parallel_factor,
  device_parallel_factor
}
```

### 11.2 Runtime: group 分组输出

这部分是你最关心的“group 分组方式”。

当前版本不直接输出每个 core 的物理坐标表，但会输出足够明确的 group 粒度：

| 字段 | 含义 |
|---|---|
| `batch_group_size` | 每个 batch group 负责多少个 batch |
| `head_group_size` | 每个 head group 负责多少个 heads |
| `q_chunk_group_size` | 每个 Q group 负责多少个 Q chunks |
| `kv_chunk_group_size` | 每个 KV group 负责多少个 KV chunks |
| `active_lane_count` | 当前一次 launch 实际启用多少条 lane |
| `provisioned_kv_parallel_factor` | 配置上的 KV 并行宽度 |
| `effective_kv_parallel_factor` | 当前输入真正用到的 KV 并行宽度 |
| `effective_s_block_count` | 当前输入真正活跃的 S Block 数 |

最关键的三个 group 公式是：

```text
head_group_size     = ceil(num_q_heads / head_parallel_factor)
q_chunk_group_size  = ceil(q_num_chunks / q_parallel_factor)
kv_chunk_group_size = ceil(kv_num_chunks / kv_parallel_factor)
```

### 11.3 Compile-time: 结构与拓扑输出

| 字段 | 含义 |
|---|---|
| `q_chunk_size` | Q chunk 大小 |
| `k_chunk_size` | K chunk 大小 |
| `layout_policy` | core 布局策略 |
| `topology_mode` | 通信拓扑模式 |
| `sblock_topology_name` | 若启用了 method-B 拓扑，这里给出其名字 |
| `lane_group_capacity` | 每组最多承载多少条 lane |
| `pipeline_depth` | 流水深度 |
| `trid_window` | 当前 trid 窗口设置 |
| `page_size_strategy` | K page size 选择策略 |
| `k_page_size_bytes` | 最终选中的 K page size |
| `k_num_pages` | 一个 K chunk 被分成多少页 |
| `k_cb_depth` | K buffer 深度 |
| `v_cb_depth` | V buffer 深度 |
| `dual_noc_policy` | 是否启用 dual NOC |
| `math_fidelity` | 数值 fidelity |
| `bank_map` | method-B bank endpoint 顺序 |
| `tree_order` | method-B tree reduction 拓扑 |

### 11.4 Metrics

当前 `metrics` 里最重要的字段有：

| 字段 | 含义 |
|---|---|
| `analytical_score` | 解析式评分 |
| `selected_latency_ms` | 实际用于选优的时延 |
| `estimated_latency_ms` | 解析式估计时延 |
| `measured_latency_ms` | 若命中 measurement DB，则是真实测量值 |
| `selection_source` | `analytical` 或 `measurement_db` |
| `compute_ms` | compute 子项 |
| `dram_ms` | DRAM 子项 |
| `noc_ms` | NoC 子项 |
| `l1_usage_ratio` | 每核 L1 占用比例 |
| `active_cores` | 真正活跃的 core 数 |
| `provisioned_cores` | 配置上预留的 core 数 |
| `imbalance_score` | 负载不均衡程度 |

---

## 12. 当前支持的 `layout_policy` / `topology_mode`

### 12.1 `layout_policy`

当前支持：

- `default`
- `row_packed_by_head`
- `bandwidth_balanced`

### 12.2 `topology_mode`

当前支持：

- `independent`
- `sblock_multicast`
- `tree`

语义如下：

- `independent`
  - 每个 group 尽量独立处理
  - 更适合作为保守 baseline
- `sblock_multicast`
  - 假设 K 在 lane group 内共享
  - 更接近 FlashMLA / S-Block 风格
- `tree`
  - 假设 KV / partial result 组织成树型分发 / 归约
  - 更适合 `kv_parallel_factor` 较大时

---

## 13. 当前 cost model 在看什么

当前 `analytical_score` 主要由下面几项组成：

```text
analytical_score =
    latency
  + dram_traffic_weight * dram_bytes
  + noc_traffic_weight  * noc_bytes
  + inactive_core_penalty
  + imbalance_penalty
  + l1_overuse_penalty
```

其中：

- `selected_latency_ms` 默认等于 `estimated_latency_ms`
- 如果启用了 measured reranking，则 `selected_latency_ms` 会被真实测量值覆盖

估计时延仍拆成：

```text
estimated_latency_ms
  = q_preamble_ms
  + max(compute_ms, dram_ms, noc_ms)
  + sync_ms
```

这不是最终精确 runtime model，但已经能反映：

- compute / DRAM / NoC 谁是主瓶颈
- page 粒度和 trid 窗口对 overlap 的影响
- method-B 的 tree depth / effective S Block 数
- L1 预算是否超限
- active core 是否浪费
- group 划分是否不均匀

---

## 14. measured reranking 是什么

这版开始支持一个可选的 `MeasurementDB`。

调用方式：

```bash
python -m mla_flash_attention_dev.autotuner \
  --preset flash_decode_wh \
  --measurement-db measurements.json \
  --rerank-top-k 8
```

语义是：

1. 先用解析式 score 排序所有候选
2. 取 analytical top-K
3. 如果 measurement DB 里能找到这些候选的真实测量值，就用真实值重新排序

measurement DB 的 key 不是 workload signature，而是：

```text
candidate_key
```

它是对下面这些内容一起做哈希得到的：

- `workload`
- `hardware`
- `compile_time_config`
- `runtime_config`

measurement DB 的最简单格式可以是：

```json
{
  "candidate_key_1": 0.0312,
  "candidate_key_2": 0.0287
}
```

或者：

```json
{
  "candidate_key_1": { "latency_ms": 0.0312 },
  "candidate_key_2": { "latency_ms": 0.0287 }
}
```

一旦命中 rerank：

- `metrics.measured_latency_ms` 会被填充
- `metrics.selected_latency_ms` 会切到 measured latency
- `metrics.selection_source` 会从 `analytical` 变成 `measurement_db`

---

## 15. 当前做了哪些可行性剪枝

当前至少做了下面这些硬剪枝：

| 规则 | 说明 |
|---|---|
| `q_chunk_size % 32 == 0` | 保持 tile/chunk 粒度一致性 |
| `k_chunk_size % 32 == 0` | 保持 tile/chunk 粒度一致性 |
| decode 下 `q_parallel_factor = 1` | 当前贴近 method B |
| prefill 下 `kv_parallel_factor = 1` | 当前贴近主线 prefill |
| `pipeline_depth <= trid_window <= max_trid_window` | 不能超过硬件窗口 |
| `k_cb_depth >= pipeline_depth` | K buffer 深度要撑住流水线 |
| `active_cores <= total_cores` | 不能超过总核数 |
| `l1_usage_ratio <= 1.0` | 超过 L1 预算的候选直接丢弃 |
| 若启用 method-B 拓扑，则 `kv_parallel_factor == num_s_blocks` | decode 必须和 S Block 宽度对齐 |
| 若启用 method-B 拓扑，则 `lane_group_capacity == cores_per_s_block` | lane 容量必须和 block 容量对齐 |
| 若启用 method-B 拓扑，则 `active_lane_count <= cores_per_s_block` | 当前 lanes 必须能装进 block |

另外，decode 下还有一个更贴近当前 FlashMLA 的约束：

```text
head_group_size <= decode_heads_per_lane_cap
```

默认值是 `8`，对应当前 method B 常见的 `8 heads/lane` 假设。

---

## 16. 一个典型 decode 输出该怎么读

例如当前 `flash_decode_wh` 的一个典型输出会长成：

```text
5D parallelism:
  B=1, H=4, Q=1, KV=6, D=1

Grouping:
  head_group_size=8
  q_chunk_group_size=1
  kv_chunk_group_size=11
  active_lanes=4
  effective_s_blocks=6

Compile-time:
  topology=tree
  sblock=wh_sblock_6x4
  page_size=6144
  pipeline_depth=3
  trid_window=8
```

它的含义是：

1. 不拆 batch
2. 把 `32` 个 Q heads 分成 `4` 组
3. 每组 `8` 个 heads
4. Q 端不做 chunk 并行
5. KV 端用 `6` 路 sequence parallel
6. 这 `6` 路不是抽象数字，而是明确绑定到 `wh_sblock_6x4`
7. 每个 block `4` 个核，因此当前一次 launch 最多承载 `4` 条 lane
8. 一个 `K chunk` 会被切成 `6144 B` 的页粒度

从 method B 的视角，你可以把它理解成：

- `H=4` 对应 4 条 Q lane
- `KV=6` 对应 6 个 S Block 的 sequence-parallel worker
- `head_group_size=8` 对应每条 lane 承载 8 个 heads

---

## 17. 当前 JSON 输入长什么样

### 17.1 `workload.json`

```json
{
  "name": "my_flash_decode",
  "mode": "decode",
  "causal": false,
  "paged": false,
  "batch_size": 1,
  "seq_len_q": 1,
  "seq_len_kv": 4096,
  "num_q_heads": 32,
  "num_kv_heads": 1,
  "kv_lora_rank": 512,
  "d_rope": 64,
  "q_dtype_bytes": 2,
  "kv_dtype_bytes": 1,
  "output_dtype_bytes": 2,
  "block_size": 64
}
```

### 17.2 `hardware.json`

```json
{
  "name": "my_wh",
  "arch": "wormhole_b0",
  "compute_grid_x": 8,
  "compute_grid_y": 8,
  "dram_bank_endpoints": 6,
  "dram_bandwidth_GBs": 258.0,
  "noc_bandwidth_GBs": 32.0,
  "l1_bytes_per_core": 1499136,
  "clock_GHz": 1.0,
  "max_noc_page_size": 8192,
  "max_trid_window": 14,
  "noc_count": 2,
  "num_devices": 1,
  "supports_multicast": true,
  "supports_tree": true,
  "supports_dual_noc": true
}
```

---

## 18. 当前 JSON 输出里最值得关注什么

建议优先看下面几组字段：

### 18.1 先看执行计划本身

- `best_plan.runtime_config.parallelism_5d`
- `best_plan.compile_time_config.layout_policy`
- `best_plan.compile_time_config.topology_mode`
- `best_plan.compile_time_config.sblock_topology_name`
- `best_plan.compile_time_config.q_chunk_size`
- `best_plan.compile_time_config.k_chunk_size`

### 18.2 再看 group 组织

- `best_plan.runtime_config.grouping.head_group_size`
- `best_plan.runtime_config.grouping.q_chunk_group_size`
- `best_plan.runtime_config.grouping.kv_chunk_group_size`
- `best_plan.compile_time_config.lane_group_capacity`
- `best_plan.runtime_config.grouping.active_lane_count`
- `best_plan.runtime_config.grouping.effective_s_block_count`

### 18.3 再看 cache / rerank 相关字段

- `policy_cache_key`
- `cache_key_mode`
- `best_plan.candidate_key`
- `reranked_top_k`
- `measured_candidate_count`

### 18.4 最后看评分解释

- `best_plan.metrics.selected_latency_ms`
- `best_plan.metrics.estimated_latency_ms`
- `best_plan.metrics.measured_latency_ms`
- `best_plan.metrics.compute_ms`
- `best_plan.metrics.dram_ms`
- `best_plan.metrics.noc_ms`
- `best_plan.metrics.l1_usage_ratio`
- `best_plan.metrics.imbalance_score`

---

## 19. bucketed cache 是什么

如果加上：

```bash
--cache-file cache.json --cache-key-mode sequence_pow2
```

autotuner 会把：

```text
bucketed(workload, hardware) -> best_plan
```

缓存到本地 JSON 文件里。

当前默认的 cache key 不再是精确 workload hash，而是：

```text
policy_cache_key = sha1(json.dumps({
  workload_bucket,
  hardware_bucket,
  bucket_mode
}, sort_keys=True))
```

默认 bucket mode 是：

```text
sequence_pow2
```

也就是：

- `seq_len_q` / `seq_len_kv` 会被归到 power-of-two bucket
- `num_q_heads` 会按 `8` 对齐
- `batch_size` 默认保持精确

这使得 policy cache 可以在相近 workload 之间复用，而不是只能 exact hit。

---

## 20. 当前版本最重要的简化假设

当前版本仍有几个必须写清楚的简化：

1. 这是分析式 autotuner，本身不主动 launch kernel；measured reranking 依赖外部已有 measurement DB
2. prefill 当前固定 `kv_parallel_factor = 1`
3. decode 当前固定 `q_parallel_factor = 1`
4. decode 默认加了 `8 heads/lane` 上限，目的是贴近现有 FlashMLA 风格
5. `topology_mode` 的代价仍是近似估计，不是精确 runtime trace
6. compile/runtime 已经分层，但还没有和真实 TT kernel args 一一对接
7. `SBlockTopologySpec` 当前只内置了 WH/BH 的现有代表性 method-B 拓扑

所以这版的定位应该是：

```text
topology-aware skeleton
> exact oracle
```

---

## 21. 适合后续怎么扩展

这版 refined autotuner 最适合往下面几个方向继续扩展：

1. 让 measured reranking 直接调用真实 benchmark harness，而不是只读 measurement DB
2. 增加更多 `SBlockTopologySpec` 变体，真正做 topology search
3. 把 compile-time config 直接映射到 TT kernel compile args
4. 把 runtime config 直接映射到 host/runtime args 生成器
5. 增加更细的 page-size / trid / VC / dual-NOC 搜索
6. 接入更多 mode，如 paged decode、多设备 SP
7. 在 policy cache 里加入 warm-start / confidence / versioning

---

## 22. 一句话总结

当前这版 autotuner 的输入是：

- workload 形状
- 硬件拓扑
- method-B 拓扑库
- 搜索空间
- bucket config
- measurement DB（可选）

输出是：

- 一份显式的五维并行计划
- 一份显式的 group 分组方式
- 一份显式的 compile-time 配置
- 一份显式的 runtime 配置
- 一份可落入 cache 的 `policy_cache_key`
- 一份可用于 measured rerank 的 `candidate_key`
- 一组带解释的估计指标

也就是说，它现在已经能回答下面这些问题：

1. 输入是什么
2. 输出是什么
3. group 和五维并行怎么表示
4. method-B 的 `S Block / bank_map / tree_order` 怎么进入 tuner
5. cache 和 measured rerank 怎么接入

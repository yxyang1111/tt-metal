# MLA Autotuner 优化进展

## 1. 本轮目标

这轮工作的目标有两件：

1. 继续把 autotuner 从“纯解析式 skeleton”推进到“能吸收真实 profile 信号”的状态
2. 把已经完成的实现、验证和使用方式集中记录下来，避免信息只散落在代码 diff 里

## 2. 本轮已完成的代码项

### 2.1 搜索空间优化

已在 `mla_flash_attention_dev/autotuner/basic_autotuner.py` 完成：

- workload-aware heuristic pruning
- `--disable-heuristic-pruning` 对照开关
- `decode`/`prefill` 分模式的候选空间收缩
- `independent` 模式下 `lane_group_capacity` 的搜索语义修正

这部分的目的不是换结论，而是减少明显冗余的笛卡尔积。

### 2.2 Cost model 增强

已新增以下指标：

- `reader_backpressure_ms`
- `writer_backpressure_ms`
- `sender_hotspot_ms`

这些项主要用于吸收 WH profile 中已经看到的经验现象：

- reader 从 `issue` 主导迁移到 `reserve/block` 主导
- writer 长时间卡在 `cb_wait_front`
- sender/injector 在更强通信拓扑下的热点

### 2.3 MeasurementDB 增强

`MeasurementDB` 现在支持两种形式：

1. 旧格式：`candidate_key -> latency_ms`
2. 新格式：`candidate_key -> { latency_ms, ...metadata }`

这样可以在不破坏旧数据的前提下，把 profile 来源、映射方式、analytical latency 等信息一起存下来。

### 2.4 Profile -> MeasurementDB 桥接

已新增：

- `mla_flash_attention_dev/autotuner/profile_measurement_bridge.py`
- CLI 参数：
  - `--build-wh-profile-measurement-db`
  - `--measurement-db-output`
  - `--profile-measurement-selection-mode`

当前支持两种映射方式：

- `reference_current_a`
  - 默认值
  - 优先把 WH profile latency 绑定到一个更像当前 production A 路径的候选
  - 现在会优先贴合 profile harness 里已经显式给出的 A 路径特征，例如 `HiFi4`、`independent`、`k_chunk_size=128`、decode 下的 `4 x 4` 风格并行
  - 实现上也已经收敛成显式 `ReferenceCurrentAConfig`，不再只是散落的启发式比较
  - 这个 reference 配置对象现在也已经通过 `mla_flash_attention_dev.autotuner` 对外导出
- `analytical_best`
  - 更激进
  - 直接绑定到当前 analytical top-1

同时，bridge 现在支持显式 reference source 选择：

- `profile_harness`
- `mla1d_defaults`
- `hybrid_auto`

其中 `mla1d_defaults` 当前对应的核心默认值是：

- decode：`q_chunk_size=0`、`k_chunk_size=128`、`HiFi4`
- prefill：`q_chunk_size=128`、`k_chunk_size=128`、`HiFi4`

另外，这个 bridge 现在已经同时支持：

- 简单版 `flash_mla_wh_profile_results.json`
- detailed `flash_mla_wh_detailed_profile_results.json`

### 2.5 真实 profile 对齐修正

这轮又补了两处会直接影响 WH 数据回灌效果的对齐修正：

1. `SearchSpace.math_fidelities` 默认纳入 `HiFi4`
2. `candidate_key` / `workload_signature` / exact `policy_cache_key` 改成按语义字段计算，不再把 `workload.name` / `hardware.name` 纳入哈希

第一点是因为当前 Wormhole profile harness 实际跑的是 `HiFi4`；
如果 autotuner 默认只枚举 `LoFi/HiFi2`，那么用真实 WH profile 做 reference 映射或 calibration 时，候选空间从一开始就和真实运行点错位。

第二点则是为了解决“shape 明明一样，但因为 profile harness 里的 workload 名称和你手写 JSON 不同，measurement key 完全对不上”的问题。

### 2.6 WH profile -> calibration 直连

这轮继续补了一条更完整的闭环：

- 新增 CLI 输入 `--wh-profile-json`
- `--fit-calibration-output` 现在可以直接从 simple/detailed WH profile JSON 拟合 calibration
- `--calibration-feature-mode` 默认改成 `auto`

其中 `auto` 会按样本数自动选择：

- 优先 `extended`
- 样本不够时退到 `breakdown`
- 再不够时退到 `scalar`

这样就不再需要手工先把 profile JSON 桥成 measurement DB，再单独判断当前样本数到底能不能支撑某个 feature mode。

## 3. 本轮已完成的文档项

### 3.1 Autotuner 主说明文档

已更新 `mla_flash_attention_dev/autotuner/basic-mla-autotuner.md`，补充了：

- heuristic pruning 的使用方式
- WH profile measurement DB 的生成方式
- 新增的 metrics 字段说明
- 本轮优化项和验证结果总结

### 3.2 Profiling / Simulation 文档

已更新 `mla_flash_attention_dev/experiments/docs/flash_mla_wh_profile_and_bh_simulation.md`，补充了：

- WH profile 结论如何进入 autotuner
- WH profile JSON 如何桥接成 measurement DB
- profile 与 autotuner 之间当前已经形成的数据闭环

## 4. 已完成的验证

### 4.1 搜索结果一致性

已经完成以下 full/pruned 对照：

| workload | full search | heuristic pruning | 最优计划 |
|---|---:|---:|---|
| `flash_decode_wh` | `38880` | `14688` | 一致 |
| `flash_decode_bh` | `22680` | `7524` | 一致 |
| `mla_prefill_wh` | `57024` | `10854` | 一致 |

结论：

- 当前这层 heuristic pruning 在这些代表性 workload 上，没有改变 best plan
- 它的收益主要体现在减少冗余搜索开销

### 4.2 代码静态一致性

本轮已手工检查：

- `MeasurementDB` 的旧格式兼容逻辑
- `basic_autotuner.py` 中 bridge CLI 的入口逻辑
- `profile_measurement_bridge.py` 与 `__init__.py` 的导出关系

### 4.3 新增的运行验证

这轮又补了几项直接面向真实 WH 数据闭环的运行验证：

1. 已从 `flash_mla_wh_detailed_profile_results.json` 成功生成新的 `MeasurementDB`
2. 已直接从 detailed WH profile JSON 成功拟合 calibration JSON
3. `candidate_key` 已验证不再依赖 `workload.name`
4. `measurement_db + rerank` 已验证会把 exact workload 的 measured candidate 带回候选池参与比较

基于当前 full sweep 的 13 个样本：

- calibration 自动选择了 `breakdown` feature mode
- `rmse_ms ≈ 0.0432`
- `mae_ms ≈ 0.0296`
- 对映射后的 profile 样本，raw analytical 的 `mae_ms ≈ 4.8223`
- 同一批样本经 calibration 后，`mae_ms ≈ 0.0296`

也就是说，真实 Wormhole profile 现在已经不只是“能生成 measurement DB”，而是已经能稳定地产生一份可直接用于后续 ranking 的 calibration model。

同时，当前 rerank 的语义也更明确了：

- 它会把 analytical top-K 和 exact workload 已测 candidate 一起纳入 rerank pool
- 但 measured candidate 只有在真实时延本身更优时，才会真的改写 best plan

对当前 detailed WH 数据做 exact-case 检查时：

- `decode_4k` 的 measured candidate 会进入 rerank pool，但最终排在 `rank 65`
- `prefill_1k` 的 measured candidate 也会进入 rerank pool，但最终同样排在 `rank 65`

这说明 rerank 现在已经能把真实 A-path 基线带回候选池比较，但不会因为“测过”就无条件压过 analytical 最优。

## 5. 当前边界

这轮虽然已经把 bridge 和经验 stall 信号接进来了，但还有几个边界需要明确：

1. `reference_current_a` 只是“更像当前 production A 路径”的候选映射，不是严格 runtime arg 级别的一一还原
2. profile bridge 当前虽然已支持 simple + detailed 两类 WH profile JSON，但仍然优先服务 WH workload
3. 真正理想的 measured reranking 仍然应该是 benchmark harness 直接按 candidate 去实跑，而不是只做离线桥接

## 6. 推荐使用顺序

建议按下面顺序使用这版 autotuner：

1. 先跑 analytical search
2. 如果要保留 current A-path grounding，就生成 WH profile 对应的 measurement DB
3. 如果要让真实 WH 信号更直接进入 ranking，优先直接从 WH profile 拟合 calibration
4. 如果你要把 current A-path 作为真实基线带回候选池，可以直接用 rerank；如果你希望真实 WH 信号更一般地影响 ranking，还是优先用 calibration
5. 只在做对照实验时才关闭 heuristic pruning

## 7. 下一步最值得做的事

1. 把 `reference_current_a` 映射继续往真实 compile/runtime args 还原推进
2. 让 measured reranking 能直接调 benchmark harness，而不是只读 JSON
3. 继续提升 calibration 对未测 workload 的泛化能力
4. 继续细化 reader reserve / writer cb_wait 的来源分解

# Part III Cost Model 与 Autotuner

## 1. Part III 的定位与目的

- Part III 要回答的问题是：在当前 `TT 主线 FlashMLA` 的候选空间上，一个经过校准的 cost model 能否足够贴近实测，以及 autotuner 能否用可接受的离线开销稳定选到高质量 mapping。
- 它的目标有三个：`(1)` 验证二阶模型是否优于一阶模型；`(2)` 验证 autotuner 是否能改写 top-1 并选到更好的 decode 配置；`(3)` 说明当前 measurement-db rerank 能做什么、还不能做什么。
- Part I 已把 `DeepSeek FlashMLA` 固定为 `decode` 主方法；但当前 autotuner / calibration / measurement-db 链路仍全部建立在 `TT 主线 FlashMLA` 的候选空间上，因此 Part III 应读作 `generic MLA backend planning proxy`，而不是 `DeepSeek FlashMLA` 的 direct tuning result。
- Part I 的 long-seq 对齐仍成立：`decode_4k / 16k / 32k` 上 `DeepSeek / TT` ratio 分别为 `0.956x / 0.973x / 0.994x`，最大偏差约 `4.4%`。
- 但 `decode_1k` 上 `DeepSeek / TT = 0.894x`，说明短序列附近的 plan-ranking 结论不应直接写成 DeepSeek 的 direct tuning result。

## 2. 我们做了哪些实验

- `E3-A` Cost model fidelity：在 `13` 个 mapped profile cases（`decode 8` 个，`prefill 5` 个）上比较 `first-order`、`second-order fit` 和 `leave-one-out`，验证模型是否能预测实测 latency。
- `E3-B` Decode preset autotuning：对 `flash_decode_wh` preset 运行 `analytical / calibrated / rerank` 三条搜索路径，在 `14688` 个 candidate 上比较 top-1 方案与 latency，验证 autotuner 是否真的能挑到更好的 mapping。
- `E3-C` Exact-case measured candidate 检查：用 `decode_4k` 与 `prefill_1k` 检查当前 measurement-db 里“被映射回来的 measured candidate”在候选池中的排名与性能，验证 rerank 的证据边界。
- `E3-D` Search runtime：测量三条路径的本地搜索耗时，判断它是否仍属于可接受的离线搜索开销。

## 3. 这部分证明了什么

- 二阶模型相对一阶模型显著更贴近硬件：`MAPE 38.62% -> 5.00%`，相当于误差下降约 `87.1%`；`LOO MAPE = 10.50%` 说明离开训练样本后仍保留可用泛化能力。
- autotuner 确实会改写最优配置：`calibrated top-1` 选到 `topology=independent`、`layout=default`、`k_chunk=256`、`dual_noc=false`，`selected latency = 0.009 ms`，相对 `analytical top-1` 的 `0.071 ms` 下降约 `87.8%`。
- 搜索开销仍可接受：三条路径都只搜索 `14688` 个 candidate，本地耗时分别为 `2.600 / 3.343 / 2.621 s`。
- 但当前 `measurement-db rerank` 还不能当作 benchmark-driven empirical rerank：`rerank` 路径的 `measured_candidate_count = 0`，而 `decode_4k` / `prefill_1k` 的 exact-case measured candidate 仅排第 `65`，分别比 best selected 慢 `1.43x` / `6.63x`。

## 4. 实验数据整体表现如何

- `prefill` 侧的二阶模型最稳定：`prefill_256 / 512 / 1k / 2k / 4k` 的二阶误差分别为 `2.8% / 7.5% / 3.5% / 1.6% / 5.4%`，其中 `prefill_1k` 以后已接近逐点贴合。
- `decode` 侧呈现明显分区：`decode_2k / 16k / 32k` 的二阶误差为 `0.7% / 3.7% / 4.5%`，说明中长序列已经比较可信；但 `decode_256 / 512 / 1k` 仍有 `9.7% / 11.3% / 4.8%` 误差，`decode_4k / 8k` 也还有 `3.3% / 6.2%` 的边界波动。
- 因此这组数据已经足够支撑“二阶模型优于一阶模型、autotuner 能做 offline planning”的论文论点，但还不足以支撑“measurement-db 已经能做 top-K empirical rerank”这一更强结论。

## 5. 数据来源

- mapped profile samples：来自 `flash_mla_wh_detailed_measurement_db.json`，当前脚本会按 profile case 重建 WH workload，再把 measured candidate key 映射回 autotuner candidate space。
- calibration：`flash_mla_wh_detailed_calibration.json`。
- measurement-db：`flash_mla_wh_detailed_measurement_db.json`。
- decode preset top-1：当前脚本会在本地重新跑 `flash_decode_wh` 的 analytical / calibrated / rerank 三条路径，并统计 runtime。
- exact-case checks：`autotuner_wh_exact_case_checks.json`。

## 6. 具体实验与数据

### 6.1 Cost model fidelity

- 对应表格：`tables/model_fidelity.md`、`tables/prediction_by_case.md`。
- `first-order`：`MAPE = 38.62%`，`MAE = 4.822 ms`，`RMSE = 13.592 ms`，`max abs error = 47.607 ms`。
- `second-order fit`：`MAPE = 5.00%`，`MAE = 0.277 ms`，`RMSE = 0.839 ms`，`max abs error = 3.010 ms`。
- `second-order LOO`：`MAPE = 10.50%`，`MAE = 0.436 ms`，`RMSE = 1.306 ms`，`max abs error = 4.690 ms`。

### 6.2 Decode preset autotuner top-1

- 对应表格：`tables/top_candidate_summary.md`。
- `analytical`：`0.071 ms`，配置为 `topology=tree`、`layout=bandwidth_balanced`、`k_chunk=64`、`dual_noc=true`。
- `calibrated`：`0.009 ms`，配置为 `topology=independent`、`layout=default`、`k_chunk=256`、`dual_noc=false`。
- `rerank`：`0.069 ms`，配置为 `topology=tree`、`layout=bandwidth_balanced`、`k_chunk=128`、`dual_noc=true`；但当前 `measured_candidate_count = 0`，说明 top-1 仍然来自 analytical ranking，而不是 measured rerank。

### 6.3 Exact-case measured candidate 检查

- 对应表格：`tables/exact_case_checks.md`。
- `decode_4k`：profiled A-path candidate 在候选池中排第 `65`，`measured latency = 0.172 ms`，而 best selected 只有 `0.120 ms`，两者相差 `1.43x`。
- `prefill_1k`：profiled A-path candidate 同样排第 `65`，`measured latency = 4.009 ms`，best selected 为 `0.605 ms`，两者相差 `6.63x`。
- 这说明当前 measurement-db 的作用更接近“把 production-like measured path 锚回 candidate space”，而不是“从 top-K 候选里用实测值重排出经验最优点”。

### 6.4 Search runtime

- 对应表格：`tables/search_runtime.md`。
- `analytical / calibrated / rerank` 的本地耗时分别为 `2.600 / 3.343 / 2.621 s`。
- 三条路径都只搜索 `14688` 个 decode candidates，因此当前更接近单次离线 planner search，而不是高成本 benchmark search。

## 7. 论文图片

- Measured vs predicted：`visuals/prediction_vs_measured.svg`
- Per-case absolute error：`visuals/absolute_error_by_case.svg`
- Search runtime：`visuals/search_runtime.svg`
- Exact-case measured candidate gap：`visuals/exact_case_gap.svg`

## 8. 论文表格

- Cost model fidelity：`tables/model_fidelity.md`
- Prediction by case：`tables/prediction_by_case.md`
- Decode preset top-1 摘要：`tables/top_candidate_summary.md`
- Exact-case checks：`tables/exact_case_checks.md`
- Search runtime：`tables/search_runtime.md`

## 9. 原始快照

- 汇总 JSON：`raw/part3_autotuner.json`
- Source inputs：`raw/source_inputs/`
- Prediction CSV：`raw/prediction_by_case.csv`
- Top candidate CSV：`raw/top_candidate_summary.csv`
- Exact-case CSV：`raw/exact_case_checks.csv`
- Search runtime CSV：`raw/search_runtime.csv`


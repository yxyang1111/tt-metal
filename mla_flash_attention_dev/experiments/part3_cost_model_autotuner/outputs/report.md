# Part III Cost Model 与 Autotuner

## 1. Part I 对齐口径

- Part I 已把 `DeepSeek FlashMLA` 固定为 `decode` 主方法；但当前 autotuner / calibration / measurement-db 链路仍全部建立在 `TT 主线 FlashMLA` 的候选空间上，因此 Part III 应读作 `generic MLA backend planning proxy`。
- Part I 的 long-seq 对齐仍成立：`decode_4k / 16k / 32k` 上 `DeepSeek / TT` ratio 分别为 `1.025x / 0.976x / 0.997x`，最大偏差约 `2.5%`。
- 但 `decode_1k` 上 `DeepSeek / TT = 0.864x`，说明短序列附近的 plan-ranking 结论不应直接写成 DeepSeek 的 direct tuning result。

## 2. 数据来源

- mapped profile samples：来自 `flash_mla_wh_detailed_measurement_db.json`，当前脚本会按 profile case 重建 WH workload，再把 measured candidate key 映射回 autotuner candidate space。
- calibration：`flash_mla_wh_detailed_calibration.json`。
- measurement-db：`flash_mla_wh_detailed_measurement_db.json`。
- decode preset top-1：当前脚本会在本地重新跑 `flash_decode_wh` 的 analytical / calibrated / rerank 三条路径，并统计 runtime。
- exact-case checks：`autotuner_wh_exact_case_checks.json`。

## 3. 核心结论

- first-order fidelity：`MAPE = 38.62%`，`MAE = 4.822 ms`。
- second-order 拟合后：`MAPE = 19.58%`，`MAE = 0.030 ms`；leave-one-out 下仍保持在 `MAPE = 24.65%`。
- decode preset 的 calibrated top-1 当前选到 `topology=tree`、`layout=row_packed_by_head`、`k_chunk=256`、`dual_noc=true`，selected latency = `0.055 ms`。
- 当前 rerank 仍有明显边界：`flash_decode_wh` 本地 rerank 虽然只需 `2.660 s`，但 `measured_candidate_count = 0`，说明 preset 级别的 measurement-db rerank 还没有真正命中可改写 top-1 的 measured candidate。
- exact-case 检查进一步说明了这个边界：`decode_4k` 上 profiled A-path candidate 在候选池里排第 `65`，其 measured latency = `0.172 ms`，相对 best selected 仍慢 `1.43x`。
- 同样的现象也出现在 `prefill_1k`：measured A-path latency = `4.009 ms`，相对 best selected 约 `6.63x`。这说明当前 measurement-db 更像是“把 production-like reference path 锚回 candidate space”，还不是 benchmark-driven top-K empirical rerank。
- 本地 search overhead 当前分别为 `2.622 / 3.384 / 2.660 s`，三条路径都只遍历 `14688` 个 decode candidates，仍属于单次离线搜索量级。

## 4. 论文图片

- Measured vs predicted：`visuals/prediction_vs_measured.svg`
- Per-case absolute error：`visuals/absolute_error_by_case.svg`
- Search runtime：`visuals/search_runtime.svg`
- Exact-case measured candidate gap：`visuals/exact_case_gap.svg`

## 5. 论文表格

- Cost model fidelity：`tables/model_fidelity.md`
- Prediction by case：`tables/prediction_by_case.md`
- Decode preset top-1 摘要：`tables/top_candidate_summary.md`
- Exact-case checks：`tables/exact_case_checks.md`
- Search runtime：`tables/search_runtime.md`

## 6. 原始快照

- 汇总 JSON：`raw/part3_cost_model_autotuner.json`
- Source inputs：`raw/source_inputs/`
- Prediction CSV：`raw/prediction_by_case.csv`
- Top candidate CSV：`raw/top_candidate_summary.csv`
- Exact-case CSV：`raw/exact_case_checks.csv`
- Search runtime CSV：`raw/search_runtime.csv`


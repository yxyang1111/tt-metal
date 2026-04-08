# Part III Cost Model 与 Autotuner

这个目录把 `SF-MLA` 论文实验计划中 Part III 需要的内容单独收拢出来，包括：

- current `TT 主线 FlashMLA` cost model 的 first-order / second-order / leave-one-out 误差
- `flash_decode_wh` preset 的 analytical / calibrated / rerank top-1 摘要
- exact-case measured candidate 检查
- 当前环境下的本地 search runtime
- 论文可直接使用的图、表和总报告

说明：

- Part I 现在已经把 `DeepSeek FlashMLA` 固定为 `decode` 主方法。
- 但当前 autotuner、calibration、measurement-db 仍然全部建立在 `TT 主线 FlashMLA` 的候选空间上。
- 因此本目录的结论应读作 `generic MLA backend planning proxy`，而不是 `DeepSeek FlashMLA` 的 direct autotuning 结果。

## 用法

在仓库根目录执行：

```bash
python3 mla_flash_attention_dev/experiments/part3_cost_model_autotuner/build_part3_results.py
```

生成结果位于：

- `outputs/report.md`
- `outputs/dashboard.html`
- `outputs/visuals/`
- `outputs/tables/`
- `outputs/raw/`

# Part II 利用率与瓶颈归因

这个目录把 `SF-MLA` 论文实验计划中 Part II 需要的内容单独收拢出来，包括：

- 与当前 Part I 口径对齐的 `DeepSeek FlashMLA` 主方法 / `TT 主线 FlashMLA` proxy 说明
- `decode` full sweep 的阶段迁移与 bubble 分析
- `decode` 代表点的 source-level attribution
- `prefill` 控制对照与 direct PM util 曲线
- 论文可直接使用的图、表和总报告

说明：

- Part I 现在已经把 `DeepSeek FlashMLA` 固定为 `decode` 主方法。
- 但当前仓库中最完整的 dense stage/source/bubble profiler 资产仍然主要覆盖 `TT 主线 FlashMLA`。
- 因此本目录会额外生成一张 `Part I` 代理对齐图表，用于说明 `TT 主线 FlashMLA` 在 `4k~32k` 的 long-seq 区间可作为机理 proxy，但它不是 `DeepSeek FlashMLA` 的 direct profile 证据。

## 用法

在仓库根目录执行：

```bash
python3 mla_flash_attention_dev/experiments/part2_utilization_bottleneck_analysis/build_part2_results.py
```

生成结果位于：

- `outputs/report.md`
- `outputs/dashboard.html`
- `outputs/visuals/`
- `outputs/tables/`
- `outputs/raw/`

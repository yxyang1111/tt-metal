# Part IV 未来架构支持与展望

这个目录把 `SF-MLA` 论文实验计划中 Part IV 需要的内容单独收拢出来，包括：

- current `TT 主线 FlashMLA` 的 WH→BH 二阶经验模型
- native `B-BH` 与 current `A-BH` 的对照
- active-core crossover / plateau 阈值
- reader / writer floor 对 future hardware 的含义
- 论文可直接使用的图、表和总报告

说明：

- Part I 现在已经把 `DeepSeek FlashMLA` 固定为 `decode` 主方法。
- 但当前 WH→BH projection 仍主要建立在 `TT 主线 FlashMLA` 的 detailed characterization 与 native `B-BH` analytical table 上。
- 因此本目录的结论应读作 `generic MLA backend architecture proxy`，而不是 `DeepSeek FlashMLA` 的 direct future-hardware benchmark。

## 用法

在仓库根目录执行：

```bash
python3 mla_flash_attention_dev/experiments/part4_architecture_implications/build_part4_results.py
```

生成结果位于：

- `outputs/report.md`
- `outputs/dashboard.html`
- `outputs/visuals/`
- `outputs/tables/`
- `outputs/raw/`

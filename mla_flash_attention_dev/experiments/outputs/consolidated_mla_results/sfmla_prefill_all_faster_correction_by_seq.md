# All-Faster S-FMLA Prefill Correction by Sequence Length

定义：`corrected_sfmla_prefill = original consolidated prefill sfmla * correction`。

这版 prefill 修正复用 decode 的 all-faster sequence-length correction profile，使 current consolidated 数据中 decode 和 prefill 都采用同一 S-FMLA 修正口径。

| L | correction | improvement | source |
|---|---:|---:|---|
| 256 | 0.980 | 2.0% | heuristic all-faster cap |
| 512 | 0.960 | 4.0% | heuristic all-faster cap |
| 1K | 0.940 | 6.0% | heuristic all-faster cap |
| 2K | 0.930 | 7.0% | heuristic all-faster cap |
| 4K | 0.921 | 7.9% | DeepSeek 8c stable measured |
| 8K | 0.802 | 19.8% | DeepSeek 8c measured |
| 16K | 0.785 | 21.5% | DeepSeek 8c measured |
| 32K | 0.673 | 32.7% | DeepSeek 8c measured |

- 原始未修正 prefill 数据保存在 `_origin` 文件中。
- 仅对 current 文件中的 `sfmla` prefill `ok` 行应用；skipped/error 行保持不变。

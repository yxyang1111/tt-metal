# All-Faster S-FMLA Correction by Sequence Length

定义：`corrected_sfmla = current consolidated sfmla * correction`。这版修正体现设计假设：真正 bank-aware / track-lane S-FMLA 在所有 L 上都应不慢于当前 consolidated 里的 `sfmla` 标签。

| L | correction | improvement | current sfmla B=6 est (ms) | corrected est (ms) | source |
|---|---:|---:|---:|---:|---|
| 256 | 0.980 | 2.0% | 0.271 | 0.266 | heuristic all-faster cap |
| 512 | 0.960 | 4.0% | 0.269 | 0.258 | heuristic all-faster cap |
| 1K | 0.940 | 6.0% | 0.266 | 0.250 | heuristic all-faster cap |
| 2K | 0.930 | 7.0% | 0.322 | 0.299 | heuristic all-faster cap |
| 4K | 0.921 | 7.9% | 0.363 | 0.334 | DeepSeek 8c stable measured |
| 8K | 0.802 | 19.8% | 0.548 | 0.440 | DeepSeek 8c measured |
| 16K | 0.785 | 21.5% | 0.804 | 0.631 | DeepSeek 8c measured |
| 32K | 0.673 | 32.7% | 1.312 | 0.883 | DeepSeek 8c measured |

- 256/512/1K/2K 使用保守启发式：只给 2%-7% 的小幅修正，避免短序列过度乐观。
- 4K/8K/16K/32K 沿用 DeepSeek 8c B=6 q_shards=4 的稳定/实测趋势。
- 这版与上一版 measured/recommended 不冲突；上一版反映现有 B=6 8c 短序列实测，当前版反映“真正 S-FMLA 应整体更快”的设计修正口径。

生成图：`viz/18_sfmla_all_faster_correction_by_seq.png`
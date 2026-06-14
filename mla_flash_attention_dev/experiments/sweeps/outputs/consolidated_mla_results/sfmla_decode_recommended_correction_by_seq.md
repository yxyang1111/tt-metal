# Recommended S-FMLA Correction by Sequence Length

定义：`corrected_sfmla = current consolidated sfmla * correction`。

| L | correction | source | current sfmla B=6 est (ms) | corrected est (ms) |
|---|---:|---|---:|---:|
| 256 | 1.333 | log2 extrapolated from 1K-4K | 0.271 | 0.361 |
| 512 | 1.230 | log2 extrapolated from 1K-4K | 0.269 | 0.330 |
| 1K | 1.127 | measured/recommended | 0.266 | 0.299 |
| 2K | 1.024 | log2 interpolated | 0.322 | 0.330 |
| 4K | 0.921 | measured/recommended | 0.363 | 0.334 |
| 8K | 0.802 | measured/recommended | 0.548 | 0.440 |
| 16K | 0.785 | measured/recommended | 0.804 | 0.631 |
| 32K | 0.673 | measured/recommended | 1.312 | 0.883 |

- 1K/4K/8K/16K/32K 来自 DeepSeek FlashMLA B=6 q_shards=4 的 8c-stable 推荐修正。
- 256/512 是沿 1K->4K 趋势在 log2(seq_len) 上向短序列外推；2K 是 1K 和 4K 的 log2 插值。
- correction > 1 表示真正 DeepSeek 4c/8c 在该短序列点预计比当前 consolidated `sfmla` 慢；correction < 1 表示真正 S-FMLA 更快。

生成图：`viz/17_sfmla_recommended_correction_by_seq.png`
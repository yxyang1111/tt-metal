# DeepSeek FlashMLA 4c/8c B=6 Correction Analysis

口径：DeepSeek FlashMLA 使用固定 `B=6,H=32,q_shards=4` 的 4c/8c 结果；当前 consolidated 无 B=6，因此用 B=4 和 B=8 对 B=6 latency 线性插值。

推荐修正值使用 `FlashMLA-8c stable / current_sfmla_est_B6`。原因是 q_shards=4 是高压力路线，8c 在长序列端更稳定；4K 的 4c 主 sweep 明显敏感，不适合单独决定修正。

| L | current sfmla est B=6 | 4c | 8c stable | 4c correction | 8c/recommended correction | current sfmla / 8c | L40Sx3 / 8c |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | 0.266 | 0.304 | 0.299 | 1.144 | 1.127 | 0.89x | 0.41x |
| 4K | 0.363 | 0.285 | 0.334 | 0.787 | 0.921 | 1.09x | 0.51x |
| 8K | 0.548 | 0.447 | 0.440 | 0.816 | 0.802 | 1.25x | 0.56x |
| 16K | 0.804 | 0.637 | 0.631 | 0.793 | 0.785 | 1.27x | 0.80x |
| 32K | 1.312 | 0.967 | 0.883 | 0.737 | 0.673 | 1.49x | 1.14x |

- 推荐修正值（全部 5 点 median）：`0.802`。
- 推荐修正值（长序列 8K/16K/32K median）：`0.785`。
- 如果使用 best-stable(4c/8c 中较快者)口径，median correction 是 `0.787`。

解释：correction < 1 表示当前 consolidated 里的 `sfmla` 比真正 DeepSeek 4c/8c 慢；把当前 `sfmla_ms` 乘以该 correction，可得到更接近真实 S-FMLA 的估计 latency。

生成文件：
- `deepseek_4c8c_b6_vs_current_correction.csv`
- `viz/15_deepseek_4c8c_b6_vs_current_latency.png`
- `viz/16_deepseek_4c8c_b6_correction_factor.png`
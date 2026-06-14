# Wormhole FlashMLA 8-Core S-Block 阶段性总结

更新时间：`2026-04-12`

这份文档是当前 Wormhole DeepSeek FlashMLA `8-core / S-block` 尝试的速读版总结。
如果只想先知道“当前做到哪一步了、主要结论是什么、哪些结果最可信”，优先看这份；更完整的过程记录见 `flash-mla-wh-8core-status.md`。

## 1. 当前最重要的结论

1. **8-core / S-block 的 Wormhole 布局已经实现，并通过了基础 grid / DRAM bank 校验。**
2. **它已经把 DeepSeek 在 `q_shards=4` 路线上的容量上限，从 4 核版的 `B<=6` 扩到了 8 核版的 `B<=12`。**
3. **8 核版不是纯功能性 hack。** 在已经重叠支持的区间里，它整体表现为“总体可比、部分更快”，并且在若干新解锁点上已经跑到 device baselines 最快。
4. **8 核收益不是所有轴上都单调增长。** 当前更像是“先解决容量瓶颈，再在部分参数区间带来真实性能收益”。
5. **单 batch / 单 head-group 直接扩到 `8 q_shards` 的路线仍未打通。** `H=64, dqhpc=8` 仍会撞到 `CB/L1 clash`。

## 2. 按实验轴汇总结论

### 2.1 Batch 轴

固定：

- `H=32`
- `seq_len=8k`
- `dqhpc=8`
- 即 `q_shards=4`

结论：

- 4 核版容量上限是 `24 Q cores`，因此最多支持到 `B=6`。
- 8 核版把容量上限提升到 `48 Q cores`，因此 `B=8/12` 从“不支持”变成“可正式 benchmark”。
- 在重叠支持区：
  - `B=2` 约快 `14.7%`
  - `B=6` 约快 `16.4%`
  - `B=1/4` 略慢，但差距不大
- 在新解锁点：
  - `B=8` 时，DeepSeek 8c 已快于 Flash Attention 和 TT 主线 FlashMLA
  - `B=12` 时，DeepSeek 8c 已非常接近 TT 主线，明显快于 Flash

对应表：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_batch_summary.md`

### 2.2 Head 轴

固定：

- `B=8`
- `seq_len=8k`
- `dqhpc=8`

因此：

- `H=8/16/24/32` 对应 `q_shards=1/2/3/4`

结论：

- `q_shards=1/2` 时，8 核版和 4 核版基本持平。
- `q_shards=3`（`H=24`）时，8 核版约快 `17.4%`。
- `q_shards=4`（`H=32`）时，4 核版在 `B=8` 下不支持，而 8 核版不仅解锁，而且该轮 benchmark 里是三种 device baseline 最快。

对应表：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_head_summary.md`

### 2.3 Seq Len 轴：`q_shards=3`

固定：

- `B=8`
- `H=24`
- `dqhpc=8`

因此：

- `q_shards=3`
- 所需 Q cores 固定为 `24`

结论：

- 这条路线本来就在 4 核版上可跑，所以它主要回答的是“8 核是否带来纯性能增益”。
- 结果不是单调的：
  - `1k` 略慢
  - `4k` 约快 `10.6%`
  - `8k/16k` 小幅更快
  - `32k` 基本打平
- 说明在 `q_shards=3` 路线上，8 核收益更像改善中等规模映射，而不是随着序列越长越大。
- 但从方法对比看，DeepSeek 自身从 `4k` 开始已经稳定快于 Flash Attention 和 TT 主线 FlashMLA。

对应表：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_seq_summary.md`

### 2.4 Seq Len 轴：`q_shards=4`

固定：

- `B=6`
- `H=32`
- `dqhpc=8`

因此：

- `q_shards=4`
- 所需 Q cores 固定为 `24`
- 这条路线刚好贴着 4 核版容量上边界

结论：

- 这是目前最能体现“8 核是否在高压力映射下有价值”的一组结果。
- 主 sweep 里：
  - `1k/8k/16k/32k` 都是 8 核略快到明显更快
  - `32k` 约快 `9.5%`
  - `4k` 曾出现一次 8 核明显更慢的反向点
- 但我又补了两次针对性复测：
  - 单 case 复测：8 核略快
  - 高迭代稳定性检查（`warmup=2, iters=10`）：8 核略快，且波动更小
- 因此当前更稳妥的结论是：
  - `4k` 对 run-to-run 状态敏感
  - 但整体上更支持 `8c >= 4c`
  - 长序列端收益最明确

对应表：

- `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_qshard4_seq_summary.md`

## 3. 当前最可信的判断

如果把现在所有结果压缩成一句话：

**Wormhole 8-core / S-block 已经证明“值得继续做”。**

原因不是只有一个：

- 它确实解决了 4 核版在 `q_shards=4` 路线上的容量瓶颈。
- 它在多个新解锁点已经具备真实竞争力，甚至拿到最快结果。
- 即便在原本 4 核就能跑的路线上，它也不是普遍退化，而是多数情况下可比，部分场景更优。

但同时也要保留边界：

- 8 核收益还没有被证明在所有轴上都稳定单调。
- `4k` 这类中间区间仍可能对 runtime 状态比较敏感。
- 真正的“单 head-group 直上 8 q_shards”还没打通。

## 4. 现在先看哪些文件

如果你想快速浏览，建议顺序：

1. 先看这份文档：`mla_flash_attention_dev/docs/flash-mla-wh-8core-phase-summary.md`
2. 再看详细状态：`mla_flash_attention_dev/docs/flash-mla-wh-8core-status.md`
3. 然后按需看四张对照表：
   - `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_batch_summary.md`
   - `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_head_summary.md`
   - `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_seq_summary.md`
   - `mla_flash_attention_dev/experiments/part1_baselines/outputs/tables/wh_4c_vs_8c_qshard4_seq_summary.md`

## 5. 下一步建议

最自然的下一步还是两条：

1. 把现在这些 `4c vs 8c` 结果继续扩成更正式的多次重复统计，把 `4k` 这种敏感点做稳。
2. 继续追 `H=64, dqhpc=8` 的 `CB/L1 clash`，尝试把“单 head-group 直上 8 q_shards”也打通。

# 结果切换说明

当前目录保存的是“公平对齐并行度之后”的 Part I 四方法对比结果。

## 目录说明

- 当前主结果：本目录
- `deepseek_num_q_heads_per_core` 正式 sweep：`../outputs_deepseek_qhpc_sweep`
- 旧的固定 `dqhpc=8` 不公平对比归档：`../archive/unfair_fixed_dqhpc8_20260411`
- 旧版与新版的并排数字摘要：`fair_vs_unfair_comparison.md`

## 当前公平口径

- `DeepSeek FlashMLA` 不再统一固定 `deepseek_num_q_heads_per_core=8`。
- 当前使用 `align_with_tt_mainline` 策略：优先选择硬件可行的 `dqhpc`，使 `num_q_shards` 尽量接近 TT 主线的 `max_cores_per_head_batch=4`。
- 在当前硬件约束下，合法 tile-height 候选主要是 `1/2/4/8/16`，因此：
  - `H=8 -> dqhpc=2 -> q_shards=4`
  - `H=16 -> dqhpc=4 -> q_shards=4`
  - `H=24 -> dqhpc=8 -> q_shards=3`
  - `H=32 -> dqhpc=8 -> q_shards=4`

## 为什么还需要单独保留 `dqhpc` sweep

- 公平对齐结果回答的是“对齐并行度之后，四方法谁更快”。
- `dqhpc` sweep 回答的是“DeepSeek 自己在不同 q-shard 粒度下会怎样变化”。
- 两者关注点不同，所以分别保留。

# 不公平对比归档说明

这份归档保存的是修正公平性之前的 Part I 四方法对比结果。

## 为什么这份结果是不公平对比

- 当时 `DeepSeek FlashMLA` 的 `deepseek_num_q_heads_per_core` 固定为 `8`，没有作为正式 sweep 轴。
- 这会导致 DeepSeek 的 `num_q_shards = H / deepseek_num_q_heads_per_core` 随 `H` 变化为：
  - `H=8 -> num_q_shards=1`
  - `H=16 -> num_q_shards=2`
  - `H=24 -> num_q_shards=3`
  - `H=32 -> num_q_shards=4`
- 而同一批对比里，`FlashMLA (TT Mainline)` 的主线并行度由 `max_cores_per_head_batch=4` 控制；在常见 `B=1` 场景下，它通常可以更稳定地吃满 `4` 路 head-batch 并行。
- 结果就是：
  - 在 `H=8/16/24` 上，DeepSeek 的 Q 并行度被固定 `dqhpc=8` 人为压低；
  - 只有到 `H=32` 时，DeepSeek 的 `num_q_shards=4` 才和 TT 主线比较接近。
- 因此，这批旧结果会系统性低估 DeepSeek FlashMLA 在较小/中等 `H` 上的表现。

## 这份归档还能用来看什么

- 它仍然可以作为“固定 `dqhpc=8` 历史口径”的参考。
- 它不应该再作为当前四方法主结论或论文主图的最终版本。

## 后续替代口径

- 正式 `dqhpc` sweep：把 `deepseek_num_q_heads_per_core` 升成正式 sweep 轴，单独比较 `dqhpc` 对四方法结果的影响。
- 公平对齐并行度版本：按 `H` 为 DeepSeek 派生硬件可行的 `dqhpc`，尽量让 `num_q_shards` 与 TT 主线的活动并行度对齐，再进行新的四方法对比。

# 公平性切换对比摘要

## 1. 对比对象

- 旧口径：固定 `deepseek_num_q_heads_per_core=8`，对应归档目录 `../archive/unfair_fixed_dqhpc8_20260411`。
- 新口径：`align_with_tt_mainline`，即按 `H` 选择硬件可行的 `dqhpc`，让 DeepSeek `num_q_shards` 尽量接近 TT 主线 `max_cores_per_head_batch=4`。
- 独立证据：`../outputs_deepseek_qhpc_sweep` 单独回答“不同 `dqhpc` 会如何影响结果”。
- 说明：归档主报告本身只有默认 config；跨 `H` 的旧口径量化对比主要来自归档中的验证实验 `raw/deepseek_vs_ttmainline_head_sweep_filtered.json`。

## 2. 旧口径为什么不公平

| `H` | 旧口径 `dqhpc` | 旧口径 `q_shards` | 新公平口径 `dqhpc` | 新公平口径 `q_shards` | 结论 |
|---|---:|---:|---:|---:|---|
| `8` | `8` | `1` | `2` | `4` | 旧口径把 DeepSeek 的 Q 并行度压得最严重 |
| `16` | `8` | `2` | `4` | `4` | 旧口径仍然少了一半并行度 |
| `24` | `8` | `3` | `8` | `3` | 当前硬件合法 tile height 下无法得到 `q_shards=4`，所以新旧一致 |
| `32` | `8` | `4` | `8` | `4` | 旧口径本来就基本对齐 |

## 3. 受控证据一：`H=16` 的 `dqhpc` sweep

口径：固定 `B=1, H=16, value_dim=512, rope_dim=64`，从 `../outputs_deepseek_qhpc_sweep` 提取。为了减少跨 shape 冷启动噪声，这里重点看 `best_ms`。

| `seq_len` | 旧口径 `dqhpc=8` | 公平口径 `dqhpc=4` | 激进上界 `dqhpc=1` | 旧 -> 公平提速 |
|---|---:|---:|---:|---:|
| `8k` | `0.554 ms` | `0.349 ms` | `0.157 ms` | `1.59x` |
| `32k` | `1.856 ms` | `0.990 ms` | `0.380 ms` | `1.88x` |
| `128k` | `6.952 ms` | `3.535 ms` | `1.021 ms` | `1.97x` |

直接结论：

- 旧口径里 `dqhpc=8` 明显压低了 DeepSeek 在 `H=16` 上的表现。
- 改成公平口径 `dqhpc=4` 之后，DeepSeek 会稳定变快。
- `dqhpc=1` 还能更快，但它对应的 `q_shards=16` 已经远高于本次公平对齐目标，所以更适合看成 DeepSeek 的“上界”，而不是公平主结果。

## 4. 受控证据二：`32k` 下旧 `H` sweep vs 新 fair `H` sweep

口径：

- 旧侧：归档验证实验 `../archive/unfair_fixed_dqhpc8_20260411/raw/deepseek_vs_ttmainline_head_sweep_filtered.json`
- 新侧：当前公平主结果 `raw/part1_four_method_results_filtered.json`
- 指标：统一使用 `best_ms`

| `H` | 旧口径 DeepSeek | 新公平 DeepSeek | 提速 | 新公平 `dqhpc / q_shards` |
|---|---:|---:|---:|---|
| `8` | `6.426 ms` | `0.984 ms` | `6.53x` | `2 / 4` |
| `16` | `1.833 ms` | `0.991 ms` | `1.85x` | `4 / 4` |
| `24` | `1.574 ms` | `1.577 ms` | `~1.00x` | `8 / 3` |
| `32` | `1.233 ms` | `1.211 ms` | `1.02x` | `8 / 4` |

直接结论：

- 公平性修正的主要收益集中在 `H=8` 和 `H=16`。
- `H=24` 基本没有改善，不是因为 fair 失败，而是因为当前硬件合法 tile 高度限制下只能落到 `dqhpc=8 -> q_shards=3`。
- `H=32` 本来就已经接近公平，所以新旧差别也很小。

## 5. 新公平版下的四方法关系

下面这张表只看当前公平主结果、`seq_len=32k`、`best_ms`：

| `H` | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA | 结论 |
|---|---:|---:|---:|---|
| `8` | `0.981 ms` | `0.992 ms` | `0.984 ms` | 三者几乎打平 |
| `16` | `0.962 ms` | `0.985 ms` | `0.991 ms` | DeepSeek 已回到同一梯队 |
| `24` | `1.180 ms` | `1.216 ms` | `1.577 ms` | 仍有剩余差距 |
| `32` | `1.178 ms` | `1.214 ms` | `1.211 ms` | DeepSeek 与两条 TT 线重新对齐 |

## 6. 总结

- 旧结果“不公平”的核心不是 DeepSeek 后端天然更慢，而是固定 `dqhpc=8` 让它在 `H=8/16` 上少用了大量 Q 并行度。
- 把 `deepseek_num_q_heads_per_core` 升成正式 sweep 轴之后，这个问题已经能被单独量化。
- 把主结果切到 `align_with_tt_mainline` 后，DeepSeek 不再表现为“系统性慢于 TT mainline”；真正剩下的差距主要集中在 `H=24` 这类受硬件 tile 约束的点。

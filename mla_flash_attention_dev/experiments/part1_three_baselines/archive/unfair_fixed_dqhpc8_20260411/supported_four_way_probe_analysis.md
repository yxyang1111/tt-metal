# 四方法共同支持参数空间分析报告

## 1. 数据范围与口径

- 数据源：`raw/capability_probe_supported_four_way_results.json`
- 分析对象：仅保留四种 `decode` 方法都成功执行的 workloads，即 `800` 个 cases、`80` 组 configs、`3200` 条四方法性能记录。
- 覆盖范围：`seq_len=256, 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k`，`B=1, 2, 4, 8, 16`，`H=8, 16, 24, 32`，`value_dim=128, 192, 256, 384, 512`。
- 重要说明：这批数据来自 `capability probe` 的单次 smoke measurement，不是正式 `10` 次 benchmark。它更适合做参数空间内的粗粒度趋势分析，不宜直接作为论文主结果。
- 因为 probe 口径是 `warmup=0, iters=1`，并且覆盖大量新 config，TT 方法的绝对时延可能受到首次运行开销影响。因此下面的结论应优先看相对趋势、分轴变化和方法间关系，而不是绝对数值本身。

## 2. 总体表现

### 2.1 总体统计

| 方法 | 平均时延 (ms) | 中位时延 (ms) | 平均吞吐量 (tok/s) | 最快 case 数 | 进入最优 3% case 数 |
|---|---:|---:|---:|---:|---:|
| `Reference Attention` | `858.597` | `139.940` | `70.002` | `619 / 800` | `622 / 800` |
| `Flash Attention` | `1907.861` | `1209.700` | `3.095` | `73 / 800` | `78 / 800` |
| `FlashMLA (TT Mainline)` | `1815.582` | `1225.809` | `3.020` | `64 / 800` | `65 / 800` |
| `DeepSeek FlashMLA` | `2038.118` | `1326.605` | `2.846` | `44 / 800` | `50 / 800` |

### 2.2 总体结论

- 在这份 supported-only probe 数据里，`Reference Attention` 整体最强，主要原因是支持子空间里仍然包含大量短到中等 `seq_len` 的 workload，而这批数据又是单次测量口径。
- 在三种 TT 方法内部，`Flash Attention` 与 `FlashMLA (TT Mainline)` 基本处于同一梯队，整体差距不到 `1%`。
- `DeepSeek FlashMLA` 在总体上落后于另外两条 TT 基线，但不是全面失速；它在长序列区间的个别 case 里仍然能进入最优或接近最优。

## 3. 方法间两两对比

上表中的“几何平均延迟优势”采用更快一方相对更慢一方的倍率。

| 对比 | 更快一方 | 胜出 case | 几何平均延迟优势 | 说明 |
|---|---|---:|---:|---|
| `Reference Attention` vs `Flash Attention` | `Reference Attention` | `657 / 800` | `7.64x` | 在 probe 口径下，Reference 在 supported 子空间中明显占优 |
| `Reference Attention` vs `FlashMLA (TT Mainline)` | `Reference Attention` | `660 / 800` | `7.62x` | 与上面一致，说明短中序列在 probe 数据里占比仍然很高 |
| `Reference Attention` vs `DeepSeek FlashMLA` | `Reference Attention` | `663 / 800` | `8.20x` | DeepSeek 在 probe 口径下整体落后于 Reference |
| `Flash Attention` vs `FlashMLA (TT Mainline)` | 基本持平 | `403 / 800` vs `397 / 800` | `< 1%` | 两者总体几乎没有稳定优势方 |
| `Flash Attention` vs `DeepSeek FlashMLA` | `Flash Attention` | `453 / 800` | `1.07x` | Flash 整体略优于 DeepSeek |
| `FlashMLA (TT Mainline)` vs `DeepSeek FlashMLA` | `FlashMLA (TT Mainline)` | `451 / 800` | `1.08x` | TT 主线 MLA 整体略优于 DeepSeek |

### 3.1 对 TT 三方法的直接解读

- `Flash Attention` 和 `FlashMLA (TT Mainline)` 的总体关系是“近乎打平”，谁更好取决于具体参数切片。
- `DeepSeek FlashMLA` 没有在总体统计里成为最优方法，但它并非始终处于明显劣势；它在更长 `seq_len` 的单 case 竞争力会提升。
- 如果只看 supported-only 参数空间，最稳妥的 TT 主比较仍然应该是 `Flash Attention` 和 `FlashMLA (TT Mainline)`，而 `DeepSeek FlashMLA` 更适合作为“在特定长序列和特定维度下可能具备竞争力”的方法来分析。

## 4. 不同参数下的表现

### 4.1 随 `seq_len` 的变化

- `seq_len <= 4k` 时，`Reference Attention` 在平均时延和最快 case 计数上都明显占优；`256/512/1k/2k/4k` 这五个切片中，Reference 都是 `80 / 80` 个 case 最快。
- `8k` 时，Reference 仍然在 `80` 个 case 里赢了 `76` 个，但 TT 方法已经开始出现零星反超；其中 `FlashMLA (TT Mainline)` 是三条 TT 方法中平均时延最低的一条。
- `16k` 时，Reference 仍然是切片平均最优，但最快 case 数已经降到 `66 / 80`；Flash 与 DeepSeek 都开始拿到可见数量的单点最优。
- `32k` 时，Reference 的最快 case 数进一步降到 `44 / 80`，已经不再是压倒性领先；TT 方法开始在相当一部分 case 里反超。
- `64k` 时，切片平均时延的第一名已经变成 `Flash Attention (1773.773 ms)`，`FlashMLA (TT Mainline)` 几乎完全打平 `1777.073 ms`，两者都优于 `Reference (2095.938 ms)`。
- `128k` 时，`Flash Attention` 仍是切片平均最优 `1911.449 ms`，`FlashMLA (TT Mainline)` 第二 `1997.422 ms`，`DeepSeek FlashMLA` 第三 `2469.061 ms`，`Reference` 最后 `4239.252 ms`。

### 4.2 `seq_len` 维度的核心结论

- 这份数据里，性能主导权的拐点大致出现在 `32k -> 64k` 之间。
- 在 `64k` 和 `128k` 两个最长序列切片上，TT 方法已经在切片平均时延上超过了 `Reference Attention`。
- 在长序列端，`Flash Attention` 与 `FlashMLA (TT Mainline)` 仍然是最强的两条 TT 线，而 `DeepSeek FlashMLA` 虽然平均值不占优，但单 case 最优次数明显上升：
  - `16k`: `5` 个 case 最快
  - `32k`: `7` 个 case 最快
  - `64k`: `10` 个 case 最快
  - `128k`: `22` 个 case 最快

### 4.3 随 `B` 的变化

- 在这份 probe 数据里，`Reference Attention` 在所有 batch 切片上仍然保持平均时延第一。
- 只看 TT 三方法：
  - `B=1/2/4/16` 时，`FlashMLA (TT Mainline)` 的切片平均时延最低。
  - `B=8` 时，`Flash Attention` 的切片平均时延最低。
- `DeepSeek FlashMLA` 在 batch 维度上没有成为任何一个切片的平均最优方法。
- 其中 `B=8` 是 DeepSeek 与另外两条 TT 线差距最明显的切片：
  - `Flash Attention`: `2885.037 ms`
  - `FlashMLA (TT Mainline)`: `3004.923 ms`
  - `DeepSeek FlashMLA`: `3854.545 ms`

### 4.4 随 `H` 的变化

- 在所有 head 切片上，`Reference Attention` 仍然保持平均时延第一。
- 只看 TT 三方法：
  - `H=8` 时，`Flash Attention` 最优
  - `H=16` 时，`FlashMLA (TT Mainline)` 最优
  - `H=24` 时，`Flash Attention` 最优
  - `H=32` 时，`FlashMLA (TT Mainline)` 最优
- `DeepSeek FlashMLA` 在 head 维度上没有拿到任何切片的平均第一，但在 `H=32` 时与 `FlashMLA (TT Mainline)` 的差距已经比较小：
  - `FlashMLA (TT Mainline)`: `1333.823 ms`
  - `DeepSeek FlashMLA`: `1391.971 ms`
  - `Flash Attention`: `1841.075 ms`

### 4.5 随 `value_dim` 的变化

- 在所有 `value_dim` 切片上，`Reference Attention` 仍然保持平均时延第一。
- 只看 TT 三方法：
  - `value_dim=128/192/256/512` 时，`FlashMLA (TT Mainline)` 平均最优
  - `value_dim=384` 时，`Flash Attention` 平均最优
- `DeepSeek FlashMLA` 在 value_dim 维度上没有拿到任何一个切片的平均第一。
- 从切片均值看，`DeepSeek FlashMLA` 在 `value_dim=512` 时更接近 `Flash Attention`，但仍落后于 `FlashMLA (TT Mainline)`：
  - `FlashMLA (TT Mainline)`: `2066.897 ms`
  - `DeepSeek FlashMLA`: `2440.037 ms`
  - `Flash Attention`: `2453.104 ms`

## 5. 各方法画像

### 5.1 `Reference Attention`

- 在这份 supported-only probe 数据中，它是整体最快的方法。
- 它在短到中等序列上优势非常明显，直到 `32k` 之前都在切片层面占主导。
- 但随着 `seq_len` 增大，它的优势会快速缩小，并在 `64k` 以后被 TT 方法在切片平均时延上超过。

### 5.2 `Flash Attention`

- 它是总体上最稳健的 TT 基线。
- 在三条 TT 方法中，它拿到了最多的最快 case：`73 / 800`。
- 它在最长序列端表现尤其强：`64k` 和 `128k` 两个切片都是它的平均时延第一。
- 它在 `H=8/24` 和 `value_dim=384` 上表现更强。

### 5.3 `FlashMLA (TT Mainline)`

- 它与 `Flash Attention` 基本处于同一档位，总体几乎打平。
- 从整体平均值和多数组合切片看，它略优于 `DeepSeek FlashMLA`，也是三条 TT 方法里最稳定的第二条主线。
- 它在 `B=1/2/4/16`、`H=16/32`、`value_dim=128/192/256/512` 上更容易成为 TT 侧最优。
- 如果要从 supported-only 空间里挑一条最有代表性的 MLA 基线，`FlashMLA (TT Mainline)` 比 `DeepSeek FlashMLA` 更稳。

### 5.4 `DeepSeek FlashMLA`

- 从 supported-only probe 的总体聚合看，它还不是最强 TT 方法。
- 它整体落后于 `Flash Attention` 和 `FlashMLA (TT Mainline)`，但不是全面落后，而是在长序列区间里变得更有竞争力。
- 它在 `128k` 拿到 `22 / 80` 个最快 case，与 `FlashMLA (TT Mainline)` 持平，仅次于 `Flash Attention (28 / 80)`。
- 因此更合理的定位是：`DeepSeek FlashMLA` 不是 supported-only 空间里的总体冠军，但在长上下文场景中值得重点跟踪。

## 6. 综合结论

- 如果只看这份 supported-only probe 数据，整体排序可以概括为：
  - 短到中等序列：`Reference Attention` 明显领先
  - 长序列：`Flash Attention` 和 `FlashMLA (TT Mainline)` 成为更强的平均性能方案
  - `DeepSeek FlashMLA`：总体略弱，但长序列竞争力上升
- 在 TT 三方法内部，最关键的结论不是“谁绝对最好”，而是：
  - `Flash Attention` 与 `FlashMLA (TT Mainline)` 近乎打平
  - `DeepSeek FlashMLA` 整体慢约 `7%~8%`
  - 但 `DeepSeek FlashMLA` 在 `32k~128k` 区间的个别 case 已经能进入第一梯队
- 如果下一步要做正式论文级 benchmark，建议：
  - 把 supported-only 的 `80` 组 configs 进一步收缩成一个更小的代表性集合
  - 对这些 configs 重新跑 `warmup + 10 次统计` 的正式 benchmark
  - 把当前这份报告作为“参数空间扫描后的筛选分析”，而不是最终性能结论

## 7. 相关文件

- 总入口：`supported_four_way_probe_report.md`
- 全量 supported-only 明细：`tables/supported_four_way_probe_all_cases.md`
- 按 `seq_len` 聚合：`tables/supported_four_way_probe_seq_summary.md`
- 按 `batch` 聚合：`tables/supported_four_way_probe_batch_summary.md`
- 按 `heads` 聚合：`tables/supported_four_way_probe_head_summary.md`
- 按 `value_dim` 聚合：`tables/supported_four_way_probe_dim_summary.md`
- `seq_len` 平均时延图：`visuals/supported_four_way_probe_seq_latency.svg`
- `seq_len` 平均吞吐量图：`visuals/supported_four_way_probe_seq_throughput.svg`
- `batch` 平均时延图：`visuals/supported_four_way_probe_batch_latency.svg`
- `heads` 平均时延图：`visuals/supported_four_way_probe_head_latency.svg`
- `value_dim` 平均时延图：`visuals/supported_four_way_probe_dim_latency.svg`

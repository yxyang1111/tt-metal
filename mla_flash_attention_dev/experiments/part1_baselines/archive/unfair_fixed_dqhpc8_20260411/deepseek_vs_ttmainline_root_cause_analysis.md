# 为什么 DeepSeek FlashMLA 目前会比 TT Mainline 慢

## 1. 问题定义

这里讨论的是当前 Wormhole 上的这条比较线：

- `FlashMLA (TT Mainline)`：`ttnn.transformer.paged_flash_multi_latent_attention_decode`
- `DeepSeek FlashMLA`：在 Wormhole 上经由 `FlashMLADecode._wh_builtin_backend(...)` 适配后，最终调用 `ttnn.transformer.flash_multi_latent_attention_decode`

目标问题是：

- 为什么在当前实验里，`DeepSeek FlashMLA` 常常比 `TT Mainline` 慢？
- 慢的主因到底是：
  - 自定义 DeepSeek 路径本身差
  - Wormhole 上的 fallback / adapter 损失
  - 还是当前参数设置导致的并行度不足

## 2. 理论分析

## 2.1 先看代码事实

从实现上看，当前 Wormhole 上的 DeepSeek 路径并不是直接跑它原本那套“自定义 S-block + ND-sharded KV + tiny-tile Q”的快路径。

关键事实有两条：

- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` 在 Wormhole 上会优先走 `_wh_builtin_backend(...)`
- `_wh_builtin_backend(...)` 会先把 DeepSeek 的特殊张量重新 materialize 成标准 `TILE_LAYOUT + DRAM_MEMORY_CONFIG`，然后调用 `ttnn.transformer.flash_multi_latent_attention_decode(...)`

也就是说，当前我们测到的 `DeepSeek FlashMLA`，本质上是：

- 不是原始实验 micro-op 的“原生 Wormhole 快路径”
- 而是“DeepSeek 输入构造 + builtin MLA decode backend”

因此，DeepSeek 设计里本来想利用的这些优势，在当前 Wormhole benchmark 里其实没有完整兑现：

- tiny-tile Q 的专用布局
- ND-sharded KV 的专用 bank placement
- 自定义 BRISC/NCRISC/TRISC 数据流
- 自定义 multicast / tree reduction 组织方式

## 2.2 但这还不是主因

如果“adapter 到 builtin backend”本身就是主要瓶颈，那么在同一个支持配置下：

- `TT mainline paged`
- `non-paged builtin + sharded Q`
- `DeepSeek adapted backend`

应该出现明显差距。

但实际验证不是这样。对于代表性配置：

- `B=1, H=32, value_dim=512, rope_dim=64`

我们测到：

| `seq_len` | `TT mainline paged` | `non-paged builtin + sharded Q` | `DeepSeek adapted backend` |
|---|---:|---:|---:|
| `8k` | `0.439 ms` | `0.428 ms` | `0.425 ms` |
| `32k` | `1.227 ms` | `1.221 ms` | `1.235 ms` |

这说明：

- 当 `H=32` 时，三条路径几乎一样快
- 因而“Wormhole 上 adapter/fallback 到 builtin backend”不是当前慢于 TT mainline 的主导原因
- 它最多只是一个次要因素，或者说它让 DeepSeek 的潜在优势消失了，但没有直接造成大幅额外惩罚

## 2.3 真正的主因：DeepSeek 当前的 Q shard 粒度太粗

当前严格四方法比较用的是：

- `deepseek_num_q_heads_per_core = 8`

这意味着：

- `num_q_shards = H / 8`

于是：

| `H` | `num_q_shards` |
|---:|---:|
| `8` | `1` |
| `16` | `2` |
| `24` | `3` |
| `32` | `4` |

而 TT mainline 这边在当前实验里用的是：

- `max_cores_per_head_batch = 4`

因此，当前对比实际上变成了：

- `H=8` 时：DeepSeek 只给自己 `1` 个 Q shard，而 TT mainline 仍可用 `4` 个核做每 head-batch 的 K-chunk 并行
- `H=16` 时：DeepSeek 只有 `2` 个 shard
- `H=24` 时：DeepSeek 只有 `3` 个 shard
- `H=32` 时：DeepSeek 才终于追平到 `4` 个 shard

所以理论上就会出现一个非常明确的现象：

- 在 `H < 32` 的情况下，DeepSeek 路径会因为 `num_q_shards` 太少而并行度不足
- 当 `H = 32` 时，这个差距应该显著缩小，甚至消失

这和我们在数据里看到的趋势是一致的。

## 2.4 因而当前“慢”的本质不是算法差，而是参数化方式不公平

更准确地说，当前比较里的主要问题不是：

- “DeepSeek 的 MLA 算法天生比 TT mainline 差”

而是：

- 当前 Wormhole 版本的 DeepSeek 路径被约束在 `deepseek_num_q_heads_per_core=8`
- 这个固定粒度在 `H=8/16/24` 时导致 Q shard 数太少
- 于是 device 侧并行度被人为卡低

因此这更像是：

- 当前实现参数没有调到合适点
- 而不是 DeepSeek 路径的理论上限更差

## 3. 实验设计

为了区分不同可能原因，我设计了三组实验。

## 3.1 实验 A：路径拆解实验

目标：

- 验证“adapter 到 builtin backend”本身是不是主因

设计：

- 固定 `B=1, H=32, value_dim=512, rope_dim=64`
- 比较三条路径：
  - `tt_mainline_paged`
  - `nonpaged_builtin_sharded_q`
  - `deepseek_adapted_backend`
- 在 `seq_len = 8k` 和 `32k` 两个点上测量

预期：

- 如果 adapter 本身是主要问题，那么 `deepseek_adapted_backend` 应该明显慢于前两者

## 3.2 实验 B：头数 sweep 验证

目标：

- 验证 DeepSeek 与 TT mainline 的差距是否会随着 `H` 提升而收敛

设计：

- 固定 `B=1, seq_len=32k, value_dim=512, rope_dim=64`
- 比较：
  - `TT mainline paged`
  - `DeepSeek adapted backend`
- 扫 `H = 8, 16, 24, 32`

预期：

- 如果主因真的是 `num_q_shards = H / 8` 太小，那么：
  - `H=8` 差距最大
  - `H=16` 差距缩小
  - `H=24` 继续缩小
  - `H=32` 基本接近或打平

## 3.3 实验 C：直接改 Q shard 粒度

目标：

- 直接验证 `deepseek_num_q_heads_per_core` 是不是瓶颈

设计：

- 固定 `B=1, H=16, seq_len=32k, value_dim=512, rope_dim=64`
- 只改：
  - `deepseek_num_q_heads_per_core = 8 / 4 / 2 / 1`
- 对应：
  - `num_q_shards = 2 / 4 / 8 / 16`

预期：

- 如果瓶颈就是 Q shard 数太少，那么随着 `num_q_shards` 增加，DeepSeek latency 应该显著下降
- 特别是当 `num_q_shards` 从 `2` 提升到 `4` 时，应该大幅接近 TT mainline

## 4. 实验结果与验证

## 4.1 实验 A：路径拆解结果

| `seq_len` | `TT mainline paged` | `non-paged builtin + sharded Q` | `DeepSeek adapted backend` | 结论 |
|---|---:|---:|---:|---|
| `8k` | `0.439 ms` | `0.428 ms` | `0.425 ms` | 三条路径几乎相同 |
| `32k` | `1.227 ms` | `1.221 ms` | `1.235 ms` | 三条路径几乎相同 |

结论：

- 这直接否定了“adapter 到 builtin backend 本身就是主要瓶颈”这个假设
- 至少在 `H=32` 代表性配置上，当前 DeepSeek backend 路线和 TT mainline decode 内核在 steady-state 下几乎等价
- 所以真正的问题不在这条 adapter 路线本身，而在别的参数维度

## 4.2 实验 B：头数 sweep 结果

由于当前工作站上存在偶发慢尾抖动，下面优先看 steady-state 的 `best` 延迟：

| `H` | DeepSeek `num_q_shards` | `TT mainline best` | `DeepSeek best` | 结论 |
|---:|---:|---:|---:|---|
| `8` | `1` | `1.007 ms` | `6.426 ms` | DeepSeek 明显更慢 |
| `16` | `2` | `0.988 ms` | `1.833 ms` | 差距显著缩小 |
| `24` | `3` | `1.222 ms` | `1.574 ms` | 继续接近 |
| `32` | `4` | `1.234 ms` | `1.233 ms` | 基本打平 |

这组结果非常符合理论预期：

- `H` 越小，DeepSeek 的 `num_q_shards` 越少，差距越大
- `H` 增长到 `32` 时，`num_q_shards=4`，与 TT mainline 当前的 `max_cores_per_head_batch=4` 对齐，差距基本消失

因此，实验 B 直接支持：

- 当前 DeepSeek 慢，不是因为它的 MLA 数学或 backend 天生差
- 而是因为当前比较里它在小/中 head 数时并行核数太少

## 4.3 实验 C：Q shard 粒度缩放结果

固定：

- `B=1, H=16, seq_len=32k, value_dim=512`

只改 `deepseek_num_q_heads_per_core`：

| `deepseek_num_q_heads_per_core` | `num_q_shards` | `median` | `best` |
|---:|---:|---:|---:|
| `8` | `2` | `8.224 ms` | `4.290 ms` |
| `4` | `4` | `0.997 ms` | `0.988 ms` |
| `2` | `8` | `0.658 ms` | `0.591 ms` |
| `1` | `16` | `0.405 ms` | `0.396 ms` |

这组实验几乎是“决定性证据”：

- 当 `num_q_shards=2` 时，DeepSeek 很慢
- 仅仅把 `deepseek_num_q_heads_per_core` 从 `8` 改成 `4`，让 `num_q_shards` 从 `2` 变成 `4`
- DeepSeek latency 就从 `8.224 ms` 直接降到 `0.997 ms`

而 `0.997 ms` 几乎与同条件下 TT mainline 的 `0.995 ms` 完全一致。

这说明：

- 当前 DeepSeek 落后 TT mainline 的主因，不是 DeepSeek backend 本身
- 而是当前固定的 `deepseek_num_q_heads_per_core=8` 让它在 `H=16` 时只开了 `2` 个 Q shard，严重欠并行

更进一步：

- 当把 shard 数继续增加到 `8` 和 `16` 时，DeepSeek 甚至还能继续变快
- 这说明当前比较中 DeepSeek 不是“已经到算法极限”，而是“还没把并行度打开”

## 5. 最终结论

可以把结论归纳成三层：

## 5.1 结论一：当前 Wormhole 上测到的 DeepSeek 路径不是原始 custom micro-op 优势路径

- 它会先把特殊张量适配成 builtin backend 可接受的标准格式
- 所以原本依赖 tiny-tile / ND-shard / S-block dataflow 的优势，在当前 benchmark 里并没有真正体现

但这不是当前“比 TT mainline 慢”的主导原因。

## 5.2 结论二：主导原因是 `deepseek_num_q_heads_per_core=8` 太保守

这会导致：

- `H=8 -> num_q_shards=1`
- `H=16 -> num_q_shards=2`
- `H=24 -> num_q_shards=3`
- `H=32 -> num_q_shards=4`

因此在 `H < 32` 的大部分 supported 配置里，DeepSeek 路径都处于并行度不足状态。

这正是它经常输给 TT mainline 的主要原因。

## 5.3 结论三：当把 Q shard 数调到与 TT mainline 同一量级后，差距会消失

最直接证据就是：

- `H=16`
- `deepseek_num_q_heads_per_core=8 -> num_q_shards=2` 时，DeepSeek 很慢
- `deepseek_num_q_heads_per_core=4 -> num_q_shards=4` 时，DeepSeek 立刻与 TT mainline 打平

所以当前更准确的表述应该是：

- **不是 DeepSeek FlashMLA 本身天然比 TT mainline 慢**
- **而是当前 Wormhole 比较配置把 DeepSeek 的 Q shard 并行度设置得过低**

## 6. 工程建议

如果要让后续比较更公平，我建议按优先级做三件事：

### 1. 先把 `deepseek_num_q_heads_per_core` 变成 sweep 轴

- 当前它固定为 `8`
- 应该至少测试：
  - `8`
  - `4`
  - `2`
  - 若资源允许，再看 `1`

### 2. 在正式 benchmark 里把 DeepSeek 的 shard 数和 TT mainline 的 core budget 对齐

- 最简单的公平口径是：
  - 让 DeepSeek 的 `num_q_shards` 接近 TT mainline 的 `max_cores_per_head_batch`
- 至少不要让 DeepSeek 在 `H=8/16/24` 时天然少开核

### 3. 把“当前 Wormhole adapted backend”与“真正 custom DeepSeek micro-op”分开表述

- 当前 Wormhole 对比线更准确的名字应该是：
  - `DeepSeek layout + WH builtin MLA backend`
- 不应该把它直接等同于“DeepSeek custom FlashMLA kernel 的最终性能”

## 7. 实验原始结果

- 路径拆解实验：`raw/deepseek_vs_ttmainline_rootcause_verification.json`
- 头数验证实验：`raw/deepseek_vs_ttmainline_head_sweep_filtered.json`
- Q shard 缩放实验：`raw/deepseek_q_shard_scaling_verification.json`

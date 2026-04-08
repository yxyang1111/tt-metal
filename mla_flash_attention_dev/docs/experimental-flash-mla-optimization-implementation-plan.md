# 实验性 FlashMLA 优化方向与实现方案

## 1. 文档目的

这份文档不是再重复一遍实验性 `FlashMLA` 的数据流，而是把前面已经识别出的优化方向继续往下压，整理成一份可落地的实现方案说明，重点回答四个问题：

1. 现有 profile 结果真正说明了什么。
2. 这些结论为什么可以指导实验性 `FlashMLA` 的优先级。
3. 每个方向具体可以怎么改，先改哪里，改哪些文件。
4. 每一类改动最合适用什么 case 验证。

本文默认你已经看过：

- `mla_flash_attention_dev/docs/experimental-flash-mla-dataflow-analysis.md`
- `mla_flash_attention_dev/docs/flash-mla-wh-component-utilization-and-bubble-analysis.md`

## 2. 使用这些 profile 结论时的边界

这里引用的细粒度 profile 证据主要来自**主线 Wormhole decode 路径**，不是实验性 unified kernel 自己的原生 profile。之所以仍然能指导实验路径，是因为两条路径在 decode 长序列上的主结构是相通的：

- 都是 `K` 路主导的 reader 压力。
- 都有 sender -> receiver 的片上广播或 multicast。
- 都有跨 `S block` 的 reduction / tail merge。
- 都会在长序列下把回压传回 `TRISC reserve-back`。

但也要明确三条边界：

1. profile 给出的优先级迁移趋势可以直接借用，不代表实验路径的绝对百分比一定相同。
2. 实验路径当前在 `Wormhole` 上仍然会走 `_wh_reference_fallback(...)`，所以真正的 device-side 验证仍要先在 `Blackhole` 上完成，再反推 `WH bring-up`。
3. 实验路径已经把 `V` 直接复用到 `K` buffer 语义里，没有独立 `cb_v_in`，因此不能把主线里一切 `V` 路问题原样投射过来。

## 3. 先给结论

### 3.1 优先级排序

| 优先级 | 方向 | 主要证据 | 首改文件 |
|---|---|---|---|
| P0 | 加深 `K` 路流水和 slot 深度 | 长序列 decode 的 `reader reserve` 和 `k_reserve` 主导 | `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`、`models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` |
| P1 | 给 `BRISC` 减负 | `writer cb_wait` 几乎全满，且主等待落在 `sender_cb_wait + tree_child_wait` | `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`、`models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py` |
| P2 | 把 tree reduction 从块级等待改成更流式 | sender/tree 是 writer 的主等待来源，不是 final gather | `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`、`models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` |
| P3 | 去掉 dummy handoff，收敛 SP 空路径 | 当前仍需要 `push_dummy_sdpa_inputs()` 才能防止下游 hang | `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`、`models/demos/deepseek_v3_b1/fused_ops/attention_block/kernels/attention_block_kernel.cpp` |
| P4 | 抽共享 planner | `flash_mla` standalone 和 fused block 在 grid / page / CB / semaphore 上重复规划 | `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`、`models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py` |

### 3.2 暂时不建议优先做的方向

下列方向现在都不是主优先级：

- 单独优化 `V` 路。
- 把主要精力放在 final output gather。
- 先做 math fidelity 改动。
- 直接拿 decode 的 PM FPU util / NOC util 百分比做决策。

理由很简单：当前证据已经很明确，主瓶颈不在这些位置。

## 4. 现在最关键的 profile 证据

下面这些数字直接决定了优先级。

### 4.1 Reader：长序列已经不是 issue 主导，而是 `K reserve` 主导

来自 `flash-mla-wh-component-utilization-and-bubble-analysis.md`：

- `decode_4k`：
  - `reader reserve = 39.78%`
  - `k_reserve = 31.64%`
- `decode_16k`：
  - `reader reserve = 59.71%`
  - `k_reserve = 52.79%`
- `decode_32k`：
  - `reader reserve = 61.54%`
  - `k_reserve = 54.82%`

同时：

- `v_reserve` 始终很小。
- 因此长序列 decode 的 reader 回压，本质上是 `K` 路 buffering / turnover / consumption 问题。

### 4.2 Writer：等待几乎都压在 sender/tree

同一份分析里，writer 的关键信号是：

- `writer cb_wait` 从 `decode_4k` 的 `98.10%` 升到 `decode_32k` 的 `99.77%`
- source-level 上：
  - `sender_cb_wait ~= 49.9%`
  - `tree_child_wait ~= 50.1%`
  - `root_cb_wait ~= 0%`
  - `output_gather_wait ~= 0%`

这说明 writer 的主要矛盾不是“最后写出太慢”，而是：

- sender 在等本地 partial output ready。
- reduction parent 在等 child partial result ready。

### 4.3 Compute：仍然在等输入，但越来越明显地感受到回压

decode compute bubble 的迁移是：

- `decode_4k`：`reserve-back share = 21.88%`
- `decode_16k`：`reserve-back share = 28.04%`
- `decode_32k`：`reserve-back share = 29.09%`

这意味着实验路径里如果继续让：

- `K` 路节拍不够深，
- `BRISC` 同时扛 Q 分发、multicast、tree reduction，

那么 `TRISC` 会越来越从“等输入”转成“同时被前后两头卡”。

## 5. 为什么这些结论能直接映射到实验路径

实验路径的实现里，有三个结构性事实和上面的 profile 结论是直接对位的。

### 5.1 它已经把 `V` 路问题基本收缩掉了

在 `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`：

- `cb_k_in` 存的是 `K/V` 共用输入。
- 注释明确写了 `V is read directly from K buffer`。
- 没有独立的 `cb_v_in`。

所以实验路径的第一优先级不应该再放在 `V` 路，而应该放在：

- `cb_k_in` 的 turnover。
- `K` page 读取和 multicast 的解耦。

### 5.2 它的 `BRISC` 职责比主线更重

`models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` 里，`BRISC` 侧同时做了：

- Q 分发 / output-core L1 读取。
- tail chunk mask 生成。
- `K` multicast。
- tree reduction sender / receiver 协议。

这和主线 profile 里 “writer 卡在 sender/tree” 的结论是正向共振的：实验路径的 `BRISC` 更容易成为回压放大器，而不是回压缓冲器。

### 5.3 它的 `K` 路 overlap 虽然已经做了，但深度还不够

当前实现已经有 page-level overlap：

- `op.py` 里 `k_page_size / k_num_pages` 是显式计算的。
- standalone 和 fused 路径里都把 `k_tiles` 固定成 `Sk_chunk_t * DHt * 2`，也就是双缓冲。
- `flash_mla.hpp` 里 `NCRISC` 和 `BRISC` 通过 `ncrisc_brisc_sync_semaphore`、`receiver_ready_semaphore` 做页级同步。

但这里仍有两个明显限制：

1. slot 深度固定为 2。
2. receiver readiness 仍然更接近“chunk 级别可接收”，而不是更细粒度的可持续 credit。

这正好对应长序列 profile 里 `K reserve` 被放大的现象。

## 6. 方向一：加深 `K` 路流水和 slot 深度

这是最应该先动手的方向。

### 6.1 当前实现长什么样

关键位置有两处。

在 `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` 里：

- `k_tiles = Sk_chunk_t * DHt * 2`
- `cb_k_in` 的 `total_size = k_tiles * k_tile_size`
- `k_page_size` 默认取 NOC 允许的最大整 tile page

在 `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` 里：

- `NCRISC` 为 sender 负责把一个 chunk 按 page 发起 ND-sharded DRAM 读取。
- `BRISC` sender 读取 `k_write_ptr_shared`，按 page 做 multicast。
- `receiver_ready_semaphore` 在 sender multicast 前要等所有 dest ready。

也就是说，当前已经不是“整 chunk 读完再发”，但也还没到“真正深流水”。

### 6.2 最小可落地的改法

第一步先不要改算法，只改 pipeline depth。

建议做法：

1. 在 `FlashMLAProgramConfig` 中新增 `k_pipeline_slots`，默认从 `2` 起，实验分支允许 `3` 或 `4`。
2. 把 `op.py` 和 `attention_block/op.py` 中的 `k_tiles = Sk_chunk_t * DHt * 2` 改成 `Sk_chunk_t * DHt * k_pipeline_slots`。
3. 把 `flash_mla.hpp` 中当前 “curr / next” 的双槽共享状态推广成 ring：
   - `page_ready_count[slot]`
   - `k_write_ptr[slot]`
   - 可选再加 `slot_epoch[slot]`，避免 wraparound 时误读旧状态。
4. `NCRISC` 每次拿一个空 slot 预留 `cb_k_in` 空间，然后向该 slot 持续灌 page。
5. `BRISC` sender 不再只在两个固定 slot 间切换，而是按 slot-ready 顺序发送。

这一步的目标不是一次性消灭所有回压，而是先把：

- `reader reserve`
- `k_reserve`
- `compute reserve-back`

往下压一截，看看 steady-state 是否更平滑。

### 6.3 第二步再做 page 粒度调参

当前 `k_page_size` 是通过 `get_max_page_size_and_num_pages(...)` 直接取到最大可整除 page，这个策略简单，但不一定总是最优。

建议把它从“固定派生值”改成“可枚举计划参数”：

- 优先尝试 `max_page_size`
- 再试 `max_page_size / 2`
- 再试更小但仍为整 tile 的 page

这么做的直觉是：

- 大 page 能降低 NOC issue 开销。
- 小 page 能提升 overlap 粒度，让 `BRISC` 更早拿到可发送数据。

这一步很适合直接挂到后面的 shared planner 或 autotune scaffold 上。

### 6.4 更激进的改法

如果 slot 加深后仍然看到 `k_reserve` 主导，可以继续往前走：

1. 把 sender 的 multicast 从“等所有 receiver ready 后开始整个 chunk”改成更细粒度 credit。
2. 把 receiver 的 ready 语义从“我这整个 chunk buffer 已 reserve”改成“我这个 slot / page window 已可接收”。
3. 尝试让 `NCRISC` 自己直接承担部分 multicast 发起逻辑，把 `BRISC` 从 `K` 路 steady-state 中再抽掉一层。

这一层风险更高，不建议作为第一刀。

### 6.5 主要风险

- `L1` 压力会直接增大，尤其是 `cb_k_in` 扩到 `3x` 或 `4x` 后可能挤压 output / reduction CB。
- slot ring 的同步写错很容易出现“读到旧页”或“slot 被重复消费”。
- page 过小可能让 NOC issue/buffer 开销反而变大。

### 6.6 最合适的验证

先看 `decode_4k / 16k / 32k`，因为这三个点最能看迁移。

重点关注：

- kernel latency
- `reader reserve share`
- `k_reserve`
- `writer cb_wait`
- `compute reserve-back share`

如果只看一个点，优先看 `decode_16k`。

## 7. 方向二：给 `BRISC` 减负

如果说方向一是“让 K 路更深”，方向二就是“别让 BRISC 在 steady-state 里同时扛太多职责”。

### 7.1 当前 `BRISC` 扛了什么

`models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` 的 `BRISC` 段依次做：

1. `reader-q-read`
2. output core 上的 Q 输入广播协商
3. 非 output core 从 output-core L1 拉 Q
4. `mask-last-chunk`
5. `mcast-sender-multicast`
6. `tree-reduction-sender / receiver`

这意味着 `BRISC` 当前并不是“纯 writer”，而是一个重控制面的 orchestrator。

### 7.2 第一种改法：把 Q 预分发到 MLA worker

这是减负最直接的一招。

思路是把当前 `BRISC` 里的 Q 分发提前：

- 对 standalone `FlashMLADecode.op(...)`，允许输入 `Q` 直接按 MLA active core 布局提供，而不是只在 output core 有 shard。
- 对 fused `AttentionBlock` 路径，优先在 `PreSDPA` 结束时就把 `Q` 物化到 MLA 将使用的所有 core，或者增加一个非常薄的 Q forwarder stage。

这样可以把 `BRISC` 里下面这几段变成 no-op 或极薄逻辑：

- `q_input_mcast_semaphore` 协调
- non-output core 的 L1-to-L1 `Q` 读
- output core 的 Q ready fanout

代价是多了一次更前置的复制，但好处是 `BRISC` steady-state 更干净。

### 7.3 第二种改法：把 tail mask 从热路径中拿掉

现在的 tail mask 完全由 `BRISC` 在最后一个 chunk 上现算，逻辑简单，但它和 Q 分发、K multicast、tree reduction 放在同一个线程里。

可选做法有三种：

1. host 或前序 stage 预先生成一张 mask tile，按 `cur_pos % k_chunk_size` 选择使用。
2. 让 `TRISC` 在看到 `mask_last_chunk` 时本地生成，而不是让 `BRISC` 生成后再塞 CB。
3. 建一个小型 mask cache，只在 `num_unmasked` 变化时更新。

这项改动本身未必带来很大绝对收益，但可以减少 `BRISC` 热路径上的杂项控制逻辑。

### 7.4 第三种改法：把 `K` multicast 的控制权往 `NCRISC` 挪

更激进的版本可以尝试让 sender reader 在 page 完成后更直接地主导 multicast：

- `NCRISC` 继续负责 ND-sharded DRAM read。
- 但 multicast 的 page issue 不再完全由 `BRISC` 驱动。
- `BRISC` 只保留 tree reduction 和少量输出整理工作。

这一招的潜在价值很大，因为它能把“读 K”和“发 K”收敛到同一侧时序里，但也最容易引入协议复杂度，因此建议排在 `k_pipeline_slots` 之后。

### 7.5 第四种改法：先补标记，再决定是否大改

当前代码里已经有 `DeviceZoneScopedN(...)`，但实验路径还没有像主线 detailed profile 那样稳定产出一套“Q 分发 / K multicast / tree sender / tree receiver”的可视化占比。

建议在真正大改前，先确保下面几类标记能稳定进入实验路径 profile：

- `reader-q-read`
- `mask-last-chunk`
- `mcast-sender-multicast`
- `tree-reduction-sender`
- `tree-reduction-receiver`

如果大部分 writer 时间已经全落在 sender/tree，那么就继续做结构改动；如果反而发现 Q path 占比远高于预期，再把 Q 预分发提前到更高优先级。

### 7.6 主要风险

- Q 预分发会增加上游的复制压力，可能把问题从 `BRISC` 挪到别的 stage。
- 如果 Q 分发提早做，standalone 和 fused 的输入布局需要统一，否则 shared planner 之前会先出现两套语义。
- 把 multicast 控制权挪给 `NCRISC` 之后，读和发之间更容易互相干扰。

### 7.7 最合适的验证

重点看：

- `writer cb_wait`
- `sender_cb_wait`
- `tree_child_wait`

如果只看功能正确性，还要补：

- 非 tail chunk 和 tail chunk 的数值一致性
- fused attention block 下 Q shard 布局是否仍然正确

## 8. 方向三：把 tree reduction 改成更流式

这是第二个真正可能改出明显收益的方向。

### 8.1 当前 tree reduction 的问题不是“算法错”，而是“推进粒度太粗”

现在的 tree reduction 是固定步数、固定 partner 的块级协议：

- sender 先等本地 `cb_out_o / cb_out_ms` 整块 ready。
- receiver 先 `cb_reserve_back(...)` 整块输入。
- receiver 再 busy-wait 自己的 `reducer_semaphore`。
- `TRISC` 在 reduction 输入 ready 后再跑 `sdpa_tail(...)`。

这套协议的优点是结构清晰，但它天然会把等待集中到：

- sender 等本地 partial ready
- parent 等 child ready

也就是主线 profile 里已经看到的 `sender_cb_wait + tree_child_wait`。

### 8.2 第一种改法：先做 active-tree schedule 压缩

这是最保守但收益常常不差的版本。

虽然当前 runtime 已经会根据 `num_active_s_blocks` 跳过一部分 partner，但 host 仍然按固定的 `TREE_REDUCTION_ORDER` 把整套拓扑塞给 runtime。

可以先做：

1. host 侧按 `num_active_s_blocks` 直接生成有效 reduction steps。
2. 对 `num_active_s_blocks < full_s_blocks` 的情况，不再传无效 step。
3. 让 tree partner 选择尽量贴合实际活跃 `S block` 和 NOC 拓扑，而不是只套固定模板。

这一步改动小，尤其适合序列不够长、活跃 block 数动态变化的 case。

### 8.3 第二种改法：把 sender/receiver 协议从整块改成分块流送

这是更有价值的版本。

建议把 `out_chunk_tiles` 再切成更小的 reduction block，例如：

- `dst_size` 大小
- `vDHt / 2`
- 或者按 tiny-tile block 固定切片

然后把协议改成：

1. `TRISC` 每完成一个 reduction block，就 push 一个 block。
2. `BRISC` sender 不再等整块 `cb_out_o` ready，而是等一个 block ready 就发一个 block。
3. `BRISC` receiver 也按 block reserve / push。
4. `TRISC sdpa_tail(...)` 改成可逐块 merge，而不是只等全块输入。

这样做的本质，是把：

- sender 等 full local output
- parent 等 full child output

改成：

- sender / parent 都沿着更小 block 持续前进

### 8.4 第三种改法：把第一轮 reduction 更早地吃进 compute

如果方向二已经把 `BRISC` 压轻，可以考虑把最靠近 local output 的第一轮 reduction 更早地交给 `TRISC`：

- 相邻 `S block` 的第一轮 merge 尽量在 compute 结果刚 ready 时就开始。
- `BRISC` 更多只做跨 core 的数据搬运，不承担太多“等待结果齐备后再统一发送”的职责。

这一步更激进，也更容易影响数值与同步稳定性，建议排在 block-streaming 之后。

### 8.5 主要风险

- `m/s/o` 的 reduction 次序变了之后，数值稳定性要重新确认。
- reduction block 太小会增加 semaphore / NOC issue 频率。
- `cb_out_in / cb_ms_in / cb_interm_*` 的碎片化会更严重。

### 8.6 最合适的验证

优先看 `decode_16k / 32k`。

重点指标：

- `sender_cb_wait`
- `tree_child_wait`
- `writer cb_wait`
- `compute wait-front`

如果这几个指标不降，只是 `root_cb_wait` 或 final gather 在变化，那就说明改动方向偏了。

## 9. 方向四：去掉 dummy handoff，把空路径收敛进 FlashMLA 自己

这个方向更多是结构正确性和工程收敛性问题，但越晚做越容易留下历史包袱。

### 9.1 当前为什么会有 dummy handoff

`models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` 里有 `push_dummy_sdpa_inputs()`。

`models/demos/deepseek_v3_b1/fused_ops/attention_block/kernels/attention_block_kernel.cpp` 和 `models/demos/deepseek_v3_b1/fused_ops/decoder_block/kernels/decoder_block_kernel.cpp` 在 “当前 device 没有 sequence data” 的情况下，会显式调用它，避免下游 `SdpaReduceWorker` hang。

这说明现在的边界仍然是：

- FlashMLA 只在“有本地序列贡献”时生产真实结果。
- 对于空设备，需要外层手动补 dummy token。

### 9.2 最好的收敛方式

推荐分两步走。

第一步：

- 让 `FlashMLADecode::Op` 自己支持“空贡献但合法退出”的统一语义。
- 即使 `skip_attention`，也要对下游产生一个可被 reduction 正常消费的 identity / empty contribution。

第二步：

- 把最终 SP reduce 的一部分直接融合到 FlashMLA 边界里。
- 至少让“哪些设备没有贡献”这件事不再要求外层 kernel 显式插 dummy tiles。

### 9.3 为什么这件事值得做

这项改动不一定立刻缩短单核 latency，但它会显著降低：

- fused path 的控制复杂度
- 多设备 SP 场景下的挂死风险
- standalone / fused 语义漂移

也会让后面做 shared planner 时边界更干净。

### 9.4 主要风险

- 多 SP 设备场景最容易出现“无贡献设备被错误当成真实贡献”。
- 下游 `SdpaReduceWorker` 的期望输入语义必须同步收敛。

### 9.5 最合适的验证

优先测：

- `seq_len < device_chunk_size`
- `seq_len` 刚好跨过一个 device chunk 边界
- 有设备空跑、有设备有贡献的多 SP case

目标不是只看速度，而是确认：

- 不 hang
- 数值正确
- 不再需要外部 dummy push

## 10. 方向五：抽共享 planner，别再让 standalone 和 fused 各算一遍

这件事不一定最先提升性能，但它会决定后续所有优化能否快速迭代。

### 10.1 当前重复在哪

现在至少有两处明显重复：

- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`
- `models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py`

重复内容包括：

- `ProgramConfig`
- `grid`
- `num_s_blocks / cores_per_s_block`
- `k_page_size / k_num_pages`
- `k_tiles`
- `CB` 和 semaphore 规划
- per-core runtime arg 组织

### 10.2 推荐拆法

建议新建一个共享 planner 模块，至少拆成四层对象：

1. `GridPlan`
   - 负责 `S block` 布局、active core 顺序、multicast 坐标、tree partner。
2. `StreamPlan`
   - 负责 `k_chunk_size`、`k_page_size`、`k_num_pages`、`k_pipeline_slots`。
3. `BufferPlan`
   - 负责 `cb_*` 容量、tile 格式、semaphore 分配。
4. `RuntimeArgPlan`
   - 负责按 core 生成 `NCRISC/BRISC/TRISC` runtime args。

然后：

- standalone op 用这套 planner 直接生成 `UnifiedKernelDescriptor` 所需材料。
- fused `AttentionBlock` / `PreSDPA` 复用同一份 planner 输出，再叠加自己独有的上游和下游阶段。

### 10.3 为什么这对优化很关键

后面不管你要试的是：

- `k_pipeline_slots = 3/4`
- 不同 `k_page_size`
- 动态 tree schedule
- 不同架构的 grid

如果没有 shared planner，每一项都得在 standalone 和 fused 路径各改一次，而且很容易改歪。

### 10.4 主要风险

- 这是重构，不是纯性能 patch。
- 如果没有做“旧 planner 和新 planner 产出的 compile/runtime args 对比”，很容易引入静默行为变化。

### 10.5 最合适的验证

最好的方式不是只跑数值，而是做 planner 对比：

1. 对同一个 shape，打印旧路径和新路径生成的：
   - active core 序列
   - `k_page_size / k_num_pages`
   - `cb_*` 大小
   - per-core runtime args
2. 确认完全一致后，再开始把新参数接进优化实验。

## 11. 逐方法实施计划

这一节把前面的 5 个主方向进一步收敛成可以开工的 implementation plan。

先说明两条口径：

- 这里的收益预估，默认是相对**当前实验性 `FlashMLA` 基线**，主要看 `decode_4k / 16k / 32k` 的 kernel latency 和 stall share。
- 各项收益**不可简单线性相加**。例如 `K` 路加深和 tree 流式化都会同时影响 `writer cb_wait`，最终叠加收益通常会小于单项相加。

### 11.1 方法一：`K` 路加深流水 + `k_page_size` 可调

| 项目 | 计划 |
|---|---|
| 如何实现 | 分两阶段做。Phase A 先把 `k_pipeline_slots` 参数化，把当前双槽 `curr/next` 协议扩成 3 或 4 槽 ring；Phase B 再把 `k_page_size` 从“自动最大值”改成“可显式选择或枚举”的计划参数。 |
| 主要修改文件 | `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`、`models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py`、`models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`、`models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/flash_mla_kernel.cpp`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla.py`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_attention_block.py` |
| 需要写哪些代码 | host 侧新增 `FlashMLAProgramConfig.k_pipeline_slots`、`FlashMLAProgramConfig.k_page_size_override` 或 `k_page_size_mode`；新增 helper，例如 `build_k_stream_plan(...)`、`resolve_k_page_size(...)`。kernel 侧新增 per-slot 状态管理，例如 `slot_ready_pages[slot]`、`slot_write_ptr[slot]`、`slot_epoch[slot]` 或等价结构；把 `NCRISC/BRISC` 当前的双槽切换逻辑改成 ring 遍历；compile-time args 里新增 `k_pipeline_slots`。 |
| 预计代码规模 | host Python 约 `120~180` 行，kernel C++ 约 `180~260` 行，测试约 `80~120` 行。 |
| 具体步骤 | 1. 在 `op.py` / `attention_block/op.py` 中把 `k_tiles = Sk_chunk_t * DHt * 2` 改成基于 `k_pipeline_slots` 推导。 2. 更新 CB 容量和 named compile-time args。 3. 在 `flash_mla.hpp` 中把 `ncrisc_brisc_sync_curr/next` 的双槽共享状态推广成 compile-time bounded ring。 4. 让 sender 侧按 slot-ready 顺序 multicast，而不是只在两个固定槽间切换。 5. 补 `k_page_size` 枚举逻辑，先支持 `auto`、`max`、`max_div_2`、`explicit_bytes` 四种模式。 |
| 测试与验证 | 先跑 `test_flash_mla.py` 和 `test_attention_block.py` 的正确性。然后只在 `decode_16k` 看第一轮性能变化，再扩到 `decode_4k / 32k`。重点观察 `reader reserve`、`k_reserve`、`compute reserve-back` 是否同步下降。 |
| 预计收益 | 保守预估：`decode_16k / 32k` latency 降 `5%~10%`，`k_reserve` 下降 `8~15` 个百分点。激进预估：如果 ring + page 粒度都调对，长序列 latency 有机会降 `10%~18%`。对 `decode_4k` 的收益通常更小，约 `2%~6%`。 |

建议的最小交付版本是：

1. 只做 `k_pipeline_slots = 3`。
2. 不先改 multicast ownership。
3. 只对 `decode_16k` 做 A/B 测试。

如果这一版没有把 `k_reserve` 明显拉下来，就不要急着推进后面的 `BRISC` 和 tree 大改。

### 11.2 方法二：给 `BRISC` 减负

| 项目 | 计划 |
|---|---|
| 如何实现 | 也分两层。Phase A 做低风险减负：先把 Q 预分发到 MLA worker，再把 tail mask 从 `BRISC` 热路径外移。Phase B 才评估是否把 `K` multicast 的 page issue 从 `BRISC` 迁给 `NCRISC`。 |
| 主要修改文件 | `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`、`models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`、`models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py`、`models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/op.py`、`models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/pre_sdpa_kernel.cpp`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_pre_sdpa.py`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_attention_block.py` |
| 需要写哪些代码 | 新增 `q_distribution_mode` 或 `pre_distributed_q` 配置；在 `PreSDPA` 或 `AttentionBlock` 侧新增一个轻量 Q fanout / forwarder 逻辑；在 `flash_mla.hpp` 的 `BRISC` 路径里增加 `if (q_is_pre_distributed)` 的快路径，跳过 `q_input_mcast_semaphore` 协调。mask 侧新增 `tail_mask_mode`，支持 `host_prebuilt` / `trisc_generate` / `brisc_generate`。如果推进 Phase B，再新增 `NCRISC` multicast helper，例如 `issue_k_multicast_page(...)`。 |
| 预计代码规模 | host Python 约 `150~240` 行，`PreSDPA/AttentionBlock` kernel 约 `80~140` 行，`flash_mla.hpp` 约 `100~180` 行，测试约 `100~150` 行。 |
| 具体步骤 | 1. 先补稳定 profile 标记，确认 `reader-q-read`、`mask-last-chunk`、`mcast-sender-multicast`、`tree-reduction-*` 都能稳定产出。 2. 在 fused 路径上先做 Q 预分发，因为 fused 更能放大 `BRISC` 控制面成本。 3. 验证 Q 预分发后再把 tail mask 从 `BRISC` 热路径拿掉。 4. 只有在 `writer cb_wait` 仍然明显落在 sender 侧时，再尝试 `NCRISC` 主导的 multicast issue。 |
| 测试与验证 | `test_pre_sdpa.py`、`test_attention_block.py`、`test_flash_mla.py`。性能上优先看 `decode_4k / 16k`，因为 `BRISC` 减负对中长序列最敏感。重点观察 `writer cb_wait`、`sender_cb_wait`、`tree_child_wait`。 |
| 预计收益 | 仅做 Q 预分发 + tail mask 外移时，保守预估 latency 降 `3%~7%`。如果这两步落地后 `BRISC` 的 sender wait 仍高，再把 multicast issue 部分挪到 `NCRISC`，整体有机会做到 `6%~12%`。对 `decode_32k` 的收益通常取决于 tree 是否同时改善，因此单独做这一步不一定超过 `8%`。 |

建议把这个方向拆成三个可独立提交的 patch：

1. `Instrumentation only`
2. `Q pre-distribution`
3. `Tail-mask offload`

这样即使最后不做 `NCRISC` multicast，也能先吃掉一部分稳定收益。

### 11.3 方法三：tree reduction 流式化

| 项目 | 计划 |
|---|---|
| 如何实现 | 先做 schedule 压缩，再做 block-streaming。Phase A 把 `TREE_REDUCTION_ORDER` 从“固定最大步数”改成“host 侧根据 `num_active_s_blocks` 生成有效步数”；Phase B 再把 sender/receiver 协议从整块 ready 改成 block ready。 |
| 主要修改文件 | `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`、`models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`、`models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/flash_mla_kernel.cpp`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_tail.py`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla.py` |
| 需要写哪些代码 | host 侧新增 `build_active_tree_schedule(...)`、`valid_tree_steps`、`reduction_block_tiles`。runtime args 里要把有效 step 数和压缩后的 partner 列表传进去。kernel 侧新增 block-streaming 协议，例如 `wait_reduce_block_ready(...)`、`send_reduce_block(...)`、`recv_reduce_block(...)`；`TRISC` 侧要让 `sdpa_tail(...)` 能按 block 增量 merge，而不是只吃完整 chunk。 |
| 预计代码规模 | host Python 约 `80~140` 行，kernel C++ 约 `220~340` 行，测试约 `120~180` 行。 |
| 具体步骤 | 1. 先在 host 侧压缩 `TREE_REDUCTION_ORDER`，加入 `valid_tree_steps`，这是低风险 patch。 2. 再选一个固定 `reduction_block_tiles`，建议先试 `dst_size` 或 `vDHt / 2`。 3. 调整 `cb_out_in / cb_ms_in / cb_interm_*` 的容量模型，使其支持 block 粒度的 reserve / push。 4. 把 sender/receiver 的 wait 改成 block 级别。 5. 只有在 block-streaming 稳定后，再尝试把第一轮 reduction 进一步往 compute 端提前。 |
| 测试与验证 | 先做 `test_sdpa_tail.py` 的数值回归，再看 `test_flash_mla.py`。性能优先测 `decode_16k / 32k`，因为 sender/tree wait 在这两个点最稳定。关键指标是 `sender_cb_wait`、`tree_child_wait`、`writer cb_wait`。 |
| 预计收益 | 只做 active-tree schedule 压缩时，收益通常只有 `1%~3%`，但风险很低。做完 block-streaming 后，保守预估长序列 latency 可降 `6%~12%`；如果当前实验路径确实和主线一样被 sender/tree 强主导，激进上限可看 `10%~18%`。 |

这个方向的判断标准很明确：

- 如果 `tree_child_wait` 不降，只是 `root_cb_wait` 在波动，就说明改错地方了。
- 如果 `sender_cb_wait` 和 `tree_child_wait` 同时下降，说明 tree 流式化是真的打在主矛盾上。

### 11.4 方法四：去掉 dummy handoff / 收敛空设备语义

| 项目 | 计划 |
|---|---|
| 如何实现 | 把“没有本地 sequence 贡献的设备”也纳入 `FlashMLA` 自己的输出协议里，不再依赖 fused 外层手工调用 `push_dummy_sdpa_inputs()`。先做 identity contribution，再评估是否把最终 SP reduce 边界向 `FlashMLA` 内收。 |
| 主要修改文件 | `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`、`models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/rt_args_common.hpp`、`models/demos/deepseek_v3_b1/fused_ops/attention_block/kernels/attention_block_kernel.cpp`、`models/demos/deepseek_v3_b1/fused_ops/decoder_block/kernels/decoder_block_kernel.cpp`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_attention_block.py`、`models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py` |
| 需要写哪些代码 | 新增 `has_local_contribution` 或等价 runtime flag；在 `FlashMLADecode::Op` 里新增统一的空贡献出口，例如 `emit_empty_sdpa_contribution()` 或 `emit_identity_sdpa_contribution()`；删掉 fused 外层对 `push_dummy_sdpa_inputs()` 的显式调用；必要时补一个 helper，把 `skip_attention`、`skip_kv_cache_update` 和 `local_cur_pos` 的语义统一起来。 |
| 预计代码规模 | kernel C++ 约 `80~140` 行，host/rt args 约 `30~60` 行，测试约 `80~120` 行。 |
| 具体步骤 | 1. 先在 `rt_args_common.hpp` 或调用侧把“空设备/无贡献设备”语义显式化。 2. 让 `FlashMLA` 在空路径上仍然产生可被下游消费的 identity contribution。 3. 从 `attention_block_kernel.cpp` 和 `decoder_block_kernel.cpp` 移除 `push_dummy_sdpa_inputs()` 分支。 4. 如有必要，再进一步把最终 SP reduce 的空路径判断内聚到 `FlashMLA` 边界。 |
| 测试与验证 | 重点不是单设备性能，而是多 SP 正确性。优先测试 `seq_len < device_chunk_size`、跨 `device_chunk_size` 边界、部分设备空跑的 case。确保“不 hang、结果正确、下游不需要 dummy push”。 |
| 预计收益 | 单设备 runtime 收益很小，通常只有 `0%~1%`。多 SP 场景若当前空设备路径较重，保守预估 `0%~4%` latency 改善，但更大的收益是稳定性和代码收敛：能直接减少一类 hang / 语义漂移问题。 |

这个方向适合在性能主线稳定后做，但不要拖到最后完全失控再做，因为一旦更多 fused kernel 复制这套 dummy 逻辑，后面清理成本会显著上升。

### 11.5 方法五：抽 shared planner

| 项目 | 计划 |
|---|---|
| 如何实现 | 把 standalone `FlashMLADecode.op(...)` 和 fused `AttentionBlock` 里重复的 grid / page / buffer / runtime arg 规划抽成共享 builder。建议新建一个 planner 模块，先做到“产出完全一致”，再让后续优化参数都从 planner 出。 |
| 主要修改文件 | 新增 `models/demos/deepseek_v3_b1/micro_ops/flash_mla/planner.py`；修改 `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`、`models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py`；可选增加 `models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla_planner.py`，或先扩展 `test_flash_mla.py` / `test_attention_block.py`。 |
| 需要写哪些代码 | 新增 dataclass / builder，例如 `FlashMLAGridPlan`、`FlashMLAStreamPlan`、`FlashMLABufferPlan`、`FlashMLARuntimeArgsPlan`、`build_flash_mla_plan(...)`。新增调试 helper，例如 `dump_flash_mla_plan(...)` 或 `compare_flash_mla_plan(...)`，保证旧路径和新路径产出的 active core 顺序、CB 大小、compile/runtime args 完全一致。 |
| 预计代码规模 | 新文件约 `250~400` 行，调用侧重构约 `120~220` 行，测试约 `120~200` 行。 |
| 具体步骤 | 1. 先把当前 `op.py` 和 `attention_block/op.py` 中重复的 grid / page / buffer 计算提取成纯函数。 2. 再把 per-core runtime args 生成收进 planner。 3. 增加对比模式，让旧路径和新 planner 同时产出 plan 并逐项比较。 4. 完全一致后，把新参数化能力如 `k_pipeline_slots`、`k_page_size_mode`、`valid_tree_steps` 接进 planner。 |
| 测试与验证 | 最重要的是“planner 等价性测试”，而不是纯数值测试。先比计划输出，再跑 `test_flash_mla.py` 和 `test_attention_block.py`。 |
| 预计收益 | 对单次 kernel latency 的直接收益通常只有 `0%~2%`，甚至可能完全没有直接 runtime 收益；真正的收益是把后续 `K` 路、tree、BRISC 改动的试验成本降低一大截。工程收益可以粗略理解为“后续参数化/实验迭代效率提升 `2x` 左右”。 |

shared planner 不建议最先做，但建议在 `K` 路参数化稳定后尽快跟上，否则后续每做一个实验都要双改两套路径。

## 12. 建议的落地顺序

如果目标是先尽快拿到一轮可解释的收益，建议按下面顺序做：

1. 先补实验路径的稳定 profile 标记和基准 case。
2. 先做 `k_pipeline_slots` 参数化，再试 `k_page_size` 小范围枚举。
3. 再做 `BRISC` 减负里的低风险项：
   - Q 预分发
   - tail mask 热路径外移
4. 然后做 dynamic / compressed tree schedule。
5. 如果 sender/tree 仍然明显主导，再做 block-streaming reduction。
6. 等边界稳定后，再去掉 dummy handoff。
7. 最后抽 shared planner，把这些参数化收敛成长期可维护方案。

这个顺序的核心原因是：

- 前三步最接近现有 profile 结论。
- 中间两步最可能真的压 writer source-level wait。
- 最后两步更偏工程收敛，适合在方向定型后做。

## 13. 一页实施清单

| 方向 | 第一刀怎么改 | 主要文件 | 重点观察 |
|---|---|---|---|
| `K` 路加深 | `k_pipeline_slots=3`，推广双槽到 ring | `micro_ops/flash_mla/op.py`、`unified_kernels/flash_mla.hpp` | `reader reserve`、`k_reserve`、`reserve-back` |
| `BRISC` 减负 | 先把 Q 预分发出去，再把 tail mask 热路径外移 | `attention_block/op.py`、`unified_kernels/flash_mla.hpp` | `writer cb_wait`、`sender_cb_wait` |
| tree 流式化 | 先压缩 active-tree step，再尝试分块 reduction | `micro_ops/flash_mla/op.py`、`unified_kernels/flash_mla.hpp` | `tree_child_wait`、`sender_cb_wait` |
| dummy handoff 去除 | 让空设备输出 identity contribution | `unified_kernels/flash_mla.hpp`、`attention_block_kernel.cpp` | 多 SP 不 hang，结果正确 |
| shared planner | 把 grid/page/buffer/runtime args 抽成统一 builder | `micro_ops/flash_mla/op.py`、`attention_block/op.py` | 新旧 planner 输出一致 |

## 14. 建议的 baseline 设计与实验设置

如果目标不是只做开发期的 A/B，而是希望后续结果可以直接整理进论文，那么 baseline 和 workload 设计也应该围绕当前已经识别出的**长序列 decode 主矛盾**来展开，而不是把 prefill、短序列、工程收敛项全混在一张主图里。

### 14.1 baseline 最好分成“主文主线”和“附录消融”两层

主文推荐保留 4 个 baseline，形成一条清晰的优化链：

| baseline | 相对前一版增加什么 | 为什么需要它 | 最该观察什么 |
|---|---|---|---|
| `B0: Exp-FlashMLA` | 当前实验性 `FlashMLA` 基线，不改 `K` 槽深、不改 `BRISC` 职责、不改 tree 协议 | 作为所有 speedup 和 stall-share 的 `1.0x` 参考 | kernel latency、`reader reserve`、`writer cb_wait` |
| `B1: +K-Stream` | 在 `B0` 上加入 `k_pipeline_slots=3`，再做小范围 `k_page_size` 枚举 | 这是最直接打在 `K reserve` 主矛盾上的版本 | `reader reserve share`、`k_reserve`、`compute reserve-back` |
| `B2: +BRISC-Offload` | 在 `B1` 上加入 Q 预分发和 tail-mask 热路径外移 | 让 `BRISC` 从 steady-state 控制面中退出一部分职责 | `writer cb_wait`、`sender_cb_wait`、`tree_child_wait` |
| `B3: +Stream-Tree` | 在 `B2` 上加入 active-tree step 压缩和 block-streaming reduction | 这是最适合作为论文 `full optimized` 版本的组合 | 长序列 latency、`sender_cb_wait`、`tree_child_wait`、`reserve-back` |

如果只想先做最小可发表版本，那么主结果先跑 `B0 / B1 / B2 / B3` 就够了。

附录里再放更细的消融，用来回答“到底是哪一刀在起作用”：

| 消融 baseline | 只打开什么 | 最适合回答的问题 |
|---|---|---|
| `A1: slots-only` | 只做 `k_pipeline_slots=3`，不调 `k_page_size` | `K` 路收益主要来自“加深流水”还是“page 粒度调参” |
| `A2: Q-only` | 只做 Q 预分发，不做 tail-mask 外移 | `BRISC` 减负里 Q 分发占了多少 |
| `A3: tree-schedule-only` | 只做 active-tree step 压缩，不做 block-streaming | tree 优化里“低风险压缩”和“真正性能改动”各自贡献多少 |
| `A4: no-dummy-handoff` | 只去掉 dummy handoff / 收敛空设备语义 | 收益是不是主要来自稳定性和多 SP 正确性，而不是单核时延 |
| `A5: shared-planner` | 只做 planner 抽象，不改 kernel 行为 | 证明它的价值主要是 plan 等价性和后续实验效率，而不是直接 runtime 提升 |

这套分层有一个额外好处：主文里讲“主矛盾怎么一步步被拆开”，附录里再讲“内部每个 patch 的边际贡献”，两条叙事不会互相打架。

### 14.2 主结果应该做在 `prefill` 还是 `decode`

主性能结果应该放在 `decode`，而且应以**中长到长序列 decode** 为主；`prefill` 更适合作为非回归和负对照。

理由很直接：

- 当前所有高优先级方向，几乎都来自长序列 decode 的 stall 迁移。
- `K` 路加深、`BRISC` 减负、tree 流式化，都是在 decode steady-state 下最容易被放大的结构问题。
- `prefill` 当然也要测，但更适合回答“这些 decode 优化有没有伤到其他路径”，而不是作为论文主卖点。

因此实验角色可以这样分：

- 主性能图：`decode`
- 主机理图：`decode`
- 系统级结果：fused `AttentionBlock` / `PreSDPA` 的 `decode`
- 补充回归表：`prefill`
- 边界正确性：多 SP / 空设备 / tail case

### 14.3 推荐 workload 矩阵

如果要兼顾“可解释”和“论文版面”，建议按下面几组 workload 来组织：

平台口径上，主 device-side 数据应优先来自 `Blackhole`；`Wormhole` 更适合作为 bring-up / reference 对照，因为当前实验路径在 `WH` 上仍可能走 fallback。

| 实验目的 | 路径 | 推荐 case | 推荐 batch | 说明 |
|---|---|---|---|---|
| 主性能曲线 | standalone `FlashMLA` decode | `decode_4k / 8k / 16k / 32k` | **固定 `batch=1`** | 最好只让 `seq_len` 一个变量变化，避免把长度效应和 batch 效应混在一起 |
| 主机理 profile | standalone `FlashMLA` decode | `decode_16k / 32k` | `batch=1` | 这两个点最稳定，最容易看出 `K reserve`、sender/tree、`reserve-back` 的联动 |
| 系统级结果 | fused `AttentionBlock` 或 `PreSDPA + FlashMLA` decode | `decode_4k / 16k / 32k` | `batch=1`，可补 `batch=2` | `BRISC` 减负和 dummy 收敛在 fused 路径更容易被放大 |
| 非回归验证 | `prefill` | `prefill_256 / 1k / 4k` | `batch=1` | 不需要扫很密，只要证明主优化没有明显伤到 prefill |
| 吞吐补图 | standalone 或 fused decode | 固定 `decode_16k`，扫 `batch=1 / 2 / 4` | 扫 batch | 这张图不要和主长度 sweep 混在一起，单独汇报 |
| 边界正确性 | decode / 多 SP | `seq_len < device_chunk_size`、刚好跨一个 `device_chunk_size`、部分设备空跑 | 贴近真实 SP 配置 | 专门验证 dummy handoff 去除和空贡献设备语义 |

如果时间有限，最小主结果可以先只保留：

1. `decode_4k`
2. `decode_16k`
3. `decode_32k`

这三个点已经足够体现“中序列开始起效，长序列收益进一步放大”的迁移趋势。

### 14.4 输入 shape 和程序配置最好固定哪些量

为了让 baseline 之间的比较尽量干净，建议先固定一套 canonical shape：

- `num_heads = 32`
- `num_kv_heads = 1`
- `kv_lora_rank = 512`
- `qk_rope_head_dim = 64`
- `block_size = 64`
- `k_chunk_size = 128`
- `max_cores_per_head_batch` 保持不变
- `math_fidelity`、`exp_approx_mode`、tensor layout、memory placement 都不要在 baseline 间切换

如果主图要做长度 sweep，推荐把 `batch` 固定在 `1`。原因不是 `batch=1` 一定最真实，而是：

- 这样最容易把“序列变长”对 reader / writer / compute 的影响单独拎出来。
- 当前开发脚本里短序列 decode 常用 `batch=2`，长序列常用 `batch=1`，这对开发排查方便，但不利于论文里做严格 apples-to-apples 对比。

如果确实还想展示吞吐随 batch 的变化，最好额外做一张固定 `seq_len=16k` 的 batch sweep 图，而不是和主长度曲线写在一起。

### 14.5 `Q/K/V` 输入应该怎么构造

论文主结果并不需要花太多精力去构造“特殊数值分布”的输入，因为这里的主要收益来自数据流、同步协议和片上缓冲节拍，首先是 shape-sensitive，而不是 value-sensitive。

比较稳妥的输入口径是：

1. 所有 baseline 使用同一个随机种子。
2. 同一组对比里复用完全相同的 `Q`、`K` 和 page table。
3. `decode` 输入按“一步 query + 全长 paged KV cache”构造：
   - `Q` 形状是 `(1, batch, num_heads, d_qk)`
   - `K` 形状是 `(batch, num_kv_heads, seq_len, d_qk)`
   - `cur_pos = seq_len - 1`
4. `prefill` 输入按“整段 query + 整段 KV”构造：
   - `Q` 形状是 `(batch, num_heads, seq_len, d_qk)`
   - `K` 形状是 `(batch, num_kv_heads, seq_len, d_qk)`
5. page table 在同一组实验内固定，不要让 baseline 之间的页映射变化。
6. `Q` 用 `bfloat16` 随机输入即可；paged `K` 的 dtype、layout、memory config 保持与当前路径一致，不要为了“调数据”去额外改实现口径。

如果后面想补一组更贴近真实模型的验证，可以再加：

- 一组由真实 decode step 捕获的 `Q/KV cache` 回放，作为附录或补充图。

但主图没必要一开始就上真实 trace，因为那会让实验可复现性和排查成本都上升。

同样，不建议把下面这些输入当主结果：

- 全零输入
- 常数输入
- 特意缩窄方差的“过干净”输入

这些更适合 debug，不适合做论文主图。

### 14.6 不需要把主要精力放在单独的 `V` 路输入变体上

当前实验路径里，`V` 路问题已经被大幅收缩到 `K` buffer 语义里，所以论文里的输入变化优先级不应该放在“单独造不同 `V` 分布”上，而应该放在：

- `seq_len`
- `batch`
- page table / page 粒度
- `Q` shard 布局
- fused 与 standalone 两条路径的切换

也就是说，真正值得扫的是**工作负载结构**，不是 `V` 值域本身。

### 14.7 哪些 baseline 只跑 standalone 就够，哪些必须上 fused

最好不要把所有 baseline 都只在一个路径上测完。更合理的分法是：

| baseline | standalone `FlashMLA` | fused `AttentionBlock` / `PreSDPA` | 为什么 |
|---|---|---|---|
| `B0` | 要跑 | 要跑 | 所有后续结果都要有共同参考点 |
| `B1: +K-Stream` | **必须跑** | 可选补跑 | 这一步主要是 `K` 路 reader / compute 节拍问题，standalone 最干净 |
| `B2: +BRISC-Offload` | 建议跑 | **必须跑** | Q 预分发和 tail-mask 外移在 fused 路径更容易放大真实收益 |
| `B3: +Stream-Tree` | **必须跑** | 建议跑 | tree 改动的机理分析最好先在 standalone 看清楚，再到 fused 看系统收益 |
| `A4: no-dummy-handoff` | 可不强调 | **必须跑** | 这个方向的价值主要在多 SP 正确性和 fused 边界收敛 |
| `A5: shared-planner` | 要做 plan 对比 | 要做 plan 对比 | 它更像工程性 baseline，而不是单纯 runtime baseline |

换句话说：

- 想看机理，优先 standalone。
- 想看真实系统收益，尤其是 `BRISC`、dummy handoff、Q shard 相关收益，必须补 fused。

### 14.8 建议汇报哪些指标

论文里不要只报一个 latency。至少要把“为什么变快”也一起讲清楚。

推荐把指标分成 4 层：

| 指标层 | 建议指标 | 用来说明什么 |
|---|---|---|
| 主结果 | kernel latency、first-to-last start、相对 `B0` 的 normalized speedup | 最终是否真的更快 |
| reader 侧 | `reader reserve share`、`k_reserve` | `K` 路是否更顺了 |
| writer / tree 侧 | `writer cb_wait`、`sender_cb_wait`、`tree_child_wait` | sender/tree 的推进是否真的更流式了 |
| compute 侧 | `compute wait-front`、`compute reserve-back share` | `TRISC` 是否从“两头都被卡”中被解开了 |

正确性和工程收敛指标则单独汇报：

- 数值一致性
- 多 SP 下不 hang
- 不再需要外部 dummy push
- planner 新旧输出完全一致

运行 protocol 也最好固定下来，不要每张图换一套口径：

- latency 和 detailed profile 最好分两轮 run，避免为了拿 stage breakdown 把所有主结果都绑在 profiler 开销上。
- 每个 case 至少做 `2` 次 warmup，再做 `4~6` 次有效测量；短序列可以多跑几次，长序列保持 `4` 次也足够。
- 主文表格或图中建议汇报 mean latency，必要时附上 std；normalized speedup 统一相对 `B0` 的 mean 计算。

要特别提醒自己的一点是：这些收益**不可线性相加**。例如 `K` 路加深和 tree 流式化都可能同时压 `writer cb_wait`，所以最终总收益通常会小于单项收益之和。

### 14.9 如果只想先跑一版最小论文实验

建议先跑下面这一版：

1. `Blackhole` 上的 standalone `decode_4k / 16k / 32k`
2. baseline 只放 `B0 / B1 / B2 / B3`
3. 所有 case 固定 `batch=1`
4. 输入使用固定 seed 的 `randn` `Q/K`
5. 对 `decode_16k / 32k` 补详细 stage profile
6. 再补一张 fused `decode_4k / 16k / 32k`
7. 最后补一个 `prefill_1k / 4k` 的 `B0 vs B3` 非回归表

这套组合的好处是：

- 主线结论足够清晰
- 版面开销可控
- 机理证据和最终收益能对上
- 后续想扩成更完整的论文实验，也不会推翻这套结构

## 15. 最后的判断

如果只允许先做一件事，就先做：

- `K` 路 slot 深度 + `k_page_size` 调参

如果允许并行开两条线，就开：

1. `K` 路深流水
2. `BRISC` sender/tree 减负

因为从当前证据看，实验性 `FlashMLA` 的主矛盾不是“算得不够快”，而是：

- `K` 路不够顺
- `BRISC` 太忙
- reduction 推进粒度太粗

这三件事一旦不拆开，长序列 decode 就会继续维持现在这种：

- reader 被 `K reserve` 主导
- writer 被 `sender/tree wait` 主导
- compute 同时看到 `wait-front` 和 `reserve-back`

的强耦合回压形态。

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

## 11. 建议的落地顺序

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

## 12. 一页实施清单

| 方向 | 第一刀怎么改 | 主要文件 | 重点观察 |
|---|---|---|---|
| `K` 路加深 | `k_pipeline_slots=3`，推广双槽到 ring | `micro_ops/flash_mla/op.py`、`unified_kernels/flash_mla.hpp` | `reader reserve`、`k_reserve`、`reserve-back` |
| `BRISC` 减负 | 先把 Q 预分发出去，再把 tail mask 热路径外移 | `attention_block/op.py`、`unified_kernels/flash_mla.hpp` | `writer cb_wait`、`sender_cb_wait` |
| tree 流式化 | 先压缩 active-tree step，再尝试分块 reduction | `micro_ops/flash_mla/op.py`、`unified_kernels/flash_mla.hpp` | `tree_child_wait`、`sender_cb_wait` |
| dummy handoff 去除 | 让空设备输出 identity contribution | `unified_kernels/flash_mla.hpp`、`attention_block_kernel.cpp` | 多 SP 不 hang，结果正确 |
| shared planner | 把 grid/page/buffer/runtime args 抽成统一 builder | `micro_ops/flash_mla/op.py`、`attention_block/op.py` | 新旧 planner 输出一致 |

## 13. 最后的判断

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

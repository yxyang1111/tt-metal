# 五种方法实现与数据流深度分析

更新时间：`2026-04-13`

## 0. 一页摘要

### 0.1 先记五句话

- `Reference Attention`：整条路径都在 host 上，用 PyTorch 直接算，没有 TT device 数据流。
- `Flash Attention`：标准 Q/K/V 分离的 **paged decode**，Q 在 L1 分片，K/V 在 DRAM 分页缓存里，Reader 先读 page table，再按页取 K/V。
- `TT-MLA`：和 `Flash Attention` 共用同一套 `sdpa_decode` 主栈，但 **V 不再独立提供**，而是把 V 当成 K hidden 维里的子集来读。
- `FlashMLA-4c`：先按 DeepSeek 4c 几何把 Q/KV 摆成 tiny-tile + L1/ND-shard，然后做一次性 adapter，**真正计时的是 non-paged MLA builtin backend**；Q 槽位上限是 `24`。
- `FlashMLA-8c`：流程和 4c 相同，但几何更宽、Q 槽位上限提高到 `48`，更适合高 `batch` / 高 `q_shards` / 长 `seq_len` 的高压力场景。

### 0.2 五种方法数据流总表

| 方法 | host 侧原始输入 | 上板前关键变换 | 计时时真正输入形态 | Reader 主要取数方式 | V 的来源 | 核组织 / 并行方式 | 中间结果与输出 |
|---|---|---|---|---|---|---|---|
| `Reference Attention` | `q_std / k_std / v_std` | 直接转 `float32` | 无 device 输入 | PyTorch 内部直接读取 host tensor | 独立 `v_std` | 无 TT 核分配 | 中间量全在 PyTorch 内部；最终输出是 host tensor |
| `Flash Attention` | `q_std / k_std / v_std` | Q `permute(2,0,1,3)`；K/V 经 `page_table_setup + to_paged_cache` | Q 为 **L1 height-sharded**；K/V 为 **DRAM paged cache**；page table / cur_pos 为 int32 device tensor | `reader_decode_all.cpp` 先读 `cur_pos`、再读 page table，再按页读 K/V chunk | 独立 V 张量 | `sdpa_decode_program_factory.cpp` 决定 worker / reducer / output core；多核沿 K chunk 并行 | `l / m / partial o` 只在片上归约；root 最终写 DRAM 输出，不回传完整 score |
| `TT-MLA` | `q_mla / k_mla` | Q 同样 `permute`；K 经 paged cache 重排；无独立 V 上板 | Q 为 **L1 height-sharded**；K 为 **DRAM paged KVPE** | 与 `Flash Attention` 同一 Reader；paged 方式读 K chunk | **V 是 K 的前 `head_dim_v` 列**，按 MLA 语义从 K 里解释 | 与 `Flash Attention` 共用同一套 `sdpa_decode` 核分配 | 中间量同样只在片上归约；省掉独立 V 的 host/device 数据流 |
| `FlashMLA-4c` | `q_deepseek / k_deepseek` | 先按 4c S-block 几何摆成 **tiny-tile Q + ND-sharded KV**；再做一次性 adapter 转成标准 TILE + DRAM | **计时阶段真正看到的是标准 TILE + DRAM backend Q/K**，且是 **non-paged MLA decode** | 计时阶段 Reader 不读 page table，直接线性读取 K chunk | V 逻辑上仍来自 K | 4c grid：`6 blocks x 4 cores = 24` 活跃 Q 槽位；`max_cores_per_head_batch` 受原始 Q shard 数影响 | 中间量仍是 `l / m / partial o`；有一次性 host round-trip adapter，但不计入 timed op |
| `FlashMLA-8c` | `q_deepseek / k_deepseek` | 与 4c 相同，但初始几何改为 8c | 与 4c 相同：计时阶段仍是标准 TILE + DRAM backend | 与 4c 相同：non-paged K 读取 | 与 4c 相同：V 来自 K | 8c grid：`6 blocks x 8 cores = 48` 活跃 Q 槽位；同样影响 builtin backend 的 program 配置 | 中间量流动方式与 4c 相同，但更宽几何更容易承接高压力映射 |

### 0.3 怎么快速区分这五种方法

1. 先问它是不是 device 方法。不是的话，只可能是 `Reference Attention`。
2. 如果是 device 方法，再问它是不是 `paged decode`。是的话只可能是 `Flash Attention` 或 `TT-MLA`。
3. 在 paged 路径里，再问 V 是否独立存在。独立就是 `Flash Attention`；V 来自 K 就是 `TT-MLA`。
4. 如果不是 paged，再问它是不是 `deepseek_flash_mla` 这条 baseline。是的话就是 `FlashMLA-4c/8c`。
5. 在 `FlashMLA-4c/8c` 里，最本质的区别不是数学，而是几何：`24` 个还是 `48` 个可用 Q 槽位，以及由此带来的映射和配置差异。

## 1. 范围与口径

这份文档对应当前 Wormhole Part1 五方法对比表里的五个名字：

1. `Reference Attention`
2. `Flash Attention`
3. `TT-MLA`
4. `FlashMLA-4c`
5. `FlashMLA-8c`

这里有三个口径先说清楚：

1. **本文讲的是当前 Part1 benchmark 真正走到的实现路径。** 入口以 `mla_flash_attention_dev/experiments/part1_baselines/run_part1_benchmarks.py` 为准。
2. **本文主要讨论 decode。** 现有 `wh_five_method_all_cases.md` 全是 decode 点，因此这里不展开 prefill。
3. **`FlashMLA-4c/8c` 需要特别区分“当前 benchmark 实际计时的路径”和“相关设计文档描述的原生/实验性 S-block 数据流”。**  
   当前 benchmark 里，这两者并不是完全一回事：
   - benchmark 先按 4c/8c 的 DeepSeek 布局把 Q/KV 摆成 tiny-tile + ND-shard；
   - 然后做一次性 host adapter，把它们转成 Wormhole 内建 `flash_multi_latent_attention_decode` 期望的标准 TILE + DRAM 张量；
   - **真正被计时的 device op 是后者。**

## 2. 五种方法与代码入口总览

| 表中名称 | benchmark key / 选择方式 | 主要入口 | 最终算子 API | 是否 paged | 是否显式提供 V |
|---|---|---|---|---|---|
| `Reference Attention` | `reference_attention` | `reference_decode()` | `torch.nn.functional.scaled_dot_product_attention` | 否 | 是 |
| `Flash Attention` | `flash_attention` | `build_decode_tt_inputs()` + `run_flash_attention_decode()` | `ttnn.transformer.paged_scaled_dot_product_attention_decode` | 是 | 是 |
| `TT-MLA` | `flash_mla` | `build_decode_tt_inputs()` + `run_flash_mla_decode()` | `ttnn.transformer.paged_flash_multi_latent_attention_decode` | 是 | 否，V 逻辑上来自 K |
| `FlashMLA-4c` | `deepseek_flash_mla` + `--deepseek-wh-cores-per-block 4` | `build_deepseek_decode_tt_inputs()` + `run_deepseek_flash_mla_decode()` | `ttnn.transformer.flash_multi_latent_attention_decode` | 否 | 否，V 逻辑上来自 K |
| `FlashMLA-8c` | `deepseek_flash_mla` + `--deepseek-wh-cores-per-block 8` | 同上 | 同上 | 否 | 否，V 逻辑上来自 K |

## 3. 五种方法共同的输入起点

所有方法的 host 输入都由 `make_torch_inputs(workload)` 构造。当前 Part1 配置里，这些输入最初都在 **host / CPU 侧**，dtype 统一来自 `torch_dtype_from_config(config)`，当前实验口径下就是 `torch.bfloat16`。

decode 下 `q_seq_len = 1`，因此所有 Q 都是“当前 token 的 query”，而 K/V 或 KV cache 覆盖整个历史 `seq_len`。

### 3.1 标准 Attention 路径输入

| 张量 | 形状 | 语义 |
|---|---|---|
| `q_std` | `(B, num_heads, q_seq_len, std_head_dim)` | 标准注意力的 Q |
| `k_std` | `(B, num_kv_heads, seq_len, std_head_dim)` | 标准注意力的 K |
| `v_std` | `(B, num_kv_heads, seq_len, std_head_dim)` | 标准注意力的 V |

### 3.2 TT-MLA 主线路径输入

| 张量 | 形状 | 语义 |
|---|---|---|
| `q_mla` | `(B, num_heads, q_seq_len, mla_head_dim_qk)` | 主线 MLA 的 Q |
| `latent_mla` | `(B, num_kv_heads, seq_len, mla_head_dim_v)` | K/V latent 部分 |
| `rope_mla` | `(B, num_kv_heads, seq_len, mla_d_rope)` | rope 部分 |
| `k_mla` | `cat([latent_mla, rope_mla], dim=-1)` | 主线 MLA 的合并 K / KVPE |

这里 `mla_head_dim_qk = mla_head_dim_v + mla_d_rope`。

### 3.3 DeepSeek FlashMLA 路径输入

| 张量 | 形状 | 语义 |
|---|---|---|
| `q_deepseek` | `(B, num_heads, q_seq_len, deepseek_kvpe_dim)` | DeepSeek Q |
| `latent_deepseek` | `(B, num_kv_heads, seq_len, deepseek_kv_lora_rank)` | KV latent 部分 |
| `rope_deepseek` | `(B, num_kv_heads, seq_len, deepseek_qk_rope_head_dim)` | rope 部分 |
| `k_deepseek` | `cat([latent_deepseek, rope_deepseek], dim=-1)` | DeepSeek 合并 KVPE |

这里：

- `deepseek_kvpe_dim = deepseek_kv_lora_rank + deepseek_qk_rope_head_dim`
- V 逻辑上是 KVPE 的前 `deepseek_kv_lora_rank` 列

### 3.4 统一的 scale

- `scale_std = std_head_dim ** -0.5`
- `scale_mla = mla_head_dim_qk ** -0.5`
- `scale_deepseek = deepseek_qk_head_dim ** -0.5`

## 4. 方法一：Reference Attention

### 4.1 数据最初在哪里，是什么形式

- 数据最初在 **host / CPU**。
- 输入使用 `q_std / k_std / v_std`，初始 dtype 是 `torch.bfloat16`。

### 4.2 谁来取，如何传递

1. `reference_decode()` 先把 `q_std / k_std / v_std` 全部 `.to(torch.float32)`。
2. 然后调用 `scaled_dot_product_attention_reference()`。
3. 这个 helper 最终落到 `torch.nn.functional.scaled_dot_product_attention`。
4. decode 时它还显式传入：
   - `start_indices=[seq_len - 1] * B`
   - `padded_layer_len=nearest_y(seq_len, k_chunk_size)`
   - `is_causal=True`

也就是说，这条路径根本不上 TT device，不存在 DRAM/L1/NoC 的数据搬运。

### 4.3 计算核如何分配

- **不分配 TT 计算核。**
- 真正的线程/向量化并行由 PyTorch/CPU runtime 自己决定，当前仓库没有对它进一步建模。

### 4.4 中间结果与输出如何流动

- score、softmax、`P @ V` 等中间量都留在 PyTorch 内部。
- 没有 page table。
- 没有 paged cache。
- 没有 host 和 device 之间的中间回传，因为整条链都在 host。
- 最终输出是一个 host 上的 `torch.Tensor`。

### 4.5 这条路径的本质

- 它是 **正确性与语义基线**，不是 device 数据流基线。
- 对五方法横向对比来说，它最大的意义是“结果参考”，不是“片上数据流参考”。

## 5. 方法二：Flash Attention

### 5.1 数据最初在哪里，是什么形式

- 初始输入来自 `q_std / k_std / v_std`，最早都在 **host / CPU**。
- decode 前，Q 会被重排成：
  - `q_for_tt = q_std.permute(2, 0, 1, 3)`，形状变为 `(q_seq_len, B, H, D)`，decode 下即 `(1, B, H, D)`。
- K/V 不直接原样上传，而是先经过：
  - `PagedAttentionConfig(block_size, max_num_blocks)`
  - `page_table_setup()`
  - `to_paged_cache()`

因此，这条路径在真正上板前，host 侧已经先把 K/V 变成了 **paged KV cache 语义**。

### 5.2 host 侧如何上板

`build_decode_tt_inputs()` 里会生成以下 device 张量：

- `tt_q`：
  - `ttnn.from_torch(...)`
  - dtype = `ttnn.bfloat16`
  - layout = `ttnn.TILE_LAYOUT`
  - memory = **L1 height-sharded**
- `tt_k` / `tt_v`：
  - dtype = `ttnn.bfloat8_b`
  - layout = `ttnn.TILE_LAYOUT`
  - memory = **DRAM**
- `tt_page_table`：
  - dtype = `ttnn.int32`
  - layout = `ttnn.ROW_MAJOR_LAYOUT`
- `tt_cur_pos`：
  - device 上的 int32 向量，值为 `[seq_len - 1] * B`

### 5.3 谁来取，如何传递

真正被 benchmark 的 op 是：

- `ttnn.transformer.paged_scaled_dot_product_attention_decode(...)`

它进一步进入：

- `ttnn::transformer::paged_scaled_dot_product_attention_decode`
- `ttnn::prim::sdpa_decode(...)`

device 侧的三段式角色是：

- `reader_decode_all.cpp`：Reader / NCRISC
- `sdpa_flash_decode.cpp`：Compute / TRISC
- `writer_decode_all.cpp`：Writer / BRISC

具体数据流如下：

1. Reader 先读取 `cur_pos`，再据此计算本核负责的 K chunk 范围。
2. Reader 从 Q 的 sharded L1 位置读 Q 到 `cb_q_in`。
3. Reader 把 page table 从 DRAM 读进 L1 的 `c_9` buffer。
4. 对每个分配到的 K chunk：
   - 通过 page table 找到物理页；
   - 从 DRAM 把 K chunk 读到 `cb_k_in`；
   - 再把对应的 V chunk 读到 `cb_v_in`；
   - 如果 factory 运行时打开了 `use_k_mcast`，还可以走 K 的片上 multicast 路径。
5. Compute 在 L1 / DEST 中执行分块 flash attention：
   - `QK^T`
   - scale
   - causal mask
   - online softmax
   - `P @ V`
   - final normalize
6. Writer 若发现一个 head/batch 被多个 worker 核分担，就会做 tree reduction：
   - 子核把局部的 `l / m / o` 写到父核的中间 buffer；
   - 父核继续合并；
   - root 最终写出结果。

### 5.4 计算核如何分配

这条路径的核分配由 `sdpa_decode_program_factory.cpp` 完成。关键量包括：

- `num_cores_per_head`
- `num_heads_per_core`
- `num_cores_per_batch`
- `num_reducer_cores`
- `num_output_cores`
- `num_active_cores`

当前这批 benchmark 配置里 `num_kv_heads=1`，因此很容易出现：

- 一个 batch 对应一个 output core；
- 多个 worker core 沿 K chunk 方向并行；
- writer 再把局部 `l/m/o` 向 root 做归约。

Q 在 host 侧已经先按 height-shard 切到了多个 core 上，program factory 再基于 grid 和 `max_cores_per_head_batch` 决定真正参与计算、归约和输出的 core 集合。

### 5.5 中间结果是否回传

- **不会把完整 attention score 矩阵写回 DRAM。**
- 中间量 `m / l / partial o` 只在片上 L1 / CB / DEST 里流动，并通过 NoC 在 worker 和 root 之间传递。
- 计时循环里不会把输出转回 host。
- benchmark 每轮只是：
  - 发起 device op
  - `synchronize_device`
  - 释放输出 tensor

所以它测到的是 device op 本身，而不是 host 回传成本。

### 5.6 这条路径的本质

- 它是 **标准 Q/K/V 分离** 的 paged decode 路径。
- 它的最大特征是：
  - K/V 是分页缓存；
  - V 是独立张量；
  - 数据流核心在 page table + DRAM 读 + 片上归约。

## 6. 方法三：TT-MLA

### 6.1 数据最初在哪里，是什么形式

- 初始输入来自 `q_mla / k_mla`，最早也都在 **host / CPU**。
- `k_mla` 已经在 host 侧把：
  - `latent_mla`
  - `rope_mla`
  拼成了一个 KVPE 张量。
- **没有独立的 `v_mla`。**

### 6.2 host 侧如何上板

`build_decode_tt_inputs()` 在 `baseline_key == "flash_mla"` 时会做：

- `q_torch = q_mla`
- `k_torch = k_mla`
- `v_torch = None`

然后和 Flash Attention 一样：

- Q 走 `permute(2, 0, 1, 3)`
- K 走 `page_table_setup + to_paged_cache`
- Q 上板到 **L1 height-sharded**
- K 上板到 **DRAM**
- page table 和 cur_pos 上板到 int32 device tensor

### 6.3 谁来取，如何传递

真正被 benchmark 的 op 是：

- `ttnn.transformer.paged_flash_multi_latent_attention_decode(...)`

它最终也进入同一套：

- `ttnn::prim::sdpa_decode(...)`

但和 Flash Attention 的关键区别是：

1. `use_mla = true`
2. `head_dim_v = mla_head_dim_v`
3. `V tensor` 可以缺省

C++ 侧明确写了：

- 如果 `use_mla` 为真且没有传 `V`，算子就把 `V` 当成 `K` hidden 维上的一个子集。

因此数据流变成：

1. Reader 仍然按 paged 方式读 K chunk。
2. 但 `read_v(...)` 可以走 `reuse_k` 语义：
   - 不必再从 DRAM 另读一份 V；
   - 而是直接把已经读入/缓存的 K 数据按 `head_dim_v` 的视角解释为 V。
3. Compute 做的还是 flash attention，但 `P @ V` 用的 V 宽度变成了 `head_dim_v`。
4. Writer 仍然用 tree reduction 合并局部 `l / m / o`，最后写出结果。

### 6.4 计算核如何分配

核分配和 Flash Attention 共享同一套 `sdpa_decode_program_factory.cpp`。

也就是说：

- worker / reducer / output core 的定义是一致的；
- batch/head/chunk 的分法是一致的；
- 真正不同的是 reader/compute 对 `V` 的理解：
  - Flash Attention：V 独立存在；
  - TT-MLA：V 逻辑上附着在 K 上。

### 6.5 中间结果是否回传

- 和 Flash Attention 一样，不会把完整 score matrix 落 DRAM。
- `l / m / partial o` 只在片上归约。
- 计时循环不做 host 回传。

### 6.6 这条路径的本质

- 它本质上不是一套独立的新 decode 栈，而是 **复用 `sdpa_decode` 主栈的 MLA-aware 变体**。
- 相对 Flash Attention，它节省掉的最关键一项是：
  - **显式 V 的 host 构造、上板与 device 读流。**

## 7. 方法四与方法五：FlashMLA-4c / FlashMLA-8c

`FlashMLA-4c` 和 `FlashMLA-8c` 在当前 benchmark 里走的是同一条 baseline：`deepseek_flash_mla`。它们的区别不是 API 名字变了，而是：

- `--deepseek-wh-cores-per-block 4`
- `--deepseek-wh-cores-per-block 8`

对应的 grid 类分别是：

- `FlashMLAOptimalGridNOC0_WH`
- `FlashMLAOptimalGridNOC0_WH_8C`

为了不把事情说乱，这里把它们拆成两层来讲。

### 7.1 当前 Part1 benchmark 里真实发生的链路

#### 7.1.1 数据最初在哪里，是什么形式

- 初始输入来自 `q_deepseek / k_deepseek`，最早都在 **host / CPU**。
- `k_deepseek = cat([latent_deepseek, rope_deepseek], dim=-1)`。
- 这里同样 **没有独立 V 张量**；V 是 KVPE 的前 `deepseek_kv_lora_rank` 列。

#### 7.1.2 第一步：按 4c/8c 几何生成 DeepSeek-native device tensor

`build_deepseek_decode_tt_inputs()` 先选择 grid：

- 4c：6 个 S block，每个 block 4 核，总共 **24** 个活跃 Q 槽位
- 8c：6 个 S block，每个 block 8 核，总共 **48** 个活跃 Q 槽位

两者都沿用相同的 6-bank 拓扑顺序：

- `(1, 2, 0, 4, 9, 8)`

也沿用相同的 3 步 tree reduction 顺序：

1. `S2 -> S1`, `S4 -> S3`, `S6 -> S5`
2. `S3 -> S1`
3. `S5 -> S1`

这一步里真正放到 device 上的张量是：

- Q：
  - `q_deepseek.permute(2, 0, 1, 3).contiguous()`
  - tiny tile = `(deepseek_num_q_heads_per_core, 32)`
  - **L1 height-sharded**
  - shard grid 直接取自 `all_active_cores[: batch * num_q_shards]`
- KV cache：
  - **DRAM ND-sharded**
  - shard shape = `[1, num_kv_heads, k_chunk_size, deepseek_kvpe_dim]`
  - shard distribution = `ROUND_ROBIN_1D`
  - grid = `grid.optimal_dram_grid()`

因此，4c/8c 的第一层差异非常明确：

- 4c 最多允许 `batch * q_shards <= 24`
- 8c 最多允许 `batch * q_shards <= 48`

#### 7.1.3 第二步：一次性 adapter，把 DeepSeek-native tensor 变成 WH 内建 backend tensor

这是当前 benchmark 最容易被忽略、但最关键的一步。

`FlashMLADecode._to_wh_backend_tensor()` 会对 Q 和 KV 各做一次：

1. `ttnn.to_torch(tensor)`：device -> host
2. `.to(torch.bfloat16)`
3. `ttnn.from_torch(...)`：host -> device
4. 目标 layout 固定为：
   - `TILE_LAYOUT`
   - `DRAM_MEMORY_CONFIG`

也就是说，进入真正计时的 device op 之前：

- 原本的 tiny-tile、L1-sharded Q 不再直接参与计时；
- 原本的 ND-sharded KV 也不再直接参与计时；
- 它们都被一次性物化成了 **标准 TILE + DRAM** 的 backend 张量。

同一阶段里，`_build_wh_sdpa_program_config()` 还会根据原始 Q shard grid 的 core 数，生成一个新的 `SDPAProgramConfig`，把 `max_cores_per_head_batch` 设成相应值。

#### 7.1.4 第三步：真正被计时的 device op

真正 benchmark 的是：

- `ttnn.transformer.flash_multi_latent_attention_decode(...)`

也就是：

- `is_paged = false`
- `use_mla = true`
- `cur_pos` 直接是 Python `list[int]`
- `V tensor = None`

它最终仍然进入：

- `ttnn::prim::sdpa_decode(...)`

因此，**当前 Part1 里的 FlashMLA-4c/8c，计时时实际上跑的是“主线 non-paged MLA decode 后端”，不是直接跑 `micro_ops/flash_mla` 的 unified kernel。**

#### 7.1.5 计时阶段里，数据是怎么流的

在真正计时的 builtin backend 阶段：

1. Reader 不再需要 page table，因为这里是 **non-paged**。
2. Reader 走 `read_kv_mask_chunks(...)` 这条 non-paged 读取路径：
   - 线性从 DRAM 读取 K chunk；
   - V 仍然逻辑上来自 K；
   - 因为 `V tensor` 缺省，底层仍然按 MLA 语义解释 `head_dim_v`。
3. Compute 继续用 flash attention 的在线 softmax 形式处理 chunk。
4. Writer 继续用 tree reduction 合并局部结果并写 DRAM 输出。

#### 7.1.6 4c 和 8c 在当前 benchmark 里到底差在哪里

在“当前真实计时路径”里，4c 和 8c 的差异主要通过三件事进入：

1. **前置合法性边界不同**
   - 4c 只允许 `required_q_cores <= 24`
   - 8c 允许到 `48`
2. **Q 原始 shard 数不同**
   - 它影响 adapter 之前的摆放
3. **`max_cores_per_head_batch` 不同**
   - 这会改变 builtin `sdpa_decode` 在计时阶段的 program 配置与可用并行度上限

所以当前 benchmark 的 4c/8c 对比，本质上是：

- **4c/8c 的几何约束 + 一次性 adapter 之后的 builtin WH MLA decode 对比**

而不是：

- **直接计时原生 experimental S-block unified kernel 对比**

### 7.2 相关设计文档中的“原生/实验性 FlashMLA 数据流”

上面讲的是“当前 benchmark 真正跑了什么”。  
下面讲的是“相关实现文档所描述的 FlashMLA-4c/8c 原生数据流长什么样”。这对理解 4c 和 8c 的设计初衷仍然非常重要。

#### 7.2.1 Q 在哪里，谁来取

相关文档把每个 batch/q-shard 组的 Q 视为：

- 先落在 output core / S1 某个核心的 L1；
- 然后由 block 内或全组 worker 从 output core 的 L1 拉取或广播。

Q 很小，因此原生设计不让每个 worker 都回 DRAM 读 Q，而是倾向于：

- **让 Q 在片上复用**

#### 7.2.2 K 在哪里，谁来取

K / KV cache 放在：

- **ND-sharded DRAM**

并且 bank 顺序故意对齐到：

- `grid.OPTIMAL_DRAM_BANK_ORDER`

每个 S block 内会有一个 sender 核：

1. 先从自己绑定的 DRAM bank 把 K page / K chunk 读到本地 L1；
2. 再通过 BRISC / NoC multicast 把 K page 扩散给 block 内其他 receiver 核。

这就是 FlashMLA 文档里反复强调的：

- **K 是唯一的大流量对象**
- **应该“少读 DRAM，多做片上 multicast”**

#### 7.2.3 V 如何传

原生 FlashMLA 设计里最关键的点之一是：

- **V 不单独占一条大数据流**

因为 KVPE 里：

- K/V 共存在同一张 tensor 里
- V 只是前 `head_dim_v` 列

所以 TRISC 在做 `P @ V` 时，直接从已经读进来的 K buffer 里按列切出 V 视图即可。  
这意味着：

- 不需要单独的 V DRAM 读
- 不需要单独的 V multicast
- 不需要单独的 `cb_v_in` 大缓冲

#### 7.2.4 核如何分配

原生设计把核心按 S block 组织：

- 4c：6 个 block × 4 核
- 8c：6 个 block × 8 核

常见的空间含义是：

- 一个 batch/q-shard 组，会占用每个 S block 的一个位置；
- 各 block 上相同“位置”的核负责不同 sequence chunk；
- chunk 以 `stride = num_s_blocks` 的方式分配

这样做的直接好处是：

- sender 核重复访问同一个 DRAM bank；
- block 内其他核只走片上 multicast；
- block 间再通过 tree reduction 把局部结果并起来。

#### 7.2.5 中间结果如何传

原生设计里不会把完整 score matrix 落到 DRAM。  
真正跨核流动的是：

- 局部输出 `O`
- online softmax 统计量 `m`
- online softmax 统计量 `l`

这些局部量会通过 tree reduction 按步合并。对 Wormhole 4c/8c，这个顺序在 grid 类里已经硬编码好了。

#### 7.2.6 4c 与 8c 的原生差异

4c 和 8c 的原生差异并不在“数学公式”上，而在 **几何和吞吐组织方式** 上：

- 4c：每个 S block 只有 4 个活跃核，几何更紧凑，单块扇出更小
- 8c：每个 S block 拓宽到 8 个活跃核，单块可容纳更多 Q shard，multicast 目的地更多，device 级 chunk 也更宽
- `FlashMLAProgramConfig` 还把 `device_chunk_size` 绑定成 `CORES_PER_BLOCK * k_chunk_size`，因此 8c 不只是 Q 槽位翻倍，单个 device stripe 的名义宽度也随之加大

因此 8c 的原生设计目标有两个：

1. 先把 `q_shards=4` 路线上的容量边界从 24 提到 48
2. 在高压力区间让 block 内负载更均匀、让长序列端更容易摊薄通信成本

## 8. 五种方法横向比较：数据流最本质的差别

| 维度 | Reference Attention | Flash Attention | TT-MLA | FlashMLA-4c | FlashMLA-8c |
|---|---|---|---|---|---|
| 输入最初位置 | host CPU | host CPU | host CPU | host CPU | host CPU |
| decode 是否 paged | 否 | 是 | 是 | 否 | 否 |
| V 是否独立存在 | 是 | 是 | 否 | 否 | 否 |
| K/V 初始 device 形式 | 无 | paged K + paged V in DRAM | paged K in DRAM，V 逻辑上附着在 K | 先 tiny/L1 + ND-shard，再 adapter 成 DRAM | 同左，但 grid 更宽 |
| Q 初始 device 形式 | 无 | L1 height-sharded | L1 height-sharded | 先 tiny-tile L1 shard，再 adapter 成 DRAM | 同左，但可占更多 active cores |
| 主要 reader | PyTorch 内部 | `reader_decode_all.cpp` | `reader_decode_all.cpp` | 计时阶段同样是 `reader_decode_all.cpp` | 同左 |
| 主要 compute | PyTorch 内部 | `sdpa_flash_decode.cpp` | `sdpa_flash_decode.cpp` | 计时阶段同样是 `sdpa_flash_decode.cpp` | 同左 |
| 主要 writer | PyTorch 内部 | `writer_decode_all.cpp` | `writer_decode_all.cpp` | 计时阶段同样是 `writer_decode_all.cpp` | 同左 |
| 是否存在 host 中间回传 | 否，整条链都在 host | 否 | 否 | **有一次性 adapter host round-trip，但不在计时里** | 同左 |
| 完整 score 是否写回外存 | 否 | 否 | 否 | 否 | 否 |
| 片上归约对象 | 无 | `l / m / partial o` | `l / m / partial o` | 计时阶段同左；原生设计也同样是这三类局部量 | 同左 |

## 9. 最关键的三条结论

1. **`Flash Attention` 和 `TT-MLA` 的 decode 栈本质上共用同一套 `sdpa_decode` 主框架。**  
   真正的分水岭不是 reader/compute/writer 三段式是否不同，而是：
   - Flash Attention 有独立 V
   - TT-MLA 把 V 当成 K hidden 维里的子视图

2. **当前 Part1 的 `FlashMLA-4c/8c`，计时上并不是直接跑 experimental unified kernel。**  
   它们先按 4c/8c 几何摆出 DeepSeek-native Q/KV，再一次性适配成 Wormhole builtin backend 张量，然后计时 `flash_multi_latent_attention_decode`。  
   因此当前 4c/8c 的收益，既来自：
   - 更宽的几何允许更多 `required_q_cores`
   也来自：
   - 适配后 builtin backend 所拿到的 program 配置变化

3. **这五种方法里，没有任何一种会把完整 attention map 当成“中间结果”回传到 host。**  
   真正流动的中间量分成两类：
   - Reference：PyTorch 内部的临时张量，只在 host 进程里存在
   - device 方法：L1 / CB / DEST 里的 `Q/K/V chunk` 与 `l/m/partial o`，它们只在片上经 NoC 流动，最后只有输出 tensor 对外可见

## 10. 建议搭配阅读的文件

如果要继续深挖，建议按这个顺序读：

1. `mla_flash_attention_dev/experiments/part1_baselines/run_part1_benchmarks.py`
2. `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp`
3. `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
4. `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`
5. `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp`
6. `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`
7. `mla_flash_attention_dev/docs/flash-mla-current-dataflow-analysis.md`
8. `mla_flash_attention_dev/docs/experimental-flash-mla-dataflow-analysis.md`
9. `mla_flash_attention_dev/docs/flash-mla-dataflow-tiling-placement-deep-dive.md`
10. `mla_flash_attention_dev/docs/tt-metal-flash-attention-dataflow-parallelism.md`

# tt-metal 中 SDPA 算子梳理

本文整理 `tt-metal/ttnn` 推理侧与 SDPA（Scaled Dot Product Attention）相关的公开 API、核心实现、张量约定、并行策略、测试入口和阅读路径，目标是帮助快速建立一张“从 Python 调用到 device kernel”的全景图。

本文聚焦 `tt-metal/ttnn` 推理路径，不展开 `tt-train` 训练侧的 ring attention 前反向实现。若只想先看单核、多核、多芯片的实现直觉，可以先读仓库内已有的 `docs/SDPA_单核多核多芯片实现.md`。

## 1. 全景图

`ttnn.transformer` 下与 SDPA 直接相关的算子大致可以分成下面几类：

| 类别 | Python API | 典型场景 | 主要实现目录 | 说明 |
| --- | --- | --- | --- | --- |
| 标准 prefill SDPA | `ttnn.transformer.scaled_dot_product_attention` | 普通 prompt prefill | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/` | 主线实现，底层是 FlashAttention 风格 |
| Chunked prefill | `ttnn.transformer.chunked_scaled_dot_product_attention` | 长 prompt、paged KV、prefix caching | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/` | 一次处理一个 Q chunk，K/V 来自 paged cache |
| Decode SDPA | `ttnn.transformer.scaled_dot_product_attention_decode` | 单 token decode | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` | 主线是 FlashDecode |
| Paged decode | `ttnn.transformer.paged_scaled_dot_product_attention_decode` | vLLM 风格 paged KV cache | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` | decode 的 paged 版本 |
| Windowed SDPA | `ttnn.transformer.windowed_scaled_dot_product_attention` | 视觉窗口注意力 | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/` | 不传显式 mask，而是传 `cu_window_seqlens` |
| Joint attention | `ttnn.transformer.joint_scaled_dot_product_attention` | 两段序列拼接后做一次 attention | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/` | 当前要求 `joint_strategy="rear"` |
| Ring distributed SDPA | `ttnn.transformer.ring_distributed_scaled_dot_product_attention` | 多设备 causal prefill | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/` | 每个设备负责两个对称 Q chunk |
| Ring joint attention | `ttnn.transformer.ring_joint_scaled_dot_product_attention` | 多设备 joint attention | `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/` | joint 的多设备扩展 |
| MLA 变体 | `flash_mla_prefill` / `chunked_flash_mla_prefill` / `flash_multi_latent_attention_decode` / `paged_flash_multi_latent_attention_decode` | MLA 模型 | `sdpa/` 与 `sdpa_decode/` | 与标准 SDPA 共用大量框架，但有自己的约束 |

如果只抓主线，可以把整个家族先压缩成 4 条：

1. `scaled_dot_product_attention`：标准 prefill。
2. `chunked_scaled_dot_product_attention`：长上下文 prefill。
3. `scaled_dot_product_attention_decode` / `paged_scaled_dot_product_attention_decode`：单 token decode。
4. `windowed` / `ring` / `joint` / `MLA`：围绕特殊模型或特殊部署形态的扩展。

## 2. 实现是怎么分层的

这套实现的一个重要特点是：**Python 层非常薄，真正的行为大多在 C++ host 侧和 device kernel 里。**

典型调用链如下：

```text
Python API (`ttnn.transformer.*`)
  -> nanobind 绑定 (`transformer_nanobind.cpp`, `sdpa_nanobind.cpp`, `sdpa_decode_nanobind.cpp`, ...)
  -> invoke 层 (`sdpa.cpp`, `sdpa_decode.cpp`)
  -> device operation (`*_device_operation.cpp`)
     - 校验输入约束
     - 计算输出 spec
     - 形成 program cache / hash key
  -> program factory (`*_program_factory.cpp`)
     - 决定多核并行划分
     - 分配 circular buffers
     - 设置 compile args / runtime args
     - 拼 reader / writer / compute kernel
  -> kernels (`dataflow/*.cpp`, `compute/*.cpp`)
```

最关键的总入口文件如下：

| 文件 | 作用 |
| --- | --- |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp` | 统一绑定 transformer 相关 op，并暴露 `SDPAProgramConfig` |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp` | 标准 SDPA、chunked、joint、ring、MLA prefill 的 Python 绑定 |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp` | decode、paged decode、MLA decode 的 Python 绑定 |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp` | 标准 prefill / chunked / ring 等 public API 的 invoke 层 |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp` | decode invoke 层 |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp` | prefill 约束校验与 program hash |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp` | prefill 多核划分、KV chain、multicast、kernel 组装 |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp` | decode 约束校验 |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` | FlashDecode 多核划分与树形归约 |
| `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/device/kernels/dataflow/reader_windowed.cpp` | windowed 路径中按 `cu_window_seqlens` 生成 block-diagonal mask |

## 3. 对外 API 和张量约定

### 3.1 标准 prefill：`scaled_dot_product_attention`

这是最基础、也最接近 PyTorch SDPA 语义的一条路径。

- `Q`：`[b, nqh, s, dh]`
- `K`：`[b, nkv, s, dh]`
- `V`：`[b, nkv, s, dh]`
- 输出：`[b, nqh, s, dh]`
- 常用参数：`attn_mask`、`is_causal`、`scale`、`sliding_window_size`、`program_config`、`compute_kernel_config`、`attention_sink`

最小调用形态如下：

```python
out = ttnn.transformer.scaled_dot_product_attention(
    input_tensor_q=q,
    input_tensor_k=k,
    input_tensor_v=v,
    is_causal=True,
    scale=head_dim ** -0.5,
    program_config=program_config,
    compute_kernel_config=compute_kernel_config,
)
```

这条路径的核心特点：

- 实现是 FlashAttention 风格，而不是“matmul + softmax + matmul”三段式拼接。
- host 侧按 `batch -> q_heads -> Q chunks` 三个维度做并行划分。
- kernel 侧在 L1 中维护 online softmax 的中间状态，避免把中间大矩阵落回 DRAM。

### 3.2 Chunked prefill：`chunked_scaled_dot_product_attention`

这条路径针对长 prompt、paged KV cache、trace replay、prefix caching。

- `Q`：`[b, nqh, chunk_s, dh]`
- `K/V`：`[max_blocks, nkv, block_s, dh]`
- `page_table_tensor`：`[b, num_pages]`
- 输出：当前 Q chunk 对应的 attention 输出

关键点：

- 一次只处理一个 Q chunk。
- `K/V` 不是连续序列，而是 paged cache。
- 支持两种 offset 传法：
  - `chunk_start_idx`：host 侧标量，适合 Python 每次 dispatch 一个 chunk。
  - `chunk_start_idx_tensor`：设备端 `int32[1]`，运行时读取，适合 trace replay 或前缀长度变化的场景。

典型调用形态：

```python
out = ttnn.transformer.chunked_scaled_dot_product_attention(
    input_tensor_q=q_chunk,
    input_tensor_k=k_cache,
    input_tensor_v=v_cache,
    page_table_tensor=page_table,
    chunk_start_idx=chunk_start_idx,
    scale=head_dim ** -0.5,
    program_config=program_config,
)
```

约束上要特别注意：

- 必须提供 `page_table_tensor`。
- `chunk_start_idx` 必须同时是 `q_chunk_size` 和 `k_chunk_size` 的整数倍。
- `page_table_tensor` 必须在 device 上，且是 `ROW_MAJOR + INT32`。

### 3.3 Decode：`scaled_dot_product_attention_decode`

decode 不是标准 prefill 的“小改版”，而是一条独立的 FlashDecode 路径。

- `Q`：`[1, b, nh, dh]`
- `K/V cache`：`[b, nkv, s, dh]`
- 输出：逻辑上可看作 `[1, b, nh, dh]`
- 常用参数：`cur_pos` 或 `cur_pos_tensor`、`scale`、`sliding_window_size`、`attn_mask`、`program_config`

典型调用形态：

```python
out = ttnn.transformer.scaled_dot_product_attention_decode(
    q,
    k_cache,
    v_cache,
    cur_pos_tensor=current_pos,
    scale=head_dim ** -0.5,
    program_config=decode_program_config,
    compute_kernel_config=decode_kernel_config,
)
```

这条路径的重点不是“并行多个 Q”，而是“并行同一个 token 对不同 KV 片段的计算，再把局部结果归并起来”。

### 3.4 Paged decode：`paged_scaled_dot_product_attention_decode`

这条路径是 decode 对 paged KV cache 的支持。

- `Q`：`[1, b, nh, dh]`
- paged `K/V`：通常是 `[num_pages, nkv, block_size, dh]`
- `page_table_tensor`：页表
- `cur_pos_tensor`：当前每个 batch/user 的位置

典型调用形态：

```python
out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q,
    k_cache,
    v_cache,
    page_table_tensor=page_table,
    cur_pos_tensor=current_pos,
    scale=head_dim ** -0.5,
    program_config=decode_program_config,
)
```

和 prefill 的 paged/chunked 相比，这里更像“KV cache 查询 + 单 token decode”。

### 3.5 Windowed：`windowed_scaled_dot_product_attention`

windowed 路径不接收显式 `attn_mask`，而是接收 `cu_window_seqlens`，由 kernel 在 device 上构造窗口化的 block-diagonal mask。

- `Q`：`[b, nqh, s, dh]`
- `K/V`：`[b, nkv, s, dh]`
- `cu_window_seqlens`：`[window_count + 1]`

典型调用形态：

```python
out = ttnn.transformer.windowed_scaled_dot_product_attention(
    q,
    k,
    v,
    cu_window_seqlens,
    scale=head_dim ** -0.5,
    program_config=program_config,
)
```

它很适合描述视觉模型里的局部窗口注意力，例如 Qwen2.5-VL 这类 block-diagonal attention 场景。

## 4. `SDPAProgramConfig` 代表什么

`SDPAProgramConfig` 定义在 `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp`，字段如下：

| 字段 | 作用 |
| --- | --- |
| `compute_with_storage_grid_size` | 参与执行的 core grid 大小 |
| `sub_core_grids` | 可选的子 core 集合 |
| `q_chunk_size` | prefill 中 Q 序列分块大小 |
| `k_chunk_size` | prefill / decode 中 K/V 序列分块大小 |
| `exp_approx_mode` | softmax 中 exp 近似模式 |
| `max_cores_per_head_batch` | decode 中每个 head-batch 最多可用多少核 |

有两个实现层面的细节值得单独记住：

1. Python 层通过 `transformer_nanobind.cpp` 暴露的构造参数目前只有前 5 个字段，`max_cores_per_head_batch` 在 C++ 结构体里存在，也会被 decode program factory 消费，但当前没有显式绑到 Python 构造参数和属性上。
2. prefill 真正常用的是 `q_chunk_size` 和 `k_chunk_size`；decode 更关键的是 `k_chunk_size`，因为它直接决定 KV cache 如何按 chunk 被不同核心分担。

## 5. 几条主实现路径

### 5.1 标准 prefill：FlashAttention 风格

prefill 主线可以概括成下面几件事：

1. Reader 把 Q/K/V chunk 从 DRAM 拉到 L1 circular buffer。
2. Compute 计算 `QK^T`、在线 softmax、`P @ V`，并在 L1 中维护 `(max, sum, out_acc)`。
3. Writer 只在最后把归一化后的输出写回。

这里的核心不是“把完整 attention score 写出来”，而是：

- 用 online softmax 逐块累计，避免中间大矩阵落盘。
- 用双缓冲把 K/V 预取和 compute 重叠。
- 把中间统计量都留在 L1。

### 5.2 Prefill 的多核并行

`sdpa_program_factory.cpp` 中的核心并行划分是：

```text
batch_parallel_factor = min(B, num_cores)
nh_parallel_factor    = min(num_cores / batch_parallel_factor, NQH)
q_parallel_factor     = min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks)
```

也就是说，prefill 的并行优先级是：

1. 先拆 batch。
2. 再拆 Q heads。
3. 最后拆 Q 序列 chunks。

这能解释为什么很多 profile 脚本会把“核心数变化”和“Q chunk 划分”绑在一起分析。

### 5.3 Causal prefill 的负载均衡

causal attention 有天然的三角形工作量不均衡问题：越靠后的 Q chunk，需要看的 K/V 越多。

为此实现里有 `BALANCED_Q_PARALLEL` 路径，会把低位和高位 chunk 配对给同一个核，例如：

```text
Core 0: Q0 + Q(n-1)
Core 1: Q1 + Q(n-2)
Core 2: Q2 + Q(n-3)
```

这样做的目的不是改变数学语义，而是让各核负载更均匀。

### 5.4 Non-causal prefill 的 KV chain / multicast

non-causal prefill 时，如果一个 head 的多个 Q chunk 被分到不同核心，这些核心往往需要读取完全相同的 K/V。

实现没有让每个核心都去 DRAM 重读，而是尝试：

- 建立 `KV chain forwarding`
- 满足条件时升级为 `multicast`

这部分逻辑基本都集中在 `sdpa_program_factory.cpp`。这也是 prefill 路径里最值得读的 host 侧优化代码之一。

### 5.5 Decode：FlashDecode + 树形归约

decode 的并行逻辑与 prefill 完全不同。

它的基本思路是：

1. 把同一个 batch/head 的 KV cache 按序列维切成多个片段。
2. 多个核心分别对自己的局部 KV 片段计算局部 attention。
3. 每个核心得到局部 `(O, M, L)` 状态：
   - `O`：局部输出
   - `M`：局部 max
   - `L`：局部 sum
4. 再通过树形归约把局部状态按 online softmax 规则合并。

`sdpa_decode_program_factory.cpp` 会计算：

- `num_cores_per_head`
- `num_tree_reduction_rounds`
- 每个核心在树里的父子关系和归约轮次

这条路径的关键价值是：**即使 query 只有一个 token，仍然可以通过在 KV 序列维上切分来充分利用多核。**

### 5.6 Windowed：mask 不再由上层显式提供

windowed 路径不是在 host 侧先拼好大 mask 再传进 kernel，而是在 `reader_windowed.cpp` 中读取 `cu_window_seqlens`，动态构造 block-diagonal window mask。

这对窗口化视觉注意力很重要，因为：

- 上层接口更轻，只传窗口边界。
- 不必显式存储完整 `s x s` mask。
- kernel 能更直接地利用窗口结构。

### 5.7 Ring distributed / joint / MLA

这些都属于 SDPA 家族里的高级分支：

- `ring_distributed_scaled_dot_product_attention`
  - 面向多设备 causal prefill。
  - 每个设备处理两个对称的 Q chunk。
  - 支持与 paged KV、prefix caching 组合。
- `joint_scaled_dot_product_attention`
  - 把两段 Q/K/V 沿序列维拼接成一次 non-causal attention。
  - 当前实现要求 `joint_strategy="rear"`。
- `ring_joint_scaled_dot_product_attention`
  - joint attention 的多设备版本。
- MLA 相关 API
  - 是共享 SDPA 框架的专门变体，不应和标准 MHA/MQA/GQA 路径混为一谈。

## 6. 常见约束与容易踩坑的地方

### 6.1 Prefill 路径

- `Q/K/V` 必须在 device 上。
- `Q/K/V` 必须是 `TILE_LAYOUT`。
- prefill 主线不接受 sharded 输入，要求 DRAM/L1 interleaved。
- batch 维、head 维、head_dim 维不支持 padding；通常只有 sequence 维允许 padding。
- `is_causal` 和显式 `attn_mask` 不能同时使用。
- `attention_sink` 若提供，必须在 device 上、`TILE_LAYOUT`、DRAM buffer，形状是 `[1, nqh, 1, 1]`。

### 6.2 Chunked prefill

- 必须带 `page_table_tensor`。
- `page_table_tensor` 必须是 `ROW_MAJOR + INT32`。
- `chunk_start_idx` 和 `chunk_start_idx_tensor` 只能二选一。
- `chunk_start_idx_tensor` 必须是 device 上的 `int32[1]`。
- `chunk_start_idx` 必须同时对齐 `q_chunk_size` 与 `k_chunk_size`。

### 6.3 Decode / paged decode

- decode 的 `Q` 形状必须是 `[1, b, nh, dh]`。
- decode 支持 Q 采用 `HEIGHT_SHARDED`，也支持 DRAM interleaved；输出也可以是 `HEIGHT_SHARDED` 或 DRAM。
- 非 causal decode 若使用显式 `attn_mask`，必须提供正的 `k_chunk_size`。
- paged causal decode 必须提供 `cur_pos_tensor`。
- paged decode 的 `page_table_tensor` 必须是 `ROW_MAJOR`：
  - 非 sharded 时通常要求 `INT32`
  - sharded 时支持 `UINT16`

### 6.4 一个值得记住的实现细节

`SDPAProgramConfig.max_cores_per_head_batch` 在 C++ 里真实存在，也被 FlashDecode 读取，但当前 Python 构造接口并没有显式暴露它。写调优文档时最好把这个点单独标出来，避免读者以为 Python 层能直接完整控制所有 decode 调参项。

## 7. 代码入口怎么读最有效

如果你要深入源码，推荐按下面顺序阅读：

1. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`
2. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`
3. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp`
4. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
5. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
6. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp`
7. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
8. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
9. `tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/device/kernels/dataflow/reader_windowed.cpp`
10. `tt-metal/tech_reports/FlashAttention/FlashAttention.md`
11. `tt-metal/tech_reports/FlashAttention/FlashDecode.md`

如果只想先理解“真实模型里怎么接这个算子”，最适合从下面两个 Python 文件切入：

- `tt-metal/models/common/modules/attention/attention_1d.py`
- `tt-metal/models/tt_transformers/tt/attention.py`

前者适合作为“统一封装层”，后者更接近生产模型的用法。

## 8. 测试、样例和资料入口

### 8.1 仓库根目录下的讲解型脚本

这些文件更适合做“原理图谱”和“教学材料”：

- `test/test_sdpa.py`
  - 最小可运行的标准 prefill 示例。
- `test/test_sdpa_single_core.py`
  - 单核 prefill 与 decode。
- `test/test_sdpa_multi_core.py`
  - 多核 causal、balanced Q、non-causal KV chain、FlashDecode tree reduction。
- `test/test_sdpa_two_chips.py`
  - 双芯片 / ring distributed 场景。
- `test/profile_sdpa.py`
  - 基础 benchmark。
- `test/profile_sdpa_single_core.py`
  - 单核 profile。
- `test/profile_sdpa_multi_core.py`
  - 多核 profile。
- `test/profile_sdpa_two_chips.py`
  - 双芯片 profile。

### 8.2 `tt-metal` 自带单测与 nightly

这些文件更适合做“能力矩阵”和“边界条件”参考：

- `tt-metal/tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py`
- `tt-metal/tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py`
- `tt-metal/tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py`
- `tt-metal/tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode.py`
- `tt-metal/tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py`
- `tt-metal/tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_ring_distributed.py`
- `tt-metal/tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode_sink.py`
- `tt-metal/tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_joint.py`

这些测试覆盖的重点包括：

- causal / non-causal prefill
- paged decode
- chunked prefill
- sliding window
- attention sink
- GQA / MQA
- sub-core grids
- long context
- tree reduction regression

### 8.3 现有文档和缺口

当前仓库里已经有几份很重要的资料：

- `docs/SDPA_单核多核多芯片实现.md`
- `tt-metal/tech_reports/FlashAttention/FlashAttention.md`
- `tt-metal/tech_reports/FlashAttention/FlashDecode.md`

另外，`tt-metal/docs/source/ttnn/ttnn/api.rst` 已经把 SDPA 家族 API 列入 autosummary，但 `tt-metal/tests/ttnn/docs_examples/examples_mapping.py` 里仍然能看到标准 SDPA / decode 示例还没有被正式纳入 docs examples，这意味着“API 已公开，但官方教程示例仍偏少”。

## 9. 一句话总结

如果用一句话概括 `tt-metal` 里的 SDPA 家族：

- prefill 主线是 **FlashAttention 风格的分块 attention 实现**；
- decode 主线是 **FlashDecode + 树形归约**；
- paged、windowed、ring、joint、MLA 则是在这两条主线之上的场景化扩展。

对源码阅读来说，最重要的不是先钻 kernel 指令，而是先把下面三件事搞清楚：

1. 这次调用属于 prefill 还是 decode。
2. 张量是连续 cache 还是 paged cache。
3. 并行是沿 `batch/head/Q chunks` 展开，还是沿 `KV sequence split + reduction` 展开。

只要这三件事清楚，后面的 program factory、kernel 参数和测试矩阵就会顺很多。

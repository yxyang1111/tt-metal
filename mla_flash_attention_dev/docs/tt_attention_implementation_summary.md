# TT Attention 能力现状总结

本文总结 `tt-metal` 仓内当前已经落地的 Attention 相关能力，重点聚焦 `FlashAttention` 风格实现与 `MLA`（Multi-Head Latent Attention）支持现状。

## 结论先行

- TT 在 `ttnn` 主线里已经实现了一整套 Transformer Attention 能力，不仅有基础的 `MHA/GQA` 相关组件，也有完整的 `SDPA prefill`、`decode`、`paged/chunked prefill`、`paged decode`、`windowed SDPA`、`joint SDPA`、`ring_joint SDPA`、`ring_distributed SDPA`。
- 仓内对 “FlashAttention” 的含义，不是直接集成外部 CUDA FlashAttention 库，而是 TT 自己在设备侧实现的 flash-style、分块、在线 softmax 的 SDPA/Decode 内核。`ttnn.transformer.scaled_dot_product_attention` 的绑定文档直接写了 `The implementation is FlashAttention-2.`，decode 文档写的是 `The implementation is Flash-Decode`。
- MLA 已经不是纯规划。`ttnn` 中已经暴露 `flash_mla_prefill`、`chunked_flash_mla_prefill`、`flash_multi_latent_attention_decode`、`paged_flash_multi_latent_attention_decode`；`models/demos/deepseek_v3/tt/mla/mla1d.py` 也已经把 prefill 和 decode 接到了模型主路径里。
- `mla_flash_attention_dev/` 当前主要是本地调研、路线图和实验笔记，并不是 MLA 的唯一实现来源。真正已落地的主代码在 `ttnn/`、`models/`、`tests/`。

## 本文范围

- 重点覆盖 `ttnn` 推理主线与模型 demo 的 Attention 实现。
- `tt-train` 中也有独立的训练侧 Attention 实现，本文在末尾单独备注，但不把它和 `ttnn` 推理路径混为一谈。

## Attention 能力矩阵

| 类别 | 已实现能力 | 主要 API / 路径 | 说明 |
| --- | --- | --- | --- |
| 基础组件 | `attention_softmax`、`split_query_key_value_and_split_heads`、`concatenate_heads` | `ttnn/cpp/ttnn/operations/transformer/attention_softmax/`、`split_query_key_value_and_split_heads/`、`concatenate_heads/` | 提供非 fused attention 或模型适配所需的基础 building blocks |
| 通用 SDPA prefill | `scaled_dot_product_attention` | `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`、`sdpa.cpp`、`sdpa_nanobind.cpp` | 主线 fused SDPA，支持 `is_causal`、`attn_mask`、`sliding_window_size`、`attention_sink`、`SDPAProgramConfig` |
| Chunked / paged prefill | `chunked_scaled_dot_product_attention` | 同上 | 面向 paged KV cache 与 prefix/chunked prefill，支持 `page_table_tensor` + `chunk_start_idx` 或设备侧 `chunk_start_idx_tensor` |
| Decode | `scaled_dot_product_attention_decode`、`paged_scaled_dot_product_attention_decode` | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` | 单 token decode 专用路径，支持 `cur_pos`/`cur_pos_tensor`、causal、paged KV、sliding window |
| Windowed Attention | `windowed_scaled_dot_product_attention` | `ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/` | 局部窗口/block-diagonal attention，不走显式 attn mask 输入 |
| Joint Attention | `joint_scaled_dot_product_attention` | `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`、`sdpa.cpp` | 适合双流输入，如 spatial + prompt |
| Ring Joint Attention | `ring_joint_scaled_dot_product_attention` | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_*` | 多卡 sequence-parallel / CCL 路径，返回 attention 输出和统计量 |
| Ring Distributed Attention | `ring_distributed_scaled_dot_product_attention` | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_distributed_*` | 支持 ring 分布式 attention，也支持和 `page_table` + `chunk_start_idx` 组合 |
| MLA Prefill | `flash_mla_prefill`、`chunked_flash_mla_prefill` | `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp` | MLA 语义下的 prefill，底层仍复用 SDPA 主管线 |
| MLA Decode | `flash_multi_latent_attention_decode`、`paged_flash_multi_latent_attention_decode` | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp` | MLA 语义下的 decode，支持 paged KV |
| MLA 附加算子 | `ttnn.experimental.deepseek.mla.matmul_wo` | `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/` | DeepSeek MLA 输出投影专用算子，不等同于 attention 核心本体 |

## FlashAttention 相关结论

### 1. TT 已经实现了 FlashAttention 风格的主 Attention 内核

最直接的证据来自 Python 绑定文档：

- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`
  - `scaled_dot_product_attention` 文档串写明：`The implementation is FlashAttention-2.`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp`
  - `scaled_dot_product_attention_decode` 文档串写明：`The implementation is Flash-Decode`

### 2. 这不是“把外部 CUDA FlashAttention 库直接搬进来”

更准确的说法是：

- TT 在自己的设备/编译/数据流体系里实现了 FlashAttention 同类算法族的 SDPA。
- 其核心特征是：
  - 使用 `SDPAProgramConfig` 指定 `q_chunk_size`、`k_chunk_size`、网格与近似模式。
  - 在设备侧按块处理 Q/K/V，并维护在线 softmax 所需的统计量。
  - prefill、decode、paged KV、chunked prefix、ring 分布式等能力都在这套原生实现上扩展。

### 3. 代码层面的具体依据

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
  - 编译期参数直接体现了分块 attention 的核心结构：`Sq_chunk_t`、`Sk_chunk_t`、`q_num_chunks`、`k_num_chunks`、`is_chunked`、`sliding_window_size`、`use_attention_sink` 等。
  - 既有标准 SDPA 路径，也有 `streaming` 版本。
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
  - decode 路径显式维护 `m/l` 这类在线 softmax 统计量，并支持动态 chunk、tree reduction、block padding。

### 4. 现有 Flash 风格能力覆盖面

- 标准 prefill SDPA
- chunked prefill
- paged KV cache
- decode / paged decode
- sliding window attention
- attention sink
- ring distributed / ring joint 等并行变体

因此，如果问题是“TT 现在有没有 FlashAttention”，答案是：

- 有，但更准确的表述应是：`TT 已经有自己原生实现的 FlashAttention 风格 SDPA / Flash-Decode 管线。`

## MLA 相关结论

### 1. MLA 已经进入 TTNN 主算子层

MLA 相关 API 已经在 `ttnn` 对外暴露：

- `ttnn.transformer.flash_mla_prefill`
- `ttnn.transformer.chunked_flash_mla_prefill`
- `ttnn.transformer.flash_multi_latent_attention_decode`
- `ttnn.transformer.paged_flash_multi_latent_attention_decode`

对应主路径：

- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp`

### 2. MLA 不是单独新写的一整套 attention 引擎，而是复用现有 SDPA/Decode 主栈

从实现看：

- `flash_mla_prefill` 最终调用 `ttnn::prim::sdpa(..., use_mla=true, head_dim_v=...)`
- `flash_multi_latent_attention_decode` / `paged_flash_multi_latent_attention_decode` 最终调用 `ttnn::prim::sdpa_decode(..., use_mla=true, head_dim_v=...)`

这说明：

- MLA 与普通 SDPA 在 TT 里是同一产品族。
- 区别主要体现在张量语义、`head_dim_v`、是否显式提供 `V`、以及 reader/kernel 对 MLA 布局的处理方式上。

### 3. MLA 已经接到模型 demo 主线

`models/demos/deepseek_v3/tt/mla/mla1d.py` 已明确接入：

- prefill 调用 `ttnn.transformer.flash_mla_prefill`
- decode 调用 `ttnn.transformer.paged_flash_multi_latent_attention_decode`

这意味着 MLA 不是只停留在算子单测，而是已经进入 DeepSeek V3 的实际模型路径。

### 4. MLA 的测试覆盖已经存在

核心测试包括：

- `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill.py`
- `tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py`
- `tests/ttnn/nightly/unit_tests/operations/sdpa/test_mla_prefill.py`
- `tests/ttnn/nightly/unit_tests/operations/sdpa/test_mla_decode.py`
- `tests/ttnn/nightly/unit_tests/operations/sdpa/test_mla_prefill_stress.py`
- `tests/ttnn/nightly/unit_tests/operations/sdpa/test_mla_decode_stress.py`
- `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_flash_mla_deepseek.py`

此外还有更偏实验/微算子方向的路径，例如：

- `models/demos/deepseek_v3_b1/tests/unit_tests/test_flash_mla.py`
- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/`
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py`

这些说明 MLA 已经有主线实现，也有实验性深化方向。

### 5. MLA 相关的额外专用算子已经存在

除了 attention 本体，仓里还有：

- `ttnn.experimental.deepseek.mla.matmul_wo`

对应路径：

- `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/`

这更像是 DeepSeek MLA 流水中的输出投影优化，而不是新的 Attention API。

### 6. MLA 当前边界

从当前代码看，比较稳妥的判断是：

- `prefill` 和 `decode` 的 MLA 主能力已经落地。
- `DeepSeek V3` 已经接入主 demo。
- 部分更复杂的多芯片/环形分布式 MLA 仍有实验痕迹，但不宜表述成“所有 distributed 变体都已完全产品化”。
- `ring_distributed_scaled_dot_product_attention` 当前并不是 MLA 版本；在 `ring_distributed_sdpa_program_factory.cpp` 中，reader compile-time args 里 `use_mla` 仍为 `0`。

因此，如果问题是“TT 现在有没有 MLA”，答案是：

- 有，而且已经到 `TTNN API + 模型接入 + 单测/夜测` 这一层级。

## 模型侧接入现状

下面这些路径能说明 Attention 能力已经真正被模型使用，而不只是算子孤立存在：

- `models/tt_transformers/tt/attention.py`
  - 通用 transformer attention 主路径。
  - decode 时使用 `scaled_dot_product_attention_decode` 或 `paged_scaled_dot_product_attention_decode`
  - prefill 时使用 `scaled_dot_product_attention` 或 `chunked_scaled_dot_product_attention`
- `models/demos/deepseek_v3/tt/mla/mla1d.py`
  - 直接接入 `flash_mla_prefill` 与 `paged_flash_multi_latent_attention_decode`
- `models/demos/llama3_70b_galaxy/tt/llama_attention.py`
  - 多芯片 Llama attention 主路径
- `models/tt_dit/models/transformers/attention_sd35.py`
  - 使用 `joint_scaled_dot_product_attention` / `ring_joint_scaled_dot_product_attention`
- `models/tt_dit/models/transformers/wan2_2/attention_wan.py`
  - 同样使用 joint/ring_joint attention
- `models/demos/qwen25_vl/tt/vision_attention.py`
  - 使用 `windowed_scaled_dot_product_attention`

## `mla_flash_attention_dev/` 的定位

`mla_flash_attention_dev/README.md` 已经明确写了：

- 该目录用于本地开发、笔记和实验
- production-ready code 应该放回正式的 `models/` 或 `ttnn/` 路径

因此应把这个目录理解为：

- 调研和规划的工作区
- 不是判断“功能是否已经落地”的唯一依据

换句话说：

- `mla_flash_attention_dev/*.md` 里写到的很多东西是路线图、设计空间和实验计划
- 真正判断“TT 已经实现了什么”，仍然应以 `ttnn/`、`models/`、`tests/` 为准

## `tt-train` 的补充说明

除了 `ttnn` 推理主线，`tt-train` 里还有另一套训练侧 Attention 体系，例如：

- `tt-train/sources/ttml/ops/scaled_dot_product_attention.hpp`
  - 提供 fused `scaled_dot_product_attention`
  - 还有 composite fallback 版本
- `tt-train/sources/ttml/ops/distributed/ring_attention_sdpa.hpp`
  - 提供训练侧 context parallel 的 ring attention SDPA
- `tt-train/sources/ttml/modules/grouped_query_attention.hpp`
  - 提供 GQA 模块

这说明从更广义的 “TT 已经实现了哪些 Attention” 来看：

- 不只是 `ttnn` 推理侧有 attention
- `tt-train` 训练侧也已有自己的 fused attention、GQA 和 ring attention 支持

但需要注意：

- `tt-train` 与 `ttnn` 是不同代码栈
- 不应把两者的 API 和成熟度直接等同

## 最终判断

可以把当前状态概括成三句话：

1. `TT 已经有完整的 Attention 主栈`，包括 prefill、decode、paged、chunked、windowed、joint、ring distributed 等多种变体。
2. `TT 已经有 FlashAttention 风格实现`，但它是 TT 自己原生实现的 SDPA/Flash-Decode，而不是简单依赖外部 CUDA FlashAttention 库。
3. `TT 已经有 MLA`，而且不仅是实验笔记，已经达到 `TTNN API + DeepSeek V3 接入 + 测试覆盖` 的落地程度；只是更广泛的分布式 MLA 形态仍有一部分处于继续演进中。

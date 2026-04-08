# Part I 实验参数总表

说明：当前文档整理的是 `run_part1_benchmarks.py` 这套 Part I 多维 sweep 在当前结果中的实际配置。
说明：记号约定为 `B=batch`，`H=num_heads`，`H_kv=num_kv_heads`，`Q=q_seq_len`，`S=kv_seq_len`。

## 1. 当前运行口径

| 项目 | 当前值 | 说明 |
|---|---|---|
| comparison mode | `strict_four_way` | 当前实现的比较模式 |
| sweep preset | `manual` | 主跑使用的 preset；`manual` 表示直接读取 CLI 轴参数 |
| probe preset | `manual` | capability probe 使用的 preset；`manual` 表示直接读取 CLI 轴参数 |
| device_id | `0` | TT 设备编号 |
| arch | `wormhole_b0` | 当前结果中的设备架构 |
| compute grid | `8 x 7` | 当前结果中的 `compute_with_storage_grid_size` |
| total configs | `1` | 当前主跑实际生成的 config 数量 |
| default config | `b1_h32_hkv1_dv512_ro64_blk64_kc128` | 默认主表固定使用的 config |
| default config label | `B=1, H=32, H_kv=1, value_dim=512, rope_dim=64, block=64, k_chunk=128` | 默认主表的人类可读说明 |
| decode seq_len sweep | `256, 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k` | decode 主 sweep |
| prefill seq_len sweep | `1k, 4k` | prefill 控制组 |
| torch input dtype | `torch.bfloat16` | host 随机输入生成 dtype |
| warmup_device / iters_device | `2 / 10` | TT baseline 预热 / 正式测量次数 |
| warmup_reference / iters_reference | `1 / 10` | torch reference 预热 / 正式测量次数 |
| outlier filter | `modified z-score > 5.0 and latency > median x 1.1` | 当前主表使用的慢尾异常点过滤规则 |
| filtered minimum kept samples | `5` | 若过滤后样本过少则回退到原始样本 |

## 2. Sweep 轴

| 轴 | 当前值 | 说明 |
|---|---|---|
| `B` | `1` | 主跑 batch sweep |
| `H` | `32` | 主跑 num_heads sweep |
| `H_kv` | `1` | 主跑 num_kv_heads sweep |
| `value_dim` | `512` | 严格四方法公共 value dim sweep |
| `rope_dim` | `64` | rope dim sweep |
| capability probe | `false` | probe 轴与结果详见 `raw/capability_probe_results.json` |

## 3. 默认 config 显式字段

| 字段 | 当前值 | 说明 |
|---|---:|---|
| `batch` | `1` | 默认主表 batch |
| `num_heads` | `32` | 默认主表 attention heads |
| `num_kv_heads` | `1` | 默认主表 kv heads |
| `std_head_dim` | `512` | 标准 attention 的 `d_q=d_k=d_v` |
| `mla_head_dim_v` | `512` | TT 主线 MLA 的 `d_v` |
| `mla_d_rope` | `64` | TT 主线 MLA rope 维度 |
| `mla_head_dim_qk` | `576` | `mla_head_dim_v + mla_d_rope` |
| `deepseek_qk_nope_head_dim` | `128` | DeepSeek 逻辑 QK 的 non-rope 部分 |
| `deepseek_qk_rope_head_dim` | `64` | DeepSeek 逻辑 QK 的 rope 部分 |
| `deepseek_qk_head_dim` | `192` | DeepSeek 缩放维度 |
| `deepseek_kv_lora_rank` | `512` | DeepSeek 输出 / value 维度 |
| `deepseek_kvpe_dim` | `576` | DeepSeek Q/KV 存储宽度 |
| `block_size` | `64` | paged attention block size |
| `k_chunk_size` | `128` | decode k chunk size |
| `max_cores_per_head_batch` | `4` | TT 主线 decode program config |
| `deepseek_num_q_heads_per_core` | `8` | DeepSeek Q shard 粒度 |

## 4. 默认 config 下的张量形状

| 张量 | 形状 | 说明 |
|---|---|---|
| `q_std` | `[B, H, Q, 512]` | Reference / Flash Attention 用 Q |
| `k_std`, `v_std` | `[B, H_kv, S, 512]` | 标准 attention 的 K/V |
| `q_mla` | `[B, H, Q, 576]` | TT-MLA 用 Q |
| `k_mla` | `[B, H_kv, S, 576]` | TT-MLA 用 K，其中前 `512` 维隐含 V |
| `q_deepseek` | `[B, H, Q, 576]` | DeepSeek 用 Q 存储宽度 |
| `k_deepseek` | `[B, H_kv, S, 576]` | DeepSeek KV cache 宽度 |
| decode `Q` | `1` | 只解当前 token |
| decode `S` | `seq_len` | 历史 cache 长度 |
| prefill `Q=S` | `seq_len` | 全序列因果 prefill |

## 5. 四条 baseline 的默认设置

| baseline | mode | Q 输入 | K / KV 输入 | V 输入 | scale | 实际调用 |
|---|---|---|---|---|---|---|
| `reference_attention` | decode + prefill | `q_std` | `k_std` | `v_std` | `512^-0.5` | torch reference SDPA |
| `flash_attention` | decode + prefill | `q_std` | `k_std` | `v_std` | `512^-0.5` | `scaled_dot_product_attention` / `paged_scaled_dot_product_attention_decode` |
| `flash_mla` | decode + prefill | `q_mla` | `k_mla` | `None`，`V` 取前 `512` 维 | `576^-0.5` | `flash_mla_prefill` / `paged_flash_multi_latent_attention_decode` |
| `deepseek_flash_mla` | decode only | `q_deepseek` | `k_deepseek` | `None`，输出宽度 `512` | `192^-0.5` | `flash_multi_latent_attention_decode` |

## 6. Program / Kernel / Guardrails

| 项目 | 当前值 | 说明 |
|---|---|---|
| decode q_num_cores | `32` | TT 主线 decode 默认 `q_num_cores=min(B*H, grid_x*grid_y)` |
| DeepSeek num_q_shards | `4` | `num_heads / deepseek_num_q_heads_per_core` |
| DeepSeek required q cores | `4` | 当前 builder 使用 `batch * num_q_shards` 个活跃核 |
| DeepSeek grid requirement | `>= 8 x 7` | 当前 Wormhole 路径要求 |
| DeepSeek k_chunk_size | `128` | 直接从 config 读取 |
| DeepSeek KV ND shard shape | `[1, H_kv, 128, 576]` | ND-sharded DRAM cache |
| measurement scope | `device_core_only_after_one_time_adapter_conversion` | benchmark 只计后端 device op，排除一次性张量适配开销 |

## 7. CLI / 结果文件

| 参数 | 当前值 | 说明 |
|---|---|---|
| `--cases` | `decode_256, decode_512, decode_1k, decode_2k, decode_4k, decode_8k, decode_16k, decode_32k, decode_64k, decode_128k, prefill_1k, prefill_4k` | 本次实际运行的 benchmark cases |
| `--detail-cases` | `decode_1k, decode_4k, decode_8k, decode_16k, decode_32k` | detailed FlashMLA profiling 选择的 case |
| `--run-capability-probe` | `false` | 是否先跑公共参数空间 probe |
| `--run-flashmla-detailed` | `false` | 是否现场重跑 detailed tracy profile |
| `--reuse-existing-flashmla-detailed` | `true` | 当前是否复用现有稳定 detailed 结果 |
| `--skip-render` | `false` | 当前结果已完成 render |

## 8. 结果文件

- 主报告：`report.md`
- 参数总表：`tables/experiment_parameters.md`
- Decode 默认主表：`tables/decode_four_methods.md`
- Prefill 默认主表：`tables/prefill_control.md`
- Capability probe：`tables/capability_probe.md`
- 多维结果目录：`multidim`
- 过滤后 JSON：`raw/part1_four_method_results_filtered.json`
- 过滤后 CSV：`raw/part1_four_method_results_filtered.csv`
- Detailed profile 来源：`/rshome/yuxin.yang/dev/tt-metal/mla_flash_attention_dev/experiments/profile_outputs/flash_mla_wh_detailed`

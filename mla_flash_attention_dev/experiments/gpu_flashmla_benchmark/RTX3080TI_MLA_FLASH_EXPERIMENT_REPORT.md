# RTX 3080 Ti MLA Flash / FlashInfer 实验报告

## 1. 目标

本实验目标是在本机 NVIDIA GeForce RTX 3080 Ti 上，使用与 L40S 实验相同的 **FlashInfer MLA** 路径，复现 DeepSeek-V3 MLA workload 的 decode 和 prefill benchmark。

实验保持与 `L40S_MLA_FLASH_EXPERIMENT_REPORT.md` 相同的 operator shape、序列长度 sweep 和主要 benchmark 参数，用来观察 3080 Ti 在本地 12GB 显存约束下的可运行范围和性能。

## 2. Workload

使用 DeepSeek-V3 MLA operator shape：

| 参数 | 值 |
|---|---:|
| Batch size, `B` | 6 |
| Query heads, `H_q` | 32 |
| KV heads, `H_kv` | 1 |
| Latent dimension, `d_c` / `kv_lora_rank` | 512 |
| RoPE dimension, `d_r` / `qk_rope_head_dim` | 64 |
| QK head dim, `d_qk=d_c+d_r` | 576 |
| Page / block size | 64 |
| Element format | BF16 |
| Decode | `Q_len=1`, `L={256,512,1K,2K,4K,8K,16K,32K}` |
| Prefill | `Q_len=L`, same `L` sweep |

FlashInfer 输入拆分为 `q_nope [total_q,Hq,512]`、`q_pe [total_q,Hq,64]`、`ckv_cache [pages,64,512]`、`kpe_cache [pages,64,64]`。

## 3. 硬件与软件环境

| 项目 | RTX 3080 Ti 实验环境 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti |
| Compute capability | sm86 |
| 可见显存 | 11.66 GiB |
| SM 数 | 80 |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| NVIDIA driver | 560.28.03 |
| FlashInfer | 0.6.11.post3 |
| FlashInfer backend | `auto` |

本次实验使用独立虚拟环境 `/rshome/yuxin.yang/.cache/flashinfer-3080ti-venv`，避免污染现有 conda 环境。FlashInfer import 时有如下 warning，但不影响本次 benchmark 运行：

```text
Failed to JIT torch c dlpack extension, EnvTensorAllocator will not be enabled.
```

### 硬件参数对比

下表列出 RTX 3080 Ti、L40S 和 TT WH 单芯片与 MLA workload 关系最直接的硬件参数。NVIDIA GPU 的片上存储列为 SM shared memory 上限和 L2 cache；TT WH 列为 Tensix core local SRAM。两者不完全等价，但都决定了 tile、KV cache 复用和中间状态驻留能力。

| 硬件 | 架构 / 单元数 | 峰值算力 | 外部内存带宽 | 片上存储 |
|---|---:|---:|---:|---:|
| NVIDIA RTX 3080 Ti | Ampere sm86, 80 SM | 约 136 TFLOPS FP16 Tensor Core；稀疏约 273 TFLOPS | 912 GB/s | shared memory 约 `80 * 100 KiB = 7.8 MiB`；L2 cache 6 MiB |
| NVIDIA L40S | Ada Lovelace sm89, 142 SM | 约 362 TFLOPS BF16/FP16 Tensor Core | 864 GB/s | shared memory 约 `142 * 100 KiB = 13.9 MiB`；L2 cache 96 MiB |
| TT WH 单芯片 | 64 Tensix cores | 65.5 TFLOPS | 288 GB/s | local SRAM `64 * 1.5 MiB = 96 MiB` |

从纸面参数看，RTX 3080 Ti 的 GDDR6X 带宽略高于 L40S，但 Tensor Core 峰值算力和片上 cache/local storage 都明显更小。L40S 的 96 MiB L2 与 WH 单芯片 96 MiB local SRAM 容量相近，但访问语义不同：L40S L2 是硬件 cache，WH local SRAM 是显式管理的片上存储，更接近 kernel tile 调度可直接利用的 scratchpad。

### Wormhole 对标 NVIDIA 卡型

如果只按算力、外部内存带宽和内存容量三项纸面指标对齐，TT Wormhole n300 最接近的 NVIDIA 卡型是 **NVIDIA A10 24GB**：n300 为 24GB GDDR6、576 GB/s、约 131 TFLOPS FP16；A10 为 24GB GDDR6、600 GB/s、约 125 TFLOPS dense FP16 Tensor Core（稀疏约 250 TFLOPS）。

单芯片 Wormhole n150 更接近 **NVIDIA RTX A2000 12GB** 的内存配置：两者都是 12GB 级别显存和 288 GB/s 外部内存带宽。算力口径上 n150 的 FP16 峰值高于 RTX A2000，更接近 RTX A4000 档位；但按内存容量和带宽对标，RTX A2000 12GB 是更贴近的参照。

## 4. Decode 结果

RTX 3080 Ti 使用 FlashInfer MLA paged decode，`warmup=10`、`iters=50`。

| Seq Len | RTX 3080 Ti FlashInfer MLA (ms) | L40S FlashInfer MLA (ms) | 3080 Ti / L40S | TT WH S-FMLA (ms) | 3080 Ti / WH S-FMLA |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.144 | 0.285 | 0.50x | 0.173 | 0.83x |
| 512 | 0.129 | 0.267 | 0.48x | 0.181 | 0.71x |
| 1K | 0.132 | 0.277 | 0.48x | 0.207 | 0.64x |
| 2K | 0.127 | 0.432 | 0.29x | 0.235 | 0.54x |
| 4K | 0.207 | 0.315 | 0.66x | 0.277 | 0.75x |
| 8K | 0.289 | 0.386 | 0.75x | 0.410 | 0.70x |
| 16K | 0.451 | 0.495 | 0.91x | 0.554 | 0.81x |
| 32K | 0.856 | 0.744 | 1.15x | 0.788 | 1.09x |

### Decode 观察

- RTX 3080 Ti 的 FlashInfer MLA decode 可以完整跑通 256 到 32K。
- 短序列上 3080 Ti 实测比本次 L40S 数据更快；这部分主要受固定开销、kernel 调度和 boost 状态影响，不应直接解读为硬件上限更高。
- 长序列上差距逐渐收敛，32K 时 3080 Ti 为 **0.856 ms**，略慢于 L40S 的 **0.744 ms**，也略慢于 TT WH S-FMLA 的 **0.788 ms**。
- 32K decode 的下界估算有效带宽约 **264.7 GB/s**，低于 L40S 报告中的 **304.9 GB/s**，但已接近 TT WH 单芯片 288 GB/s 的物理带宽量级。

## 5. Prefill 结果

RTX 3080 Ti 使用 FlashInfer MLA paged prefill，`warmup=2`、`prefill_iters=3`。full prefill 可以覆盖 256 到 16K，32K full prefill 因 12GB 显存限制 OOM。

| Seq Len | RTX 3080 Ti FlashInfer MLA Prefill (ms) | L40S FlashInfer MLA Prefill (ms) | 3080 Ti / L40S | TT WH S-FMLA (ms) | 3080 Ti / WH S-FMLA |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.579 | 0.420 | 1.38x | 1.093 | 0.53x |
| 512 | 1.786 | 0.840 | 2.13x | 2.617 | 0.68x |
| 1K | 6.497 | 3.047 | 2.13x | 8.491 | 0.77x |
| 2K | 23.257 | 9.404 | 2.47x | 33.300 | 0.70x |
| 4K | 86.509 | 35.756 | 2.42x | 129.400 | 0.67x |
| 8K | 345.566 | 134.386 | 2.57x | 510.000 | 0.68x |
| 16K | 1343.252 | 470.002 | 2.86x | 2028.000 | 0.66x |
| 32K | full prefill OOM; chunked 5687.232 | chunked 2366.419 | 2.40x | 8050.000 | 0.71x |

### 32K Chunked Prefill

32K full prefill 在 RTX 3080 Ti 上 OOM，报错为尝试额外分配约 6 GiB。使用 chunk size 4K 后可以跑通：

| Chunk | Q len | KV len | Latency (ms) |
|---:|---:|---:|---:|
| 0 | 4K | 4K | 98.598 |
| 1 | 4K | 8K | 275.951 |
| 2 | 4K | 12K | 439.080 |
| 3 | 4K | 16K | 619.016 |
| 4 | 4K | 20K | 793.119 |
| 5 | 4K | 24K | 971.478 |
| 6 | 4K | 28K | 1152.599 |
| 7 | 4K | 32K | 1337.392 |
| **Total** | **32K** | **32K** | **5687.232** |

### Prefill 观察

- RTX 3080 Ti full prefill 可稳定跑到 16K；32K 受 12GB 显存限制 OOM。
- 256 到 16K 范围内，3080 Ti FlashInfer MLA prefill 比 L40S 慢约 1.4x 到 2.9x，符合两者 SM 数、Tensor Core 能力和显存容量差异。
- 尽管慢于 L40S，3080 Ti 仍明显快于 TT WH S-FMLA prefill：16K 为 **1343 ms vs 2028 ms**，32K chunked 为 **5687 ms vs 8050 ms**。
- 32K 上能够通过 chunked prefill 跑通，说明限制主要来自 full prefill 的瞬时显存占用，而不是 FlashInfer MLA 路径完全不可用。

## 6. 总体结论

1. **Decode**：RTX 3080 Ti 上 FlashInfer MLA decode 可完整覆盖 256 到 32K，32K 为 **0.856 ms**。
2. **Prefill**：full prefill 可覆盖 256 到 16K；32K full prefill OOM。
3. **32K chunked prefill**：chunk size 4K 可跑通，总延迟 **5687.232 ms**。
4. **与 L40S 对比**：decode 在短序列上本次 3080 Ti 数据更低，长序列 32K 略慢；prefill 全范围明显慢于 L40S。
5. **与 TT WH S-FMLA 对比**：3080 Ti decode 在 32K 略慢于 S-FMLA，但 prefill 仍快于 S-FMLA，包括 32K chunked prefill。

## 7. 复现实验命令

### 环境

```bash
cd /rshome/yuxin.yang/dev/tt-metal/mla_flash_attention_dev/experiments/gpu_flashmla_benchmark

/rshome/yuxin.yang/anaconda3/envs/vllm/bin/python -m venv --system-site-packages /rshome/yuxin.yang/.cache/flashinfer-3080ti-venv
/rshome/yuxin.yang/.cache/flashinfer-3080ti-venv/bin/python -m pip install -U flashinfer-python flashinfer-cubin
/rshome/yuxin.yang/.cache/flashinfer-3080ti-venv/bin/python -m pip install -U flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
```

### Decode

```bash
/rshome/yuxin.yang/.cache/flashinfer-3080ti-venv/bin/python run_flashinfer_mla_benchmark.py \
  --backend auto \
  --batch 6 \
  --skip-prefill \
  --decode-seq-lens 256 512 1024 2048 4096 8192 16384 32768 \
  --warmup 10 \
  --iters 50 \
  --output outputs/rtx3080ti_b6_flashinfer_mla_decode.json
```

### Prefill

```bash
/rshome/yuxin.yang/.cache/flashinfer-3080ti-venv/bin/python run_flashinfer_mla_benchmark.py \
  --backend auto \
  --batch 6 \
  --skip-decode \
  --prefill-seq-lens 256 512 1024 2048 4096 8192 16384 32768 \
  --warmup 2 \
  --prefill-iters 3 \
  --output outputs/rtx3080ti_b6_flashinfer_mla_prefill.json
```

### 32K Chunked Prefill

```bash
/rshome/yuxin.yang/.cache/flashinfer-3080ti-venv/bin/python run_flashinfer_mla_chunked_prefill.py \
  --seq-len 32768 \
  --chunk-size 4096 \
  --batch 6 \
  --h-q 32 \
  --d-c 512 \
  --d-r 64 \
  --page-size 64 \
  --backend auto \
  --output outputs/rtx3080ti_b6_flashinfer_mla_chunked_prefill_32k.json
```

## 8. 产物

- Decode 结果：`outputs/rtx3080ti_b6_flashinfer_mla_decode.json`
- Prefill 结果：`outputs/rtx3080ti_b6_flashinfer_mla_prefill.json`
- 32K chunked prefill 结果：`outputs/rtx3080ti_b6_flashinfer_mla_chunked_prefill_32k.json`
- 冒烟测试结果：`outputs/rtx3080ti_flashinfer_mla_smoke.json`
- 本报告：`RTX3080TI_MLA_FLASH_EXPERIMENT_REPORT.md`

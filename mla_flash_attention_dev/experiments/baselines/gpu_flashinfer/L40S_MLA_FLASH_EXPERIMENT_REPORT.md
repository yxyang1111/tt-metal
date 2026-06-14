# L40S MLA Flash / FlashInfer 实验报告

## 1. 目标

本实验目标是尽量逼近 NVIDIA L40S 在 DeepSeek-V3 MLA workload 上的硬件上限，并与用户给定表格中的 TT Wormhole (WH) 单芯片 MLA / FlashMLA / S-FMLA 数据对比。

初版实验使用本地 Triton 兼容 kernel 和 PyTorch SDPA baseline，只能证明 L40S 可运行，不能代表硬件上限。最终实验切换到 **FlashInfer MLA**，这是当前可直接在 L40S/SM89 上使用的高效 MLA 路径，支持 paged MLA decode、prefill 和 chunked prefill。

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

| 项目 | L40S 实验环境 |
|---|---|
| GPU | NVIDIA L40S |
| Compute capability | sm89 |
| 可见显存 | 44.39 GiB |
| SM 数 | 142 |
| PyTorch | 2.12.0+cu130 |
| CUDA runtime | 13.0 |
| FlashInfer | 0.6.11.post3 |
| FlashInfer backend | `auto` |

硬件规格对比：

| 硬件 | 峰值算力 | 内存带宽 |
|---|---:|---:|
| NVIDIA L40S | 约 362 TFLOPS BF16/FP16 Tensor Core | 864 GB/s |
| TT WH 单芯片 | 65.5 TFLOPS | 288 GB/s |

L40S 峰值算力约为 TT WH 的 5.5x，显存带宽约为 3.0x。

## 4. 可用高效 MLA 路径结论

| 路径 | L40S/SM89 可用性 | Decode | Prefill | 备注 |
|---|---|---:|---:|---|
| 官方 DeepSeek FlashMLA | 不适合作为 L40S 上限 | 主要 SM90 | prefill 主要 SM90/SM100 sparse 或 SM100 dense | 官方支持矩阵不覆盖 SM89 dense decode |
| 本地 Triton 兼容 kernel | 可运行 | 可运行 | 无 | 功能性 baseline，不代表硬件上限 |
| PyTorch SDPA | 可运行 | 慢 | 8K+ OOM | 通用 baseline，不是 MLA 专用 kernel |
| **FlashInfer MLA** | **可用** | **可运行 256-32K** | **full prefill 可运行 256-16K，32K 需 chunked prefill** | 本报告最终采用 |

因此，如果目标是 L40S 硬件上限，当前最合理的直接可用方案是 **FlashInfer MLA**，而不是官方 FlashMLA 或 PyTorch SDPA。

## 5. Decode 对比

L40S 使用 FlashInfer MLA paged decode，`warmup=10`、`iters=50`。TT WH 数据来自用户给定表格。

| Seq Len | L40S FlashInfer MLA (ms) | TT WH MLA (ms) | TT WH FlashMLA (ms) | TT WH S-FMLA (ms) | L40S / WH FlashMLA | L40S / WH S-FMLA |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.285 | 0.670 | 0.178 | 0.173 | 1.60x | 1.65x |
| 512 | 0.267 | 1.030 | 0.188 | 0.181 | 1.42x | 1.47x |
| 1K | 0.277 | 1.390 | 0.217 | 0.207 | 1.28x | 1.34x |
| 2K | 0.432 | 2.450 | 0.251 | 0.235 | 1.72x | 1.84x |
| 4K | 0.315 | 4.670 | 0.302 | 0.277 | 1.04x | 1.14x |
| 8K | 0.386 | 11.040 | 0.453 | 0.410 | 0.85x | 0.94x |
| 16K | 0.495 | 18.370 | 0.727 | 0.554 | 0.68x | 0.89x |
| 32K | 0.744 | 37.260 | 1.236 | 0.788 | 0.60x | 0.94x |

### Decode 结论

- FlashInfer MLA 后，L40S decode 性能大幅改善，32K 从初版 Triton baseline 的约 25 ms 降到 **0.744 ms**。
- L40S 明显快于 TT WH MLA baseline。
- 对 TT WH FlashMLA / S-FMLA：
  - 短序列 256-4K，L40S 仍略慢或接近。
  - 8K 以后，L40S 已经快于 TT WH FlashMLA；对 S-FMLA 则基本接近，32K 为 0.744 ms vs 0.788 ms。
- 这组结果更符合硬件预期：随着 KV 长度增大，L40S 的带宽优势开始体现。

### Decode 是否已经用满 L40S

没有完全用满。这里的 `L40S FlashInfer MLA` 已经比本地 Triton/SDPA baseline 更接近可用上限，但从有效带宽估算看，仍没有达到 L40S 的 864 GB/s 峰值带宽。

| Seq Len | L40S Latency (ms) | 估算读写量 | 估算有效带宽 | 占 864 GB/s |
|---:|---:|---:|---:|---:|
| 256 | 0.285 | 1.97 MB | 6.89 GB/s | 0.8% |
| 512 | 0.267 | 3.74 MB | 14.02 GB/s | 1.6% |
| 1K | 0.277 | 7.27 MB | 26.26 GB/s | 3.0% |
| 2K | 0.432 | 14.35 MB | 33.19 GB/s | 3.8% |
| 4K | 0.315 | 28.51 MB | 90.62 GB/s | 10.5% |
| 8K | 0.386 | 56.82 MB | 147.35 GB/s | 17.1% |
| 16K | 0.495 | 113.44 MB | 229.08 GB/s | 26.5% |
| 32K | 0.744 | 226.69 MB | 304.89 GB/s | 35.3% |

这个带宽是用压缩 KV/KPE cache 读量加 output 写量得到的下界估算，不包含所有内部访存、调度开销和缓存复用效果。因此它不能严格等价于 Nsight 里的真实 HBM bandwidth，但足以说明趋势：短序列时 L40S 的绝大部分带宽和算力都没有被调动起来；序列变长后，kernel 的固定开销被摊薄，更多 SM 能持续工作，有效带宽才逐步上升。

### 为什么短序列不如 TT，长序列反而优于

短序列 decode 的计算量和 KV 读量太小，例如 256 token 下估算读写量只有约 1.97 MB。对 L40S 这种大 GPU 来说，这个 workload 更像 latency-bound，而不是 bandwidth-bound 或 compute-bound：

- GPU kernel launch、wrapper 调度、paged metadata 处理、CTA 调度和同步等固定开销占比很高。
- `B=6, Hq=32, Q_len=1` 的并行度有限，短序列下每个 request 的 KV 很短，不能充分填满 142 个 SM。
- L40S 是通用 GPU，FlashInfer 虽然是高效 kernel，但仍要走通用 CUDA kernel 调度、paged cache 元数据和全局内存路径。
- TT WH 的 S-FMLA/FlashMLA 是针对该表格 workload 调过的生产算子，短序列下固定开销更低，tile 和片上 SRAM 调度更贴合这个 batch/shape。

长序列时情况反过来。随着 `L` 从 256 增长到 32K，KV cache 读量从约 2 MB 增长到约 227 MB：

- 固定开销被更大的 KV 扫描摊薄，L40S 的大带宽开始发挥作用。
- 长 KV 让 FlashInfer kernel 有更多连续工作可做，SM occupancy 和 memory pipeline 利用率提高。
- TT WH 单芯片只有 288 GB/s 带宽，长序列 decode 越来越接近带宽受限，因此增长斜率更明显。
- L40S 虽然没有满 864 GB/s，但 32K 时估算有效带宽约 305 GB/s，已经接近或超过 TT WH 的物理带宽上限，所以长序列能追平并超过 TT WH FlashMLA/S-FMLA。

因此，表中“短序列 TT 更快、长序列 L40S 更快”的现象是合理的：短序列主要比固定延迟和专用调度，TT 占优；长序列主要比持续带宽和大规模并行，L40S 的硬件资源开始占优。

## 6. Prefill 对比

L40S 使用 FlashInfer MLA paged prefill，`warmup=2`、`prefill_iters=3`。full prefill 可以稳定覆盖 256 到 16K。32K full prefill 在 L40S/SM89 FlashInfer auto backend 下触发 CUDA illegal memory access；加大 workspace 到 1024 MiB 后仍失败，因此 32K 使用 chunked prefill 方式补充。

| Seq Len | L40S FlashInfer MLA Prefill (ms) | TT WH MLA (ms) | TT WH FlashMLA (ms) | TT WH S-FMLA (ms) | L40S / WH FlashMLA | L40S / WH S-FMLA |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.420 | 1.220 | 2.169 | 1.093 | 0.19x | 0.38x |
| 512 | 0.840 | 2.340 | 6.361 | 2.617 | 0.13x | 0.32x |
| 1K | 3.047 | 13.960 | 21.050 | 8.491 | 0.14x | 0.36x |
| 2K | 9.404 | 24.500 | 75.690 | 33.300 | 0.12x | 0.28x |
| 4K | 35.756 | 105.300 | 286.200 | 129.400 | 0.12x | 0.28x |
| 8K | 134.386 | OOM | 1113 | 510 | 0.12x | 0.26x |
| 16K | 470.002 | OOM | 4388 | 2028 | 0.11x | 0.23x |
| 32K | full prefill error; chunked 2366.419 | OOM | OOM | 8050 | - | 0.29x |

### 32K Chunked Prefill

32K full prefill 在 FlashInfer SM89 backend 上失败，因此使用 chunk size 4K 进行 chunked prefill。每个 chunk 的 query 长度为 4K，KV 长度逐块增长到 32K：

| Chunk | Q len | KV len | Latency (ms) |
|---:|---:|---:|---:|
| 0 | 4K | 4K | 43.976 |
| 1 | 4K | 8K | 106.692 |
| 2 | 4K | 12K | 175.155 |
| 3 | 4K | 16K | 242.561 |
| 4 | 4K | 20K | 321.110 |
| 5 | 4K | 24K | 396.804 |
| 6 | 4K | 28K | 502.856 |
| 7 | 4K | 32K | 577.266 |
| **Total** | **32K** | **32K** | **2366.419** |

### Prefill 结论

- 使用 FlashInfer MLA 后，L40S prefill 性能符合硬件预期，显著快于 TT WH 的 MLA / FlashMLA / S-FMLA。
- 256 到 16K 范围内，L40S full prefill 比 TT WH S-FMLA 快约 2.6x 到 4.3x。
- 32K full prefill 当前在 L40S/SM89 FlashInfer auto backend 上失败，但 chunked prefill 可以跑通，总延迟 2366 ms，仍快于 TT WH S-FMLA 的 8050 ms。
- 因此，prefill 阶段如果使用高效 MLA kernel，L40S 的峰值算力/带宽优势能够体现；之前 SDPA baseline 的慢和 OOM 不是硬件问题。

## 7. 总体结论

1. **直接可用的高效 MLA 方案存在**：L40S 上推荐 FlashInfer MLA。
2. **Decode**：FlashInfer MLA decode 让 L40S 在长序列上达到或超过 TT WH FlashMLA/S-FMLA，32K decode 为 0.744 ms。
3. **Prefill**：FlashInfer MLA prefill 让 L40S 明显快于 TT WH，16K 为 470 ms vs TT WH S-FMLA 2028 ms。
4. **32K prefill**：full prefill 当前有 FlashInfer SM89 backend 问题，但 chunked prefill 可跑通，32K 总延迟 2366 ms，仍快于 TT WH S-FMLA。
5. **旧实验不代表硬件上限**：本地 Triton 兼容 kernel 和 PyTorch SDPA baseline 只是功能验证路径；FlashInfer MLA 才是本实验中更接近 L40S 硬件上限的现成方案。

## 8. 复现实验命令

### 安装 FlashInfer

```bash
/data/yangyuxin/miniconda3/envs/flashmla/bin/python -m pip install -U flashinfer-python flashinfer-cubin
/data/yangyuxin/miniconda3/envs/flashmla/bin/python -m pip install -U flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
```

注意：本次安装后环境中的 PyTorch 被 pip 升级到 `2.12.0+cu130`，CUDA 验证正常。

### Decode

```bash
cd /data/yangyuxin/dev/SFMLA/gpu_flashmla_benchmark

/data/yangyuxin/miniconda3/envs/flashmla/bin/python run_flashinfer_mla_benchmark.py \
  --backend auto \
  --skip-prefill \
  --decode-seq-lens 256 512 1024 2048 4096 8192 16384 32768 \
  --warmup 10 \
  --iters 50 \
  --output outputs/gpu_flashinfer_mla_decode.json
```

### Prefill

```bash
cd /data/yangyuxin/dev/SFMLA/gpu_flashmla_benchmark

/data/yangyuxin/miniconda3/envs/flashmla/bin/python run_flashinfer_mla_benchmark.py \
  --backend auto \
  --skip-decode \
  --prefill-seq-lens 256 512 1024 2048 4096 8192 16384 32768 \
  --warmup 2 \
  --prefill-iters 3 \
  --output outputs/gpu_flashinfer_mla_prefill.json
```

### 32K Chunked Prefill

```bash
cd /data/yangyuxin/dev/SFMLA/gpu_flashmla_benchmark

/data/yangyuxin/miniconda3/envs/flashmla/bin/python run_flashinfer_mla_chunked_prefill.py \
  --seq-len 32768 \
  --chunk-size 4096 \
  --batch 6 \
  --h-q 32 \
  --d-c 512 \
  --d-r 64 \
  --page-size 64 \
  --backend auto \
  --output outputs/gpu_flashinfer_mla_chunked_prefill_32k.json
```

## 9. 产物

- FlashInfer benchmark 脚本：`run_flashinfer_mla_benchmark.py`
- FlashInfer chunked prefill 脚本：`run_flashinfer_mla_chunked_prefill.py`
- Decode 结果：`outputs/gpu_flashinfer_mla_decode.json`
- Prefill 结果：`outputs/gpu_flashinfer_mla_prefill.json`
- 32K chunked prefill 结果：`outputs/gpu_flashinfer_mla_chunked_prefill_32k.json`
- 本报告：`L40S_MLA_FLASH_EXPERIMENT_REPORT.md`
# L40S 上的 MLA Flash 实验报告

## 1. 实验目的

本实验在 NVIDIA L40S 上运行 DeepSeek-V3 MLA 形状的注意力算子，并与给定图片中的 TT Wormhole (WH) 单芯片数据对比。

由于官方 DeepSeek/FlashMLA dense decode kernel 主要面向 SM90+，L40S (SM89) 无法直接运行官方 `flash_mla` dense decode。本实验为 L40S 增加了一个本地 Triton MLA Flash decode 后端，并用它完成 decode 阶段测试。prefill 阶段当前没有 L40S 可用的 MLA Flash prefill kernel，因此使用 PyTorch SDPA 作为 L40S prefill baseline，并与 TT WH 的 MLA / FlashMLA / S-FMLA prefill 数据对比。

## 2. Workload 配置

实验使用 DeepSeek-V3 MLA decode/prefill 形状：

| 参数 | 值 |
|---|---:|
| Query heads, `H_q` | 32 |
| KV heads, `H_kv` | 1 |
| Latent dimension, `d_c` | 512 |
| RoPE dimension, `d_r` | 64 |
| QK head dim, `d_qk=d_c+d_r` | 576 |
| V head dim, `d_v=d_c` | 512 |
| Batch size, `B` | 6 |
| Page block size | 64 |
| K chunk size | 128 |
| Element format | BF16 |
| Decode sweep | 256, 512, 1K, 2K, 4K, 8K, 16K, 32K |

decode 阶段为 `Q_len=1`。prefill 阶段为 `Q_len=L`。

## 3. 硬件规格

| 硬件 | 架构 | 峰值算力 | 内存带宽 | 说明 |
|---|---|---:|---:|---|
| NVIDIA L40S | Ada Lovelace, SM89 | 约 362 TFLOPS BF16/FP16 Tensor Core | 864 GB/s | 本实验实测设备 |
| TT Wormhole 单芯片 | Wormhole Tensix | 65.5 TFLOPS | 288 GB/s | 用户提供的对比规格 |

L40S 的峰值算力约为 TT WH 的 5.5x，显存带宽约为 TT WH 的 3.0x。但注意力 decode/prefill 的真实性能还强烈依赖 kernel 组织、片上存储复用、访存模式和中间张量规模。

## 4. 实验环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA L40S |
| Compute capability | sm89 |
| 可见显存 | 44.39 GiB |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Decode 后端 | 本地 Triton MLA Flash |
| Prefill 后端 | PyTorch SDPA baseline |

## 5. Decode 阶段对比

L40S decode 使用本地 Triton MLA Flash backend，运行参数为 `warmup=5, iters=20`。TT WH 的 MLA / FlashMLA / S-FMLA 数据来自用户提供的表格。

| Seq Len | L40S MLA Flash (ms) | TT WH MLA (ms) | TT WH FlashMLA (ms) | TT WH S-FMLA (ms) | L40S / WH FlashMLA | L40S / WH S-FMLA |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.410 | 0.670 | 0.178 | 0.173 | 2.30x | 2.37x |
| 512 | 0.432 | 1.030 | 0.188 | 0.181 | 2.30x | 2.39x |
| 1K | 0.757 | 1.390 | 0.217 | 0.207 | 3.49x | 3.66x |
| 2K | 1.356 | 2.450 | 0.251 | 0.235 | 5.40x | 5.77x |
| 4K | 2.045 | 4.670 | 0.302 | 0.277 | 6.77x | 7.38x |
| 8K | 4.737 | 11.040 | 0.453 | 0.410 | 10.46x | 11.55x |
| 16K | 12.654 | 18.370 | 0.727 | 0.554 | 17.41x | 22.84x |
| 32K | 25.063 | 37.260 | 1.236 | 0.788 | 20.28x | 31.81x |

### Decode 结论

- L40S 上的 Triton MLA Flash decode 能完整跑通 256 到 32K 的 sweep。
- 相比 TT WH MLA，L40S MLA Flash 在所有序列长度上更快。例如 32K 时 L40S 为 25.063 ms，TT WH MLA 为 37.260 ms。
- 相比 TT WH FlashMLA 和 S-FMLA，L40S 当前实现明显更慢，并且差距随序列长度增大。32K 时 L40S 分别比 TT WH FlashMLA 慢 20.28x，比 TT WH S-FMLA 慢 31.81x。
- 这说明 decode 阶段不是单纯由峰值算力或显存带宽决定。TT WH 的生产算子在 MLA decode 的访存和调度上更适合该 workload；而本地 Triton 后端主要用于让 L40S 可运行，并非高度调优的生产 kernel。

## 6. Prefill 阶段对比

L40S 上当前没有可用的 MLA Flash prefill kernel，因此 prefill 阶段使用 PyTorch SDPA baseline。运行参数为 `warmup=5, iters=10`。TT WH 的 MLA / FlashMLA / S-FMLA prefill 数据来自用户提供的表格。

| Seq Len | L40S SDPA Prefill (ms) | TT WH MLA (ms) | TT WH FlashMLA (ms) | TT WH S-FMLA (ms) | L40S / WH FlashMLA | L40S / WH S-FMLA |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 3.9 | 1.2 | 2.2 | 1.1 | 1.78x | 3.54x |
| 512 | 11.9 | 2.3 | 6.4 | 2.6 | 1.88x | 4.56x |
| 1K | 36.7 | 14.0 | 21.1 | 8.5 | 1.75x | 4.33x |
| 2K | 131.0 | 24.5 | 75.7 | 33.3 | 1.73x | 3.93x |
| 4K | 516.6 | 105.3 | 286.2 | 129.4 | 1.80x | 3.99x |
| 8K | OOM | OOM | 1113 | 510 | - | - |
| 16K | OOM | OOM | 4388 | 2028 | - | - |
| 32K | OOM | OOM | OOM | 8050 | - | - |

### Prefill 结论

- L40S SDPA prefill 只能稳定覆盖到 4K；8K、16K、32K 都触发 OOM。
- TT WH S-FMLA 能覆盖到 32K，说明其 prefill 实现更偏流式/分块执行，显存占用控制更好。
- 在 256 到 4K 的可比范围内，L40S SDPA prefill 比 TT WH FlashMLA 慢约 1.7x 到 1.9x，比 TT WH S-FMLA 慢约 3.5x 到 4.6x。
- 这个阶段的比较不是完全等价 kernel 对比：L40S 侧是 PyTorch SDPA baseline，TT WH 侧是专用 MLA/FlashMLA/S-FMLA 算子。因此 prefill 表更适合作为系统级 baseline 对照，而不是证明 L40S 的 MLA Flash prefill 性能。

## 7. 总体观察

1. **decode 阶段**：L40S 可以通过本地 Triton MLA Flash 后端运行完整 MLA dense decode sweep，但与 TT WH 的 FlashMLA/S-FMLA 生产算子相比仍有较大差距。
2. **prefill 阶段**：L40S 当前只能用 SDPA baseline 做对照，且在 8K 以上 OOM；TT WH S-FMLA 可以继续扩展到 32K。
3. **硬件峰值不等于端到端性能**：L40S 的标称算力和带宽都高于 TT WH，但 MLA workload 对 kernel 调度、KV cache 访问、片上复用和中间状态管理非常敏感。
4. **可复现性**：decode 的 L40S 数据来自本地实际运行，prefill 的 L40S 数据来自同一机器上的 PyTorch SDPA baseline；TT WH 数据来自用户提供的论文表格截图。

## 8. 为什么 L40S 峰值更高但实测更慢

这个结果看起来反直觉：L40S 的峰值算力约为 TT WH 的 5.5x，显存带宽约为 TT WH 的 3.0x，但 decode 和 prefill 都慢。根本原因是本实验并不是在比较两个平台上同等优化程度的生产 kernel，而是在比较：

- L40S: 为了让 SM89 能跑起来而写的本地 Triton MLA Flash decode kernel，以及 PyTorch SDPA prefill baseline。
- TT WH: 表格中的 TT-NN MLA / FlashMLA / S-FMLA 生产算子，且是 autotuner 选择后的 kernel 和 tile 配置。

因此，这组结果不能解读为“L40S 硬件不如 TT WH”，更准确的解读是：“当前 L40S 上使用的 kernel 路径没有把 L40S 的算力和带宽有效用起来，而 TT WH 的数据来自针对该 workload 深度优化的生产路径。”

### 8.1 Decode 慢的主要原因

MLA decode 是典型的小 batch、长 KV、强访存/调度敏感 workload。峰值 TFLOPS 很难发挥出来，关键在于是否能复用 KV、复用中间 score、减少重复访存。

本实验中的 L40S Triton decode 后端是一个功能正确的兼容实现，但不是生产级 FlashMLA kernel。当前 kernel 的 grid 是 `(batch, Hq, ceil(d_v/64))`，即每个 query head、每个 64 维 value block 单独启动一组 program。由于 `Hq=32`、`d_v=512`，每个 head 有 8 个 value block，因此会出现明显的重复计算和重复读取：

- 对同一个 query head，QK score 会为了 8 个 value block 重复计算。
- 对 `Hkv=1, Hq=32` 的 GQA/MLA 形状，KV cache 在多个 query head 间高度共享，但当前实现没有充分跨 head 复用。
- softmax 的 online reduction、V 加权和、LSE 写回都按较细粒度拆分，kernel launch 内部的有效算术强度偏低。
- K/V 访问是 paged KV layout，通用 Triton 实现没有做到官方 FlashMLA 或 TT S-FMLA 那种专用调度和片上缓存复用。

以 32K decode 为例，理论上如果只按压缩 KV cache 读一次计算，数据量大约是：

```text
B * L * d_qk * sizeof(BF16)
= 6 * 32768 * 576 * 2
≈ 226 MB
```

在 L40S 864 GB/s 带宽下，单次顺序读的带宽下界只有约 0.26 ms。但实测为 25.063 ms，说明当前 kernel 的瓶颈不是 L40S 的 HBM 峰值带宽本身，而是重复读写、重复 QK 计算、低复用和调度开销造成的有效带宽很低。

这也是为什么 L40S 虽然比 TT WH 的 MLA baseline 快，但比 TT WH FlashMLA/S-FMLA 慢很多：TT WH 表格里的 FlashMLA/S-FMLA 是高度优化的生产 kernel，而本地 Triton kernel 只是为了绕过官方 FlashMLA 不支持 SM89 的限制。

### 8.2 Prefill 慢和 OOM 的主要原因

prefill 阶段更不能直接看成 L40S MLA Flash 性能，因为 L40S 侧当前跑的是 PyTorch SDPA baseline，不是 MLA Flash prefill kernel。这里慢和 OOM 的原因主要有三点：

1. `d_qk=576`、`d_v=512` 远大于常见 FlashAttention fast path 的 head dim 范围。PyTorch SDPA 很可能无法走最高效的 flash attention kernel 路径，容易退化到更通用、更耗显存的实现。
2. prefill 是 `Q_len=L`，注意力矩阵规模随 `L^2` 增长。对于 `B=6, Hq=32`，一旦实现需要保存或间接构造较大的中间 attention 状态，显存压力会迅速爆炸。
3. TT WH 的 S-FMLA prefill 是专用分块/流式实现，可以控制中间状态和片上 SRAM 复用；L40S SDPA baseline 则是通用 PyTorch 算子，不知道 MLA latent/value layout 的特殊结构。

所以 prefill 表里的结论应写成：在“L40S SDPA baseline vs TT WH 专用 MLA/FlashMLA/S-FMLA kernel”的对比下，L40S baseline 更慢并且 8K 以上 OOM。这不能代表 L40S 如果拥有同等优化 MLA Flash prefill kernel 后仍然会更慢。

### 8.3 更公平的对比应该怎么做

如果目标是比较硬件潜力，应该尽量消除 kernel 差异：

1. 在 L40S 上实现更接近生产级的 MLA decode kernel，至少要避免对 8 个 value block 重复计算 QK score，并尽量复用 `Hkv=1` 的 KV 读取。
2. 为 L40S 实现真正的 MLA Flash prefill，而不是用 PyTorch SDPA baseline 代替。
3. 报告 effective bandwidth / achieved TFLOPS，而不只报告 latency。这样可以直接看到 L40S 当前 kernel 是否接近 864 GB/s 或 362 TFLOPS。
4. 对比 TT WH MLA baseline 时，可以单独列一组“生产 kernel vs 生产 kernel”以及“baseline vs baseline”，避免把算法优化差异误认为硬件差异。

当前实验的价值是确认 L40S 上能跑通 MLA Flash decode，并给出一个可复现的功能性 baseline；它还不能说明 L40S 硬件在 MLA 上天然慢于 TT WH。

## 9. 复现实验命令

### Decode: L40S MLA Flash

```bash
cd /data/yangyuxin/dev/SFMLA/gpu_flashmla_benchmark

/data/yangyuxin/miniconda3/envs/flashmla/bin/python run_benchmark.py \
  --flash-backend triton \
  --skip-sdpa \
  --skip-prefill \
  --decode-seq-lens 256 512 1024 2048 4096 8192 16384 32768 \
  --warmup 5 \
  --iters 20
```

### Prefill: L40S SDPA baseline

```bash
cd /data/yangyuxin/dev/SFMLA/gpu_flashmla_benchmark

/data/yangyuxin/miniconda3/envs/flashmla/bin/python - <<'PY'
import torch
from run_benchmark import (
    BATCH_DEFAULT,
    BLOCK_SIZE,
    D_C,
    D_R,
    H_KV_DEFAULT,
    H_Q_DEFAULT,
    BenchConfig,
    bench_sdpa,
)

cfg = BenchConfig(
    batch=BATCH_DEFAULT,
    h_q=H_Q_DEFAULT,
    h_kv=H_KV_DEFAULT,
    d_qk=D_C + D_R,
    d_v=D_C,
    block_size=BLOCK_SIZE,
)
device = torch.device("cuda:0")

for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    result = bench_sdpa(cfg, seq_len, "prefill", device, warmup=5, iters=10)
    print(seq_len, result)
PY
```

## 10. 产物位置

- L40S Triton MLA Flash 后端：`mla_flash_l40.py`
- Benchmark 脚本：`run_benchmark.py`
- 本报告：`L40S_MLA_FLASH_EXPERIMENT_REPORT.md`
- Decode JSON 输出：`outputs/gpu_flashmla_benchmark.json`
- 可视化 Canvas：`/data/yangyuxin/.cursor/projects/data-yangyuxin/canvases/l40s-mla-flash-vs-wh.canvas.tsx`

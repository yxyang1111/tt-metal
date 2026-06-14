# GPU FlashMLA Benchmark

在 Hopper / Blackwell GPU 上运行 DeepSeek FlashMLA 基准测试，与论文 Section 7 的 Wormhole 数据对齐。
L40/L40S (SM89) 可使用本目录内的 Triton MLA Flash 后端运行 dense decode。

## 硬件要求

| 要求 | 说明 |
|------|------|
| **GPU** | SM90 (H100/H800) 或 SM100 (B200/B100)；L40/L40S 走 `--flash-backend triton` |
| **CUDA** | ≥ 12.8（SM100 需要 ≥ 12.9） |
| **PyTorch** | ≥ 2.0 |
| **显存** | ≥ 40 GB（推荐 80 GB 以覆盖长序列 prefill） |

> **注意**: 官方 `flash_mla` dense decode 主要面向 SM90+。L40/L40S (SM89) 请使用本地 Triton MLA Flash 后端。

## 快速开始

```bash
# 1. 安装依赖
bash install.sh

# 2. 运行完整 benchmark（decode + prefill + SDPA 对照）
python3 run_benchmark.py

# L40/L40S: 只跑 MLA Flash dense decode
python3 run_benchmark.py --flash-backend triton --skip-sdpa --skip-prefill \
  --decode-seq-lens 256 512 1024 2048 4096 8192 16384 32768

# 3. 查看结果
cat outputs/gpu_flashmla_benchmark.json    # JSON 原始数据
cat outputs/gpu_flashmla_latex_table.tex   # LaTeX 表格（直接粘到论文）
```

## 运行选项

```bash
# 只跑 decode（快速验证）
python3 run_benchmark.py --skip-prefill

# 桌面 GPU / 小显存本地冒烟验证（例如 RTX 4070 12GB）
python3 run_benchmark.py --local-auto

# 只跑 FlashMLA（不跑 SDPA 对照）
python3 run_benchmark.py --skip-sdpa

# 自定义参数
python3 run_benchmark.py --batch 1 --iters 100

# 指定序列长度
python3 run_benchmark.py --decode-seq-lens 1024 4096 16384 65536
```

## 测试参数（与论文对齐）

| 参数 | 值 | 说明 |
|------|----|------|
| `H_q` | 32 | Query heads (DeepSeek-V3) |
| `H_kv` | 1 | KV heads |
| `d_c` | 512 | Latent / value dimension |
| `d_r` | 64 | RoPE dimension |
| `d_qk` | 576 | = d_c + d_r |
| `d_v` | 512 | = d_c |
| `B` | 6 | Batch size |
| `dtype` | BF16 | Element format |
| `block_size` | 64 | Paged KV cache block size |
| `K chunk size` | 128 | Triton MLA Flash token tile |

## 输出文件

- `outputs/gpu_flashmla_benchmark.json` — 完整结果（GPU 信息 + 每个点的延迟统计）
- `outputs/gpu_flashmla_latex_table.tex` — 可直接 `\input{}` 的 LaTeX 表格，内含 Wormhole 参考数据

## FlashMLA 支持矩阵

| Kernel | SM89 (L40/L40S) | SM90 (Hopper) | SM100 (Blackwell) | 本脚本使用 |
|--------|:---:|:---:|:---:|:---:|
| Dense Decoding (BF16 KV) | ✅ Triton | ✅ official | ✅* official | ✅ |
| Sparse Decoding (FP8 KV) | — | ✅ | ✅ | — |
| Dense Prefill | — | — | ✅ | — |
| Sparse Prefill | — | ✅ | ✅ | — |

\* SM90 代码通过 CUDA 前向兼容在 SM100 上运行。

## 带宽匹配模式（模拟 WH N300 单芯片）

如果你的 GPU (如 H200) 带宽远高于 WH N300 单芯片 (288 GB/s)，可以使用带宽匹配脚本：

```bash
# 1. 探测 GPU 能力和可用的显存时钟
python3 run_bw_matched_benchmark.py --probe

# 2. 方法A: 分析缩放（不改时钟，数学换算到 288 GB/s）
python3 run_bw_matched_benchmark.py --analytical-only

# 3. 方法B: 锁定显存时钟降低带宽（需要权限）
sudo nvidia-smi -i 0 --lock-memory-clocks=405,405  # 替换为 --probe 建议的值
python3 run_bw_matched_benchmark.py --analytical-only
sudo nvidia-smi -i 0 --reset-memory-clocks

# 4. 方法C: 自动带宽扫描（需要权限，在多个时钟下测试）
sudo python3 run_bw_matched_benchmark.py --sweep
```

输出文件: `outputs/gpu_bw_matched_benchmark.json`

### 带宽匹配原理

对于 MLA decode（带宽受限操作），延迟与带宽成反比：

```
scaled_latency = measured_latency × (measured_bandwidth / target_bandwidth)
```

脚本会计算每个序列长度下的：
- **实测带宽利用率** = KV cache 数据量 / 实测延迟
- **缩放延迟** = 保持相同利用率，换算到 288 GB/s
- **理想下界** = 假设 100% 利用率的最低延迟

## 故障排查

**`flash_mla` 安装失败**
```bash
# 确认 CUDA 版本
nvcc --version
# 需要 CUDA ≥ 12.8；如果版本低，需先升级 CUDA toolkit
```

**`RuntimeError: CUDA error: no kernel image is available`**
```
说明 GPU 架构不被官方 flash-mla 支持。L40/L40S 使用 `--flash-backend triton`。
```

**Decode 延迟异常高**
```bash
# 检查 GPU 是否在节流
nvidia-smi -q -d PERFORMANCE
# 确认 GPU 处于 P0 状态，且温度正常
```

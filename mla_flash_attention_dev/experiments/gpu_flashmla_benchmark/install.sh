#!/usr/bin/env bash
set -euo pipefail

echo "=== GPU FlashMLA Benchmark — 环境安装 ==="
echo ""

# ── 1. 检测 GPU ──────────────────────────────────────────────
echo "[1/4] 检测 GPU …"
python3 -c "
import torch
cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f'  GPU:  {name}')
print(f'  SM:   sm{cc[0]}{cc[1]}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  PyTorch: {torch.__version__}')
if cc[0] < 8:
    raise SystemExit('ERROR: 需要 SM80+ (Ampere / Hopper / Blackwell)')
"

# ── 2. 安装 flash-mla (官方 pip 包) ─────────────────────────
echo ""
echo "[2/4] 安装 flash-mla …"
pip install --upgrade flash-mla 2>&1 | tail -3
echo "  flash-mla 安装完成"

# ── 3. 安装 triton (benchmark 需要) ──────────────────────────
echo ""
echo "[3/4] 检查 triton …"
python3 -c "import triton; print(f'  triton {triton.__version__} 已安装')" 2>/dev/null \
  || { echo "  安装 triton …"; pip install triton 2>&1 | tail -2; }

# ── 4. 验证 ──────────────────────────────────────────────────
echo ""
echo "[4/4] 验证 flash_mla 可导入 …"
python3 -c "
from flash_mla import get_mla_metadata, flash_mla_with_kvcache
print('  ✓ flash_mla 导入成功')
"

echo ""
echo "=== 安装完成。运行 benchmark："
echo "    python3 run_benchmark.py"
echo "==="

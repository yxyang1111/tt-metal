#!/usr/bin/env python3
"""
GPU FlashMLA benchmark — 带宽匹配模式。

通过降低显存时钟来逼近 TT Wormhole N300 单芯片的带宽 (288 GB/s)，
并在多个带宽水平下运行 FlashMLA decode，然后插值/外推到目标带宽。

用法:
    # 1. 查看 GPU 支持的显存时钟范围
    python3 run_bw_matched_benchmark.py --probe

    # 2. 带宽扫描模式：自动在多个时钟下测试（需要 sudo 或 root 权限设置时钟）
    python3 run_bw_matched_benchmark.py --sweep

    # 3. 手动指定显存时钟（需要提前用 nvidia-smi 设置）
    python3 run_bw_matched_benchmark.py --mem-clock 405

    # 4. 仅分析缩放（不改时钟，基于全速测量做数学换算）
    python3 run_bw_matched_benchmark.py --analytical-only
    python3 run_bw_matched_benchmark.py --flash-backend triton --analytical-only  # L40/L40S

DeepSeek-V3 MLA shapes: H_q=32, H_kv=1, d_qk=576, d_v=512, BF16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# ═══════════════════════════════════════════════════════════════
# Target specs
# ═══════════════════════════════════════════════════════════════
WH_N300_SINGLE_BW_GBS = 288.0   # GB/s, GDDR6
WH_N300_SINGLE_MEM_GB = 12.0    # GB

# Wormhole 论文 Table 1 参考数据 (B=6)
WH_SYSNAME_DECODE = {
    256: 0.173, 512: 0.181, 1024: 0.207, 2048: 0.235,
    4096: 0.277, 8192: 0.410, 16384: 0.554, 32768: 0.788,
    65536: 1.364, 131072: 2.549,
}
WH_FLASHMLA_DECODE = {
    256: 0.178, 512: 0.188, 1024: 0.217, 2048: 0.251,
    4096: 0.302, 8192: 0.453, 16384: 0.727, 32768: 1.236,
    65536: 2.320, 131072: 4.429,
}

# DeepSeek-V3 MLA dimensions
D_C = 512
D_R = 64
D_QK = D_C + D_R
D_V = D_C
H_Q_DEFAULT = 32
H_KV_DEFAULT = 1
BATCH_DEFAULT = 6
BLOCK_SIZE = 64

DECODE_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
WARMUP = 10
BENCH_ITERS = 50


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    h_q: int
    h_kv: int
    d_qk: int
    d_v: int
    block_size: int

    @property
    def label(self) -> str:
        return f"B={self.batch},Hq={self.h_q},Hkv={self.h_kv},dqk={self.d_qk},dv={self.d_v}"


# ═══════════════════════════════════════════════════════════════
# GPU info & bandwidth probing
# ═══════════════════════════════════════════════════════════════

def gpu_info() -> dict:
    props = torch.cuda.get_device_properties(0)
    cc = torch.cuda.get_device_capability(0)
    return {
        "name": props.name,
        "mem_gb": round(props.total_memory / 1024**3, 2),
        "sm": f"sm{cc[0]}{cc[1]}",
        "sm_count": props.multi_processor_count,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def measure_peak_bandwidth_gbs(device: torch.device, size_mb: int = 256) -> float:
    """通过大块 memcpy 测量当前配置下的实际峰值带宽。"""
    n = size_mb * 1024 * 1024 // 2  # BF16 elements
    src = torch.randn(n, dtype=torch.bfloat16, device=device)
    dst = torch.empty_like(src)

    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    iters = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    bytes_total = n * 2 * 2 * iters  # read + write
    elapsed = t1 - t0
    bw_gbs = bytes_total / elapsed / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return bw_gbs


def get_supported_mem_clocks(gpu_idx: int = 0) -> list[int]:
    """从 nvidia-smi 获取支持的显存时钟列表 (MHz)。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_idx),
             "--query-supported-clocks=mem", "--format=csv,noheader,nounits"],
            text=True,
        )
        clocks = sorted(set(int(line.strip()) for line in out.strip().split("\n") if line.strip()))
        return clocks
    except Exception as e:
        print(f"  WARNING: 无法获取支持的显存时钟: {e}")
        return []


def get_current_mem_clock(gpu_idx: int = 0) -> int:
    """获取当前显存时钟 (MHz)。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_idx),
             "--query-gpu=clocks.current.memory", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip())
    except Exception:
        return 0


def set_mem_clock(gpu_idx: int, clock_mhz: int) -> bool:
    """锁定显存时钟。需要 root/sudo 权限或 persistence mode + 管理员。"""
    try:
        subprocess.check_call(
            ["nvidia-smi", "-i", str(gpu_idx),
             f"--lock-memory-clocks={clock_mhz},{clock_mhz}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def reset_mem_clock(gpu_idx: int) -> bool:
    """解除显存时钟锁定。"""
    try:
        subprocess.check_call(
            ["nvidia-smi", "-i", str(gpu_idx), "--reset-memory-clocks"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ═══════════════════════════════════════════════════════════════
# FlashMLA decode benchmark
# ═══════════════════════════════════════════════════════════════

def _try_import_flash_mla(backend: str = "auto"):
    errors = []
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    if backend in ("auto", "triton"):
        try:
            from mla_flash_l40 import get_mla_metadata, flash_mla_with_kvcache
            if backend == "triton" or cc[0] < 9:
                return "triton", get_mla_metadata, flash_mla_with_kvcache, None
        except ImportError as e:
            errors.append(f"triton backend import failed: {e}")

    if backend in ("auto", "official"):
        try:
            from flash_mla import get_mla_metadata, flash_mla_with_kvcache
            return "official", get_mla_metadata, flash_mla_with_kvcache, None
        except ImportError as e:
            errors.append(f"official flash_mla import failed: {e}")

    if backend == "auto":
        try:
            from mla_flash_l40 import get_mla_metadata, flash_mla_with_kvcache
            return "triton", get_mla_metadata, flash_mla_with_kvcache, None
        except ImportError as e:
            errors.append(f"triton backend import failed: {e}")

    return None, None, None, "; ".join(errors) if errors else "no FlashMLA backend available"


def build_paged_kv_cache(cfg: BenchConfig, seq_len: int, device: torch.device):
    num_blocks_per_seq = math.ceil(seq_len / cfg.block_size)
    total_blocks = cfg.batch * num_blocks_per_seq
    k_cache = torch.randn(
        total_blocks, cfg.block_size, cfg.h_kv, cfg.d_qk,
        dtype=torch.bfloat16, device=device,
    )
    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device=device,
    ).view(cfg.batch, num_blocks_per_seq)
    cache_seqlens = torch.full(
        (cfg.batch,), seq_len, dtype=torch.int32, device=device,
    )
    return k_cache, block_table, cache_seqlens


def kv_cache_bytes(cfg: BenchConfig, seq_len: int) -> int:
    """FlashMLA decode 需要读取的 KV cache 字节数。"""
    num_blocks = math.ceil(seq_len / cfg.block_size)
    total_blocks = cfg.batch * num_blocks
    return total_blocks * cfg.block_size * cfg.h_kv * cfg.d_qk * 2  # BF16 = 2 bytes


def bench_flash_mla_decode(
    cfg: BenchConfig,
    seq_len: int,
    device: torch.device,
    warmup: int,
    iters: int,
    backend: str,
) -> dict:
    backend_name, get_mla_metadata, flash_mla_with_kvcache, import_error = _try_import_flash_mla(backend)
    if get_mla_metadata is None:
        return {"status": "skip", "error": import_error or "flash_mla not installed"}

    q = torch.randn(cfg.batch, 1, cfg.h_q, cfg.d_qk, dtype=torch.bfloat16, device=device)
    k_cache, block_table, cache_seqlens = build_paged_kv_cache(cfg, seq_len, device)

    sched_meta, _ = get_mla_metadata()
    scale = cfg.d_qk ** -0.5

    def run():
        return flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, cfg.d_v,
            sched_meta, None, softmax_scale=scale, causal=True,
        )

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                out, lse = run()
                del out, lse
            torch.cuda.synchronize()

            lats = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out, lse = run()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                lats.append((t1 - t0) * 1000.0)
                del out, lse

        result = {
            "kernel": "flashmla_decode",
            "seq_len": seq_len,
            "mean_ms": round(statistics.mean(lats), 4),
            "min_ms": round(min(lats), 4),
            "max_ms": round(max(lats), 4),
            "std_ms": round(statistics.pstdev(lats), 4),
            "samples": len(lats),
            "status": "ok",
            "backend": backend_name,
        }
        kv_bytes = kv_cache_bytes(cfg, seq_len)
        result["kv_cache_bytes"] = kv_bytes
        result["achieved_bw_gbs"] = round(kv_bytes / (result["mean_ms"] / 1000) / 1e9, 2)
        return result
    except Exception as e:
        torch.cuda.empty_cache()
        return {"status": "error", "error": str(e)[:200], "seq_len": seq_len, "backend": backend_name}
    finally:
        del q, k_cache, block_table, cache_seqlens
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# Analytical scaling
# ═══════════════════════════════════════════════════════════════

def analytical_scale(result: dict, target_bw_gbs: float) -> dict:
    """基于带宽利用率，将测量结果缩放到目标带宽。

    对于带宽受限的 MLA decode:
        latency_new = kv_bytes / target_bw
    这给出了理想的下界（100% 带宽利用率下的延迟）。

    更保守的估算考虑实际利用率:
        latency_new = measured_latency × (measured_bw / target_bw)
    """
    if result["status"] != "ok":
        return result

    measured_bw = result.get("achieved_bw_gbs", 0)
    if measured_bw <= 0:
        return result

    kv_bytes = result.get("kv_cache_bytes", 0)
    measured_ms = result["mean_ms"]

    # 方法1：保持相同的带宽利用率，缩放到目标带宽
    scale_factor = measured_bw / target_bw_gbs
    scaled_ms = measured_ms * scale_factor

    # 方法2：理想下界（假设 100% 带宽利用率）
    ideal_ms = (kv_bytes / target_bw_gbs / 1e9) * 1000  # -> ms

    bw_util = measured_bw / result.get("peak_bw_gbs", measured_bw) if result.get("peak_bw_gbs") else 0

    return {
        **result,
        "scaled_mean_ms": round(scaled_ms, 4),
        "ideal_lower_bound_ms": round(ideal_ms, 4),
        "bw_utilization": round(bw_util, 4) if bw_util > 0 else None,
        "scale_factor": round(scale_factor, 2),
        "target_bw_gbs": target_bw_gbs,
    }


# ═══════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════

def fmt_seq(s: int) -> str:
    return f"{s // 1024}K" if s >= 1024 and s % 1024 == 0 else str(s)


def print_results_table(title: str, results: list[dict], show_scaling: bool = False):
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")

    if show_scaling:
        header = (f"{'Seq Len':>10s}  {'Measured':>10s}  {'Achvd BW':>10s}  "
                  f"{'BW Util%':>8s}  {'Scaled':>10s}  {'Ideal LB':>10s}  "
                  f"{'WH SFMLA':>10s}  {'WH FLA':>10s}  Status")
        print(header)
        print("-" * 120)
        for r in results:
            sl = fmt_seq(r["seq_len"])
            if r["status"] == "ok":
                wh_sys = WH_SYSNAME_DECODE.get(r["seq_len"], 0)
                wh_fla = WH_FLASHMLA_DECODE.get(r["seq_len"], 0)
                bw_util = r.get("bw_utilization")
                bw_util_s = f"{bw_util*100:.1f}%" if bw_util else "---"
                print(f"{sl:>10s}  {r['mean_ms']:10.4f}  {r.get('achieved_bw_gbs',0):8.1f}GB  "
                      f"{bw_util_s:>8s}  {r.get('scaled_mean_ms',0):10.4f}  {r.get('ideal_lower_bound_ms',0):10.4f}  "
                      f"{wh_sys:10.3f}  {wh_fla:10.3f}  ok")
            else:
                err = r.get("error", "")[:40]
                print(f"{sl:>10s}  {'---':>10s}  {'---':>10s}  {'---':>8s}  "
                      f"{'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {r['status']}: {err}")
    else:
        header = (f"{'Seq Len':>10s}  {'Mean (ms)':>10s}  {'Min (ms)':>10s}  "
                  f"{'Achvd BW':>10s}  {'WH SFMLA':>10s}  Status")
        print(header)
        print("-" * 120)
        for r in results:
            sl = fmt_seq(r["seq_len"])
            if r["status"] == "ok":
                wh_sys = WH_SYSNAME_DECODE.get(r["seq_len"], 0)
                print(f"{sl:>10s}  {r['mean_ms']:10.4f}  {r['min_ms']:10.4f}  "
                      f"{r.get('achieved_bw_gbs',0):8.1f}GB  {wh_sys:10.3f}  ok")
            else:
                err = r.get("error", "")[:40]
                print(f"{sl:>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {r['status']}: {err}")


# ═══════════════════════════════════════════════════════════════
# Probe mode
# ═══════════════════════════════════════════════════════════════

def probe_gpu(gpu_idx: int):
    """显示 GPU 信息和支持的显存时钟。"""
    info = gpu_info()
    print("=" * 80)
    print("  GPU Bandwidth Probe")
    print("=" * 80)
    print(f"  GPU:     {info['name']} ({info['mem_gb']} GB, {info['sm']}, {info['sm_count']} SMs)")
    print(f"  Stack:   PyTorch {info['pytorch']}, CUDA {info['cuda']}")

    device = torch.device("cuda:0")
    peak_bw = measure_peak_bandwidth_gbs(device)
    print(f"  实测峰值带宽: {peak_bw:.1f} GB/s")
    print(f"  目标带宽 (WH N300 单芯片): {WH_N300_SINGLE_BW_GBS} GB/s")
    print(f"  带宽比值: {peak_bw / WH_N300_SINGLE_BW_GBS:.1f}x")

    cur_clock = get_current_mem_clock(gpu_idx)
    print(f"  当前显存时钟: {cur_clock} MHz")

    clocks = get_supported_mem_clocks(gpu_idx)
    if clocks:
        print(f"\n  支持的显存时钟 (MHz): {len(clocks)} 个")
        print(f"    最低: {clocks[0]} MHz")
        print(f"    最高: {clocks[-1]} MHz")

        if cur_clock > 0 and clocks[-1] > 0:
            ratio_min = clocks[0] / clocks[-1]
            est_min_bw = peak_bw * ratio_min
            print(f"\n  最低时钟下预估带宽: ~{est_min_bw:.0f} GB/s")
            print(f"  vs N300 单芯片 (288 GB/s): {est_min_bw / WH_N300_SINGLE_BW_GBS:.1f}x")

            target_clock = int(clocks[-1] * WH_N300_SINGLE_BW_GBS / peak_bw)
            print(f"\n  达到 288 GB/s 理论需要的时钟: ~{target_clock} MHz")
            closest = min(clocks, key=lambda c: abs(c - target_clock))
            print(f"  最接近的可用时钟: {closest} MHz (预估 ~{peak_bw * closest / clocks[-1]:.0f} GB/s)")

        print(f"\n  建议测试的时钟点 (从低到高):")
        test_clocks = [clocks[0]]
        for c in clocks:
            ratio = c / clocks[-1]
            est_bw = peak_bw * ratio
            if est_bw <= WH_N300_SINGLE_BW_GBS * 1.5 and c not in test_clocks:
                test_clocks.append(c)
        if clocks[-1] not in test_clocks:
            test_clocks.append(clocks[-1])
        for c in sorted(set(test_clocks)):
            ratio = c / clocks[-1]
            est_bw = peak_bw * ratio
            marker = " <-- 最接近 N300" if c == min(clocks, key=lambda x: abs(peak_bw * x / clocks[-1] - WH_N300_SINGLE_BW_GBS)) else ""
            print(f"    {c:5d} MHz  →  ~{est_bw:6.0f} GB/s{marker}")
    else:
        print("  无法获取显存时钟列表")

    print()
    print("  使用方式:")
    print("    # 锁定显存时钟 (需要权限)")
    print(f"    sudo nvidia-smi -i {gpu_idx} --lock-memory-clocks=MIN,MIN")
    print(f"    python3 {Path(__file__).name} --analytical-only")
    print(f"    sudo nvidia-smi -i {gpu_idx} --reset-memory-clocks")


# ═══════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════

def run_decode_benchmark(
    cfg: BenchConfig,
    device: torch.device,
    seq_lens: list[int],
    warmup: int,
    iters: int,
    peak_bw: float,
    backend: str,
    label: str = "",
) -> list[dict]:
    """运行 FlashMLA decode 测试并返回结果。"""
    results = []
    for sl in sorted(seq_lens):
        r = bench_flash_mla_decode(cfg, sl, device, warmup, iters, backend)
        r["seq_len"] = sl
        if r["status"] == "ok":
            r["peak_bw_gbs"] = round(peak_bw, 2)
        if label:
            r["label"] = label
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GPU FlashMLA benchmark — 带宽匹配 WH N300 模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--probe", action="store_true",
                        help="仅探测 GPU 能力和显存时钟范围")
    parser.add_argument("--sweep", action="store_true",
                        help="自动扫描多个显存时钟 (需要权限)")
    parser.add_argument("--analytical-only", action="store_true",
                        help="仅在当前时钟设置下运行，用分析缩放")
    parser.add_argument("--mem-clock", type=int, default=0,
                        help="手动指定显存时钟 MHz (0=不改)")
    parser.add_argument("--target-bw", type=float, default=WH_N300_SINGLE_BW_GBS,
                        help=f"目标带宽 GB/s (default: {WH_N300_SINGLE_BW_GBS})")
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--decode-seq-lens", nargs="+", type=int, default=DECODE_SEQ_LENS)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--flash-backend",
        choices=("auto", "triton", "official"),
        default="auto",
        help="FlashMLA 后端；auto 会在 L40/L40S(sm89) 上使用 Triton",
    )
    args = parser.parse_args()

    if args.gpu != 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda:0")
    info = gpu_info()

    # ── Probe mode ──
    if args.probe:
        probe_gpu(args.gpu)
        return

    cfg = BenchConfig(
        batch=args.batch, h_q=H_Q_DEFAULT, h_kv=H_KV_DEFAULT,
        d_qk=D_QK, d_v=D_V, block_size=BLOCK_SIZE,
    )

    backend_name, get_mla_metadata, _, import_error = _try_import_flash_mla(args.flash_backend)
    if get_mla_metadata is None:
        print(f"ERROR: FlashMLA backend 不可用: {import_error}")
        sys.exit(1)

    print("=" * 120)
    print("  GPU FlashMLA Benchmark — 带宽匹配模式 (目标: WH N300 单芯片)")
    print("=" * 120)
    print(f"  GPU:         {info['name']} ({info['mem_gb']} GB, {info['sm']}, {info['sm_count']} SMs)")
    print(f"  Stack:       PyTorch {info['pytorch']}, CUDA {info['cuda']}")
    print(f"  FlashMLA:    {backend_name} backend")
    print(f"  Config:      {cfg.label}, block_size={cfg.block_size}")
    print(f"  目标带宽:    {args.target_bw} GB/s (WH N300 单芯片 = {WH_N300_SINGLE_BW_GBS} GB/s)")

    # ── Measure peak bandwidth ──
    print(f"\n  测量峰值带宽...")
    peak_bw = measure_peak_bandwidth_gbs(device)
    print(f"  当前峰值带宽: {peak_bw:.1f} GB/s")
    print(f"  带宽比值:     {peak_bw / args.target_bw:.1f}x (vs 目标)")

    all_sweep_results = {}

    # ── Sweep mode ──
    if args.sweep:
        clocks = get_supported_mem_clocks(args.gpu)
        if not clocks:
            print("  ERROR: 无法获取显存时钟列表，回退到 analytical-only 模式")
            args.analytical_only = True
        else:
            max_clock = clocks[-1]
            test_clocks = set()

            # 找到接近 N300 带宽的时钟
            for c in clocks:
                est_bw = peak_bw * c / max_clock
                if est_bw <= args.target_bw * 2.0:
                    test_clocks.add(c)

            test_clocks.add(clocks[0])
            test_clocks.add(max_clock)

            # 添加中间点用于插值
            for c in clocks:
                est_bw = peak_bw * c / max_clock
                if est_bw <= args.target_bw * 4.0 and len(test_clocks) < 8:
                    test_clocks.add(c)

            test_clocks = sorted(test_clocks)
            print(f"\n  将在 {len(test_clocks)} 个显存时钟点测试:")
            for c in test_clocks:
                est_bw = peak_bw * c / max_clock
                print(f"    {c} MHz → ~{est_bw:.0f} GB/s")

            for clock in test_clocks:
                print(f"\n{'─' * 60}")
                print(f"  设置显存时钟: {clock} MHz")
                if not set_mem_clock(args.gpu, clock):
                    print(f"  WARNING: 设置显存时钟失败 (可能需要 sudo)，跳过")
                    continue

                time.sleep(1)
                cur_bw = measure_peak_bandwidth_gbs(device)
                print(f"  实测带宽: {cur_bw:.1f} GB/s")

                results = run_decode_benchmark(
                    cfg, device, args.decode_seq_lens,
                    args.warmup, args.iters, cur_bw, args.flash_backend,
                    label=f"{clock}MHz",
                )
                for r in results:
                    if r["status"] == "ok":
                        r["mem_clock_mhz"] = clock
                        r["peak_bw_gbs"] = round(cur_bw, 2)
                all_sweep_results[clock] = {
                    "clock_mhz": clock,
                    "peak_bw_gbs": round(cur_bw, 2),
                    "results": results,
                }
                print_results_table(f"FlashMLA Decode @ {clock} MHz (~{cur_bw:.0f} GB/s)", results)

            # 恢复
            reset_mem_clock(args.gpu)
            print(f"\n  已恢复默认显存时钟")

    # ── Analytical-only or after sweep ──
    if args.analytical_only or args.mem_clock > 0:
        if args.mem_clock > 0:
            print(f"\n  注意: 请确保已手动设置显存时钟为 {args.mem_clock} MHz")
            time.sleep(0.5)
            cur_bw = measure_peak_bandwidth_gbs(device)
            print(f"  实测带宽: {cur_bw:.1f} GB/s")
        else:
            cur_bw = peak_bw

        print(f"\n  运行 FlashMLA Decode @ 当前带宽 ({cur_bw:.0f} GB/s)...")
        results = run_decode_benchmark(
            cfg, device, args.decode_seq_lens,
            args.warmup, args.iters, cur_bw, args.flash_backend,
        )
        print_results_table("FlashMLA Decode (实测)", results)

        # Analytical scaling
        scaled = [analytical_scale(r, args.target_bw) for r in results]
        print_results_table(
            f"FlashMLA Decode → 缩放到 {args.target_bw} GB/s (WH N300 等效)",
            scaled, show_scaling=True,
        )

        all_sweep_results["analytical"] = {
            "measured_bw_gbs": round(cur_bw, 2),
            "target_bw_gbs": args.target_bw,
            "results_measured": results,
            "results_scaled": scaled,
        }

    # ── 如果既没有 sweep 也没有 analytical ──
    if not args.sweep and not args.analytical_only and args.mem_clock == 0:
        print(f"\n  运行 FlashMLA Decode @ 全速 ({peak_bw:.0f} GB/s)...")
        results = run_decode_benchmark(
            cfg, device, args.decode_seq_lens,
            args.warmup, args.iters, peak_bw, args.flash_backend,
        )
        print_results_table("FlashMLA Decode (全速)", results)

        scaled = [analytical_scale(r, args.target_bw) for r in results]
        print_results_table(
            f"FlashMLA Decode → 缩放到 {args.target_bw} GB/s (WH N300 等效)",
            scaled, show_scaling=True,
        )

        all_sweep_results["default"] = {
            "measured_bw_gbs": round(peak_bw, 2),
            "target_bw_gbs": args.target_bw,
            "results_measured": results,
            "results_scaled": scaled,
        }

    # ── 保存结果 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu": info,
        "config": asdict(cfg),
        "target_bw_gbs": args.target_bw,
        "wh_n300_single_chip": {
            "bw_gbs": WH_N300_SINGLE_BW_GBS,
            "mem_gb": WH_N300_SINGLE_MEM_GB,
        },
        "sweep_results": all_sweep_results,
        "wh_reference": {
            "sysname_decode": WH_SYSNAME_DECODE,
            "flashmla_decode": WH_FLASHMLA_DECODE,
        },
    }
    out_path = OUTPUT_DIR / "gpu_bw_matched_benchmark.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {out_path}")

    # ── 打印对比总结 ──
    print(f"\n{'=' * 120}")
    print(f"  对比总结: GPU FlashMLA (缩放到 {args.target_bw} GB/s) vs WH N300 单芯片")
    print(f"{'=' * 120}")

    final_results = None
    for key in ["analytical", "default"]:
        if key in all_sweep_results and "results_scaled" in all_sweep_results[key]:
            final_results = all_sweep_results[key]["results_scaled"]
            break

    if final_results:
        print(f"{'Seq Len':>10s}  {'GPU@288':>10s}  {'WH SFMLA':>10s}  {'WH FLA':>10s}  {'GPU/WH比':>10s}")
        print("-" * 70)
        for r in final_results:
            if r["status"] != "ok":
                continue
            sl = r["seq_len"]
            scaled = r.get("scaled_mean_ms", 0)
            wh_sys = WH_SYSNAME_DECODE.get(sl, 0)
            wh_fla = WH_FLASHMLA_DECODE.get(sl, 0)
            ratio = scaled / wh_sys if wh_sys > 0 else 0
            print(f"{fmt_seq(sl):>10s}  {scaled:10.3f}  {wh_sys:10.3f}  {wh_fla:10.3f}  {ratio:10.1f}x")


if __name__ == "__main__":
    main()

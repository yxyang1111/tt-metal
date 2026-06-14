#!/usr/bin/env python3
"""
GPU baseline benchmark for MLA decode and prefill using PyTorch SDPA on NVIDIA GPU.

Measures the same DeepSeek-V3 MLA operator shapes used in the Wormhole evaluation
(Section 7) to provide a cross-platform comparison against a commodity GPU with
similar memory capacity (~12 GB).

The MLA workload is mapped to standard GQA attention:
  Q: [B, H_q, Q_len, d_c + d_r]     (fused nope + rope query)
  K: [B, H_kv, L, d_c + d_r]         (fused latent + rope key)
  V: [B, H_kv, L, d_c]               (latent value, first d_c cols of c_t)

PyTorch's scaled_dot_product_attention dispatches to the best available backend
(FlashAttention v2, memory-efficient, or math fallback).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"


@dataclass(frozen=True)
class MLAConfig:
    num_q_heads: int = 32
    num_kv_heads: int = 1
    d_c: int = 512          # latent / value dimension
    d_r: int = 64           # RoPE dimension
    batch: int = 6
    dtype_str: str = "bfloat16"

    @property
    def d_qk(self) -> int:
        return self.d_c + self.d_r

    @property
    def d_v(self) -> int:
        return self.d_c

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_str)


@dataclass
class BenchResult:
    mode: str
    seq_len: int
    config: dict
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    status: str
    error: str = ""
    sdpa_backend: str = ""
    gpu_name: str = ""
    gpu_mem_gb: float = 0.0


DECODE_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
PREFILL_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

WARMUP_ITERS = 10
BENCH_ITERS = 50


def gpu_info() -> tuple[str, float]:
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory / 1024**3


def estimate_kv_bytes(cfg: MLAConfig, seq_len: int) -> int:
    elem_bytes = 2  # bf16
    k_bytes = cfg.batch * cfg.num_kv_heads * seq_len * cfg.d_qk * elem_bytes
    v_bytes = cfg.batch * cfg.num_kv_heads * seq_len * cfg.d_v * elem_bytes
    return k_bytes + v_bytes


def _has_enable_gqa() -> bool:
    """Check if PyTorch SDPA supports the enable_gqa kwarg (>= 2.5)."""
    try:
        q = torch.zeros(1, 2, 1, 16, device="cpu")
        k = torch.zeros(1, 1, 1, 16, device="cpu")
        v = torch.zeros(1, 1, 1, 16, device="cpu")
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except TypeError:
        return False


_ENABLE_GQA_SUPPORTED = _has_enable_gqa()


def make_inputs(
    cfg: MLAConfig, seq_len: int, q_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = cfg.torch_dtype
    q = torch.randn(cfg.batch, cfg.num_q_heads, q_len, cfg.d_qk, dtype=dt, device=device)
    k = torch.randn(cfg.batch, cfg.num_kv_heads, seq_len, cfg.d_qk, dtype=dt, device=device)
    v = torch.randn(cfg.batch, cfg.num_kv_heads, seq_len, cfg.d_v, dtype=dt, device=device)

    if not _ENABLE_GQA_SUPPORTED and cfg.num_kv_heads != cfg.num_q_heads:
        k = k.expand(cfg.batch, cfg.num_q_heads, seq_len, cfg.d_qk)
        v = v.expand(cfg.batch, cfg.num_q_heads, seq_len, cfg.d_v)

    return q, k, v


def run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    kwargs: dict = dict(scale=scale, is_causal=is_causal)
    if _ENABLE_GQA_SUPPORTED and q.shape[1] != k.shape[1]:
        kwargs["enable_gqa"] = True
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def detect_sdpa_backend() -> str:
    """Best-effort detection of which SDPA backends are enabled."""
    backends = []
    try:
        if torch.backends.cuda.flash_sdp_enabled():
            backends.append("flash_sdp")
    except Exception:
        pass
    try:
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            backends.append("mem_efficient_sdp")
    except Exception:
        pass
    return "+".join(backends) if backends else "unknown"


def benchmark_one(
    cfg: MLAConfig,
    seq_len: int,
    mode: str,
    device: torch.device,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> BenchResult:
    gpu_name, gpu_mem = gpu_info()
    q_len = 1 if mode == "decode" else seq_len
    is_causal = (mode == "prefill")
    scale = cfg.d_qk ** -0.5

    kv_bytes = estimate_kv_bytes(cfg, seq_len)
    q_bytes = cfg.batch * cfg.num_q_heads * q_len * cfg.d_qk * 2
    total_bytes = kv_bytes + q_bytes
    gpu_bytes = gpu_mem * 1024**3
    if total_bytes > gpu_bytes * 0.85:
        return BenchResult(
            mode=mode, seq_len=seq_len, config=asdict(cfg),
            mean_ms=0, min_ms=0, max_ms=0, std_ms=0,
            status="OOM_skip",
            error=f"Estimated tensor size {total_bytes/1e9:.2f} GB exceeds 85% of GPU memory",
            gpu_name=gpu_name, gpu_mem_gb=gpu_mem,
        )

    try:
        q, k, v = make_inputs(cfg, seq_len, q_len, device)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return BenchResult(
            mode=mode, seq_len=seq_len, config=asdict(cfg),
            mean_ms=0, min_ms=0, max_ms=0, std_ms=0,
            status="OOM", error=str(e),
            gpu_name=gpu_name, gpu_mem_gb=gpu_mem,
        )

    backend = detect_sdpa_backend()

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                out = run_sdpa(q, k, v, scale, is_causal)
                del out
            torch.cuda.synchronize()

            latencies = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = run_sdpa(q, k, v, scale, is_causal)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
                del out

        mean_ms = statistics.mean(latencies)
        return BenchResult(
            mode=mode, seq_len=seq_len, config=asdict(cfg),
            mean_ms=round(mean_ms, 4),
            min_ms=round(min(latencies), 4),
            max_ms=round(max(latencies), 4),
            std_ms=round(statistics.pstdev(latencies), 4),
            status="ok",
            sdpa_backend=backend,
            gpu_name=gpu_name, gpu_mem_gb=round(gpu_mem, 2),
        )
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return BenchResult(
            mode=mode, seq_len=seq_len, config=asdict(cfg),
            mean_ms=0, min_ms=0, max_ms=0, std_ms=0,
            status="OOM", error=str(e),
            gpu_name=gpu_name, gpu_mem_gb=gpu_mem,
        )
    except Exception as e:
        return BenchResult(
            mode=mode, seq_len=seq_len, config=asdict(cfg),
            mean_ms=0, min_ms=0, max_ms=0, std_ms=0,
            status="error", error=str(e),
            gpu_name=gpu_name, gpu_mem_gb=gpu_mem,
        )
    finally:
        del q, k, v
        torch.cuda.empty_cache()


def format_seq(s: int) -> str:
    if s >= 1024 and s % 1024 == 0:
        return f"{s // 1024}K"
    return str(s)


def main():
    parser = argparse.ArgumentParser(description="GPU MLA baseline benchmark")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--d-c", type=int, default=512)
    parser.add_argument("--d-r", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    parser.add_argument(
        "--decode-seq-lens", nargs="+", type=int,
        default=DECODE_SEQ_LENS,
    )
    parser.add_argument(
        "--prefill-seq-lens", nargs="+", type=int,
        default=PREFILL_SEQ_LENS,
    )
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    cfg = MLAConfig(
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        d_c=args.d_c,
        d_r=args.d_r,
        batch=args.batch,
        dtype_str=args.dtype,
    )
    device = torch.device("cuda:0")
    gpu_name, gpu_mem = gpu_info()

    print(f"GPU: {gpu_name} ({gpu_mem:.2f} GB)")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Config: B={cfg.batch}, H_q={cfg.num_q_heads}, H_kv={cfg.num_kv_heads}, "
          f"d_c={cfg.d_c}, d_r={cfg.d_r}, dtype={cfg.dtype_str}")
    print(f"Warmup={args.warmup}, Iters={args.iters}")
    print()

    results: list[dict] = []

    if not args.skip_decode:
        print("=" * 70)
        print("DECODE (Q_len=1)")
        print("=" * 70)
        print(f"{'Seq Len':>10s}  {'Mean (ms)':>10s}  {'Min (ms)':>10s}  {'Max (ms)':>10s}  {'Std (ms)':>10s}  Status")
        print("-" * 70)
        for seq_len in sorted(args.decode_seq_lens):
            r = benchmark_one(cfg, seq_len, "decode", device, warmup=args.warmup, iters=args.iters)
            results.append(asdict(r))
            if r.status == "ok":
                print(f"{format_seq(seq_len):>10s}  {r.mean_ms:10.4f}  {r.min_ms:10.4f}  {r.max_ms:10.4f}  {r.std_ms:10.4f}  {r.status}")
            else:
                print(f"{format_seq(seq_len):>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {r.status}: {r.error[:60]}")
        print()

    if not args.skip_prefill:
        print("=" * 70)
        print("PREFILL (Q_len=L)")
        print("=" * 70)
        print(f"{'Seq Len':>10s}  {'Mean (ms)':>10s}  {'Min (ms)':>10s}  {'Max (ms)':>10s}  {'Std (ms)':>10s}  Status")
        print("-" * 70)
        for seq_len in sorted(args.prefill_seq_lens):
            r = benchmark_one(cfg, seq_len, "prefill", device, warmup=args.warmup, iters=args.iters)
            results.append(asdict(r))
            if r.status == "ok":
                print(f"{format_seq(seq_len):>10s}  {r.mean_ms:10.4f}  {r.min_ms:10.4f}  {r.max_ms:10.4f}  {r.std_ms:10.4f}  {r.status}")
            else:
                print(f"{format_seq(seq_len):>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {r.status}: {r.error[:60]}")
        print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "gpu": gpu_name,
            "gpu_mem_gb": round(gpu_mem, 2),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "config": asdict(cfg),
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "results": results,
    }
    out_path = OUTPUT_DIR / "gpu_mla_benchmark.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FlashInfer MLA paged-attention benchmark for L40/L40S.

This script targets the DeepSeek-V3 MLA operator shape:
  B=6, Hq=32, Hkv=1, d_c=512, d_r=64, page size=64, BF16.

It uses FlashInfer's native BatchMLAPagedAttentionWrapper, which supports
decode, prefill, and chunked prefill through the same paged MLA API.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

D_C = 512
D_R = 64
H_Q_DEFAULT = 32
BATCH_DEFAULT = 6
PAGE_SIZE = 64
DECODE_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
PREFILL_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    h_q: int
    d_c: int
    d_r: int
    page_size: int
    dtype: str

    @property
    def d_qk(self) -> int:
        return self.d_c + self.d_r

    @property
    def label(self) -> str:
        return f"B={self.batch},Hq={self.h_q},dc={self.d_c},dr={self.d_r},page={self.page_size},{self.dtype}"


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
        "flashinfer": __import__("flashinfer").__version__,
    }


def build_paged_indices(cfg: BenchConfig, q_len: int, kv_len: int, device: torch.device):
    pages_per_seq = math.ceil(kv_len / cfg.page_size)
    total_pages = cfg.batch * pages_per_seq

    qo_indptr = torch.arange(
        0, (cfg.batch + 1) * q_len, q_len, dtype=torch.int32, device=device
    )
    kv_indptr = torch.arange(
        0, (cfg.batch + 1) * pages_per_seq, pages_per_seq, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_len_arr = torch.full((cfg.batch,), kv_len, dtype=torch.int32, device=device)
    return qo_indptr, kv_indptr, kv_indices, kv_len_arr, total_pages


def make_inputs(cfg: BenchConfig, mode: str, seq_len: int, device: torch.device):
    q_len = 1 if mode == "decode" else seq_len
    qo_indptr, kv_indptr, kv_indices, kv_len_arr, total_pages = build_paged_indices(
        cfg, q_len, seq_len, device
    )
    total_q = cfg.batch * q_len
    dtype = torch.bfloat16

    q_nope = torch.randn(total_q, cfg.h_q, cfg.d_c, dtype=dtype, device=device)
    q_pe = torch.randn(total_q, cfg.h_q, cfg.d_r, dtype=dtype, device=device)
    ckv_cache = torch.randn(total_pages, cfg.page_size, cfg.d_c, dtype=dtype, device=device)
    kpe_cache = torch.randn(total_pages, cfg.page_size, cfg.d_r, dtype=dtype, device=device)

    return q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, kv_len_arr


def summarize(kernel: str, seq_len: int, lats: list[float]) -> dict:
    return {
        "kernel": kernel,
        "seq_len": seq_len,
        "mean_ms": round(statistics.mean(lats), 4),
        "min_ms": round(min(lats), 4),
        "max_ms": round(max(lats), 4),
        "std_ms": round(statistics.pstdev(lats), 4),
        "samples": len(lats),
        "status": "ok",
    }


def safe_empty_cache():
    try:
        torch.cuda.empty_cache()
    except Exception:
        # After an illegal memory access the CUDA context may be poisoned.
        # The process can still serialize the failed benchmark row.
        pass


def estimate_mla_bytes(cfg: BenchConfig, mode: str, seq_len: int) -> int:
    # Lower-bound traffic estimate: read compressed KV/KPE cache and write output.
    q_len = 1 if mode == "decode" else seq_len
    kv_bytes = cfg.batch * seq_len * (cfg.d_c + cfg.d_r) * 2
    out_bytes = cfg.batch * q_len * cfg.h_q * cfg.d_c * 2
    return kv_bytes + out_bytes


def run_timed(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
            del out
        torch.cuda.synchronize()

        lats = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            del out
            lats.append((t1 - t0) * 1000.0)
    return lats


def bench_flashinfer_mla(
    cfg: BenchConfig,
    mode: str,
    seq_len: int,
    device: torch.device,
    backend: str,
    warmup: int,
    iters: int,
    workspace_mb: int,
) -> dict:
    try:
        inputs = make_inputs(cfg, mode, seq_len, device)
        q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, kv_len_arr = inputs
        workspace = torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchMLAPagedAttentionWrapper(workspace, backend=backend)
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            num_heads=cfg.h_q,
            head_dim_ckv=cfg.d_c,
            head_dim_kpe=cfg.d_r,
            page_size=cfg.page_size,
            causal=True,
            sm_scale=cfg.d_qk ** -0.5,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        def run():
            return wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)

        lats = run_timed(run, warmup, iters)
        result = summarize(f"flashinfer_mla_{mode}", seq_len, lats)
        result["backend"] = backend
        result["estimated_bytes"] = estimate_mla_bytes(cfg, mode, seq_len)
        result["estimated_bw_gbs"] = round(
            result["estimated_bytes"] / (result["mean_ms"] / 1000.0) / 1e9, 2
        )
        return result
    except torch.cuda.OutOfMemoryError as e:
        safe_empty_cache()
        return {"kernel": f"flashinfer_mla_{mode}", "seq_len": seq_len, "status": "OOM", "error": str(e)[:200]}
    except Exception as e:
        safe_empty_cache()
        return {"kernel": f"flashinfer_mla_{mode}", "seq_len": seq_len, "status": "error", "error": str(e)[:300]}
    finally:
        safe_empty_cache()


def fmt_seq(seq_len: int) -> str:
    return f"{seq_len // 1024}K" if seq_len >= 1024 and seq_len % 1024 == 0 else str(seq_len)


def print_table(title: str, rows: list[dict]):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(f"{'Seq Len':>10s}  {'Mean (ms)':>10s}  {'Min (ms)':>10s}  {'Max (ms)':>10s}  {'Est BW':>10s}  Status")
    print("-" * 100)
    for row in rows:
        sl = fmt_seq(row["seq_len"])
        if row["status"] == "ok":
            print(
                f"{sl:>10s}  {row['mean_ms']:10.4f}  {row['min_ms']:10.4f}  "
                f"{row['max_ms']:10.4f}  {row.get('estimated_bw_gbs', 0):8.1f}GB  ok"
            )
        else:
            print(f"{sl:>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {row['status']}: {row.get('error', '')[:50]}")


def main():
    parser = argparse.ArgumentParser(description="FlashInfer MLA benchmark on L40/L40S")
    parser.add_argument("--backend", default="auto", help="FlashInfer backend: auto/fa2/fa3/etc.")
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    parser.add_argument("--h-q", type=int, default=H_Q_DEFAULT)
    parser.add_argument("--d-c", type=int, default=D_C)
    parser.add_argument("--d-r", type=int, default=D_R)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--prefill-iters", type=int, default=5)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--decode-seq-lens", nargs="+", type=int, default=DECODE_SEQ_LENS)
    parser.add_argument("--prefill-seq-lens", nargs="+", type=int, default=PREFILL_SEQ_LENS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "gpu_flashinfer_mla_benchmark.json")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    cfg = BenchConfig(
        batch=args.batch,
        h_q=args.h_q,
        d_c=args.d_c,
        d_r=args.d_r,
        page_size=args.page_size,
        dtype="bf16",
    )
    info = gpu_info()

    print("=" * 100)
    print("  FlashInfer MLA Benchmark")
    print("=" * 100)
    print(f"  GPU:       {info['name']} ({info['mem_gb']} GB, {info['sm']}, {info['sm_count']} SMs)")
    print(f"  Stack:     PyTorch {info['pytorch']}, CUDA {info['cuda']}, FlashInfer {info['flashinfer']}")
    print(f"  Config:    {cfg.label}")
    print(f"  Backend:   {args.backend}")
    print(f"  Workspace: {args.workspace_mb} MiB")

    all_results: list[dict] = []

    if not args.skip_decode:
        rows = []
        for seq_len in sorted(args.decode_seq_lens):
            row = bench_flashinfer_mla(
                cfg, "decode", seq_len, device, args.backend, args.warmup, args.iters, args.workspace_mb
            )
            rows.append(row)
            all_results.append(row)
        print_table("FlashInfer MLA Decode (Q_len=1)", rows)

    if not args.skip_prefill:
        rows = []
        for seq_len in sorted(args.prefill_seq_lens):
            row = bench_flashinfer_mla(
                cfg, "prefill", seq_len, device, args.backend, args.warmup, args.prefill_iters, args.workspace_mb
            )
            rows.append(row)
            all_results.append(row)
        print_table("FlashInfer MLA Prefill (Q_len=L)", rows)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "gpu": info,
                "config": asdict(cfg),
                "backend": args.backend,
                "warmup": args.warmup,
                "iters": args.iters,
                "prefill_iters": args.prefill_iters,
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()

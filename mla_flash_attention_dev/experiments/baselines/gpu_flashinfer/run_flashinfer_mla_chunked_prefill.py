#!/usr/bin/env python3
"""Run FlashInfer MLA chunked prefill for long sequences."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper


def build_inputs(batch: int, heads: int, d_c: int, d_r: int, page_size: int, q_len: int, kv_len: int):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    pages_per_seq = math.ceil(kv_len / page_size)
    total_pages = batch * pages_per_seq
    qo_indptr = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, (batch + 1) * pages_per_seq, pages_per_seq, dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_len_arr = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
    q_nope = torch.randn(batch * q_len, heads, d_c, dtype=dtype, device=device)
    q_pe = torch.randn(batch * q_len, heads, d_r, dtype=dtype, device=device)
    ckv_cache = torch.randn(total_pages, page_size, d_c, dtype=dtype, device=device)
    kpe_cache = torch.randn(total_pages, page_size, d_r, dtype=dtype, device=device)
    return q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, kv_len_arr


def run_chunk(args, q_len: int, kv_len: int) -> float:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, kv_len_arr = build_inputs(
        args.batch, args.h_q, args.d_c, args.d_r, args.page_size, q_len, kv_len
    )
    workspace = torch.empty(args.workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend=args.backend)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        args.h_q,
        args.d_c,
        args.d_r,
        args.page_size,
        True,
        (args.d_c + args.d_r) ** -0.5,
        dtype,
        dtype,
    )

    def run_once():
        return wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)

    for _ in range(args.warmup):
        out = run_once()
        del out
    torch.cuda.synchronize()

    lats = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        del out
        lats.append((t1 - t0) * 1000.0)
    torch.cuda.empty_cache()
    return sum(lats) / len(lats)


def main():
    parser = argparse.ArgumentParser(description="FlashInfer MLA chunked prefill benchmark")
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--h-q", type=int, default=32)
    parser.add_argument("--d-c", type=int, default=512)
    parser.add_argument("--d-r", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu_flashinfer_mla_chunked_prefill_32k.json"))
    args = parser.parse_args()

    rows = []
    for start in range(0, args.seq_len, args.chunk_size):
        q_len = min(args.chunk_size, args.seq_len - start)
        kv_len = start + q_len
        lat_ms = run_chunk(args, q_len, kv_len)
        row = {"chunk_start": start, "q_len": q_len, "kv_len": kv_len, "latency_ms": round(lat_ms, 4)}
        rows.append(row)
        print(f"chunk start={start} q={q_len} kv={kv_len} latency={lat_ms:.4f} ms")

    total = round(sum(row["latency_ms"] for row in rows), 4)
    payload = {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": vars(args) | {"output": str(args.output), "dtype": "bf16"},
        "chunks": rows,
        "total_latency_ms": total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"total chunked prefill latency: {total:.4f} ms")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

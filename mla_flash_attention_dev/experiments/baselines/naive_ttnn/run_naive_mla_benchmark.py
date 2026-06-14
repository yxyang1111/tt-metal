#!/usr/bin/env python3
"""
Benchmark naive (unfused) MLA on TTNN vs FlashMLA.

"Naive MLA" implements attention as three separate TTNN ops:
  1. scores = Q @ K^T      (ttnn.matmul)
  2. probs  = softmax(scores * scale)   (ttnn.softmax)
  3. output = probs @ V     (ttnn.matmul)

This serves as the "TTNN MLA (naive)" baseline to replace the
host-CPU reference in the paper's evaluation table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import nearest_y
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    nearest_n,
    nearest_pow_2,
    page_table_setup,
    to_paged_cache,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

NUM_HEADS = 32
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
D_ROPE = 64
D_QK = KV_LORA_RANK + D_ROPE  # 576
BLOCK_SIZE = 64
K_CHUNK_SIZE = 128
MAX_CORES_PER_HEAD_BATCH = 4
SCALE = D_QK ** -0.5

DECODE_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
PREFILL_SEQ_LENS = [256, 512, 1024, 2048, 4096]
DEFAULT_BATCH = 1
DEFAULT_WARMUP = 2
DEFAULT_ITERS = 10


@dataclass
class BenchResult:
    method: str
    mode: str
    seq_len: int
    batch: int
    status: str
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    latencies_ms: list[float] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "method": self.method,
            "mode": self.mode,
            "seq_len": self.seq_len,
            "batch": self.batch,
            "status": self.status,
        }
        if self.status == "ok":
            d.update({
                "mean_ms": round(self.mean_ms, 6),
                "min_ms": round(self.min_ms, 6),
                "max_ms": round(self.max_ms, 6),
                "std_ms": round(self.std_ms, 6),
                "latencies_ms": [round(v, 6) for v in (self.latencies_ms or [])],
            })
        else:
            d["error"] = self.error
        return d


def safe_dealloc(*tensors: Any) -> None:
    for t in tensors:
        if t is not None:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass


def summarize(latencies: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
    }


def make_mla_inputs(batch: int, seq_len: int, q_seq_len: int = 1) -> dict[str, torch.Tensor]:
    """Generate MLA-format torch inputs."""
    gen = torch.Generator().manual_seed(42 + seq_len)
    Q = torch.randn(batch, NUM_HEADS, q_seq_len, D_QK, generator=gen, dtype=torch.bfloat16)
    K_latent = torch.randn(batch, NUM_KV_HEADS, seq_len, D_QK, generator=gen, dtype=torch.bfloat16)
    V = K_latent[..., :KV_LORA_RANK].clone()
    return {"Q": Q, "K": K_latent, "V": V}


# ---------------------------------------------------------------------------
# Naive MLA decode: unfused matmul + softmax on TTNN
# ---------------------------------------------------------------------------
def run_naive_mla_decode(
    device: Any,
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> BenchResult:
    """
    Naive MLA decode using three separate TTNN ops.

    GQA broadcast: K is expanded from (B, 1, L, d) to (B, H_q, L, d)
    via repeat_interleave so that basic matmul works.
    """
    inputs = make_mla_inputs(batch, seq_len, q_seq_len=1)
    Q_torch = inputs["Q"]  # (B, H_q, 1, d_qk)
    K_torch = inputs["K"]  # (B, 1, L, d_qk)
    V_torch = inputs["V"]  # (B, 1, L, d_v)

    K_expanded = K_torch.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)  # (B, H_q, L, d_qk)
    V_expanded = V_torch.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)  # (B, H_q, L, d_v)

    tt_Q = ttnn.from_torch(Q_torch, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_K = ttnn.from_torch(K_expanded, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_V = ttnn.from_torch(V_expanded, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def run_once():
        K_T = ttnn.transpose(tt_K, -2, -1)
        scores = ttnn.matmul(tt_Q, K_T)
        safe_dealloc(K_T)
        scores = ttnn.multiply(scores, SCALE)
        probs = ttnn.softmax(scores, dim=-1)
        safe_dealloc(scores)
        out = ttnn.matmul(probs, tt_V)
        safe_dealloc(probs)
        return out

    try:
        for _ in range(warmup):
            out = run_once()
            ttnn.synchronize_device(device)
            safe_dealloc(out)

        latencies: list[float] = []
        for _ in range(iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run_once()
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            safe_dealloc(out)

        stats = summarize(latencies)
        return BenchResult(
            method="naive_mla", mode="decode", seq_len=seq_len,
            batch=batch, status="ok", latencies_ms=latencies, **stats,
        )
    except Exception as e:
        logger.exception(f"naive_mla decode L={seq_len} failed")
        return BenchResult(
            method="naive_mla", mode="decode", seq_len=seq_len,
            batch=batch, status="error", error=str(e),
        )
    finally:
        safe_dealloc(tt_Q, tt_K, tt_V)


# ---------------------------------------------------------------------------
# Naive MLA prefill: unfused matmul + softmax on TTNN (with causal mask)
# ---------------------------------------------------------------------------
def run_naive_mla_prefill(
    device: Any,
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> BenchResult:
    """
    Naive MLA prefill using three separate TTNN ops.
    Materializes the full (L x L) attention matrix, so limited to short sequences.
    """
    inputs = make_mla_inputs(batch, seq_len, q_seq_len=seq_len)
    Q_torch = inputs["Q"]  # (B, H_q, L, d_qk)
    K_torch = inputs["K"]  # (B, 1, L, d_qk)
    V_torch = inputs["V"]  # (B, 1, L, d_v)

    K_expanded = K_torch.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    V_expanded = V_torch.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16), diagonal=1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

    tt_Q = ttnn.from_torch(Q_torch, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_K = ttnn.from_torch(K_expanded, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_V = ttnn.from_torch(V_expanded, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_mask = ttnn.from_torch(causal_mask, device=device, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def run_once():
        K_T = ttnn.transpose(tt_K, -2, -1)
        scores = ttnn.matmul(tt_Q, K_T)
        safe_dealloc(K_T)
        scores = ttnn.multiply(scores, SCALE)
        scores = ttnn.add(scores, tt_mask)
        probs = ttnn.softmax(scores, dim=-1)
        safe_dealloc(scores)
        out = ttnn.matmul(probs, tt_V)
        safe_dealloc(probs)
        return out

    try:
        for _ in range(warmup):
            out = run_once()
            ttnn.synchronize_device(device)
            safe_dealloc(out)

        latencies: list[float] = []
        for _ in range(iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run_once()
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            safe_dealloc(out)

        stats = summarize(latencies)
        return BenchResult(
            method="naive_mla", mode="prefill", seq_len=seq_len,
            batch=batch, status="ok", latencies_ms=latencies, **stats,
        )
    except Exception as e:
        logger.exception(f"naive_mla prefill L={seq_len} failed")
        return BenchResult(
            method="naive_mla", mode="prefill", seq_len=seq_len,
            batch=batch, status="error", error=str(e),
        )
    finally:
        safe_dealloc(tt_Q, tt_K, tt_V, tt_mask)


# ---------------------------------------------------------------------------
# FlashMLA decode (existing production baseline, via paged SDPA)
# ---------------------------------------------------------------------------
def run_flash_mla_decode(
    device: Any,
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> BenchResult:
    inputs = make_mla_inputs(batch, seq_len, q_seq_len=1)
    Q_torch = inputs["Q"]         # (B, H_q, 1, d_qk)
    K_torch = inputs["K"]         # (B, 1, L, d_qk)

    q_for_tt = Q_torch.permute(2, 0, 1, 3)  # (1, B, H_q, d_qk)

    max_num_blocks = seq_len // BLOCK_SIZE * batch
    paged_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks)
    page_table = page_table_setup(batch, paged_cfg)
    tt_k_paged = to_paged_cache(K_torch, page_table, paged_cfg)

    grid_size = device.compute_with_storage_grid_size()
    q_num_cores = min(batch * NUM_HEADS, grid_size.x * grid_size.y)
    block_height = nearest_y(int(np.prod(Q_torch.shape[:-1])) // q_num_cores, ttnn.TILE_SIZE)
    q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, D_QK),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, KV_LORA_RANK),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,
        k_chunk_size=K_CHUNK_SIZE,
        exp_approx_mode=False,
        max_cores_per_head_batch=MAX_CORES_PER_HEAD_BATCH,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(q_for_tt, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=q_mem_config)
    tt_k = ttnn.from_torch(tt_k_paged, device=device, dtype=ttnn.bfloat8_b,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_cur_pos = ttnn.from_torch(
        torch.tensor([seq_len - 1] * batch, dtype=torch.int32),
        device=device, dtype=ttnn.int32,
    )

    def run_once():
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q, tt_k, head_dim_v=KV_LORA_RANK,
            page_table_tensor=tt_page_table,
            cur_pos_tensor=tt_cur_pos,
            is_causal=True, scale=SCALE,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    try:
        for _ in range(warmup):
            out = run_once()
            ttnn.synchronize_device(device)
            safe_dealloc(out)

        latencies: list[float] = []
        for _ in range(iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run_once()
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            safe_dealloc(out)

        stats = summarize(latencies)
        return BenchResult(
            method="flash_mla", mode="decode", seq_len=seq_len,
            batch=batch, status="ok", latencies_ms=latencies, **stats,
        )
    except Exception as e:
        logger.exception(f"flash_mla decode L={seq_len} failed")
        return BenchResult(
            method="flash_mla", mode="decode", seq_len=seq_len,
            batch=batch, status="error", error=str(e),
        )
    finally:
        safe_dealloc(tt_q, tt_k, tt_page_table, tt_cur_pos)


# ---------------------------------------------------------------------------
# FlashMLA prefill (existing production baseline)
# ---------------------------------------------------------------------------
def run_flash_mla_prefill(
    device: Any,
    batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> BenchResult:
    inputs = make_mla_inputs(batch, seq_len, q_seq_len=seq_len)
    Q_torch = inputs["Q"]  # (B, H_q, L, d_qk)
    K_torch = inputs["K"]  # (B, 1, L, d_qk)

    padded_num_heads = nearest_pow_2(nearest_n(NUM_HEADS, 32))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=padded_num_heads,
        k_chunk_size=K_CHUNK_SIZE,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(Q_torch, device=device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_k = ttnn.from_torch(K_torch, device=device, dtype=ttnn.bfloat8_b,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def run_once():
        return ttnn.transformer.flash_mla_prefill(
            tt_q, tt_k, head_dim_v=KV_LORA_RANK,
            scale=SCALE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            is_causal=True,
        )

    try:
        for _ in range(warmup):
            out = run_once()
            ttnn.synchronize_device(device)
            safe_dealloc(out)

        latencies: list[float] = []
        for _ in range(iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run_once()
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            safe_dealloc(out)

        stats = summarize(latencies)
        return BenchResult(
            method="flash_mla", mode="prefill", seq_len=seq_len,
            batch=batch, status="ok", latencies_ms=latencies, **stats,
        )
    except Exception as e:
        logger.exception(f"flash_mla prefill L={seq_len} failed")
        return BenchResult(
            method="flash_mla", mode="prefill", seq_len=seq_len,
            batch=batch, status="error", error=str(e),
        )
    finally:
        safe_dealloc(tt_q, tt_k)


def estimate_naive_dram_bytes(batch: int, seq_len: int, mode: str) -> int:
    """Estimate DRAM usage for naive MLA (with GQA expansion)."""
    h = NUM_HEADS
    if mode == "decode":
        q_bytes = batch * h * 1 * D_QK * 2
        k_bytes = batch * h * seq_len * D_QK * 2
        v_bytes = batch * h * seq_len * KV_LORA_RANK * 2
        scores_bytes = batch * h * 1 * seq_len * 2
        return q_bytes + k_bytes + v_bytes + scores_bytes
    # prefill
    q_bytes = batch * h * seq_len * D_QK * 2
    k_bytes = batch * h * seq_len * D_QK * 2
    v_bytes = batch * h * seq_len * KV_LORA_RANK * 2
    scores_bytes = batch * h * seq_len * seq_len * 2
    mask_bytes = seq_len * seq_len * 2
    return q_bytes + k_bytes + v_bytes + scores_bytes + mask_bytes


DRAM_CAPACITY_BYTES = 12 * 1024**3  # 12 GB


def format_seq_len(seq_len: int) -> str:
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"{seq_len // 1024}K"
    return str(seq_len)


def format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive TTNN MLA vs FlashMLA benchmark")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--decode-seq-lens", nargs="+", type=int, default=DECODE_SEQ_LENS)
    parser.add_argument("--prefill-seq-lens", nargs="+", type=int, default=PREFILL_SEQ_LENS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--skip-naive", action="store_true", help="Only run FlashMLA")
    parser.add_argument("--skip-flash", action="store_true", help="Only run naive MLA")
    parser.add_argument("--skip-prefill", action="store_true", help="Skip prefill benchmarks")
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_device(device_id=args.device_id)
    grid = device.compute_with_storage_grid_size()
    logger.info(f"Device opened: arch={device.arch()}, grid={grid.x}x{grid.y}")

    results: list[dict[str, Any]] = []

    try:
        # ---- Decode benchmarks ----
        for seq_len in sorted(args.decode_seq_lens):
            naive_mem = estimate_naive_dram_bytes(args.batch, seq_len, "decode")
            logger.info(
                f"--- Decode L={format_seq_len(seq_len)} B={args.batch} "
                f"(naive est. {format_bytes(naive_mem)}) ---"
            )

            if not args.skip_naive:
                if naive_mem > DRAM_CAPACITY_BYTES * 0.85:
                    logger.warning(
                        f"Skipping naive decode L={format_seq_len(seq_len)}: "
                        f"estimated {format_bytes(naive_mem)} exceeds DRAM budget"
                    )
                    results.append(BenchResult(
                        method="naive_mla", mode="decode", seq_len=seq_len,
                        batch=args.batch, status="skipped",
                        error=f"DRAM estimate {format_bytes(naive_mem)} exceeds budget",
                    ).to_dict())
                else:
                    r = run_naive_mla_decode(device, args.batch, seq_len, args.warmup, args.iters)
                    logger.info(f"  naive_mla  decode L={format_seq_len(seq_len)}: {r.mean_ms:.3f} ms ({r.status})")
                    results.append(r.to_dict())

            if not args.skip_flash:
                r = run_flash_mla_decode(device, args.batch, seq_len, args.warmup, args.iters)
                logger.info(f"  flash_mla  decode L={format_seq_len(seq_len)}: {r.mean_ms:.3f} ms ({r.status})")
                results.append(r.to_dict())

        # ---- Prefill benchmarks ----
        if not args.skip_prefill:
            for seq_len in sorted(args.prefill_seq_lens):
                naive_mem = estimate_naive_dram_bytes(args.batch, seq_len, "prefill")
                logger.info(
                    f"--- Prefill L={format_seq_len(seq_len)} B={args.batch} "
                    f"(naive est. {format_bytes(naive_mem)}) ---"
                )

                if not args.skip_naive:
                    if naive_mem > DRAM_CAPACITY_BYTES * 0.85:
                        logger.warning(
                            f"Skipping naive prefill L={format_seq_len(seq_len)}: "
                            f"estimated {format_bytes(naive_mem)} exceeds DRAM budget"
                        )
                        results.append(BenchResult(
                            method="naive_mla", mode="prefill", seq_len=seq_len,
                            batch=args.batch, status="skipped",
                            error=f"DRAM estimate {format_bytes(naive_mem)} exceeds budget",
                        ).to_dict())
                    else:
                        r = run_naive_mla_prefill(device, args.batch, seq_len, args.warmup, args.iters)
                        logger.info(f"  naive_mla  prefill L={format_seq_len(seq_len)}: {r.mean_ms:.3f} ms ({r.status})")
                        results.append(r.to_dict())

                if not args.skip_flash:
                    r = run_flash_mla_prefill(device, args.batch, seq_len, args.warmup, args.iters)
                    logger.info(f"  flash_mla  prefill L={format_seq_len(seq_len)}: {r.mean_ms:.3f} ms ({r.status})")
                    results.append(r.to_dict())

    finally:
        try:
            ttnn.close_device(device)
        except Exception:
            pass

    # ---- Write results ----
    payload = {
        "metadata": {
            "batch": args.batch,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "kv_lora_rank": KV_LORA_RANK,
            "d_rope": D_ROPE,
            "d_qk": D_QK,
            "scale": SCALE,
            "block_size": BLOCK_SIZE,
            "k_chunk_size": K_CHUNK_SIZE,
            "warmup": args.warmup,
            "iters": args.iters,
            "note": (
                "naive_mla uses ttnn.matmul + ttnn.softmax (unfused ops, GQA-expanded K/V in DRAM). "
                "flash_mla uses ttnn.transformer.paged_flash_multi_latent_attention_decode (fused kernel). "
                "Both run on the same Wormhole device."
            ),
        },
        "results": results,
    }

    json_path = OUTPUT_DIR / "naive_vs_flash_mla_results.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Results written to {json_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 90)
    print(f"{'Mode':<8} {'SeqLen':>8} {'Naive MLA (ms)':>16} {'FlashMLA (ms)':>16} {'Ratio':>10}")
    print("-" * 90)
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        key = (r["mode"], r["seq_len"])
        by_key.setdefault(key, {})[r["method"]] = r["mean_ms"]

    for (mode, seq_len), methods in sorted(by_key.items()):
        naive = methods.get("naive_mla")
        flash = methods.get("flash_mla")
        naive_str = f"{naive:.3f}" if naive is not None else "---"
        flash_str = f"{flash:.3f}" if flash is not None else "---"
        if naive is not None and flash is not None and flash > 0:
            ratio_str = f"{naive / flash:.2f}x"
        else:
            ratio_str = "---"
        print(f"{mode:<8} {format_seq_len(seq_len):>8} {naive_str:>16} {flash_str:>16} {ratio_str:>10}")
    print("=" * 90)


if __name__ == "__main__":
    main()

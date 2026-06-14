#!/usr/bin/env python3
"""
Batch × Sequence-Length sweep profile for FlashMLADecode (our method) on Wormhole.

Profiles the S-block multicast kernel (`FlashMLADecode.op`) across a grid of
(batch, seq_len) configurations using Tracy, matching the baseline sweep
in `run_batch_seq_sweep_profile.py`.

Usage:
    python run_batch_seq_sweep_ours.py                          # full sweep
    python run_batch_seq_sweep_ours.py --batches 1 2 4 8        # subset
    python run_batch_seq_sweep_ours.py --seq-lens 1024 8192     # subset
    python run_batch_seq_sweep_ours.py --parse-only             # re-analyze
    python run_batch_seq_sweep_ours.py --child-run --batch 2 --seq-len 4096
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "batch_seq_sweep_ours"
THIS_FILE = Path(__file__).resolve()
WORMHOLE_AICLK_MHZ = 1000.0

DEVICE_ANALYSIS_TYPES = [
    "device_kernel_duration",
    "device_kernel_first_to_last_start",
    "device_brisc_kernel_duration",
    "device_ncrisc_kernel_duration",
    "device_trisc0_kernel_duration",
    "device_trisc1_kernel_duration",
    "device_trisc2_kernel_duration",
    "device_compute_cb_wait_front",
    "device_compute_cb_reserve_back",
]

CSV_FLOAT_COLUMNS = [
    "DEVICE FW DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM FPU UTIL (%)",
    "DRAM BW UTIL (%)",
    "NOC UTIL (%)",
]

# Hardware constants (Wormhole B0)
PEAK_DRAM_BW_GBS = 288.0
PEAK_FPU_TFLOPS = 65.5
NUM_COMPUTE_CORES = 64

# MLA workload constants (DeepSeek-V2/V3)
D_C = 512
D_R = 64
D_QK = D_C + D_R  # 576
H = 32
DQHPC = 8
K_CHUNK_SIZE = 128
MAX_CORES_PER_HEAD_BATCH = 4
ELEMENT_BYTES = 2  # BF16
KV_ELEMENT_BYTES = 1  # BF8

# Our method uses flash_multi_latent_attention_decode (non-paged, dense KV cache)
# which has NO redundant K reads (V is extracted from K in shared latent cache)
NUM_S_BLOCKS = 6
CORES_PER_S_BLOCK = 4
ACTIVE_CORES_OURS = NUM_S_BLOCKS * CORES_PER_S_BLOCK  # 24

DEFAULT_SEQ_LENS = [1024, 4096, 8192, 16384, 32768]
DEFAULT_BATCHES = list(range(1, 21))


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    seq_len: int
    iterations: int = 4
    warmup_iterations: int = 1


def case_name(batch: int, seq_len: int) -> str:
    suffix = f"{seq_len // 1024}k" if seq_len >= 1024 else str(seq_len)
    return f"b{batch}_L{suffix}"


def make_workload(batch: int, seq_len: int) -> Workload:
    iters = 4 if batch * seq_len <= 64 * 1024 else 3
    return Workload(name=case_name(batch, seq_len), batch=batch, seq_len=seq_len, iterations=iters)


# ── Child-run: executed under tracy ─────────────────────────────────────


def execute_child_run(batch: int, seq_len: int, iterations: int, warmup: int) -> None:
    from tracy import signpost

    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)
    print(f"[child] device opened, arch={device.arch()}, batch={batch}, seq_len={seq_len}")

    try:
        scale = D_QK ** -0.5

        # Q: [1, batch, H, D_QK] in standard tile DRAM
        q = torch.randn((1, batch, H, D_QK), dtype=torch.bfloat16)
        # K (dense, non-paged): [batch, 1, seq_len, D_QK]
        k = torch.randn((batch, 1, seq_len, D_QK), dtype=torch.bfloat16)

        tt_q = ttnn.from_torch(q, device=device, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_k = ttnn.from_torch(k, device=device, dtype=ttnn.bfloat8_b,
                               layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        grid_size = device.compute_with_storage_grid_size()
        cur_pos = [seq_len - 1] * batch

        prog_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=0,
            k_chunk_size=K_CHUNK_SIZE,
            exp_approx_mode=False,
            max_cores_per_head_batch=MAX_CORES_PER_HEAD_BATCH,
        )
        comp_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def run_one():
            return ttnn.transformer.flash_multi_latent_attention_decode(
                tt_q, tt_k, None,
                head_dim_v=D_C,
                cur_pos=cur_pos,
                scale=scale,
                program_config=prog_cfg,
                compute_kernel_config=comp_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        for _ in range(warmup):
            out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)

        signpost("start")
        for _ in range(iterations):
            out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)
        signpost("stop")

        for t in (tt_q, tt_k):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


# ── Tracy orchestration ─────────────────────────────────────────────────


def run_tracy_profile(w: Workload) -> Path:
    case_dir = OUTPUT_ROOT / w.name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TTNN_OP_PROFILER"] = "1"
    env["TT_METAL_PROFILER_TRACE_TRACKING"] = "1"

    cmd = [
        "python3", "-m", "tracy",
        "-p", "-o", str(case_dir),
        "--check-exit-code", "--op-support-count", "4000", "-t", "5000",
    ]
    for a in DEVICE_ANALYSIS_TYPES:
        cmd.extend(["-a", a])
    cmd.extend([
        str(THIS_FILE), "--child-run",
        "--batch", str(w.batch), "--seq-len", str(w.seq_len),
        "--iters", str(w.iterations), "--warmup", str(w.warmup_iterations),
    ])

    print(f"  Profiling {w.name} (batch={w.batch}, L={w.seq_len}) ...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED: {w.name}")
        print(f"    STDERR (tail):\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Tracy profiling failed for {w.name}")

    csv_path = case_dir / ".logs" / "cpp_device_perf_report.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"cpp_device_perf_report.csv not found for {w.name}")
    print(f"    OK: {csv_path}")
    return csv_path


# ── Report parsing & analysis ───────────────────────────────────────────


def summarize_numeric(series) -> dict[str, float]:
    import pandas as pd
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    vals = vals[vals > 0]
    if vals.empty:
        return {}
    return {"avg_ns": float(vals.mean()), "min_ns": float(vals.min()),
            "max_ns": float(vals.max()), "std_ns": float(vals.std(ddof=0))}


def parse_and_analyze(w: Workload, csv_path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    relevant = w.warmup_iterations + w.iterations
    if len(df) < relevant:
        raise RuntimeError(f"Expected >= {relevant} rows, got {len(df)} for {w.name}")
    df = df.tail(relevant).reset_index(drop=True)
    df = df.iloc[w.warmup_iterations:].copy()

    cs = {}
    for col in CSV_FLOAT_COLUMNS:
        if col in df.columns:
            s = summarize_numeric(df[col])
            if s:
                cs[col] = s

    kernel_ns = cs.get("DEVICE KERNEL DURATION [ns]", {}).get("avg_ns")
    ncrisc_ns = cs.get("DEVICE NCRISC KERNEL DURATION [ns]", {}).get("avg_ns")
    brisc_ns = cs.get("DEVICE BRISC KERNEL DURATION [ns]", {}).get("avg_ns")
    trisc0_ns = cs.get("DEVICE TRISC0 KERNEL DURATION [ns]", {}).get("avg_ns")
    trisc1_ns = cs.get("DEVICE TRISC1 KERNEL DURATION [ns]", {}).get("avg_ns")
    trisc2_ns = cs.get("DEVICE TRISC2 KERNEL DURATION [ns]", {}).get("avg_ns")
    pm_fpu = cs.get("PM FPU UTIL (%)", {}).get("avg_ns")
    pm_dram = cs.get("DRAM BW UTIL (%)", {}).get("avg_ns")

    kernel_us = (kernel_ns / 1000.0) if kernel_ns else 0

    active_cores = ACTIVE_CORES_OURS
    active_peak_tflops = PEAK_FPU_TFLOPS * active_cores / NUM_COMPUTE_CORES

    # FLOPs: QK = 2*L*(D_C+D_R) per head, SV = 2*L*D_C per head
    flops_per_head = 2 * w.seq_len * (2 * D_C + D_R)
    total_flops = flops_per_head * H * w.batch

    # DRAM traffic for our method (multicast eliminates redundancy):
    # Each lane source reads K once, multicasts to N_track workers
    # Ideal: batch × L × (D_QK + D_C) × kv_element_bytes
    ideal_dram = w.batch * w.seq_len * (D_QK + D_C) * KV_ELEMENT_BYTES
    # With multicast, actual DRAM ≈ ideal (1× read, no redundancy)
    total_dram = ideal_dram

    # Overall throughput
    overall_tflops = (total_flops / kernel_ns / 1000.0) if kernel_ns else 0

    # FPU/SFPU active throughput
    fpu_active_us = kernel_us * (pm_fpu / 100.0) if pm_fpu else 0
    fpu_active_ns = fpu_active_us * 1000.0
    fpu_active_tflops = (total_flops / fpu_active_ns / 1000.0) if fpu_active_ns > 0 else 0

    # Bandwidth
    actual_bw_gbs = (total_dram / kernel_ns) if kernel_ns else 0
    eff_bw_gbs = actual_bw_gbs  # no redundancy

    return {
        "case": w.name,
        "batch": w.batch,
        "seq_len": w.seq_len,
        "active_cores": active_cores,
        "active_peak_tflops": round(active_peak_tflops, 2),

        "kernel_us": round(kernel_us, 2),
        "ncrisc_us": round((ncrisc_ns / 1000.0) if ncrisc_ns else 0, 2),
        "brisc_us": round((brisc_ns / 1000.0) if brisc_ns else 0, 2),
        "trisc0_us": round((trisc0_ns / 1000.0) if trisc0_ns else 0, 2),
        "trisc1_us": round((trisc1_ns / 1000.0) if trisc1_ns else 0, 2),
        "trisc2_us": round((trisc2_ns / 1000.0) if trisc2_ns else 0, 2),

        "pm_fpu_util_pct": round(pm_fpu, 2) if pm_fpu else None,
        "pm_dram_bw_util_pct": round(pm_dram, 2) if pm_dram else None,
        "fpu_active_us": round(fpu_active_us, 2),

        "total_flops_G": round(total_flops / 1e9, 3),
        "overall_tflops": round(overall_tflops, 4),
        "overall_vs_active_peak_pct": round(overall_tflops / active_peak_tflops * 100, 2) if active_peak_tflops > 0 else 0,
        "fpu_active_tflops": round(fpu_active_tflops, 4),
        "fpu_active_vs_active_peak_pct": round(fpu_active_tflops / active_peak_tflops * 100, 2) if active_peak_tflops > 0 else 0,

        "total_dram_read_MB": round(total_dram / 1e6, 2),
        "ideal_dram_read_MB": round(ideal_dram / 1e6, 2),
        "actual_bw_gbs": round(actual_bw_gbs, 2),
        "eff_bw_gbs": round(eff_bw_gbs, 2),
        "actual_bw_vs_peak_pct": round(actual_bw_gbs / PEAK_DRAM_BW_GBS * 100, 1) if PEAK_DRAM_BW_GBS > 0 else 0,
    }


# ── Output ──────────────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    seq_lens = sorted(set(r["seq_len"] for r in results))
    batches = sorted(set(r["batch"] for r in results))
    lookup = {(r["batch"], r["seq_len"]): r for r in results}

    print("\n" + "=" * 90)
    print("  [OURS] BATCH × SEQ_LEN SWEEP — Overall Throughput (TFLOP/s)")
    print("=" * 90)
    header = f"{'Batch':>6}" + "".join(f"{'L='+str(s//1024)+'K':>12}" for s in seq_lens)
    print(header)
    for b in batches:
        row = f"{b:>6}"
        for s in seq_lens:
            r = lookup.get((b, s))
            row += f"{r['overall_tflops']:>12.3f}" if r else f"{'FAIL':>12}"
        print(row)

    print("\n" + "=" * 90)
    print("  [OURS] BATCH × SEQ_LEN SWEEP — Kernel Time (us)")
    print("=" * 90)
    print(header)
    for b in batches:
        row = f"{b:>6}"
        for s in seq_lens:
            r = lookup.get((b, s))
            row += f"{r['kernel_us']:>12.1f}" if r else f"{'FAIL':>12}"
        print(row)

    print("\n" + "=" * 90)
    print("  [OURS] BATCH × SEQ_LEN SWEEP — Actual DRAM BW (GB/s)")
    print("=" * 90)
    print(header)
    for b in batches:
        row = f"{b:>6}"
        for s in seq_lens:
            r = lookup.get((b, s))
            row += f"{r['actual_bw_gbs']:>12.1f}" if r else f"{'FAIL':>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Batch × Seq-Len sweep profile for FlashMLADecode (ours).")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--batch", type=int, help="(child-run) batch size")
    parser.add_argument("--seq-len", type=int, help="(child-run) sequence length")
    parser.add_argument("--iters", type=int, default=4, help="(child-run) measurement iterations")
    parser.add_argument("--warmup", type=int, default=1, help="(child-run) warmup iterations")
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES,
                        help="Batch sizes to sweep.")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS,
                        help="Sequence lengths to sweep.")
    parser.add_argument("--parse-only", action="store_true",
                        help="Re-analyze existing profile data without re-running.")
    args = parser.parse_args()

    if args.child_run:
        execute_child_run(args.batch, args.seq_len, args.iters, args.warmup)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    workloads = [make_workload(b, s) for b in args.batches for s in args.seq_lens]
    total = len(workloads)

    results = []
    failed = []
    for idx, w in enumerate(workloads, 1):
        print(f"\n[{idx}/{total}] ", end="")
        csv_path = OUTPUT_ROOT / w.name / ".logs" / "cpp_device_perf_report.csv"
        try:
            if not args.parse_only:
                csv_path = run_tracy_profile(w)
            elif not csv_path.exists():
                print(f"  SKIP {w.name}: no existing data")
                continue
            results.append(parse_and_analyze(w, csv_path))
        except Exception as e:
            print(f"    ERROR {w.name}: {e}")
            failed.append({"case": w.name, "batch": w.batch, "seq_len": w.seq_len, "error": str(e)})

    print_summary(results)

    out_json = OUTPUT_ROOT / "batch_seq_sweep_ours_results.json"
    with open(out_json, "w") as f:
        json.dump({"results": results, "failed": failed}, f, indent=2)
    print(f"\nResults: {out_json}")
    if failed:
        print(f"Failed cases ({len(failed)}):")
        for f_ in failed:
            print(f"  {f_['case']}: {f_['error']}")


if __name__ == "__main__":
    main()

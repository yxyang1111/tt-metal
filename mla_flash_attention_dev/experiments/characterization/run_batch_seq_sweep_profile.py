#!/usr/bin/env python3
"""
Batch × Sequence-Length sweep profile for FlashMLA decode on Wormhole.

Profiles `paged_flash_multi_latent_attention_decode` across a grid of
(batch, seq_len) configurations and extracts per-point:
  - Kernel time, NCRISC/BRISC/TRISC durations
  - PM FPU utilisation → FPU/SFPU active throughput
  - DRAM traffic and effective bandwidth

Usage:
    python run_batch_seq_sweep_profile.py                          # full sweep
    python run_batch_seq_sweep_profile.py --batches 1 2 4 8        # subset
    python run_batch_seq_sweep_profile.py --seq-lens 1024 8192     # subset
    python run_batch_seq_sweep_profile.py --parse-only             # re-analyze
    python run_batch_seq_sweep_profile.py --child-run --case b2_L4096  # (internal)
"""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "batch_seq_sweep"
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

# Hardware constants (Wormhole B0 / N300 single chip)
PEAK_DRAM_BW_GBS = 288.0
PEAK_FPU_TFLOPS = 65.5
NUM_COMPUTE_CORES = 64

# MLA workload constants (DeepSeek-V2)
D_C = 512
D_R = 64
D_QK = D_C + D_R  # 576
H = 32
DQHPC = 8
BLOCK_SIZE = 64
K_CHUNK_SIZE = 128
MAX_CORES_PER_HEAD_BATCH = 4
ELEMENT_BYTES = 2  # BF16 latent cache elements
INT32_BYTES = 4

DEFAULT_SEQ_LENS = [1024, 4096, 8192, 16384, 32768]
DEFAULT_BATCHES = list(range(1, 21))
DEFAULT_REPEATS = 3


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    seq_len: int
    iterations: int = 4
    warmup_iterations: int = 2


def case_name(batch: int, seq_len: int) -> str:
    suffix = f"{seq_len // 1024}k" if seq_len >= 1024 else str(seq_len)
    return f"b{batch}_L{suffix}"


def make_workload(batch: int, seq_len: int) -> Workload:
    iters = 4 if batch * seq_len <= 64 * 1024 else 3
    return Workload(name=case_name(batch, seq_len), batch=batch, seq_len=seq_len, iterations=iters)


def cycles_to_us(c: float) -> float:
    return c / WORMHOLE_AICLK_MHZ


def ns_to_us(v: float | None) -> float | None:
    return v / 1000.0 if v is not None else None


def ns_to_ms(v: float | None) -> float | None:
    return v / 1e6 if v is not None else None


# ── Child-run: executed under tracy ─────────────────────────────────────


def execute_child_run(batch: int, seq_len: int, iterations: int, warmup: int) -> None:
    from tracy import signpost

    import torch
    import ttnn
    from models.common.utility_functions import nearest_y
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
        page_table_setup,
        to_paged_cache,
    )

    device = ttnn.open_device(device_id=0)
    print(f"[child] device opened, arch={device.arch()}, batch={batch}, seq_len={seq_len}")

    try:
        q = torch.randn((1, batch, H, D_QK), dtype=torch.bfloat16)
        k = torch.randn((batch, 1, seq_len, D_QK), dtype=torch.bfloat16)

        max_num_blocks = (seq_len // BLOCK_SIZE) * batch
        paged_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks)
        page_table = page_table_setup(batch, paged_cfg)
        paged_cache_torch = to_paged_cache(k, page_table, paged_cfg)

        grid_size = device.compute_with_storage_grid_size()
        q_num_cores = min(batch * H, grid_size.x * grid_size.y)
        block_height = nearest_y((batch * H) // q_num_cores, ttnn.TILE_SIZE)
        q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)

        q_mem = ttnn.create_sharded_memory_config(
            shape=(block_height, D_QK), core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True,
        )
        out_mem = ttnn.create_sharded_memory_config(
            shape=(block_height, D_C), core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True,
        )

        start_indices = torch.full((batch,), seq_len - 1, dtype=torch.int32)

        tt_q = ttnn.from_torch(q, device=device, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
        tt_k = ttnn.from_torch(paged_cache_torch, device=device, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_pt = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_si = ttnn.from_torch(start_indices, device=device, dtype=ttnn.int32)

        prog_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=0, k_chunk_size=K_CHUNK_SIZE,
            exp_approx_mode=False, max_cores_per_head_batch=MAX_CORES_PER_HEAD_BATCH,
        )
        comp_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False,
        )
        scale = D_QK ** -0.5

        def run_one():
            return ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tt_q, tt_k, page_table_tensor=tt_pt, cur_pos_tensor=tt_si,
                head_dim_v=D_C, scale=scale, program_config=prog_cfg,
                compute_kernel_config=comp_cfg, memory_config=out_mem,
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

        for t in (tt_q, tt_k, tt_pt, tt_si):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


# ── Tracy orchestration ─────────────────────────────────────────────────


def repeat_dir(w: Workload, repeat_idx: int) -> Path:
    return OUTPUT_ROOT / w.name / f"repeat_{repeat_idx + 1}"


def run_tracy_profile(w: Workload, repeat_idx: int, total_repeats: int) -> Path:
    case_dir = repeat_dir(w, repeat_idx)
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

    print(
        f"  Profiling {w.name} (batch={w.batch}, L={w.seq_len}, "
        f"repeat={repeat_idx + 1}/{total_repeats}) ..."
    )
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

    n_hb = math.ceil(H / DQHPC)  # 4
    n_sc = min(MAX_CORES_PER_HEAD_BATCH, math.ceil(w.seq_len / BLOCK_SIZE))
    active_cores = min(n_hb * n_sc * w.batch, NUM_COMPUTE_CORES)
    active_peak_tflops = PEAK_FPU_TFLOPS * active_cores / NUM_COMPUTE_CORES

    # FLOPs: QK = 2*L*(D_C+D_R) per head, SV = 2*L*D_C per head
    flops_per_head = 2 * w.seq_len * (2 * D_C + D_R)
    total_flops = flops_per_head * H * w.batch

    # All kernel-level DRAM traffic for this benchmark configuration.
    #
    # Main stream: the baseline mapping has four head groups and each group
    # streams the full latent K cache from DRAM.  MLA stores V in the first
    # head_dim_v columns of K and runs with reuse_k=true, so V is reused from
    # K's L1 buffer rather than read as an independent DRAM stream.
    #
    # Small metadata streams: paged attention readers fetch page-table rows and
    # the cur_pos tensor from DRAM.  Q and output are sharded L1 tensors in this
    # benchmark, so they do not contribute DRAM traffic inside the kernel.
    k_cache_dram = w.batch * n_hb * w.seq_len * D_QK * ELEMENT_BYTES
    page_table_entries = math.ceil(w.seq_len / BLOCK_SIZE)
    page_table_dram = active_cores * page_table_entries * INT32_BYTES
    cur_pos_dram = active_cores * w.batch * INT32_BYTES
    q_dram = 0
    output_dram = 0

    total_dram = k_cache_dram + page_table_dram + cur_pos_dram + q_dram + output_dram
    ideal_dram = w.batch * w.seq_len * D_QK * ELEMENT_BYTES + page_table_dram + cur_pos_dram

    # Overall throughput
    overall_tflops = (total_flops / kernel_ns / 1000.0) if kernel_ns else 0

    # FPU/SFPU active throughput
    fpu_active_us = kernel_us * (pm_fpu / 100.0) if pm_fpu else 0
    fpu_active_ns = fpu_active_us * 1000.0
    fpu_active_tflops = (total_flops / fpu_active_ns / 1000.0) if fpu_active_ns > 0 else 0

    # Bandwidth demand. Actual delivered bandwidth is calibrated after all rows
    # are parsed because the calibration uses the batch=1 reader time per L.
    requested_bw_gbs = (total_dram / kernel_ns) if kernel_ns else 0
    ideal_requested_bw_gbs = (ideal_dram / kernel_ns) if kernel_ns else 0

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

        "k_cache_dram_read_MB": round(k_cache_dram / 1e6, 4),
        "page_table_dram_read_MB": round(page_table_dram / 1e6, 4),
        "cur_pos_dram_read_MB": round(cur_pos_dram / 1e6, 4),
        "q_dram_read_MB": round(q_dram / 1e6, 4),
        "output_dram_write_MB": round(output_dram / 1e6, 4),
        "total_dram_read_MB": round(total_dram / 1e6, 2),
        "total_dram_traffic_MB": round(total_dram / 1e6, 2),
        "ideal_dram_read_MB": round(ideal_dram / 1e6, 2),
        "requested_bw_gbs": round(requested_bw_gbs, 2),
        "ideal_requested_bw_gbs": round(ideal_requested_bw_gbs, 2),
    }


def aggregate_repeats(w: Workload, repeat_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not repeat_results:
        raise RuntimeError(f"No repeat results to aggregate for {w.name}")

    if len(repeat_results) == 1:
        result = dict(repeat_results[0])
        result["num_repeats"] = 1
        result["repeat_std"] = {}
        result["repeat_results"] = repeat_results
        return result

    result: dict[str, Any] = {
        "case": w.name,
        "batch": w.batch,
        "seq_len": w.seq_len,
        "num_repeats": len(repeat_results),
    }
    repeat_std: dict[str, float] = {}

    keys = set().union(*(r.keys() for r in repeat_results))
    for key in sorted(keys):
        if key in result or key in {"case", "batch", "seq_len"}:
            continue

        values = [r.get(key) for r in repeat_results]
        numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
        if len(numeric_values) == len(values):
            med = statistics.median(numeric_values)
            if all(isinstance(v, int) for v in numeric_values):
                result[key] = int(med)
            else:
                result[key] = round(float(med), 4)
            if len(numeric_values) > 1:
                repeat_std[key] = round(float(statistics.pstdev(numeric_values)), 4)
        else:
            result[key] = values[0]

    result["repeat_std"] = repeat_std
    result["repeat_results"] = repeat_results
    return result


def calibrate_actual_bandwidth(results: list[dict]) -> None:
    """Estimate delivered all-traffic DRAM BW from per-L batch=1 reader rate.

    Tracy does not report a valid DRAM BW counter for these runs.  The raw
    requested traffic divided by kernel time can exceed physical peak when many
    reader cores contend for DRAM, so the figure uses a per-sequence calibration:
    batch=1 all-traffic bytes / batch=1 NCRISC time, scaled by the observed
    reader share of total kernel time for each batch.
    """
    by_seq: dict[int, list[dict]] = {}
    for r in results:
        by_seq.setdefault(r["seq_len"], []).append(r)

    for seq_len, rows in by_seq.items():
        batch1 = next((r for r in rows if r["batch"] == 1), None)
        if not batch1 or batch1["ncrisc_us"] <= 0:
            continue

        batch1_bytes = batch1["total_dram_read_MB"] * 1e6
        cal_rate_gbs = batch1_bytes / (batch1["ncrisc_us"] * 1000.0)

        for r in rows:
            actual_bw_gbs = 0.0
            if r["kernel_us"] > 0:
                actual_bw_gbs = cal_rate_gbs * r["ncrisc_us"] / r["kernel_us"]
            r["actual_bw_gbs"] = round(actual_bw_gbs, 2)
            r["eff_bw_gbs"] = round(actual_bw_gbs / math.ceil(H / DQHPC), 2)
            r["actual_bw_vs_peak_pct"] = (
                round(actual_bw_gbs / PEAK_DRAM_BW_GBS * 100, 1)
                if PEAK_DRAM_BW_GBS > 0 else 0
            )

            repeat_bw = [
                cal_rate_gbs * rep["ncrisc_us"] / rep["kernel_us"]
                for rep in r.get("repeat_results", [])
                if rep.get("kernel_us", 0) > 0
            ]
            if len(repeat_bw) > 1:
                repeat_std = r.setdefault("repeat_std", {})
                repeat_std["actual_bw_gbs"] = round(float(statistics.pstdev(repeat_bw)), 4)
                repeat_std["eff_bw_gbs"] = round(
                    float(statistics.pstdev([bw / math.ceil(H / DQHPC) for bw in repeat_bw])), 4
                )


# ── Output ──────────────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    seq_lens = sorted(set(r["seq_len"] for r in results))
    batches = sorted(set(r["batch"] for r in results))
    lookup = {(r["batch"], r["seq_len"]): r for r in results}

    print("\n" + "=" * 90)
    print("  BATCH × SEQ_LEN SWEEP — Overall Throughput (TFLOP/s)")
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
    print("  BATCH × SEQ_LEN SWEEP — FPU-Active Throughput (TFLOP/s)")
    print("=" * 90)
    print(header)
    for b in batches:
        row = f"{b:>6}"
        for s in seq_lens:
            r = lookup.get((b, s))
            row += f"{r['fpu_active_tflops']:>12.3f}" if r else f"{'FAIL':>12}"
        print(row)

    print("\n" + "=" * 90)
    print("  BATCH × SEQ_LEN SWEEP — Actual DRAM BW (GB/s)")
    print("=" * 90)
    print(header)
    for b in batches:
        row = f"{b:>6}"
        for s in seq_lens:
            r = lookup.get((b, s))
            row += f"{r['actual_bw_gbs']:>12.1f}" if r else f"{'FAIL':>12}"
        print(row)

    print("\n" + "=" * 90)
    print("  BATCH × SEQ_LEN SWEEP — Kernel Time (us)")
    print("=" * 90)
    print(header)
    for b in batches:
        row = f"{b:>6}"
        for s in seq_lens:
            r = lookup.get((b, s))
            row += f"{r['kernel_us']:>12.1f}" if r else f"{'FAIL':>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Batch × Seq-Len sweep profile for FlashMLA decode.")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--batch", type=int, help="(child-run) batch size")
    parser.add_argument("--seq-len", type=int, help="(child-run) sequence length")
    parser.add_argument("--iters", type=int, default=4, help="(child-run) measurement iterations")
    parser.add_argument("--warmup", type=int, default=2, help="(child-run) warmup iterations")
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES,
                        help="Batch sizes to sweep.")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS,
                        help="Sequence lengths to sweep.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"Independent Tracy profiles per (batch, seq_len), default: {DEFAULT_REPEATS}.")
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
        try:
            if not args.parse_only:
                case_dir = OUTPUT_ROOT / w.name
                if case_dir.exists():
                    shutil.rmtree(case_dir)

            repeat_results = []
            for repeat_idx in range(args.repeats):
                csv_path = repeat_dir(w, repeat_idx) / ".logs" / "cpp_device_perf_report.csv"
                if not args.parse_only:
                    csv_path = run_tracy_profile(w, repeat_idx, args.repeats)
                elif not csv_path.exists():
                    legacy_csv_path = OUTPUT_ROOT / w.name / ".logs" / "cpp_device_perf_report.csv"
                    if args.repeats == 1 and legacy_csv_path.exists():
                        csv_path = legacy_csv_path
                    else:
                        raise FileNotFoundError(f"Missing repeat {repeat_idx + 1}/{args.repeats}: {csv_path}")
                repeat_results.append(parse_and_analyze(w, csv_path))

            results.append(aggregate_repeats(w, repeat_results))
        except Exception as e:
            print(f"    ERROR {w.name}: {e}")
            failed.append({"case": w.name, "batch": w.batch, "seq_len": w.seq_len, "error": str(e)})

    calibrate_actual_bandwidth(results)
    print_summary(results)

    out_json = OUTPUT_ROOT / "batch_seq_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump({"results": results, "failed": failed}, f, indent=2)
    print(f"\nResults: {out_json}")
    if failed:
        print(f"Failed cases ({len(failed)}):")
        for f_ in failed:
            print(f"  {f_['case']}: {f_['error']}")


if __name__ == "__main__":
    main()

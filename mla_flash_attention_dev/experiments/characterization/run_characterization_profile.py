#!/usr/bin/env python3
"""
Run comprehensive profiling for Section 3 (Characterization) of the paper.

Profiles baseline FlashMLA decode on Wormhole across L=1K,4K,8K,16K,32K
and extracts all metrics needed for the characterization tables:
  - Per-phase runtime (kernel, NCRISC, BRISC, TRISC0/1/2)
  - Reader sub-stage breakdown (page_table, reserve, issue, wait, push)
  - K vs V source split (k_reserve, k_issue, k_wait, k_push, v_*)
  - Writer sub-stage breakdown (cb_wait, issue, barrier, pop)
  - PM counters (FPU util, DRAM BW util, NOC util)
  - Compute CB wait_front / reserve_back
  - Derived: effective bandwidth, FPU utilization, stall attribution

Usage:
    python run_characterization_profile.py                    # profile all 5 cases
    python run_characterization_profile.py --cases decode_1k decode_32k
    python run_characterization_profile.py --parse-only       # re-analyze existing data
    python run_characterization_profile.py --child-run --case decode_4k   # (internal)
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "profiling"
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
    "DEVICE KERNEL FIRST TO LAST START [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
    "OP TO OP LATENCY [ns]",
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM FPU UTIL (%)",
    "NOC UTIL (%)",
    "MULTICAST NOC UTIL (%)",
    "DRAM BW UTIL (%)",
    "ETH BW UTIL (%)",
    "NPE CONG IMPACT (%)",
]

READER_STAGE_MARKERS = [
    "SDPA-PAGE-TABLE-SUM",
    "SDPA-PAGED-RESERVE-SUM",
    "SDPA-PAGED-ISSUE-SUM",
    "SDPA-PAGED-WAIT-SUM",
    "SDPA-PAGED-PUSH-SUM",
]
READER_SOURCE_MARKERS = [
    "SDPA-K-RESERVE-SUM",
    "SDPA-K-ISSUE-SUM",
    "SDPA-K-WAIT-SUM",
    "SDPA-K-PUSH-SUM",
    "SDPA-V-RESERVE-SUM",
    "SDPA-V-ISSUE-SUM",
    "SDPA-V-WAIT-SUM",
    "SDPA-V-PUSH-SUM",
]
WRITER_STAGE_MARKERS = [
    "SDPA-WRITER-CB-WAIT-SUM",
    "SDPA-WRITER-ISSUE-SUM",
    "SDPA-WRITER-BARRIER-SUM",
    "SDPA-WRITER-POP-SUM",
]

# Hardware constants (Wormhole B0)
PEAK_DRAM_BW_GBS = 200.0
PEAK_FPU_TFLOPS = 65.5
D_C = 512
D_R = 64
D_QK = D_C + D_R  # 576
H = 32
DQHPC = 8
BLOCK_SIZE = 64
K_CHUNK_SIZE = 128
MAX_CORES_PER_HEAD_BATCH = 4
ELEMENT_BYTES = 2  # BF16


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    seq_len: int
    iterations: int = 6
    warmup_iterations: int = 2

    @property
    def k_bytes_single_pass(self) -> int:
        return self.batch * self.seq_len * D_QK

    @property
    def out_bytes(self) -> int:
        return self.batch * H * D_C * 2


WORKLOADS = {
    "decode_1k": Workload("decode_1k", batch=2, seq_len=1024),
    "decode_4k": Workload("decode_4k", batch=2, seq_len=4096),
    "decode_8k": Workload("decode_8k", batch=2, seq_len=8192, iterations=4),
    "decode_16k": Workload("decode_16k", batch=1, seq_len=16384, iterations=4),
    "decode_32k": Workload("decode_32k", batch=1, seq_len=32768, iterations=4),
}


def cycles_to_us(c: float) -> float:
    return c / WORMHOLE_AICLK_MHZ


def ns_to_us(v: float | None) -> float | None:
    return v / 1000.0 if v is not None else None


def ns_to_ms(v: float | None) -> float | None:
    return v / 1e6 if v is not None else None


# ── Child-run: executed under tracy ─────────────────────────────────────


def execute_child_run(case_name: str) -> None:
    from tracy import signpost

    import torch
    import ttnn
    from models.common.utility_functions import nearest_y
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
        page_table_setup,
        to_paged_cache,
    )

    w = WORKLOADS[case_name]
    device = ttnn.open_device(device_id=0)
    print(f"[child] device opened, arch={device.arch()}, case={case_name}")

    try:
        q = torch.randn((1, w.batch, H, D_QK), dtype=torch.bfloat16)
        k = torch.randn((w.batch, 1, w.seq_len, D_QK), dtype=torch.bfloat16)

        max_num_blocks = (w.seq_len // BLOCK_SIZE) * w.batch
        paged_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks)
        page_table = page_table_setup(w.batch, paged_cfg)
        paged_cache_torch = to_paged_cache(k, page_table, paged_cfg)

        grid_size = device.compute_with_storage_grid_size()
        q_num_cores = min(w.batch * H, grid_size.x * grid_size.y)
        block_height = nearest_y((w.batch * H) // q_num_cores, ttnn.TILE_SIZE)
        q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)

        q_mem = ttnn.create_sharded_memory_config(
            shape=(block_height, D_QK), core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True,
        )
        out_mem = ttnn.create_sharded_memory_config(
            shape=(block_height, D_C), core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True,
        )

        start_indices = torch.full((w.batch,), w.seq_len - 1, dtype=torch.int32)

        tt_q = ttnn.from_torch(q, device=device, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
        tt_k = ttnn.from_torch(paged_cache_torch, device=device, dtype=ttnn.bfloat8_b,
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

        for _ in range(w.warmup_iterations):
            out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)

        signpost("start")
        for _ in range(w.iterations):
            out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)
        signpost("stop")

        for t in (tt_q, tt_k, tt_pt, tt_si):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


# ── Tracy orchestration ─────────────────────────────────────────────────


def run_tracy_profile(case_name: str) -> Path:
    case_dir = OUTPUT_ROOT / case_name
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
    cmd.extend([str(THIS_FILE), "--child-run", "--case", case_name])

    print(f"\n{'='*60}")
    print(f"  Profiling {case_name} ...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {case_name}")
        print(f"  STDOUT (tail):\n{result.stdout[-3000:]}")
        print(f"  STDERR (tail):\n{result.stderr[-3000:]}")
        raise RuntimeError(f"Tracy profiling failed for {case_name}")

    csv_path = case_dir / ".logs" / "cpp_device_perf_report.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"cpp_device_perf_report.csv not found for {case_name}")
    print(f"  OK: {csv_path}")
    return csv_path


# ── Report parsing ──────────────────────────────────────────────────────


def summarize_numeric(series) -> dict[str, float]:
    import pandas as pd
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    vals = vals[vals > 0]
    if vals.empty:
        return {}
    return {
        "avg_ns": float(vals.mean()),
        "min_ns": float(vals.min()),
        "max_ns": float(vals.max()),
        "std_ns": float(vals.std(ddof=0)),
    }


def summarize_cycles(series) -> dict[str, float]:
    import pandas as pd
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    vals = vals[vals >= 0]
    if vals.empty:
        return {}
    return {
        "avg_cycles": float(vals.mean()),
        "min_cycles": float(vals.min()),
        "max_cycles": float(vals.max()),
        "std_cycles": float(vals.std(ddof=0)),
    }


def parse_custom_breakdown(
    raw_df, measured_run_ids: list[int], proc_type: str, markers: list[str],
) -> dict[str, Any]:
    df = raw_df[
        (raw_df["RISC processor type"] == proc_type)
        & (raw_df["run host ID"].isin(measured_run_ids))
        & (raw_df["zone name"].isin(markers))
    ].copy()
    if df.empty:
        return {}

    per_core = (
        df.groupby(["run host ID", "core_x", "core_y", "zone name"], as_index=False)["data"]
        .sum().rename(columns={"data": "cycles"})
    )
    if per_core.empty:
        return {}

    stage_stats = {}
    for zone, group in per_core.groupby("zone name"):
        stage_stats[zone] = summarize_cycles(group["cycles"])

    totals = per_core.groupby(["run host ID", "core_x", "core_y"], as_index=False)["cycles"].sum()
    per_run_max = totals.groupby("run host ID", as_index=False)["cycles"].max()
    per_run_avg = totals.groupby("run host ID", as_index=False)["cycles"].mean()

    avg_sum = sum(stage_stats.get(z, {}).get("avg_cycles", 0.0) for z in markers)
    shares = {}
    if avg_sum > 0:
        for z in markers:
            shares[z] = stage_stats.get(z, {}).get("avg_cycles", 0.0) / avg_sum

    return {
        "stage_stats": stage_stats,
        "avg_profiled_cycles_per_core": float(per_run_avg["cycles"].mean()),
        "max_profiled_cycles_per_run": float(per_run_max["cycles"].mean()),
        "avg_stage_sum_cycles": avg_sum,
        "stage_shares": shares,
    }


def parse_report(case_name: str, csv_path: Path) -> dict[str, Any]:
    import pandas as pd

    w = WORKLOADS[case_name]
    df = pd.read_csv(csv_path)
    relevant = w.warmup_iterations + w.iterations
    if len(df) < relevant:
        raise RuntimeError(f"Expected >= {relevant} rows, got {len(df)}")
    df = df.tail(relevant).reset_index(drop=True)
    df = df.iloc[w.warmup_iterations:].copy()
    run_ids = [int(x) for x in df["GLOBAL CALL COUNT"].tolist()]

    col_stats = {}
    for col in CSV_FLOAT_COLUMNS:
        if col in df.columns:
            s = summarize_numeric(df[col])
            if s:
                col_stats[col] = s

    raw_log = csv_path.parent / "profile_log_device.csv"
    reader_bd = {}
    reader_src_bd = {}
    writer_bd = {}
    if raw_log.exists():
        raw_df = pd.read_csv(raw_log, skiprows=1)
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        raw_df["run host ID"] = pd.to_numeric(raw_df["run host ID"], errors="coerce").fillna(-1).astype(int)
        raw_df["data"] = pd.to_numeric(raw_df["data"], errors="coerce").fillna(0.0)
        reader_bd = parse_custom_breakdown(raw_df, run_ids, "NCRISC", READER_STAGE_MARKERS)
        reader_src_bd = parse_custom_breakdown(raw_df, run_ids, "NCRISC", READER_SOURCE_MARKERS)
        writer_bd = parse_custom_breakdown(raw_df, run_ids, "BRISC", WRITER_STAGE_MARKERS)

    return {
        "case": case_name,
        "workload": asdict(w),
        "measured_run_ids": run_ids,
        "column_stats": col_stats,
        "reader_breakdown": reader_bd,
        "reader_source_breakdown": reader_src_bd,
        "writer_breakdown": writer_bd,
    }


# ── Derived analysis ────────────────────────────────────────────────────


def analyze_profile(case_name: str, p: dict) -> dict[str, Any]:
    w = WORKLOADS[case_name]
    cs = p["column_stats"]
    rb = p["reader_breakdown"]
    rsb = p["reader_source_breakdown"]
    wb = p["writer_breakdown"]

    kernel_ns = cs.get("DEVICE KERNEL DURATION [ns]", {}).get("avg_ns")
    ncrisc_ns = cs.get("DEVICE NCRISC KERNEL DURATION [ns]", {}).get("avg_ns")
    brisc_ns = cs.get("DEVICE BRISC KERNEL DURATION [ns]", {}).get("avg_ns")
    trisc0_ns = cs.get("DEVICE TRISC0 KERNEL DURATION [ns]", {}).get("avg_ns")
    trisc1_ns = cs.get("DEVICE TRISC1 KERNEL DURATION [ns]", {}).get("avg_ns")
    trisc2_ns = cs.get("DEVICE TRISC2 KERNEL DURATION [ns]", {}).get("avg_ns")
    wait_front_ns = cs.get("DEVICE COMPUTE CB WAIT FRONT [ns]", {}).get("avg_ns")
    reserve_back_ns = cs.get("DEVICE COMPUTE CB RESERVE BACK [ns]", {}).get("avg_ns")
    pm_fpu = cs.get("PM FPU UTIL (%)", {}).get("avg_ns")
    pm_dram = cs.get("DRAM BW UTIL (%)", {}).get("avg_ns")
    pm_noc = cs.get("NOC UTIL (%)", {}).get("avg_ns")
    pm_mcast = cs.get("MULTICAST NOC UTIL (%)", {}).get("avg_ns")
    pm_ideal_ns = cs.get("PM IDEAL [ns]", {}).get("avg_ns")
    pm_compute_ns = cs.get("PM COMPUTE [ns]", {}).get("avg_ns")
    pm_bw_ns = cs.get("PM BANDWIDTH [ns]", {}).get("avg_ns")

    kernel_us = ns_to_us(kernel_ns) or 0
    kernel_ms = ns_to_ms(kernel_ns) or 0
    compute_ns = max(filter(None, [trisc0_ns, trisc1_ns, trisc2_ns]), default=0)
    compute_us = ns_to_us(compute_ns) or 0

    n_hb = math.ceil(H / DQHPC)  # 4
    n_sc = min(MAX_CORES_PER_HEAD_BATCH, math.ceil(w.seq_len / BLOCK_SIZE))  # 4

    # DRAM traffic
    k_cache_bytes = w.seq_len * D_QK * ELEMENT_BYTES
    total_k_dram = w.batch * n_hb * k_cache_bytes
    ideal_k_dram = w.batch * k_cache_bytes
    v_cache_bytes = w.seq_len * D_C * ELEMENT_BYTES
    total_v_dram = w.batch * n_hb * v_cache_bytes
    ideal_v_dram = w.batch * v_cache_bytes
    total_dram = total_k_dram + total_v_dram
    ideal_dram = ideal_k_dram + ideal_v_dram

    dram_floor_ideal_ms = ideal_dram / PEAK_DRAM_BW_GBS / 1e6
    gap = kernel_ms / dram_floor_ideal_ms if dram_floor_ideal_ms > 0 else 0

    # Throughput
    flops = 2 * w.seq_len * (2 * D_C + D_R) * H * w.batch
    achieved_tflops = flops / kernel_ns / 1000.0 if kernel_ns else 0

    # Effective BW per core
    eff_k_bw = w.k_bytes_single_pass / ncrisc_ns if ncrisc_ns else 0

    # Reader sub-stage (in us, from cycle counts)
    def rd_us(marker):
        return cycles_to_us(rb.get("stage_stats", {}).get(marker, {}).get("avg_cycles", 0))

    def rs_us(marker):
        return cycles_to_us(rsb.get("stage_stats", {}).get(marker, {}).get("avg_cycles", 0))

    def wr_us(marker):
        return cycles_to_us(wb.get("stage_stats", {}).get(marker, {}).get("avg_cycles", 0))

    reader_reserve_us = rd_us("SDPA-PAGED-RESERVE-SUM")
    reader_issue_us = rd_us("SDPA-PAGED-ISSUE-SUM")
    reader_wait_us = rd_us("SDPA-PAGED-WAIT-SUM")
    reader_push_us = rd_us("SDPA-PAGED-PUSH-SUM")
    reader_ptable_us = rd_us("SDPA-PAGE-TABLE-SUM")

    k_reserve_us = rs_us("SDPA-K-RESERVE-SUM")
    k_issue_us = rs_us("SDPA-K-ISSUE-SUM")
    k_wait_us = rs_us("SDPA-K-WAIT-SUM")
    k_push_us = rs_us("SDPA-K-PUSH-SUM")
    v_reserve_us = rs_us("SDPA-V-RESERVE-SUM")
    v_issue_us = rs_us("SDPA-V-ISSUE-SUM")
    v_wait_us = rs_us("SDPA-V-WAIT-SUM")
    v_push_us = rs_us("SDPA-V-PUSH-SUM")

    writer_cbwait_us = wr_us("SDPA-WRITER-CB-WAIT-SUM")
    writer_issue_us = wr_us("SDPA-WRITER-ISSUE-SUM")
    writer_barrier_us = wr_us("SDPA-WRITER-BARRIER-SUM")
    writer_pop_us = wr_us("SDPA-WRITER-POP-SUM")

    # FPU active time estimate = kernel * PM FPU util
    fpu_active_us = kernel_us * (pm_fpu / 100.0) if pm_fpu else 0

    # Stall attribution
    k_stall_us = reader_reserve_us
    v_stall_us = 0.0
    if k_reserve_us > 0 or v_reserve_us > 0:
        k_stall_us = k_reserve_us
        v_stall_us = v_reserve_us
    k_issue_total_us = reader_issue_us
    v_issue_total_us = 0.0
    if k_issue_us > 0 or v_issue_us > 0:
        k_issue_total_us = k_issue_us
        v_issue_total_us = v_issue_us

    return {
        "case": case_name,
        "seq_len": w.seq_len,
        "batch": w.batch,
        "n_head_batches": n_hb,
        "n_seq_cores": n_sc,
        "total_worker_cores": n_hb * n_sc * w.batch,

        # Timing (us)
        "kernel_us": round(kernel_us, 1),
        "kernel_ms": round(kernel_ms, 4),
        "ncrisc_us": round(ns_to_us(ncrisc_ns) or 0, 1),
        "brisc_us": round(ns_to_us(brisc_ns) or 0, 1),
        "trisc0_us": round(ns_to_us(trisc0_ns) or 0, 1),
        "trisc1_us": round(ns_to_us(trisc1_ns) or 0, 1),
        "trisc2_us": round(ns_to_us(trisc2_ns) or 0, 1),
        "compute_us": round(compute_us, 1),
        "wait_front_us": round(ns_to_us(wait_front_ns) or 0, 1),
        "reserve_back_us": round(ns_to_us(reserve_back_ns) or 0, 1),

        # Pipeline shares (%)
        "ncrisc_share_pct": round((ncrisc_ns / kernel_ns * 100) if kernel_ns else 0, 1),
        "brisc_share_pct": round((brisc_ns / kernel_ns * 100) if kernel_ns else 0, 1),
        "compute_share_pct": round((compute_ns / kernel_ns * 100) if kernel_ns else 0, 1),

        # Reader sub-stages (us per core avg)
        "reader_ptable_us": round(reader_ptable_us, 1),
        "reader_reserve_us": round(reader_reserve_us, 1),
        "reader_issue_us": round(reader_issue_us, 1),
        "reader_wait_us": round(reader_wait_us, 1),
        "reader_push_us": round(reader_push_us, 1),

        # K/V split (us per core avg)
        "k_reserve_us": round(k_reserve_us, 1),
        "k_issue_us": round(k_issue_us, 1),
        "k_wait_us": round(k_wait_us, 1),
        "k_push_us": round(k_push_us, 1),
        "v_reserve_us": round(v_reserve_us, 1),
        "v_issue_us": round(v_issue_us, 1),
        "v_wait_us": round(v_wait_us, 1),
        "v_push_us": round(v_push_us, 1),

        # Writer sub-stages (us per core avg)
        "writer_cbwait_us": round(writer_cbwait_us, 1),
        "writer_issue_us": round(writer_issue_us, 1),
        "writer_barrier_us": round(writer_barrier_us, 1),
        "writer_pop_us": round(writer_pop_us, 1),

        # FPU / compute utilization
        "fpu_active_us": round(fpu_active_us, 1),
        "pm_fpu_util_pct": round(pm_fpu, 2) if pm_fpu else None,
        "pm_dram_bw_util_pct": round(pm_dram, 2) if pm_dram else None,
        "pm_noc_util_pct": round(pm_noc, 2) if pm_noc else None,
        "pm_mcast_noc_util_pct": round(pm_mcast, 2) if pm_mcast else None,
        "pm_ideal_us": round(ns_to_us(pm_ideal_ns) or 0, 1),
        "pm_compute_us": round(ns_to_us(pm_compute_ns) or 0, 1),
        "pm_bw_us": round(ns_to_us(pm_bw_ns) or 0, 1),

        # Derived: DRAM traffic
        "total_dram_read_MB": round(total_dram / 1e6, 2),
        "ideal_dram_read_MB": round(ideal_dram / 1e6, 2),
        "dram_waste_pct": 75.0,
        "dram_floor_ideal_ms": round(dram_floor_ideal_ms, 4),
        "kernel_vs_ideal_gap": round(gap, 1),

        # Derived: throughput
        "total_flops_G": round(flops / 1e9, 3),
        "achieved_tflops": round(achieved_tflops, 4),
        "peak_util_pct": round(achieved_tflops / PEAK_FPU_TFLOPS * 100, 2),
        "eff_k_bw_gbs": round(eff_k_bw, 2),
        "eff_k_bw_vs_peak_pct": round(eff_k_bw / PEAK_DRAM_BW_GBS * 100, 1) if eff_k_bw else 0,

        # Stall summary for Table 2
        "k_stall_us": round(k_stall_us, 1),
        "v_stall_us": round(v_stall_us, 1),
        "k_issue_us_for_table": round(k_issue_total_us, 1),
        "v_issue_us_for_table": round(v_issue_total_us, 1),
    }


# ── Output generation ───────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  CHARACTERIZATION PROFILING RESULTS")
    print("=" * 100)

    print("\n--- Table 1: Baseline FlashMLA decode on Wormhole ---")
    print(f"{'Case':<12} {'Kernel(ms)':>11} {'Ideal(ms)':>10} {'Gap':>6} {'Thpt(TF/s)':>11} {'Eff BW(GB/s)':>13}")
    for r in results:
        print(f"L={r['seq_len']//1000}K{'':<5} {r['kernel_ms']:>11.4f} {r['dram_floor_ideal_ms']:>10.4f} "
              f"{r['kernel_vs_ideal_gap']:>5.1f}x {r['achieved_tflops']:>11.4f} {r['eff_k_bw_gbs']:>13.1f}")

    print("\n--- Table 2: Per-core pipeline time breakdown (us) ---")
    print(f"{'Metric (us)':<18} ", end="")
    for r in results:
        print(f"{'L='+str(r['seq_len']//1000)+'K':>10}", end="")
    print()
    for metric, key in [
        ("Kernel", "kernel_us"),
        ("FPU active", "fpu_active_us"),
        ("K issue", "k_issue_us_for_table"),
        ("K stall(reserve)", "k_stall_us"),
        ("V issue", "v_issue_us_for_table"),
        ("V stall(reserve)", "v_stall_us"),
        ("Writer cb_wait", "writer_cbwait_us"),
        ("Writer issue", "writer_issue_us"),
    ]:
        print(f"{metric:<18} ", end="")
        for r in results:
            print(f"{r[key]:>10.1f}", end="")
        print()

    print("\n--- Table 3: Utilization & Pipeline Shares ---")
    print(f"{'Metric':<25} ", end="")
    for r in results:
        print(f"{'L='+str(r['seq_len']//1000)+'K':>10}", end="")
    print()
    for metric, key in [
        ("PM FPU util (%)", "pm_fpu_util_pct"),
        ("PM DRAM BW util (%)", "pm_dram_bw_util_pct"),
        ("PM NOC util (%)", "pm_noc_util_pct"),
        ("Achieved TFLOP/s", "achieved_tflops"),
        ("Eff K BW/core (GB/s)", "eff_k_bw_gbs"),
        ("Eff BW / peak (%)", "eff_k_bw_vs_peak_pct"),
        ("NCRISC share (%)", "ncrisc_share_pct"),
        ("BRISC share (%)", "brisc_share_pct"),
        ("Compute share (%)", "compute_share_pct"),
    ]:
        print(f"{metric:<25} ", end="")
        for r in results:
            v = r[key]
            if v is None:
                print(f"{'n/a':>10}", end="")
            else:
                print(f"{v:>10.2f}", end="")
        print()

    print("\n--- Reader sub-stages (us per core avg) ---")
    for metric, key in [
        ("page_table", "reader_ptable_us"),
        ("reserve", "reader_reserve_us"),
        ("issue", "reader_issue_us"),
        ("wait", "reader_wait_us"),
        ("push", "reader_push_us"),
    ]:
        print(f"{metric:<18} ", end="")
        for r in results:
            print(f"{r[key]:>10.1f}", end="")
        print()

    print("\n--- Writer sub-stages (us per core avg) ---")
    for metric, key in [
        ("cb_wait", "writer_cbwait_us"),
        ("issue", "writer_issue_us"),
        ("barrier", "writer_barrier_us"),
        ("pop", "writer_pop_us"),
    ]:
        print(f"{metric:<18} ", end="")
        for r in results:
            print(f"{r[key]:>10.1f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--case", choices=sorted(WORKLOADS.keys()))
    parser.add_argument("--cases", nargs="+", choices=sorted(WORKLOADS.keys()),
                        default=["decode_1k", "decode_4k", "decode_8k", "decode_16k", "decode_32k"])
    parser.add_argument("--parse-only", action="store_true",
                        help="Re-analyze existing profile data without re-running.")
    args = parser.parse_args()

    if args.child_run:
        execute_child_run(args.case)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    profiles = []
    for cn in args.cases:
        if args.parse_only:
            csv_path = OUTPUT_ROOT / cn / ".logs" / "cpp_device_perf_report.csv"
            if not csv_path.exists():
                print(f"SKIP {cn}: no existing data")
                continue
        else:
            csv_path = run_tracy_profile(cn)
        profiles.append(parse_report(cn, csv_path))

    analyzed = [analyze_profile(p["case"], p) for p in profiles]
    print_summary(analyzed)

    out_json = OUTPUT_ROOT / "characterization_profile_results.json"
    with open(out_json, "w") as f:
        json.dump({"profiles": profiles, "analyzed": analyzed}, f, indent=2)
    print(f"\nFull results: {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

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

import numpy as np
import pandas as pd
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import nearest_y
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import page_table_setup, to_paged_cache


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "mla_flash_attention_dev" / "experiments" / "profiling" / "outputs" / "flash_mla_wh"
THIS_FILE = Path(__file__).resolve()
TARGET_OP_CODE = "SdpaDecodeDeviceOperation"

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

CUSTOM_READER_STAGE_MARKERS = [
    "SDPA-PAGE-TABLE-SUM",
    "SDPA-PAGED-RESERVE-SUM",
    "SDPA-PAGED-ISSUE-SUM",
    "SDPA-PAGED-WAIT-SUM",
    "SDPA-PAGED-PUSH-SUM",
]

A_BH_ACTIVE_CORE_SWEEP = [16, 20, 24, 32, 48, 64]


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    seq_len: int
    num_heads: int
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    block_size: int = 64
    iterations: int = 6
    warmup_iterations: int = 2

    @property
    def d_qk(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def cur_pos(self) -> int:
        return self.seq_len - 1

    @property
    def k_bytes_single_pass(self) -> int:
        return self.batch * self.seq_len * self.d_qk

    @property
    def q_bytes(self) -> int:
        return self.batch * self.num_heads * self.d_qk * 2

    @property
    def out_bytes(self) -> int:
        return self.batch * self.num_heads * self.kv_lora_rank * 2


WORKLOADS = {
    "decode_1k": Workload(name="decode_1k", batch=2, seq_len=1024, num_heads=32),
    "decode_4k": Workload(name="decode_4k", batch=2, seq_len=4096, num_heads=32),
    "decode_32k": Workload(name="decode_32k", batch=1, seq_len=32768, num_heads=32),
}


def ns_to_us(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1000.0


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def latest_cpp_device_report(case_output_dir: Path) -> Path:
    csv_path = case_output_dir / ".logs" / "cpp_device_perf_report.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"C++ device perf report not found: {csv_path}")
    return csv_path


def latest_raw_device_log(case_output_dir: Path) -> Path:
    csv_path = case_output_dir / ".logs" / "profile_log_device.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw device profiler log not found: {csv_path}")
    return csv_path


def run_tracy_profile(case_name: str) -> Path:
    case_output_dir = OUTPUT_ROOT / case_name
    if case_output_dir.exists():
        shutil.rmtree(case_output_dir)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TTNN_OP_PROFILER"] = "1"
    env["TT_METAL_PROFILER_TRACE_TRACKING"] = "1"

    cmd = [
        "python3",
        "-m",
        "tracy",
        "-p",
        "-o",
        str(case_output_dir),
        "--check-exit-code",
        "--op-support-count",
        "2000",
        "-t",
        "5000",
    ]
    for analysis in DEVICE_ANALYSIS_TYPES:
        cmd.extend(["-a", analysis])
    cmd.extend([str(THIS_FILE), "--child-run", "--case", case_name])

    logger.info(f"Running tracy profile for {case_name}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Tracy profiling failed for {case_name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return latest_cpp_device_report(case_output_dir)


def build_decode_inputs(device: Any, workload: Workload):
    d_qk = workload.d_qk
    q = torch.randn((1, workload.batch, workload.num_heads, d_qk), dtype=torch.bfloat16)
    k = torch.randn((workload.batch, 1, workload.seq_len, d_qk), dtype=torch.bfloat16)

    max_num_blocks = (workload.seq_len // workload.block_size) * workload.batch
    paged_cfg = PagedAttentionConfig(block_size=workload.block_size, max_num_blocks=max_num_blocks)
    page_table = page_table_setup(workload.batch, paged_cfg)
    paged_cache_torch = to_paged_cache(k, page_table, paged_cfg)

    grid_size = device.compute_with_storage_grid_size()
    q_num_cores = min(workload.batch * workload.num_heads, grid_size.x * grid_size.y)
    block_height = nearest_y((workload.batch * workload.num_heads) // q_num_cores, ttnn.TILE_SIZE)
    q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)

    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, d_qk),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, workload.kv_lora_rank),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    start_indices = torch.full((workload.batch,), workload.cur_pos, dtype=torch.int32)

    tt_q = ttnn.from_torch(
        q,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
    )
    tt_k = ttnn.from_torch(
        paged_cache_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_start_indices = ttnn.from_torch(start_indices, device=device, dtype=ttnn.int32)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=4,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    return {
        "tt_q": tt_q,
        "tt_k": tt_k,
        "tt_page_table": tt_page_table,
        "tt_start_indices": tt_start_indices,
        "program_config": sdpa_program_config,
        "compute_config": compute_config,
        "out_mem_config": out_mem_config,
    }


def execute_child_run(case_name: str) -> None:
    from tracy import signpost

    workload = WORKLOADS[case_name]
    device = ttnn.open_device(device_id=0)
    logger.info(f"Opened device for child run: case={case_name}, arch={device.arch()}")

    try:
        tensors = build_decode_inputs(device, workload)
        scale = workload.d_qk**-0.5
        tt_out = None

        for _ in range(workload.warmup_iterations):
            tt_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tensors["tt_q"],
                tensors["tt_k"],
                page_table_tensor=tensors["tt_page_table"],
                cur_pos_tensor=tensors["tt_start_indices"],
                head_dim_v=workload.kv_lora_rank,
                scale=scale,
                program_config=tensors["program_config"],
                compute_kernel_config=tensors["compute_config"],
                memory_config=tensors["out_mem_config"],
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)

        signpost("start")
        for _ in range(workload.iterations):
            tt_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tensors["tt_q"],
                tensors["tt_k"],
                page_table_tensor=tensors["tt_page_table"],
                cur_pos_tensor=tensors["tt_start_indices"],
                head_dim_v=workload.kv_lora_rank,
                scale=scale,
                program_config=tensors["program_config"],
                compute_kernel_config=tensors["compute_config"],
                memory_config=tensors["out_mem_config"],
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)
        signpost("stop")

        for key in ("tt_q", "tt_k", "tt_page_table", "tt_start_indices"):
            ttnn.deallocate(tensors[key])
    finally:
        ttnn.close_device(device)


def summarize_numeric_column(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    values = values[values > 0]
    if values.empty:
        return {}
    return {
        "avg_ns": float(values.mean()),
        "min_ns": float(values.min()),
        "max_ns": float(values.max()),
        "std_ns": float(values.std(ddof=0)),
    }


def summarize_cycle_values(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    numeric = numeric[numeric >= 0]
    if numeric.empty:
        return {}
    return {
        "avg_cycles": float(numeric.mean()),
        "min_cycles": float(numeric.min()),
        "max_cycles": float(numeric.max()),
        "std_cycles": float(numeric.std(ddof=0)),
    }


def derive_bottleneck(workload: Workload, column_stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    kernel_ns = column_stats.get("DEVICE KERNEL DURATION [ns]", {}).get("avg_ns")
    ncrisc_ns = column_stats.get("DEVICE NCRISC KERNEL DURATION [ns]", {}).get("avg_ns")
    brisc_ns = column_stats.get("DEVICE BRISC KERNEL DURATION [ns]", {}).get("avg_ns")
    trisc1_ns = column_stats.get("DEVICE TRISC1 KERNEL DURATION [ns]", {}).get("avg_ns")
    wait_front_ns = column_stats.get("DEVICE COMPUTE CB WAIT FRONT [ns]", {}).get("avg_ns")
    dram_util = column_stats.get("DRAM BW UTIL (%)", {}).get("avg_ns")
    noc_util = column_stats.get("NOC UTIL (%)", {}).get("avg_ns")
    multicast_util = column_stats.get("MULTICAST NOC UTIL (%)", {}).get("avg_ns")

    ratios = {}
    if kernel_ns and kernel_ns > 0:
        for key, value in {
            "ncrisc_share": ncrisc_ns,
            "brisc_share": brisc_ns,
            "trisc1_share": trisc1_ns,
            "compute_wait_front_share": wait_front_ns,
        }.items():
            ratios[key] = (value / kernel_ns) if value is not None else None

    n_share = ratios.get("ncrisc_share")
    b_share = ratios.get("brisc_share")
    t_share = ratios.get("trisc1_share")

    classification = "unknown"
    reasons = []
    if n_share is not None and b_share is not None and n_share > 0.90 and b_share > 0.95:
        classification = "reader_noc_saturated"
        reasons.append("ncrisc and brisc almost fully occupy the kernel window")
    elif n_share is not None and b_share is not None and n_share > 0.55 and b_share > 0.90:
        classification = "balanced_with_reader_noc_pressure"
        reasons.append("reader and writer are both close to the critical path")
    elif wait_front_ns and kernel_ns and wait_front_ns > 0.10 * kernel_ns:
        classification = "compute_starved_by_input"
        reasons.append("compute wait_front is significant")
    elif t_share is not None and t_share > 0.95:
        classification = "compute_on_critical_path"
        reasons.append("trisc1 is still the last stage to retire")

    effective_reader_gbps = None
    if ncrisc_ns and ncrisc_ns > 0:
        effective_reader_gbps = workload.k_bytes_single_pass / ncrisc_ns

    return {
        "classification": classification,
        "reasons": reasons,
        "ratios": ratios,
        "effective_single_pass_k_read_gbps": effective_reader_gbps,
        "effective_four_lane_k_read_gbps": (effective_reader_gbps * 4.0) if effective_reader_gbps is not None else None,
        "pm_dram_bw_util_pct": dram_util,
        "pm_noc_util_pct": noc_util,
        "pm_multicast_noc_util_pct": multicast_util,
    }


def parse_custom_reader_breakdown(case_name: str, measured_run_ids: list[int]) -> dict[str, Any]:
    raw_log_path = latest_raw_device_log(OUTPUT_ROOT / case_name)
    df = pd.read_csv(raw_log_path, skiprows=1)
    df.columns = [str(col).strip() for col in df.columns]
    df["run host ID"] = pd.to_numeric(df["run host ID"], errors="coerce").fillna(-1).astype(int)

    df = df[
        (df["RISC processor type"] == "NCRISC")
        & (df["run host ID"].isin(measured_run_ids))
        & (df["zone name"].isin(CUSTOM_READER_STAGE_MARKERS))
    ].copy()
    if df.empty:
        return {}

    df["data"] = pd.to_numeric(df["data"], errors="coerce").fillna(0.0)

    per_core = (
        df.groupby(["run host ID", "core_x", "core_y", "zone name"], as_index=False)["data"]
        .sum()
        .rename(columns={"data": "cycles"})
    )
    if per_core.empty:
        return {}

    stage_stats = {}
    for zone_name, group in per_core.groupby("zone name"):
        stage_stats[zone_name] = summarize_cycle_values(group["cycles"])

    totals = per_core.groupby(["run host ID", "core_x", "core_y"], as_index=False)["cycles"].sum()
    per_run_max_total = totals.groupby("run host ID", as_index=False)["cycles"].max()
    per_run_avg_total = totals.groupby("run host ID", as_index=False)["cycles"].mean()

    avg_stage_total = sum(stage_stats.get(zone, {}).get("avg_cycles", 0.0) for zone in CUSTOM_READER_STAGE_MARKERS)
    stage_shares = {}
    if avg_stage_total > 0:
        for zone in CUSTOM_READER_STAGE_MARKERS:
            zone_avg = stage_stats.get(zone, {}).get("avg_cycles", 0.0)
            stage_shares[zone] = zone_avg / avg_stage_total

    return {
        "raw_log_path": str(raw_log_path),
        "measured_run_ids": measured_run_ids,
        "stage_stats": stage_stats,
        "avg_profiled_cycles_per_core": float(per_run_avg_total["cycles"].mean()),
        "max_profiled_cycles_per_run": float(per_run_max_total["cycles"].mean()),
        "avg_profiled_cycles_per_core_stage_sum": avg_stage_total,
        "stage_shares": stage_shares,
    }


def parse_report(case_name: str, csv_path: Path) -> dict[str, Any]:
    workload = WORKLOADS[case_name]
    df = pd.read_csv(csv_path)
    relevant_rows = workload.warmup_iterations + workload.iterations
    if len(df) < relevant_rows:
        raise RuntimeError(
            f"Expected at least {relevant_rows} rows in {csv_path}, only found {len(df)}. "
            "This likely means unexpected extra profiling noise or missing op rows."
        )
    df = df.tail(relevant_rows).reset_index(drop=True)
    df = df.iloc[workload.warmup_iterations :].copy()
    measured_run_ids = [int(x) for x in df["GLOBAL CALL COUNT"].tolist()]

    column_stats = {}
    for column in CSV_FLOAT_COLUMNS:
        if column in df.columns:
            stats = summarize_numeric_column(df[column])
            if stats:
                column_stats[column] = stats

    analysis = derive_bottleneck(workload, column_stats)
    return {
        "case": case_name,
        "workload": asdict(workload),
        "rows_profiled": int(len(df)),
        "measured_run_ids": measured_run_ids,
        "csv_path": str(csv_path),
        "column_stats": column_stats,
        "analysis": analysis,
        "custom_reader_breakdown": parse_custom_reader_breakdown(case_name, measured_run_ids),
    }


def simulate_blackhole_native_b(workloads: list[Workload]) -> dict[str, Any]:
    clock_ghz = 1.35
    dram_bw_gbps = 512.0
    noc_bw_gbps = 86.4
    active_cores = 64
    flops_per_core_cycle = 4096
    q_remote_blocks = 7
    k_hops = 2.25
    q_hops = 5.0
    reduction_steps = 3
    reduction_hops = 4.0
    num_s_blocks = 8
    heads_per_core = 8

    cases = []
    for workload in workloads:
        h_local = 64
        d_qk = workload.d_qk
        d_v = workload.kv_lora_rank
        total_flops = 2 * h_local * workload.seq_len * (d_qk + d_v)
        t_compute_ms = total_flops / (active_cores * flops_per_core_cycle * clock_ghz * 1e9) * 1e3

        k_bytes = workload.seq_len * d_qk
        t_dram_ms = k_bytes / (dram_bw_gbps * 1e9) * 1e3

        q_lane_bytes = heads_per_core * d_qk * 2
        t_q_ms = (q_lane_bytes * q_remote_blocks * q_hops) / (noc_bw_gbps * 1e9) * 1e3

        sender_bytes = k_bytes / num_s_blocks
        t_k_mcast_ms = (sender_bytes * k_hops) / (noc_bw_gbps * 1e9) * 1e3

        o_lane_bytes = heads_per_core * d_v * 2
        t_reduce_ms = (reduction_steps * o_lane_bytes * reduction_hops) / (noc_bw_gbps * 1e9) * 1e3

        critical = {
            "dram": t_dram_ms,
            "k_mcast": t_k_mcast_ms,
            "compute": t_compute_ms,
        }
        dominant_stage = max(critical, key=critical.get)
        total_ms = t_q_ms + critical[dominant_stage] + t_reduce_ms

        cases.append(
            {
                "case": workload.name,
                "seq_len": workload.seq_len,
                "k_dram_mb": k_bytes / (1024 * 1024),
                "t_dram_ms": t_dram_ms,
                "t_q_ms": t_q_ms,
                "t_k_mcast_ms": t_k_mcast_ms,
                "t_compute_ms": t_compute_ms,
                "t_reduce_ms": t_reduce_ms,
                "predicted_core_ms": total_ms,
                "dominant_stage": dominant_stage,
            }
        )

    return {
        "model": "native_flash_mla_b",
        "assumptions": {
            "arch": "blackhole",
            "clock_ghz": clock_ghz,
            "dram_bw_gbps": dram_bw_gbps,
            "noc_bw_gbps": noc_bw_gbps,
            "active_cores": active_cores,
            "flops_per_core_cycle": flops_per_core_cycle,
            "num_s_blocks": num_s_blocks,
            "heads_per_core": heads_per_core,
        },
        "cases": cases,
    }


def simulate_blackhole_current_a(workloads: list[Workload]) -> dict[str, Any]:
    clock_ghz = 1.35
    dram_bw_gbps = 512.0
    active_cores = 16
    phi_a_core = 1024
    local_heads = 64
    reduction_ms = 0.0011

    cases = []
    for workload in workloads:
        d_qk = workload.d_qk
        d_v = workload.kv_lora_rank
        k_dram_bytes = workload.seq_len * d_qk * 4
        t_dram_ms = k_dram_bytes / (dram_bw_gbps * 1e9) * 1e3

        total_flops = 2 * local_heads * workload.seq_len * (d_qk + d_v)
        t_compute_ms = total_flops / (active_cores * phi_a_core * clock_ghz * 1e9) * 1e3
        dominant_stage = "compute" if t_compute_ms >= t_dram_ms else "dram"
        total_ms = max(t_dram_ms, t_compute_ms) + reduction_ms

        cases.append(
            {
                "case": workload.name,
                "seq_len": workload.seq_len,
                "k_dram_mb": k_dram_bytes / (1024 * 1024),
                "t_dram_ms": t_dram_ms,
                "t_compute_ms": t_compute_ms,
                "t_reduce_ms": reduction_ms,
                "predicted_core_ms": total_ms,
                "dominant_stage": dominant_stage,
            }
        )

    return {
        "model": "current_sdpa_decode_a",
        "assumptions": {
            "arch": "blackhole",
            "clock_ghz": clock_ghz,
            "dram_bw_gbps": dram_bw_gbps,
            "active_cores": active_cores,
            "phi_a_core_flop_per_cycle": phi_a_core,
            "local_heads": local_heads,
            "reduction_ms": reduction_ms,
        },
        "cases": cases,
    }


def simulate_blackhole_current_a_active_core_sweep(workloads: list[Workload], active_cores_list: list[int]) -> dict[str, Any]:
    clock_ghz = 1.35
    dram_bw_gbps = 512.0
    phi_a_core = 1024
    local_heads = 64
    reduction_ms = 0.0011

    cases = []
    for workload in workloads:
        d_qk = workload.d_qk
        d_v = workload.kv_lora_rank
        k_dram_bytes = workload.seq_len * d_qk * 4
        t_dram_ms = k_dram_bytes / (dram_bw_gbps * 1e9) * 1e3
        total_flops = 2 * local_heads * workload.seq_len * (d_qk + d_v)
        crossover_active_cores = math.ceil(total_flops / (phi_a_core * clock_ghz * 1e9 * (t_dram_ms / 1e3)))

        sweep = []
        for active_cores in active_cores_list:
            t_compute_ms = total_flops / (active_cores * phi_a_core * clock_ghz * 1e9) * 1e3
            dominant_stage = "compute" if t_compute_ms >= t_dram_ms else "dram"
            predicted_core_ms = max(t_dram_ms, t_compute_ms) + reduction_ms
            sweep.append(
                {
                    "active_cores": active_cores,
                    "t_dram_ms": t_dram_ms,
                    "t_compute_ms": t_compute_ms,
                    "t_reduce_ms": reduction_ms,
                    "predicted_core_ms": predicted_core_ms,
                    "dominant_stage": dominant_stage,
                }
            )

        cases.append(
            {
                "case": workload.name,
                "seq_len": workload.seq_len,
                "crossover_active_cores": crossover_active_cores,
                "sweep": sweep,
            }
        )

    return {
        "model": "current_sdpa_decode_a_active_core_sweep",
        "assumptions": {
            "arch": "blackhole",
            "clock_ghz": clock_ghz,
            "dram_bw_gbps": dram_bw_gbps,
            "phi_a_core_flop_per_cycle": phi_a_core,
            "local_heads": local_heads,
            "reduction_ms": reduction_ms,
        },
        "cases": cases,
    }


def write_summary(results: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "flash_mla_wh_profile_results.json"
    md_path = OUTPUT_ROOT / "flash_mla_wh_profile_report.md"

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# FlashMLA WH profile summary",
        "",
        "## WH measured profile",
        "",
        "| case | kernel us | ncrisc us | brisc us | trisc1 us | wait_front us | bottleneck | est reader GB/s |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for entry in results["wh_profile"]:
        stats = entry["column_stats"]
        analysis = entry["analysis"]
        lines.append(
            "| {case} | {kernel:.3f} | {ncrisc:.3f} | {brisc:.3f} | {trisc1:.3f} | {wait:.3f} | {cls} | {bw:.3f} |".format(
                case=entry["case"],
                kernel=ns_to_us(stats.get("DEVICE KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
                ncrisc=ns_to_us(stats.get("DEVICE NCRISC KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
                brisc=ns_to_us(stats.get("DEVICE BRISC KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
                trisc1=ns_to_us(stats.get("DEVICE TRISC1 KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
                wait=ns_to_us(stats.get("DEVICE COMPUTE CB WAIT FRONT [ns]", {}).get("avg_ns")) or 0.0,
                cls=analysis["classification"],
                bw=analysis.get("effective_single_pass_k_read_gbps") or 0.0,
            )
        )

    lines.extend(
        [
            "",
            "## WH custom NCRISC stage breakdown",
            "",
            "| case | ncrisc us | page_table us/core(avg) | reserve/block us/core(avg) | issue us/core(avg) | wait us/core(avg) | push us/core(avg) | profiled total us/core(max) | reserve share | issue share | wait share | push share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in results["wh_profile"]:
        breakdown = entry.get("custom_reader_breakdown", {})
        stages = breakdown.get("stage_stats", {})
        shares = breakdown.get("stage_shares", {})
        lines.append(
            "| {case} | {ncrisc:.3f} | {page_table:.3f} | {reserve:.3f} | {issue:.3f} | {wait:.3f} | {push:.3f} | {max_total:.3f} | {reserve_share:.1%} | {issue_share:.1%} | {wait_share:.1%} | {push_share:.1%} |".format(
                case=entry["case"],
                ncrisc=ns_to_us(entry["column_stats"].get("DEVICE NCRISC KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
                page_table=ns_to_us(stages.get("SDPA-PAGE-TABLE-SUM", {}).get("avg_cycles")) or 0.0,
                reserve=ns_to_us(stages.get("SDPA-PAGED-RESERVE-SUM", {}).get("avg_cycles")) or 0.0,
                issue=ns_to_us(stages.get("SDPA-PAGED-ISSUE-SUM", {}).get("avg_cycles")) or 0.0,
                wait=ns_to_us(stages.get("SDPA-PAGED-WAIT-SUM", {}).get("avg_cycles")) or 0.0,
                push=ns_to_us(stages.get("SDPA-PAGED-PUSH-SUM", {}).get("avg_cycles")) or 0.0,
                max_total=ns_to_us(breakdown.get("max_profiled_cycles_per_run")) or 0.0,
                reserve_share=shares.get("SDPA-PAGED-RESERVE-SUM", 0.0),
                issue_share=shares.get("SDPA-PAGED-ISSUE-SUM", 0.0),
                wait_share=shares.get("SDPA-PAGED-WAIT-SUM", 0.0),
                push_share=shares.get("SDPA-PAGED-PUSH-SUM", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## BH simulated current path (A)",
            "",
            "| case | seq_len | dram ms | compute ms | reduce ms | predicted core ms | dominant stage |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in results["bh_simulation"]["current_a_bh"]["cases"]:
        lines.append(
            "| {case} | {seq_len} | {t_dram_ms:.4f} | {t_compute_ms:.4f} | {t_reduce_ms:.4f} | {predicted_core_ms:.4f} | {dominant_stage} |".format(
                **case
            )
        )

    lines.extend(
        [
            "",
            "## BH current path active-core sweep",
            "",
            "| case | crossover active cores | 16c ms | 20c ms | 24c ms | 32c ms | 48c ms | 64c ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in results["bh_simulation"]["current_a_active_core_sweep"]["cases"]:
        sweep_lookup = {entry["active_cores"]: entry for entry in case["sweep"]}
        lines.append(
            "| {case} | {crossover_active_cores} | {c16:.4f} | {c20:.4f} | {c24:.4f} | {c32:.4f} | {c48:.4f} | {c64:.4f} |".format(
                case=case["case"],
                crossover_active_cores=case["crossover_active_cores"],
                c16=sweep_lookup[16]["predicted_core_ms"],
                c20=sweep_lookup[20]["predicted_core_ms"],
                c24=sweep_lookup[24]["predicted_core_ms"],
                c32=sweep_lookup[32]["predicted_core_ms"],
                c48=sweep_lookup[48]["predicted_core_ms"],
                c64=sweep_lookup[64]["predicted_core_ms"],
            )
        )

    lines.extend(
        [
            "",
            "## BH simulated native FlashMLA (B)",
            "",
            "| case | seq_len | dram ms | q fanout ms | k mcast ms | compute ms | reduce ms | predicted core ms | dominant stage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in results["bh_simulation"]["native_b_bh"]["cases"]:
        lines.append(
            "| {case} | {seq_len} | {t_dram_ms:.4f} | {t_q_ms:.4f} | {t_k_mcast_ms:.4f} | {t_compute_ms:.4f} | {t_reduce_ms:.4f} | {predicted_core_ms:.4f} | {dominant_stage} |".format(
                **case
            )
        )

    with md_path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile current FlashMLA decode on Wormhole and simulate Blackhole.")
    parser.add_argument("--child-run", action="store_true", help="Execute the workload body to be profiled by tracy.")
    parser.add_argument("--case", choices=sorted(WORKLOADS.keys()), default="decode_1k")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(WORKLOADS.keys()),
        default=["decode_1k", "decode_4k", "decode_32k"],
        help="Cases to profile in orchestrator mode.",
    )
    args = parser.parse_args()

    if args.child_run:
        execute_child_run(args.case)
        return

    wh_results = []
    for case_name in args.cases:
        csv_path = run_tracy_profile(case_name)
        wh_results.append(parse_report(case_name, csv_path))

    workloads = [WORKLOADS[case_name] for case_name in args.cases]
    bh_simulation = {
        "current_a_bh": simulate_blackhole_current_a(workloads),
        "current_a_active_core_sweep": simulate_blackhole_current_a_active_core_sweep(
            workloads, A_BH_ACTIVE_CORE_SWEEP
        ),
        "native_b_bh": simulate_blackhole_native_b(workloads),
    }

    results = {
        "wh_profile": wh_results,
        "bh_simulation": bh_simulation,
    }
    write_summary(results)
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

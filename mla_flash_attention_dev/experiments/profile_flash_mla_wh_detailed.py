#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import nearest_y
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import page_table_setup, to_paged_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "mla_flash_attention_dev" / "experiments" / "profile_outputs" / "flash_mla_wh_detailed"
THIS_FILE = Path(__file__).resolve()
WORMHOLE_AICLK_MHZ = 1000.0
WH_DRAM_BW_GBPS = 258.0
BH_DRAM_BW_GBPS = 512.0
WH_NOC_BW_GBPS = 32.0
BH_NOC_BW_GBPS = 86.4
BH_AICLK_MHZ = 1350.0
A_WH_ACTIVE_CORES = 16
A_BH_ACTIVE_CORE_SWEEP = [16, 20, 24, 32, 48, 64]
A_LOCAL_HEADS = 64
A_PHI_A_CORE = 1024
CANONICAL_DECODE_CASES = ["decode_1k", "decode_4k", "decode_32k"]
EMPIRICAL_ACTIVE_CORE_SCAN = list(range(4, 65))
EMPIRICAL_PLATEAU_TOLERANCE = 0.02

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

CUSTOM_WRITER_STAGE_MARKERS = [
    "SDPA-WRITER-CB-WAIT-SUM",
    "SDPA-WRITER-ISSUE-SUM",
    "SDPA-WRITER-BARRIER-SUM",
    "SDPA-WRITER-POP-SUM",
]

CUSTOM_READER_SOURCE_STAGE_MARKERS = [
    "SDPA-K-RESERVE-SUM",
    "SDPA-K-ISSUE-SUM",
    "SDPA-K-WAIT-SUM",
    "SDPA-K-PUSH-SUM",
    "SDPA-V-RESERVE-SUM",
    "SDPA-V-ISSUE-SUM",
    "SDPA-V-WAIT-SUM",
    "SDPA-V-PUSH-SUM",
]

CUSTOM_WRITER_SOURCE_STAGE_MARKERS = [
    "SDPA-WRITER-SENDER-CB-WAIT-SUM",
    "SDPA-WRITER-ROOT-CB-WAIT-SUM",
    "SDPA-WRITER-TREE-CHILD-WAIT-SUM",
    "SDPA-WRITER-OUTPUT-GATHER-WAIT-SUM",
]

READER_STAGE_LABELS = {
    "SDPA-PAGE-TABLE-SUM": "page_table",
    "SDPA-PAGED-RESERVE-SUM": "reserve",
    "SDPA-PAGED-ISSUE-SUM": "issue",
    "SDPA-PAGED-WAIT-SUM": "wait",
    "SDPA-PAGED-PUSH-SUM": "push",
}

WRITER_STAGE_LABELS = {
    "SDPA-WRITER-CB-WAIT-SUM": "cb_wait",
    "SDPA-WRITER-ISSUE-SUM": "issue",
    "SDPA-WRITER-BARRIER-SUM": "barrier",
    "SDPA-WRITER-POP-SUM": "pop",
}

READER_SOURCE_STAGE_LABELS = {
    "SDPA-K-RESERVE-SUM": "k_reserve",
    "SDPA-K-ISSUE-SUM": "k_issue",
    "SDPA-K-WAIT-SUM": "k_wait",
    "SDPA-K-PUSH-SUM": "k_push",
    "SDPA-V-RESERVE-SUM": "v_reserve",
    "SDPA-V-ISSUE-SUM": "v_issue",
    "SDPA-V-WAIT-SUM": "v_wait",
    "SDPA-V-PUSH-SUM": "v_push",
}

WRITER_SOURCE_STAGE_LABELS = {
    "SDPA-WRITER-SENDER-CB-WAIT-SUM": "sender_cb_wait",
    "SDPA-WRITER-ROOT-CB-WAIT-SUM": "root_cb_wait",
    "SDPA-WRITER-TREE-CHILD-WAIT-SUM": "tree_child_wait",
    "SDPA-WRITER-OUTPUT-GATHER-WAIT-SUM": "output_gather_wait",
}

COUPLING_SOURCE_LABELS = {
    "reader_reserve_from_compute_ms": "reserve<-compute",
    "reader_reserve_from_writer_ms": "reserve<-writer",
    "writer_cb_wait_from_compute_ms": "wait<-compute",
    "writer_cb_wait_from_reader_ms": "wait<-reader",
}


@dataclass(frozen=True)
class Workload:
    name: str
    mode: str
    variant: str
    batch: int
    seq_len: int
    num_heads: int = 32
    num_kv_heads: int = 1
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    block_size: int = 64
    k_chunk_size: int = 128
    max_cores_per_head_batch: int = 4
    iterations: int = 6
    warmup_iterations: int = 2

    @property
    def d_qk(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def q_chunk_size(self) -> int:
        return nearest_pow_2(nearest_n(self.num_heads, 32))

    @property
    def cur_pos(self) -> int:
        return self.seq_len - 1

    @property
    def k_bytes_single_pass(self) -> int:
        return self.batch * self.seq_len * self.d_qk

    @property
    def q_bytes(self) -> int:
        q_seq = self.seq_len if self.mode == "prefill" else 1
        return self.batch * self.num_heads * q_seq * self.d_qk * 2

    @property
    def out_bytes(self) -> int:
        q_seq = self.seq_len if self.mode == "prefill" else 1
        return self.batch * self.num_heads * q_seq * self.kv_lora_rank * 2

    @property
    def normalized_items(self) -> int:
        return self.batch * self.seq_len if self.mode == "prefill" else self.batch


def nearest_n(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def nearest_pow_2(x: int) -> int:
    if x < 1:
        raise ValueError("x must be >= 1")
    return 1 << (x - 1).bit_length()


def make_prefill(seq_len: int, *, batch: int = 1, iterations: int, warmup_iterations: int) -> Workload:
    suffix = f"{seq_len // 1024}k" if seq_len >= 1024 else str(seq_len)
    return Workload(
        name=f"prefill_{suffix}",
        mode="prefill",
        variant="chunked_paged",
        batch=batch,
        seq_len=seq_len,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )


def make_decode(seq_len: int, *, batch: int, iterations: int = 6, warmup_iterations: int = 2) -> Workload:
    suffix = f"{seq_len // 1024}k" if seq_len >= 1024 else str(seq_len)
    return Workload(
        name=f"decode_{suffix}",
        mode="decode",
        variant="paged_decode",
        batch=batch,
        seq_len=seq_len,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )


WORKLOADS = {
    "prefill_256": make_prefill(256, iterations=6, warmup_iterations=2),
    "prefill_512": make_prefill(512, iterations=6, warmup_iterations=2),
    "prefill_1k": make_prefill(1024, iterations=6, warmup_iterations=2),
    "prefill_2k": make_prefill(2048, iterations=4, warmup_iterations=2),
    "prefill_4k": make_prefill(4096, iterations=3, warmup_iterations=1),
    "prefill_8k": make_prefill(8192, iterations=2, warmup_iterations=1),
    "prefill_16k": make_prefill(16384, iterations=1, warmup_iterations=1),
    "prefill_32k": make_prefill(32768, iterations=1, warmup_iterations=0),
    "decode_256": make_decode(256, batch=2),
    "decode_512": make_decode(512, batch=2),
    "decode_1k": make_decode(1024, batch=2),
    "decode_2k": make_decode(2048, batch=2),
    "decode_4k": make_decode(4096, batch=2),
    "decode_8k": make_decode(8192, batch=2, iterations=4, warmup_iterations=2),
    "decode_16k": make_decode(16384, batch=1, iterations=4, warmup_iterations=2),
    "decode_32k": make_decode(32768, batch=1, iterations=4, warmup_iterations=2),
}


def ns_to_us(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1000.0


def cycles_to_us(value: float | None) -> float | None:
    if value is None:
        return None
    return value / WORMHOLE_AICLK_MHZ


def ns_to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000.0


def cycles_to_ms(value: float | None) -> float | None:
    us = cycles_to_us(value)
    if us is None:
        return None
    return us / 1000.0


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
        "4000",
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


def build_decode_inputs(device: Any, workload: Workload) -> dict[str, Any]:
    d_qk = workload.d_qk
    q = torch.randn((1, workload.batch, workload.num_heads, d_qk), dtype=torch.bfloat16)
    k = torch.randn((workload.batch, workload.num_kv_heads, workload.seq_len, d_qk), dtype=torch.bfloat16)

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

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,
        k_chunk_size=workload.k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=workload.max_cores_per_head_batch,
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
        "program_config": program_config,
        "compute_config": compute_config,
        "out_mem_config": out_mem_config,
    }


def build_prefill_inputs(device: Any, workload: Workload) -> dict[str, Any]:
    if workload.seq_len % workload.block_size != 0:
        raise ValueError(f"Prefill seq_len {workload.seq_len} must be divisible by block_size {workload.block_size}")

    d_qk = workload.d_qk
    q = torch.randn((workload.batch, workload.num_heads, workload.seq_len, d_qk), dtype=torch.bfloat16)
    k = torch.randn((workload.batch, workload.num_kv_heads, workload.seq_len, d_qk), dtype=torch.bfloat16)

    max_num_blocks = (workload.seq_len // workload.block_size) * workload.batch
    paged_cfg = PagedAttentionConfig(block_size=workload.block_size, max_num_blocks=max_num_blocks)
    page_table = page_table_setup(workload.batch, paged_cfg)
    paged_cache_torch = to_paged_cache(k, page_table, paged_cfg)

    tt_q = ttnn.from_torch(
        q,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        paged_cache_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=workload.q_chunk_size,
        k_chunk_size=workload.k_chunk_size,
        exp_approx_mode=False,
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
        "program_config": program_config,
        "compute_config": compute_config,
    }


def deallocate_tensors(tensors: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        tensor = tensors.get(key)
        if tensor is not None:
            ttnn.deallocate(tensor)


def execute_child_run(case_name: str) -> None:
    from tracy import signpost

    workload = WORKLOADS[case_name]
    device = ttnn.open_device(device_id=0)
    logger.info(f"Opened device for child run: case={case_name}, mode={workload.mode}, arch={device.arch()}")

    tt_out = None
    try:
        scale = workload.d_qk**-0.5
        if workload.mode == "prefill":
            tensors = build_prefill_inputs(device, workload)
            dealloc_keys = ("tt_q", "tt_k", "tt_page_table")

            def run_one() -> Any:
                return ttnn.transformer.chunked_flash_mla_prefill(
                    tensors["tt_q"],
                    tensors["tt_k"],
                    workload.kv_lora_rank,
                    tensors["tt_page_table"],
                    chunk_start_idx=0,
                    scale=scale,
                    program_config=tensors["program_config"],
                    compute_kernel_config=tensors["compute_config"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

        elif workload.mode == "decode":
            tensors = build_decode_inputs(device, workload)
            dealloc_keys = ("tt_q", "tt_k", "tt_page_table", "tt_start_indices")

            def run_one() -> Any:
                return ttnn.transformer.paged_flash_multi_latent_attention_decode(
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

        else:
            raise ValueError(f"Unsupported mode: {workload.mode}")

        for _ in range(workload.warmup_iterations):
            tt_out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)
            tt_out = None

        signpost("start")
        for _ in range(workload.iterations):
            tt_out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)
            tt_out = None
        signpost("stop")

        deallocate_tensors(tensors, dealloc_keys)
    finally:
        if tt_out is not None:
            ttnn.deallocate(tt_out)
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
    wait_front_ns = column_stats.get("DEVICE COMPUTE CB WAIT FRONT [ns]", {}).get("avg_ns")
    reserve_back_ns = column_stats.get("DEVICE COMPUTE CB RESERVE BACK [ns]", {}).get("avg_ns")

    trisc_lookup = {
        "trisc0": column_stats.get("DEVICE TRISC0 KERNEL DURATION [ns]", {}).get("avg_ns"),
        "trisc1": column_stats.get("DEVICE TRISC1 KERNEL DURATION [ns]", {}).get("avg_ns"),
        "trisc2": column_stats.get("DEVICE TRISC2 KERNEL DURATION [ns]", {}).get("avg_ns"),
    }
    active_triscs = {name: value for name, value in trisc_lookup.items() if value is not None}
    compute_stage = max(active_triscs, key=active_triscs.get) if active_triscs else None
    compute_ns = active_triscs.get(compute_stage) if compute_stage is not None else None

    ratios = {}
    if kernel_ns and kernel_ns > 0:
        for key, value in {
            "ncrisc_share": ncrisc_ns,
            "brisc_share": brisc_ns,
            "compute_share": compute_ns,
            "compute_wait_front_share": wait_front_ns,
            "compute_reserve_back_share": reserve_back_ns,
        }.items():
            ratios[key] = (value / kernel_ns) if value is not None else None

    n_share = ratios.get("ncrisc_share")
    b_share = ratios.get("brisc_share")
    compute_share = ratios.get("compute_share")
    wait_front_share = ratios.get("compute_wait_front_share")

    classification = "unknown"
    reasons = []
    if n_share is not None and b_share is not None and n_share > 0.9 and b_share > 0.9:
        classification = "reader_writer_saturated"
        reasons.append("reader and writer almost fully occupy the kernel window")
    elif n_share is not None and n_share > 0.85:
        classification = "reader_close_to_critical_path"
        reasons.append("ncrisc is close to the kernel window")
    elif b_share is not None and b_share > 0.85:
        classification = "writer_close_to_critical_path"
        reasons.append("brisc is close to the kernel window")
    elif wait_front_share is not None and wait_front_share > 0.10:
        classification = "compute_starved_by_input"
        reasons.append("compute wait_front is significant")
    elif compute_share is not None and compute_share > 0.95:
        classification = "compute_on_critical_path"
        reasons.append(f"{compute_stage} is the last stage to retire")

    effective_reader_gbps = None
    effective_writer_gbps = None
    if ncrisc_ns and ncrisc_ns > 0:
        effective_reader_gbps = workload.k_bytes_single_pass / ncrisc_ns
    if brisc_ns and brisc_ns > 0:
        effective_writer_gbps = workload.out_bytes / brisc_ns

    return {
        "classification": classification,
        "reasons": reasons,
        "ratios": ratios,
        "compute_stage": compute_stage,
        "effective_single_pass_k_read_gbps": effective_reader_gbps,
        "effective_output_write_gbps": effective_writer_gbps,
        "pm_dram_bw_util_pct": column_stats.get("DRAM BW UTIL (%)", {}).get("avg_ns"),
        "pm_noc_util_pct": column_stats.get("NOC UTIL (%)", {}).get("avg_ns"),
        "pm_multicast_noc_util_pct": column_stats.get("MULTICAST NOC UTIL (%)", {}).get("avg_ns"),
    }


def load_raw_device_log(case_name: str) -> tuple[Path, pd.DataFrame] | tuple[None, None]:
    try:
        raw_log_path = latest_raw_device_log(OUTPUT_ROOT / case_name)
    except FileNotFoundError:
        return None, None

    df = pd.read_csv(raw_log_path, skiprows=1)
    df.columns = [str(col).strip() for col in df.columns]
    df["run host ID"] = pd.to_numeric(df["run host ID"], errors="coerce").fillna(-1).astype(int)
    df["data"] = pd.to_numeric(df["data"], errors="coerce").fillna(0.0)
    return raw_log_path, df


def parse_custom_stage_breakdown(
    raw_log_path: Path | None,
    raw_df: pd.DataFrame | None,
    measured_run_ids: list[int],
    processor_type: str,
    stage_markers: list[str],
) -> dict[str, Any]:
    if raw_log_path is None or raw_df is None:
        return {}

    df = raw_df[
        (raw_df["RISC processor type"] == processor_type)
        & (raw_df["run host ID"].isin(measured_run_ids))
        & (raw_df["zone name"].isin(stage_markers))
    ].copy()
    if df.empty:
        return {}

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

    avg_stage_total = sum(stage_stats.get(zone, {}).get("avg_cycles", 0.0) for zone in stage_markers)
    stage_shares = {}
    if avg_stage_total > 0:
        for zone in stage_markers:
            zone_avg = stage_stats.get(zone, {}).get("avg_cycles", 0.0)
            stage_shares[zone] = zone_avg / avg_stage_total

    return {
        "raw_log_path": str(raw_log_path),
        "measured_run_ids": measured_run_ids,
        "processor_type": processor_type,
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

    raw_log_path, raw_df = load_raw_device_log(case_name)
    reader_breakdown = parse_custom_stage_breakdown(
        raw_log_path, raw_df, measured_run_ids, "NCRISC", CUSTOM_READER_STAGE_MARKERS
    )
    reader_source_breakdown = parse_custom_stage_breakdown(
        raw_log_path, raw_df, measured_run_ids, "NCRISC", CUSTOM_READER_SOURCE_STAGE_MARKERS
    )
    writer_breakdown = parse_custom_stage_breakdown(
        raw_log_path, raw_df, measured_run_ids, "BRISC", CUSTOM_WRITER_STAGE_MARKERS
    )
    writer_source_breakdown = parse_custom_stage_breakdown(
        raw_log_path, raw_df, measured_run_ids, "BRISC", CUSTOM_WRITER_SOURCE_STAGE_MARKERS
    )

    return {
        "case": case_name,
        "mode": workload.mode,
        "variant": workload.variant,
        "workload": asdict(workload),
        "rows_profiled": int(len(df)),
        "measured_run_ids": measured_run_ids,
        "csv_path": str(csv_path),
        "column_stats": column_stats,
        "analysis": derive_bottleneck(workload, column_stats),
        "custom_reader_breakdown": reader_breakdown,
        "custom_reader_source_breakdown": reader_source_breakdown,
        "custom_writer_breakdown": writer_breakdown,
        "custom_writer_source_breakdown": writer_source_breakdown,
    }


def breakdown_stage_us(breakdown: dict[str, Any], marker: str) -> float:
    return cycles_to_us(breakdown.get("stage_stats", {}).get(marker, {}).get("avg_cycles")) or 0.0


def breakdown_total_us(breakdown: dict[str, Any]) -> float:
    return cycles_to_us(breakdown.get("max_profiled_cycles_per_run")) or 0.0


def breakdown_stage_ms(breakdown: dict[str, Any], marker: str) -> float:
    return cycles_to_ms(breakdown.get("stage_stats", {}).get(marker, {}).get("avg_cycles")) or 0.0


def breakdown_total_ms(breakdown: dict[str, Any]) -> float:
    return cycles_to_ms(breakdown.get("max_profiled_cycles_per_run")) or 0.0


def max_compute_thread_ms(entry: dict[str, Any]) -> float:
    stats = entry["column_stats"]
    return max(
        ns_to_ms(stats.get("DEVICE TRISC0 KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
        ns_to_ms(stats.get("DEVICE TRISC1 KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
        ns_to_ms(stats.get("DEVICE TRISC2 KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
    )


def split_proportionally(total_ms: float, components: dict[str, float]) -> dict[str, float]:
    positive_components = {key: max(value, 0.0) for key, value in components.items()}
    denominator = sum(positive_components.values())
    if denominator <= 0 or total_ms <= 0:
        return {key: 0.0 for key in components}
    return {key: total_ms * value / denominator for key, value in positive_components.items()}


def summarize_empirical_active_core_scan(
    scan: list[dict[str, Any]], plateau_tolerance: float = EMPIRICAL_PLATEAU_TOLERANCE
) -> dict[str, Any]:
    if not scan:
        return {}

    floor_point = min(scan, key=lambda item: item["predicted_kernel_ms"])
    floor_ms = floor_point["predicted_kernel_ms"]
    floor_active_cores = floor_point["active_cores"]
    noncompute_crossover = next((item["active_cores"] for item in scan if item["dominant_stage"] != "compute"), None)
    plateau_start = next(
        (item["active_cores"] for item in scan if item["predicted_kernel_ms"] <= floor_ms * (1 + plateau_tolerance)), None
    )
    sweep_16 = next((item for item in scan if item["active_cores"] == 16), None)
    sweep_64 = next((item for item in scan if item["active_cores"] == 64), None)

    return {
        "noncompute_crossover_active_cores": noncompute_crossover,
        "plateau_floor_active_cores": floor_active_cores,
        "plateau_start_active_cores": plateau_start,
        "plateau_tolerance": plateau_tolerance,
        "predicted_floor_ms": floor_ms,
        "speedup_16c_to_64c": (
            sweep_16["predicted_kernel_ms"] / sweep_64["predicted_kernel_ms"]
            if sweep_16 is not None and sweep_64 is not None and sweep_64["predicted_kernel_ms"] > 0
            else None
        ),
    }


def summarize_ideal_active_core_scan(workload: Workload) -> dict[str, Any]:
    scan = [simulate_blackhole_current_a_ideal_case(workload, active_cores) for active_cores in EMPIRICAL_ACTIVE_CORE_SCAN]
    dram_crossover = next((item["active_cores"] for item in scan if item["dominant_stage"] != "compute"), None)
    return {
        "dram_crossover_active_cores": dram_crossover,
        "scan": scan,
    }


def simulate_blackhole_current_a_ideal_case(workload: Workload, active_cores: int) -> dict[str, float | str | int]:
    k_dram_bytes = workload.seq_len * workload.d_qk * 4
    t_dram_ms = k_dram_bytes / (BH_DRAM_BW_GBPS * 1e9) * 1e3
    total_flops = 2 * A_LOCAL_HEADS * workload.seq_len * (workload.d_qk + workload.kv_lora_rank)
    t_compute_ms = total_flops / (active_cores * A_PHI_A_CORE * (BH_AICLK_MHZ / 1000.0) * 1e9) * 1e3
    t_reduce_ms = 0.0011
    predicted_core_ms = max(t_dram_ms, t_compute_ms) + t_reduce_ms
    dominant_stage = "compute" if t_compute_ms >= t_dram_ms else "dram"
    return {
        "active_cores": active_cores,
        "t_dram_ms": t_dram_ms,
        "t_compute_ms": t_compute_ms,
        "t_reduce_ms": t_reduce_ms,
        "predicted_core_ms": predicted_core_ms,
        "dominant_stage": dominant_stage,
    }


def simulate_blackhole_current_a_coupled_case(entry: dict[str, Any], active_cores: int) -> dict[str, Any]:
    workload = WORKLOADS[entry["case"]]
    ideal = simulate_blackhole_current_a_ideal_case(workload, active_cores)

    reader_breakdown = entry.get("custom_reader_breakdown", {})
    writer_breakdown = entry.get("custom_writer_breakdown", {})

    reader_total_wh_ms = ns_to_ms(entry["column_stats"].get("DEVICE NCRISC KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0
    writer_total_wh_ms = ns_to_ms(entry["column_stats"].get("DEVICE BRISC KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0
    compute_wh_ms = max_compute_thread_ms(entry)

    reader_page_table_wh_ms = breakdown_stage_ms(reader_breakdown, "SDPA-PAGE-TABLE-SUM")
    reader_reserve_wh_ms = breakdown_stage_ms(reader_breakdown, "SDPA-PAGED-RESERVE-SUM")
    reader_issue_wh_ms = breakdown_stage_ms(reader_breakdown, "SDPA-PAGED-ISSUE-SUM")
    reader_wait_wh_ms = breakdown_stage_ms(reader_breakdown, "SDPA-PAGED-WAIT-SUM")
    reader_push_wh_ms = breakdown_stage_ms(reader_breakdown, "SDPA-PAGED-PUSH-SUM")
    reader_tracked_total_wh_ms = breakdown_total_ms(reader_breakdown)
    reader_untracked_wh_ms = max(reader_total_wh_ms - reader_tracked_total_wh_ms, 0.0)

    writer_cb_wait_wh_ms = breakdown_stage_ms(writer_breakdown, "SDPA-WRITER-CB-WAIT-SUM")
    writer_issue_wh_ms = breakdown_stage_ms(writer_breakdown, "SDPA-WRITER-ISSUE-SUM")
    writer_barrier_wh_ms = breakdown_stage_ms(writer_breakdown, "SDPA-WRITER-BARRIER-SUM")
    writer_pop_wh_ms = breakdown_stage_ms(writer_breakdown, "SDPA-WRITER-POP-SUM")
    writer_tracked_total_wh_ms = breakdown_total_ms(writer_breakdown)
    writer_untracked_wh_ms = max(writer_total_wh_ms - writer_tracked_total_wh_ms, 0.0)

    dram_scale = WH_DRAM_BW_GBPS / BH_DRAM_BW_GBPS
    noc_scale = WH_NOC_BW_GBPS / BH_NOC_BW_GBPS
    clock_scale = WORMHOLE_AICLK_MHZ / BH_AICLK_MHZ

    reader_tracked_active_wh_ms = reader_page_table_wh_ms + reader_issue_wh_ms + reader_wait_wh_ms + reader_push_wh_ms
    reader_tracked_active_bh_ms = (
        reader_page_table_wh_ms * dram_scale
        + reader_issue_wh_ms * dram_scale
        + reader_wait_wh_ms * dram_scale
        + reader_push_wh_ms * clock_scale
    )
    reader_active_scale = (
        reader_tracked_active_bh_ms / reader_tracked_active_wh_ms if reader_tracked_active_wh_ms > 0 else clock_scale
    )
    reader_active_bh_ms = reader_tracked_active_bh_ms + reader_untracked_wh_ms * reader_active_scale

    writer_tracked_active_wh_ms = writer_issue_wh_ms + writer_barrier_wh_ms + writer_pop_wh_ms
    writer_tracked_active_bh_ms = (
        writer_issue_wh_ms * noc_scale
        + writer_barrier_wh_ms * noc_scale
        + writer_pop_wh_ms * clock_scale
    )
    writer_active_scale = (
        writer_tracked_active_bh_ms / writer_tracked_active_wh_ms if writer_tracked_active_wh_ms > 0 else clock_scale
    )
    writer_active_bh_ms = writer_tracked_active_bh_ms + writer_untracked_wh_ms * writer_active_scale

    reader_reserve_coupling = (
        reader_reserve_wh_ms / max(compute_wh_ms, writer_total_wh_ms)
        if max(compute_wh_ms, writer_total_wh_ms) > 0
        else 0.0
    )
    writer_wait_coupling = (
        writer_cb_wait_wh_ms / max(compute_wh_ms, reader_total_wh_ms)
        if max(compute_wh_ms, reader_total_wh_ms) > 0
        else 0.0
    )

    compute_bh_ms = float(ideal["t_compute_ms"])
    reader_total_bh_ms = reader_active_bh_ms
    writer_total_bh_ms = writer_active_bh_ms
    for _ in range(16):
        next_reader_total_bh_ms = reader_active_bh_ms + reader_reserve_coupling * max(compute_bh_ms, writer_total_bh_ms)
        next_writer_total_bh_ms = writer_active_bh_ms + writer_wait_coupling * max(compute_bh_ms, next_reader_total_bh_ms)
        if (
            abs(next_reader_total_bh_ms - reader_total_bh_ms) < 1e-9
            and abs(next_writer_total_bh_ms - writer_total_bh_ms) < 1e-9
        ):
            reader_total_bh_ms = next_reader_total_bh_ms
            writer_total_bh_ms = next_writer_total_bh_ms
            break
        reader_total_bh_ms = next_reader_total_bh_ms
        writer_total_bh_ms = next_writer_total_bh_ms

    reader_reserve_bh_ms = max(reader_total_bh_ms - reader_active_bh_ms, 0.0)
    writer_cb_wait_bh_ms = max(writer_total_bh_ms - writer_active_bh_ms, 0.0)
    predicted_kernel_ms = max(compute_bh_ms, reader_total_bh_ms, writer_total_bh_ms)

    reader_reserve_attribution = split_proportionally(
        reader_reserve_bh_ms,
        {
            "compute": compute_bh_ms,
            "writer": writer_total_bh_ms,
        },
    )
    writer_wait_attribution = split_proportionally(
        writer_cb_wait_bh_ms,
        {
            "compute": compute_bh_ms,
            "reader": reader_total_bh_ms,
        },
    )
    coupling_breakdown = {
        "reader_reserve_from_compute_ms": reader_reserve_attribution["compute"],
        "reader_reserve_from_writer_ms": reader_reserve_attribution["writer"],
        "writer_cb_wait_from_compute_ms": writer_wait_attribution["compute"],
        "writer_cb_wait_from_reader_ms": writer_wait_attribution["reader"],
    }
    dominant_coupling_source = max(coupling_breakdown, key=coupling_breakdown.get)

    dominant_pairs = {
        "compute": compute_bh_ms,
        "reader": reader_total_bh_ms,
        "writer": writer_total_bh_ms,
    }
    dominant_stage = max(dominant_pairs, key=dominant_pairs.get)

    return {
        "case": entry["case"],
        "seq_len": workload.seq_len,
        "batch": workload.batch,
        "active_cores": active_cores,
        "ideal_first_order_ms": float(ideal["predicted_core_ms"]),
        "ideal_first_order_dominant_stage": ideal["dominant_stage"],
        "compute_bh_ms": compute_bh_ms,
        "reader_active_bh_ms": reader_active_bh_ms,
        "reader_reserve_bh_ms": reader_reserve_bh_ms,
        "reader_total_bh_ms": reader_total_bh_ms,
        "writer_active_bh_ms": writer_active_bh_ms,
        "writer_cb_wait_bh_ms": writer_cb_wait_bh_ms,
        "writer_total_bh_ms": writer_total_bh_ms,
        "predicted_kernel_ms": predicted_kernel_ms,
        "dominant_stage": dominant_stage,
        "uplift_vs_ideal": (predicted_kernel_ms / float(ideal["predicted_core_ms"])) if ideal["predicted_core_ms"] else None,
        "reader_reserve_coupling_coeff": reader_reserve_coupling,
        "writer_cb_wait_coupling_coeff": writer_wait_coupling,
        "coupling_breakdown": coupling_breakdown,
        "dominant_coupling_source": dominant_coupling_source,
    }


def simulate_blackhole_current_a_coupled(decode_entries: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    active_core_sweep = []
    for entry in decode_entries:
        workload = WORKLOADS[entry["case"]]
        baseline_case = simulate_blackhole_current_a_coupled_case(entry, A_WH_ACTIVE_CORES)
        full_scan = [
            simulate_blackhole_current_a_coupled_case(entry, active_cores) for active_cores in EMPIRICAL_ACTIVE_CORE_SCAN
        ]
        full_scan_map = {item["active_cores"]: item for item in full_scan}
        scan_summary = summarize_empirical_active_core_scan(full_scan)
        ideal_scan_summary = summarize_ideal_active_core_scan(workload)

        baseline_case["first_order_dram_crossover_active_cores"] = ideal_scan_summary.get("dram_crossover_active_cores")
        baseline_case["second_order_noncompute_crossover_active_cores"] = scan_summary.get(
            "noncompute_crossover_active_cores"
        )
        baseline_case["second_order_plateau_start_active_cores"] = scan_summary.get("plateau_start_active_cores")
        baseline_case["second_order_speedup_16c_to_64c"] = scan_summary.get("speedup_16c_to_64c")
        baseline_case["second_order_floor_ms"] = scan_summary.get("predicted_floor_ms")
        cases.append(baseline_case)

        active_core_sweep.append(
            {
                "case": entry["case"],
                "seq_len": workload.seq_len,
                "sweep": [full_scan_map[active_cores] for active_cores in A_BH_ACTIVE_CORE_SWEEP],
                "summary": scan_summary,
                "first_order_summary": ideal_scan_summary,
            }
        )

    return {
        "model": "current_a_bh_empirical_coupled",
        "assumptions": {
            "calibration_source": "wormhole detailed decode profile",
            "wh_dram_bw_gbps": WH_DRAM_BW_GBPS,
            "bh_dram_bw_gbps": BH_DRAM_BW_GBPS,
            "wh_noc_bw_gbps": WH_NOC_BW_GBPS,
            "bh_noc_bw_gbps": BH_NOC_BW_GBPS,
            "wh_aiclk_mhz": WORMHOLE_AICLK_MHZ,
            "bh_aiclk_mhz": BH_AICLK_MHZ,
            "active_core_sweep": A_BH_ACTIVE_CORE_SWEEP,
            "full_active_core_scan": EMPIRICAL_ACTIVE_CORE_SCAN,
            "plateau_tolerance": EMPIRICAL_PLATEAU_TOLERANCE,
            "notes": [
                "reader reserve is scaled as a downstream coupling term against max(compute, writer)",
                "writer cb_wait is scaled as an upstream coupling term against max(compute, reader)",
                "untracked thread time is scaled by the same factor as the tracked active sub-stages",
                "coupling attribution is a heuristic proportional split used for interpretation, not a direct hardware counter",
            ],
        },
        "cases": cases,
        "active_core_sweep": active_core_sweep,
    }


def append_profile_table(lines: list[str], title: str, entries: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| case | batch | seq_len | kernel us | us/item | ncrisc us | brisc us | compute us(max trisc) | bottleneck | est k-read GB/s | est out-write GB/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for entry in entries:
        stats = entry["column_stats"]
        analysis = entry["analysis"]
        workload = WORKLOADS[entry["case"]]
        compute_stage = analysis.get("compute_stage")
        compute_ns = None
        if compute_stage is not None:
            compute_ns = stats.get(f"DEVICE {compute_stage.upper()} KERNEL DURATION [ns]", {}).get("avg_ns")
        kernel_us = ns_to_us(stats.get("DEVICE KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0
        lines.append(
            "| {case} | {batch} | {seq_len} | {kernel:.3f} | {per_item:.6f} | {ncrisc:.3f} | {brisc:.3f} | {compute:.3f} | {cls} | {reader_bw:.3f} | {writer_bw:.3f} |".format(
                case=entry["case"],
                batch=workload.batch,
                seq_len=workload.seq_len,
                kernel=kernel_us,
                per_item=kernel_us / max(workload.normalized_items, 1),
                ncrisc=ns_to_us(stats.get("DEVICE NCRISC KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
                brisc=ns_to_us(stats.get("DEVICE BRISC KERNEL DURATION [ns]", {}).get("avg_ns")) or 0.0,
                compute=ns_to_us(compute_ns) or 0.0,
                cls=analysis["classification"],
                reader_bw=analysis.get("effective_single_pass_k_read_gbps") or 0.0,
                writer_bw=analysis.get("effective_output_write_gbps") or 0.0,
            )
        )


def append_breakdown_table(
    lines: list[str],
    title: str,
    entries: list[dict[str, Any]],
    breakdown_key: str,
    duration_column: str,
    markers: list[str],
    labels: dict[str, str],
) -> None:
    stage_labels = [labels[marker] for marker in markers]
    stage_us_headers = " | ".join(f"{label} us/core(avg)" for label in stage_labels)
    share_headers = " | ".join(f"{label} share" for label in stage_labels if label != "page_table")
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            f"| case | thread us | {stage_us_headers} | profiled total us/core(max) | coverage vs thread | {share_headers} |",
            f"|---|---:|{'---:|' * len(markers)}---:|---:|{'---:|' * (len(markers) - (1 if 'page_table' in stage_labels else 0))}",
        ]
    )

    for entry in entries:
        breakdown = entry.get(breakdown_key, {})
        if not breakdown:
            continue

        thread_us = ns_to_us(entry["column_stats"].get(duration_column, {}).get("avg_ns")) or 0.0
        total_us = breakdown_total_us(breakdown)
        coverage = (total_us / thread_us) if thread_us > 0 else 0.0

        stage_us_values = [f"{breakdown_stage_us(breakdown, marker):.3f}" for marker in markers]
        share_values = []
        for marker in markers:
            label = labels[marker]
            if label == "page_table":
                continue
            share_values.append(f"{breakdown.get('stage_shares', {}).get(marker, 0.0):.1%}")

        lines.append(
            "| {case} | {thread_us:.3f} | {stage_us} | {total_us:.3f} | {coverage:.1%} | {share_values} |".format(
                case=entry["case"],
                thread_us=thread_us,
                stage_us=" | ".join(stage_us_values),
                total_us=total_us,
                coverage=coverage,
                share_values=" | ".join(share_values),
            )
        )


def has_breakdown_data(entries: list[dict[str, Any]], breakdown_key: str) -> bool:
    return any(entry.get(breakdown_key, {}).get("stage_stats") for entry in entries)


def format_coupling_source(name: str | None) -> str:
    if not name:
        return "-"
    return COUPLING_SOURCE_LABELS.get(name, name)


def append_bh_coupled_tables(lines: list[str], simulation: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## Decode A-BH empirical coupled model",
            "",
            "这一节把 WH decode detailed profile 的 `reader reserve` / `writer cb_wait` 回灌到当前 A 路径的 BH 模型里，",
            "不再只看理想化的 `compute vs dram`，而是把 pipeline backpressure 也显式带上。",
            "下面的 source attribution 是解释性拆分，不是新的硬件计数器。",
            "",
            "| case | seq_len | ideal 1st-order ms | empirical 2nd-order ms | compute ms | reader act/reserve ms | writer act/wait ms | dominant | uplift vs ideal |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )

    for case in simulation["cases"]:
        lines.append(
            "| {case_name} | {seq_len} | {ideal:.4f} | {empirical:.4f} | {compute:.4f} | {reader_active:.4f}/{reader_reserve:.4f} | {writer_active:.4f}/{writer_wait:.4f} | {dominant} | {uplift:.2f}x |".format(
                case_name=case["case"],
                seq_len=case["seq_len"],
                ideal=case["ideal_first_order_ms"],
                empirical=case["predicted_kernel_ms"],
                compute=case["compute_bh_ms"],
                reader_active=case["reader_active_bh_ms"],
                reader_reserve=case["reader_reserve_bh_ms"],
                writer_active=case["writer_active_bh_ms"],
                writer_wait=case["writer_cb_wait_bh_ms"],
                dominant=case["dominant_stage"],
                uplift=case["uplift_vs_ideal"] or 0.0,
            )
        )

    lines.extend(
        [
            "",
            "## Decode A-BH empirical scaling summary",
            "",
            "| case | seq_len | 1st-order dram crossover | 2nd-order non-compute crossover | 2nd-order plateau start (<=2% of 64c) | 16c->64c speedup | 64c floor ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in simulation["cases"]:
        lines.append(
            "| {case_name} | {seq_len} | {first_order} | {second_order} | {plateau_start} | {speedup:.2f}x | {floor_ms:.4f} |".format(
                case_name=case["case"],
                seq_len=case["seq_len"],
                first_order=case.get("first_order_dram_crossover_active_cores") or "-",
                second_order=case.get("second_order_noncompute_crossover_active_cores") or "-",
                plateau_start=case.get("second_order_plateau_start_active_cores") or "-",
                speedup=case.get("second_order_speedup_16c_to_64c") or 0.0,
                floor_ms=case.get("second_order_floor_ms") or 0.0,
            )
        )

    lines.extend(
        [
            "",
            "## Decode A-BH empirical coupling attribution",
            "",
            "| case | seq_len | reader active | reserve<-compute | reserve<-writer | writer active | wait<-compute | wait<-reader | dominant coupling |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in simulation["cases"]:
        coupling = case.get("coupling_breakdown", {})
        lines.append(
            "| {case_name} | {seq_len} | {reader_active:.4f} | {reserve_compute:.4f} | {reserve_writer:.4f} | {writer_active:.4f} | {wait_compute:.4f} | {wait_reader:.4f} | {dominant_coupling} |".format(
                case_name=case["case"],
                seq_len=case["seq_len"],
                reader_active=case["reader_active_bh_ms"],
                reserve_compute=coupling.get("reader_reserve_from_compute_ms", 0.0),
                reserve_writer=coupling.get("reader_reserve_from_writer_ms", 0.0),
                writer_active=case["writer_active_bh_ms"],
                wait_compute=coupling.get("writer_cb_wait_from_compute_ms", 0.0),
                wait_reader=coupling.get("writer_cb_wait_from_reader_ms", 0.0),
                dominant_coupling=format_coupling_source(case.get("dominant_coupling_source")),
            )
        )

    sweep_entries = [
        entry for entry in simulation["active_core_sweep"] if entry["case"] in CANONICAL_DECODE_CASES
    ] or simulation["active_core_sweep"]

    lines.extend(
        [
            "",
            "## Decode A-BH empirical coupled active-core sweep",
            "",
            "| case | seq_len | 16c | 20c | 24c | 32c | 48c | 64c |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in sweep_entries:
        sweep_map = {item["active_cores"]: item for item in entry["sweep"]}
        lines.append(
            "| {case} | {seq_len} | {c16} | {c20} | {c24} | {c32} | {c48} | {c64} |".format(
                case=entry["case"],
                seq_len=entry["seq_len"],
                c16=f"{sweep_map[16]['predicted_kernel_ms']:.4f} ({sweep_map[16]['dominant_stage']})",
                c20=f"{sweep_map[20]['predicted_kernel_ms']:.4f} ({sweep_map[20]['dominant_stage']})",
                c24=f"{sweep_map[24]['predicted_kernel_ms']:.4f} ({sweep_map[24]['dominant_stage']})",
                c32=f"{sweep_map[32]['predicted_kernel_ms']:.4f} ({sweep_map[32]['dominant_stage']})",
                c48=f"{sweep_map[48]['predicted_kernel_ms']:.4f} ({sweep_map[48]['dominant_stage']})",
                c64=f"{sweep_map[64]['predicted_kernel_ms']:.4f} ({sweep_map[64]['dominant_stage']})",
            )
        )


def write_summary(results: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "flash_mla_wh_detailed_profile_results.json"
    md_path = OUTPUT_ROOT / "flash_mla_wh_detailed_profile_report.md"

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    prefill_entries = [entry for entry in results["profiles"] if entry["mode"] == "prefill"]
    decode_entries = [entry for entry in results["profiles"] if entry["mode"] == "decode"]
    bh_simulation = results.get("bh_empirical_decode")
    if decode_entries:
        bh_simulation = simulate_blackhole_current_a_coupled(decode_entries)
        results["bh_empirical_decode"] = bh_simulation
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)

    lines = [
        "# FlashMLA WH detailed profile summary",
        "",
        "本报告覆盖 paged/chunked MLA prefill 与 paged MLA decode 两条路径。",
        "",
        "- prefill 实测路径：`ttnn.transformer.chunked_flash_mla_prefill`（`chunk_start_idx=0`）",
        "- decode 实测路径：`ttnn.transformer.paged_flash_multi_latent_attention_decode`",
        f"- 自定义 reader marker：{', '.join(CUSTOM_READER_STAGE_MARKERS)}",
        f"- 自定义 writer marker：{', '.join(CUSTOM_WRITER_STAGE_MARKERS)}",
        f"- 可选 reader source marker：{', '.join(CUSTOM_READER_SOURCE_STAGE_MARKERS)}",
        f"- 可选 writer source marker：{', '.join(CUSTOM_WRITER_SOURCE_STAGE_MARKERS)}",
        f"- cycle -> us 换算基于 WH AICLK `{int(WORMHOLE_AICLK_MHZ)}` MHz",
    ]

    append_profile_table(lines, "Prefill measured profile", prefill_entries)
    append_profile_table(lines, "Decode measured profile", decode_entries)
    append_breakdown_table(
        lines,
        "Prefill custom NCRISC stage breakdown",
        prefill_entries,
        "custom_reader_breakdown",
        "DEVICE NCRISC KERNEL DURATION [ns]",
        CUSTOM_READER_STAGE_MARKERS,
        READER_STAGE_LABELS,
    )
    append_breakdown_table(
        lines,
        "Decode custom NCRISC stage breakdown",
        decode_entries,
        "custom_reader_breakdown",
        "DEVICE NCRISC KERNEL DURATION [ns]",
        CUSTOM_READER_STAGE_MARKERS,
        READER_STAGE_LABELS,
    )
    append_breakdown_table(
        lines,
        "Prefill custom BRISC stage breakdown",
        prefill_entries,
        "custom_writer_breakdown",
        "DEVICE BRISC KERNEL DURATION [ns]",
        CUSTOM_WRITER_STAGE_MARKERS,
        WRITER_STAGE_LABELS,
    )
    append_breakdown_table(
        lines,
        "Decode custom BRISC stage breakdown",
        decode_entries,
        "custom_writer_breakdown",
        "DEVICE BRISC KERNEL DURATION [ns]",
        CUSTOM_WRITER_STAGE_MARKERS,
        WRITER_STAGE_LABELS,
    )
    if has_breakdown_data(decode_entries, "custom_reader_source_breakdown"):
        append_breakdown_table(
            lines,
            "Decode custom reader source breakdown",
            decode_entries,
            "custom_reader_source_breakdown",
            "DEVICE NCRISC KERNEL DURATION [ns]",
            CUSTOM_READER_SOURCE_STAGE_MARKERS,
            READER_SOURCE_STAGE_LABELS,
        )
    if has_breakdown_data(decode_entries, "custom_writer_source_breakdown"):
        append_breakdown_table(
            lines,
            "Decode custom writer source breakdown",
            decode_entries,
            "custom_writer_source_breakdown",
            "DEVICE BRISC KERNEL DURATION [ns]",
            CUSTOM_WRITER_SOURCE_STAGE_MARKERS,
            WRITER_SOURCE_STAGE_LABELS,
        )
    if bh_simulation is not None:
        append_bh_coupled_tables(lines, bh_simulation)

    with md_path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def select_cases(requested_cases: list[str] | None, requested_modes: list[str]) -> list[str]:
    if requested_cases:
        return requested_cases
    return [case_name for case_name, workload in WORKLOADS.items() if workload.mode in requested_modes]


def list_cases() -> None:
    for case_name, workload in WORKLOADS.items():
        print(
            f"{case_name:>12}  mode={workload.mode:<7} batch={workload.batch:<2} seq_len={workload.seq_len:<5} "
            f"variant={workload.variant}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed WH profile for MLA prefill and decode.")
    parser.add_argument("--child-run", action="store_true", help="Execute the workload body to be profiled by tracy.")
    parser.add_argument("--case", choices=sorted(WORKLOADS.keys()), help="Single case to execute in child-run mode.")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(WORKLOADS.keys()),
        default=None,
        help="Explicit case list to profile.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["prefill", "decode"],
        default=["prefill", "decode"],
        help="When --cases is omitted, profile all cases for the selected modes.",
    )
    parser.add_argument("--list-cases", action="store_true", help="List available profile cases and exit.")
    args = parser.parse_args()

    if args.list_cases:
        list_cases()
        return

    if args.child_run:
        if not args.case:
            raise ValueError("--case is required with --child-run")
        execute_child_run(args.case)
        return

    selected_cases = select_cases(args.cases, args.modes)
    profiles = []
    for case_name in selected_cases:
        csv_path = run_tracy_profile(case_name)
        profiles.append(parse_report(case_name, csv_path))

    results = {
        "metadata": {
            "arch": "wormhole_b0",
            "aiclk_mhz": WORMHOLE_AICLK_MHZ,
            "selected_cases": selected_cases,
        },
        "profiles": profiles,
    }
    write_summary(results)
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

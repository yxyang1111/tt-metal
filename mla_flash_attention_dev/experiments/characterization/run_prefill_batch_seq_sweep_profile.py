#!/usr/bin/env python3
"""Batch x sequence-length sweep profile for FlashMLA prefill on Wormhole."""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "prefill_batch_seq_sweep"
THIS_FILE = Path(__file__).resolve()

PEAK_DRAM_BW_GBS = 288.0
PEAK_FPU_TFLOPS = 65.5
NUM_COMPUTE_CORES = 64
WORMHOLE_AICLK_MHZ = 1000.0

D_C = 512
D_R = 64
D_QK = D_C + D_R
H = 32
NUM_KV_HEADS = 1
BLOCK_SIZE = 64
K_CHUNK_SIZE = 128
Q_CHUNK_SIZE = 32
ELEMENT_BYTES = 2

DEFAULT_SEQ_LENS = [1024, 4096, 8192, 16384, 32768]
DEFAULT_BATCHES = [1, 2, 4, 6]

DEVICE_ANALYSIS_TYPES = [
    "device_kernel_duration",
    "device_brisc_kernel_duration",
    "device_ncrisc_kernel_duration",
    "device_trisc0_kernel_duration",
    "device_trisc1_kernel_duration",
    "device_trisc2_kernel_duration",
]

CSV_FLOAT_COLUMNS = [
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "PM FPU UTIL (%)",
]


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    seq_len: int
    iterations: int
    warmup_iterations: int


def case_name(batch: int, seq_len: int) -> str:
    suffix = f"{seq_len // 1024}k" if seq_len >= 1024 else str(seq_len)
    return f"prefill_b{batch}_L{suffix}"


def make_workload(batch: int, seq_len: int) -> Workload:
    # Long prefill cases are already seconds-scale at B=1, so keep profiling
    # runs short while still allowing small cases to average noise.
    tokens = batch * seq_len
    if tokens <= 4 * 1024:
        iterations, warmup = 3, 1
    elif tokens <= 16 * 1024:
        iterations, warmup = 2, 1
    else:
        iterations, warmup = 1, 0
    return Workload(case_name(batch, seq_len), batch, seq_len, iterations, warmup)


def nearest_n(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def nearest_pow_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def execute_child_run(batch: int, seq_len: int, iterations: int, warmup: int) -> None:
    from tracy import signpost

    import torch
    import ttnn
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import page_table_setup, to_paged_cache

    device = ttnn.open_device(device_id=0)
    print(f"[child] device opened, arch={device.arch()}, batch={batch}, seq_len={seq_len}")

    try:
        q = torch.randn((batch, H, seq_len, D_QK), dtype=torch.bfloat16)
        k = torch.randn((batch, NUM_KV_HEADS, seq_len, D_QK), dtype=torch.bfloat16)

        max_num_blocks = (seq_len // BLOCK_SIZE) * batch
        paged_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks)
        page_table = page_table_setup(batch, paged_cfg)
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
            q_chunk_size=nearest_pow_2(nearest_n(H, 32)),
            k_chunk_size=K_CHUNK_SIZE,
            exp_approx_mode=False,
        )
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        scale = D_QK**-0.5

        def run_one():
            return ttnn.transformer.chunked_flash_mla_prefill(
                tt_q,
                tt_k,
                D_C,
                tt_page_table,
                chunk_start_idx=0,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_config,
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

        for t in (tt_q, tt_k, tt_page_table):
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)


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
        "python3",
        "-m",
        "tracy",
        "-p",
        "-o",
        str(case_dir),
        "--check-exit-code",
        "--op-support-count",
        "4000",
        "-t",
        "5000",
    ]
    for analysis in DEVICE_ANALYSIS_TYPES:
        cmd.extend(["-a", analysis])
    cmd.extend(
        [
            str(THIS_FILE),
            "--child-run",
            "--batch",
            str(w.batch),
            "--seq-len",
            str(w.seq_len),
            "--iters",
            str(w.iterations),
            "--warmup",
            str(w.warmup_iterations),
        ]
    )

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


def summarize_numeric(series) -> dict[str, float]:
    import pandas as pd

    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    vals = vals[vals > 0]
    if vals.empty:
        return {}
    return {"avg_ns": float(vals.mean()), "std_ns": float(vals.std(ddof=0))}


def causal_pairs(seq_len: int) -> int:
    return seq_len * (seq_len + 1) // 2


def estimate_prefill_dram_bytes(batch: int, seq_len: int) -> int:
    q_bytes = batch * H * seq_len * D_QK * ELEMENT_BYTES
    out_bytes = batch * H * seq_len * D_C * ELEMENT_BYTES
    num_q_chunks = math.ceil(seq_len / Q_CHUNK_SIZE)
    scanned_tokens = 0
    for chunk in range(num_q_chunks):
        scanned_tokens += min((chunk + 1) * Q_CHUNK_SIZE, seq_len)
    k_bytes = batch * H * scanned_tokens * D_QK * ELEMENT_BYTES
    v_bytes = batch * H * scanned_tokens * D_C * ELEMENT_BYTES
    return q_bytes + k_bytes + v_bytes + out_bytes


def parse_and_analyze(w: Workload, csv_path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    relevant = w.warmup_iterations + w.iterations
    if len(df) < relevant:
        raise RuntimeError(f"Expected >= {relevant} rows, got {len(df)} for {w.name}")
    df = df.tail(relevant).reset_index(drop=True)
    df = df.iloc[w.warmup_iterations:].copy()

    cs: dict[str, dict[str, float]] = {}
    for col in CSV_FLOAT_COLUMNS:
        if col in df.columns:
            stats = summarize_numeric(df[col])
            if stats:
                cs[col] = stats

    kernel_ns = cs.get("DEVICE KERNEL DURATION [ns]", {}).get("avg_ns", 0.0)
    kernel_us = kernel_ns / 1000.0 if kernel_ns else 0.0
    pm_fpu = cs.get("PM FPU UTIL (%)", {}).get("avg_ns")

    active_cores = NUM_COMPUTE_CORES
    active_peak_tflops = PEAK_FPU_TFLOPS * active_cores / NUM_COMPUTE_CORES

    pairs = causal_pairs(w.seq_len)
    flops_per_head = 2 * pairs * (D_QK + D_C)
    total_flops = flops_per_head * H * w.batch
    overall_tflops = total_flops / kernel_ns / 1000.0 if kernel_ns else 0.0

    dram_bytes = estimate_prefill_dram_bytes(w.batch, w.seq_len)
    actual_bw_gbs = dram_bytes / kernel_ns if kernel_ns else 0.0

    return {
        "case": w.name,
        "batch": w.batch,
        "seq_len": w.seq_len,
        "kernel_us": round(kernel_us, 2),
        "pm_fpu_util_pct": round(pm_fpu, 2) if pm_fpu else None,
        "total_flops_G": round(total_flops / 1e9, 3),
        "overall_tflops": round(overall_tflops, 4),
        "overall_vs_peak_pct": round(overall_tflops / active_peak_tflops * 100, 2),
        "total_dram_traffic_MB": round(dram_bytes / 1e6, 2),
        "actual_bw_gbs": round(actual_bw_gbs, 2),
        "actual_bw_vs_peak_pct": round(actual_bw_gbs / PEAK_DRAM_BW_GBS * 100, 1),
    }


def aggregate_repeats(w: Workload, repeat_results: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(repeat_results[0])
    result["num_repeats"] = len(repeat_results)
    result["repeat_results"] = repeat_results
    if len(repeat_results) > 1:
        repeat_std = {}
        for key in ("kernel_us", "overall_tflops", "actual_bw_gbs"):
            values = [r[key] for r in repeat_results]
            result[key] = round(float(statistics.median(values)), 4)
            repeat_std[key] = round(float(statistics.pstdev(values)), 4)
        result["repeat_std"] = repeat_std
    else:
        result["repeat_std"] = {}
    return result


def print_summary(results: list[dict[str, Any]]) -> None:
    seq_lens = sorted(set(r["seq_len"] for r in results))
    batches = sorted(set(r["batch"] for r in results))
    lookup = {(r["batch"], r["seq_len"]): r for r in results}

    for title, key, fmt in [
        ("Prefill throughput (TFLOP/s)", "overall_tflops", "{:>12.3f}"),
        ("Prefill estimated DRAM BW (GB/s)", "actual_bw_gbs", "{:>12.1f}"),
        ("Prefill kernel time (ms)", "kernel_us", "{:>12.1f}"),
    ]:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)
        print(f"{'Batch':>6}" + "".join(f"{'L='+str(s//1024)+'K':>12}" for s in seq_lens))
        for b in batches:
            row = f"{b:>6}"
            for s in seq_lens:
                r = lookup.get((b, s))
                value = r[key] / 1000.0 if key == "kernel_us" and r else None
                row += fmt.format(value if key == "kernel_us" else r[key]) if r else f"{'FAIL':>12}"
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch x seq-len sweep profile for FlashMLA prefill.")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()

    if args.child_run:
        execute_child_run(args.batch, args.seq_len, args.iters, args.warmup)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    workloads = [make_workload(b, s) for b in args.batches for s in args.seq_lens]
    results = []
    failed = []

    for idx, w in enumerate(workloads, 1):
        print(f"\n[{idx}/{len(workloads)}] ", end="")
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
                    raise FileNotFoundError(f"Missing repeat {repeat_idx + 1}/{args.repeats}: {csv_path}")
                repeat_results.append(parse_and_analyze(w, csv_path))
            results.append(aggregate_repeats(w, repeat_results))
        except Exception as exc:
            print(f"    ERROR {w.name}: {exc}")
            failed.append({"case": w.name, "batch": w.batch, "seq_len": w.seq_len, "error": str(exc)})

    print_summary(results)
    out_json = OUTPUT_ROOT / "prefill_batch_seq_sweep_results.json"
    with out_json.open("w") as f:
        json.dump({"results": results, "failed": failed}, f, indent=2)
    print(f"\nResults: {out_json}")
    if failed:
        print(f"Failed cases ({len(failed)}):")
        for fail in failed:
            print(f"  {fail['case']}: {fail['error']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Measure actual DRAM read traffic for FlashMLA decode vs theoretical minimum.

Runs the TT mainline FlashMLA decode kernel with NoC event tracing enabled,
then parses the resulting noc_trace JSON files to measure total DRAM read
bytes.  Compares with the theoretical minimum (single-pass read with
multicast) to directly quantify redundancy.

Usage:
    # Run kernel with NoC trace, then parse (requires Wormhole device)
    python measure_dram_traffic.py --run --cases decode_1k decode_4k decode_32k

    # Parse existing trace directory only (no device needed)
    python measure_dram_traffic.py --parse-only --trace-dir <path>
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
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "dram_traffic"
THIS_FILE = Path(__file__).resolve()

D_C = 512
D_R = 64
H = 32
ELEMENT_BYTES = 2  # BF16 / BF8_b stored as 2-byte (BF8_b is 1 byte per element in DRAM tile layout)
ELEMENT_BYTES_K = 1  # BF8_b paged cache is 1 byte per datum in DRAM
BLOCK_SIZE = 64
K_CHUNK_SIZE = 128
MAX_CORES_PER_HEAD_BATCH = 4
PEAK_DRAM_BW_GBS = 200.0


@dataclass(frozen=True)
class DecodeCase:
    name: str
    seq_len: int
    batch: int
    iterations: int = 2
    warmup_iterations: int = 1


DECODE_CASES = {
    "decode_1k": DecodeCase("decode_1k", 1024, 2),
    "decode_4k": DecodeCase("decode_4k", 4096, 2),
    "decode_8k": DecodeCase("decode_8k", 8192, 2),
    "decode_16k": DecodeCase("decode_16k", 16384, 1),
    "decode_32k": DecodeCase("decode_32k", 32768, 1),
}

DEFAULT_CASES = ["decode_1k", "decode_4k", "decode_32k"]


# ── Theoretical minimum computation ──────────────────────────────────────


def compute_theoretical(case: DecodeCase) -> dict:
    """Compute theoretical DRAM read bytes for a decode case.

    In MLA mode (reuse_k=true), V is read from K's L1 buffer, so the only
    DRAM reads are for the K cache (d_c + d_r wide, BF8_b).
    """
    latent_width = D_C + D_R  # 576

    # Logical data volume: seq_len × latent_width × 1 byte (BF8_b)
    k_cache_bytes_logical = case.seq_len * latent_width * ELEMENT_BYTES_K

    # Tile-layout overhead: BF8_b tiles are 32×32 = 1024 elements.
    # Each tile stores: 4-byte header + 32 rows × (1 exp byte + 32 mantissa bytes) = 1060 bytes.
    # For simplicity, use the ttnn convention: BF8_b tile ≈ 1088 bytes (with alignment).
    BF8B_TILE_BYTES = 1088
    TILE_H, TILE_W = 32, 32
    seq_tiles = math.ceil(case.seq_len / TILE_H)
    dim_tiles = math.ceil(latent_width / TILE_W)  # 576 / 32 = 18 tiles
    k_cache_bytes_tiled = seq_tiles * dim_tiles * BF8B_TILE_BYTES

    n_head_batches = math.ceil(H / 8)  # dqhpc=8 for H=32 → 4 groups
    n_seq_cores = min(MAX_CORES_PER_HEAD_BATCH, math.ceil(case.seq_len / BLOCK_SIZE))

    # Actual: each head group reads the full K cache independently from DRAM.
    # V is NOT read from DRAM (reuse_k=true in MLA: V reuses K's L1 buffer).
    total_actual_logical = case.batch * n_head_batches * k_cache_bytes_logical
    total_actual_tiled = case.batch * n_head_batches * k_cache_bytes_tiled

    # Ideal: with multicast, K is read once per batch element from DRAM
    total_ideal_logical = case.batch * k_cache_bytes_logical
    total_ideal_tiled = case.batch * k_cache_bytes_tiled

    redundancy = n_head_batches
    waste_pct = (total_actual_logical - total_ideal_logical) / total_actual_logical * 100.0

    return {
        "case": case.name,
        "seq_len": case.seq_len,
        "batch": case.batch,
        "n_head_batches": n_head_batches,
        "n_seq_cores": n_seq_cores,
        "k_cache_logical_MB": round(k_cache_bytes_logical / 1e6, 3),
        "k_cache_tiled_MB": round(k_cache_bytes_tiled / 1e6, 3),
        "total_actual_logical_MB": round(total_actual_logical / 1e6, 3),
        "total_actual_tiled_MB": round(total_actual_tiled / 1e6, 3),
        "total_ideal_logical_MB": round(total_ideal_logical / 1e6, 3),
        "total_ideal_tiled_MB": round(total_ideal_tiled / 1e6, 3),
        "redundancy_factor": redundancy,
        "waste_pct": round(waste_pct, 1),
        "total_actual_bytes": total_actual_logical,
        "total_actual_tiled_bytes": total_actual_tiled,
        "total_ideal_bytes": total_ideal_logical,
        "total_ideal_tiled_bytes": total_ideal_tiled,
        "note": "V not counted: reuse_k=true in MLA, V read from K L1 buffer",
    }


# ── NoC trace parsing ────────────────────────────────────────────────────


# Wormhole B0 DRAM core NOC0 physical coordinates (from soc descriptor).
# 6 channels × 3 subchannels = 18 endpoints.
WH_DRAM_NOC0_COORDS: set[tuple[int, int]] = {
    # Channel 0
    (0, 0), (0, 1), (0, 11),
    # Channel 1
    (0, 5), (0, 6), (0, 7),
    # Channel 2
    (5, 0), (5, 1), (5, 11),
    # Channel 3
    (5, 2), (5, 9), (5, 10),
    # Channel 4
    (5, 3), (5, 4), (5, 8),
    # Channel 5
    (5, 5), (5, 6), (5, 7),
}


def load_dram_coords(trace_dir: Path) -> set[tuple[int, int]]:
    """Load DRAM bank coordinates from dram_coords.json, fall back to hardcoded WH values."""
    coords_file = trace_dir / "dram_coords.json"
    if coords_file.exists():
        data = json.loads(coords_file.read_text())
        return {(c["x"], c["y"]) for c in data["dram_bank_noc0_coords"]}
    return WH_DRAM_NOC0_COORDS


def parse_noc_trace(trace_json_path: Path, dram_coords: set[tuple[int, int]]) -> dict:
    """Parse a single noc_trace JSON file and sum DRAM read bytes."""
    events = json.loads(trace_json_path.read_text())

    total_read_bytes = 0
    total_write_bytes = 0
    dram_read_bytes = 0
    non_dram_read_bytes = 0
    read_events = 0
    dram_read_events = 0
    write_events = 0
    unknown_dst_read_bytes = 0

    for ev in events:
        ev_type = ev.get("type", "")
        num_bytes = ev.get("num_bytes", 0)

        if "READ" in ev_type:
            total_read_bytes += num_bytes
            read_events += 1

            dx = ev.get("dx")
            dy = ev.get("dy")

            if dx is not None and dy is not None:
                if (dx, dy) in dram_coords:
                    dram_read_bytes += num_bytes
                    dram_read_events += 1
                else:
                    non_dram_read_bytes += num_bytes
            else:
                unknown_dst_read_bytes += num_bytes

        elif "WRITE" in ev_type:
            total_write_bytes += num_bytes
            write_events += 1

    return {
        "file": trace_json_path.name,
        "total_read_bytes": total_read_bytes,
        "dram_read_bytes": dram_read_bytes,
        "non_dram_read_bytes": non_dram_read_bytes,
        "unknown_dst_read_bytes": unknown_dst_read_bytes,
        "total_write_bytes": total_write_bytes,
        "read_events": read_events,
        "dram_read_events": dram_read_events,
        "write_events": write_events,
    }


def parse_all_traces(trace_dir: Path) -> dict:
    """Parse all noc_trace*.json in the directory."""
    dram_coords = load_dram_coords(trace_dir)

    trace_files = sorted(trace_dir.glob("noc_trace*.json"))
    if not trace_files:
        print(f"  WARNING: No noc_trace*.json files found in {trace_dir}")
        return {"files": [], "aggregate": {}}

    results = []
    for tf in trace_files:
        r = parse_noc_trace(tf, dram_coords)
        results.append(r)

    agg_total_read = sum(r["total_read_bytes"] for r in results)
    agg_dram_read = sum(r["dram_read_bytes"] for r in results)
    agg_non_dram_read = sum(r["non_dram_read_bytes"] for r in results)
    agg_unknown = sum(r["unknown_dst_read_bytes"] for r in results)
    agg_write = sum(r["total_write_bytes"] for r in results)
    agg_read_events = sum(r["read_events"] for r in results)
    agg_dram_events = sum(r["dram_read_events"] for r in results)

    return {
        "dram_coords": sorted(dram_coords),
        "n_trace_files": len(trace_files),
        "files": results,
        "aggregate": {
            "total_read_bytes": agg_total_read,
            "total_read_MB": round(agg_total_read / 1e6, 3),
            "dram_read_bytes": agg_dram_read,
            "dram_read_MB": round(agg_dram_read / 1e6, 3),
            "non_dram_read_bytes": agg_non_dram_read,
            "non_dram_read_MB": round(agg_non_dram_read / 1e6, 3),
            "unknown_dst_read_bytes": agg_unknown,
            "unknown_dst_read_MB": round(agg_unknown / 1e6, 3),
            "total_write_bytes": agg_write,
            "total_write_MB": round(agg_write / 1e6, 3),
            "total_read_events": agg_read_events,
            "dram_read_events": agg_dram_events,
        },
    }


# ── Device run infrastructure ────────────────────────────────────────────


def run_with_noc_trace(case_name: str, output_dir: Path) -> Path:
    """Run FlashMLA decode with NoC event tracing (device-only mode, no tracy)."""
    case_dir = output_dir / case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = case_dir / ".logs"
    logs_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TTNN_OP_PROFILER"] = "1"
    env["TT_METAL_PROFILER_TRACE_TRACKING"] = "1"
    env["TT_METAL_PROFILER_DIR"] = str(case_dir)
    env["TT_METAL_DEVICE_PROFILER_NOC_EVENTS"] = "1"
    env["TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH"] = str(logs_dir.resolve())

    cmd = [
        "python3", str(THIS_FILE), "--child-run", "--case", case_name,
    ]

    print(f"\n  Running {case_name} with NoC trace (device-only)...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED: {case_name}")
        print(f"  STDOUT (last 3000 chars):\n{result.stdout[-3000:]}")
        print(f"  STDERR (last 3000 chars):\n{result.stderr[-3000:]}")
        raise RuntimeError(f"NoC trace run failed for {case_name}")

    print(f"  Completed {case_name}")

    # NoC traces are written to the RPT_PATH directory
    # Also check if they ended up in the case_dir
    trace_files = list(logs_dir.glob("noc_trace*.json"))
    if not trace_files:
        trace_files = list(case_dir.glob("noc_trace*.json"))
        if trace_files:
            logs_dir = case_dir
    print(f"  Found {len(trace_files)} trace files in {logs_dir}")
    return logs_dir


def execute_child_run(case_name: str) -> None:
    """Child process: open device, dump DRAM coords, run FlashMLA kernel."""
    import torch

    import ttnn
    from models.common.utility_functions import nearest_y
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
        page_table_setup,
        to_paged_cache,
    )

    case = DECODE_CASES[case_name]
    device = ttnn.open_device(device_id=0)
    print(f"  [child] Opened device for {case_name}, arch={device.arch()}")

    try:
        dump_dram_coordinates(device)

        d_qk = D_C + D_R
        scale = d_qk ** -0.5

        q = torch.randn((1, case.batch, H, d_qk), dtype=torch.bfloat16)
        k = torch.randn((case.batch, 1, case.seq_len, d_qk), dtype=torch.bfloat16)

        max_num_blocks = (case.seq_len // BLOCK_SIZE) * case.batch
        paged_cfg = PagedAttentionConfig(
            block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks
        )
        page_table = page_table_setup(case.batch, paged_cfg)
        paged_cache_torch = to_paged_cache(k, page_table, paged_cfg)

        grid_size = device.compute_with_storage_grid_size()

        q_num_cores = min(case.batch * H, grid_size.x * grid_size.y)
        block_height = nearest_y(
            (case.batch * H) // q_num_cores, ttnn.TILE_SIZE
        )
        q_core_grid = ttnn.num_cores_to_corerangeset(
            q_num_cores, grid_size, row_wise=True
        )

        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, d_qk),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        out_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, D_C),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        start_indices = torch.full(
            (case.batch,), case.seq_len - 1, dtype=torch.int32
        )

        tt_q = ttnn.from_torch(
            q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=q_mem_config,
        )
        tt_k = ttnn.from_torch(
            paged_cache_torch, device=device, dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = ttnn.from_torch(
            page_table, device=device, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_start_indices = ttnn.from_torch(
            start_indices, device=device, dtype=ttnn.int32,
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=0,
            k_chunk_size=K_CHUNK_SIZE,
            exp_approx_mode=False,
            max_cores_per_head_batch=MAX_CORES_PER_HEAD_BATCH,
        )
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def run_one():
            return ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tt_q, tt_k,
                page_table_tensor=tt_page_table,
                cur_pos_tensor=tt_start_indices,
                head_dim_v=D_C,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_config,
                memory_config=out_mem_config,
            )

        for _ in range(case.warmup_iterations):
            tt_out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)

        # Measured iterations (profiler captures these)
        for _ in range(case.iterations):
            tt_out = run_one()
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)

        ttnn.deallocate(tt_q)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_page_table)
        ttnn.deallocate(tt_start_indices)

    finally:
        ttnn.close_device(device)


def dump_dram_coordinates(device) -> None:
    """Dump DRAM bank NOC0 coordinates to a JSON file in the trace output dir.

    Tries the device API first; if unavailable, writes the hardcoded WH set
    so the parser can still match DRAM events.
    """
    noc_events_path = os.environ.get("TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH")
    if not noc_events_path:
        return

    coords = []
    try:
        dram_grid = device.dram_grid_size()
        for x in range(dram_grid.x):
            for y in range(dram_grid.y):
                channel = x * dram_grid.y + y
                core = device.dram_core_from_dram_channel(channel)
                coords.append({"channel": channel, "x": core.x, "y": core.y})
    except (AttributeError, Exception) as e:
        print(f"  [child] Could not read DRAM coords from device API ({e}), using hardcoded WH values")
        coords = [{"channel": i, "x": x, "y": y} for i, (x, y) in enumerate(sorted(WH_DRAM_NOC0_COORDS))]

    out_path = Path(noc_events_path) / "dram_coords.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "arch": str(device.arch()),
        "num_dram_channels": len(coords),
        "dram_bank_noc0_coords": coords,
    }, indent=2))
    print(f"  [child] Dumped {len(coords)} DRAM bank coords to {out_path}")


# ── Report generation ────────────────────────────────────────────────────


def generate_comparison_table(
    theoretical: list[dict], measured: dict[str, dict]
) -> str:
    """Generate markdown comparison table."""
    lines = [
        "# DRAM Read Traffic: Measured vs Theoretical",
        "",
        "V is NOT read from DRAM in MLA mode (`reuse_k=true`): V reuses K's L1 buffer.",
        "Only K cache DRAM reads are counted.",
        "",
        "| Case | Seq Len | B | G |"
        " Theo Logical (MB) | Theo Tiled (MB) | Theo Ideal (MB) |"
        " Measured DRAM Read (MB) | Meas / Tiled | Redundancy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for theo in theoretical:
        case_name = theo["case"]
        m = measured.get(case_name, {}).get("aggregate", {})
        m_dram = m.get("dram_read_MB", "n/a")

        if isinstance(m_dram, (int, float)) and theo["total_actual_tiled_MB"] > 0:
            ratio = f"{m_dram / theo['total_actual_tiled_MB']:.2f}x"
        else:
            ratio = "n/a"

        lines.append(
            f"| {case_name} | {theo['seq_len']} | {theo['batch']} "
            f"| {theo['n_head_batches']} "
            f"| {theo['total_actual_logical_MB']} "
            f"| {theo['total_actual_tiled_MB']} "
            f"| {theo['total_ideal_tiled_MB']} "
            f"| {m_dram} "
            f"| {ratio} | {theo['redundancy_factor']}x |"
        )

    lines.append("")
    lines.append("**Notes:**")
    lines.append("- *Theo Logical*: logical K bytes = B × G × L × (d_c+d_r) × 1B (BF8_b)")
    lines.append("- *Theo Tiled*: K bytes with tile layout overhead (BF8_b tile ≈ 1088B)")
    lines.append("- *Theo Ideal*: single-pass K read with multicast (G=1)")
    lines.append("- *Measured DRAM Read*: sum of `num_bytes` for NoC READ events targeting DRAM bank coordinates")
    lines.append("- *Meas / Tiled*: ratio close to 1.0 confirms the theoretical model; >1 includes page table + Q reads")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Measure FlashMLA DRAM read traffic")
    parser.add_argument(
        "--run", action="store_true",
        help="Run FlashMLA with NoC trace (requires device)",
    )
    parser.add_argument(
        "--parse-only", action="store_true",
        help="Only parse existing traces (no device run)",
    )
    parser.add_argument(
        "--trace-dir", type=Path,
        help="Path to trace output directory (for --parse-only)",
    )
    parser.add_argument(
        "--cases", nargs="+", default=DEFAULT_CASES,
        choices=sorted(DECODE_CASES.keys()),
        help="Decode cases to run/parse",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_run:
        execute_child_run(args.case)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Compute theoretical bounds for all cases
    print("Computing theoretical DRAM read bounds...")
    theoretical = []
    for case_name in args.cases:
        theo = compute_theoretical(DECODE_CASES[case_name])
        theoretical.append(theo)
        print(
            f"  {case_name}: actual={theo['total_actual_logical_MB']:.3f} MB, "
            f"ideal={theo['total_ideal_logical_MB']:.3f} MB, "
            f"redundancy={theo['redundancy_factor']}x"
        )

    # Run with NoC trace if requested
    measured = {}
    if args.run:
        run_output = args.output_dir / "traces"
        print(f"\nRunning FlashMLA with NoC event tracing...")
        for case_name in args.cases:
            try:
                logs_dir = run_with_noc_trace(case_name, run_output)
                result = parse_all_traces(logs_dir)
                measured[case_name] = result
                agg = result["aggregate"]
                print(
                    f"  {case_name}: total_read={agg.get('total_read_MB', 0):.3f} MB, "
                    f"dram_read={agg.get('dram_read_MB', 0):.3f} MB, "
                    f"n_files={result['n_trace_files']}"
                )
            except Exception as e:
                print(f"  ERROR running {case_name}: {e}")

    elif args.parse_only:
        trace_base = args.trace_dir or (args.output_dir / "traces")
        print(f"\nParsing NoC traces from {trace_base}...")
        for case_name in args.cases:
            logs_dir = trace_base / case_name / ".logs"
            if not logs_dir.exists():
                print(f"  SKIP {case_name}: {logs_dir} not found")
                continue
            result = parse_all_traces(logs_dir)
            measured[case_name] = result
            agg = result["aggregate"]
            print(
                f"  {case_name}: total_read={agg.get('total_read_MB', 0):.3f} MB, "
                f"dram_read={agg.get('dram_read_MB', 0):.3f} MB, "
                f"n_files={result['n_trace_files']}"
            )

    # Generate comparison table
    table = generate_comparison_table(theoretical, measured)
    table_path = args.output_dir / "dram_traffic_comparison.md"
    table_path.write_text(table)
    print(f"\nComparison table: {table_path}")

    # Save raw data
    raw_path = args.output_dir / "dram_traffic_raw.json"
    raw_data = {
        "theoretical": theoretical,
        "measured": {k: v for k, v in measured.items()},
    }
    raw_path.write_text(json.dumps(raw_data, indent=2, default=str))
    print(f"Raw data: {raw_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DRAM Traffic Summary (K-only; V reuses K L1 buffer in MLA mode)")
    print("=" * 70)
    for theo in theoretical:
        cn = theo["case"]
        m = measured.get(cn, {}).get("aggregate", {})
        m_dram = m.get("dram_read_MB")
        print(f"\n  {cn} (seq={theo['seq_len']}, batch={theo['batch']}, G={theo['n_head_batches']}):")
        print(f"    K logical (no mcast):   {theo['total_actual_logical_MB']:.3f} MB")
        print(f"    K tiled  (no mcast):    {theo['total_actual_tiled_MB']:.3f} MB")
        print(f"    K tiled  (ideal mcast): {theo['total_ideal_tiled_MB']:.3f} MB")
        if m_dram is not None:
            print(f"    Measured DRAM read:      {m_dram:.3f} MB")
            if theo["total_actual_tiled_MB"] > 0:
                ratio = m_dram / theo["total_actual_tiled_MB"]
                print(f"    Meas / Tiled:           {ratio:.2f}x")
        else:
            print(f"    Measured: (not available - run with --run)")

    print()


if __name__ == "__main__":
    main()

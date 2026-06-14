"""Architecture parameter sweep for §7.5: how hardware knobs affect MLA decode.

Sweeps four independent parameters while holding others at Wormhole defaults:
  1. L1 SRAM capacity
  2. DRAM bandwidth
  3. NoC bandwidth
  4. Core count (grid size)

Produces JSON and prints LaTeX-ready tables for each sweep.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from mla_flash_attention_dev.autotuner.basic_autotuner import (
    BasicMLAAutotuner,
    HardwareTopologySpec,
    SearchSpace,
    get_preset_hardware,
    get_preset_workload,
)
from mla_flash_attention_dev.autotuner.cost_model import CalibratedCostModel

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_JSON = (
    EXPERIMENTS_ROOT
    / "profile_outputs"
    / "flash_mla_wh_detailed"
    / "flash_mla_wh_detailed_calibration.json"
)
OUT_DIR = Path(__file__).resolve().parent

TARGET_SEQ = 32768


def _extract_top1(result) -> dict:
    m = result.best_plan.metrics
    c = result.best_plan.compile_time_config
    return {
        "latency_ms": round(m.selected_latency_ms, 6),
        "dram_ms": round(m.dram_ms, 6),
        "noc_ms": round(m.noc_ms, 6),
        "compute_ms": round(m.compute_ms, 6),
        "reader_bp_ms": round(m.reader_backpressure_ms, 6),
        "writer_bp_ms": round(m.writer_backpressure_ms, 6),
        "l1_usage_ratio": round(m.l1_usage_ratio, 4),
        "k_chunk": c.k_chunk_size,
        "pipeline_depth": c.pipeline_depth,
        "head_parallel": result.best_plan.parallelism_5d.head_parallel_factor,
        "kv_parallel": result.best_plan.parallelism_5d.kv_parallel_factor,
    }


def _dominant_bottleneck(info: dict) -> str:
    components = {
        "DRAM": info["dram_ms"],
        "NoC": info["noc_ms"],
        "compute": info["compute_ms"],
        "coupling": info["reader_bp_ms"] + info["writer_bp_ms"],
    }
    return max(components, key=components.get)


def sweep_sram(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    sram_points = [
        (384 * 1024, "384 KB (0.25x)"),
        (768 * 1024, "768 KB (0.5x)"),
        (1_499_136,  "1.5 MB (1x, WH)"),
        (3 * 1024 * 1024, "3 MB (2x)"),
        (6 * 1024 * 1024, "6 MB (4x)"),
    ]
    results = []
    workload = replace(base_workload, name="arch_sweep_sram", seq_len_kv=TARGET_SEQ)

    for sram_bytes, label in sram_points:
        hw = replace(base_hw, name=f"sram_{label}", l1_bytes_per_core=sram_bytes)
        r = tuner.tune(workload, hw, top_k=20000, calibrated_cost_model=cal_model)
        n = r.searched_candidate_count
        if n == 0:
            results.append({"label": label, "n_feasible": 0, "top1": None})
            continue

        all_bkv = set()
        for c in r.top_candidates:
            all_bkv.add(c.compile_time_config.k_chunk_size)

        r1 = tuner.tune(workload, hw, top_k=1, calibrated_cost_model=cal_model)
        info = _extract_top1(r1)
        results.append({
            "label": label,
            "n_feasible": n,
            "max_bkv": max(all_bkv),
            "top1": info,
        })
        print(f"  SRAM {label:>18}: n={n:>5} max_Bkv={max(all_bkv):>4} lat={info['latency_ms']:.4f}")

    return results


def sweep_dram_bw(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    dram_points = [
        (128.0, "128 GB/s (0.5x)"),
        (258.0, "258 GB/s (1x, WH)"),
        (512.0, "512 GB/s (2x, BH)"),
        (1024.0, "1024 GB/s (4x)"),
        (2048.0, "2048 GB/s (8x)"),
    ]
    results = []
    workload = replace(base_workload, name="arch_sweep_dram", seq_len_kv=TARGET_SEQ)

    for bw, label in dram_points:
        hw = replace(base_hw, name=f"dram_{label}", dram_bandwidth_GBs=bw)
        r1 = tuner.tune(workload, hw, top_k=1, calibrated_cost_model=cal_model)
        info = _extract_top1(r1)
        info["dominant"] = _dominant_bottleneck(info)
        results.append({"label": label, "top1": info})
        print(f"  DRAM {label:>20}: lat={info['latency_ms']:.4f} dominant={info['dominant']}")

    return results


def sweep_noc_bw(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    noc_points = [
        (16.0,  "16 GB/s (0.5x)"),
        (32.0,  "32 GB/s (1x, WH)"),
        (64.0,  "64 GB/s (2x)"),
        (128.0, "128 GB/s (4x)"),
        (256.0, "256 GB/s (8x)"),
    ]
    results = []
    workload = replace(base_workload, name="arch_sweep_noc", seq_len_kv=TARGET_SEQ)

    for bw, label in noc_points:
        hw = replace(base_hw, name=f"noc_{label}", noc_bandwidth_GBs=bw)
        r1 = tuner.tune(workload, hw, top_k=1, calibrated_cost_model=cal_model)
        info = _extract_top1(r1)
        total = info["latency_ms"]
        noc_share = (info["noc_ms"] / total * 100) if total > 0 else 0
        info["noc_share_pct"] = round(noc_share, 1)
        results.append({"label": label, "top1": info})
        print(f"  NoC  {label:>20}: lat={info['latency_ms']:.4f} noc_share={noc_share:.1f}%")

    return results


def sweep_core_count(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    grid_points = [
        (4, 4,   "4x4 (16)"),
        (6, 6,   "6x6 (36)"),
        (8, 8,   "8x8 (64, WH)"),
        (10, 10, "10x10 (100)"),
        (16, 16, "16x16 (256)"),
    ]
    results = []
    workload = replace(base_workload, name="arch_sweep_cores", seq_len_kv=TARGET_SEQ)

    for gx, gy, label in grid_points:
        hw = replace(base_hw, name=f"cores_{label}",
                     compute_grid_x=gx, compute_grid_y=gy)
        r1 = tuner.tune(workload, hw, top_k=1, calibrated_cost_model=cal_model)
        info = _extract_top1(r1)
        results.append({"label": label, "top1": info})
        print(f"  Cores {label:>16}: lat={info['latency_ms']:.4f}")

    return results


def main() -> int:
    base_workload = get_preset_workload("flash_decode_wh")
    base_hw = get_preset_hardware("wormhole_b0")
    extended_space = SearchSpace(k_chunk_sizes=(64, 128, 256, 512, 1024))
    tuner = BasicMLAAutotuner(search_space=extended_space)

    cal_model = None
    if CALIBRATION_JSON.exists():
        cal_model = CalibratedCostModel.load(str(CALIBRATION_JSON))
        print(f"Loaded calibration from {CALIBRATION_JSON}")
    else:
        print("No calibration file found; using analytical path.")

    print(f"\n{'='*60}")
    print(f"Architecture Sweep @ seq_len={TARGET_SEQ}")
    print(f"{'='*60}")

    print("\n[1/4] SRAM Capacity Sweep")
    sram_results = sweep_sram(tuner, base_hw, base_workload, cal_model)

    print("\n[2/4] DRAM Bandwidth Sweep")
    dram_results = sweep_dram_bw(tuner, base_hw, base_workload, cal_model)

    print("\n[3/4] NoC Bandwidth Sweep")
    noc_results = sweep_noc_bw(tuner, base_hw, base_workload, cal_model)

    print("\n[4/4] Core Count Sweep")
    core_results = sweep_core_count(tuner, base_hw, base_workload, cal_model)

    all_results = {
        "target_seq_len": TARGET_SEQ,
        "baseline": "wormhole_b0",
        "sram_sweep": sram_results,
        "dram_bw_sweep": dram_results,
        "noc_bw_sweep": noc_results,
        "core_count_sweep": core_results,
    }

    out_path = OUT_DIR / "architecture_sweep_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")

    _print_summary(sram_results, dram_results, noc_results, core_results)
    return 0


def _print_summary(sram, dram, noc, cores):
    baseline_lat = None
    for r in dram:
        if "1x" in r["label"]:
            baseline_lat = r["top1"]["latency_ms"]
            break

    print(f"\n{'='*60}")
    print("SUMMARY (normalized to WH baseline)")
    print(f"{'='*60}")

    if baseline_lat:
        print(f"\nBaseline latency @ 32K: {baseline_lat:.4f} ms")

        print("\nSRAM sweep:")
        for r in sram:
            if r["top1"] is None:
                print(f"  {r['label']:>18}: no feasible candidates")
            else:
                lat = r["top1"]["latency_ms"]
                ratio = lat / baseline_lat
                print(f"  {r['label']:>18}: {lat:.4f} ms ({ratio:.2f}x baseline)")

        print("\nDRAM BW sweep:")
        for r in dram:
            lat = r["top1"]["latency_ms"]
            speedup = baseline_lat / lat
            print(f"  {r['label']:>20}: {lat:.4f} ms ({speedup:.2f}x speedup) [{r['top1']['dominant']}]")

        print("\nNoC BW sweep:")
        for r in noc:
            lat = r["top1"]["latency_ms"]
            speedup = baseline_lat / lat
            print(f"  {r['label']:>20}: {lat:.4f} ms ({speedup:.2f}x) noc_share={r['top1']['noc_share_pct']}%")

        print("\nCore count sweep:")
        for r in cores:
            lat = r["top1"]["latency_ms"]
            speedup = baseline_lat / lat
            print(f"  {r['label']:>16}: {lat:.4f} ms ({speedup:.2f}x speedup)")


if __name__ == "__main__":
    raise SystemExit(main())

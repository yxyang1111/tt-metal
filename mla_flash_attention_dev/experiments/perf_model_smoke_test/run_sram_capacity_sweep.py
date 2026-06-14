"""Sweep per-core L1 SRAM capacity to study its effect on the MLA design space.

For each SRAM size, run the full DSE pipeline (enumerate → filter → score → rank)
and record:
  - Number of feasible candidates
  - Maximum feasible B_kv and G_head
  - Optimal configuration and predicted latency at representative sequence lengths
  - Bottleneck decomposition

This supports the architecture exploration subsection (§7.5) showing how SRAM
capacity shapes the design space and where diminishing returns set in.
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

SEQ_LENS = [1024, 4096, 16384, 32768]

SRAM_SWEEP_BYTES = [
    384 * 1024,       # 384 KB — 0.25× current
    768 * 1024,       # 768 KB — 0.5× current
    1_499_136,        # ~1.43 MB — current WH
    3 * 1024 * 1024,  # 3 MB — 2× current
    6 * 1024 * 1024,  # 6 MB — 4× current
    12 * 1024 * 1024, # 12 MB — 8× current
]

SRAM_LABELS = [
    "384 KB",
    "768 KB",
    "1.43 MB (WH)",
    "3 MB",
    "6 MB",
    "12 MB",
]


def _extract_top1_info(result) -> dict:
    m = result.best_plan.metrics
    c = result.best_plan.compile_time_config
    p = result.best_plan.parallelism_5d
    return {
        "selected_latency_ms": round(m.selected_latency_ms, 6),
        "dram_ms": round(m.dram_ms, 6),
        "noc_ms": round(m.noc_ms, 6),
        "compute_ms": round(m.compute_ms, 6),
        "reader_ms": round(m.reader_ms, 6),
        "reduce_ms": round(m.reduction_ms, 6),
        "reader_bp_ms": round(m.reader_backpressure_ms, 6),
        "writer_bp_ms": round(m.writer_backpressure_ms, 6),
        "overlap_residual_ms": round(m.overlap_residual_ms, 6),
        "l1_usage_ratio": round(m.l1_usage_ratio, 4),
        "l1_usage_bytes": m.l1_usage_bytes_per_core,
        "active_cores": m.active_cores,
        "provisioned_cores": m.provisioned_cores,
        "head_parallel": p.head_parallel_factor,
        "kv_parallel": p.kv_parallel_factor,
        "k_chunk": c.k_chunk_size,
        "pipeline_depth": c.pipeline_depth,
        "topology": c.topology_mode,
    }


def main() -> int:
    base_workload = get_preset_workload("flash_decode_wh")
    base_hw = get_preset_hardware("wormhole_b0")
    extended_space = SearchSpace(k_chunk_sizes=(64, 128, 256, 512, 1024))
    tuner = BasicMLAAutotuner(search_space=extended_space)

    cal_model = None
    if CALIBRATION_JSON.exists():
        cal_model = CalibratedCostModel.load(str(CALIBRATION_JSON))
        print(f"Loaded calibration model from {CALIBRATION_JSON}")
    else:
        print(f"No calibration file at {CALIBRATION_JSON}; using analytical path only.")

    results: list[dict] = []

    for sram_bytes, sram_label in zip(SRAM_SWEEP_BYTES, SRAM_LABELS):
        hw = replace(base_hw, name=f"wh_sram_{sram_label}", l1_bytes_per_core=sram_bytes)

        for seq_len in SEQ_LENS:
            workload = replace(
                base_workload,
                name=f"flash_decode_sram_sweep_s{seq_len}",
                seq_len_kv=seq_len,
            )

            r_all = tuner.tune(workload, hw, top_k=20000)
            n_feasible = r_all.searched_candidate_count

            if n_feasible == 0:
                row = {
                    "sram_label": sram_label,
                    "sram_bytes": sram_bytes,
                    "seq_len": seq_len,
                    "n_feasible": 0,
                    "top1": None,
                }
                results.append(row)
                print(f"SRAM={sram_label:>12}  S={seq_len:>6}  n_feasible=0  (no feasible candidates)")
                continue

            all_k_chunks = set()
            all_head_pars = set()
            for c in r_all.top_candidates:
                all_k_chunks.add(c.compile_time_config.k_chunk_size)
                all_head_pars.add(c.parallelism_5d.head_parallel_factor)

            r_top1 = tuner.tune(workload, hw, top_k=1)
            top1_info = _extract_top1_info(r_top1)

            r_cal_top1 = None
            cal_top1_info = None
            if cal_model is not None:
                r_cal_top1 = tuner.tune(workload, hw, top_k=1, calibrated_cost_model=cal_model)
                cal_top1_info = _extract_top1_info(r_cal_top1)

            row = {
                "sram_label": sram_label,
                "sram_bytes": sram_bytes,
                "seq_len": seq_len,
                "n_feasible": n_feasible,
                "max_k_chunk": max(all_k_chunks),
                "k_chunk_set": sorted(all_k_chunks),
                "head_parallel_set": sorted(all_head_pars),
                "top1_analytical": top1_info,
            }
            if cal_top1_info is not None:
                row["top1_calibrated"] = cal_top1_info
            results.append(row)

            hp = top1_info["head_parallel"]
            bkv = top1_info["k_chunk"]
            lat = top1_info["selected_latency_ms"]
            cal_str = ""
            if cal_top1_info:
                cal_str = f"  cal_lat={cal_top1_info['selected_latency_ms']:.4f} cal_Bkv={cal_top1_info['k_chunk']}"
            print(
                f"SRAM={sram_label:>12}  S={seq_len:>6}  "
                f"n={n_feasible:>6}  max_Bkv={max(all_k_chunks):>4}  "
                f"top1: lat={lat:.4f} Ghead(hp)={hp} Bkv={bkv}{cal_str}"
            )

    out_json = OUT_DIR / "sram_capacity_sweep.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")

    _print_latex_table(results)

    return 0


def _print_latex_table(results: list[dict]) -> None:
    """Print a LaTeX-ready summary table to stdout."""
    print("\n" + "=" * 80)
    print("LaTeX table data (analytical path):")
    print("=" * 80)

    header = f"{'SRAM':>12} | {'Seq Len':>8} | {'Feasible':>8} | {'MaxBkv':>6} | {'OptBkv':>6} | {'Ghead':>5} | {'Latency':>10} | {'L1 Ratio':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        sram = r["sram_label"]
        seq = r["seq_len"]
        n = r["n_feasible"]
        if r.get("top1_analytical") is None:
            print(f"{sram:>12} | {seq:>8} | {n:>8} | {'--':>6} | {'--':>6} | {'--':>5} | {'--':>10} | {'--':>8}")
            continue
        t1 = r["top1_analytical"]
        max_bkv = r["max_k_chunk"]
        opt_bkv = t1["k_chunk"]
        hp = t1["head_parallel"]
        lat = t1["selected_latency_ms"]
        l1r = t1["l1_usage_ratio"]
        print(f"{sram:>12} | {seq:>8} | {n:>8} | {max_bkv:>6} | {opt_bkv:>6} | {hp:>5} | {lat:>10.4f} | {l1r:>8.3f}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Sweep seq_len_kv on WH decode preset with BOTH analytical and calibrated paths.

Extends run_seq_sweep.py by adding the calibrated cost model (corrected T path)
so we can compare how the optimal configuration changes with calibration at each
sequence length.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from mla_flash_attention_dev.autotuner.basic_autotuner import (
    BasicMLAAutotuner,
    get_preset_hardware,
    get_preset_workload,
)
from mla_flash_attention_dev.autotuner.cost_model import CalibratedCostModel

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_JSON = (
    EXPERIMENTS_ROOT
    / "profiling" / "outputs"
    / "flash_mla_wh_detailed"
    / "flash_mla_wh_detailed_calibration.json"
)
OUT_DIR = Path(__file__).resolve().parent
SWEEP = [1024, 2048, 4096, 8192, 16384, 32768]


def _extract_row(seq_len: int, path_label: str, result) -> dict:
    m = result.best_plan.metrics
    c = result.best_plan.compile_time_config
    p = result.best_plan.parallelism_5d
    return {
        "seq_len_kv": seq_len,
        "path": path_label,
        "searched_candidates": result.searched_candidate_count,
        "selected_latency_ms": round(m.selected_latency_ms, 6),
        "estimated_ms": round(m.estimated_latency_ms, 6),
        "dram_ms": round(m.dram_ms, 6),
        "noc_ms": round(m.noc_ms, 6),
        "compute_ms": round(m.compute_ms, 6),
        "reader_ms": round(m.reader_ms, 6),
        "reduce_ms": round(m.reduction_ms, 6),
        "overlap_residual_ms": round(m.overlap_residual_ms, 6),
        "active_cores": f"{m.active_cores}/{m.provisioned_cores}",
        "kv_parallel": p.kv_parallel_factor,
        "head_parallel": p.head_parallel_factor,
        "q_chunk": c.q_chunk_size,
        "k_chunk": c.k_chunk_size,
        "k_page_bytes": c.k_page_size_bytes,
        "topology": c.topology_mode,
        "pipeline_depth": c.pipeline_depth,
    }


def main() -> int:
    base = get_preset_workload("flash_decode_wh")
    hw = get_preset_hardware("wormhole_b0")
    tuner = BasicMLAAutotuner()
    cal_model = CalibratedCostModel.load(str(CALIBRATION_JSON))

    rows: list[dict] = []
    for s in SWEEP:
        workload = replace(base, name=f"flash_decode_wh_s{s}", seq_len_kv=s)

        r_analytical = tuner.tune(workload, hw, top_k=1)
        r_calibrated = tuner.tune(
            workload, hw, top_k=1, calibrated_cost_model=cal_model
        )

        row_a = _extract_row(s, "analytical", r_analytical)
        row_c = _extract_row(s, "calibrated", r_calibrated)
        rows.extend([row_a, row_c])

        print(
            f"S={s:>6}  T0={row_a['selected_latency_ms']:.4f} ms  "
            f"T={row_c['selected_latency_ms']:.4f} ms  "
            f"a_topo={row_a['topology']}  c_topo={row_c['topology']}  "
            f"a_kchunk={row_a['k_chunk']}  c_kchunk={row_c['k_chunk']}  "
            f"a_pipe={row_a['pipeline_depth']}  c_pipe={row_c['pipeline_depth']}  "
            f"a_cores={row_a['active_cores']}  c_cores={row_c['active_cores']}"
        )

    out_json = OUT_DIR / "seq_sweep_wh_decode_both_paths.json"
    out_csv = OUT_DIR / "seq_sweep_wh_decode_both_paths.csv"

    out_json.write_text(json.dumps(rows, indent=2))
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out_json}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

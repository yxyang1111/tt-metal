"""Analyze the full candidate space at representative sequence lengths.

For each sequence length, enumerate ALL feasible candidates, score them with both
the analytical and calibrated cost models, and output statistics about the
latency distribution. This data supports the argument that the autotuner's
top-1 selection is genuinely optimal within the space.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

from mla_flash_attention_dev.autotuner.basic_autotuner import (
    BasicMLAAutotuner,
    get_preset_hardware,
    get_preset_workload,
)
from mla_flash_attention_dev.autotuner.cost_model import (
    StratifiedCalibratedCostModel,
    apply_calibration_to_candidate,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_JSON = (
    EXPERIMENTS_ROOT
    / "profile_outputs"
    / "flash_mla_wh_detailed"
    / "flash_mla_wh_detailed_calibration.json"
)
OUT_DIR = Path(__file__).resolve().parent
SWEEP = [1024, 4096, 16384, 32768]


def main() -> int:
    base = get_preset_workload("flash_decode_wh")
    hw = get_preset_hardware("wormhole_b0")
    tuner = BasicMLAAutotuner()
    cal_model = StratifiedCalibratedCostModel.load(str(CALIBRATION_JSON))

    results: list[dict] = []

    for s in SWEEP:
        workload = replace(base, name=f"flash_decode_wh_s{s}", seq_len_kv=s)

        # Get ALL candidates (set top_k very high)
        r_analytical = tuner.tune(workload, hw, top_k=20000)
        r_calibrated = tuner.tune(
            workload, hw, top_k=20000, calibrated_cost_model=cal_model
        )

        # Extract all analytical latencies
        a_latencies = [
            c.metrics.selected_latency_ms for c in r_analytical.top_candidates
        ]
        c_latencies = [
            c.metrics.selected_latency_ms for c in r_calibrated.top_candidates
        ]

        a_latencies_sorted = sorted(a_latencies)
        c_latencies_sorted = sorted(c_latencies)

        n = len(a_latencies_sorted)

        # Top-1, top-5, percentiles
        def pct(lst, p):
            idx = int(len(lst) * p / 100)
            idx = min(idx, len(lst) - 1)
            return lst[idx]

        row = {
            "seq_len_kv": s,
            "total_candidates": n,
            # Analytical path
            "a_top1_ms": round(a_latencies_sorted[0], 6),
            "a_top5_ms": round(a_latencies_sorted[min(4, n - 1)], 6),
            "a_top10_ms": round(a_latencies_sorted[min(9, n - 1)], 6),
            "a_median_ms": round(statistics.median(a_latencies_sorted), 6),
            "a_p90_ms": round(pct(a_latencies_sorted, 90), 6),
            "a_worst_ms": round(a_latencies_sorted[-1], 6),
            "a_mean_ms": round(statistics.mean(a_latencies_sorted), 6),
            # Calibrated path
            "c_top1_ms": round(c_latencies_sorted[0], 6),
            "c_top5_ms": round(c_latencies_sorted[min(4, n - 1)], 6),
            "c_top10_ms": round(c_latencies_sorted[min(9, n - 1)], 6),
            "c_median_ms": round(statistics.median(c_latencies_sorted), 6),
            "c_p90_ms": round(pct(c_latencies_sorted, 90), 6),
            "c_worst_ms": round(c_latencies_sorted[-1], 6),
            "c_mean_ms": round(statistics.mean(c_latencies_sorted), 6),
            # Ratios
            "a_median_over_top1": round(
                statistics.median(a_latencies_sorted) / a_latencies_sorted[0], 2
            ),
            "c_median_over_top1": round(
                statistics.median(c_latencies_sorted) / c_latencies_sorted[0], 2
            ),
            # Full sorted latency arrays for plotting
            "a_all_latencies_ms": [round(x, 6) for x in a_latencies_sorted],
            "c_all_latencies_ms": [round(x, 6) for x in c_latencies_sorted],
        }
        results.append(row)

        print(
            f"S={s:>6}  n={n}  "
            f"a_top1={row['a_top1_ms']:.4f}  a_median={row['a_median_ms']:.4f}  "
            f"a_worst={row['a_worst_ms']:.4f}  a_med/top1={row['a_median_over_top1']:.1f}x  |  "
            f"c_top1={row['c_top1_ms']:.4f}  c_median={row['c_median_ms']:.4f}  "
            f"c_worst={row['c_worst_ms']:.4f}  c_med/top1={row['c_median_over_top1']:.1f}x"
        )

    out_json = OUT_DIR / "candidate_space_analysis.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sweep seq_len_kv on WH decode preset, call the analytical cost model directly.

Dumps a compact CSV + JSON with the best-plan estimated_latency breakdown per S.
Used as a smoke-test that the perf model is live and responds to input changes.
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


OUT_DIR = Path(__file__).resolve().parent
SWEEP = [1024, 2048, 4096, 8192, 16384, 32768]


def main() -> int:
    base = get_preset_workload("flash_decode_wh")
    hw = get_preset_hardware("wormhole_b0")
    tuner = BasicMLAAutotuner()

    rows: list[dict] = []
    for s in SWEEP:
        workload = replace(base, name=f"flash_decode_wh_s{s}", seq_len_kv=s)
        result = tuner.tune(workload, hw, top_k=1)
        m = result.best_plan.metrics
        c = result.best_plan.compile_time_config
        p = result.best_plan.parallelism_5d
        rows.append({
            "seq_len_kv": s,
            "searched_candidates": result.searched_candidate_count,
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
        })
        print(
            f"S={s:>6}  est={m.estimated_latency_ms:.4f} ms  "
            f"dram={m.dram_ms:.4f}  compute={m.compute_ms:.4f}  "
            f"reader={m.reader_ms:.4f}  reduce={m.reduction_ms:.4f}  "
            f"KV={p.kv_parallel_factor} H={p.head_parallel_factor}  "
            f"cands={result.searched_candidate_count}"
        )

    (OUT_DIR / "seq_sweep_wh_decode.json").write_text(json.dumps(rows, indent=2))
    with (OUT_DIR / "seq_sweep_wh_decode.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_DIR / 'seq_sweep_wh_decode.json'}")
    print(f"wrote {OUT_DIR / 'seq_sweep_wh_decode.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

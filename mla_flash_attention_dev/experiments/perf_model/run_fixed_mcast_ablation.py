"""Same-parameter multicast ablation for prefill and decode.

The original multicast ablation re-tuned the no-multicast variant, which
mixes the effect of disabling multicast with changes in grid shape.  This
script keeps the selected parallelism, chunk sizes, pipeline depth, buffers,
and math fidelity fixed, then rebuilds only the topology-dependent part of the
candidate to estimate the isolated multicast effect.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from mla_flash_attention_dev.autotuner.basic_autotuner import (
    BasicMLAAutotuner,
    PlanCandidate,
    get_preset_hardware,
    get_preset_workload,
    get_topology_preset,
)

OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "fixed_mcast_ablation_results.json"

SEQ_LENS = {
    "decode": [1024, 2048, 4096, 8192, 16384, 32768],
    "prefill": [1024, 2048, 4096, 8192, 16384],
}


def _topology_for(candidate: PlanCandidate):
    name = candidate.compile_time_config.sblock_topology_name
    return get_topology_preset(name) if name else None


def _workload_for(mode: str, seq_len: int):
    preset = "flash_decode_wh" if mode == "decode" else "mla_prefill_wh"
    base = get_preset_workload(preset)
    return replace(
        base,
        name=f"fixed_mcast_{mode}_{seq_len}",
        mode=mode,
        causal=(mode == "prefill"),
        seq_len_q=1 if mode == "decode" else seq_len,
        seq_len_kv=seq_len,
    )


def _candidate_summary(candidate: PlanCandidate) -> dict[str, Any]:
    metrics = candidate.metrics
    return {
        "latency_ms": round(metrics.selected_latency_ms, 6),
        "dram_ms": round(metrics.dram_ms, 6),
        "noc_ms": round(metrics.noc_ms, 6),
        "reader_ms": round(metrics.reader_ms, 6),
        "compute_ms": round(metrics.compute_ms, 6),
        "reduction_ms": round(metrics.reduction_ms, 6),
        "reader_backpressure_ms": round(metrics.reader_backpressure_ms, 6),
        "writer_backpressure_ms": round(metrics.writer_backpressure_ms, 6),
        "sender_hotspot_ms": round(metrics.sender_hotspot_ms, 6),
        "overlap_residual_ms": round(metrics.overlap_residual_ms, 6),
        "dram_bytes": int(round(metrics.dram_bytes)),
        "noc_bytes": int(round(metrics.noc_bytes)),
        "dram_mb": round(metrics.dram_bytes / 1e6, 3),
        "noc_mb": round(metrics.noc_bytes / 1e6, 3),
        "active_cores": metrics.active_cores,
        "k_reuse_groups": metrics.k_reuse_groups,
        "l1_usage_ratio": round(metrics.l1_usage_ratio, 4),
        "q_num_chunks": metrics.q_num_chunks,
        "kv_num_chunks": metrics.kv_num_chunks,
    }


def _fixed_pair(tuner: BasicMLAAutotuner, mode: str, seq_len: int):
    hardware = get_preset_hardware("wormhole_b0")
    workload = _workload_for(mode, seq_len)
    best = tuner.tune(workload, hardware, top_k=1).best_plan
    parallelism = best.parallelism_5d

    if mode == "decode":
        mcast_cfg = best.compile_time_config
        mcast_topology = _topology_for(best)
        no_mcast_cfg = replace(
            mcast_cfg,
            topology_mode="independent",
            sblock_topology_name=None,
            bank_map=(),
            tree_order=(),
        )
        no_mcast_topology = None
    else:
        # Prefill is searched as independent in the current autotuner.  For the
        # what-if we add a multicast topology bit while preserving every other
        # selected knob, so K reuse shifts from repeated DRAM reads to NoC.
        no_mcast_cfg = best.compile_time_config
        no_mcast_topology = None
        mcast_cfg = replace(
            no_mcast_cfg,
            topology_mode="sblock_multicast",
            sblock_topology_name=None,
            bank_map=(),
            tree_order=(),
        )
        mcast_topology = None

    mcast = tuner._build_candidate(
        workload=workload,
        hardware=hardware,
        parallelism=parallelism,
        compile_time_config=mcast_cfg,
        topology=mcast_topology,
    )
    no_mcast = tuner._build_candidate(
        workload=workload,
        hardware=hardware,
        parallelism=parallelism,
        compile_time_config=no_mcast_cfg,
        topology=no_mcast_topology,
    )
    return best, mcast, no_mcast


def main() -> int:
    tuner = BasicMLAAutotuner()
    results: list[dict[str, Any]] = []

    for mode, seq_lens in SEQ_LENS.items():
        for seq_len in seq_lens:
            best, mcast, no_mcast = _fixed_pair(tuner, mode, seq_len)
            mcast_latency = mcast.metrics.selected_latency_ms
            no_mcast_latency = no_mcast.metrics.selected_latency_ms
            results.append(
                {
                    "mode": mode,
                    "seq_len": seq_len,
                    "seq_label": f"{seq_len // 1024}K",
                    "source": "analytical_fixed_config_what_if",
                    "speedup_no_mcast_over_mcast": round(no_mcast_latency / mcast_latency, 4),
                    "fixed_parallelism": asdict(best.parallelism_5d),
                    "fixed_compile_time_config": asdict(best.compile_time_config),
                    "mcast": _candidate_summary(mcast),
                    "no_mcast": _candidate_summary(no_mcast),
                }
            )
            print(
                f"{mode:7s} L={seq_len // 1024:>2}K "
                f"mcast={mcast_latency:.4f} ms "
                f"no_mcast={no_mcast_latency:.4f} ms "
                f"ratio={no_mcast_latency / mcast_latency:.2f}x"
            )

    payload = {
        "metadata": {
            "title": "Same-parameter multicast ablation",
            "hardware": "wormhole_b0",
            "model": "BasicMLAAutotuner analytical cost model",
            "note": (
                "This is a what-if model estimate, not an on-device kernel "
                "measurement.  Parallelism and compile-time knobs are fixed to "
                "the selected baseline candidate; only topology-dependent K "
                "reuse is toggled."
            ),
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

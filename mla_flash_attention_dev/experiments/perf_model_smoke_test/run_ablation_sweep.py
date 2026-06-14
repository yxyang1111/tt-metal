"""Ablation sweep for §7.4: isolating the contribution of each design decision.

Generates data for:
  7.4.1  Lane-wise K multicast vs no-multicast
  7.4.2  Grid shape (Ghead) at multiple sequence lengths
  7.4.3  K chunk size (Bkv) sweep
  7.4.4  Pipeline depth (D_pipe) sweep
  7.4.5  Topology and layout variants
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from mla_flash_attention_dev.autotuner.basic_autotuner import (
    BasicMLAAutotuner,
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


def _extract(result) -> dict:
    m = result.best_plan.metrics
    c = result.best_plan.compile_time_config
    p = result.best_plan.parallelism_5d
    return {
        "latency_ms": round(m.selected_latency_ms, 6),
        "dram_ms": round(m.dram_ms, 6),
        "noc_ms": round(m.noc_ms, 6),
        "compute_ms": round(m.compute_ms, 6),
        "reader_ms": round(m.reader_ms, 6),
        "reader_bp_ms": round(m.reader_backpressure_ms, 6),
        "writer_bp_ms": round(m.writer_backpressure_ms, 6),
        "reduce_ms": round(m.reduction_ms, 6),
        "l1_usage_ratio": round(m.l1_usage_ratio, 4),
        "l1_usage_bytes": m.l1_usage_bytes_per_core,
        "active_cores": m.active_cores,
        "k_chunk": c.k_chunk_size,
        "pipeline_depth": c.pipeline_depth,
        "topology": c.topology_mode,
        "head_parallel": p.head_parallel_factor,
        "kv_parallel": p.kv_parallel_factor,
    }


def sweep_bkv(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    """Sweep K chunk size at 32K, fixing other params at optimal."""
    bkv_values = [64, 128, 256, 512]
    results = []
    workload = replace(base_workload, name="ablation_bkv", seq_len_kv=32768)

    for bkv in bkv_values:
        space = SearchSpace(k_chunk_sizes=(bkv,))
        t = BasicMLAAutotuner(search_space=space)
        try:
            r = t.tune(workload, base_hw, top_k=1, calibrated_cost_model=cal_model)
        except RuntimeError:
            results.append({"bkv": bkv, "feasible": 0})
            print(f"  Bkv={bkv}: no feasible candidates")
            continue
        if r.searched_candidate_count == 0:
            results.append({"bkv": bkv, "feasible": 0})
            print(f"  Bkv={bkv}: no feasible candidates")
            continue
        info = _extract(r)
        n_chunks = 32768 // bkv
        l1_kb = info["l1_usage_bytes"] / 1024
        results.append({
            "bkv": bkv,
            "feasible": r.searched_candidate_count,
            "chunks_per_lane": n_chunks // info["kv_parallel"],
            "l1_footprint_kb": round(l1_kb, 1),
            **info,
        })
        print(f"  Bkv={bkv:>4}: lat={info['latency_ms']:.4f} chunks={n_chunks//info['kv_parallel']} L1={l1_kb:.1f}KB")

    return results


def sweep_ghead(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    """Sweep Ghead at 4K, 16K, 32K."""
    ghead_values = [4, 8, 16, 32]
    seq_lens = [4096, 16384, 32768]
    results = []

    for seq in seq_lens:
        workload = replace(base_workload, name=f"ablation_ghead_s{seq}", seq_len_kv=seq)
        for gh in ghead_values:
            space = SearchSpace(
                head_parallel_factors=(gh,),
                k_chunk_sizes=(64, 128, 256, 512),
            )
            t = BasicMLAAutotuner(search_space=space)
            try:
                r = t.tune(workload, base_hw, top_k=1, calibrated_cost_model=cal_model)
            except RuntimeError:
                results.append({"seq_len": seq, "ghead": gh, "feasible": 0})
                print(f"  S={seq:>5} Ghead={gh:>2}: no feasible candidates")
                continue
            if r.searched_candidate_count == 0:
                results.append({"seq_len": seq, "ghead": gh, "feasible": 0})
                print(f"  S={seq:>5} Ghead={gh:>2}: no feasible candidates")
                continue
            info = _extract(r)
            n_track = 32 // gh
            results.append({
                "seq_len": seq,
                "ghead": gh,
                "n_track": n_track,
                "n_lane": info["kv_parallel"],
                "feasible": r.searched_candidate_count,
                **info,
            })
            print(f"  S={seq:>5} Ghead={gh:>2}: lat={info['latency_ms']:.4f} N_t={n_track} N_l={info['kv_parallel']} Bkv={info['k_chunk']}")

    return results


def sweep_pipeline_depth(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    """Sweep pipeline depth at 32K."""
    depths = [1, 2, 3, 4]
    results = []
    workload = replace(base_workload, name="ablation_pipe", seq_len_kv=32768)

    for d in depths:
        space = SearchSpace(
            pipeline_depths=(d,),
            k_chunk_sizes=(64, 128, 256, 512),
        )
        t = BasicMLAAutotuner(search_space=space)
        try:
            r = t.tune(workload, base_hw, top_k=1, calibrated_cost_model=cal_model)
        except RuntimeError:
            results.append({"depth": d, "feasible": 0})
            print(f"  D_pipe={d}: no feasible candidates")
            continue
        if r.searched_candidate_count == 0:
            results.append({"depth": d, "feasible": 0})
            print(f"  D_pipe={d}: no feasible candidates")
            continue
        info = _extract(r)
        l1_kb = info["l1_usage_bytes"] / 1024
        results.append({
            "depth": d,
            "feasible": r.searched_candidate_count,
            "l1_per_worker_kb": round(l1_kb, 1),
            **info,
        })
        print(f"  D_pipe={d}: lat={info['latency_ms']:.4f} L1={l1_kb:.1f}KB Bkv={info['k_chunk']}")

    return results


def sweep_topology(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    """Sweep topology at 4K and 32K."""
    seq_lens = [4096, 32768]
    topologies = ["tree", "independent"]
    results = []

    for seq in seq_lens:
        workload = replace(base_workload, name=f"ablation_topo_s{seq}", seq_len_kv=seq)
        for topo in topologies:
            space = SearchSpace(
                topology_modes=(topo,),
                k_chunk_sizes=(64, 128, 256, 512),
            )
            t = BasicMLAAutotuner(search_space=space)
            try:
                r = t.tune(workload, base_hw, top_k=1, calibrated_cost_model=cal_model)
            except RuntimeError:
                results.append({"seq_len": seq, "topology": topo, "feasible": 0})
                print(f"  S={seq:>5} {topo:>12}: no feasible candidates")
                continue
            if r.searched_candidate_count == 0:
                results.append({"seq_len": seq, "topology": topo, "feasible": 0})
                print(f"  S={seq:>5} {topo:>12}: no feasible candidates")
                continue
            info = _extract(r)
            results.append({
                "seq_len": seq,
                "topology": topo,
                "feasible": r.searched_candidate_count,
                **info,
            })
            print(f"  S={seq:>5} {topo:>12}: lat={info['latency_ms']:.4f} Ghead={info['head_parallel']} Bkv={info['k_chunk']}")

    return results


def sweep_multicast(tuner, base_hw, base_workload, cal_model) -> list[dict]:
    """Compare multicast (full S-FMLA) vs no-multicast (Ghead=32, one track).

    No-multicast is approximated by setting Ghead=32 (all heads in one track,
    N_track=1), which eliminates the multicast benefit.
    """
    seq_lens = [4096, 16384, 32768]
    results = []

    for seq in seq_lens:
        workload = replace(base_workload, name=f"ablation_mcast_s{seq}", seq_len_kv=seq)

        space_full = SearchSpace(k_chunk_sizes=(64, 128, 256, 512))
        t_full = BasicMLAAutotuner(search_space=space_full)
        try:
            r_full = t_full.tune(workload, base_hw, top_k=1, calibrated_cost_model=cal_model)
            info_full = _extract(r_full)
        except RuntimeError:
            info_full = None

        space_no_mc = SearchSpace(
            head_parallel_factors=(32,),
            k_chunk_sizes=(64, 128, 256, 512),
        )
        t_no = BasicMLAAutotuner(search_space=space_no_mc)
        try:
            r_no = t_no.tune(workload, base_hw, top_k=1, calibrated_cost_model=cal_model)
            info_no = _extract(r_no)
        except RuntimeError:
            info_no = None

        results.append({
            "seq_len": seq,
            "full_sfmla": info_full,
            "no_multicast": info_no,
        })

        if info_full and info_no:
            speedup = info_no["latency_ms"] / info_full["latency_ms"]
            print(f"  S={seq:>5}: full={info_full['latency_ms']:.4f} no_mc={info_no['latency_ms']:.4f} speedup={speedup:.2f}x")
        else:
            print(f"  S={seq:>5}: {'no feasible' if not info_full else 'ok'} / {'no feasible' if not info_no else 'ok'}")

    return results


def main() -> int:
    import sys
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    base_workload = replace(
        get_preset_workload("flash_decode_wh"),
        batch_size=batch_size,
    )
    base_hw = get_preset_hardware("wormhole_b0")
    tuner = BasicMLAAutotuner()

    cal_model = None
    if CALIBRATION_JSON.exists():
        cal_model = CalibratedCostModel.load(str(CALIBRATION_JSON))
        print(f"Loaded calibration from {CALIBRATION_JSON}")
    else:
        print("No calibration file; using analytical path.")

    print(f"\n{'='*60}")
    print(f"Ablation Sweep (batch_size={batch_size})")
    print(f"{'='*60}")

    print("\n[1/5] Multicast Ablation")
    mcast = sweep_multicast(tuner, base_hw, base_workload, cal_model)

    print("\n[2/5] Ghead Sweep")
    ghead = sweep_ghead(tuner, base_hw, base_workload, cal_model)

    print("\n[3/5] Bkv Sweep")
    bkv = sweep_bkv(tuner, base_hw, base_workload, cal_model)

    print("\n[4/5] Pipeline Depth Sweep")
    pipe = sweep_pipeline_depth(tuner, base_hw, base_workload, cal_model)

    print("\n[5/5] Topology Sweep")
    topo = sweep_topology(tuner, base_hw, base_workload, cal_model)

    all_results = {
        "multicast_ablation": mcast,
        "ghead_sweep": ghead,
        "bkv_sweep": bkv,
        "pipeline_depth_sweep": pipe,
        "topology_sweep": topo,
    }

    suffix = f"_b{batch_size}" if batch_size != 1 else ""
    out_path = OUT_DIR / f"ablation_sweep_results{suffix}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

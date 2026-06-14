#!/usr/bin/env python3
"""
Plot batch-scaling characterization for each sequence length.
Generates one PDF per L value showing kernel time, throughput, utilization, and BW vs batch.
Also generates a combined overview figure.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "outputs" / "batch_seq_sweep" / "batch_seq_sweep_results.json"
FIG_DIR = SCRIPT_DIR.parents[1] / "latex" / "figures"

CHIP_PEAK_TFLOPS = 65.5
CHIP_PEAK_BW_GBS = 288.0

# MLA workload constants
D_C = 512
D_R = 64
D_QK = D_C + D_R
H = 32
N_HB = 4  # head groups per batch (H/DQHPC = 32/8)
N_SC = 4  # seq-parallel cores per head group
ELEMENT_BYTES = 2
NUM_CORES = 64

SEQ_LABELS = {1024: "1K", 4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K"}


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    return raw["results"]


def group_by_seq(results):
    groups = {}
    for r in results:
        sl = r["seq_len"]
        groups.setdefault(sl, []).append(r)
    for sl in groups:
        groups[sl].sort(key=lambda x: x["batch"])
    return groups


def compute_derived(records):
    for r in records:
        batch = r["batch"]
        seq_len = r["seq_len"]
        kernel_us = r["kernel_us"]
        ncrisc_us = r["ncrisc_us"]
        active_cores = r["active_cores"]

        # Traffic model: batch × 4 head groups × L × (D_QK+D_C) × elem_bytes
        total_traffic_bytes = batch * N_HB * seq_len * (D_QK + D_C) * ELEMENT_BYTES
        ideal_traffic_bytes = batch * seq_len * (D_QK + D_C) * ELEMENT_BYTES

        # "Modeled aggregate BW" = total_modeled_traffic / kernel_time
        r["modeled_agg_bw_gbs"] = total_traffic_bytes / (kernel_us * 1e3)  # bytes/µs = MB/s; /1e3 = GB/s
        r["ideal_agg_bw_gbs"] = ideal_traffic_bytes / (kernel_us * 1e3)

        # Per-core BW (based on NCRISC time)
        total_jobs = batch * N_HB * N_SC
        jobs_per_core = math.ceil(total_jobs / active_cores)
        per_core_bytes = jobs_per_core * (seq_len / N_SC) * (D_QK + D_C) * ELEMENT_BYTES
        r["per_core_bw_gbs"] = per_core_bytes / (ncrisc_us * 1e3) if ncrisc_us > 0 else 0

        # Implied redundancy factor = modeled_traffic / (288 × kernel_time)
        max_deliverable = CHIP_PEAK_BW_GBS * kernel_us * 1e3  # bytes
        r["implied_redundancy"] = total_traffic_bytes / max_deliverable

        # Chip utilization
        r["chip_util_pct"] = r["overall_tflops"] / CHIP_PEAK_TFLOPS * 100


def plot_per_seqlen(groups):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for seq_len, records in sorted(groups.items()):
        label = SEQ_LABELS.get(seq_len, str(seq_len))
        compute_derived(records)

        batches = [r["batch"] for r in records]
        kernel_us = [r["kernel_us"] for r in records]
        overall_tflops = [r["overall_tflops"] for r in records]
        chip_util = [r["chip_util_pct"] for r in records]
        modeled_bw = [r["modeled_agg_bw_gbs"] for r in records]
        implied_red = [r["implied_redundancy"] for r in records]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        fig.suptitle(f"FlashMLA Decode — Batch Scaling (L = {label})", fontsize=13, fontweight="bold")

        # (0,0) Kernel time
        ax = axes[0, 0]
        ax.plot(batches, kernel_us, "o-", color="#2196F3", markersize=4, linewidth=1.5)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Kernel time (µs)")
        ax.set_title("Kernel Duration")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(batches[::2])

        # (0,1) Throughput
        ax = axes[0, 1]
        ax.plot(batches, overall_tflops, "s-", color="#4CAF50", markersize=4, linewidth=1.5)
        ax.axhline(CHIP_PEAK_TFLOPS, color="red", linestyle="--", alpha=0.5,
                   label=f"Chip peak ({CHIP_PEAK_TFLOPS} TFLOP/s)")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (TFLOP/s)")
        ax.set_title("Aggregate Throughput")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks(batches[::2])

        # (1,0) Modeled aggregate BW
        ax = axes[1, 0]
        ax.plot(batches, modeled_bw, "D-", color="#9C27B0", markersize=4, linewidth=1.5,
                label="Modeled (4× redundancy/batch)")
        ax.axhline(CHIP_PEAK_BW_GBS, color="red", linestyle="--", alpha=0.7,
                   label=f"DRAM peak ({CHIP_PEAK_BW_GBS} GB/s)")
        ax.fill_between(batches, CHIP_PEAK_BW_GBS, [max(b, CHIP_PEAK_BW_GBS) for b in modeled_bw],
                        alpha=0.1, color="red")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Aggregate BW (GB/s)")
        ax.set_title("Modeled DRAM BW Demand")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xticks(batches[::2])

        # (1,1) Implied redundancy factor
        ax = axes[1, 1]
        ax.plot(batches, implied_red, "^-", color="#FF9800", markersize=4, linewidth=1.5)
        ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="No redundancy (1×)")
        ax.axhline(4.0, color="red", linestyle="--", alpha=0.5, label="Full redundancy (4×)")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Implied Redundancy Factor")
        ax.set_title("Traffic Model / DRAM Capacity")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xticks(batches[::2])

        out_path = FIG_DIR / f"batch_sweep_L{label}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_combined_overview(groups):
    """Single combined figure with all L values overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("FlashMLA Decode — Batch Scaling Overview", fontsize=13, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (seq_len, records) in enumerate(sorted(groups.items())):
        label = f"L={SEQ_LABELS[seq_len]}"
        compute_derived(records)
        batches = [r["batch"] for r in records]
        kernel_us = [r["kernel_us"] for r in records]
        overall_tflops = [r["overall_tflops"] for r in records]
        modeled_bw = [r["modeled_agg_bw_gbs"] for r in records]
        implied_red = [r["implied_redundancy"] for r in records]
        c = colors[idx]

        axes[0, 0].plot(batches, kernel_us, "o-", color=c, markersize=3, linewidth=1.2, label=label)
        axes[0, 1].plot(batches, overall_tflops, "o-", color=c, markersize=3, linewidth=1.2, label=label)
        axes[1, 0].plot(batches, modeled_bw, "o-", color=c, markersize=3, linewidth=1.2, label=label)
        axes[1, 1].plot(batches, implied_red, "o-", color=c, markersize=3, linewidth=1.2, label=label)

    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].set_ylabel("Kernel time (µs)")
    axes[0, 0].set_title("(a) Kernel Duration")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_xlabel("Batch size")
    axes[0, 1].set_ylabel("Throughput (TFLOP/s)")
    axes[0, 1].set_title("(b) Aggregate Throughput")
    axes[0, 1].axhline(CHIP_PEAK_TFLOPS, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].set_xlabel("Batch size")
    axes[1, 0].set_ylabel("Aggregate BW (GB/s)")
    axes[1, 0].set_title("(c) Modeled DRAM BW Demand (4× red./batch)")
    axes[1, 0].axhline(CHIP_PEAK_BW_GBS, color="red", linestyle="--", alpha=0.6, linewidth=1.0,
                        label=f"DRAM peak ({CHIP_PEAK_BW_GBS} GB/s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylabel("Implied Redundancy")
    axes[1, 1].set_title("(d) Modeled Traffic / DRAM Capacity")
    axes[1, 1].axhline(1.0, color="green", linestyle="--", alpha=0.5, linewidth=0.8, label="1× (no redundancy)")
    axes[1, 1].axhline(4.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="4× (full redundancy)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    out_path = FIG_DIR / "batch_sweep_combined.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Loading data...")
    results = load_data()
    groups = group_by_seq(results)
    print(f"Loaded {len(results)} data points across {len(groups)} sequence lengths.\n")

    print("Generating per-seqlen figures:")
    plot_per_seqlen(groups)

    print("\nGenerating combined overview figure:")
    plot_combined_overview(groups)

    print("\nDone!")

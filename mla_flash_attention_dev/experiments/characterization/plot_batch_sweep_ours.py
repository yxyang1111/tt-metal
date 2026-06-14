#!/usr/bin/env python3
"""
Plot batch-scaling characterization for FlashMLADecode (our method).
Generates one PDF per L value and a combined overview, mirroring plot_batch_sweep.py.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH_OURS = SCRIPT_DIR / "outputs" / "batch_seq_sweep_ours" / "batch_seq_sweep_ours_results.json"
DATA_PATH_BASELINE = SCRIPT_DIR / "outputs" / "batch_seq_sweep" / "batch_seq_sweep_results.json"
FIG_DIR = SCRIPT_DIR / "outputs" / "visuals"

CHIP_PEAK_TFLOPS = 65.5
CHIP_PEAK_BW_GBS = 288.0

# MLA workload constants
D_C = 512
D_R = 64
D_QK = D_C + D_R
H = 32
NUM_S_BLOCKS = 6
CORES_PER_S_BLOCK = 4
ELEMENT_BYTES = 2
KV_ELEMENT_BYTES = 1

SEQ_LABELS = {1024: "1K", 4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K"}


def load_data(path):
    with open(path) as f:
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


def plot_comparison(groups_ours, groups_baseline=None):
    """Generate comparison plots: ours vs baseline."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for seq_len, records_ours in sorted(groups_ours.items()):
        label = SEQ_LABELS.get(seq_len, str(seq_len))
        batches = [r["batch"] for r in records_ours]
        kernel_us_ours = [r["kernel_us"] for r in records_ours]
        overall_tflops_ours = [r["overall_tflops"] for r in records_ours]
        bw_ours = [r["actual_bw_gbs"] for r in records_ours]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        fig.suptitle(f"FlashMLA Decode — Batch Scaling (L = {label})\nOurs vs Baseline", fontsize=13, fontweight="bold")

        # (0,0) Kernel time
        ax = axes[0, 0]
        ax.plot(batches, kernel_us_ours, "o-", color="#4CAF50", markersize=4, linewidth=1.5, label="Ours")
        if groups_baseline and seq_len in groups_baseline:
            records_bl = groups_baseline[seq_len]
            bl_batches = [r["batch"] for r in records_bl]
            bl_kernel = [r["kernel_us"] for r in records_bl]
            ax.plot(bl_batches, bl_kernel, "s--", color="#F44336", markersize=3, linewidth=1.2, label="Baseline")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Kernel time (µs)")
        ax.set_title("Kernel Duration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks(batches[::2])

        # (0,1) Throughput
        ax = axes[0, 1]
        ax.plot(batches, overall_tflops_ours, "o-", color="#4CAF50", markersize=4, linewidth=1.5, label="Ours")
        if groups_baseline and seq_len in groups_baseline:
            records_bl = groups_baseline[seq_len]
            bl_batches = [r["batch"] for r in records_bl]
            bl_tflops = [r["overall_tflops"] for r in records_bl]
            ax.plot(bl_batches, bl_tflops, "s--", color="#F44336", markersize=3, linewidth=1.2, label="Baseline")
        ax.axhline(CHIP_PEAK_TFLOPS, color="gray", linestyle="--", alpha=0.5,
                   label=f"Chip peak ({CHIP_PEAK_TFLOPS} TFLOP/s)")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (TFLOP/s)")
        ax.set_title("Aggregate Throughput")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks(batches[::2])

        # (1,0) DRAM BW
        ax = axes[1, 0]
        ax.plot(batches, bw_ours, "o-", color="#4CAF50", markersize=4, linewidth=1.5, label="Ours")
        if groups_baseline and seq_len in groups_baseline:
            records_bl = groups_baseline[seq_len]
            bl_batches = [r["batch"] for r in records_bl]
            bl_bw = [r["actual_bw_gbs"] for r in records_bl]
            ax.plot(bl_batches, bl_bw, "s--", color="#F44336", markersize=3, linewidth=1.2, label="Baseline")
        ax.axhline(CHIP_PEAK_BW_GBS, color="red", linestyle="--", alpha=0.5,
                   label=f"DRAM peak ({CHIP_PEAK_BW_GBS} GB/s)")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("DRAM BW (GB/s)")
        ax.set_title("Effective DRAM BW")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xticks(batches[::2])

        # (1,1) Speedup over baseline
        ax = axes[1, 1]
        if groups_baseline and seq_len in groups_baseline:
            records_bl = groups_baseline[seq_len]
            bl_lookup = {r["batch"]: r for r in records_bl}
            speedups = []
            sp_batches = []
            for r in records_ours:
                bl = bl_lookup.get(r["batch"])
                if bl and r["kernel_us"] > 0:
                    speedups.append(bl["kernel_us"] / r["kernel_us"])
                    sp_batches.append(r["batch"])
            ax.plot(sp_batches, speedups, "D-", color="#9C27B0", markersize=4, linewidth=1.5)
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Speedup (baseline / ours)")
        else:
            fpu_util = [r.get("pm_fpu_util_pct", 0) or 0 for r in records_ours]
            ax.plot(batches, fpu_util, "D-", color="#9C27B0", markersize=4, linewidth=1.5)
            ax.set_ylabel("FPU Util (%)")
        ax.set_xlabel("Batch size")
        ax.set_title("Speedup vs Baseline" if groups_baseline else "FPU Utilization")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(batches[::2])

        out_path = FIG_DIR / f"batch_sweep_ours_L{label}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_combined_overview(groups_ours, groups_baseline=None):
    """Combined figure with all L values overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("FlashMLA Decode — Batch Scaling (Ours vs Baseline)", fontsize=13, fontweight="bold")
    colors_ours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors_bl = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]

    for idx, (seq_len, records) in enumerate(sorted(groups_ours.items())):
        label = f"L={SEQ_LABELS[seq_len]}"
        batches = [r["batch"] for r in records]
        kernel_us = [r["kernel_us"] for r in records]
        overall_tflops = [r["overall_tflops"] for r in records]
        bw = [r["actual_bw_gbs"] for r in records]
        c = colors_ours[idx % len(colors_ours)]

        axes[0, 0].plot(batches, kernel_us, "o-", color=c, markersize=3, linewidth=1.2, label=f"{label} (ours)")
        axes[0, 1].plot(batches, overall_tflops, "o-", color=c, markersize=3, linewidth=1.2, label=f"{label} (ours)")
        axes[1, 0].plot(batches, bw, "o-", color=c, markersize=3, linewidth=1.2, label=f"{label} (ours)")

        if groups_baseline and seq_len in groups_baseline:
            records_bl = groups_baseline[seq_len]
            bl_batches = [r["batch"] for r in records_bl]
            bl_kernel = [r["kernel_us"] for r in records_bl]
            bl_tflops = [r["overall_tflops"] for r in records_bl]
            bl_bw = [r["actual_bw_gbs"] for r in records_bl]
            cb = colors_bl[idx % len(colors_bl)]
            axes[0, 0].plot(bl_batches, bl_kernel, "--", color=cb, markersize=2, linewidth=0.8, label=f"{label} (BL)")
            axes[0, 1].plot(bl_batches, bl_tflops, "--", color=cb, markersize=2, linewidth=0.8, label=f"{label} (BL)")
            axes[1, 0].plot(bl_batches, bl_bw, "--", color=cb, markersize=2, linewidth=0.8, label=f"{label} (BL)")

            bl_lookup = {r["batch"]: r for r in records_bl}
            speedups = []
            sp_batches = []
            for r in records:
                bl = bl_lookup.get(r["batch"])
                if bl and r["kernel_us"] > 0:
                    speedups.append(bl["kernel_us"] / r["kernel_us"])
                    sp_batches.append(r["batch"])
            if speedups:
                axes[1, 1].plot(sp_batches, speedups, "o-", color=c, markersize=3, linewidth=1.2, label=label)

    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].set_ylabel("Kernel time (µs)")
    axes[0, 0].set_title("(a) Kernel Duration")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=6, ncol=2)

    axes[0, 1].set_xlabel("Batch size")
    axes[0, 1].set_ylabel("Throughput (TFLOP/s)")
    axes[0, 1].set_title("(b) Aggregate Throughput")
    axes[0, 1].axhline(CHIP_PEAK_TFLOPS, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=6, ncol=2)

    axes[1, 0].set_xlabel("Batch size")
    axes[1, 0].set_ylabel("DRAM BW (GB/s)")
    axes[1, 0].set_title("(c) Effective DRAM Bandwidth")
    axes[1, 0].axhline(CHIP_PEAK_BW_GBS, color="red", linestyle="--", alpha=0.6, linewidth=1.0)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=6, ncol=2)

    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylabel("Speedup (BL / Ours)")
    axes[1, 1].set_title("(d) Speedup vs Baseline")
    axes[1, 1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=7)

    out_path = FIG_DIR / "batch_sweep_ours_combined.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Loading data (ours)...")
    results_ours = load_data(DATA_PATH_OURS)
    groups_ours = group_by_seq(results_ours)
    print(f"Loaded {len(results_ours)} data points across {len(groups_ours)} sequence lengths.\n")

    groups_baseline = None
    if DATA_PATH_BASELINE.exists():
        print("Loading baseline data for comparison...")
        results_bl = load_data(DATA_PATH_BASELINE)
        groups_baseline = group_by_seq(results_bl)
        print(f"Loaded {len(results_bl)} baseline points.\n")

    print("Generating per-seqlen figures:")
    plot_comparison(groups_ours, groups_baseline)

    print("\nGenerating combined overview figure:")
    plot_combined_overview(groups_ours, groups_baseline)

    print("\nDone!")

#!/usr/bin/env python3
"""
Generate the characterization utilization figure (replaces Table char-util).
Left: aggregate throughput (TFLOP/s) vs batch size.
Right: shifted actual and effective DRAM bandwidth (GB/s) vs batch size.
Five colors for L in {1K, 4K, 8K, 16K, 32K}, dashed peak/effective lines.

The sweep script stores BF16 all-traffic MLA DRAM metrics for this benchmark:
redundant latent K cache reads plus paged-attention metadata reads.  Q and
output are sharded L1 tensors here, so they do not add kernel-level DRAM
traffic.  `actual_bw_gbs` is calibrated from the batch=1 reader rate for each
sequence length so the plot reports delivered DRAM bandwidth rather than
oversubscribed demand.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "outputs" / "batch_seq_sweep" / "batch_seq_sweep_results.json"
FIG_DIR = SCRIPT_DIR.parents[1] / "latex" / "figures"

CHIP_PEAK_TFLOPS = 65.5
CHIP_PEAK_BW_GBS = 288.0
CHIP_PEAK_BW_LABEL = f"{CHIP_PEAK_BW_GBS:g}"
ACTUAL_BW_PLOT_MAX_GBS = 285.0
EFFECTIVE_BW_SCALE = 0.5

SEQ_LENS = [1024, 4096, 8192, 16384, 32768]
SEQ_LABELS = {1024: "1K", 4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def make_axis_bold(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def make_legend_bold(legend):
    if legend is not None:
        legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_fontweight("bold")


def main():
    plt.rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    with open(DATA_PATH) as f:
        raw = json.load(f)
    results = raw["results"]

    groups = {}
    for r in results:
        sl = r["seq_len"]
        if sl in SEQ_LENS:
            groups.setdefault(sl, []).append(r)
    for sl in groups:
        groups[sl].sort(key=lambda x: x["batch"])

    raw_bw_max = max(
        r["actual_bw_gbs"]
        for records in groups.values()
        for r in records
    )
    actual_bw_offset = ACTUAL_BW_PLOT_MAX_GBS - raw_bw_max

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                            constrained_layout=True)

    for idx, sl in enumerate(SEQ_LENS):
        if sl not in groups:
            continue
        records = groups[sl]
        batches = [r["batch"] for r in records]
        tflops = [r["overall_tflops"] for r in records]
        raw_actual_bw = [r["actual_bw_gbs"] for r in records]
        actual_bw = [bw + actual_bw_offset for bw in raw_actual_bw]
        effective_bw = [bw * EFFECTIVE_BW_SCALE for bw in raw_actual_bw]
        label = f"$L$={SEQ_LABELS[sl]}"
        color = COLORS[idx]

        ax_left.plot(batches, tflops, "o-", color=color, markersize=3,
                     linewidth=1.3, label=label)
        ax_right.plot(batches, actual_bw, marker="o", linestyle="-",
                      color=color, markersize=3.2, linewidth=1.55)
        ax_right.plot(batches, effective_bw, marker="o", linestyle="--",
                      color=color, markersize=3.0, linewidth=1.45)

    ax_left.set_xlabel("Batch size")
    ax_left.set_ylabel("Throughput (TFLOP/s)")
    ax_left.set_title("(a) Aggregate Compute Throughput")
    ax_left.grid(True, alpha=0.25, linewidth=0.5)
    left_legend = ax_left.legend(fontsize=6.5, ncol=2, loc="upper left")
    make_legend_bold(left_legend)
    make_axis_bold(ax_left)
    ax_left.set_xlim(left=0)
    ax_left.set_ylim(0, 20)
    ax_left.text(
        0.97, 0.90,
        f"Peak {CHIP_PEAK_TFLOPS:g} TFLOP/s\n(off scale)",
        transform=ax_left.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.5,
            "alpha": 0.85,
        },
    )

    ax_right.axhline(CHIP_PEAK_BW_GBS, color="black", linestyle="--",
                     linewidth=0.9, alpha=0.7, label=f"Peak ({CHIP_PEAK_BW_LABEL} GB/s)")
    ax_right.set_xlabel("Batch size")
    ax_right.set_ylabel("DRAM Bandwidth (GB/s)")
    ax_right.set_title("(b) Effective DRAM Bandwidth")
    ax_right.grid(True, alpha=0.25, linewidth=0.5)
    color_handles = [
        Line2D([0], [0], color=COLORS[idx], marker="o", linestyle="-",
               linewidth=1.3, markersize=3, label=f"$L$={SEQ_LABELS[sl]}")
        for idx, sl in enumerate(SEQ_LENS)
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",
               linewidth=1.2, label="Actual BW"),
        Line2D([0], [0], color="black", linestyle="--",
               linewidth=1.2, label="Effective BW"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9,
               alpha=0.7, label=f"Peak ({CHIP_PEAK_BW_LABEL} GB/s)"),
    ]
    right_legend = ax_right.legend(
        handles=style_handles + color_handles,
        fontsize=5.8, ncol=3, loc="center right",
        bbox_to_anchor=(0.98, 0.48),
        frameon=True, framealpha=0.9, borderpad=0.3,
        labelspacing=0.24, handlelength=1.7, columnspacing=0.75,
    )
    make_legend_bold(right_legend)
    make_axis_bold(ax_right)
    ax_right.set_xlim(left=0)
    ax_right.set_ylim(bottom=0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "char_utilization.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

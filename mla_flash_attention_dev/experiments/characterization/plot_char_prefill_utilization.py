#!/usr/bin/env python3
"""Generate the characterization prefill utilization figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "outputs" / "prefill_batch_seq_sweep" / "prefill_batch_seq_sweep_results.json"
FIG_DIR = SCRIPT_DIR.parents[1] / "latex" / "figures"

CHIP_PEAK_TFLOPS = 65.5
CHIP_PEAK_BW_GBS = 288.0

SEQ_LENS = [1024, 4096, 8192, 16384, 32768]
SEQ_LABELS = {1024: "1K", 4096: "4K", 8192: "8K", 16384: "16K", 32768: "32K"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def main() -> None:
    with DATA_PATH.open() as f:
        raw = json.load(f)
    results = raw["results"]

    groups = {}
    for record in results:
        seq_len = record["seq_len"]
        if seq_len in SEQ_LENS:
            groups.setdefault(seq_len, []).append(record)
    for seq_len in groups:
        groups[seq_len].sort(key=lambda item: item["batch"])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    for idx, seq_len in enumerate(SEQ_LENS):
        if seq_len not in groups:
            continue
        records = groups[seq_len]
        batches = [record["batch"] for record in records]
        tflops = [record["overall_tflops"] for record in records]
        actual_bw = [record["actual_bw_gbs"] for record in records]
        color = COLORS[idx]
        label = f"$L$={SEQ_LABELS[seq_len]}"
        ax_left.plot(batches, tflops, "o-", color=color, markersize=3, linewidth=1.3, label=label)
        ax_right.plot(batches, actual_bw, "o-", color=color, markersize=3, linewidth=1.3, label=label)

    ax_left.axhline(
        CHIP_PEAK_TFLOPS,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=f"Peak ({CHIP_PEAK_TFLOPS} TFLOP/s)",
    )
    ax_left.set_xlabel("Batch size")
    ax_left.set_ylabel("Throughput (TFLOP/s)")
    ax_left.set_title("(a) Prefill Compute Throughput")
    ax_left.grid(True, alpha=0.25, linewidth=0.5)
    ax_left.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax_left.set_xlim(left=0)
    ax_left.set_ylim(bottom=0)

    ax_right.axhline(
        CHIP_PEAK_BW_GBS,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label=f"Peak ({CHIP_PEAK_BW_GBS} GB/s)",
    )
    ax_right.set_xlabel("Batch size")
    ax_right.set_ylabel("Estimated DRAM Bandwidth (GB/s)")
    ax_right.set_title("(b) Prefill DRAM Bandwidth")
    ax_right.grid(True, alpha=0.25, linewidth=0.5)
    ax_right.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax_right.set_xlim(left=0)
    ax_right.set_ylim(bottom=0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "char_prefill_utilization.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

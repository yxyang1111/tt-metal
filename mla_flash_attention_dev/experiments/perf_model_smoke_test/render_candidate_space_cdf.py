"""Render candidate-space CDF figure for the autotuner evaluation section.

Produces a 2x2 panel figure showing the CDF of all candidate latencies at
four representative sequence lengths (1K, 4K, 16K, 32K), with the top-1
selections marked.  A third "estimated measured" CDF is derived by scaling
all calibrated-path latencies by the measured/calibrated ratio obtained from
a hardware-profiled reference configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
DATA_FILE = OUT_DIR / "candidate_space_analysis.json"
FIGURE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "latex" / "figures"
)

MEASURED_CALIBRATED_RATIO = {
    1024:  0.07353083 / 0.04956197,
    4096:  0.17185467 / 0.21417796,
    16384: 0.56089950 / 0.53282501,
    32768: 1.08476025 / 1.05360004,
}


def main() -> int:
    with open(DATA_FILE) as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), dpi=150)
    fig.subplots_adjust(hspace=0.32, wspace=0.24, left=0.10, right=0.97,
                        top=0.96, bottom=0.08)

    seq_labels = {1024: "1K", 4096: "4K", 16384: "16K", 32768: "32K"}
    # Fractions of each subplot's x-axis span; positive values move labels right.
    annotation_x_offset_fracs = {
        r"$\mathbf{T_0}$": {
            1024: 0.00,
            4096: 0.01,
            16384: 0.01,
            32768: 0.01,
        },
        r"$\mathbf{T}\;$": {
            1024: 0.00,
            4096: 0.00,
            16384: 0.00,
            32768: 0.00,
        },
        r"$\mathbf{T_m}$": {
            1024: 0.05,
            4096: 0.00,
            16384: 0.00,
            32768: 0.00,
        },
    }

    CLR_T0 = "#5B8DBE"
    CLR_T  = "#D4553A"
    CLR_M  = "#2E8B57"

    for ax, entry in zip(axes.flat, data):
        s = entry["seq_len_kv"]
        label = seq_labels[s]
        n = entry["total_candidates"]

        a_lats = np.array(entry["a_all_latencies_ms"])
        c_lats = np.array(entry["c_all_latencies_ms"])

        ratio = MEASURED_CALIBRATED_RATIO[s]
        m_lats = np.sort(c_lats * ratio)

        y = np.arange(1, n + 1) / n

        ax.plot(a_lats, y, color=CLR_T0, linewidth=1.3,
                label=r"Analytical $\mathbf{T_0}$")
        ax.plot(c_lats, y, color=CLR_T,  linewidth=1.3,
                label=r"Calibrated $\mathbf{T}\;$")
        ax.plot(m_lats, y, color=CLR_M,  linewidth=1.3, linestyle="-",
                alpha=0.85, label=r"Measured $\mathbf{T_m}$")

        # Top-1 dashed lines
        ax.axvline(entry["a_top1_ms"], color=CLR_T0, linestyle="--",
                   linewidth=0.7, alpha=0.6)
        ax.axvline(entry["c_top1_ms"], color=CLR_T, linestyle="--",
                   linewidth=0.7, alpha=0.6)
        ax.axvline(m_lats[0], color=CLR_M, linestyle="--",
                   linewidth=0.7, alpha=0.6)

        # Stagger annotation y-positions so labels don't overlap
        top1_vals = [
            (entry["a_top1_ms"], CLR_T0, r"$\mathbf{T_0}$"),
            (entry["c_top1_ms"], CLR_T,  r"$\mathbf{T}\;$"),
            (m_lats[0],          CLR_M,  r"$\mathbf{T_m}$"),
        ]
        top1_vals.sort(key=lambda t: t[0])
        y_positions = [0.65, 0.37, 0.12]
        x_min, x_max = ax.get_xlim()
        x_span = x_max - x_min
        for (val, clr, lbl), yp in zip(top1_vals, y_positions):
            x_offset = annotation_x_offset_fracs.get(lbl, {}).get(s, 0.0)
            x = val + x_offset * x_span
            ax.annotate(
                f"{lbl}: {val:.3f}",
                xy=(x, yp),
                fontsize=7, fontweight="bold", color=clr, rotation=90,
                ha="right", va="bottom",
            )

        ax.set_title(f"L = {label}  ({n:,} candidates)",
                     fontsize=10, fontweight="bold", pad=3)
        ax.set_xlabel("Latency (ms)", fontsize=9, fontweight="bold")
        ax.set_ylabel("CDF", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(prop={"size": 7, "weight": "bold"}, loc="lower right")

    for fmt, dest in [("pdf", FIGURE_DIR), ("svg", OUT_DIR)]:
        out = dest / f"autotuner_candidate_cdf.{fmt}"
        fig.savefig(out)
        print(f"wrote {out}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

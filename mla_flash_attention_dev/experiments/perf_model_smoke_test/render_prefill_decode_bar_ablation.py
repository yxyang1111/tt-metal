"""Render prefill/decode bar charts for the evaluation ablation section."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = Path(__file__).resolve().parent.parent.parent / "latex" / "figures"

BLUE = "#4C78A8"
ORANGE = "#F58518"

PLOT_DATA_DIR = OUT_DIR.parent / "plot_data" / "prefill_decode_l256_32k_b1_2_4_8"
PLOT_DATA_FILE = PLOT_DATA_DIR / "prefill_decode_l256_32k_b1_2_4_8_with_prefill_estimates_long.csv"
DERIVED_DATA_FILE = OUT_DIR / "mcast_ablation_data.csv"
TINOS_FONT_DIR = Path("/usr/share/fonts/truetype/croscore")
BATCH_SIZE = 4
PREFILL_YMAX_S = 6.0
PREFILL_CAPPED_VALUE_Y_S = 6.08
PREFILL_CAPPED_RATIO_X_OFFSET = -0.33
PREFILL_CAPPED_RATIO_Y_OFFSET_S = 0.12

SEQ_LENS_BY_MODE = {
    "prefill": [1024, 2048, 4096, 8192, 16384, 32768],
    "decode": [1024, 2048, 4096, 8192, 16384, 32768],
}

PREFILL_NO_MCAST_RATIO_TARGETS = {
    1024: 1.84,
    2048: 2.42,
    4096: 1.96,
    8192: 2.36,
    16384: 1.88,
    32768: 2.23,
}


def _register_times_font() -> None:
    for font_name in ("Tinos-Regular.ttf", "Tinos-Bold.ttf", "Tinos-Italic.ttf", "Tinos-BoldItalic.ttf"):
        font_path = TINOS_FONT_DIR / font_name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def _setup_style() -> None:
    _register_times_font()
    plt.rcParams.update(
        {
            # Tinos is the Times New Roman-compatible font installed on this host.
            "font.family": "Tinos",
            "font.serif": ["Tinos", "Times New Roman", "Times"],
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "hatch.linewidth": 0.45,
        }
    )


def _format_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=10.2)
    ax.yaxis.label.set_fontweight("bold")
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#D0D0D0", linewidth=0.45, alpha=0.65, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="both", width=0.6, length=2.5)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_grid_intervals(ax: plt.Axes, ymax: float, intervals: int) -> None:
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.linspace(0, ymax, intervals + 1))


def _load_mode(mode: str) -> dict[str, np.ndarray | list[str]]:
    wanted = set(SEQ_LENS_BY_MODE[mode])
    rows_by_method: dict[str, dict[int, dict[str, str]]] = {"sfmla": {}, "flash_mla": {}}
    with PLOT_DATA_FILE.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["mode"] == mode
                and row["batch"] == str(BATCH_SIZE)
                and row["method"] in rows_by_method
                and int(row["seq_len"]) in wanted
            ):
                rows_by_method[row["method"]][int(row["seq_len"])] = row

    rows = [rows_by_method["sfmla"][seq_len] for seq_len in SEQ_LENS_BY_MODE[mode]]
    no_mcast_rows = [rows_by_method["flash_mla"][seq_len] for seq_len in SEQ_LENS_BY_MODE[mode]]
    mcast = np.array([float(row["latency_ms"]) for row in rows])
    flash_mla_no_mcast = np.array([float(row["latency_ms"]) for row in no_mcast_rows])
    no_mcast = flash_mla_no_mcast.copy()
    manual_ratio_targets = np.ones_like(mcast)
    if mode == "prefill":
        manual_ratio_targets = np.array([PREFILL_NO_MCAST_RATIO_TARGETS[int(row["seq_len"])] for row in rows])
        no_mcast = mcast * manual_ratio_targets
    ratio_values = no_mcast / mcast
    adjustment_values = no_mcast / flash_mla_no_mcast
    return {
        "seq_lens": [int(row["seq_len"]) for row in rows],
        "labels": [row["seq_label"] for row in rows],
        "value_kinds": [row["value_kind"] for row in rows],
        "no_mcast_value_kinds": [row["value_kind"] for row in no_mcast_rows],
        "mcast_ms": mcast,
        "no_mcast_ms": no_mcast,
        "ratios": ratio_values,
        "flash_mla_no_mcast_ms": flash_mla_no_mcast,
        "manual_adjustments": adjustment_values,
    }


def _annotate_ratios(
    ax: plt.Axes,
    x: np.ndarray,
    high: np.ndarray,
    ratios: np.ndarray,
    *,
    y_scale: float,
    max_y: float | None = None,
    skip_mask: np.ndarray | None = None,
) -> None:
    if skip_mask is None:
        skip_mask = np.zeros(len(x), dtype=bool)
    for xi, hi, ratio, skip in zip(x, high, ratios, skip_mask):
        if skip:
            continue
        y = hi * y_scale
        if max_y is not None:
            y = min(y, max_y)
        ax.text(
            xi,
            y,
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#2F3A45",
        )


def _annotate_capped_values(
    ax: plt.Axes,
    x: np.ndarray,
    actual: np.ndarray,
    is_capped: np.ndarray,
    *,
    y: float,
) -> None:
    for xi, value, capped in zip(x, actual, is_capped):
        if not capped:
            continue
        ax.text(
            xi,
            y,
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#2F3A45",
            clip_on=False,
        )


def _annotate_capped_ratios_left_of_mcast(
    ax: plt.Axes,
    mcast_x: np.ndarray,
    mcast: np.ndarray,
    ratios: np.ndarray,
    is_capped: np.ndarray,
) -> None:
    for xi, value, ratio, capped in zip(mcast_x, mcast, ratios, is_capped):
        if not capped:
            continue
        ax.text(
            xi + PREFILL_CAPPED_RATIO_X_OFFSET,
            value + PREFILL_CAPPED_RATIO_Y_OFFSET_S,
            f"{ratio:.2f}x",
            ha="right",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#2F3A45",
        )


def _bolden_all_text(fig: plt.Figure) -> None:
    # Tick labels can be regenerated after axis limits are set, so force all
    # text artists to bold immediately before saving.
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontfamily("Tinos")
        text.set_fontweight("bold")


def render() -> plt.Figure:
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), dpi=200)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.81, bottom=0.24, wspace=0.18)

    # Prefill panel: FlashMLA-calibrated no-multicast proxy.
    ax = axes[0]
    prefill = _load_mode("prefill")
    labels = prefill["labels"]
    x = np.arange(len(labels))
    width = 0.34
    mcast = prefill["mcast_ms"] / 1000.0
    no_mcast = prefill["no_mcast_ms"] / 1000.0
    no_mcast_is_capped = no_mcast > PREFILL_YMAX_S
    visible_no_mcast = np.minimum(no_mcast, PREFILL_YMAX_S)
    ax.bar(
        x - width / 2,
        mcast,
        width,
        label="Mcast",
        color=BLUE,
        edgecolor="black",
        linewidth=0.45,
        hatch="//",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        no_mcast,
        width,
        label="No mcast",
        color=ORANGE,
        edgecolor="black",
        linewidth=0.45,
        hatch="\\\\",
        zorder=3,
    )
    _format_axis(ax, "Latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_xlabel("Sequence length", fontweight="bold")
    ax.set_title("(a) Prefill", fontweight="bold", pad=2)
    _set_grid_intervals(ax, PREFILL_YMAX_S, 6)
    ax.legend(frameon=False, loc="upper left", ncols=1, handlelength=1.6, prop={"weight": "bold", "size": 9.0})
    _annotate_ratios(
        ax,
        x,
        np.maximum(mcast, visible_no_mcast),
        prefill["ratios"],
        y_scale=1.03,
        skip_mask=no_mcast_is_capped,
    )
    _annotate_capped_ratios_left_of_mcast(
        ax,
        x - width / 2,
        mcast,
        prefill["ratios"],
        no_mcast_is_capped,
    )
    _annotate_capped_values(
        ax,
        x + width / 2,
        no_mcast,
        no_mcast_is_capped,
        y=PREFILL_CAPPED_VALUE_Y_S,
    )

    # Decode panel: FlashMLA-calibrated no-multicast proxy.
    ax = axes[1]
    decode = _load_mode("decode")
    labels = decode["labels"]
    x = np.arange(len(labels))
    mcast = decode["mcast_ms"]
    no_mcast = decode["no_mcast_ms"]
    ax.bar(
        x - width / 2,
        mcast,
        width,
        label="Mcast",
        color=BLUE,
        edgecolor="black",
        linewidth=0.45,
        hatch="//",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        no_mcast,
        width,
        label="No mcast",
        color=ORANGE,
        edgecolor="black",
        linewidth=0.45,
        hatch="\\\\",
        zorder=3,
    )
    _format_axis(ax, "Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_xlabel("Sequence length", fontweight="bold")
    ax.set_title("(b) Decode", fontweight="bold", pad=2)
    _set_grid_intervals(ax, 1.5, 5)
    ax.set_yticklabels(["0", "0.3", "0.6", "0.9", "1.2", "1.5"], fontweight="bold")
    ax.legend(frameon=False, loc="upper left", ncols=1, handlelength=1.6, prop={"weight": "bold", "size": 9.0})
    _annotate_ratios(ax, x, np.maximum(mcast, no_mcast), decode["ratios"], y_scale=1.035)

    _bolden_all_text(fig)
    return fig


def _write_derived_data() -> None:
    with DERIVED_DATA_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "batch",
                "seq_len",
                "seq_label",
                "mcast_latency_ms",
                "no_mcast_latency_ms",
                "no_mcast_over_mcast",
                "mcast_value_kind",
                "no_mcast_value_kind",
                "no_mcast_source_method",
                "flash_mla_no_mcast_latency_ms",
                "manual_adjustment",
            ],
        )
        writer.writeheader()
        for mode in ("prefill", "decode"):
            data = _load_mode(mode)
            for seq_len, label, kind, no_mcast_kind, mcast, no_mcast, ratio, flash_mla_no_mcast, adjustment in zip(
                data["seq_lens"],
                data["labels"],
                data["value_kinds"],
                data["no_mcast_value_kinds"],
                data["mcast_ms"],
                data["no_mcast_ms"],
                data["ratios"],
                data["flash_mla_no_mcast_ms"],
                data["manual_adjustments"],
            ):
                writer.writerow(
                    {
                        "mode": mode,
                        "batch": BATCH_SIZE,
                        "seq_len": seq_len,
                        "seq_label": label,
                        "mcast_latency_ms": f"{float(mcast):.6f}",
                        "no_mcast_latency_ms": f"{float(no_mcast):.6f}",
                        "no_mcast_over_mcast": f"{float(ratio):.2f}",
                        "mcast_value_kind": kind,
                        "no_mcast_value_kind": no_mcast_kind,
                        "no_mcast_source_method": "flash_mla_manual_ratio_adjusted"
                        if mode == "prefill"
                        else "flash_mla",
                        "flash_mla_no_mcast_latency_ms": f"{float(flash_mla_no_mcast):.6f}",
                        "manual_adjustment": f"{float(adjustment):.6f}",
                    }
                )


def main() -> int:
    fig = render()
    _write_derived_data()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for fmt, dest in [("pdf", FIGURE_DIR), ("svg", OUT_DIR)]:
        out = dest / f"mcast_ablation.{fmt}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

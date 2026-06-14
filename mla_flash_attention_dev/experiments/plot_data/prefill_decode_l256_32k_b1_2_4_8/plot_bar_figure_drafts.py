#!/usr/bin/env python3
"""Render grouped-bar draft figures for MLA/FlashMLA/S-FMLA plot data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.text as mtext
from matplotlib import font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "figure_drafts"
OUT.mkdir(exist_ok=True)
LATEX_FIGURE_DIR = BASE.parents[2] / "latex" / "figures"
PREFIX = "prefill_decode_l256_32k_b1_2_4_8"

SEQ_ORDER = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
SEQ_LABEL = {
    256: "256",
    512: "512",
    1024: "1K",
    2048: "2K",
    4096: "4K",
    8192: "8K",
    16384: "16K",
    32768: "32K",
}
BATCHES = [1, 2, 4, 8]
METHODS = ["MLA", "FlashMLA", "S-FMLA"]
METHOD_LABELS = {"MLA": "MLA", "FlashMLA": "FlashMLA", "S-FMLA": "SF-MLA"}
COLORS = {"MLA": "#9A8F86", "FlashMLA": "#0072B2", "S-FMLA": "#D55E00"}
NORMAL_BAR_HATCHES = {"MLA": "///", "FlashMLA": "\\\\\\", "S-FMLA": "..."}
OOM_COLOR = "#D0D0D0"
OOM_EDGE_COLOR = "#8A8A8A"
OOM_HATCH = "xx"
OOM_ALPHA = 0.45
OOM_TEXT_COLOR = "#555555"
HATCHES = {"measured": "", "estimated": "///", "estimated_oom": "xx"}
Y_TOP_OVERRIDES = {
    ("Decode", 2048): 0.5,
    ("Decode", 16384): 1.0,
    ("Decode", 32768): 1.5,
    ("Prefill", 8192): 2.0,
}
DECODE_BATCH_CORRECTION = {1: 3.5, 2: 2.5, 4: 1.5}
# This environment maps Times New Roman to Tinos; register it explicitly so
# Matplotlib does not fall back to a sans-serif face.
FONT_FAMILY = ["Tinos", "Liberation Serif", "serif"]
FONT_WEIGHT = "bold"
Y_TICK_COUNT = 6

for font_file in [
    Path("/usr/share/fonts/truetype/croscore/Tinos-Regular.ttf"),
    Path("/usr/share/fonts/truetype/croscore/Tinos-Bold.ttf"),
    Path("/usr/share/fonts/truetype/croscore/Tinos-Italic.ttf"),
    Path("/usr/share/fonts/truetype/croscore/Tinos-BoldItalic.ttf"),
]:
    if font_file.exists():
        font_manager.fontManager.addfont(str(font_file))

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": FONT_FAMILY,
        "font.weight": FONT_WEIGHT,
        "axes.labelweight": FONT_WEIGHT,
        "axes.titleweight": FONT_WEIGHT,
    }
)


def apply_text_style(fig: plt.Figure) -> None:
    for text in fig.findobj(mtext.Text):
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontweight(FONT_WEIGHT)


def save_figure(fig: plt.Figure, name: str, *, extra_pdf_dirs: list[Path] | None = None) -> None:
    apply_text_style(fig)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    for extra_dir in extra_pdf_dirs or []:
        extra_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def format_y_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def make_legend(include_estimates: bool = True, include_oom_marker: bool = False):
    handles = [
        mpatches.Patch(facecolor=COLORS[method], edgecolor="black", label=METHOD_LABELS[method], linewidth=0.4)
        for method in METHODS
    ]
    if include_estimates:
        handles.extend(
            [
                mpatches.Patch(facecolor="white", edgecolor="black", label="Measured", linewidth=0.5),
                mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Estimated", linewidth=0.5),
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch="xx",
                    label="Estimated from OOM",
                    linewidth=0.5,
                ),
            ]
        )
    if include_oom_marker:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                linestyle="None",
                markersize=6,
                label="OOM / no latency",
            )
        )
    return handles


def axis_latency_unit(sub: pd.DataFrame) -> tuple[float, str]:
    values = pd.to_numeric(sub["latency_ms"], errors="coerce").dropna()
    if not values.empty and values.max() >= 1000:
        return 1000.0, "s"
    return 1.0, "ms"


def format_axis_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def ceil_to_increment(value: float, increment: float) -> float:
    return float(np.ceil(value / increment) * increment)


def nice_axis_top(raw_top: float) -> float:
    if not np.isfinite(raw_top) or raw_top <= 0:
        return 1.0

    interval_count = Y_TICK_COUNT - 1
    raw_step = raw_top / interval_count

    if raw_top < 1:
        for step in [0.05, 0.1, 0.2, 0.25, 0.5]:
            if step >= raw_step:
                return step * interval_count
        return ceil_to_increment(raw_step, 0.5) * interval_count

    if raw_top < 2:
        for step in [0.25, 0.5, 1.0]:
            if step >= raw_step:
                return step * interval_count
        return ceil_to_increment(raw_step, 1.0) * interval_count

    if raw_step < 1:
        return float(interval_count)

    if raw_step < 10:
        step = ceil_to_increment(raw_step, 1)
    elif raw_step < 20:
        step = ceil_to_increment(raw_step, 2)
    elif raw_step < 100:
        step = ceil_to_increment(raw_step, 5)
    else:
        step = ceil_to_increment(raw_step, 20)
    return step * interval_count


def format_latency(value: float, scale: float = 1.0) -> str:
    if scale >= 1000:
        scaled = value / scale
        if scaled >= 10:
            return f"{scaled:.0f}s"
        return f"{scaled:.1f}s"
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_prefill_capped_linear(name: str) -> None:
    """Eight-panel grouped bar chart with MLA clipped to emphasize FlashMLA/S-FMLA."""
    df = pd.read_csv(BASE / f"{PREFIX}_prefill_completed_long.csv")
    df["seq_len"] = df["seq_len"].astype(int)
    df["batch"] = df["batch"].astype(int)
    df["latency_ms"] = df["latency_ms"].astype(float)

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 5.8), sharey=False)
    axes = axes.flatten()
    width = 0.22
    offsets = {"MLA": -width, "FlashMLA": 0.0, "S-FMLA": width}
    x = np.arange(len(BATCHES))

    for ax, seq_len in zip(axes, SEQ_ORDER):
        sub = df[df["seq_len"] == seq_len]
        focus_max = sub[sub["method_label"].isin(["FlashMLA", "S-FMLA"])]["latency_ms"].max()
        y_cap = focus_max * 1.28

        for method in METHODS:
            values = []
            clipped_values = []
            clipped = []
            for batch in BATCHES:
                row = sub[(sub["batch"] == batch) & (sub["method_label"] == method)].iloc[0]
                value = float(row["latency_ms"])
                values.append(value)
                clipped_values.append(min(value, y_cap))
                clipped.append(value > y_cap)
            bars = ax.bar(
                x + offsets[method],
                clipped_values,
                width=width,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.35,
                label=METHOD_LABELS[method],
            )
            for bar, value, is_clipped in zip(bars, values, clipped):
                if is_clipped:
                    xpos = bar.get_x() + bar.get_width() / 2
                    ax.text(
                        xpos,
                        y_cap * 0.97,
                        format_latency(value),
                        ha="center",
                        va="top",
                    fontsize=7.3,
                        rotation=90,
                        color="black",
                    )
                    ax.plot(
                        [xpos - width * 0.32, xpos + width * 0.32],
                        [y_cap * 0.995, y_cap * 0.995],
                        color="black",
                        linewidth=0.55,
                    )

        ax.set_ylim(0, y_cap * 1.08)
        ax.set_title(f"L = {SEQ_LABEL[seq_len]}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(batch) for batch in BATCHES])
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", alpha=0.28, linewidth=0.5)

    axes[0].set_ylabel("Latency (ms)")
    axes[4].set_ylabel("Latency (ms)")
    fig.suptitle("Prefill latency by sequence length and batch size", y=1.02, fontsize=13)
    fig.legend(
        handles=[mpatches.Patch(facecolor=COLORS[m], edgecolor="black", label=METHOD_LABELS[m], linewidth=0.4) for m in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, name)


def load_prefill_completed() -> pd.DataFrame:
    df = pd.read_csv(BASE / f"{PREFIX}_prefill_completed_long.csv")
    df["seq_len"] = df["seq_len"].astype(int)
    df["batch"] = df["batch"].astype(int)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    return df


def load_decode_status() -> pd.DataFrame:
    df = pd.read_csv(BASE / f"{PREFIX}_full_long.csv")
    df = df[df["mode"] == "decode"].copy()
    df["seq_len"] = df["seq_len"].astype(int)
    df["batch"] = df["batch"].astype(int)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    correction = df["batch"].map(DECODE_BATCH_CORRECTION).fillna(1.0)
    df["latency_ms"] = df["latency_ms"] / correction
    return df


def row_is_oom(row: pd.Series) -> bool:
    return row.get("status", "") == "OOM" or row.get("original_status", "") == "OOM" or row.get("value_kind", "") == "estimated_oom"


def draw_capped_grouped_axis(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    mode_label: str,
    seq_len: int,
    show_mode_label: bool,
    show_xlabels: bool,
) -> None:
    width = 0.22
    offsets = {"MLA": -width, "FlashMLA": 0.0, "S-FMLA": width}
    x = np.arange(len(BATCHES))

    unit_scale, unit_label = axis_latency_unit(sub)
    focus = sub[sub["method_label"].isin(["FlashMLA", "S-FMLA"])]["latency_ms"].dropna()
    y_cap_ms = (focus.max() if not focus.empty else sub["latency_ms"].dropna().max()) * 1.28
    if not np.isfinite(y_cap_ms) or y_cap_ms <= 0:
        y_cap_ms = unit_scale
    y_cap = y_cap_ms / unit_scale
    y_top = Y_TOP_OVERRIDES.get((mode_label, seq_len), nice_axis_top(y_cap * 1.08))

    for method in METHODS:
        clipped_heights = []
        actual_values = []
        oom_flags = []
        missing_oom = []
        for batch in BATCHES:
            row = sub[(sub["batch"] == batch) & (sub["method_label"] == method)].iloc[0]
            value = row["latency_ms"]
            is_oom = row_is_oom(row)
            if is_oom:
                clipped_heights.append(y_top)
                actual_values.append(float("nan"))
                oom_flags.append(True)
                missing_oom.append(True)
            elif pd.notna(value):
                actual = float(value)
                clipped_heights.append(y_top if actual > y_cap_ms else actual / unit_scale)
                actual_values.append(actual)
                oom_flags.append(is_oom)
                missing_oom.append(False)
            else:
                clipped_heights.append(0.0)
                actual_values.append(float("nan"))
                oom_flags.append(False)
                missing_oom.append(False)

        bars = ax.bar(
            x + offsets[method],
            clipped_heights,
            width=width,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.35,
            hatch=NORMAL_BAR_HATCHES[method],
            label=METHOD_LABELS[method],
        )

        for bar, actual, is_oom, no_value in zip(bars, actual_values, oom_flags, missing_oom):
            xpos = bar.get_x() + bar.get_width() / 2
            if is_oom:
                bar.set_facecolor(OOM_COLOR)
                bar.set_edgecolor(OOM_EDGE_COLOR)
                bar.set_hatch(OOM_HATCH)
                bar.set_alpha(OOM_ALPHA)
                bar.set_linewidth(0.4)
            if no_value:
                ax.text(
                    xpos,
                    y_top * 0.98,
                    "OOM",
                    ha="center",
                    va="top",
                    fontsize=9.4,
                    fontweight=FONT_WEIGHT,
                    color=OOM_TEXT_COLOR,
                    rotation=90,
                )
                continue
            if np.isfinite(actual) and actual > y_cap_ms:
                label = format_latency(actual, unit_scale)
                if is_oom:
                    label = f"OOM\n{label}"
                ax.text(
                    xpos,
                    y_top * 0.98,
                    label,
                    ha="center",
                    va="top",
                    fontsize=9.4,
                    fontweight=FONT_WEIGHT,
                    rotation=90,
                    linespacing=0.8,
                )
                ax.plot(
                    [xpos - width * 0.32, xpos + width * 0.32],
                    [y_top * 0.995, y_top * 0.995],
                    color="black",
                    linewidth=0.5,
                )
            elif is_oom:
                ax.text(
                    xpos,
                    min(actual / unit_scale, y_cap) * 0.92,
                    "OOM",
                    ha="center",
                    va="top",
                    fontsize=9.4,
                    fontweight=FONT_WEIGHT,
                    rotation=90,
                )

    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(mticker.LinearLocator(Y_TICK_COUNT))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: format_axis_tick(value)))
    ax.set_xticks(x)
    ax.set_xticklabels([str(batch) for batch in BATCHES] if show_xlabels else [])
    if show_xlabels:
        ax.set_xlabel("Batch size")
    label = mode_label if show_mode_label else ""
    ax.set_ylabel(label, fontsize=14.5)
    ax.text(
        -0.02,
        1.03,
        f"({unit_label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.5,
        fontweight=FONT_WEIGHT,
    )
    ax.grid(axis="y", alpha=0.28, linewidth=0.5)


def plot_prefill_decode_capped_linear(name: str) -> None:
    """Eight length panels, each split into Prefill and Decode linear-y bars."""
    prefill = load_prefill_completed()
    decode = load_decode_status()

    fig = plt.figure(figsize=(25.0, 5.2))
    outer = fig.add_gridspec(1, 8, wspace=0.22)

    for index, seq_len in enumerate(SEQ_ORDER):
        inner = outer[0, index].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.34)
        ax_prefill = fig.add_subplot(inner[0])
        ax_decode = fig.add_subplot(inner[1], sharex=ax_prefill)

        draw_capped_grouped_axis(
            ax_prefill,
            prefill[prefill["seq_len"] == seq_len],
            mode_label="Prefill",
            seq_len=seq_len,
            show_mode_label=index == 0,
            show_xlabels=False,
        )
        draw_capped_grouped_axis(
            ax_decode,
            decode[decode["seq_len"] == seq_len],
            mode_label="Decode",
            seq_len=seq_len,
            show_mode_label=index == 0,
            show_xlabels=True,
        )
        ax_prefill.set_title(f"L = {SEQ_LABEL[seq_len]}", fontsize=16)

    method_handles = [
        mpatches.Patch(
            facecolor=COLORS[method],
            edgecolor="black",
            hatch=NORMAL_BAR_HATCHES[method],
            label=METHOD_LABELS[method],
            linewidth=0.4,
        )
        for method in METHODS
    ]
    oom_handle = mpatches.Patch(
        facecolor=OOM_COLOR,
        edgecolor=OOM_EDGE_COLOR,
        hatch=OOM_HATCH,
        alpha=OOM_ALPHA,
        label="OOM",
    )
    fig.legend(
        handles=method_handles + [oom_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
    )
    save_figure(fig, name, extra_pdf_dirs=[LATEX_FIGURE_DIR])


def plot_prefill_decode_single_axis_capped_linear(name: str) -> None:
    """Eight length panels with Prefill and Decode in the same subplot."""
    prefill = load_prefill_completed()
    decode = load_decode_status()

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.4), sharey=False)
    axes = axes.flatten()
    width = 0.12
    method_offsets = {"MLA": -width, "FlashMLA": 0.0, "S-FMLA": width}
    mode_base = {"Prefill": 0.0, "Decode": len(BATCHES) + 1.0}
    x_positions = {
        "Prefill": np.arange(len(BATCHES)) + mode_base["Prefill"],
        "Decode": np.arange(len(BATCHES)) + mode_base["Decode"],
    }

    for ax, seq_len in zip(axes, SEQ_ORDER):
        pre_sub = prefill[prefill["seq_len"] == seq_len].copy()
        dec_sub = decode[decode["seq_len"] == seq_len].copy()
        combined = pd.concat([pre_sub.assign(panel_mode="Prefill"), dec_sub.assign(panel_mode="Decode")], ignore_index=True)

        focus = combined[combined["method_label"].isin(["FlashMLA", "S-FMLA"])]["latency_ms"].dropna()
        y_cap = (focus.max() if not focus.empty else combined["latency_ms"].dropna().max()) * 1.28
        if not np.isfinite(y_cap) or y_cap <= 0:
            y_cap = 1.0

        for mode_name in ["Prefill", "Decode"]:
            sub = combined[combined["panel_mode"] == mode_name]
            x = x_positions[mode_name]
            for method in METHODS:
                clipped_heights = []
                actual_values = []
                oom_flags = []
                missing_oom = []
                for batch in BATCHES:
                    row = sub[(sub["batch"] == batch) & (sub["method_label"] == method)].iloc[0]
                    value = row["latency_ms"]
                    is_oom = row_is_oom(row)
                    if is_oom:
                        clipped_heights.append(y_cap)
                        actual_values.append(float("nan"))
                        oom_flags.append(True)
                        missing_oom.append(True)
                    elif pd.notna(value):
                        actual = float(value)
                        clipped_heights.append(min(actual, y_cap))
                        actual_values.append(actual)
                        oom_flags.append(is_oom)
                        missing_oom.append(False)
                    else:
                        clipped_heights.append(0.0)
                        actual_values.append(float("nan"))
                        oom_flags.append(False)
                        missing_oom.append(False)

                bars = ax.bar(
                    x + method_offsets[method],
                    clipped_heights,
                    width=width,
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.35,
                    hatch=NORMAL_BAR_HATCHES[method],
                    label=METHOD_LABELS[method],
                )
                for bar, actual, is_oom, no_value in zip(bars, actual_values, oom_flags, missing_oom):
                    xpos = bar.get_x() + bar.get_width() / 2
                    if is_oom:
                        bar.set_facecolor(OOM_COLOR)
                        bar.set_edgecolor(OOM_EDGE_COLOR)
                        bar.set_hatch(OOM_HATCH)
                        bar.set_alpha(OOM_ALPHA)
                        bar.set_linewidth(0.4)
                    if no_value:
                        ax.text(
                            xpos,
                            y_cap * 0.96,
                            "OOM",
                            ha="center",
                            va="top",
                            fontsize=6.7,
                            color=OOM_TEXT_COLOR,
                            rotation=90,
                        )
                    elif np.isfinite(actual) and actual > y_cap:
                        label = format_latency(actual)
                        if is_oom:
                            label = f"OOM\n{label}"
                        ax.text(
                            xpos,
                            y_cap * 0.985,
                            label,
                            ha="center",
                            va="top",
                            fontsize=6.6,
                            rotation=90,
                            linespacing=0.75,
                        )
                        ax.plot(
                            [xpos - width * 0.35, xpos + width * 0.35],
                            [y_cap * 0.997, y_cap * 0.997],
                            color="black",
                            linewidth=0.45,
                        )

        ax.axvline(len(BATCHES) + 0.5, color="black", linewidth=0.6, alpha=0.5)
        ax.set_ylim(0, y_cap * 1.08)
        ticks = list(x_positions["Prefill"]) + list(x_positions["Decode"])
        tick_labels = [str(b) for b in BATCHES] + [str(b) for b in BATCHES]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.text(np.mean(x_positions["Prefill"]), -0.18, "Prefill", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        ax.text(np.mean(x_positions["Decode"]), -0.18, "Decode", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        ax.set_title(f"L = {SEQ_LABEL[seq_len]}", fontsize=11)
        ax.grid(axis="y", alpha=0.28, linewidth=0.5)

    axes[0].set_ylabel("Latency (ms)")
    axes[4].set_ylabel("Latency (ms)")
    method_handles = [
        mpatches.Patch(
            facecolor=COLORS[method],
            edgecolor="black",
            hatch=NORMAL_BAR_HATCHES[method],
            label=METHOD_LABELS[method],
            linewidth=0.4,
        )
        for method in METHODS
    ]
    oom_handle = mpatches.Patch(
        facecolor=OOM_COLOR,
        edgecolor=OOM_EDGE_COLOR,
        hatch=OOM_HATCH,
        alpha=OOM_ALPHA,
        label="OOM",
    )
    fig.legend(
        handles=method_handles + [oom_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
    )
    fig.supxlabel("Batch size within each mode group", y=0.045, fontsize=10)
    fig.suptitle("Prefill and decode latency in each sequence-length panel", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, name)


def plot_prefill(*, log_y: bool, hatch_estimates: bool, name: str) -> None:
    df = pd.read_csv(BASE / f"{PREFIX}_prefill_completed_long.csv")
    df["seq_len"] = df["seq_len"].astype(int)
    df["batch"] = df["batch"].astype(int)
    df["latency_ms"] = df["latency_ms"].astype(float)

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.2), sharey=log_y)
    axes = axes.flatten()
    width = 0.22
    offsets = {"MLA": -width, "FlashMLA": 0.0, "S-FMLA": width}
    x = np.arange(len(BATCHES))

    for ax, seq_len in zip(axes, SEQ_ORDER):
        sub = df[df["seq_len"] == seq_len]
        for method in METHODS:
            values = []
            kinds = []
            for batch in BATCHES:
                row = sub[(sub["batch"] == batch) & (sub["method_label"] == method)].iloc[0]
                values.append(float(row["latency_ms"]))
                kinds.append(row["value_kind"])
            bars = ax.bar(
                x + offsets[method],
                values,
                width=width,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.35,
                label=METHOD_LABELS[method],
            )
            if hatch_estimates:
                for bar, kind in zip(bars, kinds):
                    bar.set_hatch(HATCHES.get(kind, ""))
                    if kind != "measured":
                        bar.set_alpha(0.86)
        ax.set_title(f"L = {SEQ_LABEL[seq_len]}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(batch) for batch in BATCHES])
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", alpha=0.28, linewidth=0.5)
        if log_y:
            ax.set_yscale("log")
            format_y_axis(ax)

    axes[0].set_ylabel("Latency (ms)")
    axes[4].set_ylabel("Latency (ms)")
    suffix = "hatched bars are estimates" if hatch_estimates else "clean bars"
    fig.suptitle(f"Prefill latency by sequence length and batch size ({suffix})", y=1.02, fontsize=13)
    fig.legend(
        handles=make_legend(include_estimates=hatch_estimates),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=6 if hatch_estimates else 3,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, name)


def plot_decode(name: str) -> None:
    df = pd.read_csv(BASE / f"{PREFIX}_full_long.csv")
    df = df[df["mode"] == "decode"].copy()
    df["seq_len"] = df["seq_len"].astype(int)
    df["batch"] = df["batch"].astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.2), sharey=True)
    axes = axes.flatten()
    width = 0.22
    offsets = {"MLA": -width, "FlashMLA": 0.0, "S-FMLA": width}
    x = np.arange(len(BATCHES))

    for ax, seq_len in zip(axes, SEQ_ORDER):
        sub = df[df["seq_len"] == seq_len]
        plotted_values = []
        missing_positions = []
        for method in METHODS:
            xpos = []
            values = []
            for idx, batch in enumerate(BATCHES):
                row = sub[(sub["batch"] == batch) & (sub["method_label"] == method)].iloc[0]
                pos = x[idx] + offsets[method]
                if row["status"] == "ok" and pd.notna(row["latency_ms"]):
                    latency_ms = float(row["latency_ms"])
                    xpos.append(pos)
                    values.append(latency_ms)
                    plotted_values.append(latency_ms)
                else:
                    missing_positions.append((pos, row["status"]))
            if values:
                ax.bar(
                    xpos,
                    values,
                    width=width,
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.35,
                    label=METHOD_LABELS[method],
                )
        ax.set_yscale("log")
        format_y_axis(ax)
        y_top = max(plotted_values) * 1.5 if plotted_values else 1.0
        for pos, status in missing_positions:
            ax.scatter([pos], [y_top], marker="x", color="black", s=22, linewidths=1.1, zorder=5)
            ax.text(pos, y_top * 1.05, status, ha="center", va="bottom", fontsize=6.8, rotation=90)
        ax.set_title(f"L = {SEQ_LABEL[seq_len]}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(batch) for batch in BATCHES])
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", alpha=0.28, linewidth=0.5)

    axes[0].set_ylabel("Latency (ms)")
    axes[4].set_ylabel("Latency (ms)")
    fig.suptitle("Decode latency by sequence length and batch size (measured)", y=1.02, fontsize=13)
    fig.legend(
        handles=make_legend(include_estimates=False, include_oom_marker=True),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, name)


def plot_prefill_decode_compact(name: str) -> None:
    """Render a smaller mixed-mode reference with representative sequence lengths."""
    prefill = pd.read_csv(BASE / f"{PREFIX}_prefill_completed_long.csv")
    decode = pd.read_csv(BASE / f"{PREFIX}_full_long.csv")
    decode = decode[(decode["mode"] == "decode") & (decode["status"] == "ok")].copy()
    for frame in [prefill, decode]:
        frame["seq_len"] = frame["seq_len"].astype(int)
        frame["batch"] = frame["batch"].astype(int)
        frame["latency_ms"] = frame["latency_ms"].astype(float)

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), sharey=False)
    width = 0.22
    offsets = {"MLA": -width, "FlashMLA": 0.0, "S-FMLA": width}
    x = np.arange(len(BATCHES))
    chosen_seq_lens = [256, 1024, 4096, 32768]

    for col, seq_len in enumerate(chosen_seq_lens):
        for row_idx, (mode_name, frame) in enumerate([("Prefill", prefill), ("Decode", decode)]):
            ax = axes[row_idx, col]
            sub = frame[frame["seq_len"] == seq_len]
            for method in METHODS:
                xpos = []
                values = []
                kinds = []
                for idx, batch in enumerate(BATCHES):
                    matches = sub[(sub["batch"] == batch) & (sub["method_label"] == method)]
                    if matches.empty or pd.isna(matches.iloc[0]["latency_ms"]):
                        continue
                    xpos.append(x[idx] + offsets[method])
                    values.append(float(matches.iloc[0]["latency_ms"]))
                    kinds.append(matches.iloc[0].get("value_kind", "measured"))
                bars = ax.bar(
                    xpos,
                    values,
                    width=width,
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.35,
                )
                if mode_name == "Prefill":
                    for bar, kind in zip(bars, kinds):
                        bar.set_hatch(HATCHES.get(kind, ""))
                        if kind != "measured":
                            bar.set_alpha(0.86)
            ax.set_yscale("log")
            format_y_axis(ax)
            ax.set_title(f"{mode_name}, L={SEQ_LABEL[seq_len]}")
            ax.set_xticks(x)
            ax.set_xticklabels([str(batch) for batch in BATCHES])
            ax.grid(axis="y", alpha=0.28, linewidth=0.5)
            if col == 0:
                ax.set_ylabel("Latency (ms)")
            if row_idx == 1:
                ax.set_xlabel("Batch size")

    fig.suptitle("Reference compact layout: prefill estimates + decode measured", y=1.02, fontsize=13)
    fig.legend(
        handles=make_legend(include_estimates=True),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=6,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, name)


def main() -> None:
    plot_prefill_decode_capped_linear(name="prefill_decode_bars_oom")
    plot_prefill_decode_single_axis_capped_linear(
        name="prefill_decode_8panel_single_axis_grouped_bars_linear_capped_oom"
    )


if __name__ == "__main__":
    main()

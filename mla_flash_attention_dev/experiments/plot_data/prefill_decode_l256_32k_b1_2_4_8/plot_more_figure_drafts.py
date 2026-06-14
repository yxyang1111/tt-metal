#!/usr/bin/env python3
"""Render additional draft figure styles for MLA/FlashMLA/S-FMLA data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "figure_drafts"
OUT.mkdir(exist_ok=True)
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
COLORS = {"MLA": "#D55E00", "FlashMLA": "#0072B2", "S-FMLA": "#009E73"}
MARKERS = {"MLA": "o", "FlashMLA": "s", "S-FMLA": "^"}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def format_seq_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQ_ORDER)
    ax.set_xticklabels([SEQ_LABEL[v] for v in SEQ_ORDER], rotation=35, ha="right")


def format_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def load_prefill() -> pd.DataFrame:
    df = pd.read_csv(BASE / f"{PREFIX}_prefill_completed_long.csv")
    df["seq_len"] = df["seq_len"].astype(int)
    df["batch"] = df["batch"].astype(int)
    df["latency_ms"] = df["latency_ms"].astype(float)
    return df


def load_decode() -> pd.DataFrame:
    df = pd.read_csv(BASE / f"{PREFIX}_full_long.csv")
    df = df[df["mode"] == "decode"].copy()
    df["seq_len"] = df["seq_len"].astype(int)
    df["batch"] = df["batch"].astype(int)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    return df


def plot_lines_by_batch(df: pd.DataFrame, mode: str, name: str, estimated: bool) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.7), sharey=True)
    for ax, batch in zip(axes, BATCHES):
        sub = df[df["batch"] == batch]
        for method in METHODS:
            m = sub[sub["method_label"] == method].sort_values("seq_len")
            ok = m[m["latency_ms"].notna()]
            if ok.empty:
                continue
            ax.plot(
                ok["seq_len"],
                ok["latency_ms"],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                markersize=4.5,
                label=method,
            )
            if estimated and "value_kind" in ok:
                est = ok[ok["value_kind"].isin(["estimated", "estimated_oom"])]
                if not est.empty:
                    ax.scatter(
                        est["seq_len"],
                        est["latency_ms"],
                        s=45,
                        facecolors="white",
                        edgecolors=COLORS[method],
                        marker=MARKERS[method],
                        linewidths=1.2,
                        zorder=5,
                    )
                    oom = est[est["value_kind"] == "estimated_oom"]
                    if not oom.empty:
                        ax.scatter(
                            oom["seq_len"],
                            oom["latency_ms"],
                            s=58,
                            color=COLORS[method],
                            marker="x",
                            linewidths=1.2,
                            zorder=6,
                        )
        format_seq_axis(ax)
        format_log_y(ax)
        ax.set_title(f"Batch = {batch}")
        ax.set_xlabel("Sequence length")
        ax.grid(True, alpha=0.28, which="both", linewidth=0.5)
    axes[0].set_ylabel("Latency (ms)")
    fig.suptitle(f"{mode} latency scaling by batch", y=1.05, fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    extra = []
    if estimated:
        extra = [
            plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white", linestyle="None", label="Estimated"),
            plt.Line2D([0], [0], marker="x", color="black", linestyle="None", label="Estimated from OOM"),
        ]
    fig.legend(handles + extra, labels + [h.get_label() for h in extra], loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
    fig.tight_layout()
    save(fig, name)


def plot_lines_by_method(df: pd.DataFrame, mode: str, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
    batch_colors = {1: "#1f77b4", 2: "#9467bd", 4: "#e377c2", 8: "#7f7f7f"}
    for ax, method in zip(axes, METHODS):
        sub = df[df["method_label"] == method]
        for batch in BATCHES:
            b = sub[(sub["batch"] == batch) & sub["latency_ms"].notna()].sort_values("seq_len")
            if b.empty:
                continue
            ax.plot(b["seq_len"], b["latency_ms"], marker="o", linewidth=1.7, markersize=4, label=f"B={batch}", color=batch_colors[batch])
        format_seq_axis(ax)
        format_log_y(ax)
        ax.set_title(method)
        ax.set_xlabel("Sequence length")
        ax.grid(True, alpha=0.28, which="both", linewidth=0.5)
    axes[0].set_ylabel("Latency (ms)")
    fig.suptitle(f"{mode} latency scaling by method", y=1.05, fontsize=12)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    fig.tight_layout()
    save(fig, name)


def pivot_latency(df: pd.DataFrame, method: str) -> np.ndarray:
    sub = df[df["method_label"] == method]
    table = sub.pivot(index="batch", columns="seq_len", values="latency_ms")
    table = table.reindex(index=BATCHES, columns=SEQ_ORDER)
    return table.to_numpy(dtype=float)


def draw_heatmap(ax: plt.Axes, data: np.ndarray, title: str, cmap: str, cbar_label: str) -> None:
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(SEQ_ORDER)))
    ax.set_xticklabels([SEQ_LABEL[v] for v in SEQ_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(BATCHES)))
    ax.set_yticklabels([str(v) for v in BATCHES])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Batch size")
    for i in range(len(BATCHES)):
        for j in range(len(SEQ_ORDER)):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6, color="black")
            else:
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=6, color="black")
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_latency_heatmaps(df: pd.DataFrame, mode: str, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8))
    for ax, method in zip(axes, METHODS):
        latency = pivot_latency(df, method)
        draw_heatmap(ax, np.log10(latency), f"{method}", "viridis", "log10 latency (ms)")
    fig.suptitle(f"{mode} latency heatmaps", y=1.04, fontsize=12)
    fig.tight_layout()
    save(fig, name)


def plot_speedup_heatmaps(df: pd.DataFrame, mode: str, name: str) -> None:
    rows = []
    for batch in BATCHES:
        for seq_len in SEQ_ORDER:
            sub = df[(df["batch"] == batch) & (df["seq_len"] == seq_len)]
            def val(method: str) -> float:
                m = sub[sub["method_label"] == method]
                if m.empty:
                    return float("nan")
                return float(m.iloc[0]["latency_ms"])
            sfmla = val("S-FMLA")
            rows.append(
                {
                    "batch": batch,
                    "seq_len": seq_len,
                    "FlashMLA / S-FMLA": val("FlashMLA") / sfmla if sfmla else float("nan"),
                    "MLA / S-FMLA": val("MLA") / sfmla if sfmla else float("nan"),
                }
            )
    ratio_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, col in zip(axes, ["FlashMLA / S-FMLA", "MLA / S-FMLA"]):
        table = ratio_df.pivot(index="batch", columns="seq_len", values=col).reindex(index=BATCHES, columns=SEQ_ORDER)
        draw_heatmap(ax, table.to_numpy(dtype=float), col, "YlOrRd", "Latency ratio")
    fig.suptitle(f"{mode} speedup heatmaps (higher means S-FMLA is faster)", y=1.04, fontsize=12)
    fig.tight_layout()
    save(fig, name)


def plot_speedup_bars_by_length(df: pd.DataFrame, mode: str, name: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.0), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(BATCHES))
    width = 0.34
    for ax, seq_len in zip(axes, SEQ_ORDER):
        flash_ratios = []
        mla_ratios = []
        for batch in BATCHES:
            sub = df[(df["batch"] == batch) & (df["seq_len"] == seq_len)]
            vals = {r["method_label"]: float(r["latency_ms"]) for _, r in sub[sub["latency_ms"].notna()].iterrows()}
            sfmla = vals.get("S-FMLA")
            flash_ratios.append(vals.get("FlashMLA", np.nan) / sfmla if sfmla else np.nan)
            mla_ratios.append(vals.get("MLA", np.nan) / sfmla if sfmla else np.nan)
        ax.bar(x - width / 2, flash_ratios, width, color=COLORS["FlashMLA"], edgecolor="black", linewidth=0.35, label="FlashMLA / S-FMLA")
        ax.bar(x + width / 2, mla_ratios, width, color=COLORS["MLA"], edgecolor="black", linewidth=0.35, label="MLA / S-FMLA")
        ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(f"L = {SEQ_LABEL[seq_len]}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in BATCHES])
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", alpha=0.28, linewidth=0.5)
    axes[0].set_ylabel("Latency ratio vs S-FMLA")
    axes[4].set_ylabel("Latency ratio vs S-FMLA")
    fig.suptitle(f"{mode} relative latency by sequence length", y=1.02, fontsize=12)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, name)


def plot_prefill_value_kind_grid(df: pd.DataFrame) -> None:
    kind_to_value = {"measured": 0, "estimated": 1, "estimated_oom": 2}
    cmap = plt.matplotlib.colors.ListedColormap(["#4daf4a", "#ffcc66", "#e41a1c"])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, method in zip(axes, METHODS):
        sub = df[df["method_label"] == method]
        table = sub.pivot(index="batch", columns="seq_len", values="value_kind").reindex(index=BATCHES, columns=SEQ_ORDER)
        values = table.map(kind_to_value.get).to_numpy(dtype=float)
        ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_title(method)
        ax.set_xticks(np.arange(len(SEQ_ORDER)))
        ax.set_xticklabels([SEQ_LABEL[v] for v in SEQ_ORDER], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(BATCHES)))
        ax.set_yticklabels([str(v) for v in BATCHES])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Batch size")
        for i in range(len(BATCHES)):
            for j in range(len(SEQ_ORDER)):
                text = {"measured": "M", "estimated": "E", "estimated_oom": "OOM"}[table.iloc[i, j]]
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
    legend = [
        mpatches.Patch(color="#4daf4a", label="Measured"),
        mpatches.Patch(color="#ffcc66", label="Estimated"),
        mpatches.Patch(color="#e41a1c", label="Estimated from MLA OOM"),
    ]
    fig.suptitle("Prefill value provenance grid", y=1.04, fontsize=12)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "draft_prefill_value_kind_grid")


def main() -> None:
    prefill = load_prefill()
    decode = load_decode()
    plot_lines_by_batch(prefill, "Prefill", "draft_prefill_lines_by_batch", estimated=True)
    plot_lines_by_batch(decode, "Decode", "draft_decode_lines_by_batch", estimated=False)
    plot_lines_by_method(prefill, "Prefill", "draft_prefill_lines_by_method")
    plot_lines_by_method(decode, "Decode", "draft_decode_lines_by_method")
    plot_latency_heatmaps(prefill, "Prefill", "draft_prefill_latency_heatmaps_by_method")
    plot_latency_heatmaps(decode, "Decode", "draft_decode_latency_heatmaps_by_method")
    plot_speedup_heatmaps(prefill, "Prefill", "draft_prefill_speedup_heatmaps")
    plot_speedup_heatmaps(decode, "Decode", "draft_decode_speedup_heatmaps")
    plot_speedup_bars_by_length(prefill, "Prefill", "draft_prefill_speedup_bars_by_length")
    plot_speedup_bars_by_length(decode, "Decode", "draft_decode_speedup_bars_by_length")
    plot_prefill_value_kind_grid(prefill)


if __name__ == "__main__":
    main()

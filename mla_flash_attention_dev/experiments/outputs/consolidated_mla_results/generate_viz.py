#!/usr/bin/env python3
"""Generate MLA benchmark visualizations from consolidated CSV data."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent
VIZ = BASE / "viz"
VIZ.mkdir(exist_ok=True)

# Style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
    }
)
PALETTE = {
    "S-FMLA (WH)": "#2563eb",
    "FlashMLA (WH)": "#7c3aed",
    "MLA baseline (WH)": "#dc2626",
    "L40S FlashMLA": "#059669",
    "L40S FlashMLA /3": "#34d399",
}

METHOD_COLS = {
    "sfmla_ms": "S-FMLA (WH)",
    "flash_mla_ms": "FlashMLA (WH)",
    "mla_ms": "MLA baseline (WH)",
    "l40s_flashmla_ms": "L40S FlashMLA",
    "l40s_div3_flashmla_ms": "L40S FlashMLA /3",
}


def load_data():
    decode = pd.read_csv(BASE / "plot_ready_decode.csv")
    prefill = pd.read_csv(BASE / "plot_ready_prefill.csv")
    speedup = pd.read_csv(BASE / "sfmla_speedup_summary.csv")
    long_df = pd.read_csv(BASE / "all_methods_long.csv")
    return decode, prefill, speedup, long_df


def seq_order(df):
    return df.sort_values("seq_len")


def plot_latency_curves(df, mode, batches=(1, 8, 32), fname=""):
    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4.5), sharey=True)
    if len(batches) == 1:
        axes = [axes]

    for ax, batch in zip(axes, batches):
        sub = seq_order(df[df["batch"] == batch])
        for col, label in METHOD_COLS.items():
            status_col = col.replace("_ms", "_status")
            if status_col not in sub.columns:
                continue
            mask = sub[status_col] == "ok"
            y = sub.loc[mask, col]
            if y.empty:
                continue
            ax.plot(
                sub.loc[mask, "seq_len"],
                y,
                marker="o",
                markersize=4,
                label=label,
                color=PALETTE[label],
                linewidth=1.8,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sequence length")
        ax.set_title(f"Batch = {batch}")
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x // 1024)}K" if x >= 1024 else str(int(x)))
        )

    axes[0].set_ylabel("Latency (ms)")
    fig.suptitle(f"{mode.capitalize()} latency vs sequence length (Wormhole + L40S reference)", y=1.02)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(VIZ / fname, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_heatmap(speedup, mode, ratio_col, title, fname, vmin=0, vmax=3):
    sub = speedup[speedup["mode"] == mode].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="batch", columns="seq_label", values=ratio_col)
    # order columns by seq_len
    order = sub.drop_duplicates("seq_label").sort_values("seq_len")["seq_label"].tolist()
    pivot = pivot[[c for c in order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.7), 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"label": "Latency ratio (method / S-FMLA)"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Batch size")
    fig.tight_layout()
    fig.savefig(VIZ / fname, bbox_inches="tight")
    plt.close(fig)


def plot_sfmla_vs_l40s(speedup, mode, fname):
    sub = speedup[(speedup["mode"] == mode) & speedup["l40s_div3_over_sfmla"].notna()].copy()
    if sub.empty:
        return
    sub["wh_faster"] = sub["l40s_div3_over_sfmla"] < 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    for batch in sorted(sub["batch"].unique()):
        bsub = seq_order(sub[sub["batch"] == batch])
        ax.plot(
            bsub["seq_len"],
            bsub["l40s_div3_over_sfmla"],
            marker="o",
            label=f"B={batch}",
            linewidth=1.5,
            markersize=4,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Parity (ratio=1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("L40S FlashMLA/3 ÷ S-FMLA latency")
    ax.set_title(f"{mode.capitalize()}: bandwidth-normalized L40S vs Wormhole S-FMLA")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x // 1024)}K" if x >= 1024 else str(int(x)))
    )
    fig.tight_layout()
    fig.savefig(VIZ / fname, bbox_inches="tight")
    plt.close(fig)


def plot_flash_vs_sfmla_scatter(speedup, mode, fname):
    sub = speedup[(speedup["mode"] == mode) & speedup["flashmla_over_sfmla"].notna()].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        sub["sfmla_ms"],
        sub["sfmla_ms"] * sub["flashmla_over_sfmla"],
        c=sub["batch"],
        cmap="viridis",
        s=40,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )
    lim = max(sub["sfmla_ms"].max(), (sub["sfmla_ms"] * sub["flashmla_over_sfmla"]).max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Parity")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("S-FMLA latency (ms)")
    ax.set_ylabel("FlashMLA latency (ms)")
    ax.set_title(f"{mode.capitalize()}: FlashMLA vs S-FMLA (WH)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Batch size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ / fname, bbox_inches="tight")
    plt.close(fig)


def plot_mla_speedup_vs_batch(speedup, mode, fname):
    sub = speedup[(speedup["mode"] == mode) & speedup["mla_over_sfmla"].notna()].copy()
    if sub.empty:
        return
    # pick representative seq lengths
    picks = [256, 4096, 32768, 131072]
    labels = ["256", "4K", "32K", "128K"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sl, lab in zip(picks, labels):
        s = sub[sub["seq_len"] == sl].sort_values("batch")
        if s.empty:
            continue
        ax.plot(s["batch"], s["mla_over_sfmla"], marker="o", label=f"L={lab}", linewidth=1.5)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("MLA baseline / S-FMLA latency")
    ax.set_title(f"{mode.capitalize()}: baseline MLA slowdown vs S-FMLA")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Seq len")
    fig.tight_layout()
    fig.savefig(VIZ / fname, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(long_df, fname):
    wh = long_df[long_df["platform"].str.contains("Wormhole", na=False)]
    summary = (
        wh.groupby(["method", "mode", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["ok", "error", "skipped"], fill_value=0)
    )
    methods = ["sfmla", "flash_mla", "mla"]
    modes = ["decode", "prefill"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"ok": "#22c55e", "error": "#ef4444", "skipped": "#94a3b8"}
    for ax, mode in zip(axes, modes):
        data = []
        for m in methods:
            row = summary.loc[(m, mode)] if (m, mode) in summary.index else pd.Series({"ok": 0, "error": 0, "skipped": 0})
            data.append([row.get(s, 0) for s in ["ok", "error", "skipped"]])
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods))
        for i, st in enumerate(["ok", "error", "skipped"]):
            vals = [d[i] for d in data]
            ax.bar(x, vals, bottom=bottom, label=st, color=colors[st], width=0.6)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(["S-FMLA", "FlashMLA", "MLA baseline"])
        ax.set_title(mode.capitalize())
        ax.set_ylabel("Case count")
    axes[0].legend(title="Status", loc="upper left")
    fig.suptitle("Wormhole benchmark coverage by method and mode", y=1.02)
    fig.tight_layout()
    fig.savefig(VIZ / fname, bbox_inches="tight")
    plt.close(fig)


def compute_summary_stats(decode, prefill, speedup):
    lines = []

    def stat_block(df, mode):
        ok_sfmla = df[df["sfmla_status"] == "ok"]
        ok_flash = df[df["flash_mla_status"] == "ok"]
        ok_sfmla_sub = ok_sfmla[["batch", "seq_len", "sfmla_ms", "l40s_div3_flashmla_ms"]]
        both = ok_sfmla_sub.merge(
            ok_flash[["batch", "seq_len", "flash_mla_ms"]],
            on=["batch", "seq_len"],
        )
        if both.empty:
            return
        ratio = both["flash_mla_ms"] / both["sfmla_ms"]
        lines.append(f"### {mode.capitalize()}")
        lines.append(
            f"- 有效 S-FMLA 测点: {len(ok_sfmla)}；FlashMLA 有效: {len(ok_flash)}；两者同时有效: {len(both)}"
        )
        lines.append(
            f"- FlashMLA / S-FMLA: median={ratio.median():.3f}x, mean={ratio.mean():.3f}x, "
            f"min={ratio.min():.3f}x, max={ratio.max():.3f}x"
        )
        faster = (ratio < 1.0).sum()
        lines.append(f"- FlashMLA 快于 S-FMLA 的配置: {faster}/{len(ratio)} ({100*faster/len(ratio):.1f}%)")

        mla_ok = df[df["mla_status"] == "ok"]
        mla_both = ok_sfmla_sub.merge(
            mla_ok[["batch", "seq_len", "mla_ms"]], on=["batch", "seq_len"], how="inner"
        )
        if not mla_both.empty:
            mla_ratio = mla_both["mla_ms"] / mla_both["sfmla_ms"]
            lines.append(
                f"- MLA baseline / S-FMLA: median={mla_ratio.median():.1f}x, max={mla_ratio.max():.1f}x"
            )

        l40s = ok_sfmla_sub[ok_sfmla_sub["l40s_div3_flashmla_ms"].notna()]
        if not l40s.empty:
            l40s_ratio = l40s["l40s_div3_flashmla_ms"] / l40s["sfmla_ms"]
            wh_wins = (l40s_ratio > 1.0).sum()
            lines.append(
                f"- L40S/3 vs S-FMLA: median ratio={l40s_ratio.median():.3f}x; "
                f"S-FMLA 更快于 L40S/3 的配置: {wh_wins}/{len(l40s_ratio)} ({100*wh_wins/len(l40s_ratio):.1f}%)"
            )
        lines.append("")

    lines.append("## 核心统计摘要\n")
    stat_block(decode, "decode")
    stat_block(prefill, "prefill")

    # decode: where S-FMLA beats bandwidth-normalized L40S
    dec = speedup[(speedup["mode"] == "decode") & speedup["l40s_div3_over_sfmla"].notna()].copy()
    if not dec.empty:
        sfmla_wins = dec.nlargest(5, "l40s_div3_over_sfmla")[
            ["batch", "seq_label", "sfmla_ms", "l40s_div3_over_sfmla"]
        ]
        lines.append("### Decode — S-FMLA 快于 L40S/3 的 Top 5 配置")
        lines.append("(ratio = L40S/3 ÷ S-FMLA，>1 表示 S-FMLA 更快)\n")
        for _, r in sfmla_wins.iterrows():
            lines.append(
                f"- B={int(r.batch)}, L={r.seq_label}: S-FMLA={r.sfmla_ms:.3f}ms, ratio={r.l40s_div3_over_sfmla:.3f}x"
            )
        lines.append("")

        l40s_wins = dec.nsmallest(5, "l40s_div3_over_sfmla")[
            ["batch", "seq_label", "sfmla_ms", "l40s_div3_over_sfmla"]
        ]
        lines.append("### Decode — L40S/3 快于 S-FMLA 的 Top 5 配置")
        lines.append("(ratio < 1 表示 L40S/3 更快)\n")
        for _, r in l40s_wins.iterrows():
            lines.append(
                f"- B={int(r.batch)}, L={r.seq_label}: S-FMLA={r.sfmla_ms:.3f}ms, ratio={r.l40s_div3_over_sfmla:.3f}x"
            )
        lines.append("")

    pref = speedup[(speedup["mode"] == "prefill") & speedup["l40s_div3_over_sfmla"].notna()].copy()
    if not pref.empty:
        l40s_wins_pref = pref.nsmallest(3, "l40s_div3_over_sfmla")[
            ["batch", "seq_label", "sfmla_ms", "l40s_div3_over_sfmla"]
        ]
        lines.append("### Prefill — L40S/3 快于 S-FMLA 的配置 (短 seq)")
        for _, r in l40s_wins_pref.iterrows():
            lines.append(
                f"- B={int(r.batch)}, L={r.seq_label}: ratio={r.l40s_div3_over_sfmla:.2f}x"
            )
        sfmla_wins_pref = pref.nlargest(3, "l40s_div3_over_sfmla")[
            ["batch", "seq_label", "sfmla_ms", "l40s_div3_over_sfmla"]
        ]
        lines.append("")
        lines.append("### Prefill — S-FMLA 快于 L40S/3 的配置 (长 seq / 大 batch)")
        for _, r in sfmla_wins_pref.iterrows():
            lines.append(
                f"- B={int(r.batch)}, L={r.seq_label}: ratio={r.l40s_div3_over_sfmla:.2f}x"
            )

    return "\n".join(lines)


def main():
    decode, prefill, speedup, long_df = load_data()

    plot_latency_curves(decode, "decode", batches=(1, 8, 32), fname="01_decode_latency_curves.png")
    plot_latency_curves(prefill, "prefill", batches=(1, 8, 32), fname="02_prefill_latency_curves.png")

    plot_speedup_heatmap(
        speedup,
        "decode",
        "flashmla_over_sfmla",
        "Decode: FlashMLA / S-FMLA latency ratio (<1 = FlashMLA faster)",
        "03_decode_flash_over_sfmla_heatmap.png",
        vmin=0.8,
        vmax=1.6,
    )
    plot_speedup_heatmap(
        speedup,
        "decode",
        "mla_over_sfmla",
        "Decode: MLA baseline / S-FMLA latency ratio",
        "04_decode_mla_over_sfmla_heatmap.png",
        vmin=1,
        vmax=60,
    )
    plot_speedup_heatmap(
        speedup,
        "prefill",
        "flashmla_over_sfmla",
        "Prefill: FlashMLA / S-FMLA latency ratio",
        "05_prefill_flash_over_sfmla_heatmap.png",
        vmin=0.9,
        vmax=2.5,
    )
    plot_speedup_heatmap(
        speedup,
        "decode",
        "l40s_div3_over_sfmla",
        "Decode: L40S FlashMLA/3 / S-FMLA (<1 = S-FMLA faster than normalized L40S)",
        "06_decode_l40s_div3_over_sfmla_heatmap.png",
        vmin=0,
        vmax=2.5,
    )

    plot_sfmla_vs_l40s(speedup, "decode", "07_decode_sfmla_vs_l40s_div3.png")
    plot_sfmla_vs_l40s(speedup, "prefill", "08_prefill_sfmla_vs_l40s_div3.png")
    plot_flash_vs_sfmla_scatter(speedup, "decode", "09_decode_flash_vs_sfmla_scatter.png")
    plot_flash_vs_sfmla_scatter(speedup, "prefill", "10_prefill_flash_vs_sfmla_scatter.png")
    plot_mla_speedup_vs_batch(speedup, "decode", "11_decode_mla_slowdown_vs_batch.png")
    plot_coverage(long_df, "12_wh_coverage_status.png")

    stats = compute_summary_stats(decode, prefill, speedup)
    analysis_path = BASE / "analysis_report.md"
    header = """# MLA Flash Attention 基准测试结果分析

> 数据来源: `consolidated_mla_results/`  
> 平台: Wormhole N300s (WH) vs NVIDIA L40S FlashInfer MLA 参考  
> 方法: S-FMLA, FlashMLA, MLA baseline, L40S FlashMLA (raw & /3 带宽归一化)

---

## 数据概览

| 指标 | 数值 |
|------|------|
| 总测点 | 560 (ok=383, error=9, skipped=168) |
| L40S 参考 | 140 点 (全部 ok) |
| Wormhole | 420 点 (ok=243, error=9, skipped=168) |
| Decode sweep | batch ∈ {1,2,4,8,16,32,64}, seq ∈ {256..128K} |
| Prefill sweep | 同上；seq>4K 或 batch×seq>16K 多为 skipped (安全上限) |

### 方法对比说明

- **S-FMLA**: Wormhole 上优化的 sparse flash MLA 实现（本报告主要 baseline）
- **FlashMLA**: Wormhole flash MLA 实现
- **MLA baseline**: Wormhole 标准 MLA
- **L40S FlashMLA**: GPU 参考；**/3** 为带宽归一化（约 3× 原始延迟），用于与 WH 公平对比

---

## 可视化图表

图表保存在 `viz/` 目录:

| 文件 | 内容 |
|------|------|
| `01_decode_latency_curves.png` | Decode 延迟随 seq_len 变化 (B=1,8,32) |
| `02_prefill_latency_curves.png` | Prefill 延迟曲线 |
| `03_decode_flash_over_sfmla_heatmap.png` | Decode FlashMLA/S-FMLA 比值热力图 |
| `04_decode_mla_over_sfmla_heatmap.png` | Decode MLA baseline 相对 S-FMLA 慢多少 |
| `05_prefill_flash_over_sfmla_heatmap.png` | Prefill FlashMLA/S-FMLA |
| `06_decode_l40s_div3_over_sfmla_heatmap.png` | Decode L40S/3 vs S-FMLA |
| `07-08_*` | L40S/3 vs S-FMLA 曲线 (decode/prefill) |
| `09-10_*` | FlashMLA vs S-FMLA 散点图 |
| `11_*` | MLA baseline 随 batch 放大倍数 |
| `12_wh_coverage_status.png` | WH 测试覆盖/跳过/失败统计 |

---

"""
    analysis_path.write_text(header + stats, encoding="utf-8")
    print(f"Wrote {analysis_path}")
    print(f"Plots saved to {VIZ}")


if __name__ == "__main__":
    main()

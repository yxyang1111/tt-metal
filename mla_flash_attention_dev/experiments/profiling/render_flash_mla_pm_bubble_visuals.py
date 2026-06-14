#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from render_flash_mla_profile_visuals import (
    MARGIN_LEFT,
    MARGIN_RIGHT,
    SVG_HEIGHT,
    SVG_WIDTH,
    format_seq_len,
    render_legend,
    render_line_chart,
    render_stacked_bar_chart,
    svg_escape,
    write_svg,
    write_text,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "profiling" / "outputs" / "flash_mla_pm_bubble_probe_decode_safe"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
DETAILED_JSON = ROOT / "profiling" / "outputs" / "flash_mla_wh_detailed" / "flash_mla_wh_detailed_profile_results.json"
VISUAL_DIR = OUTPUT_DIR / "visuals"
SUMMARY_CSV = OUTPUT_DIR / "flash_mla_pm_bubble_decode_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "flash_mla_pm_bubble_summary.md"
DASHBOARD_HTML = OUTPUT_DIR / "flash_mla_pm_bubble_dashboard.html"

STATE_COLORS = {
    "ok": "#2ec4b6",
    "placeholder": "#ffca3a",
    "missing": "#ff595e",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mean_numeric(series: pd.Series, *, positive_only: bool = False) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if positive_only:
        values = values[values > 0]
    if values.empty:
        return None
    return float(values.mean())


def stage_share(entry: dict[str, Any], breakdown_key: str, marker: str) -> float:
    return float(entry.get(breakdown_key, {}).get("stage_shares", {}).get(marker, 0.0))


def measured_rows(csv_path: Path, workload: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    df = df[df["OP TYPE"] != "signpost"].copy()

    relevant_rows = int(workload["warmup_iterations"]) + int(workload["iterations"])
    if len(df) < relevant_rows:
        raise RuntimeError(
            f"Expected at least {relevant_rows} profiled rows in {csv_path}, only found {len(df)}."
        )

    df = df.tail(relevant_rows).reset_index(drop=True)
    return df.iloc[int(workload["warmup_iterations"]) :].copy()


def pm_triplet_state(summary: dict[str, Any]) -> str:
    values = [summary["pm_ideal_ns"], summary["pm_compute_ns"], summary["pm_bandwidth_ns"]]
    if all(value is None for value in values):
        return "missing"
    if values == [1.0, 1.0, 1.0]:
        return "placeholder"
    return "ok"


def pm_fpu_state(summary: dict[str, Any]) -> str:
    value = summary["pm_fpu_util_pct"]
    if value is None:
        return "missing"
    if value <= 0.01:
        return "placeholder"
    return "ok"


def util_state(value: float | None) -> str:
    return "ok" if value is not None else "missing"


def fmt_or_missing(value: float | None, fmt: str) -> str:
    if value is None:
        return "missing"
    return fmt.format(value)


def render_status_grid(
    title: str,
    subtitle: str,
    row_labels: list[str],
    column_labels: list[str],
    states: list[list[str]],
) -> str:
    left = 190
    top = 108
    grid_width = SVG_WIDTH - left - MARGIN_RIGHT
    grid_height = SVG_HEIGHT - top - 36
    col_width = grid_width / max(len(column_labels), 1)
    row_height = grid_height / max(len(row_labels), 1)

    parts = [
        f'<rect width="100%" height="100%" rx="22" fill="#111722"></rect>',
        f'<text x="{MARGIN_LEFT}" y="34" font-size="24" font-weight="700" fill="#f5f7fb">{svg_escape(title)}</text>',
        f'<text x="{MARGIN_LEFT}" y="56" font-size="13" fill="#9fb0c3">{svg_escape(subtitle)}</text>',
        render_legend(
            [
                ("ok", STATE_COLORS["ok"]),
                ("placeholder", STATE_COLORS["placeholder"]),
                ("missing", STATE_COLORS["missing"]),
            ],
            MARGIN_LEFT,
            76,
        ),
    ]

    for index, label in enumerate(column_labels):
        x = left + col_width * index + col_width / 2.0
        parts.append(
            f'<text x="{x:.2f}" y="{top - 14:.2f}" text-anchor="middle" font-size="12" fill="#c7d2df">{svg_escape(label)}</text>'
        )

    for row_index, row_label in enumerate(row_labels):
        y = top + row_height * row_index
        parts.append(
            f'<text x="{left - 12}" y="{y + row_height / 2.0 + 4:.2f}" text-anchor="end" font-size="12" fill="#c7d2df">{svg_escape(row_label)}</text>'
        )
        for col_index, state in enumerate(states[row_index]):
            x = left + col_width * col_index
            parts.append(
                f'<rect x="{x + 6:.2f}" y="{y + 4:.2f}" width="{col_width - 12:.2f}" height="{row_height - 8:.2f}" '
                f'rx="10" fill="{STATE_COLORS[state]}" opacity="0.92"></rect>'
            )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">{"".join(parts)}</svg>'
    )


def build_case_summaries() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(MANIFEST_PATH)
    detailed = load_json(DETAILED_JSON)
    detailed_map = {entry["case"]: entry for entry in detailed["profiles"] if entry["mode"] == "decode"}

    completed_cases = [case for case in manifest["cases"] if case["status"] == "completed"]
    completed_cases.sort(key=lambda item: detailed_map[item["case"]]["workload"]["seq_len"])

    summaries: list[dict[str, Any]] = []
    for case_meta in completed_cases:
        entry = detailed_map[case_meta["case"]]
        workload = entry["workload"]
        df = measured_rows(Path(case_meta["ops_csv"]), workload)

        trisc_candidates = [
            mean_numeric(df[f"DEVICE TRISC{idx} KERNEL DURATION [ns]"], positive_only=True) for idx in range(3)
        ]
        trisc_candidates = [value for value in trisc_candidates if value is not None]
        if not trisc_candidates:
            raise RuntimeError(f"No TRISC duration found in {case_meta['ops_csv']}.")

        compute_ns = max(trisc_candidates)
        kernel_ns = mean_numeric(df["DEVICE KERNEL DURATION [ns]"], positive_only=True)
        wait_front_ns = mean_numeric(df["DEVICE COMPUTE CB WAIT FRONT [ns]"], positive_only=True)
        reserve_back_ns = mean_numeric(df["DEVICE COMPUTE CB RESERVE BACK [ns]"], positive_only=True)
        pm_ideal_ns = mean_numeric(df["PM IDEAL [ns]"])
        pm_compute_ns = mean_numeric(df["PM COMPUTE [ns]"])
        pm_bandwidth_ns = mean_numeric(df["PM BANDWIDTH [ns]"])
        pm_fpu_util_pct = mean_numeric(df["PM FPU UTIL (%)"])
        noc_util_pct = mean_numeric(df["NOC UTIL (%)"])
        multicast_noc_util_pct = mean_numeric(df["MULTICAST NOC UTIL (%)"])
        dram_bw_util_pct = mean_numeric(df["DRAM BW UTIL (%)"])

        bubble_total_ns = (wait_front_ns or 0.0) + (reserve_back_ns or 0.0)
        wait_front_share_in_bubble = (wait_front_ns / bubble_total_ns) if bubble_total_ns > 0 else None
        reserve_back_share_in_bubble = (reserve_back_ns / bubble_total_ns) if bubble_total_ns > 0 else None

        summaries.append(
            {
                "case": case_meta["case"],
                "seq_len": int(workload["seq_len"]),
                "batch": int(workload["batch"]),
                "rows_profiled": int(len(df)),
                "kernel_us": (kernel_ns or 0.0) / 1000.0,
                "compute_us": compute_ns / 1000.0,
                "wait_front_us": (wait_front_ns or 0.0) / 1000.0,
                "reserve_back_us": (reserve_back_ns or 0.0) / 1000.0,
                "bubble_density_vs_kernel": (bubble_total_ns / kernel_ns) if kernel_ns and bubble_total_ns else None,
                "wait_front_share_in_bubble_pct": (
                    wait_front_share_in_bubble * 100.0 if wait_front_share_in_bubble is not None else None
                ),
                "reserve_back_share_in_bubble_pct": (
                    reserve_back_share_in_bubble * 100.0 if reserve_back_share_in_bubble is not None else None
                ),
                "reader_reserve_share_pct": stage_share(
                    entry, "custom_reader_breakdown", "SDPA-PAGED-RESERVE-SUM"
                )
                * 100.0,
                "writer_cb_wait_share_pct": stage_share(
                    entry, "custom_writer_breakdown", "SDPA-WRITER-CB-WAIT-SUM"
                )
                * 100.0,
                "pm_ideal_ns": pm_ideal_ns,
                "pm_compute_ns": pm_compute_ns,
                "pm_bandwidth_ns": pm_bandwidth_ns,
                "pm_fpu_util_pct": pm_fpu_util_pct,
                "noc_util_pct": noc_util_pct,
                "multicast_noc_util_pct": multicast_noc_util_pct,
                "dram_bw_util_pct": dram_bw_util_pct,
            }
        )

    return manifest, summaries


def write_summary_csv(case_summaries: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case_summaries).to_csv(SUMMARY_CSV, index=False)


def write_summary_md(manifest: dict[str, Any], case_summaries: list[dict[str, Any]]) -> None:
    decode_labels = [format_seq_len(summary["seq_len"]) for summary in case_summaries]
    bubble_ok_count = sum(
        1 for summary in case_summaries if summary["wait_front_us"] > 0 or summary["reserve_back_us"] > 0
    )
    pm_triplet_placeholder_count = sum(1 for summary in case_summaries if pm_triplet_state(summary) == "placeholder")
    pm_fpu_placeholder_count = sum(1 for summary in case_summaries if pm_fpu_state(summary) == "placeholder")
    noc_ok_count = sum(1 for summary in case_summaries if util_state(summary["noc_util_pct"]) == "ok")
    dram_ok_count = sum(1 for summary in case_summaries if util_state(summary["dram_bw_util_pct"]) == "ok")

    lines = [
        "# FlashMLA Decode PM/Bubble Summary",
        "",
        "## 产物位置",
        "",
        "- 汇总 CSV：`flash_mla_pm_bubble_decode_summary.csv`",
        "- 图表目录：`visuals/`",
        "- Dashboard：`flash_mla_pm_bubble_dashboard.html`",
        "",
        "## 执行状态",
        "",
        f"- preflight：`{manifest['preflight']['case']}`，状态 `{manifest['preflight']['status']}`，尝试次数 `{manifest['preflight']['attempts']}`。",
        f"- decode guarded rerun：`{len(case_summaries)}/8` 个 case 完成，覆盖 `{', '.join(decode_labels)}`。",
        f"- compute bubble 计数：`{bubble_ok_count}/8` 可用。",
        f"- `PM IDEAL/COMPUTE/BANDWIDTH` placeholder：`{pm_triplet_placeholder_count}/8`。",
        f"- `PM FPU UTIL` placeholder：`{pm_fpu_placeholder_count}/8`。",
        f"- `NOC UTIL` 可用：`{noc_ok_count}/8`；`DRAM BW UTIL` 可用：`{dram_ok_count}/8`。",
        "",
        "## Bubble 总表",
        "",
        "| case | seq_len | kernel us | compute us(max trisc) | wait-front counter us | reserve-back counter us | bubble density vs kernel | wait-front share in bubble | reserve-back share in bubble | reader reserve share | writer cb_wait share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in case_summaries:
        lines.append(
            "| {case} | {seq_len} | {kernel:.3f} | {compute:.3f} | {wait:.3f} | {reserve:.3f} | {density:.2f}x | {wait_share:.1f}% | {reserve_share:.1f}% | {reader_reserve:.1f}% | {writer_wait:.1f}% |".format(
                case=summary["case"],
                seq_len=summary["seq_len"],
                kernel=summary["kernel_us"],
                compute=summary["compute_us"],
                wait=summary["wait_front_us"],
                reserve=summary["reserve_back_us"],
                density=summary["bubble_density_vs_kernel"] or 0.0,
                wait_share=summary["wait_front_share_in_bubble_pct"] or 0.0,
                reserve_share=summary["reserve_back_share_in_bubble_pct"] or 0.0,
                reader_reserve=summary["reader_reserve_share_pct"],
                writer_wait=summary["writer_cb_wait_share_pct"],
            )
        )

    lines.extend(
        [
            "",
            "说明：`wait-front/reserve-back` 这里按 counter 密度来解读，而不是直接当作 wall-time share。",
            "因为短序列点上 `wait-front + reserve-back` 会超过单次 kernel wall time，说明它们是跨 compute 线程累计后的 stall counter。",
            "",
            "## PM / NOC / DRAM 原始捕获状态",
            "",
            "| case | PM IDEAL [ns] | PM COMPUTE [ns] | PM BANDWIDTH [ns] | PM FPU UTIL (%) | NOC UTIL (%) | MULTICAST NOC UTIL (%) | DRAM BW UTIL (%) | 判读 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for summary in case_summaries:
        lines.append(
            "| {case} | {ideal} | {compute} | {bandwidth} | {fpu} | {noc} | {mcast} | {dram} | {status} |".format(
                case=summary["case"],
                ideal=fmt_or_missing(summary["pm_ideal_ns"], "{:.1f}"),
                compute=fmt_or_missing(summary["pm_compute_ns"], "{:.1f}"),
                bandwidth=fmt_or_missing(summary["pm_bandwidth_ns"], "{:.1f}"),
                fpu=fmt_or_missing(summary["pm_fpu_util_pct"], "{:.3f}"),
                noc=fmt_or_missing(summary["noc_util_pct"], "{:.3f}"),
                mcast=fmt_or_missing(summary["multicast_noc_util_pct"], "{:.3f}"),
                dram=fmt_or_missing(summary["dram_bw_util_pct"], "{:.3f}"),
                status=(
                    "bubble-ok / PM-placeholder / NOC-DRAM-missing"
                    if pm_triplet_state(summary) == "placeholder"
                    and pm_fpu_state(summary) == "placeholder"
                    and util_state(summary["noc_util_pct"]) == "missing"
                    and util_state(summary["dram_bw_util_pct"]) == "missing"
                    else "mixed"
                ),
            )
        )

    first = case_summaries[0]
    last = case_summaries[-1]
    lines.extend(
        [
            "",
            "## 关键结论",
            "",
            f"- compute bubble 计数已经在 `decode_256 -> decode_32k` 的 `8/8` 个点上补齐；`wait-front share in bubble` 从 `{first['wait_front_share_in_bubble_pct']:.1f}%` 下降到 `{last['wait_front_share_in_bubble_pct']:.1f}%`，但始终保持主导。",
            f"- `reserve-back share in bubble` 会从 `{first['reserve_back_share_in_bubble_pct']:.1f}%` 抬升到 `{last['reserve_back_share_in_bubble_pct']:.1f}%`；这和 detailed profile 里 `reader reserve share` 从 `{first['reader_reserve_share_pct']:.1f}%` 抬升到 `{last['reader_reserve_share_pct']:.1f}%` 的趋势是一致的。",
            f"- writer 侧的 `cb_wait` 仍然几乎一路饱和，从 `{first['writer_cb_wait_share_pct']:.1f}%` 上升到 `{last['writer_cb_wait_share_pct']:.1f}%`，说明长序列 decode 仍然是强耦合流水线，而不是单纯的 PM/FPU 问题。",
            "- 这批 guarded rerun 目前应当被解读成 `bubble-complete / PM-incomplete`：`PM IDEAL/COMPUTE/BANDWIDTH` 固定在 `1.0 ns`，`PM FPU UTIL` 只有 `0.000 ~ 0.002`，`NOC/MULTICAST/DRAM BW UTIL` 仍为空，不宜当作真实利用率。",
        ]
    )

    write_text(SUMMARY_MD, "\n".join(lines) + "\n")


def write_visuals_and_dashboard(manifest: dict[str, Any], case_summaries: list[dict[str, Any]]) -> None:
    labels = [format_seq_len(summary["seq_len"]) for summary in case_summaries]

    bubble_density_svg = render_line_chart(
        "Decode Compute Bubble Density",
        "把 compute wait-front / reserve-back counter 归一化到 kernel wall time；这是 stall-density，不是 wall-time share。",
        labels,
        [
            {
                "label": "wait-front / kernel",
                "color": "#ff595e",
                "values": [summary["wait_front_us"] / summary["kernel_us"] for summary in case_summaries],
            },
            {
                "label": "reserve-back / kernel",
                "color": "#ffca3a",
                "values": [summary["reserve_back_us"] / summary["kernel_us"] for summary in case_summaries],
            },
            {
                "label": "total bubble / kernel",
                "color": "#5c7cfa",
                "values": [summary["bubble_density_vs_kernel"] for summary in case_summaries],
            },
        ],
        "counter / kernel",
        lambda value: f"{value:.1f}x",
        y_min=0.0,
    )

    bubble_composition_svg = render_stacked_bar_chart(
        "Decode Compute Bubble Composition",
        "wait-front 始终占主导，但 reserve-back 会随着序列增长持续抬升。",
        labels,
        [
            {
                "label": "wait-front share",
                "color": "#ff595e",
                "values": [summary["wait_front_share_in_bubble_pct"] / 100.0 for summary in case_summaries],
            },
            {
                "label": "reserve-back share",
                "color": "#ffca3a",
                "values": [summary["reserve_back_share_in_bubble_pct"] / 100.0 for summary in case_summaries],
            },
        ],
        "bubble 占比",
    )

    status_svg = render_status_grid(
        "Decode PM / Telemetry Capture Status",
        "这轮 rerun 真正补齐的是 compute bubble；PM triplet 仍是 placeholder，NOC/DRAM util 仍未落出来。",
        labels,
        ["bubble", "PM triplet", "PM FPU", "NOC", "MCAST", "DRAM"],
        [
            [
                "ok" if summary["wait_front_us"] > 0 or summary["reserve_back_us"] > 0 else "missing",
                pm_triplet_state(summary),
                pm_fpu_state(summary),
                util_state(summary["noc_util_pct"]),
                util_state(summary["multicast_noc_util_pct"]),
                util_state(summary["dram_bw_util_pct"]),
            ]
            for summary in case_summaries
        ],
    )

    visual_paths = {
        "bubble_density": VISUAL_DIR / "decode_compute_bubble_density.svg",
        "bubble_composition": VISUAL_DIR / "decode_compute_bubble_composition.svg",
        "capture_status": VISUAL_DIR / "decode_pm_capture_status.svg",
    }
    for path, content in {
        visual_paths["bubble_density"]: bubble_density_svg,
        visual_paths["bubble_composition"]: bubble_composition_svg,
        visual_paths["capture_status"]: status_svg,
    }.items():
        write_svg(path, content)

    completed_count = len(case_summaries)
    bubble_ok_count = sum(
        1 for summary in case_summaries if summary["wait_front_us"] > 0 or summary["reserve_back_us"] > 0
    )
    pm_placeholder_count = sum(1 for summary in case_summaries if pm_triplet_state(summary) == "placeholder")
    noc_ok_count = sum(1 for summary in case_summaries if util_state(summary["noc_util_pct"]) == "ok")
    last = case_summaries[-1]

    dashboard_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlashMLA Decode PM Bubble Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #121925;
      --panel-2: #0f141d;
      --text: #f5f7fb;
      --muted: #9fb0c3;
      --border: #223046;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #162033 0%, var(--bg) 45%);
      color: var(--text);
    }}
    .shell {{
      max-width: 1480px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 40px;
    }}
    .lead {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
      max-width: 1020px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px 18px 16px;
    }}
    .card .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .card .hint {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    figure {{
      margin: 0;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 16px;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 18px;
    }}
    .footnote {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <h1>FlashMLA Decode PM/Bubble Dashboard</h1>
    <p class="lead">这页只对应 guarded `decode_safe` rerun。重点不是重新解释 reader/writer stage，而是把这轮真正补齐的 compute bubble 计数和仍然缺失的 PM/NOC/DRAM util 状态单独拎出来看。</p>
    <section class="cards">
      <div class="card">
        <div class="label">Case Completion</div>
        <div class="value">{completed_count}/8</div>
        <div class="hint">preflight=`{manifest['preflight']['status']}`，decode guarded rerun 全部完成。</div>
      </div>
      <div class="card">
        <div class="label">Bubble Counters</div>
        <div class="value">{bubble_ok_count}/8</div>
        <div class="hint">`DEVICE COMPUTE CB WAIT FRONT/RESERVE BACK` 在所有 decode 点可用。</div>
      </div>
      <div class="card">
        <div class="label">PM Triplet</div>
        <div class="value">{pm_placeholder_count}/8 placeholder</div>
        <div class="hint">`PM IDEAL/COMPUTE/BANDWIDTH` 全部固定在 `1.0 ns`，不能当作真实 PM 利用率。</div>
      </div>
      <div class="card">
        <div class="label">NOC / DRAM Util</div>
        <div class="value">{noc_ok_count}/8 usable</div>
        <div class="hint">`NOC/MULTICAST/DRAM BW UTIL` 当前仍为空；这轮应解读成 bubble-only rerun。</div>
      </div>
      <div class="card">
        <div class="label">32k Bubble Mix</div>
        <div class="value">{last['wait_front_share_in_bubble_pct']:.1f}% / {last['reserve_back_share_in_bubble_pct']:.1f}%</div>
        <div class="hint">`decode_32k` 的 bubble 里，wait-front 仍然占主导，但 reserve-back 已经明显抬升。</div>
      </div>
    </section>
    <section class="grid">
      <figure><img src="visuals/decode_compute_bubble_density.svg" alt="Decode compute bubble density"></figure>
      <figure><img src="visuals/decode_compute_bubble_composition.svg" alt="Decode compute bubble composition"></figure>
      <figure><img src="visuals/decode_pm_capture_status.svg" alt="Decode PM capture status"></figure>
    </section>
    <p class="footnote">解释口径：compute bubble 这里按 counter 密度而不是 wall-time share 来读，因为短序列点上 counter 总和会超过单次 kernel wall time。Stage-level 的 reader reserve / writer cb_wait 仍然以 detailed JSON 里的分解表为准。</p>
  </div>
</body>
</html>
"""
    write_text(DASHBOARD_HTML, dashboard_html)


def main() -> None:
    manifest, case_summaries = build_case_summaries()
    write_summary_csv(case_summaries)
    write_summary_md(manifest, case_summaries)
    write_visuals_and_dashboard(manifest, case_summaries)
    print(
        json.dumps(
            {
                "summary_csv": str(SUMMARY_CSV),
                "summary_md": str(SUMMARY_MD),
                "dashboard_html": str(DASHBOARD_HTML),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

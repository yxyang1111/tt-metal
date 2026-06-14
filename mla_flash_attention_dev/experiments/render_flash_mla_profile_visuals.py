#!/usr/bin/env python3

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DETAILED_JSON = ROOT / "profile_outputs" / "flash_mla_wh_detailed" / "flash_mla_wh_detailed_profile_results.json"
SIMPLE_JSON = ROOT / "profile_outputs" / "flash_mla_wh" / "flash_mla_wh_profile_results.json"
OUTPUT_DIR = ROOT / "profile_outputs" / "flash_mla_wh_detailed"
VISUAL_DIR = OUTPUT_DIR / "visuals"
SUMMARY_MD = OUTPUT_DIR / "flash_mla_visual_summary.md"
DASHBOARD_HTML = OUTPUT_DIR / "flash_mla_visual_dashboard.html"
AUDIT_MD = OUTPUT_DIR / "flash_mla_profile_completeness_and_utilization.md"

SVG_WIDTH = 920
SVG_HEIGHT = 420
MARGIN_LEFT = 72
MARGIN_RIGHT = 28
MARGIN_TOP = 76
MARGIN_BOTTOM = 68

COLORS = {
    "compute": "#2ec4b6",
    "brisc": "#ff9f1c",
    "ncrisc": "#5c7cfa",
    "page_table": "#adb5bd",
    "reserve": "#ff595e",
    "issue": "#1982c4",
    "wait": "#ffca3a",
    "push": "#6a4c93",
    "cb_wait": "#ff595e",
    "barrier": "#ffca3a",
    "pop": "#2ec4b6",
    "prefill": "#ff595e",
    "decode": "#1982c4",
    "a1": "#6c757d",
    "a2": "#c1121f",
    "b": "#2a9d8f",
    "seq_1k": "#1982c4",
    "seq_4k": "#ff595e",
    "seq_32k": "#6a4c93",
}

PM_UTIL_COLUMNS = [
    "PM FPU UTIL (%)",
    "NOC UTIL (%)",
    "MULTICAST NOC UTIL (%)",
    "DRAM BW UTIL (%)",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def format_seq_len(seq_len: int) -> str:
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"{seq_len // 1024}k"
    return str(seq_len)


def get_us_per_item(entry: dict[str, Any]) -> float:
    kernel_us = entry["column_stats"]["DEVICE KERNEL DURATION [ns]"]["avg_ns"] / 1000.0
    workload = entry["workload"]
    if entry["mode"] == "prefill":
        normalized_items = workload["batch"] * workload["seq_len"]
    else:
        normalized_items = workload["batch"]
    return kernel_us / max(normalized_items, 1)


def get_share(entry: dict[str, Any], breakdown_key: str, marker: str) -> float:
    return entry.get(breakdown_key, {}).get("stage_shares", {}).get(marker, 0.0)


def get_case_map(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["case"]: entry for entry in entries}


def has_breakdown(entry: dict[str, Any], key: str) -> bool:
    return bool(entry.get(key, {}).get("stage_stats"))


def has_pm_util(entry: dict[str, Any]) -> bool:
    stats = entry.get("column_stats", {})
    return any(stats.get(column, {}).get("avg_ns") is not None for column in PM_UTIL_COLUMNS)


def has_compute_bubble_counters(entry: dict[str, Any]) -> bool:
    ratios = entry.get("analysis", {}).get("ratios", {})
    return ratios.get("compute_wait_front_share") is not None or ratios.get("compute_reserve_back_share") is not None


def present(flag: bool) -> str:
    return "ok" if flag else "missing"


def x_positions(labels: list[str], left: int, width: int) -> list[float]:
    if len(labels) == 1:
        return [left + width / 2.0]
    step = width / (len(labels) - 1)
    return [left + step * i for i in range(len(labels))]


def y_linear(value: float, y_min: float, y_max: float, top: int, height: int) -> float:
    if y_max <= y_min:
        return top + height
    ratio = (value - y_min) / (y_max - y_min)
    return top + height - ratio * height


def y_ticks(y_min: float, y_max: float, count: int) -> list[float]:
    if count <= 1:
        return [y_min, y_max]
    return [y_min + (y_max - y_min) * i / (count - 1) for i in range(count)]


def svg_escape(text: str) -> str:
    return html.escape(text, quote=True)


def polyline_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(commands)


def render_legend(items: list[tuple[str, str]], x: int, y: int) -> str:
    parts = []
    cursor_x = x
    for label, color in items:
        parts.append(
            f'<rect x="{cursor_x}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}"></rect>'
            f'<text x="{cursor_x + 18}" y="{y}" font-size="13" fill="#dfe7f1">{svg_escape(label)}</text>'
        )
        cursor_x += 18 + len(label) * 8 + 26
    return "".join(parts)


def render_line_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[dict[str, Any]],
    y_label: str,
    value_formatter,
    *,
    y_min: float = 0.0,
    y_max: float | None = None,
    tick_count: int = 5,
) -> str:
    chart_w = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_h = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    x_vals = x_positions(labels, MARGIN_LEFT, chart_w)

    all_values = [
        value
        for item in series
        for value in item["values"]
        if value is not None
    ]
    if y_max is None:
        ymax = max(all_values) if all_values else 1.0
        y_max = ymax * 1.08 if ymax > 0 else 1.0

    grid_parts = []
    for tick in y_ticks(y_min, y_max, tick_count):
        y = y_linear(tick, y_min, y_max, MARGIN_TOP, chart_h)
        grid_parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{y:.2f}" '
            f'stroke="#273142" stroke-width="1"></line>'
        )
        grid_parts.append(
            f'<text x="{MARGIN_LEFT - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#9fb0c3">'
            f'{svg_escape(value_formatter(tick))}</text>'
        )

    axis_parts = [
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP + chart_h}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<text x="20" y="{MARGIN_TOP + chart_h / 2:.2f}" transform="rotate(-90 20 {MARGIN_TOP + chart_h / 2:.2f})" font-size="13" fill="#c7d2df">{svg_escape(y_label)}</text>',
    ]
    for label, x in zip(labels, x_vals):
        axis_parts.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - 26}" text-anchor="middle" font-size="12" fill="#c7d2df">{svg_escape(label)}</text>'
        )

    series_parts = []
    for item in series:
        point_parts = []
        active_points = []
        dash_attr = ' stroke-dasharray="8 6"' if item.get("dashed") else ""
        for x, value in zip(x_vals, item["values"]):
            if value is None:
                if active_points:
                    series_parts.append(
                        f'<path d="{polyline_path(active_points)}" fill="none" stroke="{item["color"]}" stroke-width="3" '
                        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}></path>'
                    )
                    active_points = []
                continue
            y = y_linear(float(value), y_min, y_max, MARGIN_TOP, chart_h)
            active_points.append((x, y))
            point_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{item["color"]}" stroke="#10151d" stroke-width="1.5"></circle>'
            )
        if active_points:
            series_parts.append(
                f'<path d="{polyline_path(active_points)}" fill="none" stroke="{item["color"]}" stroke-width="3" '
                f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}></path>'
            )
        series_parts.extend(point_parts)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <rect width="100%" height="100%" rx="22" fill="#111722"></rect>
  <text x="{MARGIN_LEFT}" y="34" font-size="24" font-weight="700" fill="#f5f7fb">{svg_escape(title)}</text>
  <text x="{MARGIN_LEFT}" y="56" font-size="13" fill="#9fb0c3">{svg_escape(subtitle)}</text>
  {render_legend([(item["label"], item["color"]) for item in series], MARGIN_LEFT, 76)}
  {''.join(grid_parts)}
  {''.join(axis_parts)}
  {''.join(series_parts)}
</svg>
"""


def render_stacked_bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[dict[str, Any]],
    y_label: str,
    *,
    y_min: float = 0.0,
    y_max: float | None = 1.0,
    tick_count: int = 5,
    value_formatter=None,
) -> str:
    chart_w = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_h = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    bar_group_width = chart_w / max(len(labels), 1)
    bar_width = min(66.0, bar_group_width * 0.64)
    x_vals = [MARGIN_LEFT + bar_group_width * i + bar_group_width / 2.0 for i in range(len(labels))]

    stacked_totals = []
    for idx in range(len(labels)):
        total = 0.0
        for item in series:
            value = item["values"][idx]
            if value is None:
                continue
            total += max(float(value), 0.0)
        stacked_totals.append(total)

    if y_max is None:
        ymax = max(stacked_totals) if stacked_totals else 1.0
        y_max = ymax * 1.08 if ymax > 0 else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    if value_formatter is None:
        if y_min == 0.0 and y_max == 1.0:
            value_formatter = lambda tick: f"{tick * 100:.0f}%"
        else:
            value_formatter = lambda tick: f"{tick:.1f}"

    grid_parts = []
    for tick in y_ticks(y_min, y_max, tick_count):
        y = y_linear(tick, y_min, y_max, MARGIN_TOP, chart_h)
        grid_parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{y:.2f}" stroke="#273142" stroke-width="1"></line>'
        )
        grid_parts.append(
            f'<text x="{MARGIN_LEFT - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#9fb0c3">{svg_escape(value_formatter(tick))}</text>'
        )

    axis_parts = [
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP + chart_h}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<text x="20" y="{MARGIN_TOP + chart_h / 2:.2f}" transform="rotate(-90 20 {MARGIN_TOP + chart_h / 2:.2f})" font-size="13" fill="#c7d2df">{svg_escape(y_label)}</text>',
    ]
    for label, x in zip(labels, x_vals):
        axis_parts.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - 26}" text-anchor="middle" font-size="12" fill="#c7d2df">{svg_escape(label)}</text>'
        )

    bar_parts = []
    for idx, x in enumerate(x_vals):
        running = y_min
        for item in series:
            raw_value = item["values"][idx]
            if raw_value is None:
                continue
            value = float(raw_value)
            if value <= 0:
                continue
            y1 = y_linear(running + value, y_min, y_max, MARGIN_TOP, chart_h)
            y0 = y_linear(running, y_min, y_max, MARGIN_TOP, chart_h)
            bar_parts.append(
                f'<rect x="{x - bar_width / 2:.2f}" y="{y1:.2f}" width="{bar_width:.2f}" height="{(y0 - y1):.2f}" '
                f'rx="8" fill="{item["color"]}"></rect>'
            )
            running += value

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <rect width="100%" height="100%" rx="22" fill="#111722"></rect>
  <text x="{MARGIN_LEFT}" y="34" font-size="24" font-weight="700" fill="#f5f7fb">{svg_escape(title)}</text>
  <text x="{MARGIN_LEFT}" y="56" font-size="13" fill="#9fb0c3">{svg_escape(subtitle)}</text>
  {render_legend([(item["label"], item["color"]) for item in series], MARGIN_LEFT, 76)}
  {''.join(grid_parts)}
  {''.join(axis_parts)}
  {''.join(bar_parts)}
</svg>
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_svg(path: Path, content: str) -> None:
    write_text(path, content)


def build_visuals() -> dict[str, Path]:
    detailed = load_json(DETAILED_JSON)
    simple = load_json(SIMPLE_JSON)

    profiles = detailed["profiles"]
    decode_entries = sorted((entry for entry in profiles if entry["mode"] == "decode"), key=lambda item: item["workload"]["seq_len"])
    prefill_entries = sorted((entry for entry in profiles if entry["mode"] == "prefill"), key=lambda item: item["workload"]["seq_len"])
    decode_labels = [format_seq_len(entry["workload"]["seq_len"]) for entry in decode_entries]
    prefill_decode_labels = [format_seq_len(seq) for seq in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]]

    decode_thread_share_svg = render_line_chart(
        "Decode 阶段迁移",
        "观察 compute、BRISC、NCRISC 在 kernel window 里的占比变化。",
        decode_labels,
        [
            {
                "label": "compute 占比",
                "color": COLORS["compute"],
                "values": [entry["analysis"]["ratios"].get("compute_share", 0.0) for entry in decode_entries],
            },
            {
                "label": "BRISC 占比",
                "color": COLORS["brisc"],
                "values": [entry["analysis"]["ratios"].get("brisc_share", 0.0) for entry in decode_entries],
            },
            {
                "label": "NCRISC 占比",
                "color": COLORS["ncrisc"],
                "values": [entry["analysis"]["ratios"].get("ncrisc_share", 0.0) for entry in decode_entries],
            },
        ],
        "kernel 占比",
        lambda value: f"{value * 100:.0f}%",
        y_max=1.0,
    )

    decode_reader_breakdown_svg = render_stacked_bar_chart(
        "Decode Reader 分解",
        "随着序列增长，reserve 会逐步超过 issue。",
        decode_labels,
        [
            {"label": "page_table", "color": COLORS["page_table"], "values": [get_share(entry, "custom_reader_breakdown", "SDPA-PAGE-TABLE-SUM") for entry in decode_entries]},
            {"label": "issue", "color": COLORS["issue"], "values": [get_share(entry, "custom_reader_breakdown", "SDPA-PAGED-ISSUE-SUM") for entry in decode_entries]},
            {"label": "wait", "color": COLORS["wait"], "values": [get_share(entry, "custom_reader_breakdown", "SDPA-PAGED-WAIT-SUM") for entry in decode_entries]},
            {"label": "reserve", "color": COLORS["reserve"], "values": [get_share(entry, "custom_reader_breakdown", "SDPA-PAGED-RESERVE-SUM") for entry in decode_entries]},
            {"label": "push", "color": COLORS["push"], "values": [get_share(entry, "custom_reader_breakdown", "SDPA-PAGED-PUSH-SUM") for entry in decode_entries]},
        ],
        "阶段占比",
    )

    decode_writer_breakdown_svg = render_stacked_bar_chart(
        "Decode Writer 分解",
        "整条 decode sweep 上，writer 基本一直被 cb_wait_front 主导。",
        decode_labels,
        [
            {"label": "cb_wait", "color": COLORS["cb_wait"], "values": [get_share(entry, "custom_writer_breakdown", "SDPA-WRITER-CB-WAIT-SUM") for entry in decode_entries]},
            {"label": "issue", "color": COLORS["issue"], "values": [get_share(entry, "custom_writer_breakdown", "SDPA-WRITER-ISSUE-SUM") for entry in decode_entries]},
            {"label": "barrier", "color": COLORS["barrier"], "values": [get_share(entry, "custom_writer_breakdown", "SDPA-WRITER-BARRIER-SUM") for entry in decode_entries]},
            {"label": "pop", "color": COLORS["pop"], "values": [get_share(entry, "custom_writer_breakdown", "SDPA-WRITER-POP-SUM") for entry in decode_entries]},
        ],
        "阶段占比",
    )

    prefill_map = {entry["workload"]["seq_len"]: get_us_per_item(entry) for entry in prefill_entries}
    decode_map = {entry["workload"]["seq_len"]: get_us_per_item(entry) for entry in decode_entries}
    normalized_latency_svg = render_line_chart(
        "归一化时延",
        "把 prefill 和 decode 放到同一条 sequence 轴上，用 us/item 对比。",
        prefill_decode_labels,
        [
            {
                "label": "prefill us/item",
                "color": COLORS["prefill"],
                "values": [prefill_map.get(seq) for seq in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]],
            },
            {
                "label": "decode us/item",
                "color": COLORS["decode"],
                "values": [decode_map.get(seq) for seq in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]],
            },
        ],
        "us / item",
        lambda value: f"{value:.1f}",
    )

    bh_empirical_cases = get_case_map(detailed["bh_empirical_decode"]["cases"])
    bh_a1_cases = get_case_map(simple["bh_simulation"]["current_a_bh"]["cases"])
    bh_b_cases = get_case_map(simple["bh_simulation"]["native_b_bh"]["cases"])
    canonical_cases = ["decode_1k", "decode_4k", "decode_32k"]
    bh_labels = [case.replace("decode_", "") for case in canonical_cases]
    bh_latency_svg = render_line_chart(
        "BH 时延对比",
        "把 A-BH 一阶、A-BH 二阶、B-BH 放到一张图里。",
        bh_labels,
        [
            {
                "label": "A-BH 一阶",
                "color": COLORS["a1"],
                "values": [bh_a1_cases[case]["predicted_core_ms"] for case in canonical_cases],
                "dashed": True,
            },
            {
                "label": "A-BH 二阶",
                "color": COLORS["a2"],
                "values": [bh_empirical_cases[case]["predicted_kernel_ms"] for case in canonical_cases],
            },
            {
                "label": "B-BH",
                "color": COLORS["b"],
                "values": [bh_b_cases[case]["predicted_core_ms"] for case in canonical_cases],
            },
        ],
        "时延 (ms)",
        lambda value: f"{value:.3f}",
    )

    empirical_sweeps = {
        entry["case"]: {item["active_cores"]: item["predicted_kernel_ms"] for item in entry["sweep"]}
        for entry in detailed["bh_empirical_decode"]["active_core_sweep"]
    }
    active_core_labels = [str(core) for core in [16, 20, 24, 32, 48, 64]]
    bh_sweep_svg = render_line_chart(
        "A-BH 二阶 Active-Core Sweep",
        "一旦 floor 形成，继续加 active cores 几乎不再改变曲线。",
        active_core_labels,
        [
            {
                "label": "decode_1k",
                "color": COLORS["seq_1k"],
                "values": [empirical_sweeps["decode_1k"][core] for core in [16, 20, 24, 32, 48, 64]],
            },
            {
                "label": "decode_4k",
                "color": COLORS["seq_4k"],
                "values": [empirical_sweeps["decode_4k"][core] for core in [16, 20, 24, 32, 48, 64]],
            },
            {
                "label": "decode_32k",
                "color": COLORS["seq_32k"],
                "values": [empirical_sweeps["decode_32k"][core] for core in [16, 20, 24, 32, 48, 64]],
            },
        ],
        "时延 (ms)",
        lambda value: f"{value:.3f}",
    )

    visual_paths = {
        "decode_thread_shares": VISUAL_DIR / "decode_thread_shares.svg",
        "decode_reader_breakdown": VISUAL_DIR / "decode_reader_breakdown.svg",
        "decode_writer_breakdown": VISUAL_DIR / "decode_writer_breakdown.svg",
        "normalized_latency": VISUAL_DIR / "normalized_latency.svg",
        "bh_latency_comparison": VISUAL_DIR / "bh_latency_comparison.svg",
        "bh_active_core_sweep": VISUAL_DIR / "bh_active_core_sweep.svg",
    }
    write_svg(visual_paths["decode_thread_shares"], decode_thread_share_svg)
    write_svg(visual_paths["decode_reader_breakdown"], decode_reader_breakdown_svg)
    write_svg(visual_paths["decode_writer_breakdown"], decode_writer_breakdown_svg)
    write_svg(visual_paths["normalized_latency"], normalized_latency_svg)
    write_svg(visual_paths["bh_latency_comparison"], bh_latency_svg)
    write_svg(visual_paths["bh_active_core_sweep"], bh_sweep_svg)

    decode_1k = bh_empirical_cases["decode_1k"]
    decode_4k = bh_empirical_cases["decode_4k"]
    decode_32k = bh_empirical_cases["decode_32k"]
    b_32k = bh_b_cases["decode_32k"]

    summary_md = f"""# FlashMLA Profile 可视化摘要

## 产物位置

- Dashboard：`flash_mla_visual_dashboard.html`
- 图表目录：`visuals/`

## 核心结论

- Decode 的 critical-path 迁移非常清楚：`256~1k` 仍然偏 compute-critical，`2k~8k` 转成 writer-close，`16k` 开始 reader-close，`32k` 则 reader 和 writer 一起饱和。
- Decode reader 的主导项从 `issue` 迁移到 `reserve`：`reserve share` 在 `1k` 是 `{get_share(get_case_map(decode_entries)['decode_1k'], 'custom_reader_breakdown', 'SDPA-PAGED-RESERVE-SUM') * 100:.1f}%`，到 `4k` 变成 `{get_share(get_case_map(decode_entries)['decode_4k'], 'custom_reader_breakdown', 'SDPA-PAGED-RESERVE-SUM') * 100:.1f}%`，到 `32k` 变成 `{get_share(get_case_map(decode_entries)['decode_32k'], 'custom_reader_breakdown', 'SDPA-PAGED-RESERVE-SUM') * 100:.1f}%`。
- Decode writer 几乎始终被 `cb_wait_front` 主导：`cb_wait share` 从 `1k` 的 `{get_share(get_case_map(decode_entries)['decode_1k'], 'custom_writer_breakdown', 'SDPA-WRITER-CB-WAIT-SUM') * 100:.1f}%` 上升到 `32k` 的 `{get_share(get_case_map(decode_entries)['decode_32k'], 'custom_writer_breakdown', 'SDPA-WRITER-CB-WAIT-SUM') * 100:.1f}%`。
- `A-BH` 的二阶经验值已经明显 floor-dominated：`decode_1k={decode_1k['predicted_kernel_ms']:.4f} ms`，`decode_4k={decode_4k['predicted_kernel_ms']:.4f} ms`，`decode_32k={decode_32k['predicted_kernel_ms']:.4f} ms`。
- 在 `32k` 点上，`B-BH={b_32k['predicted_core_ms']:.4f} ms`，相对经验二阶 `A-BH` 大约快 `{decode_32k['predicted_kernel_ms'] / b_32k['predicted_core_ms']:.1f}x`。

## 图表

### 1. Decode 阶段迁移

![](visuals/decode_thread_shares.svg)

这张图最适合回答“什么时候不再只是 compute 在拖”：随着序列增长，`BRISC` 和 `NCRISC` 会逐渐把原来的 slack 吃掉。

### 2. Decode Reader 分解

![](visuals/decode_reader_breakdown.svg)

这张图把 reader 的关键迁移直接画出来了：长序列 decode 不再主要由 `issue` 主导，而是越来越被 `reserve` 主导，也就是 backpressure 信号。

### 3. Decode Writer 分解

![](visuals/decode_writer_breakdown.svg)

Writer 不是主要被 write issue 带宽限制，而是主要停在 `cb_wait`，所以更准确的表述是“强耦合流水线”，而不是“单纯写带宽瓶颈”。

### 4. Prefill vs Decode 归一化时延

![](visuals/normalized_latency.svg)

这张图把两条路径放到了同一条 `us/item` 轴上。Prefill 的 per-item 曲线明显更重，而 decode 在更长序列前都相对平缓。

### 5. BH 模型对比

![](visuals/bh_latency_comparison.svg)

这张图把最关键的三条 BH 线放到一起：`A-BH 一阶`、`A-BH 二阶`、`B-BH`。

### 6. A-BH 二阶 Active-Core Sweep

![](visuals/bh_active_core_sweep.svg)

这里最重要的观察是：一旦 floor 形成，继续增加 active cores 几乎不会继续缩短曲线。

## 备注

- 当前可视化基于现有的 WH detailed JSON 和 simple WH/BH simulation JSON。
- 新接入的 source-level marker，例如 `K/V reserve` 和 `sender/root/tree/output` wait，要在下一次重跑 detailed sweep 后才会真正出现在图里。
"""
    write_text(SUMMARY_MD, summary_md)

    total_cases = len(profiles)
    decode_case_count = len(decode_entries)
    base_profile_count = sum(
        1
        for entry in profiles
        if entry.get("column_stats", {}).get("DEVICE KERNEL DURATION [ns]", {}).get("avg_ns") is not None
    )
    reader_breakdown_count = sum(1 for entry in profiles if has_breakdown(entry, "custom_reader_breakdown"))
    writer_breakdown_count = sum(1 for entry in profiles if has_breakdown(entry, "custom_writer_breakdown"))
    reader_source_count = sum(1 for entry in decode_entries if has_breakdown(entry, "custom_reader_source_breakdown"))
    writer_source_count = sum(1 for entry in decode_entries if has_breakdown(entry, "custom_writer_source_breakdown"))
    pm_util_count = sum(1 for entry in profiles if has_pm_util(entry))
    compute_bubble_count = sum(1 for entry in profiles if has_compute_bubble_counters(entry))

    audit_lines = [
        "# FlashMLA Profile 完整性与利用率说明",
        "",
        "## 最后建议阅读顺序",
        "",
        "- 全量结果表：`flash_mla_wh_detailed_profile_report.md`",
        "- 完整性与利用率说明：`flash_mla_profile_completeness_and_utilization.md`",
        "- 读图版摘要：`flash_mla_visual_summary.md`",
        "- 可视化总入口：`flash_mla_visual_dashboard.html`",
        "",
        "## 这轮结果完整到哪一层",
        "",
        f"- `base measured profile`：`{base_profile_count}/{total_cases}`，包含 `kernel / BRISC / NCRISC / TRISC* / classification`。",
        f"- `reader stage breakdown`：`{reader_breakdown_count}/{total_cases}`，覆盖 `page_table / reserve / issue / wait / push`。",
        f"- `writer stage breakdown`：`{writer_breakdown_count}/{total_cases}`，覆盖 `cb_wait / issue / barrier / pop`。",
        f"- `decode reader source breakdown`：`{reader_source_count}/{decode_case_count}`，当前结果里还没有真实 `K/V` source 数据。",
        f"- `decode writer source breakdown`：`{writer_source_count}/{decode_case_count}`，当前结果里还没有真实 `sender/root/tree/output` source 数据。",
        f"- `PM 利用率计数`：`{pm_util_count}/{total_cases}`，当前结果里 `PM FPU/NOC/DRAM` 都还是空。",
        f"- `compute 空泡计数`：`{compute_bubble_count}/{total_cases}`，当前结果里 `DEVICE COMPUTE CB WAIT FRONT/RESERVE BACK` 都还是空。",
        "",
        "结论：这轮数据已经 **完整到 stage-level**，但还没有完整到 **PM utilization / compute-side bubble / source-level attribution** 这三层。",
        "",
        "## Case 完整性矩阵",
        "",
        "| case | base profile | reader stages | writer stages | reader source | writer source | PM util | compute bubble |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in profiles:
        is_decode = entry["mode"] == "decode"
        audit_lines.append(
            f"| {entry['case']} | "
            f"{present(entry.get('column_stats', {}).get('DEVICE KERNEL DURATION [ns]', {}).get('avg_ns') is not None)} | "
            f"{present(has_breakdown(entry, 'custom_reader_breakdown'))} | "
            f"{present(has_breakdown(entry, 'custom_writer_breakdown'))} | "
            f"{present(has_breakdown(entry, 'custom_reader_source_breakdown')) if is_decode else 'n/a'} | "
            f"{present(has_breakdown(entry, 'custom_writer_source_breakdown')) if is_decode else 'n/a'} | "
            f"{present(has_pm_util(entry))} | "
            f"{present(has_compute_bubble_counters(entry))} |"
        )

    audit_lines.extend(
        [
            "",
            "## 能不能用 profile 看利用率和空泡？",
            "",
            "可以，但要分成两层来看。",
            "",
            "### 现在已经能直接看的",
            "",
            "- `线程窗口占比`：`BRISC/NCRISC/TRISC` 相对 `kernel window` 的占比，能看谁在逼近 critical path，谁还有 slack。",
            "- `reader 空泡/反压代理量`：`reserve share`。它表示 reader 在等下游 CB 空间，本质上就是 backpressure stall。",
            "- `writer 空泡/等待代理量`：`cb_wait share`。它表示 writer 在等 compute/reduction 把结果推到输出 CB。",
            "- `hot-path coverage`：详细报告里的 `coverage vs thread` 能告诉你 marker 已经解释了多少线程时间；剩余部分是未跟踪控制路径，不能直接等同于空泡。",
            "",
            "### 当前还不能直接看的",
            "",
            "- `精确硬件利用率百分比`：当前结果里的 `PM FPU UTIL (%)`、`NOC UTIL (%)`、`MULTICAST NOC UTIL (%)`、`DRAM BW UTIL (%)` 都没有值，所以现在不能直接回答“FPU 利用率是 63% 还是 81%”这种问题。",
            "- `compute-side 空泡`：当前结果里的 `DEVICE COMPUTE CB WAIT FRONT` 和 `DEVICE COMPUTE CB RESERVE BACK` 也没有值，所以还不能直接量化 compute 自己在等输入还是等输出。",
            "- `source-level stall attribution`：虽然内核里已经接好了 `K/V reserve` 和 `sender/root/tree/output wait` 的 marker，但这批 JSON 还没有下一次重跑后的真实值。",
            "",
            "### 这轮最可靠的空泡结论",
            "",
            f"- `decode_32k`：reader 的 `reserve share={get_share(get_case_map(decode_entries)['decode_32k'], 'custom_reader_breakdown', 'SDPA-PAGED-RESERVE-SUM') * 100:.1f}%`，writer 的 `cb_wait share={get_share(get_case_map(decode_entries)['decode_32k'], 'custom_writer_breakdown', 'SDPA-WRITER-CB-WAIT-SUM') * 100:.1f}%`。这说明长序列 decode 的主要空泡不是纯 compute idle，而是 **reader downstream backpressure + writer output-availability wait**。",
            f"- `prefill_4k`：reader 的 `wait share={get_share(get_case_map(prefill_entries)['prefill_4k'], 'custom_reader_breakdown', 'SDPA-PAGED-WAIT-SUM') * 100:.1f}%`，writer 的 `cb_wait share={get_share(get_case_map(prefill_entries)['prefill_4k'], 'custom_writer_breakdown', 'SDPA-WRITER-CB-WAIT-SUM') * 100:.1f}%`。这说明 prefill 从一开始就更像强耦合饱和流水线，而不是短 decode 那种 compute-dominant 形态。",
            "",
            "## 如果你下一步要真正看“利用率百分比”和“compute 空泡”",
            "",
            "- 继续保留现有 detailed stage-level profile，因为这层已经完整。",
            "- 重新跑一次能稳定产出 `PM FPU/NOC/DRAM` 的 profile，确认这些列不再是空值。",
            "- 用下一次 detailed rerun 把新的 `source-level marker` 数据真正灌进 JSON，这样就能把 `reader reserve` 和 `writer cb_wait` 拆到更直接的来源。",
        ]
    )
    write_text(AUDIT_MD, "\n".join(audit_lines) + "\n")

    dashboard_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlashMLA Profile Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #121925;
      --panel-2: #0f141d;
      --text: #f5f7fb;
      --muted: #9fb0c3;
      --border: #223046;
      --accent: #5c7cfa;
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
      max-width: 980px;
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
    <h1>FlashMLA Profile Dashboard</h1>
    <p class="lead">这份 dashboard 把当前 WH detailed profile 和 BH 模型放到一页里，重点突出 decode 阶段迁移、reader/writer 内部分解，以及 A-BH 一阶/二阶/B 的差异。</p>
    <section class="cards">
      <div class="card">
        <div class="label">Decode 迁移</div>
        <div class="value">compute -&gt; writer -&gt; reader</div>
        <div class="hint">`256~1k` compute-critical，`2k~8k` writer-close，`16k` reader-close，`32k` reader/writer saturated。</div>
      </div>
      <div class="card">
        <div class="label">Reader Reserve</div>
        <div class="value">{get_share(get_case_map(decode_entries)['decode_32k'], 'custom_reader_breakdown', 'SDPA-PAGED-RESERVE-SUM') * 100:.1f}%</div>
        <div class="hint">`decode_32k` 的 reader 主导项已经是 reserve，而不是 issue。</div>
      </div>
      <div class="card">
        <div class="label">Writer cb_wait</div>
        <div class="value">{get_share(get_case_map(decode_entries)['decode_32k'], 'custom_writer_breakdown', 'SDPA-WRITER-CB-WAIT-SUM') * 100:.1f}%</div>
        <div class="hint">长序列 decode 的 writer 主要是在等结果 ready，而不是等写带宽。</div>
      </div>
      <div class="card">
        <div class="label">BH 32k 差距</div>
        <div class="value">{decode_32k['predicted_kernel_ms'] / b_32k['predicted_core_ms']:.1f}x</div>
        <div class="hint">`A-BH empirical` 相对 `B-BH ideal` 的结构差距。</div>
      </div>
    </section>
    <section class="grid">
      <figure><img src="visuals/decode_thread_shares.svg" alt="Decode stage migration"></figure>
      <figure><img src="visuals/decode_reader_breakdown.svg" alt="Decode reader breakdown"></figure>
      <figure><img src="visuals/decode_writer_breakdown.svg" alt="Decode writer breakdown"></figure>
      <figure><img src="visuals/normalized_latency.svg" alt="Prefill vs decode normalized latency"></figure>
      <figure><img src="visuals/bh_latency_comparison.svg" alt="BH latency comparison"></figure>
      <figure><img src="visuals/bh_active_core_sweep.svg" alt="BH active core sweep"></figure>
    </section>
    <p class="footnote">说明：当前 dashboard 使用现有 detailed JSON 和 simple BH simulation JSON。新接入的 source-level marker 要在下次重跑 detailed sweep 后才会真正出现在结果里。</p>
  </div>
</body>
</html>
"""
    write_text(DASHBOARD_HTML, dashboard_html)

    return {
        "summary_md": SUMMARY_MD,
        "audit_md": AUDIT_MD,
        "dashboard_html": DASHBOARD_HTML,
        **visual_paths,
    }


def main() -> None:
    paths = build_visuals()
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

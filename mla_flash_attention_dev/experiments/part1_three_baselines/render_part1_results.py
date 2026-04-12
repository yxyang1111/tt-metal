#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

from mla_flash_attention_dev.experiments.part1_three_baselines.experiment_config import (
    DEFAULT_DECODE_SEQ_LENS,
    DEFAULT_PREFILL_SEQ_LENS,
    ExperimentConfig,
    format_seq_len,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("PART1_OUTPUT_DIR", str(ROOT / "outputs"))).expanduser()
RAW_JSON = OUTPUT_DIR / "raw" / "part1_four_method_results.json"
FILTERED_JSON = OUTPUT_DIR / "raw" / "part1_four_method_results_filtered.json"
FILTERED_CSV = OUTPUT_DIR / "raw" / "part1_four_method_results_filtered.csv"
CAPABILITY_PROBE_JSON = OUTPUT_DIR / "raw" / "capability_probe_results.json"
CAPABILITY_PROBE_CHECKPOINT_JSON = OUTPUT_DIR / "raw" / "capability_probe_checkpoint.json"
SUPPORTED_PROBE_JSON = OUTPUT_DIR / "raw" / "capability_probe_supported_four_way_results.json"
SUPPORTED_PROBE_CSV = OUTPUT_DIR / "raw" / "capability_probe_supported_four_way_results.csv"
DETAIL_JSON = OUTPUT_DIR / "flashmla_detailed" / "flash_mla_wh_detailed_profile_results.json"
VISUAL_DIR = OUTPUT_DIR / "visuals"
TABLE_DIR = OUTPUT_DIR / "tables"
MULTIDIM_DIR = OUTPUT_DIR / "multidim"
MULTIDIM_TABLE_DIR = MULTIDIM_DIR / "tables"
MULTIDIM_VISUAL_DIR = MULTIDIM_DIR / "visuals"
REPORT_MD = OUTPUT_DIR / "report.md"
SUPPORTED_PROBE_REPORT_MD = OUTPUT_DIR / "supported_four_way_probe_report.md"
DASHBOARD_HTML = OUTPUT_DIR / "dashboard.html"
DECODE_TABLE_MD = TABLE_DIR / "decode_four_methods.md"
DECODE_SUMMARY_MD = TABLE_DIR / "decode_best_worst_ties.md"
PREFILL_TABLE_MD = TABLE_DIR / "prefill_control.md"
FLASHMLA_SUBANALYSIS_MD = TABLE_DIR / "flashmla_subanalysis.md"
PARAMETER_TABLE_MD = TABLE_DIR / "experiment_parameters.md"
CAPABILITY_PROBE_MD = TABLE_DIR / "capability_probe.md"
CAPABILITY_PROBE_SUMMARY_MD = TABLE_DIR / "capability_probe_summary.md"
FAIR_ALL_CASES_MD = TABLE_DIR / "fair_four_method_all_cases.md"
FAIR_HEAD_SLICE_SUMMARY_MD = TABLE_DIR / "fair_four_method_head_slice_summary.md"
FAIR_SEQ_SUMMARY_MD = TABLE_DIR / "fair_four_method_seq_summary.md"
FAIR_BATCH_SUMMARY_MD = TABLE_DIR / "fair_four_method_batch_summary.md"
FAIR_DIM_SUMMARY_MD = TABLE_DIR / "fair_four_method_dim_summary.md"
SUPPORTED_PROBE_ALL_CASES_MD = TABLE_DIR / "supported_four_way_probe_all_cases.md"
SUPPORTED_PROBE_SEQ_SUMMARY_MD = TABLE_DIR / "supported_four_way_probe_seq_summary.md"
SUPPORTED_PROBE_BATCH_SUMMARY_MD = TABLE_DIR / "supported_four_way_probe_batch_summary.md"
SUPPORTED_PROBE_HEAD_SUMMARY_MD = TABLE_DIR / "supported_four_way_probe_head_summary.md"
SUPPORTED_PROBE_DIM_SUMMARY_MD = TABLE_DIR / "supported_four_way_probe_dim_summary.md"
MULTIDIM_BATCH_TABLE_MD = MULTIDIM_TABLE_DIR / "decode_batch_sweep.md"
MULTIDIM_HEAD_TABLE_MD = MULTIDIM_TABLE_DIR / "decode_head_sweep.md"
MULTIDIM_DIM_TABLE_MD = MULTIDIM_TABLE_DIR / "decode_dim_sweep.md"
MULTIDIM_DEEPSEEK_QHPC_TABLE_MD = MULTIDIM_TABLE_DIR / "decode_deepseek_q_heads_per_core_sweep.md"

SVG_WIDTH = 980
SVG_HEIGHT = 440
MARGIN_LEFT = 76
MARGIN_RIGHT = 36
MARGIN_TOP = 82
MARGIN_BOTTOM = 72

COLORS = {
    "reference_attention": "#6c757d",
    "flash_attention": "#1982c4",
    "flash_mla": "#ff595e",
    "deepseek_flash_mla": "#8ac926",
    "probe_ok": "#2ec4b6",
    "probe_blocked": "#ff9f1c",
    "compute": "#2ec4b6",
    "brisc": "#ff9f1c",
    "ncrisc": "#5c7cfa",
}

LABELS = {
    "reference_attention": "Reference Attention",
    "flash_attention": "Flash Attention",
    "flash_mla": "FlashMLA (TT Mainline)",
    "deepseek_flash_mla": "DeepSeek FlashMLA",
}
DECODE_BASELINES = (
    "reference_attention",
    "flash_attention",
    "flash_mla",
    "deepseek_flash_mla",
)
TIE_THRESHOLD_RATIO = 1.03
OUTLIER_MODIFIED_Z_THRESHOLD = 5.0
OUTLIER_MEDIAN_RATIO_THRESHOLD = 1.10
MIN_FILTERED_SAMPLES = 5


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            serialized: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    serialized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def default_config_from_payload(payload: dict[str, Any]) -> ExperimentConfig | None:
    metadata = payload.get("metadata", {})
    default_config = metadata.get("default_config")
    if isinstance(default_config, dict):
        return ExperimentConfig.from_dict(default_config)
    configs = metadata.get("configs") or []
    if configs:
        return ExperimentConfig.from_dict(configs[0])
    for row in payload.get("results", []):
        if "config_signature" in row:
            return ExperimentConfig.from_dict(row)
    return None


def config_from_row(row: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.from_dict(row)


def unique_configs_from_results(payload: dict[str, Any]) -> list[ExperimentConfig]:
    metadata = payload.get("metadata", {})
    configs = metadata.get("configs") or []
    if configs:
        return [ExperimentConfig.from_dict(config) for config in configs]
    seen: dict[str, ExperimentConfig] = {}
    for row in payload.get("results", []):
        signature = row.get("config_signature")
        if signature and signature not in seen:
            seen[signature] = config_from_row(row)
    return list(seen.values())


def config_signature_from_row(row: dict[str, Any]) -> str:
    try:
        return ExperimentConfig.from_dict(row).config_signature
    except Exception:
        return str(row.get("config_signature", "unknown"))


def result_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (row["mode"], config_signature_from_row(row), int(row["seq_len"]), row["baseline"])


def results_by_key(results: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    return {result_key(row): row for row in results if row.get("status") == "ok"}


def rows_for_signature(results: list[dict[str, Any]], *, mode: str, config_signature: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in results
        if row.get("status") == "ok" and row.get("mode") == mode and row.get("config_signature") == config_signature
    ]
    return sorted(rows, key=lambda row: (int(row["seq_len"]), row["baseline"]))


def seq_lens_for_signature(results: list[dict[str, Any]], *, mode: str, config_signature: str) -> list[int]:
    return sorted(
        {
            int(row["seq_len"])
            for row in results
            if row.get("status") == "ok" and row.get("mode") == mode and row.get("config_signature") == config_signature
        }
    )


def first_available_seq_len(results: list[dict[str, Any]], preferred: int, *, mode: str, config_signature: str) -> int | None:
    seq_lens = seq_lens_for_signature(results, mode=mode, config_signature=config_signature)
    if not seq_lens:
        return None
    if preferred in seq_lens:
        return preferred
    return seq_lens[min(range(len(seq_lens)), key=lambda idx: abs(seq_lens[idx] - preferred))]


def filter_rows(results: list[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    selected = []
    for row in results:
        if row.get("status") != "ok":
            continue
        if any(row.get(key) != value for key, value in criteria.items()):
            continue
        selected.append(row)
    return selected


def svg_escape(text: str) -> str:
    return html.escape(text, quote=True)


def x_positions(labels: list[str], left: int, width: int) -> list[float]:
    if len(labels) == 1:
        return [left + width / 2.0]
    step = width / (len(labels) - 1)
    return [left + i * step for i in range(len(labels))]


def y_linear(value: float, y_min: float, y_max: float, top: int, height: int) -> float:
    if y_max <= y_min:
        return top + height
    ratio = (value - y_min) / (y_max - y_min)
    return top + height - ratio * height


def y_ticks(y_min: float, y_max: float, count: int) -> list[float]:
    if count <= 1:
        return [y_min, y_max]
    return [y_min + (y_max - y_min) * i / (count - 1) for i in range(count)]


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
    formatter,
    *,
    y_min: float = 0.0,
    y_max: float | None = None,
    tick_count: int = 5,
) -> str:
    chart_w = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_h = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    x_vals = x_positions(labels, MARGIN_LEFT, chart_w)
    all_values = [value for item in series for value in item["values"] if value is not None]
    if y_max is None:
        ymax = max(all_values) if all_values else 1.0
        y_max = ymax * 1.08 if ymax > 0 else 1.0

    grid_parts = []
    for tick in y_ticks(y_min, y_max, tick_count):
        y = y_linear(tick, y_min, y_max, MARGIN_TOP, chart_h)
        grid_parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{y:.2f}" stroke="#273142" stroke-width="1"></line>'
        )
        grid_parts.append(
            f'<text x="{MARGIN_LEFT - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#9fb0c3">{svg_escape(formatter(tick))}</text>'
        )

    axis_parts = [
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP + chart_h}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<text x="22" y="{MARGIN_TOP + chart_h / 2:.2f}" transform="rotate(-90 22 {MARGIN_TOP + chart_h / 2:.2f})" font-size="13" fill="#c7d2df">{svg_escape(y_label)}</text>',
    ]
    for label, x in zip(labels, x_vals):
        axis_parts.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - 24}" text-anchor="middle" font-size="12" fill="#c7d2df">{svg_escape(label)}</text>'
        )

    series_parts = []
    for item in series:
        points = []
        for x, value in zip(x_vals, item["values"]):
            if value is None:
                continue
            y = y_linear(float(value), y_min, y_max, MARGIN_TOP, chart_h)
            points.append((x, y))
        if not points:
            continue
        series_parts.append(
            f'<path d="{polyline_path(points)}" fill="none" stroke="{item["color"]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>'
        )
        for x, y in points:
            series_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{item["color"]}" stroke="#10151d" stroke-width="1.5"></circle>'
            )

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
) -> str:
    chart_w = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_h = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    bar_group_width = chart_w / max(len(labels), 1)
    bar_width = min(70.0, bar_group_width * 0.64)
    x_vals = [MARGIN_LEFT + bar_group_width * i + bar_group_width / 2.0 for i in range(len(labels))]

    grid_parts = []
    for tick in y_ticks(0.0, 1.0, 5):
        y = y_linear(tick, 0.0, 1.0, MARGIN_TOP, chart_h)
        grid_parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{y:.2f}" stroke="#273142" stroke-width="1"></line>'
        )
        grid_parts.append(
            f'<text x="{MARGIN_LEFT - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#9fb0c3">{tick * 100:.0f}%</text>'
        )

    axis_parts = [
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP + chart_h}" x2="{SVG_WIDTH - MARGIN_RIGHT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{MARGIN_TOP + chart_h}" stroke="#5b6b80" stroke-width="1.2"></line>',
        f'<text x="22" y="{MARGIN_TOP + chart_h / 2:.2f}" transform="rotate(-90 22 {MARGIN_TOP + chart_h / 2:.2f})" font-size="13" fill="#c7d2df">{svg_escape(y_label)}</text>',
    ]
    for label, x in zip(labels, x_vals):
        axis_parts.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - 24}" text-anchor="middle" font-size="12" fill="#c7d2df">{svg_escape(label)}</text>'
        )

    bar_parts = []
    for idx, x in enumerate(x_vals):
        bar_bottom = MARGIN_TOP + chart_h
        cursor = bar_bottom
        for item in series:
            value = float(item["values"][idx])
            height = chart_h * value
            cursor -= height
            bar_parts.append(
                f'<rect x="{x - bar_width / 2:.2f}" y="{cursor:.2f}" width="{bar_width:.2f}" height="{height:.2f}" fill="{item["color"]}" rx="4"></rect>'
            )

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


def parse_hex_color(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


def interpolate_hex_color(start: str, end: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    start_rgb = parse_hex_color(start)
    end_rgb = parse_hex_color(end)
    mixed = tuple(round(start_rgb[idx] + (end_rgb[idx] - start_rgb[idx]) * ratio) for idx in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def render_heatmap(
    title: str,
    subtitle: str,
    x_labels: list[str],
    y_labels: list[str],
    values: list[list[float | None]],
    *,
    low_color: str = "#6d1f33",
    high_color: str = "#2ec4b6",
) -> str:
    chart_left = 138
    chart_top = 108
    chart_w = SVG_WIDTH - chart_left - MARGIN_RIGHT
    chart_h = SVG_HEIGHT - chart_top - MARGIN_BOTTOM
    cols = max(len(x_labels), 1)
    rows = max(len(y_labels), 1)
    gap = 10.0
    cell_w = (chart_w - gap * (cols - 1)) / cols
    cell_h = (chart_h - gap * (rows - 1)) / rows

    axis_parts = [
        f'<text x="22" y="{chart_top + chart_h / 2:.2f}" transform="rotate(-90 22 {chart_top + chart_h / 2:.2f})" font-size="13" fill="#c7d2df">Batch</text>',
        f'<text x="{chart_left + chart_w / 2:.2f}" y="{SVG_HEIGHT - 22}" text-anchor="middle" font-size="13" fill="#c7d2df">Heads</text>',
    ]
    for col_idx, label in enumerate(x_labels):
        x = chart_left + col_idx * (cell_w + gap) + cell_w / 2.0
        axis_parts.append(
            f'<text x="{x:.2f}" y="{chart_top - 14}" text-anchor="middle" font-size="12" fill="#c7d2df">{svg_escape(label)}</text>'
        )
    for row_idx, label in enumerate(y_labels):
        y = chart_top + row_idx * (cell_h + gap) + cell_h / 2.0 + 4.0
        axis_parts.append(
            f'<text x="{chart_left - 14}" y="{y:.2f}" text-anchor="end" font-size="12" fill="#c7d2df">{svg_escape(label)}</text>'
        )

    cell_parts = []
    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            x = chart_left + col_idx * (cell_w + gap)
            y = chart_top + row_idx * (cell_h + gap)
            fill = "#1a2230" if value is None else interpolate_hex_color(low_color, high_color, float(value))
            label = "-" if value is None else f"{value * 100:.0f}%"
            text_fill = "#f5f7fb" if value is None or value < 0.72 else "#0b1118"
            cell_parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" rx="12" fill="{fill}" stroke="#2c394d" stroke-width="1"></rect>'
                f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h / 2 + 5:.2f}" text-anchor="middle" font-size="16" font-weight="700" fill="{text_fill}">{svg_escape(label)}</text>'
            )

    legend = render_legend([("0% all-four-ok", low_color), ("100% all-four-ok", high_color)], MARGIN_LEFT, 76)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <rect width="100%" height="100%" rx="22" fill="#111722"></rect>
  <text x="{MARGIN_LEFT}" y="34" font-size="24" font-weight="700" fill="#f5f7fb">{svg_escape(title)}</text>
  <text x="{MARGIN_LEFT}" y="56" font-size="13" fill="#9fb0c3">{svg_escape(subtitle)}</text>
  {legend}
  {''.join(axis_parts)}
  {''.join(cell_parts)}
</svg>
"""


def summarize_latencies(latencies_ms: list[float], normalized_items: int) -> dict[str, float]:
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    variance = sum((value - mean_ms) ** 2 for value in latencies_ms) / len(latencies_ms)
    std_ms = variance ** 0.5
    us_per_item = (mean_ms * 1000.0 / normalized_items) if normalized_items else 0.0
    throughput_tokens_per_s = (normalized_items * 1000.0 / mean_ms) if mean_ms > 0 else 0.0
    return {
        "mean_ms": mean_ms,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": std_ms,
        "us_per_item": us_per_item,
        "throughput_tokens_per_s": throughput_tokens_per_s,
    }


def filter_slow_tail_outliers(latencies_ms: list[float]) -> tuple[list[float], list[float]]:
    if len(latencies_ms) < 4:
        return list(latencies_ms), []
    med = median(latencies_ms)
    if med <= 0:
        return list(latencies_ms), []
    deviations = [abs(value - med) for value in latencies_ms]
    mad = median(deviations)
    if mad <= 0:
        return list(latencies_ms), []

    kept: list[float] = []
    removed: list[float] = []
    for value in latencies_ms:
        modified_z = 0.6745 * (value - med) / mad
        is_slow_tail_outlier = (
            value > med
            and value > med * OUTLIER_MEDIAN_RATIO_THRESHOLD
            and modified_z > OUTLIER_MODIFIED_Z_THRESHOLD
        )
        if is_slow_tail_outlier:
            removed.append(value)
        else:
            kept.append(value)

    if len(kept) < MIN_FILTERED_SAMPLES:
        return list(latencies_ms), []
    return kept, removed


def apply_outlier_filter(payload: dict[str, Any]) -> dict[str, Any]:
    filtered_results: list[dict[str, Any]] = []
    removed_total = 0
    rows_with_outliers = 0

    for row in payload["results"]:
        updated = dict(row)
        if row.get("status") != "ok" or not row.get("latencies_ms"):
            filtered_results.append(updated)
            continue

        raw_latencies = [float(value) for value in row["latencies_ms"]]
        filtered_latencies, removed_outliers = filter_slow_tail_outliers(raw_latencies)

        updated["raw_latencies_ms"] = [round(value, 6) for value in raw_latencies]
        updated["raw_num_samples"] = len(raw_latencies)
        updated["removed_outliers_ms"] = [round(value, 6) for value in removed_outliers]
        updated["num_outliers_removed"] = len(removed_outliers)
        updated["latencies_ms"] = [round(value, 6) for value in filtered_latencies]
        updated["num_samples"] = len(filtered_latencies)

        if removed_outliers:
            rows_with_outliers += 1
            removed_total += len(removed_outliers)

        summary = summarize_latencies(filtered_latencies, row["normalized_items"])
        updated.update({key: round(value, 6) for key, value in summary.items()})
        filtered_results.append(updated)

    metadata = dict(payload.get("metadata", {}))
    metadata["outlier_filter"] = {
        "type": "slow_tail_mad_filter",
        "modified_z_threshold": OUTLIER_MODIFIED_Z_THRESHOLD,
        "median_ratio_threshold": OUTLIER_MEDIAN_RATIO_THRESHOLD,
        "min_filtered_samples": MIN_FILTERED_SAMPLES,
        "removed_points_total": removed_total,
        "rows_with_outliers": rows_with_outliers,
    }
    return {"metadata": metadata, "results": filtered_results}


def synthesize_legacy_default_config(results: list[dict[str, Any]]) -> ExperimentConfig | None:
    ok_rows = [row for row in results if row.get("status") == "ok"]
    if not ok_rows:
        return None

    def first(baseline: str) -> dict[str, Any] | None:
        for row in ok_rows:
            if row.get("baseline") == baseline:
                return row
        return None

    any_row = ok_rows[0]
    ref_row = first("reference_attention") or first("flash_attention") or any_row
    mla_row = first("flash_mla") or any_row
    deepseek_row = first("deepseek_flash_mla") or mla_row

    std_head_dim = int(ref_row.get("head_dim_qk") or ref_row.get("head_dim_v") or 512)
    mla_head_dim_v = int(mla_row.get("head_dim_v") or std_head_dim)
    mla_head_dim_qk = int(mla_row.get("head_dim_qk") or mla_head_dim_v)
    mla_d_rope = max(mla_head_dim_qk - mla_head_dim_v, 0)
    deepseek_kv_lora_rank = int(deepseek_row.get("head_dim_v") or mla_head_dim_v)
    if (
        std_head_dim == 512
        and mla_head_dim_v == 512
        and mla_d_rope == 64
        and deepseek_kv_lora_rank == 512
        and "config_signature" not in any_row
    ):
        deepseek_qk_rope_head_dim = 64
        deepseek_qk_nope_head_dim = 128
    else:
        deepseek_qk_storage_width = int(deepseek_row.get("head_dim_qk") or (deepseek_kv_lora_rank + mla_d_rope))
        deepseek_qk_rope_head_dim = mla_d_rope
        deepseek_qk_nope_head_dim = max(deepseek_qk_storage_width - deepseek_qk_rope_head_dim, 0)

    return ExperimentConfig(
        batch=int(any_row.get("batch", 1)),
        num_heads=int(any_row.get("num_heads", 32)),
        num_kv_heads=int(any_row.get("num_kv_heads", 1)),
        std_head_dim=std_head_dim,
        mla_head_dim_v=mla_head_dim_v,
        mla_d_rope=mla_d_rope,
        deepseek_qk_nope_head_dim=deepseek_qk_nope_head_dim,
        deepseek_qk_rope_head_dim=deepseek_qk_rope_head_dim,
        deepseek_kv_lora_rank=deepseek_kv_lora_rank,
        block_size=int(any_row.get("block_size", 64)),
        k_chunk_size=int(any_row.get("k_chunk_size", 128)),
        max_cores_per_head_batch=int(any_row.get("max_cores_per_head_batch", 4)),
        deepseek_num_q_heads_per_core=int(any_row.get("deepseek_num_q_heads_per_core", 8)),
        torch_input_dtype=str(any_row.get("torch_input_dtype", "torch.bfloat16")),
    )


def backfill_config_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    metadata = dict(payload.get("metadata", {}))

    def has_signature_mismatch() -> bool:
        default_config_data = metadata.get("default_config")
        if isinstance(default_config_data, dict):
            computed = ExperimentConfig.from_dict(default_config_data).config_signature
            if default_config_data.get("config_signature") != computed:
                return True
        for row in results:
            if row.get("status") != "ok":
                continue
            if "config_signature" not in row:
                return True
            try:
                computed = ExperimentConfig.from_dict(row).config_signature
            except Exception:
                continue
            if row.get("config_signature") != computed:
                return True
        return False

    if (
        results
        and all("config_signature" in row for row in results if row.get("status") == "ok")
        and metadata.get("default_config")
        and not has_signature_mismatch()
    ):
        return payload

    default_config = synthesize_legacy_default_config(results)
    if default_config is None:
        return payload

    updated_results = []
    for row in results:
        updated = dict(row)
        updated.setdefault("case_group", row.get("case"))
        updated.setdefault("q_seq_len", row.get("seq_len") if row.get("mode") == "prefill" else 1)
        for key, value in default_config.to_dict().items():
            updated.setdefault(key, value)
        try:
            normalized_config = ExperimentConfig.from_dict(updated)
            updated["config_signature"] = normalized_config.config_signature
            updated["config_label"] = normalized_config.config_label
            updated["deepseek_num_q_shards"] = normalized_config.deepseek_num_q_shards
        except Exception:
            pass
        updated_results.append(updated)

    metadata.setdefault("comparison_mode", "strict_four_way")
    metadata["configs"] = [default_config.to_dict()]
    metadata["default_config"] = default_config.to_dict()
    metadata.setdefault(
        "decode_seq_lens",
        sorted({row["seq_len"] for row in updated_results if row.get("status") == "ok" and row.get("mode") == "decode"}),
    )
    metadata.setdefault(
        "prefill_seq_lens",
        sorted({row["seq_len"] for row in updated_results if row.get("status") == "ok" and row.get("mode") == "prefill"}),
    )
    metadata.setdefault(
        "sweep_axes",
        {
            "batches": sorted({row["batch"] for row in updated_results if row.get("status") == "ok"}),
            "num_heads": sorted({row["num_heads"] for row in updated_results if row.get("status") == "ok"}),
            "num_kv_heads": sorted({row["num_kv_heads"] for row in updated_results if row.get("status") == "ok"}),
            "value_dims": sorted({row["std_head_dim"] for row in updated_results if row.get("status") == "ok"}),
            "rope_dims": sorted({row["mla_d_rope"] for row in updated_results if row.get("status") == "ok"}),
            "deepseek_num_q_heads_per_core": sorted(
                {row["deepseek_num_q_heads_per_core"] for row in updated_results if row.get("status") == "ok"}
            ),
            "decode_seq_lens": metadata.get("decode_seq_lens", []),
            "prefill_seq_lens": metadata.get("prefill_seq_lens", []),
        },
    )
    metadata.setdefault("deepseek_parallelism_policy", "fixed")
    return {"metadata": metadata, "results": updated_results}


def first_row_for_baseline(results: list[dict[str, Any]], baseline: str) -> dict[str, Any] | None:
    for row in results:
        if row.get("status") == "ok" and row.get("baseline") == baseline:
            return row
    return None


def default_config_signature(payload: dict[str, Any]) -> str | None:
    config = default_config_from_payload(payload)
    return config.config_signature if config else None


def default_case_rows(payload: dict[str, Any], *, mode: str) -> list[tuple[str, int]]:
    signature = default_config_signature(payload)
    if signature is None:
        return []
    rows = {}
    for row in payload.get("results", []):
        if row.get("status") != "ok" or row.get("mode") != mode or row.get("config_signature") != signature:
            continue
        rows[row.get("case_group", row["case"])] = int(row["seq_len"])
    return sorted(rows.items(), key=lambda item: item[1])


def format_seq_list(seq_lens: list[int]) -> str:
    labels = [format_seq_len(value) for value in seq_lens]
    return ", ".join(labels)


def lookup_row(
    results_map: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    mode: str,
    config_signature: str,
    seq_len: int,
    baseline: str,
) -> dict[str, Any] | None:
    return results_map.get((mode, config_signature, seq_len, baseline))


def payload_has_multidim_sweep(payload: dict[str, Any]) -> bool:
    configs = unique_configs_from_results(payload)
    return len(configs) > 1


def deepseek_parallelism_policy(payload: dict[str, Any]) -> str:
    return str(payload.get("metadata", {}).get("deepseek_parallelism_policy", "fixed"))


def metric_or_dash(row: dict[str, Any] | None, key: str, fmt: str) -> str:
    if not row or row.get(key) is None:
        return "-"
    return format(row[key], fmt)


def sample_count_or_dash(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    if row.get("num_samples") is not None:
        return str(row["num_samples"])
    if row.get("latencies_ms") is not None:
        return str(len(row["latencies_ms"]))
    return "-"


def format_kept_ratio(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    kept = row.get("num_samples")
    raw = row.get("raw_num_samples")
    if kept is None:
        kept = len(row.get("latencies_ms", []))
    if raw is None:
        raw = len(row.get("raw_latencies_ms", row.get("latencies_ms", [])))
    return f"{kept}/{raw} kept"


def format_latency_stats_cell(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    return (
        f"avg {row['mean_ms']:.3f} / best {row['min_ms']:.3f} / worst {row['max_ms']:.3f} ms "
        f"({format_kept_ratio(row)})"
    )


def format_throughput_cell(row: dict[str, Any] | None) -> str:
    if not row or row.get("throughput_tokens_per_s") is None:
        return "-"
    return f"{row['throughput_tokens_per_s']:.1f} tok/s"


def format_ratio_value(value: float) -> str:
    if value >= 0.1:
        return f"{value:.2f}x"
    if value >= 0.01:
        return f"{value:.3f}x"
    return f"{value:.4f}x"


def ratio_or_dash(numerator: dict[str, Any] | None, denominator: dict[str, Any] | None) -> str:
    if not numerator or not denominator:
        return "-"
    return format_ratio_value(numerator["mean_ms"] / denominator["mean_ms"])


def summarize_methods(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: row["mean_ms"])
    best_latency = ordered[0]["mean_ms"]
    worst_latency = ordered[-1]["mean_ms"]
    best_group = [row for row in ordered if row["mean_ms"] <= best_latency * TIE_THRESHOLD_RATIO]
    worst_group = [row for row in ordered if row["mean_ms"] >= worst_latency / TIE_THRESHOLD_RATIO]
    tie_group = best_group if len(best_group) >= 2 else []
    return best_group, worst_group, tie_group


def format_method_group(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    return ", ".join(
        f"{row['baseline_label']} ({row['mean_ms']:.3f} ms, {row['throughput_tokens_per_s']:.1f} tok/s)"
        for row in rows
    )


def make_fixed_config_table(
    payload: dict[str, Any],
    results_map: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    mode: str,
    baselines: tuple[str, ...],
    title: str,
) -> str:
    default_config = default_config_from_payload(payload)
    if default_config is None:
        return f"# {title}\n\n未检测到可渲染的默认配置。\n"

    seq_lens = seq_lens_for_signature(payload["results"], mode=mode, config_signature=default_config.config_signature)
    if not seq_lens:
        return f"# {title}\n\n当前结果中没有 `{mode}` 的成功记录。\n"

    label_map = {baseline: LABELS[baseline] for baseline in baselines}
    lines = [
        f"# {title}",
        "",
        "说明：每个 workload 下各方法都重复运行，并在剔除慢尾异常点后重新计算 `avg / best / worst`。",
        "说明：`best=最小延迟`，`worst=最大延迟`，括号里的 `8/10 kept` 表示过滤后保留样本数。",
    ]
    if payload_has_multidim_sweep(payload):
        lines.append(
            f"说明：这张默认主表固定在 `config={default_config.config_signature}`，即 `{default_config.config_label}`；更多 B/H/dims 对比见 `multidim/` 目录。"
        )
    if "deepseek_flash_mla" in baselines:
        lines.append(
            "说明：`DeepSeek FlashMLA` 采用“预先完成一次张量适配后，只计时 backend device op”的口径，因此可与其它 TT device baseline 更公平比较。"
        )
    lines.extend(
        [
            "",
            "## 延迟统计",
            "",
            "| case | " + " | ".join(label_map[baseline] for baseline in baselines) + " |",
            "|---|" + "|".join("---" for _ in baselines) + "|",
        ]
    )
    for seq_len in seq_lens:
        rows = [
            lookup_row(results_map, mode=mode, config_signature=default_config.config_signature, seq_len=seq_len, baseline=baseline)
            for baseline in baselines
        ]
        if not any(rows):
            continue
        lines.append(
            "| {case} | {cells} |".format(
                case=f"{mode}_{format_seq_len(seq_len)}",
                cells=" | ".join(format_latency_stats_cell(row) for row in rows),
            )
        )
    lines.extend(
        [
            "",
            "## 平均吞吐量",
            "",
            "| case | " + " | ".join(label_map[baseline] for baseline in baselines) + " |",
            "|---|" + "|".join("---:" for _ in baselines) + "|",
        ]
    )
    for seq_len in seq_lens:
        rows = [
            lookup_row(results_map, mode=mode, config_signature=default_config.config_signature, seq_len=seq_len, baseline=baseline)
            for baseline in baselines
        ]
        if not any(rows):
            continue
        lines.append(
            "| {case} | {cells} |".format(
                case=f"{mode}_{format_seq_len(seq_len)}",
                cells=" | ".join(format_throughput_cell(row) for row in rows),
            )
        )
    return "\n".join(lines) + "\n"


def make_decode_table(payload: dict[str, Any], results_map: dict[tuple[str, str, int, str], dict[str, Any]]) -> str:
    return make_fixed_config_table(
        payload,
        results_map,
        mode="decode",
        baselines=DECODE_BASELINES,
        title="Decode 四方法 10 次统计（已去除慢尾异常点）",
    )


def make_decode_summary_table(payload: dict[str, Any]) -> str:
    del payload
    return "\n".join(
        [
            "# Decode 统计口径说明",
            "",
            "旧版这里曾经写成“方法间最好 / 最差 / 平局”汇总，但这不是当前需要的口径。",
            "当前请直接查看 `decode_four_methods.md`：它展示默认配置下按 `seq_len` 聚合的四方法统计；如果当前结果包含多维 sweep，请同时查看 `multidim/` 目录中的分轴表和图。",
            "",
        ]
    )


def make_prefill_table(payload: dict[str, Any], results_map: dict[tuple[str, str, int, str], dict[str, Any]]) -> str:
    return make_fixed_config_table(
        payload,
        results_map,
        mode="prefill",
        baselines=("reference_attention", "flash_attention", "flash_mla"),
        title="Prefill 三方法 10 次统计（已去除慢尾异常点）",
    )


def format_probe_ratio(ok: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{100.0 * ok / total:.1f}%"


def probe_axis_value(case: dict[str, Any], axis: str) -> int:
    config = case.get("config", {})
    if axis == "batch":
        return int(config["batch"])
    if axis == "heads":
        return int(config["num_heads"])
    if axis == "dims":
        return int(config["common_value_dim"])
    if axis == "seq_len":
        return int(case["seq_len"])
    raise ValueError(f"Unknown probe axis: {axis}")


def probe_axis_stats(probe_payload: dict[str, Any], axis: str) -> list[tuple[int, dict[str, int]]]:
    grouped: dict[int, dict[str, int]] = {}
    for case in probe_payload.get("cases", []):
        value = probe_axis_value(case, axis)
        stats = grouped.setdefault(value, {"ok": 0, "total": 0})
        stats["total"] += 1
        stats["ok"] += int(case.get("supported_by_all", False))
    return sorted(grouped.items(), key=lambda item: item[0])


def probe_bh_stats(probe_payload: dict[str, Any]) -> tuple[list[int], list[int], dict[tuple[int, int], dict[str, int]]]:
    batches: set[int] = set()
    heads: set[int] = set()
    grouped: dict[tuple[int, int], dict[str, int]] = {}
    for case in probe_payload.get("cases", []):
        batch = probe_axis_value(case, "batch")
        head = probe_axis_value(case, "heads")
        batches.add(batch)
        heads.add(head)
        stats = grouped.setdefault((batch, head), {"ok": 0, "total": 0})
        stats["total"] += 1
        stats["ok"] += int(case.get("supported_by_all", False))
    return sorted(batches), sorted(heads), grouped


def make_capability_probe_summary_table(probe_payload: dict[str, Any] | None) -> str:
    if not probe_payload:
        return "# Capability Probe 汇总\n\n未检测到 capability probe 结果，跳过该表。\n"

    metadata = probe_payload.get("metadata", {})
    cases = probe_payload.get("cases", [])
    total_cases = len(cases)
    all_ok_cases = sum(1 for case in cases if case.get("supported_by_all"))
    status = metadata.get("status", "-")
    batch_stats = probe_axis_stats(probe_payload, "batch")
    head_stats = probe_axis_stats(probe_payload, "heads")
    dim_stats = probe_axis_stats(probe_payload, "dims")
    seq_len_stats = probe_axis_stats(probe_payload, "seq_len")
    batches, heads, bh_stats = probe_bh_stats(probe_payload)

    lines = [
        "# Capability Probe 汇总",
        "",
        "说明：这里汇总 strict four-way capability probe 的结果，用来筛出四种 decode 方法共同支持的公共参数空间。",
        "说明：逐 case 明细仍保留在 `capability_probe.md`；这里优先展示支持边界与失败模式。",
        "",
        "## 1. 总览",
        "",
        "| 指标 | 当前值 | 说明 |",
        "|---|---|---|",
        f"| status | `{status}` | `completed` 表示 probe 已全部跑完；否则当前为 checkpoint 快照 |",
        f"| probe preset | `{metadata.get('probe_preset', 'manual')}` | 本次 probe 使用的 preset |",
        f"| device arch | `{metadata.get('device', {}).get('arch', '-')}` | probe 结果对应的设备架构 |",
        f"| all four ok | `{all_ok_cases}/{total_cases}` | 四种 decode 方法都可运行的 case 数量 |",
        f"| all four ok ratio | `{format_probe_ratio(all_ok_cases, total_cases)}` | 四方法公共支持比例 |",
        f"| baselines | `{', '.join(metadata.get('probe_baselines', [])) or '-'}` | 本次 probe 检查的 baseline 集合 |",
        "",
        "## 2. 共同支持边界（按 `B × H`）",
        "",
    ]
    for batch in batches:
        supported_heads = []
        blocked_heads = []
        for head in heads:
            stats = bh_stats.get((batch, head), {"ok": 0, "total": 0})
            if stats["total"] <= 0:
                continue
            head_label = f"H={head}"
            if stats["ok"] == stats["total"]:
                supported_heads.append(head_label)
            else:
                blocked_heads.append(head_label)
        lines.append(
            "- `B={batch}`: 四方法全支持 `{supported}`；不支持 `{blocked}`。".format(
                batch=batch,
                supported=" / ".join(supported_heads) if supported_heads else "无",
                blocked=" / ".join(blocked_heads) if blocked_heads else "无",
            )
        )

    lines.extend(
        [
            "",
            "## 3. 按 `B` 汇总",
            "",
            "| `B` | all four ok | total | support rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for batch, stats in batch_stats:
        lines.append(
            f"| `{batch}` | `{stats['ok']}` | `{stats['total']}` | `{format_probe_ratio(stats['ok'], stats['total'])}` |"
        )

    lines.extend(
        [
            "",
            "## 4. 按 `H` 汇总",
            "",
            "| `H` | all four ok | total | support rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for head, stats in head_stats:
        lines.append(
            f"| `{head}` | `{stats['ok']}` | `{stats['total']}` | `{format_probe_ratio(stats['ok'], stats['total'])}` |"
        )

    lines.extend(
        [
            "",
            "## 5. 按 `value_dim` 汇总",
            "",
            "| `value_dim` | all four ok | total | support rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for dim, stats in dim_stats:
        lines.append(
            f"| `{dim}` | `{stats['ok']}` | `{stats['total']}` | `{format_probe_ratio(stats['ok'], stats['total'])}` |"
        )

    lines.extend(
        [
            "",
            "## 6. 按 `decode seq_len` 汇总",
            "",
            "| `seq_len` | all four ok | total | support rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for seq_len, stats in seq_len_stats:
        lines.append(
            f"| `{format_seq_len(seq_len)}` | `{stats['ok']}` | `{stats['total']}` | `{format_probe_ratio(stats['ok'], stats['total'])}` |"
        )

    lines.extend(
        [
            "",
            "## 7. `B × H` 支持矩阵",
            "",
            "| `B \\ H` | " + " | ".join(f"`H={head}`" for head in heads) + " |",
            "|" + "---|" * (len(heads) + 1),
        ]
    )
    for batch in batches:
        row_cells = [f"`B={batch}`"]
        for head in heads:
            stats = bh_stats.get((batch, head), {"ok": 0, "total": 0})
            row_cells.append(f"`{format_probe_ratio(stats['ok'], stats['total'])} ({stats['ok']}/{stats['total']})`")
        lines.append("| " + " | ".join(row_cells) + " |")

    unsupported_by_baseline: dict[str, int] = {}
    reason_by_baseline: dict[str, dict[str, int]] = {}
    for case in cases:
        for baseline in case.get("unsupported_baselines", []):
            unsupported_by_baseline[baseline] = unsupported_by_baseline.get(baseline, 0) + 1
            baseline_reasons = reason_by_baseline.setdefault(baseline, {})
            reason = case.get("baseline_errors", {}).get(baseline, "-")
            baseline_reasons[reason] = baseline_reasons.get(reason, 0) + 1

    lines.extend(
        [
            "",
            "## 8. 失败模式",
            "",
            "| baseline | unsupported cases | top reasons |",
            "|---|---:|---|",
        ]
    )
    if unsupported_by_baseline:
        for baseline, count in sorted(unsupported_by_baseline.items(), key=lambda item: (-item[1], item[0])):
            top_reasons = sorted(reason_by_baseline.get(baseline, {}).items(), key=lambda item: (-item[1], item[0]))[:3]
            reason_text = "; ".join(f"`{reason}` x {reason_count}" for reason, reason_count in top_reasons) if top_reasons else "-"
            lines.append(f"| `{baseline}` | `{count}` | {reason_text} |")
    else:
        lines.append("| `-` | `0` | 所有 baseline 全通过 |")

    return "\n".join(lines) + "\n"


def make_capability_probe_table(probe_payload: dict[str, Any] | None) -> str:
    if not probe_payload:
        return "# Capability Probe\n\n未检测到 capability probe 结果，跳过该表。\n"

    metadata = probe_payload.get("metadata", {})
    lines = [
        "# Strict Four-Way Capability Probe",
        "",
        "说明：probe 只做功能可运行性检查，不作为论文主结果；每个候选点对四种 decode 方法各跑一次，输出 `ok / unsupported / error`。",
        "说明：若只想先看支持边界和失败模式，请优先查看 `capability_probe_summary.md`。",
    ]
    if metadata.get("status") != "completed":
        lines.extend(
            [
                (
                    "说明：当前展示的是 checkpoint 快照，"
                    f"已完成 `{metadata.get('processed_workloads', 0)}/{metadata.get('total_workloads', 0)}` 个 workloads，"
                    f"剩余 `{metadata.get('remaining_workloads', 0)}` 个。"
                ),
                "",
            ]
        )
    else:
        lines.append("")
    lines.extend(
        [
        "| case | config | all four ok | Reference | Flash | TT-MLA | DeepSeek |",
        "|---|---|---|---|---|---|---|",
        ]
    )
    for case in probe_payload.get("cases", []):
        status = case.get("baseline_status", {})
        lines.append(
            "| {case_name} | `{config}` | `{all_ok}` | `{ref}` | `{flash}` | `{mla}` | `{deepseek}` |".format(
                case_name=case["case"],
                config=case["config_signature"],
                all_ok=str(case.get("supported_by_all", False)).lower(),
                ref=status.get("reference_attention", "-"),
                flash=status.get("flash_attention", "-"),
                mla=status.get("flash_mla", "-"),
                deepseek=status.get("deepseek_flash_mla", "-"),
            )
        )
    return "\n".join(lines) + "\n"


def make_experiment_parameter_table(
    payload: dict[str, Any], detail_payload: dict[str, Any] | None, probe_payload: dict[str, Any] | None
) -> str:
    metadata = payload.get("metadata", {})
    results = payload.get("results", [])
    device = metadata.get("device", {})
    outlier_filter = metadata.get("outlier_filter", {})
    probe_metadata = probe_payload.get("metadata", {}) if probe_payload else {}
    default_config = default_config_from_payload(payload)
    configs = unique_configs_from_results(payload)
    sweep_axes = metadata.get("sweep_axes", {})
    decode_seq_lens = metadata.get("decode_seq_lens") or sorted(
        {row["seq_len"] for row in results if row.get("status") == "ok" and row.get("mode") == "decode"}
    ) or list(DEFAULT_DECODE_SEQ_LENS)
    prefill_seq_lens = metadata.get("prefill_seq_lens") or sorted(
        {row["seq_len"] for row in results if row.get("status") == "ok" and row.get("mode") == "prefill"}
    ) or list(DEFAULT_PREFILL_SEQ_LENS)
    grid_x = device.get("grid_x")
    grid_y = device.get("grid_y")
    grid_display = f"{grid_x} x {grid_y}" if grid_x is not None and grid_y is not None else "-"

    if default_config is None:
        return "# Part I 实验参数总表\n\n未检测到可用的默认配置。\n"

    decode_q_num_cores = (
        min(default_config.batch * default_config.num_heads, grid_x * grid_y)
        if isinstance(grid_x, int) and isinstance(grid_y, int)
        else "-"
    )
    deepseek_required_q_cores = (
        default_config.batch * default_config.deepseek_num_q_shards
        if default_config.deepseek_num_q_shards is not None
        else "-"
    )

    detail_metadata = detail_payload.get("metadata", {}) if detail_payload else {}
    detail_cases = detail_metadata.get("selected_cases", ["decode_1k", "decode_4k", "decode_8k", "decode_16k", "decode_32k"])
    reuse_existing = bool(detail_metadata.get("reused_existing"))
    run_flashmla_detailed = bool(detail_payload) and not reuse_existing
    reuse_source = detail_metadata.get("reused_existing_source", "-")

    def axis_display(name: str) -> str:
        values = sweep_axes.get(name) or []
        return ", ".join(str(value) for value in values) if values else "-"

    lines = [
        "# Part I 实验参数总表",
        "",
        "说明：当前文档整理的是 `run_part1_benchmarks.py` 这套 Part I 多维 sweep 在当前结果中的实际配置。",
        "说明：记号约定为 `B=batch`，`H=num_heads`，`H_kv=num_kv_heads`，`Q=q_seq_len`，`S=kv_seq_len`。",
        "",
        "## 1. 当前运行口径",
        "",
        "| 项目 | 当前值 | 说明 |",
        "|---|---|---|",
        f"| comparison mode | `{metadata.get('comparison_mode', '-')}` | 当前实现的比较模式 |",
        f"| sweep preset | `{metadata.get('sweep_preset', 'manual')}` | 主跑使用的 preset；`manual` 表示直接读取 CLI 轴参数 |",
        f"| probe preset | `{probe_metadata.get('probe_preset', metadata.get('probe_preset', 'manual'))}` | capability probe 使用的 preset；`manual` 表示直接读取 CLI 轴参数 |",
        f"| DeepSeek parallelism policy | `{metadata.get('deepseek_parallelism_policy', 'fixed')}` | `fixed`=固定 q_heads_per_core；`align_with_tt_mainline`=按 TT 主线并行度派生 |",
        f"| device_id | `{device.get('device_id', '-')}` | TT 设备编号 |",
        f"| arch | `{device.get('arch', '-')}` | 当前结果中的设备架构 |",
        f"| compute grid | `{grid_display}` | 当前结果中的 `compute_with_storage_grid_size` |",
        f"| total configs | `{len(configs)}` | 当前主跑实际生成的 config 数量 |",
        f"| default config | `{default_config.config_signature}` | 默认主表固定使用的 config |",
        f"| default config label | `{default_config.config_label}` | 默认主表的人类可读说明 |",
        f"| decode seq_len sweep | `{format_seq_list(decode_seq_lens)}` | decode 主 sweep |",
        f"| prefill seq_len sweep | `{format_seq_list(prefill_seq_lens)}` | prefill 控制组 |",
        f"| torch input dtype | `{default_config.torch_input_dtype}` | host 随机输入生成 dtype |",
        f"| warmup_device / iters_device | `{metadata.get('warmup_device', '-')} / {metadata.get('iters_device', '-')}` | TT baseline 预热 / 正式测量次数 |",
        f"| warmup_reference / iters_reference | `{metadata.get('warmup_reference', '-')} / {metadata.get('iters_reference', '-')}` | torch reference 预热 / 正式测量次数 |",
        f"| outlier filter | `modified z-score > {outlier_filter.get('modified_z_threshold', '-')} and latency > median x {outlier_filter.get('median_ratio_threshold', '-')}` | 当前主表使用的慢尾异常点过滤规则 |",
        f"| filtered minimum kept samples | `{outlier_filter.get('min_filtered_samples', '-')}` | 若过滤后样本过少则回退到原始样本 |",
        "",
        "## 2. Sweep 轴",
        "",
        "| 轴 | 当前值 | 说明 |",
        "|---|---|---|",
        f"| `B` | `{axis_display('batches')}` | 主跑 batch sweep |",
        f"| `H` | `{axis_display('num_heads')}` | 主跑 num_heads sweep |",
        f"| `H_kv` | `{axis_display('num_kv_heads')}` | 主跑 num_kv_heads sweep |",
        f"| `value_dim` | `{axis_display('value_dims')}` | 严格四方法公共 value dim sweep |",
        f"| `rope_dim` | `{axis_display('rope_dims')}` | rope dim sweep |",
        f"| `deepseek_num_q_heads_per_core` | `{axis_display('deepseek_num_q_heads_per_core')}` | DeepSeek Q shard 粒度 sweep / 派生结果 |",
        f"| capability probe | `{str(probe_payload is not None).lower()}` | status=`{probe_metadata.get('status', '-')}`；汇总见 `tables/capability_probe_summary.md`，原始结果见 `raw/capability_probe_results.json` |",
        "",
        "## 3. 默认 config 显式字段",
        "",
        "| 字段 | 当前值 | 说明 |",
        "|---|---:|---|",
        f"| `batch` | `{default_config.batch}` | 默认主表 batch |",
        f"| `num_heads` | `{default_config.num_heads}` | 默认主表 attention heads |",
        f"| `num_kv_heads` | `{default_config.num_kv_heads}` | 默认主表 kv heads |",
        f"| `std_head_dim` | `{default_config.std_head_dim}` | 标准 attention 的 `d_q=d_k=d_v` |",
        f"| `mla_head_dim_v` | `{default_config.mla_head_dim_v}` | TT 主线 MLA 的 `d_v` |",
        f"| `mla_d_rope` | `{default_config.mla_d_rope}` | TT 主线 MLA rope 维度 |",
        f"| `mla_head_dim_qk` | `{default_config.mla_head_dim_qk}` | `mla_head_dim_v + mla_d_rope` |",
        f"| `deepseek_qk_nope_head_dim` | `{default_config.deepseek_qk_nope_head_dim}` | DeepSeek 逻辑 QK 的 non-rope 部分 |",
        f"| `deepseek_qk_rope_head_dim` | `{default_config.deepseek_qk_rope_head_dim}` | DeepSeek 逻辑 QK 的 rope 部分 |",
        f"| `deepseek_qk_head_dim` | `{default_config.deepseek_qk_head_dim}` | DeepSeek 缩放维度 |",
        f"| `deepseek_kv_lora_rank` | `{default_config.deepseek_kv_lora_rank}` | DeepSeek 输出 / value 维度 |",
        f"| `deepseek_kvpe_dim` | `{default_config.deepseek_kvpe_dim}` | DeepSeek Q/KV 存储宽度 |",
        f"| `block_size` | `{default_config.block_size}` | paged attention block size |",
        f"| `k_chunk_size` | `{default_config.k_chunk_size}` | decode k chunk size |",
        f"| `max_cores_per_head_batch` | `{default_config.max_cores_per_head_batch}` | TT 主线 decode program config |",
        f"| `deepseek_num_q_heads_per_core` | `{default_config.deepseek_num_q_heads_per_core}` | DeepSeek Q shard 粒度 |",
        "",
        "## 4. 默认 config 下的张量形状",
        "",
        "| 张量 | 形状 | 说明 |",
        "|---|---|---|",
        f"| `q_std` | `[B, H, Q, {default_config.std_head_dim}]` | Reference / Flash Attention 用 Q |",
        f"| `k_std`, `v_std` | `[B, H_kv, S, {default_config.std_head_dim}]` | 标准 attention 的 K/V |",
        f"| `q_mla` | `[B, H, Q, {default_config.mla_head_dim_qk}]` | TT-MLA 用 Q |",
        f"| `k_mla` | `[B, H_kv, S, {default_config.mla_head_dim_qk}]` | TT-MLA 用 K，其中前 `{default_config.mla_head_dim_v}` 维隐含 V |",
        f"| `q_deepseek` | `[B, H, Q, {default_config.deepseek_kvpe_dim}]` | DeepSeek 用 Q 存储宽度 |",
        f"| `k_deepseek` | `[B, H_kv, S, {default_config.deepseek_kvpe_dim}]` | DeepSeek KV cache 宽度 |",
        f"| decode `Q` | `1` | 只解当前 token |",
        f"| decode `S` | `seq_len` | 历史 cache 长度 |",
        f"| prefill `Q=S` | `seq_len` | 全序列因果 prefill |",
        "",
        "## 5. 四条 baseline 的默认设置",
        "",
        "| baseline | mode | Q 输入 | K / KV 输入 | V 输入 | scale | 实际调用 |",
        "|---|---|---|---|---|---|---|",
        f"| `reference_attention` | decode + prefill | `q_std` | `k_std` | `v_std` | `{default_config.std_head_dim}^-0.5` | torch reference SDPA |",
        f"| `flash_attention` | decode + prefill | `q_std` | `k_std` | `v_std` | `{default_config.std_head_dim}^-0.5` | `scaled_dot_product_attention` / `paged_scaled_dot_product_attention_decode` |",
        f"| `flash_mla` | decode + prefill | `q_mla` | `k_mla` | `None`，`V` 取前 `{default_config.mla_head_dim_v}` 维 | `{default_config.mla_head_dim_qk}^-0.5` | `flash_mla_prefill` / `paged_flash_multi_latent_attention_decode` |",
        f"| `deepseek_flash_mla` | decode only | `q_deepseek` | `k_deepseek` | `None`，输出宽度 `{default_config.deepseek_kv_lora_rank}` | `{default_config.deepseek_qk_head_dim}^-0.5` | `flash_multi_latent_attention_decode` |",
        "",
        "## 6. Program / Kernel / Guardrails",
        "",
        "| 项目 | 当前值 | 说明 |",
        "|---|---|---|",
        f"| decode q_num_cores | `{decode_q_num_cores}` | TT 主线 decode 默认 `q_num_cores=min(B*H, grid_x*grid_y)` |",
        f"| DeepSeek num_q_shards | `{default_config.deepseek_num_q_shards}` | `num_heads / deepseek_num_q_heads_per_core` |",
        f"| DeepSeek required q cores | `{deepseek_required_q_cores}` | 当前 builder 使用 `batch * num_q_shards` 个活跃核 |",
        f"| DeepSeek grid requirement | `>= 8 x 7` | 当前 Wormhole 路径要求 |",
        f"| DeepSeek k_chunk_size | `{default_config.k_chunk_size}` | 直接从 config 读取 |",
        f"| DeepSeek KV ND shard shape | `[1, H_kv, {default_config.k_chunk_size}, {default_config.deepseek_kvpe_dim}]` | ND-sharded DRAM cache |",
        "| measurement scope | `device_core_only_after_one_time_adapter_conversion` | benchmark 只计后端 device op，排除一次性张量适配开销 |",
        "",
        "## 7. CLI / 结果文件",
        "",
        "| 参数 | 当前值 | 说明 |",
        "|---|---|---|",
        f"| `--cases` | `{', '.join(metadata.get('cases', []))}` | 本次实际运行的 benchmark cases |",
        f"| `--detail-cases` | `{', '.join(detail_cases)}` | detailed FlashMLA profiling 选择的 case |",
        f"| `--run-capability-probe` | `{str(probe_payload is not None or bool(metadata.get('run_capability_probe'))).lower()}` | 当前工作区是否存在 capability probe 结果 |",
        f"| `--run-flashmla-detailed` | `{str(run_flashmla_detailed).lower()}` | 是否现场重跑 detailed tracy profile |",
        f"| `--reuse-existing-flashmla-detailed` | `{str(reuse_existing).lower()}` | 当前是否复用现有稳定 detailed 结果 |",
        f"| `--skip-render` | `false` | 当前结果已完成 render |",
        "",
        "## 8. 结果文件",
        "",
        f"- 主报告：`{REPORT_MD.relative_to(OUTPUT_DIR)}`",
        f"- 参数总表：`{PARAMETER_TABLE_MD.relative_to(OUTPUT_DIR)}`",
        f"- Decode 默认主表：`{DECODE_TABLE_MD.relative_to(OUTPUT_DIR)}`",
        f"- Prefill 默认主表：`{PREFILL_TABLE_MD.relative_to(OUTPUT_DIR)}`",
        f"- Capability probe 汇总：`{CAPABILITY_PROBE_SUMMARY_MD.relative_to(OUTPUT_DIR)}`",
        f"- Capability probe：`{CAPABILITY_PROBE_MD.relative_to(OUTPUT_DIR)}`",
        f"- 多维结果目录：`{MULTIDIM_DIR.relative_to(OUTPUT_DIR)}`",
        f"- 过滤后 JSON：`{FILTERED_JSON.relative_to(OUTPUT_DIR)}`",
        f"- 过滤后 CSV：`{FILTERED_CSV.relative_to(OUTPUT_DIR)}`",
    ]
    if reuse_existing:
        lines.append(f"- Detailed profile 来源：`{reuse_source}`")
    return "\n".join(lines) + "\n"


def same_except_axis(config: ExperimentConfig, default: ExperimentConfig, axis: str, *, policy: str) -> bool:
    if axis == "batch":
        return (
            config.num_heads == default.num_heads
            and config.num_kv_heads == default.num_kv_heads
            and config.common_value_dim == default.common_value_dim
            and config.mla_d_rope == default.mla_d_rope
            and config.block_size == default.block_size
            and config.k_chunk_size == default.k_chunk_size
            and config.deepseek_num_q_heads_per_core == default.deepseek_num_q_heads_per_core
        )
    if axis == "heads":
        return (
            config.batch == default.batch
            and config.num_kv_heads == default.num_kv_heads
            and config.common_value_dim == default.common_value_dim
            and config.mla_d_rope == default.mla_d_rope
            and config.block_size == default.block_size
            and config.k_chunk_size == default.k_chunk_size
            and (
                policy == "align_with_tt_mainline"
                or config.deepseek_num_q_heads_per_core == default.deepseek_num_q_heads_per_core
            )
        )
    if axis == "dims":
        return (
            config.batch == default.batch
            and config.num_heads == default.num_heads
            and config.num_kv_heads == default.num_kv_heads
            and config.block_size == default.block_size
            and config.k_chunk_size == default.k_chunk_size
            and config.deepseek_num_q_heads_per_core == default.deepseek_num_q_heads_per_core
        )
    if axis == "deepseek_qhpc":
        return (
            config.batch == default.batch
            and config.num_heads == default.num_heads
            and config.num_kv_heads == default.num_kv_heads
            and config.common_value_dim == default.common_value_dim
            and config.mla_d_rope == default.mla_d_rope
            and config.block_size == default.block_size
            and config.k_chunk_size == default.k_chunk_size
        )
    raise ValueError(f"Unknown axis: {axis}")


def axis_value(config: ExperimentConfig, axis: str) -> tuple[int, ...]:
    if axis == "batch":
        return (config.batch,)
    if axis == "heads":
        return (config.num_heads,)
    if axis == "dims":
        return (config.common_value_dim, config.mla_d_rope)
    if axis == "deepseek_qhpc":
        return (config.deepseek_num_q_heads_per_core,)
    raise ValueError(f"Unknown axis: {axis}")


def axis_label(config: ExperimentConfig, axis: str, *, policy: str) -> str:
    if axis == "batch":
        return f"B={config.batch}"
    if axis == "heads":
        if policy == "align_with_tt_mainline":
            return (
                f"H={config.num_heads} "
                f"(dqhpc={config.deepseek_num_q_heads_per_core}, q_shards={config.deepseek_num_q_shards})"
            )
        return f"H={config.num_heads}"
    if axis == "dims":
        return f"value={config.common_value_dim}, rope={config.mla_d_rope}"
    if axis == "deepseek_qhpc":
        return f"dqhpc={config.deepseek_num_q_heads_per_core}, q_shards={config.deepseek_num_q_shards}"
    raise ValueError(f"Unknown axis: {axis}")


def make_axis_sweep_table(
    payload: dict[str, Any],
    results_map: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    axis: str,
    title: str,
) -> str:
    default_config = default_config_from_payload(payload)
    configs = unique_configs_from_results(payload)
    policy = deepseek_parallelism_policy(payload)
    if default_config is None:
        return f"# {title}\n\n未检测到默认配置。\n"

    candidates = sorted(
        [config for config in configs if same_except_axis(config, default_config, axis, policy=policy)],
        key=lambda config: axis_value(config, axis),
    )
    distinct_values = {axis_value(config, axis) for config in candidates}
    if len(distinct_values) <= 1:
        return f"# {title}\n\n当前结果里 `{axis}` 只有一个取值，暂无可展示的 sweep。\n"

    seq_len = first_available_seq_len(payload["results"], 8192, mode="decode", config_signature=default_config.config_signature)
    if seq_len is None:
        return f"# {title}\n\n当前结果里没有可用于 `{axis}` 视图的 decode 记录。\n"

    labels = [axis_label(config, axis, policy=policy) for config in candidates]
    lines = [
        f"# {title}",
        "",
        f"说明：固定 `seq_len={format_seq_len(seq_len)}`，其余轴保持默认 config：`{default_config.config_label}`。",
        "",
        "## 延迟统计",
        "",
        "| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
        "|---|---|---|---|---|",
    ]
    for config, label in zip(candidates, labels):
        rows = [
            lookup_row(results_map, mode="decode", config_signature=config.config_signature, seq_len=seq_len, baseline=baseline)
            for baseline in DECODE_BASELINES
        ]
        if not any(rows):
            continue
        lines.append("| {label} | {cells} |".format(label=label, cells=" | ".join(format_latency_stats_cell(row) for row in rows)))
    lines.extend(
        [
            "",
            "## 平均吞吐量",
            "",
            "| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for config, label in zip(candidates, labels):
        rows = [
            lookup_row(results_map, mode="decode", config_signature=config.config_signature, seq_len=seq_len, baseline=baseline)
            for baseline in DECODE_BASELINES
        ]
        if not any(rows):
            continue
        lines.append("| {label} | {cells} |".format(label=label, cells=" | ".join(format_throughput_cell(row) for row in rows)))
    return "\n".join(lines) + "\n"


def make_axis_latency_chart(
    payload: dict[str, Any],
    results_map: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    axis: str,
    title: str,
    subtitle: str,
) -> str | None:
    default_config = default_config_from_payload(payload)
    configs = unique_configs_from_results(payload)
    policy = deepseek_parallelism_policy(payload)
    if default_config is None:
        return None

    candidates = sorted(
        [config for config in configs if same_except_axis(config, default_config, axis, policy=policy)],
        key=lambda config: axis_value(config, axis),
    )
    distinct_values = {axis_value(config, axis) for config in candidates}
    if len(distinct_values) <= 1:
        return None

    seq_len = first_available_seq_len(payload["results"], 8192, mode="decode", config_signature=default_config.config_signature)
    if seq_len is None:
        return None

    labels = [axis_label(config, axis, policy=policy) for config in candidates]
    series = []
    for baseline in DECODE_BASELINES:
        values = [
            (
                lookup_row(results_map, mode="decode", config_signature=config.config_signature, seq_len=seq_len, baseline=baseline) or {}
            ).get("mean_ms")
            for config in candidates
        ]
        series.append({"label": LABELS[baseline], "color": COLORS[baseline], "values": values})
    return render_line_chart(
        title,
        subtitle.format(seq_len=format_seq_len(seq_len), config=default_config.config_label),
        labels,
        series,
        "Latency (ms)",
        lambda value: f"{value:.2f}",
        y_min=0.0,
    )


def build_multidim_artifacts(payload: dict[str, Any]) -> dict[str, Path]:
    if not payload_has_multidim_sweep(payload):
        return {}

    MULTIDIM_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MULTIDIM_VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    results_map = results_by_key(payload["results"])

    write_text(MULTIDIM_BATCH_TABLE_MD, make_axis_sweep_table(payload, results_map, axis="batch", title="Decode Batch Sweep"))
    write_text(MULTIDIM_HEAD_TABLE_MD, make_axis_sweep_table(payload, results_map, axis="heads", title="Decode Head Sweep"))
    write_text(MULTIDIM_DIM_TABLE_MD, make_axis_sweep_table(payload, results_map, axis="dims", title="Decode Dim Sweep"))
    write_text(
        MULTIDIM_DEEPSEEK_QHPC_TABLE_MD,
        make_axis_sweep_table(
            payload,
            results_map,
            axis="deepseek_qhpc",
            title="Decode DeepSeek Q-Heads-Per-Core Sweep",
        ),
    )

    artifacts: dict[str, Path] = {
        "batch_table": MULTIDIM_BATCH_TABLE_MD,
        "head_table": MULTIDIM_HEAD_TABLE_MD,
        "dim_table": MULTIDIM_DIM_TABLE_MD,
        "deepseek_qhpc_table": MULTIDIM_DEEPSEEK_QHPC_TABLE_MD,
    }

    charts = {
        "batch_latency": (
            MULTIDIM_VISUAL_DIR / "decode_batch_sweep_latency.svg",
            make_axis_latency_chart(
                payload,
                results_map,
                axis="batch",
                title="Decode Batch Sweep Latency",
                subtitle="固定 `seq_len={seq_len}`，仅变化 batch；默认其余参数为 `{config}`。",
            ),
        ),
        "head_latency": (
            MULTIDIM_VISUAL_DIR / "decode_head_sweep_latency.svg",
            make_axis_latency_chart(
                payload,
                results_map,
                axis="heads",
                title="Decode Head Sweep Latency",
                subtitle="固定 `seq_len={seq_len}`，仅变化 num_heads；默认其余参数为 `{config}`。",
            ),
        ),
        "dim_latency": (
            MULTIDIM_VISUAL_DIR / "decode_dim_sweep_latency.svg",
            make_axis_latency_chart(
                payload,
                results_map,
                axis="dims",
                title="Decode Dim Sweep Latency",
                subtitle="固定 `seq_len={seq_len}`，仅变化 value/rope dims；默认其余参数为 `{config}`。",
            ),
        ),
        "deepseek_qhpc_latency": (
            MULTIDIM_VISUAL_DIR / "decode_deepseek_q_heads_per_core_sweep_latency.svg",
            make_axis_latency_chart(
                payload,
                results_map,
                axis="deepseek_qhpc",
                title="Decode DeepSeek Q-Heads-Per-Core Sweep Latency",
                subtitle="固定 `seq_len={seq_len}`，仅变化 DeepSeek q_heads_per_core；默认其余参数为 `{config}`。",
            ),
        ),
    }
    for key, (path, svg) in charts.items():
        if svg is None:
            continue
        write_text(path, svg)
        artifacts[key] = path
    return artifacts


def build_capability_probe_visuals(probe_payload: dict[str, Any] | None) -> dict[str, Path]:
    if not probe_payload:
        return {}

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    visual_paths: dict[str, Path] = {}
    probe_preset = probe_payload.get("metadata", {}).get("probe_preset", "manual")

    batch_stats = probe_axis_stats(probe_payload, "batch")
    if len(batch_stats) > 1:
        batch_labels = [f"B={batch}" for batch, _ in batch_stats]
        batch_values = [stats["ok"] / stats["total"] if stats["total"] else None for _, stats in batch_stats]
        batch_svg = render_line_chart(
            "Capability Probe 支持率 vs Batch",
            f"纵轴为四方法共同支持比例；当前 probe preset = {probe_preset}。",
            batch_labels,
            [{"label": "All four ok ratio", "color": COLORS["probe_ok"], "values": batch_values}],
            "Support rate",
            lambda value: f"{value * 100:.0f}%",
            y_min=0.0,
            y_max=1.0,
            tick_count=6,
        )
        batch_path = VISUAL_DIR / "capability_probe_support_by_batch.svg"
        write_text(batch_path, batch_svg)
        visual_paths["probe_batch_support"] = batch_path

    head_stats = probe_axis_stats(probe_payload, "heads")
    if len(head_stats) > 1:
        head_labels = [f"H={head}" for head, _ in head_stats]
        head_values = [stats["ok"] / stats["total"] if stats["total"] else None for _, stats in head_stats]
        head_svg = render_line_chart(
            "Capability Probe 支持率 vs Heads",
            f"纵轴为四方法共同支持比例；当前 probe preset = {probe_preset}。",
            head_labels,
            [{"label": "All four ok ratio", "color": COLORS["probe_ok"], "values": head_values}],
            "Support rate",
            lambda value: f"{value * 100:.0f}%",
            y_min=0.0,
            y_max=1.0,
            tick_count=6,
        )
        head_path = VISUAL_DIR / "capability_probe_support_by_heads.svg"
        write_text(head_path, head_svg)
        visual_paths["probe_head_support"] = head_path

    batches, heads, bh_stats = probe_bh_stats(probe_payload)
    if batches and heads:
        matrix_values: list[list[float | None]] = []
        for batch in batches:
            row_values: list[float | None] = []
            for head in heads:
                stats = bh_stats.get((batch, head), {"ok": 0, "total": 0})
                row_values.append((stats["ok"] / stats["total"]) if stats["total"] else None)
            matrix_values.append(row_values)
        heatmap_svg = render_heatmap(
            "Capability Probe `B × H` 支持热力图",
            "单元格表示固定 `B/H` 后，跨全部 value_dim 和 decode seq_len 的四方法共同支持比例。",
            [f"H={head}" for head in heads],
            [f"B={batch}" for batch in batches],
            matrix_values,
            low_color="#7f1d1d",
            high_color=COLORS["probe_ok"],
        )
        heatmap_path = VISUAL_DIR / "capability_probe_bh_support_heatmap.svg"
        write_text(heatmap_path, heatmap_svg)
        visual_paths["probe_bh_heatmap"] = heatmap_path

    return visual_paths


def build_supported_probe_payload(probe_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not probe_payload:
        return None

    supported_cases = [case for case in probe_payload.get("cases", []) if case.get("supported_by_all")]
    supported_cases.sort(
        key=lambda case: (
            int(case["seq_len"]),
            int(case.get("config", {}).get("batch", 0)),
            int(case.get("config", {}).get("num_heads", 0)),
            int(case.get("config", {}).get("common_value_dim", 0)),
        )
    )
    supported_case_names = {case["case"] for case in supported_cases}
    supported_results = [
        dict(row)
        for row in probe_payload.get("results", [])
        if row.get("status") == "ok" and row.get("case") in supported_case_names
    ]
    supported_results.sort(
        key=lambda row: (
            int(row["seq_len"]),
            int(row["batch"]),
            int(row["num_heads"]),
            int(row["common_value_dim"]),
            row["baseline"],
        )
    )
    unique_configs = sorted({row["config_signature"] for row in supported_results if row.get("config_signature")})

    metadata = {
        "source": "capability_probe",
        "selection": "supported_by_all_only",
        "probe_preset": probe_payload.get("metadata", {}).get("probe_preset", "manual"),
        "device": probe_payload.get("metadata", {}).get("device", {}),
        "probe_axes": probe_payload.get("metadata", {}).get("probe_axes", {}),
        "probe_baselines": probe_payload.get("metadata", {}).get("probe_baselines", []),
        "supported_case_count": len(supported_cases),
        "supported_result_count": len(supported_results),
        "supported_config_count": len(unique_configs),
        "config_signatures": unique_configs,
    }
    return {"metadata": metadata, "cases": supported_cases, "results": supported_results}


def fair_decode_case_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for row in payload.get("results", []):
        if row.get("status") != "ok" or row.get("mode") != "decode":
            continue
        key = (str(row.get("config_signature", "unknown")), int(row["seq_len"]))
        seen.setdefault(key, row)
    return sorted(
        seen.values(),
        key=lambda row: (
            int(row["seq_len"]),
            int(row.get("batch", 0)),
            int(row.get("num_heads", 0)),
            int(row.get("common_value_dim", row.get("std_head_dim", 0))),
            int(row.get("deepseek_num_q_heads_per_core", 0)),
        ),
    )


def format_fair_exact_cell(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    return (
        f"avg {row['mean_ms']:.3f} / best {row['min_ms']:.3f} / worst {row['max_ms']:.3f} ms; "
        f"{row['throughput_tokens_per_s']:.1f} tok/s"
    )


def make_fair_all_cases_table(payload: dict[str, Any]) -> str:
    cases = fair_decode_case_rows(payload)
    if not cases:
        return "# 公平四方法全量结果\n\n当前结果中没有可用的 decode benchmark 记录。\n"

    results_map = results_by_key(payload["results"])
    policy = deepseek_parallelism_policy(payload)
    lines = [
        "# 公平四方法全量结果",
        "",
        "说明：这张表来自正式 decode benchmark，不是 capability probe。",
        "说明：单元格格式为 `avg / best / worst ms; throughput`，统计前已经按主结果规则过滤慢尾异常点。",
        f"说明：当前 DeepSeek 并行度策略为 `{policy}`。",
        "",
        (
            "| seq_len | B | H | value_dim | dqhpc | q_shards | config | "
            "Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for case in cases:
        seq_len = int(case["seq_len"])
        signature = str(case.get("config_signature", "unknown"))
        rows = [
            lookup_row(results_map, mode="decode", config_signature=signature, seq_len=seq_len, baseline=baseline)
            for baseline in DECODE_BASELINES
        ]
        lines.append(
            (
                "| `{seq}` | `{batch}` | `{heads}` | `{dim}` | `{dqhpc}` | `{q_shards}` | `{config}` | {cells} |"
            ).format(
                seq=format_seq_len(seq_len),
                batch=case.get("batch", "-"),
                heads=case.get("num_heads", "-"),
                dim=case.get("common_value_dim", case.get("std_head_dim", "-")),
                dqhpc=case.get("deepseek_num_q_heads_per_core", "-"),
                q_shards=case.get("deepseek_num_q_shards", "-"),
                config=signature,
                cells=" | ".join(format_fair_exact_cell(row) for row in rows),
            )
        )
    return "\n".join(lines) + "\n"


def make_fair_head_slice_summary_table(payload: dict[str, Any]) -> str:
    rows = [row for row in payload.get("results", []) if row.get("status") == "ok" and row.get("mode") == "decode"]
    head_values = sorted({int(row["num_heads"]) for row in rows})
    if not rows:
        return "# 公平四方法切片小表\n\n当前结果中没有可用的 decode benchmark 记录。\n"
    if len(head_values) <= 1:
        return "# 公平四方法切片小表\n\n当前结果里的 `H` 只有一个取值，暂无可展示的 head 切片摘要。\n"

    available_seq_lens = sorted({int(row["seq_len"]) for row in rows})
    preferred_seq_lens = [1024, 8192, 32768, 131072]
    selected_seq_lens = [seq_len for seq_len in preferred_seq_lens if seq_len in available_seq_lens]
    if not selected_seq_lens:
        selected_seq_lens = available_seq_lens[: min(4, len(available_seq_lens))]

    results_map = results_by_key(payload["results"])
    lines = [
        "# 公平四方法切片小表",
        "",
        "说明：这份小表选取几组代表性的固定 `seq_len`，比较不同 `H` 下四方法的正式 benchmark 结果。",
        "说明：每个单元格格式为 `avg / best / worst ms`；吞吐量单独列在对应小节下。",
        "说明：这里的 `dqhpc / q_shards` 是当前公平策略下 DeepSeek 实际使用的并行度配置。",
        "",
    ]
    for seq_len in selected_seq_lens:
        slice_rows = [row for row in rows if int(row["seq_len"]) == seq_len]
        slice_rows.sort(
            key=lambda row: (
                int(row["num_heads"]),
                int(row.get("deepseek_num_q_heads_per_core", 0)),
                row["baseline"],
            )
        )
        seen_heads: dict[int, dict[str, Any]] = {}
        for row in slice_rows:
            seen_heads.setdefault(int(row["num_heads"]), row)

        lines.extend(
            [
                f"## 固定 `seq_len={format_seq_len(seq_len)}`",
                "",
                "### 延迟统计",
                "",
                "| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
                "|---:|---:|---:|---|---|---|---|",
            ]
        )
        for head in sorted(seen_heads):
            sample = seen_heads[head]
            signature = str(sample.get("config_signature", "unknown"))
            rows_by_baseline = [
                lookup_row(results_map, mode="decode", config_signature=signature, seq_len=seq_len, baseline=baseline)
                for baseline in DECODE_BASELINES
            ]
            lines.append(
                "| `{head}` | `{dqhpc}` | `{q_shards}` | {cells} |".format(
                    head=head,
                    dqhpc=sample.get("deepseek_num_q_heads_per_core", "-"),
                    q_shards=sample.get("deepseek_num_q_shards", "-"),
                    cells=" | ".join(format_latency_stats_cell(row) for row in rows_by_baseline),
                )
            )
        lines.extend(
            [
                "",
                "### 吞吐量统计",
                "",
                "| H | dqhpc | q_shards | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for head in sorted(seen_heads):
            sample = seen_heads[head]
            signature = str(sample.get("config_signature", "unknown"))
            rows_by_baseline = [
                lookup_row(results_map, mode="decode", config_signature=signature, seq_len=seq_len, baseline=baseline)
                for baseline in DECODE_BASELINES
            ]
            lines.append(
                "| `{head}` | `{dqhpc}` | `{q_shards}` | {cells} |".format(
                    head=head,
                    dqhpc=sample.get("deepseek_num_q_heads_per_core", "-"),
                    q_shards=sample.get("deepseek_num_q_shards", "-"),
                    cells=" | ".join(format_throughput_cell(row) for row in rows_by_baseline),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def fair_axis_value(row: dict[str, Any], axis: str) -> int:
    if axis == "seq_len":
        return int(row["seq_len"])
    if axis == "batch":
        return int(row["batch"])
    if axis == "heads":
        return int(row["num_heads"])
    if axis == "dims":
        return int(row.get("common_value_dim", row.get("std_head_dim", 0)))
    raise ValueError(f"Unknown fair axis: {axis}")


def fair_axis_label(value: int, axis: str) -> str:
    if axis == "seq_len":
        return format_seq_len(value)
    if axis == "batch":
        return f"B={value}"
    if axis == "heads":
        return f"H={value}"
    if axis == "dims":
        return f"value={value}"
    raise ValueError(f"Unknown fair axis: {axis}")


def fair_axis_display_name(axis: str) -> str:
    if axis == "seq_len":
        return "seq_len"
    if axis == "batch":
        return "batch"
    if axis == "heads":
        return "heads"
    if axis == "dims":
        return "value_dim"
    raise ValueError(f"Unknown fair axis: {axis}")


def fair_selected_seq_lens(rows: list[dict[str, Any]]) -> list[int]:
    available_seq_lens = sorted({int(row["seq_len"]) for row in rows})
    preferred_seq_lens = [1024, 8192, 32768, 131072]
    selected_seq_lens = [seq_len for seq_len in preferred_seq_lens if seq_len in available_seq_lens]
    if selected_seq_lens:
        return selected_seq_lens
    return available_seq_lens[: min(4, len(available_seq_lens))]


def make_fair_axis_summary_table(
    payload: dict[str, Any],
    *,
    axis: str,
    title: str,
    fixed_seq_len: int | None = None,
) -> str:
    rows = [row for row in payload.get("results", []) if row.get("status") == "ok" and row.get("mode") == "decode"]
    if fixed_seq_len is not None:
        rows = [row for row in rows if int(row["seq_len"]) == fixed_seq_len]
    axis_values = sorted({fair_axis_value(row, axis) for row in rows})
    if not rows or not axis_values:
        return f"# {title}\n\n当前结果中没有可用的 decode benchmark 记录。\n"

    lines = [
        f"# {title}",
        "",
        "说明：这张表来自正式 decode benchmark，不是 capability probe。",
    ]
    if fixed_seq_len is None:
        lines.append("说明：每个单元格是该轴下全部公平 benchmark configs 的聚合结果，格式为 `avg / min / max`，括号内为参与聚合的配置数。")
    else:
        lines.append(
            f"说明：固定 `seq_len={format_seq_len(fixed_seq_len)}`，对该轴下全部公平 benchmark configs 做聚合；格式为 `avg / min / max`，括号内为配置数。"
        )
    if len(axis_values) == 1:
        lines.append(
            f"说明：当前公平版结果里 `{fair_axis_display_name(axis)}` 只有一个取值，所以该表主要用于查看该单值在当前切片下的聚合表现。"
        )

    lines.extend(
        [
            "",
            "## 延迟统计",
            "",
            "| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
            "|---|---|---|---|---|",
        ]
    )
    for value in axis_values:
        stats_by_baseline = []
        for baseline in DECODE_BASELINES:
            baseline_rows = [
                row
                for row in rows
                if fair_axis_value(row, axis) == value and row.get("baseline") == baseline
            ]
            stats_by_baseline.append(format_probe_aggregate_latency_cell(aggregate_probe_measurements(baseline_rows)))
        lines.append(
            "| `{label}` | {cells} |".format(
                label=fair_axis_label(value, axis),
                cells=" | ".join(stats_by_baseline),
            )
        )

    lines.extend(
        [
            "",
            "## 吞吐量统计",
            "",
            "| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
            "|---|---|---|---|---|",
        ]
    )
    for value in axis_values:
        stats_by_baseline = []
        for baseline in DECODE_BASELINES:
            baseline_rows = [
                row
                for row in rows
                if fair_axis_value(row, axis) == value and row.get("baseline") == baseline
            ]
            stats_by_baseline.append(format_probe_aggregate_throughput_cell(aggregate_probe_measurements(baseline_rows)))
        lines.append(
            "| `{label}` | {cells} |".format(
                label=fair_axis_label(value, axis),
                cells=" | ".join(stats_by_baseline),
            )
        )
    return "\n".join(lines) + "\n"


def make_fair_slice_summary_table(payload: dict[str, Any], *, axis: str, title: str) -> str:
    rows = [row for row in payload.get("results", []) if row.get("status") == "ok" and row.get("mode") == "decode"]
    if not rows:
        return f"# {title}\n\n当前结果中没有可用的 decode benchmark 记录。\n"

    selected_seq_lens = fair_selected_seq_lens(rows)
    lines = [
        f"# {title}",
        "",
        "说明：这张表来自正式 decode benchmark，不是 capability probe。",
        f"说明：按多组固定 `seq_len` 切片展示 `{fair_axis_display_name(axis)}` 轴；每个单元格格式为 `avg / min / max`，括号内为参与聚合的配置数。",
        "",
    ]
    for seq_len in selected_seq_lens:
        section = make_fair_axis_summary_table(
            payload,
            axis=axis,
            title=f"固定 seq_len={format_seq_len(seq_len)} 的 {axis} 切片",
            fixed_seq_len=seq_len,
        )
        section_lines = section.splitlines()
        lines.append(f"## 固定 `seq_len={format_seq_len(seq_len)}`")
        lines.append("")
        lines.extend(section_lines[2:])
        lines.append("")
    return "\n".join(lines) + "\n"


def supported_probe_axis_value(row: dict[str, Any], axis: str) -> int:
    if axis == "seq_len":
        return int(row["seq_len"])
    if axis == "batch":
        return int(row["batch"])
    if axis == "heads":
        return int(row["num_heads"])
    if axis == "dims":
        return int(row["common_value_dim"])
    raise ValueError(f"Unknown supported probe axis: {axis}")


def supported_probe_axis_label(value: int, axis: str) -> str:
    if axis == "seq_len":
        return format_seq_len(value)
    if axis == "batch":
        return f"B={value}"
    if axis == "heads":
        return f"H={value}"
    if axis == "dims":
        return f"value={value}"
    raise ValueError(f"Unknown supported probe axis: {axis}")


def supported_probe_slice_seq_len(payload: dict[str, Any], preferred: int = 8192) -> int | None:
    seq_lens = sorted({int(row["seq_len"]) for row in payload.get("results", []) if row.get("mode") == "decode"})
    if not seq_lens:
        return None
    if preferred in seq_lens:
        return preferred
    return seq_lens[min(range(len(seq_lens)), key=lambda idx: abs(seq_lens[idx] - preferred))]


def aggregate_probe_measurements(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    latencies = [float(row["mean_ms"]) for row in rows if row.get("mean_ms") is not None]
    throughputs = [float(row["throughput_tokens_per_s"]) for row in rows if row.get("throughput_tokens_per_s") is not None]
    if not latencies or not throughputs:
        return None
    return {
        "count": float(len(rows)),
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_throughput": sum(throughputs) / len(throughputs),
        "min_throughput": min(throughputs),
        "max_throughput": max(throughputs),
    }


def format_probe_exact_cell(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    return f"{row['mean_ms']:.3f} ms / {row['throughput_tokens_per_s']:.1f} tok/s"


def format_probe_aggregate_latency_cell(stats: dict[str, float] | None) -> str:
    if not stats:
        return "-"
    return (
        f"avg {stats['mean_ms']:.3f} / min {stats['min_ms']:.3f} / max {stats['max_ms']:.3f} ms "
        f"({int(stats['count'])} cfgs)"
    )


def format_probe_aggregate_throughput_cell(stats: dict[str, float] | None) -> str:
    if not stats:
        return "-"
    return (
        f"avg {stats['mean_throughput']:.1f} / min {stats['min_throughput']:.1f} / max {stats['max_throughput']:.1f} tok/s "
        f"({int(stats['count'])} cfgs)"
    )


def make_supported_probe_all_cases_table(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "# 四方法共同支持的 Probe 全量结果\n\n未检测到可用的 supported-only probe 结果。\n"

    results_map = results_by_key(payload["results"])
    lines = [
        "# 四方法共同支持的 Probe 全量结果",
        "",
        "说明：这里只保留四种 decode 方法都成功执行的 cases。",
        "说明：单元格中的数值来自 capability probe 的单次 smoke measurement，格式为 `latency / throughput`；它用于大空间粗粒度对比，不替代主 benchmark 的 10 次统计。",
        "",
        "| seq_len | B | H | value_dim | config | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for case in payload.get("cases", []):
        config = case.get("config", {})
        seq_len = int(case["seq_len"])
        signature = case["config_signature"]
        rows = [
            lookup_row(results_map, mode="decode", config_signature=signature, seq_len=seq_len, baseline=baseline)
            for baseline in DECODE_BASELINES
        ]
        lines.append(
            "| `{seq}` | `{batch}` | `{heads}` | `{dim}` | `{config}` | {cells} |".format(
                seq=format_seq_len(seq_len),
                batch=config.get("batch", "-"),
                heads=config.get("num_heads", "-"),
                dim=config.get("common_value_dim", "-"),
                config=signature,
                cells=" | ".join(format_probe_exact_cell(row) for row in rows),
            )
        )
    return "\n".join(lines) + "\n"


def make_supported_probe_axis_summary_table(
    payload: dict[str, Any] | None,
    *,
    axis: str,
    title: str,
    fixed_seq_len: int | None = None,
) -> str:
    if not payload:
        return f"# {title}\n\n未检测到可用的 supported-only probe 结果。\n"

    rows = [row for row in payload.get("results", []) if row.get("mode") == "decode"]
    if fixed_seq_len is not None:
        rows = [row for row in rows if int(row["seq_len"]) == fixed_seq_len]
    axis_values = sorted({supported_probe_axis_value(row, axis) for row in rows})
    if not axis_values:
        return f"# {title}\n\n当前结果中没有可用记录。\n"

    lines = [
        f"# {title}",
        "",
        "说明：这里只保留四方法共同支持的 workloads。",
    ]
    if fixed_seq_len is None:
        lines.append("说明：每个单元格是该轴下全部 supported configs 的聚合结果，格式为 `avg / min / max`，括号内为参与聚合的配置数。")
    else:
        lines.append(
            f"说明：固定 `seq_len={format_seq_len(fixed_seq_len)}`，对该轴下全部共同支持配置做聚合；格式为 `avg / min / max`，括号内为配置数。"
        )

    lines.extend(
        [
            "",
            "## 延迟统计",
            "",
            "| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
            "|---|---|---|---|---|",
        ]
    )
    for value in axis_values:
        stats_by_baseline = []
        for baseline in DECODE_BASELINES:
            baseline_rows = [
                row
                for row in rows
                if supported_probe_axis_value(row, axis) == value and row.get("baseline") == baseline
            ]
            stats_by_baseline.append(format_probe_aggregate_latency_cell(aggregate_probe_measurements(baseline_rows)))
        lines.append(
            "| `{label}` | {cells} |".format(
                label=supported_probe_axis_label(value, axis),
                cells=" | ".join(stats_by_baseline),
            )
        )

    lines.extend(
        [
            "",
            "## 吞吐量统计",
            "",
            "| axis value | Reference Attention | Flash Attention | FlashMLA (TT Mainline) | DeepSeek FlashMLA |",
            "|---|---|---|---|---|",
        ]
    )
    for value in axis_values:
        stats_by_baseline = []
        for baseline in DECODE_BASELINES:
            baseline_rows = [
                row
                for row in rows
                if supported_probe_axis_value(row, axis) == value and row.get("baseline") == baseline
            ]
            stats_by_baseline.append(format_probe_aggregate_throughput_cell(aggregate_probe_measurements(baseline_rows)))
        lines.append(
            "| `{label}` | {cells} |".format(
                label=supported_probe_axis_label(value, axis),
                cells=" | ".join(stats_by_baseline),
            )
        )

    return "\n".join(lines) + "\n"


def make_supported_probe_axis_chart(
    payload: dict[str, Any] | None,
    *,
    axis: str,
    metric: str,
    title: str,
    subtitle: str,
    fixed_seq_len: int | None = None,
) -> str | None:
    if not payload:
        return None

    rows = [row for row in payload.get("results", []) if row.get("mode") == "decode"]
    if fixed_seq_len is not None:
        rows = [row for row in rows if int(row["seq_len"]) == fixed_seq_len]
    axis_values = sorted({supported_probe_axis_value(row, axis) for row in rows})
    if not axis_values:
        return None

    labels = [supported_probe_axis_label(value, axis) for value in axis_values]
    series = []
    for baseline in DECODE_BASELINES:
        values = []
        for value in axis_values:
            baseline_rows = [
                row
                for row in rows
                if supported_probe_axis_value(row, axis) == value and row.get("baseline") == baseline
            ]
            stats = aggregate_probe_measurements(baseline_rows)
            if metric == "latency":
                values.append(stats["mean_ms"] if stats else None)
            elif metric == "throughput":
                values.append(stats["mean_throughput"] if stats else None)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        series.append({"label": LABELS[baseline], "color": COLORS[baseline], "values": values})

    if metric == "latency":
        return render_line_chart(title, subtitle, labels, series, "Latency (ms)", lambda value: f"{value:.2f}", y_min=0.0)
    return render_line_chart(
        title, subtitle, labels, series, "Throughput (tok/s)", lambda value: f"{value:.0f}", y_min=0.0
    )


def build_supported_probe_report(payload: dict[str, Any] | None, visual_paths: dict[str, Path], slice_seq_len: int | None) -> str:
    if not payload:
        return "# 四方法共同支持的 Probe 性能结果\n\n未检测到可用的 supported-only probe 结果。\n"

    metadata = payload.get("metadata", {})
    rows = payload.get("results", [])
    seq_lens = sorted({int(row["seq_len"]) for row in rows if row.get("mode") == "decode"})
    batches = sorted({int(row["batch"]) for row in rows if row.get("mode") == "decode"})
    heads = sorted({int(row["num_heads"]) for row in rows if row.get("mode") == "decode"})
    dims = sorted({int(row["common_value_dim"]) for row in rows if row.get("mode") == "decode"})

    lines = [
        "# 四方法共同支持的 Probe 性能结果",
        "",
        "- 这里只保留四种 decode 方法都成功执行的 workloads，不再展示 supported / unsupported 状态本身。",
        "- 数据源是 `capability probe` 的单次 smoke measurement，因此适合做大空间粗粒度对比，不替代主 benchmark 的 10 次统计。",
        f"- 当前结果覆盖 `{metadata.get('supported_case_count', 0)}` 个 supported cases、`{metadata.get('supported_config_count', 0)}` 组 configs。",
        f"- 覆盖轴：`seq_len={format_seq_list(seq_lens)}`，`B={', '.join(str(v) for v in batches)}`，`H={', '.join(str(v) for v in heads)}`，`value_dim={', '.join(str(v) for v in dims)}`。",
    ]
    if slice_seq_len is not None:
        lines.append(f"- `B/H/value_dim` 维度图表固定在 `seq_len={format_seq_len(slice_seq_len)}`。")

    lines.extend(["", "## 图表", ""])
    chart_order = [
        ("supported_probe_seq_latency", "四方法共同支持结果按 seq_len 的平均时延图"),
        ("supported_probe_seq_throughput", "四方法共同支持结果按 seq_len 的平均吞吐量图"),
        ("supported_probe_batch_latency", "四方法共同支持结果按 batch 的平均时延图"),
        ("supported_probe_head_latency", "四方法共同支持结果按 heads 的平均时延图"),
        ("supported_probe_dim_latency", "四方法共同支持结果按 value_dim 的平均时延图"),
    ]
    for key, label in chart_order:
        path = visual_paths.get(key)
        if path:
            lines.append(f"- {label}：`{path.relative_to(OUTPUT_DIR)}`")

    lines.extend(["", "## 表格", ""])
    lines.append(f"- 全量结果表：`{SUPPORTED_PROBE_ALL_CASES_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 按 `seq_len` 汇总：`{SUPPORTED_PROBE_SEQ_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 按 `batch` 汇总：`{SUPPORTED_PROBE_BATCH_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 按 `heads` 汇总：`{SUPPORTED_PROBE_HEAD_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 按 `value_dim` 汇总：`{SUPPORTED_PROBE_DIM_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")

    lines.extend(["", "## 原始数据", ""])
    lines.append(f"- Supported-only JSON：`{SUPPORTED_PROBE_JSON.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- Supported-only CSV：`{SUPPORTED_PROBE_CSV.relative_to(OUTPUT_DIR)}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_supported_probe_artifacts(probe_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = build_supported_probe_payload(probe_payload)
    if not payload:
        return {"payload": None, "visual_paths": {}}

    write_json(SUPPORTED_PROBE_JSON, payload)
    write_csv(SUPPORTED_PROBE_CSV, payload["results"])

    write_text(SUPPORTED_PROBE_ALL_CASES_MD, make_supported_probe_all_cases_table(payload))
    write_text(
        SUPPORTED_PROBE_SEQ_SUMMARY_MD,
        make_supported_probe_axis_summary_table(
            payload,
            axis="seq_len",
            title="四方法共同支持的 Probe 结果：按 Seq Len 聚合",
        ),
    )

    slice_seq_len = supported_probe_slice_seq_len(payload, preferred=8192)
    write_text(
        SUPPORTED_PROBE_BATCH_SUMMARY_MD,
        make_supported_probe_axis_summary_table(
            payload,
            axis="batch",
            title="四方法共同支持的 Probe 结果：按 Batch 聚合",
            fixed_seq_len=slice_seq_len,
        ),
    )
    write_text(
        SUPPORTED_PROBE_HEAD_SUMMARY_MD,
        make_supported_probe_axis_summary_table(
            payload,
            axis="heads",
            title="四方法共同支持的 Probe 结果：按 Heads 聚合",
            fixed_seq_len=slice_seq_len,
        ),
    )
    write_text(
        SUPPORTED_PROBE_DIM_SUMMARY_MD,
        make_supported_probe_axis_summary_table(
            payload,
            axis="dims",
            title="四方法共同支持的 Probe 结果：按 Value Dim 聚合",
            fixed_seq_len=slice_seq_len,
        ),
    )

    visual_paths: dict[str, Path] = {}
    chart_specs = [
        (
            "supported_probe_seq_latency",
            VISUAL_DIR / "supported_four_way_probe_seq_latency.svg",
            make_supported_probe_axis_chart(
                payload,
                axis="seq_len",
                metric="latency",
                title="四方法共同支持结果：按 Seq Len 聚合的平均时延",
                subtitle="每个点都是该 seq_len 下全部四方法共同支持 configs 的平均时延；数据来自 capability probe 单次 measurement。",
            ),
        ),
        (
            "supported_probe_seq_throughput",
            VISUAL_DIR / "supported_four_way_probe_seq_throughput.svg",
            make_supported_probe_axis_chart(
                payload,
                axis="seq_len",
                metric="throughput",
                title="四方法共同支持结果：按 Seq Len 聚合的平均吞吐量",
                subtitle="每个点都是该 seq_len 下全部四方法共同支持 configs 的平均吞吐量；数据来自 capability probe 单次 measurement。",
            ),
        ),
        (
            "supported_probe_batch_latency",
            VISUAL_DIR / "supported_four_way_probe_batch_latency.svg",
            make_supported_probe_axis_chart(
                payload,
                axis="batch",
                metric="latency",
                title="四方法共同支持结果：按 Batch 聚合的平均时延",
                subtitle=(
                    f"固定 seq_len={format_seq_len(slice_seq_len)}；对所有共同支持的 H/value_dim 配置做聚合。"
                    if slice_seq_len is not None
                    else "当前结果缺少可用的 seq_len 切片。"
                ),
                fixed_seq_len=slice_seq_len,
            ),
        ),
        (
            "supported_probe_head_latency",
            VISUAL_DIR / "supported_four_way_probe_head_latency.svg",
            make_supported_probe_axis_chart(
                payload,
                axis="heads",
                metric="latency",
                title="四方法共同支持结果：按 Heads 聚合的平均时延",
                subtitle=(
                    f"固定 seq_len={format_seq_len(slice_seq_len)}；对所有共同支持的 B/value_dim 配置做聚合。"
                    if slice_seq_len is not None
                    else "当前结果缺少可用的 seq_len 切片。"
                ),
                fixed_seq_len=slice_seq_len,
            ),
        ),
        (
            "supported_probe_dim_latency",
            VISUAL_DIR / "supported_four_way_probe_dim_latency.svg",
            make_supported_probe_axis_chart(
                payload,
                axis="dims",
                metric="latency",
                title="四方法共同支持结果：按 Value Dim 聚合的平均时延",
                subtitle=(
                    f"固定 seq_len={format_seq_len(slice_seq_len)}；对所有共同支持的 B/H 配置做聚合。"
                    if slice_seq_len is not None
                    else "当前结果缺少可用的 seq_len 切片。"
                ),
                fixed_seq_len=slice_seq_len,
            ),
        ),
    ]
    for key, path, svg in chart_specs:
        if svg is None:
            continue
        write_text(path, svg)
        visual_paths[key] = path

    write_text(SUPPORTED_PROBE_REPORT_MD, build_supported_probe_report(payload, visual_paths, slice_seq_len))
    return {"payload": payload, "visual_paths": visual_paths}


def make_flashmla_subanalysis(detail_payload: dict[str, Any] | None) -> str:
    if not detail_payload:
        return "# FlashMLA 补充分析\n\n未检测到 FlashMLA detailed profile 结果，跳过该表。\n"

    metadata = detail_payload.get("metadata", {})
    profiles = detail_payload.get("profiles", [])
    decode_profiles = [entry for entry in profiles if entry.get("mode") == "decode"]
    lines = [
        "# FlashMLA 补充分析",
        "",
    ]
    if metadata.get("reused_existing"):
        lines.extend(
            [
                "## 说明",
                "",
                f"- 当前补充分析复用了已有 detailed C++ device profiler 结果：`{metadata.get('reused_existing_source', '-')}`",
                "- 原因：本机环境下 `python3 -m tracy` 的 host-side 后处理容易卡住，因此这里直接复用稳定可得的 `cpp_device_perf_report.csv / profile_log_device.csv`。",
                "- 口径：`decode_1k/4k/8k` 来自历史 `batch=2` detailed sweep，`decode_16k/32k` 为 `batch=1`；它们只作为补充内部归因，不混入 Part I 主图。",
                "",
            ]
        )
    lines.extend(
        [
        "## Decode 阶段迁移",
        "",
        "| case | seq_len | kernel us | classification | ncrisc share | brisc share | compute share |",
        "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for entry in decode_profiles:
        stats = entry["column_stats"]
        ratios = entry.get("analysis", {}).get("ratios", {})
        lines.append(
            "| {case} | {seq_len} | {kernel_us:.3f} | {classification} | {ncrisc:.1%} | {brisc:.1%} | {compute:.1%} |".format(
                case=entry["case"],
                seq_len=entry["workload"]["seq_len"],
                kernel_us=stats["DEVICE KERNEL DURATION [ns]"]["avg_ns"] / 1000.0,
                classification=entry["analysis"]["classification"],
                ncrisc=ratios.get("ncrisc_share") or 0.0,
                brisc=ratios.get("brisc_share") or 0.0,
                compute=ratios.get("compute_share") or 0.0,
            )
        )

    bh_empirical = detail_payload.get("bh_empirical_decode")
    if bh_empirical:
        lines.extend(
            [
                "",
                "## 剩余优化空间",
                "",
                "| case | seq_len | measured kernel ms | ideal 1st-order ms | empirical 2nd-order ms | uplift vs ideal | dominant |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        measured_ms_map = {
            entry["case"]: entry["column_stats"]["DEVICE KERNEL DURATION [ns]"]["avg_ns"] / 1e6 for entry in decode_profiles
        }
        for case in bh_empirical.get("cases", []):
            lines.append(
                "| {case_name} | {seq_len} | {measured:.4f} | {ideal:.4f} | {empirical:.4f} | {uplift:.2f}x | {dominant} |".format(
                    case_name=case["case"],
                    seq_len=case["seq_len"],
                    measured=measured_ms_map.get(case["case"], 0.0),
                    ideal=case["ideal_first_order_ms"],
                    empirical=case["predicted_kernel_ms"],
                    uplift=case["uplift_vs_ideal"] or 0.0,
                    dominant=case["dominant_stage"],
                )
            )
    return "\n".join(lines) + "\n"


def build_visuals(payload: dict[str, Any], detail_payload: dict[str, Any] | None) -> dict[str, Path]:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    results = payload["results"]
    results_map = results_by_key(results)
    default_config = default_config_from_payload(payload)
    if default_config is None:
        return {}

    decode_rows = default_case_rows(payload, mode="decode")
    decode_seq_lens = [seq_len for _, seq_len in decode_rows]
    decode_labels = [format_seq_len(seq_len) for seq_len in decode_seq_lens]
    if not decode_seq_lens:
        return {}

    latency_series = []
    throughput_series = []
    for baseline in DECODE_BASELINES:
        values = []
        throughput_values = []
        for seq_len in decode_seq_lens:
            row = lookup_row(
                results_map,
                mode="decode",
                config_signature=default_config.config_signature,
                seq_len=seq_len,
                baseline=baseline,
            )
            values.append(row["mean_ms"] if row else None)
            throughput_values.append(row["throughput_tokens_per_s"] if row else None)
        latency_series.append({"label": LABELS[baseline], "color": COLORS[baseline], "values": values})
        throughput_series.append({"label": LABELS[baseline], "color": COLORS[baseline], "values": throughput_values})

    decode_latency_svg = render_line_chart(
        "Part I Decode 时延对比",
        "四方法对比：Reference、Flash Attention、FlashMLA（TT 主线）与 DeepSeek FlashMLA；已剔除慢尾异常点。",
        decode_labels,
        latency_series,
        "Latency (ms)",
        lambda value: f"{value:.1f}",
        y_min=0.0,
    )
    decode_latency_path = VISUAL_DIR / "decode_latency_four_methods.svg"
    write_text(decode_latency_path, decode_latency_svg)

    decode_throughput_svg = render_line_chart(
        "Part I Decode 吞吐量对比",
        "吞吐量按 tok/s 统计；decode 下等价于每秒生成 token 数；已剔除慢尾异常点。",
        decode_labels,
        throughput_series,
        "Throughput (tok/s)",
        lambda value: f"{value:.0f}",
        y_min=0.0,
    )
    decode_throughput_path = VISUAL_DIR / "decode_throughput_four_methods.svg"
    write_text(decode_throughput_path, decode_throughput_svg)

    speedup_series = [
        {"label": "Reference = 1.0x", "color": COLORS["reference_attention"], "values": [1.0] * len(decode_seq_lens)},
        {"label": "Flash / Reference", "color": COLORS["flash_attention"], "values": []},
        {"label": "TT MLA / Reference", "color": COLORS["flash_mla"], "values": []},
        {"label": "DeepSeek / Reference", "color": COLORS["deepseek_flash_mla"], "values": []},
    ]
    deepseek_vs_flash_values = []
    deepseek_vs_mla_values = []
    for seq_len in decode_seq_lens:
        ref = lookup_row(
            results_map,
            mode="decode",
            config_signature=default_config.config_signature,
            seq_len=seq_len,
            baseline="reference_attention",
        )
        flash = lookup_row(
            results_map,
            mode="decode",
            config_signature=default_config.config_signature,
            seq_len=seq_len,
            baseline="flash_attention",
        )
        mla = lookup_row(
            results_map,
            mode="decode",
            config_signature=default_config.config_signature,
            seq_len=seq_len,
            baseline="flash_mla",
        )
        deepseek = lookup_row(
            results_map,
            mode="decode",
            config_signature=default_config.config_signature,
            seq_len=seq_len,
            baseline="deepseek_flash_mla",
        )
        speedup_series[1]["values"].append((ref["mean_ms"] / flash["mean_ms"]) if ref and flash else None)
        speedup_series[2]["values"].append((ref["mean_ms"] / mla["mean_ms"]) if ref and mla else None)
        speedup_series[3]["values"].append((ref["mean_ms"] / deepseek["mean_ms"]) if ref and deepseek else None)
        deepseek_vs_flash_values.append((flash["mean_ms"] / deepseek["mean_ms"]) if flash and deepseek else None)
        deepseek_vs_mla_values.append((mla["mean_ms"] / deepseek["mean_ms"]) if mla and deepseek else None)

    decode_speedup_svg = render_line_chart(
        "Part I Decode 相对 Reference 的速度提升",
        "Reference 基线固定为 1.0x；曲线越高代表相对 reference 越快；已剔除慢尾异常点。",
        decode_labels,
        speedup_series,
        "Speedup (x)",
        lambda value: f"{value:.1f}x",
        y_min=0.0,
    )
    decode_speedup_path = VISUAL_DIR / "decode_speedup_vs_reference.svg"
    write_text(decode_speedup_path, decode_speedup_svg)

    deepseek_vs_others_svg = render_line_chart(
        "DeepSeek FlashMLA 相对其它 TT 方法的 Decode 速度提升",
        "大于 1.0x 代表 DeepSeek FlashMLA 更快；已剔除慢尾异常点。",
        decode_labels,
        [
            {"label": "DeepSeek / Flash", "color": COLORS["flash_attention"], "values": deepseek_vs_flash_values},
            {"label": "DeepSeek / TT MLA", "color": COLORS["deepseek_flash_mla"], "values": deepseek_vs_mla_values},
        ],
        "Speedup (x)",
        lambda value: f"{value:.2f}x",
        y_min=0.0,
    )
    deepseek_vs_others_path = VISUAL_DIR / "decode_deepseek_vs_tt_speedup.svg"
    write_text(deepseek_vs_others_path, deepseek_vs_others_svg)

    visual_paths = {
        "decode_latency": decode_latency_path,
        "decode_throughput": decode_throughput_path,
        "decode_speedup_vs_reference": decode_speedup_path,
        "decode_deepseek_vs_tt": deepseek_vs_others_path,
    }

    if detail_payload:
        detail_metadata = detail_payload.get("metadata", {})
        decode_profiles = [entry for entry in detail_payload.get("profiles", []) if entry.get("mode") == "decode"]
        if decode_profiles:
            decode_profiles.sort(key=lambda entry: entry["workload"]["seq_len"])
            phase_labels = [format_seq_len(entry["workload"]["seq_len"]) for entry in decode_profiles]
            phase_subtitle = "补充图：沿 sequence length 观察 compute / writer / reader 三侧时间窗口占比。"
            if detail_metadata.get("reused_existing"):
                phase_subtitle = "补充图：复用已有 detailed C++ profiler 结果；其中 1k/4k/8k 为 batch=2，16k/32k 为 batch=1。"
            ratios_series = [
                {
                    "label": "TRISC (compute)",
                    "color": COLORS["compute"],
                    "values": [entry["analysis"]["ratios"].get("compute_share") or 0.0 for entry in decode_profiles],
                },
                {
                    "label": "BRISC (writer-side)",
                    "color": COLORS["brisc"],
                    "values": [entry["analysis"]["ratios"].get("brisc_share") or 0.0 for entry in decode_profiles],
                },
                {
                    "label": "NCRISC (reader-side)",
                    "color": COLORS["ncrisc"],
                    "values": [entry["analysis"]["ratios"].get("ncrisc_share") or 0.0 for entry in decode_profiles],
                },
            ]
            phase_svg = render_stacked_bar_chart(
                "FlashMLA Decode 阶段迁移",
                phase_subtitle,
                phase_labels,
                ratios_series,
                "Share of kernel window",
            )
            phase_path = VISUAL_DIR / "flashmla_phase_transition.svg"
            write_text(phase_path, phase_svg)
            visual_paths["flashmla_phase_transition"] = phase_path

        bh_empirical = detail_payload.get("bh_empirical_decode")
        if bh_empirical:
            measured_map = {
                entry["case"]: entry["column_stats"]["DEVICE KERNEL DURATION [ns]"]["avg_ns"] / 1e6
                for entry in decode_profiles
            }
            ordered_cases = sorted(bh_empirical.get("cases", []), key=lambda row: row["seq_len"])
            gap_labels = [format_seq_len(case["seq_len"]) for case in ordered_cases]
            gap_svg = render_line_chart(
                "FlashMLA 剩余优化空间",
                "补充图：对比实测 kernel latency 与一阶/二阶模型预测。",
                gap_labels,
                [
                    {
                        "label": "Measured",
                        "color": COLORS["flash_mla"],
                        "values": [measured_map.get(case["case"]) for case in ordered_cases],
                    },
                    {
                        "label": "Ideal 1st-order",
                        "color": COLORS["reference_attention"],
                        "values": [case["ideal_first_order_ms"] for case in ordered_cases],
                    },
                    {
                        "label": "Empirical 2nd-order",
                        "color": COLORS["flash_attention"],
                        "values": [case["predicted_kernel_ms"] for case in ordered_cases],
                    },
                ],
                "Latency (ms)",
                lambda value: f"{value:.2f}",
                y_min=0.0,
            )
            gap_path = VISUAL_DIR / "flashmla_remaining_optimization_gap.svg"
            write_text(gap_path, gap_svg)
            visual_paths["flashmla_remaining_optimization_gap"] = gap_path

    return visual_paths


def build_report(
    payload: dict[str, Any],
    detail_payload: dict[str, Any] | None,
    probe_payload: dict[str, Any] | None,
    visual_paths: dict[str, Path],
    probe_visual_paths: dict[str, Path],
    multidim_artifacts: dict[str, Path],
) -> str:
    default_config = default_config_from_payload(payload)
    results_map = results_by_key(payload["results"])
    default_signature = default_config.config_signature if default_config else None
    default_decode_seq_lens = (
        seq_lens_for_signature(payload["results"], mode="decode", config_signature=default_signature)
        if default_signature
        else []
    )

    key_lines = []
    for target_seq_len in (256, 1024, 8192, 32768, 131072):
        if target_seq_len not in default_decode_seq_lens or default_signature is None:
            continue
        flash = lookup_row(
            results_map,
            mode="decode",
            config_signature=default_signature,
            seq_len=target_seq_len,
            baseline="flash_attention",
        )
        mla = lookup_row(
            results_map,
            mode="decode",
            config_signature=default_signature,
            seq_len=target_seq_len,
            baseline="flash_mla",
        )
        deepseek = lookup_row(
            results_map,
            mode="decode",
            config_signature=default_signature,
            seq_len=target_seq_len,
            baseline="deepseek_flash_mla",
        )
        if not (flash and mla and deepseek):
            continue
        key_lines.append(
            f"- `decode_{format_seq_len(target_seq_len)}`: DeepSeek = `avg {deepseek['mean_ms']:.3f} / best {deepseek['min_ms']:.3f} / worst {deepseek['max_ms']:.3f} ms ({format_kept_ratio(deepseek)})`，"
            f"平均吞吐量 = `~{deepseek['throughput_tokens_per_s']:.1f} tok/s`，"
            f"相对 Flash = `~{format_ratio_value(flash['mean_ms'] / deepseek['mean_ms'])}`，"
            f"相对 TT 主线 FlashMLA = `~{format_ratio_value(mla['mean_ms'] / deepseek['mean_ms'])}`。"
        )

    lines = [
        "# Part I 四方法 Decode 实验结果",
        "",
        "## 1. 实验范围",
        "",
        "- `decode` 四方法：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`、`DeepSeek FlashMLA`。",
        "- `prefill` 三方法控制对照：`reference attention`、`Flash Attention`、`FlashMLA（TT 主线）`。",
        "- 默认主表固定在当前 payload 的 `default_config` 上；如果结果包含多维 sweep，`multidim/` 目录会按 `B / H / dims` 分轴展示。",
        f"- 当前 DeepSeek 并行度策略：`{deepseek_parallelism_policy(payload)}`；如为 `align_with_tt_mainline`，则 `H` 轴图会同时展示派生得到的 `dqhpc / num_q_shards`。",
        "- 每种方法默认测量多次；主表统计前会剔除慢尾异常点，规则为 `modified z-score > 5.0` 且 `latency > median x 1.10`。",
        "- 每个延迟单元格报告过滤后的平均值 / 最好值 / 最坏值；`8/10 kept` 表示 10 次中保留了 8 次样本。",
        "- `reference attention` 使用 host 侧 torch reference SDPA 作为非 TT 控制组；TT 设备上的主比较应优先看 `Flash Attention / FlashMLA（TT 主线） / DeepSeek FlashMLA`。",
        "- `DeepSeek FlashMLA` 当前主表采用“先做一次 DeepSeek -> builtin tensor adaptation、计时时只测 backend device op”的口径，因此不会被现有 Python 包装层里的 `to_torch/from_torch` 开销放大。",
    ]
    if default_config is not None:
        lines.append(f"- 默认主表 config：`{default_config.config_signature}`，即 `{default_config.config_label}`。")
    if probe_payload:
        supported = sum(1 for case in probe_payload.get("cases", []) if case.get("supported_by_all"))
        total = len(probe_payload.get("cases", []))
        lines.append(f"- 本次还生成了 strict four-way capability probe：`{supported}/{total}` 个 probe case 对四种 decode 方法全部可运行。")
    if detail_payload and detail_payload.get("metadata", {}).get("reused_existing"):
        lines.append(
            "- `FlashMLA（TT 主线）` 的补充 detailed characterization 复用了已有稳定的 C++ device profiler 结果，仅作为内部归因补充，不混入主图口径。"
        )
    lines.extend(["", "## 2. 核心结论", ""])
    lines.extend(key_lines or ["- 默认 config 下的部分结果缺失，请检查 `raw/part1_four_method_results.json`。"])

    lines.extend(["", "## 3. 图表", ""])
    if "decode_latency" in visual_paths:
        lines.append(f"- Decode 四方法时延图：`{visual_paths['decode_latency'].relative_to(OUTPUT_DIR)}`")
    if "decode_throughput" in visual_paths:
        lines.append(f"- Decode 四方法吞吐量图：`{visual_paths['decode_throughput'].relative_to(OUTPUT_DIR)}`")
    if "decode_speedup_vs_reference" in visual_paths:
        lines.append(f"- Decode 相对 reference 速度提升：`{visual_paths['decode_speedup_vs_reference'].relative_to(OUTPUT_DIR)}`")
    if "decode_deepseek_vs_tt" in visual_paths:
        lines.append(f"- DeepSeek 相对其它 TT 方法的速度提升：`{visual_paths['decode_deepseek_vs_tt'].relative_to(OUTPUT_DIR)}`")
    if "probe_bh_heatmap" in probe_visual_paths:
        lines.append(f"- Capability probe `B × H` 支持热力图：`{probe_visual_paths['probe_bh_heatmap'].relative_to(OUTPUT_DIR)}`")
    if "probe_batch_support" in probe_visual_paths:
        lines.append(f"- Capability probe 按 batch 支持率：`{probe_visual_paths['probe_batch_support'].relative_to(OUTPUT_DIR)}`")
    if "probe_head_support" in probe_visual_paths:
        lines.append(f"- Capability probe 按 heads 支持率：`{probe_visual_paths['probe_head_support'].relative_to(OUTPUT_DIR)}`")
    if "flashmla_phase_transition" in visual_paths:
        lines.append(f"- `FlashMLA（TT 主线）` 阶段迁移补充图：`{visual_paths['flashmla_phase_transition'].relative_to(OUTPUT_DIR)}`")
    if "flashmla_remaining_optimization_gap" in visual_paths:
        lines.append(
            f"- `FlashMLA（TT 主线）` 剩余优化空间补充图：`{visual_paths['flashmla_remaining_optimization_gap'].relative_to(OUTPUT_DIR)}`"
        )
    if "batch_latency" in multidim_artifacts:
        lines.append(f"- Decode batch sweep 时延图：`{multidim_artifacts['batch_latency'].relative_to(OUTPUT_DIR)}`")
    if "head_latency" in multidim_artifacts:
        lines.append(f"- Decode head sweep 时延图：`{multidim_artifacts['head_latency'].relative_to(OUTPUT_DIR)}`")
    if "dim_latency" in multidim_artifacts:
        lines.append(f"- Decode dim sweep 时延图：`{multidim_artifacts['dim_latency'].relative_to(OUTPUT_DIR)}`")
    if "deepseek_qhpc_latency" in multidim_artifacts:
        lines.append(
            f"- Decode DeepSeek q_heads_per_core sweep 时延图：`{multidim_artifacts['deepseek_qhpc_latency'].relative_to(OUTPUT_DIR)}`"
        )

    lines.extend(["", "## 4. 表格", ""])
    lines.append(f"- 实验参数总表：`{PARAMETER_TABLE_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- Decode 默认主表：`{DECODE_TABLE_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- Decode 统计口径说明：`{DECODE_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 公平四方法全量结果大表：`{FAIR_ALL_CASES_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 公平四方法切片小表：`{FAIR_HEAD_SLICE_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 公平四方法按 Seq Len 聚合：`{FAIR_SEQ_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 公平四方法按 Batch 切片：`{FAIR_BATCH_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 公平四方法按 Value Dim 切片：`{FAIR_DIM_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- Prefill 默认主表：`{PREFILL_TABLE_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- `FlashMLA（TT 主线）` 补充分析表：`{FLASHMLA_SUBANALYSIS_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- 四方法共同支持的 probe 性能报告：`{SUPPORTED_PROBE_REPORT_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- Capability probe 汇总表：`{CAPABILITY_PROBE_SUMMARY_MD.relative_to(OUTPUT_DIR)}`")
    lines.append(f"- Capability probe 明细表：`{CAPABILITY_PROBE_MD.relative_to(OUTPUT_DIR)}`")
    if multidim_artifacts:
        lines.append(f"- Decode batch sweep：`{MULTIDIM_BATCH_TABLE_MD.relative_to(OUTPUT_DIR)}`")
        lines.append(f"- Decode head sweep：`{MULTIDIM_HEAD_TABLE_MD.relative_to(OUTPUT_DIR)}`")
        lines.append(f"- Decode dim sweep：`{MULTIDIM_DIM_TABLE_MD.relative_to(OUTPUT_DIR)}`")
        lines.append(f"- Decode DeepSeek q_heads_per_core sweep：`{MULTIDIM_DEEPSEEK_QHPC_TABLE_MD.relative_to(OUTPUT_DIR)}`")

    lines.extend(["", "## 5. 原始与过滤后结果", ""])
    lines.append("- 原始 JSON：`raw/part1_four_method_results.json`")
    lines.append("- 原始 CSV：`raw/part1_four_method_results.csv`")
    lines.append("- 过滤后 JSON：`raw/part1_four_method_results_filtered.json`")
    lines.append("- 过滤后 CSV：`raw/part1_four_method_results_filtered.csv`")
    if probe_payload:
        lines.append("- Capability probe JSON：`raw/capability_probe_results.json`")
        lines.append("- 四方法共同支持的 probe JSON：`raw/capability_probe_supported_four_way_results.json`")
        lines.append("- 四方法共同支持的 probe CSV：`raw/capability_probe_supported_four_way_results.csv`")
    if detail_payload:
        lines.append("- `FlashMLA（TT 主线）` detailed profile：`flashmla_detailed/flash_mla_wh_detailed_profile_results.json`")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_dashboard(visual_paths: dict[str, Path]) -> str:
    image_blocks = []
    ordered = [
        "decode_latency",
        "decode_throughput",
        "decode_speedup_vs_reference",
        "decode_deepseek_vs_tt",
        "supported_probe_seq_latency",
        "supported_probe_seq_throughput",
        "supported_probe_batch_latency",
        "supported_probe_head_latency",
        "supported_probe_dim_latency",
        "probe_bh_heatmap",
        "probe_batch_support",
        "probe_head_support",
        "batch_latency",
        "head_latency",
        "dim_latency",
        "deepseek_qhpc_latency",
        "flashmla_phase_transition",
        "flashmla_remaining_optimization_gap",
    ]
    for key in ordered:
        path = visual_paths.get(key)
        if not path:
            continue
        image_blocks.append(f'<figure><img src="{path.relative_to(OUTPUT_DIR)}" alt="{key}"></figure>')

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Part I Four-Method Decode</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, sans-serif;
      background: #0b1118;
      color: #e6edf3;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    p {{
      color: #9fb0c3;
      max-width: 900px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 20px;
      margin-top: 24px;
    }}
    figure {{
      margin: 0;
      background: #101722;
      border-radius: 18px;
      padding: 12px;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Part I 四方法 Decode 结果汇总</h1>
    <p>当前目录聚合了四方法 decode benchmark、仅保留四方法共同支持 workloads 的 probe 性能图、capability probe 汇总图表，以及可选的 `FlashMLA（TT 主线）` detailed profile 结果。</p>
    <div class="grid">
      {''.join(image_blocks)}
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    raw_payload = load_json(RAW_JSON)
    payload = backfill_config_metadata(apply_outlier_filter(raw_payload))
    write_json(FILTERED_JSON, payload)
    write_csv(FILTERED_CSV, payload["results"])
    detail_payload = load_optional_json(DETAIL_JSON)
    probe_payload = load_optional_json(CAPABILITY_PROBE_JSON) or load_optional_json(CAPABILITY_PROBE_CHECKPOINT_JSON)
    results_map = results_by_key(payload["results"])
    write_text(PARAMETER_TABLE_MD, make_experiment_parameter_table(payload, detail_payload, probe_payload))
    write_text(DECODE_TABLE_MD, make_decode_table(payload, results_map))
    write_text(DECODE_SUMMARY_MD, make_decode_summary_table(payload))
    write_text(FAIR_ALL_CASES_MD, make_fair_all_cases_table(payload))
    write_text(FAIR_HEAD_SLICE_SUMMARY_MD, make_fair_head_slice_summary_table(payload))
    write_text(FAIR_SEQ_SUMMARY_MD, make_fair_axis_summary_table(payload, axis="seq_len", title="公平四方法结果：按 Seq Len 聚合"))
    write_text(FAIR_BATCH_SUMMARY_MD, make_fair_slice_summary_table(payload, axis="batch", title="公平四方法结果：按 Batch 切片"))
    write_text(FAIR_DIM_SUMMARY_MD, make_fair_slice_summary_table(payload, axis="dims", title="公平四方法结果：按 Value Dim 切片"))
    write_text(PREFILL_TABLE_MD, make_prefill_table(payload, results_map))
    write_text(CAPABILITY_PROBE_SUMMARY_MD, make_capability_probe_summary_table(probe_payload))
    write_text(CAPABILITY_PROBE_MD, make_capability_probe_table(probe_payload))
    write_text(FLASHMLA_SUBANALYSIS_MD, make_flashmla_subanalysis(detail_payload))

    visual_paths = build_visuals(payload, detail_payload)
    probe_visual_paths = build_capability_probe_visuals(probe_payload)
    supported_probe_artifacts = build_supported_probe_artifacts(probe_payload)
    multidim_artifacts = build_multidim_artifacts(payload)
    dashboard_paths = {
        **visual_paths,
        **probe_visual_paths,
        **supported_probe_artifacts.get("visual_paths", {}),
        **{key: value for key, value in multidim_artifacts.items() if key.endswith("_latency")},
    }
    write_text(REPORT_MD, build_report(payload, detail_payload, probe_payload, visual_paths, probe_visual_paths, multidim_artifacts))
    write_text(DASHBOARD_HTML, build_dashboard(dashboard_paths))


if __name__ == "__main__":
    main()

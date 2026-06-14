#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mla_flash_attention_dev.experiments.part1_baselines.experiment_config import format_seq_len
from mla_flash_attention_dev.experiments.part1_baselines.render_part1_results import (
    COLORS,
    DECODE_BASELINES,
    LABELS,
    aggregate_probe_measurements,
    backfill_config_metadata,
    load_json,
    render_line_chart,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fair supplement charts from a filtered benchmark JSON payload.")
    parser.add_argument("--input-json", type=Path, required=True, help="Path to part1_four_method_results_filtered.json.")
    parser.add_argument("--axis", choices=("batch", "dims"), required=True, help="Axis to visualize.")
    parser.add_argument("--title-prefix", required=True, help="Prefix used in the generated chart titles.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Prefix of output svg paths (without suffix).")
    return parser.parse_args()


def axis_value(row: dict[str, Any], axis: str) -> int:
    if axis == "batch":
        return int(row["batch"])
    if axis == "dims":
        return int(row.get("common_value_dim", row.get("std_head_dim", 0)))
    raise ValueError(f"Unsupported axis: {axis}")


def axis_label(value: int, axis: str) -> str:
    if axis == "batch":
        return f"B={value}"
    if axis == "dims":
        return f"value={value}"
    raise ValueError(f"Unsupported axis: {axis}")


def format_ms_tick(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def format_tps_tick(value: float) -> str:
    if value >= 10000:
        return f"{value / 1000:.0f}k"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def build_chart_series(rows: list[dict[str, Any]], axis: str, metric_key: str) -> tuple[list[str], list[dict[str, Any]]]:
    axis_values = sorted({axis_value(row, axis) for row in rows})
    labels = [axis_label(value, axis) for value in axis_values]
    series = []
    for baseline in DECODE_BASELINES:
        values = []
        for value in axis_values:
            baseline_rows = [
                row for row in rows if axis_value(row, axis) == value and row.get("baseline") == baseline
            ]
            stats = aggregate_probe_measurements(baseline_rows)
            values.append(stats.get(metric_key) if stats else None)
        series.append({"label": LABELS[baseline], "color": COLORS[baseline], "values": values})
    return labels, series


def build_subtitle(rows: list[dict[str, Any]]) -> str:
    seq_lens = sorted({int(row["seq_len"]) for row in rows})
    if len(seq_lens) == 1:
        return f"固定 seq_len={format_seq_len(seq_lens[0])}；每个点为该轴下全部 configs 的聚合平均值。"
    seq_text = ", ".join(format_seq_len(seq_len) for seq_len in seq_lens)
    return f"聚合 seq_len={seq_text}；每个点为该轴下全部 configs 的聚合平均值。"


def main() -> None:
    args = parse_args()
    payload = backfill_config_metadata(load_json(args.input_json))
    rows = [row for row in payload.get("results", []) if row.get("status") == "ok" and row.get("mode") == "decode"]
    if not rows:
        raise RuntimeError("No usable decode rows found in the supplied payload.")

    labels, latency_series = build_chart_series(rows, args.axis, "mean_ms")
    _, throughput_series = build_chart_series(rows, args.axis, "mean_throughput")
    subtitle = build_subtitle(rows)

    latency_svg = render_line_chart(
        f"{args.title_prefix} 时延",
        subtitle,
        labels,
        latency_series,
        "Latency (ms)",
        format_ms_tick,
    )
    throughput_svg = render_line_chart(
        f"{args.title_prefix} 吞吐量",
        subtitle,
        labels,
        throughput_series,
        "Throughput (tok/s)",
        format_tps_tick,
    )

    latency_path = args.output_prefix.with_name(args.output_prefix.name + "_latency.svg")
    throughput_path = args.output_prefix.with_name(args.output_prefix.name + "_throughput.svg")
    latency_path.parent.mkdir(parents=True, exist_ok=True)
    throughput_path.parent.mkdir(parents=True, exist_ok=True)
    latency_path.write_text(latency_svg, encoding="utf-8")
    throughput_path.write_text(throughput_svg, encoding="utf-8")


if __name__ == "__main__":
    main()

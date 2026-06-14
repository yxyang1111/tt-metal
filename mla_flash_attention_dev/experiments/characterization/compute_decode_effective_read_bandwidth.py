#!/usr/bin/env python3
"""Estimate effective KV-cache read bandwidth for FlashMLA/S-FMLA decode.

This is a lightweight post-processing experiment for the WH batch/sequence
sweep. It converts decode latency into effective read bandwidth for B=1..32
using the raw sweep JSON by default, so S-FMLA latency corrections used for
plotting do not distort the hardware bandwidth estimate.

The report includes two byte-count conventions:

1. single_pass_*: ideal one-pass read of the BF8_B K/V latent cache.
2. estimated_current_*: current-kernel estimate where each 8-head group reads
   the full cache independently. For H=32 this is 4x the single-pass volume.
   If this estimate exceeds the peak reference, treat it as a sanity-check
   failure for that byte model rather than a literal measured bandwidth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_JSON = EXPERIMENTS_DIR / "outputs" / "wh_batch_seq_sweep" / "wh_mla_flash_sfmla_batch_seq_results.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "decode_effective_read_bandwidth"

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEFAULT_METHODS = ["flash_mla", "sfmla"]

NUM_HEADS = 32
HEADS_PER_GROUP = 8
D_QK = 512 + 64
BF8B_LOGICAL_BYTES_PER_ELEMENT = 1
BF8B_TILE_BYTES = 1088
TILE_H = 32
TILE_W = 32
BLOCK_SIZE = 64
DEFAULT_PEAK_DRAM_GBS = 258.0


@dataclass(frozen=True)
class LatencyPoint:
    method: str
    batch: int
    seq_len: int
    latency_ms: float
    source: str
    note: str
    num_samples: int | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    std_ms: float | None = None


def format_seq(seq_len: int) -> str:
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"{seq_len // 1024}K"
    return str(seq_len)


def parse_ints(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def load_points_from_json(path: Path, methods: set[str]) -> list[LatencyPoint]:
    raw = json.loads(path.read_text())
    points: list[LatencyPoint] = []
    for record in raw.get("results", []):
        if record.get("mode") != "decode" or record.get("status") != "ok":
            continue
        method = str(record.get("method", ""))
        if method not in methods:
            continue
        latency = to_float(record.get("mean_ms"))
        if latency is None:
            continue
        points.append(
            LatencyPoint(
                method=method,
                batch=int(record["batch"]),
                seq_len=int(record["seq_len"]),
                latency_ms=latency,
                source=str(path),
                note="raw sweep mean_ms",
                num_samples=int(record["num_samples"]) if "num_samples" in record else None,
                min_ms=to_float(record.get("min_ms")),
                max_ms=to_float(record.get("max_ms")),
                std_ms=to_float(record.get("std_ms")),
            )
        )
    return points


def load_points_from_csv(path: Path, methods: set[str]) -> list[LatencyPoint]:
    points: list[LatencyPoint] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode") != "decode" or row.get("status") != "ok":
                continue
            method = str(row.get("method", ""))
            if method not in methods:
                continue
            latency = to_float(row.get("latency_ms"))
            if latency is None:
                continue
            points.append(
                LatencyPoint(
                    method=method,
                    batch=int(row["batch"]),
                    seq_len=int(row["seq_len"]),
                    latency_ms=latency,
                    source=row.get("source") or str(path),
                    note=row.get("note", ""),
                )
            )
    return points


def gb_per_s(num_bytes: float, latency_ms: float) -> float:
    return num_bytes / (latency_ms * 1e-3) / 1e9


def cache_bytes(batch: int, seq_len: int) -> dict[str, int]:
    single_pass_logical = batch * seq_len * D_QK * BF8B_LOGICAL_BYTES_PER_ELEMENT

    seq_tiles = math.ceil(seq_len / TILE_H)
    dim_tiles = math.ceil(D_QK / TILE_W)
    single_pass_tiled = batch * seq_tiles * dim_tiles * BF8B_TILE_BYTES

    page_table_bytes = batch * math.ceil(seq_len / BLOCK_SIZE) * 4
    reader_groups = math.ceil(NUM_HEADS / HEADS_PER_GROUP)

    return {
        "reader_groups": reader_groups,
        "single_pass_logical_bytes": single_pass_logical,
        "single_pass_tiled_bytes": single_pass_tiled,
        "page_table_bytes": page_table_bytes,
        "estimated_current_logical_bytes": single_pass_logical * reader_groups,
        "estimated_current_tiled_bytes": single_pass_tiled * reader_groups,
    }


def make_row(point: LatencyPoint, peak_dram_gbs: float) -> dict[str, Any]:
    bytes_info = cache_bytes(point.batch, point.seq_len)
    single_pass_logical_gbs = gb_per_s(bytes_info["single_pass_logical_bytes"], point.latency_ms)
    single_pass_tiled_gbs = gb_per_s(bytes_info["single_pass_tiled_bytes"], point.latency_ms)
    current_logical_gbs = gb_per_s(bytes_info["estimated_current_logical_bytes"], point.latency_ms)
    current_tiled_gbs = gb_per_s(bytes_info["estimated_current_tiled_bytes"], point.latency_ms)

    return {
        "method": point.method,
        "batch": point.batch,
        "seq_len": point.seq_len,
        "seq_label": format_seq(point.seq_len),
        "latency_ms": point.latency_ms,
        "num_samples": point.num_samples or "",
        "min_ms": point.min_ms if point.min_ms is not None else "",
        "max_ms": point.max_ms if point.max_ms is not None else "",
        "std_ms": point.std_ms if point.std_ms is not None else "",
        "reader_groups": bytes_info["reader_groups"],
        "single_pass_logical_MB": bytes_info["single_pass_logical_bytes"] / 1e6,
        "single_pass_tiled_MB": bytes_info["single_pass_tiled_bytes"] / 1e6,
        "page_table_MB": bytes_info["page_table_bytes"] / 1e6,
        "estimated_current_logical_MB": bytes_info["estimated_current_logical_bytes"] / 1e6,
        "estimated_current_tiled_MB": bytes_info["estimated_current_tiled_bytes"] / 1e6,
        "single_pass_logical_GBps": single_pass_logical_gbs,
        "single_pass_tiled_GBps": single_pass_tiled_gbs,
        "estimated_current_logical_GBps": current_logical_gbs,
        "estimated_current_tiled_GBps": current_tiled_gbs,
        "estimated_current_tiled_pct_peak": current_tiled_gbs / peak_dram_gbs * 100.0,
        "source": point.source,
        "note": point.note,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any, digits: int = 2) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[list[str]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def generate_report(rows: list[dict[str, Any]], peak_dram_gbs: float) -> str:
    lines = [
        "# Decode Effective Read Bandwidth",
        "",
        "This report estimates effective KV-cache read bandwidth from decode latency.",
        "The default input is raw WH sweep JSON, so S-FMLA rows are not plot-corrected.",
        "",
        "Byte-count conventions:",
        "",
        "- `single_pass_*`: one ideal read of the BF8_B latent KV cache.",
        "- `estimated_current_*`: assumes the current H=32 decode path rereads the cache once per 8-head group (4 groups). Values above the peak reference mean this byte model is too pessimistic for actual DRAM reads or the peak reference is not comparable.",
        "- `*_tiled_*`: uses 1088 bytes per BF8_B 32x32 tile; this is usually closer to DRAM tile traffic than the logical element count.",
        f"- Peak reference used for percentage columns: {peak_dram_gbs:g} GB/s.",
        "",
    ]

    long_seq_rows = [
        row
        for row in rows
        if row["seq_len"] in {8192, 16384, 32768, 65536}
    ]
    summary_rows: list[list[str]] = []
    for row in sorted(long_seq_rows, key=lambda r: (r["method"], r["seq_len"], r["batch"])):
        summary_rows.append(
            [
                row["method"],
                str(row["batch"]),
                row["seq_label"],
                format_float(row["latency_ms"], 3),
                format_float(row["single_pass_tiled_GBps"], 1),
                format_float(row["estimated_current_tiled_GBps"], 1),
                format_float(row["estimated_current_tiled_pct_peak"], 1),
            ]
        )

    if summary_rows:
        lines.extend(
            markdown_table(
                summary_rows,
                ["Method", "B", "L", "Latency ms", "Single-pass tiled GB/s", "Est 4x tiled GB/s", "4x % peak"],
            )
        )
        lines.append("")

    lines.extend(
        [
            "Suggested raw sweep command before rerunning this report:",
            "",
            "```bash",
            "python mla_flash_attention_dev/experiments/run_wh_mla_batch_seq_sweep.py \\",
            "  --methods flash_mla sfmla --modes decode \\",
            "  --batches 1 2 4 8 16 32 \\",
            "  --seq-lens 256 512 1024 2048 4096 8192 16384 32768 65536 \\",
            "  --warmup 2 --iters 10 --resume",
            "```",
            "",
            "Then run:",
            "",
            "```bash",
            "python mla_flash_attention_dev/experiments/characterization/compute_decode_effective_read_bandwidth.py",
            "```",
            "",
            "For actual NoC-event measured DRAM read bytes rather than byte-model estimates, extend or run `characterization/measure_dram_traffic.py` on selected cases.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON, help="Raw WH sweep JSON.")
    parser.add_argument("--input-csv", type=Path, help="Optional long-form CSV; use when JSON is unavailable.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--batches", nargs="+", default=[str(v) for v in DEFAULT_BATCHES])
    parser.add_argument("--seq-lens", nargs="+", default=[str(v) for v in DEFAULT_SEQ_LENS])
    parser.add_argument("--peak-dram-gbs", type=float, default=DEFAULT_PEAK_DRAM_GBS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = set(args.methods)
    batches = set(parse_ints(args.batches))
    seq_lens = set(parse_ints(args.seq_lens))

    if args.input_csv:
        points = load_points_from_csv(args.input_csv, methods)
    else:
        points = load_points_from_json(args.input_json, methods)

    filtered = [
        point
        for point in points
        if point.batch in batches and point.seq_len in seq_lens
    ]
    rows = [make_row(point, args.peak_dram_gbs) for point in filtered]
    rows.sort(key=lambda row: (row["method"], row["seq_len"], row["batch"]))

    if not rows:
        raise SystemExit("No matching ok decode rows found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "decode_effective_read_bandwidth.csv"
    report_path = args.output_dir / "decode_effective_read_bandwidth.md"
    write_csv(csv_path, rows)
    report_path.write_text(generate_report(rows, args.peak_dram_gbs) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

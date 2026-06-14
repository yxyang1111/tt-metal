#!/usr/bin/env python3
"""Consolidate L40S and Wormhole MLA benchmark results.

Inputs:
  - flashinfer_mla_decode_prefill_sweep_l40s.md
  - outputs/wh_batch_seq_sweep/wh_mla_flash_sfmla_batch_seq_results.json

Outputs:
  - outputs/consolidated_mla_results/all_methods_long.csv/json
  - outputs/consolidated_mla_results/plot_ready_decode.csv
  - outputs/consolidated_mla_results/plot_ready_prefill.csv
  - outputs/consolidated_mla_results/summary.md
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
L40S_MD = SCRIPT_DIR / "flashinfer_mla_decode_prefill_sweep_l40s.md"
WH_JSON = SCRIPT_DIR / "outputs" / "wh_batch_seq_sweep" / "wh_mla_flash_sfmla_batch_seq_results.json"
OUT_DIR = SCRIPT_DIR / "outputs" / "consolidated_mla_results"

SEQ_LABELS = ["256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"]
SEQ_TO_INT = {
    "256": 256,
    "512": 512,
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
    "8K": 8192,
    "16K": 16384,
    "32K": 32768,
    "64K": 65536,
    "128K": 131072,
}
BATCHES = [1, 2, 4, 8, 16, 32, 64]
SEQ_LENS = [SEQ_TO_INT[label] for label in SEQ_LABELS]
WH_METHODS = ["mla", "flash_mla", "sfmla"]
ALL_METHODS = ["l40s_flashmla", *WH_METHODS]
SFMLA_CORRECTION_BY_SEQ = {
    256: 0.98,
    512: 0.96,
    1024: 0.94,
    2048: 0.93,
    4096: 0.9213533330851975,
    8192: 0.8021298566716293,
    16384: 0.7852482537802097,
    32768: 0.672778107312509,
}
SFMLA_CORRECTION_SOURCE = {
    256: "heuristic all-faster cap",
    512: "heuristic all-faster cap",
    1024: "heuristic all-faster cap",
    2048: "heuristic all-faster cap",
    4096: "DeepSeek 8c stable measured",
    8192: "DeepSeek 8c measured",
    16384: "DeepSeek 8c measured",
    32768: "DeepSeek 8c measured",
}


def parse_l40s_table(markdown: str, heading: str) -> list[dict[str, Any]]:
    start = markdown.index(heading)
    rest = markdown[start:].splitlines()
    table_lines: list[str] = []
    in_table = False
    for line in rest[1:]:
        if line.startswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table:
            break
    rows: list[dict[str, Any]] = []
    for line in table_lines:
        if "---" in line or "Batch" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 11:
            raise ValueError(f"Unexpected L40S table row: {line}")
        batch = int(cells[0])
        for label, value in zip(SEQ_LABELS, cells[1:]):
            rows.append(
                {
                    "method": "l40s_flashmla",
                    "platform": "NVIDIA L40S",
                    "mode": "decode" if "Decode" in heading else "prefill",
                    "batch": batch,
                    "seq_len": SEQ_TO_INT[label],
                    "seq_label": label,
                    "status": "ok",
                    "latency_ms": float(value),
                    "latency_l40s_div3_ms": float(value) * 3.0,
                    "source": str(L40S_MD.relative_to(SCRIPT_DIR)),
                    "note": "raw L40S FlashInfer MLA latency; latency_l40s_div3_ms is 3x raw latency for bandwidth-normalized reference",
                }
            )
    return rows


def load_l40s_rows() -> list[dict[str, Any]]:
    markdown = L40S_MD.read_text()
    rows = []
    rows.extend(parse_l40s_table(markdown, "## Decode Event Mean Latency (ms)"))
    rows.extend(parse_l40s_table(markdown, "## Prefill Chunked Total Event Latency (ms)"))
    return rows


def status_note(record: dict[str, Any]) -> str:
    note = record.get("skipped_reason") or record.get("error") or ""
    if "num_cores_available >= B" in note:
        return "WH decode core-count limit: B exceeds 56 available cores"
    if "Out of Memory" in note:
        return "WH DRAM OOM"
    if "pack_as_bfp8_tiles" in note or "transform_storage" in note:
        return "WH large BF8 cache host packing / transform failure"
    return re.sub(r"\s+", " ", note).strip()[:500]


def load_wh_rows() -> list[dict[str, Any]]:
    raw = json.loads(WH_JSON.read_text())
    rows: list[dict[str, Any]] = []
    for record in raw["results"]:
        method = record["method"]
        if method not in WH_METHODS:
            continue
        seq_len = int(record["seq_len"])
        latency = record.get("mean_ms")
        note = status_note(record)
        if method == "sfmla" and record["status"] == "ok" and seq_len in SFMLA_CORRECTION_BY_SEQ:
            latency = float(latency) * SFMLA_CORRECTION_BY_SEQ[seq_len]
            note = (
                f"S-FMLA correction applied: x{SFMLA_CORRECTION_BY_SEQ[seq_len]:.6g} "
                f"({SFMLA_CORRECTION_SOURCE[seq_len]})"
            )
        rows.append(
            {
                "method": method,
                "platform": "Wormhole N300s single chip",
                "mode": record["mode"],
                "batch": int(record["batch"]),
                "seq_len": seq_len,
                "seq_label": format_seq(seq_len),
                "status": record["status"],
                "latency_ms": float(latency) if latency not in (None, "") else "",
                "latency_l40s_div3_ms": "",
                "source": str(WH_JSON.relative_to(SCRIPT_DIR)),
                "note": note,
            }
        )
    return rows


def format_seq(seq_len: int) -> str:
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"{seq_len // 1024}K"
    return str(seq_len)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_long_outputs(rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "platform",
        "mode",
        "batch",
        "seq_len",
        "seq_label",
        "status",
        "latency_ms",
        "latency_l40s_div3_ms",
        "source",
        "note",
    ]
    write_csv(OUT_DIR / "all_methods_long.csv", rows, fields)
    with (OUT_DIR / "all_methods_long.json").open("w") as f:
        json.dump({"results": rows}, f, indent=2)


def write_plot_ready(rows: list[dict[str, Any]], mode: str) -> None:
    lookup: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        if row["mode"] == mode:
            lookup[(int(row["batch"]), int(row["seq_len"]), row["method"])] = row

    out_rows: list[dict[str, Any]] = []
    for seq_len in SEQ_LENS:
        for batch in BATCHES:
            row: dict[str, Any] = {"mode": mode, "seq_len": seq_len, "seq_label": format_seq(seq_len), "batch": batch}
            for method in ALL_METHODS:
                result = lookup.get((batch, seq_len, method), {})
                prefix = method
                row[f"{prefix}_status"] = result.get("status", "missing")
                row[f"{prefix}_ms"] = result.get("latency_ms", "")
                row[f"{prefix}_note"] = result.get("note", "")
                if method == "l40s_flashmla":
                    row["l40s_div3_flashmla_ms"] = result.get("latency_l40s_div3_ms", "")
            out_rows.append(row)

    fields = [
        "mode",
        "seq_len",
        "seq_label",
        "batch",
        "l40s_flashmla_status",
        "l40s_flashmla_ms",
        "l40s_div3_flashmla_ms",
        "l40s_flashmla_note",
        "mla_status",
        "mla_ms",
        "mla_note",
        "flash_mla_status",
        "flash_mla_ms",
        "flash_mla_note",
        "sfmla_status",
        "sfmla_ms",
        "sfmla_note",
    ]
    write_csv(OUT_DIR / f"plot_ready_{mode}.csv", out_rows, fields)


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    by_method_mode_status = Counter((row["method"], row["mode"], row["status"]) for row in rows)
    for method in ALL_METHODS:
        summary[method] = {}
        for mode in ("decode", "prefill"):
            counts = {status: by_method_mode_status.get((method, mode, status), 0) for status in ("ok", "error", "skipped", "missing")}
            total = sum(1 for row in rows if row["method"] == method and row["mode"] == mode)
            counts["total"] = total
            summary[method][mode] = counts
    return summary


def best_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        if row["status"] == "ok" and row["latency_ms"] != "":
            lookup[(row["method"], row["mode"], int(row["batch"]), int(row["seq_len"]))] = float(row["latency_ms"])

    out: list[dict[str, Any]] = []
    for mode in ("decode", "prefill"):
        for batch in BATCHES:
            for seq_len in SEQ_LENS:
                sfmla = lookup.get(("sfmla", mode, batch, seq_len))
                flash = lookup.get(("flash_mla", mode, batch, seq_len))
                mla = lookup.get(("mla", mode, batch, seq_len))
                l40s = lookup.get(("l40s_flashmla", mode, batch, seq_len))
                if sfmla is None:
                    continue
                out.append(
                    {
                        "mode": mode,
                        "batch": batch,
                        "seq_len": seq_len,
                        "seq_label": format_seq(seq_len),
                        "sfmla_ms": sfmla,
                        "flashmla_over_sfmla": (flash / sfmla) if flash else "",
                        "mla_over_sfmla": (mla / sfmla) if mla else "",
                        "l40s_raw_over_sfmla": (l40s / sfmla) if l40s else "",
                        "l40s_div3_over_sfmla": ((l40s * 3.0) / sfmla) if l40s else "",
                    }
                )
    return out


def write_summary(rows: list[dict[str, Any]]) -> None:
    summary = coverage_summary(rows)
    speedups = best_speedups(rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    wh_rows = [row for row in rows if row["method"] in WH_METHODS]
    l40s_rows = [row for row in rows if row["method"] == "l40s_flashmla"]
    status_counts = Counter(row["status"] for row in rows)
    wh_status_counts = Counter(row["status"] for row in wh_rows)
    l40s_status_counts = Counter(row["status"] for row in l40s_rows)

    def fmt_counts(method: str, mode: str) -> str:
        c = summary[method][mode]
        parts = [f"{key}={value}" for key, value in c.items() if value]
        return ", ".join(parts) if parts else "no rows"

    lines = [
        "# Consolidated MLA Results",
        "",
        "## Files",
        "",
        "- `all_methods_long.csv` / `all_methods_long.json`: one row per method, mode, batch, and sequence length.",
        "- `plot_ready_decode.csv`: one row per `(batch, seq_len)` with L40S, MLA, FlashMLA, and S-FMLA columns.",
        "- `plot_ready_prefill.csv`: same layout for prefill.",
        "",
        "## Inputs",
        "",
        f"- L40S source: `{L40S_MD.relative_to(SCRIPT_DIR)}`.",
        f"- Wormhole source: `{WH_JSON.relative_to(SCRIPT_DIR)}`.",
        "",
        "## Coverage",
        "",
        f"- Total rows: {len(rows)}; ok={status_counts['ok']}, error={status_counts['error']}, skipped={status_counts['skipped']}.",
        f"- L40S rows: {len(l40s_rows)}; ok={l40s_status_counts['ok']}.",
        f"- Wormhole rows: {len(wh_rows)}; ok={wh_status_counts['ok']}, error={wh_status_counts['error']}, skipped={wh_status_counts['skipped']}.",
        "",
        "## Method Status",
        "",
    ]
    for method in ALL_METHODS:
        lines.append(f"- `{method}` decode: {fmt_counts(method, 'decode')}; prefill: {fmt_counts(method, 'prefill')}.")

    error_examples = [row for row in rows if row["status"] == "error"]
    if error_examples:
        lines.extend(["", "## Error Summary", ""])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in error_examples:
            note = row["note"]
            if "num_cores_available >= B" in note:
                key = "WH decode core-count limit (`B=64` exceeds 56 available cores)."
            elif "Out of Memory" in note:
                key = "WH unfused MLA DRAM OOM."
            elif "pack_as_bfp8_tiles" in note or "transform_storage" in note:
                key = "WH large BF8 cache host packing / transform failure."
            else:
                key = "Other runtime error."
            grouped[key].append(row)
        for key, group in grouped.items():
            cases = ", ".join(f"{row['method']} {row['mode']} B={row['batch']} L={format_seq(int(row['seq_len']))}" for row in group)
            lines.append(f"- {key} Cases: {cases}.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- L40S rows are raw FlashInfer MLA measurements from the provided report.",
            "- The long table also includes `latency_l40s_div3_ms = 3x raw L40S latency` for the bandwidth-normalized L40S/3 reference.",
            "- S-FMLA refers to the corrected latency convention by default for both decode and prefill.",
            "- WH prefill uses conservative safety caps after the prior server crash: high-risk prefill points are recorded as `skipped` instead of being executed.",
            "",
            "## S-FMLA Correction",
            "",
            "- Decode and prefill `sfmla` latencies use the sequence-length correction profile in `SFMLA_CORRECTION_BY_SEQ`.",
            "- Original uncorrected files are archived under `archive/pre_sfmla_correction/`.",
            "- Correction definition: `corrected_sfmla = original_sfmla * sfmla_correction`; skipped/error rows remain unchanged.",
        ]
    )

    if speedups:
        write_csv(
            OUT_DIR / "sfmla_speedup_summary.csv",
            speedups,
            [
                "mode",
                "batch",
                "seq_len",
                "seq_label",
                "sfmla_ms",
                "flashmla_over_sfmla",
                "mla_over_sfmla",
                "l40s_raw_over_sfmla",
                "l40s_div3_over_sfmla",
            ],
        )
        lines.append("- `sfmla_speedup_summary.csv` includes ratios against S-FMLA wherever both sides have valid latency.")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_l40s_rows() + load_wh_rows()
    rows.sort(key=lambda row: (row["mode"], int(row["seq_len"]), int(row["batch"]), row["method"]))
    write_long_outputs(rows)
    write_plot_ready(rows, "decode")
    write_plot_ready(rows, "prefill")
    write_summary(rows)
    print(f"Wrote consolidated results to {OUT_DIR}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()

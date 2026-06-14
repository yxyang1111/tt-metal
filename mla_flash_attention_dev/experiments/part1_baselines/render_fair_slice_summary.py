#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mla_flash_attention_dev.experiments.part1_baselines.render_part1_results import (
    backfill_config_metadata,
    load_json,
    make_fair_slice_summary_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a fair slice summary table from a benchmark JSON payload.")
    parser.add_argument("--input-json", type=Path, required=True, help="Path to part1_four_method_results_filtered.json.")
    parser.add_argument("--axis", choices=("batch", "dims", "heads"), required=True, help="Axis to summarize.")
    parser.add_argument("--title", required=True, help="Markdown title for the generated table.")
    parser.add_argument("--output", type=Path, required=True, help="Output markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = backfill_config_metadata(load_json(args.input_json))
    content = make_fair_slice_summary_table(payload, axis=args.axis, title=args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

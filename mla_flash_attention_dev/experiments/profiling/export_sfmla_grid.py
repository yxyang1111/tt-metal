#!/usr/bin/env python3
"""Export or preview custom S-FMLA grid placements (N_S × C_S on chip)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ttnn
from mla_flash_attention_dev.sfmla import GridLayoutSpec, auto_place_wormhole, build_grid_class, layout_from_rect_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-s-blocks", type=int, required=True, help="N_S")
    parser.add_argument("--lane-cols", type=int, required=True, help="Lane rectangle width (C_S factor)")
    parser.add_argument("--lane-rows", type=int, required=True, help="Lane rectangle height (C_S factor)")
    parser.add_argument(
        "--dram-banks",
        type=str,
        default=None,
        help="Comma-separated DRAM bank ids per S-block (default: auto-spread)",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        default=None,
        help='Manual anchors "x,y;x,y;..." — skips auto-placement',
    )
    parser.add_argument("--output", type=Path, default=None, help="Write GridLayoutSpec JSON")
    parser.add_argument("--print-tree", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dram_banks = None
    if args.dram_banks:
        dram_banks = tuple(int(v) for v in args.dram_banks.split(","))

    if args.anchors:
        anchors = tuple(tuple(int(v) for v in pair.split(",")) for pair in args.anchors.split(";"))
        if dram_banks is None:
            dram_banks = tuple(range(len(anchors)))
        spec = layout_from_rect_blocks(
            anchors=anchors,
            dram_banks=dram_banks,
            lane_cols=args.lane_cols,
            lane_rows=args.lane_rows,
        )
    else:
        device = ttnn.open_device(device_id=args.device_id)
        try:
            spec = auto_place_wormhole(
                device,
                num_s_blocks=args.num_s_blocks,
                lane_cols=args.lane_cols,
                lane_rows=args.lane_rows,
                dram_banks=dram_banks,
            )
        finally:
            ttnn.close_device(device)

    payload = spec.to_dict()
    payload["summary"] = {
        "N_S": spec.num_s_blocks,
        "C_S": spec.cores_per_lane,
        "active_cores": spec.num_s_blocks * spec.cores_per_lane,
    }
    if args.print_tree:
        grid_cls = build_grid_class(spec)
        payload["tree_reduction_order"] = [list(step) for step in grid_cls.TREE_REDUCTION_ORDER]

    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

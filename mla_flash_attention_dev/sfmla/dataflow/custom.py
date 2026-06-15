"""Build FlashMLA-compatible grid classes from arbitrary N_S × C_S layouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLAOptimalGridNOC0_WH


@dataclass(frozen=True)
class SBlockPlacement:
    """One S-block: worker cores (sender first) and paired DRAM bank."""

    cores: tuple[tuple[int, int], ...]
    dram_bank: int


@dataclass(frozen=True)
class GridLayoutSpec:
    """Full chip placement for S-FMLA Block-Lane grid (N_S S-blocks × C_S lanes)."""

    s_blocks: tuple[SBlockPlacement, ...]

    @property
    def num_s_blocks(self) -> int:
        return len(self.s_blocks)

    @property
    def cores_per_lane(self) -> int:
        widths = {len(block.cores) for block in self.s_blocks}
        if len(widths) != 1:
            raise ValueError(f"All S-blocks must share lane width C_S, got {sorted(widths)}")
        return widths.pop()

    @property
    def dram_bank_order(self) -> tuple[int, ...]:
        return tuple(block.dram_bank for block in self.s_blocks)

    def validate(self, *, device: Any | None = None) -> None:
        if self.num_s_blocks < 1:
            raise ValueError("Grid must have at least one S-block")

        seen: set[tuple[int, int]] = set()
        for idx, block in enumerate(self.s_blocks):
            if not block.cores:
                raise ValueError(f"S-block {idx} has no cores")
            for core in block.cores:
                if core in seen:
                    raise ValueError(f"Duplicate worker core {core}")
                seen.add(core)

        if device is not None:
            grid_size = device.compute_with_storage_grid_size()
            for idx, block in enumerate(self.s_blocks):
                for x, y in block.cores:
                    if x >= grid_size.x or y >= grid_size.y:
                        raise ValueError(
                            f"S-block {idx} core ({x},{y}) exceeds device grid "
                            f"{grid_size.x}x{grid_size.y}"
                        )
            optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
            num_banks = len(optimal_workers)
            for bank in self.dram_bank_order:
                if bank < 0 or bank >= num_banks:
                    raise ValueError(f"DRAM bank {bank} out of range [0, {num_banks})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "s_blocks": [
                {"cores": [list(c) for c in block.cores], "dram_bank": block.dram_bank}
                for block in self.s_blocks
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GridLayoutSpec:
        blocks = []
        for entry in data["s_blocks"]:
            cores = tuple(tuple(c) for c in entry["cores"])
            blocks.append(SBlockPlacement(cores=cores, dram_bank=int(entry["dram_bank"])))
        return cls(s_blocks=tuple(blocks))

    @classmethod
    def from_json(cls, path: str | Path) -> GridLayoutSpec:
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2)


def rect_lane_cores(anchor_x: int, anchor_y: int, *, cols: int, rows: int) -> tuple[tuple[int, int], ...]:
    """Row-major rectangle; first core is the K DRAM reader (sender)."""
    if cols < 1 or rows < 1:
        raise ValueError(f"lane rectangle must be positive, got {cols}x{rows}")
    return tuple((anchor_x + col, anchor_y + row) for row in range(rows) for col in range(cols))


def build_tree_reduction_order(num_blocks: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Balanced pairwise tournament with block 0 as final accumulator."""
    if num_blocks <= 1:
        return ()

    steps: list[tuple[tuple[int, int], ...]] = []
    active = list(range(num_blocks))
    while len(active) > 1:
        pairs: list[tuple[int, int]] = []
        next_active: list[int] = []
        for i in range(0, len(active), 2):
            if i + 1 < len(active):
                dst, src = active[i], active[i + 1]
                pairs.append((dst, src))
                next_active.append(dst)
            else:
                next_active.append(active[i])
        if pairs:
            steps.append(tuple(pairs))
        active = next_active
    return tuple(steps)


def build_grid_class(
    spec: GridLayoutSpec,
    *,
    base_cls: type = FlashMLAOptimalGridNOC0_WH,
    name: str | None = None,
) -> type:
    """Materialize a FlashMLA grid class from an explicit placement spec."""
    spec.validate()
    blocks = tuple((block.cores, block.dram_bank) for block in spec.s_blocks)
    tree_order = build_tree_reduction_order(spec.num_s_blocks)
    class_name = name or f"SFMLACustomGrid_N{spec.num_s_blocks}xC{spec.cores_per_lane}"

    return type(
        class_name,
        (base_cls,),
        {
            "BLOCKS": blocks,
            "NUM_BLOCKS": spec.num_s_blocks,
            "CORES_PER_BLOCK": spec.cores_per_lane,
            "OPTIMAL_DRAM_BANK_ORDER": spec.dram_bank_order,
            "TREE_REDUCTION_ORDER": tree_order,
            "NUM_TREE_REDUCTION_STEPS": len(tree_order),
        },
    )


def layout_from_rect_blocks(
    *,
    anchors: tuple[tuple[int, int], ...],
    dram_banks: tuple[int, ...],
    lane_cols: int,
    lane_rows: int,
) -> GridLayoutSpec:
    """Build a layout from per-block anchor coordinates."""
    if len(anchors) != len(dram_banks):
        raise ValueError("anchors and dram_banks must have the same length")
    blocks = []
    for (ax, ay), bank in zip(anchors, dram_banks):
        cores = rect_lane_cores(ax, ay, cols=lane_cols, rows=lane_rows)
        blocks.append(SBlockPlacement(cores=cores, dram_bank=bank))
    return GridLayoutSpec(s_blocks=tuple(blocks))

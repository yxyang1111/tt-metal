"""Auto-place N_S × C_S S-FMLA grids on Wormhole worker cores."""

from __future__ import annotations

from typing import Any

import ttnn

from .custom import GridLayoutSpec, SBlockPlacement, rect_lane_cores


def _fits(
    anchor_x: int,
    anchor_y: int,
    *,
    cols: int,
    rows: int,
    grid_x: int,
    grid_y: int,
    used: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    cores = rect_lane_cores(anchor_x, anchor_y, cols=cols, rows=rows)
    for x, y in cores:
        if x >= grid_x or y >= grid_y or (x, y) in used:
            return None
    return cores


def find_rect_near_anchor(
    anchor_x: int,
    anchor_y: int,
    *,
    cols: int,
    rows: int,
    grid_x: int,
    grid_y: int,
    used: set[tuple[int, int]],
    search_radius: int = 8,
) -> tuple[tuple[int, int], ...]:
    """Search for a non-overlapping lane rectangle near ``(anchor_x, anchor_y)``."""
    for radius in range(search_radius + 1):
        candidates: list[tuple[int, int]] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidates.append((anchor_x + dx, anchor_y + dy))
        if radius == 0:
            candidates = [(anchor_x, anchor_y)]

        for ax, ay in candidates:
            cores = _fits(ax, ay, cols=cols, rows=rows, grid_x=grid_x, grid_y=grid_y, used=used)
            if cores is not None:
                return cores

    raise RuntimeError(
        f"Could not place {cols}x{rows} lane rectangle near ({anchor_x},{anchor_y}) "
        f"within radius {search_radius} on {grid_x}x{grid_y} grid"
    )


def spread_dram_banks(num_banks: int, num_s_blocks: int) -> tuple[int, ...]:
    """Pick ``num_s_blocks`` DRAM banks spread across the chip."""
    if num_s_blocks > num_banks:
        raise ValueError(f"num_s_blocks={num_s_blocks} exceeds available DRAM banks={num_banks}")
    if num_s_blocks == 1:
        return (0,)
    step = max(1, num_banks // num_s_blocks)
    banks = [min(i * step, num_banks - 1) for i in range(num_s_blocks)]
    # ensure uniqueness while preserving spread
    seen: set[int] = set()
    unique: list[int] = []
    for bank in banks:
        candidate = bank
        while candidate in seen and candidate + 1 < num_banks:
            candidate += 1
        if candidate in seen:
            candidate = bank
            while candidate in seen and candidate > 0:
                candidate -= 1
        seen.add(candidate)
        unique.append(candidate)
    return tuple(unique)


def auto_place_wormhole(
    device: Any,
    *,
    num_s_blocks: int,
    lane_cols: int,
    lane_rows: int,
    dram_banks: tuple[int, ...] | None = None,
) -> GridLayoutSpec:
    """Place ``num_s_blocks`` rectangular lanes (``lane_cols × lane_rows``) near DRAM banks."""
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")

    grid_size = device.compute_with_storage_grid_size()
    optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(optimal_workers)

    if dram_banks is None:
        dram_banks = spread_dram_banks(num_banks, num_s_blocks)
    if len(dram_banks) != num_s_blocks:
        raise ValueError(f"dram_banks length {len(dram_banks)} must equal num_s_blocks {num_s_blocks}")

    used: set[tuple[int, int]] = set()
    blocks: list[SBlockPlacement] = []
    for bank in dram_banks:
        if bank < 0 or bank >= num_banks:
            raise ValueError(f"DRAM bank {bank} out of range [0, {num_banks})")
        anchor = optimal_workers[bank]
        cores = find_rect_near_anchor(
            anchor.x,
            anchor.y,
            cols=lane_cols,
            rows=lane_rows,
            grid_x=grid_size.x,
            grid_y=grid_size.y,
            used=used,
        )
        used.update(cores)
        blocks.append(SBlockPlacement(cores=cores, dram_bank=bank))

    spec = GridLayoutSpec(s_blocks=tuple(blocks))
    spec.validate(device=device)
    return spec

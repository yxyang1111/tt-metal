"""Virtual grid φ: (i, b) → core — paper §3.3.

Wraps Wormhole S-block layouts from the experimental FlashMLA micro-op.
N_S = number of S-blocks (sequence-parallel DRAM streams).
C_S = cores per S-block (lane width / multicast fan-out).

Supports catalog floorplans (``wh_6x4``, ``wh_6x8``), truncated subsets, and
fully custom N_S × C_S placements via :mod:`custom` / :mod:`placement`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import get_flash_mla_wormhole_grid

from ..config import SFMLAWorkloadConfig
from .catalog import CORES_PER_BLOCK_TO_TOPOLOGY, get_topology_spec
from .custom import GridLayoutSpec, SBlockPlacement, build_grid_class
from .placement import auto_place_wormhole
from .subset import subset_topology


@dataclass(frozen=True)
class SFMLAGrid:
    """Paper-facing view of the Wormhole FlashMLA core grid."""

    grid_cls: type
    layout_spec: GridLayoutSpec | None = None

    @property
    def num_s_blocks(self) -> int:
        """N_S: K-sequence parallel S-block count."""
        return self.grid_cls.NUM_BLOCKS

    @property
    def cores_per_lane(self) -> int:
        """C_S: Q-shard lane width (max B lanes per S-block column)."""
        return self.grid_cls.CORES_PER_BLOCK

    @property
    def num_active_cores(self) -> int:
        return self.num_s_blocks * self.cores_per_lane

    @property
    def dram_bank_order(self) -> tuple[int, ...]:
        return self.grid_cls.OPTIMAL_DRAM_BANK_ORDER

    def all_worker_cores(self) -> list[tuple[int, int]]:
        return [core for block_cores, _ in self.grid_cls.BLOCKS for core in block_cores]

    def optimal_dram_grid(self) -> Any:
        return self.grid_cls.optimal_dram_grid()

    def validate_grid(self, device: Any) -> None:
        """Check device compute grid can host all S-block worker cores."""
        if self.layout_spec is not None:
            self.layout_spec.validate(device=device)
        if hasattr(self.grid_cls, "validate_grid"):
            self.grid_cls.validate_grid(device)

    def sender_cores(self, s_block_idx: int) -> tuple[tuple[int, int], ...]:
        """Physical workers in S-block i; sender(i) is the DRAM K reader (first core)."""
        return self.grid_cls.get_cores(s_block_idx)


def resolve_layout_spec(config: SFMLAWorkloadConfig, device: Any | None = None) -> GridLayoutSpec | None:
    """Build a custom layout spec from workload fields, if requested."""
    if config.grid_layout_json:
        spec = GridLayoutSpec.from_json(config.grid_layout_json)
        if device is not None:
            spec.validate(device=device)
        return spec

    if config.custom_core_coords is not None:
        banks = config.custom_dram_banks
        if banks is None:
            banks = tuple(range(len(config.custom_core_coords)))
        if len(banks) != len(config.custom_core_coords):
            raise ValueError("custom_dram_banks length must match custom_core_coords")
        blocks = tuple(
            SBlockPlacement(cores=tuple(cores), dram_bank=bank)
            for cores, bank in zip(config.custom_core_coords, banks)
        )
        spec = GridLayoutSpec(s_blocks=blocks)
        if device is not None:
            spec.validate(device=device)
        return spec

    if config.num_s_blocks is not None and config.lane_cols is not None and config.lane_rows is not None:
        if device is None:
            raise ValueError("device is required to auto-place a custom N_S × C_S grid")
        return auto_place_wormhole(
            device,
            num_s_blocks=config.num_s_blocks,
            lane_cols=config.lane_cols,
            lane_rows=config.lane_rows,
            dram_banks=config.custom_dram_banks,
        )

    return None


def get_sfmla_grid(
    *,
    cores_per_block: int = 4,
    num_s_blocks_active: int | None = None,
    topology: str | None = None,
    config: SFMLAWorkloadConfig | None = None,
    device: Any | None = None,
    layout_spec: GridLayoutSpec | None = None,
) -> SFMLAGrid:
    """Select or build an S-FMLA grid layout.

    Priority:
      1. Explicit ``layout_spec``
      2. Custom fields on ``config`` (JSON / manual coords / auto N_S×C_S)
      3. Catalog topology or ``cores_per_block`` floorplan (+ optional subset)
    """
    if layout_spec is None and config is not None:
        layout_spec = resolve_layout_spec(config, device=device)

    if layout_spec is not None:
        grid_cls = build_grid_class(layout_spec)
        return SFMLAGrid(grid_cls=grid_cls, layout_spec=layout_spec)

    if topology is not None:
        spec = get_topology_spec(topology)
        if spec.arch != "wormhole_b0":
            raise ValueError(f"Topology {topology!r} is not runnable on Wormhole WH S-FMLA decode")
        grid_cls = spec.grid_cls
    else:
        cs = config.effective_cores_per_block if config is not None else cores_per_block
        if cs not in CORES_PER_BLOCK_TO_TOPOLOGY:
            raise ValueError(
                f"cores_per_block must be one of {sorted(CORES_PER_BLOCK_TO_TOPOLOGY)}, got {cs}. "
                "For arbitrary N_S×C_S, set num_s_blocks + lane_cols + lane_rows on SFMLAWorkloadConfig."
            )
        grid_cls = get_flash_mla_wormhole_grid(cores_per_block=cs)

    if num_s_blocks_active is not None:
        grid_cls = subset_topology(grid_cls, num_s_blocks_active)

    return SFMLAGrid(grid_cls=grid_cls)

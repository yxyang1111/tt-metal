"""Virtual grid φ: (i, b) → core — paper §3.3.

Wraps Wormhole S-block layouts from the experimental FlashMLA micro-op.
N_S = number of S-blocks (sequence-parallel DRAM streams).
C_S = cores per S-block (lane width / multicast fan-out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLAOptimalGridNOC0_WH,
    FlashMLAOptimalGridNOC0_WH_8C,
    get_flash_mla_wormhole_grid,
)


@dataclass(frozen=True)
class SFMLAGrid:
    """Paper-facing view of the Wormhole FlashMLA core grid."""

    grid_cls: type

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
        if hasattr(self.grid_cls, "validate_grid"):
            self.grid_cls.validate_grid(device)

    def sender_cores(self, s_block_idx: int) -> tuple[tuple[int, int], ...]:
        """Physical workers in S-block i; sender(i) is the DRAM K reader (first core)."""
        return self.grid_cls.get_cores(s_block_idx)


def get_sfmla_grid(*, cores_per_block: int = 4) -> SFMLAGrid:
    """Select WH grid layout: 4-core (default) or 8-core lanes."""
    if cores_per_block not in (
        FlashMLAOptimalGridNOC0_WH.CORES_PER_BLOCK,
        FlashMLAOptimalGridNOC0_WH_8C.CORES_PER_BLOCK,
    ):
        raise ValueError(
            f"cores_per_block must be {FlashMLAOptimalGridNOC0_WH.CORES_PER_BLOCK} "
            f"or {FlashMLAOptimalGridNOC0_WH_8C.CORES_PER_BLOCK}, got {cores_per_block}"
        )
    return SFMLAGrid(grid_cls=get_flash_mla_wormhole_grid(cores_per_block=cores_per_block))

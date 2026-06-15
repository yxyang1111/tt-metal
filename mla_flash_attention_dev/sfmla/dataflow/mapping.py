"""Block-Lane placement constraints (paper §3.3, §3.5)."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SFMLAWorkloadConfig
from .grid import SFMLAGrid, get_sfmla_grid


@dataclass(frozen=True)
class SFMLAMapping:
    """Derived mapping for one launch: B lanes, N_S S-blocks, active core count."""

    num_q_shards: int  # B = H_q / τ
    num_s_blocks: int  # N_S
    cores_per_lane: int  # C_S
    required_q_cores: int  # batch × B
    max_active_cores: int

    def validate(self) -> None:
        if self.num_q_shards > self.cores_per_lane:
            raise RuntimeError(
                f"num_q_shards B={self.num_q_shards} exceeds C_S={self.cores_per_lane}"
            )
        if self.required_q_cores > self.max_active_cores:
            raise RuntimeError(
                f"batch × B = {self.required_q_cores} exceeds active cores {self.max_active_cores}"
            )


def derive_mapping(config: SFMLAWorkloadConfig, grid: SFMLAGrid | None = None) -> SFMLAMapping:
    grid = grid or get_sfmla_grid(cores_per_block=config.cores_per_block)
    num_q_shards = config.num_q_shards
    mapping = SFMLAMapping(
        num_q_shards=num_q_shards,
        num_s_blocks=grid.num_s_blocks,
        cores_per_lane=grid.cores_per_lane,
        required_q_cores=config.batch * num_q_shards,
        max_active_cores=grid.num_active_cores,
    )
    mapping.validate()
    return mapping

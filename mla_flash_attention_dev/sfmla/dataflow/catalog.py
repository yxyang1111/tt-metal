"""Registered Wormhole / reference topologies for S-FMLA (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLAOptimalGridNOC0,
    FlashMLAOptimalGridNOC0_WH,
    FlashMLAOptimalGridNOC0_WH_8C,
)


@dataclass(frozen=True)
class TopologySpec:
    name: str
    arch: str
    grid_cls: type
    cores_per_lane: int
    num_s_blocks: int
    description: str

    @property
    def num_active_cores(self) -> int:
        return self.num_s_blocks * self.cores_per_lane


WH_TOPOLOGY_CATALOG: dict[str, TopologySpec] = {
    "wh_6x4": TopologySpec(
        name="wh_6x4",
        arch="wormhole_b0",
        grid_cls=FlashMLAOptimalGridNOC0_WH,
        cores_per_lane=FlashMLAOptimalGridNOC0_WH.CORES_PER_BLOCK,
        num_s_blocks=FlashMLAOptimalGridNOC0_WH.NUM_BLOCKS,
        description="Wormhole 6 S-blocks × 4 lanes (default)",
    ),
    "wh_6x8": TopologySpec(
        name="wh_6x8",
        arch="wormhole_b0",
        grid_cls=FlashMLAOptimalGridNOC0_WH_8C,
        cores_per_lane=FlashMLAOptimalGridNOC0_WH_8C.CORES_PER_BLOCK,
        num_s_blocks=FlashMLAOptimalGridNOC0_WH_8C.NUM_BLOCKS,
        description="Wormhole 6 S-blocks × 8 lanes",
    ),
    "bh_8x8": TopologySpec(
        name="bh_8x8",
        arch="blackhole",
        grid_cls=FlashMLAOptimalGridNOC0,
        cores_per_lane=FlashMLAOptimalGridNOC0.CORES_PER_BLOCK,
        num_s_blocks=FlashMLAOptimalGridNOC0.NUM_BLOCKS,
        description="Blackhole reference 8×8 (cost-model / bring-up only on WH)",
    ),
}

CORES_PER_BLOCK_TO_TOPOLOGY: dict[int, str] = {
    FlashMLAOptimalGridNOC0_WH.CORES_PER_BLOCK: "wh_6x4",
    FlashMLAOptimalGridNOC0_WH_8C.CORES_PER_BLOCK: "wh_6x8",
}


def list_topology_names(*, arch: str | None = "wormhole_b0") -> tuple[str, ...]:
    if arch is None:
        return tuple(WH_TOPOLOGY_CATALOG)
    return tuple(name for name, spec in WH_TOPOLOGY_CATALOG.items() if spec.arch == arch)


def get_topology_spec(name: str) -> TopologySpec:
    if name not in WH_TOPOLOGY_CATALOG:
        raise KeyError(f"Unknown topology {name!r}; choose from {list(WH_TOPOLOGY_CATALOG)}")
    return WH_TOPOLOGY_CATALOG[name]

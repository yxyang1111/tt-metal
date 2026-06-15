"""Virtual grid and Block-Lane mapping (paper §3.3)."""

from .catalog import TopologySpec, get_topology_spec, list_topology_names
from .custom import GridLayoutSpec, SBlockPlacement, build_grid_class, build_tree_reduction_order, layout_from_rect_blocks
from .grid import SFMLAGrid, get_sfmla_grid, resolve_layout_spec
from .mapping import SFMLAMapping, derive_mapping
from .placement import auto_place_wormhole, find_rect_near_anchor, spread_dram_banks
from .subset import subset_topology

__all__ = [
    "TopologySpec",
    "get_topology_spec",
    "list_topology_names",
    "GridLayoutSpec",
    "SBlockPlacement",
    "build_grid_class",
    "build_tree_reduction_order",
    "layout_from_rect_blocks",
    "resolve_layout_spec",
    "auto_place_wormhole",
    "find_rect_near_anchor",
    "spread_dram_banks",
    "subset_topology",
    "SFMLAGrid",
    "get_sfmla_grid",
    "SFMLAMapping",
    "derive_mapping",
]

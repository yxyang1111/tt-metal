"""S-FMLA: spatial Flash Multi-head Latent Attention on Tenstorrent Wormhole.

Paper-aligned Python API for the Block-Lane dataflow (N_S S-blocks × C_S lanes),
tensor placement, and decode runtime. Device kernels live under
``ttnn/cpp/ttnn/operations/transformer/sdpa_decode/``.
"""

from .config import SFMLAWorkloadConfig
from .dataflow.custom import GridLayoutSpec, SBlockPlacement, build_grid_class, build_tree_reduction_order
from .dataflow.grid import SFMLAGrid, get_sfmla_grid, resolve_layout_spec
from .dataflow.mapping import SFMLAMapping, derive_mapping
from .dataflow.placement import auto_place_wormhole
from .dse import list_valid_configs, validate_config
from .runtime.decode import SFMLADecodeInputs, build_decode_inputs, run_decode
from .reference import golden_decode

__all__ = [
    "SFMLAWorkloadConfig",
    "GridLayoutSpec",
    "SBlockPlacement",
    "build_grid_class",
    "build_tree_reduction_order",
    "auto_place_wormhole",
    "resolve_layout_spec",
    "SFMLAGrid",
    "get_sfmla_grid",
    "SFMLAMapping",
    "derive_mapping",
    "list_valid_configs",
    "validate_config",
    "SFMLADecodeInputs",
    "build_decode_inputs",
    "run_decode",
    "golden_decode",
]

"""S-FMLA: spatial Flash Multi-head Latent Attention on Tenstorrent Wormhole.

Paper-aligned Python API for the Block-Lane dataflow (N_S S-blocks × C_S lanes),
tensor placement, and decode runtime. Device kernels live under
``ttnn/cpp/ttnn/operations/transformer/sdpa_decode/``.
"""

from .config import SFMLAWorkloadConfig
from .dataflow.grid import SFMLAGrid, get_sfmla_grid
from .dataflow.mapping import SFMLAMapping, derive_mapping
from .runtime.decode import SFMLADecodeInputs, build_decode_inputs, run_decode
from .reference import golden_decode

__all__ = [
    "SFMLAWorkloadConfig",
    "SFMLAGrid",
    "get_sfmla_grid",
    "SFMLAMapping",
    "derive_mapping",
    "SFMLADecodeInputs",
    "build_decode_inputs",
    "run_decode",
    "golden_decode",
]

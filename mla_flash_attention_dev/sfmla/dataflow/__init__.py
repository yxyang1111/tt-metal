"""Virtual grid and Block-Lane mapping (paper §3.3)."""

from .grid import SFMLAGrid, get_sfmla_grid
from .mapping import SFMLAMapping, derive_mapping

__all__ = ["SFMLAGrid", "get_sfmla_grid", "SFMLAMapping", "derive_mapping"]

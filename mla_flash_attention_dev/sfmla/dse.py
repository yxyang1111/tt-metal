"""Enumerate and validate hardware-feasible S-FMLA decode configurations."""

from __future__ import annotations

from itertools import product
from typing import Any

from .config import SFMLAWorkloadConfig
from .dataflow.catalog import WH_TOPOLOGY_CATALOG, CORES_PER_BLOCK_TO_TOPOLOGY
from .dataflow.grid import get_sfmla_grid
from .dataflow.mapping import derive_mapping

VALID_TAU_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)
VALID_K_CHUNK_SIZES: tuple[int, ...] = (64, 128, 256, 512)


def valid_tau_values(num_heads: int) -> tuple[int, ...]:
    return tuple(tau for tau in VALID_TAU_VALUES if num_heads % tau == 0)


def is_valid_k_chunk_size(k_chunk_size: int) -> bool:
    return k_chunk_size > 0 and (k_chunk_size & (k_chunk_size - 1)) == 0 and k_chunk_size % 32 == 0


def _validate_grid_fields(config: SFMLAWorkloadConfig) -> list[str]:
    errors: list[str] = []

    if config.uses_custom_grid:
        if config.grid_layout_json and (config.custom_core_coords or config.num_s_blocks):
            errors.append("grid_layout_json cannot be combined with other custom grid fields")
        if config.custom_core_coords is not None and config.num_s_blocks is not None:
            errors.append("custom_core_coords cannot be combined with num_s_blocks auto-placement")
        if config.num_s_blocks is not None:
            if config.lane_cols is None or config.lane_rows is None:
                errors.append("num_s_blocks requires lane_cols and lane_rows for auto-placement")
            elif config.num_s_blocks < 1:
                errors.append(f"num_s_blocks must be >= 1, got {config.num_s_blocks}")
            elif config.lane_cols < 1 or config.lane_rows < 1:
                errors.append(f"lane_cols/lane_rows must be >= 1, got {config.lane_cols}x{config.lane_rows}")
        if config.custom_core_coords is not None:
            widths = {len(cores) for cores in config.custom_core_coords}
            if len(widths) != 1:
                errors.append("all custom_core_coords entries must have the same C_S")
            if config.custom_dram_banks is not None and len(config.custom_dram_banks) != len(
                config.custom_core_coords
            ):
                errors.append("custom_dram_banks length must match custom_core_coords")
        return errors

    if config.cores_per_block not in CORES_PER_BLOCK_TO_TOPOLOGY:
        errors.append(
            f"cores_per_block must be one of {sorted(CORES_PER_BLOCK_TO_TOPOLOGY)}, got {config.cores_per_block}"
        )

    if config.num_s_blocks_active is not None:
        base_name = CORES_PER_BLOCK_TO_TOPOLOGY.get(config.cores_per_block)
        if base_name:
            max_ns = WH_TOPOLOGY_CATALOG[base_name].num_s_blocks
            if not (1 <= config.num_s_blocks_active <= max_ns):
                errors.append(f"num_s_blocks_active must be in [1, {max_ns}], got {config.num_s_blocks_active}")

    return errors


def validate_config(config: SFMLAWorkloadConfig, *, device: Any | None = None) -> list[str]:
    """Return human-readable validation errors; empty list means feasible."""
    errors: list[str] = list(_validate_grid_fields(config))

    if config.num_q_heads_per_core not in valid_tau_values(config.num_heads):
        errors.append(
            f"num_q_heads_per_core must divide num_heads={config.num_heads} "
            f"and be one of {VALID_TAU_VALUES}, got {config.num_q_heads_per_core}"
        )

    if not is_valid_k_chunk_size(config.k_chunk_size):
        errors.append(f"k_chunk_size must be a power-of-two multiple of 32, got {config.k_chunk_size}")

    if errors:
        return errors

    if config.uses_custom_grid and device is None and config.num_s_blocks is not None:
        cs = config.effective_cores_per_block
        ns = config.num_s_blocks
        if config.num_q_shards > cs:
            errors.append(f"num_q_shards B={config.num_q_shards} exceeds C_S={cs}")
        required = config.batch * config.num_q_shards
        if required > ns * cs:
            errors.append(f"batch × B = {required} exceeds N_S×C_S = {ns * cs}")
        return errors

    try:
        grid = get_sfmla_grid(
            cores_per_block=config.effective_cores_per_block,
            num_s_blocks_active=config.num_s_blocks_active,
            config=config,
            device=device,
        )
        if device is not None:
            grid.validate_grid(device)
        derive_mapping(config, grid)
    except (ValueError, RuntimeError) as exc:
        errors.append(str(exc))

    return errors


def list_valid_configs(
    *,
    batch_options: tuple[int, ...] = (1, 2, 4, 8),
    seq_len: int = 4096,
    num_heads: int = 32,
    num_kv_heads: int = 1,
    cores_per_block_options: tuple[int, ...] = (4, 8),
    num_s_blocks_active_options: tuple[int | None, ...] = (None, 1, 2, 3, 4, 5, 6),
    custom_grid_options: tuple[tuple[int, int, int], ...] = (),
    tau_options: tuple[int, ...] | None = None,
    k_chunk_options: tuple[int, ...] = VALID_K_CHUNK_SIZES,
    device: Any | None = None,
) -> list[SFMLAWorkloadConfig]:
    """Enumerate feasible ``SFMLAWorkloadConfig`` points for decode DSE.

    ``custom_grid_options`` entries are ``(num_s_blocks, lane_cols, lane_rows)`` tuples.
    When ``device`` is provided, auto-placement feasibility is checked for custom grids.
    """
    tau_options = tau_options or valid_tau_values(num_heads)
    configs: list[SFMLAWorkloadConfig] = []

    grid_variants: list[dict[str, Any]] = []
    for cores_per_block, num_s_blocks_active in product(cores_per_block_options, num_s_blocks_active_options):
        if num_s_blocks_active is not None:
            base_name = CORES_PER_BLOCK_TO_TOPOLOGY.get(cores_per_block)
            if base_name is None:
                continue
            max_ns = WH_TOPOLOGY_CATALOG[base_name].num_s_blocks
            if num_s_blocks_active < 1 or num_s_blocks_active > max_ns:
                continue
        grid_variants.append(
            {
                "cores_per_block": cores_per_block,
                "num_s_blocks_active": num_s_blocks_active,
                "num_s_blocks": None,
                "lane_cols": None,
                "lane_rows": None,
            }
        )

    for num_s_blocks, lane_cols, lane_rows in custom_grid_options:
        grid_variants.append(
            {
                "cores_per_block": lane_cols * lane_rows,
                "num_s_blocks_active": None,
                "num_s_blocks": num_s_blocks,
                "lane_cols": lane_cols,
                "lane_rows": lane_rows,
            }
        )

    for batch, grid, tau, k_chunk in product(batch_options, grid_variants, tau_options, k_chunk_options):
        config = SFMLAWorkloadConfig(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_q_heads_per_core=tau,
            cores_per_block=grid["cores_per_block"],
            k_chunk_size=k_chunk,
            num_s_blocks_active=grid["num_s_blocks_active"],
            num_s_blocks=grid["num_s_blocks"],
            lane_cols=grid["lane_cols"],
            lane_rows=grid["lane_rows"],
        )
        if not validate_config(config, device=device):
            configs.append(config)

    return configs

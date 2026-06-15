"""Build a truncated WH topology with fewer active S-blocks (paper N_S sweep)."""

from __future__ import annotations


def subset_topology(base_cls: type, num_s_blocks: int) -> type:
    """Return a grid class using the first ``num_s_blocks`` S-blocks of ``base_cls``."""
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")
    if num_s_blocks >= base_cls.NUM_BLOCKS:
        return base_cls

    blocks = base_cls.BLOCKS[:num_s_blocks]
    bank_order = tuple(block[1] for block in blocks)

    def _filter_tree_reduction(order: tuple) -> tuple:
        filtered_steps: list[tuple] = []
        for step in order:
            pairs = tuple((dst, src) for dst, src in step if dst < num_s_blocks and src < num_s_blocks)
            if pairs:
                filtered_steps.append(pairs)
        if not filtered_steps and num_s_blocks == 1:
            return ()
        return tuple(filtered_steps)

    name = f"{base_cls.__name__}_S{num_s_blocks}"

    return type(
        name,
        (base_cls,),
        {
            "BLOCKS": blocks,
            "NUM_BLOCKS": num_s_blocks,
            "CORES_PER_BLOCK": base_cls.CORES_PER_BLOCK,
            "OPTIMAL_DRAM_BANK_ORDER": bank_order,
            "TREE_REDUCTION_ORDER": _filter_tree_reduction(base_cls.TREE_REDUCTION_ORDER),
            "NUM_TREE_REDUCTION_STEPS": len(_filter_tree_reduction(base_cls.TREE_REDUCTION_ORDER)),
        },
    )

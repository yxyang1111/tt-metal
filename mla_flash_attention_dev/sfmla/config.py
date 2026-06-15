"""Workload and compile-time parameters for S-FMLA (paper §3.2–3.5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SFMLAWorkloadConfig:
    """Semantic + mapping parameters for one S-FMLA decode launch.

    Paper symbols:
      - ``num_heads`` → H_q
      - ``num_kv_heads`` → H_kv (DeepSeek MLA uses 1)
      - ``kv_lora_rank`` → d_v
      - ``qk_rope_head_dim`` + ``kv_lora_rank`` → d_k = kvpe_dim
      - ``num_q_heads_per_core`` → τ (heads per lane shard); C_S must cover B = H_q / τ
      - ``cores_per_block`` → C_S (lane width / max Q shards per S-block column)
      - ``k_chunk_size`` → L_k (K sequence chunk in tiles)
    """

    batch: int
    seq_len: int
    num_heads: int = 32
    num_kv_heads: int = 1
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    num_q_heads_per_core: int = 8
    cores_per_block: int = 4
    k_chunk_size: int = 128
    block_size: int = 64
    num_s_blocks_active: int | None = None
    """If set, use only the first N_S S-blocks (1..6 on WH). None = full topology."""

    # Custom N_S × C_S grid (overrides catalog floorplans when set).
    num_s_blocks: int | None = None
    lane_cols: int | None = None
    lane_rows: int | None = None
    custom_dram_banks: tuple[int, ...] | None = None
    custom_core_coords: tuple[tuple[tuple[int, int], ...], ...] | None = None
    grid_layout_json: str | None = None
    """Path to JSON from ``GridLayoutSpec.to_json`` for fully manual placement."""

    @property
    def uses_custom_grid(self) -> bool:
        return bool(
            self.grid_layout_json
            or self.custom_core_coords is not None
            or (self.num_s_blocks is not None and self.lane_cols is not None and self.lane_rows is not None)
        )

    @property
    def effective_cores_per_block(self) -> int:
        if self.lane_cols is not None and self.lane_rows is not None:
            return self.lane_cols * self.lane_rows
        return self.cores_per_block

    @property
    def kvpe_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        return self.kvpe_dim**-0.5

    @property
    def num_q_shards(self) -> int:
        if self.num_heads % self.num_q_heads_per_core != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by "
                f"num_q_heads_per_core={self.num_q_heads_per_core}"
            )
        return self.num_heads // self.num_q_heads_per_core

    @property
    def cur_pos(self) -> list[int]:
        return [self.seq_len - 1] * self.batch

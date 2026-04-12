#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_DEVICE_ID = 0
DEFAULT_TORCH_INPUT_DTYPE_STR = "torch.bfloat16"
DEFAULT_BATCHES = (1,)
DEFAULT_NUM_HEADS = (32,)
DEFAULT_NUM_KV_HEADS = (1,)
DEFAULT_VALUE_DIMS = (512,)
DEFAULT_ROPE_DIMS = (64,)
DEFAULT_DECODE_SEQ_LENS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
DEFAULT_PREFILL_SEQ_LENS = (1024, 4096)
DEFAULT_BLOCK_SIZE = 64
DEFAULT_K_CHUNK_SIZE = 128
DEFAULT_MAX_CORES_PER_HEAD_BATCH = 4
DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE = 8
DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE_VALUES = (DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE,)

DEFAULT_PROBE_BATCHES = (1, 2, 4)
DEFAULT_PROBE_NUM_HEADS = (8, 16, 24, 32)
DEFAULT_PROBE_NUM_KV_HEADS = (1,)
DEFAULT_PROBE_VALUE_DIMS = (256, 512)
DEFAULT_PROBE_ROPE_DIMS = (64,)
DEFAULT_PROBE_DECODE_SEQ_LENS = (1024, 8192, 32768)
DEFAULT_WIDE_BATCHES = (1, 2, 4, 8, 16)
DEFAULT_EXPANDED_VALUE_DIMS = (128, 192, 256, 384, 512)


@dataclass(frozen=True)
class SweepPreset:
    name: str
    description: str
    batches: tuple[int, ...]
    num_heads: tuple[int, ...]
    num_kv_heads: tuple[int, ...]
    value_dims: tuple[int, ...]
    rope_dims: tuple[int, ...]
    deepseek_num_q_heads_per_core_values: tuple[int, ...]
    decode_seq_lens: tuple[int, ...]
    prefill_seq_lens: tuple[int, ...] = ()


SWEEP_PRESETS: dict[str, SweepPreset] = {
    "plan_phase2": SweepPreset(
        name="plan_phase2",
        description=(
            "Plan 里的第一阶段公共参数空间：B=1/2/4, H=8/16/24/32, "
            "value_dim=256/512, rope_dim=64, decode=1k/8k/32k"
        ),
        batches=DEFAULT_PROBE_BATCHES,
        num_heads=DEFAULT_PROBE_NUM_HEADS,
        num_kv_heads=DEFAULT_PROBE_NUM_KV_HEADS,
        value_dims=DEFAULT_PROBE_VALUE_DIMS,
        rope_dims=DEFAULT_PROBE_ROPE_DIMS,
        deepseek_num_q_heads_per_core_values=DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE_VALUES,
        decode_seq_lens=DEFAULT_PROBE_DECODE_SEQ_LENS,
        prefill_seq_lens=(),
    ),
    "wide_bh": SweepPreset(
        name="wide_bh",
        description=(
            "扩展 batch/head 候选空间：B=1/2/4/8/16, H=8/16/24/32, "
            "value_dim=256/512, rope_dim=64, decode=1k/8k/32k"
        ),
        batches=DEFAULT_WIDE_BATCHES,
        num_heads=(8, 16, 24, 32),
        num_kv_heads=(1,),
        value_dims=(256, 512),
        rope_dims=(64,),
        deepseek_num_q_heads_per_core_values=DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE_VALUES,
        decode_seq_lens=(1024, 8192, 32768),
        prefill_seq_lens=(),
    ),
    "wide_bh_full": SweepPreset(
        name="wide_bh_full",
        description=(
            "更完整的主 sweep：B=1/2/4/8/16, H=8/16/24/32, "
            "value_dim=128/192/256/384/512, rope_dim=64, decode 使用完整 256..128k"
        ),
        batches=DEFAULT_WIDE_BATCHES,
        num_heads=(8, 16, 24, 32),
        num_kv_heads=(1,),
        value_dims=DEFAULT_EXPANDED_VALUE_DIMS,
        rope_dims=(64,),
        deepseek_num_q_heads_per_core_values=DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE_VALUES,
        decode_seq_lens=DEFAULT_DECODE_SEQ_LENS,
        prefill_seq_lens=(),
    ),
}


def sweep_preset_names() -> tuple[str, ...]:
    return tuple(SWEEP_PRESETS.keys())


def get_sweep_preset(name: str) -> SweepPreset:
    return SWEEP_PRESETS[name]


def format_seq_len(seq_len: int) -> str:
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"{seq_len // 1024}k"
    return str(seq_len)


@dataclass(frozen=True)
class ExperimentConfig:
    batch: int
    num_heads: int
    num_kv_heads: int
    std_head_dim: int
    mla_head_dim_v: int
    mla_d_rope: int
    deepseek_qk_nope_head_dim: int
    deepseek_qk_rope_head_dim: int
    deepseek_kv_lora_rank: int
    block_size: int = DEFAULT_BLOCK_SIZE
    k_chunk_size: int = DEFAULT_K_CHUNK_SIZE
    max_cores_per_head_batch: int = DEFAULT_MAX_CORES_PER_HEAD_BATCH
    deepseek_num_q_heads_per_core: int = DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE
    torch_input_dtype: str = DEFAULT_TORCH_INPUT_DTYPE_STR

    @property
    def common_value_dim(self) -> int:
        return self.std_head_dim

    @property
    def mla_head_dim_qk(self) -> int:
        return self.mla_head_dim_v + self.mla_d_rope

    @property
    def deepseek_qk_head_dim(self) -> int:
        return self.deepseek_qk_nope_head_dim + self.deepseek_qk_rope_head_dim

    @property
    def deepseek_kvpe_dim(self) -> int:
        return self.deepseek_kv_lora_rank + self.deepseek_qk_rope_head_dim

    @property
    def config_signature(self) -> str:
        return (
            f"b{self.batch}_h{self.num_heads}_hkv{self.num_kv_heads}"
            f"_dv{self.common_value_dim}_ro{self.mla_d_rope}"
            f"_blk{self.block_size}_kc{self.k_chunk_size}"
            f"_dqhpc{self.deepseek_num_q_heads_per_core}"
        )

    @property
    def config_label(self) -> str:
        return (
            f"B={self.batch}, H={self.num_heads}, H_kv={self.num_kv_heads}, "
            f"value_dim={self.common_value_dim}, rope_dim={self.mla_d_rope}, "
            f"block={self.block_size}, k_chunk={self.k_chunk_size}, "
            f"deepseek_q_heads_per_core={self.deepseek_num_q_heads_per_core}"
        )

    @property
    def deepseek_num_q_shards(self) -> int | None:
        if self.num_heads % self.deepseek_num_q_heads_per_core != 0:
            return None
        return self.num_heads // self.deepseek_num_q_heads_per_core

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "common_value_dim": self.common_value_dim,
                "mla_head_dim_qk": self.mla_head_dim_qk,
                "deepseek_qk_head_dim": self.deepseek_qk_head_dim,
                "deepseek_kvpe_dim": self.deepseek_kvpe_dim,
                "config_signature": self.config_signature,
                "config_label": self.config_label,
                "deepseek_num_q_shards": self.deepseek_num_q_shards,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            batch=int(data["batch"]),
            num_heads=int(data["num_heads"]),
            num_kv_heads=int(data["num_kv_heads"]),
            std_head_dim=int(data["std_head_dim"]),
            mla_head_dim_v=int(data["mla_head_dim_v"]),
            mla_d_rope=int(data["mla_d_rope"]),
            deepseek_qk_nope_head_dim=int(data["deepseek_qk_nope_head_dim"]),
            deepseek_qk_rope_head_dim=int(data["deepseek_qk_rope_head_dim"]),
            deepseek_kv_lora_rank=int(data["deepseek_kv_lora_rank"]),
            block_size=int(data.get("block_size", DEFAULT_BLOCK_SIZE)),
            k_chunk_size=int(data.get("k_chunk_size", DEFAULT_K_CHUNK_SIZE)),
            max_cores_per_head_batch=int(data.get("max_cores_per_head_batch", DEFAULT_MAX_CORES_PER_HEAD_BATCH)),
            deepseek_num_q_heads_per_core=int(
                data.get("deepseek_num_q_heads_per_core", DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE)
            ),
            torch_input_dtype=str(data.get("torch_input_dtype", DEFAULT_TORCH_INPUT_DTYPE_STR)),
        )


def make_strict_four_way_config(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    value_dim: int,
    rope_dim: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    k_chunk_size: int = DEFAULT_K_CHUNK_SIZE,
    max_cores_per_head_batch: int = DEFAULT_MAX_CORES_PER_HEAD_BATCH,
    deepseek_num_q_heads_per_core: int = DEFAULT_DEEPSEEK_NUM_Q_HEADS_PER_CORE,
    torch_input_dtype: str = DEFAULT_TORCH_INPUT_DTYPE_STR,
) -> ExperimentConfig:
    return ExperimentConfig(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        std_head_dim=value_dim,
        mla_head_dim_v=value_dim,
        mla_d_rope=rope_dim,
        deepseek_qk_nope_head_dim=value_dim,
        deepseek_qk_rope_head_dim=rope_dim,
        deepseek_kv_lora_rank=value_dim,
        block_size=block_size,
        k_chunk_size=k_chunk_size,
        max_cores_per_head_batch=max_cores_per_head_batch,
        deepseek_num_q_heads_per_core=deepseek_num_q_heads_per_core,
        torch_input_dtype=torch_input_dtype,
    )


def workload_case_name(mode: str, seq_len: int, config: ExperimentConfig) -> str:
    return f"{mode}_s{format_seq_len(seq_len)}_{config.config_signature}"


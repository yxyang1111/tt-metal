#!/usr/bin/env python3
"""Timed S-FMLA decode probe with full ``SFMLAWorkloadConfig`` (τ, C_S, L_k, N_S)."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from mla_flash_attention_dev.experiments.baselines.naive_ttnn import run_naive_mla_benchmark as baseline
from mla_flash_attention_dev.sfmla import (
    SFMLAWorkloadConfig,
    build_decode_inputs,
    derive_mapping,
    get_sfmla_grid,
    list_valid_configs,
    run_decode,
    validate_config,
)
from mla_flash_attention_dev.sfmla.runtime.decode import deallocate_decode_inputs

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "sfmla_wh"


def timed_decode(
    device: Any,
    *,
    batch: int,
    seq_len: int,
    cores_per_block: int,
    num_q_heads_per_core: int,
    k_chunk_size: int,
    num_s_blocks_active: int | None,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    config = SFMLAWorkloadConfig(
        batch=batch,
        seq_len=seq_len,
        num_heads=baseline.NUM_HEADS,
        num_kv_heads=baseline.NUM_KV_HEADS,
        kv_lora_rank=baseline.KV_LORA_RANK,
        qk_rope_head_dim=baseline.D_ROPE,
        num_q_heads_per_core=num_q_heads_per_core,
        cores_per_block=cores_per_block,
        k_chunk_size=k_chunk_size,
        num_s_blocks_active=num_s_blocks_active,
    )
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))

    mapping = derive_mapping(config)
    grid = get_sfmla_grid(
        cores_per_block=config.cores_per_block,
        num_s_blocks_active=config.num_s_blocks_active,
    )
    q = torch.randn((1, batch, baseline.NUM_HEADS, baseline.D_QK), dtype=torch.bfloat16)
    k = torch.randn((batch, baseline.NUM_KV_HEADS, seq_len, baseline.D_QK), dtype=torch.bfloat16)

    decode_inputs = build_decode_inputs(device, q, k, config)
    try:
        for _ in range(warmup):
            out = run_decode(device, decode_inputs)
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)

        latencies_ms: list[float] = []
        for _ in range(iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run_decode(device, decode_inputs)
            ttnn.synchronize_device(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(out)
    finally:
        deallocate_decode_inputs(decode_inputs)

    return {
        "batch": batch,
        "seq_len": seq_len,
        "tau": num_q_heads_per_core,
        "C_S": cores_per_block,
        "k_chunk_size": k_chunk_size,
        "num_s_blocks_active": num_s_blocks_active,
        "N_S": grid.num_s_blocks,
        "B": mapping.num_q_shards,
        "required_q_cores": mapping.required_q_cores,
        "max_cores_per_head_batch": cores_per_block,
        "mapping_note": decode_inputs.mapping_note,
        "latencies_ms": [round(v, 6) for v in latencies_ms],
        "mean_ms": round(statistics.mean(latencies_ms), 6),
        "min_ms": round(min(latencies_ms), 6),
        "max_ms": round(max(latencies_ms), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--sfmla-cores-per-block", type=int, default=4, choices=(4, 8))
    parser.add_argument("--sfmla-num-q-heads-per-core", type=int, default=8)
    parser.add_argument("--sfmla-k-chunk-size", type=int, default=128)
    parser.add_argument("--sfmla-num-s-blocks-active", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--list-valid-configs", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_valid_configs:
        configs = list_valid_configs(
            batch_options=(args.batch,),
            seq_len=args.seq_len,
            cores_per_block_options=(args.sfmla_cores_per_block,),
        )
        print(json.dumps([c.__dict__ for c in configs], indent=2))
        return

    device = ttnn.open_device(device_id=args.device_id)
    try:
        result = timed_decode(
            device,
            batch=args.batch,
            seq_len=args.seq_len,
            cores_per_block=args.sfmla_cores_per_block,
            num_q_heads_per_core=args.sfmla_num_q_heads_per_core,
            k_chunk_size=args.sfmla_k_chunk_size,
            num_s_blocks_active=args.sfmla_num_s_blocks_active,
            warmup=args.warmup,
            iters=args.iters,
        )
    finally:
        ttnn.close_device(device)

    output_path = args.output or (OUTPUT_ROOT / "sfmla_wh_profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote {}", output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

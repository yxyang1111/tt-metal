#!/usr/bin/env python3

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode,
    FlashMLAProgramConfig,
    get_flash_mla_wormhole_grid,
)
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    run_flash_mla_decode_impl,
    run_flash_mla_prefill_impl,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "mla_flash_attention_dev" / "experiments" / "profile_outputs" / "flash_mla_wh_smoke"
DEFAULT_CASES = ("standalone_decode_wh", "decode", "prefill")


def deallocate_tensors(tensors: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        tensor = tensors.get(key)
        if tensor is not None:
            ttnn.deallocate(tensor)


def ensure_wormhole(device: Any) -> None:
    arch_name = str(device.arch()).lower()
    if "wormhole" not in arch_name:
        raise RuntimeError(f"Expected a Wormhole device, got {device.arch()}")


def build_standalone_wh_decode_inputs(
    device: Any,
    *,
    batch_size: int,
    decode_position: int,
    k_chunk_size: int,
    max_seq_len: int,
    num_heads: int,
    num_q_heads_per_core: int,
    wh_cores_per_block: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kvpe_dim = kv_lora_rank + qk_rope_head_dim
    scale = qk_head_dim**-0.5

    if num_heads % num_q_heads_per_core != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_q_heads_per_core={num_q_heads_per_core}"
        )

    num_q_shards = num_heads // num_q_heads_per_core
    grid = get_flash_mla_wormhole_grid(cores_per_block=wh_cores_per_block)
    if num_q_shards > grid.CORES_PER_BLOCK:
        raise RuntimeError(f"num_q_shards {num_q_shards} exceeds cores_per_block {grid.CORES_PER_BLOCK}")
    all_active_cores = [core for block_cores, _ in grid.BLOCKS for core in block_cores]
    required_q_cores = batch_size * num_q_shards
    if required_q_cores > len(all_active_cores):
        raise RuntimeError(
            f"batch_size * num_q_shards {batch_size} * {num_q_shards} exceeds active_q_cores {len(all_active_cores)}"
        )

    tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))
    q_cores = all_active_cores[:required_q_cores]
    q_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in q_cores]
    )
    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR),
    )

    q_shape = (1, batch_size, num_heads, kvpe_dim)
    cache_shape = (batch_size, 1, max_seq_len, kvpe_dim)
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_cache = torch.randn(cache_shape, dtype=torch.bfloat16)

    tt_q = ttnn.from_torch(
        torch_q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=q_mem_config,
        tile=tiny_tile,
    )

    program_config = FlashMLAProgramConfig(
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        grid=grid,
        allow_wh_fallback=False,
    )

    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, program_config.k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )
    tt_cache = ttnn.from_torch(
        torch_cache,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=kv_mem_config,
    )

    position_ids = torch.full((batch_size,), decode_position, dtype=torch.int32)
    tt_position_ids = ttnn.from_torch(
        position_ids,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    out_shape = (1, batch_size, num_heads, kv_lora_rank)
    tt_out = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem_config,
        tile=tiny_tile,
    )

    compute_kernel_config = ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tensors = {
        "tt_q": tt_q,
        "tt_cache": tt_cache,
        "tt_position_ids": tt_position_ids,
        "tt_out": tt_out,
        "program_config": program_config,
        "compute_kernel_config": compute_kernel_config,
    }
    metadata = {
        "batch_size": batch_size,
        "decode_position": decode_position,
        "k_chunk_size": k_chunk_size,
        "max_seq_len": max_seq_len,
        "num_heads": num_heads,
        "num_q_heads_per_core": num_q_heads_per_core,
        "num_q_shards": num_q_shards,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "scale": scale,
        "torch_q": torch_q,
        "torch_cache": torch_cache,
        "position_ids": position_ids,
        "wh_cores_per_block": wh_cores_per_block,
        "grid_name": grid.__name__,
    }
    return tensors, metadata


def run_standalone_decode_wh(
    device: Any,
    *,
    batch_size: int,
    decode_position: int,
    max_seq_len: int,
    num_heads: int,
    num_q_heads_per_core: int,
    wh_cores_per_block: int,
) -> dict[str, Any]:
    if decode_position >= max_seq_len:
        raise ValueError(f"decode_position {decode_position} must be < max_seq_len {max_seq_len}")

    torch.manual_seed(0)

    tensors, metadata = build_standalone_wh_decode_inputs(
        device,
        batch_size=batch_size,
        decode_position=decode_position,
        k_chunk_size=128,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_q_heads_per_core=num_q_heads_per_core,
        wh_cores_per_block=wh_cores_per_block,
    )

    reference_output = FlashMLADecode.golden(
        q=metadata["torch_q"],
        kv_cache=metadata["torch_cache"],
        position_ids=metadata["position_ids"],
        head_dim_v=metadata["kv_lora_rank"],
        scale=metadata["scale"],
    )

    def _fail_wh_fallback(cls, *args, **kwargs):
        raise RuntimeError("WH fallback path was invoked during standalone_decode_wh smoke run")

    original_fallback = FlashMLADecode.__dict__["_wh_reference_fallback"]
    iteration_latencies_ms = []
    first_output = None
    try:
        FlashMLADecode._wh_reference_fallback = classmethod(_fail_wh_fallback)
        for _ in range(2):
            start = time.perf_counter()
            attn_out = FlashMLADecode.op(
                q_tensor=tensors["tt_q"],
                kv_cache_tensor=tensors["tt_cache"],
                head_dim_v=metadata["kv_lora_rank"],
                cur_pos_tensor=tensors["tt_position_ids"],
                output_tensor=tensors["tt_out"],
                scale=metadata["scale"],
                program_config=tensors["program_config"],
                compute_kernel_config=tensors["compute_kernel_config"],
            )
            ttnn.synchronize_device(device)
            iteration_latencies_ms.append((time.perf_counter() - start) * 1000.0)

            output_torch = ttnn.to_torch(attn_out)
            if first_output is None:
                pcc_required = 0.995
                passing, pcc_message = comp_pcc(reference_output, output_torch, pcc_required)
                if not passing:
                    raise AssertionError(f"standalone WH decode PCC check failed: {pcc_message}")
                first_output = output_torch.clone()
                max_abs_diff = torch.max(torch.abs(output_torch - reference_output)).item()
                mean_abs_diff = torch.mean(torch.abs(output_torch - reference_output)).item()
                reference_shape = tuple(reference_output.shape)
                output_shape = tuple(output_torch.shape)
            else:
                if not torch.equal(output_torch, first_output):
                    diff = (output_torch - first_output).abs().max().item()
                    raise AssertionError(f"standalone WH decode output changed across iterations, max diff={diff}")
    finally:
        FlashMLADecode._wh_reference_fallback = original_fallback
        deallocate_tensors(tensors, ("tt_q", "tt_cache", "tt_position_ids", "tt_out"))

    return {
        "case": "standalone_decode_wh",
        "status": "passed",
        "batch_size": batch_size,
        "grid": metadata["grid_name"],
        "decode_position": decode_position,
        "max_seq_len": max_seq_len,
        "output_shape": output_shape,
        "reference_shape": reference_shape,
        "pcc": pcc_message,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "iteration_latencies_ms": iteration_latencies_ms,
    }


def run_decode_smoke(device: Any, *, seq_len: int) -> dict[str, Any]:
    torch.manual_seed(0)
    start = time.perf_counter()
    run_flash_mla_decode_impl(
        device=device,
        batch=1,
        seq_len=seq_len,
        nh=32,
        nkv=1,
        kv_lora_rank=512,
        d_rope=64,
        q_num_cores=32,
        q_dtype=ttnn.bfloat16,
        q_mem_config=None,
        dtype=ttnn.bfloat8_b,
        use_paged_attention=True,
        block_size=64,
        reuse_k=False,
        max_cores_per_head_batch=4,
    )
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "case": "decode",
        "status": "passed",
        "seq_len": seq_len,
        "batch": 1,
        "nh": 32,
        "nkv": 1,
        "elapsed_ms": elapsed_ms,
        "path": "ttnn.transformer.paged_flash_multi_latent_attention_decode",
    }


def run_prefill_smoke(device: Any, *, seq_len: int) -> dict[str, Any]:
    torch.manual_seed(0)
    start = time.perf_counter()
    run_flash_mla_prefill_impl(
        device=device,
        batch=1,
        seq_len=seq_len,
        nh=32,
        nkv=1,
        kv_lora_rank=512,
        d_rope=64,
        q_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat8_b,
        use_paged_attention=True,
        block_size=64,
    )
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "case": "prefill",
        "status": "passed",
        "seq_len": seq_len,
        "batch": 1,
        "nh": 32,
        "nkv": 1,
        "elapsed_ms": elapsed_ms,
        "path": "ttnn.transformer.chunked_flash_mla_prefill",
    }


def write_summary(results: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "flash_mla_wh_smoke_results.json"
    md_path = OUTPUT_ROOT / "flash_mla_wh_smoke_report.md"

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# FlashMLA WH Smoke Results",
        "",
        f"- arch: `{results['metadata']['arch']}`",
        f"- cases: `{', '.join(results['metadata']['selected_cases'])}`",
        "",
        "| case | status | key result |",
        "|---|---|---|",
    ]

    for result in results["results"]:
        if result["status"] == "passed":
            if result["case"] == "standalone_decode_wh":
                key_result = (
                    f"pcc={result['pcc']}, max_abs_diff={result['max_abs_diff']:.6f}, "
                    f"latencies_ms={result['iteration_latencies_ms']}"
                )
            else:
                key_result = f"elapsed_ms={result['elapsed_ms']:.3f}, path={result['path']}"
        else:
            key_result = result["error"].splitlines()[-1]
        lines.append(f"| {result['case']} | {result['status']} | {key_result} |")

    with md_path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wormhole FlashMLA smoke coverage for standalone decode, decode, and prefill.")
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--standalone-max-seq-len", type=int, default=4096)
    parser.add_argument("--standalone-decode-position", type=int, default=2047)
    parser.add_argument("--standalone-batch-size", type=int, default=1)
    parser.add_argument("--standalone-num-heads", type=int, default=32)
    parser.add_argument("--standalone-num-q-heads-per-core", type=int, default=8)
    parser.add_argument("--wh-cores-per-block", type=int, choices=(4, 8), default=4)
    parser.add_argument("--decode-seq-len", type=int, default=4096)
    parser.add_argument("--prefill-seq-len", type=int, default=1024)
    args = parser.parse_args()

    case_runners = {
        "standalone_decode_wh": lambda device: run_standalone_decode_wh(
            device,
            batch_size=args.standalone_batch_size,
            decode_position=args.standalone_decode_position,
            max_seq_len=args.standalone_max_seq_len,
            num_heads=args.standalone_num_heads,
            num_q_heads_per_core=args.standalone_num_q_heads_per_core,
            wh_cores_per_block=args.wh_cores_per_block,
        ),
        "decode": lambda device: run_decode_smoke(device, seq_len=args.decode_seq_len),
        "prefill": lambda device: run_prefill_smoke(device, seq_len=args.prefill_seq_len),
    }

    device = ttnn.open_device(device_id=0)
    try:
        ensure_wormhole(device)
        results = {
            "metadata": {
                "arch": str(device.arch()),
                "selected_cases": args.cases,
                "wh_cores_per_block": args.wh_cores_per_block,
            },
            "results": [],
        }

        had_failure = False
        for case_name in args.cases:
            logger.info(f"Running WH smoke case: {case_name}")
            try:
                result = case_runners[case_name](device)
            except Exception:
                had_failure = True
                result = {
                    "case": case_name,
                    "status": "failed",
                    "error": traceback.format_exc(),
                }
            results["results"].append(result)

        write_summary(results)
        logger.info(json.dumps(results, indent=2))

        if had_failure:
            raise SystemExit(1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

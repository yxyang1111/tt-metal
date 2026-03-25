# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

"""
Test for flash_multi_latent_attention_decode op on Wormhole B0.

Validates the WH-specific S block layout (6 blocks × 4 cores) with 6 DRAM banks.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_wormhole_b0, run_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode,
    FlashMLAOptimalGridNOC0_WH,
    FlashMLAProgramConfig,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "decode_position",
    [
        # Aligned chunks (multiples of 128 - 1)
        127,
        255,
        511,
        1023,
        2047,
        # Unaligned chunks
        0,
        1,
        7,
        16,
        128,
        564,
        1203,
        2046,
    ],
)
@pytest.mark.parametrize("k_chunk_size", [128])
@pytest.mark.parametrize("max_seq_len", [32 * 1024])
def test_flash_mla_decode_wh(device, batch_size, decode_position, k_chunk_size, max_seq_len):
    """Test FlashMLADecode op on Wormhole B0 with 6-block S layout."""
    torch.manual_seed(0)

    optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    for bank_id, worker_core in enumerate(optimal_workers):
        logger.info(f"DRAM bank {bank_id} -> optimal worker core ({worker_core.x}, {worker_core.y})")

    # WH with TP=4 (T3K): 128 / 8 = 16 heads per device, or TP=2: 64 heads
    # Use 32 heads for a lighter WH config that fits 4 Q shards × 8 heads/core
    num_heads = 32
    num_q_heads_per_core = 8
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
    kvpe_dim = kv_lora_rank + qk_rope_head_dim  # 576
    scale = qk_head_dim**-0.5

    num_q_shards = num_heads // num_q_heads_per_core  # 4
    grid = FlashMLAOptimalGridNOC0_WH
    assert num_q_shards <= grid.CORES_PER_BLOCK, (
        f"num_q_shards ({num_q_shards}) must be <= cores_per_block ({grid.CORES_PER_BLOCK})"
    )

    logger.info(
        f"Testing FlashMLADecode (WH) with batch_size={batch_size}, k_chunk_size={k_chunk_size}, "
        f"decode_position={decode_position}, max_seq_len={max_seq_len}, "
        f"num_heads={num_heads}, num_q_shards={num_q_shards}"
    )

    tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

    # Q sharded onto S1 output cores (first S block)
    s1_cores, _ = grid.BLOCKS[0]
    q_cores = s1_cores[:num_q_shards]
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

    # Create Q tensor: [1, batch_size, num_heads, kvpe_dim]
    logger.info("Creating Q tensor...")
    q_shape = (1, batch_size, num_heads, kvpe_dim)
    torch_q = torch.randn(q_shape, dtype=torch.bfloat16)

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
    )

    # Create KV cache with ND sharding across 6 DRAM banks
    logger.info(f"Creating KV cache with seq_len={max_seq_len}...")
    cache_shape = (batch_size, 1, max_seq_len, kvpe_dim)
    torch_cache = torch.randn(cache_shape, dtype=torch.bfloat16)

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
    num_chunks_total = max_seq_len // program_config.k_chunk_size
    num_banks = len(grid.OPTIMAL_DRAM_BANK_ORDER)
    logger.info(
        f"KV cache: ND sharded, DRAM banks: {num_banks} (order: {grid.OPTIMAL_DRAM_BANK_ORDER}), "
        f"chunks: {num_chunks_total}, shard_shape: [1, 1, {program_config.k_chunk_size}, {kvpe_dim}]"
    )

    tt_cache = ttnn.from_torch(
        torch_cache,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=kv_mem_config,
    )

    # Position tensor replicated on every core
    grid_size = device.compute_with_storage_grid_size()
    position_ids = torch.ones(batch_size, dtype=torch.int32) * decode_position
    position_replicated = position_ids.repeat(grid_size.x * grid_size.y, 1)
    pos_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )
    pos_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_position_ids = ttnn.from_torch(
        position_replicated,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=pos_mem_config,
    )

    # Create output tensor
    logger.info("Creating output tensor...")
    out_shape = (1, batch_size, num_heads, kv_lora_rank)
    torch_output_zeros = torch.zeros(out_shape, dtype=torch.bfloat16)
    tt_out = ttnn.from_torch(
        torch_output_zeros,
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

    # Golden reference
    logger.info("Computing PyTorch reference...")
    reference_output = FlashMLADecode.golden(
        q=torch_q,
        kv_cache=torch_cache,
        position_ids=position_ids,
        head_dim_v=kv_lora_rank,
        scale=scale,
    )

    # Stress test with multiple iterations
    num_iterations = 10
    first_output = None
    logger.info(f"Running FlashMLADecode.op {num_iterations} times for stress test...")
    for i in range(num_iterations):
        if i % 10 == 0:
            logger.info(f"  Iteration {i}/{num_iterations}...")
        attn_out = FlashMLADecode.op(
            q_tensor=tt_q,
            kv_cache_tensor=tt_cache,
            head_dim_v=kv_lora_rank,
            cur_pos_tensor=tt_position_ids,
            output_tensor=tt_out,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        output_torch = ttnn.to_torch(attn_out)

        expected_shape = (1, batch_size, num_heads, kv_lora_rank)
        assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"
        assert not torch.isnan(output_torch).any(), f"Iteration {i}: Output contains NaN values"
        assert not torch.all(output_torch == 0), f"Iteration {i}: Output is all zeros"

        if i == 0:
            has_inf_out = torch.isinf(output_torch).any().item()
            has_inf_ref = torch.isinf(reference_output).any().item()
            logger.info(f"Output has inf: {has_inf_out}, count: {torch.isinf(output_torch).sum().item()}")
            logger.info(f"Reference has inf: {has_inf_ref}")
            logger.info(f"Output range: [{output_torch.min().item()}, {output_torch.max().item()}]")
            logger.info(f"Reference range: [{reference_output.min().item()}, {reference_output.max().item()}]")
            logger.info(f"Output[0,0,0,:8]: {output_torch[0,0,0,:8]}")
            logger.info(f"Reference[0,0,0,:8]: {reference_output[0,0,0,:8]}")
            if has_inf_out:
                inf_mask = torch.isinf(output_torch)
                inf_indices = torch.nonzero(inf_mask)
                logger.info(f"First 20 inf positions: {inf_indices[:20]}")
                logger.info(f"Inf values: {output_torch[inf_mask][:20]}")
            out_max_diff = torch.max(torch.abs(output_torch - reference_output)).item()
            out_mean_diff = torch.mean(torch.abs(output_torch - reference_output)).item()
            logger.info(f"Out Max absolute difference: {out_max_diff}")
            logger.info(f"Out Mean absolute difference: {out_mean_diff}")
            pcc_required = 0.995
            passing, pcc_message = comp_pcc(reference_output, output_torch, pcc_required)
            logger.info(f"    PCC vs golden: {pcc_message}")
            first_output = output_torch.clone()
        else:
            assert torch.equal(output_torch, first_output), (
                f"Iteration {i}: Output differs from first iteration! "
                f"Max diff: {(output_torch - first_output).abs().max().item()}"
            )

    logger.info(f"  Completed {num_iterations} iterations!")
    logger.info("FlashMLADecode WH test passed!")


@run_for_wormhole_b0()
def test_flash_mla_wh_grid_layout(device):
    """Validate WH S block grid layout: no core overlaps, valid rectangles."""
    grid = FlashMLAOptimalGridNOC0_WH

    # Verify no core overlaps across blocks
    all_cores = set()
    for s_idx in range(grid.NUM_BLOCKS):
        cores = grid.get_cores(s_idx)
        for x, y in cores:
            assert (x, y) not in all_cores, f"S block {s_idx}: core ({x},{y}) already used by another block"
            all_cores.add((x, y))

    logger.info(f"Total active cores: {len(all_cores)}")
    assert len(all_cores) == grid.NUM_BLOCKS * grid.CORES_PER_BLOCK

    # Verify multicast coordinates resolve to valid physical rectangles
    for s_idx in range(grid.NUM_BLOCKS):
        start_x, start_y, end_x, end_y, num_dests = grid.physical_multicast_coords(device, s_idx)
        logger.info(
            f"S{s_idx+1}: mcast phys ({start_x},{start_y})->({end_x},{end_y}), "
            f"num_dests={num_dests}, bank={grid.BLOCKS[s_idx][1]}"
        )
        assert num_dests == grid.CORES_PER_BLOCK - 1

    # Verify tree reduction covers all blocks
    for s_idx in range(grid.NUM_BLOCKS):
        roles = grid.get_tree_reduction_role(s_idx)
        logger.info(f"S{s_idx+1} tree roles: {roles}")

    # S1 (idx 0) must be the final receiver
    assert grid.is_tree_reduction_receiver(0)
    assert not grid.is_tree_reduction_sender(0)

    logger.info("WH grid layout validation passed!")

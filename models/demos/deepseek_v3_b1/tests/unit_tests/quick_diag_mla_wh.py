"""Quick diagnostic: identify which output tiles have garbage values."""
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode, FlashMLAOptimalGridNOC0_WH, FlashMLAProgramConfig,
)

def main():
    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0)

    batch_size = 1
    decode_position = 127
    k_chunk_size = 128
    max_seq_len = 32 * 1024
    num_heads = 32
    num_q_heads_per_core = 8
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kvpe_dim = kv_lora_rank + qk_rope_head_dim
    scale = qk_head_dim**-0.5

    num_q_shards = num_heads // num_q_heads_per_core
    grid = FlashMLAOptimalGridNOC0_WH
    tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

    s1_cores, _ = grid.BLOCKS[0]
    q_cores = s1_cores[:num_q_shards]
    q_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in q_cores]
    )
    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch_q = torch.randn((1, batch_size, num_heads, kvpe_dim), dtype=torch.bfloat16)
    tt_q = ttnn.from_torch(torch_q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=q_mem_config, tile=tiny_tile)

    program_config = FlashMLAProgramConfig(k_chunk_size=k_chunk_size, exp_approx_mode=False,
                                           grid=grid, allow_wh_fallback=False)

    torch_cache = torch.randn((batch_size, 1, max_seq_len, kvpe_dim), dtype=torch.bfloat16)
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, k_chunk_size, kvpe_dim],
        grid=grid.optimal_dram_grid(),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kv_nd_shard_spec)
    tt_cache = ttnn.from_torch(torch_cache, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=kv_mem_config)

    grid_size = device.compute_with_storage_grid_size()
    position_ids = torch.ones(batch_size, dtype=torch.int32) * decode_position
    position_replicated = position_ids.repeat(grid_size.x * grid_size.y, 1)
    pos_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )
    pos_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_position_ids = ttnn.from_torch(position_replicated, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=device, memory_config=pos_mem_config)

    torch_output_zeros = torch.zeros((1, batch_size, num_heads, kv_lora_rank), dtype=torch.bfloat16)
    tt_out = ttnn.from_torch(torch_output_zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=out_mem_config, tile=tiny_tile)

    compute_kernel_config = ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=False, dst_full_sync_en=True,
    )

    reference_output = FlashMLADecode.golden(q=torch_q, kv_cache=torch_cache,
                                             position_ids=position_ids, head_dim_v=kv_lora_rank, scale=scale)

    attn_out = FlashMLADecode.op(
        q_tensor=tt_q, kv_cache_tensor=tt_cache, head_dim_v=kv_lora_rank,
        cur_pos_tensor=tt_position_ids, output_tensor=tt_out, scale=scale,
        program_config=program_config, compute_kernel_config=compute_kernel_config,
    )
    output_torch = ttnn.to_torch(attn_out)

    logger.info(f"Output shape: {output_torch.shape}")
    logger.info(f"Output range: [{output_torch.min().item():.6e}, {output_torch.max().item():.6e}]")
    logger.info(f"Reference range: [{reference_output.min().item():.6e}, {reference_output.max().item():.6e}]")

    # Per-head analysis
    for h in range(num_heads):
        out_h = output_torch[0, 0, h, :]
        ref_h = reference_output[0, 0, h, :]
        max_abs = out_h.abs().max().item()
        ref_max_abs = ref_h.abs().max().item()
        logger.info(f"Head {h:2d}: out_max_abs={max_abs:.6e}, ref_max_abs={ref_max_abs:.6e}, "
                     f"out_range=[{out_h.min().item():.4e},{out_h.max().item():.4e}]")

    # Per-tile analysis (each tile = 32 output dimensions)
    logger.info("\n--- Per-tile analysis (8 heads × 16 tiles) ---")
    for shard in range(num_q_shards):
        head_start = shard * num_q_heads_per_core
        head_end = head_start + num_q_heads_per_core
        for tile_idx in range(kv_lora_rank // 32):
            dim_start = tile_idx * 32
            dim_end = dim_start + 32
            out_tile = output_torch[0, 0, head_start:head_end, dim_start:dim_end]
            ref_tile = reference_output[0, 0, head_start:head_end, dim_start:dim_end]
            max_abs = out_tile.abs().max().item()
            if max_abs > 100:
                logger.warning(f"  Shard {shard}, Tile {tile_idx} (dims {dim_start}-{dim_end}): "
                               f"LARGE max_abs={max_abs:.4e}")
            else:
                _, pcc_msg = comp_pcc(ref_tile, out_tile, 0.9)
                logger.info(f"  Shard {shard}, Tile {tile_idx} (dims {dim_start}-{dim_end}): "
                            f"max_abs={max_abs:.4e}, pcc={pcc_msg}")

    # Check face pattern (first 16 dims = face0, next 16 = face1)
    logger.info("\n--- Face analysis (face0=dims[0:16], face1=dims[16:32]) ---")
    for shard in range(num_q_shards):
        head_start = shard * num_q_heads_per_core
        head_end = head_start + num_q_heads_per_core
        for tile_idx in range(min(4, kv_lora_rank // 32)):
            dim_start = tile_idx * 32
            face0 = output_torch[0, 0, head_start:head_end, dim_start:dim_start+16]
            face1 = output_torch[0, 0, head_start:head_end, dim_start+16:dim_start+32]
            logger.info(f"  Shard {shard}, Tile {tile_idx}: "
                        f"face0_max={face0.abs().max().item():.4e}, "
                        f"face1_max={face1.abs().max().item():.4e}")

    ttnn.close_device(device)

if __name__ == "__main__":
    main()

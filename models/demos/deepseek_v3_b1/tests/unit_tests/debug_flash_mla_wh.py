#!/usr/bin/env python3
"""Minimal debug script for FlashMLADecode on WH — isolate the failure."""

import torch
import ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode,
    FlashMLAOptimalGridNOC0_WH,
    FlashMLAProgramConfig,
    get_noc_max_page_size,
    get_max_page_size_and_num_pages,
)

torch.manual_seed(0)

device = ttnn.open_device(device_id=0)
grid_size = device.compute_with_storage_grid_size()
logger.info(f"Device grid: {grid_size.x}x{grid_size.y}")
arch_name = ttnn.get_arch_name()
logger.info(f"Architecture: {arch_name}")

decode_position = 127
batch_size = 1
k_chunk_size = 128
max_seq_len = 2048  # smaller for faster debug

num_heads = 32
num_q_heads_per_core = 8
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
kvpe_dim = kv_lora_rank + qk_rope_head_dim  # 576
scale = qk_head_dim**-0.5

num_q_shards = num_heads // num_q_heads_per_core  # 4
grid = FlashMLAOptimalGridNOC0_WH

# Print key computed values
Sk_chunk_t = k_chunk_size // 32
DHt = kvpe_dim // 32
vDHt = kv_lora_rank // 32
k_chunk_tiles = Sk_chunk_t * DHt
logger.info(f"Sk_chunk_t={Sk_chunk_t}, DHt={DHt}, vDHt={vDHt}, k_chunk_tiles={k_chunk_tiles}")

# Compute page size for WH
noc_max = get_noc_max_page_size(arch_name)
logger.info(f"NOC max page size: {noc_max}")

# We need to know the k_tile_size to calculate pages
# bfloat8_b 32x32 tile: approximately 1088 bytes (1024 data + 64 exponent)
# Let's calculate after creating the tensor to get the exact tile size

tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))

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

torch_q = torch.randn((1, batch_size, num_heads, kvpe_dim), dtype=torch.bfloat16)
tt_q = ttnn.from_torch(torch_q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=q_mem_config, tile=tiny_tile)

program_config = FlashMLAProgramConfig(
    k_chunk_size=k_chunk_size, exp_approx_mode=False,
    grid=grid, allow_wh_fallback=False,
)

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
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False,
)

# Golden reference
reference_output = FlashMLADecode.golden(
    q=torch_q, kv_cache=torch_cache, position_ids=position_ids,
    head_dim_v=kv_lora_rank, scale=scale,
)

# Also test the fallback path
logger.info("=== Testing WH fallback path ===")
program_config_fallback = FlashMLAProgramConfig(
    k_chunk_size=k_chunk_size, exp_approx_mode=False,
    grid=grid, allow_wh_fallback=True,
)
fallback_out = FlashMLADecode.op(
    q_tensor=tt_q, kv_cache_tensor=tt_cache, head_dim_v=kv_lora_rank,
    cur_pos_tensor=tt_position_ids, output_tensor=tt_out, scale=scale,
    program_config=program_config_fallback, compute_kernel_config=compute_kernel_config,
)
fallback_torch = ttnn.to_torch(fallback_out)
passing_fb, pcc_fb = comp_pcc(reference_output, fallback_torch, 0.99)
logger.info(f"Fallback PCC vs golden: {pcc_fb} (pass={passing_fb})")

# Recreate output tensor (fallback may have changed it)
tt_out = ttnn.from_torch(torch_output_zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=device, memory_config=out_mem_config, tile=tiny_tile)

# Now test real kernel
logger.info("=== Testing real WH kernel ===")
program_config_real = FlashMLAProgramConfig(
    k_chunk_size=k_chunk_size, exp_approx_mode=False,
    grid=grid, allow_wh_fallback=False,
)
attn_out = FlashMLADecode.op(
    q_tensor=tt_q, kv_cache_tensor=tt_cache, head_dim_v=kv_lora_rank,
    cur_pos_tensor=tt_position_ids, output_tensor=tt_out, scale=scale,
    program_config=program_config_real, compute_kernel_config=compute_kernel_config,
)
output_torch = ttnn.to_torch(attn_out)

logger.info(f"Output shape: {output_torch.shape}")
logger.info(f"Output dtype: {output_torch.dtype}")
has_inf = torch.isinf(output_torch).any().item()
has_nan = torch.isnan(output_torch).any().item()
logger.info(f"Has inf: {has_inf} (count: {torch.isinf(output_torch).sum().item()})")
logger.info(f"Has nan: {has_nan}")
logger.info(f"Output range: [{output_torch.min().item():.4f}, {output_torch.max().item():.4f}]")
logger.info(f"Reference range: [{reference_output.min().item():.4f}, {reference_output.max().item():.4f}]")

# Per-head analysis
for h in range(min(4, num_heads)):
    out_h = output_torch[0, 0, h, :]
    ref_h = reference_output[0, 0, h, :]
    inf_count = torch.isinf(out_h).sum().item()
    if inf_count > 0:
        first_inf = torch.nonzero(torch.isinf(out_h))[0].item()
    else:
        first_inf = -1
    p, m = comp_pcc(ref_h.unsqueeze(0), out_h.unsqueeze(0), 0.9)
    logger.info(f"  Head {h}: inf_count={inf_count}, first_inf_pos={first_inf}, pcc={m}")

# Detailed diff for head 0
logger.info(f"Output[0,0,0,:16]:    {output_torch[0,0,0,:16].tolist()}")
logger.info(f"Reference[0,0,0,:16]: {reference_output[0,0,0,:16].tolist()}")
logger.info(f"Output[0,0,0,490:512]:    {output_torch[0,0,0,490:512].tolist()}")
logger.info(f"Reference[0,0,0,490:512]: {reference_output[0,0,0,490:512].tolist()}")

passing, pcc_message = comp_pcc(reference_output, output_torch, 0.995)
logger.info(f"Overall PCC: {pcc_message} (pass={passing})")

ttnn.close_device(device)

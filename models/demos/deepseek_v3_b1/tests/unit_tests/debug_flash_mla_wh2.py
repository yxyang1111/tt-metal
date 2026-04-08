#!/usr/bin/env python3
"""Focused debug: verify ND shard roundtrip + identify broken step in FlashMLA WH."""

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

grid = FlashMLAOptimalGridNOC0_WH
batch_size = 1
k_chunk_size = 128
max_seq_len = 2048
kvpe_dim = 576
kv_lora_rank = 512

# ---- Step 1: Verify ND shard roundtrip ----
logger.info("=== Step 1: ND shard roundtrip ===")
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
roundtrip = ttnn.to_torch(tt_cache).to(torch.bfloat16)
p, m = comp_pcc(torch_cache, roundtrip, 0.99)
logger.info(f"KV cache roundtrip PCC: {m} (pass={p})")
logger.info(f"  Original range:  [{torch_cache.min():.4f}, {torch_cache.max():.4f}]")
logger.info(f"  Roundtrip range:  [{roundtrip.min():.4f}, {roundtrip.max():.4f}]")
logger.info(f"  Max diff: {(torch_cache - roundtrip).abs().max():.6f}")

# Verify specific chunks
for chunk_id in [0, 1, 5, 6]:
    start = chunk_id * k_chunk_size
    end = start + k_chunk_size
    orig_chunk = torch_cache[0, 0, start:end, :]
    rt_chunk = roundtrip[0, 0, start:end, :]
    p_c, m_c = comp_pcc(orig_chunk, rt_chunk, 0.99)
    logger.info(f"  Chunk {chunk_id} (rows {start}-{end}): PCC={m_c}")

# ---- Step 2: Compute k_page_size and verify ----
logger.info("=== Step 2: Page size computation ===")
Sk_chunk_t = k_chunk_size // 32
DHt = kvpe_dim // 32
vDHt = kv_lora_rank // 32
k_chunk_tiles = Sk_chunk_t * DHt
arch_name = ttnn.get_arch_name()
noc_max = get_noc_max_page_size(arch_name)

k_tile = ttnn.Tile((32, 32))
k_tile_size = k_tile.get_tile_size(ttnn.bfloat8_b)
logger.info(f"k_tile_size (bfloat8_b 32x32): {k_tile_size} bytes")
logger.info(f"k_chunk_tiles: {k_chunk_tiles}, total bytes: {k_chunk_tiles * k_tile_size}")
k_page_size, k_num_pages = get_max_page_size_and_num_pages(noc_max, k_chunk_tiles, k_tile_size)
logger.info(f"k_page_size: {k_page_size}, k_num_pages: {k_num_pages}")
logger.info(f"Verify: {k_page_size} * {k_num_pages} = {k_page_size * k_num_pages} == {k_chunk_tiles * k_tile_size}")

# ---- Step 3: Run with num_heads=8 (B=1, no Q multicast within S block) ----
logger.info("=== Step 3: Minimal config (num_heads=8, B=1) ===")
num_heads = 8
num_q_heads_per_core = 8
scale = (128 + 64)**-0.5

tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))
s1_cores, _ = grid.BLOCKS[0]
q_cores = s1_cores[:1]  # Only 1 Q shard
q_core_grid = ttnn.CoreRangeSet(
    [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in q_cores]
)
q_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR))
out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(q_core_grid, (num_q_heads_per_core, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR))

torch_q_small = torch.randn((1, 1, num_heads, kvpe_dim), dtype=torch.bfloat16)
tt_q_small = ttnn.from_torch(torch_q_small, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=q_mem, tile=tiny_tile)

position_ids = torch.tensor([127], dtype=torch.int32)
position_replicated = position_ids.repeat(grid_size.x * grid_size.y, 1)
pos_core_grid = ttnn.CoreRangeSet(
    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
)
pos_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(pos_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR))
tt_pos = ttnn.from_torch(position_replicated, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=device, memory_config=pos_mem)

tt_out_small = ttnn.from_torch(
    torch.zeros((1, 1, num_heads, kv_lora_rank), dtype=torch.bfloat16),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=out_mem, tile=tiny_tile)

compute_cfg = ttnn.types.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=False)

prog = FlashMLAProgramConfig(k_chunk_size=k_chunk_size, exp_approx_mode=False,
                              grid=grid, allow_wh_fallback=False)

ref_small = FlashMLADecode.golden(q=torch_q_small, kv_cache=torch_cache,
                                   position_ids=position_ids, head_dim_v=kv_lora_rank, scale=scale)

out_small = FlashMLADecode.op(q_tensor=tt_q_small, kv_cache_tensor=tt_cache, head_dim_v=kv_lora_rank,
    cur_pos_tensor=tt_pos, output_tensor=tt_out_small, scale=scale,
    program_config=prog, compute_kernel_config=compute_cfg)
out_torch_small = ttnn.to_torch(out_small)

inf_cnt = torch.isinf(out_torch_small).sum().item()
p_s, m_s = comp_pcc(ref_small, out_torch_small, 0.99)
logger.info(f"Minimal config (B=1): inf_count={inf_cnt}, PCC={m_s} (pass={p_s})")
logger.info(f"  Output range: [{out_torch_small.min():.4f}, {out_torch_small.max():.4f}]")
logger.info(f"  Ref range: [{ref_small.min():.4f}, {ref_small.max():.4f}]")
if inf_cnt > 0:
    first_inf = torch.nonzero(torch.isinf(out_torch_small.flatten()))[0].item()
    logger.info(f"  First inf at flat position: {first_inf}")
logger.info(f"  Out[:8]:  {out_torch_small[0,0,0,:8].tolist()}")
logger.info(f"  Ref[:8]:  {ref_small[0,0,0,:8].tolist()}")
logger.info(f"  Out[490:512]:  {out_torch_small[0,0,0,490:512].tolist()}")
logger.info(f"  Ref[490:512]:  {ref_small[0,0,0,490:512].tolist()}")

ttnn.close_device(device)

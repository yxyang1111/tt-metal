"""Analyze inf pattern in FlashMLA output to debug WH compute issue."""
import torch
import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode, FlashMLAOptimalGridNOC0_WH, FlashMLAProgramConfig,
)

torch.manual_seed(0)
device = ttnn.open_device(device_id=0)
gs = device.compute_with_storage_grid_size()
grid = FlashMLAOptimalGridNOC0_WH
k_chunk_size = 128
max_seq_len = 2048
num_heads = 32
nqhpc = 8
kv_lora_rank = 512
kvpe_dim = 576
scale = (128 + 64) ** -0.5
nqs = num_heads // nqhpc
tiny = ttnn.Tile((nqhpc, 32))
s1_cores, _ = grid.BLOCKS[0]
qc = s1_cores[:nqs]
qcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in qc])
qm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(qcrs, (nqhpc, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR))
om = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(qcrs, (nqhpc, kv_lora_rank), ttnn.ShardOrientation.ROW_MAJOR))
tq = torch.randn((1, 1, num_heads, kvpe_dim), dtype=torch.bfloat16)
tt_q = ttnn.from_torch(tq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=qm, tile=tiny)
tc = torch.randn((1, 1, max_seq_len, kvpe_dim), dtype=torch.bfloat16)
kvnd = ttnn.NdShardSpec(shard_shape=[1, 1, k_chunk_size, kvpe_dim], grid=grid.optimal_dram_grid(),
    orientation=ttnn.ShardOrientation.ROW_MAJOR, shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
kvm = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kvnd)
tt_c = ttnn.from_torch(tc, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=kvm)
pcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gs.x - 1, gs.y - 1))])
pm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(pcrs, (1, 1), ttnn.ShardOrientation.ROW_MAJOR))
cc = ttnn.types.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=False, dst_full_sync_en=True)

pos = 127
pids = torch.tensor([pos], dtype=torch.int32)
pr = pids.repeat(gs.x * gs.y, 1)
tp = ttnn.from_torch(pr, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=pm)
to_k = ttnn.from_torch(torch.zeros((1, 1, num_heads, kv_lora_rank), dtype=torch.bfloat16),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=om, tile=tiny)
prog = FlashMLAProgramConfig(k_chunk_size=k_chunk_size, exp_approx_mode=False, grid=grid, allow_wh_fallback=False)
out = FlashMLADecode.op(q_tensor=tt_q, kv_cache_tensor=tt_c, head_dim_v=kv_lora_rank,
    cur_pos_tensor=tp, output_tensor=to_k, scale=scale, program_config=prog, compute_kernel_config=cc)
out_t = ttnn.to_torch(out)

print("=== Inf pattern analysis for pos=127 ===")
h0 = out_t[0, 0, 0, :]
inf_mask = torch.isinf(h0)
inf_positions = torch.nonzero(inf_mask).squeeze()
print(f"Head 0: shape={h0.shape}, num_inf={inf_mask.sum().item()}")
print(f"Inf positions: {inf_positions.tolist()}")

for tile_idx in range(16):
    tile_start = tile_idx * 32
    tile_end = tile_start + 32
    tile_infs = inf_mask[tile_start:tile_end].sum().item()
    if tile_infs > 0:
        tile_inf_cols = torch.nonzero(inf_mask[tile_start:tile_end]).squeeze().tolist()
        if isinstance(tile_inf_cols, int):
            tile_inf_cols = [tile_inf_cols]
        print(f"  Tile {tile_idx} (cols {tile_start}-{tile_end-1}): {tile_infs} infs at local cols {tile_inf_cols}")

print("\nSame inf pattern across heads 0-7 (shard 0)?")
for h in range(8):
    hx = out_t[0, 0, h, :]
    inf_p = torch.nonzero(torch.isinf(hx)).squeeze().tolist()
    if isinstance(inf_p, int):
        inf_p = [inf_p]
    print(f"  Head {h}: {len(inf_p)} infs at {inf_p}")

print("\nSame inf pattern across heads 8-15 (shard 1)?")
for h in range(8, 16):
    hx = out_t[0, 0, h, :]
    inf_p = torch.nonzero(torch.isinf(hx)).squeeze().tolist()
    if isinstance(inf_p, int):
        inf_p = [inf_p]
    print(f"  Head {h}: {len(inf_p)} infs at {inf_p}")

ttnn.close_device(device)

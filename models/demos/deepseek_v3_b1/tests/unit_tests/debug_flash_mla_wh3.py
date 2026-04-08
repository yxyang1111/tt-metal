"""Detailed diagnostic for FlashMLA WH inf pattern at positions 497-511."""
import torch
import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import (
    FlashMLADecode, FlashMLAOptimalGridNOC0_WH, FlashMLAProgramConfig,
)

torch.manual_seed(42)
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
kvnd = ttnn.NdShardSpec(shard_shape=[1, 1, k_chunk_size, kvpe_dim], grid=grid.optimal_dram_grid(),
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D)
kvm = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kvnd)
pcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gs.x - 1, gs.y - 1))])
pm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(pcrs, (1, 1), ttnn.ShardOrientation.ROW_MAJOR))
cc = ttnn.types.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=False, dst_full_sync_en=True)
prog = FlashMLAProgramConfig(k_chunk_size=k_chunk_size, exp_approx_mode=False, grid=grid, allow_wh_fallback=False)


def run_test(label, torch_q, torch_cache, pos):
    print(f"\n{'='*60}")
    print(f"TEST: {label} (pos={pos})")
    print(f"{'='*60}")
    batch = torch_q.shape[1]
    tt_q = ttnn.from_torch(torch_q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=qm, tile=tiny)
    tt_c = ttnn.from_torch(torch_cache, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=kvm)
    pids = torch.tensor([pos], dtype=torch.int32)
    pr = pids.repeat(gs.x * gs.y, 1)
    tp = ttnn.from_torch(pr, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=pm)
    to_k = ttnn.from_torch(torch.zeros((1, batch, num_heads, kv_lora_rank), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=om, tile=tiny)
    out = FlashMLADecode.op(q_tensor=tt_q, kv_cache_tensor=tt_c, head_dim_v=kv_lora_rank,
        cur_pos_tensor=tp, output_tensor=to_k, scale=scale, program_config=prog, compute_kernel_config=cc)
    out_t = ttnn.to_torch(out)
    ttnn.deallocate(tt_q)
    ttnn.deallocate(tt_c)
    ttnn.deallocate(tp)

    h0 = out_t[0, 0, 0, :]
    print(f"\nHead 0 stats: min={h0.min():.6f}, max={h0.max():.6f}, mean={h0.mean():.6f}")
    print(f"  num_inf={torch.isinf(h0).sum().item()}, num_nan={torch.isnan(h0).sum().item()}")

    posinf = (h0 == float('inf')).sum().item()
    neginf = (h0 == float('-inf')).sum().item()
    print(f"  +inf count={posinf}, -inf count={neginf}")

    print(f"\nHead 0 tile breakdown (last 3 tiles):")
    for tile_idx in [13, 14, 15]:
        s, e = tile_idx * 32, (tile_idx + 1) * 32
        t = h0[s:e]
        f0 = t[:16]
        f1 = t[16:]
        print(f"  Tile {tile_idx} (cols {s}-{e-1}):")
        print(f"    Face 0: min={f0.min():.6f} max={f0.max():.6f} inf={torch.isinf(f0).sum().item()}")
        print(f"    Face 1: min={f1.min():.6f} max={f1.max():.6f} inf={torch.isinf(f1).sum().item()}")
        if torch.isinf(t).any():
            print(f"    Face 1 values: {f1.tolist()}")

    print(f"\nHead 0 first tile values (tile 0):")
    t0 = h0[:32]
    print(f"  Face 0[:4]: {t0[:4].tolist()}")
    print(f"  Face 1[:4]: {t0[16:20].tolist()}")

    all_heads_inf = []
    for h in range(min(num_heads, 32)):
        hx = out_t[0, 0, h, :]
        inf_count = torch.isinf(hx).sum().item()
        all_heads_inf.append(inf_count)
    print(f"\nInf count per head: {all_heads_inf}")

    ref = FlashMLADecode.golden(q=torch_q, kv_cache=torch_cache,
        position_ids=torch.tensor([pos] * batch), head_dim_v=kv_lora_rank, scale=scale)
    finite_mask = torch.isfinite(out_t)
    if finite_mask.any():
        pcc = torch.corrcoef(torch.stack([out_t[finite_mask].float(), ref[finite_mask].float()]))[0, 1].item()
        print(f"\nPCC (finite elements only): {pcc:.6f}")
    else:
        print("\nNo finite elements to compute PCC")

    return out_t


# Test 1: Random data, pos=127
tq = torch.randn((1, 1, num_heads, kvpe_dim), dtype=torch.bfloat16)
tc = torch.randn((1, 1, max_seq_len, kvpe_dim), dtype=torch.bfloat16)
run_test("Random Q, Random KV, B=1", tq, tc, 127)

# Test 2: Zero Q (uniform softmax), Random KV
tq_zero = torch.zeros((1, 1, num_heads, kvpe_dim), dtype=torch.bfloat16)
run_test("Zero Q (uniform softmax), Random KV, B=1", tq_zero, tc, 127)

# Test 3: Random Q, Constant KV=1.0
tc_ones = torch.ones((1, 1, max_seq_len, kvpe_dim), dtype=torch.bfloat16)
run_test("Random Q, KV=ones, B=1", tq, tc_ones, 127)

# Test 4: Very small pos (pos=3)
run_test("Random Q, Random KV, pos=3, B=1", tq, tc, 3)

ttnn.close_device(device)

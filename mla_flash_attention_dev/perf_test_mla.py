"""
MLA (Multi-Latent Attention) Performance Benchmark on Wormhole B0

Measures wall-clock latency for flash_mla_prefill and paged_flash_multi_latent_attention_decode
across multiple workload configurations representative of DeepSeek V3.
"""

import time
import math
import json
import torch
import numpy as np
from loguru import logger

import ttnn
from models.common.utility_functions import nearest_y
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    page_table_setup,
    to_paged_cache,
    nearest_n,
    nearest_pow_2,
)


def benchmark_mla_prefill(device, batch, seq_len, nh, nkv, kv_lora_rank, d_rope,
                          q_dtype, kv_dtype, is_causal, num_warmup=2, num_iters=5):
    """Benchmark flash_mla_prefill and return timing + metadata."""

    d_qk = kv_lora_rank + d_rope
    q = torch.randn(batch, nh, seq_len, d_qk).float()
    k = torch.randn(batch, nkv, seq_len, d_qk).float()

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    q_chunk_size = padded_num_heads
    k_chunk_size = 128
    scale = d_qk ** -0.5

    sdpa_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(q, device=device, dtype=q_dtype,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_k = ttnn.from_torch(k, device=device, dtype=kv_dtype,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Warmup
    for _ in range(num_warmup):
        tt_out = ttnn.transformer.flash_mla_prefill(
            tt_q, tt_k, head_dim_v=kv_lora_rank, scale=scale,
            program_config=sdpa_config, compute_kernel_config=compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, attn_mask=None, is_causal=is_causal,
        )
        ttnn.synchronize_device(device)

    # Benchmark
    latencies_ms = []
    for _ in range(num_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        tt_out = ttnn.transformer.flash_mla_prefill(
            tt_q, tt_k, head_dim_v=kv_lora_rank, scale=scale,
            program_config=sdpa_config, compute_kernel_config=compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, attn_mask=None, is_causal=is_causal,
        )
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    ttnn.deallocate(tt_q)
    ttnn.deallocate(tt_k)
    ttnn.deallocate(tt_out)

    return latencies_ms


def benchmark_mla_decode(device, batch, seq_len, nh, nkv, kv_lora_rank, d_rope,
                         q_dtype, kv_dtype, block_size=64, num_warmup=2, num_iters=10):
    """Benchmark paged_flash_multi_latent_attention_decode and return timing + metadata."""

    d_qk = kv_lora_rank + d_rope
    q = torch.randn(batch, nh, 1, d_qk).float()
    k = torch.randn(batch, nkv, seq_len, d_qk).float()
    v = k[..., :kv_lora_rank]

    q_for_tt = q.permute(2, 0, 1, 3)  # [1, B, NH, D]

    max_num_blocks = seq_len // block_size * batch
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    page_table = page_table_setup(batch, paged_cfg)
    tt_k_torch = to_paged_cache(k, page_table, paged_cfg)

    q_chunk_size = 0
    k_chunk_size = 128
    scale = d_qk ** -0.5

    max_start_idx = seq_len // 2
    start_indices = np.linspace(0, max_start_idx, batch, dtype=np.int32).tolist() if batch > 1 else [max_start_idx]
    padded_layer_len = nearest_y(max_start_idx + 1, k_chunk_size)

    grid_size = device.compute_with_storage_grid_size()
    q_num_cores = min(batch * nh, grid_size.x * grid_size.y)
    block_height = nearest_y(np.prod(q.shape[:-1]) // q_num_cores, ttnn.TILE_SIZE)
    q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, q.shape[-1]),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, v.shape[-1]),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    sdpa_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=4,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(q_for_tt, device=device, dtype=q_dtype,
                           layout=ttnn.TILE_LAYOUT, memory_config=q_mem_config)
    tt_k = ttnn.from_torch(tt_k_torch, device=device, dtype=kv_dtype,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_start_indices = ttnn.from_torch(torch.tensor(start_indices), device=device, dtype=ttnn.int32)

    # Warmup
    for _ in range(num_warmup):
        tt_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q, tt_k, page_table_tensor=tt_page_table, cur_pos_tensor=tt_start_indices,
            head_dim_v=kv_lora_rank, scale=scale,
            program_config=sdpa_config, compute_kernel_config=compute_config,
            memory_config=out_mem_config,
        )
        ttnn.synchronize_device(device)

    # Benchmark
    latencies_ms = []
    for _ in range(num_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        tt_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q, tt_k, page_table_tensor=tt_page_table, cur_pos_tensor=tt_start_indices,
            head_dim_v=kv_lora_rank, scale=scale,
            program_config=sdpa_config, compute_kernel_config=compute_config,
            memory_config=out_mem_config,
        )
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    ttnn.deallocate(tt_q)
    ttnn.deallocate(tt_k)
    ttnn.deallocate(tt_out)

    return latencies_ms


def compute_theoretical_metrics(mode, batch, seq_len, nh, nkv, kv_lora_rank, d_rope,
                                is_causal, num_cores, kv_dtype_bytes):
    """Compute theoretical FLOPs, memory bytes, and ideal time."""

    d_qk = kv_lora_rank + d_rope
    d_v = kv_lora_rank

    if mode == "prefill":
        Sq = seq_len
        Sk = seq_len
    else:  # decode
        Sq = 1
        Sk = seq_len // 2  # effective KV length (midpoint start_idx)

    # FLOPs: Q*K^T + Attn*V
    flops_qk = 2 * d_qk * Sq * Sk * nh * batch
    flops_av = 2 * d_v * Sq * Sk * nh * batch
    total_flops = flops_qk + flops_av
    if is_causal and mode == "prefill":
        total_flops //= 2

    # Memory bytes (DRAM reads)
    # Q: batch * nh * Sq * d_qk (bf16=2B)
    # K: batch * nkv * Sk * d_qk (bf8=1B or bf16=2B)
    # V is derived from K in MLA (reuse_k), so no extra V read
    q_bytes = batch * nh * Sq * d_qk * 2  # Q always bf16
    k_bytes = batch * nkv * Sk * d_qk * kv_dtype_bytes
    # Output: batch * nh * Sq * d_v * 2
    out_bytes = batch * nh * Sq * d_v * 2
    total_mem_bytes = q_bytes + k_bytes + out_bytes

    # WH_B0 specs
    DRAM_BW_GBs = 258.0  # GB/s
    CLOCK_GHz = 1.0
    FMA_PER_CYCLE = 4096  # per core, LoFi
    FIDELITY_MULT = 4.0   # HiFi4

    ideal_compute_cycles = total_flops / (num_cores * FMA_PER_CYCLE / FIDELITY_MULT)
    ideal_compute_ms = ideal_compute_cycles / (CLOCK_GHz * 1e6)

    ideal_mem_ms = (total_mem_bytes / 1e9) / DRAM_BW_GBs * 1000

    arithmetic_intensity = total_flops / total_mem_bytes if total_mem_bytes > 0 else float('inf')

    return {
        "total_flops": total_flops,
        "flops_qk": flops_qk,
        "flops_av": flops_av,
        "total_mem_bytes": total_mem_bytes,
        "q_bytes": q_bytes,
        "k_bytes": k_bytes,
        "out_bytes": out_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "ideal_compute_ms": ideal_compute_ms,
        "ideal_mem_ms": ideal_mem_ms,
        "bottleneck": "compute" if ideal_compute_ms > ideal_mem_ms else "memory",
        "ideal_ms": max(ideal_compute_ms, ideal_mem_ms),
    }


def main():
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    logger.info(f"Device grid: {grid.x}x{grid.y} = {num_cores} cores")

    results = {"device": "WORMHOLE_B0", "grid": f"{grid.x}x{grid.y}", "num_cores": num_cores, "tests": []}

    # ========== Prefill workloads ==========
    prefill_configs = [
        # (batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_dtype, kv_dtype, is_causal, label)
        (1, 1024, 16, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, True, "P-small-causal"),
        (1, 4096, 16, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, True, "P-med-causal"),
        (1, 1024, 32, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, True, "P-32h-causal"),
        (1, 4096, 32, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, True, "P-32h-4k-causal"),
        (1, 1024, 128, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, True, "P-128h-causal"),
        (1, 512, 32, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, False, "P-32h-512-noncausal"),
        (1, 1024, 32, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, False, "P-32h-1k-noncausal"),
        (2, 1024, 128, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, True, "P-B2-128h-causal"),
    ]

    for cfg in prefill_configs:
        batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_dt, kv_dt, is_causal, label = cfg
        logger.info(f"[Prefill] {label}: B={batch} S={seq_len} NH={nh} lora={kv_lora_rank} rope={d_rope} causal={is_causal}")

        try:
            lats = benchmark_mla_prefill(device, batch, seq_len, nh, nkv, kv_lora_rank, d_rope,
                                         q_dt, kv_dt, is_causal, num_warmup=2, num_iters=5)
            theory = compute_theoretical_metrics("prefill", batch, seq_len, nh, nkv, kv_lora_rank, d_rope,
                                                 is_causal, num_cores, 1)  # bf8=1byte
            avg_ms = sum(lats) / len(lats)
            min_ms = min(lats)
            results["tests"].append({
                "mode": "prefill", "label": label,
                "batch": batch, "seq_len": seq_len, "nh": nh, "nkv": nkv,
                "kv_lora_rank": kv_lora_rank, "d_rope": d_rope, "is_causal": is_causal,
                "latencies_ms": [round(x, 3) for x in lats],
                "avg_ms": round(avg_ms, 3), "min_ms": round(min_ms, 3),
                "theory": {k: round(v, 4) if isinstance(v, float) else v for k, v in theory.items()},
            })
            logger.info(f"  avg={avg_ms:.3f}ms  min={min_ms:.3f}ms  ideal={theory['ideal_ms']:.3f}ms  "
                        f"bottleneck={theory['bottleneck']}  AI={theory['arithmetic_intensity']:.1f}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results["tests"].append({"mode": "prefill", "label": label, "error": str(e)})

    # ========== Decode workloads ==========
    decode_configs = [
        # (batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_dtype, kv_dtype, block_size, label)
        (2, 1024, 8, 1, 128, 64, ttnn.bfloat16, ttnn.bfloat8_b, 64, "D-small"),
        (2, 1024, 32, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, 64, "D-32h-1k"),
        (2, 4096, 32, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, 64, "D-32h-4k"),
        (4, 1024, 32, 1, 512, 64, ttnn.bfloat16, ttnn.bfloat8_b, 64, "D-B4-32h"),
    ]

    for cfg in decode_configs:
        batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_dt, kv_dt, block_size, label = cfg
        logger.info(f"[Decode] {label}: B={batch} S={seq_len} NH={nh} lora={kv_lora_rank} rope={d_rope}")

        try:
            lats = benchmark_mla_decode(device, batch, seq_len, nh, nkv, kv_lora_rank, d_rope,
                                        q_dt, kv_dt, block_size, num_warmup=3, num_iters=10)
            theory = compute_theoretical_metrics("decode", batch, seq_len, nh, nkv, kv_lora_rank, d_rope,
                                                 False, num_cores, 1)  # bf8=1byte
            avg_ms = sum(lats) / len(lats)
            min_ms = min(lats)
            results["tests"].append({
                "mode": "decode", "label": label,
                "batch": batch, "seq_len": seq_len, "nh": nh, "nkv": nkv,
                "kv_lora_rank": kv_lora_rank, "d_rope": d_rope,
                "latencies_ms": [round(x, 3) for x in lats],
                "avg_ms": round(avg_ms, 3), "min_ms": round(min_ms, 3),
                "theory": {k: round(v, 4) if isinstance(v, float) else v for k, v in theory.items()},
            })
            logger.info(f"  avg={avg_ms:.3f}ms  min={min_ms:.3f}ms  ideal={theory['ideal_ms']:.3f}ms  "
                        f"bottleneck={theory['bottleneck']}  AI={theory['arithmetic_intensity']:.1f}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results["tests"].append({"mode": "decode", "label": label, "error": str(e)})

    ttnn.close_device(device)

    # Save results
    out_path = "mla_flash_attention_dev/perf_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Label':<25} {'Mode':<8} {'B':>2} {'S':>6} {'NH':>4} {'LoRA':>5} "
          f"{'Avg(ms)':>9} {'Min(ms)':>9} {'Ideal(ms)':>10} {'Bottleneck':>10} {'Eff%':>7} {'AI':>8}")
    print("=" * 120)
    for t in results["tests"]:
        if "error" in t:
            print(f"{t['label']:<25} {t['mode']:<8} {'ERROR':>60}")
            continue
        eff = t["theory"]["ideal_ms"] / t["min_ms"] * 100 if t["min_ms"] > 0 else 0
        print(f"{t['label']:<25} {t['mode']:<8} {t['batch']:>2} {t['seq_len']:>6} {t['nh']:>4} {t['kv_lora_rank']:>5} "
              f"{t['avg_ms']:>9.3f} {t['min_ms']:>9.3f} {t['theory']['ideal_ms']:>10.3f} "
              f"{t['theory']['bottleneck']:>10} {eff:>6.1f}% {t['theory']['arithmetic_intensity']:>8.1f}")
    print("=" * 120)


if __name__ == "__main__":
    main()

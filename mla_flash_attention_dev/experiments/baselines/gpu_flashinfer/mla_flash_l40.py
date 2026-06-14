from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@dataclass
class TritonMLASchedMeta:
    """Compatibility placeholder for the official flash_mla API."""


def get_mla_metadata(*args, **kwargs) -> Tuple[TritonMLASchedMeta, None]:
    return TritonMLASchedMeta(), None


@triton.jit
def _mla_decode_kernel(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    out,
    lse,
    stride_block_table_b: tl.constexpr,
    max_seq_len: tl.constexpr,
    d_qk: tl.constexpr,
    d_v: tl.constexpr,
    h_q: tl.constexpr,
    h_kv: tl.constexpr,
    page_block_size: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DQ: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_q_idx = tl.program_id(1)
    value_block_idx = tl.program_id(2)

    kv_group = h_q // h_kv
    head_kv_idx = head_q_idx // kv_group

    offs_n = tl.arange(0, BLOCK_N)
    offs_dq = tl.arange(0, BLOCK_DQ)
    offs_dv = value_block_idx * BLOCK_DV + tl.arange(0, BLOCK_DV)

    seqlen = tl.load(cache_seqlens + batch_idx)
    q_base = (batch_idx * h_q + head_q_idx) * d_qk
    out_base = (batch_idx * h_q + head_q_idx) * d_v

    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)

    for start_n in tl.range(0, max_seq_len, BLOCK_N, loop_unroll_factor=1):
        token_idx = start_n + offs_n
        valid_n = token_idx < seqlen
        page_idx = token_idx // page_block_size
        page_offset = token_idx - page_idx * page_block_size
        block_idx = tl.load(block_table + batch_idx * stride_block_table_b + page_idx)
        cache_base = ((block_idx * page_block_size + page_offset) * h_kv + head_kv_idx) * d_qk

        scores = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for start_d in tl.range(0, d_qk, BLOCK_DQ):
            d_idx = start_d + offs_dq
            q_vec = tl.load(q + q_base + d_idx, mask=d_idx < d_qk, other=0.0).to(tl.float32)
            k_tile = tl.load(
                k_cache + cache_base[:, None] + d_idx[None, :],
                mask=valid_n[:, None] & (d_idx[None, :] < d_qk),
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(k_tile * q_vec[None, :], axis=1)

        scores = tl.where(valid_n, scores * softmax_scale, -float("inf"))
        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        v_idx = offs_dv
        v_tile = tl.load(
            k_cache + cache_base[:, None] + v_idx[None, :],
            mask=valid_n[:, None] & (v_idx[None, :] < d_v),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    result = acc / l_i
    tl.store(out + out_base + offs_dv, result, mask=offs_dv < d_v)
    tl.store(lse + batch_idx * h_q + head_q_idx, m_i + tl.log(l_i), mask=value_block_idx == 0)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Optional[TritonMLASchedMeta] = None,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del tile_scheduler_metadata, num_splits, causal, kwargs

    if q.device.type != "cuda" or k_cache.device.type != "cuda":
        raise ValueError("Triton MLA Flash backend requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype:
        raise ValueError("q and k_cache must both be float16 or bfloat16")
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError(f"q must have shape [B, 1, Hq, Dqk], got {tuple(q.shape)}")
    if k_cache.ndim != 4:
        raise ValueError(f"k_cache must have shape [blocks, block_size, Hkv, Dqk], got {tuple(k_cache.shape)}")
    if block_table is None or cache_seqlens is None:
        raise ValueError("block_table and cache_seqlens are required for dense decode")
    if not q.is_contiguous() or not k_cache.is_contiguous() or not block_table.is_contiguous():
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        block_table = block_table.contiguous()

    batch, _, h_q, d_qk = q.shape
    _, page_block_size, h_kv, cache_d_qk = k_cache.shape
    if cache_d_qk != d_qk:
        raise ValueError(f"q head dim ({d_qk}) must match k_cache head dim ({cache_d_qk})")
    if h_q % h_kv != 0:
        raise ValueError(f"Hq ({h_q}) must be divisible by Hkv ({h_kv})")
    if head_dim_v > d_qk:
        raise ValueError(f"head_dim_v ({head_dim_v}) must be <= q/k head dim ({d_qk})")
    if page_block_size & (page_block_size - 1):
        raise ValueError("page_block_size must be a power of two for this Triton backend")

    if softmax_scale is None:
        softmax_scale = d_qk ** -0.5

    out = torch.empty((batch, 1, h_q, head_dim_v), device=q.device, dtype=q.dtype)
    lse = torch.empty((batch, h_q, 1), device=q.device, dtype=torch.float32)

    max_seq_len = block_table.shape[1] * page_block_size
    block_n = 128
    block_dq = 64
    block_dv = 64
    grid = (batch, h_q, triton.cdiv(head_dim_v, block_dv))

    _mla_decode_kernel[grid](
        q,
        k_cache,
        block_table,
        cache_seqlens,
        out,
        lse,
        block_table.stride(0),
        max_seq_len,
        d_qk,
        head_dim_v,
        h_q,
        h_kv,
        page_block_size,
        float(softmax_scale),
        BLOCK_N=block_n,
        BLOCK_DQ=block_dq,
        BLOCK_DV=block_dv,
        num_warps=4,
    )
    return out, lse

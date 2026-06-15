"""PyTorch golden reference for S-FMLA decode semantics."""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode


def golden_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    head_dim_v: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference output for MLA decode with V as leading columns of K."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return FlashMLADecode.golden(
        q=q,
        kv_cache=kv_cache,
        position_ids=position_ids,
        head_dim_v=head_dim_v,
        scale=scale,
    )

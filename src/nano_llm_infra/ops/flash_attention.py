from __future__ import annotations

import torch

from nano_llm_infra.ops.pytorch_ref.flash_attention_ref import flash_attention_ref

from nano_llm_infra.ops.triton.flash_attention import flash_attention_triton


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    impl: str = "triton",
    block_m: int = 16,
    block_n: int = 32,
) -> torch.Tensor:
    """Mini-FlashAttention dispatch，输入 shape 为 [B, H, T, D]。"""
    if impl == "ref":
        return flash_attention_ref(q, k, v, causal)
    if impl == "triton":
        return flash_attention_triton(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal=causal,
            block_m=block_m,
            block_n=block_n,
        )
    raise ValueError(f"Unsupported FlashAttention implementation: {impl}")


def flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    return flash_attention(q, k, v, causal=causal, impl="ref")

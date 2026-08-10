from __future__ import annotations

import math

import torch


def flash_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """PyTorch reference，显式物化 [T, T] attention matrix。"""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])

    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)

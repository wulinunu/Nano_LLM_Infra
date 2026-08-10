from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    sm_scale,
    SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # 一个 program 负责 [BLOCK_M, D] 的 Q Tile。
    query_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    offsets_m = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    valid_m = offsets_m < SEQ_LEN
    mask_d = offsets_d < HEAD_DIM
    q_ptrs = (
        Q
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + offsets_m[:, None] * stride_qt
        + offsets_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None] & mask_d[None, :], other=0.0)

    # 每个 Query Row 都维护自己的 Online Softmax 状态。
    m_i = tl.where(valid_m, -float("inf"), 0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, SEQ_LEN, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offsets_n < SEQ_LEN

        k_ptrs = (
            K
            + batch_idx * stride_kb
            + head_idx * stride_kh
            + offsets_n[:, None] * stride_kt
            + offsets_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=valid_n[:, None] & mask_d[None, :], other=0.0)

        # [BLOCK_M, D] @ [D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        score_mask = valid_m[:, None] & valid_n[None, :]
        if CAUSAL:
            score_mask = score_mask & (offsets_n[None, :] <= offsets_m[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
            V
            + batch_idx * stride_vb
            + head_idx * stride_vh
            + offsets_n[:, None] * stride_vt
            + offsets_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=valid_n[:, None] & mask_d[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32), input_precision="ieee")
        m_i = m_ij

    out_ptrs = (
        Out
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + offsets_m[:, None] * stride_ot
        + offsets_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc / l_i[:, None], mask=valid_m[:, None] & mask_d[None, :])


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 16,
    block_n: int = 32,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = q.shape
    block_d = triton.next_power_of_2(head_dim)

    out = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, block_m), num_heads, batch_size)
    _flash_attention_kernel[grid](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        1.0 / math.sqrt(head_dim),
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        num_warps=4,
        num_stages=1,
    )
    return out

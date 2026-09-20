from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
)


class _RingExchange(torch.autograd.Function):
    """前向把 Tensor 传给下一 Rank，反向把梯度传回上一 Rank。"""

    # 当前 Rank 把 input 发给下一个 Rank，同时从上一个 Rank 接收新的 input
    @staticmethod
    def forward(ctx, input_, group, global_ranks):
        rank_index = global_ranks.index(dist.get_rank())
        next_rank = global_ranks[(rank_index + 1) % len(global_ranks)]
        previous_rank = global_ranks[(rank_index - 1) % len(global_ranks)]
        contiguous_input = input_.contiguous()
        output = torch.empty_like(contiguous_input)
        requests = dist.batch_isend_irecv( #非阻塞操作
            [
                dist.P2POp(dist.isend, contiguous_input, next_rank, group),
                dist.P2POp(dist.irecv, output, previous_rank, group),
            ]
        )
        for request in requests:
            request.wait()

        ctx.group = group
        ctx.global_ranks = global_ranks
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rank_index = ctx.global_ranks.index(dist.get_rank())
        next_rank = ctx.global_ranks[(rank_index + 1) % len(ctx.global_ranks)]
        previous_rank = ctx.global_ranks[(rank_index - 1) % len(ctx.global_ranks)]
        contiguous_grad = grad_output.contiguous()
        grad_input = torch.empty_like(contiguous_grad)
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(
                    dist.isend, contiguous_grad, previous_rank, ctx.group
                ),
                dist.P2POp(dist.irecv, grad_input, next_rank, ctx.group),
            ]
        )
        for request in requests:
            request.wait()
        return grad_input, None, None


def ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """每张卡保留本地 Q，沿 CP 环轮转各段 K/V 并执行 Online Softmax。"""
    # query/key/value: [batch, num_heads, local_seq_len, head_dim]
    group = get_context_parallel_group()
    global_ranks = get_context_parallel_global_ranks()
    rank = global_ranks.index(dist.get_rank())
    world_size = len(global_ranks)
    local_seq_len = query.size(2)
    scale = 1.0 / math.sqrt(query.size(-1))

    max_score = torch.full(query.shape[:-1], -float("inf"), device=query.device, dtype=torch.float32)  # [batch, num_heads, local_seq_len]
    softmax_sum = torch.zeros_like(max_score)  # [batch, num_heads, local_seq_len]
    output = torch.zeros_like(query, dtype=torch.float32)  # [batch, num_heads, local_seq_len, head_dim]
    current_key, current_value = key, value  # [batch, num_heads, local_seq_len, head_dim]

    query_positions = rank * local_seq_len + torch.arange(local_seq_len, device=query.device)  # [local_seq_len]
    for step in range(world_size):
        source_rank = (rank - step) % world_size
        key_positions = source_rank * local_seq_len + torch.arange(local_seq_len, device=query.device)  # [local_seq_len]
        causal_mask = key_positions[None, :] <= query_positions[:, None]  # [local_seq_len, local_seq_len]
        scores = torch.matmul(query.float(), current_key.float().transpose(-1, -2)) * scale  # [batch, num_heads, local_seq_len, local_seq_len]
        scores = scores.masked_fill(~causal_mask, -float("inf"))

        block_max = scores.max(dim=-1).values  # [batch, num_heads, local_seq_len]
        new_max = torch.maximum(max_score, block_max)  # [batch, num_heads, local_seq_len]
        old_scale = torch.exp(max_score - new_max)  # [batch, num_heads, local_seq_len]
        # 当前乘以value前的分子矩阵（未归一化的softmax权重）
        probabilities = torch.exp(scores - new_max.unsqueeze(-1))  # [batch, num_heads, local_seq_len, local_seq_len]
        # 乘以value后的分子 每个位置一个head dim向量
        output = output * old_scale.unsqueeze(-1) + torch.matmul(probabilities, current_value.float())  # [batch, num_heads, local_seq_len, head_dim]
        # 累计分母
        softmax_sum = softmax_sum * old_scale + probabilities.sum(dim=-1)  # [batch, num_heads, local_seq_len]
        max_score = new_max

        if step + 1 < world_size:
            current_key = _RingExchange.apply(current_key, group, global_ranks)
            current_value = _RingExchange.apply(current_value, group, global_ranks)

    return (output / softmax_sum.unsqueeze(-1)).to(query.dtype)  # [batch, num_heads, local_seq_len, head_dim]


class DenseCausalAttention(nn.Module):
    """完整序列上的 Attention，用于正确性对照。"""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden.shape
        qkv = self.qkv(hidden).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
        ) # [batch, num_heads, seq_len, head_dim]
        attention = attention.transpose(1, 2).reshape(
            batch_size, seq_len, hidden_size
        ) # [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
        return self.out_proj(attention)


class ContextParallelAttention(nn.Module):
    """序列维切分的 Ring Attention；每张卡只输入本地 Token。"""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.group = get_context_parallel_group()
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, local_seq_len, hidden_size = hidden.shape
        qkv = self.qkv(hidden).view(
            batch_size, local_seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        attention = ring_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ) # [batch, num_heads, seq_len, head_dim]
        attention = attention.transpose(1, 2).reshape(
            batch_size, local_seq_len, hidden_size
        )
        return self.out_proj(attention)

def copy_dense_attention_weights(
    dense_attention: DenseCausalAttention,
    cp_attention: ContextParallelAttention,
) -> None:
    with torch.no_grad():
        cp_attention.qkv.weight.copy_(dense_attention.qkv.weight)
        cp_attention.out_proj.weight.copy_(dense_attention.out_proj.weight)


def check_context_parallel(
    dense_attention: DenseCausalAttention,
    cp_attention: ContextParallelAttention,
    local_hidden: torch.Tensor,
) -> tuple[float, float, float, float]:
    """验证输出、输入梯度以及 QKV/输出投影的参数梯度。"""
    group = get_context_parallel_group()
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    local_seq_len = local_hidden.size(1)
    gathered_hidden = torch.empty(
        (
            local_hidden.size(0) * world_size,
            local_seq_len,
            local_hidden.size(2),
        ),
        device=local_hidden.device,
        dtype=local_hidden.dtype,
    )
    dist.all_gather_into_tensor(gathered_hidden, local_hidden, group=group)
    full_hidden = (
        gathered_hidden.view(
            world_size,
            local_hidden.size(0),
            local_seq_len,
            local_hidden.size(2),
        ) # [world_size, batch, local_seq, hidden]
        .transpose(0, 1) # [batch, world_size, local_seq, hidden]
        .reshape(local_hidden.size(0), world_size * local_seq_len, -1) # [batch, seq_len, hidden]
    )

    # 输出的diff
    dense_input = full_hidden.detach().requires_grad_()
    cp_input = local_hidden.detach().requires_grad_()
    dense_output = dense_attention(dense_input)
    cp_output = cp_attention(cp_input)
    start, end = rank * local_seq_len, (rank + 1) * local_seq_len
    output_diff = (dense_output[:, start:end] - cp_output).abs().max().item()

    # 输入和参数的梯度diff
    dense_output.sum().backward()
    cp_output.sum().backward()
    for parameter in cp_attention.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, group=cp_attention.group)
    input_grad_diff = (
        dense_input.grad[:, start:end] - cp_input.grad
    ).abs().max().item()
    qkv_grad_diff = (
        dense_attention.qkv.weight.grad - cp_attention.qkv.weight.grad
    ).abs().max().item()
    out_grad_diff = (
        dense_attention.out_proj.weight.grad - cp_attention.out_proj.weight.grad
    ).abs().max().item()
    return output_diff, input_grad_diff, qkv_grad_diff, out_grad_diff

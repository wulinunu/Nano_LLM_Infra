from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from ..parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)


# TP region是指列并行+激活函数+行并行的一整个模块 这里只定义进和出region的正反向行为
class _CopyToTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, group=get_tensor_model_parallel_group())
        return grad_output


class _ReduceFromTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=get_tensor_model_parallel_group())
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class ColumnParallelLinear(nn.Module):
    """按输出维切分权重；前向得到局部输出。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if out_features % tp_size != 0:
            raise ValueError("out_features must be divisible by TP size")
        self.weight = nn.Parameter(torch.empty(out_features // tp_size, in_features))
        self.bias = nn.Parameter(torch.empty(out_features // tp_size)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.size(1))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(_CopyToTPRegion.apply(input_), self.weight.t())
        return output if self.bias is None else output + self.bias


class RowParallelLinear(nn.Module):
    """按输入维切分权重；前向 all-reduce 局部输出。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if in_features % tp_size != 0:
            raise ValueError("in_features must be divisible by TP size")
        self.weight = nn.Parameter(torch.empty(out_features, in_features // tp_size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = _ReduceFromTPRegion.apply(torch.matmul(input_, self.weight.t()))
        return output if self.bias is None else output + self.bias


class TPMLP(nn.Module):
    """Megatron 风格：ColumnParallelLinear → GELU → RowParallelLinear。"""

    def __init__(self, hidden_size: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        intermediate_size = hidden_size * mlp_ratio
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.act = nn.GELU()
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(hidden)))


def shard_dense_mlp_weights_to_tp(tp_mlp: TPMLP, dense_up: nn.Linear, dense_down: nn.Linear) -> None:
    """把 dense MLP 权重按 TP rank 切到 Column / Row shard。初始化时定死 后面个卡间只流动激活值"""
    tp_size = get_tensor_model_parallel_world_size()
    rank = dist.get_rank(group=get_tensor_model_parallel_group())
    up_chunk = dense_up.weight.shape[0] // tp_size
    down_chunk = dense_down.weight.shape[1] // tp_size
    start_up, end_up = rank * up_chunk, (rank + 1) * up_chunk
    start_down, end_down = rank * down_chunk, (rank + 1) * down_chunk
    with torch.no_grad():
        tp_mlp.up_proj.weight.copy_(dense_up.weight[start_up:end_up])
        tp_mlp.down_proj.weight.copy_(dense_down.weight[:, start_down:end_down])


def check_tp_mlp(
    dense_mlp: nn.Module,
    tp_mlp: TPMLP,
    x: torch.Tensor,
) -> tuple[float, float]:
    """对比 dense / TP 的 forward 与 down_proj 梯度，返回 (fwd_max_diff, grad_max_diff)。"""
    tp_size = get_tensor_model_parallel_world_size()
    rank = dist.get_rank(group=get_tensor_model_parallel_group())
    down_chunk = dense_mlp.down_proj.weight.shape[1] // tp_size
    start_down, end_down = rank * down_chunk, (rank + 1) * down_chunk

    dense_out = dense_mlp(x)
    tp_out = tp_mlp(x)
    fwd_diff = (dense_out - tp_out).abs().max().item()

    dense_out.sum().backward()
    tp_out.sum().backward()
    dense_shard = dense_mlp.down_proj.weight.grad[:, start_down:end_down]
    grad_diff = (dense_shard - tp_mlp.down_proj.weight.grad).abs().max().item()
    return fwd_diff, grad_diff


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TPMLP",
    "shard_dense_mlp_weights_to_tp",
    "check_tp_mlp",
]

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from ..parallel_state import get_tensor_model_parallel_group


# TP region是指列并行+激活函数+行并行的一整个模块 这里只定义进和出region的正反向行为
# torch.autograd.Function是为了自定义反向传播接口 nn.module只能定义前向 反向需要自己实现
class _CopyToTPRegion(torch.autograd.Function):
    @staticmethod # 调用方式是Function.apply(tensor) 不通过实例调用 所以是staticmethod
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
        tp_size = dist.get_world_size(group=get_tensor_model_parallel_group())
        if out_features % tp_size != 0:
            raise ValueError("out_features must be divisible by TP size")
        self.weight = nn.Parameter(torch.empty(out_features // tp_size, in_features))
        self.bias = nn.Parameter(torch.empty(out_features // tp_size)) if bias else None
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
        tp_size = dist.get_world_size(group=get_tensor_model_parallel_group())
        if in_features % tp_size != 0:
            raise ValueError("in_features must be divisible by TP size")
        self.weight = nn.Parameter(torch.empty(out_features, in_features // tp_size))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.size(1))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = _ReduceFromTPRegion.apply(torch.matmul(input_, self.weight.t())) #先乘再reduce
        return output if self.bias is None else output + self.bias


class TPMLP(nn.Module):
    """Megatron 风格：ColumnParallelLinear → GELU → RowParallelLinear。"""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.act = nn.GELU()
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(hidden)))


def shard_dense_mlp_weights_to_tp(tp_mlp: TPMLP, dense_up: nn.Linear, dense_down: nn.Linear) -> None:
    """把 dense MLP 权重按 TP rank 切到 Column / Row shard。初始化时定死 后面个卡间只流动激活值"""
    group = get_tensor_model_parallel_group()
    tp_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    assert dense_up.weight.shape[0] == dense_down.weight.shape[1]
    chunk_size = dense_up.weight.shape[0] // tp_size  # linear 权重：[out_features, in_features]
    start, end = rank * chunk_size, (rank + 1) * chunk_size
    with torch.no_grad():
        tp_mlp.up_proj.weight.copy_(dense_up.weight[start:end])# 切输出维 即列并行，输出被切分
        tp_mlp.down_proj.weight.copy_(dense_down.weight[:, start:end])# 切输入维 列并行输出的每一部分各自计算然后相加


def check_tp_mlp(
    dense_mlp: nn.Module,
    tp_mlp: TPMLP,
    x: torch.Tensor,
) -> tuple[float, float, float, float]:
    """对比 dense / TP 的 forward、x.grad、up/down 权重梯度。"""
    group = get_tensor_model_parallel_group()
    tp_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    assert dense_mlp.up_proj.weight.shape[0] == dense_mlp.down_proj.weight.shape[1]
    chunk_size = dense_mlp.up_proj.weight.shape[0] // tp_size
    start, end = rank * chunk_size, (rank + 1) * chunk_size

    x_dense = x.detach().requires_grad_()
    x_tp = x.detach().requires_grad_()
    dense_out = dense_mlp(x_dense)
    tp_out = tp_mlp(x_tp)
    fwd_diff = (dense_out - tp_out).abs().max().item()

    dense_out.sum().backward()
    tp_out.sum().backward()
    x_grad_diff = (x_dense.grad - x_tp.grad).abs().max().item()
    up_grad_diff = (
        (dense_mlp.up_proj.weight.grad[start:end] - tp_mlp.up_proj.weight.grad).abs().max().item()
    )
    down_grad_diff = (
        (dense_mlp.down_proj.weight.grad[:, start:end] - tp_mlp.down_proj.weight.grad).abs().max().item()
    )
    return fwd_diff, x_grad_diff, up_grad_diff, down_grad_diff

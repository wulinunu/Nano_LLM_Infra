from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from ..parallel_state import get_expert_model_parallel_group


class _AllToAll(torch.autograd.Function):
    """前向发送 Token，反向沿相反方向送回梯度。"""

    @staticmethod
    def forward(ctx, input_, send_splits, recv_splits, group):
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.group = group
        output = input_.new_empty((sum(recv_splits), *input_.shape[1:])) # [当前rank总共收到的token数量， 其他维度如hiddensize]
        dist.all_to_all_single(
            output,
            input_.contiguous(), #输入必须内存连续，否则 NCCL 无法正确读取
            output_split_sizes=list(recv_splits), #输入如何切分发给各 rank
            input_split_sizes=list(send_splits), #输出如何按来源 rank 拼接
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty((sum(ctx.send_splits), *grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=list(ctx.send_splits),
            input_split_sizes=list(ctx.recv_splits),
            group=ctx.group,
        )
        return grad_input, None, None, None

# 每个expert有自己独立的权重
class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(hidden)))


class DenseMoE(nn.Module):
    """用于正确性对照：所有 Expert 都放在当前 Rank。"""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        # 每个expert独立创建 享有独立的参数
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, hidden: torch.Tensor, capacity_factor: float) -> torch.Tensor:
        original_shape = hidden.shape # [batch_size, seq_len, hidden_size]
        hidden = hidden.reshape(-1, original_shape[-1]) # [batch_size * seq_len, hidden_size]
        router_probs = self.router(hidden).softmax(dim=-1) # [batch_size * seq_len, num_experts]
        routes = router_probs.argmax(dim=-1) # [batch_size * seq_len] 每个token对应的expert
        capacity = math.ceil(capacity_factor * hidden.size(0) / len(self.experts)) # 接收的token数量
        output = torch.zeros_like(hidden)

        for expert_id, expert in enumerate(self.experts):
            token_indices = torch.where(routes == expert_id)[0][:capacity] # 路由到这个expert的token id
            if token_indices.numel() == 0:
                continue
            expert_output = expert(hidden[token_indices]) # [token_indices.numel(), hidden_size]
            #乘以这个router的概率，router参数才能获得梯度
            expert_output = expert_output * router_probs[token_indices, expert_id, None] # [token_indices.numel(), hidden_size] * [token_indices.numel(), 1] = [token_indices.numel(), hidden_size]
            output.index_copy_(0, token_indices, expert_output) #写入这些token原来的位置
        return output.view(original_shape)


class ExpertParallelMoE(nn.Module):
    """Top-1 MoE：每个 Rank 只保存一部分 Expert。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.group = get_expert_model_parallel_group()
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        if num_experts % self.world_size != 0:
            raise ValueError("num_experts must be divisible by EP size")

        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.world_size
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.local_experts = nn.ModuleList(
            [
                ExpertMLP(hidden_size, intermediate_size)
                for _ in range(self.num_local_experts)
            ]
        )
        self.last_routed_expert_load = torch.zeros(num_experts, dtype=torch.long) # 最近一次前向中，丢弃前每个expert的负载
        self.last_dropped_tokens = 0 # 最近一次前向中 被丢弃的全局token数量

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # ---------- 1. Router：每个 token 选一个 expert ----------
        original_shape = hidden.shape # [batch_size_local, seq_len, hidden_size]
        hidden = hidden.reshape(-1, original_shape[-1]) # [batch_size_local * seq_len, hidden_size]
        router_probs = self.router(hidden).softmax(dim=-1) # [batch_size_local * seq_len, num_experts]
        routes = router_probs.argmax(dim=-1) # [batch_size_local * seq_len] 每个token对应的expert

        # ---------- 2. 统计全局负载，算 capacity，决定哪些 token 保留 ----------
        tokens_per_expert_on_this_rank = torch.bincount(routes, minlength=self.num_experts)
        tokens_per_expert_by_rank = torch.empty(
            self.world_size * self.num_experts,
            device=hidden.device,
            dtype=torch.long,
        )
        dist.all_gather_into_tensor(tokens_per_expert_by_rank, tokens_per_expert_on_this_rank, group=self.group)
        tokens_per_expert_by_rank = tokens_per_expert_by_rank.view(self.world_size, self.num_experts)
        capacity = math.ceil(self.capacity_factor * hidden.size(0) * self.world_size / self.num_experts) #理论上每个expert需要负载多少个token 乘以系数后向上取整

        positions_in_expert_on_this_rank = torch.empty_like(routes) #长度为当前rank中token的数量
        for expert_id in range(self.num_experts):
            mask = routes == expert_id # 当前rank中 属于这个expert的token的位置
            positions_in_expert_on_this_rank[mask] = torch.arange(mask.sum(), device=hidden.device) # 按顺序写这个rank中  token被分到这个expert的位置
        tokens_per_expert_on_previous_ranks = tokens_per_expert_by_rank[:self.rank].sum(dim=0) # [self.rank, num_experts] -> [num_experts] 前self.rank张卡中每个expert的负载
        global_positions_in_expert = positions_in_expert_on_this_rank + tokens_per_expert_on_previous_ranks[routes] # 当前rank中每个token在全局的expert中的位置
        kept_indices = torch.where(global_positions_in_expert < capacity)[0] # 当前rank中保留的token id

        # ---------- 3. 按目标 rank 排好序，交换“发/收多少个” ----------
        destinations = routes[kept_indices] // self.num_local_experts # 计算路由到的expert所在rank
        order = destinations.argsort() # 按rank排序 结果是排序后的下标
        ordered_indices = kept_indices[order] # 排序后的token id
        send_splits = torch.bincount(destinations, minlength=self.world_size).tolist() # 当前rank中分别要向其它rank发送多少个token
        send_counts = torch.tensor(send_splits, device=hidden.device, dtype=torch.long) # 将python list 变成GPU tensor
        recv_counts = torch.empty_like(send_counts) # 长度为rank的数量
        dist.all_to_all_single(recv_counts, send_counts, group=self.group) # 每个 rank 把自己的 send_counts[i] 发给 rank i，同时接收别人发来的计数，填到 recv_counts
        recv_splits = recv_counts.tolist() #长度为rank的数量

        # ---------- 4. Dispatch：token + 本地 expert 下标 All-to-All 到目标 rank ----------
        dispatched_hidden = _AllToAll.apply(hidden[ordered_indices], send_splits, recv_splits, self.group) #[sum(recv_splits), hidden_size]
        sent_local_expert_ids = (routes[ordered_indices] % self.num_local_experts).contiguous() #发出去的token都在对应的本地rank上是第几个expert
        received_local_expert_ids = torch.empty(sum(recv_splits), device=hidden.device, dtype=torch.long) #收过来的token要给我本地的哪一个expert
        dist.all_to_all_single( #交换给哪个expert
            received_local_expert_ids,
            sent_local_expert_ids,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.group,
        )

        # ---------- 5. 本地 Expert 计算 ----------
        expert_output = torch.empty_like(dispatched_hidden) #输出缓冲区
        for expert_id, expert in enumerate(self.local_experts):
            mask = received_local_expert_ids == expert_id
            if mask.any():
                expert_output[mask] = expert(dispatched_hidden[mask])

        # ---------- 6. Combine：结果送回原 rank，乘 router 概率，写回原顺序 ----------
        returned_output = _AllToAll.apply(
            expert_output, recv_splits, send_splits, self.group
        )
        returned_output = (
            returned_output
            * router_probs[ordered_indices, routes[ordered_indices], None]
        )
        output = torch.zeros_like(hidden)
        output.index_copy_(0, ordered_indices, returned_output)

        # ---------- 7. 统计负载 / 丢弃数（不影响计算） ----------
        self.last_routed_expert_load = tokens_per_expert_by_rank.sum(dim=0).detach().cpu()
        dropped = torch.tensor(hidden.size(0) - kept_indices.numel(), device=hidden.device, dtype=torch.long)
        dist.all_reduce(dropped, group=self.group)
        self.last_dropped_tokens = int(dropped.item())

        return output.view(original_shape)


def copy_dense_moe_weights(dense_moe: DenseMoE, ep_moe: ExpertParallelMoE) -> None:
    """把 Dense Router 和当前 Rank 对应的 Expert 权重复制到 EP 模型。"""
    start = ep_moe.rank * ep_moe.num_local_experts
    with torch.no_grad():
        ep_moe.router.weight.copy_(dense_moe.router.weight)
        for local_id, expert in enumerate(ep_moe.local_experts):
            dense_expert = dense_moe.experts[start + local_id]
            expert.up_proj.weight.copy_(dense_expert.up_proj.weight)
            expert.down_proj.weight.copy_(dense_expert.down_proj.weight)


def check_expert_parallel(
    dense_moe: DenseMoE,
    ep_moe: ExpertParallelMoE,
    hidden: torch.Tensor, # 仅本卡的输入
) -> tuple[float, float, float, float]:
    """验证输出、输入梯度、Router 梯度和本地 Expert 梯度。"""
    local_batch = hidden.size(0)
    gathered_hidden = torch.empty(
        (local_batch * ep_moe.world_size, *hidden.shape[1:]),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    dist.all_gather_into_tensor(gathered_hidden, hidden, group=ep_moe.group)

    # 输出的diff
    dense_input = gathered_hidden.requires_grad_()
    ep_input = hidden.requires_grad_()
    dense_output = dense_moe(dense_input, ep_moe.capacity_factor)
    ep_output = ep_moe(ep_input)
    start, end = ep_moe.rank * local_batch, (ep_moe.rank + 1) * local_batch
    output_diff = (dense_output[start:end] - ep_output).abs().max().item()

    # 输入和router参数的梯度diff
    dense_output.sum().backward()
    ep_output.sum().backward()
    if ep_moe.router.weight.grad is not None:
        dist.all_reduce(ep_moe.router.weight.grad, group=ep_moe.group)
    input_grad_diff = (
        dense_input.grad[start:end] - ep_input.grad
    ).abs().max().item()
    router_grad_diff = (
        dense_moe.router.weight.grad - ep_moe.router.weight.grad
    ).abs().max().item()

    # expert参数的梯度diff
    expert_grad_diff = 0.0
    first_global_expert = ep_moe.rank * ep_moe.num_local_experts
    for local_id, local_expert in enumerate(ep_moe.local_experts):
        dense_expert = dense_moe.experts[first_global_expert + local_id]
        dense_grad = dense_expert.up_proj.weight.grad
        local_grad = local_expert.up_proj.weight.grad
        if dense_grad is not None and local_grad is not None:
            expert_grad_diff = max(
                expert_grad_diff, (dense_grad - local_grad).abs().max().item()
            )
    return output_diff, input_grad_diff, router_grad_diff, expert_grad_diff

from __future__ import annotations

from itertools import product

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------- #
# 全局并行状态维护
# ---------------------------------------------------------------------------- #

_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GROUP = None
_EXPERT_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None

_PP_GLOBAL_RANKS: list[int] = []
_CP_GLOBAL_RANKS: list[int] = []


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
) -> None:
    """
    初始化正交的 DP、PP、CP、EP、TP 通信组。

    Rank 布局为 ``[dp][pp][cp][ep][tp]``，TP 是变化最快的维度。
    当前 demo 只单独运行一种模型并行，但通信组可以表达五维组合。
    """
    if not dist.is_initialized():
        raise RuntimeError("Call dist.init_process_group() before initializing parallel groups.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    global _TENSOR_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GROUP
    global _EXPERT_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _PP_GLOBAL_RANKS
    global _CP_GLOBAL_RANKS

    model_parallel_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * context_parallel_size
        * expert_model_parallel_size
    )
    if world_size % model_parallel_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by PP x CP x EP x TP "
            f"({pipeline_model_parallel_size} x {context_parallel_size} x "
            f"{expert_model_parallel_size} x {tensor_model_parallel_size})"
        )

    data_parallel_size = world_size // model_parallel_size
    rank_grid = torch.arange(world_size).reshape(
        data_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        expert_model_parallel_size,
        tensor_model_parallel_size,
    )

    for dp_rank, pp_rank, cp_rank, ep_rank in product(
        range(data_parallel_size),
        range(pipeline_model_parallel_size),
        range(context_parallel_size),
        range(expert_model_parallel_size),
    ):
        ranks = rank_grid[dp_rank, pp_rank, cp_rank, ep_rank, :].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    for dp_rank, pp_rank, cp_rank, tp_rank in product(
        range(data_parallel_size),
        range(pipeline_model_parallel_size),
        range(context_parallel_size),
        range(tensor_model_parallel_size),
    ):
        ranks = rank_grid[dp_rank, pp_rank, cp_rank, :, tp_rank].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group

    for dp_rank, pp_rank, ep_rank, tp_rank in product(
        range(data_parallel_size),
        range(pipeline_model_parallel_size),
        range(expert_model_parallel_size),
        range(tensor_model_parallel_size),
    ):
        ranks = rank_grid[dp_rank, pp_rank, :, ep_rank, tp_rank].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CP_GLOBAL_RANKS = ranks

    for dp_rank, cp_rank, ep_rank, tp_rank in product(
        range(data_parallel_size),
        range(context_parallel_size),
        range(expert_model_parallel_size),
        range(tensor_model_parallel_size),
    ):
        ranks = rank_grid[dp_rank, :, cp_rank, ep_rank, tp_rank].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PP_GLOBAL_RANKS = ranks

    for pp_rank, cp_rank, ep_rank, tp_rank in product(
        range(pipeline_model_parallel_size),
        range(context_parallel_size),
        range(expert_model_parallel_size),
        range(tensor_model_parallel_size),
    ):
        ranks = rank_grid[:, pp_rank, cp_rank, ep_rank, tp_rank].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group


def get_data_parallel_group():
    """获取当前进程所在的 DP 组。"""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """当前进程所在的 PP 通信组。"""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, "pipeline model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_global_ranks() -> list[int]:
    """这条流水线各 stage 的全局 rank，按前后顺序排列。"""
    assert _PP_GLOBAL_RANKS, "pipeline model parallel group is not initialized"
    return _PP_GLOBAL_RANKS


def get_context_parallel_group():
    """获取当前进程所在的 CP 组。"""
    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks() -> list[int]:
    """当前 CP 环中各进程的全局 rank，按序列分片顺序排列。"""
    assert _CP_GLOBAL_RANKS, "context parallel group is not initialized"
    return _CP_GLOBAL_RANKS


def get_expert_model_parallel_group():
    """获取当前进程所在的 EP 组。"""
    assert _EXPERT_MODEL_PARALLEL_GROUP is not None, "expert parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """获取当前进程所在的 TP 组。"""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP

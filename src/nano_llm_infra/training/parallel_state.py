from __future__ import annotations

import torch.distributed as dist

# ---------------------------------------------------------------------------- #
# 全局并行状态维护
# ---------------------------------------------------------------------------- #

_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None

_TP_WORLD_SIZE = 1
_PP_WORLD_SIZE = 1
_DP_WORLD_SIZE = 1
_PP_GLOBAL_RANKS: list[int] = []


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> None:
    """
    初始化正交的 TP、PP、DP 通信组。

    Rank 布局为 ``[dp][pp][tp]``。例如 DP=2、PP=2、TP=2 时：
    - TP group: [0, 1]、[2, 3]、[4, 5]、[6, 7]
    - PP group: [0, 2]、[1, 3]、[4, 6]、[5, 7]
    - DP group: [0, 4]、[1, 5]、[2, 6]、[3, 7]
    """
    if not dist.is_initialized():
        raise RuntimeError("Call dist.init_process_group() before initializing parallel groups.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    global _TENSOR_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _TP_WORLD_SIZE
    global _PP_WORLD_SIZE
    global _DP_WORLD_SIZE
    global _PP_GLOBAL_RANKS

    # 确保总卡数能被切分方式整除
    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_parallel_size ({pipeline_model_parallel_size})"
        )

    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

    _TP_WORLD_SIZE = tensor_model_parallel_size
    _PP_WORLD_SIZE = pipeline_model_parallel_size
    _DP_WORLD_SIZE = data_parallel_size

    for dp_rank in range(data_parallel_size):
        for pp_rank in range(pipeline_model_parallel_size):
            base = (dp_rank * pipeline_model_parallel_size + pp_rank) * tensor_model_parallel_size
            ranks = list(range(base, base + tensor_model_parallel_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group

    for dp_rank in range(data_parallel_size):
        for tp_rank in range(tensor_model_parallel_size):
            ranks = [
                (dp_rank * pipeline_model_parallel_size + pp_rank)
                * tensor_model_parallel_size
                + tp_rank
                for pp_rank in range(pipeline_model_parallel_size)
            ]
            group = dist.new_group(ranks)
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PP_GLOBAL_RANKS = ranks #因为pp需要指定前后顺序

    for pp_rank in range(pipeline_model_parallel_size):
        for tp_rank in range(tensor_model_parallel_size):
            ranks = [
                (dp_rank * pipeline_model_parallel_size + pp_rank)
                * tensor_model_parallel_size
                + tp_rank
                for dp_rank in range(data_parallel_size)
            ]
            group = dist.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group


def get_tensor_model_parallel_group():
    """获取当前进程所在的 TP 组。"""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """获取当前进程所在的 DP 组。"""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_data_parallel_rank() -> int:
    if _DP_WORLD_SIZE == 1:
        return 0
    return dist.get_rank(group=get_data_parallel_group())


def get_data_parallel_world_size() -> int:
    return _DP_WORLD_SIZE


def get_pipeline_model_parallel_group():
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, "pipeline model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_rank() -> int:
    if _PP_WORLD_SIZE == 1:
        return 0
    return dist.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_world_size() -> int:
    return _PP_WORLD_SIZE


# dist.send/resv明确要求指定src/dst 所以必须要有global rank
def get_pipeline_model_parallel_global_ranks() -> list[int]:
    return _PP_GLOBAL_RANKS


def get_tensor_model_parallel_world_size():
    return _TP_WORLD_SIZE


def get_tensor_model_parallel_rank():
    if _TP_WORLD_SIZE == 1:
        return 0
    return dist.get_rank(group=get_tensor_model_parallel_group())

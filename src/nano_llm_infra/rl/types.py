from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class Phase(str, Enum):
    ROLLOUT = "rollout"
    REWARD = "reward"
    TRAIN = "train"
    SYNC = "sync"


@dataclass
class RLConfig:
    vocab_size: int = 64
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    group_size: int = 4
    response_length: int = 12
    temperature: float = 1.0
    learning_rate: float = 3e-3
    kl_beta: float = 0.04
    reward_workers: int = 4
    kv_block_size: int = 4
    kv_num_blocks: int = 256
    rollout_batch_size: int = 16
    zero_stage: int = 2
    zero_bucket_size_mb: float = 0.05


@dataclass
class Prompt:
    group_id: int
    prompt_ids: list[int]
    target_token: int


@dataclass
class Experience:
    prompt_ids: list[int]
    response_ids: list[int]
    old_logprobs: list[float]
    group_id: int
    policy_version: int
    reward: float = 0.0
    completion_mask: list[bool] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.completion_mask:
            self.completion_mask = [False] * len(self.prompt_ids) + [True] * len(self.response_ids)


@dataclass
class PackedBatch:
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    old_logprobs: torch.Tensor
    completion_mask: torch.Tensor
    token_group_ids: torch.Tensor
    max_seqlen: int


@dataclass
class StepMetrics:
    step: int
    phase: Phase
    rollout_ms: float
    reward_ms: float
    train_ms: float
    sync_ms: float
    policy_version: int
    mean_reward: float
    loss: float
    kl: float
    grad_norm: float
    gpu_memory_mb: float
    kv_pool_mb: float
    kv_blocks_peak: int
    rollout_memory_mb: float
    released_memory_mb: float
    zero_backward_memory_mb: float

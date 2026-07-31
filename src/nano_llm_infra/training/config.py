from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    zero_stage: int = 0


@dataclass
class RuntimeConfig:
    precision: str = "fp16"
    bucket_size_mb: float = 1.0
    use_activation_checkpoint: bool = False

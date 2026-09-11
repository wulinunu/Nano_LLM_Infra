from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil

import torch


class BlockAllocationError(RuntimeError):
    """Raised when the KV cache pool cannot satisfy a block allocation."""


@dataclass
class BlockTable:
    """Logical-token-block to physical-cache-block mapping for one request."""

    block_size: int
    block_ids: list[int] = field(default_factory=list)

    def num_blocks_needed(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return ceil(num_tokens / self.block_size) if num_tokens else 0

    def ensure_block_capacity(self, num_tokens: int, allocator: BlockAllocator) -> None:
        required = self.num_blocks_needed(num_tokens)
        missing = required - len(self.block_ids)
        if missing > 0:
            self.block_ids.extend(allocator.allocate(missing))

    def physical_block_for_token(self, token_idx: int) -> int:
        if token_idx < 0:
            raise ValueError("token_idx must be non-negative")
        logical_block = token_idx // self.block_size
        if logical_block >= len(self.block_ids):
            raise IndexError("token_idx is not allocated in this block table")
        return self.block_ids[logical_block]

    def clear(self) -> list[int]:
        released = self.block_ids
        self.block_ids = []
        return released


class BlockAllocator:
    """O(1) free-list allocator for fixed-size KV cache blocks."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free_blocks: deque[int] = deque(range(num_blocks))
        self._allocated: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self._allocated)

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if num_blocks > self.num_free_blocks:
            raise BlockAllocationError(
                f"requested {num_blocks} blocks, but only {self.num_free_blocks} are free"
            )

        block_ids = [self._free_blocks.popleft() for _ in range(num_blocks)]
        self._allocated.update(block_ids)
        return block_ids

    def free(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            if block_id not in self._allocated:
                raise ValueError(f"block {block_id} is not currently allocated")
            self._allocated.remove(block_id)
            self._free_blocks.append(block_id)


class KVCachePool:
    """Preallocated KV cache pool addressed by physical block ids."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.cache = torch.zeros(
            (
                num_blocks,
                num_layers,
                2,
                num_heads,
                block_size,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )

    @property
    def keys(self) -> torch.Tensor:
        return self.cache[:, :, 0]

    @property
    def values(self) -> torch.Tensor:
        return self.cache[:, :, 1]

    def get_block(self, block_id: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(f"block_id {block_id} is out of range")
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} is out of range")
        return self.keys[block_id, layer_idx], self.values[block_id, layer_idx]

    def write_token(
        self,
        block_id: int,
        layer_idx: int,
        token_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        block_key, block_value = self.get_block(block_id, layer_idx)
        if not 0 <= token_offset < self.block_size:
            raise IndexError(f"token_offset {token_offset} is out of range")
        expected_shape = (self.num_heads, self.head_dim)
        if tuple(key.shape) != expected_shape:
            raise ValueError(f"key shape {tuple(key.shape)} does not match {expected_shape}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"value shape {tuple(value.shape)} does not match {expected_shape}")
        block_key[:, token_offset, :].copy_(key)
        block_value[:, token_offset, :].copy_(value)

    def release(self) -> None:
        self.cache = torch.empty(0)

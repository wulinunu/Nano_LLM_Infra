from __future__ import annotations

import torch

from nano_llm_infra.inference.block_manager import BlockTable, KVCachePool
from nano_llm_infra.ops.pytorch_ref.paged_attention_ref import paged_attention_ref

from nano_llm_infra.ops.triton.paged_attention import paged_attention_triton

from nano_llm_infra import _paged_attention


def _build_decode_inputs(
    query: torch.Tensor,
    kv_cache: KVCachePool,
    block_table: BlockTable,
    layer_idx: int,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    q = query.unsqueeze(0).contiguous()
    k_cache, v_cache = kv_cache.get_block(0, layer_idx)  # seed shape only
    del k_cache, v_cache  # avoid accidental use below

    block_tables = torch.tensor(
        [block_table.block_ids],
        device=query.device,
        dtype=torch.int32,
    )
    context_lens = torch.tensor([num_tokens], device=query.device, dtype=torch.int32)
    return (
        q,
        kv_cache.keys[:, layer_idx].contiguous(),
        kv_cache.values[:, layer_idx].contiguous(),
        block_tables,
        context_lens,
        kv_cache.block_size,
    )


def paged_attention(
    query: torch.Tensor,
    kv_cache: KVCachePool,
    block_table: BlockTable,
    layer_idx: int,
    num_tokens: int,
    impl: str = "ref",
) -> torch.Tensor:
    q, k_cache, v_cache, block_tables, context_lens, block_size = _build_decode_inputs(
        query, kv_cache, block_table, layer_idx, num_tokens
    )

    if impl == "ref":
        return paged_attention_ref(q, k_cache, v_cache, block_tables, context_lens, block_size).squeeze(0)
    if impl == "triton":
        return paged_attention_triton(q, k_cache, v_cache, block_tables, context_lens, block_size).squeeze(0)
    if impl == "cuda":
        return _paged_attention.paged_attention_cuda(
            q, k_cache, v_cache, block_tables, context_lens, int(block_size)
        ).squeeze(0)
    raise ValueError(f"Unsupported paged attention implementation: {impl}")


def paged_attention_reference(
    query: torch.Tensor,
    kv_cache: KVCachePool,
    block_table: BlockTable,
    layer_idx: int,
    num_tokens: int,
) -> torch.Tensor:
    return paged_attention(query, kv_cache, block_table, layer_idx, num_tokens, impl="ref")

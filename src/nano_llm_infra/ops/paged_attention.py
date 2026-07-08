from __future__ import annotations

import torch

from nano_llm_infra.inference.block_manager import BlockTable, KVCachePool
from nano_llm_infra.ops.paged_attention_ref import paged_attention_ref

try:
    from nano_llm_infra.ops.triton.paged_attention import paged_attention_triton
except ImportError as exc:  # pragma: no cover - exercised when Triton unavailable
    _TRITON_BACKEND = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_BACKEND = paged_attention_triton
    _TRITON_IMPORT_ERROR = None

try:
    from nano_llm_infra import _paged_attention
except ImportError as exc:  # pragma: no cover - exercised before extension build
    _PAGED_ATTENTION_EXT = None
    _PAGED_ATTENTION_IMPORT_ERROR = exc
else:
    _PAGED_ATTENTION_EXT = _paged_attention
    _PAGED_ATTENTION_IMPORT_ERROR = None


def _require_cuda_backend() -> None:
    if _PAGED_ATTENTION_EXT is None:
        raise ImportError(
            "nano_llm_infra._paged_attention is not built. "
            "Run `python setup.py build_ext --inplace` first."
        ) from _PAGED_ATTENTION_IMPORT_ERROR


def _require_triton_backend() -> None:
    if _TRITON_BACKEND is None:
        raise ImportError(
            "Triton paged attention backend is unavailable. "
            "Install Triton and verify `src/nano_llm_infra/ops/triton/page_attention.py` imports cleanly."
        ) from _TRITON_IMPORT_ERROR


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
        _require_triton_backend()
        return _TRITON_BACKEND(q, k_cache, v_cache, block_tables, context_lens, block_size).squeeze(0)
    if impl == "cuda":
        _require_cuda_backend()
        return _PAGED_ATTENTION_EXT.paged_attention_cuda(
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

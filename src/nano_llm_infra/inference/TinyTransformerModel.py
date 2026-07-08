from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nano_llm_infra.inference.block_manager import KVCachePool
from nano_llm_infra.ops.rmsnorm import rms_norm
from nano_llm_infra.ops.paged_attention import paged_attention


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float = 1e-6, impl: str = "torch") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon
        self.impl = impl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl in {"shared", "warp_shuffle"}:
            if not x.is_cuda:
                raise ValueError("custom RMSNorm kernels require CUDA tensors")
            original_shape = x.shape
            x_3d = x.reshape(1, -1, original_shape[-1]).contiguous().float()
            weight = self.weight.contiguous().float()
            y = rms_norm(x_3d, weight, epsilon=self.epsilon, impl=self.impl)
            return y.reshape(original_shape).to(dtype=x.dtype)

        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.epsilon) * self.weight


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class TinyTransformerModel(nn.Module):
    """A tiny transformer model with explicit prefill and decode paths."""

    def __init__(
        self,
        vocab_size: int,
        device: str | torch.device = "cpu",
        hidden_size: int | None = None,
        num_layers: int = 1,
        num_heads: int = 2,
        mlp_hidden_size: int | None = None,
        rmsnorm_impl: str = "torch",
        paged_attention_impl: str = "ref",
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.hidden_size = hidden_size or 16
        self._num_layers = num_layers
        self._num_heads = num_heads
        if self.hidden_size % self._num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self._head_dim = self.hidden_size // self._num_heads
        self.mlp_hidden_size = mlp_hidden_size or 32
        self.rmsnorm_impl = rmsnorm_impl
        self.paged_attention_impl = paged_attention_impl

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn_norms = nn.ModuleList(
            [TinyRMSNorm(self.hidden_size, impl=self.rmsnorm_impl) for _ in range(self._num_layers)]
        )
        self.q_projs = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self._num_layers)]
        )
        self.k_projs = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self._num_layers)]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self._num_layers)]
        )
        self.o_projs = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self._num_layers)]
        )
        self.mlp_norms = nn.ModuleList(
            [TinyRMSNorm(self.hidden_size, impl=self.rmsnorm_impl) for _ in range(self._num_layers)]
        )
        self.mlps = nn.ModuleList(
            [TinyMLP(self.hidden_size, self.mlp_hidden_size) for _ in range(self._num_layers)]
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.to(self.device)
        self.eval()

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    def prefill(self, request, kv_cache: KVCachePool) -> torch.Tensor:
        block_table = request.block_table
        seq_len = request.num_tokens
        token_ids = torch.tensor(request.total_token_ids, device=self.device, dtype=torch.long)
        with torch.no_grad():
            hidden = self.embedding(token_ids)

        for layer_idx in range(self.num_layers):
            with torch.no_grad():
                attn_hidden = self.attn_norms[layer_idx](hidden)
                query = self.q_projs[layer_idx](attn_hidden).view(seq_len, self.num_heads, self.head_dim)
                key = self.k_projs[layer_idx](attn_hidden).view(seq_len, self.num_heads, self.head_dim)
                value = self.v_projs[layer_idx](attn_hidden).view(seq_len, self.num_heads, self.head_dim)

            for token_idx in range(seq_len):
                block_id = block_table.physical_block_for_token(token_idx)
                token_offset = token_idx % kv_cache.block_size
                kv_cache.write_token(block_id, layer_idx, token_offset, key[token_idx], value[token_idx])

            with torch.no_grad():
                q = query.transpose(0, 1).unsqueeze(0)
                k = key.transpose(0, 1).unsqueeze(0)
                v = value.transpose(0, 1).unsqueeze(0)
                attention_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attention_output = attention_output.squeeze(0).transpose(0, 1).reshape(seq_len, self.hidden_size)
                attn_output = self.o_projs[layer_idx](attention_output)
                hidden = hidden + attn_output
                hidden = hidden + self.mlps[layer_idx](self.mlp_norms[layer_idx](hidden))

        request.cached_tokens = seq_len
        with torch.no_grad():
            return self.lm_head(hidden[-1])

    def decode(self, request, kv_cache: KVCachePool) -> torch.Tensor:
        block_table = request.block_table
        token_idx = request.cached_tokens
        token_id = request.total_token_ids[token_idx]
        block_id = block_table.physical_block_for_token(token_idx)
        token_offset = token_idx % kv_cache.block_size
        with torch.no_grad():
            input_id = torch.tensor([token_id], device=self.device, dtype=torch.long)
            hidden = self.embedding(input_id).squeeze(0)

        for layer_idx in range(self.num_layers):
            with torch.no_grad():
                attn_hidden = self.attn_norms[layer_idx](hidden)
                query = self.q_projs[layer_idx](attn_hidden).view(self.num_heads, self.head_dim)
                key = self.k_projs[layer_idx](attn_hidden).view(self.num_heads, self.head_dim)
                value = self.v_projs[layer_idx](attn_hidden).view(self.num_heads, self.head_dim)
            kv_cache.write_token(block_id, layer_idx, token_offset, key, value)

            attention_output = paged_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                layer_idx=layer_idx,
                num_tokens=request.num_tokens,
                impl=self.paged_attention_impl,
            )
            with torch.no_grad():
                attn_output = self.o_projs[layer_idx](attention_output.reshape(self.hidden_size))
                hidden = hidden + attn_output
                hidden = hidden + self.mlps[layer_idx](self.mlp_norms[layer_idx](hidden))

        request.cached_tokens = request.num_tokens
        with torch.no_grad():
            return self.lm_head(hidden)

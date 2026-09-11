from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from nano_llm_infra.ops.paged_attention import paged_attention
from nano_llm_infra.ops.rmsnorm import rms_norm


class TinyRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        impl: str = "torch",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon
        self.impl = impl

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.impl in {"shared", "warp_shuffle"}:
            shape = hidden.shape
            output = rms_norm(
                hidden.reshape(1, -1, shape[-1]).contiguous().float(),
                self.weight.contiguous().float(),
                epsilon=self.epsilon,
                impl=self.impl,
            )
            return output.reshape(shape).to(hidden.dtype)

        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + self.epsilon) * self.weight


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(hidden)))


class TinyTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_size: int,
        rmsnorm_impl: str = "torch",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = TinyRMSNorm(hidden_size, impl=rmsnorm_impl)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = TinyRMSNorm(hidden_size, impl=rmsnorm_impl)
        self.mlp = TinyMLP(hidden_size, mlp_hidden_size)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden.shape
        residual = hidden
        qkv = self.qkv(self.norm1(hidden))
        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
        )
        attention = attention.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            hidden_size,
        )
        hidden = residual + self.out_proj(attention)
        return hidden + self.mlp(self.norm2(hidden))


class TinyTransformerModel(nn.Module):
    """Training、Inference 和 RL 共用的一份模型定义。"""

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_hidden_size: int | None = None,
        device: str | torch.device | None = None,
        use_activation_checkpoint: bool = False,
        rmsnorm_impl: str = "torch",
        paged_attention_impl: str = "ref",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.paged_attention_impl = paged_attention_impl
        self.use_activation_checkpoint = use_activation_checkpoint
        mlp_hidden_size = mlp_hidden_size or hidden_size * 4

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                TinyTransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_hidden_size,
                    rmsnorm_impl,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = TinyRMSNorm(hidden_size, impl=rmsnorm_impl)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if device is not None:
            self.to(device)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        for block in self.blocks:
            if self.use_activation_checkpoint and self.training:
                hidden = checkpoint(
                    block,
                    hidden,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, attention_mask)
        return self.lm_head(self.norm(hidden))

    @torch.no_grad()
    def prefill(self, request, kv_cache) -> torch.Tensor:
        seq_len = request.num_tokens
        token_ids = torch.tensor(
            request.total_token_ids,
            device=self.embedding.weight.device,
            dtype=torch.long,
        )
        hidden = self.embedding(token_ids)

        for layer_idx, block in enumerate(self.blocks):
            qkv = block.qkv(block.norm1(hidden))
            query, key, value = qkv.view(
                seq_len,
                3,
                self.num_heads,
                self.head_dim,
            ).unbind(dim=1)
            for token_idx in range(seq_len):
                block_id = request.block_table.physical_block_for_token(token_idx)
                kv_cache.write_token(
                    block_id,
                    layer_idx,
                    token_idx % kv_cache.block_size,
                    key[token_idx],
                    value[token_idx],
                )

            attention = F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                is_causal=True,
            )
            attention = attention.squeeze(0).transpose(0, 1).reshape(
                seq_len,
                self.hidden_size,
            )
            hidden = hidden + block.out_proj(attention)
            hidden = hidden + block.mlp(block.norm2(hidden))

        request.cached_tokens = seq_len
        return self.lm_head(self.norm(hidden[-1]))

    @torch.no_grad()
    def decode(self, request, kv_cache) -> torch.Tensor:
        token_idx = request.cached_tokens
        token_id = request.total_token_ids[token_idx]
        block_id = request.block_table.physical_block_for_token(token_idx)
        hidden = self.embedding(
            torch.tensor(
                token_id,
                device=self.embedding.weight.device,
                dtype=torch.long,
            )
        )

        for layer_idx, block in enumerate(self.blocks):
            query, key, value = block.qkv(block.norm1(hidden)).view(
                3,
                self.num_heads,
                self.head_dim,
            )
            kv_cache.write_token(
                block_id,
                layer_idx,
                token_idx % kv_cache.block_size,
                key,
                value,
            )
            attention = paged_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=request.block_table,
                layer_idx=layer_idx,
                num_tokens=request.num_tokens,
                impl=self.paged_attention_impl,
            )
            hidden = hidden + block.out_proj(attention.reshape(self.hidden_size))
            hidden = hidden + block.mlp(block.norm2(hidden))

        request.cached_tokens = request.num_tokens
        return self.lm_head(self.norm(hidden))

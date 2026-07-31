from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from nano_llm_infra.ops.rmsnorm import rms_norm_reference


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return rms_norm_reference(hidden, self.weight, self.epsilon)


class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = TinyRMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = TinyRMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size, bias=False),
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden.shape
        residual = hidden
        qkv = self.qkv(self.norm1(hidden))
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        attention = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
        )
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        hidden = residual + self.out_proj(attention)
        return hidden + self.mlp(self.norm2(hidden))


class TinyTrainingTransformer(nn.Module):
    """共享训练模型；TP/PP/DP 组件均围绕它组织。"""

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        use_activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.norm = TinyRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.use_activation_checkpoint = use_activation_checkpoint

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        for block in self.blocks:
            if self.use_activation_checkpoint and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.lm_head(self.norm(hidden))

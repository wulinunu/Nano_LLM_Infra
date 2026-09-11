from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from .packing import pack_experiences, segmented_causal_mask
from .types import Experience, PackedBatch, Prompt, RLConfig


class TinyPolicy(nn.Module):
    """足够演示 Rollout 与 GRPO 的单层 causal attention。"""

    def __init__(self, config: RLConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = 4
        self.head_dim = hidden // self.num_heads
        self.token_embedding = nn.Embedding(config.vocab_size, hidden)
        self.position_embedding = nn.Embedding(config.max_seq_len, hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, config.vocab_size, bias=False)

    def forward(self, batch: PackedBatch) -> torch.Tensor:
        hidden = self.token_embedding(batch.token_ids)
        hidden = hidden + self.position_embedding(batch.position_ids)
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        shape = (-1, self.num_heads, self.head_dim)
        query, key, value = query.view(shape), key.view(shape), value.view(shape)

        scores = torch.einsum("thd,shd->hts", query, key) * self.head_dim**-0.5
        mask = segmented_causal_mask(batch.cu_seqlens, len(batch.token_ids), hidden.device)
        scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        context = torch.einsum("hts,shd->thd", scores.softmax(dim=-1), value)
        hidden = self.output(context.reshape(len(hidden), -1))
        return self.lm_head(hidden)


@torch.no_grad()
def generate(
    model: TinyPolicy,
    prompts: list[Prompt],
    config: RLConfig,
    policy_version: int,
    device: torch.device,
) -> list[Experience]:
    model.eval()
    experiences = []
    for prompt in prompts:
        for _ in range(config.group_size):
            tokens = list(prompt.prompt_ids)
            response_ids: list[int] = []
            old_logprobs: list[float] = []
            for _ in range(config.response_length):
                batch = pack_token_sequence(tokens, device)
                logits = model(batch)[-1] / config.temperature
                logprobs = logits.log_softmax(dim=-1)
                token = int(torch.multinomial(logprobs.exp(), 1))
                tokens.append(token)
                response_ids.append(token)
                old_logprobs.append(float(logprobs[token]))

            experiences.append(
                Experience(
                    prompt_ids=prompt.prompt_ids,
                    response_ids=response_ids,
                    old_logprobs=old_logprobs,
                    group_id=prompt.group_id,
                    policy_version=policy_version,
                )
            )
    return experiences


def pack_token_sequence(tokens: list[int], device: torch.device) -> PackedBatch:
    length = len(tokens)
    return PackedBatch(
        token_ids=torch.tensor(tokens, dtype=torch.long, device=device),
        position_ids=torch.arange(length, device=device),
        cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=device),
        old_logprobs=torch.zeros(length, device=device),
        completion_mask=torch.zeros(length, dtype=torch.bool, device=device),
        token_group_ids=torch.zeros(length, dtype=torch.long, device=device),
        max_seqlen=length,
    )


def compute_group_advantages(experiences: list[Experience]) -> list[float]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(experiences):
        groups[item.group_id].append(index)

    advantages = torch.zeros(len(experiences))
    for indices in groups.values():
        values = torch.tensor([experiences[index].reward for index in indices])
        normalized = (values - values.mean()) / (values.std(unbiased=False) + 1e-6)
        advantages[indices] = normalized
    return advantages.tolist()


def grpo_loss(
    model: TinyPolicy,
    reference_model: TinyPolicy,
    experiences: list[Experience],
    config: RLConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = pack_experiences(experiences, device)
    logits = model(batch)
    with torch.no_grad():
        reference_logits = reference_model(batch)
    advantages = compute_group_advantages(experiences)

    losses = []
    kls = []
    for index, experience in enumerate(experiences):
        start = int(batch.cu_seqlens[index])
        prompt_end = start + len(experience.prompt_ids)
        end = int(batch.cu_seqlens[index + 1])
        predictors = torch.arange(prompt_end - 1, end - 1, device=device)
        targets = batch.token_ids[predictors + 1]
        new_logprobs = logits[predictors].log_softmax(dim=-1)
        new_logprobs = new_logprobs.gather(-1, targets[:, None]).squeeze(-1)
        reference_logprobs = reference_logits[predictors].log_softmax(dim=-1)
        reference_logprobs = reference_logprobs.gather(
            -1, targets[:, None]
        ).squeeze(-1)
        old_logprobs = batch.old_logprobs[predictors + 1]
        ratio = (new_logprobs - old_logprobs).exp()
        advantage = torch.tensor(advantages[index], device=device)
        clipped = ratio.clamp(1.0 - 0.2, 1.0 + 0.2)
        policy_loss = -torch.minimum(
            ratio * advantage,
            clipped * advantage,
        ).mean()
        log_ratio = reference_logprobs - new_logprobs
        kl = (log_ratio.exp() - log_ratio - 1.0).mean()
        losses.append(policy_loss + config.kl_beta * kl)
        kls.append(kl)
    return torch.stack(losses).mean(), torch.stack(kls).mean()

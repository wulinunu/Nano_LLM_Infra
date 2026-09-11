from __future__ import annotations

from collections import defaultdict

import torch

from nano_llm_infra.models.tiny_transformer import TinyTransformerModel
from .packing import pack_experiences, segmented_causal_mask
from .types import Experience, RLConfig


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
    model: TinyTransformerModel,
    reference_model: TinyTransformerModel,
    experiences: list[Experience],
    config: RLConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = pack_experiences(experiences, device)
    attention_mask = segmented_causal_mask(
        batch.cu_seqlens,
        len(batch.token_ids),
        device,
    )
    logits = model(batch.token_ids.unsqueeze(0), attention_mask).squeeze(0)
    with torch.no_grad():
        reference_logits = reference_model(
            batch.token_ids.unsqueeze(0),
            attention_mask,
        ).squeeze(0)
    advantages = compute_group_advantages(experiences)

    losses = []
    kls = []
    for index, experience in enumerate(experiences):
        start = int(batch.cu_seqlens[index])
        prompt_end = start + len(experience.prompt_ids)
        end = int(batch.cu_seqlens[index + 1])
        predictors = torch.arange(prompt_end - 1, end - 1, device=device)
        targets = batch.token_ids[predictors + 1]
        raw_new_logprobs = logits[predictors].log_softmax(dim=-1)
        new_logprobs = (logits[predictors] / config.temperature).log_softmax(dim=-1)
        new_logprobs = new_logprobs.gather(-1, targets[:, None]).squeeze(-1)
        raw_new_logprobs = raw_new_logprobs.gather(
            -1,
            targets[:, None],
        ).squeeze(-1)
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
        log_ratio = reference_logprobs - raw_new_logprobs
        kl = (log_ratio.exp() - log_ratio - 1.0).mean()
        losses.append(policy_loss + config.kl_beta * kl)
        kls.append(kl)
    return torch.stack(losses).mean(), torch.stack(kls).mean()

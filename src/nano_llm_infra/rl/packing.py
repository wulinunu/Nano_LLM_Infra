from __future__ import annotations

import torch

from .types import Experience, PackedBatch


def pack_experiences(
    experiences: list[Experience],
    device: torch.device | str,
) -> PackedBatch:
    token_ids: list[int] = []
    position_ids: list[int] = []
    completion_mask: list[bool] = []
    old_logprobs: list[float] = []
    token_group_ids: list[int] = []
    cu_seqlens = [0]
    max_seqlen = 0

    for experience in experiences:
        if len(experience.response_ids) != len(experience.old_logprobs):
            raise ValueError("response_ids and old_logprobs must have the same length")

        sequence = experience.prompt_ids + experience.response_ids
        token_ids.extend(sequence)
        position_ids.extend(range(len(sequence)))
        completion_mask.extend(experience.completion_mask)
        old_logprobs.extend([-1.0] * len(experience.prompt_ids) + experience.old_logprobs)
        token_group_ids.extend([experience.group_id] * len(sequence))
        cu_seqlens.append(cu_seqlens[-1] + len(sequence))
        max_seqlen = max(max_seqlen, len(sequence))

    return PackedBatch(
        token_ids=torch.tensor(token_ids, dtype=torch.long, device=device),
        position_ids=torch.tensor(position_ids, dtype=torch.long, device=device),
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        old_logprobs=torch.tensor(old_logprobs, dtype=torch.float32, device=device),
        completion_mask=torch.tensor(completion_mask, dtype=torch.bool, device=device),
        token_group_ids=torch.tensor(token_group_ids, dtype=torch.long, device=device),
        max_seqlen=max_seqlen,
    )


def segmented_causal_mask(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """True 表示两个 token 属于同一序列且满足 causal 关系。"""
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        start_index, end_index = int(start), int(end)
        length = end_index - start_index
        mask[start_index:end_index, start_index:end_index] = torch.tril(
            torch.ones((length, length), dtype=torch.bool, device=device)
        )
    return mask


def padding_stats(experiences: list[Experience]) -> tuple[int, int, float]:
    lengths = [len(item.prompt_ids) + len(item.response_ids) for item in experiences]
    packed_tokens = sum(lengths)
    padded_tokens = max(lengths) * len(lengths)
    ratio = 1.0 - packed_tokens / padded_tokens
    return packed_tokens, padded_tokens, ratio


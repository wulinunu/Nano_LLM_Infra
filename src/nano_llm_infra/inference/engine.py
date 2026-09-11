from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch.profiler import record_function

from nano_llm_infra.inference.block_manager import (
    BlockAllocationError,
    BlockAllocator,
    BlockTable,
    KVCachePool,
)
from nano_llm_infra.models.tiny_transformer import TinyTransformerModel


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    status: RequestStatus = RequestStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    generated_logprobs: list[float] = field(default_factory=list)
    cached_tokens: int = 0
    block_table: BlockTable | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
 
    @property
    def total_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.total_token_ids)

    @property
    def is_finished(self) -> bool:
        return len(self.generated_token_ids) >= self.max_new_tokens

    def append_token(self, token_id: int, logprob: float) -> None:
        if self.is_finished:
            raise RuntimeError("cannot append token to a finished request")
        self.generated_token_ids.append(token_id)
        self.generated_logprobs.append(logprob)
        if self.is_finished:
            self.status = RequestStatus.FINISHED

class IterationLevelScheduler:
    """Minimal continuous batching scheduler backed by a fixed block allocator."""

    def __init__(self, allocator: BlockAllocator, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.allocator = allocator
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.preempted: deque[Request] = deque()
        self.finished: list[Request] = []

    def add_request(self, request: Request) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def step(self) -> list[Request]:
        for queue in (self.preempted, self.waiting):
            while queue and len(self.running) < self.max_batch_size:
                request = queue[0]
                try:
                    request.block_table.ensure_block_capacity(request.num_tokens, self.allocator)
                except BlockAllocationError:
                    break

                queue.popleft()
                request.status = RequestStatus.RUNNING
                self.running.append(request)

        self.running = [request for request in self.running if not request.is_finished]
        return list(self.running)

    def finish_request(self, request: Request) -> None:
        if request in self.running:
            self.running.remove(request)
        released = request.block_table.clear()
        self.allocator.free(released)
        request.status = RequestStatus.FINISHED
        self.finished.append(request)

    def preempt_request(self, request: Request) -> Request:
        if request not in self.running:
            raise ValueError("request is not running")
        self.running.remove(request)
        released = request.block_table.clear()
        self.allocator.free(released)
        request.cached_tokens = 0
        request.status = RequestStatus.PREEMPTED
        self.preempted.append(request)
        return request


class Sampler:
    """Minimal sampler supporting temperature, top-k, and top-p."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 1.0:
            return logits
        return logits / self.temperature

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_k is None or self.top_k >= logits.size(-1):
            return logits
        topk_values, _ = torch.topk(logits, self.top_k, dim=-1)
        threshold = topk_values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > self.top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        return logits.masked_fill(mask, float("-inf"))

    def sample(
        self,
        logits: torch.Tensor,
    ) -> tuple[list[int], list[float]]:
        if logits.dim() != 2:
            raise ValueError("logits must have shape [batch_size, vocab_size]")

        filtered_logits = self._apply_temperature(logits)
        filtered_logits = self._apply_top_k(filtered_logits)
        filtered_logits = self._apply_top_p(filtered_logits)

        if self.do_sample:
            probs = torch.softmax(filtered_logits, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            token_ids = torch.argmax(filtered_logits, dim=-1)
        logprobs = torch.log_softmax(filtered_logits, dim=-1)
        sampled_logprobs = logprobs.gather(1, token_ids[:, None]).squeeze(1)
        return token_ids.tolist(), sampled_logprobs.tolist()


@dataclass(frozen=True)
class EngineStepStats:
    running: int
    waiting: int
    finished: int
    free_blocks: int
    generated: dict[str, int]


class NanoEngine:
    """Minimal continuous-batching inference engine."""

    def __init__(
        self,
        allocator: BlockAllocator,
        scheduler: IterationLevelScheduler,
        model_runner: TinyTransformerModel,
        sampler: Sampler,
        block_size: int,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.allocator = allocator
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.sampler = sampler
        self.block_size = block_size
        self.kv_cache = KVCachePool(
            num_blocks=allocator.num_blocks,
            block_size=block_size,
            num_layers=model_runner.num_layers,
            num_heads=model_runner.num_heads,
            head_dim=model_runner.head_dim,
            device=model_runner.embedding.weight.device,
        )
        self._next_request_id = 0

    def add_request(self, prompt_token_ids: list[int], max_new_tokens: int) -> Request:
        request = Request(
            request_id=str(self._next_request_id),
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
        )
        request.block_table = BlockTable(block_size=self.block_size)
        self._next_request_id += 1
        self.scheduler.add_request(request)
        return request

    def step(self) -> EngineStepStats:
        with record_function("Scheduler"):
            batch = self.scheduler.step()
        generated: dict[str, int] = {}
        if batch:
            logits = []
            for request in batch:
                if request.cached_tokens == 0:
                    with record_function("Prefill"):
                        logits.append(self.model_runner.prefill(request, self.kv_cache))
                else:
                    with record_function("Decode"):
                        logits.append(self.model_runner.decode(request, self.kv_cache))
            with record_function("Stack_and_Sample"):
                logits = torch.stack(logits, dim=0)
                token_ids, logprobs = self.sampler.sample(logits)
            with record_function("Update_Request_State"):
                for request, token_id, logprob in zip(
                    batch,
                    token_ids,
                    logprobs,
                    strict=True,
                ):
                    try:
                        request.block_table.ensure_block_capacity(request.num_tokens + 1, self.allocator)
                    except BlockAllocationError:
                        self.scheduler.preempt_request(request)
                        continue
                    request.append_token(token_id, logprob)
                    generated[request.request_id] = token_id
                    if request.is_finished:
                        self.scheduler.finish_request(request)

        return EngineStepStats(
            running=len(self.scheduler.running),
            waiting=len(self.scheduler.waiting),
            finished=len(self.scheduler.finished),
            free_blocks=self.allocator.num_free_blocks,
            generated=generated,
        )

    def release_kv_cache(self) -> int:
        released_bytes = self.kv_cache.cache.numel() * self.kv_cache.cache.element_size()
        self.kv_cache.release()
        return released_bytes

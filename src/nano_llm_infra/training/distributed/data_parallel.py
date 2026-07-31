from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from nano_llm_infra.training.amp import AmpEngine

from ..parallel_state import get_data_parallel_group


@dataclass
class GradientBucket:
    parameters: list[nn.Parameter]
    offsets: list[tuple[int, int]]
    buffer: torch.Tensor
    ready_count: int = 0
    work: object | None = None
    done_event: torch.cuda.Event | None = None


class GradientReducer:
    """DP group 内用 bucket、hook 和 comm stream 同步完整梯度。"""

    def __init__(self, model: nn.Module, bucket_size_mb: float = 1.0) -> None:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters or not parameters[0].is_cuda:
            raise ValueError("GradientReducer requires CUDA model parameters")

        self.parameters = parameters
        self.device = parameters[0].device
        self.group = get_data_parallel_group()
        self.world_size = dist.get_world_size(group=self.group)
        self.comm_stream = torch.cuda.Stream(device=self.device)
        self.buckets = self._build_buckets(bucket_size_mb)
        self.hooks = []
        self.timeline: list[str] = []

        for bucket_index, bucket in enumerate(self.buckets):
            for parameter_index, parameter in enumerate(bucket.parameters):
                hook = partial(self._on_gradient_ready, bucket_index, parameter_index)
                self.hooks.append(parameter.register_hook(hook))

    def _build_buckets(self, bucket_size_mb: float) -> list[GradientBucket]:
        max_bytes = max(1, int(bucket_size_mb * 1024 * 1024))
        buckets: list[GradientBucket] = []
        parameters: list[nn.Parameter] = []
        current_bytes = 0
        for parameter in self.parameters:
            parameter_bytes = parameter.numel() * parameter.element_size()
            if parameters and current_bytes + parameter_bytes > max_bytes:
                buckets.append(self._make_bucket(parameters))
                parameters, current_bytes = [], 0
            parameters.append(parameter)
            current_bytes += parameter_bytes
        if parameters:
            buckets.append(self._make_bucket(parameters))
        return buckets

    def _make_bucket(self, parameters: list[nn.Parameter]) -> GradientBucket:
        offsets: list[tuple[int, int]] = []
        offset = 0
        for parameter in parameters:
            next_offset = offset + parameter.numel()
            offsets.append((offset, next_offset))
            offset = next_offset
        buffer = torch.empty(offset, device=self.device, dtype=parameters[0].dtype)
        return GradientBucket(parameters, offsets, buffer)

    def begin_backward(self) -> None:
        self.timeline.clear()
        for bucket in self.buckets:
            bucket.ready_count = 0
            bucket.work = None
            bucket.done_event = None

    def _on_gradient_ready(
        self,
        bucket_index: int,
        parameter_index: int,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        bucket = self.buckets[bucket_index]
        start, end = bucket.offsets[parameter_index]
        bucket.buffer[start:end].copy_(gradient.reshape(-1))
        bucket.ready_count += 1
        if bucket.ready_count == len(bucket.parameters):
            ready_event = torch.cuda.Event()
            torch.cuda.current_stream(self.device).record_event(ready_event)
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(ready_event)
                bucket.work = dist.all_reduce(bucket.buffer, group=self.group, async_op=True)
                bucket.done_event = torch.cuda.Event()
                bucket.done_event.record(self.comm_stream)
            self.timeline.append(f"bucket {bucket_index}: async all_reduce launched")
        return gradient

    def finish_gradient_sync(self) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        for bucket_index, bucket in enumerate(self.buckets):
            if bucket.work is None or bucket.done_event is None:
                raise RuntimeError(f"bucket {bucket_index} did not receive every gradient")
            bucket.work.wait()
            current_stream.wait_event(bucket.done_event)
            bucket.buffer.div_(self.world_size)
            for parameter, (start, end) in zip(bucket.parameters, bucket.offsets, strict=True):
                parameter.grad = bucket.buffer[start:end].view_as(parameter).clone()
            self.timeline.append(f"bucket {bucket_index}: gradient average finished")

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class DataParallelRuntime:
    """DP：协调 AMP、梯度 bucket all-reduce 与 optimizer step。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        amp: AmpEngine,
        gradient_reducer: GradientReducer | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.amp = amp
        self.gradient_reducer = gradient_reducer

    def train_step(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        if self.gradient_reducer is not None:
            self.gradient_reducer.begin_backward()

        with self.amp.autocast():
            logits = self.model(token_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                token_ids[:, 1:].reshape(-1),
            )

        self.amp.backward(loss)
        if self.gradient_reducer is not None:
            self.gradient_reducer.finish_gradient_sync()
        self.amp.step(self.optimizer)
        return loss.detach()

    def close(self) -> None:
        if self.gradient_reducer is not None:
            self.gradient_reducer.close()

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nano_llm_infra.models.tiny_transformer import TinyTransformerModel
from nano_llm_infra.training.parallel_state import (
    get_pipeline_model_parallel_global_ranks,
    get_pipeline_model_parallel_group,
)


class PipelineStage(nn.Module):
    """一个连续 Transformer Block 区间组成的 PP stage。"""

    def __init__(
        self,
        embedding: nn.Embedding | None,
        blocks: list[nn.Module],
        norm: nn.Module | None,
        lm_head: nn.Linear | None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm
        self.lm_head = lm_head

    def forward_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            raise RuntimeError("only the first pipeline stage accepts token ids")
        return self.embedding(token_ids)

    def forward_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        if self.norm is not None:
            hidden = self.norm(hidden)
        if self.lm_head is not None:
            hidden = self.lm_head(hidden)
        return hidden


def build_pipeline_stage_model(model: TinyTransformerModel) -> PipelineStage:
    """从完整模型中取出当前 PP rank 应持有的连续 Block。"""
    global_ranks = get_pipeline_model_parallel_global_ranks()
    pp_rank = global_ranks.index(dist.get_rank())
    pp_world_size = len(global_ranks)
    if len(model.blocks) % pp_world_size != 0:
        raise ValueError("num_layers must be divisible by pipeline parallel size")  # 均分简化

    # 确定当前stage起始结束的位置
    blocks_per_stage = len(model.blocks) // pp_world_size
    start = pp_rank * blocks_per_stage
    end = start + blocks_per_stage

    return PipelineStage(
        embedding=model.embedding if pp_rank == 0 else None,
        blocks=list(model.blocks[start:end]),
        norm=model.norm if pp_rank == pp_world_size - 1 else None,
        lm_head=model.lm_head if pp_rank == pp_world_size - 1 else None,
    )


class PipelineRuntime:
    """
    最小 PP 运行时，同一套 send/recv 上提供三种调度：

    - naive: 每个 micro-batch 都完整走完 F 再 B（气泡最大）
    - gpipe: 先做完所有 F，再按逆序做完所有 B
    - 1f1b: warmup + F/B 交错 + cooldown
    """

    def __init__(self, stage: PipelineStage) -> None:
        self.stage = stage
        self.group = get_pipeline_model_parallel_group()
        self.global_ranks = get_pipeline_model_parallel_global_ranks()
        self.rank = self.global_ranks.index(dist.get_rank())
        self.world_size = len(self.global_ranks)
        self.input_hiddens: dict[int, torch.Tensor] = {}
        self.output_hiddens: dict[int, torch.Tensor] = {}
        self.logits: dict[int, torch.Tensor] = {}
        self.timeline: list[str] = []

    @property
    def is_first_stage(self) -> bool:
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.rank == self.world_size - 1

    def _previous_global_rank(self) -> int:
        return self.global_ranks[self.rank - 1]

    def _next_global_rank(self) -> int:
        return self.global_ranks[self.rank + 1]

    def _reset_state(self) -> None:
        self.input_hiddens.clear()
        self.output_hiddens.clear()
        self.logits.clear()
        self.timeline.clear()

    def forward_microbatch(
        self,
        mb_id: int,
        token_ids: torch.Tensor | None,
        hidden_shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        if self.is_first_stage: # 第一层不需要接 
            if token_ids is None:
                raise ValueError("the first pipeline stage needs token ids")
            hidden = self.stage.forward_embedding(token_ids)
            hidden = self.stage.forward_hidden(hidden)
        else: #接激活值并计算
            hidden = torch.empty(hidden_shape, device=device)
            dist.recv(hidden, src=self._previous_global_rank(), group=self.group) #成对的resv和send 谁先执行到 就阻塞等对方
            self.input_hiddens[mb_id] = hidden.requires_grad_() # 输入的tensor需要梯度 才能回传。存入self.input_hiddens因为有多个micro-batch
            hidden = self.stage.forward_hidden(self.input_hiddens[mb_id])

        if self.is_last_stage:
            self.logits[mb_id] = hidden
        else: #不是最后一层都需要传
            self.output_hiddens[mb_id] = hidden # 正向保存
            dist.send(hidden.detach(), dst=self._next_global_rank(), group=self.group)

        self.timeline.append(f"F{mb_id}")

    def backward_microbatch(
        self,
        mb_id: int,
        token_ids: torch.Tensor | None,
        num_microbatches: int,
    ) -> torch.Tensor | None:
        loss = None
        if self.is_last_stage: # 最后一个不需要接只需要传
            if token_ids is None or mb_id not in self.logits:
                raise ValueError("last stage needs token ids and stored logits")
            logits = self.logits.pop(mb_id)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                token_ids[:, 1:].reshape(-1),
            ) / num_microbatches
            loss.backward()
            if not self.is_first_stage:
                grad = self.input_hiddens.pop(mb_id).grad
                dist.send(grad, dst=self._previous_global_rank(), group=self.group)
            self.timeline.append(f"B{mb_id}")
            return loss.detach()

        # 接梯度并回传
        output_hidden = self.output_hiddens.pop(mb_id)
        output_grad = torch.empty_like(output_hidden)
        dist.recv(output_grad, src=self._next_global_rank(), group=self.group)
        output_hidden.backward(output_grad)

        # 传 第一个不需要传
        if not self.is_first_stage:
            grad = self.input_hiddens.pop(mb_id).grad
            dist.send(grad, dst=self._previous_global_rank(), group=self.group)

        self.timeline.append(f"B{mb_id}")
        return None

    def run_naive(
        self,
        micro_batches: list[torch.Tensor],
        hidden_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        """每个 micro-batch：整条流水线先 F 再 B。"""
        self._reset_state()
        num_microbatches = len(micro_batches)
        last_loss = None
        for mb_id, tokens in enumerate(micro_batches): # 同样的批次 不同stage拿到同样的token
            tokens_f = tokens if self.is_first_stage else None
            self.forward_microbatch(mb_id, tokens_f, hidden_shape, device)
            tokens_b = tokens if self.is_last_stage else None
            loss = self.backward_microbatch(mb_id, tokens_b, num_microbatches)
            if loss is not None:
                last_loss = loss
        return last_loss

    def run_gpipe(
        self,
        micro_batches: list[torch.Tensor],
        hidden_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        """先全部 Forward，再逆序全部 Backward。"""
        self._reset_state()
        num_microbatches = len(micro_batches)
        last_loss = None

        for mb_id, tokens in enumerate(micro_batches):
            tokens_f = tokens if self.is_first_stage else None
            self.forward_microbatch(mb_id, tokens_f, hidden_shape, device)

        for mb_id in range(num_microbatches - 1, -1, -1):
            tokens_b = micro_batches[mb_id] if self.is_last_stage else None
            loss = self.backward_microbatch(mb_id, tokens_b, num_microbatches)
            if loss is not None:
                last_loss = loss
        return last_loss

    def run_1f1b(
        self,
        micro_batches: list[torch.Tensor],
        hidden_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        Megatron 风格 1F1B：
          warmup_forwards = p - 1 - rank
          steady = m - warmup
          cooldown_backwards = warmup
        """
        self._reset_state()
        num_microbatches = len(micro_batches)
        if num_microbatches < self.world_size:
            raise ValueError("1f1b requires num_microbatches >= pipeline parallel size")

        warmup = self.world_size - 1 - self.rank
        next_forward = 0
        next_backward = 0
        last_loss = None

        for _ in range(warmup):
            tokens = micro_batches[next_forward] if self.is_first_stage else None
            self.forward_microbatch(next_forward, tokens, hidden_shape, device)
            next_forward += 1

        steady = num_microbatches - warmup
        for _ in range(steady):
            tokens_f = micro_batches[next_forward] if self.is_first_stage else None
            self.forward_microbatch(next_forward, tokens_f, hidden_shape, device)
            next_forward += 1

            tokens_b = micro_batches[next_backward] if self.is_last_stage else None
            loss = self.backward_microbatch(next_backward, tokens_b, num_microbatches)
            if loss is not None:
                last_loss = loss
            next_backward += 1

        for _ in range(warmup):
            tokens_b = micro_batches[next_backward] if self.is_last_stage else None
            loss = self.backward_microbatch(next_backward, tokens_b, num_microbatches)
            if loss is not None:
                last_loss = loss
            next_backward += 1

        return last_loss

    def run(
        self,
        schedule: str,
        micro_batches: list[torch.Tensor],
        hidden_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        if schedule == "naive":
            return self.run_naive(micro_batches, hidden_shape, device)
        if schedule == "gpipe":
            return self.run_gpipe(micro_batches, hidden_shape, device)
        if schedule == "1f1b":
            return self.run_1f1b(micro_batches, hidden_shape, device)
        raise ValueError(f"unsupported pp schedule: {schedule}")

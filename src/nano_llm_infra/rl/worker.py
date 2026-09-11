from __future__ import annotations

import os
from time import perf_counter

import torch
import torch.distributed as dist

from nano_llm_infra.inference.block_manager import BlockAllocator
from nano_llm_infra.inference.engine import (
    IterationLevelScheduler,
    NanoEngine,
    Request,
    Sampler,
)
from nano_llm_infra.models.tiny_transformer import TinyTransformerModel
from nano_llm_infra.training.distributed.zero import ZeroRuntime
from nano_llm_infra.training.parallel_state import initialize_model_parallel

from .policy import grpo_loss
from .types import Experience, Phase, Prompt, RLConfig


class ColocatedWorker:
    """同一张 GPU 上分时运行 rollout engine 和 ZeRO-2 training。"""

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        config: RLConfig,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        initialize_model_parallel()

        model_args = {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "device": self.device,
        }
        torch.manual_seed(0)
        self.training_policy = TinyTransformerModel(**model_args)
        self.rollout_policy = TinyTransformerModel(**model_args)
        self.reference_policy = TinyTransformerModel(**model_args)
        self.rollout_policy.load_state_dict(self.training_policy.state_dict())
        self.reference_policy.load_state_dict(self.training_policy.state_dict())
        self.reference_policy.requires_grad_(False)
        self.reference_policy.eval()
        self.zero_runtime = ZeroRuntime(
            self.training_policy,
            zero_stage=config.zero_stage,
            bucket_size_mb=config.zero_bucket_size_mb,
            lr=config.learning_rate,
        )
        torch.manual_seed(rank + 1)

        self.phase = Phase.ROLLOUT
        self.policy_version = 0
        self.last_rollout_metrics: dict[str, float] = {}

    @torch.no_grad()
    def rollout(
        self,
        prompts: list[Prompt],
    ) -> tuple[list[Experience], dict[str, float]]:
        self.phase = Phase.ROLLOUT
        self.rollout_policy.eval()
        before_mb = torch.cuda.memory_allocated(self.device) / 1024**2

        allocator = BlockAllocator(self.config.kv_num_blocks)
        scheduler = IterationLevelScheduler(
            allocator,
            self.config.rollout_batch_size,
        )
        engine = NanoEngine(
            allocator=allocator,
            scheduler=scheduler,
            model_runner=self.rollout_policy,
            sampler=Sampler(
                temperature=self.config.temperature,
                do_sample=True,
            ),
            block_size=self.config.kv_block_size,
        )
        allocated_mb = torch.cuda.memory_allocated(self.device) / 1024**2

        requests: list[tuple[Prompt, Request]] = []
        for prompt in prompts:
            for _ in range(self.config.group_size):
                request = engine.add_request(
                    prompt.prompt_ids,
                    self.config.response_length,
                )
                requests.append((prompt, request))

        peak_blocks = 0
        while scheduler.waiting or scheduler.running or scheduler.preempted:
            engine.step()
            peak_blocks = max(peak_blocks, allocator.num_allocated_blocks)

        experiences = [
            Experience(
                prompt_ids=prompt.prompt_ids,
                response_ids=request.generated_token_ids,
                old_logprobs=request.generated_logprobs,
                group_id=prompt.group_id,
                policy_version=self.policy_version,
            )
            for prompt, request in requests
        ]

        before_release_mb = torch.cuda.memory_allocated(self.device) / 1024**2
        kv_pool_mb = engine.release_kv_cache() / 1024**2
        del engine
        torch.cuda.empty_cache()
        released_mb = (
            before_release_mb
            - torch.cuda.memory_allocated(self.device) / 1024**2
        )
        self.last_rollout_metrics = {
            "kv_pool_mb": kv_pool_mb,
            "kv_blocks_peak": float(peak_blocks),
            "rollout_memory_mb": allocated_mb,
            "released_memory_mb": released_mb,
            "base_memory_mb": before_mb,
        }
        return experiences, self.last_rollout_metrics

    def train(self, experiences: list[Experience]) -> dict[str, float]:
        self.phase = Phase.TRAIN
        self.training_policy.train()
        self.zero_runtime.optimizer.zero_grad(set_to_none=True)
        loss, kl = grpo_loss(
            self.training_policy,
            self.reference_policy,
            experiences,
            self.config,
            self.device,
        )

        local_samples = torch.tensor(float(len(experiences)), device=self.device)
        global_samples = local_samples.clone()
        dist.all_reduce(global_samples, group=self.zero_runtime.group)
        scaled_loss = loss * local_samples * self.world_size / global_samples
        self.zero_runtime.backward_and_step(scaled_loss)

        grad_norm_sq = torch.zeros((), device=self.device)
        for bucket in self.zero_runtime.buckets:
            if bucket.param_shard.grad is not None:
                grad_norm_sq += bucket.param_shard.grad.float().square().sum()
        dist.all_reduce(grad_norm_sq, group=self.zero_runtime.group)

        return {
            "loss": float(loss.detach()),
            "kl": float(kl.detach()),
            "reward": sum(item.reward for item in experiences) / len(experiences),
            "grad_norm": float(grad_norm_sq.sqrt()),
            "samples": float(len(experiences)),
            "zero_backward_memory_mb": self.zero_runtime.backward_memory_mb[-1],
        }

    @torch.no_grad()
    def sync_weights(self, mode: str = "gpu") -> dict[str, float]:
        self.phase = Phase.SYNC
        torch.cuda.synchronize()
        start = perf_counter()

        train_parameters = dict(self.training_policy.named_parameters())
        rollout_parameters = dict(self.rollout_policy.named_parameters())
        cpu_bytes = 0
        if mode == "cpu":
            state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in train_parameters.items()
            }
            cpu_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
            for name, parameter in rollout_parameters.items():
                parameter.copy_(state[name].to(self.device))
        else:
            total = sum(parameter.numel() for parameter in train_parameters.values())
            buffer = torch.empty(total, device=self.device)
            offset = 0
            for parameter in train_parameters.values():
                size = parameter.numel()
                buffer[offset : offset + size].copy_(parameter.reshape(-1))
                offset += size

            offset = 0
            for parameter in rollout_parameters.values():
                size = parameter.numel()
                parameter.copy_(buffer[offset : offset + size].view_as(parameter))
                offset += size

        torch.cuda.synchronize()
        self.policy_version += 1
        return {
            "latency_ms": (perf_counter() - start) * 1000,
            "cpu_copy_mb": cpu_bytes / 1024**2,
            "version": float(self.policy_version),
        }

    def memory_mb(self) -> float:
        return torch.cuda.memory_allocated(self.device) / 1024**2

    @torch.no_grad()
    def weights_match(self) -> bool:
        return all(
            torch.equal(train, rollout)
            for train, rollout in zip(
                self.training_policy.parameters(),
                self.rollout_policy.parameters(),
                strict=True,
            )
        )

    def close(self) -> None:
        self.zero_runtime.close()
        dist.destroy_process_group()

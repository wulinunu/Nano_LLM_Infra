from __future__ import annotations

import os
from time import perf_counter

import torch
import torch.distributed as dist

from .policy import TinyPolicy, generate, grpo_loss
from .types import Experience, Phase, Prompt, RLConfig


class ColocatedWorker:
    """一个 Worker 在同一设备上分时运行 Rollout 与 Training。"""

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

        torch.manual_seed(0)
        self.training_policy = TinyPolicy(config).to(self.device)
        self.rollout_policy = TinyPolicy(config).to(self.device)
        self.reference_policy = TinyPolicy(config).to(self.device)
        self.rollout_policy.load_state_dict(self.training_policy.state_dict())
        self.reference_policy.load_state_dict(self.training_policy.state_dict())
        self.reference_policy.requires_grad_(False)
        self.reference_policy.eval()
        self.optimizer = torch.optim.AdamW(
            self.training_policy.parameters(),
            lr=config.learning_rate,
        )
        torch.manual_seed(rank + 1)

        self.phase = Phase.ROLLOUT
        self.policy_version = 0

    def rollout(self, prompts: list[Prompt]) -> list[Experience]:
        self.phase = Phase.ROLLOUT
        experiences = generate(
            self.rollout_policy,
            prompts,
            self.config,
            self.policy_version,
            self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return experiences

    def train(
        self,
        experiences: list[Experience],
    ) -> dict[str, float]:
        self.phase = Phase.TRAIN
        self.training_policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, kl = grpo_loss(
            self.training_policy,
            self.reference_policy,
            experiences,
            self.config,
            self.device,
        )
        loss.backward()

        gradients = [
            parameter.grad.reshape(-1)
            for parameter in self.training_policy.parameters()
            if parameter.grad is not None
        ]
        flat_grad = torch.cat(gradients)
        sample_count = torch.tensor(float(len(experiences)), device=self.device)
        flat_grad *= sample_count
        dist.all_reduce(flat_grad)
        dist.all_reduce(sample_count)
        flat_grad /= sample_count

        offset = 0
        for parameter in self.training_policy.parameters():
            size = parameter.numel()
            parameter.grad.copy_(flat_grad[offset : offset + size].view_as(parameter))
            offset += size
        self.optimizer.step()

        metrics = {
            "loss": float(loss.detach()),
            "kl": float(kl.detach()),
            "reward": sum(item.reward for item in experiences) / len(experiences),
            "grad_norm": float(flat_grad.norm()),
            "samples": float(len(experiences)),
        }
        return metrics

    @torch.no_grad()
    def sync_weights(self, mode: str = "gpu") -> dict[str, float]:
        self.phase = Phase.SYNC
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()

        train_parameters = dict(self.training_policy.named_parameters())
        rollout_parameters = dict(self.rollout_policy.named_parameters())
        if train_parameters.keys() != rollout_parameters.keys():
            raise ValueError("training and rollout parameter names must match")

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

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.policy_version += 1
        return {
            "latency_ms": (perf_counter() - start) * 1000,
            "cpu_copy_mb": cpu_bytes / 1024**2,
            "version": float(self.policy_version),
        }

    def memory_mb(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0

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
        dist.destroy_process_group()

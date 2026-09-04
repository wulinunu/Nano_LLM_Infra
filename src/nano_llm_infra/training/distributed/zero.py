from __future__ import annotations

"""基于 Bucket 的 ZeRO-0/1/2/3 极简实现。"""

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Adam

from nano_llm_infra.training.parallel_state import get_data_parallel_group


class ZeroBucket:
    def __init__(
        self,
        parameters: list[nn.Parameter],
        world_size: int,
        rank: int,
        device: torch.device,
        stage: int,
    ) -> None:
        self.parameters = parameters
        self.offsets: list[tuple[int, int]] = []
        self.original_shapes: dict[nn.Parameter, torch.Size] = {}
        offset = 0
        for p in parameters:
            self.original_shapes[p] = p.shape
            next_offset = offset + p.numel()
            self.offsets.append((offset, next_offset))
            offset = next_offset

        padding_numel = (world_size - offset % world_size) % world_size
        self.total_numel_with_padding = offset + padding_numel
        shard_size = self.total_numel_with_padding // world_size if stage != 0 else self.total_numel_with_padding

        with torch.no_grad():
            flat_param = torch.zeros(
                self.total_numel_with_padding,
                device=device,
                dtype=parameters[0].dtype,
            )
            for p, (start, end) in zip(parameters, self.offsets):
                flat_param[start:end].copy_(p.data.flatten())

        # 根据元素拆分 分片有可能把一整个参数分开
        start = rank * shard_size if stage != 0 else 0
        end = start + shard_size
        self.param_shard = nn.Parameter(flat_param[start:end].clone())

        self.grad_buffer: torch.Tensor | None = None # 当前bucket内的完整梯度（尚未归约和分片）
        self.ready_count = 0 #执行时每张卡上都是完整参数 需要凑满len(bucket.parameters)


class ZeroRuntime:
    def __init__(
        self,
        module: nn.Module,
        zero_stage: int = 2,
        bucket_size_mb: float = 1.0,
        lr: float = 1e-3,
    ) -> None:
        if zero_stage not in {0, 1, 2, 3}:
            raise ValueError("zero_stage must be 0, 1, 2, or 3")
        self.module = module
        self.stage = zero_stage
        self.group = get_data_parallel_group()
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.device = next(module.parameters()).device
        self.backward_memory_mb: list[float] = []

        self.buckets: list[ZeroBucket] = []
        self.param_locations: dict[nn.Parameter, tuple[ZeroBucket, int]] = {} #记录属于哪个bucket和在这个bucket的哪一个位置
        max_bytes = max(1, int(bucket_size_mb * 1024 * 1024))

        current_bytes = 0
        current_params: list[nn.Parameter] = []
        
        # 分桶
        for p in module.parameters():
            if not p.requires_grad:
                continue
            p_bytes = p.numel() * p.element_size()
            if current_params and current_bytes + p_bytes > max_bytes:
                self.buckets.append(
                    ZeroBucket(
                        current_params,
                        self.world_size,
                        self.rank,
                        self.device,
                        self.stage,
                    )
                )
                current_params, current_bytes = [], 0
            current_params.append(p)
            current_bytes += p_bytes
        if current_params:
            self.buckets.append(
                ZeroBucket(
                    current_params,
                    self.world_size,
                    self.rank,
                    self.device,
                    self.stage,
                )
            )

        self.optimizer = Adam([b.param_shard for b in self.buckets], lr=lr) # 直接更新到桶的分片梯度上

        # ZeRO-3 初始化后只长期保存参数分片 桶构造完之后就可以删
        if self.stage == 3:
            for bucket in self.buckets:
                for p in bucket.parameters:
                    p.data = torch.empty(0, device=self.device, dtype=p.dtype)

        self.hooks = []
        for bucket in self.buckets:
            for i, p in enumerate(bucket.parameters):
                self.param_locations[p] = (bucket, i)
                self.hooks.append(p.register_post_accumulate_grad_hook(self._copy_grad_to_bucket))

    def _copy_grad_to_bucket(self, param: nn.Parameter) -> None:
        if param.grad is None:
            return

        bucket, param_index = self.param_locations[param]
        if bucket.grad_buffer is None:
            bucket.grad_buffer = torch.zeros(
                bucket.total_numel_with_padding,
                device=self.device,
                dtype=param.dtype,
            )

        start, end = bucket.offsets[param_index]
        bucket.grad_buffer[start:end].copy_(param.grad.flatten()) # 将梯度复制到桶内
        if self.stage in (2, 3):
            param.grad = None

        bucket.ready_count += 1
        if bucket.ready_count >= len(bucket.parameters): #桶满
            if self.stage == 3:
                for p in bucket.parameters:
                    p.data = torch.empty(0, device=self.device, dtype=p.dtype)

            if self.stage == 0:
                dist.all_reduce(bucket.grad_buffer, group=self.group)
                bucket.grad_buffer.div_(self.world_size)
                bucket.param_shard.grad = bucket.grad_buffer
            else:
                bucket.param_shard.grad = torch.empty_like(bucket.param_shard)
                dist.reduce_scatter_tensor(
                    bucket.param_shard.grad,
                    bucket.grad_buffer,
                    group=self.group,
                )
                bucket.param_shard.grad.div_(self.world_size)

            bucket.grad_buffer = None
            bucket.ready_count = 0
        else:
            return

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.stage == 3:
            # 极简 ZeRO-3：前向前一次性拉取完整参数，反向后释放
            for bucket in self.buckets:
                # 从各卡汇总参数
                flat_param = torch.empty(
                    bucket.total_numel_with_padding,
                    device=self.device,
                    dtype=bucket.param_shard.dtype,
                )
                dist.all_gather_into_tensor(flat_param, bucket.param_shard.data, group=self.group)
                for p, (start, end) in zip(bucket.parameters, bucket.offsets):
                    p.data = flat_param[start:end].view(bucket.original_shapes[p])
        return self.module(token_ids)

    def backward_and_step(self, loss: torch.Tensor) -> None:
        for bucket in self.buckets:
            bucket.ready_count = 0
        loss.backward() #调用hook 将梯度复制到桶内
        self.backward_memory_mb.append(
            torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        )
        #这里如果是zero2 3 的话在hook里面已经释放过了 只有1在这里释放
        for p in self.module.parameters():
            p.grad = None 

        self.optimizer.step()

        if self.stage in (1, 2):
            for bucket in self.buckets:
                flat_param = torch.empty(
                    bucket.total_numel_with_padding,
                    device=self.device,
                    dtype=bucket.param_shard.dtype,
                )
                dist.all_gather_into_tensor(flat_param, bucket.param_shard.data, group=self.group)
                for p, (start, end) in zip(bucket.parameters, bucket.offsets):
                    p.data.copy_(flat_param[start:end].view(bucket.original_shapes[p]))
        elif self.stage == 0:
            for bucket in self.buckets:
                for p, (start, end) in zip(bucket.parameters, bucket.offsets):
                    p.data.copy_(bucket.param_shard.data[start:end].view(bucket.original_shapes[p]))

    def run_step(self, token_ids: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        out = self.forward(token_ids)
        self.backward_and_step(out.sum())

    def close(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

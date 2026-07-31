from __future__ import annotations

"""
基于 Bucket 机制的 ZeRO-1 / ZeRO-2 极简实现。
- 利用 Bucket 将多个散碎的参数拼成大块 (Buffer)
- 凑满一个 Bucket 后，调用 reduce_scatter_tensor 获得本卡对应的梯度分片
- 优化器只更新本卡的 param_shard，然后通过 all_gather_into_tensor 将更新后的参数广播回全量模型
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Adam

from nano_llm_infra.training.parallel_state import get_data_parallel_group


class ZeroBucket:
    def __init__(self, parameters: list[nn.Parameter], bucket_index: int, world_size: int, rank: int, device: torch.device):
        self.parameters = parameters
        self.bucket_index = bucket_index
        self.world_size = world_size
        self.rank = rank
        self.device = device
        
        # 1. 记录每个参数在 Bucket 里的偏移量和原始形状 (专门为 ZeRO-3 恢复用)
        self.offsets = []
        self.original_shapes = {}
        offset = 0
        for p in parameters:
            self.original_shapes[p] = p.shape
            next_offset = offset + p.numel()
            self.offsets.append((offset, next_offset))
            offset = next_offset
            
        # 2. 补齐 Buffer 尺寸使其能被 world_size 整除
        self.total_numel = offset
        pad = (world_size - self.total_numel % world_size) % world_size
        self.padded_numel = self.total_numel + pad
        
        # ZeRO-0 不切分，shard_size 就是全量大小
        self.shard_size = self.padded_numel if bucket_index == -1 else self.padded_numel // world_size
        
        # 3. 构建全量参数 Buffer (仅在初始化使用，用于切分初始权重)
        with torch.no_grad():
            flat_param = torch.zeros(self.padded_numel, device=device, dtype=parameters[0].dtype)
            for p, (start, end) in zip(parameters, self.offsets):
                flat_param[start:end].copy_(p.data.flatten())
                
        # 4. 本卡负责优化的 Parameter Shard
        if bucket_index == -1: # ZeRO-0 的特殊标记
            start = 0
            end = self.shard_size
        else:
            start = rank * self.shard_size
            end = start + self.shard_size
        self.param_shard = nn.Parameter(flat_param[start:end].clone())
        
        # 5. 梯度缓冲（动态分配，用完即焚）
        self.grad_buffer = None
        # 切片梯度 (Optimizer 需要长久持有)
        self.grad_shard = torch.zeros(self.shard_size, device=device, dtype=parameters[0].dtype)
        
        self.ready_count = 0
        self.flat_param_cache = None  # ZeRO-3 缓存拉取到的全量参数


class ZeROWrapper:
    def __init__(self, module: nn.Module, stage: int = 2, bucket_size_mb: float = 1.0, lr: float = 1e-3) -> None:
        if stage not in {0, 1, 2, 3}:
            raise ValueError("This bucketed ZeroRuntime currently supports stage 0, 1, 2, and 3.")
            
        self.module = module
        self.stage = stage
        self.group = get_data_parallel_group()
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.device = next(module.parameters()).device
        
        # 1. 构建 Buckets
        self.buckets: list[ZeroBucket] = []
        max_bytes = max(1, int(bucket_size_mb * 1024 * 1024))
        
        current_bytes = 0
        current_params = []
        for p in module.parameters():
            if not p.requires_grad:
                continue
            p_bytes = p.numel() * p.element_size()
            if current_params and current_bytes + p_bytes > max_bytes:
                idx = -1 if self.stage == 0 else len(self.buckets)
                self.buckets.append(ZeroBucket(current_params, idx, self.world_size, self.rank, self.device))
                current_params, current_bytes = [], 0
            current_params.append(p)
            current_bytes += p_bytes
        if current_params:
            idx = -1 if self.stage == 0 else len(self.buckets)
            self.buckets.append(ZeroBucket(current_params, idx, self.world_size, self.rank, self.device))
            
        # 2. 优化器只管理所有 bucket 切分出的 param_shard
        self.optimizer = Adam([b.param_shard for b in self.buckets], lr=lr)
        
        # ZeRO-3 初始清空全量参数显存 (清空的是原始 tensor 本身的数据)
        if self.stage == 3:
            for bucket in self.buckets:
                for p in bucket.parameters:
                    # 不能直接改变 Parameter 对象，只是把底层的 Tensor 变空
                    p.data = torch.empty(0, device=self.device, dtype=p.dtype)
        
        # 3. 挂载 Hook 进行通信
        self.hooks = []
        for bucket in self.buckets:
            for i, p in enumerate(bucket.parameters):
                # 使用 register_post_accumulate_grad_hook
                self.hooks.append(
                    p.register_post_accumulate_grad_hook(self._make_hook(bucket, i))
                )

    def _make_hook(self, bucket: ZeroBucket, param_index: int):
        def hook(param: torch.Tensor):
            grad = param.grad
            if grad is None:
                return
            
            # 按需动态分配 grad_buffer
            if bucket.grad_buffer is None:
                bucket.grad_buffer = torch.zeros(bucket.padded_numel, device=self.device, dtype=param.dtype)

            # 把算好的梯度拷入 bucket buffer
            start, end = bucket.offsets[param_index]
            bucket.grad_buffer[start:end].copy_(grad.flatten())  # 不用 detach 了，原样拷
            
            # ZeRO-2/3 核心：算完立马释放全量梯度！
            # 【终极修复】：不但要在 param 上把它设为 None，我们还要 del 掉刚才捕获到的 grad 本身！
            # 否则它在 Python 的局部变量里还存活着！
            if self.stage in (2, 3):
                param.grad = None
            del grad
                
            bucket.ready_count += 1
            
            # Bucket 满了，触发通信
            if bucket.ready_count == len(bucket.parameters):
                if self.stage == 0:
                    # ZeRO-0 (即纯 DP): 使用 All-Reduce 拿到全量梯度总和
                    dist.all_reduce(bucket.grad_buffer, group=self.group)
                    bucket.grad_buffer.div_(self.world_size)
                    # 全量 Buffer 就是我们要更新的 Shard
                    bucket.grad_shard.copy_(bucket.grad_buffer)
                else:
                    # ZeRO-1/2/3: 调用真实的 reduce_scatter_tensor (把全量 buffer 缩减成本卡负责的 shard)
                    dist.reduce_scatter_tensor(bucket.grad_shard, bucket.grad_buffer, group=self.group)
                    bucket.grad_shard.div_(self.world_size)
                
                # 【内存优化的关键所在】：通信完毕后，立刻释放局部的 grad_buffer
                # 这样 ZeRO-2 真正的内存优势就能体现出来了！
                bucket.grad_buffer = None
                bucket.ready_count = 0  # 重置以便下一次反向传播
                
                # ZeRO-3 核心：反向传播算完梯度后，连前向拉取的全量参数也一并释放！
                if self.stage == 3:
                    bucket.flat_param_cache = None
                    for p in bucket.parameters:
                        p.data = torch.empty(0, device=self.device, dtype=p.dtype)
                
        return hook

    def _gather_bucket(self, bucket: ZeroBucket):
        """ZeRO-3 用：在使用该 bucket 前，临时通过 all_gather 拉取全量参数。"""
        if bucket.flat_param_cache is None:
            bucket.flat_param_cache = torch.empty(bucket.padded_numel, device=self.device, dtype=bucket.param_shard.dtype)
            dist.all_gather_into_tensor(bucket.flat_param_cache, bucket.param_shard.data, group=self.group)
            
            # 由于一开始 p.data 被设为了空 tensor，直接 view_as 会报错 size 没法匹配
            # 必须用我们初始化时记录的 offset 来计算正确的 shape 赋回去
            for p, (start, end) in zip(bucket.parameters, bucket.offsets):
                # 获取该 parameter 原始的 shape
                original_shape = bucket.original_shapes[p]
                p.data = bucket.flat_param_cache[start:end].view(original_shape)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.stage == 3:
            # 极简版 ZeRO-3：前向传播前，一把梭哈拉回所有 bucket 的参数
            for bucket in self.buckets:
                self._gather_bucket(bucket)
            out = self.module(token_ids)
            # 前向算完后，参数还得留在内存里供反向传播用，直到 hook 里才释放
            return out
            
        # ZeRO 1/2 前向就是普通的前向，参数在 step() 完之后已经被 all_gather 就位了
        return self.module(token_ids)

    def backward_and_step(self, loss: torch.Tensor) -> None:
        # 1. 重置 bucket 状态
        for bucket in self.buckets:
            bucket.ready_count = 0
            
        # 2. 反向传播，触发 hooks，完成 reduce_scatter
        loss.backward()
        
        # 【核心探针】：在这个瞬间，所有的中间激活都已经被释放了，
        # ZeRO-1 和 ZeRO-2 真正的差异就是在这里体现的！
        # 我们把这个瞬间的显存记录下来。
        if hasattr(self, "mem_probe_list"):
            self.mem_probe_list.append(torch.cuda.memory_allocated(self.device) / (1024 * 1024))
        
        # ZeRO-0/1 清理 grad (如果是 ZeRO-2，在 hook 里已经早早释放了)
        # 如果你想在 Profiler 里看清楚 ZeRO-1 的梯度占用，这里绝对不能提前把梯度清了！
        # 注释掉下面这段：
        # if self.stage in (0, 1):
        #     for p in self.module.parameters():
        #         p.grad = None
        
        # 3. 优化器更新本地 param_shard
        for bucket in self.buckets:
            bucket.param_shard.grad = bucket.grad_shard
            
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        # 4. Step 完之后
        if self.stage in (1, 2):
            # ZeRO-1/2 将更新后的 param_shard 重新 all_gather 拼回模型全量参数，为下次前向准备
            for bucket in self.buckets:
                flat_param = torch.empty(bucket.padded_numel, device=self.device, dtype=bucket.param_shard.dtype)
                dist.all_gather_into_tensor(flat_param, bucket.param_shard.data, group=self.group)
                for p, (start, end) in zip(bucket.parameters, bucket.offsets):
                    p.data.copy_(flat_param[start:end].view(bucket.original_shapes[p]))
        elif self.stage == 0:
            # ZeRO-0：param_shard 就是全量参数，直接拷回即可，无需通信
            for bucket in self.buckets:
                for p, (start, end) in zip(bucket.parameters, bucket.offsets):
                    p.data.copy_(bucket.param_shard.data[start:end].view(bucket.original_shapes[p]))
        else:
            # ZeRO-3 维持空壳，下次前向再拉
            pass

    def run_step(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        # 手动清理原模型的梯度，这对 ZeRO-0 和 ZeRO-1 非常重要
        # ZeRO-2/3 在 backward 期间已经设置为 None 了，但是保险起见再跑一次
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad = None

        out = self.forward(token_ids)
        self.backward_and_step(out.sum())
        
        # 【重点】：我们在这里手动调用一下 PyTorch 的垃圾回收！
        # 否则因为 Python 的引用延迟，算出来的张量其实还是占着显存池
        # torch.cuda.empty_cache() 是大忌（极慢），所以我们只等它自然回收
        # 但有些游离的变量，我们可以等函数 return out.detach() 后让它失去作用域。
        
        return out.detach()

    def close(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class ZeroRuntime:
    def __init__(self, model: nn.Module, zero_stage: int, bucket_size_mb: float = 1.0, lr: float = 1e-3) -> None:
        if zero_stage not in {0, 1, 2, 3}:
            print(f"Warning: Bucketed ZeroRuntime supports stage 0, 1, 2 and 3. Fallback to stage 2.")
            zero_stage = 2
        self.engine = ZeROWrapper(model, stage=zero_stage, bucket_size_mb=bucket_size_mb, lr=lr)
        self.engine.mem_probe_list = []  # 注入探针列表（backward后）

    def run_step(self, token_ids: torch.Tensor) -> torch.Tensor | None:
        return self.engine.run_step(token_ids)

    def close(self) -> None:
        if hasattr(self.engine, "close"):
            self.engine.close()

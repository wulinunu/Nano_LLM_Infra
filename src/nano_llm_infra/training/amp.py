from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.optim import Optimizer


class AmpEngine:
    """在torch.amp/torch.autocast的基础上简单包了一层"""

    def __init__(self, device: torch.device, precision: str = "fp16") -> None:
        if precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError("precision must be fp16, bf16, or fp32")

        self.device = device
        self.precision = precision
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
        self.enabled = device.type == "cuda" and precision != "fp32"
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.enabled and precision == "fp16",
        )

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype) #自动判断并作精度转换

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(self, optimizer: Optimizer) -> None:
        # 在这里 unscale，optimizer 看到的是正常尺度的 FP32 梯度。
        self.scaler.unscale_(optimizer) # 将缩放的梯度变回去
        self.scaler.step(optimizer) # 如果梯度没有溢出则正常更新 如果梯度溢出则不更新 避免权重被inf/nan污染
        self.scaler.update() #看是否需要再变化scale 如果出现溢出则减少 如果没有溢出则增大

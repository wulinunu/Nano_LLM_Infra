from __future__ import annotations

import torch

from nano_llm_infra import _C


def rms_norm_reference(input: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """PyTorch reference implementation for correctness checks."""
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    return input * torch.rsqrt(variance + epsilon) * gamma


def rms_norm_shared(input: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Run the shared-memory CUDA RMSNorm implementation."""
    return _C.rms_norm_shared(input, gamma, float(epsilon))


def rms_norm_warp_shuffle(input: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Run the warp-shuffle CUDA RMSNorm implementation."""
    return _C.rms_norm_warp_shuffle(input, gamma, float(epsilon))


def rms_norm(
    input: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
    impl: str = "warp_shuffle",
) -> torch.Tensor:
    """Dispatch to one of the custom CUDA RMSNorm implementations."""
    if impl == "warp_shuffle":
        return rms_norm_warp_shuffle(input, gamma, epsilon)
    if impl == "shared":
        return rms_norm_shared(input, gamma, epsilon)
    raise ValueError(f"Unsupported RMSNorm implementation: {impl}")

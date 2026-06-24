from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from nano_llm_infra.ops import rms_norm_reference, rms_norm_shared, rms_norm_warp_shuffle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark custom CUDA RMSNorm kernels.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> None:
    max_error = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    print(f"{name:>16} correctness: {passed} | max_error={max_error:.6e}")
    if not passed:
        raise AssertionError(f"{name} output does not match PyTorch reference")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Please run this benchmark on a CUDA machine.")

    shape = (args.batch_size, args.seq_len, args.hidden_dim)
    print(f"shape={shape}, warmup={args.warmup}, iters={args.iters}, epsilon={args.epsilon}")

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    gamma = torch.randn(args.hidden_dim, device="cuda", dtype=torch.float32)

    expected = rms_norm_reference(x, gamma, args.epsilon)
    shared = rms_norm_shared(x, gamma, args.epsilon)
    warp_shuffle = rms_norm_warp_shuffle(x, gamma, args.epsilon)

    check_close("shared", shared, expected, args.rtol, args.atol)
    check_close("warp_shuffle", warp_shuffle, expected, args.rtol, args.atol)

    benchmarks: dict[str, Callable[[], torch.Tensor]] = {
        "torch_reference": lambda: rms_norm_reference(x, gamma, args.epsilon),
        "shared": lambda: rms_norm_shared(x, gamma, args.epsilon),
        "warp_shuffle": lambda: rms_norm_warp_shuffle(x, gamma, args.epsilon),
    }

    print("\nLatency:")
    for name, fn in benchmarks.items():
        latency_ms = time_cuda(fn, args.warmup, args.iters)
        print(f"{name:>16}: {latency_ms:.4f} ms")


if __name__ == "__main__":
    main()

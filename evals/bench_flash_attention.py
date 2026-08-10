from __future__ import annotations

import argparse

import torch
import triton

from nano_llm_infra.ops.flash_attention import flash_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Mini-FlashAttention.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--block-m", type=int, nargs="+", default=[16])
    parser.add_argument("--block-n", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--causal", action="store_true")
    return parser.parse_args()


def peak_memory_mb(fn) -> float:
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def benchmark_case(args: argparse.Namespace, seq_len: int, head_dim: int) -> None:
    shape = (args.batch_size, args.num_heads, seq_len, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    reference = flash_attention(q, k, v, causal=args.causal, impl="ref")
    implementations = [("ref", lambda: flash_attention(q, k, v, args.causal, impl="ref"))]
    implementations.extend(
        (
            f"triton_m{block_m}_n{block_n}",
            lambda current_m=block_m, current_n=block_n: flash_attention(
                q,
                k,
                v,
                causal=args.causal,
                impl="triton",
                block_m=current_m,
                block_n=current_n,
            ),
        )
        for block_m in args.block_m
        for block_n in args.block_n
    )

    print(f"\nshape={shape} causal={args.causal}")

    # 先让所有实现完成 JIT、cuBLAS 初始化和缓存建立，再正式计时。
    for _, fn in implementations:
        fn()
    torch.cuda.synchronize()

    for name, fn in implementations:
        out = fn()
        max_error = (out - reference).abs().max().item()
        torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-3)

        p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
            fn,
            warmup=args.warmup_ms,
            rep=args.rep_ms,
            quantiles=[0.5, 0.2, 0.8],
        )
        tokens_per_s = args.batch_size * seq_len / (p50_ms / 1000)
        memory_mb = peak_memory_mb(fn)
        print(
            f"{name:>16}: p50={p50_ms:8.4f} ms "
            f"[p20={p20_ms:.4f}, p80={p80_ms:.4f}] "
            f"{tokens_per_s:10.1f} tok/s peak_mem={memory_mb:8.1f} MB "
            f"max_error={max_error:.3e}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mini-FlashAttention benchmark")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    print(
        "输出列说明：\n"
        "  implementation : ref 或 Triton 的 Tile 配置\n"
        "  p50             : 延迟中位数，主要性能指标\n"
        "  p20 / p80       : 延迟波动范围，差距越小越稳定\n"
        "  tok/s           : 每秒处理的 Token 数\n"
        "  peak_mem        : 峰值显存占用\n"
        "  max_error       : 相对 PyTorch Reference 的最大绝对误差"
    )
    for seq_len in args.seq_lens:
        for head_dim in args.head_dims:
            benchmark_case(args, seq_len, head_dim)


if __name__ == "__main__":
    main()

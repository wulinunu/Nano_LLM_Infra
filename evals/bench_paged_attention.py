from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from nano_llm_infra import _paged_attention
from nano_llm_infra.inference.block_manager import BlockTable, KVCachePool
from nano_llm_infra.ops.pytorch_ref.paged_attention_ref import paged_attention_ref
from nano_llm_infra.ops.triton.paged_attention import paged_attention_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PagedAttention backends.")
    # 增加 context_lens 的范围，测一下更长的序列
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--block-size", type=int, default=16)
    # 大幅增加 num_heads，模拟真实大模型 (比如 Llama-2-7B 有 32 个 head)
    # 这样可以增加 Grid Size，把 GPU 的计算单元 (SM) 喂饱
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    # 增加物理块总数，确保能装下 4096 的长序列
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def build_case(
    num_tokens: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, KVCachePool, BlockTable, int]:
    required_blocks = (num_tokens + block_size - 1) // block_size
    if required_blocks > num_blocks:
        raise ValueError("num_blocks must cover the requested context length")

    query = torch.randn((num_heads, head_dim), device=device, dtype=dtype)
    kv_cache = KVCachePool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    block_table = BlockTable(block_size=block_size, block_ids=list(range(required_blocks)))

    for token_idx in range(num_tokens):
        block_id = block_table.physical_block_for_token(token_idx)
        token_offset = token_idx % block_size
        key = torch.randn((num_heads, head_dim), device=device, dtype=dtype)
        value = torch.randn((num_heads, head_dim), device=device, dtype=dtype)
        kv_cache.write_token(block_id, 0, token_offset, key, value)

    return query, kv_cache, block_table, num_tokens


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


def check_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> float:
    max_abs_diff = (actual - expected).abs().max().item()
    passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    print(f"{name:>7}: allclose={passed!s:<5} max_abs_diff={max_abs_diff:.6e}")
    if not passed:
        raise AssertionError(f"{name} output does not match ref backend")
    return max_abs_diff


def compare_case(
    num_tokens: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
    num_blocks: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    rtol: float,
    atol: float,
) -> None:
    device = torch.device("cuda")
    query, kv_cache, block_table, num_tokens = build_case(
        num_tokens=num_tokens,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        dtype=dtype,
        device=device,
    )

    print(
        f"\n=== num_tokens={num_tokens} block_size={block_size} "
        f"num_heads={num_heads} head_dim={head_dim} dtype={dtype} ==="
    )

    q = query.unsqueeze(0).contiguous()
    k_cache = kv_cache.keys[:, 0].contiguous()
    v_cache = kv_cache.values[:, 0].contiguous()
    block_tables = torch.tensor(
        [block_table.block_ids],
        device=device,
        dtype=torch.int32,
    )
    context_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)

    backend_fns = {
        "ref": lambda: paged_attention_ref(
            q, k_cache, v_cache, block_tables, context_lens, block_size
        ),
        "triton": lambda: paged_attention_triton(
            q, k_cache, v_cache, block_tables, context_lens, block_size
        ),
        "cuda": lambda: _paged_attention.paged_attention_cuda(
            q, k_cache, v_cache, block_tables, context_lens, block_size
        ),
    }
    ref_out = backend_fns["ref"]()
    results: list[tuple[str, float]] = []

    for impl, fn in backend_fns.items():
        try:
            out = fn()
            check_close(impl, out, ref_out, rtol=rtol, atol=atol)
            latency_ms = time_cuda(fn, warmup=warmup, iters=iters)
            results.append((impl, latency_ms))
            print(f"{impl:>7}: latency={latency_ms:.3f} ms")
        except ImportError:
            print(f"{impl:>7}: unavailable")
        except Exception as exc:
            print(f"{impl:>7}: failed ({type(exc).__name__}: {exc})")

    ref_latency = next((latency for name, latency in results if name == "ref"), None)
    if ref_latency is not None:
        for name, latency in results:
            if name == "ref":
                continue
            print(f"{name:>7}: speedup_vs_ref={ref_latency / latency:.2f}x")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark because Triton/CUDA backends run on GPU.")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    
    # 关闭 PyTorch 的自动 Attention 融合 (SDPA)，强制走原始的 matmul + softmax + matmul 路径
    # 这样才能公平对比我们手写的 PagedAttention 和最原始的 PyTorch 实现
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    dtype = torch.float32

    for num_tokens in args.context_lens:
        compare_case(
            num_tokens=num_tokens,
            block_size=args.block_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_blocks=args.num_blocks,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            rtol=args.rtol,
            atol=args.atol,
        )


if __name__ == "__main__":
    main()

"""Micro-benchmark for one VSA-H3 attention call against the dense backends.

    python benchmarks/vsa_h3_microbench.py

The defaults are the geometry a FastH3 768x1344x124 run with Ulysses degree 8
reaches, i.e. 7 heads per rank. Set XFUSER_VSA_H3_BACKEND=flex to measure the
FlexAttention path instead of the Triton kernel.
"""

import argparse
import time

import torch

from xfuser.core import vsa_h3_attention as vsa


def _time(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def dense_call(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)


def aiter_dense_call(query, key, value):
    """The dense backend FastH3 actually falls back to, for a fair comparison."""
    from xfuser.core.distributed.attention_backend import _aiter_attn_call

    return _aiter_attn_call(query, key, value, 0.0, False, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--video", type=int, nargs=3, default=(37, 24, 42))
    parser.add_argument("--prefix", type=int, nargs=2, default=(11, 414))
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    video_shape = tuple(args.video)
    prefix = tuple(args.prefix)

    metadata = vsa.build_h3_vsa_metadata(prefix, video_shape, device)
    shape = (1, args.heads, metadata.total_seq_length, args.head_dim)
    query, key, value, gate = (
        torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(4)
    )

    partial_tiles = (
        metadata.num_tiles
        - metadata.num_full_video_tiles
        - metadata.num_prefix_tiles
        + metadata.num_prefix_partial_tiles
    )
    print(
        f"prefix={prefix} video={video_shape} "
        f"tokens={metadata.total_seq_length} tiles={metadata.num_tiles} "
        f"(partial={partial_tiles}) "
        f"heads={args.heads} head_dim={args.head_dim}"
    )

    dense_ms = _time(lambda: dense_call(query, key, value), args.iters)
    print(f"dense SDPA        {dense_ms:8.2f} ms")
    try:
        aiter_ms = _time(lambda: aiter_dense_call(query, key, value), args.iters)
        print(f"dense AITER FA    {aiter_ms:8.2f} ms")
    except Exception as error:  # noqa: BLE001 - informational benchmark only
        aiter_ms = None
        print(f"dense AITER FA    unavailable ({error})")

    def sparse_call():
        return vsa.h3_vsa_attention(query, key, value, gate, metadata)

    sparse_ms = _time(sparse_call, args.iters)
    torch.cuda.reset_peak_memory_stats()
    sparse_call()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"VSA-H3            {sparse_ms:8.2f} ms  peak {peak:5.2f} GiB")

    print(f"speedup vs SDPA     {dense_ms / sparse_ms:5.2f}x")
    if aiter_ms is not None:
        print(f"speedup vs AITER FA {aiter_ms / sparse_ms:5.2f}x")


if __name__ == "__main__":
    main()

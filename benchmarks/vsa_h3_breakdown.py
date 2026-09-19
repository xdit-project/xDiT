"""Per-stage timing of one VSA-H3 attention call.

    python benchmarks/vsa_h3_breakdown.py

The defaults are the geometry a FastH3 768x1344x124 run with Ulysses degree 8
reaches, i.e. 7 heads per rank. Requires the Triton backend.
"""

import argparse
import time

import torch

from xfuser.core import vsa_h3_attention as vsa


def _time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--video", type=int, nargs=3, default=(37, 24, 42))
    parser.add_argument("--prefix", type=int, nargs=2, default=(11, 414))
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    metadata = vsa.build_h3_vsa_metadata(
        tuple(args.prefix), tuple(args.video), device
    )
    packed_shape = (1, args.heads, metadata.total_seq_length, args.head_dim)
    packed = [
        torch.randn(packed_shape, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    ]
    query, key, value, gate = packed

    from xfuser.core.vsa_h3_triton import (
        triton_h3_vsa_attention,
        triton_pool_h3_vsa_tiles,
    )

    pooled = [triton_pool_h3_vsa_tiles(tensor, metadata) for tensor in packed[:3]]
    kv_indices = vsa.build_h3_vsa_kv_list(pooled[0], pooled[1], metadata)
    compressed = torch.nn.functional.scaled_dot_product_attention(
        *(tensor.to(torch.bfloat16) for tensor in pooled)
    )

    width = metadata.num_prefix_tiles + vsa.compute_h3_vsa_topk(
        vsa.FASTH3_VSA_SPARSITY, metadata.num_video_tiles
    )
    print(
        f"tokens={metadata.total_seq_length} tiles={metadata.num_tiles} "
        f"width={width} (padded {kv_indices.shape[-1]}) "
        f"density={width / metadata.num_tiles:.1%} "
        f"heads={args.heads} head_dim={args.head_dim}"
    )

    stages = {
        "pool x3": lambda: [
            triton_pool_h3_vsa_tiles(tensor, metadata) for tensor in packed[:3]
        ],
        "kv-list select": lambda: vsa.build_h3_vsa_kv_list(
            pooled[0], pooled[1], metadata
        ),
        "compressed SDPA": lambda: torch.nn.functional.scaled_dot_product_attention(
            *(tensor.to(torch.bfloat16) for tensor in pooled)
        ),
    }
    stages["triton kernel"] = lambda: triton_h3_vsa_attention(
        query, key, value, kv_indices, compressed, gate, metadata
    )

    total = 0.0
    for name, fn in stages.items():
        milliseconds = _time(fn, args.iters)
        total += milliseconds
        print(f"  {name:<18} {milliseconds:8.3f} ms")
    print(f"  {'stage sum':<18} {total:8.3f} ms")

    whole = _time(
        lambda: vsa.h3_vsa_attention(query, key, value, gate, metadata),
        args.iters,
    )
    print(f"  {'end to end':<18} {whole:8.3f} ms")


if __name__ == "__main__":
    main()

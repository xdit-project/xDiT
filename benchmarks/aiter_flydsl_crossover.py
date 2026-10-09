"""Measure the end-to-end BF16/FP8 FlyDSL attention crossover on gfx1201.

FP8 timings include Q/K/V quantization, matching the xDiT runtime path.
By default the sweep covers the calibrated RDNA4 head-count/head-dimension
classes, including Ulysses-local shapes.
"""

import argparse
import statistics

import torch
from aiter.ops.flydsl import flydsl_flash_attn_func, flydsl_fp8_quant


RDNA4_SHAPE_CLASSES = (
    (6, 128),
    (8, 128),
    (12, 128),
    (15, 128),
    (16, 128),
    (24, 128),
    (30, 128),
    (32, 128),
    (48, 128),
    (19, 64),
    (38, 64),
)


def _bf16_attention(query, key, value):
    return flydsl_flash_attn_func(query, key, value, causal=False, waves_per_eu=2, daz=True)


def _fp8_attention(query, key, value):
    query_fp8, key_fp8, value_fp8, query_scale, key_scale, value_scale = flydsl_fp8_quant(
        query, key, value, rotation=True
    )
    return flydsl_flash_attn_func(
        query_fp8,
        key_fp8,
        value_fp8,
        causal=False,
        q_descale=query_scale,
        k_descale=key_scale,
        v_descale=value_scale,
        waves_per_eu=2,
        daz=True,
    )


def _measure(function, query, key, value, iterations):
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function(query, key, value)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _parse_pairs(value):
    return [tuple(map(int, pair.split(":"))) for pair in value.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument(
        "--pairs",
        type=_parse_pairs,
        default=_parse_pairs(
            "512:512,768:768,1024:1024,1280:1280,1536:1536,1792:1792,"
            "2048:2048,2304:2304,2432:2432,2560:2560,2688:2688,2816:2816,"
            "2944:2944,3072:3072,3328:3328,3584:3584,3840:3840,4096:4096,"
            "4352:4352,4608:4608,512:1280,768:2304,1024:3328,1280:4352,"
            "1536:5376,1664:5888,1792:6400,1920:6912,2048:7424,2304:8448,"
            "2560:9472,3072:11520,3584:13568,4096:15616,4352:16640"
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if (args.heads is None) != (args.head_dim is None):
        parser.error("--heads and --head-dim must be supplied together")
    shape_classes = ((args.heads, args.head_dim),) if args.heads is not None else RDNA4_SHAPE_CLASSES

    torch.manual_seed(0)
    print("Sq,Sk,H,D,bf16_ms,fp8_ms,fp8_speedup")
    with torch.inference_mode():
        for heads, head_dim in shape_classes:
            for query_length, key_length in args.pairs:
                query_shape = (1, query_length, heads, head_dim)
                key_shape = (1, key_length, heads, head_dim)
                query = torch.randn(query_shape, device="cuda", dtype=torch.bfloat16)
                key = torch.randn(key_shape, device="cuda", dtype=torch.bfloat16)
                value = torch.randn(key_shape, device="cuda", dtype=torch.bfloat16)

                for _ in range(args.warmup):
                    _bf16_attention(query, key, value)
                    _fp8_attention(query, key, value)
                torch.cuda.synchronize()

                bf16_ms = _measure(_bf16_attention, query, key, value, args.iterations)
                fp8_ms = _measure(_fp8_attention, query, key, value, args.iterations)
                print(
                    f"{query_length},{key_length},{heads},{head_dim},{bf16_ms:.4f},{fp8_ms:.4f},{bf16_ms / fp8_ms:.4f}"
                )


if __name__ == "__main__":
    main()

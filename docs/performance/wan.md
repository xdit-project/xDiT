# Wan Ulysses SDMA Overlap

Wan can overlap one-group SDMA input transfers with the V projection:

```text
--ulysses_a2a sdma --ulysses_sdma_head_groups 1 --ulysses_sdma_overlap_qkv
```

This is opt-in. The default RCCL path and SDMA configurations without these
options are unchanged. Unlike head-group projection in FLUX.3 and MiniMax-H3,
Wan keeps its normalization across the full head width and does not duplicate
projection weights.

## Execution

Q and K are projected and normalized over all heads. The eligible FlyDSL
norm/RoPE kernel writes directly into a peer-major Q/K buffer. Q/K are submitted
to SDMA before a token-dependent V projection. V is packed and sent separately;
only its completion flag makes the complete Q/K/V input ready for attention.
The attention and output exchange use the existing SDMA implementation.

The fast path requires inference, batch one, BF16 activations and norm weights,
unfused resident plain linear projections, and Ulysses-only dense self-attention.
Unsupported settings retain the existing path, including quantized or sharded
projections, projection hooks, FP8 communication, ring attention, and head
balancing. FP8 attention with BF16 communication is supported. Other head-group
counts do not use this optimization.

The producer retains the original norm/RoPE eligibility checks and reference
fallback. During external CUDA graph capture, split sends use the same packed
layout over RCCL; this is not SDMA graph replay.

## Reproduction

The full-model harness loads cached Wan2.2-T2V-A14B Diffusers weights, uses seed
42 and a fixed prompt, and generates 81 frames at 1280x720 with 40 steps. Run
each variant in a separate process; `sdmaq` enables the options above, `sdma`
uses the parent one-group SDMA path, and `rccl` uses the default transport.

```bash
torchrun --standalone --nproc_per_node 8 tests/e2e/run_wan_sdma.py \
  --backend AITER_FP8 --variant sdmaq --steps 40 --repeats 3 \
  --output results/wan-sdmaq
```

On a shared node with brr, put the command after
`brr run --exclusive --gpus 8 --max-wait 10m --`. Use `AITER_BF16` for BF16
attention. Compilation and the first complete generation are warmup, not
measured samples. The JSON report includes every rank's timing, split-send
activation counts, decoded shape/range/finite checks, and repeat differences.
The final decoded video and three representative PNGs are saved locally.

Reported latency is the slowest rank's GPU-event time around the pipeline
call, summarized by the median of three measured generations. Model loading,
compilation warmup, output gathering after the timed call, and artifact writing
are excluded; this is not process startup-to-exit wall time.

## MI350X Measurements

Eight MI350X GPUs, Ulysses 8, pretrained Wan2.2-T2V-A14B, AITER_FP8 attention
with BF16 weights and communication, using the protocol above:

| Process order | RCCL (s) | Parent SDMA (s) | Q/K overlap (s) | Reduction vs parent |
| --- | ---: | ---: | ---: | ---: |
| Q/K overlap, parent SDMA, RCCL | 54.1393 | 53.8827 | 52.8472 | 1.92% |
| RCCL, parent SDMA, Q/K overlap | 54.0796 | 53.7790 | 52.4190 | 2.53% |

Each value is the median of three measured generations in a fresh process.
The second comparison reverses process order; it reduces latency by 3.07%
against RCCL. Each optimized generation executed 3200 split-QK calls. These
are measurements on one node and one prompt/seed, not a general speedup or
quality guarantee.

Runtime: `amdsiloai/pytorch-xdit:v26.10.1`, torch `2.13.0+git3369fa2`, HIP
`7.16.26323`, RCCL `2.30.4`, diffusers `0.41.0.dev0`, AITER
`e2d019f15599b73360ae027b73b8d72961903bd2` with the image's existing functional
varlen patch.

## Focused Validation

The complete SDMA integration module passed all 17 tests on eight MI350X GPUs.
The new cases cover alternating original/split exchanges, compiled tensor
lifetimes, exact producer layouts for 2/4/8 peers, actual two-rank transfers,
BF16 and FP8 processor equality, external graph capture/replay, and fallback
for unsupported settings. The graph checks exercise packed RCCL fallback.

## Numerical Scope

The transport, producer, and eager/compiled attention tests check exact equality
against the matching packed layout. This does not imply pixel-exact equality
with the original RCCL pipeline: the parent SDMA path changes attention input
layout, and full diffusion amplifies small differences. Full-model experiments
also showed repeat variation on both parent and optimized SDMA paths. The
harness records this rather than imposing an uncalibrated pixel threshold;
`--max-repeat-rmse` can enforce a chosen repeat tolerance. One seed and prompt
are a smoke test, not a general video-quality evaluation.
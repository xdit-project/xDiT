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

Eight MI350X GPUs, Ulysses 8, pretrained Wan2.2-T2V-A14B, BF16 weights and
communication, using the protocol above:

| Attention and process order | RCCL (s) | Parent SDMA (s) | Q/K overlap (s) | Reduction vs parent |
| --- | ---: | ---: | ---: | ---: |
| FP8: Q/K overlap, parent SDMA, RCCL | 54.1393 | 53.8827 | 52.8472 | 1.92% |
| FP8: RCCL, parent SDMA, Q/K overlap | 54.0796 | 53.7790 | 52.4190 | 2.53% |
| BF16: RCCL, parent SDMA, Q/K overlap | 81.6193 | 80.7359 | 80.0860 | 0.80% |

Each value is the median of three measured generations in a fresh process.
The second comparison reverses process order; it reduces latency by 3.07%
against RCCL. Each optimized generation executed 3200 split-QK calls. These
are measurements on one node and one prompt/seed, not a general speedup or
quality guarantee. The BF16 samples overlap substantially (parent 80.48-82.19 s,
optimized 79.85-81.49 s); its three-sample gain is not a tight significance claim.

Runtime: `amdsiloai/pytorch-xdit:v26.10.1`, torch `2.13.0+git3369fa2`, HIP
`7.16.26323`, RCCL `2.30.4`, diffusers `0.41.0.dev0`, AITER
`e2d019f15599b73360ae027b73b8d72961903bd2` with the image's existing functional
varlen patch.

## Two-Block Comparisons

These are full-shape random-weight two-block Wan forwards, not pretrained
generation: 75,600 tokens, Ulysses 8, compilation enabled, eight warmups and
80 deterministically shuffled measurements per variant. The reported metric
is the maximum across ranks of each rank's median whole-forward GPU-event time.
Both SDMA variants match the same-layout packed control exactly.

| GPUs | Attention | RCCL (ms) | Parent SDMA (ms) | Q/K overlap (ms) | Reduction vs parent |
| --- | --- | ---: | ---: | ---: | ---: |
| 8x MI350X | BF16 | 49.28 | 49.27 | 48.58 | 1.40% |
| 8x MI350X | FP8 | 32.99 | 33.12 | 32.26 | 2.59% |
| 8x MI355X | BF16 | 43.47-43.49 | 43.65-43.66 | 42.68-42.71 | 2.16-2.21% |
| 8x MI355X | FP8 | 28.93-28.98 | 29.42-29.49 | 28.39-28.46 | 3.44-3.50% |

MI350X rows are one final production confirmation per backend; MI355X ranges
cover three fresh processes per backend. On MI355X, improvement over RCCL is
1.75-1.81% BF16 and 1.81-1.86% FP8. Duplicate RCCL controls differ by less than
0.04% in every MI355X run. The two nodes have matching Torch/HIP/RCCL/Diffusers
and AITER revisions, with byte-identical existing AITER functional-varlen patches,
but different container image IDs. Wan weights were absent on MI355X, so no
pretrained end-to-end gain is claimed there.

## Focused Validation

The complete SDMA integration module passed all 17 tests on both eight MI350X
and eight MI355X GPUs.
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
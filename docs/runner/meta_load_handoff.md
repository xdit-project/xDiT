# Memory-efficient load: node handoff

Written when the MI300X node this work ran on was reclaimed. Everything below is either committed on
`feat/rdna4-fp8-meta-load` or reproducible from it; the node's `/validation` scratch is gone.

## The node is now MI355X (gfx950)

Work resumed on gfx950, not the gfx942 the results below were taken on, which changes what the
matrix should expect. `_AITER_FP4_UNSUPPORTED_ARCHS` in `loading/format_backends.py` lists only
`gfx942`, so the five cases that assert FP4 is rejected are gfx942 behaviour, not ROCm behaviour.
`probe_format_backend_capabilities()` on this node confirms it: `aiter_mxfp4` is `True` with no
reason, where gfx942 refused it with "AITER builds no FP4 kernels for gfx942". Keep those five cases
pinned to `gfx942` and add gfx950 cases rather than editing them in place.

Do not expect FP4 to pass everywhere on gfx950, though. `aiter_mxfp4_fsdp` is still `False`, for an
unrelated reason: FSDP2 here cannot shard non-floating-point parameters without pytorch/pytorch#177948,
which landed in torch 2.12, and this image carries 2.9.1. So on gfx950 the new FP4 cases should expect
success only on the eager and replicated placements, and FP4 with `fsdp_blockwise` should still expect
a rejection — attributed to the torch version, not the architecture. That distinction is worth keeping
in the case notes, because it moves as soon as the image gets a newer torch.

Everything above is a capability probe. No FP4 inference has run on gfx950 yet.

Any result recorded here should name the arch it was seen on. The existing evidence is gfx942, and
`observed_on_gfx942` notes in `tests/gpu_validation/matrix.json` mark cases whose JSONL did not
survive the reclaim.

## Do these two things first

1. **Set `FLYDSL_GPU_ARCH` in the container environment** (`gfx950` on MI355X, `gfx942` on MI300X,
   `gfx1201` on RDNA4).
   Without it `flydsl.runtime.device` shells out to `rocm_agent_enumerator` with a 300 s timeout, and
   under load the calls pile up faster than they drain. One node reached 1336 concurrent enumerators
   holding 1693 GB of RSS on a 1 TB host. It stalls runs and it dominates every host-memory sample.
   Verify with `python3 -c "import flydsl.runtime.device as d; print(d.get_rocm_arch())"` and confirm
   no `rocm_agent_enumerator` process appears.
2. **Set `PYTHONPATH` to the checkout.** The image ships its own editable xfuser install, so pytest
   validates that copy instead of the branch unless `PYTHONPATH` points at the working tree.

## What the cache can reach

`tools/cache_inventory.py` maps every registered runner to the HF hub cache. On this node 25 of the
33 have usable weights; 8 do not, and the root filesystem is 98% full with 90 GB free, so fetching
them is not safe without reclaiming space first.

Four of the eight are withheld from memory-efficient load anyway, so nothing is lost by leaving them:
CausalWan, LTX-2, and both non-sparse HunyuanVideo 1.5 variants. Only the Diffusers-format 1.5 repos
are missing; `tencent/HunyuanVideo-1.5` is cached at 254 GB and serves the sparse variant.

The other four do declare meta-load support and cannot be exercised here: **Cosmos3-Nano**,
**Cosmos3-Super**, **Wan2.1-VACE**, and **FLUX.2-klein-4B**. klein-4B is the one that matters, since
its three matrix cases were observed on gfx942 and its cache entry is a 36 KB metadata-only stub, so
re-confirming them on gfx950 needs a ~24 GB download.

Read the tool's verdict as a floor on what is missing rather than a promise a given invocation will
load: runners that pick their repo from CLI flags have several candidates, and one counts as cached
when any candidate is present.

## Test suite

The suite has now run against the merge on gfx950: 557 passed, 18 failed, 15 skipped. Eleven
failures were merge and branch defects and are fixed; none of the remaining 18 is a code defect.

Eight fail identically before the merge and are unrelated to this work: `test_envs` cpu and mps
device selection, two `test_gilbert` sliced-neighbor mapping cases, `feedforward_test`, both
`usp_test` hybrid-equivalence cases, and `attention_processor_test[HunyuanDiT]`.

Ten need a newer image rather than a code change: the nine `test_minimax_h3` cases need a diffusers
build carrying MiniMax-H3, and `test_vsa_density_collection_is_opt_in` needs an aiter with
`jenga_sparse_attention`.

Four files cannot be collected at all, three of them predating this branch: `test_sharding.py`
imports `shard_transformer_blocks`, deleted upstream in #638; `test_diffusers_adapters.py` imports
`xFuserFluxAttnProcessor2_0`, removed in `d2bf3c3`; and `test_ring_flash_attn.py` and
`test_xfuser_attn.py` need `flash_attn`, which has no ROCm build here.

## State of the branch

`feat/rdna4-fp8-meta-load` is merged with upstream through MiniMax-H3 (#747). The merge was chosen
over a rebase: the branch is 59 commits past the merge base with 17 touching `base_model.py`, so a
rebase replays the same conflict repeatedly and rewrites history already on `origin`.
`backup/pre-upstream-rebase-20260807` is the pre-merge state if that needs revisiting.

Three resolutions in that merge are judgement calls worth re-reading if something looks wrong:
upstream's `_build_fsdp_quantize_fn` was dropped as dead (its call site is the FSDP path this branch
replaced with the per-block adapter path); upstream's `fp8_gemm_include_suffixes` was threaded through
`_conversion_filter` rather than left at the two direct call sites this branch had already replaced;
and in `runner_utils.py` this branch's `filter_fn` and upstream's `include_suffixes` were made to
compose, with an explicit filter taking precedence.

The three new upstream models (MiniMax-H3, LingBot-Video, Ideogram 4) arrived without load
declarations, and because they subclass `xFuserModel` they inherited its declaration rather than
reporting as withheld. That is what `test_every_registered_runner_has_its_own_load_declaration`
forbids, and the inherited declaration was derived from the base's empty `ModelCapabilities`, so it
also contradicted the fp8/fp4 support these runners enable — the failure
`test_runner_load_capabilities_match_model_quantization_capabilities` reported. All five registered
classes now carry an explicit `@LoadCapability.declare(unsupported_reason=...)`, which withholds meta
load while re-deriving the quantization contracts from each runner's own `ModelCapabilities`. The gap
report covers 33 runner models, 13 of them withheld.

## Measurements that do not hold up

Two earlier conclusions were wrong and should not be carried forward.

Host-memory peaks recorded before the `flydsl` fix are not measurements of the load path. The
enumerator storm put ~313 GB of unrelated anon memory in the trace before a model began loading, so
figures like "582 GB peak for a 24 GB model" are mostly storm. Re-measure anything that matters.

Reading safetensors through a pinned staging buffer looked 9x faster than the current direct-to-GPU
read (12.3 GB/s against 1.35 GB/s), but that was warm page cache only. Repeated with a cold cache and
a busy disk, all strategies collapsed to 0.1-0.7 GB/s and the difference vanished: the fill is
disk-bound, not H2D-bound. Do not change `_read_device` on the strength of the warm number.

## Coverage

`python tools/load_support_matrix.py --format gaps --backend <backend>` reports where testing does not
reach. The larger gap is not capability but coverage: most models declaring a memory-efficient load
have no matrix case at all.

`tests/gpu_validation/supplementary/` holds the probe matrices this node used, with per-case
`observed_on_gfx942` notes. FLUX.2-klein-4B passed eager, `fsdp_blockwise` at 4 ranks and `replicated`
at 4 ranks, which is real coverage for a model the checked-in matrix has none for; those three cases
can be landed. Z-Image-Turbo, Wan2.2-TI2V and FLUX.1-Kontext are staged but unproven.

Run supplementary cases **serially**: the single-process eager path binds port 29500 regardless of
`PET_MASTER_PORT`, which only reaches torchrun. `seed_cache.py` re-seeds the four models these
matrices need (about 150 GB, roughly four minutes).

## Open threads

The 8-rank divergence between eager and memory-efficient loads is unresolved and may not be real: the
runs attributed to it failed on a port collision and on a kill during the enumerator storm. Re-run
with `FLYDSL_GPU_ARCH` set before assuming a numerical bug. The per-tensor fingerprint hook used to
chase it gathered full tensors on rank 0 and was too expensive to complete; hash shard-local tensors
per rank instead.

Enabling collective mapped-checkpoint loads for CausalWan and Wan2.2-Distilled is the most tractable
way to widen support. Both are withheld for collective-safe key discovery, which is what
`resolve_mapped_checkpoint` and `CheckpointManifest` already provide, and only rank 0 reads while
`_require_checkpoint_keys` broadcasts the missing-key list from rank 0, so the mapping only has to be
correct there. Note the withheld declarations are load-bearing, not stale: `LoaderAdapter`
distinguishes `supports_local_blockwise` from `supports_standard_collectives`, and
`validate_materialization_contract` rejects the latter for anything but `STANDARD_TRANSFORMER`.

Still open from the earlier plan: routing text encoders through the blockwise fill on the replicated
path, which still uses `broadcast_load` and so materializes a whole encoder before scattering.

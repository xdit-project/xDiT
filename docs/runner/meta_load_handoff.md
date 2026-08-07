# Memory-efficient load: node handoff

Written when the MI300X node this work ran on was reclaimed. Everything below is either committed on
`feat/rdna4-fp8-meta-load` or reproducible from it; the node's `/validation` scratch is gone.

## Do these two things first

1. **Set `FLYDSL_GPU_ARCH` in the container environment** (`gfx942` on MI300X, `gfx1201` on RDNA4).
   Without it `flydsl.runtime.device` shells out to `rocm_agent_enumerator` with a 300 s timeout, and
   under load the calls pile up faster than they drain. One node reached 1336 concurrent enumerators
   holding 1693 GB of RSS on a 1 TB host. It stalls runs and it dominates every host-memory sample.
   Verify with `python3 -c "import flydsl.runtime.device as d; print(d.get_rocm_arch())"` and confirm
   no `rocm_agent_enumerator` process appears.
2. **Run the test suite.** The upstream merge (`2128b08`) was resolved and statically checked, but
   pytest and torch live only in the container, so no test has run against it.

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

The three new upstream models (MiniMax-H3, LingBot-Video, Ideogram 4) declare no load capability, so
they report as withheld and the gap report now covers 31 runner models rather than 28.

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

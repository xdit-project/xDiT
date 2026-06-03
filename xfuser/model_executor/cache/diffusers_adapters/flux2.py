"""
FBCache adapter for FLUX.2 (Flux2Transformer2DModel / xFuserFlux2Transformer2DWrapper).

Adapted from flux.py which targets FLUX.1's FluxTransformer2DModel.

Key differences from FLUX.1:
  - Dual-stream block signature: block(hidden_states, encoder_hidden_states,
        temb_mod_img, temb_mod_txt, image_rotary_emb, joint_attention_kwargs)
    Returns: (encoder_hidden_states, hidden_states)   [encoder FIRST]

  - Single-stream block signature: block(hidden_states, encoder_hidden_states=None,
        temb_mod, image_rotary_emb, joint_attention_kwargs)
    Input hidden_states is already [txt || img] concatenated.
    Returns: hidden_states (no split)

  - Modulation tensors are pre-computed ONCE before all block loops:
        double_stream_mod_img  -> passed to every dual-stream block
        double_stream_mod_txt  -> passed to every dual-stream block
        single_stream_mod      -> passed to every single-stream block
    These are three distinct tensors from three distinct Flux2Modulation heads.

  - The FBCache wraps transformer_blocks (dual) AND single_transformer_blocks
    inside a single Flux2FBCachedTransformerBlocks module placed as the sole
    element of transformer_blocks.  single_transformer_blocks is emptied.

  - The monkey-patched forward() intercepts the FLUX.2 model's block loops,
    builds the right per-type kwargs, and calls Flux2FBCachedTransformerBlocks.

Implementation note:
  We store `single_stream_mod` (and joint_attention_kwargs for single blocks)
  on the Flux2FBCachedTransformerBlocks instance as instance variables that are
  set by the patched forward() immediately before the block loop.  This avoids
  changing the `process_blocks` / `forward` signatures of the base class.
"""

import torch
from torch import nn

from xfuser.model_executor.cache import utils
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY


class Flux2FBCachedTransformerBlocks(utils.FBCachedTransformerBlocks):
    """
    FBCachedTransformerBlocks specialised for FLUX.2.

    Call convention differences from FLUX.1:
      - Dual blocks return (encoder_hidden_states, hidden_states), i.e.
        `return_hidden_states_first=False`.
      - Single blocks need a different modulation kwarg (`temb_mod` instead of
        `temb_mod_img`/`temb_mod_txt`) and take a single concatenated
        hidden_states tensor.

    Permanent hook design (set by apply_cache_on_transformer):
        self._single_stream_mod           : torch.Tensor  (injected by pre-hook)
        self._single_joint_attn_kwargs    : dict | None   (injected by pre-hook)

    process_blocks() reads these to call single blocks correctly.

    CUDA Graph compatibility:
      torch.compile(mode="reduce-overhead") uses CUDA Graphs which allocate a
      FIXED pool of GPU buffers for each compiled graph segment. Tensors stored
      from inside the graph (e.g. first_hidden_states_residual) have buffers that
      are overwritten on the next graph replay. Cross-step cache storage therefore
      must happen OUTSIDE the CUDA graph.

      This is achieved by:
        1. Pre-allocating persistent GPU buffers (_persistent_*) whose data
           pointers are stable across steps (same buffer, updated in-place).
        2. Decorating the store methods with @torch._dynamo.disable so they run
           in EAGER mode (causing a graph break), allocating and copying into the
           persistent buffers OUTSIDE the graph's memory pool.
        3. Reading from the persistent buffers inside the next step's compiled
           graph — the data pointer is stable so the guard holds and the graph
           correctly sees the updated data.

    Device compatibility:
      The base class initialises rel_l1_thresh with get_device(0) = cuda:0 for
      ALL ranks. In Ulysses SP with 8 ranks using cuda:0..cuda:7, the l1_distance
      result on rank N (cuda:N) cannot be compared to a threshold on cuda:0.
      are_two_tensor_similar is overridden to move threshold to t1's device.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Injected by permanent pre-hook in apply_cache_on_transformer.
        self._single_stream_mod = None
        self._single_joint_attn_kwargs = None
        # Persistent GPU buffers for cross-step cache storage.
        # Allocated lazily; same object/data_ptr reused across steps via .copy_().
        self._persistent_modulated_inputs = None
        self._persistent_hidden_residual = None
        self._persistent_encoder_residual = None

    # ------------------------------------------------------------------
    # Override: are_two_tensor_similar (device fix for multi-GPU Ulysses)
    # ------------------------------------------------------------------
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold) -> torch.Tensor:
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(t1.device)
        return self.l1_distance(t1, t2) < threshold

    # ------------------------------------------------------------------
    # Eager-only store helpers (break out of CUDA graph pool)
    # ------------------------------------------------------------------
    @torch._dynamo.disable
    def _store_modulated_inputs(self, tensor: torch.Tensor) -> None:
        """
        Copy tensor into a persistent buffer OUTSIDE any CUDA graph.

        @torch._dynamo.disable causes a graph break when called from within a
        compiled segment. The copy runs eagerly so the destination buffer is
        allocated from the regular CUDA allocator (not the graph's static pool).
        The persistent buffer has a stable data_ptr so the graph guard holds on
        subsequent calls — the graph reads from the same pointer, getting updated
        data because we .copy_() into it before each replay.
        """
        if (self._persistent_modulated_inputs is None
                or self._persistent_modulated_inputs.shape != tensor.shape
                or self._persistent_modulated_inputs.device != tensor.device):
            self._persistent_modulated_inputs = torch.empty_like(tensor)
        self._persistent_modulated_inputs.copy_(tensor)
        self.cache_context.modulated_inputs = self._persistent_modulated_inputs

    @torch._dynamo.disable
    def _store_block_residuals(
        self,
        hidden_residual: torch.Tensor,
        encoder_residual: torch.Tensor,
    ) -> None:
        """Copy block residuals into persistent buffers outside any CUDA graph."""
        if (self._persistent_hidden_residual is None
                or self._persistent_hidden_residual.shape != hidden_residual.shape
                or self._persistent_hidden_residual.device != hidden_residual.device):
            self._persistent_hidden_residual = torch.empty_like(hidden_residual)
        self._persistent_hidden_residual.copy_(hidden_residual)
        self.cache_context.hidden_states_residual = self._persistent_hidden_residual

        if (self._persistent_encoder_residual is None
                or self._persistent_encoder_residual.shape != encoder_residual.shape
                or self._persistent_encoder_residual.device != encoder_residual.device):
            self._persistent_encoder_residual = torch.empty_like(encoder_residual)
        self._persistent_encoder_residual.copy_(encoder_residual)
        self.cache_context.encoder_hidden_states_residual = self._persistent_encoder_residual

    # ------------------------------------------------------------------
    # Override: get_modulated_inputs
    # ------------------------------------------------------------------
    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        """
        Run block[0] (dual-stream) and use its residual as the cache key.
        """
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]

        # FLUX.2 dual block returns (encoder_hidden_states, hidden_states)
        result = first_transformer_block(hidden_states, encoder_hidden_states, *args, **kwargs)
        encoder_out, hidden_out = result  # FLUX.2: encoder first
        hidden_states_out = hidden_out
        encoder_hidden_states_out = encoder_out

        first_hidden_states_residual = hidden_states_out - original_hidden_states
        prev_first_hidden_states_residual = self.cache_context.modulated_inputs

        if not self.use_cache:
            # Store via @torch._dynamo.disable helper so the copy runs eagerly
            # (outside any CUDA graph's static buffer pool). This ensures the
            # persistent buffer holds step-N's data when step-(N+1) reads it.
            self._store_modulated_inputs(first_hidden_states_residual)

        return (
            first_hidden_states_residual,
            prev_first_hidden_states_residual,
            hidden_states_out,
            encoder_hidden_states_out,
        )

    # ------------------------------------------------------------------
    # Override: process_blocks
    # ------------------------------------------------------------------
    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """
        Run dual-stream blocks from start_idx, then all single-stream blocks.
        """
        # --- Dual-stream blocks ---
        for block in self.transformer_blocks[start_idx:]:
            # FLUX.2 dual block returns (encoder_hidden_states, hidden_states)
            encoder_out, hidden_out = block(hidden, encoder, *args, **kwargs)
            hidden, encoder = hidden_out, encoder_out

        # --- Single-stream blocks ---
        if self.single_transformer_blocks:
            combined = torch.cat([encoder, hidden], dim=1)
            encoder_seq_len = encoder.shape[1]

            image_rotary_emb = kwargs.get("image_rotary_emb", None)
            single_mod = self._single_stream_mod
            single_jkw = self._single_joint_attn_kwargs

            for block in self.single_transformer_blocks:
                combined = block(
                    hidden_states=combined,
                    encoder_hidden_states=None,
                    temb_mod=single_mod,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=single_jkw,
                )

            encoder = combined[:, :encoder_seq_len, :]
            hidden = combined[:, encoder_seq_len:, :]

        # Compute residuals inside the compiled segment, then store them via
        # @torch._dynamo.disable helper to escape the CUDA graph buffer pool.
        hidden_residual = hidden - self.cache_context.original_hidden_states
        encoder_residual = encoder - self.cache_context.original_encoder_hidden_states
        self._store_block_residuals(hidden_residual, encoder_residual)
        return hidden, encoder


def apply_cache_on_transformer(
    transformer,
    *,
    rel_l1_thresh: float = 0.12,
    return_hidden_states_first: bool = False,  # FLUX.2 dual blocks return (encoder, hidden)
    num_steps: int = 50,
    use_cache: str = "Fb",
):
    """
    Apply FBCache step-caching to a FLUX.2 transformer.

    Works on both Flux2Transformer2DModel and xFuserFlux2Transformer2DWrapper.

    torch.compile-compatible design — uses permanent hooks only:
      1. Permanently replaces transformer.transformer_blocks with a ModuleList
         containing a single Flux2FBCachedTransformerBlocks module.
      2. Permanently empties transformer.single_transformer_blocks (single blocks
         are handled inside Flux2FBCachedTransformerBlocks.process_blocks).
      3. Registers PERMANENT forward hooks (registered once at apply time, not
         per-call). Permanent hooks cause torch.compile graph breaks (dynamo runs
         them eagerly), which is harmless for correctness. Per-call hook
         registration inside a compiled function would make dynamo fail with
         "graph mutation during tracing" errors — this approach avoids that.

    Hook ordering (per denoising step):
      a. transformer.single_stream_modulation forward_hook fires:
         captures the pre-computed single_stream_mod into _state.
         (In Flux2Transformer2DModel.forward, single_stream_modulation is called
         BEFORE the transformer_blocks loop.)
      b. cached_blocks forward_pre_hook (with_kwargs=True) fires:
         reads _state["single_stream_mod"] and sets it on cached_blocks, then
         reads joint_attention_kwargs from the block-loop call kwargs and stores
         it as _single_joint_attn_kwargs for single-stream block invocations.
      c. Flux2FBCachedTransformerBlocks.forward runs:
         calls get_modulated_inputs (block[0]) + process_blocks (blocks[1:] + all
         single blocks) using the injected single_stream_mod.

    Only FBCache is supported for FLUX.2; TeaCache is not implemented because
    TeaCache requires running norm1 on a single block to get modulation params,
    and FLUX.2's modulation is shared/pre-computed differently.
    """
    if use_cache != "Fb":
        raise ValueError(
            f"FLUX.2 cache adapter only supports use_cache='Fb' (FBCache). "
            f"Got '{use_cache}'. TeaCache is not supported for FLUX.2."
        )

    adapter_name = TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer), "flux")

    cached_blocks = Flux2FBCachedTransformerBlocks(
        transformer.transformer_blocks,
        transformer.single_transformer_blocks,
        transformer=transformer,
        rel_l1_thresh=rel_l1_thresh,
        return_hidden_states_first=return_hidden_states_first,
        num_steps=num_steps,
        name=adapter_name,
    )

    # ── 1. Permanently swap the block lists ─────────────────────────────
    # Flux2Transformer2DModel.forward loops over self.transformer_blocks and
    # self.single_transformer_blocks. After this swap:
    #   - The dual-stream loop runs once, calling cached_blocks.
    #   - The single-stream loop is a no-op (single blocks run inside
    #     cached_blocks.process_blocks).
    transformer.transformer_blocks = nn.ModuleList([cached_blocks])
    transformer.single_transformer_blocks = nn.ModuleList()

    # ── 2. Persistent state — each distributed rank has its own copy ─────
    _state = {"single_stream_mod": None}

    # ── 3. Permanent hook: capture single_stream_modulation output ────────
    # Fires during Flux2Transformer2DModel.forward BEFORE the block loops.
    # In multi-GPU Ulysses SP, each rank runs this independently on its device.
    def _capture_single_mod(module, input, output):
        _state["single_stream_mod"] = output

    transformer.single_stream_modulation.register_forward_hook(_capture_single_mod)

    # ── 4. Permanent pre-hook: inject into cached_blocks ─────────────────
    # with_kwargs=True is required because the dual-stream block loop calls:
    #   block(hidden_states=..., encoder_hidden_states=...,
    #         temb_mod_img=..., temb_mod_txt=...,
    #         image_rotary_emb=..., joint_attention_kwargs=...)
    # i.e. all-keyword, so args=() and everything is in kwargs.
    def _inject_single_mod(module, args, kwargs):
        mod = _state["single_stream_mod"]
        if mod is None:
            raise RuntimeError(
                "[FBCache FLUX.2] single_stream_mod was not captured before "
                "Flux2FBCachedTransformerBlocks.forward was called. "
                "Verify that single_stream_modulation hook is correctly registered."
            )
        cached_blocks._single_stream_mod = mod
        # Pass joint_attention_kwargs through unchanged for single-stream blocks.
        # (For normal inference kv_cache_mode is None and no modification is needed.)
        cached_blocks._single_joint_attn_kwargs = kwargs.get("joint_attention_kwargs", None)

    cached_blocks.register_forward_pre_hook(_inject_single_mod, with_kwargs=True)

    # ── 5. Per-step CUDA Graph boundary marker ───────────────────────────
    # torch.compile(mode="reduce-overhead") uses CUDA Graphs with a static
    # buffer pool per graph segment. Without a step boundary signal, the CUDA
    # graph system considers all outputs from run N as "live" when run N+1
    # starts. If dynamo needs to recompile at step N+1 (e.g. because a guard
    # on cache_context.modulated_inputs changed from None to a tensor), it
    # will try to read run-N output tensors that have been overwritten by the
    # N+1 graph replay, triggering:
    #   "accessing tensor output of CUDAGraphs that has been overwritten"
    #
    # torch.compiler.cudagraph_mark_step_begin() tells the CUDA graph system
    # that run N's outputs are no longer live, so subsequent reads/recompiles
    # can proceed safely. It must be called BEFORE each transformer forward.
    # Registering it as a pre-hook ensures it fires eagerly (pre-hooks run
    # by nn.Module._call_impl BEFORE entering the compiled forward_call).
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        def _mark_cudagraph_step_begin(module, args, kwargs):
            torch.compiler.cudagraph_mark_step_begin()

        transformer.register_forward_pre_hook(
            _mark_cudagraph_step_begin, with_kwargs=True, prepend=True
        )

    return transformer

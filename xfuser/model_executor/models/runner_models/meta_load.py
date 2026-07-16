"""Memory-efficient FSDP load: build components on meta, then fill sharded weights without ever
materializing a full copy on host.

Two fill strategies, both collective (every fs-group rank must call in identical order):

* Transformer (``_TransformerDiskFiller``): self-fill. Each block's real weights are read from disk
  on fs-rank0 ONLY and broadcast GPU->GPU to the group. rank0-only read is required because the full
  block must exist on every rank before block-128 fp8 quantization (a shard boundary splitting a
  128x128 tile invalidates the tile scale, so per-rank slice reads are impossible), and if every rank
  read the full block from disk host anon would scale with N ranks (measured +3.5GB per block, enough
  to trip the cgroup OOM killer). Reading on rank0 then broadcasting keeps host disk-read anon at 1x.

* Text encoders (``MemoryEfficientSharder.broadcast_load``): rank0 loads once via from_pretrained
  (resolves tied weights), then scatters one wrapped block at a time via broadcast_from_rank0, so
  peers never receive the whole model at once. fp8-targeted TEs stream rank0 straight to fp8.

``MemoryEfficientSharder`` holds the ``xFuserModel`` so it can reuse the model's fp8 predicates
(``_component_wants_fp8`` / ``_fp8_targets_for_component``) and settings without duplicating them.
"""

import gc

import torch

from xfuser.core.distributed.parallel_state import get_fs_group, get_world_group
from xfuser.core.utils.checkpoint_io import (
    host_mem_gb,
    drop_file_page_cache,
    resolve_checkpoint_weight_map,
    component_shard_paths,
)
from xfuser.core.utils.runner_utils import log, rgetattr


def _is_bcast_src(group) -> bool:
    """True only on the single global rank-0 of `group` — the broadcast source.

    Uses rank_in_group, not local_rank. local_rank is node-local, so on a multi-node
    group every node's local-rank-0 would self-elect as source: each reads the full
    checkpoint (host anon scales with node count) and every read but the group's global
    rank-0 is silently discarded, since broadcast src=0 always means the group's global
    rank 0.
    """
    return group.rank_in_group == 0


class MemoryEfficientSharder:
    """Builds pipeline components on meta and fills their FSDP shards from disk without a full copy.

    Owns the ``xFuserModel`` (``model``) to reuse its settings and fp8 predicates.
    """

    def __init__(self, model) -> None:
        self.model = model

    def build_meta_transformer(self, wrapper_cls, subfolder: str = "transformer", init_kwargs: dict | None = None):
        """Build the (diffusers) transformer wrapper on meta from its config only (no weights).

        Real weights are streamed per block from disk during sharding (see _TransformerDiskFiller),
        so the full model never materializes. Uses the diffusers-public from_config; fp8 quantization
        happens per block on the real weights during sharding, so no fp8 swap is done here.

        init_kwargs: extra wrapper __init__ args (e.g. wan's attention_kwargs) not in the on-disk
        config; forwarded to from_config so the meta model matches the from_pretrained path.
        """
        from accelerate import init_empty_weights
        config = wrapper_cls.load_config(self.model.settings.model_name, subfolder=subfolder)
        with init_empty_weights():
            model = wrapper_cls.from_config(config, **(init_kwargs or {}))
        # from_config defaults to fp32; cast to bf16 on meta (dtype-only, legal) so the per-block
        # disk fill and the AITER quantize see bf16 (matching the on-disk checkpoint dtype).
        return model.to(torch.bfloat16)

    def build_meta_component(self, component_name: str, fp8: bool = True):
        """Instantiate a transformers pipeline component on meta from its config (no weights).

        Resolves the component's class from the pipeline's model_index.json and builds it under
        init_empty_weights so every param lands on meta. When fp8-targeted (RDNA4 AITER) and
        ``fp8`` is True, its targeted Linears are swapped to meta fp8 layers so the model is sharded
        and filled as fp8. ``fp8=False`` keeps the component bf16 on meta (replicated broadcast path:
        rank0 broadcasts bf16 and the per-rank fp8 walk quantizes locally afterwards).
        Returns None (caller falls back to a normal from_pretrained load) unless the component is a
        transformers model we can rebuild from config; real weights arrive later via broadcast_load.
        """
        try:
            from diffusers import DiffusionPipeline
            import importlib
            from accelerate import init_empty_weights

            model_name = self.model.settings.model_name
            index = DiffusionPipeline.load_config(model_name)
            entry = index.get(component_name)
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                return None
            library, class_name = entry
            if library != "transformers":
                return None
            cls = getattr(importlib.import_module(library), class_name)
            config = cls.config_class.from_pretrained(model_name, subfolder=component_name)
            with init_empty_weights():
                component = cls._from_config(config)
            # _from_config defaults to fp32; align meta params to bf16 (dtype-only .to is legal on
            # meta) so their DTensor dtype matches the broadcast source.
            component = component.to(torch.bfloat16)
            if fp8 and self.model._component_wants_fp8(component_name):
                self._swap_meta_te_to_fp8(
                    component, self.model._fp8_targets_for_component(component_name)
                )
            return component
        except Exception as e:
            log(f"Meta-init of component '{component_name}' failed ({e}); using normal load.")
            return None

    def meta_te_kwargs(self, normal):
        """Build text-encoder(s) on meta for the pipeline's from_pretrained (meta FSDP load path).

        Returns (pipe_component_kwargs, None): the kwargs carry meta modules so the pipeline skips
        loading those components, and te_quant is None (the meta module is filled by broadcast_load,
        not streamed by the pipe). Falls back to `normal` when there are no TE components or any
        component cannot be meta-built.
        """
        te_components = [
            name for name in self.model.settings.fsdp_strategy
            if name != "transformer" and not name.startswith("transformer_")
        ]
        if not te_components:
            return normal
        kwargs = {}
        for name in te_components:
            meta = self.build_meta_component(name)
            if meta is None:
                return normal
            kwargs[name] = meta
        return kwargs, None

    def meta_te_kwargs_replicated(self, normal):
        """Text-encoder kwargs for the replicated broadcast-load path (fits-in-GPU, multi-GPU).

        rank0 loads TEs real via the pipeline's from_pretrained, fp8-streamed when targeted
        (te_quant = _te_pipeline_quant_config()), so its host peak is one fp8 copy. Peers build meta
        components with the MATCHING layout (build_meta_component fp8-swaps targeted linears), so the
        later broadcast fills fp8 shards param-for-param with no re-quantize.

        A peer that cannot meta-build a component raises: the per-tensor broadcast walks
        named_parameters/buffers in lockstep, so a real bf16 fallback (no fp8 weight_scale buffers)
        against rank0's fp8 source would diverge the tensor count and desync the collective.
        """
        if _is_bcast_src(get_world_group()):
            return {}, self.model._te_pipeline_quant_config()
        te_components = [
            name for name in self.model.settings.fsdp_strategy
            if name != "transformer" and not name.startswith("transformer_")
        ]
        kwargs = {}
        for name in te_components:
            meta = self.build_meta_component(name, fp8=True)
            if meta is None:
                raise RuntimeError(
                    f"replicated broadcast-load: peer failed to meta-build text encoder "
                    f"'{name}'; its layout would diverge from rank0's and hang the broadcast"
                )
            kwargs[name] = meta
        return kwargs, None

    def broadcast_fill_replicated(self, offload: bool = False) -> None:
        """Fill every replicated big component with rank0's real weights via GPU->GPU broadcast,
        one component at a time. Bounds both host and VRAM peak to ~1x the model.

        Transformer: built on meta on every rank. Filled per block: rank0 reads one block off disk,
        broadcasts it over the world group, then ALL ranks fp8-quantize that block (symmetric, since
        the same bf16 yields the same fp8) before the next block. Peak = accumulating fp8 model + one
        transient bf16 block, so it fits a single GPU where the full bf16 model would not (~24 vs
        ~12 GB). No fully_shard: replicated keeps the full quantized block on every rank. Reuses the
        FSDP disk filler (_TransformerDiskFiller) and per-block quantize_fn (_build_fsdp_quantize_fn).

        Text encoders: rank0 loaded them real (fp8-streamed when targeted) via the pipeline; peers
        built matching-layout meta. Materialize peer meta to real-empty on device, then broadcast
        every param/buffer from rank0. No re-quantize, since both sides already carry the fp8 layout.
        """
        from diffusers.models.model_loading_utils import set_module_tensor_to_device
        world = get_world_group()
        device = f"cuda:{world.local_rank}"
        strategy = self.model.settings.fsdp_strategy
        for name in strategy:
            component = getattr(self.model.pipe, name, None)
            if component is None or not hasattr(component, "named_parameters"):
                continue
            if name == "transformer" or name.startswith("transformer_"):
                if self._all_ranks_loaded_real(component, world, device):
                    # Unwired runner: loaded real on EVERY rank (e.g. a composition-wrapper pipeline
                    # whose _load_model built the whole pipeline real). _fill_transformer_replicated's
                    # to_empty() would wipe those weights; skip the destructive fill and keep the real
                    # all-rank-symmetric weights (the post-load fp8 walk still quantizes them).
                    log(f"{name} loaded real on all ranks (unwired for replicated meta load); "
                        f"skipping broadcast fill, keeping real weights.")
                    continue
                self._fill_transformer_replicated(component, name, strategy[name], device, world)
            else:
                if self._all_ranks_loaded_real(component, world, device):
                    # Same unwired-runner case: TE loaded real on EVERY rank; skipping avoids
                    # broadcasting rank0's bytes over each peer's already-correct weights.
                    log(f"{name} loaded real on all ranks (unwired for replicated meta load); "
                        f"skipping broadcast fill, keeping real weights.")
                    continue
                self._fill_te_replicated(component, device, world, set_module_tensor_to_device)
            torch.cuda.empty_cache()
            log(f"Broadcast-filled {name} from rank0 (replicated). "
                f"host {host_mem_gb()} GB, VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def _all_ranks_loaded_real(self, component, world, device) -> bool:
        """True only if EVERY rank has this component fully real (no meta params).

        A per-rank `any(is_meta)` check diverges in the replicated path: rank0 loads the TEs real
        (fp8-streamed) while peers build them on meta, so rank0 would skip the fill's collectives
        while peers enter them -> the count-guard broadcast reads garbage and aborts (or hangs).
        All-reduce the local real flag so the skip decision is identical on every rank."""
        local_real = 0 if any(p.is_meta for p in component.parameters()) else 1
        flag = torch.tensor([local_real], device=device)
        return int(world.all_reduce(flag).item()) == world.world_size

    def _fill_transformer_replicated(self, component, name, strategy, device, world) -> None:
        """Per-block rank0-disk-read + world broadcast + symmetric per-block fp8 quantize (no shard)."""
        wrap_attrs = strategy.get("wrap_attrs", [])
        fill_block, finalize = self.build_transformer_disk_loaders(
            component, wrap_attrs, name, device, group=world
        )
        quantize_fn = self.model._build_fsdp_quantize_fn(name, wrap_attrs, world.local_rank)
        wrapped = []
        for attr in wrap_attrs:
            wrapped.extend(rgetattr(component, attr))
        for i, block in enumerate(wrapped):
            block.to_empty(device=device, recurse=True)
            fill_block(block, i)
            if quantize_fn is not None:
                quantize_fn(block, i)
            torch.cuda.empty_cache()
        finalize(component)

    def _fill_te_replicated(self, component, device, world, set_module_tensor_to_device) -> None:
        """Materialize peer meta to real-empty on device (move any real-CPU tensor on-device), then
        broadcast every param/buffer from world-rank0. Layout already matches (rank0 fp8-streamed,
        peers meta fp8-swapped), so no re-quantize.

        Two hazards, because rank0 builds the TE real via the pipeline's from_pretrained while peers
        build it on meta via _from_config (build_meta_component):

        1. dtype divergence (the fix below). The meta build instantiates at the pipeline's compute
           dtype (bf16), so auto-computed buffers can differ from the real build -- Qwen3 rotary
           inv_freq / original_inv_freq are fp32 real but bf16 on meta. world.broadcast copies rank0's
           element bytes into the peer tensor's EXISTING storage, so a bf16 slot receiving fp32 bytes
           is silently corrupted -> garbage RoPE frequencies -> NaN/black (survived ulysses<=2, broke
           at ulysses=4 where the SP chunk amplifies the corrupt rotary positions). Broadcast rank0's
           (shape, dtype) per name and reallocate any mismatched peer tensor before the data broadcast.

        2. enumeration order. The two builders are not guaranteed to enumerate named_* in the same
           order, so drive the broadcast off a sorted name order: every rank iterates an identical
           sequence over the (count-guarded, class-identical) name set, landing each broadcast into the
           peer tensor of the same name regardless of per-builder registration order.

        Capture the name lists ONCE and drive the materialize pass off them (via rgetattr), mirroring
        _TransformerDiskFiller.finalize. Re-enumerating named_parameters() for the broadcast is unsafe:
        set_module_tensor_to_device replaces param objects, so a re-enumeration can hand back a
        different (still-CPU) object than the one the materialize pass moved, and world.broadcast on a
        CPU tensor aborts the NCCL group.

        remove_duplicate=False: T5-family TEs tie shared.weight to encoder.embed_tokens.weight. rank0
        (from_pretrained) keeps the tie -> dedup yields one name; peers (_from_config on meta) build
        them untied -> two names. Enumerating with duplicates makes both sides expose the SAME name
        SET; each tied name then broadcasts rank0's shared tensor into the matching peer name, leaving
        every peer alias holding rank0's value (effectively tied) instead of desyncing the collective.
        """
        param_names = [n for n, _ in component.named_parameters(recurse=True, remove_duplicate=False)]
        buffer_names = [n for n, _ in component.named_buffers(recurse=True, remove_duplicate=False)]
        for name in param_names + buffer_names:
            t = rgetattr(component, name)
            if t.is_meta:
                set_module_tensor_to_device(
                    component, name, device,
                    value=torch.empty(t.shape, dtype=t.dtype, device=device),
                )
            elif t.device.type != "cuda":
                set_module_tensor_to_device(component, name, device, value=t.to(device))
        self._assert_tensor_count_agrees(component, world, device)
        ordered = sorted(param_names) + sorted(buffer_names)
        # dtype/shape realign before the byte-level broadcast. rank0 builds the TE real via
        # from_pretrained while peers build it on meta at the pipeline's compute dtype (bf16), so
        # auto-computed buffers can diverge in dtype -- e.g. Qwen3 rotary inv_freq is fp32 on the
        # real build but bf16 on the meta build. world.broadcast copies rank0's element bytes into
        # the peer tensor's existing storage; a bf16 slot receiving fp32 bytes is silently corrupted
        # (garbage RoPE frequencies -> black output). Broadcast rank0's (shape, dtype) per name and
        # reallocate any peer tensor that differs so every broadcast lands into a matching-layout slot.
        spec = (
            {n: (tuple(rgetattr(component, n).shape), rgetattr(component, n).dtype) for n in ordered}
            if world.rank_in_group == 0 else None
        )
        box = [spec]
        world.broadcast_object_list(box, src=0)
        spec = box[0]
        if world.rank_in_group != 0:
            for name in ordered:
                t = rgetattr(component, name)
                shape, dtype = spec[name]
                if tuple(t.shape) != shape or t.dtype != dtype:
                    # dtype= is required: set_module_tensor_to_device otherwise casts `value` back to
                    # the EXISTING buffer dtype (bf16), silently no-oping the fp32 realloc.
                    set_module_tensor_to_device(
                        component, name, device,
                        value=torch.empty(shape, dtype=dtype, device=device),
                        dtype=dtype,
                    )
        for name in ordered:
            # .contiguous() on the src: rank0's from_pretrained real tensors are trusted as-is and a
            # non-contiguous buffer (strided/transposed view) would broadcast the wrong element bytes
            # into peers. No-op on peers (freshly materialized -> already contiguous, filled in place).
            world.broadcast(rgetattr(component, name).data.contiguous(), src=0)

    def _assert_tensor_count_agrees(self, module, group, device) -> None:
        """Collective guard that every rank sees the same param+buffer count for `module`.

        Broadcasts rank0's count and compares. A peer whose meta tree diverged (e.g. an uneven
        fp8 swap that added or dropped weight_scale buffers) raises here, before the per-tensor
        broadcast loop, turning a silent NCCL hang into a clear error.

        Counts with remove_duplicate=False to match the broadcast loop's enumeration exactly:
        counting deduped while broadcasting non-deduped would let a tied-weight mismatch (rank0
        tied, peer untied) pass this guard yet still hang the per-tensor loop."""
        local_n = (
            sum(1 for _ in module.named_parameters(recurse=True, remove_duplicate=False))
            + sum(1 for _ in module.named_buffers(recurse=True, remove_duplicate=False))
        )
        ref = torch.tensor([local_n], device=device)
        group.broadcast(ref, src=0)
        if int(ref.item()) != local_n:
            raise RuntimeError(
                f"replicated broadcast-load: param/buffer count mismatch "
                f"(rank0={int(ref.item())}, local={local_n}); meta layout diverged from rank0"
            )

    def build_transformer_disk_loaders(self, component, wrap_attrs, subfolder, device, group=None):
        """(load_block_fn, load_epilogue_fn) filling a meta transformer from disk (rank0-read + bcast).

        group: broadcast group (default get_fs_group() for the FSDP path). The replicated path passes
        get_world_group() — get_fs_group() has world_size 1 when fully_shard_degree==1, so its
        broadcast would be a no-op and peers would receive garbage."""
        filler = _TransformerDiskFiller(self.model, component, wrap_attrs, subfolder, device, group)
        return filler.fill_block, filler.finalize

    def broadcast_load(self, component, component_name: str, offload: bool) -> None:
        """Fill a meta-initialized, FSDP-sharded text-encoder component with real weights.

        rank0 loads the component once and scatters it block-by-block; the op is collective.
        Buffers stay on the GPU regardless of
        offload: CPUOffloadPolicy manages params only, and buffers (fp8 weight_scale, rotary caches)
        are tiny and consumed on-device each forward.
        """
        self._broadcast_load_component(component, component_name, offload)

        if offload and torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.current_device()}"
            for buf in component.buffers():
                if buf.device.type == "cpu":
                    buf.data = buf.data.to(dev)

    def _swap_meta_te_to_fp8(self, module, targets: list) -> None:
        """In-place swap targeted meta nn.Linear leaves to meta xFuserFP8BlockScaleLinear in the
        TE plain layout (fp8 `weight` + fp32 `weight_scale`), so FSDP shards fp8 and the rank0
        broadcast fills fp8 shards. The placeholder weight is flipped bf16->fp8 so its DTensor
        dtype matches the fp8 source state dict."""
        import torch.nn as nn
        from xfuser.model_executor.quant.aiter_fp8_quantizer import _swap_linears_to_fp8
        from xfuser.model_executor.layers.fp8_linear import (
            xFuserFP8BlockScaleLinear, _fp8_dtype,
        )
        for t in targets:
            _swap_linears_to_fp8(module.get_submodule(t), preshuffle=False, add_scale_buffer=True)
        fp8 = _fp8_dtype()
        for m in module.modules():
            if isinstance(m, xFuserFP8BlockScaleLinear) and m.weight is not None and m.weight.is_meta:
                m.weight = nn.Parameter(m.weight.to(fp8), requires_grad=False)
                # Normalize to rank0's post-load layout: fp8 in `weight_fp8` (param) + `weight_scale`
                # (buffer) + a plain-attr `weight` sentinel. rank0 builds the real component via the
                # HfQuantizer whose _process_model_after_weight_loading absorbs the same way, so peers
                # and rank0 expose identical named_parameters/named_buffers (name, order, shape) — a
                # prerequisite for both the positional replicated broadcast and set_model_state_dict.
                m.absorb_fp8_weight_from_weight_attr()

    def _load_rank0_source(self, component, component_name: str):
        """rank0's full host copy of a component, fp8-quantized when the component wants fp8.

        from_pretrained resolves tied weights; the fp8 HfQuantizer streams straight to fp8 so the
        source is fp8-sized, not bf16-sized, on rank0.
        """
        kwargs = {}
        if self.model._component_wants_fp8(component_name):
            from xfuser.model_executor.quant import AiterFp8BlockScaleTEConfig
            kwargs["quantization_config"] = AiterFp8BlockScaleTEConfig(
                target_modules=self.model._fp8_targets_for_component(component_name)
            )
        return type(component).from_pretrained(
            self.model.settings.model_name,
            subfolder=component_name,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

    def _broadcast_load_component(
        self, component, component_name: str, offload: bool
    ) -> None:
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict, StateDictOptions,
        )
        wrap_attrs = self.model.settings.fsdp_strategy[component_name].get("wrap_attrs", [])
        is_src = _is_bcast_src(get_fs_group())
        full_sd: dict = {}
        src = None
        if is_src:
            src = self._load_rank0_source(component, component_name)
            full_sd = src.state_dict()

        # broadcast_from_rank0 scatters rank0's full tensors into each rank's DTensor shard; a
        # partial dict + strict=False lets us scatter one module (block/tail) at a time so peers
        # never receive the whole model at once. cpu_offload places filled params on CPU to satisfy
        # CPUOffloadPolicy (broadcast_from_rank0 otherwise defaults them to cuda).
        opts = StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=True, strict=False, cpu_offload=offload
        )
        block_prefixes = tuple(f"{a}." for a in wrap_attrs)

        for attr in wrap_attrs:
            prefix = f"{attr}."
            for idx, block in enumerate(rgetattr(component, attr)):
                bp = f"{prefix}{idx}."
                # Block-relative keys (from_pretrained gives full paths; block.state_dict is relative).
                block_sd = (
                    {k[len(bp):]: v for k, v in full_sd.items() if k.startswith(bp)}
                    if is_src else {}
                )
                set_model_state_dict(block, block_sd, options=opts)

        # Non-block params/buffers: embeddings, norms, lm_head.
        tail_sd = (
            {k: v for k, v in full_sd.items() if not k.startswith(block_prefixes)}
            if is_src else {}
        )
        set_model_state_dict(component, tail_sd, options=opts)

        del full_sd, src
        self._release_rank0_source(is_src, component_name)

    def _release_rank0_source(self, is_src: bool, component_name: str) -> None:
        """Release rank0's transient full host copy after a component is broadcast.

        from_pretrained models can survive `del` via ref cycles (hooks/config/tied weights), so
        force a collect before the next component loads, then drop the checkpoint page cache.
        """
        if is_src:
            gc.collect()
            drop_file_page_cache(
                component_shard_paths(self.model.settings.model_name, component_name, "model")
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _TransformerDiskFiller:
    """Fills a meta transformer's blocks with real weights, reading on fs-rank0 and broadcasting
    each tensor GPU->GPU to the fs group. Holds the checkpoint weight_map, fs group, and the open-
    handle ExitStack across the per-block fill and the epilogue. See module docstring for why the
    read is rank0-only (block-128 fp8 tile constraint + host-anon N-scaling)."""

    def __init__(self, model, component, wrap_attrs, subfolder, device, group=None) -> None:
        from contextlib import ExitStack

        self.model = model
        self.subfolder = subfolder
        self.device = device
        self.group = group or get_fs_group()
        self.is_src = _is_bcast_src(self.group)
        # Only rank0 reads the checkpoint; peers receive via broadcast and never open a file
        # (no per-peer mmap page cache, no redundant hub revalidation HEADs).
        self.weight_map = (
            resolve_checkpoint_weight_map(model.settings.model_name, subfolder)
            if self.is_src else {}
        )
        self.shard_paths = set(self.weight_map.values())
        self._handle_cache: dict[str, object] = {}
        self._stack = ExitStack()
        self._block_prefixes = tuple(f"{a}." for a in wrap_attrs)
        self._id2fqn: dict[int, str] = {}
        for attr in wrap_attrs:
            for idx, mod in enumerate(rgetattr(component, attr)):
                self._id2fqn[id(mod)] = f"{attr}.{idx}"

    def _handle(self, path):
        from safetensors import safe_open
        h = self._handle_cache.get(path)
        if h is None:
            h = self._stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            self._handle_cache[path] = h
        return h

    def _ckpt_key(self, root, name):
        """Map a live (possibly wrapped) param/buffer name to its checkpoint key.

        xFuser layer/model wrappers register the real module as a submodule named
        'module', so multi-GPU named_* emit '...module...' segments the checkpoint
        never has. Drop each 'module' segment whose parent is an xFuser wrapper;
        real submodules literally named 'module' are left intact.
        """
        from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
        cur, out = root, []
        for seg in name.split("."):
            if seg == "module" and isinstance(cur, xFuserBaseWrapper):
                cur = getattr(cur, seg)
                continue
            out.append(seg)
            cur = getattr(cur, seg)
        return ".".join(out)

    def _fill(self, module, local_name, key, required):
        from diffusers.models.model_loading_utils import set_module_tensor_to_device
        path = self.weight_map.get(key)
        if path is None:
            if required:
                raise RuntimeError(f"missing checkpoint weight for {key} in {self.subfolder}")
            return
        set_module_tensor_to_device(
            module, local_name, self.device, value=self._handle(path).get_tensor(key)
        )

    def _broadcast(self, module):
        # Collective: all group ranks must call in the same order. Module structure is identical
        # across ranks (meta -> to_empty), so named_* iteration order matches. remove_duplicate=False
        # so tied weights emit the same name count on every rank regardless of per-rank tie state.
        for _, p in module.named_parameters(recurse=True, remove_duplicate=False):
            self.group.broadcast(p.data, src=0)
        for _, b in module.named_buffers(recurse=True, remove_duplicate=False):
            self.group.broadcast(b.data, src=0)

    def fill_block(self, block, i):
        """Fill + broadcast one wrapped block. Missing keys (non-persistent buffers) skipped."""
        fqn = self._id2fqn.get(id(block))
        if fqn is None:
            raise RuntimeError(f"block {i} not found in wrap_attrs index (id mismatch)")
        if self.is_src:
            prefix = fqn + "."
            for local_name, _ in block.named_parameters():
                self._fill(block, local_name, prefix + self._ckpt_key(block, local_name), required=True)
            for local_name, _ in block.named_buffers():
                self._fill(block, local_name, prefix + self._ckpt_key(block, local_name), required=False)
        self._broadcast(block)
        if i % 8 == 0:
            log(f"  self-fill {self.subfolder} block {i}: host cur/anon/file "
                f"{host_mem_gb()} GB, VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def finalize(self, comp):
        """Fill the non-block remainder before the component-level shard.

        Only blocks are to_empty'd by shard_component; the non-block remainder is still meta on
        peers (rank0 gets real tensors from _fill). Broadcast can't run on meta, so materialize
        every non-block tensor to real-empty on all ranks first, then rank0 fills, then broadcast.
        Non-persistent buffers absent from disk stay empty and are recomputed on forward, so
        broadcasting their garbage is harmless.
        """
        from diffusers.models.model_loading_utils import set_module_tensor_to_device
        # Block-membership test must run on the unwrapped name: xFuser wrappers insert 'module'
        # segments, so a raw wrapped name (module.transformer_blocks.0...) never matches the
        # 'transformer_blocks.' prefix and every block param would leak into the tail (then miss,
        # e.g. runtime-only weight_fp8 which has no checkpoint key). fill_block already handled blocks.
        tail = [
            name for name, _ in comp.named_parameters(remove_duplicate=False)
            if not self._ckpt_key(comp, name).startswith(self._block_prefixes)
        ]
        tail_bufs = [
            name for name, _ in comp.named_buffers(remove_duplicate=False)
            if not self._ckpt_key(comp, name).startswith(self._block_prefixes)
        ]
        target_type = torch.device(self.device).type
        for name in tail + tail_bufs:
            t = rgetattr(comp, name)
            if t.is_meta:
                set_module_tensor_to_device(
                    comp, name, self.device,
                    value=torch.empty(t.shape, dtype=t.dtype, device=self.device),
                )
            elif t.device.type != target_type:
                # Non-persistent buffers (e.g. Wan rope freqs_cos/freqs_sin) are created real on
                # CPU by init_empty_weights (include_buffers=False), not meta. Their values are
                # correct and identical across ranks, but the broadcast below is GPU-only, so move
                # them on-device first.
                set_module_tensor_to_device(comp, name, self.device, value=t.to(self.device))
        if self.is_src:
            for name in tail:
                self._fill(comp, name, self._ckpt_key(comp, name), required=True)
            for name in tail_bufs:
                self._fill(comp, name, self._ckpt_key(comp, name), required=False)
        for name in tail + tail_bufs:
            self.group.broadcast(rgetattr(comp, name).data, src=0)
        self._stack.close()
        if self.is_src:
            drop_file_page_cache(self.shard_paths)

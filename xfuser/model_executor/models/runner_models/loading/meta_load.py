"""Memory-efficient load: build components on meta, then fill their real weights without ever
materializing a full copy on host.

Serves two load shapes that never both apply, since one needs weight-splitting parallelism and the
other rules it out:

* Sharded (``fsdp_meta_load``, needs fully_shard_degree > 1): ranks legitimately hold different
  weights, so each fills its own shard.
* Replicated (``replicated_broadcast_load``, needs no weight-splitting parallelism): every rank holds
  the same weights, so rank0 loads once and broadcasts, keeping host peak at 1x the model instead of
  Nx. Nothing is sharded on this path.

Two fill strategies, both collective (every fs-group rank must call in identical order):

* Transformer (``_TransformerDiskFiller``): self-fill. Each block's real weights are read from disk
  on fs-rank0 ONLY and broadcast GPU->GPU to the group. rank0-only read is required because the full
  block must exist on every rank before block-128 fp8 quantization (a shard boundary splitting a
  128x128 tile invalidates the tile scale, so per-rank slice reads are impossible), and if every rank
  read the full block from disk host anon would scale with N ranks (measured +3.5GB per block, enough
  to trip the cgroup OOM killer). Reading on rank0 then broadcasting keeps host disk-read anon at 1x.

* Text encoders (``MemoryEfficientLoader.broadcast_load``): rank0 loads once via from_pretrained
  (resolves tied weights), then scatters one wrapped block at a time via broadcast_from_rank0, so
  peers never receive the whole model at once. fp8-targeted TEs stream rank0 straight to fp8.

``MemoryEfficientLoader`` holds the ``xFuserModel`` so it can reuse the run's FP8 plan
(``model.fp8``, see ``fp8_plan``) and settings without duplicating them.
"""

import gc
import weakref

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


def _tensor_layout(module) -> tuple[tuple[str, str], ...]:
    """Ordered parameter/buffer names, preserving aliases on both builders."""
    return tuple(
        [("parameter", name) for name, _ in module.named_parameters(
            recurse=True, remove_duplicate=False
        )]
        + [("buffer", name) for name, _ in module.named_buffers(
            recurse=True, remove_duplicate=False
        )]
    )


def _collective_assert_same_layout(local_layout, group, device) -> None:
    """Collectively reject any ordered layout mismatch before data broadcasts begin."""
    box = [local_layout if group.rank_in_group == 0 else None]
    group.broadcast_object_list(box, src=0)
    reference = box[0]
    local_mismatch = int(local_layout != reference)
    mismatch = torch.tensor([local_mismatch], device=device)
    mismatch_count = int(group.all_reduce(mismatch).item())
    if mismatch_count:
        detail = ""
        if local_mismatch:
            first = next(
                (
                    i, expected, actual
                )
                for i, (expected, actual) in enumerate(
                    zip(reference, local_layout)
                )
                if expected != actual
            ) if len(reference) == len(local_layout) else (
                min(len(reference), len(local_layout)),
                reference[min(len(reference), len(local_layout)):]
                or "<end>",
                local_layout[min(len(reference), len(local_layout)):]
                or "<end>",
            )
            detail = f"; first local difference at {first[0]}: rank0={first[1]!r}, local={first[2]!r}"
        raise RuntimeError(
            "replicated broadcast-load: ordered parameter/buffer layout mismatch on "
            f"{mismatch_count} of {group.world_size} ranks{detail}"
        )


def _collective_source_call(group, is_src, operation, context):
    """Run a rank0 operation and broadcast its failure status before any rank continues."""
    result = None
    source_error = None
    status = None
    if is_src:
        try:
            result = operation()
        except Exception as error:
            source_error = error
            status = (type(error).__name__, str(error))
    box = [status]
    group.broadcast_object_list(box, src=0)
    status = box[0]
    if status is not None:
        error_type, message = status
        raise RuntimeError(
            f"{context} failed on rank0: {error_type}: {message}"
        ) from source_error
    return result


def _collective_build_call(group, operation, context):
    """Run a build locally, then make every participating rank agree on any failure."""
    if (
        group is None
        or group.world_size <= 1
        or not hasattr(group, "broadcast_object_list")
    ):
        return operation()

    result = None
    local_error = None
    try:
        result = operation()
    except Exception as error:
        local_error = (type(error).__name__, str(error))

    failures = []
    for src in range(group.world_size):
        box = [local_error if group.rank_in_group == src else None]
        group.broadcast_object_list(box, src=src)
        if box[0] is not None:
            failures.append((src, *box[0]))
    if failures:
        rank, error_type, message = failures[0]
        raise RuntimeError(
            f"{context} failed on rank {rank}: {error_type}: {message}"
        )
    return result


def _collective_reconcile_tensor_specs(module, names, group, device):
    """Make peer tensor storage match rank0 shape/dtype before positional data broadcasts."""
    from diffusers.models.model_loading_utils import set_module_tensor_to_device

    spec = (
        [
            (name, tuple(rgetattr(module, name).shape), rgetattr(module, name).dtype)
            for name in names
        ]
        if group.rank_in_group == 0 else None
    )
    box = [spec]
    group.broadcast_object_list(box, src=0)

    def reconcile():
        if group.rank_in_group != 0:
            for name, shape, dtype in box[0]:
                tensor = rgetattr(module, name)
                if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    set_module_tensor_to_device(
                        module,
                        name,
                        device,
                        value=torch.empty(shape, dtype=dtype, device=device),
                        dtype=dtype,
                    )

    _collective_build_call(group, reconcile, context="transformer tensor-spec reconciliation")


def _persistent_named_buffers(module):
    """Named buffers saved by state_dict, using each buffer owner's persistence set."""
    persistent = []
    for name, buffer in module.named_buffers(recurse=True, remove_duplicate=False):
        parent_name, _, local_name = name.rpartition(".")
        owner = module.get_submodule(parent_name) if parent_name else module
        if local_name not in owner._non_persistent_buffers_set:
            persistent.append((name, buffer))
    return persistent


class MemoryEfficientLoader:
    """Builds pipeline components on meta and fills their real weights without a full host copy.

    Fills FSDP shards from disk on the sharded path, and broadcasts rank0's weights to replicated
    peers on the unsharded one; ``fsdp_meta_load`` and ``replicated_broadcast_load`` say which
    applies. Owns the ``xFuserModel`` (``model``) to reuse its settings and fp8 predicates.
    """

    def __init__(self, model) -> None:
        self.model = model
        # The meta transformers this loader built, so the shard step can recognize them by identity
        # rather than by guessing from the component's name. Weak so a component the pipeline
        # replaces or drops is not kept alive by the bookkeeping.
        self._meta_transformers = weakref.WeakSet()
        # Resolved on first use and cached: several load-time seams consult it, so this keeps the
        # decision identical everywhere and logs the reason once.
        self._replicated_decision = None

    def fsdp_meta_load(self) -> bool:
        """True when the memory-efficient sharded (meta-init + per-block rank0-read/broadcast fill)
        load path is on."""
        config = self.model.config
        return bool(config.memory_efficient_sharding and config.fully_shard_degree > 1)

    def replicated_broadcast_load(self) -> bool:
        """Whether to load replicated components once on rank0 and broadcast them to peers.

        A model that fits one GPU but runs multi-GPU for pure sequence/CFG/data parallelism is
        replicated across ranks, so every rank would otherwise from_pretrained its own full host
        copy: host RAM = N x model, which trips the container's OOM killer during load long before
        VRAM is the binding constraint (observed: flux 2gpu 93.6GB, 4gpu 171GB). Instead rank0 loads
        real weights to GPU while peers build on meta and receive every param/buffer over a
        GPU->GPU broadcast, so host peak is 1x the model whatever the rank count.

        Off unless --memory_efficient_replicated_load is passed, like --memory_efficient_sharding:
        a mismatched collective on this path hangs rather than raises, so it is the user's call
        rather than something inferred from the hardware. Even when asked for, it is skipped where
        it cannot apply: weight-splitting parallelism (FSDP, PipeFusion, tensor parallel), where
        ranks legitimately hold different weights and a broadcast would overwrite them; a single
        rank, where there is no peer to broadcast to; and runners not wired for the path (the
        model's _supports_replicated_meta_load).
        """
        if self._replicated_decision is None:
            self._replicated_decision = self._resolve_replicated_broadcast_load()
        return self._replicated_decision

    def _resolve_replicated_broadcast_load(self) -> bool:
        config = self.model.config
        if not config.memory_efficient_replicated_load:
            return False
        splits_weights_per_rank = (
            config.fully_shard_degree > 1
            or config.pipefusion_parallel_degree > 1
            or config.tensor_parallel_degree > 1
        )
        if splits_weights_per_rank:
            log("--memory_efficient_replicated_load ignored: this run splits weights per rank "
                "(FSDP/PipeFusion/tensor parallel), so peers hold different weights than rank0 and "
                "a broadcast would overwrite them. Loading per rank.")
            return False
        if get_world_group().world_size == 1:
            log("--memory_efficient_replicated_load ignored: single-rank run has no peer to "
                "broadcast to. Loading normally.")
            return False
        if not self.model._supports_replicated_meta_load():
            log(f"--memory_efficient_replicated_load ignored: "
                f"{type(self.model).__name__} loads its components directly rather than through "
                f"the meta-build seams this path needs. Loading per rank.")
            return False
        log("Replicated rank0-broadcast load enabled by --memory_efficient_replicated_load "
            "(host peak 1x the model, not Nx).")
        return True

    def self_fills_from_disk(self, component) -> bool:
        """Whether this component is one we built on meta via build_meta_transformer, and can
        therefore fill per block from disk (see _TransformerDiskFiller) rather than by broadcasting
        a rank0 from_pretrained. True only for the component object we built: the two fill paths need
        different collectives, so this must not guess from the component's name."""
        return component in self._meta_transformers

    def build_meta_transformer(self, wrapper_cls, subfolder: str = "transformer", init_kwargs: dict | None = None):
        """Build the (diffusers) transformer wrapper on meta from its config only (no weights).

        Real weights are streamed per block from disk during sharding (see _TransformerDiskFiller),
        so the full model never materializes. Uses the diffusers-public from_config; fp8 quantization
        happens per block on the real weights during sharding, so no fp8 swap is done here.

        init_kwargs: extra wrapper __init__ args (e.g. wan's attention_kwargs) not in the on-disk
        config; forwarded to from_config so the meta model matches the from_pretrained path.
        """
        def build():
            from accelerate import init_empty_weights
            config = wrapper_cls.load_config(
                self.model.settings.model_name, subfolder=subfolder
            )
            with init_empty_weights():
                model = wrapper_cls.from_config(config, **(init_kwargs or {}))
            # Match the checkpoint dtype before disk fill and quantization.
            return model.to(torch.bfloat16)

        model = _collective_build_call(
            get_world_group(), build, context=f"meta transformer '{subfolder}'"
        )
        self._meta_transformers.add(model)
        return model

    def build_meta_component(self, component_name: str, fp8: bool = True):
        """Instantiate a transformers pipeline component on meta from its config (no weights).

        Resolves the component's class from the pipeline's model_index.json and builds it under
        init_empty_weights so every param lands on meta. When fp8-targeted (RDNA4 AITER) and ``fp8``
        is True, its targeted Linears are swapped to meta fp8 layers so the model is sharded and
        filled as fp8. ``fp8=False`` keeps the component bf16 on meta (replicated broadcast path:
        rank0 broadcasts bf16 and the per-rank fp8 walk quantizes locally afterwards).
        Returns None (caller falls back to a normal from_pretrained load) unless the component is a
        transformers model we can rebuild from config; real weights arrive later via broadcast_load.

        Only "this component is not rebuildable here" is caught, and only around the class/config
        resolution that can report it: a missing/unreadable config (OSError), a library the
        environment lacks (ImportError), or a class exposing no config_class (AttributeError).
        Building the module and swapping it to fp8 happen outside that guard, so a typo'd config key
        or a bug in the swap propagates rather than hiding behind a load that merely looks slower.
        Note the fallback is rank-local, so a rank that takes it while its peers build meta
        diverges; ``agreed_is_meta`` catches that downstream and fails every rank.
        """
        from diffusers import DiffusionPipeline
        import importlib
        from accelerate import init_empty_weights

        # Only resolving the component's class and config is guarded: that is where "this component
        # is not rebuildable here" shows up. Construction and the fp8 swap run outside, so a bug in
        # either surfaces instead of degrading every rank to a normal load behind one log line
        # (symmetrically, so agreed_is_meta would not catch it either).
        try:
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
        except (OSError, ImportError, AttributeError) as e:
            log(f"Meta-init of component '{component_name}' failed "
                f"({type(e).__name__}: {e}); using normal load.")
            return None

        with init_empty_weights():
            component = cls._from_config(config)
        # _from_config defaults to fp32; align meta params to bf16 (dtype-only .to is legal on
        # meta) so their DTensor dtype matches the broadcast source.
        component = component.to(torch.bfloat16)
        if fp8 and self.model.fp8.aiter_covers(component_name):
            self._swap_meta_te_to_fp8(
                component, self.model.fp8.targets_for(component_name)
            )
        return component

    def meta_te_kwargs(self):
        """Build text-encoder(s) on meta for the pipeline's from_pretrained (meta FSDP load path).

        Returns (pipe_component_kwargs, None): the kwargs carry meta modules so the pipeline skips
        loading those components, and te_quant is None (the meta module is filled by broadcast_load,
        not streamed by the pipe). Returns None when there are no TE components or any one of them
        cannot be meta-built, leaving the caller to take its normal load.
        """
        te_components = [
            name for name in self.model.settings.fsdp_strategy
            if name != "transformer" and not name.startswith("transformer_")
        ]
        if not te_components:
            return None
        def build():
            kwargs = {}
            for name in te_components:
                meta = self.build_meta_component(name)
                if meta is None:
                    raise RuntimeError(f"could not meta-build text encoder '{name}'")
                kwargs[name] = meta
            return kwargs, None

        return _collective_build_call(
            get_world_group(), build, context="FSDP text-encoder meta construction"
        )

    def meta_te_kwargs_replicated(self):
        """Text-encoder kwargs for the replicated broadcast-load path (fits-in-GPU, multi-GPU).

        rank0 loads TEs real via the pipeline's from_pretrained, fp8-streamed when targeted
        (te_quant from the run's fp8 plan), so its host peak is one fp8 copy. Peers build meta
        components with the MATCHING layout (build_meta_component fp8-swaps targeted linears), so the
        later broadcast fills fp8 shards param-for-param with no re-quantize.

        A peer that cannot meta-build a component raises: the per-tensor broadcast walks
        named_parameters/buffers in lockstep, so a real bf16 fallback (no fp8 weight_scale buffers)
        against rank0's fp8 source would diverge the tensor count and desync the collective.
        """
        world = get_world_group()

        def build():
            if _is_bcast_src(world):
                return {}, self.model.fp8.aiter_te_pipeline_config()
            te_components = [
                name for name in self.model.settings.fsdp_strategy
                if name != "transformer" and not name.startswith("transformer_")
            ]
            kwargs = {}
            for name in te_components:
                meta = self.build_meta_component(name, fp8=True)
                if meta is None:
                    raise RuntimeError(
                        "replicated broadcast-load: peer failed to meta-build text encoder "
                        f"'{name}'; its layout would diverge from rank0's"
                    )
                kwargs[name] = meta
            return kwargs, None

        return _collective_build_call(
            world, build, context="replicated text-encoder meta construction"
        )

    def broadcast_fill_replicated(self, offload: bool = False) -> None:
        """Fill every replicated big component with rank0's real weights via GPU->GPU broadcast,
        one component at a time. Bounds both host and VRAM peak to ~1x the model.

        Transformer: built on meta on every rank. Filled per block: rank0 reads one block off disk,
        broadcasts it over the world group, then ALL ranks fp8-quantize that block (symmetric, since
        the same bf16 yields the same fp8) before the next block. Peak = accumulating fp8 model + one
        transient bf16 block, so it fits a single GPU where the full bf16 model would not (~24 vs
        ~12 GB). No fully_shard: replicated keeps the full quantized block on every rank. Reuses the
        FSDP disk filler (_TransformerDiskFiller) and per-block quantize_fn (``shard`` module).

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
        All-reduce the local real flag so the skip decision is identical on every rank.

        Deliberately tolerant of disagreement, unlike ``agreed_is_meta``: rank0-real/peer-meta is
        the normal shape of the replicated path, and it means "fall through to the fill"."""
        local_real = 0 if any(p.is_meta for p in component.parameters()) else 1
        flag = torch.tensor([local_real], device=device)
        return int(world.all_reduce(flag).item()) == world.world_size

    def agreed_is_meta(self, component, component_name: str, group, device) -> bool:
        """Whether `component` was built on meta, as agreed by every rank in `group`.

        The answer selects between collective branches (FSDP meta_init, the per-block disk fill,
        the rank0 broadcast fill), so a rank-local answer that diverges pairs a broadcast on some
        ranks with no broadcast on others — an unrecoverable hang with no traceback. Ranks can
        diverge here because ``build_meta_component`` falls back to a real load per rank when a
        component is not rebuildable, which one rank can hit alone (a transient filesystem or Hub
        error). All-reduce the local flag and refuse to continue unless every rank agrees, so that
        case fails loudly, naming the component, instead of hanging.
        """
        local_meta = 1 if any(p.is_meta for p in component.parameters()) else 0
        n_meta = int(group.all_reduce(torch.tensor([local_meta], device=device)).item())
        if n_meta not in (0, group.world_size):
            raise RuntimeError(
                f"'{component_name}' was built on meta on {n_meta} of {group.world_size} ranks; "
                f"the meta and real load paths use different collectives, so continuing would "
                f"hang. This rank built it "
                f"{'on meta' if local_meta else 'real'}; a rank that fell back to a real load "
                f"logged why above ('Meta-init of component ... failed')."
            )
        return n_meta == group.world_size

    def _fill_transformer_replicated(self, component, name, strategy, device, world) -> None:
        """Per-block rank0-disk-read + world broadcast + symmetric per-block fp8 quantize (no shard)."""
        wrap_attrs = strategy.get("wrap_attrs", [])
        fill_block, finalize = self.build_transformer_disk_loaders(
            component, wrap_attrs, name, device, group=world
        )
        from .shard import build_block_quantize_fn
        from xfuser.core.distributed.sharding import (
            _restore_nonpersistent_buffers,
            _save_nonpersistent_buffers,
        )
        quantize_fn = build_block_quantize_fn(self.model, name, wrap_attrs, world.local_rank)
        wrapped = []
        for attr in wrap_attrs:
            wrapped.extend(rgetattr(component, attr))
        for i, block in enumerate(wrapped):
            nonpersistent_buffers = _save_nonpersistent_buffers(block, device)
            block.to_empty(device=device, recurse=True)
            _restore_nonpersistent_buffers(nonpersistent_buffers)
            fill_block(block, i)
            if quantize_fn is not None:
                _collective_build_call(
                    world,
                    lambda: quantize_fn(block, i),
                    context=f"quantizing replicated transformer block {i}",
                )
            torch.cuda.empty_cache()
        finalize(component)

    def _fill_te_replicated(self, component, device, world, set_module_tensor_to_device) -> None:
        """Materialize peer meta to real-empty on device (move any real-CPU tensor on-device), then
        broadcast every param/buffer from world-rank0. Layout already matches (rank0 fp8-streamed,
        peers meta fp8-swapped), so no re-quantize.

        Two hazards, because rank0 builds the TE real via the pipeline's from_pretrained while peers
        build it on meta via _from_config (build_meta_component):

        1. dtype divergence. The meta build instantiates at the pipeline's compute dtype (bf16),
           so auto-computed buffers can differ from the real build -- Qwen3 rotary
           inv_freq / original_inv_freq are fp32 real but bf16 on meta. world.broadcast copies rank0's
           element bytes into the peer tensor's EXISTING storage, so a bf16 slot receiving fp32 bytes
           is silently corrupted -> garbage RoPE frequencies -> NaN/black (survived ulysses<=2, broke
           at ulysses=4 where the SP chunk amplifies the corrupt rotary positions). Broadcast rank0's
           (shape, dtype) per name and reallocate any mismatched peer tensor before the data broadcast.

        2. layout divergence. Compare the exact ordered parameter/buffer names collectively before
           materializing or broadcasting anything. This catches same-count reorderings and aliases,
           and makes every rank raise together instead of letting rank0 enter a later broadcast.

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
        layout = _tensor_layout(component)
        _collective_assert_same_layout(layout, world, device)
        param_names = [name for kind, name in layout if kind == "parameter"]
        buffer_names = [name for kind, name in layout if kind == "buffer"]
        for name in param_names + buffer_names:
            t = rgetattr(component, name)
            if t.is_meta:
                set_module_tensor_to_device(
                    component, name, device,
                    value=torch.empty(t.shape, dtype=t.dtype, device=device),
                )
            elif t.device.type != "cuda":
                set_module_tensor_to_device(component, name, device, value=t.to(device))
        ordered = param_names + buffer_names
        # dtype/shape realign: broadcast rank0's (shape, dtype) per name and reallocate any peer
        # tensor that differs so every broadcast lands into a matching-layout slot (hazard 1 above).
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
        # Peers must receive in place: .contiguous() on a peer would return a copy and the broadcast
        # would fill that temporary, leaving the real tensor unwritten. Check every destination
        # before the loop and agree the result, because raising from inside the loop would abort the
        # peer while rank0 blocks in broadcast, trading a silent corruption for a silent hang.
        strided = [
            name for name in ordered
            if world.rank_in_group != 0 and not rgetattr(component, name).data.is_contiguous()
        ]
        n_bad = int(world.all_reduce(torch.tensor([len(strided)], device=device)).item())
        if n_bad:
            raise RuntimeError(
                f"replicated broadcast-load: {n_bad} peer destination(s) are not contiguous and "
                f"cannot receive the broadcast in place"
                + (f"; on this rank: {strided[:3]}" if strided else " (offenders on other ranks)")
            )
        for name in ordered:
            tensor = rgetattr(component, name).data
            if world.rank_in_group == 0:
                # rank0's tensors come from from_pretrained and may be strided views, whose element
                # bytes are not the buffer bytes; send a contiguous copy instead.
                tensor = tensor.contiguous()
            world.broadcast(tensor, src=0)

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
        if self.model.fp8.aiter_covers(component_name):
            from xfuser.model_executor.quant import AiterFp8BlockScaleTEConfig
            kwargs["quantization_config"] = AiterFp8BlockScaleTEConfig(
                target_modules=self.model.fp8.targets_for(component_name)
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
        group = get_fs_group()
        is_src = _is_bcast_src(group)
        full_sd: dict = {}
        src = None

        def load_source_state_dict():
            nonlocal src
            src = self._load_rank0_source(component, component_name)
            return src.state_dict()

        try:
            full_sd = _collective_source_call(
                group,
                is_src,
                load_source_state_dict,
                context=f"loading {component_name} source state dict",
            ) or {}

            # broadcast_from_rank0 scatters rank0's full tensors into each rank's DTensor shard; a
            # partial dict + strict=False lets us scatter one module (block/tail) at a time so peers
            # never receive the whole model at once. cpu_offload places filled params on CPU to
            # satisfy CPUOffloadPolicy (broadcast_from_rank0 otherwise defaults them to cuda).
            opts = StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False, cpu_offload=offload
            )
            block_prefixes = tuple(f"{a}." for a in wrap_attrs)

            for attr in wrap_attrs:
                prefix = f"{attr}."
                for idx, block in enumerate(rgetattr(component, attr)):
                    bp = f"{prefix}{idx}."
                    # from_pretrained has full paths; block.state_dict uses block-relative keys.
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
        finally:
            del full_sd, src
            self._release_rank0_source(is_src, component_name)

    def _release_rank0_source(self, is_src: bool, component_name: str) -> None:
        """Release rank0's transient full host copy after a component is broadcast.

        from_pretrained models can survive `del` via ref cycles (hooks/config/tied weights), so
        force a collect before the next component loads, then drop the checkpoint page cache.

        Both shard basenames are tried because this fills whatever the caller did not route to the
        per-block disk fill: transformers components ("model") today, but a diffusers component
        ("diffusion_pytorch_model") if one ever lands here. An absent basename resolves to no paths,
        so the miss would otherwise be a silent no-op that leaves the cache full.
        """
        if is_src:
            gc.collect()
            paths = set()
            for basename in ("model", "diffusion_pytorch_model"):
                paths |= component_shard_paths(
                    self.model.settings.model_name, component_name, basename
                )
            drop_file_page_cache(paths)
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
        self.weight_map = _collective_source_call(
            self.group,
            self.is_src,
            lambda: resolve_checkpoint_weight_map(
                model.settings.model_name, subfolder
            ),
            context=f"resolving checkpoint map for {subfolder}",
        ) or {}
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

    def _require_checkpoint_keys(self, keys):
        """Collectively reject missing persistent tensors before any rank enters data broadcast."""
        box = [[key for key in keys if key not in self.weight_map] if self.is_src else None]
        self.group.broadcast_object_list(box, src=0)
        missing = box[0]
        if missing:
            preview = ", ".join(missing[:3])
            suffix = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            raise RuntimeError(
                f"missing checkpoint tensors in {self.subfolder}: {preview}{suffix}"
            )

    def _broadcast(self, module):
        # Collective: all group ranks must call in the same order. Module structure is identical
        # across ranks (meta -> to_empty), so named_* iteration order matches. remove_duplicate=False
        # so tied weights emit the same name count on every rank regardless of per-rank tie state.
        for _, p in module.named_parameters(recurse=True, remove_duplicate=False):
            self.group.broadcast(p.data, src=0)
        for _, b in _persistent_named_buffers(module):
            self.group.broadcast(b.data, src=0)

    def fill_block(self, block, i):
        """Fill + broadcast one wrapped block, excluding only non-persistent buffers."""
        fqn = self._id2fqn.get(id(block))
        if fqn is None:
            raise RuntimeError(f"block {i} not found in wrap_attrs index (id mismatch)")
        layout = _tensor_layout(block)
        _collective_assert_same_layout(layout, self.group, self.device)
        prefix = fqn + "."
        required = [
            (local_name, prefix + self._ckpt_key(block, local_name))
            for local_name, _ in block.named_parameters(remove_duplicate=False)
        ] + [
            (local_name, prefix + self._ckpt_key(block, local_name))
            for local_name, _ in _persistent_named_buffers(block)
        ]
        self._require_checkpoint_keys([key for _, key in required])
        for local_name, key in required:
            _collective_source_call(
                self.group,
                self.is_src,
                lambda local_name=local_name, key=key: self._fill(
                    block, local_name, key, required=True
                ),
                context=f"loading checkpoint tensor {key}",
            )
        broadcast_names = [
            name for kind, name in layout if kind == "parameter"
        ] + [name for name, _ in _persistent_named_buffers(block)]
        _collective_reconcile_tensor_specs(
            block, broadcast_names, self.group, self.device
        )
        self._broadcast(block)
        if i % 8 == 0:
            log(f"  self-fill {self.subfolder} block {i}: host cur/anon/file "
                f"{host_mem_gb()} GB, VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def finalize(self, comp):
        """Fill the non-block remainder before the component-level shard.

        Only blocks are to_empty'd by shard_component; the non-block remainder is still meta on
        peers (rank0 gets real tensors from _fill). Broadcast can't run on meta, so materialize
        every non-block tensor to real-empty on all ranks first, then rank0 fills, then broadcast.
        Non-persistent buffers absent from disk stay local and are recomputed on forward; they are
        neither required from the checkpoint nor broadcast as uninitialized storage.
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
        all_tail_bufs = [
            name for name, _ in comp.named_buffers(remove_duplicate=False)
            if not self._ckpt_key(comp, name).startswith(self._block_prefixes)
        ]
        tail_bufs = [
            name for name, _ in _persistent_named_buffers(comp)
            if not self._ckpt_key(comp, name).startswith(self._block_prefixes)
        ]
        tail_layout = tuple(
            [("parameter", name) for name in tail]
            + [("buffer", name) for name in all_tail_bufs]
        )
        _collective_assert_same_layout(tail_layout, self.group, self.device)
        self._require_checkpoint_keys(
            [self._ckpt_key(comp, name) for name in tail + tail_bufs]
        )
        target_type = torch.device(self.device).type
        for name in tail + all_tail_bufs:
            t = rgetattr(comp, name)
            if t.is_meta:
                set_module_tensor_to_device(
                    comp, name, self.device,
                    value=torch.empty(t.shape, dtype=t.dtype, device=self.device),
                )
            elif t.device.type != target_type:
                # Non-persistent buffers (e.g. Wan rope freqs_cos/freqs_sin) are created real on
                # CPU by init_empty_weights (include_buffers=False), not meta. Their values are
                # correct and identical across ranks, so move them on-device without broadcasting.
                set_module_tensor_to_device(comp, name, self.device, value=t.to(self.device))
        for name in tail + tail_bufs:
            key = self._ckpt_key(comp, name)
            _collective_source_call(
                self.group,
                self.is_src,
                lambda name=name, key=key: self._fill(
                    comp, name, key, required=True
                ),
                context=f"loading checkpoint tensor {key}",
            )
        _collective_reconcile_tensor_specs(
            comp, tail + tail_bufs, self.group, self.device
        )
        for name in tail + tail_bufs:
            self.group.broadcast(rgetattr(comp, name).data, src=0)
        self._stack.close()
        if self.is_src:
            drop_file_page_cache(self.shard_paths)

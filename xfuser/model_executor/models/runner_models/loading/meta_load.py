"""Memory-efficient load: build components on meta, then fill their real weights without ever
materializing a full copy on host.

Serves two load shapes that never both apply, since one needs weight-splitting parallelism and the
other rules it out:

* Sharded (``fsdp_meta_load``, needs fully_shard_degree > 1): ranks legitimately hold different
  weights, so each fills its own shard.
* Replicated (``replicated_broadcast_load``, needs no weight-splitting parallelism): every rank holds
  the same weights, so rank0 loads once and broadcasts, keeping host peak at 1x the model instead of
  Nx. Nothing is sharded on this path.

Two fill strategies, both collective (every fs-group rank must call in identical order). Which one a
component takes is not decided by what kind of component it is, but by whether its live tensor names
can be mapped onto checkpoint keys:

* Self-fill (``_BlockwiseDiskFiller``), the cheaper path. Each block's real weights are read from disk
  on fs-rank0 alone and broadcast device to device across the group, so no rank ever holds more
  than a block.
  rank0-only read is required because the full block must exist on every rank before block-128 fp8
  quantization (a shard boundary splitting a 128x128 tile invalidates the tile scale, so per-rank slice
  reads are impossible), and if every rank read the full block from disk host anon would scale with N
  ranks (measured +3.5GB per block, enough to trip the cgroup OOM killer). Reading on rank0 then
  broadcasting keeps host disk-read anon at 1x.

  Transformers always qualify: they are built from their own config, so their live names already are
  checkpoint keys. Text encoders qualify when Transformers' renaming rules can be reproduced and
  proven exactly (``text_encoder_adapter.resolve_transformers_manifest``).

* Broadcast fill (``ModelLoader.broadcast_load``), the fallback for text encoders whose keys
  could not be mapped. rank0 loads the whole component via from_pretrained (resolving tied weights),
  then scatters one wrapped block at a time via broadcast_from_rank0, so peers never receive the whole
  model at once even though rank0 held it. fp8-targeted components stream rank0 straight to fp8.

``ModelLoader`` owns the run's contract, quantization plan, backend selection, and route state.
"""

import collections
import gc
import threading
import time
import weakref
from contextlib import contextmanager

import torch

from xfuser.core.distributed.parallel_state import get_fs_group, get_world_group
from xfuser.core.utils.checkpoint_io import (
    host_mem_gb,
    drop_file_page_cache,
    warm_file_page_cache,
    resolve_checkpoint_weight_map,
    component_shard_paths,
)
from xfuser.core.utils.dtype_policy import cast_preserving_fp32_modules
from xfuser.core.utils.runner_utils import log, rgetattr, _use_aiter_fp8_rdna4
from xfuser.envs import _is_cuda
from .checkpoint import CheckpointManifest, CheckpointRequest
from .contracts import (
    LoadDeclaration,
    LoadRoute,
    MaterializationMode,
    UnsupportedLoadContract,
    assert_offload_is_compatible_with_format,
    assert_offload_is_compatible_with_sharding,
    assert_requested_materialization_is_honoured,
    select_effective_materialization_mode,
    select_load_contract,
    select_runtime_quantization,
    validate_materialization_contract,
)
from .backend_selection import QuantizationBackends
from .quantization_ledger import QuantizationLedger
from .quantization_plan import QuantizationPlan


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
        [
            ("parameter", name)
            for name, _ in module.named_parameters(recurse=True, remove_duplicate=False)
        ]
        + [
            ("buffer", name)
            for name, _ in module.named_buffers(recurse=True, remove_duplicate=False)
        ]
    )


def _tensor_layout_contract(module) -> tuple[tuple, ...]:
    """Ordered collective contract including tensor and alias metadata."""

    canonical: dict[int, str] = {}
    entries = []
    for kind, named in (
        (
            "parameter",
            module.named_parameters(recurse=True, remove_duplicate=False),
        ),
        (
            "buffer",
            module.named_buffers(recurse=True, remove_duplicate=False),
        ),
    ):
        for name, tensor in named:
            alias = canonical.setdefault(id(tensor), name)
            persistent = None
            if kind == "buffer":
                parent_name, _, local_name = name.rpartition(".")
                owner = module.get_submodule(parent_name) if parent_name else module
                persistent = local_name not in owner._non_persistent_buffers_set
            entries.append(
                (
                    kind,
                    name,
                    tuple(tensor.shape),
                    tensor.dtype,
                    alias,
                    persistent,
                )
            )
    return tuple(entries)


def _collective_assert_same_layout(
    local_layout, group, device, reference_layout=None
) -> None:
    """Collectively reject any ordered layout mismatch before data broadcasts begin."""
    box = [
        (
            (reference_layout if reference_layout is not None else local_layout)
            if group.rank_in_group == 0
            else None
        )
    ]
    group.broadcast_object_list(box, src=0)
    reference = box[0]
    local_mismatch = int(local_layout != reference)
    mismatch = torch.tensor([local_mismatch], device=device)
    mismatch_count = int(group.all_reduce(mismatch).item())
    if mismatch_count:
        detail = ""
        if local_mismatch:
            first = (
                next(
                    (i, expected, actual)
                    for i, (expected, actual) in enumerate(zip(reference, local_layout))
                    if expected != actual
                )
                if len(reference) == len(local_layout)
                else (
                    min(len(reference), len(local_layout)),
                    reference[min(len(reference), len(local_layout)) :] or "<end>",
                    local_layout[min(len(reference), len(local_layout)) :] or "<end>",
                )
            )
            detail = f"; first local difference at {first[0]}: rank0={first[1]!r}, local={first[2]!r}"
        raise RuntimeError(
            "replicated broadcast-load: ordered parameter/buffer layout mismatch on "
            f"{mismatch_count} of {group.world_size} ranks{detail}"
        )


def _collective_source_call(group, is_src, operation, context, src: int = 0):
    """Run the reading rank's operation and broadcast its failure status before any rank continues."""
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
    group.broadcast_object_list(box, src=src)
    status = box[0]
    if status is not None:
        error_type, message = status
        raise RuntimeError(
            f"{context} failed on rank{src}: {error_type}: {message}"
        ) from source_error
    return result


def _fill_phase_timing_enabled() -> bool:
    from xfuser import envs

    return str(envs.environment_variables["XDIT_FILL_PHASE_TIMING"]()).lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def _warm_shard_depth() -> int:
    """How many shards may be held warm: 0 off, 1 the one being read, 2 also the next one.

    Unparseable means the default rather than off, since a typo in a performance knob should not
    quietly hand back the slow path.
    """
    from xfuser import envs

    raw = str(envs.environment_variables["XDIT_WARM_SHARDS"]()).strip().lower()
    if raw in ("", "false", "no"):
        return 0
    if raw == "true":
        return 2
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


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

    # One all_gather_object rather than a broadcast per rank, which would cost world_size
    # collectives per block -- eight at the rank count this path runs at -- to carry a failure that
    # is almost always absent. Matches _collective_quantize_call on the FSDP path. The per-rank
    # loop is the fallback, so a group without a backing process group still agrees.
    dist = torch.distributed
    device_group = getattr(group, "device_group", None)
    failures: list = [None] * group.world_size
    if device_group is not None and dist.is_available() and dist.is_initialized():
        dist.all_gather_object(failures, local_error, group=device_group)
    else:
        for src in range(group.world_size):
            box = [local_error if group.rank_in_group == src else None]
            group.broadcast_object_list(box, src=src)
            failures[src] = box[0]
    for rank, failure in enumerate(failures):
        if failure is not None:
            error_type, message = failure
            raise RuntimeError(
                f"{context} failed on rank {rank}: {error_type}: {message}"
            )
    return result


def _collective_reconcile_tensor_specs(module, names, group, device, src: int = 0):
    """Make peer tensor storage match the reading rank's shape/dtype before positional broadcasts."""
    spec = (
        [
            (name, tuple(rgetattr(module, name).shape), rgetattr(module, name).dtype)
            for name in names
        ]
        if group.rank_in_group == src
        else None
    )
    box = [spec]
    group.broadcast_object_list(box, src=src)

    def reconcile():
        if group.rank_in_group != src:
            for name, shape, dtype in box[0]:
                tensor = rgetattr(module, name)
                if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    parent_name, _, local_name = name.rpartition(".")
                    owner = module.get_submodule(parent_name) if parent_name else module
                    replacement = torch.empty(shape, dtype=dtype, device=device)
                    if local_name in owner._parameters:
                        owner._parameters[local_name] = torch.nn.Parameter(
                            replacement,
                            requires_grad=tensor.requires_grad,
                        )
                    elif local_name in owner._buffers:
                        owner._buffers[local_name] = replacement
                    else:
                        raise RuntimeError(
                            f"{name} is not a registered parameter or buffer"
                        )

    _collective_build_call(
        group, reconcile, context="transformer tensor-spec reconciliation"
    )


def _collective_reconcile_replicated_tensor_specs(
    module, group, device
) -> tuple[tuple, ...]:
    """Rebuild peer storage and aliases from rank0's authoritative contract."""

    source_contract = (
        _tensor_layout_contract(module) if group.rank_in_group == 0 else None
    )
    box = [source_contract]
    group.broadcast_object_list(box, src=0)
    source_contract = box[0]

    def reconcile():
        if group.rank_in_group == 0:
            return
        local_contract = _tensor_layout_contract(module)
        local_by_name = {entry[1]: entry for entry in local_contract}
        source_alias_groups: dict[str, list[tuple]] = {}
        for entry in source_contract:
            source_alias_groups.setdefault(entry[4], []).append(entry)

        for sources in source_alias_groups.values():
            entries = [local_by_name.get(source[1]) for source in sources]
            if any(entry is None for entry in entries):
                continue
            if any(entry[0] != source[0] for entry, source in zip(entries, sources)):
                continue
            desired = {(source[2], source[3]) for source in sources}
            if len(desired) != 1:
                continue

            shape, dtype = desired.pop()
            tensor = rgetattr(module, entries[0][1])
            replacement = torch.empty(shape, dtype=dtype, device=device)
            registered = (
                torch.nn.Parameter(
                    replacement,
                    requires_grad=tensor.requires_grad,
                )
                if entries[0][0] == "parameter"
                else replacement
            )
            for kind, name, *_ in entries:
                parent_name, _, local_name = name.rpartition(".")
                owner = module.get_submodule(parent_name) if parent_name else module
                registry = owner._parameters if kind == "parameter" else owner._buffers
                registry[local_name] = registered

    _collective_build_call(
        group,
        reconcile,
        context="replicated text-encoder tensor-spec reconciliation",
    )
    return source_contract


def _persistent_named_buffers(module):
    """Named buffers saved by state_dict, using each buffer owner's persistence set."""
    persistent = []
    for name, buffer in module.named_buffers(recurse=True, remove_duplicate=False):
        parent_name, _, local_name = name.rpartition(".")
        owner = module.get_submodule(parent_name) if parent_name else module
        if local_name not in owner._non_persistent_buffers_set:
            persistent.append((name, buffer))
    return persistent


class ModelLoader:
    """Own one run's loading decisions, routes, and memory-efficient fill state.

    Fills FSDP shards from disk on the sharded path, and broadcasts rank0's weights to replicated
    peers on the unsharded one. The resolved declaration and contract, quantization targets,
    backend adapters, checkpoint identity, and route bookkeeping all live on this single object.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.load_declaration = LoadDeclaration.for_runner(
            model.capabilities,
            load_support=model.load_support,
            fsdp_strategy=model.settings.fsdp_strategy,
        )
        self.load_contract = None
        self.quantization_ledger = QuantizationLedger()
        self.quantization_plan = QuantizationPlan(model)
        self.backends = QuantizationBackends(self)
        # Meta components that can fill themselves per block, mapped to where their weights come
        # from: a CheckpointRequest for transformers this loader built, a CheckpointManifest for text
        # encoders whose keys had to be resolved. Keyed by identity rather than by name, so the shard
        # step recognizes the object it was handed instead of guessing from the component's name.
        # Weak so a component the pipeline replaces or drops is not kept alive by the bookkeeping.
        self._blockwise_sources = weakref.WeakKeyDictionary()
        # Resolved on first use and cached: several load-time seams consult it, so this keeps the
        # decision identical everywhere and logs the reason once.
        self._replicated_decision = None
        self._local_blockwise_transformers = weakref.WeakKeyDictionary()
        # Text-encoder fill routes, resolved once (see te_blockwise_routes): the fp8 plan and the
        # load both read them, and must not reach different answers.
        self._te_routes = None

    def checkpoint_request(
        self, subfolder: str | None = None, **kwargs
    ) -> CheckpointRequest:
        """Build the run's checkpoint identity, preserving explicit caller choices."""
        from huggingface_hub.constants import HF_HUB_OFFLINE

        defaults = dict(
            getattr(type(self.model), "checkpoint_request_defaults", {}) or {}
        )
        defaults.setdefault("local_files_only", HF_HUB_OFFLINE)
        defaults.update(kwargs)
        return CheckpointRequest(
            self.model.settings.model_name, subfolder=subfolder, **defaults
        )

    def preflight(self, *, world_size: int) -> None:
        """Resolve and validate all load decisions before model allocation."""
        config = self.model.config
        assert_requested_materialization_is_honoured(config, world_size=world_size)
        assert_offload_is_compatible_with_sharding(config)
        mode = select_effective_materialization_mode(config, world_size=world_size)
        requested_format, backend = select_runtime_quantization(
            config,
            aiter_fp8_active=bool(
                config.use_fp8_gemms and _use_aiter_fp8_rdna4()
            ),
            cuda_active=_is_cuda(),
        )
        assert_offload_is_compatible_with_format(
            config,
            requested_format=requested_format,
            selected_backend=backend,
        )
        self.load_contract = select_load_contract(
            requested_format=requested_format,
            selected_backend=backend,
            materialization_mode=mode,
            declaration=self.load_declaration,
            fsdp_strategy=self.model.settings.fsdp_strategy,
            runner_name=type(self.model).__name__,
        )
        self.backends.preflight()

    def transformer_quantization_adapter(self, component_name: str):
        return self.backends.transformer_adapter(component_name)

    def _validate_mode(self, mode: MaterializationMode) -> None:
        validate_materialization_contract(
            self.load_declaration,
            mode,
            self.model.settings.fsdp_strategy,
            runner_name=type(self.model).__name__,
        )

    def fsdp_meta_load(self) -> bool:
        """True when the memory-efficient sharded (meta-init + per-block rank0-read/broadcast fill)
        load path is on."""
        config = self.model.config
        enabled = (
            select_effective_materialization_mode(
                config, world_size=get_world_group().world_size
            )
            is MaterializationMode.FSDP_META
        )
        if enabled:
            self._validate_mode(MaterializationMode.FSDP_META)
        return enabled

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
        model's explicit load declaration).
        """
        if self._replicated_decision is None:
            self._replicated_decision = self._resolve_replicated_broadcast_load()
        return self._replicated_decision

    def _resolve_replicated_broadcast_load(self) -> bool:
        config = self.model.config
        if not config.memory_efficient_replicated_load:
            return False
        world_size = get_world_group().world_size
        effective_mode = select_effective_materialization_mode(
            config, world_size=world_size
        )
        if effective_mode is not MaterializationMode.REPLICATED_META:
            splits_weights_per_rank = (
                config.fully_shard_degree > 1
                or config.pipefusion_parallel_degree > 1
                or config.tensor_parallel_degree > 1
            )
            if splits_weights_per_rank:
                log(
                    "--memory_efficient_replicated_load ignored: this run splits weights per rank "
                    "(FSDP/PipeFusion/tensor parallel), so peers hold different weights than rank0 and "
                    "a broadcast would overwrite them. Loading per rank."
                )
            elif world_size == 1:
                log(
                    "--memory_efficient_replicated_load ignored: single-rank run has no peer to "
                    "broadcast to. Loading normally."
                )
            return False
        self._validate_mode(effective_mode)
        log(
            "Replicated rank0-broadcast load enabled by --memory_efficient_replicated_load "
            "(host peak 1x the model, not Nx)."
        )
        return True

    def self_fills_from_disk(self, component) -> bool:
        """Whether this meta component can fill per block from disk (see _BlockwiseDiskFiller) rather
        than by broadcasting a rank0 from_pretrained.

        True for transformers built by build_meta_transformer and for text encoders whose checkpoint
        mapping was proven (register_blockwise_fill). Keyed on the component object, not its name: the
        two fill paths need different collectives, so this must not guess.
        """
        return component in self._blockwise_sources

    def plan_eager_blockwise_fallback(self, prepared, targets, wrap_attrs):
        """Return the component-level local fallback decision."""

        from .format_backends import plan_eager_blockwise_fallback

        config = self.model.config
        offload_requested = any(
            getattr(config, name, False)
            for name in (
                "enable_model_cpu_offload",
                "enable_sequential_cpu_offload",
                "enable_group_cpu_offload",
            )
        )
        component_name = prepared.descriptor.component_name
        declaration = self.load_declaration
        return plan_eager_blockwise_fallback(
            prepared=prepared,
            targets=targets,
            wrap_attrs=wrap_attrs,
            world_size=get_world_group().world_size,
            standard_loader=(
                bool(
                    declaration.routes
                    & LoadRoute.LOCAL_BLOCKWISE
                )
                and component_name in declaration.local_meta_transformers
            ),
            offload_requested=offload_requested,
        )

    def mark_local_blockwise(self, component) -> None:
        if component not in self._blockwise_sources:
            raise RuntimeError("local blockwise transformer was not built on meta")
        self._local_blockwise_transformers[component] = True

    def load_transformer(self, wrapper_cls, **kwargs):
        """Route one transformer's weights onto the device (see transformer_load)."""

        from .transformer_load import load_transformer

        return load_transformer(self, wrapper_cls, **kwargs)

    def plan_text_encoders(self, existing_quantization_config=None):
        """Plan each declared text encoder's quantization and fill (see text_encoder_plan)."""

        from .text_encoder_plan import plan_text_encoders

        return plan_text_encoders(self, existing_quantization_config)

    def materialize_pipeline(self) -> None:
        """Place or shard the loaded pipeline according to the current run config."""
        if self.model.config.use_fp4_gemms:
            self.quantization_plan.log_fp8_overrides()
        if self.model.config.fully_shard_degree > 1:
            from .shard import shard_pipeline_components

            shard_pipeline_components(self)
        else:
            from .placement import place_pipeline_components

            place_pipeline_components(self)

    def fill_eager_transformers(self) -> None:
        """Fill all component-level eager blockwise plans before device placement."""

        local_rank = get_world_group().local_rank
        device = f"cuda:{local_rank}"
        for name, component in self.model.pipe.components.items():
            if component not in self._local_blockwise_transformers:
                continue
            strategy = self.model.settings.fsdp_strategy[name]
            self.fill_transformer_local(
                component,
                name,
                strategy,
                device,
            )
            self._local_blockwise_transformers.pop(component, None)
            log(
                f"Blockwise-filled {name} locally. "
                f"host {host_mem_gb()} GB, "
                f"VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB"
            )

    def build_meta_transformer(
        self,
        wrapper_cls,
        request: CheckpointRequest | str = "transformer",
        init_kwargs: dict | None = None,
        *,
        subfolder: str | None = None,
        weight_source: CheckpointManifest | None = None,
    ):
        """Build the (diffusers) transformer wrapper on meta from its config only (no weights).

        Real weights are streamed per block from disk during sharding (see _BlockwiseDiskFiller),
        so the full model never materializes. Uses the diffusers-public from_config; fp8 quantization
        happens per block on the real weights during sharding, so no fp8 swap is done here.

        init_kwargs: extra wrapper __init__ args (e.g. wan's attention_kwargs) not in the on-disk
        config; forwarded to from_config so the meta model matches the from_pretrained path.
        """
        if isinstance(request, str):
            request = self.checkpoint_request(subfolder or request)
        elif subfolder is not None and request.subfolder != subfolder:
            request = request.with_subfolder(subfolder)
        declaration = self.load_declaration
        routes = declaration.routes
        local_custom_load = (
            weight_source is not None
            and bool(routes & LoadRoute.LOCAL_BLOCKWISE)
            and get_world_group().world_size == 1
        )
        if not (
            routes & LoadRoute.STANDARD_COLLECTIVES
        ) and not local_custom_load:
            raise UnsupportedLoadContract(
                f"{type(self.model).__name__} does not declare standard "
                "collective loading"
            )
        component_name = request.subfolder or "transformer"
        if component_name not in declaration.all_meta_transformers:
            raise UnsupportedLoadContract(
                f"{type(self.model).__name__} attempted meta construction of "
                f"'{component_name}' without declaring it in load_declaration"
            )
        strategy = self.model.settings.fsdp_strategy.get(component_name)
        if strategy is None or not strategy.get("wrap_attrs"):
            raise UnsupportedLoadContract(
                f"{type(self.model).__name__} cannot meta-build '{component_name}': "
                "fsdp_strategy must declare non-empty wrap_attrs before construction"
            )

        def build():
            from accelerate import init_empty_weights

            config = wrapper_cls.load_config(
                request.model_name_or_path, **request.config_kwargs()
            )
            with init_empty_weights():
                model = wrapper_cls.from_config(config, **(init_kwargs or {}))
            # Match the checkpoint dtype before disk fill and quantization.
            cast_preserving_fp32_modules(model, torch.bfloat16)
            # from_config leaves nn.Module's training default; from_pretrained
            # ends with eval(), and only that path normally reaches inference.
            return model.eval()

        model = _collective_build_call(
            get_world_group(), build, context=f"meta transformer '{component_name}'"
        )
        self._blockwise_sources[model] = weight_source or request
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
        if component_name not in self.load_declaration.meta_text_encoders:
            raise UnsupportedLoadContract(
                f"{type(self.model).__name__} attempted meta construction of "
                f"'{component_name}' without declaring it in "
                "load_declaration.meta_text_encoders"
            )

        from diffusers import DiffusionPipeline
        from accelerate import init_empty_weights
        from .text_encoder_adapter import resolve_transformers_component

        # Only resolving the component's class and config is guarded: that is where "this component
        # is not rebuildable here" shows up. Construction and the fp8 swap run outside, so a bug in
        # either surfaces instead of degrading every rank to a normal load behind one log line
        # (symmetrically, so agreed_is_meta would not catch it either).
        try:
            request = self.checkpoint_request()
            model_name = request.model_name_or_path
            resolved = resolve_transformers_component(
                DiffusionPipeline,
                component_name,
                request,
            )
            if resolved is None:
                return None
            cls, config = resolved
        except (OSError, ImportError, AttributeError) as e:
            log(
                f"Meta-init of component '{component_name}' failed "
                f"({type(e).__name__}: {e}); using normal load."
            )
            return None

        with init_empty_weights():
            component = cls._from_config(config)
        # _from_config defaults to fp32 and to training mode; align meta params to bf16 (dtype-only
        # .to is legal on meta) so their DTensor dtype matches the broadcast source, and match
        # from_pretrained's eval().
        component = cast_preserving_fp32_modules(component, torch.bfloat16).eval()
        if fp8:
            self.apply_meta_te_fp8(component, component_name)
        return component

    def _meta_te_fp8_targets(self, component_name: str) -> tuple:
        """This component's fp8-streamed target paths, component-relative."""
        streamed_targets = self.quantization_ledger.fp8_streaming_targets
        prefix = f"{component_name}."
        return tuple(
            "" if target == component_name else target[len(prefix) :]
            for target in streamed_targets
            if target == component_name or target.startswith(prefix)
        )

    def apply_meta_te_fp8(self, component, component_name: str) -> bool:
        """Swap this component's targeted meta Linears to fp8, reporting whether anything changed.

        Separate from build_meta_component so the swap can be deferred: the blockwise fill needs a
        bf16 layout to match the checkpoint, and only falls back to this fp8 layout if the checkpoint
        mapping is refused (see meta_te_kwargs).
        """
        targets = self._meta_te_fp8_targets(component_name)
        if not targets:
            return False
        self._swap_meta_te_to_fp8(component, targets)
        return True

    def _te_component_names(self) -> list[str]:
        """Text encoders explicitly opted into shared meta loading."""
        return list(self.load_declaration.meta_text_encoders)

    def meta_te_kwargs(self):
        """Build text-encoder(s) on meta for the pipeline's from_pretrained (meta FSDP load path).

        Each is built bf16 and offered to the blockwise disk fill, which is the path worth being on:
        one block of the encoder is real at a time, on the rank that reads it, and fp8 quantization
        happens per block on the way through. The alternative below it costs rank 0 a full host copy
        of the encoder before any of it is scattered.

        Taking that path needs the encoder's live tensor names mapped onto its checkpoint keys, which
        is only sometimes provable (see resolve_transformers_manifest). So the layout follows the
        mapping rather than the reverse: bf16 while the mapping holds, and if it is refused, the fp8
        swap is applied after the fact and the component falls back to broadcast_load. Building bf16
        first costs nothing to undo, since meta parameters have no storage.

        The fp8 swap has to be the fallback and not the default. It replaces each targeted Linear's
        weight with weight_fp8 + weight_scale, names no bf16 checkpoint has, so a component built fp8
        can never be mapped -- which would have left the blockwise fill reachable only on bf16 runs,
        and unreachable on the small-GPU fp8 runs that need it most.

        Returns (pipe_component_kwargs, None): the kwargs carry meta modules so the pipeline skips
        loading those components, and te_quant is None (the pipe streams nothing; the meta module is
        filled by whichever path was chosen). Returns None when there are no TE components or any one
        of them cannot be meta-built, leaving the caller to take its normal load.
        """
        if not self._te_component_names():
            return None
        routes = self.te_blockwise_routes()
        if any(component is None for component, _, _ in routes.values()):
            return None
        kwargs = {}
        for name, (component, manifest, refusal) in routes.items():
            kwargs[name] = component
            if refusal is None:
                self._blockwise_sources[component] = manifest
                log(f"Text encoder '{name}' will be filled blockwise from disk.")
                continue
            # Only now, after the fp8 plan has run and recorded its streaming targets, can the
            # fallback layout be built: the swap needs to know what rank 0 will send.
            self.apply_meta_te_fp8(component, name)
            log(
                f"Text encoder '{name}' cannot be filled blockwise from disk ({refusal}); "
                f"rank 0 will load it whole and broadcast."
            )
        return kwargs, None

    def te_blockwise_routes(self) -> dict:
        """Per text encoder: its meta component, its checkpoint mapping, and why it was refused.

        Computed once and cached, because two decisions depend on this answer and must not disagree.
        The fp8 plan needs it to know how the component gets quantized (per block during the fill, or
        by a config the pipeline streams), and the load needs it to know which collective fills the
        component. Deciding twice would risk planning for one path and loading down the other.

        Each component is built bf16 on meta here and handed back for use, rather than rebuilt later:
        the mapping is resolved against this object's tensor names, so the object that was judged has
        to be the object that gets filled.

        Refusals are unanimous across the world group (see _agreed_refusal).
        """
        if self._te_routes is not None:
            return self._te_routes
        from .text_encoder_adapter import resolve_transformers_manifest

        routes = {}
        for name in self._te_component_names():
            component = _collective_build_call(
                get_world_group(),
                lambda name=name: self.build_meta_component(name, fp8=False),
                context=f"text-encoder meta construction '{name}'",
            )
            if component is None:
                # Not rebuildable from config at all; the caller falls back to a normal load, and
                # agreed_is_meta catches any rank that disagreed.
                routes[name] = (None, None, "component is not rebuildable from config")
                continue
            strategy = self.model.settings.fsdp_strategy.get(name) or {}
            if not strategy.get("wrap_attrs"):
                manifest = None
                refusal = "no wrap_attrs declared, so it has no blocks to fill one at a time"
            else:
                manifest, refusal = resolve_transformers_manifest(
                    component, self.checkpoint_request(name)
                )
            routes[name] = (
                component,
                manifest,
                self._agreed_refusal(refusal, name),
            )
        self._te_routes = routes
        return routes

    def will_fill_blockwise(self, component_name: str) -> bool:
        """Whether this text encoder's weights will be read per block from disk.

        Asked by the fp8 plan before the component is loaded: a blockwise-filled component is
        quantized per block on the way through, so it needs neither a streaming config nor a
        post-load conversion walk.
        """
        route = self.te_blockwise_routes().get(component_name)
        return route is not None and route[2] is None

    def _agreed_refusal(self, refusal: str | None, component_name: str) -> str | None:
        """The refusal every rank will act on: any rank refusing makes all of them refuse.

        Unanimity is the only safe resolution, since the two paths run different collectives. Falling
        back together is always available, while proceeding together is not, so one rank's refusal
        becomes everyone's.
        """
        world = get_world_group()
        if world is None or world.world_size <= 1:
            return refusal
        local = 0 if refusal is None else 1
        n_refused = int(
            world.all_reduce(
                torch.tensor([local], device=f"cuda:{world.local_rank}")
            ).item()
        )
        if n_refused == 0:
            return None
        if refusal is not None:
            return refusal
        return (
            f"{n_refused} of {world.world_size} ranks could not map '{component_name}' onto its "
            f"checkpoint (this rank could); falling back together to stay collective-safe"
        )

    def meta_te_kwargs_replicated(self, te_quant_config=None):
        """Text-encoder kwargs for the replicated broadcast-load path (fits-in-GPU, multi-GPU).

        Rank 0 loads TEs real via the pipeline's from_pretrained, using native FP8 streaming when
        available. Otherwise it loads bf16 and peers build the matching bf16 meta layout; the normal
        post-load walk converts each replica after broadcast.

        A peer that cannot meta-build a component raises: the per-tensor broadcast walks
        named_parameters/buffers in lockstep, so a real bf16 fallback (no fp8 weight_scale buffers)
        against rank0's fp8 source would diverge the tensor count and desync the collective.
        """
        world = get_world_group()

        def build():
            if _is_bcast_src(world):
                return {}, te_quant_config
            kwargs = {}
            for name in self._te_component_names():
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
        broadcasts it over the world group, then every rank fp8-quantizes that block (symmetric,
        since the same bf16 gives the same fp8) before the next block. Peak = accumulating fp8 + one
        transient bf16 block, so it fits a single GPU where the full bf16 model would not (~24 vs
        ~12 GB). No fully_shard: replicated keeps the full quantized block on every rank. Reuses the
        FSDP disk filler (_BlockwiseDiskFiller) and per-block quantize_fn (``shard`` module).

        Text encoders: rank0 loaded them real via the pipeline and peers built a matching-layout
        meta component. Materialize peer meta to real-empty on device, then broadcast every
        param/buffer from rank0. Native-streamed FP8 layouts need no later conversion; bf16 fallback
        layouts are converted by the normal post-load walk.
        """
        from diffusers.models.model_loading_utils import set_module_tensor_to_device

        world = get_world_group()
        device = f"cuda:{world.local_rank}"
        strategy = self.model.settings.fsdp_strategy
        for name in strategy:
            component = getattr(self.model.pipe, name, None)
            if component is None or not hasattr(component, "named_parameters"):
                continue
            if self._is_meta_denoiser(name):
                if self._all_ranks_loaded_real(component, world, device):
                    # Unwired runner: loaded real on every rank (e.g. a composition-wrapper pipeline
                    # whose _load_model built the whole pipeline real). _fill_transformer_replicated's
                    # to_empty() would wipe those weights; skip the destructive fill and keep the real
                    # all-rank-symmetric weights (the post-load fp8 walk still quantizes them).
                    log(
                        f"{name} loaded real on all ranks (unwired for replicated meta load); "
                        f"skipping broadcast fill, keeping real weights."
                    )
                    continue
                self._fill_transformer_replicated(
                    component, name, strategy[name], device, world
                )
            else:
                if name not in self.load_declaration.meta_text_encoders:
                    continue
                if self._all_ranks_loaded_real(component, world, device):
                    # Same unwired-runner case: the encoder loaded real on every rank, so skipping
                    # avoids broadcasting rank0's bytes over each peer's already-correct weights.
                    log(
                        f"{name} loaded real on all ranks (unwired for replicated meta load); "
                        f"skipping broadcast fill, keeping real weights."
                    )
                    continue
                self._fill_te_replicated(
                    component, device, world, set_module_tensor_to_device
                )
            torch.cuda.empty_cache()
            log(
                f"Broadcast-filled {name} from rank0 (replicated). "
                f"host {host_mem_gb()} GB, VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB"
            )

    def _is_meta_denoiser(self, name: str) -> bool:
        """Whether this component is filled per block from disk or broadcast whole.

        Asked of the runner's declaration first and of the name only as a fallback,
        because a name is not a reliable answer: Ideogram 4's second denoiser is
        called `unconditional_transformer`, which no prefix of `transformer` matches,
        so it took the text-encoder branch and was broadcast from a rank0 copy that
        was itself still on meta. That produced a black image rather than an error.
        """

        if name in self.load_declaration.all_meta_transformers:
            return True
        return name == "transformer" or name.startswith("transformer_")

    def _all_ranks_loaded_real(self, component, world, device) -> bool:
        """True only if EVERY rank has this component fully real (no meta params).

        A per-rank `any(is_meta)` check diverges in the replicated path: rank0 loads the TEs real
        (fp8-streamed) while peers build them on meta, so rank0 would skip the fill's collectives
        while peers enter them -> the count-guard broadcast reads garbage and aborts (or hangs).
        All-reduce the local real flag so the skip decision is identical on every rank.

        Deliberately tolerant of disagreement, unlike ``agreed_is_meta``: rank0-real/peer-meta is
        the normal shape of the replicated path, and it means "fall through to the fill".
        """
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

    def _fill_transformer_replicated(
        self, component, name, strategy, device, world
    ) -> None:
        """Per-block rank0-disk-read + world broadcast + symmetric per-block fp8 quantize (no shard)."""
        from .shard import build_block_quantize_fn

        quantize_fn = build_block_quantize_fn(
            self,
            name,
            strategy.get("wrap_attrs", []),
            world.local_rank,
            component=component,
        )
        self._fill_blocks(
            component,
            name,
            strategy,
            device,
            group=world,
            quantize_fn=quantize_fn,
        )

    def fill_transformer_local(
        self,
        component,
        name,
        strategy,
        device,
        *,
        quantize_fn=None,
    ) -> None:
        """Fill and quantize one eager meta transformer without collectives."""

        if quantize_fn is None:
            from .shard import build_block_quantize_fn

            quantize_fn = build_block_quantize_fn(
                self,
                name,
                strategy.get("wrap_attrs", []),
                get_world_group().local_rank,
                component=component,
            )
        self._fill_blocks(
            component,
            name,
            strategy,
            device,
            group=None,
            quantize_fn=quantize_fn,
        )

    def _fill_blocks(
        self,
        component,
        name,
        strategy,
        device,
        *,
        group,
        quantize_fn,
    ) -> None:
        """Materialize, fill, and quantize blocks through one transport."""

        wrap_attrs = strategy.get("wrap_attrs", [])
        fill_block, finalize = self.build_blockwise_disk_loaders(
            component,
            wrap_attrs,
            name,
            device,
            group=group,
            collective=group is not None,
        )
        from xfuser.core.distributed.sharding import (
            _restore_nonpersistent_buffers,
            _save_nonpersistent_buffers,
        )

        wrapped = []
        for attr in wrap_attrs:
            wrapped.extend(rgetattr(component, attr))
        for i, block in enumerate(wrapped):
            nonpersistent_buffers = _save_nonpersistent_buffers(block, device)
            block.to_empty(device=device, recurse=True)
            _restore_nonpersistent_buffers(nonpersistent_buffers)
            fill_block(block, i)
            if quantize_fn is not None:
                if group is None:
                    quantize_fn(block, i)
                else:
                    _collective_build_call(
                        group,
                        lambda: quantize_fn(block, i),
                        context=f"quantizing replicated transformer block {i}",
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finalize(component)

    def _fill_te_replicated(
        self, component, device, world, set_module_tensor_to_device
    ) -> None:
        """Materialize peer meta to real-empty on device, then broadcast every param and buffer from
        world-rank0. Layouts already match (rank0 fp8-streamed, peers meta fp8-swapped), so nothing
        is re-quantized.

        Two hazards follow from rank0 building the encoder real through the pipeline's
        from_pretrained while peers build it on meta through _from_config:

        1. Divergent dtypes. A meta build instantiates at the pipeline's compute dtype, so
           auto-computed buffers can differ from the real build: Qwen3's rotary inv_freq is fp32
           when built real and bf16 on meta. Since broadcast copies rank0's element bytes into the
           peer tensor's existing storage, a bf16 slot receiving fp32 bytes is silently corrupted,
           which reached inference as garbage rotary frequencies and a black render. It survived
           ulysses<=2 and broke at 4, where the sequence chunk amplifies the corrupt positions.
           Rank0's (shape, dtype) per name therefore goes first, and any mismatched peer tensor is
           reallocated before the data broadcast.

        2. Divergent layout. Peer placeholders are reconciled to rank0's tensor specs, then ordered
           names, kinds, aliases, persistence and final shape/dtype are compared collectively. That
           permits legitimate real-versus-meta storage differences while still rejecting structural
           divergence before rank0 enters a data broadcast.

        The name lists are captured once and drive the materialize pass through rgetattr, because
        set_module_tensor_to_device replaces param objects: re-enumerating named_parameters() for
        the broadcast can hand back a different, still-CPU object than the one the pass moved, and
        broadcasting a CPU tensor aborts the NCCL group.

        remove_duplicate=False exposes every tied name. Materialization preserves each rank's alias
        groups, and the final contract rejects any rank whose tie structure differs from rank0.
        """
        source_contract = _collective_reconcile_replicated_tensor_specs(
            component, world, device
        )
        layout = _tensor_layout(component)
        param_names = [name for kind, name in layout if kind == "parameter"]
        buffer_names = [name for kind, name in layout if kind == "buffer"]
        local_contract = _tensor_layout_contract(component)
        alias_groups: dict[str, list[tuple]] = {}
        for entry in local_contract:
            alias_groups.setdefault(entry[4], []).append(entry)

        def materialize():
            for entries in alias_groups.values():
                name = entries[0][1]
                t = rgetattr(component, name)
                if t.is_meta:
                    set_module_tensor_to_device(
                        component,
                        name,
                        device,
                        value=torch.empty(t.shape, dtype=t.dtype, device=device),
                    )
                elif t.device.type != "cuda":
                    set_module_tensor_to_device(
                        component, name, device, value=t.to(device)
                    )
                materialized = rgetattr(component, name)
                for kind, alias_name, *_ in entries[1:]:
                    parent_name, _, local_name = alias_name.rpartition(".")
                    owner = (
                        component.get_submodule(parent_name)
                        if parent_name
                        else component
                    )
                    registry = (
                        owner._parameters if kind == "parameter" else owner._buffers
                    )
                    registry[local_name] = materialized

        _collective_build_call(
            world,
            materialize,
            context="replicated text-encoder materialization",
        )
        _collective_assert_same_layout(
            _tensor_layout_contract(component),
            world,
            device,
            reference_layout=source_contract,
        )
        ordered = param_names + buffer_names
        # Peers must receive in place: .contiguous() on a peer would return a copy and the broadcast
        # would fill that temporary, leaving the real tensor unwritten. Check every destination
        # before the loop and agree the result, because raising from inside the loop would abort the
        # peer while rank0 blocks in broadcast, trading a silent corruption for a silent hang.
        strided = [
            name
            for name in ordered
            if world.rank_in_group != 0
            and not rgetattr(component, name).data.is_contiguous()
        ]
        n_bad = int(
            world.all_reduce(torch.tensor([len(strided)], device=device)).item()
        )
        if n_bad:
            raise RuntimeError(
                f"replicated broadcast-load: {n_bad} peer destination(s) are not contiguous and "
                f"cannot receive the broadcast in place"
                + (
                    f"; on this rank: {strided[:3]}"
                    if strided
                    else " (offenders on other ranks)"
                )
            )
        for name in ordered:
            tensor = rgetattr(component, name).data
            if world.rank_in_group == 0:
                # rank0's tensors come from from_pretrained and may be strided views, whose element
                # bytes are not the buffer bytes; send a contiguous copy instead.
                tensor = tensor.contiguous()
            world.broadcast(tensor, src=0)

    def build_blockwise_disk_loaders(
        self,
        component,
        wrap_attrs,
        subfolder,
        device,
        group=None,
        *,
        collective=True,
    ):
        """(load_block_fn, load_epilogue_fn) filling a meta component from disk (rank0-read + bcast).

        Works for any component registered as self-filling, transformer or text encoder; the recorded
        source is what tells the filler how live names reach checkpoint keys.

        group: broadcast group (default get_fs_group() for the FSDP path). The replicated path passes
        get_world_group() — get_fs_group() has world_size 1 when fully_shard_degree==1, so its
        broadcast would be a no-op and peers would receive garbage."""
        source = self._blockwise_sources.get(component)
        if source is None:
            raise UnsupportedLoadContract(
                f"no checkpoint source recorded for meta component '{subfolder}'; "
                "refusing to reconstruct checkpoint identity before disk fill"
            )
        filler_kwargs = {} if collective else {"collective": False}
        filler = _BlockwiseDiskFiller(
            self.model,
            component,
            wrap_attrs,
            source,
            device,
            group,
            **filler_kwargs,
        )
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
            xFuserFP8BlockScaleLinear,
            _fp8_dtype,
        )

        for t in targets:
            _swap_linears_to_fp8(
                module.get_submodule(t), preshuffle=False, add_scale_buffer=True
            )
        fp8 = _fp8_dtype()
        for m in module.modules():
            if (
                isinstance(m, xFuserFP8BlockScaleLinear)
                and m.weight is not None
                and m.weight.is_meta
            ):
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
        request = self.checkpoint_request(component_name)
        quantization_config = getattr(
            self.model, "_text_encoder_quantization_configs", {}
        ).get(component_name)
        from .text_encoder_adapter import load_transformers_component

        return load_transformers_component(
            type(component),
            request,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

    def _broadcast_load_component(
        self, component, component_name: str, offload: bool
    ) -> None:
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict,
            StateDictOptions,
        )

        wrap_attrs = self.model.settings.fsdp_strategy[component_name].get(
            "wrap_attrs", []
        )
        group = get_fs_group()
        is_src = _is_bcast_src(group)
        full_sd: dict = {}
        src = None

        def load_source_state_dict():
            nonlocal src
            src = self._load_rank0_source(component, component_name)
            return src.state_dict()

        try:
            full_sd = (
                _collective_source_call(
                    group,
                    is_src,
                    load_source_state_dict,
                    context=f"loading {component_name} source state dict",
                )
                or {}
            )

            # broadcast_from_rank0 scatters rank0's full tensors into each rank's DTensor shard; a
            # partial dict + strict=False lets us scatter one module (block/tail) at a time so peers
            # never receive the whole model at once. cpu_offload places filled params on CPU to
            # satisfy CPUOffloadPolicy (broadcast_from_rank0 otherwise defaults them to cuda).
            opts = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=False,
                cpu_offload=offload,
            )
            block_prefixes = tuple(f"{a}." for a in wrap_attrs)

            for attr in wrap_attrs:
                prefix = f"{attr}."
                for idx, block in enumerate(rgetattr(component, attr)):
                    bp = f"{prefix}{idx}."
                    # from_pretrained has full paths; block.state_dict uses block-relative keys.
                    block_sd = (
                        {
                            k[len(bp) :]: v
                            for k, v in full_sd.items()
                            if k.startswith(bp)
                        }
                        if is_src
                        else {}
                    )
                    set_model_state_dict(block, block_sd, options=opts)

            # Non-block params/buffers: embeddings, norms, lm_head.
            tail_sd = (
                {k: v for k, v in full_sd.items() if not k.startswith(block_prefixes)}
                if is_src
                else {}
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
                    self.checkpoint_request(component_name),
                    basename=basename,
                )
            drop_file_page_cache(paths)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _BlockwiseDiskFiller:
    """Fills a meta component's blocks with real weights, reading on fs-rank0 and broadcasting
    each tensor GPU->GPU to the fs group. Holds the checkpoint weight_map and fs group across the
    per-block fill and the epilogue; at most one shard stays mapped, since an open handle retains
    every tensor read through it (see _handle). See module docstring for why the read is rank0-only
    (block-128 fp8 tile constraint + host-anon N-scaling).

    Nothing here is specific to a transformer. It needs a component whose repeated blocks are named
    by wrap_attrs and a source that maps live tensor names to checkpoint keys, which is why the same
    class fills diffusers transformers (whose live names are already checkpoint keys, via a
    CheckpointRequest) and transformers text encoders (whose names need the renaming a
    CheckpointManifest carries; see text_encoder_adapter.resolve_transformers_manifest).
    """

    def __init__(
        self,
        model,
        component,
        wrap_attrs,
        source: CheckpointRequest | CheckpointManifest,
        device,
        group=None,
        *,
        collective=True,
    ) -> None:
        from contextlib import ExitStack

        self.model = model
        self.source = source
        self.subfolder = (
            source.label or "mapped checkpoint"
            if isinstance(source, CheckpointManifest)
            else source.subfolder or "transformer"
        )
        self.device = device
        self.group = (group or get_fs_group()) if collective else None
        self.is_src = self.group is None or _is_bcast_src(self.group)
        # rank0 resolves the checkpoint map and hands it to every rank, rather than each rank
        # resolving its own: resolving once avoids redundant hub revalidation HEADs, and
        # broadcasting the result keeps that while letting any rank read a block.
        manifest = (
            source
            if isinstance(source, CheckpointManifest)
            else CheckpointManifest(
                self._share_from_source(
                    lambda: resolve_checkpoint_weight_map(source),
                    context=f"resolving checkpoint map for {self.subfolder}",
                )
                or {}
            )
        )
        self.weight_map = manifest.weight_map
        self.checkpoint_keys = manifest.checkpoint_keys
        self.derived = manifest.derived
        self.strict = manifest.strict
        self._used_keys = set()
        self.shard_paths = set(self.weight_map.values())
        # Which keys each shard still owes, so a shard's page cache is dropped when the fill is
        # globally done with it rather than when one rank happens to move on. Every rank sees every
        # block's key list, so all of them reach zero for a shard at the same point without talking.
        self._unread_by_shard: dict[str, set[str]] = collections.defaultdict(set)
        for key, path in self.weight_map.items():
            self._unread_by_shard[path].add(key)
        # Which shard follows which, so one can be streamed in while the previous one is consumed.
        # The checkpoint index lists keys in shard order, so first appearance is consumption order;
        # being wrong here costs a wasted prefetch and not a wrong answer, since _handle still warms
        # whatever it is actually handed.
        order = list(dict.fromkeys(self.weight_map.values()))
        self._next_shard = dict(zip(order, order[1:]))
        self._streamed: set[str] = set()
        self._prefetch_thread = None
        # Wall seconds per fill phase, reported once in finalize. Without a split, a slow fill gives
        # no clue whether to attack the reading, the transport or the collective bookkeeping.
        self._phase_seconds: dict[str, float] = collections.defaultdict(float)
        self._handle_cache: dict[str, object] = {}
        self._stack = ExitStack()
        self._block_prefixes = tuple(f"{a}." for a in wrap_attrs)
        self._id2fqn: dict[int, str] = {}
        for attr in wrap_attrs:
            for idx, mod in enumerate(rgetattr(component, attr)):
                self._id2fqn[id(mod)] = f"{attr}.{idx}"

    @contextmanager
    def _timed(self, phase: str):
        """Charge wall time to a fill phase, under XDIT_FILL_PHASE_TIMING.

        Opt-in because a truthful breakdown and a fast fill are in conflict here. Reads and
        broadcasts queue asynchronous device work, so timing without a synchronise measures only
        submissions: the broadcast read as 0.0s and the reads silently absorbed it, which is worse
        than no breakdown because it points at the wrong phase. Synchronising fixes the attribution
        but serialises a fill that otherwise overlaps its reads, broadcasts and sharding, and that
        measured 2.3x slower end to end. So the default is neither: no timing and no stalls.
        """
        if not _fill_phase_timing_enabled():
            yield
            return
        started = time.monotonic()
        try:
            yield
        finally:
            device = torch.device(getattr(self, "device", "cpu"))
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            # Lazily, because timing must never be the reason a fill fails: callers build this class
            # attribute-by-attribute in places, and a phase tally is not worth an AttributeError.
            phases = getattr(self, "_phase_seconds", None)
            if phases is None:
                phases = self._phase_seconds = collections.defaultdict(float)
            phases[phase] += time.monotonic() - started

    def _is_reader(self, src: int) -> bool:
        """Whether this rank is the one reading for `src`.

        Falls back to the fixed is_src for a group that exposes no rank, which keeps a caller that
        only ever reads on rank0 working without having to describe a rank it does not have.
        """
        if self.group is None:
            return True
        rank = getattr(self.group, "rank_in_group", None)
        if rank is None:
            return getattr(self, "is_src", True)
        return rank == src

    def _source_call(self, fn, *, context, src: int = 0):
        if self.group is None:
            return fn()
        return _collective_source_call(
            self.group,
            self._is_reader(src),
            fn,
            context=context,
            src=src,
        )

    def _share_from_source(self, fn, *, context):
        """Run fn on rank0 and give every rank the result.

        Distinct from _source_call, which discards the peers' None: the checkpoint map has to exist on
        every rank now that any rank may be the one to read a block.
        """
        result = self._source_call(fn, context=context)
        if self.group is None:
            return result
        box = [result if self._is_reader(0) else None]
        self.group.broadcast_object_list(box, src=0)
        return box[0]

    def _reader_for_block(self, index: int) -> int:
        """Which rank reads block `index`. Always rank 0; the hook exists to keep that decision named.

        Rotating this across ranks looks like free parallelism and is not. Measured cold on 8 ranks it
        cost 37.2s against 32.7s for a single reader: block-strided rotation puts every rank at a
        different offset in the same shard, which defeats readahead, and it grew resident page cache
        from ~10GB to 35.6GB because every rank ends up mapping every shard.

        The fill is bounded by mmap fault latency rather than by having one reader. A single reader
        sees ~0.6 GB/s through mmap where the same file streams at ~3.2 GB/s via pread and the device
        tops out near 6 GB/s, so the headroom is in how the bytes are fetched, not in who fetches them.
        """
        return 0

    def _assert_same_layout(self, module):
        if self.group is not None:
            _collective_assert_same_layout(
                _tensor_layout_contract(module),
                self.group,
                getattr(self, "device", "cpu"),
            )

    def _reconcile_tensor_specs(self, module, names, src: int = 0):
        if self.group is not None:
            _collective_reconcile_tensor_specs(
                module,
                names,
                self.group,
                getattr(self, "device", "cpu"),
                src=src,
            )

    def _read_device(self) -> str:
        """Where safe_open should place the tensors it hands back.

        Reading to the fill's own device rather than to host is both cheaper and smaller: a handle
        opened on cpu retains a host copy of every tensor read through it (measured 1:1 with the
        bytes read, released only on close), while one opened on the accelerator retains nothing and
        skips the host staging copy entirely. On a FLUX.2 shard that is 9.9 GB read for +0.6 GB of
        host anon in 2.4s, against +10.2 GB in 9.0s through host.

        Falls back to host when the fill targets CPU (offload), where _release_handles bounds what
        the retention can reach.
        """
        device = torch.device(self.device)
        return "cpu" if device.type == "cpu" else str(device)

    def _handle(self, path):
        """Open `path`, keeping at most one shard mapped at a time.

        Only the host-read fallback actually accumulates (see _read_device), but one shard is the
        right bound either way, and releasing on shard change rather than per block keeps the
        teardown count at one per shard: blocks are laid out in key order.

        Streaming the shard before mapping it is what makes the read fast rather than fault-bound
        (see warm_file_page_cache). It belongs here because this is the one place that knows a shard
        is about to be read in full, and being per-shard is what bounds it: the previous shard's
        handle is already closed and its pages already retired.
        """
        from safetensors import safe_open

        h = self._handle_cache.get(path)
        if h is None:
            self._release_handles()
            depth = _warm_shard_depth()
            if depth:
                # Charged as its own phase, but it happens inside a "read" span and so is counted in
                # that total too: it is a component of the read, not a sibling of it.
                with self._timed("warm"):
                    self._await_or_warm(path)
            h = self._stack.enter_context(
                safe_open(path, framework="pt", device=self._read_device())
            )
            self._handle_cache[path] = h
            if depth > 1:
                self._prefetch_after(path)
        return h

    def _await_or_warm(self, path) -> None:
        """Make `path` warm, unless it already is because the previous shard streamed it ahead.

        A prefetch in flight is deliberately not waited on. Its pread stays ahead of the mmap that
        follows it, so letting the read start immediately costs nothing, where stalling for the tail
        of a stream we are about to consume anyway only adds latency.

        Tracked as a set rather than "which shard is in flight" because a shard outlives its
        prefetch: the tail fill reopens shards the block walk already closed, and asking whether a
        stream is running would restream those in full.
        """
        streamed = self._streamed_shards()
        if path in streamed:
            return
        warm_file_page_cache(path)
        streamed.add(path)

    def _prefetch_after(self, path) -> None:
        """Stream the next shard while this one is being consumed.

        Warming is the one part of the read still serialised against it: the shard has to be in cache
        before the mmap walks it, so the fill pays that stream up front with the device idle. The work
        after it -- quantise, broadcast, shard -- needs no disk, which is what the next shard's stream
        can hide under.

        Exactly one shard is ever in flight, so this holds two shards in page cache rather than the
        whole checkpoint. Shards with nothing left unread are skipped: they have already been retired
        by _retire_keys and re-reading one would put back the cache that drop exists to release.
        """
        nxt = getattr(self, "_next_shard", {}).get(path)
        streamed = self._streamed_shards()
        if nxt is None or nxt in streamed or not self._unread_by_shard.get(nxt):
            return
        self._join_prefetch()
        streamed.add(nxt)
        self._prefetch_thread = threading.Thread(
            target=warm_file_page_cache, args=(nxt,), daemon=True
        )
        self._prefetch_thread.start()

    def _streamed_shards(self) -> set:
        """Shards believed to be in page cache.

        Lazily, for the same reason the phase tally is: this class gets built attribute-by-attribute
        in places and a prefetch is never worth an AttributeError.
        """
        streamed = getattr(self, "_streamed", None)
        if streamed is None:
            streamed = self._streamed = set()
        return streamed

    def _join_prefetch(self) -> None:
        """Wait for any in-flight prefetch, so a stream never outlives the fill that wanted it.

        A thread left running past teardown would repopulate cache that finalize just dropped.
        """
        thread = getattr(self, "_prefetch_thread", None)
        self._prefetch_thread = None
        if thread is not None:
            thread.join()

    def _release_handles(self):
        """Close any open shard handle, freeing the tensor copies it retains.

        A tensor already read stays valid: get_tensor hands back an owned copy, not a view into the
        mapping, so weights already assigned to the module survive the close.

        Dropping the closed shard's page cache here rather than only at the end of the component is
        what keeps the cache near one shard instead of the whole checkpoint: blocks are read in key
        order with one shard open at a time, so a shard being closed means the fill is done with it.
        If a later block did reopen it, the cost is a re-read and not a wrong answer.
        """
        self._handle_cache.clear()
        self._stack.close()

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
                raise RuntimeError(
                    f"missing checkpoint weight for {key} in {self.subfolder}"
                )
            return
        set_module_tensor_to_device(
            module,
            local_name,
            self.device,
            value=self._tensor_for(key, path),
        )
        self._used_keys.add(key)

    def _tensor_for(self, key: str, path: str):
        """This live tensor, read from the shard or built from tensors in it."""

        handle = self._handle(path)
        derived = self.derived.get(key)
        if derived is None:
            return handle.get_tensor(self.checkpoint_keys.get(key, key))
        return derived.build(*(handle.get_tensor(name) for name in derived.sources))

    def _require_checkpoint_keys(self, keys, src: int = 0):
        """Collectively reject missing persistent tensors before any rank enters data broadcast."""
        missing = (
            [key for key in keys if key not in self.weight_map]
            if self._is_reader(src)
            else None
        )
        if self.group is not None:
            box = [missing]
            self.group.broadcast_object_list(box, src=src)
            missing = box[0]
        if missing:
            preview = ", ".join(missing[:3])
            suffix = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            raise RuntimeError(
                f"missing checkpoint tensors in {self.subfolder}: {preview}{suffix}"
            )

    def _broadcast(self, module, src: int = 0):
        if self.group is None:
            return
        # Collective: all group ranks must call in the same order. Module structure is identical
        # across ranks (meta -> to_empty), so named_* iteration order matches. remove_duplicate=False
        # so tied weights emit the same name count on every rank regardless of per-rank tie state.
        self._broadcast_tensors(
            [
                p.data
                for _, p in module.named_parameters(
                    recurse=True, remove_duplicate=False
                )
            ]
            + [b.data for _, b in _persistent_named_buffers(module)],
            src=src,
        )

    def _broadcast_tensors(self, tensors, src: int = 0) -> None:
        """Broadcast a module's tensors from rank0 as one batch of collectives.

        One broadcast per tensor is fifteen launches per block on Z-Image, each paying its own
        latency for a few hundred megabytes. _coalescing_manager submits them as a group, which is
        preferred over flattening into one contiguous buffer precisely because flattening would cost
        an extra block-sized allocation on every rank during a fill whose whole point is to hold at
        most one block.

        Falls back to the per-tensor loop when the manager is unavailable or the group exposes no
        process group to coalesce on, so this stays an optimisation rather than a requirement.
        """
        if not tensors:
            return
        device_group = getattr(self.group, "device_group", None)
        manager = getattr(torch.distributed, "_coalescing_manager", None)
        if manager is None or device_group is None:
            for tensor in tensors:
                self.group.broadcast(tensor, src=src)
            return
        global_src = torch.distributed.get_global_rank(device_group, src)
        with manager(group=device_group, device=torch.device(self.device)):
            for tensor in tensors:
                torch.distributed.broadcast(tensor, src=global_src, group=device_group)

    def _retire_keys(self, keys) -> None:
        """Drop a shard's page cache once nothing still needs it.

        Tied to the keys rather than to a handle closing, because a shard's keys outlive any one
        handle: the tail fill reaches back for keys in shards the block walk already closed, and
        dropping on close evicted pages that were about to be read again. Keying on "every key in
        this shard has been consumed" is the condition that actually means finished.
        """
        # Consumption is recorded here rather than in _fill because only the reading
        # rank runs the read, while every rank retires the same keys. A strict
        # manifest asks whether the fill ever wanted a mapped key, which is a
        # question about the mapping and must get the same answer on every rank.
        self._used_keys.update(keys)
        by_shard = getattr(self, "_unread_by_shard", None)
        if by_shard is None:
            return
        for key in keys:
            path = self.weight_map.get(key)
            remaining = by_shard.get(path) if path else None
            if remaining is None:
                continue
            remaining.discard(key)
            if not remaining:
                del by_shard[path]
                # Forget that it was streamed: its pages are going away, so a reopen has to pay for
                # them again rather than trusting a warm that no longer holds.
                self._streamed_shards().discard(path)
                drop_file_page_cache([path])

    def _read_tensors(self, module, required, src: int = 0) -> None:
        """Read every tensor a module needs under one collective status exchange.

        A _source_call per tensor meant a pickled broadcast_object_list per tensor: fifteen of them
        per block on Z-Image, so several hundred across a transformer, all to report a failure that
        almost never happens. One exchange per module says the same thing.

        The key has to move into the exception for that to be free of cost to diagnosis, since peers
        only ever see the message text that _collective_source_call forwards. Without it a missing
        tensor would name the block and leave the reader to guess which of fifteen it was.
        """

        def read_all():
            for local_name, key in required:
                try:
                    self._fill(module, local_name, key, required=True)
                except Exception as error:
                    raise RuntimeError(
                        f"reading checkpoint tensor {key}: {type(error).__name__}: {error}"
                    ) from error

        self._source_call(
            read_all,
            context=f"loading {self.subfolder} checkpoint tensors",
            src=src,
        )

    def fill_block(self, block, i):
        """Fill + broadcast one wrapped block, excluding only non-persistent buffers."""
        device = getattr(self, "device", "cpu")
        fqn = self._id2fqn.get(id(block))
        if fqn is None:
            raise RuntimeError(f"block {i} not found in wrap_attrs index (id mismatch)")
        layout = _tensor_layout(block)
        src = self._reader_for_block(i)
        with self._timed("agree"):
            self._assert_same_layout(block)
        prefix = fqn + "."
        required = [
            (local_name, prefix + self._ckpt_key(block, local_name))
            for local_name, _ in block.named_parameters(remove_duplicate=False)
        ] + [
            (local_name, prefix + self._ckpt_key(block, local_name))
            for local_name, _ in _persistent_named_buffers(block)
        ]
        with self._timed("agree"):
            self._require_checkpoint_keys([key for _, key in required], src=src)
        with self._timed("read"):
            self._read_tensors(block, required, src=src)
        broadcast_names = [name for kind, name in layout if kind == "parameter"] + [
            name for name, _ in _persistent_named_buffers(block)
        ]
        with self._timed("agree"):
            self._reconcile_tensor_specs(block, broadcast_names, src=src)
        with self._timed("broadcast"):
            self._broadcast(block, src=src)
        self._retire_keys([key for _, key in required])
        if i % 8 == 0:
            log(
                f"  self-fill {self.subfolder} block {i}: host cur/anon/file "
                f"{host_mem_gb()} GB, VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB"
            )

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
            name
            for name, _ in comp.named_parameters(remove_duplicate=False)
            if not self._ckpt_key(comp, name).startswith(self._block_prefixes)
        ]
        all_tail_bufs = [
            name
            for name, _ in comp.named_buffers(remove_duplicate=False)
            if not self._ckpt_key(comp, name).startswith(self._block_prefixes)
        ]
        tail_bufs = [
            name
            for name, _ in _persistent_named_buffers(comp)
            if not self._ckpt_key(comp, name).startswith(self._block_prefixes)
        ]
        tail_layout = tuple(
            [("parameter", name) for name in tail]
            + [("buffer", name) for name in all_tail_bufs]
        )
        self._assert_same_layout(comp)
        self._require_checkpoint_keys(
            [self._ckpt_key(comp, name) for name in tail + tail_bufs]
        )
        target_type = torch.device(self.device).type
        for name in tail + all_tail_bufs:
            t = rgetattr(comp, name)
            if t.is_meta:
                set_module_tensor_to_device(
                    comp,
                    name,
                    self.device,
                    value=torch.empty(t.shape, dtype=t.dtype, device=self.device),
                )
            elif t.device.type != target_type:
                # Non-persistent buffers (e.g. Wan rope freqs_cos/freqs_sin) are created real on
                # CPU by init_empty_weights (include_buffers=False), not meta. Their values are
                # correct and identical across ranks, so move them on-device without broadcasting.
                set_module_tensor_to_device(
                    comp, name, self.device, value=t.to(self.device)
                )
        with self._timed("read"):
            self._read_tensors(
                comp,
                [(name, self._ckpt_key(comp, name)) for name in tail + tail_bufs],
            )
        with self._timed("agree"):
            self._reconcile_tensor_specs(comp, tail + tail_bufs)
        if self.group is not None:
            with self._timed("broadcast"):
                self._broadcast_tensors(
                    [rgetattr(comp, name).data for name in tail + tail_bufs]
                )
        self._retire_keys([self._ckpt_key(comp, name) for name in tail + tail_bufs])
        if self.strict:
            unused = sorted(set(self.weight_map) - self._used_keys)
            if unused:
                preview = ", ".join(unused[:3])
                suffix = f" (+{len(unused) - 3} more)" if len(unused) > 3 else ""
                raise RuntimeError(
                    f"unexpected checkpoint tensors in {self.subfolder}: "
                    f"{preview}{suffix}"
                )
        self._retie_weights(comp)
        self._release_handles()
        # Before the drop, not after: a prefetch still streaming would put back the cache this is
        # about to release. Joining here rather than in _release_handles is what keeps the prefetch
        # overlapped, since that runs on every shard change.
        self._join_prefetch()
        # Backstop: _retire_keys drops each shard as its last key is consumed, but a shard holding
        # keys this fill never asks for would otherwise keep its cache to the end.
        drop_file_page_cache(self.shard_paths)
        self._log_phase_breakdown()

    def _log_phase_breakdown(self) -> None:
        """Report where the fill spent its time, so a slow fill points at what to change."""
        if not self._phase_seconds:
            return
        parts = " ".join(
            f"{phase} {self._phase_seconds[phase]:.1f}s"
            for phase in ("read", "broadcast", "agree")
            if self._phase_seconds.get(phase)
        )
        if parts:
            log(f"  self-fill {self.subfolder} phases: {parts}")

    def _retie_weights(self, comp) -> None:
        """Restore tied weights, which the fill unpicked.

        set_module_tensor_to_device rebinds each name to a new Parameter, so names that shared one
        tensor before the fill no longer do afterwards. Both copies hold the right values (a tie maps
        to the same checkpoint key, so each alias is filled from it), which is why this is a memory
        and identity fix rather than a correctness one: an embedding-sized tensor per tie, and a
        model whose declared ties no longer hold.

        Runs after the tail is filled and broadcast, so both aliases already agree; re-tying earlier
        would alias a filled tensor to an unfilled one. Only ever affects the tail, since block
        Linears are what quantization touches and ties live in embeddings and heads.
        """
        tie = getattr(comp, "tie_weights", None)
        if callable(tie) and getattr(comp, "_tied_weights_keys", None):
            tie()

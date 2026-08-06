import abc
import torch
import copy
import argparse
import json
import functools
from PIL.Image import Image
from typing import Callable, List, Optional, Tuple, Generator
from dataclasses import dataclass, field, replace
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import numpy as np
from xfuser.config import args, xFuserArgs
from xfuser.envs import (
    PACKAGES_CHECKER,
    _TORCH_GROUPNORM,
    get_platform,
    _is_hip,
    _is_cuda,
)
from xfuser.core.utils.runner_utils import (
    log,
    load_dataset_prompts,
    convert_model_convs_to_channels_last,
    _use_aiter_fp8_rdna4,
    rgetattr,
)

from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_world_size,
    initialize_runtime_state,
    get_runtime_state,
    init_distributed_environment,
)
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.attention_schedule import AttentionSchedule, create_hybrid_attn_schedule, create_hybrid_gemm_schedule
from xfuser.model_executor.models.runner_models.loading.checkpoint import CheckpointRequest
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadCapability,
    select_effective_materialization_mode,
    select_load_contract,
    select_runtime_quantization,
)
from xfuser.model_executor.models.runner_models.loading.format_backends import (
    module_path_is_covered,
    module_paths_overlap,
)


def _component_target_paths(component_name, targets):
    return {
        component_name if not target else f"{component_name}.{target}"
        for target in targets
    }


def _record_streaming_targets(model, attribute, component_name, targets):
    tracked = getattr(model, attribute, None)
    if tracked is not None:
        tracked.update(_component_target_paths(component_name, targets))


def _blockwise_owned_targets(targets, wrap_attrs):
    owned = []
    for target in targets:
        for wrap_attr in wrap_attrs:
            if module_path_is_covered(target, wrap_attr):
                owned.append(target)
            elif module_path_is_covered(wrap_attr, target):
                owned.append(wrap_attr)
    minimal = []
    for target in sorted(set(owned), key=lambda path: (path.count("."), path)):
        if not any(
            module_path_is_covered(target, owner)
            for owner in minimal
        ):
            minimal.append(target)
    return tuple(minimal)


def _record_blockwise_ownership(
    model,
    adapter,
    component_name,
    targets,
    wrap_attrs,
    descriptor,
):
    log(descriptor.log_message())
    if adapter.format.value == "fp8" and hasattr(
        model, "_fp8_descriptor_components"
    ):
        model._fp8_descriptor_components.add(component_name)
    if hasattr(model, "_quantization_descriptor_components"):
        model._quantization_descriptor_components.add(component_name)
    if descriptor.materialization_mode not in {"streaming", "blockwise"}:
        return
    owned_targets = _blockwise_owned_targets(targets, wrap_attrs)
    _record_streaming_targets(
        model,
        "_quantization_streaming_targets",
        component_name,
        owned_targets,
    )
    if descriptor.materialization_mode == "blockwise" and getattr(
        model.config, "use_fp4_gemms", False
    ):
        fp8_targets = tuple(model.fp8.targets_for(component_name))
        owned_fp8_targets = _blockwise_owned_targets(fp8_targets, wrap_attrs)
        _record_streaming_targets(
            model,
            "_quantization_streaming_targets",
            component_name,
            owned_fp8_targets,
        )
        _record_streaming_targets(
            model,
            "_fp8_streaming_targets",
            component_name,
            owned_fp8_targets,
        )
    if adapter.format.value == "fp8":
        _record_streaming_targets(
            model,
            "_fp8_streaming_targets",
            component_name,
            owned_targets,
        )


def _blockwise_transformer_descriptor(
    adapter,
    component_name,
    targets,
    wrap_attrs,
    *,
    local=False,
):
    if adapter.format.value == "fp8":
        from .loading.fp8_backends import (
            plan_blockwise_transformer_fp8_load,
        )

        descriptor = plan_blockwise_transformer_fp8_load(
            adapter,
            component_name=component_name,
            targets=targets,
            wrap_attrs=wrap_attrs,
        )
        return (
            replace(descriptor, materialization_mode="blockwise")
            if local and descriptor.materialization_mode == "streaming"
            else descriptor
        )
    from .loading.format_backends import describe_blockwise_format_load

    return describe_blockwise_format_load(
        adapter,
        component_name=component_name,
        targets=targets,
        wrap_attrs=wrap_attrs,
    )


def _conversion_filter(module_path, excluded_paths):
    overlapping = tuple(
        path
        for path in excluded_paths
        if module_paths_overlap(module_path, path)
    )
    if any(
        module_path_is_covered(module_path, path)
        for path in overlapping
    ):
        return False, None
    descendants = tuple(
        path
        for path in overlapping
        if module_path_is_covered(path, module_path)
    )
    if not descendants:
        return True, None

    def filter_fn(_module, fqn):
        full_path = module_path if not fqn else f"{module_path}.{fqn}"
        return not any(
            module_path_is_covered(full_path, path)
            for path in descendants
        )

    return True, filter_fn


def _fp8_adapter_for_contract(model):
    """Return the backend that owns FP8 storage for the active contract."""

    if model.load_contract is None:
        return None
    format_value = model.load_contract.requested_format.value
    if format_value == "fp8":
        return model.fp8_backend
    if format_value in {"fp4", "fp8_fp4"}:
        return model.blockwise_fp8_backend
    return None


packages_info = PACKAGES_CHECKER.get_packages_info()

MODEL_REGISTRY = {}

def register_model(name: str) -> Callable:
    """ Decorator to register a model in the registry. """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


_SPARSE_ATTENTION_BACKENDS = frozenset({
    AttentionBackendType.AITER_SPARSE_SAGE,
    AttentionBackendType.AITER_SPARSE_SAGE_V2,
    AttentionBackendType.FLEX_BLOCK_ATTN
})
_SPARGE_ATTENTION_BACKENDS = frozenset({
    AttentionBackendType.AITER_SPARGE,
    AttentionBackendType.AITER_SPARGE_V2,
    AttentionBackendType.FLEX_BLOCK_SPARGE,
})


def _parse_attention_backend(name: Optional[str], kind: str) -> Optional[AttentionBackendType]:
    if name is None:
        return None
    try:
        return AttentionBackendType[name.upper()]
    except KeyError:
        raise ValueError(f"Invalid {kind}: {name}")


def _validate_cross_attention_for_sparge(config: xFuserArgs) -> None:
    """Cross-attention must be set and must not itself be a Sparge backend
    whenever Sparge Attention is in play (either as the explicit backend or
    via the hybrid schedule)."""
    if config.cross_attention_backend is None:
        raise ValueError(
            "When Sparge Attention is used, --cross_attention_backend must be "
            "set to a non-Sparge backend."
        )
    cross = _parse_attention_backend(
        config.cross_attention_backend, "cross attention backend",
    )
    if cross in _SPARGE_ATTENTION_BACKENDS:
        raise ValueError(
            f"--cross_attention_backend cannot be {cross.name} when Sparge "
            f"Attention is used. Pick a non-Sparge cross attention backend."
        )


@dataclass(frozen=True)
class ModelCapabilities:
    """ Class to define model capabilities """
    # Parallelization
    ulysses_degree: bool = True  # All xDiT models support these
    ring_degree: bool = True
    pipefusion_parallel_degree: bool = False
    data_parallel_degree: bool = True
    tensor_parallel_degree: bool = False
    use_cfg_parallel: bool = False
    use_parallel_vae: bool = False
    use_parallel_vae_encoder: bool = False
    fully_shard_degree: bool = False
    # Memory optimizations
    enable_slicing: bool = False
    enable_tiling: bool = False
    use_vae_channels_last_format: bool = True
    # Other features
    use_int8_gemms: bool = False
    use_fp8_gemms: bool = False
    use_fp4_gemms: bool = False
    use_fbcache: bool = False
    use_hybrid_attn_schedule: bool = False
    use_hybrid_gemm_schedule: bool = False
    cross_attention_backend: bool = False
    supports_sparse_attention_backends: bool = False
    supports_sparge_attention_backends: bool = False
    supports_distilled_weights: bool = False

@dataclass(frozen=True)
class DefaultInputValues:
    """ Class to define model specific default input values """
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    flow_shift: Optional[float] = None
    max_sequence_length: Optional[int] = None
    num_hybrid_attn_high_precision_steps: Optional[int] = None
    num_hybrid_gemm_high_precision_steps: Optional[int] = None
    ssta_tile_thw: Optional[Tuple[int, int, int]] = None

@dataclass
class ModelSettings:
    """ Class to define model options """
    model_name: Optional[str] = None
    output_name: Optional[str] = None
    model_output_type: Optional[str] = None
    mod_value: Optional[int] = None
    fps: Optional[int] = None
    int8_gemm_module_list: List[str] = None
    fp8_gemm_module_list: List[str] = None
    fp8_text_encoder_module_list: List[str] = None
    fp4_gemm_module_list: List[str] = None
    fp8_precision_overrides: Tuple[str] = None
    fp8_precision_override_suffixes: Tuple[str] = None
    fbcache_thresh: float = 0.12
    # FSDP strategy is just for the components to be sharded - other components will be moved to correct device automatically
    fsdp_strategy: dict = field(default_factory=lambda: {
        "": { # name, e.g. transformer
            "shard_submodule_key": None, # submodule to shard, e.g encoder -> transformer.encoder will be sharded
            "block_attr": None, # attribute name of blocks to shard, e.g. blocks
            "dtype": None, # Target dtype to convert the model to before sharding
            "children_to_device": [{ # Move other children to device
                "submodule_key": None, # e.g "encoder" -> children of transformer.encoder
                "exclude_keys": [] # exclude these children from being moved
            }]
        }
    })
    valid_tasks: List[str] = field(default_factory=list)
    resolution_divisor: Optional[int] = None

class DiffusionOutput:
    """ Class to encapsulate diffusion model outputs """
    def __init__(self, images: List[Image] = None, videos: List[np.ndarray]|np.ndarray = None, pipe_args: List[dict]|dict = []) -> None:
        self.images = images
        if isinstance(videos, np.ndarray) and videos.ndim == 5:
            videos = list(videos)
        elif not isinstance(videos, list):
            videos = [videos]
        self.videos = videos
        if not isinstance(pipe_args, list):
            pipe_args = [pipe_args]
        output_count = len(self.images or self.videos or [])
        if len(pipe_args) == 1 and output_count > 1:
            pipe_args = pipe_args * output_count
        self.pipe_args = pipe_args

    @classmethod
    def from_outputs(cls, outputs: List["DiffusionOutput"], output_type: str) -> "DiffusionOutput":
        if output_type == "image":
            args_list = []
            all_images = []
            for out in outputs:
                all_images.extend(out.images)
                args_list.extend(out.pipe_args)
            return DiffusionOutput(images=all_images, pipe_args=args_list)
        elif output_type == "video":
            all_videos = []
            args_list = []
            for out in outputs:
                all_videos.extend(out.videos)
                args_list.extend(out.pipe_args)
            return DiffusionOutput(videos=all_videos, pipe_args=args_list)
        else:
            raise NotImplementedError(f"DiffusionOutput does not support output type: {output_type}")

    def get_outputs(self) -> Generator[Tuple[Image|np.ndarray, dict], None, None]:
        """ Returns a generator that yields output items along with their used input arguments """
        if self.images:
            for image, single_pipe_args in zip(self.images, self.pipe_args):
                yield (image, single_pipe_args)
        elif self.videos:
            for video, single_pipe_args in zip(self.videos, self.pipe_args):
                yield (video, single_pipe_args)

class xFuserModel(abc.ABC):
    """ Base class for xFuser models """

    capabilities: ModelCapabilities = ModelCapabilities()
    load_capability: LoadCapability = LoadCapability.for_runner(
        capabilities,
        unsupported_reason=(
            "runner has not declared a compatible meta construction seam"
        ),
    )
    default_input_values: DefaultInputValues = DefaultInputValues()
    settings: ModelSettings = ModelSettings()
    model_output_type: str = ""
    fps: int = 0

    def __init__(self, config: xFuserArgs) -> None:
        self.settings = copy.deepcopy(self.settings)
        self._customize_settings(config)
        self._refresh_load_capability()
        self._validate_config(config)
        self._update_model_settings(config)
        self.config = config
        self.pipe = None
        self.load_contract = None
        self._fp8_descriptor_components = set()
        self._fp8_streaming_targets = set()
        self._quantization_descriptor_components = set()
        self._quantization_streaming_targets = set()

    def _refresh_load_capability(self) -> None:
        """Re-derive FSDP support from instance-customized settings."""
        declaration = type(self).load_capability
        self.load_capability = LoadCapability.for_runner(
            self.capabilities,
            meta_transformers=declaration.meta_transformers,
            replicated=bool(
                declaration.replicated_meta_transformers
            ),
            fsdp_strategy=self.settings.fsdp_strategy,
            loader_adapter=declaration.loader_adapter,
            component_exclusions=declaration.component_exclusions,
            unsupported_reason=declaration.unsupported_reason,
        )

    def _select_preload_contract(self, *, world_size: int):
        """Resolve and validate loading before model allocation or load collectives."""
        mode = select_effective_materialization_mode(
            self.config, world_size=world_size
        )
        requested_format, backend = select_runtime_quantization(
            self.config,
            aiter_fp8_active=bool(
                self.config.use_fp8_gemms and _use_aiter_fp8_rdna4()
            ),
            cuda_active=_is_cuda(),
        )
        return select_load_contract(
            requested_format=requested_format,
            selected_backend=backend,
            materialization_mode=mode,
            capability=self.load_capability,
            fsdp_strategy=self.settings.fsdp_strategy,
            runner_name=type(self).__name__,
        )

    def _customize_settings(self, config: xFuserArgs) -> None:
        """Hook for subclasses to mutate self.settings before validation and CLI overrides.

        Runs on the instance-local deepcopy, before _validate_config and
        _update_model_settings, so subclass model_name/valid_tasks/overrides are in
        place when those consumers run. Subclasses must use the `config` parameter;
        self.config is not assigned until __init__ completes.
        """
        pass

    def _update_model_settings(self, config: xFuserArgs) -> None:
        if config.use_fp4_gemms:
            self._apply_fp8_override_cli_from_config(config)
        te_targets = self.settings.fp8_text_encoder_module_list
        if config.use_fp8_text_encoder and not te_targets:
            log(f"--use_fp8_text_encoder has no effect for {type(self).__name__}: it declares no "
                f"text-encoder FP8 targets.")
        elif te_targets and config.use_fp8_gemms and not config.use_fp8_text_encoder:
            # Says so out loud because text-encoder FP8 used to ride along with --use_fp8_gemms on
            # RDNA4, and is now opt-in everywhere; a run that silently kept a bf16 text encoder
            # would otherwise look like the flag had regressed.
            log(f"--use_fp8_gemms covers the transformer; {type(self).__name__}'s "
                f"{len(te_targets)} text-encoder target(s) stay bf16. Add --use_fp8_text_encoder "
                f"to quantize them too, for less memory at some risk to text conditioning.")

    def initialize(self, input_args: dict) -> None:
        """ Load the model pipeline """

        if not torch.distributed.is_initialized():
            log("Initializing distributed environment...")
            init_distributed_environment()

        self.load_contract = self._select_preload_contract(
            world_size=get_world_group().world_size
        )
        # Capability/backend mismatches fail here, before _load_model allocates
        # transformer weights.
        _ = self.fp8_backend
        _ = self.format_backend
        if self._uses_blockwise_fp8_backend():
            _ = self.blockwise_fp8_backend
        self.engine_config, _ = self.config.create_config()
        log("Loading model pipeline...")
        self.pipe = self._load_model()

        log("Initializing runtime state...")
        initialize_runtime_state(self.pipe, self.engine_config)

        self._post_load_and_state_initialization(input_args)
        self._enable_options()

        if self.config.use_torch_compile:
            log("Torch.compile enabled. Warming up torch compiler ...")
            compile_input_args = copy.deepcopy(input_args)
            compile_input_args = self._split_prompts_for_dp(compile_input_args)
            if self.config.batch_size and isinstance(compile_input_args.get("prompt"), list):
                compile_input_args["prompt"] = compile_input_args["prompt"][: self.config.batch_size]
            self._compile_model(compile_input_args)

    @property
    def fp8(self):
        """This run's FP8 coverage and the loader configs that apply it (see fp8_plan.Fp8Plan)."""
        from xfuser.model_executor.models.runner_models.loading.fp8_plan import Fp8Plan
        return Fp8Plan(self)

    @functools.cached_property
    def fp8_backend(self):
        """The task-3-selected FP8 implementation, validated before allocation."""
        if (
            self.load_contract is None
            or self.load_contract.requested_format.value != "fp8"
        ):
            return None
        from .loading.fp8_backends import (
            probe_fp8_backend_capabilities,
            select_fp8_backend,
        )

        return select_fp8_backend(
            self.load_contract,
            capabilities=probe_fp8_backend_capabilities(),
        )

    @functools.cached_property
    def format_backend(self):
        """Primary FP4/INT8 implementation, validated before allocation."""

        if (
            self.load_contract is None
            or self.load_contract.requested_format.value
            not in {"fp4", "fp8_fp4", "int8"}
        ):
            return None
        from .loading.format_backends import (
            probe_format_backend_capabilities,
            select_format_backend,
            validate_format_fsdp_placement,
        )

        capabilities = probe_format_backend_capabilities()
        adapter = select_format_backend(
            self.load_contract,
            capabilities=capabilities,
            hybrid=self.config.use_hybrid_gemm_schedule,
        )
        validate_format_fsdp_placement(
            self.load_contract,
            adapter,
            capabilities=capabilities,
            required=self._places_format_backend_under_fsdp2(),
        )
        return adapter

    @functools.cached_property
    def blockwise_fp8_backend(self):
        """FP8 converter for pure FP8 and FP8-only portions of hybrid loads."""

        if self.load_contract is None:
            return None
        from .loading.fp8_backends import (
            probe_fp8_backend_capabilities,
            select_blockwise_fp8_backend,
            validate_torchao_fsdp2_patches,
        )

        capabilities = probe_fp8_backend_capabilities()
        adapter = select_blockwise_fp8_backend(
            self.load_contract,
            capabilities=capabilities,
        )
        validate_torchao_fsdp2_patches(
            self.load_contract,
            capabilities=capabilities,
            required=self._places_torchao_tensor_subclass_under_fsdp2(
                adapter
            ),
        )
        return adapter

    def _places_torchao_tensor_subclass_under_fsdp2(
        self,
        fp8_adapter,
        *,
        assume_torchao_fp8: bool = False,
    ) -> bool:
        """Whether configured FSDP2 blocks will contain TorchAO Float8Tensor."""

        if self.config.fully_shard_degree <= 1:
            return False
        fsdp_target_paths = {
            f"{component_name}.{wrap_attr}"
            for component_name, strategy in (
                self.settings.fsdp_strategy or {}
            ).items()
            for wrap_attr in strategy.get("wrap_attrs", ())
        }
        if not fsdp_target_paths:
            return False

        fp4_targets = set(self.settings.fp4_gemm_module_list or ())
        fp8_only_targets = {
            target
            for target in self.fp8.module_list()
            if not any(
                module_path_is_covered(target, fp4_target)
                for fp4_target in fp4_targets
            )
        }
        if (
            (
                assume_torchao_fp8
                or (
                    fp8_adapter is not None
                    and fp8_adapter.backend.value == "torchao"
                )
            )
            and any(
                module_paths_overlap(target, fsdp_path)
                for target in fp8_only_targets
                for fsdp_path in fsdp_target_paths
            )
        ):
            return True

        fp4_can_emit_fp8 = bool(
            self.settings.fp8_precision_overrides
            or self.settings.fp8_precision_override_suffixes
            or self.config.use_hybrid_gemm_schedule
        )
        return bool(
            self.config.use_fp4_gemms
            and fp4_can_emit_fp8
            and any(
                module_paths_overlap(target, fsdp_path)
                for target in fp4_targets
                for fsdp_path in fsdp_target_paths
            )
        )

    def _requires_blockwise_fp8_backend(self) -> bool:
        """Whether FP4 mode declares whole components owned only by FP8."""

        if not self.config.use_fp4_gemms:
            return False
        fp4_targets = set(self.settings.fp4_gemm_module_list or ())
        return any(
            not any(
                module_path_is_covered(target, fp4_target)
                for fp4_target in fp4_targets
            )
            for target in self.fp8.module_list()
        )

    def _places_format_backend_under_fsdp2(self) -> bool:
        if self.config.fully_shard_degree <= 1:
            return False
        fsdp_target_paths = {
            f"{component_name}.{wrap_attr}"
            for component_name, strategy in (
                self.settings.fsdp_strategy or {}
            ).items()
            for wrap_attr in strategy.get("wrap_attrs", ())
        }
        if self.load_contract.requested_format.value in {"fp4", "fp8_fp4"}:
            targets = set(self.settings.fp4_gemm_module_list or ())
        elif self.load_contract.requested_format.value == "int8":
            targets = set(self.settings.int8_gemm_module_list or ())
        else:
            return False
        return any(
            module_paths_overlap(target, fsdp_path)
            for target in targets
            for fsdp_path in fsdp_target_paths
        )

    def _format_targets_for(self, component_name: str) -> tuple[str, ...]:
        if self.load_contract.requested_format.value in {"fp4", "fp8_fp4"}:
            entries = self.settings.fp4_gemm_module_list or ()
        elif self.load_contract.requested_format.value == "int8":
            entries = self.settings.int8_gemm_module_list or ()
        else:
            return ()
        prefix = f"{component_name}."
        return tuple(
            ""
            if entry == component_name
            else entry[len(prefix):]
            for entry in entries
            if entry == component_name or entry.startswith(prefix)
        )

    def _transformer_quantization_adapter(self, component_name: str):
        """Return the adapter and relative targets owning one transformer."""

        format_targets = self._format_targets_for(component_name)
        if format_targets:
            return self.format_backend, format_targets
        fp8_targets = tuple(self.fp8.targets_for(component_name))
        if not fp8_targets:
            return None, ()
        return _fp8_adapter_for_contract(self), fp8_targets

    def _uses_blockwise_fp8_backend(self) -> bool:
        if self._requires_blockwise_fp8_backend():
            return True
        if self._places_torchao_tensor_subclass_under_fsdp2(None):
            return True
        if (
            self.load_contract.requested_format.value == "fp8"
            and self._places_torchao_tensor_subclass_under_fsdp2(
                None, assume_torchao_fp8=True
            )
        ):
            return True
        return (
            self.load_contract.materialization_mode.value != "eager"
            and self.load_contract.requested_format.value == "fp8"
        )

    def _memory_efficient_fsdp_load(self) -> bool:
        """True when the memory-efficient sharded (meta-init + rank0-broadcast) load path is on."""
        return self._loader.fsdp_meta_load()

    def _replicated_broadcast_load(self) -> bool:
        """True when replicated components load once on rank0 and broadcast to peers
        (see MemoryEfficientLoader.replicated_broadcast_load)."""
        return self._loader.replicated_broadcast_load()

    @functools.cached_property
    def _loader(self):
        """Lazy MemoryEfficientLoader bound to this model (meta-init + rank0-broadcast load).

        Must stay cached, unlike ``fp8``: the loader records which transformers it built on meta, so
        a fresh instance per access would report none of them and silently route every component to
        the wrong fill path.
        """
        from xfuser.model_executor.models.runner_models.loading.meta_load import MemoryEfficientLoader
        return MemoryEfficientLoader(self)

    def _checkpoint_request(self, subfolder: str | None = None, **kwargs) -> CheckpointRequest:
        """Checkpoint identity shared by discovery and from_pretrained calls."""
        return CheckpointRequest(
            self.settings.model_name, subfolder=subfolder, **kwargs
        )

    def _build_transformer_structure(
        self,
        wrapper_cls,
        request: CheckpointRequest,
        init_kwargs: dict | None,
    ):
        """Build only meta tensors so FP8 target prefixes can be mapped safely."""

        from accelerate import init_empty_weights
        from contextlib import ExitStack

        config = wrapper_cls.load_config(
            request.model_name_or_path, **request.config_kwargs()
        )
        with ExitStack() as stack:
            try:
                stack.enter_context(
                    init_empty_weights(include_buffers=True)
                )
            except TypeError as exc:
                raise RuntimeError(
                    "accelerate.init_empty_weights(include_buffers=True) is "
                    "required for bounded structure inspection"
                ) from exc
            return wrapper_cls.from_config(config, **(init_kwargs or {}))

    def _native_quantization_device_map(self):
        """Place TorchAO quantize-on-load weights on this rank's accelerator."""

        return {"": get_world_group().local_rank}

    def _build_transformer(
        self,
        wrapper_cls,
        subfolder: str | None = None,
        init_kwargs: dict | None = None,
        stream_quant: bool = True,
        checkpoint_request: CheckpointRequest | None = None,
    ):
        """Load a transformer through the selected materialization backend.

        FSDP/replicated paths retain the xDiT meta-build and blockwise filler.
        Ordinary backends pass a native Diffusers quantization config to
        ``from_pretrained`` only when the format's exact semantics permit it.

        init_kwargs: extra wrapper __init__ args (e.g. wan's attention_kwargs) forwarded on both paths.
        stream_quant: gates native quantize-on-load for ordinary loading. Meta
        paths always convert targeted blocks through the backend adapter.
        """
        request = checkpoint_request or self._checkpoint_request(
            subfolder or "transformer"
        )
        if subfolder is not None and request.subfolder != subfolder:
            request = request.with_subfolder(subfolder)
        elif request.subfolder is None:
            request = request.with_subfolder("transformer")
        component_name = request.subfolder
        adapter_selector = getattr(
            self, "_transformer_quantization_adapter", None
        )
        if adapter_selector is None:
            adapter = self.fp8_backend
            targets = tuple(self.fp8.targets_for(component_name))
        else:
            adapter, targets = adapter_selector(component_name)
            targets = tuple(targets)
        strategy = self.settings.fsdp_strategy.get(component_name, {})
        wrap_attrs = tuple(strategy.get("wrap_attrs", ()))
        fsdp_meta = self._memory_efficient_fsdp_load()
        replicated_meta = (
            False if fsdp_meta else self._replicated_broadcast_load()
        )
        if fsdp_meta or replicated_meta:
            if adapter is not None:
                descriptor = _blockwise_transformer_descriptor(
                    adapter,
                    component_name,
                    targets,
                    wrap_attrs,
                )
                _record_blockwise_ownership(
                    self,
                    adapter,
                    component_name,
                    targets,
                    wrap_attrs,
                    descriptor,
                )
            return self._loader.build_meta_transformer(
                wrapper_cls, request, init_kwargs
            )
        quantization_config = None
        if adapter is not None:
            if adapter.format.value == "fp8":
                from .loading.fp8_backends import (
                    prepare_native_transformer_fp8_load,
                )

                prepared = prepare_native_transformer_fp8_load(
                    adapter,
                    component_name=component_name,
                    targets=targets,
                    stream_quant=stream_quant,
                    model_factory=lambda: self._build_transformer_structure(
                        wrapper_cls, request, init_kwargs
                    ),
                )
            else:
                from .loading.format_backends import (
                    prepare_native_transformer_format_load,
                )

                prepared = prepare_native_transformer_format_load(
                    adapter,
                    component_name=component_name,
                    targets=targets,
                    stream_quant=stream_quant,
                    precision_prefixes=(
                        self.settings.fp8_precision_overrides or ()
                    ),
                    precision_suffixes=(
                        self.settings.fp8_precision_override_suffixes or ()
                    ),
                    hybrid=self.config.use_hybrid_gemm_schedule,
                    model_factory=lambda: self._build_transformer_structure(
                        wrapper_cls, request, init_kwargs
                    ),
                )
            loader = getattr(self, "_loader", None)
            local_plan_factory = getattr(
                loader, "plan_eager_blockwise_fallback", None
            )
            local_plan = (
                local_plan_factory(prepared, targets, wrap_attrs)
                if local_plan_factory is not None
                else None
            )
            if local_plan is not None and local_plan.enabled:
                descriptor = _blockwise_transformer_descriptor(
                    adapter,
                    component_name,
                    targets,
                    wrap_attrs,
                    local=True,
                )
                _record_blockwise_ownership(
                    self,
                    adapter,
                    component_name,
                    targets,
                    wrap_attrs,
                    descriptor,
                )
                component = loader.build_meta_transformer(
                    wrapper_cls, request, init_kwargs
                )
                loader.mark_local_blockwise(component)
                return component
            log(prepared.descriptor.log_message())
            quantization_config = prepared.quantization_config
            if (
                adapter.format.value == "fp8"
                and hasattr(self, "_fp8_descriptor_components")
            ):
                self._fp8_descriptor_components.add(component_name)
            if hasattr(self, "_quantization_descriptor_components"):
                self._quantization_descriptor_components.add(component_name)
            if (
                quantization_config is not None
            ):
                streamed_targets = (
                    getattr(prepared, "streamed_targets", ()) or targets
                )
                _record_streaming_targets(
                    self,
                    "_quantization_streaming_targets",
                    component_name,
                    streamed_targets,
                )
                if adapter.format.value == "fp8":
                    _record_streaming_targets(
                        self,
                        "_fp8_streaming_targets",
                        component_name,
                        streamed_targets,
                    )
        load_kwargs = request.from_pretrained_kwargs()
        device_map_factory = getattr(
            self, "_native_quantization_device_map", None
        )
        if quantization_config is not None and device_map_factory is not None:
            load_kwargs.setdefault("device_map", device_map_factory())
        return wrapper_cls.from_pretrained(
            request.model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            **load_kwargs,
            **(init_kwargs or {}),
        )

    def _meta_te_kwargs(self, existing_quantization_config=None):
        """Build text-encoder(s) on meta for the memory-efficient FSDP load path.

        Returns (pipe_component_kwargs, te_quant_config). On the meta path the kwargs carry meta
        modules to hand to the pipeline's from_pretrained (so it skips loading those components)
        and te_quant is None — the pipe does not stream the TE; instead the meta module (fp8 when
        targeted, else bf16) is filled by the rank0-broadcast sharded load, then FSDP-sharded
        (CPU-offloaded). The transformer is unaffected either way; it keeps its own streaming-fp8
        from_pretrained path.

        The normal-path config is built last, and only if it is reached: constructing it registers
        the transformers quantizer process-globally, which the meta paths have no use for.
        """
        replicated_meta = self._replicated_broadcast_load()
        fsdp_meta = (
            False if replicated_meta else self._memory_efficient_fsdp_load()
        )
        adapter = _fp8_adapter_for_contract(self)
        component_configs = {}
        entries = self.settings.fp8_text_encoder_module_list or ()
        component_names = tuple(
            dict.fromkeys(
                entry.partition(".")[0]
                for entry in entries
                if "." in entry
            )
        )
        if adapter is not None:
            from .loading.fp8_backends import (
                prepare_text_encoder_fp8_load,
            )

            for component_name in component_names:
                targets = tuple(self.fp8.targets_for(component_name))
                if not targets:
                    continue
                # Existing meta layouts only mirror AITER's plain fp8+scale
                # representation. Replicated TorchAO falls back after broadcast;
                # memory-efficient FSDP rejects that layout-changing fallback.
                stream_quant = not (replicated_meta or fsdp_meta) or (
                    adapter.backend.value == "aiter"
                )
                prepared = prepare_text_encoder_fp8_load(
                    adapter,
                    component_name=component_name,
                    targets=targets,
                    stream_quant=stream_quant,
                    supports_post_load=not fsdp_meta,
                    model_factory=lambda name=component_name: (
                        self._loader.build_meta_component(name, fp8=False)
                    ),
                )
                log(prepared.descriptor.log_message())
                self._fp8_descriptor_components.add(component_name)
                if prepared.descriptor.materialization_mode == "streaming":
                    _record_streaming_targets(
                        self,
                        "_fp8_streaming_targets",
                        component_name,
                        targets,
                    )
                if prepared.quantization_config is not None:
                    component_configs[component_name] = (
                        prepared.quantization_config
                    )
        self._text_encoder_quantization_configs = dict(component_configs)

        pipeline_config = existing_quantization_config
        if component_configs:
            from .loading.text_encoder_adapter import (
                TextEncoderFrameworkAdapter,
            )

            pipeline_config = (
                TextEncoderFrameworkAdapter()
                .pipeline_quantization_config(
                    component_configs,
                    existing=existing_quantization_config,
                )
            )

        if replicated_meta:
            return self._loader.meta_te_kwargs_replicated(pipeline_config)
        if fsdp_meta:
            meta_kwargs = self._loader.meta_te_kwargs()
            if meta_kwargs is not None:
                return meta_kwargs
        return {}, pipeline_config

    def _enable_options(self) -> None:
        """ Enable model options based on config"""
        if getattr(self.config, "use_spargeattn_head_balance", False):
            log("Enabling Sparge block-sparse head balancing...")

        if self.config.enable_slicing:
            log("Enabling VAE slicing...")
            self.pipe.vae.enable_slicing()

        if self.config.enable_tiling:
            log("Enabling VAE tiling...")
            self.pipe.vae.enable_tiling()

        if self.config.enable_group_cpu_offload:
            # block_level groups only top-level ModuleLists: fits compiled transformers
            # (blocks are top-level) and avoids the per-block-compile recompile storm that
            # leaf-level hooks trigger. Eager components nest their layers (e.g. Mistral-3 at
            # model.language_model.layers) where block_level can't reach -> whole component in
            # one unmatched group -> OOM; they use leaf_level, which recurses.
            from diffusers.hooks import apply_group_offloading
            log("Enabling group CPU offload (transformer block-level, others leaf-level, streamed)...")
            local_rank = get_world_group().local_rank
            onload_device = torch.device(f"cuda:{local_rank}")
            block_level_names = set(self._get_compiled_pipe_components())
            for name, component in self.pipe.components.items():
                if not isinstance(component, torch.nn.Module):
                    continue
                offload_type = "block_level" if name in block_level_names else "leaf_level"
                kwargs = dict(
                    onload_device=onload_device,
                    offload_type=offload_type,
                    use_stream=True,
                    record_stream=True,
                    # Pin each tensor as it is offloaded instead of pre-pinning the whole component:
                    # host RAM stays flat where it is the binding constraint, at some of the
                    # streaming win. Opt-in because it costs latency where host RAM is plentiful.
                    low_cpu_mem_usage=self.config.group_offload_low_cpu_mem,
                    non_blocking=True,
                )
                if offload_type == "block_level":
                    kwargs["num_blocks_per_group"] = 1
                if hasattr(component, "enable_group_offload"):
                    component.enable_group_offload(**kwargs)
                else:
                    apply_group_offloading(module=component, **kwargs)
        elif self.config.enable_sequential_cpu_offload:
            log("Enabling sequential CPU offload...")
            self.pipe.enable_sequential_cpu_offload()
        elif self.config.enable_model_cpu_offload:
            log("Enabling model CPU offload...")
            self.pipe.enable_model_cpu_offload()


    def _validate_config(self, config: xFuserArgs) -> None:
        """ Validate if the model supports requested config """
        config._validate_gemm_quantization_flags()
        for key in ModelCapabilities.__annotations__.keys():
            config_value = getattr(config, key, None)  # Some config options might not be set in the CLI, such as support for specific attention backends.
            if isinstance(config_value, int):
                if not getattr(self.capabilities, key) and config_value > 1:
                    raise ValueError(f"Model {self.settings.model_name} does not support {key}.")
            else:
                if config_value and not getattr(self.capabilities, key):
                    raise ValueError(f"Model {self.settings.model_name} does not support {key}.")

        backend = _parse_attention_backend(config.attention_backend, "attention backend")
        supports_sparse = self.capabilities.supports_sparse_attention_backends
        supports_sparge = self.capabilities.supports_sparge_attention_backends

        if backend is None:
            if supports_sparse:
                raise ValueError(
                    f"Model {config.model} supports sparse attention backends, "
                    f"but no attention backend was specified. Please specify a "
                    f"sparse attention backend to take advantage of the model's "
                    f"capabilities. If you want to use a dense attention backend, "
                    f"use the dense model equivalent."
                )
            if config.use_hybrid_attn_schedule:
                low = _parse_attention_backend(
                    config.hybrid_attn_low_precision_backend,
                    "hybrid low-precision attention backend",
                )
                high = _parse_attention_backend(
                    config.hybrid_attn_high_precision_backend,
                    "hybrid high-precision attention backend",
                )
                if (low in _SPARGE_ATTENTION_BACKENDS
                        or high in _SPARGE_ATTENTION_BACKENDS):
                    _validate_cross_attention_for_sparge(config)
        else:
            if backend in _SPARSE_ATTENTION_BACKENDS and not supports_sparse:
                raise ValueError(
                    f"Model {config.model} does not support sparse attention backends."
                )
            if supports_sparse and backend not in _SPARSE_ATTENTION_BACKENDS:
                raise ValueError(
                    f"Model {config.model} supports sparse attention backends, but "
                    f"attention backend '{config.attention_backend}' was specified. "
                    f"This is not an error per se, but you should use a sparse "
                    f"attention backend to take advantage of the model's capabilities. "
                    f"If you want to use a dense attention backend, use the dense "
                    f"model equivalent."
                )
            if backend in _SPARGE_ATTENTION_BACKENDS:
                if not supports_sparge:
                    raise ValueError(
                        f"Model {config.model} does not support Sparge attention backend."
                    )
                if self.capabilities.cross_attention_backend:
                    _validate_cross_attention_for_sparge(config)

        possible_task = getattr(config, "task", None)
        if possible_task and self.settings.valid_tasks:
            if possible_task not in self.settings.valid_tasks:
                raise ValueError(f"Model {self.settings.model_name} does not support task '{possible_task}'. Supported tasks: {self.settings.valid_tasks}")
        if possible_task and not self.settings.valid_tasks:
            raise ValueError(f"Model {self.settings.model_name} does not support multiple tasks, but task '{possible_task}' was specified.")
        if not possible_task and self.settings.valid_tasks:
            raise ValueError(f"Model {self.settings.model_name} requires a task to be specified. Supported tasks: {self.settings.valid_tasks}")
        if config.dataset_path and not config.batch_size:
            raise ValueError(f"Dataset path specified without batch size. Please specify batch size for dataset inference.")

        if self.model_output_type == "video" and not self.fps:
            raise ValueError(f"Model {self.settings.model_name} produces video output but fps is not set.")

        if config.use_int8_gemms and _is_hip():
            raise ValueError("Int8 GEMMs on ROCm are not supported.")
            
        if config.use_fp4_gemms:
            if _is_hip() and not packages_info.get("has_aiter", False):
                raise ValueError("FP4 GEMMs on ROCm require AITER.")
            if _is_cuda():
                major, _ = torch.cuda.get_device_capability()
                if major < 10:
                    raise ValueError(
                        f"NVFP4 GEMMs require CUDA capability >= 10.0 (Blackwell). "
                        f"Detected: {torch.cuda.get_device_capability()}"
                    )
        if config.use_parallel_vae:
            if not packages_info.get("has_distvae", False):
                raise ValueError("DistVAE is not installed. Please install it before using parallel VAE.")
            if torch.nn.GroupNorm.__module__ == "aiter.ops.groupnorm":
                log("AITER GroupNorm is not supported with parallel VAE. Reverting to torch GroupNorm.")
                torch.nn.GroupNorm = _TORCH_GROUPNORM
        
        if config.distilled_transformer_path or config.distilled_transformer_2_path:
            if not self.capabilities.supports_distilled_weights:
                raise ValueError(f"Model {self.settings.model_name} does not support distilled_transformer_path or distilled_transformer_2_path params.")


    def _get_compile_mode(self) -> str:
        # Overrides should return "default" when PACKAGES_CHECKER._on_rdna4():
        # CUDA graphs are slow on RDNA4.
        return "default"  # TODO: Configurable

    def _get_compile_dynamic(self) -> Optional[bool]:
        return None  # torch default (auto)

    def _get_compiled_pipe_components(self) -> List[str]:
        return ["transformer"]

    def _get_compile_warmup_steps(self, input_args: dict) -> Optional[int]:
        return 2  # None = skip step reduction, run full warmup cycle

    def _enable_compute_comm_overlap(self) -> None:
        """Enables compute-communication overlap for the model while caring for
        pipeline-parallel models that don't respect the SPMD assumption and could
        deadlock in torch's compiler spmd_check()."""
        torch._inductor.config.reorder_for_compute_comm_overlap = True

        # torch >= ~2.13: enabling the overlap machinery activates an SPMD
        # graph-consistency check that issues a WORLD-group all_gather_object at
        # compile time. Pipeline parallelism is non-SPMD (stages compile different
        # graphs at data-dependent times), so that collective deadlocks. For SPMD
        # runs the check is a cheap, useful guard, so only disable it under PP.
        if get_pipeline_parallel_world_size() > 1:
            _ado = getattr(torch._inductor.config, "aten_distributed_optimizations", None)
            if _ado is not None and hasattr(_ado, "spmd_check"):
                _ado.spmd_check = False

    def _compile_model(self, input_args: dict) -> None:
        """Compile pipe components with torch.compile.

        When FSDP is active (fully_shard_degree > 1), compiles each component's
        FSDP-wrapped block lists individually (read from fsdp_strategy wrap_attrs)
        to avoid dynamo tracing through FSDP2 forward_pre_hooks and fragmenting
        the graph at every block boundary.
        """
        self._enable_compute_comm_overlap()

        mode = self._get_compile_mode()
        dynamic = self._get_compile_dynamic()
        for component_name in self._get_compiled_pipe_components():
            component = getattr(self.pipe, component_name, None)
            if component is None:
                continue
            if self.config.fully_shard_degree > 1:
                wrap_attrs = self.settings.fsdp_strategy.get(component_name, {}).get("wrap_attrs", [])
                compiled_any = False
                for attr in wrap_attrs:
                    try:
                        block_list = rgetattr(component, attr)
                    except AttributeError:
                        block_list = None
                    if block_list is not None:
                        for i in range(len(block_list)):
                            block_list[i] = torch.compile(block_list[i], mode=mode, dynamic=dynamic)
                        compiled_any = True
                if not compiled_any:
                    setattr(self.pipe, component_name, torch.compile(component, mode=mode, dynamic=dynamic))
            else:
                setattr(self.pipe, component_name, torch.compile(component, mode=mode, dynamic=dynamic))
        compile_args = copy.deepcopy(input_args)
        warmup_steps = self._get_compile_warmup_steps(input_args)
        if warmup_steps is not None:
            compile_args["num_inference_steps"] = warmup_steps
        self._run_timed_pipe(compile_args)


    def run(self, input_args: dict) -> Tuple[DiffusionOutput, list]:
        """ Run the model with given input arguments and return output and timings """
        self._validate_args(input_args)
        input_args = self._split_prompts_for_dp(input_args)
        timings = []
        output: DiffusionOutput = None

        if self.config.warmup_calls:
            warmup_args = copy.deepcopy(input_args)
            if self.config.batch_size and isinstance(warmup_args.get("prompt"), list):
                warmup_args["prompt"] = warmup_args["prompt"][: self.config.batch_size]
            self._run_warmup_calls(warmup_args)

        inference_start = torch.cuda.Event(enable_timing=True)
        inference_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        inference_start.record()
        for iteration in range(self.config.num_iterations):
            log(f"Running iteration {iteration + 1}/{self.config.num_iterations}")

            if self.config.batch_size: # Run in batched mode
                output, batch_timings = self._run_pipe_batched(input_args)
                timings += batch_timings
            else: # Run all in one go
                output, timing = self._run_timed_pipe(input_args)
                timings.append(timing)
                log(f"Iteration {iteration + 1} completed in {timing:.2f}s")

        inference_end.record()
        torch.cuda.synchronize()

        output = self._gather_dp_outputs(output)

        if len(timings) > 1:
            timings.pop(0) # Remove first timing for more accurate average # TODO: fix
        log(f"Average time over {self.config.num_iterations} runs: {sum(timings) / len(timings):.2f}s")
        log(f"Total time spent: {inference_start.elapsed_time(inference_end) / 1000:.2f}s")

        return output, timings

    def _run_pipe_batched(self, input_args: dict) -> Tuple[List[DiffusionOutput], list]:
        """ Run the pipeline in batches """
        batch_size = self.config.batch_size
        all_prompts = input_args["prompt"]
        timings = []
        all_outputs = []
        batch_count = len(all_prompts) // batch_size + (1 if len(all_prompts) % batch_size != 0 else 0)

        for batch_index in range(0, batch_count):
            batch_args = copy.deepcopy(input_args)
            prompts = batch_args["prompt"][batch_index*batch_size:(batch_index+1)*batch_size]
            batch_args["prompt"] = prompts

            log(f"Processing batch {batch_index} with prompts {batch_index*batch_size} to {(batch_index+1)*batch_size}")
            output, timing = self._run_timed_pipe(batch_args)
            timings.append(timing)
            all_outputs.append(output)
            log(f"Batch {batch_index} completed in {timing:.2f}s")

        return DiffusionOutput.from_outputs(all_outputs, self.settings.model_output_type), timings

    def _run_warmup_calls(self, input_args: dict) -> None:
        """ Run initial warmup calls if specified """
        if self.config.warmup_calls:
            log(f"Warming up model with {self.config.warmup_calls} calls...")
            for iteration in range(self.config.warmup_calls):
                log(f"Warmup iteration {iteration + 1}/{self.config.warmup_calls}")
                self._run_timed_pipe(input_args)
            log(f"Warmup complete.")

    def profile(self, input_args: dict) -> Tuple[DiffusionOutput, list, torch.profiler.profiler.profile]:
        """ Profile the model execution """
        self._validate_args(input_args)
        input_args = self._split_prompts_for_dp(input_args)

        schedule = torch.profiler.schedule(
            wait=self.config.profile_wait,
            warmup=self.config.profile_warmup,
            active=self.config.profile_active,
        )
        num_repetitions = self.config.profile_wait + self.config.profile_warmup + self.config.profile_active

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
            record_shapes=True,
            with_stack=False,
        ) as profile_object:
            for iteration in range(num_repetitions):
                log(f"Profiling iteration {iteration + 1}/{num_repetitions}")
                with record_function("model_inference"):
                    if self.config.batch_size: # Run in batched mode
                        output, batch_timings = self._run_pipe_batched(input_args)
                        timing = sum(batch_timings)
                    else: # Run all in one go
                        output, timing = self._run_timed_pipe(input_args)
                profile_object.step()
                log(f"Profiling iteration {iteration + 1} completed in {timing:.2f}s")

        output = self._gather_dp_outputs(output)

        return output, [], profile_object

    def preprocess_args(self, input_args: dict) -> dict:
        """ Preprocess input arguments before passing them to the model """
        args = copy.deepcopy(input_args)

        # Apply model specific default input values
        for default_key, _ in DefaultInputValues.__annotations__.items():
            if args.get(default_key, None) is None:
                default_value = getattr(self.default_input_values, default_key)
                if default_value is not None:
                    args[default_key] = default_value
                    log(f"Parameter '{default_key}' not specified. Using model-specific default value: {default_value}")

        # Dataset to prompts
        if input_args.get("dataset_path", None):
            args["prompt"] = load_dataset_prompts(input_args["dataset_path"])

        negative_prompt = args.get("negative_prompt")
        if negative_prompt and isinstance(negative_prompt, list) and len(negative_prompt) == 1:
            args["negative_prompt"] = negative_prompt[0]

        args = self._preprocess_args_images(args)
        return args

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess image inputs if necessary """
        self._validate_args(input_args)
        images = [load_image(path) for path in input_args.get("input_images", [])]
        input_args["input_images"] = images
        return input_args

    def save_output(self, output: DiffusionOutput) -> None:
        """ Saves the output based on its type """
        # Assumes output only has images or videos, not both
        if output.images:
            for image_index, (image, pipe_args) in enumerate(output.get_outputs()):
                output_name = self.get_output_name(pipe_args)
                output_path = f"{self.config.output_directory}/{output_name}_{image_index}.png"
                image.save(output_path)
                log(f"Output image saved to {output_path}")
        elif output.videos:
            for video_index, (video, pipe_args) in enumerate(output.get_outputs()):
                output_name = self.get_output_name(pipe_args)
                output_path = f"{self.config.output_directory}/{output_name}_{video_index}.mp4"
                export_to_video(video, output_path, fps=self.settings.fps)
                log(f"Output video saved to {output_path}")
        else:
            raise NotImplementedError(f"No output to save.")

    def save_timings(self, timings: list) -> None:
        timing_file_name = f"{self.config.output_directory}/timings.json"
        with open(timing_file_name, "w") as timing_file:
            json.dump(timings, timing_file, indent=2)
        log(f"Timings saved to {self.config.output_directory}/timings.json")

    def save_profile(self, profile: torch.profiler.profiler.profile) -> None:
        profile_file = f"{self.config.output_directory}/profile_trace_rank_{get_world_group().rank}.json.gz"
        profile.export_chrome_trace(profile_file)
        log(f"Profile trace saved to {profile_file}", log_from_all_processes=True)

    def _run_timed_pipe(self, input_args: dict) -> Tuple[DiffusionOutput, float]:
        """ Run a a full pipeline with timing information """

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start.record()
        out = self._run_pipe(input_args)
        end.record()

        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000  # Convert to seconds
        return out, elapsed_time

    def get_output_name(self, input_args) -> str:
        """ Generate a unique output name based on model and config """
        use_compile = self.config.use_torch_compile
        ulysses_degree = self.config.ulysses_degree or 1
        ring_degree = self.config.ring_degree or 1
        height = input_args["height"]
        width = input_args["width"]
        name = f"{self.settings.output_name}_u{ulysses_degree}r{ring_degree}_tc_{use_compile}_{height}x{width}"
        if self.config.task:
            name += f"_{self.config.task}"
        return name

    def _apply_fp8_override_cli_from_config(self, config: xFuserArgs) -> None:
        """Apply optional CLI FP8 override patterns (per-slot) into ModelSettings."""

        def _parse_csv_patterns(raw: Optional[str]) -> Optional[Tuple[str, ...]]:
            if raw is None or not raw.strip():
                return None
            patterns = tuple(p.strip() for p in raw.split(",") if p.strip())
            return patterns or None

        if config.fp8_precision_override_prefix_patterns is not None:
            self.settings.fp8_precision_overrides = _parse_csv_patterns(
                config.fp8_precision_override_prefix_patterns
            )
        if config.fp8_precision_override_suffix_patterns is not None:
            self.settings.fp8_precision_override_suffixes = _parse_csv_patterns(
                config.fp8_precision_override_suffix_patterns
            )

    def _post_load_and_state_initialization(self, input_args: dict) -> None: ##TODO: should this be renamed?
        """ Hook for any post model-load and state initialization """

        local_rank = get_world_group().local_rank
        # Log FP8 precision overrides once here rather than per module/block in the
        # quantization and FSDP-sharding loops below (avoids duplicate log spam).
        if self.config.use_fp4_gemms:
            self._log_fp8_overrides(
                self.settings.fp8_precision_overrides,
                self.settings.fp8_precision_override_suffixes,
            )
        # FSDP path handles device placement and quantization (per-block for FSDP2).
        if self.config.fully_shard_degree > 1:
            from .loading.shard import shard_pipeline_components
            shard_pipeline_components(self)
        else:
            offload_requested = (
                self.config.enable_model_cpu_offload
                or self.config.enable_sequential_cpu_offload
                or self.config.enable_group_cpu_offload
            )
            fill_eager = getattr(
                getattr(self, "_loader", None),
                "fill_eager_transformers",
                None,
            )
            if fill_eager is not None:
                fill_eager()
            # Replicated multi-GPU: rank0's real bf16 weights are broadcast to peers' meta components
            # (GPU->GPU) and fp8-quantized per component in place. Bounds VRAM to one bf16 component.
            # The AITER walk below then no-ops (components are already fp8).
            if self._replicated_broadcast_load():
                self._loader.broadcast_fill_replicated(offload_requested)
            # AITER converts layer-by-layer CPU→GPU before pipe.to; torchao
            # converts after placement. Both use the selected backend adapter.
            adapter = self.fp8_backend
            if adapter is not None and adapter.converts_before_device_move:
                for module_name in self.fp8.module_list():
                    convert, filter_fn = _conversion_filter(
                        module_name,
                        getattr(self, "_fp8_streaming_targets", ()),
                    )
                    if not convert:
                        continue
                    convert_kwargs = {}
                    if filter_fn is not None:
                        convert_kwargs["filter_fn"] = filter_fn
                    replaced = adapter.convert_module(
                        rgetattr(self.pipe, module_name),
                        device=f"cuda:{local_rank}",
                        offload_to_cpu=offload_requested,
                        **convert_kwargs,
                    )
                    if replaced:
                        log(
                            f"Quantized {replaced} layers in {module_name} "
                            f"to FP8 ({adapter.storage_semantics})."
                        )
                    else:
                        log(f"{module_name} already FP8 (streamed quantize-on-load); post-load walk no-op.")
            if not offload_requested:
                self.pipe = self.pipe.to(f"cuda:{local_rank}")
            if self.config.use_fp4_gemms:
                if _is_cuda():
                    self._setup_nvfp4_gemms(local_rank=local_rank)
                else:
                    self._setup_mxfp4_gemms(local_rank=local_rank)
            # FP4 setup also owns its explicit hybrid FP8 path and any declared FP8-only modules.
            # Running the generic walk afterwards would re-quantize inside the hybrid wrappers.
            if (
                adapter is not None
                and not adapter.converts_before_device_move
                and not self.config.use_fp4_gemms
            ):
                for module_name in self.fp8.module_list():
                    component_name = module_name.partition(".")[0]
                    convert, filter_fn = _conversion_filter(
                        module_name,
                        getattr(self, "_fp8_streaming_targets", ()),
                    )
                    if not convert:
                        continue
                    if (
                        component_name.startswith("transformer")
                        and component_name
                        not in self._fp8_descriptor_components
                    ):
                        log(
                            "Transformer quantization: requested=fp8, "
                            f"backend={adapter.backend.value}, "
                            f"storage={adapter.storage_semantics}, "
                            "materialization=post_load; fallback=runner did "
                            "not use the transformer construction seam"
                        )
                        self._fp8_descriptor_components.add(component_name)
                    convert_kwargs = {}
                    if filter_fn is not None:
                        convert_kwargs["filter_fn"] = filter_fn
                    adapter.convert_module(
                        rgetattr(self.pipe, module_name),
                        device=f"cuda:{local_rank}",
                        **convert_kwargs,
                    )
            if self.config.use_int8_gemms:
                adapter = self.format_backend
                for module_name in self.settings.int8_gemm_module_list:
                    component_name = module_name.partition(".")[0]
                    convert, filter_fn = _conversion_filter(
                        module_name,
                        getattr(
                            self, "_quantization_streaming_targets", ()
                        ),
                    )
                    if not convert:
                        continue
                    if component_name not in self._quantization_descriptor_components:
                        from .loading.format_backends import (
                            prepare_native_transformer_format_load,
                        )

                        descriptor = prepare_native_transformer_format_load(
                            adapter,
                            component_name=component_name,
                            targets=self._format_targets_for(component_name),
                            stream_quant=False,
                        ).descriptor
                        log(descriptor.log_message())
                        self._quantization_descriptor_components.add(
                            component_name
                        )
                    convert_kwargs = {}
                    if filter_fn is not None:
                        convert_kwargs["filter_fn"] = filter_fn
                    adapter.convert_module(
                        rgetattr(self.pipe, module_name),
                        device=f"cuda:{local_rank}",
                        **convert_kwargs,
                    )

        if self.config.use_hybrid_attn_schedule:
            self._setup_hybrid_attn_schedule(input_args)

        if self.config.use_hybrid_gemm_schedule:
            self._setup_hybrid_gemm_schedule(input_args)

        if self.config.use_vae_channels_last_format:
            self._convert_vae_to_channels_last()

    def _log_fp8_overrides(self, prefixes, suffixes) -> None:
        """Log the FP8 precision-override patterns (prefix and suffix) consistently."""
        if prefixes:
            log(
                "The following layers will be quantized to FP8, to maintain output quality: "
                f"{prefixes} (prefix match)"
            )
        if suffixes:
            log(
                "The following layers will be quantized to FP8, to maintain output quality: "
                f"{suffixes} (suffix match)"
            )


    def _setup_mxfp4_gemms(self, local_rank):
        adapter = self.format_backend
        for module_name in self.settings.fp4_gemm_module_list:
            component_name = module_name.partition(".")[0]
            convert, filter_fn = _conversion_filter(
                module_name,
                getattr(self, "_quantization_streaming_targets", ()),
            )
            if not convert:
                continue
            # Certain models benefit from a hybrid quantization strategy: applying FP8 to
            # a number of transformer blocks while using FP4 for others. This mixed-precision
            # approach balances performance and output quality better than uniform quantization.
            if component_name not in self._quantization_descriptor_components:
                from .loading.format_backends import (
                    prepare_native_transformer_format_load,
                )

                descriptor = prepare_native_transformer_format_load(
                    adapter,
                    component_name=component_name,
                    targets=self._format_targets_for(component_name),
                    stream_quant=True,
                    precision_prefixes=(
                        self.settings.fp8_precision_overrides or ()
                    ),
                    precision_suffixes=(
                        self.settings.fp8_precision_override_suffixes or ()
                    ),
                    hybrid=self.config.use_hybrid_gemm_schedule,
                ).descriptor
                log(descriptor.log_message())
                self._quantization_descriptor_components.add(component_name)
            module = rgetattr(self.pipe, module_name)
            convert_kwargs = {}
            if filter_fn is not None:
                convert_kwargs["filter_fn"] = filter_fn
            adapter.convert_module(
                module,
                fp8_layers=self.settings.fp8_precision_overrides,
                fp8_suffix_layers=self.settings.fp8_precision_override_suffixes,
                hybrid=self.config.use_hybrid_gemm_schedule,
                device=f"cuda:{local_rank}",
                **convert_kwargs,
            )
        self._setup_fp8_only_gemm_modules(local_rank)

    def _setup_fp8_only_gemm_modules(self, local_rank):
        # Any module specified in fp8 gemms modules list and not specified in fp4 gemms module list,
        # will be quantized to fp8, this is specially beneficial for MoE models like Wan2.2,
        # where the low-noise transformer should use FP8 quantization.
        # This transformer generates fine details and requires higher precision to maintain quality.
        fp4_modules = set(self.settings.fp4_gemm_module_list or ())
        fp8_only_modules = [
            name
            for name in self.fp8.module_list()
            if not any(
                module_path_is_covered(name, fp4_module)
                for fp4_module in fp4_modules
            )
        ]
        if not fp8_only_modules:
            return
        adapter = self.blockwise_fp8_backend
        for module_name in fp8_only_modules:
            excluded_paths = fp4_modules | set(
                getattr(self, "_quantization_streaming_targets", ())
            ) | set(
                getattr(self, "_fp8_streaming_targets", ())
            )
            convert, filter_fn = _conversion_filter(
                module_name, excluded_paths
            )
            if not convert:
                continue
            log(f"Quantizing linear layers in {module_name} to FP8...")
            module = rgetattr(self.pipe, module_name)
            convert_kwargs = {}
            if filter_fn is not None:
                convert_kwargs["filter_fn"] = filter_fn
            adapter.convert_module(
                module,
                device=f"cuda:{local_rank}",
                **convert_kwargs,
            )

    def _setup_nvfp4_gemms(self, local_rank):
        adapter = self.format_backend
        for module_name in self.settings.fp4_gemm_module_list:
            component_name = module_name.partition(".")[0]
            convert, filter_fn = _conversion_filter(
                module_name,
                getattr(self, "_quantization_streaming_targets", ()),
            )
            if not convert:
                continue
            if component_name not in self._quantization_descriptor_components:
                from .loading.format_backends import (
                    prepare_native_transformer_format_load,
                )

                descriptor = prepare_native_transformer_format_load(
                    adapter,
                    component_name=component_name,
                    targets=self._format_targets_for(component_name),
                    stream_quant=False,
                    precision_prefixes=(
                        self.settings.fp8_precision_overrides or ()
                    ),
                    precision_suffixes=(
                        self.settings.fp8_precision_override_suffixes or ()
                    ),
                    hybrid=self.config.use_hybrid_gemm_schedule,
                ).descriptor
                log(descriptor.log_message())
                self._quantization_descriptor_components.add(component_name)
            module = rgetattr(self.pipe, module_name)
            convert_kwargs = {}
            if filter_fn is not None:
                convert_kwargs["filter_fn"] = filter_fn
            adapter.convert_module(
                module,
                fp8_layers=self.settings.fp8_precision_overrides,
                fp8_suffix_layers=self.settings.fp8_precision_override_suffixes,
                hybrid=self.config.use_hybrid_gemm_schedule,
                device=f"cuda:{local_rank}",
                **convert_kwargs,
            )
        self._setup_fp8_only_gemm_modules(local_rank)

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        return 1

    def _setup_hybrid_attn_schedule(self, input_args: dict) -> None:
        """
        Setup hybrid attention schedule: high precision backend at start/end, low precision backend in the middle,
        or a custom schedule provided by the user.
        """
        if input_args["num_hybrid_attn_high_precision_steps"] is None:
            raise ValueError("You must provide 'num_hybrid_attn_high_precision_steps' to use the hybrid attention schedule.")
        multiplier = self._calculate_hybrid_attention_step_multiplier(input_args)
        total_steps = input_args["num_inference_steps"] * multiplier
        if self.config.hybrid_attn_low_precision_backend is None or self.config.hybrid_attn_high_precision_backend is None:
            attention_schedule = AttentionSchedule.from_comma_delimited_string(self.config.hybrid_attn_schedule)
            if attention_schedule.total_steps != total_steps:
                raise ValueError(f"Hybrid attention schedule total steps {attention_schedule.total_steps} does not match input steps {total_steps} (input_args['num_inference_steps']={input_args['num_inference_steps']}, multiplier={multiplier}).")
        else:
            num_high_precision_steps = input_args["num_hybrid_attn_high_precision_steps"] * multiplier
            low_precision_backend = AttentionBackendType[self.config.hybrid_attn_low_precision_backend.upper()]
            high_precision_backend = AttentionBackendType[self.config.hybrid_attn_high_precision_backend.upper()]
            attention_schedule = create_hybrid_attn_schedule(
                num_high_precision_steps=num_high_precision_steps,
                low_precision_backend=low_precision_backend,
                high_precision_backend=high_precision_backend,
                total_steps=total_steps,
                check_compat=get_runtime_state()._check_if_backend_compatible_with_current_configuration,
            )

        log("Enabling hybrid attention schedule")
        log(f"Hybrid attention schedule: {attention_schedule.backends}", debug=True)
        get_runtime_state().set_attention_schedule(attention_schedule, total_steps=total_steps)

    def _setup_hybrid_gemm_schedule(self, input_args: dict) -> None:
        """
        Setup hybrid GEMM schedule: high precision FP8 GEMMs at start/end, MXFP4 GEMMs in the middle.
        """
        if input_args["num_hybrid_gemm_high_precision_steps"] is None:
            raise ValueError("You must provide 'num_hybrid_gemm_high_precision_steps' to use the hybrid GEMM schedule.")
        multiplier = self._calculate_hybrid_attention_step_multiplier(input_args)
        total_steps = input_args["num_inference_steps"] * multiplier
        num_high_precision_steps = input_args["num_hybrid_gemm_high_precision_steps"] * multiplier

        gemm_schedule = create_hybrid_gemm_schedule(
            num_high_precision_steps=num_high_precision_steps,
            total_steps=total_steps,
        )

        log("Enabling hybrid GEMM schedule")
        log(f"Hybrid GEMM schedule (high precision=True): {gemm_schedule.use_high_precision_schedule}", debug=True)
        get_runtime_state().set_gemm_schedule(gemm_schedule, total_steps=total_steps)

    def _convert_vae_to_channels_last(self) -> None:
        """ Convert the VAE to channels last """
        convert_model_convs_to_channels_last(self.pipe.vae)

        original_decode = self.pipe.vae.decode
        memory_format = torch.channels_last if self.settings.model_output_type == "image" else torch.channels_last_3d

        @functools.wraps(original_decode)
        def decode_wrapper(*args, **kwargs):
            if args:
                args = list(args)
                args[0] = args[0].to(memory_format=memory_format)
                args = tuple(args)
            elif "z" in kwargs:
                kwargs["z"] = kwargs["z"].to(memory_format=memory_format)
            output = original_decode(*args, **kwargs)
            return output

        self.pipe.vae.decode = decode_wrapper

    def _make_generator(self, seed: int) -> torch.Generator:
        """Generator on the pipe's execution device (cuda normally, cpu under offload).

        randn_tensor requires the generator device to match the tensor's; hardcoding cuda
        breaks when CPU offload runs the pipeline on cpu.
        """
        return torch.Generator(device=self.pipe._execution_device).manual_seed(seed)

    @abc.abstractmethod
    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        """ Execute the pipeline. Must be implemented by subclasses. """
        pass

    @abc.abstractmethod
    def _load_model(self) -> DiffusionPipeline:
        """ Load the model. Must be implemented by subclasses. """
        pass

    def _split_prompts_for_dp(self, input_args: dict) -> dict:
        """Shard prompts across data-parallel groups so each group processes a subset."""
        if self.config.data_parallel_degree == 1:
            return input_args

        dp_world_size = get_data_parallel_world_size()
        dp_rank = get_data_parallel_rank()
        prompts = input_args.get("prompt")
        negative_prompts = input_args.get("negative_prompt")

        if isinstance(prompts, str):
            log(f"Single prompt with dp_world_size={dp_world_size}: all DP groups will process the same prompt.")
            return input_args

        if len(prompts) < dp_world_size:
            raise ValueError(
                f"Number of prompts ({len(prompts)}) is less than data_parallel_world_size ({dp_world_size}). "
            )

        local_prompts = prompts[dp_rank::dp_world_size]
        if isinstance(negative_prompts, list) and len(negative_prompts) != 1:
            local_negative_prompts = negative_prompts[dp_rank::dp_world_size]
        else:
            local_negative_prompts = negative_prompts
        log(f"Each DP group will process {len(local_prompts)} prompts out of {len(prompts)} total prompts.")

        split_args = copy.copy(input_args)
        split_args["prompt"] = local_prompts
        split_args["negative_prompt"] = local_negative_prompts
        return split_args

    def _gather_dp_outputs(self, output: DiffusionOutput) -> Optional[DiffusionOutput]:
        """
        Gathers DiffusionOutput objects from all DP groups onto the last rank.

        Within each SP group every rank holds an identical copy of the output
        Only the first rank in the SP group sends the real payload,
        the other ranks send None to keep the collective valid.

        """
        if self.config.data_parallel_degree == 1:
            return output

        world_group = get_world_group()
        last_rank = world_group.world_size - 1

        is_representative = get_sequence_parallel_rank() == 0 and get_classifier_free_guidance_rank() == 0
        send_obj = output if is_representative else None

        gather_list = [None] * world_group.world_size if world_group.rank == last_rank else None

        torch.distributed.gather_object(send_obj, gather_list, dst=last_rank)

        if world_group.rank == last_rank:
            real_outputs = [o for o in gather_list if o is not None]
            return DiffusionOutput.from_outputs(real_outputs, self.settings.model_output_type)
        return None

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments. Can be overridden by subclasses. """
        if input_args["prompt"] is None and input_args["dataset_path"] is None:
            raise ValueError("Either 'prompt' or 'dataset_path' must be provided in input arguments.")

        if self.settings.resolution_divisor:
            if (input_args["height"] % self.settings.resolution_divisor != 0 or input_args["width"] % self.settings.resolution_divisor != 0):
                raise ValueError(f"Model {self.settings.model_name} requires height and width to be divisible by {self.settings.resolution_divisor}.")

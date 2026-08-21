import abc
import torch
import copy
import json
from PIL.Image import Image
from typing import Callable, List, Optional, Tuple, Generator
from dataclasses import dataclass, field, replace
from torch.profiler import profile, record_function, ProfilerActivity
import diffusers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import numpy as np
from xfuser.compat import is_diffusers_import_error
from xfuser.config import xFuserArgs
from xfuser.envs import (
    PACKAGES_CHECKER,
    _is_hip,
    _is_cuda,
)
from xfuser.core.utils.runner_utils import (
    log,
    load_dataset_prompts,
    rgetattr,
)
from xfuser.model_executor.models.runner_models.vae_manager import (
    VAEManager,
    validate_vae_config,
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
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    LoadRoute,
)
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    apply_fp8_override_cli_to_settings,
)

packages_info = PACKAGES_CHECKER.get_packages_info()

MODEL_REGISTRY = {}

# Value for min_diffusers_version when a model needs diffusers symbols that no release
# ships yet, so the only way to run it is a source install of diffusers. Distinct from
# None, which means no particular floor is known.
DIFFUSERS_FROM_SOURCE = "source"


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
    AttentionBackendType.AITER_VSA,
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
    text_encoder_tp_degree: bool = False
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
    use_fp8_text_encoder: bool = False
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
    fp8_gemm_include_suffixes: Optional[Tuple[str, ...]] = None
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

    # torch.compile modes that run the graph under CUDA Graphs, whose outputs live in a fixed
    # buffer pool and are therefore only valid until the next replay.
    CUDAGRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})

    # Shared loading is opt-in; subclasses must declare verified routes explicitly.
    load_support: LoadSupport = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )
    capabilities: ModelCapabilities = ModelCapabilities()
    default_input_values: DefaultInputValues = DefaultInputValues()
    settings: ModelSettings = ModelSettings()
    model_output_type: str = ""
    fps: int = 0
    checkpoint_request_defaults: dict = {}

    # Lowest diffusers release this model is expected to run on, used only to name an
    # upgrade target when a load fails. It never gates a load, so a value above the
    # true minimum costs an over-stated recommendation and blocks nothing, while a
    # value below it is a bug that tests/core/test_diffusers_floors.py catches. Use
    # DIFFUSERS_FROM_SOURCE when the model's support has not been released yet, and
    # leave None when no floor is known.
    min_diffusers_version: Optional[str] = None

    def __init__(self, config: xFuserArgs) -> None:
        self.settings = copy.deepcopy(self.__class__.settings)
        self._customize_settings(config)
        self._vae_manager = VAEManager(config, self.capabilities, self.settings)
        self._validate_config(config)
        self._update_model_settings(config)
        self.config = config
        self.pipe = None
        from .loading.meta_load import ModelLoader

        self.loader = ModelLoader(self)

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
            apply_fp8_override_cli_to_settings(config, self.settings)
        te_targets = self.settings.fp8_text_encoder_module_list
        if te_targets and config.use_fp8_gemms and not config.use_fp8_text_encoder:
            # Said out loud because text-encoder FP8 is opt-in: an encoder left bf16 is otherwise
            # indistinguishable from --use_fp8_gemms failing to take effect.
            log(f"--use_fp8_gemms covers the transformer; {type(self).__name__}'s "
                f"{len(te_targets)} text-encoder target(s) stay bf16. Add --use_fp8_text_encoder "
                f"to quantize them too, for less memory at some risk to text conditioning.")

    def _load_model_checked(self) -> DiffusionPipeline:
        """Load the pipeline, reporting a missing diffusers symbol as a version problem.

        Runner modules import their pipelines and transformer wrappers inside
        _load_model, so a model too new for the installed diffusers still registers,
        and fails here instead. The error names the model and the symbol that is
        missing; min_diffusers_version, when set, adds where to get a diffusers
        that has it.
        """
        try:
            return self._load_model()
        except ImportError as e:
            if not is_diffusers_import_error(e):
                raise
            if self.min_diffusers_version == DIFFUSERS_FROM_SOURCE:
                remedy = (
                    "Support for this model has not landed in a diffusers release yet; "
                    "it needs diffusers installed from source."
                )
            elif self.min_diffusers_version:
                remedy = f"Requires diffusers>={self.min_diffusers_version}."
            else:
                remedy = "A newer diffusers is required."
            raise ImportError(
                f"{self.settings.model_name} is unavailable with diffusers "
                f"{diffusers.__version__}: {e}. {remedy}"
            ) from e

    def initialize(self, input_args: dict) -> None:
        """ Load the model pipeline """

        if not torch.distributed.is_initialized():
            log("Initializing distributed environment...")
            init_distributed_environment()

        self.loader.preflight(world_size=get_world_group().world_size)
        self.engine_config, _ = self.config.create_config()
        log("Loading model pipeline...")
        self.pipe = self._load_model_checked()

        log("Initializing runtime state...")
        initialize_runtime_state(self._get_runtime_state_pipeline(), self.engine_config)

        self._post_load_and_state_initialization(input_args)
        if self.config.use_parallel_vae:
            self._vae_manager.setup_parallel_vae(self._decoding_vaes())
        self._enable_options()

        if self.config.use_torch_compile:
            log("Torch.compile enabled. Warming up torch compiler ...")
            compile_input_args = copy.deepcopy(input_args)
            compile_input_args = self._split_prompts_for_dp(compile_input_args)
            if self.config.batch_size and isinstance(compile_input_args.get("prompt"), list):
                compile_input_args["prompt"] = compile_input_args["prompt"][: self.config.batch_size]
            self._compile_model(compile_input_args)

    def _local_onload_device(self) -> torch.device:
        """The device this rank offloads to and from, which is never implicitly cuda:0."""
        return torch.device(f"cuda:{get_world_group().local_rank}")

    def _enable_options(self) -> None:
        """ Enable model options based on config"""
        if getattr(self.config, "use_spargeattn_head_balance", False):
            log("Enabling Sparge block-sparse head balancing...")

        self._vae_manager.enable_options(self._decoding_vaes())

        if self.config.enable_group_cpu_offload:
            # block_level groups only top-level ModuleLists: fits compiled transformers
            # (blocks are top-level) and avoids the per-block-compile recompile storm that
            # leaf-level hooks trigger. Eager components nest their layers (e.g. Mistral-3 at
            # model.language_model.layers) where block_level cannot reach, leaving the whole
            # component in one unmatched group and OOMing; they use leaf_level, which recurses.
            from diffusers.hooks import apply_group_offloading
            log("Enabling group CPU offload (transformer block-level, others leaf-level, streamed)...")
            onload_device = self._local_onload_device()
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
            # Diffusers defaults the onload device to cuda:0, which every rank would
            # then share, so the first collective fails with a duplicate GPU. The
            # group offload above already names the local device; these must too.
            self.pipe.enable_sequential_cpu_offload(device=self._local_onload_device())
        elif self.config.enable_model_cpu_offload:
            log("Enabling model CPU offload...")
            self.pipe.enable_model_cpu_offload(device=self._local_onload_device())

    def _get_runtime_state_pipeline(self):
        return self.pipe


    def _decoding_vaes(self) -> List:
        """Forward staged VAE discovery to the VAE manager."""
        return self._vae_manager.decoding_vaes(
            [self.pipe, getattr(self, "second_pipe", None)]
        )

    def _validate_config(self, config: xFuserArgs) -> None:
        """ Validate if the model supports requested config """
        config._validate_gemm_quantization_flags()
        for key in ModelCapabilities.__annotations__.keys():
            config_value = getattr(config, key, None)  # Some config options might not be set in the CLI, such as support for specific attention backends.
            if isinstance(config_value, int) and not isinstance(config_value, bool):
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
            raise ValueError("Dataset path specified without batch size. Please specify batch size for dataset inference.")

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
        validate_vae_config(config, self.capabilities, self.settings)
        
        if config.distilled_transformer_path or config.distilled_transformer_2_path:
            if not self.capabilities.supports_distilled_weights:
                raise ValueError(f"Model {self.settings.model_name} does not support distilled_transformer_path or distilled_transformer_2_path params.")

    def _get_compile_mode(self) -> str:
        # Overrides should return "default" when PACKAGES_CHECKER._on_rdna4():
        # CUDA graphs are slow on RDNA4.
        return "default"  # TODO: Configurable

    def _get_compile_dynamic(self) -> Optional[bool]:
        return None  # torch default (auto)

    def _mark_cudagraph_steps(self, component: torch.nn.Module) -> None:
        """Tell CUDA Graphs where one inference step ends, so the next may reuse its buffers.

        Compiling a component blockwise makes every block its own graph segment, and a segment
        recorded on a later step copies its inputs from the previous block's output buffer. Without a
        step boundary the graph system still considers the earlier step's outputs live and refuses
        the read with "accessing tensor output of CUDAGraphs that has been overwritten by a
        subsequent run". Reaching it takes both halves: a runner that asks for reduce-overhead, and
        sharding, which is what turns one compiled transformer into a graph per block. A pre-hook
        rather than a wrapper because pre-hooks run before the compiled forward is entered.
        """
        if getattr(component, "_xfuser_marks_cudagraph_steps", False):
            return
        if not hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            return

        def _mark(module, args, kwargs):
            torch.compiler.cudagraph_mark_step_begin()

        component.register_forward_pre_hook(_mark, with_kwargs=True, prepend=True)
        component._xfuser_marks_cudagraph_steps = True

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
                if compiled_any and mode in self.CUDAGRAPH_COMPILE_MODES:
                    self._mark_cudagraph_steps(component)
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
            log("Warmup complete.")

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
            raise NotImplementedError("No output to save.")

    def save_timings(self, timings: list) -> None:
        timing_file_name = f"{self.config.output_directory}/timings.json"
        with open(timing_file_name, "w") as timing_file:
            json.dump(timings, timing_file, indent=2)
        log(f"Timings saved to {self.config.output_directory}/timings.json")

    def save_profile(self, profile: torch.profiler.profiler.profile) -> None:
        profile_file = f"{self.config.output_directory}/profile_trace_rank_{get_world_group().rank}.json.gz"
        profile.export_chrome_trace(profile_file)
        log(f"Profile trace saved to {profile_file}", log_from_all_processes=True)

    def prepare_run(self, input_args: dict) -> None:
        """Prepare model state before a pipeline invocation."""
        self._vae_manager.prepare_run(self._decoding_vaes(), input_args)

    def _run_timed_pipe(self, input_args: dict) -> Tuple[DiffusionOutput, float]:
        """ Run a a full pipeline with timing information """

        self.prepare_run(input_args)
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

    def _post_load_and_state_initialization(self, input_args: dict) -> None: ##TODO: should this be renamed?
        """ Hook for any post model-load and state initialization """

        self.loader.materialize_pipeline()

        if self.config.use_hybrid_attn_schedule:
            self._setup_hybrid_attn_schedule(input_args)

        if self.config.use_hybrid_gemm_schedule:
            self._setup_hybrid_gemm_schedule(input_args)

        if self.config.use_vae_channels_last_format:
            self._convert_vae_to_channels_last()

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
        """Forward channels-last conversion for subclass compatibility."""
        self._vae_manager.convert_to_channels_last(self._decoding_vaes())

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

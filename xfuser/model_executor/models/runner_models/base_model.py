import abc
import torch
import copy
import argparse
import json
import functools
from PIL.Image import Image
from typing import Callable, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
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
from xfuser.core.distributed.parallel_state import get_fs_group
from xfuser.core.utils.checkpoint_io import host_mem_gb
from xfuser.core.utils.runner_utils import (
    log,
    load_dataset_prompts,
    quantize_linear_layers_to_int8,
    quantize_linear_layers_to_fp8,
    quantize_linear_layers_to_fp8_blockscale,
    quantize_linear_layers_to_fp4,
    quantize_linear_layers_to_nvfp4,
    convert_model_convs_to_channels_last,
    _use_aiter_fp8_rdna4,
    rgetattr,
)

from xfuser.model_executor.quant import AiterFp8BlockScaleConfig  # noqa: F401  (import registers the quantizer)
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
    shard_component,
)
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.attention_schedule import AttentionSchedule, create_hybrid_attn_schedule, create_hybrid_gemm_schedule


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
    default_input_values: DefaultInputValues = DefaultInputValues()
    settings: ModelSettings = ModelSettings()
    model_output_type: str = ""
    fps: int = 0

    def __init__(self, config: xFuserArgs) -> None:
        self.settings = copy.deepcopy(self.settings)
        self._customize_settings(config)
        self._validate_config(config)
        self._update_model_settings(config)
        self.config = config
        self.pipe = None

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
        self._gate_te_fp8_to_rdna4()

    def _gate_te_fp8_to_rdna4(self) -> None:
        """Keep text-encoder entries in fp8_gemm_module_list only on RDNA4.

        TE fp8 is worthwhile only via the AITER block-scale path (RDNA4). Off RDNA4 the
        list is consumed by the torchao / fp4 walks, which would then quantize a text
        encoder we don't want touched, so strip every non-transformer entry there. Runs
        after _customize_settings (which may rebuild the list, e.g. Wan2.2 dual), so the
        gate covers runtime-assigned lists too.
        """
        if _is_hip() and PACKAGES_CHECKER._on_rdna4():
            return
        lst = self.settings.fp8_gemm_module_list
        if not lst:
            return
        self.settings.fp8_gemm_module_list = [
            m for m in lst if m.split(".", 1)[0].startswith("transformer")
        ]

    def initialize(self, input_args: dict) -> None:
        """ Load the model pipeline """

        if not torch.distributed.is_initialized():
            log("Initializing distributed environment...")
            init_distributed_environment()

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

    def _aiter_fp8_active(self) -> bool:
        """True when RDNA4 AITER FP8 quantization applies (fp8 gemms requested + supported)."""
        return bool(self.config.use_fp8_gemms and _use_aiter_fp8_rdna4())

    def _fp8_stream_quant_config(
        self, attr_prefix: str = "transformer"
    ) -> Optional[AiterFp8BlockScaleConfig]:
        """Config for streaming FP8 quantize-on-load, or None when not applicable.

        Passing this config to the transformer's from_pretrained quantizes each weight as it
        streams off disk, so the full bf16 transformer never materializes (peak ~= one weight
        + accumulating fp8) — cheaper than loading bf16 then quantizing in the _post_load walk.
        Targets exactly the fp8_gemm_module_list sub-modules for this transformer (stripping
        the pipe-level prefix, e.g. "transformer." / "transformer_2."), so the later
        _post_load AITER walk is a safe no-op (leaves are already fp8, not nn.Linear).
        Applies to both single-GPU and FSDP: the streamed fp8 module is what FSDP shards,
        so the per-block quantize_fn is a no-op on those leaves.
        """
        if not self._aiter_fp8_active():
            return None
        targets = self._fp8_targets_for_component(attr_prefix)
        if not targets:
            return None
        return AiterFp8BlockScaleConfig(target_modules=targets)

    def _te_pipeline_quant_config(self):
        """PipelineQuantizationConfig routing the AITER FP8 streaming quantizer to the pipeline's
        transformers sub-models (text encoders), or None when not applicable.

        A text encoder is a transformers model loaded by the diffusers pipeline; streaming it to
        fp8 (instead of loading full bf16 then quantizing post-load) is the load-time host-RAM win
        on multi-GPU FSDP, where every node-local rank would otherwise hold a full bf16 copy.
        Groups fp8_gemm_module_list entries by pipeline component, excluding the transformer(s)
        (the DiT streams via _fp8_stream_quant_config), and keys the mapping by whatever the pipe
        names each component. The later _post_load AITER walk is a safe no-op on those leaves.
        """
        if not self._aiter_fp8_active():
            return None
        component_targets: dict[str, list[str]] = {}
        for entry in (self.settings.fp8_gemm_module_list or []):
            component, _, rest = entry.partition(".")
            if not rest or component.startswith("transformer"):
                continue
            component_targets.setdefault(component, []).append(rest)
        if not component_targets:
            return None
        from diffusers.quantizers import PipelineQuantizationConfig
        from xfuser.model_executor.quant import AiterFp8BlockScaleTEConfig
        return PipelineQuantizationConfig(
            quant_mapping={
                component: AiterFp8BlockScaleTEConfig(target_modules=targets)
                for component, targets in component_targets.items()
            },
        )

    def _memory_efficient_fsdp_load(self) -> bool:
        """True when the memory-efficient sharded (meta-init + rank0-broadcast) load path is on."""
        return bool(self.config.memory_efficient_sharding and self.config.fully_shard_degree > 1)

    def _replicated_broadcast_load(self) -> bool:
        """Auto-on replicated multi-GPU load: the model fits one GPU and is replicated across ranks
        (pure sequence/CFG/data parallel), so every rank would otherwise from_pretrained a full CPU
        copy -> host RAM = N x model -> cgroup OOM. Instead rank0 loads real weights to GPU, peers
        build on meta and receive every param/buffer via a GPU->GPU broadcast (host peak = 1x).
        RDNA4-only. Excludes weight-splitting parallelism (FSDP/pipefusion/TP), where per-rank
        weights differ and a broadcast would be wrong (FSDP has its own meta-load path)."""
        return bool(
            PACKAGES_CHECKER._on_rdna4()
            and get_world_group().world_size > 1
            and self.config.fully_shard_degree == 1
            and self.config.pipefusion_parallel_degree == 1
            and self.config.tensor_parallel_degree == 1
        )

    def _fp8_targets_for_component(self, component_name: str) -> list[str]:
        """fp8_gemm_module_list entries under this component, with the pipe-level prefix stripped
        (e.g. "text_encoder.model.language_model.layers" -> "model.language_model.layers")."""
        prefix = f"{component_name}."
        return [
            m[len(prefix):] for m in (self.settings.fp8_gemm_module_list or [])
            if m.startswith(prefix)
        ]

    def _component_wants_fp8(self, component_name: str) -> bool:
        return bool(self._aiter_fp8_active() and self._fp8_targets_for_component(component_name))

    @functools.cached_property
    def _sharder(self):
        """Lazy MemoryEfficientSharder bound to this model (meta-init + rank0-broadcast load)."""
        from xfuser.model_executor.models.runner_models.meta_load import MemoryEfficientSharder
        return MemoryEfficientSharder(self)

    def _build_transformer(self, wrapper_cls, subfolder: str = "transformer", init_kwargs: dict | None = None, stream_quant: bool = True):
        """Load the transformer for a pipeline. On the memory-efficient FSDP path (multi-GPU
        fully-shard) build it on meta — weights are then streamed per block from disk on every rank
        during sharding, so the full model never materializes on host or any single GPU. Otherwise
        load via from_pretrained (single-GPU / non-FSDP path, unchanged).

        init_kwargs: extra wrapper __init__ args (e.g. wan's attention_kwargs) forwarded on both paths.
        stream_quant: on the non-meta path, stream-quantize to fp8 when True (models that already load
        fp8 today); False keeps the plain bf16 load (models that load bf16 today). The meta path always
        quantizes per block from fp8_gemm_module_list, so stream_quant only gates the non-meta config.
        """
        if self._memory_efficient_fsdp_load():
            return self._sharder.build_meta_transformer(wrapper_cls, subfolder, init_kwargs)
        # Replicated broadcast load: build on meta on ALL ranks. Weights are streamed per block from
        # disk on rank0 and broadcast GPU->GPU, then fp8-quantized per block (broadcast_fill_replicated),
        # so the full bf16 transformer never materializes on host or any single GPU.
        if self._replicated_broadcast_load():
            return self._sharder.build_meta_transformer(wrapper_cls, subfolder, init_kwargs)
        return wrapper_cls.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder=subfolder,
            quantization_config=(
                self._fp8_stream_quant_config(subfolder) if stream_quant else None
            ),
            **(init_kwargs or {}),
        )

    def _meta_te_kwargs(self):
        """Build text-encoder(s) on meta for the memory-efficient FSDP load path.

        Returns (pipe_component_kwargs, te_quant_config). On the meta path the kwargs carry meta
        modules to hand to the pipeline's from_pretrained (so it skips loading those components)
        and te_quant is None — the pipe does not stream the TE; instead the meta module (fp8 when
        targeted, else bf16) is filled by the rank0-broadcast sharded load, then FSDP-sharded
        (CPU-offloaded). On the normal path returns ({}, self._te_pipeline_quant_config()). The
        transformer is unaffected either way; it keeps its own streaming-fp8 from_pretrained path.
        """
        normal = ({}, self._te_pipeline_quant_config())
        if self._replicated_broadcast_load():
            return self._sharder.meta_te_kwargs_replicated(normal)
        if not self._memory_efficient_fsdp_load():
            return normal
        return self._sharder.meta_te_kwargs(normal)

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
            low_cpu_mem_usage = PACKAGES_CHECKER._on_rdna4()
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
                    low_cpu_mem_usage=low_cpu_mem_usage,
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

        if config.use_int8_gemms:
            if config.use_fp8_gemms or config.use_fp4_gemms:
                raise ValueError("Cannot use int8 gemms with fp8 or fp4 gemms.")
            if _is_hip():
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
            self._shard_model_with_fsdp()
        else:
            offload_requested = (
                self.config.enable_model_cpu_offload
                or self.config.enable_sequential_cpu_offload
                or self.config.enable_group_cpu_offload
            )
            # Replicated multi-GPU: rank0's real bf16 weights are broadcast to peers' meta components
            # (GPU->GPU) and fp8-quantized per component in place. Bounds VRAM to one bf16 component.
            # The AITER walk below then no-ops (components are already fp8).
            if self._replicated_broadcast_load():
                self._sharder.broadcast_fill_replicated(offload_requested)
            # AITER FP8: quantizes layer-by-layer CPU→GPU individually before pipe.to(cuda).
            # All other quant paths (FP4, torchao FP8) need weights on GPU first.
            if self._aiter_fp8_active():
                for module_name in self.settings.fp8_gemm_module_list:
                    replaced = quantize_linear_layers_to_fp8_blockscale(
                        rgetattr(self.pipe, module_name), device=f"cuda:{local_rank}",
                        offload_to_cpu=offload_requested,
                    )
                    if replaced:
                        log(f"Quantized {replaced} layers in {module_name} to FP8 block-scale (AITER).")
                    else:
                        log(f"{module_name} already FP8 (streamed quantize-on-load); post-load walk no-op.")
            if not offload_requested:
                self.pipe = self.pipe.to(f"cuda:{local_rank}")
            if self.config.use_fp4_gemms:
                if _is_cuda():
                    self._setup_nvfp4_gemms(local_rank=local_rank)
                else:
                    self._setup_mxfp4_gemms(local_rank=local_rank)
            if self.config.use_fp8_gemms and not _use_aiter_fp8_rdna4():
                for module_name in self.settings.fp8_gemm_module_list:
                    log(f"Quantizing {module_name} to FP8 (torchao)...")
                    quantize_linear_layers_to_fp8(rgetattr(self.pipe, module_name), device=f"cuda:{local_rank}")
            if self.config.use_int8_gemms:
                for module_name in self.settings.int8_gemm_module_list:
                    log(f"Quantizing {module_name} to W8A8 INT8 (torchao)...")
                    quantize_linear_layers_to_int8(
                        rgetattr(self.pipe, module_name), device=f"cuda:{local_rank}",
                        min_layer_size=512,
                    )

        if self.config.use_hybrid_attn_schedule:
            self._setup_hybrid_attn_schedule(input_args)

        if self.config.use_hybrid_gemm_schedule:
            self._setup_hybrid_gemm_schedule(input_args)

        if self.config.use_vae_channels_last_format:
            self._convert_vae_to_channels_last()


    def _shard_model_with_fsdp(self) -> None:
        """ Shard the model with FSDP based on settings """
        if self.config.use_fp8_gemms and _is_cuda():
            from xfuser.core.utils.runner_utils import _TORCHAO_FLOAT8_FSDP2_PATCHES
            assert _TORCHAO_FLOAT8_FSDP2_PATCHES, (
                "FSDP2 + FP8 requires torchao Float8Tensor patches but they failed to apply at "
                "import time. Check for torchao import errors in runner_utils."
            )
        local_rank = get_world_group().local_rank
        fs_local_rank = get_fs_group().local_rank
        device_group = get_fs_group().device_group
        for component_name, component in self.pipe.components.items():
            if component_name in self.settings.fsdp_strategy:
                log(f"Sharding {component_name} with FSDP... "
                    f"(host cur/anon/file: {host_mem_gb()} GB, "
                    f"VRAM: {torch.cuda.memory_allocated(local_rank)/1e9:.2f}GB)")
                strategy = self.settings.fsdp_strategy[component_name]
                wrap_attrs = strategy.get("wrap_attrs", [])
                dtype = strategy.get("dtype", None)
                offload_policy = strategy.get("offload_policy", None)
                # A meta component was built on-config to avoid a full bf16 copy per rank. Two meta
                # paths: the transformer self-fills each block from disk on every rank (never full
                # anywhere, quantized per block), while text encoders are filled by a rank0
                # broadcast (no per-block quantize; the filled TE stays bf16/streamed-fp8 on rank0).
                is_meta = any(p.is_meta for p in component.parameters())
                is_transformer_selffill = is_meta and component_name.startswith("transformer")
                load_block_fn = load_epilogue_fn = None
                if is_transformer_selffill:
                    quantize_fn = self._build_fsdp_quantize_fn(
                        component_name, wrap_attrs, fs_local_rank
                    )
                    load_block_fn, load_epilogue_fn = self._sharder.build_transformer_disk_loaders(
                        component, wrap_attrs, component_name, f"cuda:{fs_local_rank}"
                    )
                else:
                    quantize_fn = (
                        None if is_meta
                        else self._build_fsdp_quantize_fn(component_name, wrap_attrs, fs_local_rank)
                    )
                fsdp_object = shard_component(
                    component, wrap_attrs, device_group, fs_local_rank, dtype,
                    quantize_fn=quantize_fn,
                    reshard_after_forward=self.config.reshard_after_forward,
                    memory_efficient_init=self.config.memory_efficient_sharding,
                    offload_policy=offload_policy,
                    # All ranks load from the same checkpoint so states are already
                    # identical. No broadcast needed regardless of offload policy.
                    sync_module_states=False,
                    meta_init=is_meta and not is_transformer_selffill,
                    load_block_fn=load_block_fn,
                    load_epilogue_fn=load_epilogue_fn,
                )
                if is_meta and not is_transformer_selffill:
                    self._sharder.broadcast_load(
                        fsdp_object, component_name, offload_policy == "cpu"
                    )
                setattr(self.pipe, component_name, fsdp_object)
                torch.cuda.empty_cache()
                log(f"Sharded {component_name}. "
                    f"(host cur/anon/file: {host_mem_gb()} GB, "
                    f"VRAM: {torch.cuda.memory_allocated(local_rank)/1e9:.2f}GB)")
            else:
                log(f"Skipping FSDP wrapping for {component_name}...")
                if hasattr(component, "to"):
                    component.to(f"cuda:{local_rank}")
                else:
                    log(f"Component {component_name} has no .to() method, skipping device move.")
                    pass

        # diffusers' _execution_device short-circuits on the first nn.Module component
        # that lacks _hf_hook, returning self.device (= first module's .device).
        # With CPUOffloadPolicy, text_encoder.device = cpu, breaking latent generation.
        # Fix: give every nn.Module component a minimal _hf_hook so _execution_device
        # continues past them, with cpu-offloaded components advertising cuda.
        cpu_offloaded = {
            name for name, s in self.settings.fsdp_strategy.items()
            if s.get("offload_policy") == "cpu"
        }
        if cpu_offloaded:
            cuda_device = f"cuda:{local_rank}"

            class _ExecDeviceHook:
                def __init__(self, execution_device):
                    self.execution_device = execution_device

            for name, component in self.pipe.components.items():
                if not isinstance(component, torch.nn.Module):
                    continue
                if not hasattr(component, "_hf_hook"):
                    component._hf_hook = _ExecDeviceHook(
                        cuda_device if name in cpu_offloaded else None
                    )

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

    def _build_fsdp_quantize_fn(
        self, component_name: str, wrap_attrs: list, local_rank: int
    ):
        """
        Return a per-block quantize callable (block, block_idx) -> None for this
        component, or None if no quantization is configured for it.

        fp8_precision_overrides entries like "5." apply to block index 5. We strip
        the block-index prefix before passing to the quantize functions so they see
        the same local FQN paths they would in the non-FSDP path.

        Suffix patterns (e.g. .net.0.proj) are block-local FQNs and are passed
        through unchanged on every block; only prefix patterns are stripped.
        """
        if not (self.config.use_fp4_gemms or self.config.use_fp8_gemms or self.config.use_int8_gemms):
            return None

        device = f"cuda:{local_rank}"
        fp4_list = set(self.settings.fp4_gemm_module_list or [])
        fp8_list = set(self.settings.fp8_gemm_module_list or [])
        fp8_overrides = self.settings.fp8_precision_overrides or ()
        fp8_suffix_overrides = self.settings.fp8_precision_override_suffixes
        int8_list = set(self.settings.int8_gemm_module_list or [])

        paths = [f"{component_name}.{a}" for a in wrap_attrs]

        use_fp4_here = self.config.use_fp4_gemms and any(p in fp4_list for p in paths)
        # fp8-only: in fp8 list but not fp4 list (e.g. transformer_2 in Wan2.2 FP4 mode)
        use_fp8_here = (
            self.config.use_fp8_gemms and any(p in fp8_list for p in paths)
        ) or (
            self.config.use_fp4_gemms and any(p in fp8_list and p not in fp4_list for p in paths)
        )
        use_int8_here = self.config.use_int8_gemms and any(p in int8_list for p in paths)

        if not use_fp4_here and not use_fp8_here and not use_int8_here:
            return None

        def quantize_fn(block, block_idx: int) -> None:
            block_prefix = f"{block_idx}."
            # Strip the block-index prefix so the quantize functions see local FQN paths.
            local_fp8 = tuple(
                o[len(block_prefix):] for o in fp8_overrides if o.startswith(block_prefix)
            ) or None
            if use_fp4_here:
                if _is_cuda():
                    quantize_linear_layers_to_nvfp4(
                        block,
                        fp8_layers=local_fp8,
                        fp8_suffix_layers=fp8_suffix_overrides,
                        device=device,
                    )
                else:
                    quantize_linear_layers_to_fp4(
                        block,
                        fp8_layers=local_fp8,
                        fp8_suffix_layers=fp8_suffix_overrides,
                        use_hybrid_schedule=self.config.use_hybrid_gemm_schedule,
                        device=device,
                    )
            elif use_fp8_here:
                if _use_aiter_fp8_rdna4():
                    quantize_linear_layers_to_fp8_blockscale(block, device=device)
                else:
                    quantize_linear_layers_to_fp8(block, device=device)
            else:
                # use_int8_here
                quantize_linear_layers_to_int8(block, device=device, min_layer_size=512)

        return quantize_fn

    def _setup_mxfp4_gemms(self, local_rank):
        for module_name in self.settings.fp4_gemm_module_list:
            # Certain models benefit from a hybrid quantization strategy: applying FP8 to
            # a number of transformer blocks while using FP4 for others. This mixed-precision
            # approach balances performance and output quality better than uniform quantization.
            log(f"Quantizing linear layers in {module_name} to FP4...")
            module = rgetattr(self.pipe, module_name)
            quantize_linear_layers_to_fp4(
                module,
                fp8_layers=self.settings.fp8_precision_overrides,
                fp8_suffix_layers=self.settings.fp8_precision_override_suffixes,
                use_hybrid_schedule=self.config.use_hybrid_gemm_schedule,
                device=f"cuda:{local_rank}",
            )
        # Any module specified in fp8 gemms modules list and not specified in fp4 gemms module list,
        # will be quantized to fp8, this is specially beneficial for MoE models like Wan2.2,
        # where the low-noise transformer should use FP8 quantization.
        # This transformer generates fine details and requires higher precision to maintain quality.
        for module_name in self.settings.fp8_gemm_module_list:
            if module_name in self.settings.fp4_gemm_module_list:
                continue
            log(f"Quantizing linear layers in {module_name} to FP8...")
            module = rgetattr(self.pipe, module_name)
            quantize_linear_layers_to_fp8(module, device=f"cuda:{local_rank}")

    def _setup_nvfp4_gemms(self, local_rank):
        for module_name in self.settings.fp4_gemm_module_list:
            log(f"Quantizing linear layers in {module_name} to NVFP4 (torchao)...")
            module = rgetattr(self.pipe, module_name)
            quantize_linear_layers_to_nvfp4(
                module,
                fp8_layers=self.settings.fp8_precision_overrides,
                fp8_suffix_layers=self.settings.fp8_precision_override_suffixes,
                device=f"cuda:{local_rank}",
            )
        for module_name in self.settings.fp8_gemm_module_list:
            if module_name in self.settings.fp4_gemm_module_list:
                continue
            log(f"Quantizing linear layers in {module_name} to FP8...")
            module = rgetattr(self.pipe, module_name)
            quantize_linear_layers_to_fp8(module, device=f"cuda:{local_rank}")

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

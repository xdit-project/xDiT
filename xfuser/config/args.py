import sys
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
import torch.distributed

from xfuser.logger import init_logger
from xfuser.core.distributed import init_distributed_environment
from xfuser.config.config import (
    EngineConfig,
    FastAttnConfig,
    ParallelConfig,
    TensorParallelConfig,
    PipeFusionParallelConfig,
    SequenceParallelConfig,
    DataParallelConfig,
    FullyShardConfig,
    VaeParallelConfig,
    ModelConfig,
    InputConfig,
    RuntimeConfig,
)

logger = init_logger(__name__)


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def _normalize_name(self, name: str) -> str:
        # First, try the standard normalization (all hyphens to underscores)
        fully_normalized = "--" + name[len("--"):].replace("-", "_")
        if fully_normalized in self._option_string_actions:
            return fully_normalized

        # If not found, check if it's a BooleanOptionalAction (starts with --no-)
        if name.startswith("--no-"):
            no_dash_normalized = "--no-" + name[len("--no-"):].replace("-", "_")
            if no_dash_normalized in self._option_string_actions:
                return no_dash_normalized

        return fully_normalized

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        processed_args = []
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    processed_args.append(f"{self._normalize_name(key)}={value}")
                else:
                    processed_args.append(self._normalize_name(arg))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class xFuserArgs:
    """Arguments for xFuser engine."""

    # Model arguments
    model: str = ""
    download_dir: Optional[str] = None
    trust_remote_code: bool = False
    # Runtime arguments
    warmup_steps: int = 1
    # use_cuda_graph: bool = True
    use_parallel_vae: bool = False
    # use_profiler: bool = False
    use_torch_compile: bool = False
    use_onediff: bool = False
    # Parallel arguments
    # data parallel
    data_parallel_degree: int = 1
    use_cfg_parallel: bool = False
    # sequence parallel
    shard_dit: Optional[bool] = False
    ulysses_degree: Optional[int] = 1
    ring_degree: Optional[int] = 1
    # tensor parallel
    tensor_parallel_degree: int = 1
    text_encoder_tp_degree: int = 1
    split_scheme: Optional[str] = "row"
    # ray arguments
    use_ray: bool = False
    ray_world_size: int = 1
    vae_parallel_size: int = 0
    dit_parallel_size: int = 0
    # pipefusion parallel
    pipefusion_parallel_degree: int = 1
    num_pipeline_patch: Optional[int] = None
    attn_layer_num_for_pp: Optional[List[int]] = None
    # Input arguments
    height: int = 1024
    width: int = 1024
    num_frames: int = 49
    num_inference_steps: int = 20
    max_sequence_length: int = 256
    img_file_path: Optional[str] = None
    prompt: Union[str, List[str]] = ""
    negative_prompt: Union[str, List[str]] = ""
    no_use_resolution_binning: bool = False
    seed: int = 42
    output_type: str = "pil"
    guidance_scale: float = 3.5
    guidance_scale_2: Optional[float] = None
    flow_shift: Optional[float] = None
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_group_cpu_offload: bool = False
    group_offload_low_cpu_mem: bool = False
    enable_tiling: bool = False
    enable_slicing: bool = False
    vae_tile_size_height: Optional[int] = None
    vae_tile_size_width: Optional[int] = None
    vae_tile_overlap_height: Optional[int] = None
    vae_tile_overlap_width: Optional[int] = None
    # DiTFastAttn arguments
    use_fast_attn: bool = False
    n_calib: int = 8
    threshold: float = 0.5
    window_size: int = 64
    coco_path: Optional[str] = None
    use_cache: bool = False
    use_teacache: bool = False
    use_fbcache: bool = False
    # Other arguments
    use_fp8_t5_encoder: bool = False
    shard_t5_encoder: bool = False
    attention_backend: Optional[str] = None
    cross_attention_backend: Optional[str] = None
    use_int8_gemms: bool = False
    use_fp8_gemms: bool = False
    use_fp8_text_encoder: bool = False
    use_fp4_gemms: bool = False
    fp8_precision_override_prefix_patterns: Optional[str] = None
    fp8_precision_override_suffix_patterns: Optional[str] = None
    # Model runner specific
    num_iterations: int = 1
    profile: bool = False
    profile_wait: int = 2
    profile_warmup: int = 2
    profile_active: int = 1
    warmup_calls: int = 0
    output_directory: str = "."
    input_images: Optional[List[str]] = None
    resize_input_images: bool = False
    task: Optional[str] = None
    batch_size: Optional[int] = None
    dataset_path: Optional[str] = None
    use_fsdp: bool = False
    fully_shard_degree: int = 1
    reshard_after_forward: bool = True
    memory_efficient_sharding: bool = False
    memory_efficient_replicated_load: bool = False
    use_vae_channels_last_format: bool = False
    # Hybrid attention schedule
    use_hybrid_attn_schedule: bool = False
    hybrid_attn_low_precision_backend: Optional[str] = None
    hybrid_attn_high_precision_backend: Optional[str] = None
    num_hybrid_attn_high_precision_steps: Optional[int] = None
    hybrid_attn_schedule: Optional[str] = None
    # Hybrid GEMM schedule (FP8 high precision + FP4 low precision)
    use_hybrid_gemm_schedule: bool = False
    num_hybrid_gemm_high_precision_steps: Optional[int] = None
    # SSTA arguments
    use_ssta_sparse_text_to_image: Optional[bool] = False
    # Sparge attention
    spargeattn_reorder_sequence: bool = True
    use_spargeattn_static_block_mask: bool = True
    spargeattn_simthreshold: float = 0.3
    spargeattn_cdfthreshold: float = 0.92
    use_spargeattn_head_balance: bool = False
    # AITER CK-Tile VSA attention
    vsa_block_size: int = 128
    vsa_top_k: int = 1
    vsa_top_k_ratio: float = 0.0
    vsa_drop_rates: Optional[List[float]] = None
    vsa_prob_threshold: float = 0.9
    vsa_reorder_sequence: bool = True
    use_vsa_static_block_mask: bool = True
    use_vsa_first_frame_mask: bool = True
    vsa_collect_density: bool = False
    # Distilled model weight paths
    distilled_transformer_path: Optional[str] = None
    distilled_transformer_2_path: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser):
        """Shared CLI arguments for xFuser engine."""
        # Model arguments
        model_group = parser.add_argument_group("Model Options")
        model_group.add_argument(
            "--model",
            type=str,
            default="PixArt-alpha/PixArt-XL-2-1024-MS",
            help="Name or path of the huggingface model to use.",
            required=True,
        )
        model_group.add_argument(
            "--download-dir",
            type=nullable_str,
            default=xFuserArgs.download_dir,
            help="Directory to download and load the weights, default to the default cache dir of huggingface.",
        )
        model_group.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code from huggingface.",
        )

        # Runtime arguments
        runtime_group = parser.add_argument_group("Runtime Options")
        runtime_group.add_argument(
            "--warmup_steps", type=int, default=1, help="Warmup steps in generation."
        )
        # runtime_group.add_argument("--use_cuda_graph", action="store_true")
        runtime_group.add_argument("--use_parallel_vae", action="store_true")
        # runtime_group.add_argument("--use_profiler", action="store_true")
        runtime_group.add_argument(
            "--use_torch_compile",
            action="store_true",
            help="Enable torch.compile to accelerate inference in a single card",
        )
        runtime_group.add_argument(
            "--use_onediff",
            action="store_true",
            help="Enable onediff to accelerate inference in a single card",
        )
        runtime_group.add_argument(
            "--use_teacache",
            action="store_true",
            help="Enable teacache to accelerate inference in a single card",
        )
        runtime_group.add_argument(
            "--use_fbcache",
            action="store_true",
            help="Enable teacache to accelerate inference in a single card",
        )
        runtime_group.add_argument(
            "--attention_backend",
            type=str,
            default=None,
            help="Attention backend to use. If not specified, the best available backend will be selected automatically.",
        )
        # Parallel arguments
        parallel_group = parser.add_argument_group("Parallel Processing Options")
        runtime_group.add_argument(
            "--use_ray",
            action="store_true",
            help="Enable ray to run inference in multi-card",
        )
        parallel_group.add_argument(
            "--ray_world_size",
            type=int,
            default=1,
            help="The number of ray workers (world_size for ray)",
        )
        parallel_group.add_argument(
            "--dit_parallel_size",
            type=int,
            default=0,
            help="The number of processes for DIT parallelization.",
        )
        parallel_group.add_argument(
            "--use_cfg_parallel",
            action="store_true",
            help="Use split batch in classifier_free_guidance. cfg_degree will be 2 if set",
        )
        parallel_group.add_argument(
            "--data_parallel_degree", type=int, default=1, help="Data parallel degree."
        )
        parallel_group.add_argument(
            "--ulysses_degree",
            type=int,
            default=1,
            help="Ulysses sequence parallel degree. Used in attention layer.",
        )
        parallel_group.add_argument(
            "--ring_degree",
            type=int,
            default=1,
            help="Ring sequence parallel degree. Used in attention layer.",
        )
        parallel_group.add_argument(
            "--shard_dit",
            action="store_true",
            help="Enable DiT model sharding. Used together with sequence parallelism.",
        )
        parallel_group.add_argument(
            "--shard_t5_encoder",
            action="store_true",
            help="Enable t5 encoder sharding.",
        )
        parallel_group.add_argument(
            "--pipefusion_parallel_degree",
            type=int,
            default=1,
            help="Pipefusion parallel degree. Indicates the number of pipeline stages.",
        )
        parallel_group.add_argument(
            "--num_pipeline_patch",
            type=int,
            default=None,
            help="Number of patches the feature map should be segmented in pipefusion parallel.",
        )
        parallel_group.add_argument(
            "--attn_layer_num_for_pp",
            default=None,
            nargs="*",
            type=int,
            help="List representing the number of layers per stage of the pipeline in pipefusion parallel",
        )
        parallel_group.add_argument(
            "--tensor_parallel_degree",
            type=int,
            default=1,
            help="Tensor parallel degree.",
        )
        parallel_group.add_argument(
            "--text_encoder_tp_degree",
            type=int,
            default=1,
            help="Tensor parallel degree for supported text encoders, reusing the existing model ranks.",
        )
        parallel_group.add_argument(
            "--split_scheme",
            type=str,
            default="row",
            help="Split scheme for tensor parallel.",
        )
        parallel_group.add_argument(
            "--vae_parallel_size",
            type=int,
            default=0,
            help="Number of processes for VAE parallelization. 0: no seperate process for VAE, 1: run VAE in a separate process, >1: distribute VAE across multiple processes.",
        )

        # Input arguments
        input_group = parser.add_argument_group("Input Options")
        input_group.add_argument(
            "--height", type=int, default=1024, help="The height of image"
        )
        input_group.add_argument(
            "--width", type=int, default=1024, help="The width of image"
        )
        input_group.add_argument(
            "--num_frames", type=int, default=49, help="The frames of video"
        )
        input_group.add_argument(
            "--img_file_path", type=str, default=None, help="Path for the input image."
        )
        input_group.add_argument(
            "--prompt", type=str, nargs="*", default="", help="Prompt for the model."
        )
        input_group.add_argument("--no_use_resolution_binning", action="store_true")
        input_group.add_argument(
            "--negative_prompt",
            type=str,
            nargs="*",
            default="",
            help="Negative prompt for the model.",
        )
        input_group.add_argument(
            "--num_inference_steps",
            type=int,
            default=20,
            help="Number of inference steps.",
        )
        input_group.add_argument(
            "--max_sequence_length",
            type=int,
            default=256,
            help="Max sequencen length of prompt",
        )
        runtime_group.add_argument(
            "--seed", type=int, default=42, help="Random seed for operations."
        )
        runtime_group.add_argument(
            "--output_type",
            type=str,
            default="pil",
            help="Output type of the pipeline.",
        )
        input_group.add_argument(
            "--guidance_scale",
            type=float,
            default=3.5,
            help="Guidance scale for classifier free guidance.",
        )
        runtime_group.add_argument(
            "--enable_sequential_cpu_offload",
            action="store_true",
            help="Offloading the weights to the CPU.",
        )
        runtime_group.add_argument(
            "--enable_model_cpu_offload",
            action="store_true",
            help="Offloading the weights to the CPU.",
        )
        runtime_group.add_argument(
            "--enable_group_cpu_offload",
            action="store_true",
            help="Async leaf-level group CPU offload (streamed, per-group pinned). Overlaps H2D "
                 "transfer with compute; upstream diffusers group offloading.",
        )
        runtime_group.add_argument(
            "--enable_tiling",
            action="store_true",
            help="Making VAE decode a tile at a time to save GPU memory.",
        )
        runtime_group.add_argument(
            "--enable_slicing",
            action="store_true",
            help="Decode one batch item at a time to reduce GPU memory use. This has no effect "
                 "when the batch size is 1.",
        )
        runtime_group.add_argument(
            "--vae_tile_size_height",
            type=int,
            default=None,
            help="Exact output-pixel height of each VAE tile. Requires --enable_tiling. "
                 "Must be used with --vae_tile_size_width.",
        )
        runtime_group.add_argument(
            "--vae_tile_size_width",
            type=int,
            default=None,
            help="Exact output-pixel width of each VAE tile. Requires --enable_tiling. "
                 "Must be used with --vae_tile_size_height.",
        )
        runtime_group.add_argument(
            "--vae_tile_overlap_height",
            type=int,
            default=None,
            help="Exact VAE tile overlap along the height axis, in output pixels. Requires "
                 "--enable_tiling. Must be used with --vae_tile_overlap_width. Height and width "
                 "may differ; use 0 for an inactive strip axis.",
        )
        runtime_group.add_argument(
            "--vae_tile_overlap_width",
            type=int,
            default=None,
            help="Exact VAE tile overlap along the width axis, in output pixels. Requires "
                 "--enable_tiling. Must be used with --vae_tile_overlap_height. Height and width "
                 "may differ; use 0 for an inactive strip axis.",
        )
        runtime_group.add_argument(
            "--use_fp8_t5_encoder",
            action="store_true",
            help="Quantize the T5 text encoder.",
        )
        runtime_group.add_argument(
            "--use_fp8_gemms",
            action="store_true",
            help="Quantize the transformer linear layers (selected models only).",
        )
        runtime_group.add_argument(
            "--use_int8_gemms",
            action="store_true",
            help="Quantize the transformer linear layers (selected models only).",
        )

        # DiTFastAttn arguments
        fast_attn_group = parser.add_argument_group("DiTFastAttn Options")
        fast_attn_group.add_argument(
            "--use_fast_attn",
            action="store_true",
            help="Use DiTFastAttn to accelerate single inference. Only data parallelism can be used with DITFastAttn.",
        )
        fast_attn_group.add_argument(
            "--n_calib",
            type=int,
            default=8,
            help="Number of prompts for compression method seletion.",
        )
        fast_attn_group.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Threshold for selecting attention compression method.",
        )
        fast_attn_group.add_argument(
            "--window_size",
            type=int,
            default=64,
            help="Size of window attention.",
        )
        fast_attn_group.add_argument(
            "--coco_path",
            type=str,
            default=None,
            help="Path of MS COCO annotation json file.",
        )
        fast_attn_group.add_argument(
            "--use_cache",
            action="store_true",
            help="Use cache config for attention compression.",
        )

        return parser


    @staticmethod
    def add_runner_args(parser: FlexibleArgumentParser):
        parser.add_argument(
            "--model",
            type=str,
            help="Name or path of the huggingface model to use.",
            required=True,
        )
        parser.add_argument(
            "--use_parallel_vae",
            help="Enable parallel VAE.",
            action="store_true")
        parser.add_argument(
            "--use_torch_compile",
            action="store_true",
            help="Enable torch.compile to accelerate inference in a single card",
        )
        parser.add_argument(
            "--attention_backend",
            type=str,
            default=None,
            help="Attention backend to use. If not specified, the best available backend will be selected automatically.",
        )
        parser.add_argument(
            "--cross_attention_backend",
            type=str,
            default=None,
            help="Attention backend to use for cross-attention. If not specified, falls back to --attention_backend.",
        )
        parser.add_argument(
            "--use_cfg_parallel",
            action="store_true",
            help="Use split batch in classifier_free_guidance. cfg_degree will be 2 if set",
        )
        parser.add_argument(
            "--data_parallel_degree", type=int, default=1, help="Data parallel degree."
        )
        parser.add_argument(
            "--ulysses_degree",
            type=int,
            default=1,
            help="Ulysses sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--ring_degree",
            type=int,
            default=1,
            help="Ring sequence parallel degree. Used in attention layer.",
        )
        parser.add_argument(
            "--pipefusion_parallel_degree",
            type=int,
            default=1,
            help="Pipefusion parallel degree. Indicates the number of pipeline stages.",
        )
        parser.add_argument(
            "--tensor_parallel_degree",
            type=int,
            default=1,
            help="Tensor parallel degree.",
        )
        parser.add_argument(
            "--text_encoder_tp_degree",
            type=int,
            default=1,
            help="Tensor parallel degree for supported text encoders, reusing the existing model ranks.",
        )
        parser.add_argument(
            "--fully_shard_degree",
            type=int,
            default=1,
            help="Fully sharding (sharding) degree."
        )
        parser.add_argument(
            "--no_reshard_after_forward",
            dest="reshard_after_forward",
            action="store_false",
            help="Keep parameters gathered after each block's forward instead of resharding. "
                 "Trades memory for latency by eliminating repeated all-gathers. "
                 "Only valid with --fully_shard_degree > 1.",
        )
        parser.add_argument(
            "--memory_efficient_sharding",
            action="store_true",
            default=False,
            help="Reduce peak VRAM during load: shard transformer blocks one at a time on GPU "
                 "during init, so the full unsharded component never materializes on device. "
                 "Slightly slower to load - use if the model OOMs on GPU during init. Requires "
                 "--fully_shard_degree > 1.",
        )
        parser.add_argument(
            "--memory_efficient_replicated_load",
            action="store_true",
            default=False,
            help="Reduce peak host RAM when a model that fits one GPU is replicated across ranks "
                 "(pure sequence/CFG/data parallelism): rank0 loads the real weights and peers "
                 "build on meta and receive them over a GPU->GPU broadcast, so host peak is 1x the "
                 "model instead of Nx. Use if the load is OOM-killed on host as rank count grows. "
                 "No effect with weight-splitting parallelism (FSDP/PipeFusion/tensor parallel), "
                 "which loads per-rank weights anyway, or on a single rank.",
        )
        parser.add_argument(
            "--height",
            type=int,
            help="The height of image",
        )
        parser.add_argument(
            "--width",
            type=int,
            help="The width of image",
        )
        parser.add_argument(
            "--num_frames",
            type=int,
            help="The frames of video",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            nargs="*",
            help="Prompt for the model.",
        )
        parser.add_argument(
            "--negative_prompt", type=str,
            nargs="*",
            help="Negative prompt for the model.",
        )
        parser.add_argument(
            "--num_inference_steps",
            type=int,
            help="Number of inference steps.",
        )
        parser.add_argument(
            "--max_sequence_length",
            type=int,
            help="Max sequence length of prompt",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for operations."
        )
        parser.add_argument(
            "--guidance_scale",
            type=float,
            help="Guidance scale for classifier free guidance.",
        )
        parser.add_argument(
            "--guidance_scale_2",
            type=float,
            help="Guidance scale for classifier free guidance (second scale).",
        )
        parser.add_argument(
            "--flow_shift",
            type=float,
            help="Flow shift for the scheduler.",
        )
        parser.add_argument(
            "--enable_sequential_cpu_offload",
            action="store_true",
            help="Offloading the weights to the CPU.",
        )
        parser.add_argument(
            "--enable_model_cpu_offload",
            action="store_true",
            help="Offloading the weights to the CPU.",
        )
        parser.add_argument(
            "--enable_group_cpu_offload",
            action="store_true",
            help="Async leaf-level group CPU offload (streamed).",
        )
        parser.add_argument(
            "--group_offload_low_cpu_mem",
            action="store_true",
            help="With --enable_group_cpu_offload, pin each tensor as it is offloaded instead of "
                 "pre-pinning whole components. Keeps host RAM flat on hosts where it is the "
                 "binding constraint, at some of the streaming speedup.",
        )
        parser.add_argument(
            "--enable_tiling",
            action="store_true",
            help="Enable VAE tiling to save GPU memory.",
        )
        parser.add_argument(
            "--enable_slicing",
            action="store_true",
            help="Decode one batch item at a time to reduce GPU memory use. This has no effect "
                 "when the batch size is 1.",
        )
        parser.add_argument(
            "--vae_tile_size_height",
            type=int,
            default=None,
            help="Exact output-pixel height of each VAE tile. Requires --enable_tiling. "
                 "Must be used with --vae_tile_size_width.",
        )
        parser.add_argument(
            "--vae_tile_size_width",
            type=int,
            default=None,
            help="Exact output-pixel width of each VAE tile. Requires --enable_tiling. "
                 "Must be used with --vae_tile_size_height.",
        )
        parser.add_argument(
            "--vae_tile_overlap_height",
            type=int,
            default=None,
            help="Exact VAE tile overlap along the height axis, in output pixels. Requires "
                 "--enable_tiling. Must be used with --vae_tile_overlap_width. Height and width "
                 "may differ; use 0 for an inactive strip axis.",
        )
        parser.add_argument(
            "--vae_tile_overlap_width",
            type=int,
            default=None,
            help="Exact VAE tile overlap along the width axis, in output pixels. Requires "
                 "--enable_tiling. Must be used with --vae_tile_overlap_height. Height and width "
                 "may differ; use 0 for an inactive strip axis.",
        )
        parser.add_argument(
            "--use_int8_gemms",
            action="store_true",
            help="Quantize the transformer linear layers (selected models only).",
        )
        parser.add_argument(
            "--use_fp8_gemms",
            action="store_true",
            help="Quantize the transformer linear layers (selected models only).",
        )
        parser.add_argument(
            "--use_fp8_text_encoder",
            action="store_true",
            help="Also quantize the text encoder's linear layers to FP8 (selected models only). "
                 "Requires --use_fp8_gemms, which covers the transformer alone. Frees several GB "
                 "for large bf16 text encoders, at whatever output-quality cost FP8 carries for "
                 "the encoder; off by default because that is a quality trade-off, not a free win.",
        )
        parser.add_argument(
            "--use_fp4_gemms",
            action="store_true",
            help="Quantize the transformer linear layers (selected models only).",
        )
        parser.add_argument(
            "--fp8_precision_override_prefix_patterns",
            type=nullable_str,
            default=None,
            help="Comma-delimited FQN prefix patterns to keep in FP8 during FP4 GEMMs.",
        )
        parser.add_argument(
            "--fp8_precision_override_suffix_patterns",
            type=nullable_str,
            default=None,
            help="Comma-delimited FQN suffix patterns to keep in FP8 during FP4 GEMMs.",
        )

        parser.add_argument(
            "--num_iterations",
            type=int,
            default=1,
            help="Number of iterations to run the model."
        )
        parser.add_argument(
            "--profile",
            default=False,
            action="store_true",
            help="Whether to run Pytorch profiler. See --profile_wait, --profile_warmup and --profile_active for profiler specific warmup."
        )
        parser.add_argument(
            "--profile_wait",
            type=int,
            default=2,
            help="wait argument for torch.profiler.schedule. Only used with --profile.",
        )
        parser.add_argument(
            "--profile_warmup",
            type=int,
            default=2,
            help="warmup argument for torch.profiler.schedule. Only used with --profile.",
        )
        parser.add_argument(
            "--profile_active",
            type=int,
            default=1,
            help="active argument for torch.profiler.schedule. Only used with --profile.",
        )
        parser.add_argument(
            "--warmup_calls",
            help="The number of full pipe calls to warmup the model.",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--output_directory",
            type=str,
            default=".",
            help="Directory where to save outputs, profiles and timings.",
        )
        parser.add_argument(
            "--input_images",
            default=[],
            nargs="+",
            help="Path(s)/URL(s) to input image(s).",
        )
        parser.add_argument(
            "--resize_input_images",
            default=False,
            action="store_true",
            help="Whether to resize and crop the input image(s) to the specified width and height.",
        )
        parser.add_argument(
            "--task",
            default=None,
            help="Task to perform. Only applicable if the model supports multiple tasks."
        )
        parser.add_argument(
            "--batch_size",
            default=None,
            type=int,
            help="Batch size for inference. If not specified, defaults to 1.",
        )
        parser.add_argument(
            "--dataset_path",
            default=None,
            help="Path to a csv dataset file containing prompts. If specified, prompts will be loaded from the dataset. Consider using --batch_size accordingly.",
        )
        parser.add_argument(
            "--use_hybrid_attn_schedule",
            action="store_true",
            default=False,
            help="Enable hybrid attention schedule for faster inference and improved quality."
        )
        parser.add_argument(
            "--hybrid_attn_low_precision_backend",
            type=nullable_str,
            default=None,
            help="Low-precision attention backend for hybrid schedule. If set, high-precision backend must also be set.",
        )
        parser.add_argument(
            "--hybrid_attn_high_precision_backend",
            type=nullable_str,
            default=None,
            help="High-precision attention backend for hybrid schedule. If set, low-precision backend must also be set.",
        )
        parser.add_argument(
            "--num_hybrid_attn_high_precision_steps",
            type=int,
            default=None,
            help="Number of high-precision attention steps in hybrid schedule.",
        )
        parser.add_argument(
            "--hybrid_attn_schedule",
            type=str,
            default=None,
            help="An explicit comma-delimited string of attention backend names for the hybrid schedule. Example: 'FLASH_3,FLASH_3_FP8,FLASH_3_FP8,FLASH_3'. Use only if both low-precision and high-precision backends are not set.",
        )
        parser.add_argument(
            "--use_hybrid_gemm_schedule",
            action="store_true",
            default=False,
            help="Enable hybrid GEMM schedule: FP8 at start/end and MXFP4 in the middle.",
        )
        parser.add_argument(
            "--num_hybrid_gemm_high_precision_steps",
            type=int,
            default=None,
            help="Number of high-precision GEMM steps at both the start and end of denoising.",
        )
        parser.add_argument(
            "--use_vae_channels_last_format",
            action="store_true",
            default=False,
            help="Use channels last memory format for the VAE.",
        )
        parser.add_argument(
            "--use_ssta_sparse_text_to_image",
            action="store_true",
            default=False,
            help="Use sparse attention for text-to-image attention path in SSTA.",
        )
        parser.add_argument(
            "--spargeattn_simthreshold",
            type=float,
            default=0.3,
            help="Similarity threshold for sparge attention.",
        )
        parser.add_argument(
            "--spargeattn_cdfthreshold",
            type=float,
            default=0.92,
            help="CDF threshold for sparge attention.",
        )
        parser.add_argument(
            "--spargeattn_reorder_sequence",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reorder image tokens via the gilbert space-filling curve "
                 "before Sparge attention. Use --no-spargeattn_reorder_sequence to disable."
        )
        parser.add_argument(
            "--use_spargeattn_static_block_mask",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="OR a static gilbert block-neighbor mask into the dynamic "
                 "Sparge block mask. Only meaningful when "
                 "--spargeattn_reorder_sequence is set. Use --no-use_spargeattn_static_block_mask to disable."
        )
        parser.add_argument(
            "--use_spargeattn_head_balance",
            action="store_true",
            help="Balance per-rank attention work across Ulysses ranks by "
                 "permuting heads (block-sparse load balancing). Only has an "
                 "effect with ulysses_degree>1 and a Sparge attention backend.",
        )
        parser.add_argument(
            "--vsa_block_size",
            type=int,
            default=128,
            help="Query/KV block size for the AITER CK-Tile VSA backend.",
        )
        parser.add_argument(
            "--vsa_top_k",
            type=int,
            default=1,
            help="Minimum KV blocks selected per query block by AITER VSA.",
        )
        parser.add_argument(
            "--vsa_top_k_ratio",
            type=float,
            default=0.0,
            help="Minimum selected KV-block fraction for AITER VSA. "
                 "For Jenga drop rate r, set this to 1-r.",
        )
        parser.add_argument(
            "--vsa_drop_rates",
            type=float,
            nargs="+",
            default=None,
            help="Enable Jenga's per-step sparse schedule. One value applies "
                 "to all steps; two values switch after the midpoint. "
                 "Drop rates <=0.25 use dense AITER attention.",
        )
        parser.add_argument(
            "--vsa_prob_threshold",
            type=float,
            default=0.9,
            help="Jenga cumulative probability threshold for AITER VSA.",
        )
        parser.add_argument(
            "--vsa_reorder_sequence",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reorder Wan video tokens with the sliced Gilbert curve before VSA.",
        )
        parser.add_argument(
            "--use_vsa_static_block_mask",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Union Gilbert physical-neighbor blocks into the dynamic VSA mask.",
        )
        parser.add_argument(
            "--use_vsa_first_frame_mask",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Preserve first-frame block relations in the VSA mask.",
        )
        parser.add_argument(
            "--vsa_collect_density",
            action="store_true",
            help="Record the selected VSA block density for profiling.",
        )
        parser.add_argument(
            "--distilled_transformer_path",
            type=nullable_str,
            default=None,
            help="Path to the high-noise distilled transformer safetensors file.",
        )
        parser.add_argument(
            "--distilled_transformer_2_path",
            type=nullable_str,
            default=None,
            help="Path to the low-noise distilled transformer_2 safetensors file.",
        )
        parser.add_argument(
            "--use_fbcache",
            action="store_true",
            help="Enable FBCache to accelerate the diffusion loop",
        )
        return parser


    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})
        return engine_args

    @classmethod
    def from_runner_args(cls, args: dict):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{arg_name: arg_value for arg_name, arg_value in args.items() if arg_name in attrs})
        return engine_args


    def _validate_gemm_quantization_flags(self) -> None:
        """Validate ownership of mutually exclusive generic GEMM quantizers."""
        if self.use_int8_gemms and (self.use_fp8_gemms or self.use_fp4_gemms):
            raise ValueError(
                "--use_int8_gemms cannot be combined with --use_fp8_gemms or "
                "--use_fp4_gemms, including explicit hybrid FP8/FP4 mode."
            )
        if self.use_fp8_gemms and self.use_fp4_gemms and not self.use_hybrid_gemm_schedule:
            raise ValueError(
                "--use_fp8_gemms and --use_fp4_gemms cannot both be enabled unless "
                "--use_hybrid_gemm_schedule explicitly owns the mixed FP8/FP4 mode."
            )
        if self.use_hybrid_gemm_schedule and not self.use_fp4_gemms:
            raise ValueError(
                "When use_hybrid_gemm_schedule is True, use_fp4_gemms must be set."
            )

    def create_config(
        self,
    ) -> Tuple[EngineConfig, InputConfig]:
        self._validate_gemm_quantization_flags()
        if not self.use_ray and not torch.distributed.is_initialized():
            logger.warning(
                "Distributed environment is not initialized. " "Initializing..."
            )
            init_distributed_environment()
        if self.use_ray:
            self.world_size = self.ray_world_size
        else:
            self.world_size = torch.distributed.get_world_size()

        if self.dit_parallel_size == 0 and (not self.use_parallel_vae or self.vae_parallel_size == 0):
            self.dit_parallel_size = self.world_size
        assert self.dit_parallel_size+self.vae_parallel_size == self.world_size, (
            f"DIT parallel size {self.dit_parallel_size} and VAE parallel size {self.vae_parallel_size} must sum to world size {self.world_size}"
        )

        # Hybrid attention schedule validation
        if self.use_hybrid_attn_schedule:
            if self.attention_backend is not None:
                raise ValueError(
                    "When use_hybrid_attn_schedule is True, attention_backend must not be set."
                )
            if self.hybrid_attn_schedule is not None:
                if self.hybrid_attn_low_precision_backend is not None or self.hybrid_attn_high_precision_backend is not None:
                    raise ValueError("When an explicit hybrid attention schedule is provided, neither hybrid_attn_low_precision_backend nor hybrid_attn_high_precision_backend may be set.")
            elif self.hybrid_attn_low_precision_backend is None or self.hybrid_attn_high_precision_backend is None:
                raise ValueError(
                    "When use_hybrid_attn_schedule is True, both hybrid_attn_low_precision_backend and "
                    "hybrid_attn_high_precision_backend must be set."
                )

        if self.group_offload_low_cpu_mem and not self.enable_group_cpu_offload:
            raise ValueError(
                "--group_offload_low_cpu_mem only affects group CPU offload; pass "
                "--enable_group_cpu_offload too."
            )

        if self.use_fp8_text_encoder and not self.use_fp8_gemms:
            raise ValueError(
                "--use_fp8_text_encoder extends --use_fp8_gemms, which covers the transformer "
                "alone, to the text encoder; pass --use_fp8_gemms too."
            )

        if (
            self.fp8_precision_override_prefix_patterns is not None
            or self.fp8_precision_override_suffix_patterns is not None
        ) and not self.use_fp4_gemms:
            raise ValueError(
                "FP8 precision override patterns require --use_fp4_gemms: "
                "overrides apply when quantizing linear layers for FP4 GEMMs."
            )

        model_config = ModelConfig(
            model=self.model,
            download_dir=self.download_dir,
            trust_remote_code=self.trust_remote_code,
        )

        runtime_config = RuntimeConfig(
            warmup_steps=self.warmup_steps,
            # use_cuda_graph=self.use_cuda_graph,
            use_hybrid_attn_schedule=self.use_hybrid_attn_schedule,
            use_parallel_vae=self.use_parallel_vae,
            use_torch_compile=self.use_torch_compile,
            use_onediff=self.use_onediff,
            # use_profiler=self.use_profiler,
            use_fp8_t5_encoder=self.use_fp8_t5_encoder,
            attention_backend=self.attention_backend,
            cross_attention_backend=self.cross_attention_backend,
            spargeattn_reorder_sequence=self.spargeattn_reorder_sequence,
            use_spargeattn_static_block_mask=self.use_spargeattn_static_block_mask,
            spargeattn_simthreshold=self.spargeattn_simthreshold,
            spargeattn_cdfthreshold=self.spargeattn_cdfthreshold,
            use_spargeattn_head_balance=self.use_spargeattn_head_balance,
            vsa_block_size=self.vsa_block_size,
            vsa_top_k=self.vsa_top_k,
            vsa_top_k_ratio=self.vsa_top_k_ratio,
            vsa_drop_rates=self.vsa_drop_rates,
            vsa_prob_threshold=self.vsa_prob_threshold,
            vsa_reorder_sequence=self.vsa_reorder_sequence,
            use_vsa_static_block_mask=self.use_vsa_static_block_mask,
            use_vsa_first_frame_mask=self.use_vsa_first_frame_mask,
            vsa_collect_density=self.vsa_collect_density,
        )

        parallel_config = ParallelConfig(
            dp_config=DataParallelConfig(
                dp_degree=self.data_parallel_degree,
                use_cfg_parallel=self.use_cfg_parallel,
                dit_parallel_size=self.dit_parallel_size,
            ),
            sp_config=SequenceParallelConfig(
                ulysses_degree=self.ulysses_degree,
                ring_degree=self.ring_degree,
                shard_dit=self.shard_dit,
                dit_parallel_size=self.dit_parallel_size,
            ),
            tp_config=TensorParallelConfig(
                tp_degree=self.tensor_parallel_degree,
                split_scheme=self.split_scheme,
                dit_parallel_size=self.dit_parallel_size,
            ),
            pp_config=PipeFusionParallelConfig(
                pp_degree=self.pipefusion_parallel_degree,
                num_pipeline_patch=self.num_pipeline_patch,
                attn_layer_num_for_pp=self.attn_layer_num_for_pp,
                dit_parallel_size=self.dit_parallel_size,
            ),
            fs_config=FullyShardConfig(
                fs_degree=self.fully_shard_degree,
                dit_parallel_size=self.dit_parallel_size,
            ),
            vae_config=VaeParallelConfig(
                use_parallel_vae=self.use_parallel_vae,
            ),
            world_size=self.world_size,
            dit_parallel_size=self.dit_parallel_size,
            vae_parallel_size=self.vae_parallel_size,
            shard_t5_encoder=self.shard_t5_encoder,
        )

        fast_attn_config = FastAttnConfig(
            use_fast_attn=self.use_fast_attn,
            n_step=self.num_inference_steps,
            n_calib=self.n_calib,
            threshold=self.threshold,
            window_size=self.window_size,
            coco_path=self.coco_path,
            use_cache=self.use_cache,
        )

        engine_config = EngineConfig(
            model_config=model_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
            fast_attn_config=fast_attn_config,
        )

        input_config = InputConfig(
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            use_resolution_binning=not self.no_use_resolution_binning,
            batch_size=len(self.prompt) if isinstance(self.prompt, list) else 1,
            img_file_path=self.img_file_path,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            seed=self.seed,
            output_type=self.output_type,
            guidance_scale=self.guidance_scale,
        )

        return engine_config, input_config

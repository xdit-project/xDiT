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
    ModelConfig,
    InputConfig,
    RuntimeConfig,
)

logger = init_logger(__name__)


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = "--" + key[len("--") :].replace("-", "_")
                    processed_args.append(f"{key}={value}")
                else:
                    processed_args.append("--" + arg[len("--") :].replace("-", "_"))
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
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_tiling: bool = False
    enable_slicing: bool = False
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
    use_fp8_gemms: bool = False
    # Model runner specific
    num_iterations: int = 1
    repetition_sleep_duration: Optional[int] = None
    profile: bool = False
    profile_wait: int = 2
    profile_warmup: int = 2
    profile_active: int = 1
    warmup_calls: int = 0
    output_directory: str = "."
    input_images: Optional[List[str]] = None
    resize_input_images: bool = False
    task: Optional[str] = None


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
            "--vae_parallel_size",
            type=int,
            default=0,
            help="Number of processes for VAE parallelization. 0: no seperate process for VAE, 1: run VAE in a separate process, >1: distribute VAE across multiple processes.",
        )
        parallel_group.add_argument(
            "--split_scheme",
            type=str,
            default="row",
            help="Split scheme for tensor parallel.",
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
            "--enable_tiling",
            action="store_true",
            help="Making VAE decode a tile at a time to save GPU memory.",
        )
        runtime_group.add_argument(
            "--enable_slicing",
            action="store_true",
            help="Making VAE decode a tile at a time to save GPU memory.",
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
        parser = xFuserArgs.add_cli_args(parser)
        parser.set_defaults(model=None) # No default model for runner
        parser.add_argument(
            "--num_iterations",
            type=int,
            default=1,
            help="Number of iterations to run the model."
        )
        parser.add_argument(
            "--repetition_sleep_duration",
            type=int,
            default=None,
            help="The duration to sleep in between different pipe calls in seconds."
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
        return parser


    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    @classmethod
    def from_runner_args(cls, args: dict):
        engine_args = cls(**{arg_name: arg_value for arg_name, arg_value in args.items()})
        return engine_args



    def create_config(
        self,
    ) -> Tuple[EngineConfig, InputConfig]:
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
        assert self.dit_parallel_size+self.vae_parallel_size == self.world_size, f"DIT parallel size {self.dit_parallel_size} and VAE parallel size {self.vae_parallel_size} must sum to world size {self.world_size}"
        model_config = ModelConfig(
            model=self.model,
            download_dir=self.download_dir,
            trust_remote_code=self.trust_remote_code,
        )

        runtime_config = RuntimeConfig(
            warmup_steps=self.warmup_steps,
            # use_cuda_graph=self.use_cuda_graph,
            use_parallel_vae=self.use_parallel_vae,
            use_torch_compile=self.use_torch_compile,
            use_onediff=self.use_onediff,
            # use_profiler=self.use_profiler,
            use_fp8_t5_encoder=self.use_fp8_t5_encoder,
            attention_backend=self.attention_backend,
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

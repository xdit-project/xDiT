import sys
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Union

import torch
import torch.distributed

from pipefuser.logger import init_logger
from pipefuser.distributed import init_distributed_environment
from pipefuser.config.config import (
    EngineConfig,
    ParallelConfig,
    TensorParallelConfig,
    PipeFusionParallelConfig,
    SequenceParallelConfig,
    DataParallelConfig,
    ModelConfig,
    InputConfig,
    RuntimeConfig
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
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = '--' + key[len('--'):].replace('-', '_')
                    processed_args.append(f'{key}={value}')
                else:
                    processed_args.append('--' +
                                          arg[len('--'):].replace('-', '_'))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val

@dataclass
class EngineArgs:
    """Arguments for PipeFuser engine."""
    # Model arguments
    model: str
    download_dir: Optional[str] = None
    trust_remote_code: bool = False
    # Input arguments
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20
    prompt: Union[str, List[str]] = ""
    negative_prompt: Union[str, List[str]] = ""
    no_use_resolution_binning: bool = False
    # Runtime arguments
    seed: int = 42
    warmup_steps: int = 1
    output_type: str = "pil"
    # use_cuda_graph: bool = True
    # use_parallel_vae: bool = False
    # use_profiler: bool = False
    # Parallel arguments
        # data parallel
    data_parallel_degree: int = 1
    use_split_batch: bool = False
    do_classifier_free_guidance: bool = True
        # sequence parallel
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None
        # tensor parallel
    tensor_parallel_degree: int = 1
    split_scheme: Optional[str] = 'row'
        # pipefusion parallel
    pipefusion_parallel_degree: int = 1
    num_pipeline_patch: Optional[int] = None
    attn_layer_num_for_pp: Optional[List[int]] = None


    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser):
        """Shared CLI arguments for PipeFuser engine."""
        # Model arguments
        model_group = parser.add_argument_group('Model Options')
        model_group.add_argument('--model', type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS", help='Name or path of the huggingface model to use.')
        model_group.add_argument('--download-dir', type=nullable_str, default=EngineArgs.download_dir, help='Directory to download and load the weights, default to the default cache dir of huggingface.')
        model_group.add_argument('--trust-remote-code', action='store_true', help='Trust remote code from huggingface.')

        # Input arguments
        input_group = parser.add_argument_group('Input Options')
        input_group.add_argument("--height", type=int, default=1024, help="The height of image")
        input_group.add_argument("--width", type=int, default=1024, help="The width of image")
        input_group.add_argument("--prompt", type=str, nargs="*", default="", help="Prompt for the model.")
        input_group.add_argument("--negative_prompt", type=str, nargs="*", default="", help="Negative prompt for the model.")
        input_group.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps.")
        input_group.add_argument("--no_use_resolution_binning", action="store_true")

        # Runtime arguments
        runtime_group = parser.add_argument_group('Runtime Options')
        runtime_group.add_argument("--seed", type=int, default=42, help="Random seed for operations.")
        runtime_group.add_argument("--warmup_steps", type=int, default=1, help="Warmup steps in generation.")
        runtime_group.add_argument("--output_type", type=str, default="pil", help="Output type of the pipeline.")
        # runtime_group.add_argument("--use_cuda_graph", action="store_true")
        # runtime_group.add_argument("--use_parallel_vae", action="store_true")
        # runtime_group.add_argument("--use_profiler", action="store_true")

        # Parallel arguments
        parallel_group = parser.add_argument_group('Parallel Processing Options')
        parallel_group.add_argument("--do_classifier_free_guidance", action="store_true")
        parallel_group.add_argument("--use_split_batch", action="store_true", help="Use split batch in classifier_free_guidance. cfg_degree will be 2 if set")
        parallel_group.add_argument("--data_parallel_degree", type=int, default=1, help="Data parallel degree.")
        parallel_group.add_argument("--ulysses_degree", type=int, default=None, help="Ulysses sequence parallel degree. Used in attention layer.")
        parallel_group.add_argument("--ring_degree", type=int, default=None, help="Ring sequence parallel degree. Used in attention layer.")
        parallel_group.add_argument("--pipefusion_parallel_degree", type=int, default=1, help="Pipefusion parallel degree. Indicates the number of pipeline stages.")
        parallel_group.add_argument("--num_pipeline_patch", type=int, default=None, help="Number of patches the feature map should be segmented in pipefusion parallel.")
        parallel_group.add_argument("--attn_layer_num_for_pp", default=None, nargs="*", type=int, help="List representing the number of layers per stage of the pipeline in pipefusion parallel")
        parallel_group.add_argument("--tensor_parallel_degree", type=int, default=1, help="Tensor parallel degree.")
        parallel_group.add_argument("--split_scheme", type=str, default='row', help="Split scheme for tensor parallel.")
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_config(self, ) -> EngineConfig:
        if not torch.distributed.is_initialized():
            logger.warning("Distributed environment is not initialized. "
                           "Initializing...")
            init_distributed_environment(random_seed=self.seed)

        model_config = ModelConfig(
            model=self.model,
            download_dir=self.download_dir,
            trust_remote_code=self.trust_remote_code,
        )
        
        input_config = InputConfig(
            height=self.height,
            width=self.width,
            batch_size=len(self.prompt) if isinstance(self.prompt, list) else 1,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            use_resolution_binning=not self.no_use_resolution_binning,
        )

        runtime_config = RuntimeConfig(
            seed=self.seed,
            warmup_steps=self.warmup_steps,
            output_type=self.output_type,
            # use_cuda_graph=self.use_cuda_graph,
            # use_parallel_vae=self.use_parallel_vae,
            # use_profiler=self.use_profiler,
        )
        
        parallel_config = ParallelConfig(
            dp_config=DataParallelConfig(
                dp_degree=self.data_parallel_degree,
                use_split_batch=self.use_split_batch,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            ),
            sp_config=SequenceParallelConfig(
                ulysses_degree=self.ulysses_degree,
                ring_degree=self.ring_degree,
            ),
            tp_config=TensorParallelConfig(
                tp_degree=self.tensor_parallel_degree,
                split_scheme=self.split_scheme,
            ),
            pp_config=PipeFusionParallelConfig(
                pp_degree=self.pipefusion_parallel_degree,
                num_pipeline_patch=self.num_pipeline_patch,
                attn_layer_num_for_pp=self.attn_layer_num_for_pp,
            ),
        )

        return EngineConfig(
            model_config=model_config,
            input_config=input_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
        )
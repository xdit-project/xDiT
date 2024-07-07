import sys
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.distributed

from pipefuser.logger import init_logger
from pipefuser.distributed.parallel_state import init_distributed_environment
from pipefuser.config.config import (
    EngineConfig,
    ParallelConfig,
    TensorParallelConfig,
    PipeFusionParallelConfig,
    SequenceParallelConfig,
    DataParallelConfig,
    ModelConfig,
    DataConfig,
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
                    key = '--' + key[len('--'):].replace('_', '-')
                    processed_args.append(f'{key}={value}')
                else:
                    processed_args.append('--' +
                                          arg[len('--'):].replace('_', '-'))
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
    scheduler: Optional[str] = "dpmsolver_multistep"
    # Data arguments
    height: int = 1024
    width: int = 1024
    batch_size: Optional[int] = None
    use_resolution_binning: bool = True
    # Runtime arguments
    seed: int = 42
    warmup_steps: int = 1
    use_cuda_graph: bool = True
    use_parallel_vae: bool = False
    use_profiler: bool = False
    # Parallel arguments
        # data parallel
    data_parallel_degree: int = 1
    use_split_batch: bool = False
    do_classifier_free_guidance: bool = True
        # sequence parallel
    ulysse_degree: Optional[int] = None
    ring_degree: Optional[int] = None
        # tensor parallel
    tensor_parallel_degree: int = 1
    split_scheme: Optional[str] = 'row'
        # pipefusion parallel
    pipefusion_parallel_degree: int = 1
    pipeline_patch_num: Optional[int] = None
    attn_layer_num_for_pp: Optional[List[int]] = None


    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser):
        """Shared CLI arguments for PipeFuser engine."""
        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            help='Name or path of the huggingface model to use.')
        parser.add_argument('--download-dir',
                            type=nullable_str,
                            default=EngineArgs.download_dir,
                            help='Directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='Trust remote code from huggingface.')
        parser.add_argument("--scheduler",
                            "-s",
                            default="dpm-solver",
                            type=str,
                            choices=["dpm-solver", 
                                     "ddim", 
                                     "dpmsolver_multistep"],
                            help="Scheduler to use.")
        # Data arguments
        parser.add_argument("--height",
                            type=int,
                            default=1024,
                            help="The height of image")
        parser.add_argument("--width",
                            type=int,
                            default=1024,
                            help="The width of image")
        parser.add_argument("--no_use_resolution_binning",
                            action="store_true")
        # Runtime arguments
        parser.add_argument("--seed",
                            type=int,
                            default=42,
                            help="Random seed for operations.")
        parser.add_argument("--warmup_steps",
                            type=int,
                            default=1,
                            help="Warmup steps.")
        parser.add_argument("--use_cuda_graph",
                            action="store_true")
        parser.add_argument("--use_parallel_vae",
                            action="store_true")
        parser.add_argument("--use_profiler",
                            action="store_true")
        # Parallel arguments
        parser.add_argument("--data_parallel_degree",
                            type=int,
                            default=1,
                            help="Data parallel degree.")
        parser.add_argument("--use_split_batch",
                            action="store_true")
        parser.add_argument("--do_classifier_free_guidance",
                            action="store_true")
        parser.add_argument("--ulysses_degree",
                            type=int,
                            default=None)
        parser.add_argument("--ring_degree",
                            type=int,
                            default=None)
        parser.add_argument("--tensor_parallel_degree",
                            type=int,
                            default=1)
        parser.add_argument("--split_scheme",
                            type=str,
                            default='row')
        parser.add_argument("--pipefusion_parallel_degree",
                            type=int,
                            default=1)
        parser.add_argument("--pipeline_patch_num",
                            type=int,
                            default=None)
        parser.add_argument("--attn_layer_num_for_pp",
                            default=None,
                            nargs="*",
                            type=int)
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
            init_distributed_environment()

        model_config = ModelConfig(
            model=self.model,
            download_dir=self.download_dir,
            trust_remote_code=self.trust_remote_code,
            scheduler=self.scheduler,
        )
        
        data_config = DataConfig(
            height=self.height,
            width=self.width,
            batch_size=self.batch_size,
            use_resolution_binning=self.use_resolution_binning,
        )

        runtime_config = RuntimeConfig(
            seed=self.seed,
            warmup_steps=self.warmup_steps,
            use_cuda_graph=self.use_cuda_graph,
            use_parallel_vae=self.use_parallel_vae,
            use_profiler=self.use_profiler,
        )
        
        parallel_config = ParallelConfig(
            dp_config=DataParallelConfig(
                dp_degree=self.data_parallel_degree,
                use_split_batch=self.use_split_batch,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            ),
            sp_config=SequenceParallelConfig(
                ulysses_degree=self.ulysse_degree,
                ring_degree=self.ring_degree,
            ),
            tp_config=TensorParallelConfig(
                tp_degree=self.tensor_parallel_degree,
                split_scheme=self.split_scheme,
            ),
            pp_config=PipeFusionParallelConfig(
                pp_degree=self.pipefusion_parallel_degree,
                pipeline_patch_num=self.pipeline_patch_num,
                attn_layer_num_for_pp=self.attn_layer_num_for_pp,
            ),
        )

        return EngineConfig(
            model_config=model_config,
            data_config=data_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
        )
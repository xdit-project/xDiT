import argparse
import logging
import torch
import os
import gc
from typing import Tuple, Any


# Allow single-GPU runs to work without torchrun, debugging
if not os.environ.get("RANK"):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"


from xfuser.model_executor.models.runner_models.base_model import (
    MODEL_REGISTRY,
)
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_runtime_state,
)
from xfuser.core.utils.runner_utils import log, is_last_process
from xfuser import xFuserArgs


class xFuserModelRunner:
    """ A generic model runner for models supported by xDiT """

    def __init__(self, config: dict):
        self.config = config
        self.model = self._select_model(config.model, config)
        self.is_initialized = False

    def _select_model(self, model_name: str, config: dict) -> "xFuserModel":
        """ Select and instantiate model from registry"""
        model = MODEL_REGISTRY.get(model_name, None)
        if not model:
            raise ValueError(f"Model {model_name} not found in registry.")
        return model(config)


    def initialize(self, input_args: dict) -> None:
        """ Initialize the model and runtime state """
        if self.is_initialized:
            log("Model already initialized, skipping...")
            return

        log(f"Initializing model: {self.config.model}")
        self.model.initialize(input_args)

        self.is_initialized = True
        log("Model initialization complete.")

    def run(self, input_args: dict[str, Any]) -> Tuple[Any, list]:
        """ Run the model with given input arguments """
        if not self.is_initialized:
            raise RuntimeError("ModelRunner not initialized. Call initialize() before run().")

        log("Running model...")
        output, timings = self.model.run(input_args)
        return output, timings

    def preprocess_args(self, input_args: dict|argparse.Namespace) -> dict:
        """ Preprocess input arguments before passing them to the model """
        if type(input_args) == argparse.Namespace:
            input_args = vars(input_args)
        return self.model.preprocess_args(input_args)

    def validate_args(self, input_args: dict|argparse.Namespace) -> None:
        """ Validate input arguments """
        if type(input_args) == argparse.Namespace:
            input_args = vars(input_args)
        return self.model.validate_args(input_args)

    def profile(self, input_args: dict) -> Tuple[Any, list, Any]:
        """ Profile the model execution """
        output, timings, profile_object = self.model.profile(input_args)
        return output, timings, profile_object

    def print_args(self, args: argparse.Namespace) -> None:
        """ Print the arguments from the Namespace """
        log("Model Runner Arguments:")
        for arg, value in vars(args).items():
            log(f"  {arg}: {value}")

    def cleanup(self) -> None:
        """ Cleanup resources after model execution """
        get_runtime_state().destroy_distributed_env()
        del self.model.pipe
        gc.collect()
        torch.cuda.empty_cache()
        log("Cleaned up resources.")


    def save(self, output=None, timings=None, profile=None, save_once: bool = True) -> None:
        """ Save model output, timings and profiles to file, if applicable """
        if save_once: # TODO: add rank info to file names so this can even make sense
            if not is_last_process():
                return
        if output:
            self.model.save_output(output) # Handle different output types
        if timings:
            self.model.save_timings(timings)
        if profile:
            self.model.save_profile(profile)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of iterations to run the model.")
    parser.add_argument("--repetition_sleep_duration", type=int, default=None, help="The duration to sleep in between different pipe calls in seconds.")
    parser.add_argument("--profile", default=False, action="store_true", help="Whether to run Pytorch profiler. See --profile_wait, --profile_warmup and --profile_active for profiler specific warmup.")
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
    args = xFuserArgs.add_cli_args(parser).parse_args()
    runner = xFuserModelRunner(args)
    runner.print_args(args)
    runner.validate_args(args)
    input_args = runner.preprocess_args(args)
    runner.initialize(input_args)
    if args.profile:
        out, timing, profile = runner.profile(input_args)
        runner.save(profile=profile)
    else:
        output, timings = runner.run(input_args)
        runner.save(output=output, timings=timings)

    runner.cleanup()

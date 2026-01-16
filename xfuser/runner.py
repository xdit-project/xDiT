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


from diffusers.utils import BaseOutput
from xfuser.model_executor.models.runner_models.base_model import (
    MODEL_REGISTRY,
    xFuserModel,
)
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_runtime_state,
)
from xfuser.core.utils.runner_utils import log, is_last_process
from xfuser import xFuserArgs


class xFuserModelRunner:
    """ A generic model runner for models supported by xDiT """

    def __init__(self, config: dict) -> None:
        xfuser_config = xFuserArgs.from_runner_args(config)
        # Runs the config through argument parsing and validation - not the cleanest solution but has to be done inside the runner
        engine_config, input_config = xfuser_config.create_config()

        self.config = xfuser_config
        self.model = self._select_model(xfuser_config.model, xfuser_config)
        self.is_initialized = False

    def _select_model(self, model_name: str, config: xFuserArgs) -> xFuserModel:
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

    def run(self, input_args: dict) -> Tuple[BaseOutput, list]:
        """ Run the model with given input arguments """
        if not self.is_initialized:
            raise RuntimeError("ModelRunner not initialized. Call initialize() before run().")

        log("Running model...")
        output, timings = self.model.run(input_args)
        return output, timings

    def preprocess_args(self, input_args: dict) -> dict:
        """ Preprocess input arguments before passing them to the model """
        return self.model.preprocess_args(input_args)

    def profile(self, input_args: dict) -> Tuple[BaseOutput, list, Any]:
        """ Profile the model execution """
        output, timings, profile_object = self.model.profile(input_args)
        return output, timings, profile_object

    def print_args(self, args: dict) -> None:
        """ Print the arguments from the Namespace """
        log("Model Runner Arguments:")
        for arg, value in args.items():
            log(f"  {arg}: {value}")

    def cleanup(self) -> None:
        """ Cleanup resources after model execution """
        get_runtime_state().destroy_distributed_env()
        del self.model.pipe
        gc.collect()
        torch.cuda.empty_cache()
        log("Cleaned up resources.")


    def save(self, output: BaseOutput = None, timings: list = None, profile: Any = None, save_once: bool = True) -> None:
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
    xfuser_args = xFuserArgs.add_runner_args(parser).parse_args()
    args = vars(xfuser_args)
    runner = xFuserModelRunner(args)
    runner.print_args(args)

    input_args = runner.preprocess_args(args)
    runner.initialize(input_args)

    if xfuser_args.profile:
        out, timing, profile = runner.profile(input_args)
        runner.save(profile=profile)
    else:
        output, timings = runner.run(input_args)
        runner.save(output=output, timings=timings)

    runner.cleanup()

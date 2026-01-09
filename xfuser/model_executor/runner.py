import abc
import argparse
import logging
import torch
from typing import Optional, Tuple, Any
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser

from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    is_dp_last_group,
    initialize_runtime_state,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s - %(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}

def log(message: str):
    """Log message only from the last process to avoid duplicates."""
    if is_dp_last_group():
        logger.info(message)

def register_model(name: str):
    """ Decorator to register a model in the registry. """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model: {name}")
        return cls
    return decorator


class xFuserModelRunner:
    """ A generic model runner for models supported by xDiT """

    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.model: Optional[xFuserModel] = None
        self.is_initialized = False

    def _select_model(self, model_name: str, config: argparse.Namespace) -> "xFuserModel":
        """ Select and instantiate model from registry"""
        model = MODEL_REGISTRY.get(model_name, None)
        if not model:
            raise ValueError(f"Model {model_name} not found in registry.")
        return model(config)

    def initialize(self) -> None:
        """ Initialize the model and runtime state """
        if self.is_initialized:
            log("Model already initialized, skipping...")
            return
        log(f"Initializing model: {self.config.model}")
        self.model = self._select_model(self.config.model, self.config)
        self.model.initialize()

        log("Initializing runtime state...")
        engine_args = xFuserArgs.from_cli_args(self.config)
        engine_config, input_config = engine_args.create_config()
        initialize_runtime_state(self.model.pipe, engine_config)

        self.is_initialized = True
        log("Model initialization complete.")

    def run(self, input_args: dict[str, Any]) -> Tuple[Any, list]:
        """ Run the model with given input arguments """
        if not self.is_initialized:
            raise RuntimeError("ModelRunner not initialized. Call initialize() before run().")

        timings = []
        preprocessed_args = self._preprocess_args(input_args)
        for iteration in range(self.config.num_repetitions):
            log(f"Running iteration {iteration + 1}/{self.config.num_repetitions}")
            out, timing = self.model.run_timed_pipe(preprocessed_args)
            timings.append(timing)
            log(f"Iteration {iteration + 1} completed in {timing:.2f} ms")

        log(f"Average time over {self.config.num_repetitions} runs: {sum(timings) / len(timings):.2f} ms")
        return out, timings

    def profile(self, input_args: dict):
        """ Profile the model execution """
        raise NotImplementedError("Profiling not implemented yet.")

    def _preprocess_args(self, input_args: dict):
        """ Preprocess input arguments before passing them to the model """

        # Returning the values as dict instead of Namespace for easier handling
        return vars(input_args)

    def save(self, output):
        """ Save model output, timings and profiles to file """
        raise NotImplementedError("Save method not implemented yet.")

    def cleanup(self):
        """ Cleanup resources after model execution """
        raise NotImplementedError("Cleanup method not implemented yet.")





class xFuserModel(abc.ABC):
    """ Base class for xFuser models """

    def __init__(self, config: dict):
        self.config = config
        self.pipe = None

    def initialize(self):
        """ Load the model pipeline """
        self.pipe = self._load_model()

    def run_timed_pipe(self, input_args: dict):
        """ Run a a full pipeline with timing information """

        events = {
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        torch.cuda.synchronize()
        events["start"].record()
        out = self._run_pipe(input_args)
        events["end"].record()
        torch.cuda.synchronize()
        elapsed_time = events["start"].elapsed_time(events["end"])
        return out, elapsed_time

    @abc.abstractmethod
    def _run_pipe(self, input_args: dict):
        """ Execute the pipeline. Muyst be implemented by subclasses. """
        pass

    #@abc.abstractmethod
    def profile(self, input_args: dict):
        pass

    @abc.abstractmethod
    def _load_model(self):
        """ Load the model. Must be implemented by subclasses. """
        pass

    def enable_slicing(self):
        raise NotImplementedError("This model does not support slicing.")


from diffusers import ZImagePipeline
from xfuser.model_executor.models.transformers.transformer_z_image import xFuserZImageTransformer2DWrapper

@register_model("Tongyi-MAI/Z-Image-Turbo")
class xFuserZImageTurboModel(xFuserModel):

    model_name: str = "Tongyi-MAI/Z-Image-Turbo"
    model_type: str = "image"

    def _load_model(self):
        transformer = xFuserZImageTransformer2DWrapper.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe = pipe.to("cuda")
        return pipe
        #pipe = opipe.to(f"cuda:{local_rank}")

    def _run_pipe(self, input_args: dict):
        prompt = str(input_args["prompt"])
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=prompt,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output



if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument("--num_repetitions", type=int, default=1, help="Number of repetitions to run the model.")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    print(args)
    runner = xFuserModelRunner(args)
    runner.initialize()
    out, timing = runner.run(args)
    print(out)
import abc
import math
import argparse
import logging
import functools
import torch
import json
import copy
import os
import gc
import numpy as np
from typing import Optional, Tuple, Any, Union, Dict
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers.utils import load_image

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    initialize_runtime_state,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s - %(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {}

# Allow single-GPU runs to work without torchrun, debugging
if not os.environ.get("RANK"):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

def is_last_process():
    """ Checks based on env rank and world size if this is last process in """
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    if rank == world_size - 1:
        return True
    return False


def log(message: str):
    """Log message only from the last process to avoid duplicates."""
    if is_last_process():
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

    def preprocess_args(self, input_args: dict) -> dict:
        """ Preprocess input arguments before passing them to the model """
        return self.model.preprocess_args(input_args)

    def validate_args(self, input_args: dict):
        """ Validate input arguments """
        return self.model.validate_args(input_args)

    def profile(self, input_args: dict):
        """ Profile the model execution """
        output, timings, profile_object = self.model.profile(input_args)
        return output, timings, profile_object

    def print_args(self, args: argparse.Namespace):
        """ Print the arguments from the Namespace """
        log("Model Runner Arguments:")
        for arg, value in vars(args).items():
            log(f"  {arg}: {value}")

    def cleanup(self):
        """ Cleanup resources after model execution """
        get_runtime_state().destroy_distributed_env()
        del self.model.pipe
        gc.collect()
        torch.cuda.empty_cache()
        log("Cleaned up resources.")


    def save(self, output=None, timings=None, profile=None, save_once: bool = True):
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

class xFuserModel(abc.ABC):
    """ Base class for xFuser models """

    def __init__(self, config: dict):
        self.config = config
        self.pipe = None

    def initialize(self, input_args):
        """ Load the model pipeline """
        # These initialize some internal states as a side-effect ...
        engine_args = xFuserArgs.from_cli_args(self.config)
        engine_config, input_config = engine_args.create_config()

        self.pipe = self._load_model()


        log("Initializing runtime state...")
        initialize_runtime_state(self.pipe, engine_config)

        if self.config.use_torch_compile:
            log("Torch.compile enabled. Warming up torch compiler ...")
            self._compile_model(input_args)

    def _compile_model(self, input_args):
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe = torch.compile(self.pipe, mode="default") # TODO: Configurable

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2  # Reduce steps for warmup # TODO: make this more generic
        self._run_timed_pipe(compile_args)


    def run(self, input_args):
        """ Run the model with given input arguments and return output and timings """
        timings = []

        self._run_warmup_calls(input_args)
        for iteration in range(self.config.num_iterations):
            log(f"Running iteration {iteration + 1}/{self.config.num_iterations}")
            out, timing = self._run_timed_pipe(input_args)
            timings.append(timing)
            log(f"Iteration {iteration + 1} completed in {timing:.2f}s")

        log(f"Average time over {self.config.num_iterations} runs: {sum(timings) / len(timings):.2f}s")
        return out, timings

    def _run_warmup_calls(self, input_args):
        """ Run initial warmup calls if specified """
        if self.config.warmup_calls:
            log(f"Warming up model with {self.config.warmup_calls} calls...")
            for iteration in range(self.config.warmup_calls):
                log(f"Warmup iteration {iteration + 1}/{self.config.warmup_calls}")
                self._run_timed_pipe(input_args)
            log(f"Warmup complete.")

    def profile(self, input_args: dict):
        """ Profile the model execution """
        schedule = torch.profiler.schedule(
            wait=self.config.profile_wait,
            warmup=self.config.profile_warmup,
            active=self.config.profile_active,
        )
        num_repetitions = self.config.profile_wait + self.config.profile_warmup + self.config.profile_active

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
            record_shapes=True, #TODO: make configurable
            with_stack=False, #TODO: experimental config necessary?
        ) as profile_object:
            for iteration in range(num_repetitions):
                log(f"Profiling iteration {iteration + 1}/{num_repetitions}")
                with record_function("model_inference"):
                    out, timing = self._run_timed_pipe(input_args)
                profile_object.step()
                log(f"Profiling iteration {iteration + 1} completed in {timing:.2f}s")
        return out, [], profile_object

    def preprocess_args(self, input_args: dict) -> dict:
        """ Preprocess input arguments before passing them to the model """
        if type(input_args) == argparse.Namespace:
            input_args = vars(input_args)
        args = copy.deepcopy(input_args)
        args = self._preprocess_args_images(args)
        args = self._preprocess_args_custom(args)
        return args


    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess image inputs if necessary """
        # For legacy reasons, we support several ways to pass images
        images = []
        image_path_keys = ["image_path", "image_paths", "input_images"]
        for key in image_path_keys:
            if key in input_args:
                paths = input_args[key]
                if isinstance(paths, str):
                    paths = [paths]
                for path in paths:
                    image = load_image(path)
                    images.append(image)
        input_args["input_images"] = images
        return input_args

    def _preprocess_args_custom(self, input_args: dict) -> dict:
        """ Preprocess custom inputs if necessary """
        return input_args

    def save_output(self, output):
        """ Saves the output based on its type """
        if self.model_output_type == "image":
            from PIL import Image
            output_image = output.images[0] # TODO: BS > 1
            output_name = self.get_output_name()
            output_path = f"{self.config.output_directory}/{output_name}.png"
            output_image.save(output_path)
            log(f"Output image saved to {output_path}")

        elif self.model_output_type == "video":
            output_name = self.get_output_name()
            output_path = f"{self.config.output_directory}/{output_name}.mp4"
            output.save(output_path)
            log(f"Output video saved to {output_path}")

    def save_timings(self, timings: list):
        timing_file = open(f"{self.config.output_directory}/timings.json", "w")
        json.dump(timings, timing_file)
        timing_file.close()
        log(f"Timings saved to {self.config.output_directory}/timings.json")

    def save_profile(self, profile):
        profile_file = f"{self.config.output_directory}/profile_trace_rank_{get_world_group().rank}.json"
        profile.export_chrome_trace(profile_file)
        log(f"Profile trace saved to {profile_file}")

    def _run_timed_pipe(self, input_args: dict):
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
        elapsed_time = events["start"].elapsed_time(events["end"]) / 1000  # Convert to seconds
        return out, elapsed_time

    def get_output_name(self) -> str:
        """ Generate a unique output name based on model and config """
        use_compile = self.config.use_torch_compile
        ulysses_degree = self.config.ulysses_degree or 1
        ring_degree = self.config.ring_degree or 1
        height = self.config.height
        width = self.config.width
        return f"{self.output_name}_u{ulysses_degree}r{ring_degree}_tc_{use_compile}_{height}x{width}"

    @abc.abstractmethod
    def _run_pipe(self, input_args: dict):
        """ Execute the pipeline. Muyst be implemented by subclasses. """
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
@register_model("Z-Image-Turbo")
class xFuserZImageTurboModel(xFuserModel):

    model_name: str = "Tongyi-MAI/Z-Image-Turbo"
    output_name: str = "z_image_turbo"
    model_output_type: str = "image"

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
        local_rank = get_world_group().local_rank
        pipe = pipe.to(f"cuda:{local_rank}")
        return pipe

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
    args = xFuserArgs.add_cli_args(parser).parse_args()
    runner = xFuserModelRunner(args)
    runner.print_args(args)
    input_args = runner.preprocess_args(args)
    runner.initialize(input_args)
    if args.profile:
        out, timing, profile = runner.profile(input_args)
        runner.save(profile=profile)
    else:
        output, timings = runner.run(input_args)
        runner.save(output=output, timings=timings)

    runner.cleanup()

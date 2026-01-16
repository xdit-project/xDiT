import abc
import torch
import copy
import argparse
import json
from PIL.Image import Image
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import load_image, export_to_video, BaseOutput
import numpy as np
from xfuser.config import args, xFuserArgs
from xfuser.core.utils.runner_utils import (
    log,
)

from xfuser.core.distributed import (
    get_world_group,
    initialize_runtime_state,
    init_distributed_environment,
)

MODEL_REGISTRY = {}

def register_model(name: str) -> callable:
    """ Decorator to register a model in the registry. """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        log(f"Registered model: {name}", debug=True)
        return cls
    return decorator

@dataclass
class ModelCapabilities:
    """ Class to define model capabilities """
    ulysses_degree: bool = True  # All xDiT models support these
    ring_degree: bool = True
    pipefusion_parallel_degree: bool = False
    tensor_parallel_degree: bool = False
    use_cfg_parallel: bool = False
    use_parallel_vae: bool = False
    enable_slicing: bool = False
    enable_tiling: bool = False
    use_fp8_gemms: bool = False

@dataclass
class DefaultInputValues:
    """ Class to define model specific default input values """
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    max_sequence_length: Optional[int] = None

class xFuserModel(abc.ABC):
    """ Base class for xFuser models """

    capabilities: ModelCapabilities = ModelCapabilities()
    default_input_values: DefaultInputValues = DefaultInputValues()
    valid_tasks: list = []
    model_output_type: str = ""
    fps: int = 24

    def __init__(self, config: xFuserArgs) -> None:
        self._validate_config(config)
        self.config = config
        self.pipe = None

    def initialize(self, input_args: dict) -> None:
        """ Load the model pipeline """
        # These initialize some internal states as a side-effect ...

        if not torch.distributed.is_initialized():
            log("Initializing distributed environment.. .")
            init_distributed_environment()

        log("Loading model pipeline...")
        self.pipe = self._load_model()

        log("Initializing runtime state...")
        self.engine_config, _ = self.config.create_config()
        initialize_runtime_state(self.pipe, self.engine_config)

        self._post_load_and_state_initialization(input_args)
        self._enable_options()

        if self.config.use_torch_compile:
            log("Torch.compile enabled. Warming up torch compiler ...")
            self._compile_model(input_args)

    def _enable_options(self) -> None:
        """ Enable model options based on config"""
        if self.config.enable_slicing:
            log("Enabling VAE slicing...")
            self.pipe.vae.enable_slicing()

        if self.config.enable_tiling:
            log("Enabling VAE tiling...")
            self.pipe.vae.enable_tiling()

        if self.config.enable_sequential_cpu_offload:
            log("Enabling sequential CPU offload...")
            self.pipe.enable_sequential_cpu_offload()
        elif self.config.enable_model_cpu_offload:
            log("Enabling model CPU offload...")
            self.pipe.enable_model_cpu_offload()


    def _validate_config(self, config: xFuserArgs) -> None:
        """ Validate if the model supports requested config """
        for key in ModelCapabilities.__annotations__.keys():
            config_value = getattr(config, key)
            if type(config_value) == int:
                if not getattr(self.capabilities, key) and config_value > 1:
                    raise ValueError(f"Model {self.model_name} does not support {key}.")
            if type(config_value) == bool:
                if config_value and not getattr(self.capabilities, key):
                    raise ValueError(f"Model {self.model_name} does not support {key}.")

        possible_task = getattr(config, "task", None)
        if possible_task and self.valid_tasks:
            if possible_task not in self.valid_tasks:
                raise ValueError(f"Model {self.model_name} does not support task '{possible_task}'. Supported tasks: {self.valid_tasks}")
        if possible_task and not self.valid_tasks:
            raise ValueError(f"Model {self.model_name} does not support multiple tasks, but task '{possible_task}' was specified.")
        if not possible_task and self.valid_tasks:
            raise ValueError(f"Model {self.model_name} requires a task to be specified. Supported tasks: {self.valid_tasks}")


    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe = torch.compile(self.pipe, mode="default") # TODO: Configurable

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2  # Reduce steps for warmup # TODO: make this more generic
        self._run_timed_pipe(compile_args)


    def run(self, input_args: dict) -> Tuple[BaseOutput, list]:
        """ Run the model with given input arguments and return output and timings """
        self._validate_args(input_args)
        timings = []

        self._run_warmup_calls(input_args)
        for iteration in range(self.config.num_iterations):
            log(f"Running iteration {iteration + 1}/{self.config.num_iterations}")
            out, timing = self._run_timed_pipe(input_args)
            timings.append(timing)
            log(f"Iteration {iteration + 1} completed in {timing:.2f}s")

        log(f"Average time over {self.config.num_iterations} runs: {sum(timings) / len(timings):.2f}s")
        return out, timings

    def _run_warmup_calls(self, input_args: dict) -> None:
        """ Run initial warmup calls if specified """
        if self.config.warmup_calls:
            log(f"Warming up model with {self.config.warmup_calls} calls...")
            for iteration in range(self.config.warmup_calls):
                log(f"Warmup iteration {iteration + 1}/{self.config.warmup_calls}")
                self._run_timed_pipe(input_args)
            log(f"Warmup complete.")

    def profile(self, input_args: dict) -> Tuple[BaseOutput, list, torch.profiler.profiler.profile]:
        """ Profile the model execution """
        self._validate_args(input_args)

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
        args = copy.deepcopy(input_args)

        # Apply model specific default input values
        for default_key, _ in DefaultInputValues.__annotations__.items():
            if args.get(default_key, None) is None:
                default_value = getattr(self.default_input_values, default_key)
                if default_value is not None:
                    args[default_key] = default_value
                    log(f"Parameter '{default_key}' not specified. Using model-specific default value: {default_value}")

        args = self._preprocess_args_images(args)
        return args


    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess image inputs if necessary """
        images = [load_image(path) for path in input_args.get("input_images", [])]
        input_args["input_images"] = images
        return input_args

    def save_output(self, output: BaseOutput) -> None:
        """ Saves the output based on its type """
        if self.model_output_type == "image":
            output_image = output.images[0] # TODO: BS > 1
            output_name = self.get_output_name()
            output_path = f"{self.config.output_directory}/{output_name}.png"
            output_image.save(output_path)
            log(f"Output image saved to {output_path}")

        elif self.model_output_type == "video":
            output_video = output.frames[0] # TODO: BS > 1
            output_name = self.get_output_name()
            output_path = f"{self.config.output_directory}/{output_name}.mp4"
            export_to_video(output_video, output_path, fps=self.fps)
            log(f"Output video saved to {output_path}")
        else:
            raise NotImplementedError(f"Saving output of type {self.model_output_type} is not implemented.")

    def save_timings(self, timings: list) -> None:
        timing_file = open(f"{self.config.output_directory}/timings.json", "w")
        json.dump(timings, timing_file, indent=2)
        timing_file.close()
        log(f"Timings saved to {self.config.output_directory}/timings.json")

    def save_profile(self, profile: torch.profiler.profiler.profile) -> None:
        profile_file = f"{self.config.output_directory}/profile_trace_rank_{get_world_group().rank}.json"
        profile.export_chrome_trace(profile_file)
        log(f"Profile trace saved to {profile_file}")

    def _run_timed_pipe(self, input_args: dict) -> Tuple[BaseOutput, float]:
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
        name = f"{self.output_name}_u{ulysses_degree}r{ring_degree}_tc_{use_compile}_{height}x{width}"
        if self.config.task:
            name += f"_{self.config.task}"
        return name

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        """ Hook for any post model-load and state initialization """
        local_rank = get_world_group().local_rank
        self.pipe = self.pipe.to(f"cuda:{local_rank}")

    @abc.abstractmethod
    def _run_pipe(self, input_args: dict) -> BaseOutput:
        """ Execute the pipeline. Muyst be implemented by subclasses. """
        pass

    @abc.abstractmethod
    def _load_model(self) -> DiffusionPipeline:
        """ Load the model. Must be implemented by subclasses. """
        pass

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments. Can be overridden by subclasses. """
        pass
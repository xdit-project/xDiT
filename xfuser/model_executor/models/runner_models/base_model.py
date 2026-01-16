import abc
import torch
import copy
import argparse
import json
from PIL.Image import Image
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from torch.profiler import profile, record_function, ProfilerActivity
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_, _is_linear
from torchao.quantization.quantize_.common import KernelPreference
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import load_image, export_to_video, BaseOutput
import numpy as np
from xfuser.config import args, xFuserArgs
from xfuser.core.utils.runner_utils import (
    log,
    is_last_process,
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


        self._enable_options(self.pipe)

        if self.config.use_torch_compile:
            log("Torch.compile enabled. Warming up torch compiler ...")
            self._compile_model(input_args)

    def _enable_options(self, pipe: DiffusionPipeline) -> None:
        """ Enable model options based on config"""
        if self.config.enable_slicing:
            log("Enabling VAE slicing...")
            pipe.vae.enable_slicing()

        if self.config.enable_tiling:
            log("Enabling VAE tiling...")
            pipe.vae.enable_tiling()

        if self.config.enable_sequential_cpu_offload:
            log("Enabling sequential CPU offload...")
            pipe.enable_sequential_cpu_offload()
        elif self.config.enable_model_cpu_offload:
            log("Enabling model CPU offload...")
            pipe.enable_model_cpu_offload()


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

    def profile(self, input_args: dict) -> Tuple[BaseOutput, list, torch.profiler.profile]:
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
        # For legacy reasons, we support several ways to pass images
        images = [load_image(path) for path in input_args.get("input_images", [])]
        input_args["input_images"] = images
        return input_args

    def save_output(self, output: BaseOutput) -> None:
        """ Saves the output based on its type """
        if self.model_output_type == "image":
            from PIL import Image
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
        json.dump(timings, timing_file)
        timing_file.close()
        log(f"Timings saved to {self.config.output_directory}/timings.json")

    def save_profile(self, profile) -> None:
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

    def _resize_and_crop_image(self, image: Image, target_height: int, target_width:int , mod_value: int) -> Image:
        """ Resize and center-crop image to target dimensions """

        ##TODO: move this func to utils
        target_height_aligned = target_height // mod_value * mod_value
        target_width_aligned = target_width // mod_value * mod_value

        if is_last_process():
            print("Force output size mode enabled.")
            print(f"Input image resolution: {image.height}x{image.width}")
            print(f"Requested output resolution: {target_height}x{target_width}")
            print(f"Aligned output resolution (multiple of {mod_value}): {target_height_aligned}x{target_width_aligned}")

        # Step 1: Resize image maintaining aspect ratio so both dimensions >= target
        img_width, img_height = image.size

        # Calculate scale factor to ensure both dimensions are at least target size
        scale_width = target_width_aligned / img_width
        scale_height = target_height_aligned / img_height
        scale = max(scale_width, scale_height)  # Use max to ensure both dims are >= target

        # Resize with aspect ratio preserved
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        image = image.resize((new_width, new_height))

        if is_last_process():
            print(f"Resized image to: {new_height}x{new_width} (maintaining aspect ratio)")

        # Step 2: Crop from center to get exact target dimensions
        left = (new_width - target_width_aligned) // 2
        top = (new_height - target_height_aligned) // 2
        image = image.crop((left, top, left + target_width_aligned, top + target_height_aligned))

        if is_last_process():
            print(f"Cropped from center to: {target_height_aligned}x{target_width_aligned}")
        return image



    def _resize_image_to_max_area(self, image: Image, input_height: int, input_width: int, mod_value: int) -> Image:
        """ Resize image to fit within max area while retaining aspect ratio """
        ##TODO: move to utils

        max_area = input_height * input_width
        width, height = image.size
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area /aspect_ratio)) // mod_value * mod_value

        image = image.resize((width, height))
        if is_last_process():
            log(f"Resized image to {image.width}x{image.height} to fit within max area {width}x{height}")
        return image


    def _quantize_linear_layers_to_fp8(self, module_or_module_list_to_quantize: torch.nn.Module | torch.nn.ModuleList,
        filter_fn: Callable[[torch.nn.Module, str], bool] = _is_linear,
        device: Optional[torch.device] = None) -> None:
        config = Float8DynamicActivationFloat8WeightConfig(
                  granularity=PerTensor(),
                  set_inductor_config=False,
                  kernel_preference=KernelPreference.AUTO
            )
        """Quantize all linear layers in the given module or module list to FP8."""
        log("Quantizing linear layers to FP8...")
        if isinstance(module_or_module_list_to_quantize, torch.nn.Module):
            quantize_(module_or_module_list_to_quantize, config=config, filter_fn=filter_fn, device=device)
        elif isinstance(module_or_module_list_to_quantize, torch.nn.ModuleList):
            for module in module_or_module_list_to_quantize:
                quantize_(module, config=config, filter_fn=filter_fn, device=device)
        else:
            raise ValueError(f"Failed to quantize linear layers. Invalid module type: {type(module_or_module_list_to_quantize)}, expected torch.nn.Module or torch.nn.ModuleList")

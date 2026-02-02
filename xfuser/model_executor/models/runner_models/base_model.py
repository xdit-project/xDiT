import abc
import torch
import copy
import argparse
import json
from PIL.Image import Image
from typing import Callable, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import numpy as np
from xfuser.config import args, xFuserArgs
from xfuser.core.distributed.parallel_state import get_sp_group
from xfuser.core.utils.runner_utils import (
    log,
    load_dataset_prompts,
    quantize_linear_layers_to_fp8,
    rgetattr,
)

from xfuser.core.distributed import (
    get_world_group,
    initialize_runtime_state,
    get_runtime_state,
    init_distributed_environment,
    children_to_device,
    shard_transformer_blocks,
)


MODEL_REGISTRY = {}

def register_model(name: str) -> Callable:
    """ Decorator to register a model in the registry. """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

@dataclass(frozen=True)
class ModelCapabilities:
    """ Class to define model capabilities """
    # Parallelization
    ulysses_degree: bool = True  # All xDiT models support these
    ring_degree: bool = True
    pipefusion_parallel_degree: bool = False
    data_parallel_degree: bool = False
    tensor_parallel_degree: bool = False
    use_cfg_parallel: bool = False
    use_parallel_vae: bool = False
    use_fsdp: bool = False
    # Memory optimizations
    enable_slicing: bool = False
    enable_tiling: bool = False
    # Other features
    use_fp8_gemms: bool = False
    use_hybrid_fp8_attn: bool = False

@dataclass(frozen=True)
class DefaultInputValues:
    """ Class to define model specific default input values """
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    max_sequence_length: Optional[int] = None
    num_hybrid_bf16_attn_steps: Optional[int] = None

@dataclass
class ModelSettings:
    """ Class to define model options """
    model_name: Optional[str] = None
    output_name: Optional[str] = None
    model_output_type: Optional[str] = None
    mod_value: Optional[int] = None
    fps: Optional[int] = None
    fp8_gemm_module_list: List[str] = None
    # FSDP strategy is just for the components to be sharded - other components will be moved to correct device automatically
    fsdp_strategy: dict = field(default_factory=lambda: {
        "": { # name, e.g. transformer
            "shard_submodule_key": None, # submodule to shard, e.g encoder -> transformer.encoder will be sharded
            "block_attr": None, # attribute name of blocks to shard, e.g. blocks
            "dtype": None, # Target dtype to convert the model to before sharding
            "children_to_device": [{ # Move other children to device
                "submodule_key": None, # e.g "encoder" -> children of transformer.encoder
                "exclude_keys": [] # exclude these children from being moved
            }]
        }
    })
    valid_tasks: List[str] = field(default_factory=list)
    resolution_divisor: Optional[int] = None

class DiffusionOutput:
    """ Class to encapsulate diffusion model outputs """
    def __init__(self, images: List[Image] = None, videos: List[np.ndarray]|np.ndarray = None, pipe_args: List[dict]|dict = []) -> None:
        self.images = images
        if not isinstance(videos, list):
            videos = [videos]
        self.videos = videos
        if not isinstance(pipe_args, list):
            pipe_args = [pipe_args]
        self.pipe_args = pipe_args

    @classmethod
    def from_outputs(cls, outputs: List["DiffusionOutput"], output_type: str) -> "DiffusionOutput":
        if output_type == "image":
            args_list = []
            all_images = []
            for out in outputs:
                all_images.extend(out.images)
                args_list.extend(out.pipe_args)
            return DiffusionOutput(images=all_images, pipe_args=args_list)
        elif output_type == "video":
            all_videos = []
            args_list = []
            for out in outputs:
                all_videos.extend(out.videos)
                args_list.extend(out.pipe_args)
            return DiffusionOutput(videos=all_videos, pipe_args=args_list)
        else:
            raise NotImplementedError(f"DiffusionOutput does not support output type: {output_type}")

    def get_outputs(self) -> Generator[Tuple[Image|np.ndarray, dict], None, None]:
        """ Returns a generator that yields output items along with their used input arguments """
        if self.images:
            for image, single_pipe_args in zip(self.images, self.pipe_args):
                yield (image, single_pipe_args)
        elif self.videos:
            for video, single_pipe_args in zip(self.videos, self.pipe_args):
                yield (video, single_pipe_args)

class xFuserModel(abc.ABC):
    """ Base class for xFuser models """

    capabilities: ModelCapabilities = ModelCapabilities()
    default_input_values: DefaultInputValues = DefaultInputValues()
    settings: ModelSettings = ModelSettings()
    model_output_type: str = ""
    fps: int = 0

    def __init__(self, config: xFuserArgs) -> None:
        self._validate_config(config)
        self.config = config
        self.pipe = None

    def initialize(self, input_args: dict) -> None:
        """ Load the model pipeline """

        if not torch.distributed.is_initialized():
            log("Initializing distributed environment...")
            init_distributed_environment()

        self.engine_config, _ = self.config.create_config()
        log("Loading model pipeline...")
        self.pipe = self._load_model()

        log("Initializing runtime state...")
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
            if isinstance(config_value, int):
                if not getattr(self.capabilities, key) and config_value > 1:
                    raise ValueError(f"Model {self.settings.model_name} does not support {key}.")
            if isinstance(config_value, bool):
                if config_value and not getattr(self.capabilities, key):
                    raise ValueError(f"Model {self.settings.model_name} does not support {key}.")

        possible_task = getattr(config, "task", None)
        if possible_task and self.settings.valid_tasks:
            if possible_task not in self.settings.valid_tasks:
                raise ValueError(f"Model {self.settings.model_name} does not support task '{possible_task}'. Supported tasks: {self.settings.valid_tasks}")
        if possible_task and not self.settings.valid_tasks:
            raise ValueError(f"Model {self.settings.model_name} does not support multiple tasks, but task '{possible_task}' was specified.")
        if not possible_task and self.settings.valid_tasks:
            raise ValueError(f"Model {self.settings.model_name} requires a task to be specified. Supported tasks: {self.settings.valid_tasks}")
        if config.dataset_path and not config.batch_size:
            raise ValueError(f"Dataset path specified without batch size. Please specify batch size for dataset inference.")

        if self.model_output_type == "video" and not self.fps:
            raise ValueError(f"Model {self.settings.model_name} produces video output but fps is not set.")

        if self.settings.resolution_divisor and (config.height % self.settings.resolution_divisor != 0 or config.width % self.settings.resolution_divisor != 0):
            raise ValueError(f"Model {self.settings.model_name} requires height and width to be divisible by {self.settings.resolution_divisor}.")


    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default") # TODO: Configurable

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2  # Reduce steps for warmup # TODO: make this more generic
        self._run_timed_pipe(compile_args)


    def run(self, input_args: dict) -> Tuple[DiffusionOutput, list]:
        """ Run the model with given input arguments and return output and timings """
        self._validate_args(input_args)
        timings = []
        output: DiffusionOutput = None

        self._run_warmup_calls(input_args)
        for iteration in range(self.config.num_iterations):
            log(f"Running iteration {iteration + 1}/{self.config.num_iterations}")

            if self.config.batch_size: # Run in batched mode
                output, batch_timings = self._run_pipe_batched(input_args)
                timings += batch_timings
            else: # Run all in one go
                output, timing = self._run_timed_pipe(input_args)
                timings.append(timing)
                log(f"Iteration {iteration + 1} completed in {timing:.2f}s")

        if len(timings) > 1:
            timings.pop(0) # Remove first timing for more accurate average # TODO: fix
        log(f"Average time over {self.config.num_iterations} runs: {sum(timings) / len(timings):.2f}s")
        return output, timings

    def _run_pipe_batched(self, input_args: dict) -> Tuple[List[DiffusionOutput], list]:
        """ Run the pipeline in batches """
        batch_size = self.config.batch_size
        all_prompts = input_args["prompt"]
        timings = []
        all_outputs = []
        batch_count = len(all_prompts) // batch_size + (1 if len(all_prompts) % batch_size != 0 else 0)

        for batch_index in range(0, batch_count):
            batch_args = copy.deepcopy(input_args)
            prompts = batch_args["prompt"][batch_index*batch_size:(batch_index+1)*batch_size]
            batch_args["prompt"] = prompts

            log(f"Processing batch {batch_index} with prompts {batch_index*batch_size} to {(batch_index+1)*batch_size}")
            output, timing = self._run_timed_pipe(batch_args)
            timings.append(timing)
            all_outputs.append(output)
            log(f"Batch {batch_index} completed in {timing:.2f}s")

        return DiffusionOutput.from_outputs(all_outputs, self.settings.model_output_type), timings

    def _run_warmup_calls(self, input_args: dict) -> None:
        """ Run initial warmup calls if specified """
        if self.config.warmup_calls:
            log(f"Warming up model with {self.config.warmup_calls} calls...")
            for iteration in range(self.config.warmup_calls):
                log(f"Warmup iteration {iteration + 1}/{self.config.warmup_calls}")
                self._run_timed_pipe(input_args)
            log(f"Warmup complete.")

    def profile(self, input_args: dict) -> Tuple[DiffusionOutput, list, torch.profiler.profiler.profile]:
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
            record_shapes=True,
            with_stack=False,
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

        # Dataset to prompts
        if input_args.get("dataset_path", None):
            args["prompt"] = load_dataset_prompts(input_args["dataset_path"])

        args = self._preprocess_args_images(args)
        return args

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess image inputs if necessary """
        self._validate_args(input_args)
        images = [load_image(path) for path in input_args.get("input_images", [])]
        input_args["input_images"] = images
        return input_args

    def save_output(self, output: DiffusionOutput) -> None:
        """ Saves the output based on its type """
        # Assumes output only has images or videos, not both
        if output.images:
            for image_index, (image, pipe_args) in enumerate(output.get_outputs()):
                output_name = self.get_output_name(pipe_args)
                output_path = f"{self.config.output_directory}/{output_name}_{image_index}.png"
                image.save(output_path)
                log(f"Output image saved to {output_path}")
        elif output.videos:
            for video_index, (video, pipe_args) in enumerate(output.get_outputs()):
                if isinstance(video, np.ndarray):
                    video = video[0] # Remove batch dimension
                output_name = self.get_output_name(pipe_args)
                output_path = f"{self.config.output_directory}/{output_name}_{video_index}.mp4"
                export_to_video(video, output_path, fps=self.settings.fps)
                log(f"Output video saved to {output_path}")
        else:
            raise NotImplementedError(f"No output to save.")

    def save_timings(self, timings: list) -> None:
        timing_file_name = f"{self.config.output_directory}/timings.json"
        with open(timing_file_name, "w") as timing_file:
            json.dump(timings, timing_file, indent=2)
        log(f"Timings saved to {self.config.output_directory}/timings.json")

    def save_profile(self, profile: torch.profiler.profiler.profile) -> None:
        profile_file = f"{self.config.output_directory}/profile_trace_rank_{get_world_group().rank}.json"
        profile.export_chrome_trace(profile_file)
        log(f"Profile trace saved to {profile_file}")

    def _run_timed_pipe(self, input_args: dict) -> Tuple[DiffusionOutput, float]:
        """ Run a a full pipeline with timing information """

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start.record()
        out = self._run_pipe(input_args)
        end.record()

        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000  # Convert to seconds
        return out, elapsed_time

    def get_output_name(self, input_args) -> str:
        """ Generate a unique output name based on model and config """
        use_compile = self.config.use_torch_compile
        ulysses_degree = self.config.ulysses_degree or 1
        ring_degree = self.config.ring_degree or 1
        height = input_args["height"]
        width = input_args["width"]
        name = f"{self.settings.output_name}_u{ulysses_degree}r{ring_degree}_tc_{use_compile}_{height}x{width}"
        if self.config.task:
            name += f"_{self.config.task}"
        return name

    def _post_load_and_state_initialization(self, input_args: dict) -> None: ##TODO: should this be renamed?
        """ Hook for any post model-load and state initialization """

        local_rank = get_world_group().local_rank
        if self.config.use_fsdp:
            self._shard_model_with_fsdp()
        else:
            self.pipe = self.pipe.to(f"cuda:{local_rank}")

        if self.config.use_fp8_gemms:
            for module_name in self.settings.fp8_gemm_module_list:
                log(f"Quantizing linear layers in {module_name} to FP8...")
                module = rgetattr(self.pipe, module_name)
                quantize_linear_layers_to_fp8(module, device=f"cuda:{local_rank}")

        if self.config.use_hybrid_fp8_attn:
            self._setup_hybrid_fp8_attn(input_args)

    def _shard_model_with_fsdp(self) -> None:
        """ Shard the model with FSDP based on settings """
        local_rank = get_world_group().local_rank
        sp_local_rank = get_sp_group().local_rank
        sp_device_group = get_sp_group().device_group
        sp_device = f"cuda:{sp_local_rank}"
        for component_name, component in self.pipe.components.items():
            if component_name in self.settings.fsdp_strategy:
                strategy = self.settings.fsdp_strategy[component_name]
                log(f"Wrapping {component_name} with FSDP...")
                # Moving non FSPD'd children to device
                for child in strategy.get("children_to_device", []): # Iterate over list of children to move to device
                    submodule_key = child.get("submodule_key", None)
                    exclude_keys = child.get("exclude_keys", [])
                    if submodule_key:
                        log(f"Moving children of {component_name}.{submodule_key} to device, excluding {exclude_keys}...")
                        children_to_device(getattr(component, submodule_key), sp_device, exclude_keys)
                    else:
                        log(f"Moving children of {component_name} to device, excluding {exclude_keys}...")
                        children_to_device(component, sp_device, exclude_keys)

                # FSDP
                submodule_key = strategy.get("shard_submodule_key", None)
                block_attr = strategy.get("block_attr", None)
                dtype = strategy.get("dtype", None)
                shard_obj = component if not submodule_key else getattr(component, submodule_key)
                log(f"Sharding {component_name} submodule {submodule_key} with block attribute {block_attr} to dtype {dtype}...")
                fsdp_object = shard_transformer_blocks(
                    shard_obj,
                    block_attr=block_attr,
                    device_id=sp_local_rank,
                    dtype=dtype,
                    process_group=sp_device_group,
                    use_orig_params=True,
                    sync_module_states=True,
                    forward_prefetch=True,
                )
                setattr(self.pipe, component_name, fsdp_object)
            else:
                log(f"Skipping FSDP wrapping for {component_name}...")
                if hasattr(component, "to"):
                    component.to(f"cuda:{local_rank}")
                else:
                    log(f"Component {component_name} has no .to() method, skipping device move.")
                    pass

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        return 1

    def _setup_hybrid_fp8_attn(self, input_args: dict) -> None:
        """
        Setup hybrid FP8 attention, where initial and final attention steps use bf16 for stability,
        and middle steps use FP8 for performance. To keep track of which steps to use which attention,
        a boolean decision vector is created and stored in the runtime state. We keep track of the current
        step during inference in the transformer forward pass, and when CFG is used, the transformer is called
        twice per denoising step, so we need to account for that in the decision vector.
        """
        number_of_initial_and_final_bf16_attn_steps = input_args["num_hybrid_bf16_attn_steps"] # Number of initial and final steps to use bf16 attention for stability
        multiplier = self._calculate_hybrid_attention_step_multiplier(input_args) # If CFG is switched on, double the transformers are called
        fp8_steps_threshold = number_of_initial_and_final_bf16_attn_steps * multiplier
        total_steps = input_args["num_inference_steps"] * multiplier # Total number of transformer calls during the denoising process
        # Create a boolean vector indicating which steps should use fp8 attention
        fp8_decision_vector = torch.tensor(
        [i >= fp8_steps_threshold and i < (total_steps - fp8_steps_threshold)
            for i in range(total_steps)], dtype=torch.bool
        )
        get_runtime_state().set_hybrid_attn_parameters(fp8_decision_vector)

    @abc.abstractmethod
    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        """ Execute the pipeline. Must be implemented by subclasses. """
        pass

    @abc.abstractmethod
    def _load_model(self) -> DiffusionPipeline:
        """ Load the model. Must be implemented by subclasses. """
        pass

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments. Can be overridden by subclasses. """
        if input_args["prompt"] is None and input_args["dataset_path"] is None:
            raise ValueError("Either 'prompt' or 'dataset_path' must be provided in input arguments.")
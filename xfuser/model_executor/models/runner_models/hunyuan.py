import torch
import copy
from diffusers import HunyuanVideoPipeline, HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser import xFuserArgs
from xfuser.model_executor.models.transformers.transformer_hunyuan_video import xFuserHunyuanVideoTransformer3DWrapper
from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import xFuserHunyuanVideo15Transformer3DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
)
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.core.utils.runner_utils import (
    resize_and_crop_image,
)

@register_model("tencent/HunyuanVideo")
@register_model("HunyuanVideo")
class xFuserHunyuanvideoModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
        use_hybrid_attn_schedule=True
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=50,
        guidance_scale=6.0,
        num_hybrid_attn_high_precision_steps = 5,
    )
    settings = ModelSettings(
        model_name="tencent/HunyuanVideo",
        output_name="hunyuan_video",
        model_output_type="video",
        fps=24,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserHunyuanVideoTransformer3DWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            revision="refs/pr/18",
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            revision="refs/pr/18",
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            prompt=input_args["prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer.compile()

        compile_args = copy.deepcopy(input_args)
        # If a per-step attention schedule is active, do a full warmup to trigger all backend paths.
        if not get_runtime_state().has_attention_schedule():
            compile_args["num_inference_steps"] = 2 # Reduce steps for warmup
        self._run_timed_pipe(compile_args)


@register_model("tencent/HunyuanVideo-1.5")
@register_model("Hunyuanvideo-1.5")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v")
@register_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v")
class xFuserHunyuanvideo15Model(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_frames=121,
        num_inference_steps=50,
    )
    settings = ModelSettings(
        output_name="hunyuan_video_1_5",
        model_output_type="video",
        fps=24,
        mod_value=16,
        valid_tasks=["i2v", "t2v"],
    )


    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        if self.config.task == "i2v": # TODO: different model for 480p
            self.settings.model_name = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v"
        else:
            self.settings.model_name = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"


    def _load_model(self) -> DiffusionPipeline:
        task = self.config.task
        pipeline = HunyuanVideo15Pipeline if task == "t2v" else HunyuanVideo15ImageToVideoPipeline
        transformer = xFuserHunyuanVideo15Transformer3DWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "num_inference_steps": input_args["num_inference_steps"],
            "num_frames": input_args["num_frames"],
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            "prompt": input_args["prompt"],
        }
        if self.config.task == "i2v":
            kwargs["image"] = input_args["image"]
        else: #t2v task
            kwargs["height"] = input_args["height"]
            kwargs["width"] = input_args["width"]

        output = self.pipe(**kwargs)
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess input images if necessary based on task and other args """
        input_args = super()._preprocess_args_images(input_args)
        if self.config.task == "i2v":
            image = input_args["input_images"][0]
            if input_args.get("resize_input_images", False):
                image = resize_and_crop_image(image, input_args["width"], input_args["height"], self.settings.mod_value)
            input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        if self.config.task == "i2v":
            images = input_args.get("input_images", [])
            if len(images) != 1:
                raise ValueError("Exactly one input image is required for HunyuanVideo-1.5 model when task is 'i2v'.")

    def _compile_model(self, input_args: dict) -> None:
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)
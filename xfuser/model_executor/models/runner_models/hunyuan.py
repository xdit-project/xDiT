import torch
import copy
from diffusers import HunyuanVideoPipeline, HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
from xfuser.model_executor.models.transformers.transformer_hunyuan_video import xFuserHunyuanVideoTransformer3DWrapper
from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import xFuserHunyuanVideo15Transformer3DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    ModelCapabilities,
)
from xfuser.core.distributed import (
    get_world_group,
)

@register_model("tencent/HunyuanVideo")
@register_model("Hunyuanvideo")
class xFuserHunyuanvideoModel(xFuserModel):

    fps = 24
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
    )

    model_name: str = "tencent/HunyuanVideo"
    output_name: str = "hunyuan_video"
    model_output_type: str = "video"

    def _load_model(self):
        transformer = xFuserHunyuanVideoTransformer3DWrapper.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            revision="refs/pr/18",
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            revision="refs/pr/18",
        )
        return pipe

    def _run_pipe(self, input_args: dict):
        output = self.pipe(
            prompt=input_args["prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return output

    def _compile_model(self, input_args):
        """ Compile the model using torch.compile """
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer.compile()

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2  # Reduce steps for warmup # TODO: make this more generic
        self._run_timed_pipe(compile_args)


@register_model("tencent/HunyuanVideo-1.5")
@register_model("Hunyuanvideo-1.5")
class xFuserHunyuanvideo15Model(xFuserModel):

    fps = 24
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_slicing=True,
        enable_tiling=True,
    )

    valid_tasks = ["i2v", "t2v"]
    model_name: str = "tencent/HunyuanVideo-1.5"
    output_name: str = "hunyuan_video_1_5"
    model_output_type: str = "video"

    def _load_model(self):
        task = self.config.get("task")
        pipeline = HunyuanVideo15Pipeline if task == "t2v" else HunyuanVideo15ImageToVideoPipeline
        transformer = xFuserHunyuanVideo15Transformer3DWrapper.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = pipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        return pipe

    def _run_pipe(self, input_args: dict):
        kwargs = {
            "num_inference_steps": input_args["num_inference_steps"],
            "guidance_scale": input_args["guidance_scale"],
            "num_frames": input_args["num_frames"],
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            "prompt": input_args["prompt"],
        }
        if self.config.task == "i2v":
            kwargs["image"] = input_args["image"]
        else: #t2v task
            kwargs["height"] = input_args["height"]
            kwargs["width"] = input_args["width"]

        return self.pipe(**kwargs)

    def _preprocess_args_images(self, input_args):
        """ Preprocess input images if necessary based on task and other args """
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        if self.config.task == "i2v" and input_args.get("resize_input_images", False):
            image = self._resize_and_crop_image(image, input_args["width"], input_args["height"], self.mod_value)
        input_args["image"] = image
        return input_args

    def validate_args(self, input_args: dict):
        """ Validate input arguments """
        if self.config.task == "i2v":
            images = input_args.get("input_images", [])
            if len(images) != 1:
                raise ValueError("Exactly one input image is required for HunyuanVideo-1.5 model when task is 'i2v'.")
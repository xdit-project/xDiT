import copy
import torch
from typing import List
from PIL import Image
from diffusers import WanPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.utils import load_image
from xfuser import xFuserArgs
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper
from xfuser.model_executor.models.transformers.transformer_wan_vace import xFuserWanVACETransformer3DWrapper
from xfuser.model_executor.pipelines.pipeline_wan_i2v import (
    xFuserWanImageToVideoPipeline,
)
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
)
from xfuser.core.utils.runner_utils import (
    resize_and_crop_image,
    resize_image_to_max_area,
)

COMMON_FSDP_STRATEGY = {
    "transformer": {
        "wrap_attrs": ["blocks"],
        "dtype": torch.bfloat16,
    },
    "text_encoder": {
        "wrap_attrs": ["encoder.block"],
    }
}


@register_model("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
@register_model("Wan2.1-I2V")
class xFuserWan21I2VModel(xFuserModel):

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        do_cfg = input_args["guidance_scale"] > 1.0
        if do_cfg:
            return 2
        return 1

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_cfg_parallel=True,
        use_hybrid_fp8_attn=True,
        use_fp4_gemms=True
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=40,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=3.5,
        num_hybrid_bf16_attn_steps = 5,
    )
    settings = ModelSettings(
        model_name = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        output_name = "wan2.1_i2v",
        model_output_type = "video",
        mod_value = 16, # vae_scale_factor_spatial * patch_size[1] = 8
        fps = 16,
        fp8_gemm_module_list=["transformer.blocks"],
        fp4_gemm_module_list=["transformer.blocks"],
        fp8_precision_overrides=("0.", "1.", "2.", "3.", "4.",
                                 "5.", "6.", "7.", "8.", "9.",
                                 "30.", "31.", "32.", "33.", "34.",
                                 "35.", "36.", "37.", "38.", "39."),
        fsdp_strategy=COMMON_FSDP_STRATEGY,
        flow_shift=5,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = xFuserWanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
        )
        pipe.scheduler.config.flow_shift = self.settings.flow_shift
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            image=input_args["image"],
            height=input_args["height"],
            width=input_args["width"],
            prompt=str(input_args["prompt"]),
            negative_prompt=input_args["negative_prompt"],
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        image = input_args["input_images"][0]
        width, height = input_args["width"], input_args["height"]
        if input_args.get("resize_input_images", False):
            image = resize_and_crop_image(image, width, height, self.settings.mod_value)
        else:
            image = resize_image_to_max_area(image, height, width, self.settings.mod_value)
        input_args["height"] = image.height
        input_args["width"] = image.width
        input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Wan I2V model.")

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        compile_args = copy.deepcopy(input_args)
        # If hybrid attention is being used, we need to do a full cycle to warmup the compiler
        # to trigger both bf16 and fp8 attention paths. Reduce steps for warmup if not using hybrid attention.
        if not self.config.use_hybrid_fp8_attn:
            compile_args["num_inference_steps"] = 2 # Reduce steps for warmup
        self._run_timed_pipe(compile_args)


@register_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
@register_model("Wan2.2-I2V")
class xFuserWan22I2VModel(xFuserWan21I2VModel):

    def __init__(self, config: xFuserArgs) -> None:
        self.settings.model_name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        self.settings.output_name = "wan2.2_i2v"
        super().__init__(config)
        self.settings.fsdp_strategy["transformer_2"] = {
                "wrap_attrs": ["blocks"],
                "dtype": torch.bfloat16,
        }
        self.settings.fp8_gemm_module_list=["transformer.blocks", "transformer_2.blocks"]
        self.settings.fp8_precision_overrides=None


    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer_2",
        )
        pipe = xFuserWanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
                transformer_2=transformer_2,
        )
        pipe.scheduler.config.flow_shift = self.settings.flow_shift
        return pipe

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self.pipe.transformer_2 = torch.compile(self.pipe.transformer_2, mode="default")
        # Full cycle to warmup the torch compiler
        self._run_timed_pipe(input_args)


@register_model("Wan-AI/Wan2.1-T2V-14B-Diffusers")
@register_model("Wan2.1-T2V")
class xFuserWan21T2VModel(xFuserModel):

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        do_cfg = input_args["guidance_scale"] > 1.0
        if do_cfg:
            return 2
        return 1

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_hybrid_fp8_attn=True,
    )
    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=40,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=3.5,
        num_hybrid_bf16_attn_steps = 5,
    )
    settings = ModelSettings(
        mod_value=8,
        fps=16,
        model_output_type="video",
        model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        output_name="wan2.1_t2v",
        fp8_gemm_module_list=["transformer.blocks"],
        fp4_gemm_module_list=["transformer.blocks"],
        fp8_precision_overrides=("0.", "1.", "2.", "3.", "4.",
                                 "5.", "6.", "7.", "8.", "9.",
                                 "30.", "31.", "32.", "33.", "34.",
                                 "35.", "36.", "37.", "38.", "39."),
        fsdp_strategy=COMMON_FSDP_STRATEGY,
        flow_shift=12,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        pipe.scheduler.config.flow_shit = self.settings.flow_shift
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=str(input_args["prompt"]),
            negative_prompt=str(input_args["negative_prompt"]),
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        compile_args = copy.deepcopy(input_args)
        # If hybrid attention is being used, we need to do a full cycle to warmup the compiler
        # to trigger both bf16 and fp8 attention paths. Reduce steps for warmup if not using hybrid attention.
        if not self.config.use_hybrid_fp8_attn:
            compile_args["num_inference_steps"] = 2 # Reduce steps for warmup # TODO: make this more generic
        self._run_timed_pipe(compile_args)


@register_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
@register_model("Wan2.2-T2V")
class xFuserWan22T2VModel(xFuserWan21T2VModel):

    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        self.settings.model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.settings.output_name = "wan2.2_t2v"
        self.settings.fsdp_strategy["transformer_2"] = {
                "wrap_attrs": ["blocks"],
                "dtype": torch.bfloat16,
        }
        self.settings.fp8_gemm_module_list=["transformer.blocks", "transformer_2.blocks"]
        self.settings.fp8_precision_overrides=None

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        transformer_2 = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer_2",
        )
        pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            transformer_2=transformer_2,
        )
        pipe.scheduler.config.flow_shit = self.settings.flow_shift
        return pipe

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self.pipe.transformer_2 = torch.compile(self.pipe.transformer_2, mode="default")
        # Full cycle to warmup the torch compiler
        self._run_timed_pipe(input_args)


@register_model("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
@register_model("Wan2.2-TI2V")
class xFuserWan22TI2VModel(xFuserWan21T2VModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
    )
    default_input_values = DefaultInputValues(
        height=736,
        width=1280,
        num_inference_steps=50,
        num_frames=121,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=5.0,
        num_hybrid_bf16_attn_steps=5,
    )
    settings = ModelSettings(
        mod_value=32,
        fps=24,
        model_output_type="video",
        model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        output_name="wan2.2_ti2v",
        fp8_gemm_module_list=["transformer.blocks"],
        fsdp_strategy=COMMON_FSDP_STRATEGY,
        valid_tasks=["i2v", "t2v"],
        flow_shift=5,
    )

    def _load_model(self) -> DiffusionPipeline:
        torch.set_float32_matmul_precision('high')
        transformer = xFuserWanTransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe_class = xFuserWanImageToVideoPipeline if self.config.task == "i2v" else WanPipeline
        pipe = pipe_class.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                torch_dtype=torch.bfloat16,
                transformer=transformer,
        )
        pipe.scheduler.config.flow_shit = self.settings.flow_shift
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "height": input_args["height"],
            "width": input_args["width"],
            "prompt": str(input_args["prompt"]),
            "negative_prompt": str(input_args["negative_prompt"]),
            "num_inference_steps": input_args["num_inference_steps"],
            "num_frames": input_args["num_frames"],
            "guidance_scale": input_args["guidance_scale"],
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        }
        if self.config.task == "i2v":
            kwargs["image"] = input_args["image"]
        output = self.pipe(**kwargs)
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self._run_timed_pipe(input_args)

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        if self.config.task != "i2v":
            return input_args

        image = input_args["input_images"][0]
        width, height = input_args["width"], input_args["height"]
        if input_args.get("resize_input_images", False):
            image = resize_and_crop_image(image, width, height, self.settings.mod_value)
        else:
            image = resize_image_to_max_area(image, height, width, self.settings.mod_value)
        input_args["height"] = image.height
        input_args["width"] = image.width
        input_args["image"] = image
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if self.config.task == "i2v":
            if len(images) != 1:
                raise ValueError("Exactly one input image is required for Wan TI2V model when using i2v task.")
        else:
            if len(images) != 0:
                raise ValueError("No input images should be provided for Wan TI2V model when using t2v task.")



@register_model("Wan-AI/Wan2.1-VACE-14B-diffusers")
@register_model("Wan-AI/Wan2.1-VACE-1.3B-diffusers")
@register_model("Wan2.1-VACE-14B")
@register_model("Wan2.1-VACE-1.3B")
class xFuserWan21VACEModel(xFuserModel):

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_fp8_gemms=True,
    )

    default_input_values = DefaultInputValues(
        height=720,
        width=1280,
        num_inference_steps=30,
        num_frames=81,
        negative_prompt="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        guidance_scale=5.0,
    )

    settings = ModelSettings(
        fps=16,
        model_output_type="video",
        fp8_gemm_module_list=["transformer.blocks", "transformer.vace_blocks"],
    )

    def __init__(self, config: xFuserArgs) -> None:
        super().__init__(config)
        if "14B" in config.model:
            self.settings.model_name = "Wan-AI/Wan2.1-VACE-14B-diffusers"
            self.settings.output_name = "wan.2.1_vace_14b"
        else:
            self.settings.model_name = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
            self.settings.output_name = "wan.2.1_vace_1.3b"

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserWanVACETransformer3DWrapper.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = WanVACEPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        pipe.scheduler.flow_shift = 5.0 # 5.0 for 720p, 3.0 for 480p
        return pipe

    def _prepare_video_and_mask(self, first_img: Image, last_img: Image, height: int, width: int, num_frames: int) -> tuple[List[Image.Image], List[Image.Image]]:
        """ Prepare video and mask for Wan VACE model """
        first_img = first_img.resize((width, height))
        last_img = last_img.resize((width, height))
        frames = []
        frames.append(first_img)
        # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
        # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
        # match the original code.
        frames.extend([Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
        frames.append(last_img)
        mask_black = Image.new("L", (width, height), 0)
        mask_white = Image.new("L", (width, height), 255)
        mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
        return frames, mask

    def _preprocess_args_images(self, input_args: dict) -> dict:
        """ Preprocess image inputs if necessary """
        self._validate_args(input_args)
        images = [load_image(path) for path in input_args.get("input_images", [])]
        video, mask = self._prepare_video_and_mask(images[0], images[1], input_args["height"], input_args["width"], input_args["num_frames"])
        input_args["video"] = video
        input_args["mask"] = mask
        return input_args

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(
            height=input_args["height"],
            width=input_args["width"],
            prompt=str(input_args["prompt"]),
            negative_prompt=str(input_args["negative_prompt"]),
            num_inference_steps=input_args["num_inference_steps"],
            num_frames=input_args["num_frames"],
            guidance_scale=input_args["guidance_scale"],
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
            video=input_args["video"],
            mask=input_args["mask"],
        )
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if len(images) != 2:
            raise ValueError("Exactly two input images are required for Wan VACE model (first frame and last frame).")
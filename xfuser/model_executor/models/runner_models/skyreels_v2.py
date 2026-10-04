import torch
from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser import xFuserArgs
from xfuser.core.utils.runner_utils import (
    resize_and_crop_image,
    resize_image_to_max_area,
)
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)

# The Wan 2.1 negative prompt, which SkyReels-V2 inherits.
NEGATIVE_PROMPT = "bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards"

T2V_14B_MODEL_ID = "Skywork/SkyReels-V2-T2V-14B-540P-Diffusers"
I2V_14B_MODEL_ID = "Skywork/SkyReels-V2-I2V-14B-540P-Diffusers"
I2V_1_3B_MODEL_ID = "Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers"

# Attention heads per checkpoint; Ulysses splits heads, so its degree must divide them.
ATTENTION_HEADS = {
    T2V_14B_MODEL_ID: 40,
    I2V_14B_MODEL_ID: 40,
    I2V_1_3B_MODEL_ID: 12,
}


class xFuserSkyReelsV2Model(xFuserModel):
    """Non-causal SkyReels-V2 (T2V and I2V) on the plain diffusers pipelines.

    Diffusion forcing is not offered: it needs per-frame timesteps and a
    block-causal attention mask that sequence-parallel attention cannot apply.
    """

    # SkyReelsV2Attention, with the attention_mask argument the wrapper relies on,
    # first shipped in diffusers 0.36.0.
    min_diffusers_version = "0.36.0"
    attention_head_dims = frozenset({128})

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    settings = ModelSettings(
        model_output_type="video",
        mod_value=16,  # vae_scale_factor_spatial (8) * patch_size (2)
        resolution_divisor=16,
        fps=24,
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["blocks"],
                "dtype": torch.bfloat16,
            },
            "text_encoder": {
                "wrap_attrs": ["encoder.block"],
                "offload_policy": "cpu",
            },
        },
    )

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        self.settings.model_name = self._resolve_model_name(config.model)
        self.settings.output_name = self.settings.model_name.split("/")[-1].lower()
        self.attention_heads = ATTENTION_HEADS[self.settings.model_name]

    def _resolve_model_name(self, requested: str) -> str:
        raise NotImplementedError

    def _validate_config(self, config: xFuserArgs) -> None:
        super()._validate_config(config)
        ulysses_degree = config.ulysses_degree or 1
        if self.attention_heads % ulysses_degree != 0:
            raise ValueError(
                f"{self.settings.model_name} has {self.attention_heads} attention heads, so "
                f"--ulysses_degree must divide {self.attention_heads}; got {ulysses_degree}."
            )

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        # The checkpoints ship flow_shift=1.0; SkyReels-V2 samples with a larger shift.
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=input_args["flow_shift"]
        )

    def _load_transformer(self):
        from xfuser.model_executor.models.transformers.transformer_skyreels_v2 import (
            xFuserSkyReelsV2Transformer3DWrapper,
        )

        return self.loader.load_transformer(xFuserSkyReelsV2Transformer3DWrapper)

    def _pipe_kwargs(self, input_args: dict) -> dict:
        return {
            "height": input_args["height"],
            "width": input_args["width"],
            "prompt": input_args["prompt"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "num_frames": input_args["num_frames"],
            "guidance_scale": input_args["guidance_scale"],
            "generator": self._make_generator(input_args["seed"]),
        }


@register_model(T2V_14B_MODEL_ID)
@register_model("SkyReels-V2-T2V-14B")
class xFuserSkyReelsV2T2VModel(xFuserSkyReelsV2Model):
    default_input_values = DefaultInputValues(
        height=544,
        width=960,
        num_frames=97,
        num_inference_steps=50,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=6.0,
        flow_shift=8.0,
    )

    def _resolve_model_name(self, requested: str) -> str:
        return T2V_14B_MODEL_ID

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import SkyReelsV2Pipeline

        transformer = self._load_transformer()
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        return SkyReelsV2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            quantization_config=te_quant,
            **te_kwargs,
        )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        if input_args.get("input_images"):
            raise ValueError(
                f"{self.settings.model_name} is text-to-video and takes no input images; "
                "use a SkyReels-V2 I2V model for image-to-video."
            )

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(**self._pipe_kwargs(input_args))
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)


@register_model(I2V_14B_MODEL_ID)
@register_model(I2V_1_3B_MODEL_ID)
@register_model("SkyReels-V2-I2V-14B")
@register_model("SkyReels-V2-I2V-1.3B")
class xFuserSkyReelsV2I2VModel(xFuserSkyReelsV2Model):
    default_input_values = DefaultInputValues(
        height=544,
        width=960,
        num_frames=97,
        num_inference_steps=50,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=5.0,
        flow_shift=5.0,
    )

    def _resolve_model_name(self, requested: str) -> str:
        return I2V_1_3B_MODEL_ID if "1.3B" in requested else I2V_14B_MODEL_ID

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import SkyReelsV2ImageToVideoPipeline

        transformer = self._load_transformer()
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        return SkyReelsV2ImageToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
            quantization_config=te_quant,
            **te_kwargs,
        )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        if len(input_args.get("input_images", [])) != 1:
            raise ValueError(f"{self.settings.model_name} requires exactly one input image.")

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

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        output = self.pipe(image=input_args["image"], **self._pipe_kwargs(input_args))
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

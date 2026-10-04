import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)

# Negative prompt used by the LTX-Video 0.9.7 model card examples.
LTX_VIDEO_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

# Values from the 0.9.7-dev checkpoint's transformer/config.json and vae/config.json.
LTX_VIDEO_097_NUM_ATTENTION_HEADS = 32
LTX_VIDEO_097_ATTENTION_HEAD_DIM = 128
LTX_VIDEO_SPATIAL_COMPRESSION = 32
LTX_VIDEO_TEMPORAL_COMPRESSION = 8

# The VAE of the 0.9.5+ checkpoints conditions decoding on a timestep; these are
# the decode settings the 0.9.7 model card uses.
LTX_VIDEO_DECODE_TIMESTEP = 0.05
LTX_VIDEO_DECODE_NOISE_SCALE = 0.025


def ltx_video_token_count(height: int, width: int, num_frames: int) -> int:
    """Video tokens the transformer sees (patch size 1 in time and space)."""
    latent_frames = (num_frames - 1) // LTX_VIDEO_TEMPORAL_COMPRESSION + 1
    return latent_frames * (height // LTX_VIDEO_SPATIAL_COMPRESSION) * (width // LTX_VIDEO_SPATIAL_COMPRESSION)


@register_model("Lightricks/LTX-Video-0.9.7-dev")
@register_model("LTX-Video-0.9.7-dev")
class xFuserLTXVideoModel(xFuserModel):
    """LTX-Video 0.9.7-dev (13B) single-stage text-to-video and image-to-video.

    t2v runs diffusers' LTXPipeline and i2v LTXImageToVideoPipeline, with the
    transformer swapped for the xDiT wrapper. The distilled checkpoints and the
    two-stage latent-upsampler flow are not covered here.
    """

    # 0.9.7's checkpoint was converted with diffusers 0.34.0.dev0.
    min_diffusers_version = "0.34.0"

    attention_head_dims = frozenset({LTX_VIDEO_097_ATTENTION_HEAD_DIM})

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_cfg_parallel=True,
        fully_shard_degree=True,
        enable_slicing=True,
        enable_tiling=True,
    )
    default_input_values = DefaultInputValues(
        height=512,
        width=704,
        num_frames=121,
        num_inference_steps=30,
        guidance_scale=3.0,
        negative_prompt=LTX_VIDEO_NEGATIVE_PROMPT,
    )
    settings = ModelSettings(
        model_name="Lightricks/LTX-Video-0.9.7-dev",
        output_name="ltx_video_0_9_7_dev",
        model_output_type="video",
        # Also passed to the pipeline as frame_rate, which scales the temporal RoPE;
        # 25 is the LTX pipelines' default.
        fps=25,
        resolution_divisor=LTX_VIDEO_SPATIAL_COMPRESSION,
        valid_tasks=["t2v", "i2v"],
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
                "dtype": torch.bfloat16,
            },
        },
    )

    def _validate_config(self, config: xFuserArgs) -> None:
        super()._validate_config(config)
        ulysses_degree = config.ulysses_degree or 1
        if LTX_VIDEO_097_NUM_ATTENTION_HEADS % ulysses_degree != 0:
            raise ValueError(
                f"{self.settings.model_name} has {LTX_VIDEO_097_NUM_ATTENTION_HEADS} attention "
                f"heads, which ulysses_degree {ulysses_degree} does not divide."
            )

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        num_frames = input_args["num_frames"]
        if (num_frames - 1) % LTX_VIDEO_TEMPORAL_COMPRESSION != 0:
            raise ValueError(
                f"{self.settings.model_name} needs num_frames of the form "
                f"{LTX_VIDEO_TEMPORAL_COMPRESSION}k+1 (e.g. 121), got {num_frames}."
            )

        images = input_args.get("input_images") or []
        if self.config.task == "i2v" and len(images) != 1:
            raise ValueError(f"{self.settings.model_name} i2v needs exactly one input image, got {len(images)}.")
        if self.config.task == "t2v" and images:
            raise ValueError(f"{self.settings.model_name} t2v does not take input images.")

        if self.config.use_cfg_parallel and input_args["guidance_scale"] <= 1.0:
            raise ValueError(
                "use_cfg_parallel needs guidance_scale > 1; without guidance there is "
                "no negative branch to run on the second rank."
            )

        ring_degree = self.config.ring_degree or 1
        sp_degree = (self.config.ulysses_degree or 1) * ring_degree
        tokens = ltx_video_token_count(input_args["height"], input_args["width"], num_frames)
        if ring_degree > 1 and tokens % sp_degree != 0:
            raise ValueError(
                f"With ring_degree > 1 the {tokens} video tokens of a {input_args['height']}x"
                f"{input_args['width']}x{num_frames} video must be divisible by the "
                f"sequence-parallel degree {sp_degree}. Pick another resolution or frame "
                "count, or use Ulysses only, which pads the sequence."
            )

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import LTXImageToVideoPipeline, LTXPipeline

        from xfuser.model_executor.models.transformers.transformer_ltx_video import (
            xFuserLTXVideoTransformer3DWrapper,
        )

        transformer = self.loader.load_transformer(xFuserLTXVideoTransformer3DWrapper)
        pipe_cls = LTXImageToVideoPipeline if self.config.task == "i2v" else LTXPipeline
        return pipe_cls.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        if self.config.task == "i2v":
            # The pipeline resizes the image to height x width itself.
            input_args["image"] = input_args["input_images"][0]
        return input_args

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = dict(
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            frame_rate=self.settings.fps,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            decode_timestep=LTX_VIDEO_DECODE_TIMESTEP,
            decode_noise_scale=LTX_VIDEO_DECODE_NOISE_SCALE,
            generator=self._make_generator(input_args["seed"]),
            output_type="np",
        )
        if self.config.task == "i2v":
            kwargs["image"] = input_args["image"]
        output = self.pipe(**kwargs)
        return DiffusionOutput(videos=output.frames, pipe_args=input_args)

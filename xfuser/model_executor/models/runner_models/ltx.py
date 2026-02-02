import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from xfuser.model_executor.models.transformers.transformer_ltx2 import xFuserLTX2VideoTransformer3DWrapper
from xfuser.model_executor.models.runner_models.base_model import (
    xFuserModel,
    register_model,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
    ModelCapabilities,
)

from xfuser.core.utils.runner_utils import (
    log,
)

@register_model("Lightricks/LTX-2")
@register_model("LTX-2")
class xFuserLTX2VideoModel(xFuserModel):

    default_input_values = DefaultInputValues(
        height=1024,
        width=1536,
        num_frames=121,
        num_inference_steps=40,
        guidance_scale=3.0,
        negative_prompt="shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

    )
    settings = ModelSettings(
        model_name="Lightricks/LTX-2",
        output_name="ltx_2_video",
        model_output_type="video",
        fps=24,
        resolution_divisor=64,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
    )

    def _load_model(self) -> DiffusionPipeline:
        transformer = xFuserLTX2VideoTransformer3DWrapper.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = LTX2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.settings.model_name,
            subfolder="latent_upsampler",
            torch_dtype=torch.bfloat16,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        self.upsample_pipe = upsample_pipe

        self.first_scheduler = self.pipe.scheduler
        self.second_scheduler = FlowMatchEulerDiscreteScheduler.from_config( # Scheduler for the 2nd stage
            pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
        )

        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        video_latent, audio_latent = self.pipe(
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            height=input_args["height"] // 2,
            width=input_args["width"] // 2,
            num_frames=input_args["num_frames"],
            frame_rate=self.settings.fps,
            sigmas=None,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            output_type="latent",
            return_dict=False,
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )

        video_latent = self.upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]

        self.pipe.scheduler = self.second_scheduler
        output = self.pipe(
            latents=video_latent,
            audio_latents=audio_latent,
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            num_inference_steps=3,
            guidance_scale=1.0,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            output_type="np",
            generator=torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        )
        self.pipe.scheduler = self.first_scheduler

        return DiffusionOutput(videos=output, pipe_args=input_args)


    def save_output(self, output: DiffusionOutput) -> None:
        pipe_args = output.pipe_args
        output = output.videos
        for i, video_object in enumerate(output):
            video, audio = video_object.frames, video_object.audio
            video = (video * 255).round().astype("uint8")
            video = torch.from_numpy(video)
            output_name = self.get_output_name(pipe_args[i])
            output_path = f"{self.config.output_directory}/{output_name}_{i}.mp4"
            encode_video(video[0], audio=audio[0].float().cpu(), audio_sample_rate=24000, fps=self.settings.fps, output_path=output_path)
            log(f"Output video saved to {output_path}")

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        self.upsample_pipe.to(self.pipe.device)
import torch
import copy
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

DEFAULT_NEGATIVE_PROMPT = "" \
"blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, " \
"grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, " \
"deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, " \
"wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of " \
"field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent " \
"lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny " \
"valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, " \
"mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, " \
"off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward " \
"pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, " \
"inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."



@register_model("Lightricks/LTX-2")
@register_model("LTX-2")
class xFuserLTX2VideoModel(xFuserModel):

    default_input_values = DefaultInputValues(
        height=1024,
        width=1536,
        num_frames=121,
        num_inference_steps=40,
        guidance_scale=4.0,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
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
        second_pipe = LTX2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        second_pipe.load_lora_weights(
            self.settings.model_name, adapter_name="stage_2_distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors"
        )
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.settings.model_name,
            subfolder="latent_upsampler",
            torch_dtype=torch.bfloat16,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)

        second_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config( # Scheduler for the 2nd stage
            pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
        )
        self.second_pipe = second_pipe
        self.upsample_pipe = upsample_pipe

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

        output = self.second_pipe(
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
        return DiffusionOutput(videos=output, pipe_args=input_args)

    def _compile_model(self, input_args: dict) -> None:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self.second_pipe.transformer = torch.compile(self.second_pipe.transformer, mode="default")

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2  # Reduce steps for warmup # TODO: make this more generic
        self._run_timed_pipe(compile_args)

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
        self.second_pipe.to(self.pipe.device)
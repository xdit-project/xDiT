import copy
from types import SimpleNamespace
from typing import ClassVar

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.core.utils.runner_utils import log
from xfuser.core.utils.video_utils import encode_video_with_audio
from xfuser.envs import PACKAGES_CHECKER
from xfuser.model_executor.models.runner_models.base_model import (
    DIFFUSERS_FROM_SOURCE,
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)

DEFAULT_NEGATIVE_PROMPT = (
    ""
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


@register_model("dg845/LTX-2.3-Diffusers")
@register_model("LTX-2.3")
class xFuserLTX23VideoModel(xFuserModel):
    min_diffusers_version = "0.37.0"

    default_input_values = DefaultInputValues(
        height=1024,
        width=1536,
        num_frames=121,
        num_inference_steps=40,
        guidance_scale=4.0,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )

    settings = ModelSettings(
        model_name="dg845/LTX-2.3-Diffusers",
        output_name="ltx_2_3_video",
        model_output_type="video",
        fps=24,
        resolution_divisor=64,
    )

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_tiling=True,
        enable_slicing=True,
    )

    _STG_SCALE = 1.0
    _SPATIO_TEMPORAL_GUIDANCE_BLOCKS: ClassVar[list[int]] = [28]
    _MODALITY_SCALE = 3.0
    _GUIDANCE_RESCALE = 0.7
    _AUDIO_GUIDANCE_SCALE = 7.0
    _AUDIO_STG_SCALE = 1.0
    _AUDIO_MODALITY_SCALE = 3.0
    _AUDIO_GUIDANCE_RESCALE = 0.7

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        from xfuser.model_executor.models.transformers.transformer_ltx2 import (
            xFuserLTX2VideoTransformer3DWrapper,
        )

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
            "Lightricks/LTX-2.3",
            adapter_name="stage_2_distilled",
            weight_name="ltx-2.3-22b-distilled-lora-384.safetensors",
        )
        second_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
        )

        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            "dg845/LTX-2.3-Spatial-Upsampler-Diffusers",
            subfolder="latent_upsampler",
            torch_dtype=torch.bfloat16,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=pipe.vae, latent_upsampler=latent_upsampler
        )

        second_pipe.vae.enable_tiling()

        self.second_pipe = second_pipe
        self.upsample_pipe = upsample_pipe

        return pipe

    def _enable_options(self) -> None:
        super()._enable_options()
        if self.config.enable_slicing:
            self.second_pipe.vae.enable_slicing()

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

        generator = torch.Generator(device="cuda").manual_seed(input_args["seed"])

        # self.pipe and self.second_pipe share the same transformer object, so the
        # stage 2 distilled LoRA loaded on second_pipe is also visible to pipe.
        # Toggle adapters so stage 1 runs on the base model.
        self.pipe.transformer.disable_adapters()

        video_latent, audio_latent = self.pipe(
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            height=input_args["height"] // 2,
            width=input_args["width"] // 2,
            num_frames=input_args["num_frames"],
            frame_rate=self.settings.fps,
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            stg_scale=self._STG_SCALE,
            spatio_temporal_guidance_blocks=self._SPATIO_TEMPORAL_GUIDANCE_BLOCKS,
            modality_scale=self._MODALITY_SCALE,
            guidance_rescale=self._GUIDANCE_RESCALE,
            audio_guidance_scale=self._AUDIO_GUIDANCE_SCALE,
            audio_stg_scale=self._AUDIO_STG_SCALE,
            audio_modality_scale=self._AUDIO_MODALITY_SCALE,
            audio_guidance_rescale=self._AUDIO_GUIDANCE_RESCALE,
            use_cross_timestep=True,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )

        video_latent = self.upsample_pipe(
            latents=video_latent, output_type="latent", return_dict=False
        )[0]

        self.second_pipe.transformer.enable_adapters()

        output = self.second_pipe(
            latents=video_latent,
            audio_latents=audio_latent,
            prompt=input_args["prompt"],
            negative_prompt=input_args["negative_prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            frame_rate=self.settings.fps,
            num_inference_steps=3,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            stg_scale=0.0,
            modality_scale=1.0,
            guidance_rescale=0.0,
            audio_guidance_scale=1.0,
            audio_stg_scale=0.0,
            audio_modality_scale=1.0,
            audio_guidance_rescale=0.0,
            spatio_temporal_guidance_blocks=None,
            use_cross_timestep=True,
            generator=generator,
            output_type="np",
        )

        return DiffusionOutput(videos=output, pipe_args=input_args)

    def _get_compile_mode(self) -> str:
        if PACKAGES_CHECKER._on_rdna4():
            return "default"
        return "reduce-overhead"

    def _compile_model(self, input_args: dict) -> None:
        super()._enable_compute_comm_overlap()
        self.pipe.transformer.compile_repeated_blocks(mode=self._get_compile_mode())

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = (
            2  # Reduce steps for warmup # TODO: make this more generic
        )
        self._run_timed_pipe(compile_args)

    def save_output(self, output: DiffusionOutput) -> None:
        pipe_args = output.pipe_args
        output = output.videos
        audio_sample_rate = self.pipe.vocoder.config.output_sampling_rate
        for i, video_object in enumerate(output):
            video, audio = video_object.frames, video_object.audio
            video = (video * 255).round().astype("uint8")
            video = torch.from_numpy(video)
            output_name = self.get_output_name(pipe_args[i])
            output_path = f"{self.config.output_directory}/{output_name}_{i}.mp4"
            encode_video_with_audio(
                video[0],
                audio=audio[0].float().cpu(),
                audio_sample_rate=audio_sample_rate,
                fps=self.settings.fps,
                output_path=output_path,
            )
            log(f"Output video saved to {output_path}")

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        self.upsample_pipe.to(self.pipe.device)
        self.second_pipe.to(self.pipe.device)


@register_model("Lightricks/LTX-2")
@register_model("LTX-2")
class xFuserLTX2VideoModel(xFuserModel):
    min_diffusers_version = "0.37.0"

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
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fps=24,
        resolution_divisor=64,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_tiling=True,
        enable_slicing=True,
        use_fp8_gemms=True,
    )

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        from xfuser.model_executor.models.transformers.transformer_ltx2 import (
            xFuserLTX2VideoTransformer3DWrapper,
        )

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
            self.settings.model_name,
            adapter_name="stage_2_distilled",
            weight_name="ltx-2-19b-distilled-lora-384.safetensors",
        )
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.settings.model_name,
            subfolder="latent_upsampler",
            torch_dtype=torch.bfloat16,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=pipe.vae, latent_upsampler=latent_upsampler
        )

        second_pipe.scheduler = (
            FlowMatchEulerDiscreteScheduler.from_config(  # Scheduler for the 2nd stage
                pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
            )
        )
        self.second_pipe = second_pipe
        self.upsample_pipe = upsample_pipe

        return pipe

    def _enable_options(self) -> None:
        super()._enable_options()
        if self.config.enable_tiling:
            self.second_pipe.vae.enable_tiling()
        if self.config.enable_slicing:
            self.second_pipe.vae.enable_slicing()

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

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

        video_latent = self.upsample_pipe(
            latents=video_latent, output_type="latent", return_dict=False
        )[0]

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

    def _get_compile_mode(self) -> str:
        if PACKAGES_CHECKER._on_rdna4():
            return "default"
        return "reduce-overhead"

    def _compile_model(self, input_args: dict) -> None:
        super()._enable_compute_comm_overlap()
        self.pipe.transformer.compile_repeated_blocks(mode=self._get_compile_mode())

        # two steps to warmup the torch compiler
        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = (
            2  # Reduce steps for warmup # TODO: make this more generic
        )
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
            encode_video_with_audio(
                video[0],
                audio=audio[0].float().cpu(),
                audio_sample_rate=24000,
                fps=self.settings.fps,
                output_path=output_path,
            )
            log(f"Output video saved to {output_path}")

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        self.upsample_pipe.to(self.pipe.device)
        self.second_pipe.to(self.pipe.device)


class _xFuserLTX25VideoModelBase(xFuserModel):
    """Shared lifecycle logic for LTX-2.5 distilled and full runner variants.

    Subclasses set:
        _TRANSFORMER_SUBFOLDER : str   – "transformer" or "transformer_full"
        _DISTILLED             : bool  – selects the two-stage vs single-stage
                                         inference recipe

    Guidance knobs default to distilled (no-guidance) values here.
    xFuserLTX25FullVideoModel overrides them with the full-model reference
    parameters from LTX-2 package constants.py.
    """

    _TRANSFORMER_SUBFOLDER: str = "transformer"
    _DISTILLED: bool = True
    # Guidance defaults for the distilled path (no CFG / STG / modality boost).
    # xFuserLTX25FullVideoModel overrides these with LTX-2.4/2.5 params.
    _STG_SCALE: float = 0.0
    _SPATIO_TEMPORAL_GUIDANCE_BLOCKS: ClassVar[list[int] | None] = None
    _MODALITY_SCALE: float = 1.0
    _GUIDANCE_RESCALE: float = 0.0
    _AUDIO_GUIDANCE_SCALE: float = 1.0
    _AUDIO_STG_SCALE: float = 0.0
    _AUDIO_MODALITY_SCALE: float = 1.0
    _AUDIO_GUIDANCE_RESCALE: float = 0.0

    min_diffusers_version = DIFFUSERS_FROM_SOURCE

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        enable_tiling=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
    )

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import (
            LTX2ImageToVideoPipeline,
            LTX2LatentUpsamplePipeline,
            LTX2Pipeline,
        )
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        use_sp = self.config.ulysses_degree > 1 or self.config.ring_degree > 1
        if use_sp:
            from xfuser.model_executor.models.transformers.transformer_ltx2 import (
                xFuserLTX2VideoTransformer3DWrapper,
            )

            transformer_cls = xFuserLTX2VideoTransformer3DWrapper
        else:
            from diffusers.models.transformers.transformer_ltx2 import (
                LTX2VideoTransformer3DModel,
            )

            transformer_cls = LTX2VideoTransformer3DModel

        transformer = transformer_cls.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.bfloat16,
            subfolder=self._TRANSFORMER_SUBFOLDER,
        )

        pipe_cls = (
            LTX2ImageToVideoPipeline if self.config.task == "i2v" else LTX2Pipeline
        )
        pipe = pipe_cls.from_pretrained(
            self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

        if not self._DISTILLED:
            # Full model uses a dynamic-shifting schedule
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                use_dynamic_shifting=True,
                shift_terminal=0.1,
            )

        if self._DISTILLED:
            latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                self.settings.model_name,
                subfolder="latent_upsampler",
                torch_dtype=torch.bfloat16,
            )
            self.upsample_pipe = LTX2LatentUpsamplePipeline(
                vae=pipe.vae, latent_upsampler=latent_upsampler
            )

        # Diffusion decoder — replaces convolutional VAE decode for both distilled
        # and full model pipelines.
        from diffusers import LTX2VideoDiffusionDecoderModel
        from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import (
            LTX2VideoDiffusionDecodePipeline,
        )

        diff_decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
            self.settings.model_name,
            subfolder="diffusion_decoder",
            torch_dtype=torch.bfloat16,
        )
        try:
            from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
                LTX2VideoVaeNeighborhoodNattenProcessor,
            )

            diff_decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
            log("Diffusion decoder: using NATTEN attention processor.")
        except (ImportError, RuntimeError, FileNotFoundError):
            # NATTEN is not available
            # Fall back to tiled PyTorch SDPA
            # Works on CUDA, ROCm and CPU. Ported from LTX-2 EagerSdpaAttention.
            from xfuser.model_executor.layers.ltx2_na3d_eager_attn import (
                LTX2VideoVaeEagerSdpaAttnProcessor,
            )

            diff_decoder.set_attn_processor(LTX2VideoVaeEagerSdpaAttnProcessor())
            log(
                "Diffusion decoder: NATTEN unavailable; using Triton na3d attention fallback."
            )
        self.decode_pipe = LTX2VideoDiffusionDecodePipeline(
            diffusion_decoder=diff_decoder,
            scheduler=pipe.scheduler,
        )

        return pipe

    def _enable_options(self) -> None:
        super()._enable_options()
        if self.config.enable_tiling:
            self.pipe.vae.enable_tiling()
            self.decode_pipe.diffusion_decoder.enable_tiling()

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        if self.config.task == "i2v":
            input_args["image"] = input_args["input_images"][0]
        return input_args

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        if self.config.task == "i2v":
            images = input_args.get("input_images") or []
            if len(images) != 1:
                raise ValueError(
                    f"LTX-2.5 I2V requires exactly one input image, got {len(images)}."
                )
        if self._DISTILLED:
            steps = input_args.get("num_inference_steps")
            if steps != 8:
                raise ValueError(
                    f"LTX-2.5 distilled uses a fixed 8-step schedule; "
                    f"num_inference_steps must be 8, got {steps}."
                )
            guidance_scale = input_args.get("guidance_scale")
            if guidance_scale != 1.0:
                log(
                    "Using guidance_scale=1.0. Other guindance scale values are not supported with this model."
                )

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        from diffusers.pipelines.ltx2.utils import (
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
        )

        is_i2v = self.config.task == "i2v"
        generator = torch.Generator(device="cuda").manual_seed(input_args["seed"])

        shared = {
            "prompt": input_args["prompt"],
            "frame_rate": self.settings.fps,
            "stg_scale": self._STG_SCALE,
            "modality_scale": self._MODALITY_SCALE,
            "audio_guidance_scale": self._AUDIO_GUIDANCE_SCALE,
            "audio_stg_scale": self._AUDIO_STG_SCALE,
            "audio_modality_scale": self._AUDIO_MODALITY_SCALE,
            "generator": generator,
        }
        if is_i2v:
            shared["image"] = input_args["image"]

        if self._DISTILLED:
            # Distilled model: no guidance, fixed 8-step sigma schedule.
            shared["guidance_scale"] = 1.0
            video_latent, audio_latent = self.pipe(
                height=input_args["height"] // 2,
                width=input_args["width"] // 2,
                num_frames=input_args["num_frames"],
                sigmas=DISTILLED_SIGMA_VALUES,
                output_type="latent",
                return_dict=False,
                **shared,
            )

            video_latent = self.upsample_pipe(
                latents=video_latent, output_type="latent", return_dict=False
            )[0]

            # Stage-2: guidance disabled.
            stage2_shared = {
                "prompt": input_args["prompt"],
                "frame_rate": self.settings.fps,
                "guidance_scale": 1.0,
                "stg_scale": 0.0,
                "modality_scale": 1.0,
                "audio_guidance_scale": 1.0,
                "audio_stg_scale": 0.0,
                "audio_modality_scale": 1.0,
                "generator": generator,
            }
            if is_i2v:
                stage2_shared["image"] = input_args["image"]

            video_latents, audio_latents = self.pipe(
                num_frames=input_args["num_frames"],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                latents=video_latent,
                audio_latents=audio_latent,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                output_type="latent",
                return_dict=False,
                **stage2_shared,
            )
        else:
            # Full model: single-stage with full guidance (STG, modality, CFG).
            shared["guidance_scale"] = input_args["guidance_scale"]
            shared["guidance_rescale"] = self._GUIDANCE_RESCALE
            shared["audio_guidance_rescale"] = self._AUDIO_GUIDANCE_RESCALE
            shared["use_cross_timestep"] = True
            shared["negative_prompt"] = input_args["negative_prompt"]
            if self._SPATIO_TEMPORAL_GUIDANCE_BLOCKS is not None:
                shared["spatio_temporal_guidance_blocks"] = (
                    self._SPATIO_TEMPORAL_GUIDANCE_BLOCKS
                )
            video_latents, audio_latents = self.pipe(
                height=input_args["height"],
                width=input_args["width"],
                num_frames=input_args["num_frames"],
                num_inference_steps=input_args["num_inference_steps"],
                output_type="latent",
                return_dict=False,
                **shared,
            )

        # Diffusion decode (replaces convolutional VAE decode)
        video_np = self.decode_pipe(
            video_latents,
            generator=generator,
            output_type="np",
            denormalize=False,
            return_dict=False,
        )[0]  # (B, T, H, W, C) float32 in [0, 1]

        # Audio: pipeline returned raw audio latents when output_type="latent".
        # Decode via audio_vae -> mel -> vocoder -> waveform.
        mel = self.pipe.audio_vae.decode(
            audio_latents.to(self.pipe.audio_vae.dtype), return_dict=False
        )[0]
        audio_waveform = self.pipe.vocoder(mel)

        output = SimpleNamespace(frames=video_np, audio=audio_waveform)
        return DiffusionOutput(videos=output, pipe_args=input_args)

    def _get_compile_mode(self) -> str:
        if PACKAGES_CHECKER._on_rdna4():
            return "default"
        return "reduce-overhead"

    def _compile_model(self, input_args: dict) -> None:
        super()._enable_compute_comm_overlap()

        if hasattr(self.pipe.transformer, "compile_repeated_blocks"):
            # xFuser wrapper: compiles each block; the wrapper forward adds
            # .clone() between calls to prevent pytorch#152887 CUDAGraph aliasing.
            self.pipe.transformer.compile_repeated_blocks(mode=self._get_compile_mode())
            # Register a pre-hook so the CUDAGraph system gets a step-boundary
            # signal before EVERY transformer forward (not just before the pipe
            # call).  The denoising loop calls the transformer once per step
            # (×8 for distilled stage-1, ×3 for stage-2, ×3× for full model
            # with guidance), so the hook must fire per-invocation.
            # Mirrors diffusers_adapters/flux2.py
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):

                def _mark_cudagraph_step(module, args, kwargs):
                    torch.compiler.cudagraph_mark_step_begin()

                self.pipe.transformer.register_forward_pre_hook(
                    _mark_cudagraph_step, with_kwargs=True, prepend=True
                )

        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)

    def save_output(self, output: DiffusionOutput) -> None:
        pipe_args = output.pipe_args
        output = output.videos
        audio_sample_rate = self.pipe.vocoder.config.output_sampling_rate
        for i, video_object in enumerate(output):
            video, audio = video_object.frames, video_object.audio
            video = (video * 255).round().astype("uint8")
            video = torch.from_numpy(video)
            output_name = self.get_output_name(pipe_args[i])
            output_path = f"{self.config.output_directory}/{output_name}_{i}.mp4"
            encode_video_with_audio(
                video[0],
                audio=audio[0].float().cpu(),
                audio_sample_rate=audio_sample_rate,
                fps=self.settings.fps,
                output_path=output_path,
            )
            log(f"Output video saved to {output_path}")

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        if self._DISTILLED:
            self.upsample_pipe.to(self.pipe.device)
        self.decode_pipe.diffusion_decoder.to(self.pipe.device)


@register_model("Lightricks/LTX-2.5-Diffusers")
@register_model("LTX-2.5-distilled")
@register_model("LTX-2.5")
class xFuserLTX25DistilledVideoModel(_xFuserLTX25VideoModelBase):
    """LTX-2.5 distilled (default) T2V / I2V runner.

    Uses the two-stage flow: stage-1 at half resolution with
    DISTILLED_SIGMA_VALUES, spatial latent upsample, then stage-2 refine at
    full resolution with STAGE_2_DISTILLED_SIGMA_VALUES.
    """

    _TRANSFORMER_SUBFOLDER = "transformer"
    _DISTILLED = True

    default_input_values = DefaultInputValues(
        height=1024,
        width=1536,
        num_frames=121,
        num_inference_steps=8,
        guidance_scale=1.0,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )

    settings = ModelSettings(
        model_name="Lightricks/LTX-2.5-Diffusers",
        output_name="ltx_2_5_distilled_video",
        model_output_type="video",
        fps=24,
        resolution_divisor=64,
        valid_tasks=["t2v", "i2v"],
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fp4_gemm_module_list=["transformer.transformer_blocks"],
    )


@register_model("LTX-2.5-full")
class xFuserLTX25FullVideoModel(_xFuserLTX25VideoModelBase):
    """LTX-2.5 full/SFT T2V / I2V runner.

    Single-stage inference: full transformer, dynamic-shifting schedule,
    full guidance (cfg=3.0, stg=1.0, modality=3.0).
    """

    _TRANSFORMER_SUBFOLDER = "transformer_full"
    _DISTILLED = False  # single-stage, no upsampler

    # Full-model guidance parameters - LTX-2.4/2.5 params:
    # video:  cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0, stg_blocks=[28]
    # audio:  cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0, stg_blocks=[28]
    _STG_SCALE = 1.0
    _SPATIO_TEMPORAL_GUIDANCE_BLOCKS: ClassVar[list[int]] = [28]
    _MODALITY_SCALE = 3.0
    _GUIDANCE_RESCALE = 0.7
    _AUDIO_GUIDANCE_SCALE = 7.0
    _AUDIO_STG_SCALE = 1.0
    _AUDIO_MODALITY_SCALE = 3.0
    _AUDIO_GUIDANCE_RESCALE = 0.7

    default_input_values = DefaultInputValues(
        height=1024,
        width=1536,
        num_frames=121,
        num_inference_steps=30,
        guidance_scale=3.0,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )

    settings = ModelSettings(
        model_name="Lightricks/LTX-2.5-Diffusers",
        output_name="ltx_2_5_full_video",
        model_output_type="video",
        fps=24,
        resolution_divisor=64,
        valid_tasks=["t2v", "i2v"],
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fp4_gemm_module_list=["transformer.transformer_blocks"],
    )

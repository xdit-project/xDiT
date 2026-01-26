import torch
from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline
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
        width=1024,
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
        valid_tasks=["i2v", "t2v"]
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
        if self.config.task == "i2v":
            pipe = LTX2ImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            pipe = LTX2Pipeline.from_pretrained(
                pretrained_model_name_or_path=self.settings.model_name,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "prompt": input_args["prompt"],
            "height": input_args["height"],
            "width": input_args["width"],
            "num_frames": input_args["num_frames"],
            "frame_rate": self.settings.fps,
            "num_inference_steps": input_args["num_inference_steps"],
            "guidance_scale": input_args["guidance_scale"],
            "output_type": "np",
            "generator": torch.Generator(device="cuda").manual_seed(input_args["seed"]),
        }
        if self.config.task == "i2v":
            kwargs["image"] = input_args["input_images"][0]
        output = self.pipe(**kwargs)
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


    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if self.config.task == "i2v" and len(images) != 1:
            raise ValueError("Exactly one input image is required for LTX-2 I2V task.")
        elif self.config.task == "t2v" and len(images) != 0:
            raise ValueError("Input images are not supported for LTX-2 T2V task.")
        return input_args
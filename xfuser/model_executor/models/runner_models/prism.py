import os
from types import SimpleNamespace

import torch

from xfuser.core.distributed import get_world_group
from xfuser.core.utils.runner_utils import log
from xfuser.core.utils.video_utils import encode_video_with_audio
from xfuser.model_executor.models.runner_models.base_model import (
    DefaultInputValues,
    DiffusionOutput,
    ModelCapabilities,
    ModelSettings,
    register_model,
    xFuserModel,
)

PRISM_REPO = "FrancisRing/Prism"
# Prism ships its weights as one state dict over the MOVA base model's module tree;
# the base folder supplies the configs, VAEs, text encoder, tokenizer and scheduler.
_BASE_SUBFOLDER = "pretrained_models/MOVA-360p"
_VIDEO_HEADS = 40
# Block shape Prism's block-sparse attention was trained and sampled with.
_BSA_CHUNK_THW = (4, 4, 4)

# Prism's sampler default (hymm/sample/sample_mova_single.py), the Wan negative prompt.
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)


class PrismDiffusionOutput(DiffusionOutput):
    def __init__(self, videos, audio, audio_sample_rate: int, pipe_args) -> None:
        super().__init__(videos=videos, pipe_args=pipe_args)
        self.audio = audio
        self.audio_sample_rate = audio_sample_rate


def snap_num_frames(num_frames: int, temporal_scale: int = 4) -> int:
    """Round down to the nearest ``k * temporal_scale + 1`` frames the video VAE accepts."""
    if num_frames < temporal_scale + 1:
        return temporal_scale + 1
    return num_frames - (num_frames - 1) % temporal_scale


@register_model(PRISM_REPO)
@register_model("Prism-preview-alpha")
@register_model("Prism")
class xFuserPrismModel(xFuserModel):
    """Prism preview-alpha: reference image + text to 24 fps video with 48 kHz audio."""

    # AutoencoderKLWan, the video VAE, first shipped in diffusers 0.33.0.
    min_diffusers_version = "0.33.0"
    checkpoint_subfolder = "preview_alpha"
    attention_head_dims = frozenset({128})
    # Ulysses splits the video tower's heads; the 12 audio heads are zero-padded instead.
    attention_heads = _VIDEO_HEADS

    capabilities = ModelCapabilities(
        ulysses_degree=True,
        # Ring would need the audio-length key trimming and the head padding done per ring step.
        ring_degree=False,
        data_parallel_degree=False,
        # Everything but video self-attention runs on it, e.g. dense beside TRITON_BSA.
        cross_attention_backend=True,
        supports_bsa_attention_backends=True,
        enable_tiling=True,
    )
    default_input_values = DefaultInputValues(
        height=480,
        width=848,
        num_frames=205,
        num_inference_steps=50,
        guidance_scale=5.0,
        # The video tower's noise-schedule shift; the audio tower keeps _AUDIO_SHIFT.
        flow_shift=9.0,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )
    settings = ModelSettings(
        model_name=PRISM_REPO,
        output_name="prism_preview_alpha",
        model_output_type="video",
        fps=24,
        # VAE stride 8 times patch size 2.
        resolution_divisor=16,
    )
    _AUDIO_SHIFT = 7.0

    def _get_compile_warmup_steps(self, input_args: dict) -> int | None:
        """Fewest steps whose schedule reaches the low-noise expert.

        Each video expert compiles to its own graph, so a warmup that stays on the
        high-noise one leaves the other to compile in the middle of the first timed run.
        """
        boundary = self.pipe.transformer.boundary_ratio * self.pipe.scheduler.config.num_train_timesteps
        for steps in range(2, input_args["num_inference_steps"] + 1):
            pairs = self.pipe.scheduler.set_pair_timesteps(steps, input_args["flow_shift"], self._AUDIO_SHIFT)
            if (pairs[:, 0] < boundary).any():
                return steps
        return None

    def _validate_config(self, config) -> None:
        super()._validate_config(config)
        if config.enable_model_cpu_offload or config.enable_sequential_cpu_offload or config.enable_group_cpu_offload:
            raise ValueError("Prism does not support CPU offloading.")
        if config.batch_size is not None or config.dataset_path is not None:
            raise ValueError("Prism generates one clip per request and does not support batching or datasets.")
        if not 0.0 <= config.bsa_sparsity < 1.0 or not 0.0 <= config.bsa_cdf_threshold < 1.0:
            raise ValueError(
                f"--bsa_sparsity and --bsa_cdf_threshold must lie in [0, 1), "
                f"got {config.bsa_sparsity} and {config.bsa_cdf_threshold}."
            )

    def preprocess_args(self, input_args: dict) -> dict:
        args = super().preprocess_args(input_args)
        if isinstance(args.get("prompt"), list) and len(args["prompt"]) == 1:
            args["prompt"] = args["prompt"][0]
        snapped = snap_num_frames(args["num_frames"])
        if snapped != args["num_frames"]:
            log(f"Prism needs 4k+1 frames; using {snapped} instead of {args['num_frames']}.")
            args["num_frames"] = snapped
        return args

    def _validate_args(self, input_args: dict) -> None:
        super()._validate_args(input_args)
        if isinstance(input_args["prompt"], list) and len(input_args["prompt"]) != 1:
            raise ValueError("Prism generates one clip per request; pass a single --prompt.")
        if len(input_args.get("input_images") or []) != 1:
            raise ValueError("Prism animates a reference image: pass exactly one --input_images.")

    def _checkpoint_dir(self) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.settings.model_name,
            allow_patterns=[f"{_BASE_SUBFOLDER}/*", f"{self.checkpoint_subfolder}/*"],
        )

    def _load_model(self):
        from diffusers import AutoencoderKLWan
        from safetensors.torch import load_file
        from transformers import T5TokenizerFast, UMT5EncoderModel

        from xfuser.model_executor.models.customized.prism import (
            DAC,
            FlowMatchPairScheduler,
            MOVABridge,
            PrismPipeline,
        )

        root = self._checkpoint_dir()
        base = os.path.join(root, _BASE_SUBFOLDER)
        weights = os.path.join(root, self.checkpoint_subfolder, "diffusion_pytorch_model.safetensors")
        device = f"cuda:{get_world_group().local_rank}"

        # Build the 33B-parameter bridge on meta and read the Prism weights straight onto the
        # device, rather than load the MOVA base DiTs only for Prism's to overwrite them.
        log(f"Loading {self.settings.model_name} {self.checkpoint_subfolder}")
        transformer = MOVABridge.from_mova_config(base)
        missing, unexpected = transformer.load_state_dict(load_file(weights, device=device), strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                f"{weights} does not match the MOVA module tree: "
                f"{len(missing)} missing keys (e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})."
            )
        transformer.eval().requires_grad_(False)
        # Read by TRITON_BSA on the video self-attention and ignored by dense backends.
        transformer.video_attention_kwargs = {
            "bsa_sparsity": self.config.bsa_sparsity,
            "bsa_cdf_threshold": self.config.bsa_cdf_threshold or None,
            "bsa_chunk_thw": _BSA_CHUNK_THW,
        }

        return PrismPipeline(
            transformer=transformer,
            vae=AutoencoderKLWan.from_pretrained(base, subfolder="video_vae", torch_dtype=torch.bfloat16),
            audio_vae=DAC.from_pretrained(base, subfolder="audio_vae", torch_dtype=torch.bfloat16),
            text_encoder=UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.bfloat16),
            tokenizer=T5TokenizerFast.from_pretrained(base, subfolder="tokenizer"),
            scheduler=FlowMatchPairScheduler.from_pretrained(base, subfolder="scheduler"),
        )

    def _get_runtime_state_pipeline(self):
        # DiTRuntimeState reads diffusers-style transformer config fields.
        config = SimpleNamespace(
            num_attention_heads=_VIDEO_HEADS,
            attention_head_dim=128,
            patch_size=2,
            in_channels=self.pipe.transformer.video_dit.config.in_dim,
        )
        return SimpleNamespace(
            transformer=SimpleNamespace(config=config),
            vae_scale_factor=self.pipe.vae_scale_factor_spatial,
        )

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        video, audio = self.pipe(
            prompt=input_args["prompt"],
            image=input_args["input_images"][0],
            # None lets the pipeline reuse the video prompt embedding rather than re-encode it.
            audio_prompt=input_args.get("audio_prompt") or None,
            negative_prompt=input_args["negative_prompt"],
            height=input_args["height"],
            width=input_args["width"],
            num_frames=input_args["num_frames"],
            video_fps=float(self.settings.fps),
            num_inference_steps=input_args["num_inference_steps"],
            visual_shift=input_args["flow_shift"],
            audio_shift=self._AUDIO_SHIFT,
            cfg_scale=input_args["guidance_scale"],
            generator=torch.Generator(device=self.pipe._execution_device).manual_seed(input_args["seed"]),
        )
        return PrismDiffusionOutput(
            videos=[video],
            audio=[audio],
            audio_sample_rate=self.pipe.audio_sample_rate,
            pipe_args=input_args,
        )

    def save_output(self, output: DiffusionOutput) -> None:
        for index, (video, pipe_args) in enumerate(output.get_outputs()):
            frames = torch.from_numpy((video * 255).round().clip(0, 255).astype("uint8"))
            # DAC decodes mono; the AAC stream is stereo.
            audio = output.audio[index].float().cpu().expand(2, -1)
            output_path = f"{self.config.output_directory}/{self.get_output_name(pipe_args)}_{index}.mp4"
            encode_video_with_audio(
                frames,
                audio=audio,
                audio_sample_rate=output.audio_sample_rate,
                fps=self.settings.fps,
                output_path=output_path,
            )
            log(f"Output video with audio saved to {output_path}")

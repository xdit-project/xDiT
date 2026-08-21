import copy
import json
import os
import torch
from diffusers import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from xfuser import xFuserArgs
from xfuser.model_executor.pipelines.pipeline_lingbot_video import (
    xFuserLingBotVideoPipeline,
    get_lingbot_video_pipeline_class,
)
from xfuser.model_executor.models.runner_models.base_model import (
    ModelSettings,
    xFuserModel,
    register_model,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    LoadRoute,
)
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.core.distributed.parallel_state import get_vae_parallel_group
from xfuser.core.utils.runner_utils import log


LINGBOT_FSDP_STRATEGY = {
    "transformer": {
        "wrap_attrs": ["blocks"],
        "dtype": torch.bfloat16,
    },
}

# MoE backend config
os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
os.environ.setdefault("LINGBOT_MOE_PAD_BACKEND", "vectorized")

# Refiner constants
REFINER_BASE_HEIGHT = 480
REFINER_BASE_WIDTH = 832
REFINER_STEPS = 8        # canonical: refiner runs few steps on partially-noised input
REFINER_T_THRESH = 0.85  # canonical: noise level threshold for refiner init

DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", '
    '"blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", '
    '"color flicker", "underexposed", "overexposed", "invisible subject", '
    '"subject hidden in darkness"], "artistic_style": ["painting", "illustration", '
    '"drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], '
    '"composition_and_content": ["text", "watermark", "signature", "logo", '
    '"subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], '
    '"temporal_and_motion_stability": ["flickering", "jittery", "motion blur", '
    '"temporal inconsistency", "warping", "morphing", "incoherent motion", '
    '"unnatural movement", "static object with sudden jump", '
    '"frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", '
    '"unrealistic texture", "deformed bottle", "liquid freezing improperly", '
    '"distorted reflections"]}}'
)


def _load_json_prompt(prompt: str) -> str:
    """If prompt is a path to a .json file, load and serialize the caption.

    LingBot-Video expects structured JSON captions, not plain text.
    The JSON file should have {"caption": {...}, "duration": N} format.
    The caption dict is serialized to a compact JSON string.
    """
    if isinstance(prompt, str) and prompt.endswith(".json") and os.path.isfile(prompt):
        with open(prompt, encoding="utf-8") as f:
            data = json.load(f)
        caption = data.get("caption", data)
        if isinstance(caption, (dict, list)):
            return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
        return str(caption)
    return prompt


@register_model("robbyant/lingbot-video-moe-30b-a3b")
@register_model("LingBot-Video-MoE")
class xFuserLingBotVideoMoEModel(xFuserModel):
    def save_output(self, output):
        # Stock TI2V CFG parallel puts output on rank 0, but xDiT's runner
        # only saves from the last rank. Skip gracefully when no output.
        if output.videos or output.images:
            super().save_output(output)
        else:
            log("No output on this rank (CFG parallel secondary), skipping save.")

    def _validate_config(self, config):
        # Temporarily clear task to skip base class task validation,
        # then restore — we handle --task refine ourselves.
        saved_task = config.task
        config.task = None
        super()._validate_config(config)
        config.task = saved_task

    def _calculate_hybrid_attention_step_multiplier(self, input_args: dict) -> int:
        if input_args["guidance_scale"] > 1.0 and not self.config.use_cfg_parallel:
            return 2
        return 1

    # Composed loading and custom FSDP wrapping bypass xDiT's shared load seam.
    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        use_cfg_parallel=True,
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_hybrid_attn_schedule=True,
        use_hybrid_gemm_schedule=True,
        fully_shard_degree=True,
        use_parallel_vae=True,
        use_parallel_vae_encoder=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=480,
        width=832,
        num_inference_steps=40,
        num_frames=81,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance_scale=3.0,
        flow_shift=3.0,
    )
    settings = ModelSettings(
        model_name="robbyant/lingbot-video-moe-30b-a3b",
        output_name="lingbot_video_moe",
        model_output_type="video",
        fps=24,
        fp8_gemm_module_list=["transformer.blocks"],
        fp4_gemm_module_list=["transformer.blocks"],
        fsdp_strategy=LINGBOT_FSDP_STRATEGY,
    )

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        # Cache bulk_dtype on each block before quantization replaces nn.Linear
        # modules — LingBotVideoBlock.forward reads self.attn.to_q.weight.dtype
        # which breaks after FP4/FP8 quantization removes .weight.
        if self.config.use_fp4_gemms or self.config.use_fp8_gemms:
            for block in self.pipe.transformer.blocks:
                if hasattr(block, "attn") and hasattr(block.attn, "to_q"):
                    w = getattr(block.attn.to_q, "weight", None)
                    if w is not None:
                        block._cached_bulk_dtype = w.dtype
        # Use LingBot's own FSDP approach instead of xDiT's: shard per-block
        # with ignored_params for minority-dtype (FP32 norms/router).
        if self.config.fully_shard_degree > 1:
            from lingbot_video.fsdp_inference import apply_fsdp_inference, init_fsdp_inference_mesh
            local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self.pipe.transformer.to(f"cuda:{local_rank}")
            mesh = init_fsdp_inference_mesh()
            info = apply_fsdp_inference(self.pipe.transformer, mesh)
            log(f"FSDP: {info.wrapped_blocks} blocks sharded, "
                f"{info.ignored_params} FP32 params excluded, "
                f"VRAM: {torch.cuda.memory_allocated(local_rank)/1e9:.1f}GB")
            # Move remaining components to GPU
            self.pipe.vae.to(f"cuda:{local_rank}")
            self.pipe.text_encoder.to(f"cuda:{local_rank}")
            # Skip xDiT's own FSDP sharding
            saved_fsdp = self.config.fully_shard_degree
            self.config.fully_shard_degree = 1
            super()._post_load_and_state_initialization(input_args)
            self.config.fully_shard_degree = saved_fsdp
        else:
            super()._post_load_and_state_initialization(input_args)
        # After quantization, patch blocks that lost .weight.dtype
        if self.config.use_fp4_gemms or self.config.use_fp8_gemms:
            from xfuser.model_executor.models.transformers.transformer_lingbot_video import _patch_block_bulk_dtype
            for block in self.pipe.transformer.blocks:
                if hasattr(block, "_cached_bulk_dtype"):
                    _patch_block_bulk_dtype(block)
        # Cache pre-transposed expert weights to eliminate per-call copies
        self.pipe.transformer.cache_expert_weights()

        # Load refiner if requested via --task refine
        self.refiner_pipe = None
        use_refiner = self.config.task == "refine"
        if use_refiner:
            self._load_refiner(input_args)

    def _load_refiner(self, input_args):
        from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
        from xfuser.model_executor.models.transformers.transformer_lingbot_video import (
            xFuserLingBotVideoTransformer3DWrapper,
        )

        log("Loading refiner transformer...")
        model_name = self.settings.model_name
        refiner_transformer = xFuserLingBotVideoTransformer3DWrapper.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, subfolder="refiner",
        )
        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        refiner_transformer = refiner_transformer.to(f"cuda:{local_rank}")
        refiner_scheduler = FlowUniPCMultistepScheduler.from_pretrained(
            model_name, subfolder="scheduler",
        )
        LingBotVideoPipeline = get_lingbot_video_pipeline_class()
        refiner_pipe = LingBotVideoPipeline(
            transformer=refiner_transformer,
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            processor=self.pipe.processor,
            scheduler=refiner_scheduler,
        )
        refiner_pipe.__class__ = type(
            "xFuserLingBotVideoPipeline",
            (xFuserLingBotVideoPipeline, LingBotVideoPipeline),
            {},
        )
        # Quantize refiner linears (same path as base transformer)
        if self.config.use_fp4_gemms or self.config.use_fp8_gemms:
            from xfuser.model_executor.models.transformers.transformer_lingbot_video import _patch_block_bulk_dtype
            from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp4, quantize_linear_layers_to_fp8
            # Cache bulk dtype before quantization replaces nn.Linear
            for block in refiner_transformer.blocks:
                if hasattr(block, "attn") and hasattr(block.attn, "to_q"):
                    w = getattr(block.attn.to_q, "weight", None)
                    if w is not None:
                        block._cached_bulk_dtype = w.dtype
            device = f"cuda:{local_rank}"
            if self.config.use_fp4_gemms:
                log("Quantizing refiner blocks to FP4...")
                quantize_linear_layers_to_fp4(
                    refiner_transformer.blocks,
                    fp8_layers=self.settings.fp8_precision_overrides,
                    use_hybrid_schedule=self.config.use_hybrid_gemm_schedule,
                    device=device,
                )
            elif self.config.use_fp8_gemms:
                log("Quantizing refiner blocks to FP8...")
                quantize_linear_layers_to_fp8(refiner_transformer.blocks, device=device)
            # Patch blocks that lost .weight.dtype
            for block in refiner_transformer.blocks:
                if hasattr(block, "_cached_bulk_dtype"):
                    _patch_block_bulk_dtype(block)
        # FSDP shard the refiner transformer if enabled
        if self.config.fully_shard_degree > 1:
            from xfuser.core.distributed.parallel_state import get_fs_group
            from xfuser.core.distributed.sharding import shard_component
            device_group = get_fs_group().device_group
            fs_local_rank = get_fs_group().local_rank
            log("Sharding refiner transformer with FSDP...")
            refiner_pipe.transformer = shard_component(
                refiner_transformer, ["blocks"], device_group, fs_local_rank,
                torch.bfloat16, sync_module_states=False,
            )

        self.refiner_pipe = refiner_pipe
        refiner_pipe.transformer.cache_expert_weights()
        log("Refiner loaded.")

    def _build_pipe(self, model_name, transformer_subfolder="transformer", use_i2v=False):
        from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
        from xfuser.model_executor.models.transformers.transformer_lingbot_video import (
            xFuserLingBotVideoTransformer3DWrapper,
        )

        transformer = xFuserLingBotVideoTransformer3DWrapper.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, subfolder=transformer_subfolder,
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_name, torch_dtype=torch.float32, subfolder="vae",
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, subfolder="text_encoder",
        )
        processor = Qwen3VLProcessor.from_pretrained(
            model_name, subfolder="processor",
        )
        scheduler = FlowUniPCMultistepScheduler.from_pretrained(
            model_name, subfolder="scheduler",
        )
        if use_i2v:
            from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline
            BasePipeClass = LingBotVideoImageToVideoPipeline
        else:
            BasePipeClass = get_lingbot_video_pipeline_class()
        pipe = BasePipeClass(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            scheduler=scheduler,
        )
        pipe.__class__ = type(
            "xFuserLingBotVideoPipeline",
            (xFuserLingBotVideoPipeline, BasePipeClass),
            {},
        )
        return pipe

    def _load_model(self) -> DiffusionPipeline:
        use_i2v = bool(self.config.img_file_path) or bool(self.config.input_images)
        if use_i2v:
            log("TI2V mode: loading image-to-video pipeline")
        return self._build_pipe(self.settings.model_name, use_i2v=use_i2v)

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        raw_prompt = input_args["prompt"]
        if isinstance(raw_prompt, list):
            raw_prompt = raw_prompt[0] if len(raw_prompt) == 1 else " ".join(raw_prompt)
        prompt = _load_json_prompt(raw_prompt)
        generator = torch.Generator(device="cuda").manual_seed(input_args["seed"])

        # In refiner mode, --height/--width is the final output resolution;
        # base pass runs at fixed 480x832, refiner upscales to target.
        use_refiner = hasattr(self, "refiner_pipe") and self.refiner_pipe is not None
        if use_refiner:
            refiner_height = input_args["height"]
            refiner_width = input_args["width"]
            base_height = REFINER_BASE_HEIGHT
            base_width = REFINER_BASE_WIDTH
        else:
            base_height = input_args["height"]
            base_width = input_args["width"]

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=input_args["negative_prompt"],
            height=base_height,
            width=base_width,
            num_frames=input_args["num_frames"],
            num_inference_steps=input_args["num_inference_steps"],
            guidance_scale=input_args["guidance_scale"],
            shift=input_args.get("flow_shift", 3.0),
            generator=generator,
        )

        # TI2V: pass image to pipeline
        images = input_args.get("input_images", [])
        if images:
            pipe_kwargs["image"] = images[0]

        # Base pass
        output = self.pipe(**pipe_kwargs)

        # Refiner pass
        if use_refiner and output.frames:
            output = self._run_refiner(output, input_args, prompt, generator,
                                       refiner_height, refiner_width)

        return DiffusionOutput(videos=output.frames or None, pipe_args=input_args)

    def _run_refiner(self, base_output, input_args, prompt, generator,
                     refiner_height, refiner_width):
        from lingbot_video.utils import prepare_refiner_latent
        import numpy as np

        torch.cuda.empty_cache()
        device = next(self.refiner_pipe.vae.parameters()).device
        vae = self.refiner_pipe.vae

        base_frames = base_output.frames
        frames_np = np.array(base_frames[0] if isinstance(base_frames, list) else base_frames)
        # (T, H, W, 3) -> (T, 3, H, W) on CPU
        frames_cpu = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        del frames_np, base_frames
        num_frames = frames_cpu.shape[0]

        # Resize on CPU (avoids GPU allocation of full 1080p tensor)
        frames_cpu = torch.nn.functional.interpolate(
            frames_cpu, size=(refiner_height, refiner_width),
            mode="bilinear", align_corners=False,
        ).mul_(2.0).sub_(1.0)
        # (T, 3, H, W) -> (1, 3, T, H, W), transfer to GPU only for VAE encode
        video_tensor = frames_cpu.unsqueeze(0).permute(0, 2, 1, 3, 4)
        del frames_cpu

        refiner_generator = torch.Generator(device="cuda").manual_seed(input_args["seed"])
        with torch.no_grad():
            x_up = self.refiner_pipe._vae_latent_to_dit(
                vae.encode(video_tensor.to(device=device, dtype=vae.dtype)).latent_dist.mode()
            )
            del video_tensor
            torch.cuda.empty_cache()
            noise = torch.randn(
                x_up.shape, device=x_up.device, dtype=x_up.dtype,
                generator=refiner_generator,
            )
            initial_latent = prepare_refiner_latent(x_up, noise, REFINER_T_THRESH)

        guidance = input_args.get("guidance_scale_2") or input_args["guidance_scale"]
        shift = input_args.get("flow_shift", 3.0)
        log(f"Running refiner: {refiner_height}x{refiner_width}, "
            f"{REFINER_STEPS} steps, guidance={guidance}, t_thresh={REFINER_T_THRESH}")
        refiner_output = self.refiner_pipe(
            prompt=prompt,
            negative_prompt=input_args["negative_prompt"],
            height=refiner_height,
            width=refiner_width,
            num_frames=num_frames,
            num_inference_steps=REFINER_STEPS,
            guidance_scale=guidance,
            shift=shift,
            t_thresh=REFINER_T_THRESH,
            generator=refiner_generator,
            latents=initial_latent,
        )
        return refiner_output

    def _prepare_and_compile_transformer(self, pipe, height, width, num_frames):
        """Prepare RoPE + MoE patches for compile, then compile and warmup."""
        pipe.transformer.prepare_for_compile(
            height=height,
            width=width,
            num_frames=num_frames,
            vae_scale_temporal=getattr(pipe, "vae_scale_factor_temporal", 4),
            vae_scale_spatial=getattr(pipe, "vae_scale_factor_spatial", 8),
        )
        pipe.transformer = torch.compile(pipe.transformer, mode="default")

    def _reset_scheduler(self, pipe):
        scheduler = pipe.scheduler
        if hasattr(scheduler, "timestep_list"):
            scheduler.timestep_list = [None] * len(scheduler.timestep_list)

    def _compile_model(self, input_args):
        torch._inductor.config.reorder_for_compute_comm_overlap = True

        # In refiner mode, --height/--width is the target; base uses 480x832
        if self.refiner_pipe is not None:
            base_h, base_w = REFINER_BASE_HEIGHT, REFINER_BASE_WIDTH
            refiner_h, refiner_w = input_args["height"], input_args["width"]
        else:
            base_h, base_w = input_args["height"], input_args["width"]

        # Compile base transformer
        self._prepare_and_compile_transformer(
            self.pipe, base_h, base_w, input_args["num_frames"],
        )

        # Compile refiner BEFORE warmup so warmup covers both
        if self.refiner_pipe is not None:
            log("Compiling refiner transformer...")
            self._prepare_and_compile_transformer(
                self.refiner_pipe, refiner_h, refiner_w,
                input_args["num_frames"],
            )

        compile_args = copy.deepcopy(input_args)
        compile_args["num_inference_steps"] = 2
        self._run_timed_pipe(compile_args)
        self._reset_scheduler(self.pipe)
        if self.refiner_pipe is not None:
            self._reset_scheduler(self.refiner_pipe)


@register_model("robbyant/lingbot-video-dense-1.3b")
@register_model("LingBot-Video-Dense")
class xFuserLingBotVideoDenseModel(xFuserLingBotVideoMoEModel):
    # The dense runner shares the composed loading and custom FSDP limitation.
    load_support = LoadSupport(
        meta_transformers=(),
        meta_text_encoders=(),
        replicated_meta=False,
        routes=LoadRoute.NONE,
    )

    settings = ModelSettings(
        model_name="robbyant/lingbot-video-dense-1.3b",
        output_name="lingbot_video_dense",
        model_output_type="video",
        fps=24,
        fp8_gemm_module_list=["transformer.blocks"],
        fp4_gemm_module_list=["transformer.blocks"],
        fsdp_strategy=LINGBOT_FSDP_STRATEGY,
    )

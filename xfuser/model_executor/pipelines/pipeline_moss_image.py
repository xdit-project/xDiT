"""Text-to-image pipeline for MOSS-Image.

Sampling follows the OpenMOSS reference: fp32 noise, fp32 CFG, and fp32 Euler
updates, with FLUX.2 latent unpacking before the VAE. CFG degree 2 runs the
positive and negative DiT passes on separate ranks. The optional third
skip-layer pass stays on the positive rank.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from xfuser.model_executor.models.transformers.transformer_moss_image import (
    MOSSImageScheduler,
    MOSSImageTransformer,
    validate_skip_layers,
)


def skip_window_active(step_index, num_steps, slg_start, slg_stop) -> bool:
    """True while ``slg_start <= step_index / num_steps < slg_stop``."""

    return slg_start <= step_index / num_steps < slg_stop


def combine_moss_guidance(conditional, uncond, guidance_scale, skipped, slg_scale, *, do_cfg, slg_active):
    """fp32 CFG, plus the optional positive-branch skip-layer correction."""

    prediction = conditional
    if do_cfg:
        prediction = uncond + guidance_scale * (conditional - uncond)
    if slg_active:
        if not do_cfg:
            prediction = skipped
        prediction = prediction + slg_scale * (conditional - skipped)
    return prediction


def _parallel_ready():
    from xfuser.core.distributed import model_parallel_is_initialized

    return model_parallel_is_initialized()


def _broadcast(group, tensor, src):
    return group.broadcast(tensor.contiguous(), src=src)


class xFuserMOSSImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["text_encoder", "tokenizer", "vae"]

    def __init__(self, transformer: MOSSImageTransformer, scheduler: MOSSImageScheduler, text_encoder=None, tokenizer=None, vae=None):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, torch_dtype=torch.bfloat16, vae_dtype=torch.float32, **kwargs):
        cache_dir = kwargs.get("cache_dir")
        transformer = MOSSImageTransformer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        ).eval()
        # Runtime Ulysses checks read num_attention_heads. This checkpoint publishes num_heads.
        transformer.register_to_config(num_attention_heads=transformer.config.num_heads)
        tokenizer = _load_tokenizer(pretrained_model_name_or_path, cache_dir)
        text_encoder = _load_text_encoder(pretrained_model_name_or_path, torch_dtype, cache_dir).eval()
        vae = _load_vae(pretrained_model_name_or_path, vae_dtype, cache_dir).eval()
        return cls(transformer, MOSSImageScheduler(), text_encoder, tokenizer, vae)

    @torch.no_grad()
    def encode_prompt(self, prompt, max_sequence_length=4096):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text encoder and tokenizer are required for string prompts.")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(value, str) for value in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")
        rows = []
        # One prompt at a time: Qwen3.5 linear attention changes if sequences are padded together.
        for value in prompts:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length)
            inputs = inputs.to(self._execution_device)
            output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
        for index, row in enumerate(rows):
            embeds[index, : row.shape[0]] = row
            mask[index, : row.shape[0]] = True
        return embeds, mask

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        *,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.0,
        negative_prompt="",
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        max_sequence_length=4096,
        output_type="pil",
        return_dict=True,
        uncond_skip_layers=(22,),
        slg_layers=None,
        slg_scale=0.0,
        slg_start=0.0,
        slg_stop=1.0,
        shift=None,
    ):
        if not math.isfinite(slg_scale) or slg_scale < 0:
            raise ValueError("slg_scale must be finite and nonnegative.")
        if not 0 <= slg_start <= slg_stop <= 1:
            raise ValueError("Require 0 <= slg_start <= slg_stop <= 1.")
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be pil, np, pt, or latent.")
        if num_images_per_prompt < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Invalid image count or guidance scale.")
        if not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096.")
        uncond_skip_layers = validate_skip_layers(
            uncond_skip_layers, len(self.transformer.layers), "uncond_skip_layers"
        )
        slg_layers = validate_skip_layers(slg_layers, len(self.transformer.layers), "slg_layers")
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt and prompt_embeds.")
        if shift is not None:
            _set_scheduler_shift(self.scheduler, shift)

        device = self._execution_device
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt, max_sequence_length)
        batch = prompt_embeds.shape[0]

        def prepare(embeds, mask):
            if embeds.ndim != 3 or embeds.shape[0] != batch:
                raise ValueError("Prompt embeddings must be [B, S, C] with matching batch size.")
            if mask is None:
                mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
            return (
                embeds.to(device).repeat_interleave(num_images_per_prompt, 0),
                mask.to(device).repeat_interleave(num_images_per_prompt, 0),
            )

        positive, positive_mask = _share_embeddings(*prepare(prompt_embeds, prompt_attention_mask))
        do_cfg = guidance_scale != 1.0
        negative = negative_mask = None
        if do_cfg:
            if negative_prompt_embeds is None:
                negatives = [negative_prompt] * batch if isinstance(negative_prompt, str) else negative_prompt
                if len(negatives) != batch:
                    raise ValueError("negative_prompt must match the prompt batch size.")
                negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                    negatives, max_sequence_length
                )
            negative, negative_mask = _share_embeddings(*prepare(negative_prompt_embeds, negative_prompt_attention_mask))

        shape = (batch * num_images_per_prompt, self.transformer.config.latent_channels, height // 16, width // 16)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        elif tuple(latents.shape) != shape:
            raise ValueError(f"latents must have shape {shape}.")
        else:
            latents = latents.to(device=device, dtype=torch.float32)
        latents = _share_latents(latents)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_steps = len(timesteps)
        cfg_world, cfg_rank, cfg_group = _cfg_layout()
        if cfg_world not in (1, 2):
            raise RuntimeError(f"MOSS-Image CFG parallelism requires a CFG degree of 1 or 2, got {cfg_world}.")
        split_cfg = do_cfg and cfg_world == 2

        autocast = torch.autocast(device_type=latents.device.type, enabled=False)
        with autocast:
            for step_index, timestep in enumerate(self.progress_bar(timesteps)):
                window = skip_window_active(step_index, num_steps, slg_start, slg_stop)
                slg_active = bool(slg_layers) and slg_scale > 0 and window
                uncond_skip = uncond_skip_layers if uncond_skip_layers and window else None
                if split_cfg:
                    prediction = _cfg_parallel_prediction(
                        self.transformer,
                        latents,
                        timestep,
                        positive,
                        positive_mask,
                        negative,
                        negative_mask,
                        cfg_rank,
                        cfg_group,
                        guidance_scale,
                        uncond_skip,
                        slg_layers if slg_active else None,
                        slg_scale,
                        slg_active,
                    )
                elif cfg_world == 2:
                    if cfg_rank == 1:
                        prediction = _positive_prediction(
                            self.transformer,
                            latents,
                            timestep,
                            positive,
                            positive_mask,
                            guidance_scale,
                            slg_layers if slg_active else None,
                            slg_scale,
                            slg_active,
                            do_cfg=False,
                        )
                    else:
                        prediction = torch.empty_like(latents)
                    prediction = _broadcast(cfg_group, prediction, src=1)
                else:
                    prediction = _serial_prediction(
                        self.transformer,
                        latents,
                        timestep,
                        positive,
                        positive_mask,
                        negative,
                        negative_mask,
                        guidance_scale,
                        do_cfg,
                        uncond_skip,
                        slg_layers if slg_active else None,
                        slg_scale,
                        slg_active,
                    )
                latents = self.scheduler.step(prediction, timestep, latents, return_dict=False)[0]

        if output_type == "latent":
            images = latents
        else:
            if self.vae is None:
                raise ValueError("A FLUX.2 VAE is required for decoded output.")
            images = self.vae.decode(_unpack_flux2_latents(latents, self.vae), return_dict=False)[0].clamp(-1, 1)
            images = self.image_processor.postprocess(images, output_type=output_type)
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=images) if return_dict else (images,)


def _serial_prediction(
    transformer,
    latents,
    timestep,
    positive,
    positive_mask,
    negative,
    negative_mask,
    guidance_scale,
    do_cfg,
    uncond_skip,
    slg_layers,
    slg_scale,
    slg_active,
):
    conditional = _forward(transformer, latents, timestep, positive, positive_mask)
    uncond = None
    if do_cfg:
        uncond = _forward(transformer, latents, timestep, negative, negative_mask, skip_layers=uncond_skip)
    skipped = None
    if slg_active:
        skipped = _forward(transformer, latents, timestep, positive, positive_mask, skip_layers=slg_layers)
    return combine_moss_guidance(
        conditional,
        uncond,
        guidance_scale,
        skipped,
        slg_scale,
        do_cfg=do_cfg,
        slg_active=slg_active,
    )


def _positive_prediction(
    transformer,
    latents,
    timestep,
    positive,
    positive_mask,
    guidance_scale,
    slg_layers,
    slg_scale,
    slg_active,
    do_cfg,
):
    conditional = _forward(transformer, latents, timestep, positive, positive_mask)
    skipped = None
    if slg_active:
        skipped = _forward(transformer, latents, timestep, positive, positive_mask, skip_layers=slg_layers)
    return combine_moss_guidance(
        conditional,
        None,
        guidance_scale,
        skipped,
        slg_scale,
        do_cfg=do_cfg,
        slg_active=slg_active,
    )


def _cfg_parallel_prediction(
    transformer,
    latents,
    timestep,
    positive,
    positive_mask,
    negative,
    negative_mask,
    cfg_rank,
    cfg_group,
    guidance_scale,
    uncond_skip,
    slg_layers,
    slg_scale,
    slg_active,
):
    skipped = None
    if cfg_rank == 1:
        local = _forward(transformer, latents, timestep, positive, positive_mask)
        if slg_active:
            skipped = _forward(transformer, latents, timestep, positive, positive_mask, skip_layers=slg_layers)
    else:
        local = _forward(transformer, latents, timestep, negative, negative_mask, skip_layers=uncond_skip)
    uncond, conditional = cfg_group.all_gather(local, separate_tensors=True)
    if slg_active:
        if cfg_rank != 1:
            skipped = torch.empty_like(conditional)
        skipped = _broadcast(cfg_group, skipped, src=1)
    return combine_moss_guidance(
        conditional,
        uncond,
        guidance_scale,
        skipped,
        slg_scale,
        do_cfg=True,
        slg_active=slg_active,
    )


def _forward(transformer, latents, timestep, embeds, mask, skip_layers=None):
    kwargs = {"skip_layers": skip_layers} if skip_layers else {}
    return transformer(latents, timestep, embeds, mask, **kwargs).sample.float()


def _cfg_layout():
    if not _parallel_ready():
        return 1, 0, None
    from xfuser.core.distributed import (
        get_cfg_group,
        get_classifier_free_guidance_rank,
        get_classifier_free_guidance_world_size,
    )

    return get_classifier_free_guidance_world_size(), get_classifier_free_guidance_rank(), get_cfg_group()


def _share_latents(latents):
    shared = _share_from_rank0(latents)
    return latents if shared is None else shared


def _share_embeddings(embeds, mask):
    shared_embeds = _share_from_rank0(embeds)
    shared_mask = _share_from_rank0(mask)
    if shared_embeds is None:
        return embeds, mask
    return shared_embeds, shared_mask


def _share_from_rank0(tensor):
    if tensor is None or not _parallel_ready():
        return None
    from xfuser.core.distributed import get_world_group

    group = get_world_group()
    if group.world_size == 1:
        return None
    return _broadcast(group, tensor, src=0)


def _set_scheduler_shift(scheduler, shift):
    shift = float(shift)
    if shift <= 0:
        raise ValueError("flow shift must be positive.")
    if float(scheduler.config.shift) == shift:
        return
    scheduler._internal_dict = FrozenDict({**dict(scheduler.config), "shift": shift})


def _unpack_flux2_latents(latents, vae):
    z = latents.to(dtype=vae.dtype)
    batch_norm = vae.bn
    eps = vae.config.batch_norm_eps
    scale = (batch_norm.running_var + eps).sqrt().to(z).view(1, -1, 1, 1)
    shift = batch_norm.running_mean.to(z).view(1, -1, 1, 1)
    z = z * scale + shift
    batch, channels, height, width = z.shape
    return z.reshape(batch, channels // 4, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3).reshape(
        batch, channels // 4, 2 * height, 2 * width
    )


def _load_tokenizer(repo, cache_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, subfolder="tokenizer", cache_dir=cache_dir)


def _load_text_encoder(repo, dtype, cache_dir):
    from transformers import AutoModel

    kwargs = dict(subfolder="text_encoder", cache_dir=cache_dir)
    try:
        return AutoModel.from_pretrained(repo, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModel.from_pretrained(repo, torch_dtype=dtype, **kwargs)


def _load_vae(repo, dtype, cache_dir):
    from diffusers import AutoencoderKLFlux2

    path = Path(repo)
    subfolder = None if (path / "config.json").is_file() else "vae"
    try:
        return AutoencoderKLFlux2.from_pretrained(repo, subfolder=subfolder, torch_dtype=dtype, cache_dir=cache_dir)
    except ImportError as exc:
        raise ImportError(
            "MOSS-Image decodes with AutoencoderKLFlux2, which needs diffusers>=0.38.0."
        ) from exc

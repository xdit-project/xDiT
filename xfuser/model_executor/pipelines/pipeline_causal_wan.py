from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np

from diffusers import WanImageToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_torch_xla_available, logging
import torch


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


class xFuserCausalWanPipeline(WanImageToVideoPipeline):

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
        guidance_scale_2=None,
    ):
        if guidance_scale is not None and guidance_scale > 0:
            raise ValueError(
                "CFG is not supported for causal WAN. Please set guidance_scale to 0."
            )
        if image is not None and image_embeds is not None:
            raise ValueError(
                "Cannot forward both `image` and `image_embeds`. Please provide only one."
            )
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please provide only one.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`. Please provide only one.")

    def _compute_block_sizes(self, num_latent_frames: int, num_frames_per_block: int, boundary_timestep: float) -> List[int]:
        """
        Compute the sizes of the blocks to process.
        """
        if num_latent_frames % num_frames_per_block != 0:
            raise ValueError(
                f"num_latent_frames ({num_latent_frames}) must be divisible by "
                f"num_frames_per_block ({num_frames_per_block})"
            )
        block_sizes = [num_frames_per_block] * (num_latent_frames // num_frames_per_block)
        # First block is just 1 frame
        if boundary_timestep is not None:
            block_sizes[0] = 1
        return block_sizes

    def _compute_dmd_timesteps(self, dmd_denoising_steps: List[int], boundary_timestep: float, flow_shift: float, device: torch.device) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Compute DMD timesteps, number of high-noise steps, and sigma boundary.
        """
        # Apply flow_shift to scheduler before computing sigma schedule
        if flow_shift is not None:
            self.scheduler._shift = flow_shift

        # Set up scheduler with dense 1000-step schedule for sigma lookups
        self.scheduler.set_timesteps(1000, device=device)

        # Build shifted timestep schedule matching FastVideo's scheduler init
        # (linspace [1..1000] with shift applied), used for warping DMD steps.
        num_train = self.scheduler.config.num_train_timesteps
        shift = flow_shift if flow_shift is not None else 1.0
        ts = np.linspace(1, num_train, num_train, dtype=np.float32)[::-1].copy()
        sigmas = ts / num_train
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        init_timesteps = torch.from_numpy((sigmas * num_train).astype(np.float32))

        # Warp DMD timesteps through the shifted schedule
        raw_dmd_timesteps = torch.tensor(dmd_denoising_steps, dtype=torch.long)
        scheduler_timesteps = torch.cat((
            init_timesteps,
            torch.tensor([0], dtype=torch.float32),
        ))
        dmd_timesteps = scheduler_timesteps[1000 - raw_dmd_timesteps].to(device)

        # Precompute which DMD timesteps are in the high-noise regime
        num_high_noise_steps = int((dmd_timesteps >= boundary_timestep).sum().item())
        # Precompute sigma for boundary
        sigma_boundary = self._get_sigma_for_timestep(
            self.scheduler,
            torch.tensor([boundary_timestep], dtype=torch.long, device=device),
        )

        return dmd_timesteps, num_high_noise_steps, sigma_boundary

    def _build_causal_attn_args(self, current_kv_cache, crossattn_cache, start_idx, frame_seq_length, local_attn_size, sink_size, max_attention_size, attention_kwargs):
        """
        Build causal attention argument dictionary.
        """
        return {
            **(attention_kwargs or {}),
            "kv_cache": current_kv_cache,
            "crossattn_cache": crossattn_cache,
            "current_start": start_idx * frame_seq_length,
            "start_frame": start_idx,
            "local_attn_size": local_attn_size,
            "sink_size": sink_size,
            "max_attention_size": max_attention_size,
        }

    def _initialize_kv_cache(
        self,
        batch_size: int,
        num_blocks: int,
        num_heads: int,
        head_dim: int,
        kv_cache_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list:
        """Initialize per-layer KV caches for causal self-attention."""
        kv_cache = []
        for _ in range(num_blocks):
            kv_cache.append({
                "k": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        return kv_cache

    def _initialize_crossattn_cache(
        self,
        batch_size: int,
        max_text_len: int,
        num_blocks: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list:
        """Initialize per-layer cross-attention caches."""
        crossattn_cache = []
        for _ in range(num_blocks):
            crossattn_cache.append({
                "k": torch.zeros(
                    [batch_size, max_text_len, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, max_text_len, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "is_init": False,
            })
        return crossattn_cache

    def _get_sigma_for_timestep(self, scheduler, timestep):
        """Look up sigma from scheduler's dense schedule. Uses float64 for precision."""
        device = timestep.device
        sigmas = scheduler.sigmas.double().to(device)
        timesteps = scheduler.timesteps.double().to(device)
        timestep = timestep.double()
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        return sigmas[timestep_id]

    def _pred_noise_to_pred_video(self, noise_pred, noisy, sigma):
        """Denoise: pred_video = noisy - sigma * noise_pred (standard flow matching)."""
        dtype = noise_pred.dtype
        noise_pred = noise_pred.double()
        noisy = noisy.double()
        sigma = sigma.reshape(-1, 1, 1, 1).double()
        pred_video = noisy - sigma * noise_pred
        return pred_video.to(dtype)

    def _pred_noise_to_x_bound(self, noise_pred, noisy, sigma, sigma_boundary):
        """Bounded denoise for high-noise MoE transformer.
        pred_video = noisy - (sigma - sigma_boundary) * noise_pred
        """
        dtype = noise_pred.dtype
        noise_pred = noise_pred.double()
        noisy = noisy.double()
        sigma = sigma.reshape(-1, 1, 1, 1).double()
        sigma_boundary = sigma_boundary.reshape(-1, 1, 1, 1).double()
        pred_video = noisy - (sigma - sigma_boundary) * noise_pred
        return pred_video.to(dtype)

    def _add_noise(self, clean, noise, sigma):
        """Standard flow matching forward process: sample = (1 - sigma) * clean + sigma * noise."""
        sigma = sigma.reshape(-1, 1, 1, 1)
        return ((1 - sigma) * clean + sigma * noise).to(clean.dtype)

    def _add_noise_high(self, clean, noise, sigma, sigma_boundary):
        """Bounded re-noise for high-noise MoE regime.
        alpha = (1 - sigma) / (1 - sigma_boundary)
        beta = sqrt(sigma^2 - (alpha * sigma_boundary)^2)
        sample = alpha * clean + beta * noise
        """
        sigma = sigma.reshape(-1, 1, 1, 1).double()
        sigma_boundary = sigma_boundary.reshape(-1, 1, 1, 1).double()
        alpha = (1 - sigma) / (1 - sigma_boundary)
        beta = torch.sqrt(sigma ** 2 - (alpha * sigma_boundary) ** 2)
        return (alpha * clean.double() + beta * noise.double()).to(clean.dtype)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        num_frames_per_block: int = 1,
        sliding_window_num_frames: int = 21,
        context_noise: int = 0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        max_attention_size: int = 32760,
        dmd_denoising_steps: Optional[List[int]] = None,
        flow_shift: Optional[float] = None,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 3b. Encode image embeddings (CLIP) for i2v
        if image is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                image_embeds = self.encode_image(image, device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare latent variables

        num_channels_latents = self.vae.config.z_dim
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 5. Compute frame sequence length for cache sizing
        p_t, p_h, p_w = self.transformer.config.patch_size
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        frame_seq_length = (height // (self.vae_scale_factor_spatial * p_h)) * (width // (self.vae_scale_factor_spatial * p_w))

        # 6. Initialize caches
        num_transformer_blocks = len(self.transformer.blocks)
        num_heads = self.transformer.config.num_attention_heads
        head_dim = self.transformer.config.attention_head_dim

        kv_cache_size = frame_seq_length * sliding_window_num_frames

        kv_cache = self._initialize_kv_cache(
            batch_size=batch_size,
            num_blocks=num_transformer_blocks,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_cache_size=kv_cache_size,
            dtype=transformer_dtype,
            device=device,
        )
        kv_cache_2 = self._initialize_kv_cache(
                batch_size=batch_size,
                num_blocks=num_transformer_blocks,
                num_heads=num_heads,
                head_dim=head_dim,
                kv_cache_size=kv_cache_size,
                dtype=transformer_dtype,
                device=device,
        )
        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=batch_size,
            max_text_len=max_sequence_length,
            num_blocks=num_transformer_blocks,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=transformer_dtype,
            device=device,
        )

        boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps

        # 7. Setup blocks and DMD
        block_sizes = self._compute_block_sizes(num_latent_frames, num_frames_per_block, boundary_timestep)
        dmd_timesteps, num_high_noise_steps, sigma_boundary = self._compute_dmd_timesteps(dmd_denoising_steps, boundary_timestep, flow_shift, device)

        # 7b. First-frame KV cache seeding for i2v
        start_idx = 0
        if image is not None:
            preprocessed = self.video_processor.preprocess(image, height=height, width=width)
            preprocessed = preprocessed.to(device, dtype=torch.float32)
            # VAE encode: add temporal dim -> encode -> extract mean
            first_frame_latent = self.vae.encode(
                preprocessed.unsqueeze(2).to(self.vae.dtype)
            ).latent_dist.mean.float()
            # Normalize using the same latents_mean/latents_std as the decode path
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(first_frame_latent.device, first_frame_latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(first_frame_latent.device, first_frame_latent.dtype)
            )
            first_frame_latent = (first_frame_latent - latents_mean) / latents_std

            # Seed KV caches by running first frame through both transformers at t=0
            t_zero = torch.zeros([batch_size], device=device, dtype=torch.long)
            seed_attn_args = self._build_causal_attn_args(
                kv_cache, crossattn_cache, 0, frame_seq_length,
                local_attn_size, sink_size, max_attention_size, attention_kwargs,
            )
            with self.transformer.cache_context("cond"):
                self.transformer(
                    hidden_states=first_frame_latent.to(transformer_dtype),
                    timestep=t_zero,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=seed_attn_args,
                    return_dict=False,
                )
            seed_attn_args_2 = self._build_causal_attn_args(
                kv_cache_2, crossattn_cache, 0, frame_seq_length,
                local_attn_size, sink_size, max_attention_size, attention_kwargs,
            )
            with self.transformer_2.cache_context("cond"):
                self.transformer_2(
                    hidden_states=first_frame_latent.to(transformer_dtype),
                    timestep=t_zero,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=seed_attn_args_2,
                    return_dict=False,
                )

            # Write first frame into latents and advance past it
            latents[:, :, :1] = first_frame_latent
            start_idx = 1
            block_sizes.pop(0)

        total_steps = len(block_sizes) * len(dmd_timesteps)
        with self.progress_bar(total=total_steps) as progress_bar:
            for block_idx, block_size in enumerate(block_sizes):
                end_idx = start_idx + block_size

                # Extract block latent slice
                block_latents = latents[:, :, start_idx:end_idx].clone()

                # Work in BTCHW for DMD math, BCTHW for model
                noise_latents_btchw = block_latents.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
                raw_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(dmd_timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t_cur

                    if t_cur >= boundary_timestep:
                        current_model = self.transformer
                        current_guidance_scale = guidance_scale
                        current_kv_cache = kv_cache
                    else:
                        current_model = self.transformer_2
                        current_guidance_scale = guidance_scale_2
                        current_kv_cache = kv_cache_2

                    # Save noisy state for DMD conversion (flat: [B*T, C, H, W])
                    noise_latents_flat = noise_latents_btchw.flatten(0, 1).clone()

                    # Build model input in BCTHW format
                    current_latents_bcthw = noise_latents_btchw.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
                    latent_model_input = current_latents_bcthw.to(transformer_dtype)
                    timestep = t_cur.expand(block_latents.shape[0]) if isinstance(t_cur, torch.Tensor) else torch.tensor([t_cur], device=device).expand(block_latents.shape[0])

                    attention_args = self._build_causal_attn_args(
                        current_kv_cache,
                        crossattn_cache,
                        start_idx,
                        frame_seq_length,
                        local_attn_size,
                        sink_size,
                        max_attention_size,
                        attention_kwargs,
                    )

                    with current_model.cache_context("cond"):
                        noise_pred = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_args,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_image=image_embeds,
                                attention_kwargs=attention_args,
                                return_dict=False,
                            )[0]
                            noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                    # Convert noise_pred from BCTHW to BTCHW then flatten
                    noise_pred_btchw = noise_pred.permute(0, 2, 1, 3, 4)
                    noise_pred_flat = noise_pred_btchw.flatten(0, 1)

                    # Get sigma for current timestep
                    sigma_cur = self._get_sigma_for_timestep(self.scheduler, t_cur.unsqueeze(0))
                    sigma_cur_expanded = sigma_cur.expand(noise_pred_flat.shape[0])

                    # Denoise: convert noise prediction to clean video prediction
                    if t_cur >= boundary_timestep:
                        sigma_bound_expanded = sigma_boundary.expand(noise_pred_flat.shape[0])
                        pred_video_flat = self._pred_noise_to_x_bound(
                            noise_pred_flat, noise_latents_flat,
                            sigma_cur_expanded, sigma_bound_expanded,
                        )
                    else:
                        pred_video_flat = self._pred_noise_to_pred_video(
                            noise_pred_flat, noise_latents_flat, sigma_cur_expanded,
                        )

                    pred_video_btchw = pred_video_flat.unflatten(0, raw_shape[:2])

                    # Re-noise or finalize
                    if i < len(dmd_timesteps) - 1:
                        next_t = dmd_timesteps[i + 1]
                        sigma_next = self._get_sigma_for_timestep(
                            self.scheduler, next_t.unsqueeze(0)
                        )
                        sigma_next_expanded = sigma_next.expand(noise_pred_flat.shape[0])

                        noise = randn_tensor(raw_shape, generator=generator, device=device, dtype=pred_video_btchw.dtype)
                        noise_flat = noise.flatten(0, 1)

                        if i < num_high_noise_steps - 1:
                            # Still in high-noise regime -> bounded re-noise
                            sigma_bound_expanded = sigma_boundary.expand(noise_pred_flat.shape[0])
                            noise_latents_btchw = self._add_noise_high(
                                pred_video_btchw.flatten(0, 1), noise_flat,
                                sigma_next_expanded, sigma_bound_expanded,
                            ).unflatten(0, raw_shape[:2])
                        elif i == num_high_noise_steps - 1:
                            # Transitioning past boundary -> use clean prediction
                            noise_latents_btchw = pred_video_btchw
                        else:
                            # Standard re-noise
                            noise_latents_btchw = self._add_noise(
                                pred_video_btchw.flatten(0, 1), noise_flat,
                                sigma_next_expanded,
                            ).unflatten(0, raw_shape[:2])
                    else:
                        # Last step: use clean prediction
                        noise_latents_btchw = pred_video_btchw

                    progress_bar.update()

                # Convert final result back to BCTHW
                block_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)

                # Write denoised block back
                latents[:, :, start_idx:end_idx] = block_latents

                t_context = torch.zeros([batch_size], device=device, dtype=torch.long) + context_noise

                context_input_model = block_latents.to(transformer_dtype)
                context_timestep = t_context

                # Update KV caches with clean context
                attention_args = self._build_causal_attn_args(kv_cache, crossattn_cache, start_idx, frame_seq_length, local_attn_size, sink_size, max_attention_size, attention_kwargs)
                with self.transformer.cache_context("cond"):
                    self.transformer(
                        hidden_states=context_input_model,
                        timestep=context_timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_args,
                        return_dict=False,
                    )

                attention_args = self._build_causal_attn_args(kv_cache_2, crossattn_cache, start_idx, frame_seq_length, local_attn_size, sink_size, max_attention_size, attention_kwargs)
                with self.transformer_2.cache_context("cond"):
                    self.transformer_2(
                        hidden_states=context_input_model,
                        timestep=context_timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_args,
                        return_dict=False,
                    )

                start_idx += block_size

        self._current_timestep = None

        # Remove trailing unprocessed frames from first-block shortening
        if boundary_timestep is not None:
            num_frames_to_remove = num_frames_per_block - 1
            if num_frames_to_remove > 0:
                latents = latents[:, :, :-num_frames_to_remove, :, :]

        # VAE
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
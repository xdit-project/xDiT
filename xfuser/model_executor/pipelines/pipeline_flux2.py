"""xFuser pipeline wrapper for Flux2 with PipeFusion support.

Modeled on pipeline_flux.py. The denoising loop is replaced by the sync/async
patch-level pipeline (_sync_pipeline / _async_pipeline) inherited from the
Flux1 implementation, adapted to Flux2's transformer signature (guidance
embedding instead of pooled projections, modulation parameters).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import Flux2Pipeline, Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2PipelineOutput
from diffusers.utils import is_torch_xla_available

from xfuser.config import EngineConfig, InputConfig
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_runtime_state,
    get_pp_group,
    get_sp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    is_dp_last_group,
)
from xfuser.logger import init_logger
from .base_pipeline import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister

logger = init_logger(__name__)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class xFuserFlux2PipelineBase(xFuserPipelineBaseWrapper):
    """Shared PipeFusion logic for Flux2 pipelines. Subclasses bind a concrete
    diffusers pipeline class via ``_diffusers_cls`` and ``from_pretrained``."""

    _diffusers_cls = Flux2Pipeline

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        engine_config,
        cache_args: Dict = {},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = cls._diffusers_cls.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config, cache_args)

    def prepare_run(
        self,
        input_config,
        steps: int = 3,
        sync_steps: int = 1,
    ):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            num_inference_steps=steps,
            max_sequence_length=input_config.max_sequence_length,
            generator=torch.Generator(device="cuda").manual_seed(42),
            output_type=input_config.output_type,
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        image=None,
        prompt=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[list] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[dict] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple = (10, 20, 30),
        caption_upsample_temperature: float = None,
        **kwargs,
    ):
        # 1. Check inputs
        # klein's check_inputs takes an extra `guidance_scale` arg; dev's does not.
        # Conditionally pass it so the wrapper works for both pipeline variants.
        import inspect

        _check_inputs_kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        if "guidance_scale" in inspect.signature(self.check_inputs).parameters:
            _check_inputs_kwargs["guidance_scale"] = guidance_scale
        self.check_inputs(**_check_inputs_kwargs)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # * set runtime state input parameters (PipeFusion bookkeeping)
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        # 3. text embeddings
        if caption_upsample_temperature:
            prompt = self.upsample_prompt(
                prompt,
                images=image,
                temperature=caption_upsample_temperature,
                device=device,
            )
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        # 4. prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 5. timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1]
        try:
            from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

            mu = compute_empirical_mu(
                image_seq_len=image_seq_len, num_steps=num_inference_steps
            )
        except Exception:
            mu = None
        from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # guidance embedding
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 6. Denoising loop -> sync/async patch pipeline
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if (
                get_pipeline_parallel_world_size() > 1
                and len(timesteps) > num_pipeline_warmup_steps
            ):
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_ids,
                    guidance=guidance,
                    timesteps=timesteps[:num_pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
                latents = self._async_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_ids,
                    guidance=guidance,
                    timesteps=timesteps[num_pipeline_warmup_steps:],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_ids,
                    guidance=guidance,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    sync_only=True,
                )

        # 7. output (only on the last dp group)
        image_out = None
        if is_dp_last_group():
            if output_type == "latent":
                image_out = latents
            else:
                latents = self._unpack_latents_with_ids(latents, latent_ids)
                latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents_bn_std = torch.sqrt(
                    self.vae.bn.running_var.view(1, -1, 1, 1)
                    + self.vae.config.batch_norm_eps
                ).to(latents.device, latents.dtype)
                latents = latents * latents_bn_std + latents_bn_mean
                latents = self._unpatchify_latents(latents)
                image_out = self.vae.decode(latents, return_dict=False)[0]
                image_out = self.image_processor.postprocess(
                    image_out, output_type=output_type
                )

            self.maybe_free_model_hooks()
            if not return_dict:
                return (image_out,)
            return Flux2PipelineOutput(images=image_out)
        else:
            return None

    # ---- patch-level pipeline drivers (adapted from pipeline_flux.py) ----

    def _init_sync_pipeline(self, latents, latent_image_ids, prompt_embeds, text_ids):
        get_runtime_state().set_patched_mode(patch_mode=False)

        latents_list = [
            latents[:, start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        ]
        latents = torch.cat(latents_list, dim=-2)
        latent_image_ids_list = [
            latent_image_ids[..., start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        ]
        latent_image_ids = torch.cat(latent_image_ids_list, dim=-2)

        if get_runtime_state().split_text_embed_in_sp:
            if prompt_embeds.shape[-2] % get_sequence_parallel_world_size() == 0:
                prompt_embeds = torch.chunk(
                    prompt_embeds, get_sequence_parallel_world_size(), dim=-2
                )[get_sequence_parallel_rank()]
            else:
                get_runtime_state().split_text_embed_in_sp = False
            if text_ids.shape[-2] % get_sequence_parallel_world_size() == 0:
                text_ids = torch.chunk(
                    text_ids, get_sequence_parallel_world_size(), dim=-2
                )[get_sequence_parallel_rank()]
            else:
                get_runtime_state().split_text_embed_in_sp = False
        return latents, latent_image_ids, prompt_embeds, text_ids

    def _sync_pipeline(
        self,
        latents,
        prompt_embeds,
        text_ids,
        latent_image_ids,
        guidance,
        timesteps,
        num_warmup_steps,
        progress_bar,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
        sync_only=False,
    ):
        latents, latent_image_ids, prompt_embeds, text_ids = self._init_sync_pipeline(
            latents, latent_image_ids, prompt_embeds, text_ids
        )
        for i, t in enumerate(timesteps):
            if getattr(self, "_interrupt", False):
                continue
            if is_pipeline_last_stage():
                last_timestep_latents = latents

            if get_pipeline_parallel_world_size() == 1:
                pass
            elif is_pipeline_first_stage() and i == 0:
                pass
            else:
                latents = get_pp_group().pipeline_recv()
                if not is_pipeline_first_stage():
                    encoder_hidden_state = get_pp_group().pipeline_recv(
                        0, "encoder_hidden_state"
                    )

            latents, encoder_hidden_state = self._backbone_forward(
                latents=latents,
                encoder_hidden_states=(
                    prompt_embeds if is_pipeline_first_stage() else encoder_hidden_state
                ),
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                guidance=guidance,
                t=t,
            )

            if is_pipeline_last_stage():
                latents_dtype = latents.dtype
                latents = self._scheduler_step(latents, last_timestep_latents, t)
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                if callback_on_step_end is not None:
                    callback_kwargs = {
                        k: locals()[k] for k in callback_on_step_end_tensor_inputs
                    }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
            if XLA_AVAILABLE:
                pass

            if sync_only and is_pipeline_last_stage() and i == len(timesteps) - 1:
                pass
            elif get_pipeline_parallel_world_size() > 1:
                get_pp_group().pipeline_send(latents)
                if not is_pipeline_last_stage():
                    get_pp_group().pipeline_send(
                        encoder_hidden_state, name="encoder_hidden_state"
                    )

        if (
            sync_only
            and get_sequence_parallel_world_size() > 1
            and is_pipeline_last_stage()
        ):
            sp_degree = get_sequence_parallel_world_size()
            sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True)
            latents_list = []
            for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                latents_list += [
                    sp_latents_list[sp_patch_idx][
                        :,
                        get_runtime_state()
                        .pp_patches_token_start_idx_local[
                            pp_patch_idx
                        ] : get_runtime_state()
                        .pp_patches_token_start_idx_local[pp_patch_idx + 1],
                        :,
                    ]
                    for sp_patch_idx in range(sp_degree)
                ]
            latents = torch.cat(latents_list, dim=-2)
        return latents

    def _init_async_pipeline(
        self, num_timesteps, latents, num_pipeline_warmup_steps, latent_image_ids
    ):
        get_runtime_state().set_patched_mode(patch_mode=True)

        if is_pipeline_first_stage():
            latents = (
                get_pp_group().pipeline_recv()
                if num_pipeline_warmup_steps > 0
                else latents
            )
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
        elif is_pipeline_last_stage():
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
        else:
            patch_latents = [
                None for _ in range(get_runtime_state().num_pipeline_patch)
            ]

        patch_latent_image_ids = list(
            latent_image_ids[..., start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        )

        recv_timesteps = (
            num_timesteps - 1 if is_pipeline_first_stage() else num_timesteps
        )
        if is_pipeline_first_stage():
            for _ in range(recv_timesteps):
                for patch_idx in range(get_runtime_state().num_pipeline_patch):
                    get_pp_group().add_pipeline_recv_task(patch_idx)
        else:
            for _ in range(recv_timesteps):
                get_pp_group().add_pipeline_recv_task(0, "encoder_hidden_states")
                for patch_idx in range(get_runtime_state().num_pipeline_patch):
                    get_pp_group().add_pipeline_recv_task(patch_idx)
        return patch_latents, patch_latent_image_ids

    def _async_pipeline(
        self,
        latents,
        prompt_embeds,
        text_ids,
        latent_image_ids,
        guidance,
        timesteps,
        num_warmup_steps,
        progress_bar,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
    ):
        if len(timesteps) == 0:
            return latents
        num_pipeline_patch = get_runtime_state().num_pipeline_patch
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        patch_latents, patch_latent_image_ids = self._init_async_pipeline(
            num_timesteps=len(timesteps),
            latents=latents,
            num_pipeline_warmup_steps=num_pipeline_warmup_steps,
            latent_image_ids=latent_image_ids,
        )
        last_patch_latents = (
            [None for _ in range(num_pipeline_patch)]
            if is_pipeline_last_stage()
            else None
        )

        first_async_recv = True
        for i, t in enumerate(timesteps):
            if getattr(self, "_interrupt", False):
                continue
            for patch_idx in range(num_pipeline_patch):
                if is_pipeline_last_stage():
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if first_async_recv:
                        if not is_pipeline_first_stage() and patch_idx == 0:
                            get_pp_group().recv_next()
                        get_pp_group().recv_next()
                        first_async_recv = False
                    if not is_pipeline_first_stage() and patch_idx == 0:
                        last_encoder_hidden_states = (
                            get_pp_group().get_pipeline_recv_data(
                                idx=patch_idx, name="encoder_hidden_states"
                            )
                        )
                    patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(
                        idx=patch_idx
                    )

                patch_latents[patch_idx], next_encoder_hidden_states = (
                    self._backbone_forward(
                        latents=patch_latents[patch_idx],
                        encoder_hidden_states=(
                            prompt_embeds
                            if is_pipeline_first_stage()
                            else last_encoder_hidden_states
                        ),
                        text_ids=text_ids,
                        latent_image_ids=patch_latent_image_ids[patch_idx],
                        guidance=guidance,
                        t=t,
                    )
                )
                if is_pipeline_last_stage():
                    latents_dtype = patch_latents[patch_idx].dtype
                    patch_latents[patch_idx] = self._scheduler_step(
                        patch_latents[patch_idx], last_patch_latents[patch_idx], t
                    )
                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            latents = latents.to(latents_dtype)
                    if i != len(timesteps) - 1:
                        get_pp_group().pipeline_isend(
                            patch_latents[patch_idx], segment_idx=patch_idx
                        )
                else:
                    if patch_idx == 0:
                        get_pp_group().pipeline_isend(
                            next_encoder_hidden_states, name="encoder_hidden_states"
                        )
                    get_pp_group().pipeline_isend(
                        patch_latents[patch_idx], segment_idx=patch_idx
                    )

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                        pass
                    elif is_pipeline_first_stage():
                        get_pp_group().recv_next()
                    else:
                        if patch_idx == num_pipeline_patch - 1:
                            get_pp_group().recv_next()
                        get_pp_group().recv_next()

                get_runtime_state().next_patch()

            if i == len(timesteps) - 1 or (
                (i + num_pipeline_warmup_steps + 1) > num_warmup_steps
                and (i + num_pipeline_warmup_steps + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

        latents = None
        if is_pipeline_last_stage():
            latents = torch.cat(patch_latents, dim=-2)
            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = get_sp_group().all_gather(
                    latents, separate_tensors=True
                )
                latents_list = []
                for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                    latents_list += [
                        sp_latents_list[sp_patch_idx][
                            ...,
                            get_runtime_state()
                            .pp_patches_token_start_idx_local[
                                pp_patch_idx
                            ] : get_runtime_state()
                            .pp_patches_token_start_idx_local[pp_patch_idx + 1],
                            :,
                        ]
                        for sp_patch_idx in range(sp_degree)
                    ]
                latents = torch.cat(latents_list, dim=-2)
        return latents

    def _backbone_forward(
        self, latents, encoder_hidden_states, text_ids, latent_image_ids, guidance, t
    ):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        ret = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=getattr(self, "_attention_kwargs", None),
            return_dict=False,
        )[0]
        if self.engine_config.parallel_config.dit_parallel_size > 1:
            noise_pred, encoder_hidden_states = ret
        else:
            noise_pred, encoder_hidden_states = ret, None
        return noise_pred, encoder_hidden_states

    def _scheduler_step(self, noise_pred, latents, t):
        return self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]


@xFuserPipelineWrapperRegister.register(Flux2Pipeline)
class xFuserFlux2Pipeline(xFuserFlux2PipelineBase):
    _diffusers_cls = Flux2Pipeline


@xFuserPipelineWrapperRegister.register(Flux2KleinPipeline)
class xFuserFlux2KleinPipeline(xFuserFlux2PipelineBase):
    _diffusers_cls = Flux2KleinPipeline

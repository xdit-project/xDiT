import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput


from xfuser.model_executor.layers.usp import USP
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
    local_sp_pad_rows,
    zero_sp_pad_rows_hook,
)
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.model_executor.layers.fused_qk_rope_zimage_flydsl import (
    flydsl_fused_qk_norm_rope,
)

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_runtime_state,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.model_executor.models.transformers.base_transformer import xFuserTransformerBaseWrapper
from xfuser.model_executor.models.transformers.register import xFuserTransformerWrappersRegister

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


def _scatter_pad_token(x: torch.Tensor, mask: torch.Tensor, pad_token: torch.Tensor) -> torch.Tensor:
    """``x[mask] = pad_token`` without the device->host sync.

    Boolean-mask ``index_put_`` lowers to ``nonzero()`` on CUDA, which must read the
    match count back to the host to size its output -- a full blocking sync on the
    hot path, twice per denoise step (x tokens and caption tokens).

    ``torch.where`` computes exactly the same result as a data-independent
    elementwise select: no ``nonzero``, no host round-trip, no dynamic shape (which
    also removes a torch.compile graph break).  Rows selected by ``mask`` get
    ``pad_token`` broadcast across the feature dim, all other rows are copied
    verbatim, so the produced tensor is bit-identical to the in-place form.

    Ungated and selected purely from tensor properties; anything that does not
    match the expected 2-D (tokens, dim) / 1-D bool-mask layout falls through to
    the original in-place reference path.
    """
    if (
        x.dim() == 2
        and mask.dim() == 1
        and mask.dtype == torch.bool
        and mask.shape[0] == x.shape[0]
        and pad_token.dim() == 2
        and pad_token.shape[0] == 1
        and pad_token.shape[1] == x.shape[1]
    ):
        return torch.where(mask.unsqueeze(-1), pad_token.to(x.dtype), x)
    # _reference fallback: unchanged upstream behaviour.
    x[mask] = pad_token
    return x


class xFuserZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    def __init__(self, attention_kwargs: Optional[dict] = None):
        self.attention_kwargs = attention_kwargs

    @staticmethod
    def _pad_heads(x: torch.Tensor, pad_heads: int) -> torch.Tensor:
        if pad_heads <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[1] = pad_heads
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms + RoPE.
        #
        # The wrapper selects from tensor properties alone and falls back to the
        # unfused diffusers path (norm then complex rope) whenever the FlyDSL
        # stack is absent or the shape is out of envelope, so this call is
        # numerically interchangeable with the code it replaced.
        query, key = flydsl_fused_qk_norm_rope(query, key, attn.norm_q, attn.norm_k, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Transpose for attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        ulysses_world_size = get_ulysses_parallel_world_size()
        pad_heads = 0
        if ulysses_world_size > 1:
            pad_heads = (ulysses_world_size - (query.shape[1] % ulysses_world_size)) % ulysses_world_size
            if pad_heads:
                query = self._pad_heads(query, pad_heads)
                key = self._pad_heads(key, pad_heads)
                value = self._pad_heads(value, pad_heads)

        hidden_states = USP(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attn_layer=attn,
            attention_kwargs=self.attention_kwargs,
        )

        if pad_heads:
            hidden_states = hidden_states[:, :-pad_heads, :, :]

        # Transpose back to original shape
        hidden_states = hidden_states.transpose(1, 2)

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


def z_image_attn_modules(transformer) -> list[torch.nn.Module]:
    """Return every Z-Image attention module that executes USP."""
    return [
        layer.attention
        for layer in (
            *transformer.noise_refiner,
            *transformer.context_refiner,
            *transformer.layers,
        )
    ]


class xFuserZImageTransformer2DWrapper(ZImageTransformer2DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._self_attn_kwargs = {"valid_kv_len": None}
        self._sp_pad = {"local_rows": 0}
        for layer in self.layers + self.context_refiner + self.noise_refiner:
            layer.attention.processor = xFuserZSingleStreamAttnProcessor(attention_kwargs=self._self_attn_kwargs)
            layer.register_forward_hook(zero_sp_pad_rows_hook(self._sp_pad))
        register_fp8_comms_eligible_modules(self, z_image_attn_modules(self))

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size
        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        cfg_world_size = get_classifier_free_guidance_world_size()
        cfg_rank = get_classifier_free_guidance_rank()
        do_cfg_parallel = cfg_world_size > 1

        if do_cfg_parallel:
            B = len(x) // cfg_world_size
            x = x[cfg_rank * B : (cfg_rank + 1) * B]
            cap_feats = cap_feats[cfg_rank * B : (cfg_rank + 1) * B]
            t = t.chunk(cfg_world_size, dim=0)[cfg_rank]

        bsz = len(x)
        device = x[0].device
        # The pipeline keeps the timestep schedule host-resident so that its
        # per-step ``t_norm = timestep[0].item()`` is not a blocking D2H sync
        # (see xfuser/model_executor/models/runner_models/z_image.py::
        # _keep_timesteps_host_resident).  Pay for that with one cheap
        # non-blocking H2D copy of the (B,) timestep vector instead of a full
        # queue drain per denoise step.  Selected purely from tensor properties;
        # if ``t`` is already on-device this is a no-op and the original
        # behaviour is preserved byte for byte.
        if isinstance(t, torch.Tensor) and t.device != device:
            t = t.to(device, non_blocking=True)
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        # assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x = _scatter_pad_token(x, torch.cat(x_inner_pad_mask), self.x_pad_token)
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # SP support
        seq_len = x.shape[1]
        pad_amount = (sp_world_size - (seq_len % sp_world_size)) % sp_world_size
        self._self_attn_kwargs["valid_kv_len"] = seq_len if pad_amount else None
        x = chunk_and_pad_sequence(x, sp_world_rank, sp_world_size, pad_amount, dim=-2)
        self._sp_pad["local_rows"] = local_sp_pad_rows(seq_len, x.shape[1], sp_world_rank)
        x_attn_mask = chunk_and_pad_sequence(x_attn_mask, sp_world_rank, sp_world_size, pad_amount, dim=-1)
        x_freqs_cis_chunked = chunk_and_pad_sequence(x_freqs_cis, sp_world_rank, sp_world_size, pad_amount, dim=-2)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                x = self._gradient_checkpointing_func(layer, x, x_attn_mask, x_freqs_cis_chunked, adaln_input)
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis_chunked, adaln_input)

        # Gather SP outputs and remove padding
        x = gather_and_unpad(x, pad_amount, dim=-2)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        # assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats = _scatter_pad_token(cap_feats, torch.cat(cap_inner_pad_mask), self.cap_pad_token)
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)
        )

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        # SP support
        seq_len = cap_feats.shape[1]
        pad_amount = (sp_world_size - (seq_len % sp_world_size)) % sp_world_size
        self._self_attn_kwargs["valid_kv_len"] = seq_len if pad_amount else None
        cap_feats = chunk_and_pad_sequence(cap_feats, sp_world_rank, sp_world_size, pad_amount, dim=-2)
        self._sp_pad["local_rows"] = local_sp_pad_rows(seq_len, cap_feats.shape[1], sp_world_rank)
        cap_attn_mask = chunk_and_pad_sequence(cap_attn_mask, sp_world_rank, sp_world_size, pad_amount, dim=-1)
        cap_freqs_cis_chunked = chunk_and_pad_sequence(cap_freqs_cis, sp_world_rank, sp_world_size, pad_amount, dim=-2)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                cap_feats = self._gradient_checkpointing_func(layer, cap_feats, cap_attn_mask, cap_freqs_cis_chunked)
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis_chunked)

        # Gather SP outputs and remove padding
        cap_feats = gather_and_unpad(cap_feats, pad_amount, dim=-2)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        # SP support
        seq_len = unified.shape[1]
        pad_amount = (sp_world_size - (seq_len % sp_world_size)) % sp_world_size
        self._self_attn_kwargs["valid_kv_len"] = seq_len if pad_amount else None
        unified = chunk_and_pad_sequence(unified, sp_world_rank, sp_world_size, pad_amount, dim=-2)
        self._sp_pad["local_rows"] = local_sp_pad_rows(seq_len, unified.shape[1], sp_world_rank)
        unified_attn_mask = chunk_and_pad_sequence(unified_attn_mask, sp_world_rank, sp_world_size, pad_amount, dim=-1)
        unified_freqs_cis = chunk_and_pad_sequence(unified_freqs_cis, sp_world_rank, sp_world_size, pad_amount, dim=-2)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.layers:
                unified = self._gradient_checkpointing_func(
                    layer, unified, unified_attn_mask, unified_freqs_cis, adaln_input
                )
        else:
            for layer in self.layers:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        # Gather SP outputs and remove padding
        unified = gather_and_unpad(unified, pad_amount, dim=-2)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if do_cfg_parallel:
            x_stacked = torch.stack(x, dim=0)
            x_stacked = get_cfg_group().all_gather(x_stacked, dim=0)
            x = list(x_stacked.unbind(dim=0))

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)


def z_image_pipefusion_patch_major(tensor, *, grid, patch_rows):
    """Reorder image tokens into padded height stripes for stale-KV updates.

    The synchronous cache repeats each stripe's last real token into its local
    padding. The first asynchronous visit replaces that approximation with the
    K/V produced from Z-Image's learned pad token while preserving equal patch
    shapes for transport and compilation.
    """
    rows, width = grid
    tensor = tensor[:, : rows * width].reshape(
        tensor.shape[0],
        rows,
        width,
        tensor.shape[-2],
        tensor.shape[-1],
    )
    patches = []
    for stripe in tensor.split(patch_rows, dim=1):
        stripe = stripe.flatten(1, 2)
        pad = (-stripe.shape[1]) % SEQ_MULTI_OF
        if pad:
            stripe = torch.cat((stripe, stripe[:, -1:].expand(-1, pad, -1, -1)), dim=1)
        patches.append(stripe)
    return torch.cat(patches, dim=1)


class xFuserZImagePipeFusionAttnProcessor(xFuserZSingleStreamAttnProcessor):
    """Cache image K/V by spatial patch while keeping caption K/V fresh."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del encoder_hidden_states, attention_mask
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query, key = flydsl_fused_qk_norm_rope(query, key, attn.norm_q, attn.norm_k, freqs_cis)
        dtype = query.dtype

        image_tokens = attn._xfuser_z_image_tokens
        image_key, caption_key = key.split((image_tokens, key.shape[1] - image_tokens), dim=1)
        image_value, caption_value = value.split((image_tokens, value.shape[1] - image_tokens), dim=1)
        state = get_runtime_state()
        if not state.patch_mode:
            image_key_for_cache = z_image_pipefusion_patch_major(
                image_key,
                grid=attn._xfuser_z_image_grid,
                patch_rows=attn._xfuser_z_image_patch_rows,
            )
            image_value_for_cache = z_image_pipefusion_patch_major(
                image_value,
                grid=attn._xfuser_z_image_grid,
                patch_rows=attn._xfuser_z_image_patch_rows,
            )
        else:
            image_key_for_cache = image_key
            image_value_for_cache = image_value
        cached_key, cached_value = get_cache_manager().update_and_get_kv_cache(
            new_kv=[image_key_for_cache, image_value_for_cache],
            layer=attn,
            slice_dim=1,
            layer_type="attn",
        )
        if state.patch_mode:
            key = torch.cat((cached_key, caption_key), dim=1)
            value = torch.cat((cached_value, caption_value), dim=1)

        hidden_states = USP(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            attn_layer=None,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


@xFuserTransformerWrappersRegister.register(ZImageTransformer2DModel)
class xFuserZImagePipeFusionTransformerWrapper(xFuserTransformerBaseWrapper):
    """Stage-local Z-Image backbone with first-stage refiners."""

    def __init__(self, transformer: ZImageTransformer2DModel):
        for layer in transformer.noise_refiner:
            layer.attention.processor = xFuserZImagePipeFusionAttnProcessor()
        for layer in transformer.context_refiner:
            layer.attention.processor = xFuserZSingleStreamAttnProcessor()
        for layer in transformer.layers:
            layer.attention.processor = xFuserZImagePipeFusionAttnProcessor()
        super().__init__(
            transformer=transformer,
            transformer_blocks_name=["layers"],
        )
        if not is_pipeline_first_stage():
            self.module.noise_refiner = nn.ModuleList()
            self.module.context_refiner = nn.ModuleList()
        cache_manager = get_cache_manager()
        for layer in (*self.noise_refiner, *self.layers):
            if not cache_manager.has_cache_entry(layer.attention):
                cache_manager.register_cache_entry(layer.attention, "attn")
        register_fp8_comms_eligible_modules(
            self,
            [layer.attention for layer in (*self.noise_refiner, *self.context_refiner, *self.layers)],
        )
        self._pipefusion_context_cache = None

    @staticmethod
    def _padded_tokens(value):
        return value + (-value) % SEQ_MULTI_OF

    def pipefusion_compile_capture_forward(
        self,
        x,
        t,
        cap_feats,
        patch_size=2,
        f_patch_size=1,
        *,
        pipeline_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        del t, kwargs
        if is_pipeline_first_stage():
            lengths = []
            for image, caption in zip(x, cap_feats):
                image_tokens = (
                    (image.shape[1] // f_patch_size) * (image.shape[2] // patch_size) * (image.shape[3] // patch_size)
                )
                lengths.append(self._padded_tokens(image_tokens) + self._padded_tokens(len(caption)))
            output = x[0].new_zeros((len(x), max(lengths), self.config.dim))
        else:
            output = pipeline_hidden_states
        if is_pipeline_last_stage():
            output = [torch.zeros_like(image) for image in x]
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def _prepare_patch_geometry(
        self,
        x,
        cap_feats,
        patch_size,
        f_patch_size,
        patch_start_height,
    ):
        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_pad_mask,
            cap_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
        if patch_start_height:
            row_offset = patch_start_height // patch_size
            for pos_ids, pad_mask in zip(x_pos_ids, x_pad_mask):
                pos_ids[:, 1] = torch.where(
                    pad_mask,
                    pos_ids[:, 1],
                    pos_ids[:, 1] + row_offset,
                )
        x_freqs = pad_sequence(
            list(self.rope_embedder(torch.cat(x_pos_ids)).split([len(value) for value in x_pos_ids])),
            batch_first=True,
            padding_value=0.0,
        )
        x_freqs = x_freqs[:, : max(len(value) for value in x)]
        cap_freqs = pad_sequence(
            list(self.rope_embedder(torch.cat(cap_pos_ids)).split([len(value) for value in cap_pos_ids])),
            batch_first=True,
            padding_value=0.0,
        )
        cap_freqs = cap_freqs[:, : max(len(value) for value in cap_feats)]
        return x, cap_feats, x_size, x_pad_mask, cap_pad_mask, x_freqs, cap_freqs

    @staticmethod
    def _embed_padded(items, pad_masks, embedder, pad_token):
        lengths = [len(value) for value in items]
        values = embedder(torch.cat(items))
        values = _scatter_pad_token(values, torch.cat(pad_masks), pad_token)
        return pad_sequence(
            list(values.split(lengths)),
            batch_first=True,
            padding_value=0.0,
        )

    @staticmethod
    def _attention_mask(items, device):
        lengths = [len(value) for value in items]
        mask = torch.zeros((len(items), max(lengths)), dtype=torch.bool, device=device)
        for index, length in enumerate(lengths):
            mask[index, :length] = 1
        return mask

    def _set_image_attention_metadata(self, layers, image_tokens, x_size, patch_size):
        state = get_runtime_state()
        rows = x_size[0][1] // patch_size
        width = x_size[0][2] // patch_size
        patch_rows = tuple(height // patch_size for height in state.pp_patches_height)
        for module in layers.modules():
            attention = getattr(module, "attention", None)
            if attention is not None:
                attention._xfuser_z_image_tokens = image_tokens
                attention._xfuser_z_image_grid = (rows, width)
                attention._xfuser_z_image_patch_rows = patch_rows

    def _refine_context(self, cap_feats, cap_pad_mask, cap_mask, cap_freqs):
        if self._pipefusion_context_cache is None:
            cap_hidden = self._embed_padded(
                cap_feats,
                cap_pad_mask,
                self.cap_embedder,
                self.cap_pad_token,
            )
            for layer in self.context_refiner:
                cap_hidden = layer(cap_hidden, cap_mask, cap_freqs)
            self._pipefusion_context_cache = (cap_hidden, cap_mask, cap_freqs)
        return self._pipefusion_context_cache

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        pipeline_hidden_states: Optional[torch.Tensor] = None,
        patch_start_height: int = 0,
        return_dict: bool = True,
    ):
        (
            x,
            cap_feats,
            x_size,
            x_pad_mask,
            cap_pad_mask,
            x_freqs,
            cap_freqs,
        ) = self._prepare_patch_geometry(
            x,
            cap_feats,
            patch_size,
            f_patch_size,
            patch_start_height,
        )
        device = x[0].device
        adaln_input = self.t_embedder((t.to(device) * self.t_scale)).type_as(x[0])
        x_mask = self._attention_mask(x, device)
        cap_mask = self._attention_mask(cap_feats, device)
        image_tokens = x_mask.shape[1]

        if is_pipeline_first_stage():
            x_hidden = self._embed_padded(
                x,
                x_pad_mask,
                self.all_x_embedder[f"{patch_size}-{f_patch_size}"],
                self.x_pad_token,
            )
            self._set_image_attention_metadata(
                self.noise_refiner,
                image_tokens,
                x_size,
                patch_size,
            )
            for layer in self.noise_refiner:
                x_hidden = layer(x_hidden, x_mask, x_freqs, adaln_input)
            cap_hidden, cap_mask, cap_freqs = self._refine_context(
                cap_feats,
                cap_pad_mask,
                cap_mask,
                cap_freqs,
            )
            unified = torch.cat((x_hidden, cap_hidden), dim=1)
        else:
            if pipeline_hidden_states is None:
                raise ValueError("A non-first Z-Image PipeFusion stage requires pipeline_hidden_states.")
            unified = pipeline_hidden_states

        unified_mask = torch.cat((x_mask, cap_mask), dim=1)
        unified_freqs = torch.cat((x_freqs, cap_freqs), dim=1)
        self._set_image_attention_metadata(self.layers, image_tokens, x_size, patch_size)
        for layer in self.layers:
            unified = layer(unified, unified_mask, unified_freqs, adaln_input)

        if is_pipeline_last_stage():
            unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
            output = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size)
        else:
            output = unified
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

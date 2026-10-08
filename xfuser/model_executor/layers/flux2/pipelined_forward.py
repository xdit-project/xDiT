"""Pipelined replacement for ``Flux2Transformer2DModel.forward``.

VENDORED CODE.  Everything outside the two block loops below is a transcription
of upstream's forward (diffusers 0.37) and must track it.  It is deliberately
confined to this module, and it is deliberately partial: the reference-image KV
cache paths (``kv_cache``, ``kv_cache_mode``, ``num_ref_tokens``) are *not*
reimplemented.  :func:`run_pipelined_forward` returns None for anything it does
not model, and the wrapper then runs upstream's forward untouched -- so the
features this file omits keep working through the code that owns them, and the
blast radius of the copy is the plain text-to-image path only.
"""

from typing import Any, Optional

import torch

from xfuser.model_executor.layers.flux2.pipelined_stacks import (
    run_double_stack,
    run_single_stack,
    stacks_are_pipelinable,
)

_SENTINEL_FALLBACK = None


def run_pipelined_forward(
    model,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.Tensor] = None,
    img_ids: Optional[torch.Tensor] = None,
    txt_ids: Optional[torch.Tensor] = None,
    guidance: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[dict[str, Any]] = None,
    return_dict: bool = True,
    kv_cache=None,
    kv_cache_mode: Optional[str] = None,
    num_ref_tokens: int = 0,
    ref_fixed_timestep: float = 0.0,
    **unknown,
):
    """Upstream's forward with the block stacks pipelined, or None to fall back.

    None means "this request uses something this module does not implement";
    the caller must then run the real ``Flux2Transformer2DModel.forward``.
    """
    if unknown:
        # A diffusers version that takes an argument this copy predates: let the
        # real forward have it rather than silently ignoring it.
        return _SENTINEL_FALLBACK
    if kv_cache is not None or kv_cache_mode is not None or num_ref_tokens:
        return _SENTINEL_FALLBACK  # reference-image KV cache: not reimplemented
    if encoder_hidden_states is None or timestep is None:
        return _SENTINEL_FALLBACK
    if torch.is_grad_enabled() and getattr(model, "gradient_checkpointing", False):
        return _SENTINEL_FALLBACK
    if not stacks_are_pipelinable(model.transformer_blocks, model.single_transformer_blocks):
        return _SENTINEL_FALLBACK

    num_txt_tokens = encoder_hidden_states.shape[1]

    # 1. Timestep embedding and modulation parameters.  Evaluated once for the
    #    whole stack, which is what lets the runners rotate a block's opening
    #    norm into its predecessor's closing kernel.
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = model.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = model.double_stream_modulation_img(temb)
    double_stream_mod_txt = model.double_stream_modulation_txt(temb)
    single_stream_mod = model.single_stream_modulation(temb)

    # 2. Input projections.
    hidden_states = model.x_embedder(hidden_states)
    encoder_hidden_states = model.context_embedder(encoder_hidden_states)

    # 3. RoPE tables for the joint [text, image] stream.
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    image_rotary_emb = model.pos_embed(img_ids)
    text_rotary_emb = model.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # 4. Double stream blocks.
    encoder_hidden_states, hidden_states = run_double_stack(
        model.transformer_blocks,
        hidden_states,
        encoder_hidden_states,
        double_stream_mod_img,
        double_stream_mod_txt,
        concat_rotary_emb,
        joint_attention_kwargs,
    )

    # 5. Single stream blocks, on the concatenated [text, image] stream.
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    hidden_states = run_single_stack(
        model.single_transformer_blocks,
        hidden_states,
        single_stream_mod,
        concat_rotary_emb,
        joint_attention_kwargs,
    )
    hidden_states = hidden_states[:, num_txt_tokens:, ...]

    # 6. Output layers.
    hidden_states = model.norm_out(hidden_states, temb)
    output = model.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Transformer2DModelOutput,
    )

    return Flux2Transformer2DModelOutput(sample=output)

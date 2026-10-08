"""Pipelined FLUX.2 block stacks.

VENDORED CODE.  The two loops below are adapted from
``Flux2Transformer2DModel.forward`` and the two block ``forward`` methods in
``diffusers/models/transformers/transformer_flux2.py`` (0.37).  They live in
their own module, away from the wrapper, so the copy is visible and auditable
rather than smeared through xDiT's own code.  :func:`stacks_are_pipelinable`
gates them; anything it does not recognise runs upstream's forward untouched.

Why copy at all
---------------
A FLUX.2 block opens with ``LayerNorm(x) * (1 + scale) + shift`` and closes with
``x + gate * out``.  Run block by block those are two separate trips over the
residual stream per boundary.  They are really one producer-consumer pair, so
these runners rotate the opening norm of block ``i + 1`` into the closing kernel
of block ``i``: one fused launch per boundary instead of several row passes
(see ``fused_residual_norm``).  Only the peeled first opening and last closing
run on their own.

What makes it cheap here
------------------------
FLUX.2 computes modulation *once per stack*, not once per block --
``double_stream_modulation_img`` / ``_txt`` and ``single_stream_modulation`` are
each evaluated a single time in the forward and the same tensor is handed to
every block.  Its norms are ``nn.LayerNorm(dim, elementwise_affine=False)``, so
they carry no parameters either.  Between them, block ``i + 1``'s opening norm
is a function of tensors block ``i`` already holds, and the rotation needs no
lookahead at all.
"""

from typing import Optional

import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Modulation,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)

from xfuser.model_executor.layers.flux2.fused_residual_norm import (
    fused_gated_residual_layernorm,
)


def stacks_are_pipelinable(double_blocks, single_blocks) -> bool:
    """True when the runners may drive the blocks' internals directly.

    False for anything that has taken ownership of a block as a unit: a
    per-block ``torch.compile`` (FSDP, ``--cache_method``) wraps it in an
    ``OptimizedModule``, and reaching past it to ``block.attn`` would step
    around that compiled graph.  Those stacks run upstream's loops.
    """
    if not hasattr(Flux2Modulation, "split"):
        return False  # pre-0.37 modulation API; the runners assume the new one
    return (
        bool(double_blocks)
        and bool(single_blocks)
        and all(type(b) is Flux2TransformerBlock for b in double_blocks)
        and all(type(b) is Flux2SingleTransformerBlock for b in single_blocks)
    )


def _clip_fp16(x: torch.Tensor) -> torch.Tensor:
    # Upstream clips both streams after their residuals when running fp16.
    if x.dtype == torch.float16:
        return x.clip(-65504, 65504)
    return x


def run_double_stack(
    blocks,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb_mod_img: torch.Tensor,
    temb_mod_txt: torch.Tensor,
    image_rotary_emb,
    joint_attention_kwargs: Optional[dict],
):
    """``Flux2TransformerBlock`` stack, software-pipelined by one norm."""
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(
        temb_mod_img, 2
    )
    (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = (
        Flux2Modulation.split(temb_mod_txt, 2)
    )
    kwargs = joint_attention_kwargs or {}
    last = len(blocks) - 1

    # Peeled prologue: the first block's opening norms have no predecessor.
    norm_hidden_states = blocks[0].norm1(hidden_states) * (1 + scale_msa) + shift_msa
    norm_encoder_hidden_states = (
        blocks[0].norm1_context(encoder_hidden_states) * (1 + c_scale_msa) + c_shift_msa
    )

    for i, block in enumerate(blocks):
        attn_output, context_attn_output = block.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )

        # Close attention, open the feed-forward, in one kernel per stream.
        hidden_states, norm_hidden_states = fused_gated_residual_layernorm(
            hidden_states, attn_output, gate_msa, block.norm2, scale_mlp, shift_mlp
        )
        encoder_hidden_states, norm_encoder_hidden_states = fused_gated_residual_layernorm(
            encoder_hidden_states,
            context_attn_output,
            c_gate_msa,
            block.norm2_context,
            c_scale_mlp,
            c_shift_mlp,
        )

        ff_output = block.ff(norm_hidden_states)
        context_ff_output = block.ff_context(norm_encoder_hidden_states)

        # Close the feed-forward, and open the *next* block's attention: the
        # modulation is shared across the stack, so this needs no lookahead.
        emit = i < last
        hidden_states, norm_hidden_states = fused_gated_residual_layernorm(
            hidden_states, ff_output, gate_mlp, block.norm1, scale_msa, shift_msa, emit_norm=emit
        )
        encoder_hidden_states, norm_encoder_hidden_states = fused_gated_residual_layernorm(
            encoder_hidden_states,
            context_ff_output,
            c_gate_mlp,
            block.norm1_context,
            c_scale_msa,
            c_shift_msa,
            emit_norm=emit,
        )
        encoder_hidden_states = _clip_fp16(encoder_hidden_states)

    return encoder_hidden_states, hidden_states


def run_single_stack(
    blocks,
    hidden_states: torch.Tensor,
    temb_mod: torch.Tensor,
    image_rotary_emb,
    joint_attention_kwargs: Optional[dict],
):
    """``Flux2SingleTransformerBlock`` stack, software-pipelined by one norm."""
    mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
    kwargs = joint_attention_kwargs or {}
    last = len(blocks) - 1

    norm_hidden_states = blocks[0].norm(hidden_states) * (1 + mod_scale) + mod_shift

    for i, block in enumerate(blocks):
        attn_output = block.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )
        hidden_states, norm_hidden_states = fused_gated_residual_layernorm(
            hidden_states,
            attn_output,
            mod_gate,
            block.norm,
            mod_scale,
            mod_shift,
            emit_norm=i < last,
        )
        hidden_states = _clip_fp16(hidden_states)

    return hidden_states

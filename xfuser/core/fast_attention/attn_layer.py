import torch
from diffusers.models.attention_processor import Attention
from typing import Optional
import torch.nn.functional as F
import flash_attn
from enum import Flag, auto
from .fast_attn_state import get_fast_attn_window_size


class FastAttnMethod(Flag):
    FULL_ATTN = auto()
    RESIDUAL_WINDOW_ATTN = auto()
    OUTPUT_SHARE = auto()
    CFG_SHARE = auto()
    RESIDUAL_WINDOW_ATTN_CFG_SHARE = RESIDUAL_WINDOW_ATTN | CFG_SHARE
    FULL_ATTN_CFG_SHARE = FULL_ATTN | CFG_SHARE

    def has(self, method: "FastAttnMethod"):
        return bool(self & method)


class xFuserFastAttention:
    window_size: list[int] = [-1, -1]
    steps_method: list[FastAttnMethod] = []
    cond_first: bool = False
    need_compute_residual: list[bool] = []
    need_cache_output: bool = False

    def __init__(
        self,
        steps_method: list[FastAttnMethod] = [],
        cond_first: bool = False,
    ):
        window_size = get_fast_attn_window_size()
        self.window_size = [window_size, window_size]
        self.steps_method = steps_method
        # CFG order flag (conditional first or unconditional first)
        self.cond_first = cond_first
        self.need_compute_residual = self.compute_need_compute_residual()
        self.need_cache_output = True

    def set_methods(
        self,
        steps_method: list[FastAttnMethod],
        selecting: bool = False,
    ):
        self.steps_method = steps_method
        if selecting:
            if len(self.need_compute_residual) != len(self.steps_method):
                self.need_compute_residual = [False] * len(self.steps_method)
        else:
            self.need_compute_residual = self.compute_need_compute_residual()

    def compute_need_compute_residual(self):
        """Check at which timesteps do we need to compute the full-window residual of this attention module"""
        need_compute_residual = []
        for i, method in enumerate(self.steps_method):
            need = False
            if method.has(FastAttnMethod.FULL_ATTN):
                for j in range(i + 1, len(self.steps_method)):
                    if self.steps_method[j].has(FastAttnMethod.RESIDUAL_WINDOW_ATTN):
                        # If encountered a step that conduct WA-RS,
                        # this step needs the residual computation
                        need = True
                    if self.steps_method[j].has(FastAttnMethod.FULL_ATTN):
                        # If encountered another step using the `full-attn` strategy,
                        # this step doesn't need the residual computation
                        break
            need_compute_residual.append(need)
        return need_compute_residual

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):

        # Before calculating the attention, prepare the related parameters
        method = self.steps_method[attn.stepi] if attn.stepi < len(self.steps_method) else FastAttnMethod.FULL_ATTN
        need_compute_residual = self.need_compute_residual[attn.stepi] if attn.stepi < len(self.need_compute_residual) else False

        # Run the forward method according to the selected strategy
        residual = hidden_states
        if method.has(FastAttnMethod.OUTPUT_SHARE):
            hidden_states = attn.cached_output
        else:
            if method.has(FastAttnMethod.CFG_SHARE):
                # Directly use the unconditional branch's attention output
                # as the conditional branch's attention output

                batch_size = hidden_states.shape[0]
                if self.cond_first:
                    hidden_states = hidden_states[: batch_size // 2]
                else:
                    hidden_states = hidden_states[batch_size // 2 :]
                if encoder_hidden_states is not None:
                    if self.cond_first:
                        encoder_hidden_states = encoder_hidden_states[: batch_size // 2]
                    else:
                        encoder_hidden_states = encoder_hidden_states[batch_size // 2 :]
                if attention_mask is not None:
                    if self.cond_first:
                        attention_mask = attention_mask[: batch_size // 2]
                    else:
                        attention_mask = attention_mask[batch_size // 2 :]

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim)

            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)

            if attention_mask is not None:
                assert (
                    method.has(FastAttnMethod.RESIDUAL_WINDOW_ATTN) == False
                ), "Attention mask is not supported in windowed attention"

                hidden_states = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                ).transpose(1, 2)
            elif method.has(FastAttnMethod.FULL_ATTN):
                all_hidden_states = flash_attn.flash_attn_func(query, key, value)
                if need_compute_residual:
                    # Compute the full-window attention residual
                    w_hidden_states = flash_attn.flash_attn_func(query, key, value, window_size=self.window_size)
                    window_residual = all_hidden_states - w_hidden_states
                    if method.has(FastAttnMethod.CFG_SHARE):
                        window_residual = torch.cat([window_residual, window_residual], dim=0)
                    # Save the residual for usage in follow-up steps
                    attn.cached_residual = window_residual
                hidden_states = all_hidden_states
            elif method.has(FastAttnMethod.RESIDUAL_WINDOW_ATTN):
                w_hidden_states = flash_attn.flash_attn_func(query, key, value, window_size=self.window_size)
                hidden_states = w_hidden_states + attn.cached_residual[:batch_size].view_as(w_hidden_states)

            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if method.has(FastAttnMethod.CFG_SHARE):
                hidden_states = torch.cat([hidden_states, hidden_states], dim=0)

            if self.need_cache_output:
                attn.cached_output = hidden_states

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # After been call once, add the timestep index of this attention module by 1
        attn.stepi += 1

        return hidden_states


# TODO: Implement classes to support DiTFastAttn in different diffusion models
class xFuserJointFastAttention(xFuserFastAttention):
    pass


class xFuserFluxFastAttention(xFuserFastAttention):
    pass


class xFuserHunyuanFastAttention(xFuserFastAttention):
    pass

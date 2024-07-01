import torch.cuda
from diffusers.models.attention import Attention
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from pipefuser.modules.base_module import BaseModule
from pipefuser.utils import DistriConfig


class DistriAttentionTP(BaseModule):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriAttentionTP, self).__init__(module, distri_config)

        heads = module.heads
        sliced_heads = heads // distri_config.n_device_per_batch
        remainder_heads = heads % distri_config.n_device_per_batch
        if distri_config.split_idx() < remainder_heads:
            sliced_heads += 1
        self.sliced_heads = sliced_heads

        if sliced_heads > 0:
            if distri_config.split_idx() < remainder_heads:
                start_head = distri_config.split_idx() * sliced_heads
            else:
                start_head = (
                    remainder_heads * (sliced_heads + 1)
                    + (distri_config.split_idx() - remainder_heads) * sliced_heads
                )
            end_head = start_head + sliced_heads

            dim = module.to_q.out_features // heads

            sharded_to_q = nn.Linear(
                module.to_q.in_features,
                sliced_heads * dim,
                bias=module.to_q.bias is not None,
                device=module.to_q.weight.device,
                dtype=module.to_q.weight.dtype,
            )
            sharded_to_q.weight.data.copy_(
                module.to_q.weight.data[start_head * dim : end_head * dim]
            )
            if module.to_q.bias is not None:
                sharded_to_q.bias.data.copy_(
                    module.to_q.bias.data[start_head * dim : end_head * dim]
                )

            sharded_to_k = nn.Linear(
                module.to_k.in_features,
                sliced_heads * dim,
                bias=module.to_k.bias is not None,
                device=module.to_k.weight.device,
                dtype=module.to_k.weight.dtype,
            )
            sharded_to_k.weight.data.copy_(
                module.to_k.weight.data[start_head * dim : end_head * dim]
            )
            if module.to_k.bias is not None:
                sharded_to_k.bias.data.copy_(
                    module.to_k.bias.data[start_head * dim : end_head * dim]
                )

            sharded_to_v = nn.Linear(
                module.to_v.in_features,
                sliced_heads * dim,
                bias=module.to_v.bias is not None,
                device=module.to_v.weight.device,
                dtype=module.to_v.weight.dtype,
            )
            sharded_to_v.weight.data.copy_(
                module.to_v.weight.data[start_head * dim : end_head * dim]
            )
            if module.to_v.bias is not None:
                sharded_to_v.bias.data.copy_(
                    module.to_v.bias.data[start_head * dim : end_head * dim]
                )

            sharded_to_out = nn.Linear(
                sliced_heads * dim,
                module.to_out[0].out_features,
                bias=module.to_out[0].bias is not None,
                device=module.to_out[0].weight.device,
                dtype=module.to_out[0].weight.dtype,
            )
            sharded_to_out.weight.data.copy_(
                module.to_out[0].weight.data[:, start_head * dim : end_head * dim]
            )
            if module.to_out[0].bias is not None:
                sharded_to_out.bias.data.copy_(module.to_out[0].bias.data)

            del module.to_q
            del module.to_k
            del module.to_v

            old_to_out = module.to_out[0]

            module.to_q = sharded_to_q
            module.to_k = sharded_to_k
            module.to_v = sharded_to_v
            module.to_out[0] = sharded_to_out
            module.heads = sliced_heads

            del old_to_out

            torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor or None = None,
        attention_mask: torch.FloatTensor or None = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        distri_config = self.distri_config
        module = self.module
        residual = hidden_states

        if self.sliced_heads > 0:
            input_ndim = hidden_states.ndim

            assert input_ndim == 3

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = module.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, module.heads, -1, attention_mask.shape[-1]
                )

            if module.group_norm is not None:
                hidden_states = module.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = module.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif module.norm_cross:
                encoder_hidden_states = module.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = module.to_k(encoder_hidden_states)
            value = module.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // module.heads

            query = query.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, module.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = F.linear(hidden_states, module.to_out[0].weight, bias=None)
            # dropout
            hidden_states = module.to_out[1](hidden_states)
        else:
            hidden_states = torch.zeros(
                [
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    module.to_out[0].out_features,
                ],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        dist.all_reduce(
            hidden_states,
            op=dist.ReduceOp.SUM,
            group=distri_config.batch_parallel_group,
            async_op=False,
        )
        if module.to_out[0].bias is not None:
            hidden_states = hidden_states + module.to_out[0].bias.view(1, 1, -1)

        if module.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / module.rescale_output_factor

        self.counter += 1

        return hidden_states

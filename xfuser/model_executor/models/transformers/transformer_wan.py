from typing import Optional, Dict, Any, Union
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)

from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

from xfuser.core.distributed import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from .register import xFuserTransformerWrappersRegister
from .base_transformer import xFuserTransformerBaseWrapper
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.logger import init_logger
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)

logger = init_logger(__name__)


@xFuserTransformerWrappersRegister.register(WanTransformer3DModel)
class xFuserWanTransformer3DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: WanTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_name_to_wrap=["attn1", "attn2"]# "WanTransformerBlock"],
        )

    def _split_transformer_blocks(
        self,
        transformer: nn.Module,
        blocks_name: list[str] = [],
    ):
        if not hasattr(transformer, "blocks"):
            raise AttributeError(
                f"'{transformer.__class__.__name__}' object has no attribute 'blocks'"
            )

        # transformer layer split
        attn_layer_num_for_pp = (
            get_runtime_state().parallel_config.pp_config.attn_layer_num_for_pp
        )
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()
        if attn_layer_num_for_pp is not None:
            assert sum(attn_layer_num_for_pp) == len(transformer.blocks), (
                "Sum of attn_layer_num_for_pp should be equal to the "
                "number of transformer blocks"
            )
            if is_pipeline_first_stage():
                transformer.blocks = transformer.blocks[: attn_layer_num_for_pp[0]]
            else:
                transformer.blocks = transformer.blocks[
                    sum(attn_layer_num_for_pp[: pp_rank - 1]) : sum(
                        attn_layer_num_for_pp[:pp_rank]
                    )
                ]
        else:
            num_blocks_per_stage = (
                len(transformer.blocks) + pp_world_size - 1
            ) // pp_world_size
            start_idx = pp_rank * num_blocks_per_stage
            end_idx = min(
                (pp_rank + 1) * num_blocks_per_stage,
                len(transformer.blocks),
            )
            transformer.blocks = transformer.blocks[start_idx:end_idx]
        # position embedding
        if not is_pipeline_first_stage():
            transformer.pos_embed = None
        if not is_pipeline_last_stage():
            transformer.norm_out = None
            transformer.proj_out = None
        return transformer



    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
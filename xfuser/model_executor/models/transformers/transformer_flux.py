from typing import Optional, Dict, Any, Union
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)

from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.logger import init_logger
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)

logger = init_logger(__name__)
from diffusers.models.attention import FeedForward

# FluxTransformer2DModel(
#   (pos_embed): EmbedND()
#   (time_text_embed): CombinedTimestepTextProjEmbeddings(
#     (time_proj): Timesteps()
#     (timestep_embedder): TimestepEmbedding(
#       (linear_1): Linear(in_features=256, out_features=3072, bias=True)
#       (act): SiLU()
#       (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
#     )
#     (text_embedder): PixArtAlphaTextProjection(
#       (linear_1): Linear(in_features=768, out_features=3072, bias=True)
#       (act_1): SiLU()
#       (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
#     )
#   )
#   (context_embedder): Linear(in_features=4096, out_features=3072, bias=True)
#   (x_embedder): Linear(in_features=64, out_features=3072, bias=True)
#   (transformer_blocks): ModuleList(
#     (0-18): 19 x FluxTransformerBlock(
#       (norm1): AdaLayerNormZero(
#         (silu): SiLU()
#         (linear): Linear(in_features=3072, out_features=18432, bias=True)
#         (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
#       )
#       (norm1_context): AdaLayerNormZero(
#         (silu): SiLU()
#         (linear): Linear(in_features=3072, out_features=18432, bias=True)
#         (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
#       )
#       (attn): Attention(
#         (norm_q): RMSNorm()
#         (norm_k): RMSNorm()
#         (to_q): Linear(in_features=3072, out_features=3072, bias=True)
#         (to_k): Linear(in_features=3072, out_features=3072, bias=True)
#         (to_v): Linear(in_features=3072, out_features=3072, bias=True)
#         (add_k_proj): Linear(in_features=3072, out_features=3072, bias=True)
#         (add_v_proj): Linear(in_features=3072, out_features=3072, bias=True)
#         (add_q_proj): Linear(in_features=3072, out_features=3072, bias=True)
#         (to_out): ModuleList(
#           (0): Linear(in_features=3072, out_features=3072, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#         (to_add_out): Linear(in_features=3072, out_features=3072, bias=True)
#         (norm_added_q): RMSNorm()
#         (norm_added_k): RMSNorm()
#       )
#       (norm2): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
#       (ff): FeedForward(
#         (net): ModuleList(
#           (0): GELU(
#             (proj): Linear(in_features=3072, out_features=12288, bias=True)
#           )
#           (1): Dropout(p=0.0, inplace=False)
#           (2): Linear(in_features=12288, out_features=3072, bias=True)
#         )
#       )
#       (norm2_context): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
#       (ff_context): FeedForward(
#         (net): ModuleList(
#           (0): GELU(
#             (proj): Linear(in_features=3072, out_features=12288, bias=True)
#           )
#           (1): Dropout(p=0.0, inplace=False)
#           (2): Linear(in_features=12288, out_features=3072, bias=True)
#         )
#       )
#     )
#   )
#   (single_transformer_blocks): ModuleList(
#     (0-37): 38 x FluxSingleTransformerBlock(
#       (norm): AdaLayerNormZeroSingle(
#         (silu): SiLU()
#         (linear): Linear(in_features=3072, out_features=9216, bias=True)
#         (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
#       )
#       (proj_mlp): Linear(in_features=3072, out_features=12288, bias=True)
#       (act_mlp): GELU(approximate='tanh')
#       (proj_out): Linear(in_features=15360, out_features=3072, bias=True)
#       (attn): Attention(
#         (norm_q): RMSNorm()
#         (norm_k): RMSNorm()
#         (to_q): Linear(in_features=3072, out_features=3072, bias=True)
#         (to_k): Linear(in_features=3072, out_features=3072, bias=True)
#         (to_v): Linear(in_features=3072, out_features=3072, bias=True)
#       )
#     )
#   )
#   (norm_out): AdaLayerNormContinuous(
#     (silu): SiLU()
#     (linear): Linear(in_features=3072, out_features=6144, bias=True)
#     (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
#   )
#   (proj_out): Linear(in_features=3072, out_features=64, bias=True)
# )


@xFuserTransformerWrappersRegister.register(FluxTransformer2DModel)
class xFuserFluxTransformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[FeedForward],
            # submodule_classes_to_wrap=[FeedForward] if get_tensor_model_parallel_world_size() > 1 else [],
            submodule_name_to_wrap=["attn", "ff_context"],
        )
        self.encoder_hidden_states_cache = [
            None for _ in range(len(self.transformer_blocks))
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # print(f"{txt_ids.shape=}, {img_ids.shape=}")
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                encoder_hidden_states, hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        output = output, None

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


if __name__ == "__main__":
    # print module in FluxTransformer2DModel
    model = FluxTransformer2DModel()

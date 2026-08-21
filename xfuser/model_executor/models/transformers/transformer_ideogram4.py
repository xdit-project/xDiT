import torch
import torch.nn.functional as F

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.model_executor.layers.usp import USP
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister,
)
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    half = hidden_states.shape[-1] // 2
    return torch.cat((-hidden_states[..., half:], hidden_states[..., :half]), dim=-1)


def _make_xfuser_ideogram4_attention_processor():
    from diffusers.models.transformers.transformer_ideogram4 import (
        Ideogram4AttnProcessor,
    )

    @xFuserAttentionProcessorRegister.register(Ideogram4AttnProcessor)
    class xFuserIdeogram4AttnProcessor(Ideogram4AttnProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.attention_function = USP

        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
            image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            query = attn.to_q(hidden_states).unflatten(
                -1,
                (attn.num_heads, attn.head_dim),
            )
            key = attn.to_k(hidden_states).unflatten(
                -1,
                (attn.num_heads, attn.head_dim),
            )
            value = attn.to_v(hidden_states).unflatten(
                -1,
                (attn.num_heads, attn.head_dim),
            )

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            cos, sin = image_rotary_emb
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
            query = (query * cos) + (_rotate_half(query) * sin)
            key = (key * cos) + (_rotate_half(key) * sin)

            hidden_states = self.attention_function(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
            ).transpose(1, 2)
            hidden_states = hidden_states.flatten(2, 3).type_as(query)
            return attn.to_out[0](hidden_states)

    return xFuserIdeogram4AttnProcessor


def _make_xfuser_ideogram4_transformer_wrapper():
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.transformers.transformer_ideogram4 import (
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
        Ideogram4Transformer2DModel,
    )
    from diffusers.utils import apply_lora_scale

    processor_class = _make_xfuser_ideogram4_attention_processor()

    class xFuserIdeogram4Transformer2DWrapper(Ideogram4Transformer2DModel):
        def _install_xfuser_processors(self) -> None:
            for layer in self.layers:
                layer.attention.set_processor(processor_class())

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = super().from_pretrained(*args, **kwargs)
            model.__class__ = cls
            model._install_xfuser_processors()
            return model

        @classmethod
        def from_config(cls, *args, **kwargs):
            # The meta path builds from config and never calls from_pretrained, and a
            # processor swap touches no weights, so it is safe on meta tensors.
            model = super().from_config(*args, **kwargs)
            model._install_xfuser_processors()
            return model

        def _set_sequence_layout(
            self,
            num_pad_tokens: int,
            num_text_tokens: int,
            num_image_tokens: int,
        ) -> None:
            self._num_pad_tokens = num_pad_tokens
            self._num_text_tokens = num_text_tokens
            self._num_image_tokens = num_image_tokens

        def _init_sp_state(self) -> None:
            try:
                self._sp_rank = get_sequence_parallel_rank()
                self._sp_size = get_sequence_parallel_world_size()
            except AssertionError:
                self._sp_rank = 0
                self._sp_size = 1

        def _get_sequence_layout(
            self,
            indicator: torch.Tensor,
        ) -> tuple[int, int, int]:
            if hasattr(self, "_num_image_tokens"):
                return (
                    self._num_pad_tokens,
                    self._num_text_tokens,
                    self._num_image_tokens,
                )

            text_counts = (indicator == LLM_TOKEN_INDICATOR).sum(dim=1)
            image_counts = (indicator == OUTPUT_IMAGE_INDICATOR).sum(dim=1)
            pad_counts = indicator.shape[1] - text_counts - image_counts
            if not (
                torch.equal(text_counts, text_counts[:1].expand_as(text_counts))
                and torch.equal(image_counts, image_counts[:1].expand_as(image_counts))
                and torch.equal(pad_counts, pad_counts[:1].expand_as(pad_counts))
            ):
                raise ValueError(
                    "Ideogram 4 xDiT requires prompts in the same batch to have equal token lengths."
                )

            layout = (
                int(pad_counts[0].item()),
                int(text_counts[0].item()),
                int(image_counts[0].item()),
            )
            self._set_sequence_layout(*layout)
            return layout

        @apply_lora_scale("attention_kwargs")
        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            segment_ids: torch.Tensor,
            indicator: torch.Tensor,
            attention_kwargs: dict | None = None,
            return_dict: bool = True,
        ):
            sp_rank = self._sp_rank
            sp_size = self._sp_size

            batch_size, _, in_channels = hidden_states.shape
            if in_channels != self.in_channels:
                raise ValueError(
                    f"Expected last dim {self.in_channels}, got {in_channels}."
                )

            num_pad_tokens, num_text_tokens, num_image_tokens = (
                self._get_sequence_layout(indicator)
            )
            text_start = num_pad_tokens
            image_start = text_start + num_text_tokens

            text_hidden_states = encoder_hidden_states[:, text_start:image_start]
            text_hidden_states = self.llm_cond_norm(text_hidden_states)
            text_hidden_states = self.llm_cond_proj(text_hidden_states)

            image_hidden_states = hidden_states[:, image_start:]
            image_hidden_states = self.input_proj(image_hidden_states)

            t_cond = self.t_embedding(timestep)
            if timestep.dim() == 1:
                t_cond = t_cond.unsqueeze(1)
            adaln_input = F.silu(self.adaln_proj(t_cond))

            text_indicator_embedding = self.embed_image_indicator(
                torch.zeros(
                    batch_size,
                    num_text_tokens,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            )
            image_indicator_embedding = self.embed_image_indicator(
                torch.ones(
                    batch_size,
                    num_image_tokens,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            )

            hidden_states = torch.cat(
                [
                    text_hidden_states + text_indicator_embedding,
                    image_hidden_states + image_indicator_embedding,
                ],
                dim=1,
            )
            position_ids = torch.cat(
                [
                    position_ids[:, text_start:image_start],
                    position_ids[:, image_start:],
                ],
                dim=1,
            )

            sequence_length = num_text_tokens + num_image_tokens
            pad_amount = (sp_size - sequence_length % sp_size) % sp_size
            hidden_states = chunk_and_pad_sequence(
                hidden_states,
                sp_rank,
                sp_size,
                pad_amount,
                dim=1,
            )
            position_ids = chunk_and_pad_sequence(
                position_ids,
                sp_rank,
                sp_size,
                pad_amount,
                dim=1,
            )

            cos, sin = self.rotary_emb(position_ids)
            image_rotary_emb = (
                cos.to(hidden_states.dtype),
                sin.to(hidden_states.dtype),
            )

            for block in self.layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        None,
                        image_rotary_emb,
                        adaln_input,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        None,
                        image_rotary_emb,
                        adaln_input,
                    )

            output = self.final_layer(hidden_states, conditioning=adaln_input)
            if sp_size > 1:
                output = gather_and_unpad(output, pad_amount, dim=1)

            pad_output = torch.zeros(
                batch_size,
                num_pad_tokens,
                output.shape[-1],
                dtype=output.dtype,
                device=output.device,
            )
            output = torch.cat([pad_output, output], dim=1)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

    return xFuserIdeogram4Transformer2DWrapper


_wrapper_cls = None


def get_ideogram4_transformer_wrapper_class():
    global _wrapper_cls
    if _wrapper_cls is None:
        _wrapper_cls = _make_xfuser_ideogram4_transformer_wrapper()
    return _wrapper_cls

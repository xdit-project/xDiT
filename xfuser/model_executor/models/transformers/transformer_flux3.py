from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux3 import (
    Flux3Attention,
    Flux3Transformer2DModel,
    _get_qkv_projections,
)

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.model_executor.layers.usp import USP
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
)


class xFuserFlux3AttnProcessor:
    """Flux3AttnProcessor with the attention call routed through USP.

    Each sequence stage (text, image and reference-image modes, then the joint
    blocks) shares one instance. Before a stage runs, the transformer sets
    ``valid_kv_len`` to the stage's real length when the SP pad made the sequence
    longer, so the pad keys are dropped after the all-to-all.
    """

    def __init__(self):
        self.valid_kv_len: Optional[int] = None

    def __call__(
        self,
        attn: Flux3Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("FLUX.3 attention takes no mask.")

        query, key, value = _get_qkv_projections(attn, hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # QK norm and RoPE in fp32 with one cast back, as upstream.
        query = attn.norm_q(query.float())
        key = attn.norm_k(key.float())

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query, key = query.to(value.dtype), key.to(value.dtype)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_kwargs = None if self.valid_kv_len is None else {"valid_kv_len": self.valid_kv_len}
        hidden_states = USP(query, key, value, attn_layer=attn, attention_kwargs=attention_kwargs)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def flux3_attn_modules(transformer) -> list[torch.nn.Module]:
    """Return every FLUX.3 attention module that executes USP, in execution order."""
    blocks = (
        *transformer.txt_mode_blocks,
        *transformer.mode_blocks["image"],
        *transformer.mode_blocks["image_cond"],
        *transformer.single_transformer_blocks,
    )
    return [block.attn for block in blocks]


class xFuserFlux3Transformer2DWrapper(Flux3Transformer2DModel):
    """Sequence-parallel FLUX.3 transformer.

    Every sequence is padded at its end to a multiple of the SP degree and split
    evenly across ranks, so after the Ulysses all-to-all the pad is a uniform
    suffix that ``valid_kv_len`` removes:

    - Per-mode blocks: the text, image and reference-image streams are sharded
      and attended separately, then gathered.
    - Joint blocks: ``[text, image, reference]`` is sharded as one sequence. A
      rank's chunk can straddle the text/content boundary, so each block gets
      the chunk's text part (possibly empty) as ``encoder_hidden_states`` and the
      rest as ``hidden_states``. The blocks run unmodified; with one rank this is
      exactly the upstream call.
    """

    # The arguments are spelled out because diffusers reads the config keys to pass
    # from this signature; behind **kwargs, flow_map would fall back to False.
    def __init__(
        self,
        in_channels: int = 128,
        cond_in_channels: int = 128,
        num_layers: int = 5,
        num_single_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 20480,
        timestep_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 10000,
        eps: float = 1e-6,
        flow_map: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            cond_in_channels=cond_in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_channels=timestep_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            flow_map=flow_map,
        )
        self._txt_processor = xFuserFlux3AttnProcessor()
        self._image_processor = xFuserFlux3AttnProcessor()
        self._image_cond_processor = xFuserFlux3AttnProcessor()
        self._joint_processor = xFuserFlux3AttnProcessor()
        stages = (
            (self.txt_mode_blocks, self._txt_processor),
            (self.mode_blocks["image"], self._image_processor),
            (self.mode_blocks["image_cond"], self._image_cond_processor),
            (self.single_transformer_blocks, self._joint_processor),
        )
        for blocks, processor in stages:
            for block in blocks:
                block.attn.processor = processor
        register_fp8_comms_eligible_modules(self, flux3_attn_modules(self))

    @staticmethod
    def _shard(
        sp_rank: int,
        sp_world_size: int,
        pad: int,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ):
        hidden_states = chunk_and_pad_sequence(hidden_states, sp_rank, sp_world_size, pad, dim=1)
        temb_mod = chunk_and_pad_sequence(temb_mod, sp_rank, sp_world_size, pad, dim=1)
        rotary_emb = tuple(chunk_and_pad_sequence(r, sp_rank, sp_world_size, pad, dim=0) for r in rotary_emb)
        return hidden_states, temb_mod, rotary_emb

    def _run_mode_blocks(
        self,
        blocks: torch.nn.ModuleList,
        processor: xFuserFlux3AttnProcessor,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sp_world_size = get_sequence_parallel_world_size()
        seq_len = hidden_states.shape[1]
        pad = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        processor.valid_kv_len = seq_len if pad else None

        hidden_states, temb_mod, rotary_emb = self._shard(
            get_sequence_parallel_rank(), sp_world_size, pad, hidden_states, temb_mod, rotary_emb
        )
        for block in blocks:
            hidden_states = block(hidden_states, temb_mod, rotary_emb)
        return gather_and_unpad(hidden_states, pad, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: Optional[torch.Tensor] = None,
        image_cond_ids: Optional[torch.Tensor] = None,
        target_timestep: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        if image_cond_hidden_states is not None and image_cond_ids is None:
            raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
        if image_cond_hidden_states is None and image_cond_ids is not None:
            raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError("`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`).")

        batch_size, image_seq_len = hidden_states.shape[:2]
        dtype = hidden_states.dtype

        # 1-4. Timestep embeddings, modulations, input projections and RoPE, as upstream.
        image_temb = self.time_proj(timestep.to(dtype) * 1000)
        image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
        zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
        zero_temb = self.timestep_embedder(zero_temb.to(dtype))[:, None, :]
        if target_timestep is not None:
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_temb = image_temb.expand(batch_size, image_seq_len, -1)
        zero_temb_act = F.silu(zero_temb)

        txt_temb_act = zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
        txt_early_mod = self.early_stream_modulations["txt"](txt_temb_act)
        txt_single_mod = self.single_stream_modulations["txt"](txt_temb_act)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        txt_rotary_emb = self.pos_embed(txt_ids)

        image_temb_act = F.silu(image_temb)
        image_early_mod = self.early_stream_modulations["image"](image_temb_act)
        image_single_mod = self.single_stream_modulations["image"](image_temb_act)
        hidden_states = self.x_embedders["image"](hidden_states)
        image_rotary_emb = self.pos_embed(img_ids)

        if image_cond_hidden_states is not None:
            cond_temb_act = zero_temb_act.expand(batch_size, image_cond_hidden_states.shape[1], -1)
            cond_early_mod = self.early_stream_modulations["image_cond"](cond_temb_act)
            cond_single_mod = self.single_stream_modulations["image_cond"](cond_temb_act)
            image_cond_hidden_states = self.x_embedders["image_cond"](image_cond_hidden_states)
            cond_rotary_emb = self.pos_embed(image_cond_ids)

        # 5. Per-mode blocks, each mode sharded across the SP ranks on its own.
        encoder_hidden_states = self._run_mode_blocks(
            self.txt_mode_blocks, self._txt_processor, encoder_hidden_states, txt_early_mod, txt_rotary_emb
        )
        hidden_states = self._run_mode_blocks(
            self.mode_blocks["image"], self._image_processor, hidden_states, image_early_mod, image_rotary_emb
        )
        streams = [
            (encoder_hidden_states, txt_single_mod, txt_rotary_emb),
            (hidden_states, image_single_mod, image_rotary_emb),
        ]
        if image_cond_hidden_states is not None:
            image_cond_hidden_states = self._run_mode_blocks(
                self.mode_blocks["image_cond"],
                self._image_cond_processor,
                image_cond_hidden_states,
                cond_early_mod,
                cond_rotary_emb,
            )
            streams.append((image_cond_hidden_states, cond_single_mod, cond_rotary_emb))

        # 6. Joint blocks over [text, image, reference], sharded as one sequence.
        txt_seq_len = encoder_hidden_states.shape[1]
        joint_states = torch.cat([s[0] for s in streams], dim=1)
        joint_mod = torch.cat([s[1] for s in streams], dim=1)
        joint_rotary_emb = (
            torch.cat([s[2][0] for s in streams], dim=0),
            torch.cat([s[2][1] for s in streams], dim=0),
        )
        joint_seq_len = joint_states.shape[1]

        sp_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()
        pad = (sp_world_size - joint_seq_len % sp_world_size) % sp_world_size
        self._joint_processor.valid_kv_len = joint_seq_len if pad else None
        joint_states, joint_mod, joint_rotary_emb = self._shard(
            sp_rank, sp_world_size, pad, joint_states, joint_mod, joint_rotary_emb
        )

        # Text tokens of this rank's chunk: the leading part, possibly none.
        chunk_len = joint_states.shape[1]
        local_txt_len = min(max(txt_seq_len - sp_rank * chunk_len, 0), chunk_len)
        context_states = joint_states[:, :local_txt_len]
        content_states = joint_states[:, local_txt_len:]
        context_mod = joint_mod[:, :local_txt_len]
        content_mod = joint_mod[:, local_txt_len:]

        for block in self.single_transformer_blocks:
            context_states, content_states = block(
                hidden_states=content_states,
                encoder_hidden_states=context_states,
                temb_mod=content_mod,
                temb_mod_context=context_mod,
                image_rotary_emb=joint_rotary_emb,
            )

        joint_states = torch.cat([context_states, content_states], dim=1)
        joint_states = gather_and_unpad(joint_states, pad, dim=1)

        # 7. Output layers on the image tokens only.
        hidden_states = joint_states[:, txt_seq_len : txt_seq_len + image_seq_len]
        hidden_states = self.norm_out(hidden_states, image_temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

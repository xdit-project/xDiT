import functools
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch._dynamo import mark_static_address
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux3 import (
    Flux3Attention,
    Flux3Transformer2DModel,
    _get_qkv_projections,
)

from xfuser.core.attention.spec import EXACT_KV_TOKENS_KEY, EXACT_Q_TOKENS_KEY
from xfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.model_executor.layers import usp_sdma
from xfuser.model_executor.layers.usp import PROCESS_GROUP, USP, attention
from xfuser.model_executor.models.transformers.transformers_utils import (
    PerCallCache,
    StaticBlockInputs,
    chunk_and_pad_sequence,
    gather_and_unpad,
)


class xFuserFlux3AttnProcessor:
    """Flux3AttnProcessor with the attention call routed through USP.

    Each sequence stage (text, image and reference-image modes, then the joint
    blocks) shares one instance. Before a stage runs, the transformer sets
    ``valid_kv_len`` to the stage's real length when the SP pad made the sequence
    longer, so the pad keys are dropped after the all-to-all, and ``local`` when
    every rank holds the whole stream, so attention runs without collectives.

    ``text_only`` marks the text stage's instance, which attends the prompt alone
    and runs on the cross-attention backend: the main backend unless
    ``--cross_attention_backend`` names another. ``exact_tokens`` marks the joint
    stage's prompt tokens, over the padded sequence, for backends that select KV
    blocks: their keys and their own query rows are computed exactly. Every other
    backend ignores it.
    """

    def __init__(self, text_only: bool = False):
        self.valid_kv_len: Optional[int] = None
        self.local = False
        self.text_only = text_only
        self.exact_tokens: Optional[torch.Tensor] = None

    def _backend(self):
        # The prompt is 80-512 tokens: too few KV blocks for a backend that selects blocks, so a
        # Sol or Sparge run names a dense one for it with --cross_attention_backend.
        return get_runtime_state().get_cross_attention_backend() if self.text_only else None

    def _attention_kwargs(self) -> Optional[Dict[str, Any]]:
        if self.valid_kv_len is None and self.exact_tokens is None:
            return None
        return {
            "valid_kv_len": self.valid_kv_len,
            EXACT_KV_TOKENS_KEY: self.exact_tokens,
            EXACT_Q_TOKENS_KEY: self.exact_tokens,
        }

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
        query, key, value = self._norm_rope(attn, query, key, value, image_rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.local:
            hidden_states = attention(query, key, value, backend=self._backend())
        else:
            hidden_states = USP(
                query,
                key,
                value,
                attn_layer=attn,
                backend=self._backend(),
                attention_kwargs=self._attention_kwargs(),
                combine_qkv_a2a=True,
            )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _norm_rope(attn, query, key, value, image_rotary_emb):
        """QK norm and RoPE of [b, s, heads, d] projections, for any subset of heads."""
        # QK norm and RoPE in fp32 with one cast back, as upstream.
        query = attn.norm_q(query.float())
        key = attn.norm_k(key.float())

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        return query.to(value.dtype), key.to(value.dtype), value


@usp_sdma.sdma_processor(xFuserFlux3AttnProcessor)
class xFuserFlux3SdmaAttnProcessor(xFuserFlux3AttnProcessor):
    """xFuserFlux3AttnProcessor under --ulysses_a2a sdma --ulysses_sdma_overlap_qkv.

    The QK norm and RoPE act per head, so Q/K/V are projected one head group at a time and
    each group's exchange overlaps the next group's projection
    (usp_sdma.grouped_qkv_attention). Calls that path does not cover take the base path.
    """

    sdma_grouped_qkv = True

    def __call__(
        self,
        attn: Flux3Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not self.local and attention_mask is None:
            grouped = usp_sdma.grouped_qkv_attention(
                attn,
                hidden_states,
                lambda q, k, v: self._norm_rope(attn, q, k, v, image_rotary_emb),
                backend=self._backend(),
                attention_kwargs=self._attention_kwargs(),
                group=PROCESS_GROUP.ULYSSES_PG,
            )
            if grouped is not None:
                hidden_states = grouped.transpose(1, 2).flatten(2, 3).to(hidden_states.dtype)
                hidden_states = attn.to_out[0](hidden_states)
                return attn.to_out[1](hidden_states)
        return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)


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

    - Per-mode blocks: the text, image and reference-image streams are attended
      separately. The image and reference streams are sharded, then gathered. A
      text stream of at most ``replicate_text_max_tokens`` runs whole on every
      rank instead, which skips its collectives for a little repeated compute.
    - Joint blocks: ``[text, image, reference]`` is sharded as one sequence. A
      rank's chunk can straddle the text/content boundary, so each block gets
      the chunk's text part (possibly empty) as ``encoder_hidden_states`` and the
      rest as ``hidden_states``. The blocks run unmodified; with one rank this is
      exactly the upstream call.
    """

    # Measured on 8x MI350X at 1024² (u8): replicating 80 or 400 text tokens saves about
    # 1.5 ms per eager forward of 38 ms; under HIP graphs the difference vanishes.
    replicate_text_max_tokens = 512

    # ConfigMixin picks the config keys to pass from this signature. Wrapping upstream's
    # __init__ shows it upstream's parameters; behind a bare **kwargs every key, flow_map
    # included, would fall back to its default.
    @functools.wraps(Flux3Transformer2DModel.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._txt_processor = xFuserFlux3AttnProcessor(text_only=True)
        self._image_processor = xFuserFlux3AttnProcessor()
        self._image_cond_processor = xFuserFlux3AttnProcessor()
        self._joint_processor = xFuserFlux3AttnProcessor()
        self._exact_tokens: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}
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
        # The RoPE tables and modulations every compiled block of a step reads, and the
        # RoPE tables (float64, many small kernels) and text projection of a call's ids and
        # prompt embeddings, which its steps share.
        self._block_inputs = StaticBlockInputs()
        self._step_constants = PerCallCache()

    def release_step_constants(self) -> None:
        """Drop the per-call cache, and with it the prompt embeddings and ids it holds."""
        self._step_constants.clear()

    def _modulation(self, name: str, mod_row: torch.Tensor, seq_len: int) -> torch.Tensor:
        """The (batch, seq_len, 3 * dim) modulation of a mode whose tokens share one row."""
        mod = self._block_inputs(name, mod_row).expand(-1, seq_len, -1)
        mark_static_address(mod)
        return mod

    def _content_modulation(self, mod_rows, stream_lens, start: int, length: int) -> torch.Tensor:
        """The joint-block modulation of this rank's content tokens: positions ``start`` to
        ``start + length`` of the concatenated content streams, each with its own row.
        SP pad positions past the end take the last stream's row; their outputs are dropped."""
        pieces = []
        offset = 0
        for index, (mod_row, stream_len) in enumerate(zip(mod_rows, stream_lens)):
            end = offset + stream_len if index < len(mod_rows) - 1 else start + length
            overlap = min(end, start + length) - max(offset, start)
            if overlap > 0:
                pieces.append((index, mod_row, overlap))
            offset += stream_len
        if len(pieces) <= 1:
            index, mod_row, _ = pieces[0] if pieces else (0, mod_rows[0], 0)
            return self._modulation(f"joint.content_mod{index}", mod_row, length)
        return torch.cat([mod_row.expand(-1, overlap, -1) for _, mod_row, overlap in pieces], dim=1)

    def _rotary(self, name: str, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._block_inputs(f"{name}.cos", rotary_emb[0]), self._block_inputs(f"{name}.sin", rotary_emb[1])

    def _joint_exact_tokens(self, txt_seq_len: int, padded_len: int, device: torch.device) -> torch.Tensor:
        # The prompt is a few percent of the joint sequence. A backend that selects KV blocks keeps
        # its keys exact for every query, and computes its own rows exactly: they become the text
        # stream every later joint block conditions on.
        # One tensor per shape, kept: the CUDA-graph-compiled joint blocks read it as a static
        # input, and a new tensor would make every block re-record its graph. Sequential CFG
        # alternates two shapes when the prompts' text windows differ (a long prompt against the
        # empty negative one).
        key = (txt_seq_len, padded_len, device)
        exact_tokens = self._exact_tokens.get(key)
        if exact_tokens is None:
            exact_tokens = torch.zeros(padded_len, dtype=torch.bool, device=device)
            exact_tokens[:txt_seq_len] = True
            self._exact_tokens[key] = exact_tokens
        return exact_tokens

    def _run_mode_blocks(
        self,
        name: str,
        blocks: torch.nn.ModuleList,
        processor: xFuserFlux3AttnProcessor,
        hidden_states: torch.Tensor,
        mod_row: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        local: bool = False,
    ) -> torch.Tensor:
        processor.local = local
        sp_world_size = get_sequence_parallel_world_size()
        seq_len = hidden_states.shape[1]
        pad = 0 if local else (sp_world_size - seq_len % sp_world_size) % sp_world_size
        processor.valid_kv_len = seq_len if pad else None
        if not local:
            sp_rank = get_sequence_parallel_rank()
            hidden_states = chunk_and_pad_sequence(hidden_states, sp_rank, sp_world_size, pad, dim=1)
            rotary_emb = tuple(chunk_and_pad_sequence(r, sp_rank, sp_world_size, pad, dim=0) for r in rotary_emb)

        temb_mod = self._modulation(f"{name}.mod", mod_row, hidden_states.shape[1])
        rotary_emb = self._rotary(name, rotary_emb)
        for block in blocks:
            hidden_states = block(hidden_states, temb_mod, rotary_emb)
        return hidden_states if local else gather_and_unpad(hidden_states, pad, dim=1)

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
        # One forward is one entry of the per-step schedules (--solattn_beta_schedule).
        get_runtime_state().increment_step_counter()

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
        zero_temb_act = F.silu(zero_temb)

        # Every token of a mode has the same timestep, so each modulation is computed on
        # one row and broadcast instead of on every token of every rank.
        txt_early_mod = self.early_stream_modulations["txt"](zero_temb_act)
        txt_single_mod = self.single_stream_modulations["txt"](zero_temb_act)
        encoder_hidden_states = self._step_constants("context", encoder_hidden_states, self.context_embedder)
        txt_rotary_emb = self._step_constants("txt_ids", txt_ids, self.pos_embed)

        image_temb_act = F.silu(image_temb)
        image_early_mod = self.early_stream_modulations["image"](image_temb_act)
        image_single_mod = self.single_stream_modulations["image"](image_temb_act)
        hidden_states = self.x_embedders["image"](hidden_states)
        image_rotary_emb = self._step_constants("img_ids", img_ids, self.pos_embed)

        if image_cond_hidden_states is not None:
            cond_early_mod = self.early_stream_modulations["image_cond"](zero_temb_act)
            cond_single_mod = self.single_stream_modulations["image_cond"](zero_temb_act)
            image_cond_hidden_states = self.x_embedders["image_cond"](image_cond_hidden_states)
            cond_rotary_emb = self._step_constants("image_cond_ids", image_cond_ids, self.pos_embed)

        # 5. Per-mode blocks, each mode attended on its own; short text runs whole on every rank.
        replicate_text = (
            get_sequence_parallel_world_size() > 1 and encoder_hidden_states.shape[1] <= self.replicate_text_max_tokens
        )
        encoder_hidden_states = self._run_mode_blocks(
            "txt",
            self.txt_mode_blocks,
            self._txt_processor,
            encoder_hidden_states,
            txt_early_mod,
            txt_rotary_emb,
            local=replicate_text,
        )
        hidden_states = self._run_mode_blocks(
            "image", self.mode_blocks["image"], self._image_processor, hidden_states, image_early_mod, image_rotary_emb
        )
        streams = [
            (encoder_hidden_states, txt_single_mod, txt_rotary_emb),
            (hidden_states, image_single_mod, image_rotary_emb),
        ]
        if image_cond_hidden_states is not None:
            image_cond_hidden_states = self._run_mode_blocks(
                "image_cond",
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
        joint_rotary_emb = (
            torch.cat([s[2][0] for s in streams], dim=0),
            torch.cat([s[2][1] for s in streams], dim=0),
        )
        joint_seq_len = joint_states.shape[1]

        sp_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()
        pad = (sp_world_size - joint_seq_len % sp_world_size) % sp_world_size
        self._joint_processor.valid_kv_len = joint_seq_len if pad else None
        self._joint_processor.exact_tokens = self._joint_exact_tokens(
            txt_seq_len, joint_seq_len + pad, joint_states.device
        )
        joint_states = chunk_and_pad_sequence(joint_states, sp_rank, sp_world_size, pad, dim=1)
        joint_rotary_emb = self._rotary(
            "joint", tuple(chunk_and_pad_sequence(r, sp_rank, sp_world_size, pad, dim=0) for r in joint_rotary_emb)
        )

        # Text tokens of this rank's chunk: the leading part, possibly none.
        chunk_len = joint_states.shape[1]
        local_txt_len = min(max(txt_seq_len - sp_rank * chunk_len, 0), chunk_len)
        context_states = joint_states[:, :local_txt_len]
        content_states = joint_states[:, local_txt_len:]
        context_mod = self._modulation("joint.txt_mod", txt_single_mod, local_txt_len)
        content_mod = self._content_modulation(
            [s[1] for s in streams[1:]],
            [s[0].shape[1] for s in streams[1:]],
            start=max(sp_rank * chunk_len - txt_seq_len, 0),
            length=chunk_len - local_txt_len,
        )

        for block in self.single_transformer_blocks:
            context_states, content_states = block(
                hidden_states=content_states,
                encoder_hidden_states=context_states,
                temb_mod=content_mod,
                temb_mod_context=context_mod,
                image_rotary_emb=joint_rotary_emb,
            )

        # 7. Output layers on this rank's image tokens, which follow its text tokens. They
        #    are per token, so only the output channels need gathering. Every image token
        #    has the same modulation, so the one image_temb row broadcasts.
        local_image_len = min(max(txt_seq_len + image_seq_len - sp_rank * chunk_len, 0), chunk_len) - local_txt_len
        hidden_states = content_states[:, :local_image_len]
        hidden_states = self.norm_out(hidden_states, image_temb)
        output = self.proj_out(hidden_states)
        output = F.pad(output, (0, 0, local_txt_len, chunk_len - local_txt_len - local_image_len))
        output = gather_and_unpad(output, pad, dim=1)[:, txt_seq_len : txt_seq_len + image_seq_len]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

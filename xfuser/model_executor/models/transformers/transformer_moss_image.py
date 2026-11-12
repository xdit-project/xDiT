"""MOSS-Image DiT and flow-matching scheduler.

The block layout, RoPE, and V2 nearest-grid schedule follow the Apache-2.0
reference published with ``OpenMOSS-Team/MOSS-Image-Base``
(``pipeline_moss_image.py``). Joint DiT attention uses xDiT sequence
parallelism; the two text-refiner blocks stay replicated on every rank.
"""

from __future__ import annotations

import os

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm as _DiffusersRMSNorm
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from torch import nn
from torch.nn import functional as F

from xfuser.model_executor.layers.fused_qk_rope_flydsl import (
    _HAS_FLYDSL,
    flydsl_fused_qk_norm_rope,
    prepare_neox_rope_tables,
)

_COMBINE_QKV_A2A = os.environ.get("MOSS_COMBINE_QKV_A2A", "0") == "1"
# One compiled graph for the whole layer stack instead of one per block.
_FUSED_LAYER_LOOP = os.environ.get("MOSS_FUSED_LAYER_LOOP", "0") == "1"
# Paired (CFG-interleaved) passes share token-wise GEMMs/FFN on the concatenated sequences.
_PAIR_BATCH = os.environ.get("MOSS_PAIR_BATCH", "0") == "1"
# With _PAIR_BATCH: one Ulysses input and one output all-to-all shared by both sequences.
_PAIR_A2A = os.environ.get("MOSS_PAIR_A2A", "0") == "1"


class RMSNorm(_DiffusersRMSNorm):
    # Rounds like diffusers' RMSNorm for bf16 weights (after the norm and after the
    # affine), which is the schedule the FlyDSL QK-norm kernel keys on.
    def __init__(self, dim):
        super().__init__(dim, eps=1e-6)

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def validate_skip_layers(layers, depth, name="skip_layers"):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return frozenset(layers)


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _neox_rope_tables(rope):
    """(cos, sin, cos_pad, sin_fold) for ``flydsl_fused_qk_norm_rope(rope_style="neox")``.

    ``_rotate`` is rotate-half RoPE over the whole head, so the plain pair feeds the
    kernel's fallback and the folded pair feeds the kernel.
    """
    sin, cos = rope[0].chunk(2, -1)
    cos, sin = torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)
    return (cos, sin, *prepare_neox_rope_tables(cos, sin, cos.shape[-1]))


def _sequence_parallel_world_size():
    from xfuser.core.distributed import model_parallel_is_initialized

    if not model_parallel_is_initialized():
        return 1
    from xfuser.core.distributed import get_sequence_parallel_world_size

    return get_sequence_parallel_world_size()


_ULYSSES_BSHD = os.environ.get("MOSS_ULYSSES_BSHD", "0") == "1"
_FUSED_QK_PROLOGUE = os.environ.get("MOSS_FUSED_QK_ROPE", "0") == "1"


def _ulysses_bshd_enabled(x):
    if not (_ULYSSES_BSHD and x.shape[0] == 1 and _sequence_parallel_world_size() > 1):
        return False
    from xfuser.core.distributed import get_ring_parallel_world_size

    return get_ring_parallel_world_size() == 1


def _to_send_layout(x, world):
    """[1, S/P, H, D] -> [P, S/P, H/P, D], the chunk each rank receives being token-major."""
    _, local_len, heads, head_dim = x.shape
    return x.view(local_len, world, heads // world, head_dim).transpose(0, 1).contiguous()


def _ulysses_bshd_attention(query, key, value, valid_kv_len):
    """Ulysses attention exchanging token-major [P, S/P, H/P, D] chunks.

    The input all-to-all then lands as one BSHD [1, S, H/P, D] tensor that the
    kernel reads in place, and the kernel's BSHD output is already the output
    all-to-all's send buffer, so only the pre-exchange permute and the final
    head regroup copy remain. Inputs are send buffers [P, S/P, H/P, D]; the
    output is [1, S/P, H*D].
    """
    from xfuser.model_executor.layers.usp import _sdpa_all_to_all_single, attention

    world, local_len, local_heads, head_dim = query.shape

    def exchange_in(x):
        return _sdpa_all_to_all_single(x).view(1, world * local_len, local_heads, head_dim)

    query, key, value = exchange_in(query), exchange_in(key), exchange_in(value)
    if valid_kv_len:
        key, value = key[:, :valid_kv_len], value[:, :valid_kv_len]
    hidden = attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=False
    )
    hidden = hidden.transpose(1, 2).contiguous().view(world, local_len, local_heads, head_dim)
    hidden = _sdpa_all_to_all_single(hidden)
    return hidden.transpose(0, 1).reshape(1, local_len, world * local_heads * head_dim)


def _attend(q, k, v, rope, heads, *, shard_sequence, valid_kv_len):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    if shard_sequence and _ulysses_bshd_enabled(q):
        world = _sequence_parallel_world_size()
        query = _to_send_layout(_rotate(q.reshape(shape), rope).to(v.dtype), world)
        key = _to_send_layout(_rotate(k.reshape(shape), rope).to(v.dtype), world)
        return _ulysses_bshd_attention(query, key, _to_send_layout(v.reshape(shape), world), valid_kv_len)
    query = _rotate(q.reshape(shape), rope).transpose(1, 2)
    key = _rotate(k.reshape(shape), rope).transpose(1, 2)
    value = v.reshape(shape).transpose(1, 2)
    query = query.to(value.dtype)
    key = key.to(value.dtype)
    if shard_sequence and _sequence_parallel_world_size() > 1:
        from xfuser.model_executor.layers.usp import USP

        kwargs = {"valid_kv_len": valid_kv_len} if valid_kv_len else None
        hidden = USP(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=kwargs,
            combine_qkv_a2a=_COMBINE_QKV_A2A,
        )
    else:
        from xfuser.core.distributed import runtime_state_is_initialized

        if runtime_state_is_initialized():
            from xfuser.model_executor.layers.usp import attention

            hidden = attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            hidden = F.scaled_dot_product_attention(query, key, value)
    return hidden.transpose(1, 2).flatten(2)


def _attend_pair(inputs_a, inputs_b, heads):
    from xfuser.model_executor.layers.usp import USP_pair

    prepared = []
    for q, k, v, rope, valid_kv_len in (inputs_a, inputs_b):
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        value = v.reshape(shape).transpose(1, 2)
        query = _rotate(q.reshape(shape), rope).transpose(1, 2).to(value.dtype)
        key = _rotate(k.reshape(shape), rope).transpose(1, 2).to(value.dtype)
        prepared.append(((query, key, value), {"valid_kv_len": valid_kv_len} if valid_kv_len else None))
    hidden_a, hidden_b = USP_pair(prepared[0][0], prepared[1][0], prepared[0][1], prepared[1][1])
    return hidden_a.transpose(1, 2).flatten(2), hidden_b.transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attend(q, k, v, rope, self.heads, shard_sequence=False, valid_kv_len=None))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, valid_kv_len=None, rope_tables=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["o"].weight.dtype
        y = y.to(dtype)
        if rope_tables is not None and dtype == torch.bfloat16 and _ulysses_bshd_enabled(y):
            attended = self._fused_attend(y, rope_tables, valid_kv_len)
        else:
            q, k, v = self._project_qkv(y)
            q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
            k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
            attended = _attend(q, k, v, rope, self.heads, shard_sequence=True, valid_kv_len=valid_kv_len)
        x = x + self.proj["o"](attended) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]

    def _project_qkv(self, y):
        if "qkv" in self.proj:
            return self.proj["qkv"](y).chunk(3, -1)
        if "qk" in self.proj:
            q, k = self.proj["qk"](y).chunk(2, -1)
            return q, k, self.proj["v"](y)
        return self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)

    def forward_pair(self, x_a, x_b, rope_a, rope_b, modulation, valid_a, valid_b):
        """Two sequences sharing ``modulation``: token-wise projections and the FFN run once on
        their concatenation; attention (and its Ulysses exchange) stays per sequence."""
        split = x_a.shape[1]
        x = torch.cat((x_a, x_b), 1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["o"].weight.dtype
        y = y.to(dtype)
        q, k, v = self._project_qkv(y)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        if _PAIR_A2A and _sequence_parallel_world_size() > 1:
            attn_a, attn_b = _attend_pair(
                (q[:, :split], k[:, :split], v[:, :split], rope_a, valid_a),
                (q[:, split:], k[:, split:], v[:, split:], rope_b, valid_b),
                self.heads,
            )
        else:
            attn_a = _attend(
                q[:, :split], k[:, :split], v[:, :split], rope_a, self.heads, shard_sequence=True, valid_kv_len=valid_a
            )
            attn_b = _attend(
                q[:, split:], k[:, split:], v[:, split:], rope_b, self.heads, shard_sequence=True, valid_kv_len=valid_b
            )
        x = x + self.proj["o"](torch.cat((attn_a, attn_b), 1)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        x = x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]
        return x[:, :split], x[:, split:]

    def _fused_attend(self, y, rope_tables, valid_kv_len):
        world = _sequence_parallel_world_size()
        q, k, value = self._project_qkv(y)
        cos, sin, cos_pad, sin_fold = rope_tables
        query, key = flydsl_fused_qk_norm_rope(
            q.unflatten(-1, (self.heads, -1)),
            k.unflatten(-1, (self.heads, -1)),
            self.q_norm,
            self.k_norm,
            (cos, sin),
            rope_style="neox",
            rotary_dim=cos.shape[-1],
            neox_tables=(cos_pad, sin_fold),
            send_world=world,
        )
        value = _to_send_layout(value.unflatten(-1, (self.heads, -1)), world)
        return _ulysses_bshd_attention(query, key, value, valid_kv_len)

    @torch.no_grad()
    def fuse_qkv_projections(self, names=("q", "k", "v")):
        fused_name = "".join(names)
        if fused_name in self.proj:
            return
        parts = [self.proj.pop(name) for name in names]
        weight = torch.cat([part.weight for part in parts], 0)
        fused = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
        fused.weight.copy_(weight)
        self.proj[fused_name] = fused


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def _shard_sequence(hidden, rope):
    from xfuser.core.distributed import get_sequence_parallel_rank
    from xfuser.model_executor.models.transformers.transformers_utils import chunk_and_pad_sequence

    length = hidden.shape[1]
    world = _sequence_parallel_world_size()
    if world == 1:
        return hidden, rope, length
    pad = (world - (length % world)) % world
    rank = get_sequence_parallel_rank()
    hidden = chunk_and_pad_sequence(hidden, rank, world, pad, dim=1)
    if rope is not None:
        rope = chunk_and_pad_sequence(rope, rank, world, pad, dim=1)
    return hidden, rope, length


# Traced into a graph, this subgroup all-gather returns wrong data under CFG parallelism.
@torch._dynamo.disable
def _gather_sequence(hidden, valid_length):
    from xfuser.model_executor.models.transformers.transformers_utils import gather_and_unpad

    if _sequence_parallel_world_size() == 1:
        return hidden
    pad = hidden.shape[1] * _sequence_parallel_world_size() - valid_length
    return gather_and_unpad(hidden.contiguous(), pad, dim=1)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    """Dense MOSS-Image DiT.

    Each sample keeps its true text length through the refiner and the joint
    blocks, so padding cannot attend. ``skip_layers`` drops whole DiT blocks
    and is how unconditional skip-layer guidance is applied.
    """

    @register_to_config
    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    @torch.no_grad()
    def encode_context(self, encoder_hidden_states, encoder_attention_mask, height, width):
        """Per-sample (refined text tokens, full RoPE table).

        Both depend only on the prompt and the latent grid, not on the timestep or latents, so a
        sampler can compute them once and pass them to every ``forward`` call as ``context``.
        """
        if encoder_hidden_states.ndim != 3:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        device = encoder_hidden_states.device
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        contexts = []
        for index in range(encoder_hidden_states.shape[0]):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            rope = self.rope(height, width, context.shape[1], device).to(self.video_embedder.weight.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            # This rank's shard of the joint-sequence RoPE, materialized once.
            rope = _shard_sequence(rope, None)[0].contiguous()
            rope_tables = _neox_rope_tables(rope) if _FUSED_QK_PROLOGUE and _HAS_FLYDSL else None
            contexts.append((text, rope, rope_tables))
        return tuple(contexts)

    def fuse_qkv_projections(self, names=("q", "k", "v")):
        """Merge each block's projections ``names`` (q,k,v or q,k) into one GEMM. Call after loading weights.

        Fusing only q,k keeps the v GEMM free to overlap the q/k all-to-all under Ulysses.
        """
        for layer in self.layers:
            layer.fuse_qkv_projections(names)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
        pair_encoder_hidden_states=None,
        pair_encoder_attention_mask=None,
        pair_skip_layers=None,
        context=None,
        pair_context=None,
    ):
        """``pair_*`` runs a second text condition on the same latents (batch 1 only),
        interleaving the two passes layer by layer so one pass's Ulysses all-to-all
        overlaps the other's compute. The returned sample stacks [primary, pair]."""
        skip_layers = validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        if context is None:
            context = self.encode_context(encoder_hidden_states, encoder_attention_mask, height, width)
        if len(context) != batch:
            raise ValueError("context must hold one (text, rope, rope_tables) entry per sample.")
        if pair_encoder_hidden_states is not None:
            if batch != 1 or pair_encoder_hidden_states.ndim != 3 or pair_encoder_hidden_states.shape[0] != 1:
                raise ValueError("pair_encoder_hidden_states requires batch size 1.")
            if pair_context is None:
                pair_context = self.encode_context(
                    pair_encoder_hidden_states, pair_encoder_attention_mask, height, width
                )
            pair_skip_layers = validate_skip_layers(pair_skip_layers, len(self.layers))
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        if pair_encoder_hidden_states is not None:
            sample_mod = mod[:1]
            first = self._embed_sample(hidden_states, *context[0])
            second = self._embed_sample(hidden_states, *pair_context[0])
            torch._dynamo.graph_break()
            tokens_a, tokens_b = self._run_layers_pair(first, skip_layers, second, pair_skip_layers, sample_mod)
            sample = torch.cat(
                [
                    self._finalize_sample(tokens_a, first[2], head[:1], height, width),
                    self._finalize_sample(tokens_b, second[2], head[:1], height, width),
                ]
            )
            return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)
        outputs = []
        for index in range(batch):
            text, rope, rope_tables = context[index]
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            tokens = torch.cat((video, text), 1)
            tokens, _, valid_length = _shard_sequence(tokens, None)
            # Without a host sync left in forward, Dynamo traces the whole DiT as one graph, and
            # that graph produces noise under Ulysses 4 + CFG on ROCm (u2 is fine). Splitting
            # it after sharding keeps the result correct.
            torch._dynamo.graph_break()
            sample_mod = mod[index : index + 1]
            if _FUSED_LAYER_LOOP:
                tokens = self._run_layers(tokens, rope, rope_tables, sample_mod, valid_length, skip_layers)
            else:
                for layer_index, layer in enumerate(self.layers):
                    if layer_index not in skip_layers:
                        tokens = layer(tokens, rope, sample_mod, valid_kv_len=valid_length, rope_tables=rope_tables)
            tokens = _gather_sequence(tokens, valid_length)
            tokens = self.final_norm_video(tokens[:, : height * width].to(self.final_norm_video.weight.dtype))
            tokens = tokens * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            tokens = self.final_linear_video(tokens.to(self.final_linear_video.weight.dtype))
            outputs.append(tokens.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)

    @staticmethod
    def _validated_mask(encoder_hidden_states, encoder_attention_mask):
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=encoder_hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        return mask

    def _embed_sample(self, hidden_states, text, rope, rope_tables):
        video = hidden_states.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
        tokens = torch.cat((video, text), 1)
        tokens, _, valid_length = _shard_sequence(tokens, None)
        return tokens, rope, valid_length, rope_tables

    def _finalize_sample(self, tokens, valid_length, sample_head, height, width):
        tokens = _gather_sequence(tokens, valid_length)
        tokens = self.final_norm_video(tokens[:, : height * width].to(self.final_norm_video.weight.dtype))
        tokens = tokens * (1 + sample_head[:, 1, None]) + sample_head[:, 0, None]
        tokens = self.final_linear_video(tokens.to(self.final_linear_video.weight.dtype))
        return tokens.transpose(1, 2).reshape(1, -1, height, width)

    def _run_layers_pair(self, sample_a, skip_a, sample_b, skip_b, sample_mod):
        tokens_a, rope_a, valid_a, tables_a = sample_a
        tokens_b, rope_b, valid_b, tables_b = sample_b
        for layer_index, layer in enumerate(self.layers):
            run_a = layer_index not in skip_a
            run_b = layer_index not in skip_b
            if _PAIR_BATCH and run_a and run_b:
                tokens_a, tokens_b = layer.forward_pair(
                    tokens_a, tokens_b, rope_a, rope_b, sample_mod, valid_a, valid_b
                )
                continue
            if run_a:
                tokens_a = layer(tokens_a, rope_a, sample_mod, valid_kv_len=valid_a, rope_tables=tables_a)
            if run_b:
                tokens_b = layer(tokens_b, rope_b, sample_mod, valid_kv_len=valid_b, rope_tables=tables_b)
        return tokens_a, tokens_b

    def _run_layers(self, tokens, rope, rope_tables, sample_mod, valid_length, skip_layers):
        # Kept free of data-dependent control flow so Dynamo traces it as one graph
        # even though forward() itself graph-breaks on the text mask.
        for layer_index, layer in enumerate(self.layers):
            if layer_index not in skip_layers:
                tokens = layer(tokens, rope, sample_mod, valid_kv_len=valid_length, rope_tables=rope_tables)
        return tokens


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler integration using the training project's V2, nearest-grid schedule."""

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        if num_inference_steps < 1 or self.config.shift <= 0 or self.config.training_shift <= 0:
            raise ValueError("Steps and shifts must be positive.")
        total = self.config.num_train_timesteps

        def grid(count, shift):
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count)
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(num_inference_steps, self.config.shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where(
            (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
            lower,
            upper,
        )
        sigmas = torch.unique_consecutive(training[nearest])
        # Sigmas are already shifted and snapped. A second shift would move them again.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())

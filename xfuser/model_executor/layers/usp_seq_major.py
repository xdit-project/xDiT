"""USP's sequence-major Ulysses exchange, for self-attention on backends that read strided BSHD.

The exchange sends q, k and v as one ``[P, s, b, 3, h/P, d]`` buffer -- destination rank, then
sequence position -- so the received chunks join into strided BSHD views over the whole
sequence, which the kernel reads without a copy, and the output returns as ``[P, s, b, h/P, d]``.
``plan`` decides per call whether the exchange serves it; a call it does not serve takes ``USP``.
"""

from typing import NamedTuple, Optional

import torch

from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import AttnCall, Spec
from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed.fp8_comms import Fp8CommsCall, _per_tensor_quant
from xfuser.model_executor.layers import usp_sdma
from xfuser.model_executor.layers.usp import (
    _compile_capture_bypasses_attention,
    _current_step,
    _has_kv_cache,
    _parallel_degrees,
    _sdpa_all_to_all_single,
    _trim_trailing_kv_padding,
)


def _qkv_send(q, k, v, world_size):
    """BSHD q, k, v as one ``[P, s, b, 3, h/P, d]`` send buffer: destination rank, then
    sequence position, so each received chunk is sequence-major."""
    b, s, h, d = q.shape
    qkv = torch.stack([q, k, v], dim=2).view(b, s, 3, world_size, h // world_size, d)
    return qkv.permute(3, 1, 0, 2, 4, 5).contiguous()


def _qkv_views(received):
    """The q, k, v BSHD views ``[b, P*s, h/P, d]`` of a received ``[P, s, b, 3, h/P, d]`` buffer.

    Source rank and position merge into one sequence dimension of stride ``b * 3 * h/P * d``,
    so nothing is copied; the views are strided, not contiguous.
    """
    joined = received.flatten(0, 1)
    return tuple(joined[:, :, index].transpose(0, 1) for index in range(3))


def attention_output(received):
    """The ``[b, s, P * h/P * d]`` rows of a received ``[P, s, b, h/P, d]`` attention output."""
    world_size, s, b, heads, d = received.shape
    return received.permute(2, 1, 0, 3, 4).reshape(b, s, world_size * heads * d)


class SeqMajorOutput(NamedTuple):
    """``attention``'s result: the output exchange's receive buffer ``[P, s, b, h/P, d]``,
    whose chunk ``p`` holds heads ``p * h/P`` onwards for this rank's ``s`` tokens, and the
    one-element descale of its values when they are FP8 (fp8 comms), else None.

    A consumer may read ``received`` in place, dequantizing as ``rows`` does.
    """

    received: torch.Tensor
    descale: Optional[torch.Tensor]

    def rows(self, dtype: torch.dtype) -> torch.Tensor:
        """The ``[b, s, h * d]`` rows in ``dtype``, with one copy; FP8 values dequantize as
        ``fp8_comms_output_all_to_all`` dequantizes ``USP``'s output exchange."""
        rows = attention_output(self.received)
        if self.descale is not None:
            rows = rows.float() * self.descale
        return rows.to(dtype)


class SeqMajorPlan(NamedTuple):
    """How ``attention`` runs one attention call; see ``plan``."""

    spec: Spec
    world_size: int
    # The layer's calibrated fp8-comms scales when q, k, v and the output travel in FP8.
    fp8_comms: Optional[Fp8CommsCall]
    # The call's attention_kwargs, limited to _ATTENTION_KWARGS.
    attention_kwargs: dict


# The attention_kwargs the exchange honours. It builds no VarlenPacking and carries no
# ULYSSES_EXTRA_INPUTS_KEY tensors, so a call with masks, varlen packing or extra Ulysses
# inputs takes USP.
_ATTENTION_KWARGS = frozenset({"valid_kv_len"})


def plan(attn_layer, heads, attention_kwargs=None, backend=None) -> Optional[SeqMajorPlan]:
    """The plan of a self-attention call that may take the sequence-major exchange, or None
    when it takes ``USP``.

    The exchange serves non-causal self-attention without dropout, in Ulysses-only calls on a
    backend whose spec ``reads_strided_bshd``, without what ``USP`` adds around its exchange
    (joint tensors, KV caching, head balancing, SDMA, the compile-capture bypass). Of
    ``attention_kwargs`` it honours ``valid_kv_len`` only; any other entry that is not None
    makes the call take ``USP``. While ``attn_layer``'s model calibrates fp8 comms its calls
    take ``USP``, which observes them; once calibrated, a backend that ``accepts_prequantized``
    exchanges FP8 q, k, v and output.
    """
    attention_kwargs = {name: value for name, value in (attention_kwargs or {}).items() if value is not None}
    if not attention_kwargs.keys() <= _ATTENTION_KWARGS:
        return None
    if _compile_capture_bypasses_attention() or usp_sdma.selected():
        return None
    world_size, ring_world_size = _parallel_degrees()
    if world_size <= 1 or ring_world_size > 1 or heads % world_size:
        return None
    runtime_state = get_runtime_state()
    backend = backend if backend is not None else runtime_state.attention_backend
    spec = attention_registry.find(backend)
    if spec is None or not spec.reads_strided_bshd or _has_kv_cache(attn_layer):
        return None
    if spec.head_balanced and runtime_state.runtime_config.use_sparseattn_head_balance:
        return None
    model_state = _fp8_comms_model_state(runtime_state.fp8_comms, attn_layer)
    if model_state is not None and not model_state.synced:
        return None
    fp8_comms = None
    if model_state is not None and spec.accepts_prequantized:
        fp8_comms = Fp8CommsCall(
            attn_layer.fp8_q_scale, attn_layer.fp8_k_scale, attn_layer.fp8_v_scale, attn_layer.fp8_o_scale
        )
    return SeqMajorPlan(spec, world_size, fp8_comms, attention_kwargs)


def _fp8_comms_model_state(fp8_comms, attn_layer):
    """The fp8-comms calibration state of ``attn_layer``'s model, or None for a layer without
    fp8 comms."""
    owner = getattr(attn_layer, "fp8_comms_owner", None)
    if fp8_comms is None or owner is None or not hasattr(attn_layer, "fp8_comms_layer_idx"):
        return None
    return fp8_comms.get_model_state(owner)


def send(query, key, value, plan: SeqMajorPlan):
    """``attention``'s ``[P, s, b, 3, h/P, d]`` send buffer from BSHD q, k, v. Under the
    plan's fp8 comms, q and k are rotated for the backend and q, k, v quantized with the
    layer's scales, as ``USP``'s FP8 exchange quantizes them."""
    fp8_comms = plan.fp8_comms
    if fp8_comms is not None and plan.spec.prequant_rotate is not None:
        query, key = plan.spec.prequant_rotate(query, key)
    buffer = _qkv_send(query, key, value, plan.world_size)
    if fp8_comms is None:
        return buffer
    scales = torch.cat([fp8_comms.q_scale, fp8_comms.k_scale, fp8_comms.v_scale]).view(3, 1, 1)
    return _per_tensor_quant(buffer, scales)[0]


def attention(send, plan: SeqMajorPlan) -> SeqMajorOutput:
    """Ulysses attention through the sequence-major exchange of ``send``, built by ``send()``
    or by a kernel writing the same layout and quantization.

    Same result as ``USP`` with the plan's backend and attention_kwargs: the selected spec runs
    the call, through its ``accepts``, on the received views. ``valid_kv_len`` drops a uniform
    trailing K/V pad, as in ``USP``.
    """
    fp8_comms = plan.fp8_comms
    received = _sdpa_all_to_all_single(send)
    query, key, value = (view.transpose(1, 2) for view in _qkv_views(received))
    kwargs = dict(plan.attention_kwargs)
    if fp8_comms is not None:
        kwargs |= {
            "pre_quantized": True,
            "q_descale": fp8_comms.q_scale,
            "k_descale": fp8_comms.k_scale,
            "v_descale": fp8_comms.v_scale,
        }
    key, value, kwargs = _trim_trailing_kv_padding(key, value, kwargs, plan.spec.type)
    batch, heads, seq_len, head_dim = query.shape
    dtype = torch.bfloat16 if fp8_comms is not None else query.dtype
    # The output send buffer: chunk p holds rank p's tokens.
    out = torch.empty(seq_len, batch, heads, head_dim, dtype=dtype, device=query.device)
    call = AttnCall(
        ulysses_world_size=plan.world_size,
        step=_current_step(),
        attention_kwargs=kwargs,
        seq_major_out=out.transpose(0, 1),
    )
    plan.spec.run(query, key, value, call)
    descale = None
    if fp8_comms is not None:
        out, descale = _per_tensor_quant(out, fp8_comms.o_scale)
    received = _sdpa_all_to_all_single(out)
    return SeqMajorOutput(received.view(plan.world_size, seq_len // plan.world_size, batch, heads, head_dim), descale)

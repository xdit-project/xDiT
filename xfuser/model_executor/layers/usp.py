# This file implements USP with torch version >= '2.5.0'
import functools
from typing import NamedTuple, Optional, Tuple

import torch
import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    model_parallel_is_initialized,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_ring_parallel_rank,
    get_ulysses_parallel_rank,
    get_runtime_state,
    runtime_state_is_initialized,
)

from xfuser.compat import version_at_least
from xfuser.core.attention.backends.aiter_mha_v4.attention_a2a import (
    trim_aiter_mha_v4_a2a_padding,
)
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.logger import init_logger
from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import VarlenPacking
from xfuser.core.attention.spec import AttnCall
from xfuser.core.attention.spec import Sparsity
from xfuser.core.attention.spec import Spec
from xfuser.core.distributed.fp8_comms import (
    Fp8CommsCall,
    _per_tensor_quant,
    fp8_attention_kwargs,
    fp8_comms_input_all_to_all,
    fp8_comms_output_all_to_all,
    fp8_observe_output,
)
from xfuser.core.sparge_attention.head_balance import (
    apply_head_balance,
    revert_head_balance,
)
from xfuser.core.distributed.attention_a2a import (
    fused_a2a_input,
    fused_a2a_input_role,
    get_fused_a2a_codecs,
    get_fused_a2a_mode,
    get_fused_a2a_profile,
    launch_attention_a2a_packed,
    use_fused_a2a_interleave,
    use_fused_a2a_packed,
)
from xfuser.model_executor.layers import usp_sdma

# Sparge backends whose kernel cost can be load-balanced across Ulysses ranks.
# These all build a block mask via _build_sparge_block_mask and write the
# per-head cost into the head-balance "cost sink". Non-sparge backends are
# excluded so head balancing is a clean no-op for them.
_HEAD_BALANCE_BACKENDS = attention_registry.types_where(head_balanced=True)

_FP8_NCCL_NEEDS_VIEW = not version_at_least(torch.__version__, "2.11.0")
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)
_warned_fp8_comms_missing_attn = False
logger = init_logger(__name__)


def _compile_capture_bypasses_attention() -> bool:
    return runtime_state_is_initialized() and getattr(
        get_runtime_state(),
        "_xdit_compile_capture_bypass_attention",
        False,
    )


def _warn_fp8_comms_missing_attn():
    global _warned_fp8_comms_missing_attn
    if _warned_fp8_comms_missing_attn:
        return
    runtime_state = get_runtime_state()
    if runtime_state.fp8_comms is None or get_ulysses_parallel_world_size() <= 1:
        return
    _warned_fp8_comms_missing_attn = True
    logger.warning(
        "use_fp8_comms is enabled but USP was called without attn_layer "
        "(or a module with fp8 scale buffers). FP8 all-to-all will not run on this path."
    )


# A backend that needs per-head tensors of its own alongside query/key/value
# lists their ``attention_kwargs`` keys under this one, and USP carries them
# through the same Ulysses exchange without knowing what they mean.
ULYSSES_EXTRA_INPUTS_KEY = "ulysses_extra_inputs"


def ring_attn(
    attention_function, query, key, value, dropout_p=0.0, is_causal=False, joint_attn_kwargs=None, attention_kwargs=None
):
    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "joint_attn_kwargs": joint_attn_kwargs,
        "attention_kwargs": attention_kwargs,
    }
    if version_at_least(torch.__version__, "2.6.0"):
        from torch.distributed.tensor.experimental._attention import _cp_options

        _cp_options.enable_load_balance = False
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.flatten()
    # NCCL does not support FP8 collectives before PyTorch 2.11, view as uint8 (same width) for the transfer.
    if _FP8_NCCL_NEEDS_VIEW and x_dtype in _FP8_DTYPES:
        x = x.view(torch.uint8)
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.view(x_dtype).reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _combined_qkv_all_to_all(q, k, v, *extra):
    """Concatenate query, key, value tensors and perform a single all-to-all communication.

    Extra tensors shaped like q ride the same exchange and are returned after v.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return (q, k, v, *extra)

    assert q.ndim == 4, f"q must have 4 dimensions, got {q.ndim}"
    b, h, s, d = q.shape
    assert h % world_size == 0, f"h must be divisible by world_size, got {h} and {world_size}"

    n = 3 + len(extra)
    # [n, b, h, s, d]
    qkv = torch.stack([q, k, v, *extra], dim=0)
    # [n, b, P, h/P, s, d]
    qkv = qkv.view(n, b, world_size, h // world_size, s, d)
    # [P, n, b, h/P, s, d]
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()

    qkv = _sdpa_all_to_all_single(qkv)

    # [n, b, h/P, P*s, d]  — reshape directly avoids the intermediate
    # contiguous copy that the separate permute+view required.
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).reshape(n, b, h // world_size, -1, d)

    return torch.unbind(qkv, dim=0)


def _seq_major_qkv_send(q, k, v, world_size):
    """BSHD q, k, v as one ``[P, s, b, 3, h/P, d]`` send buffer: destination rank, then
    sequence position, so each received chunk is sequence-major."""
    b, s, h, d = q.shape
    qkv = torch.stack([q, k, v], dim=2).view(b, s, 3, world_size, h // world_size, d)
    return qkv.permute(3, 1, 0, 2, 4, 5).contiguous()


def _seq_major_qkv_views(received):
    """The q, k, v BSHD views ``[b, P*s, h/P, d]`` of a received ``[P, s, b, 3, h/P, d]`` buffer.

    Source rank and position merge into one sequence dimension of stride ``b * 3 * h/P * d``,
    so nothing is copied; the views are strided, not contiguous.
    """
    joined = received.flatten(0, 1)
    return tuple(joined[:, :, index].transpose(0, 1) for index in range(3))


def seq_major_attention_output(received):
    """The ``[b, s, P * h/P * d]`` rows of a received ``[P, s, b, h/P, d]`` attention output."""
    world_size, s, b, heads, d = received.shape
    return received.permute(2, 1, 0, 3, 4).reshape(b, s, world_size * heads * d)


class SeqMajorOutput(NamedTuple):
    """``usp_seq_major``'s result: the output exchange's receive buffer ``[P, s, b, h/P, d]``,
    whose chunk ``p`` holds heads ``p * h/P`` onwards for this rank's ``s`` tokens, and the
    one-element descale of its values when they are FP8 (fp8 comms), else None.

    A consumer may read ``received`` in place, dequantizing as ``rows`` does.
    """

    received: torch.Tensor
    descale: Optional[torch.Tensor]

    def rows(self, dtype: torch.dtype) -> torch.Tensor:
        """The ``[b, s, h * d]`` rows in ``dtype``, with one copy; FP8 values dequantize as
        ``fp8_comms_output_all_to_all`` dequantizes ``USP``'s output exchange."""
        rows = seq_major_attention_output(self.received)
        if self.descale is not None:
            rows = rows.float() * self.descale
        return rows.to(dtype)


class SeqMajorPlan(NamedTuple):
    """How ``usp_seq_major`` runs one attention call; see ``usp_seq_major_plan``."""

    spec: Spec
    world_size: int
    # The layer's calibrated fp8-comms scales when q, k, v and the output travel in FP8.
    fp8_comms: Optional[Fp8CommsCall]
    # The call's attention_kwargs, limited to _SEQ_MAJOR_ATTENTION_KWARGS.
    attention_kwargs: dict


# The attention_kwargs the sequence-major exchange honours. It builds no VarlenPacking and
# carries no ULYSSES_EXTRA_INPUTS_KEY tensors, so a call with masks, varlen packing or extra
# Ulysses inputs takes USP.
_SEQ_MAJOR_ATTENTION_KWARGS = frozenset({"valid_kv_len"})


def usp_seq_major_plan(attn_layer, heads, attention_kwargs=None, backend=None) -> Optional[SeqMajorPlan]:
    """The plan of a self-attention call that may take USP's sequence-major Ulysses exchange,
    or None when it takes ``USP``.

    The sequence-major exchange sends q, k and v as one ``[P, s, b, 3, h/P, d]`` buffer --
    destination rank, then sequence position -- so the received chunks join into strided BSHD
    views over the whole sequence, which the kernel reads without a copy, and the output
    returns as ``[P, s, b, h/P, d]``. It serves non-causal self-attention without dropout, in
    Ulysses-only calls on a backend whose spec ``reads_strided_bshd``, without what ``USP``
    adds around its exchange (joint tensors, KV caching, head balancing, SDMA, the
    compile-capture bypass). Of ``attention_kwargs`` it honours ``valid_kv_len`` only; any
    other entry that is not None makes the call take ``USP``. While ``attn_layer``'s model
    calibrates fp8 comms its calls take ``USP``, which observes them; once calibrated, a
    backend that ``accepts_prequantized`` exchanges FP8 q, k, v and output.
    """
    attention_kwargs = {name: value for name, value in (attention_kwargs or {}).items() if value is not None}
    if not attention_kwargs.keys() <= _SEQ_MAJOR_ATTENTION_KWARGS:
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
    if spec.head_balanced and runtime_state.runtime_config.use_spargeattn_head_balance:
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


def usp_seq_major_send(query, key, value, plan: SeqMajorPlan):
    """``usp_seq_major``'s ``[P, s, b, 3, h/P, d]`` send buffer from BSHD q, k, v. Under the
    plan's fp8 comms, q and k are rotated for the backend and q, k, v quantized with the
    layer's scales, as ``USP``'s FP8 exchange quantizes them."""
    fp8_comms = plan.fp8_comms
    if fp8_comms is not None and plan.spec.prequant_rotate is not None:
        query, key = plan.spec.prequant_rotate(query, key)
    send = _seq_major_qkv_send(query, key, value, plan.world_size)
    if fp8_comms is None:
        return send
    scales = torch.cat([fp8_comms.q_scale, fp8_comms.k_scale, fp8_comms.v_scale]).view(3, 1, 1)
    return _per_tensor_quant(send, scales)[0]


def usp_seq_major(send, plan: SeqMajorPlan) -> SeqMajorOutput:
    """Ulysses attention through the sequence-major exchange of ``send``, built by
    ``usp_seq_major_send`` or by a kernel writing the same layout and quantization.

    Same result as ``USP`` with the plan's backend and attention_kwargs: the selected spec runs
    the call, through its ``accepts``, on the received views. ``valid_kv_len`` drops a uniform
    trailing K/V pad, as in ``USP``.
    """
    fp8_comms = plan.fp8_comms
    received = _sdpa_all_to_all_single(send)
    query, key, value = (view.transpose(1, 2) for view in _seq_major_qkv_views(received))
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
    call = AttnCall(ulysses_world_size=plan.world_size, attention_kwargs=kwargs, seq_major_out=out.transpose(0, 1))
    plan.spec.run(query, key, value, call)
    descale = None
    if fp8_comms is not None:
        out, descale = _per_tensor_quant(out, fp8_comms.o_scale)
    received = _sdpa_all_to_all_single(out)
    return SeqMajorOutput(received.view(plan.world_size, seq_len // plan.world_size, batch, heads, head_dim), descale)


def _combined_gqa_qkv_all_to_all(q, k, v, *extra):
    """Exchange GQA Q/K/V in one collective without expanding KV heads.

    Every destination receives its contiguous query-head shard and the
    corresponding compact KV-head shard from every source rank. Query-shaped
    extra tensors can share the same collective.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return (q, k, v, *extra)

    tensors = (q, k, v, *extra)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("GQA all-to-all inputs must all have four dimensions.")
    if k.shape != v.shape:
        raise ValueError(
            f"GQA key and value tensors must have identical shapes, got {tuple(k.shape)} and {tuple(v.shape)}."
        )

    batch_size, _, _, head_dim = q.shape
    if any(tensor.shape[0] != batch_size or tensor.shape[-1] != head_dim for tensor in tensors):
        raise ValueError("GQA all-to-all inputs must share batch and head dimensions.")
    if any(tensor.shape[1] % world_size != 0 for tensor in tensors):
        raise ValueError("Every GQA head count must be divisible by the Ulysses world size.")
    if any(tensor.shape != q.shape for tensor in extra):
        raise ValueError("Extra GQA all-to-all inputs must match the query shape.")

    packed_chunks = []
    metadata = []
    for tensor in tensors:
        _, heads, sequence_length, tensor_head_dim = tensor.shape
        local_heads = heads // world_size
        # Match _ft_c_input_all_to_all's destination-major layout, then pack
        # unequal Q and KV payloads into one equally split collective.
        packed_chunks.append(tensor.permute(1, 0, 2, 3).contiguous().reshape(world_size, -1))
        metadata.append((local_heads, sequence_length, tensor_head_dim))

    chunk_sizes = [chunk.shape[1] for chunk in packed_chunks]
    exchanged = _sdpa_all_to_all_single(torch.cat(packed_chunks, dim=1))

    outputs = []
    for chunk, (local_heads, sequence_length, tensor_head_dim) in zip(exchanged.split(chunk_sizes, dim=1), metadata):
        outputs.append(
            chunk.view(
                world_size,
                local_heads,
                batch_size,
                sequence_length,
                tensor_head_dim,
            )
            .permute(2, 1, 0, 3, 4)
            .reshape(batch_size, local_heads, -1, tensor_head_dim)
        )
    return tuple(outputs)


def _repeat_kv_heads(key, value, repeats):
    if repeats == 1:
        return key, value
    return (
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
    )


def _validate_gqa_params(
    query,
    key,
    value,
    kv_head_repeat,
    joint_strategy,
):
    if isinstance(kv_head_repeat, bool) or not isinstance(kv_head_repeat, int):
        raise TypeError("kv_head_repeat must be an integer.")
    if kv_head_repeat < 1:
        raise ValueError("kv_head_repeat must be at least 1.")
    if kv_head_repeat == 1:
        return

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("GQA query, key, and value must have four dimensions.")
    if key.shape[1] != value.shape[1]:
        raise ValueError("GQA key and value must have the same head count.")
    if query.shape[1] != key.shape[1] * kv_head_repeat:
        raise ValueError(
            f"Query heads ({query.shape[1]}) must equal KV heads "
            f"({key.shape[1]}) times kv_head_repeat ({kv_head_repeat})."
        )

    ulysses_world_size = get_ulysses_parallel_world_size()
    if ulysses_world_size > 1 and key.shape[1] % ulysses_world_size != 0:
        raise ValueError(
            f"KV heads ({key.shape[1]}) must be divisible by the Ulysses world size ({ulysses_world_size})."
        )
    if joint_strategy is not None:
        raise NotImplementedError("GQA KV repetition does not support joint tensors.")


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = joint_key.shape[1] // ulysses_world_size
    joint_key = joint_key.transpose(1, 2)
    joint_value = joint_value.transpose(1, 2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
        :,
    ].transpose(1, 2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
        :,
    ].transpose(1, 2)
    return joint_key, joint_value


def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor


def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value


def _has_kv_cache(attn_layer) -> bool:
    """Return whether PipeFusion registered a KV cache for this attention layer."""
    return attn_layer is not None and get_cache_manager().has_cache_entry(attn_layer)


def _serves_packed_keys(backend, key, attention_kwargs) -> bool:
    """Whether the selected backend will honour the packing itself."""
    spec = attention_registry.find(backend)
    if spec is None:
        return False
    probe = AttnCall(
        varlen=VarlenPacking.from_kwargs(attention_kwargs),
        attention_kwargs=attention_kwargs,
    )
    return spec.rejects(key, key, key, probe) is None


def _trim_trailing_kv_padding(key, value, attention_kwargs, backend=None):
    """Slice a uniform padded K/V suffix while retaining every query row.

    Returns the kwargs the backend should see: once the pad has been sliced
    the packing has been honoured, so it is cleared -- left in place, it would
    describe keys that are no longer there. Keys are set to None rather than
    removed, because torch.compile guards on this dict's key set.
    """
    kwargs = attention_kwargs or {}
    valid_kv_len = kwargs.get("valid_kv_len")
    if valid_kv_len is None:
        return key, value, attention_kwargs
    if kwargs.get("indices_k") is not None and _serves_packed_keys(backend, key, kwargs):
        # A backend that accepts packed keys handles the pad itself, and
        # slicing first would leave its indices pointing past the end of K.
        return key, value, attention_kwargs
    if not 0 < valid_kv_len <= key.shape[2]:
        raise ValueError(f"valid_kv_len must be in [1, {key.shape[2]}], got {valid_kv_len}.")

    consumed = kwargs
    if kwargs.get("indices_k") is not None:
        consumed = dict(kwargs)
        consumed["indices_k"] = None
        consumed["cu_seqlens_k"] = None
        consumed["max_seqlen_k"] = None
    return key[:, :, :valid_kv_len], value[:, :, :valid_kv_len], consumed


# Spatial sparse setup strips the rows past the published `thw` layout itself
# before reordering Q and K/V. SSTA also drops `encoder_sp_padding` from text.
# Trimming K/V first breaks the equal Q/K/V lengths required by these paths.
_SPARSITY_STRIPPING_SP_PAD = (Sparsity.SSTA, Sparsity.SPARGE, Sparsity.VSA)


def _backend_strips_sp_pad(backend, attention_kwargs) -> bool:
    spec = attention_registry.find(backend)
    return spec is not None and spec.sparsity in _SPARSITY_STRIPPING_SP_PAD and attention_kwargs.get("thw") is not None


def _ring_trim_trailing_kv_padding(attention_function, valid_kv_len, shard_len):
    """Drop the trailing pad from whichever K/V shard each ring step attends to.

    No single slice before the ring is right, because the shards rotate. With
    yunchang's Ulysses-low layout the ring shards are contiguous in global
    order, so only the last ring rank's shard ends in pad, and torch's ring
    hands step ``i`` the shard of ring rank ``(rank - i) % size``. The slice is
    taken inside the step, so the shard sent on keeps the equal shape the ring
    exchange needs. Steps are counted in place, as concat_joint_tensors_decorator
    does, which is how the count stays traceable by torch.compile.
    """
    ring_rank = get_ring_parallel_rank()
    ring_size = get_ring_parallel_world_size()
    if valid_kv_len <= (ring_size - 1) * shard_len:
        # A shard holding nothing but pad would need an empty attention call.
        raise NotImplementedError(
            f"The SP pad spans more than one ring shard (valid_kv_len={valid_kv_len}, "
            f"shard length {shard_len}, ring degree {ring_size})."
        )
    state = [0]  # {"step": 0}

    @functools.wraps(attention_function)
    def wrapper(query, key, value, *args, **kwargs):
        source = (ring_rank - state[0]) % ring_size
        state[0] += 1
        valid = valid_kv_len - source * shard_len
        if valid < shard_len:
            key, value = key[:, :, :valid], value[:, :, :valid]
        return attention_function(query, key, value, *args, **kwargs)

    return wrapper


def _get_attention_function(backend=None):
    """
    Get the attention function based on the runtime state or from a given explicit backend.
    """
    if backend is not None:
        attention_backend = backend
    else:
        attention_backend = get_runtime_state().attention_backend

    spec = attention_registry.find(attention_backend)
    if spec is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return concat_joint_tensors_decorator(_spec_adapter(spec))


def _parallel_degrees() -> Tuple[int, int]:
    """(ulysses, ring) sequence-parallel degrees, or the single-rank default.

    `attention()` is the entry point for calls that need no sequence
    parallelism, and it is reached before -- or entirely without -- an
    initialised process group. Asking for the degrees there must not be what
    fails.
    """
    if not model_parallel_is_initialized():
        return 1, 1
    return get_ulysses_parallel_world_size(), get_ring_parallel_world_size()


def _spec_adapter(spec):
    """Present a Spec with the calling convention ring_attn and the direct
    call sites already use.

    This is also where the two things kernels used to fetch for themselves get
    supplied: the varlen packing, previously rebuilt inside every kernel from
    attention_kwargs, and the sequence-parallel degrees, previously read from
    globals. Both now arrive on the call, which is what makes a backend
    testable without an initialised process group.
    """

    def call(query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=None):
        kwargs = attention_kwargs if attention_kwargs is not None else {}
        ulysses_world_size, ring_world_size = _parallel_degrees()
        return spec.run(
            query,
            key,
            value,
            AttnCall(
                dropout_p=dropout_p,
                is_causal=is_causal,
                varlen=VarlenPacking.from_kwargs(kwargs),
                ulysses_world_size=ulysses_world_size,
                ring_world_size=ring_world_size,
                attention_kwargs=kwargs,
            ),
        )

    return call


def concat_joint_tensors_decorator(func):
    """
    Decorator to handle joint tensor concatenation
    This is needed for ring attention with 'rear' joint_strategy, as it
    needs to concat the joint tensors before calling the attention function
    but only on the last step.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query, key, value = args[0:3]
        is_causal = kwargs.get("is_causal")
        dropout_p = kwargs.get("dropout_p")
        joint_attn_kwargs = kwargs.get("joint_attn_kwargs", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)

        if joint_attn_kwargs is not None:
            joint_strategy = joint_attn_kwargs.get("joint_strategy", None)
            joint_key = joint_attn_kwargs.get("joint_key", None)
            joint_value = joint_attn_kwargs.get("joint_value", None)
            step = joint_attn_kwargs.get("step", 0)
            total_steps = joint_attn_kwargs.get("total_steps", 1)
            if (joint_strategy == "front" and step == 0) or (joint_strategy == "rear" and step == total_steps - 1):
                key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
                value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)
            joint_attn_kwargs["step"] = step + 1  # In place increment step

        return func(query, key, value, dropout_p=dropout_p, is_causal=is_causal, attention_kwargs=attention_kwargs)

    return wrapper


def _ulysses_extra_inputs(attention_kwargs, query):
    """Return the (name, tensor) pairs the backend asked to join the Ulysses exchange."""
    if not attention_kwargs:
        return []

    extras = []
    for name in attention_kwargs.get(ULYSSES_EXTRA_INPUTS_KEY) or ():
        tensor = attention_kwargs.get(name)
        if tensor is None:
            continue
        if tensor.shape != query.shape:
            raise ValueError(
                f"attention_kwargs['{name}'] must match the query shape to be "
                f"exchanged with it, got {tuple(tensor.shape)} vs {tuple(query.shape)}."
            )
        extras.append((name, tensor))
    return extras


def usp_fused_a2a_input_role(
    input,
    role,
    pending=None,
    valid_kv_len=None,
):
    """Submit one sequence-major Wan role to the active Ulysses A2A input hop."""
    return fused_a2a_input_role(
        input,
        role,
        PROCESS_GROUP.ULYSSES_PG,
        get_ulysses_parallel_rank(),
        pending,
        valid_kv_len,
    )


def usp_fused_a2a_input_q(input, valid_kv_len=None):
    return usp_fused_a2a_input_role(
        input,
        0,
        valid_kv_len=valid_kv_len,
    )


def usp_fused_a2a_input_k(input, pending, valid_kv_len=None):
    return usp_fused_a2a_input_role(
        input,
        1,
        pending,
        valid_kv_len,
    )


def usp_fused_a2a_input_v(input, pending, valid_kv_len=None):
    return usp_fused_a2a_input_role(
        input,
        2,
        pending,
        valid_kv_len,
    )


def _attention_a2a_usp(
    query,
    key,
    value,
    *,
    dropout_p,
    is_causal,
    joint_strategy,
    attn_layer,
    attention_kwargs,
    kv_head_repeat,
    pending,
):
    """Run the deliberately narrow AITER/MORI Ulysses input path."""
    runtime_state = get_runtime_state()
    ulysses_world_size = get_ulysses_parallel_world_size()
    ring_world_size = get_ring_parallel_world_size()

    if ulysses_world_size <= 1:
        raise NotImplementedError("Attention A2A requires Ulysses parallelism")
    if ring_world_size != 1:
        raise NotImplementedError("Attention A2A does not support ring parallelism")
    if joint_strategy is not None:
        raise NotImplementedError("Attention A2A does not support joint attention")
    if kv_head_repeat != 1:
        raise NotImplementedError("Attention A2A does not support grouped-query KV repetition")
    if runtime_state.runtime_config.use_spargeattn_head_balance:
        raise NotImplementedError("Attention A2A does not support sparse head balancing")
    if query.shape != key.shape or query.shape != value.shape:
        raise NotImplementedError("Attention A2A requires equal self-attention Q/K/V shapes")
    if not use_fused_a2a_packed():
        raise RuntimeError("Attention A2A requires packed per-role transport results")

    profile = get_fused_a2a_profile()
    if profile in ("none", "auto"):
        raise RuntimeError("Attention A2A runtime did not activate a concrete profile")
    if pending is not None and not use_fused_a2a_interleave():
        raise ValueError("interleaved input requires eligible Attention A2A transport")

    unsupported = []
    if _ulysses_extra_inputs(attention_kwargs, query):
        unsupported.append("extra Ulysses inputs")
    if _has_kv_cache(attn_layer):
        unsupported.append("KV caching")
    if dropout_p not in (None, 0.0):
        unsupported.append("dropout")
    if is_causal:
        unsupported.append("causal masking")
    if (attention_kwargs or {}).get("indices_k") is not None:
        unsupported.append("arbitrary varlen key packing")
    if unsupported:
        raise NotImplementedError("packed Attention A2A does not support " + ", ".join(unsupported))

    valid_kv_len = (attention_kwargs or {}).get("valid_kv_len")
    (query, key, value), packed_scales = fused_a2a_input(
        query,
        key,
        value,
        PROCESS_GROUP.ULYSSES_PG,
        get_ulysses_parallel_rank(),
        softmax_scale=query.shape[-1] ** -0.5,
        pending=pending,
        valid_kv_len=valid_kv_len,
    )
    key, value, packed_scales = trim_aiter_mha_v4_a2a_padding(
        key,
        value,
        packed_scales,
        valid_kv_len,
        get_fused_a2a_codecs(),
        profile,
    )
    output = launch_attention_a2a_packed(
        query,
        key,
        value,
        *packed_scales,
        profile,
        query.shape[-1] ** -0.5,
    ).transpose(1, 2)
    return _ft_c_output_all_to_all(output)


def USP(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    joint_query: torch.Tensor | None = None,
    joint_key: torch.Tensor | None = None,
    joint_value: torch.Tensor | None = None,
    joint_strategy: str | None = None,
    attn_layer=None,
    combine_qkv_a2a: bool | None = None,
    backend=None,
    attention_kwargs: dict | None = None,
    head_balance_layer=None,
    kv_head_repeat: int = 1,
    attention_a2a_enabled: bool = False,
    attention_a2a_pending=None,
):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    Explicit backend can be provided to specify the attention backend to use.

    ``attention_kwargs["valid_kv_len"]`` declares the global number of valid
    main K/V tokens before uniform SP suffix padding, across both Ulysses and
    ring ranks. Joint tokens are excluded from this count and remain intact.

    ``attn_layer`` (optional): the attention module. Used to resolve Ulysses FP8
    communication state and to update the PipeFusion KV cache for modules that
    registered one. Callers only pass the module; USP no-ops when FP8 comms is
    off, the module has no scales, or ``joint_strategy`` is set, and it skips
    the KV cache outside PipeFusion, where no entry exists.

    ``head_balance_layer`` (optional): a stable per-layer handle (e.g. the
    attention module). When provided and --use_spargeattn_head_balance is set, the
    Ulysses head dimension is permuted so each rank gets a cost-balanced subset
    of heads (block-sparse load balancing); the permutation is inverted on the
    output. No-op for non-sparse backends (no cost is published) and for ring/
    joint paths. Also used as the FP8-comms module when ``attn_layer`` is None
    (KV cache already updated by the caller).

    ``kv_head_repeat`` keeps grouped-query attention K/V heads compact during
    the Ulysses input all-to-all, then repeats each local KV head immediately
    before attention. With ``combine_qkv_a2a=True``, unequal Q and KV payloads
    are packed into one collective.
    """
    if combine_qkv_a2a is None:
        combine_qkv_a2a = False
    _validate_gqa_params(query, key, value, kv_head_repeat, joint_strategy)

    if _compile_capture_bypasses_attention():
        # Capture needs the public attention output contract, not attention
        # values. Bypass before communication or backend-specific quantization
        # so every activation dtype is preserved without guessing how a
        # low-precision backend represents Q/K/V internally.
        if joint_strategy:
            query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        return torch.zeros_like(query)

    use_attention_a2a = attention_a2a_enabled and get_fused_a2a_mode() > 0
    if attention_a2a_pending is not None and not use_attention_a2a:
        raise ValueError("interleaved input requires eligible Attention A2A transport")
    if use_attention_a2a:
        return _attention_a2a_usp(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            joint_strategy=joint_strategy,
            attn_layer=attn_layer,
            attention_kwargs=attention_kwargs,
            kv_head_repeat=kv_head_repeat,
            pending=attention_a2a_pending,
        )

    runtime_state = get_runtime_state()
    ulysses_world_size = get_ulysses_parallel_world_size()
    ring_world_size = get_ring_parallel_world_size()
    hb_backend = backend if backend is not None else runtime_state.attention_backend
    attention_function = _get_attention_function(backend=backend)

    fp8_module = attn_layer if attn_layer is not None else head_balance_layer
    fp8_comms = None
    if not joint_strategy:
        if fp8_module is None:
            _warn_fp8_comms_missing_attn()
        else:
            fp8_backend = backend if backend is not None else runtime_state.attention_backend
            fp8_comms = fp8_attention_kwargs(
                runtime_state.fp8_comms,
                fp8_module,
                query,
                key,
                value,
                False,
                fp8_backend,
            ).get("fp8_comms")

    if kv_head_repeat > 1 and fp8_comms is not None:
        raise NotImplementedError("GQA KV repetition does not support FP8 communication.")

    hb_uly = ulysses_world_size
    query, key, value, hb_applied, attention_kwargs = apply_head_balance(
        query,
        key,
        value,
        head_balance_layer,
        enabled=(runtime_state.runtime_config.use_spargeattn_head_balance and kv_head_repeat == 1),
        ulysses_world_size=hb_uly,
        ring_world_size=ring_world_size,
        is_sparge_backend=hb_backend in _HEAD_BALANCE_BACKENDS,
        joint_strategy=joint_strategy,
        attention_kwargs=attention_kwargs,
    )

    if fp8_comms is not None and joint_strategy:
        # query is fp8-quantized before the all-to-all but joint_key/value stay bf16.
        raise NotImplementedError("fp8 comms does not support joint attention.")

    joint_attn_kwargs = None
    if joint_strategy:
        if joint_query is not None:
            query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)
        joint_attn_kwargs = {
            "joint_value": joint_value,
            "joint_key": joint_key,
            "joint_strategy": joint_strategy,
            "step": 0,
            "total_steps": get_ring_parallel_world_size(),
        }

    extra_inputs = _ulysses_extra_inputs(attention_kwargs, query)

    if usp_sdma.selected():
        out = usp_sdma.run(
            query,
            key,
            value,
            group=PROCESS_GROUP.ULYSSES_PG,
            backend=hb_backend,
            dropout_p=dropout_p,
            is_causal=is_causal,
            attention_kwargs=attention_kwargs,
            extra_inputs=extra_inputs,
            joint_key=joint_key,
            joint_value=joint_value,
            joint_strategy=joint_strategy,
            kv_head_repeat=kv_head_repeat,
            ring_world_size=ring_world_size,
            fp8_comms=fp8_comms,
            has_kv_cache=_has_kv_cache(attn_layer),
            head_balanced=hb_applied,
        )
        if out is not None:
            if fp8_module is not None and not joint_strategy:
                fp8_observe_output(runtime_state.fp8_comms, fp8_module, out, False)
            return out

    qkv_amaxes = None
    if ulysses_world_size > 1:
        if fp8_comms is not None:
            if extra_inputs:
                raise NotImplementedError(
                    f"fp8 comms does not support extra Ulysses inputs: {', '.join(name for name, _ in extra_inputs)}."
                )
            fp8_comms_backend = backend if backend is not None else get_runtime_state().attention_backend
            query, key, value, attn_kwargs_update, qkv_amaxes = fp8_comms_input_all_to_all(
                query,
                key,
                value,
                fp8_comms.q_scale,
                fp8_comms.k_scale,
                fp8_comms.v_scale,
                fp8_comms_backend,
            )
            attention_kwargs = (attention_kwargs or {}) | attn_kwargs_update
        elif combine_qkv_a2a and kv_head_repeat > 1:
            exchanged = _combined_gqa_qkv_all_to_all(query, key, value, *(tensor for _, tensor in extra_inputs))
            query, key, value = exchanged[:3]
            for (name, _), tensor in zip(extra_inputs, exchanged[3:]):
                attention_kwargs[name] = tensor
        elif combine_qkv_a2a and query.shape == key.shape == value.shape:
            exchanged = _combined_qkv_all_to_all(query, key, value, *(tensor for _, tensor in extra_inputs))
            query, key, value = exchanged[:3]
            for (name, _), tensor in zip(extra_inputs, exchanged[3:]):
                attention_kwargs[name] = tensor
        else:
            query = _ft_c_input_all_to_all(query)
            key = _ft_c_input_all_to_all(key)
            value = _ft_c_input_all_to_all(value)
            for name, tensor in extra_inputs:
                attention_kwargs[name] = _ft_c_input_all_to_all(tensor)

    if _has_kv_cache(attn_layer):
        key, value = _update_and_get_kv_cache(key, value, attn_layer)

    # Uniform trailing padding needs no mask or varlen packing. Keeping all Q
    # rows but slicing K/V is equivalent to masking those keys and lets dense
    # backends retain their optimized cross-attention path.
    valid_kv_len = (attention_kwargs or {}).get("valid_kv_len")
    if valid_kv_len is None:
        pass
    elif joint_strategy and (
        is_causal
        or attention_kwargs.get("indices_k") is not None
        or _backend_strips_sp_pad(hb_backend, attention_kwargs)
    ):
        # In these three cases the pad stays in K for the backend to handle
        # (below), which cannot tell it from joint tokens joined afterwards.
        raise NotImplementedError(
            "valid_kv_len cannot be combined with joint tensors for causal "
            "attention, packed keys, or a backend that strips the pad against "
            "a spatial layout (Sparge, VSA, SSTA)."
        )
    elif is_causal:
        # Causality already hides trailing pad from every real query.
        pass
    elif _backend_strips_sp_pad(hb_backend, attention_kwargs):
        pass
    elif get_ring_parallel_world_size() > 1 and attention_kwargs.get("indices_k") is None:
        attention_function = _ring_trim_trailing_kv_padding(
            attention_function,
            valid_kv_len,
            key.shape[2],
        )
    else:
        key, value, attention_kwargs = _trim_trailing_kv_padding(
            key,
            value,
            attention_kwargs,
            hb_backend,
        )

    if joint_strategy and valid_kv_len is not None:
        # The published length counts main K/V only. Once consumed, a backend
        # must not apply it again to K/V with joint tokens appended.
        attention_kwargs = {**attention_kwargs, "valid_kv_len": None}

    if kv_head_repeat > 1:
        key, value = _repeat_kv_heads(key, value, kv_head_repeat)

    if get_sequence_parallel_world_size() == 1:  # No SP
        out, _ = attention_function(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            joint_attn_kwargs=joint_attn_kwargs,
            attention_kwargs=attention_kwargs,
        )

    elif ulysses_world_size == 1:  # Ring only
        out = ring_attn(
            attention_function,
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            joint_attn_kwargs=joint_attn_kwargs,
            attention_kwargs=attention_kwargs,
        )

    else:
        if ring_world_size == 1:  # Ulysses only
            out, _ = attention_function(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                joint_attn_kwargs=joint_attn_kwargs,
                attention_kwargs=attention_kwargs,
            )
        else:  # USP
            out = ring_attn(
                attention_function,
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                joint_attn_kwargs=joint_attn_kwargs,
                attention_kwargs=attention_kwargs,
            )
        if fp8_comms is not None:
            out = fp8_comms_output_all_to_all(out, fp8_comms.o_scale, qkv_amaxes)
        else:
            out = _ft_c_output_all_to_all(out)
        if hb_applied:
            # Restore global head order on the output, gather this step's per-head
            # costs across the Ulysses group, and plan next step's permutation.
            out = revert_head_balance(out, attention_kwargs, head_balance_layer, hb_uly)

    if fp8_module is not None and not joint_strategy:
        fp8_observe_output(get_runtime_state().fp8_comms, fp8_module, out, False)

    return out


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend=None,
    attention_kwargs=None,
    head_balance_layer=None,
    attn_layer=None,
):
    """
    Runs attention call without any parallelism.
    This can be used when the logic necessitates no Ulysses or Ring parallelism in any case.
    Explicit backend can be provided to specify the attention backend to use.

    ``attn_layer`` and ``head_balance_layer`` are accepted for call-site signature
    parity with ``USP`` but ignored here: with no Ulysses parallelism there is
    no head sharding or FP8 all-to-all.
    """
    if _compile_capture_bypasses_attention():
        return torch.zeros_like(query)

    attention_function = _get_attention_function(backend=backend)
    # Same rule as USP: a backend that cannot serve packed keys gets the pad
    # sliced instead, which is the same computation.
    resolved = backend if backend is not None else get_runtime_state().attention_backend
    key, value, attention_kwargs = _trim_trailing_kv_padding(key, value, attention_kwargs, resolved)
    out, _ = attention_function(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
    )
    return out

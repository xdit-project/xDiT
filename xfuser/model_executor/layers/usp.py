# This file implements USP with torch version >= '2.5.0'
import torch
import functools

import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_rank,
    get_runtime_state,
)

from xfuser.compat import version_at_least
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.logger import init_logger
from xfuser.core.distributed.attention_backend import (
    AITER_MHA_V4_SPARGE_BACKEND_SET,
    ATTENTION_FUNCTION_REGISTRY,
    AttentionBackendType,
)
from xfuser.core.distributed.fp8_comms import (
    fp8_attention_kwargs,
    fp8_comms_input_all_to_all,
    fp8_comms_output_all_to_all,
    fp8_observe_output,
)
from xfuser.core.sparge_attention.head_balance import (
    apply_head_balance,
    revert_head_balance,
)

# Sparge backends whose kernel cost can be load-balanced across Ulysses ranks.
# These all build a block mask via _build_sparge_block_mask and write the
# per-head cost into the head-balance "cost sink". Non-sparge backends are
# excluded so head balancing is a clean no-op for them.
_HEAD_BALANCE_BACKENDS = frozenset({
    AttentionBackendType.AITER_SPARGE,
    AttentionBackendType.AITER_SPARGE_V2,
    AttentionBackendType.FLEX_BLOCK_SPARGE,
}) | AITER_MHA_V4_SPARGE_BACKEND_SET

_FP8_NCCL_NEEDS_VIEW = not version_at_least(torch.__version__, "2.11.0")
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)
_warned_fp8_comms_missing_attn = False
logger = init_logger(__name__)


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


def ring_attn(attention_function, query, key, value, dropout_p=0.0, is_causal=False, joint_attn_kwargs=None, attention_kwargs=None):
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
            "GQA key and value tensors must have identical shapes, got "
            f"{tuple(k.shape)} and {tuple(v.shape)}."
        )

    batch_size, _, _, head_dim = q.shape
    if any(
        tensor.shape[0] != batch_size or tensor.shape[-1] != head_dim
        for tensor in tensors
    ):
        raise ValueError("GQA all-to-all inputs must share batch and head dimensions.")
    if any(tensor.shape[1] % world_size != 0 for tensor in tensors):
        raise ValueError(
            "Every GQA head count must be divisible by the Ulysses world size."
        )
    if any(tensor.shape != q.shape for tensor in extra):
        raise ValueError("Extra GQA all-to-all inputs must match the query shape.")

    packed_chunks = []
    metadata = []
    for tensor in tensors:
        _, heads, sequence_length, tensor_head_dim = tensor.shape
        local_heads = heads // world_size
        # Match _ft_c_input_all_to_all's destination-major layout, then pack
        # unequal Q and KV payloads into one equally split collective.
        packed_chunks.append(
            tensor.permute(1, 0, 2, 3).contiguous().reshape(world_size, -1)
        )
        metadata.append((local_heads, sequence_length, tensor_head_dim))

    chunk_sizes = [chunk.shape[1] for chunk in packed_chunks]
    exchanged = _sdpa_all_to_all_single(torch.cat(packed_chunks, dim=1))

    outputs = []
    for chunk, (local_heads, sequence_length, tensor_head_dim) in zip(
        exchanged.split(chunk_sizes, dim=1), metadata
    ):
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
            f"KV heads ({key.shape[1]}) must be divisible by the Ulysses "
            f"world size ({ulysses_world_size})."
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
    attn_heads_per_ulysses_rank = (
        joint_key.shape[1] // ulysses_world_size
    )
    joint_key = joint_key.transpose(1,2)
    joint_value = joint_value.transpose(1,2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :, ].transpose(1,2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :,
    ].transpose(1,2)
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
    return (
        attn_layer is not None
        and get_cache_manager().has_cache_entry(attn_layer)
    )


def _trim_trailing_kv_padding(key, value, attention_kwargs):
    """Slice a uniform padded K/V suffix while retaining every query row."""
    kwargs = attention_kwargs or {}
    valid_kv_len = kwargs.get("valid_kv_len")
    if valid_kv_len is None:
        return key, value
    if kwargs.get("indices_k") is not None:
        # A producer that publishes both leaves the choice to the backend: a
        # varlen-capable one packs K/V itself, and slicing here would leave its
        # indices pointing past the end of K.
        return key, value
    if not 0 < valid_kv_len <= key.shape[2]:
        raise ValueError(
            f"valid_kv_len must be in [1, {key.shape[2]}], got {valid_kv_len}."
        )
    return key[:, :, :valid_kv_len], value[:, :, :valid_kv_len]


def _get_attention_function(backend=None):
    """
    Get the attention function based on the runtime state or from a given explicit backend.
    """
    if backend is not None:
        attention_backend = backend
    else:
        attention_backend = get_runtime_state().attention_backend
    func = ATTENTION_FUNCTION_REGISTRY.get(attention_backend, None)
    if func is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return concat_joint_tensors_decorator(func)

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
            joint_attn_kwargs["step"] = step + 1 # In place increment step

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
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    Explicit backend can be provided to specify the attention backend to use.

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
    _validate_gqa_params(
        query, key, value, kv_head_repeat, joint_strategy
    )

    attention_function = _get_attention_function(backend=backend)

    fp8_module = attn_layer if attn_layer is not None else head_balance_layer
    fp8_comms = None
    if not joint_strategy:
        if fp8_module is None:
            _warn_fp8_comms_missing_attn()
        else:
            runtime_state = get_runtime_state()
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
        raise NotImplementedError(
            "GQA KV repetition does not support FP8 communication."
        )

    hb_uly = get_ulysses_parallel_world_size()
    hb_backend = backend if backend is not None else get_runtime_state().attention_backend
    query, key, value, hb_applied, attention_kwargs = apply_head_balance(
        query, key, value, head_balance_layer,
        enabled=(
            get_runtime_state().runtime_config.use_spargeattn_head_balance
            and kv_head_repeat == 1
        ),
        ulysses_world_size=hb_uly,
        ring_world_size=get_ring_parallel_world_size(),
        is_sparge_backend=hb_backend in _HEAD_BALANCE_BACKENDS,
        joint_strategy=joint_strategy,
        attention_kwargs=attention_kwargs,
    )

    if fp8_comms is not None and joint_strategy:
        # query is fp8-quantized before the all-to-all but joint_key/value stay bf16.
        raise NotImplementedError("fp8 comms does not support joint attention.")

    joint_attn_kwargs = None
    if joint_strategy:
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

    qkv_amaxes = None
    if get_ulysses_parallel_world_size() > 1:
        if fp8_comms is not None:
            if extra_inputs:
                raise NotImplementedError(
                    "fp8 comms does not support extra Ulysses inputs: "
                    f"{', '.join(name for name, _ in extra_inputs)}."
                )
            fp8_comms_backend = backend if backend is not None else get_runtime_state().attention_backend
            query, key, value, attn_kwargs_update, qkv_amaxes = fp8_comms_input_all_to_all(
                query, key, value,
                fp8_comms.q_scale, fp8_comms.k_scale, fp8_comms.v_scale,
                fp8_comms_backend,
            )
            attention_kwargs = (attention_kwargs or {}) | attn_kwargs_update
        elif combine_qkv_a2a and kv_head_repeat > 1:
            exchanged = _combined_gqa_qkv_all_to_all(
                query, key, value, *(tensor for _, tensor in extra_inputs)
            )
            query, key, value = exchanged[:3]
            for (name, _), tensor in zip(extra_inputs, exchanged[3:]):
                attention_kwargs[name] = tensor
        elif combine_qkv_a2a and query.shape == key.shape == value.shape:
            exchanged = _combined_qkv_all_to_all(
                query, key, value, *(tensor for _, tensor in extra_inputs)
            )
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
    key, value = _trim_trailing_kv_padding(key, value, attention_kwargs)

    if kv_head_repeat > 1:
        key, value = _repeat_kv_heads(key, value, kv_head_repeat)

    if get_sequence_parallel_world_size() == 1: # No SP
        out, _ = attention_function(query,
                                    key,
                                    value,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    joint_attn_kwargs=joint_attn_kwargs,
                                    attention_kwargs=attention_kwargs)

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(attention_function,
                        query,
                        key,
                        value,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        joint_attn_kwargs=joint_attn_kwargs,
                        attention_kwargs=attention_kwargs)

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out, _ = attention_function(query,
                                        key,
                                        value,
                                        dropout_p=dropout_p,
                                        is_causal=is_causal,
                                        joint_attn_kwargs=joint_attn_kwargs,
                                        attention_kwargs=attention_kwargs)
        else: # USP
            out = ring_attn(attention_function,
                            query,
                            key,
                            value,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                            joint_attn_kwargs=joint_attn_kwargs,
                            attention_kwargs=attention_kwargs)
        if fp8_comms is not None:
            out = fp8_comms_output_all_to_all(out, fp8_comms.o_scale, qkv_amaxes)
        else:
            out = _ft_c_output_all_to_all(out)
        if hb_applied:
            # Restore global head order on the output, gather this step's per-head
            # costs across the Ulysses group, and plan next step's permutation.
            out = revert_head_balance(
                out, attention_kwargs, head_balance_layer, hb_uly
            )

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
    attention_function = _get_attention_function(backend=backend)
    out, _ = attention_function(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
    )
    return out


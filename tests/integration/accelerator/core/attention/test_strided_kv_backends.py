"""Dense backends read strided K/V views as they would contiguous tensors.

USP drops a sequence-parallel pad by slicing K/V, so a backend gets
`key[:, :, :valid]` views whose head stride still spans the padded length; Wan's
cross-attention hands over Q/K/V transposed from BSHD. A kernel that assumes
dense strides (AITER's Sage v1 does) returns garbage or faults on them.
"""

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall
from xfuser.model_executor.layers import usp

HEADS, HEAD_DIM, SEQ_LEN, SP_PAD, TEXT_LEN = 8, 128, 1024, 4, 512
DENSE_BACKENDS = [
    backend for backend in AttentionBackendType if (spec := registry.find(backend)) is not None and not spec.is_sparse
]


def _available(spec):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    reason = spec.unavailable()
    if reason is not None:
        pytest.skip(reason)


def _randn(*shape, generator):
    return torch.randn(*shape, generator=generator).to(device="cuda", dtype=torch.bfloat16)


def _strided_qkv(layout, batch):
    generator = torch.Generator().manual_seed(0)
    if layout == "pad-dropped":
        query, key, value = (_randn(batch, HEADS, SEQ_LEN, HEAD_DIM, generator=generator) for _ in range(3))
        return query, key[:, :, : SEQ_LEN - SP_PAD], value[:, :, : SEQ_LEN - SP_PAD]
    query = _randn(batch, SEQ_LEN, HEADS, HEAD_DIM, generator=generator).transpose(1, 2)
    key, value = (_randn(batch, TEXT_LEN, HEADS, HEAD_DIM, generator=generator).transpose(1, 2) for _ in range(2))
    return query, key, value


def _error(output, reference):
    return (output.float() - reference).abs().max().item()


# A second batch row puts the batch stride, which spans the padded length too, in play.
BATCH_SIZES = [pytest.param(1, id="batch1"), pytest.param(2, id="batch2")]


@pytest.mark.parametrize("batch", BATCH_SIZES)
@pytest.mark.parametrize("layout", ["pad-dropped", "transposed-cross-attention"])
@pytest.mark.parametrize("backend", DENSE_BACKENDS, ids=lambda backend: backend.name)
def test_strided_kv_gives_what_contiguous_kv_gives(backend, layout, batch):
    spec = registry.get(backend)
    _available(spec)
    query, key, value = _strided_qkv(layout, batch)
    reason = spec.rejects(query, key, value, AttnCall())
    if reason is not None:
        pytest.skip(reason)
    reference = F.scaled_dot_product_attention(query.float(), key.float(), value.float())
    full_len = [tensor.contiguous() for tensor in (query, key, value)]
    full_reference = F.scaled_dot_product_attention(*(tensor.float() for tensor in full_len))

    view_error = _error(usp.attention(query, key, value, backend=backend), reference)
    copy_error = _error(
        usp.attention(query.contiguous(), key.contiguous(), value.contiguous(), backend=backend), reference
    )
    noise = _error(usp.attention(*full_len, backend=backend), full_reference)

    # Low-precision backends err by up to ~1e-1 even on contiguous input, so a view is held to the error of
    # its own contiguous copy, or twice the backend's error on an unsliced call where that is larger.
    assert view_error <= max(1.5 * copy_error, 2 * noise), (
        f"{backend.name} on {layout} views: {view_error:.3e}; contiguous copies {copy_error:.3e}, noise {noise:.3e}"
    )


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_prequantized_fp8_attention_reads_pad_dropped_views_as_copies(batch):
    """FP8 comms quantizes Q/K/V per tensor before the Ulysses exchange and hands AITER_FP8 the fp8 tensors
    with their descales; USP then slices the pad off those."""
    spec = registry.get(AttentionBackendType.AITER_FP8)
    _available(spec)
    from xfuser.core.distributed.fp8_comms import _FP8_DTYPE, _per_tensor_quant

    generator = torch.Generator().manual_seed(0)
    fp8_max = torch.finfo(_FP8_DTYPE).max
    quantized = []
    for _ in range(3):
        tensor = _randn(batch, HEADS, SEQ_LEN, HEAD_DIM, generator=generator)
        quantized.append(_per_tensor_quant(tensor, (tensor.float().abs().amax() / fp8_max).reshape(1)))
    (query, q_descale), (key, k_descale), (value, v_descale) = quantized
    dequantized = [tensor.float() * descale for tensor, descale in quantized]
    attention_kwargs = {"pre_quantized": True, "q_descale": q_descale, "k_descale": k_descale, "v_descale": v_descale}

    def attend(key, value):
        return usp.attention(
            query, key, value, backend=AttentionBackendType.AITER_FP8, attention_kwargs=dict(attention_kwargs)
        ).float()

    def reference(kv_len):
        return F.scaled_dot_product_attention(
            dequantized[0], dequantized[1][:, :, :kv_len], dequantized[2][:, :, :kv_len]
        )

    valid = SEQ_LEN - SP_PAD
    view = attend(key[:, :, :valid], value[:, :, :valid])
    copy = attend(key[:, :, :valid].contiguous(), value[:, :, :valid].contiguous())
    noise = _error(attend(key, value), reference(SEQ_LEN))

    assert torch.equal(view, copy)
    assert _error(view, reference(valid)) <= 2 * noise

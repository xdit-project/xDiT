"""A key-padding mask keeps masked keys out of USP attention on the selected backend, or the run is refused.

A backend that applies ``attn_mask`` gets the mask; one that attends over packed
keys gets them, packed from the mask, through the same varlen path Krea-2 and
LTX-2 use. Neither kind is ever swapped for another backend.
"""

import dataclasses
import importlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.requirements import ALWAYS
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers import usp
from xfuser.model_executor.layers.attention_mask import AttentionMaskWithMeta, MaskMetaCache
from xfuser.model_executor.models.transformers.transformers_utils import prefix_key_mask, text_image_key_mask

MASK_APPLYING = [
    AttentionBackendType.SDPA,
    AttentionBackendType.SDPA_MATH,
    AttentionBackendType.SDPA_EFFICIENT,
    # The default on NVIDIA without FlashAttention 3 or 4.
    AttentionBackendType.CUDNN,
]
PACKED_KEYS = [
    AttentionBackendType.FLASH,
    AttentionBackendType.FLASH_3,
    AttentionBackendType.FLASH_4,
    AttentionBackendType.AITER,
    AttentionBackendType.AITER_BF16,
]
NEITHER = [
    AttentionBackendType.SDPA_FLASH,
    AttentionBackendType.SAGE,
    AttentionBackendType.FLASH_3_FP8,
    AttentionBackendType.NVTE_FP8,
    # Its fallback is SDPA, which would apply the mask: the selection is refused all the same.
    AttentionBackendType.SOL_ATTN,
]


@pytest.fixture
def runtime(monkeypatch):
    """USP on one rank: no process group, and the runtime state the key-padding helpers read.

    Returns ``configure(fp8_comms=None, ring_degree=1, backend=SDPA)``.
    """
    for name in ("get_sequence_parallel_world_size", "get_ulysses_parallel_world_size"):
        monkeypatch.setattr(usp, name, lambda: 1)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: 0)

    def configure(fp8_comms=None, ring_degree=1, backend=AttentionBackendType.SDPA):
        state = SimpleNamespace(
            fp8_comms=fp8_comms,
            attention_backend=backend,
            runtime_config=SimpleNamespace(use_spargeattn_head_balance=False),
        )
        monkeypatch.setattr(usp, "get_runtime_state", lambda: state)
        monkeypatch.setattr(usp, "get_ring_parallel_world_size", lambda: ring_degree)

    configure()
    return configure


def _padding(cache=None, lengths=(3, 5), keys=6):
    """The key padding of two rows whose first 3 and 5 of 6 keys are real."""
    return usp.key_padding_mask(cache or MaskMetaCache(), None, prefix_key_mask, lengths, keys, torch.device("cpu"))


@pytest.mark.parametrize("selected", MASK_APPLYING)
def test_a_backend_that_applies_the_mask_gets_the_mask_alone(runtime, selected):
    runtime(backend=selected)
    shared = {"valid_kv_len": 5, "other": 1}

    padding = _padding()
    kwargs = usp.key_padding_attention_kwargs(shared, padding)

    expected = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    assert isinstance(padding, torch.Tensor)
    torch.testing.assert_close(padding, expected[:, None, None, :])
    # The mask covers the sequence-parallel pad, so USP must not trim it.
    assert kwargs == {"valid_kv_len": None, "other": 1, "attn_mask": padding}
    # The wrapper's shared dict outlives the call.
    assert shared == {"valid_kv_len": 5, "other": 1}


@pytest.mark.parametrize("selected", PACKED_KEYS)
def test_a_backend_that_attends_over_packed_keys_gets_them_packed_once_per_layout(runtime, selected):
    runtime(backend=selected)
    cache = MaskMetaCache()

    with mock.patch.object(usp, "make_attn_mask_with_meta", wraps=usp.make_attn_mask_with_meta) as build:
        padding = _padding(cache)
        # A later step: the same lengths in a new tuple.
        assert _padding(cache, lengths=tuple([3, 5])) is padding
        assert build.call_count == 1
        _padding(cache, lengths=(2, 5))
        assert build.call_count == 2
    kwargs = usp.key_padding_attention_kwargs({"valid_kv_len": 5}, padding)

    assert isinstance(padding, AttentionMaskWithMeta)
    assert padding.indices_k.tolist() == [0, 1, 2, 6, 7, 8, 9, 10]
    assert padding.cu_seqlens_k.tolist() == [0, 3, 8]
    assert padding.max_seqlen_k == 5
    assert kwargs == {
        "valid_kv_len": None,
        "attn_mask": padding.attn_mask,
        "indices_k": padding.indices_k,
        "cu_seqlens_k": padding.cu_seqlens_k,
        "max_seqlen_k": 5,
    }


def test_without_padding_the_kwargs_are_left_alone():
    shared = {"valid_kv_len": 5}

    assert usp.key_padding_attention_kwargs(shared, None) is shared


@pytest.mark.parametrize("selected", NEITHER)
def test_a_backend_that_neither_applies_the_mask_nor_packs_is_refused_naming_those_that_do(runtime, selected):
    runtime(backend=selected)

    with pytest.raises(NotImplementedError) as refusal:
        _padding()

    message = str(refusal.value)
    assert message.startswith(f"{selected.name} cannot be combined with batched prompts of different lengths")
    # SDPA runs everywhere; the rest of the list depends on the machine.
    backends = message.split("one of ")[1].split(",")[0]
    assert backends == "SDPA"
    assert selected.name not in message.split("one of ")[1]
    assert "--use_cfg_parallel" in message


@pytest.mark.parametrize(
    "setup, match",
    [
        pytest.param(dict(fp8_comms=object()), r"--use_fp8_comms cannot be combined with batched prompts", id="fp8"),
        pytest.param(
            dict(ring_degree=2), r"cannot run with ring attention.*ring degree 2.*--use_cfg_parallel", id="ring"
        ),
        pytest.param(
            dict(backend=AttentionBackendType.AITER_SPARGE),
            r"AITER_SPARGE cannot be combined with batched prompts.*only applied by dense backends",
            id="sparse",
        ),
    ],
)
def test_a_setup_that_cannot_keep_the_keys_out_is_refused(runtime, setup, match):
    runtime(**setup)

    with pytest.raises((ValueError, NotImplementedError), match=match):
        _padding()


def _cpu_flash_attn_2(monkeypatch):
    """The FLASH backend's own kernel, with FlashAttention 2 replaced by SDPA on CPU, one sequence at a time.

    Returns the packings the kernel was handed.
    """
    kernel = importlib.import_module("xfuser.core.attention.backends.flash_attn.kernel")
    packings = []

    def pack_kv(query, key, value, packing):
        packings.append(packing)
        return kernel_pack_kv(query, key, value, packing)

    def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal):
        out = torch.empty_like(q)
        for i in range(len(cu_seqlens_q) - 1):
            rows = slice(int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1]))
            keys = slice(int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1]))
            out[rows] = F.scaled_dot_product_attention(
                q[rows].transpose(0, 1), k[keys].transpose(0, 1), v[keys].transpose(0, 1), scale=softmax_scale
            ).transpose(0, 1)
        return out

    kernel_pack_kv = kernel.pack_kv
    monkeypatch.setattr(kernel, "pack_kv", pack_kv)
    monkeypatch.setattr(kernel, "FA2_VARLEN", varlen)
    flash = attention_registry.get(AttentionBackendType.FLASH)
    specs = [spec for spec in attention_registry.REGISTRY.values() if spec is not flash]
    return packings, [*specs, dataclasses.replace(flash, requires=ALWAYS, _resolved=None)]


HEADS, HEAD_DIM, TEXT, IMAGE, SP_PAD = 2, 8, 5, 7, 2
# Two prompts of 5 and 3 tokens beside 7 image tokens, as a sequence-parallel run pads them by 2 rows: the
# prompt sharded with the image ("sharded", at degree 1 simply ahead of it), or joined as joint keys.
LAYOUTS = ["sharded", "front", "rear"]


def _layout_inputs(layout):
    """Q/K/V as USP takes them, the joint ones, and the keys the backend sees in order, with their mask."""
    generator = torch.Generator().manual_seed(0)

    def randn(length):
        return torch.randn(2, HEADS, length, HEAD_DIM, generator=generator)

    text, image = [randn(TEXT) for _ in range(3)], [randn(IMAGE + SP_PAD) for _ in range(3)]
    for tensor in image:
        # Garbage in the pad rows, which no key may contribute.
        tensor[:, :, IMAGE:] = 1e3
    text_mask = prefix_key_mask(None, (5, 3), TEXT, torch.device("cpu"))
    mask = text_image_key_mask(text_mask, IMAGE, SP_PAD, 1, None if layout == "sharded" else layout)
    if layout == "sharded":
        main = [torch.cat([t, i], dim=2) for t, i in zip(text, image)]
        return main, {}, main, mask
    joint = dict(joint_query=text[0], joint_key=text[1], joint_value=text[2], joint_strategy=layout)
    order = (lambda t, i: (t, i)) if layout == "front" else (lambda t, i: (i, t))
    seen = [torch.cat(order(t, i), dim=2) for t, i in zip(text, image)]
    # The joint queries join the main ones the same way.
    return image, joint, seen, mask


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_packed_key_backend_gets_the_keys_it_sees_packed_and_matches_masked_attention(runtime, monkeypatch, layout):
    packings, specs = _cpu_flash_attn_2(monkeypatch)
    runtime(backend=AttentionBackendType.FLASH)
    main, joint, (query, key, value), mask = _layout_inputs(layout)

    with attention_registry.using(specs):
        padding = usp.key_padding_mask(MaskMetaCache(), mask, lambda key: key)
        # valid_kv_len is what a wrapper declares for its pad when no prompt is masked.
        kwargs = usp.key_padding_attention_kwargs(
            {"valid_kv_len": IMAGE + (TEXT if layout == "sharded" else 0)}, padding
        )
        output = usp.USP(*main, attention_kwargs=kwargs, **joint)

    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None, :])
    (packing,) = packings
    assert packing.indices_k.tolist() == mask.flatten().nonzero().flatten().tolist()
    assert packing.cu_seqlens_k.tolist() == [0, *mask.sum(dim=1).cumsum(0).tolist()]
    torch.testing.assert_close(output, expected)


def test_the_mask_alone_compiles_into_one_graph(runtime):
    """The default backends get the mask built in place: no host sync, no graph break."""
    runtime(backend=AttentionBackendType.SDPA)
    # Selecting a backend resolves it, outside any compiled region.
    attention_registry.get(AttentionBackendType.SDPA).resolved()

    def masked_usp(query, key, value, lengths):
        padding = usp.key_padding_mask(MaskMetaCache(), None, prefix_key_mask, lengths, key.shape[2], key.device)
        return usp.USP(query, key, value, attention_kwargs=usp.key_padding_attention_kwargs({}, padding))

    generator = torch.Generator().manual_seed(0)
    query, key, value = (torch.randn(2, 2, 8, 16, generator=generator) for _ in range(3))
    torch._dynamo.reset()
    output = torch.compile(masked_usp, backend="aot_eager", fullgraph=True)(query, key, value, (5, 8))
    torch._dynamo.reset()

    mask = prefix_key_mask(None, (5, 8), 8, torch.device("cpu"))
    torch.testing.assert_close(output, F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None]))

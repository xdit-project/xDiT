"""Regression tests for the optional FlashAttention-3 ring arguments.

The production module imports accelerator-only dependencies at import time.  These
tests execute the small ring wrapper definitions with lightweight CPU stubs so the
argument contract is checked on every platform.
"""

import ast
import inspect
import math
from enum import Enum
from pathlib import Path

import torch

_SOURCE = (
    Path(__file__).parents[2]
    / "xfuser"
    / "core"
    / "long_ctx_attention"
    / "ring"
    / "ring_flash_attn.py"
)


def _load_ring_symbols():
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    wanted = {
        "_call_fa3_forward",
        "xdit_ring_flash_attn_forward",
        "xFuserRingFlashAttnFunc",
        "xdit_ring_flash_attn_func",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]

    class AttnType(Enum):
        FA = "fa"
        FA3 = "fa3"
        SPARSE_SAGE = "sparse_sage"

    class RingComm:
        world_size = 1
        rank = 0

        def __init__(self, _group):
            pass

    def update_out_and_lse(_out, _lse, block_out, block_lse):
        return block_out, block_lse

    namespace = {
        "Enum": Enum,
        "List": list,
        "inspect": inspect,
        "math": math,
        "torch": torch,
        "F": torch.nn.functional,
        "AttnType": AttnType,
        "RingComm": RingComm,
        "RingFlashAttnFunc": torch.autograd.Function,
        "update_out_and_lse": update_out_and_lse,
        "select_flash_attn_impl": lambda *_args, **_kwargs: None,
        "get_cache_manager": lambda: None,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(_SOURCE), "exec"), namespace)  # noqa: S102
    return namespace, AttnType


def _attention_result(q, _key, _value, **_kwargs):
    # The wrapper only needs a correctly shaped output/LSE pair for this contract test.
    return q + 0, torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1)


def test_fa3_adapter_omits_none_descales_for_legacy_signatures():
    namespace, _ = _load_ring_symbols()
    seen = {}

    def legacy_adapter(
        q,
        key,
        value,
        *,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_softmax,
    ):
        seen.update(
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
        return _attention_result(q, key, value)

    result = namespace["_call_fa3_forward"](
        legacy_adapter,
        torch.ones(1, 2, 1, 4),
        torch.ones(1, 2, 1, 4),
        torch.ones(1, 2, 1, 4),
        dropout_p=0.0,
        softmax_scale=0.5,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )

    assert result[0].shape == (1, 2, 1, 4)
    assert "q_descale" not in seen


def test_fa3_adapter_forwards_supported_descales():
    namespace, _ = _load_ring_symbols()
    seen = {}

    def modern_adapter(q, key, value, **kwargs):
        seen.update(kwargs)
        return _attention_result(q, key, value)

    qd, kd, vd = (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))
    namespace["_call_fa3_forward"](
        modern_adapter,
        torch.ones(1, 1, 1, 4),
        torch.ones(1, 1, 1, 4),
        torch.ones(1, 1, 1, 4),
        dropout_p=0.0,
        softmax_scale=0.5,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        q_descale=qd,
        k_descale=kd,
        v_descale=vd,
    )

    assert seen["q_descale"] is qd
    assert seen["k_descale"] is kd
    assert seen["v_descale"] is vd


def test_ring_forward_works_with_legacy_fa3_adapter_without_none_kwargs():
    namespace, attn_type = _load_ring_symbols()
    seen = {}

    def legacy_adapter(
        q,
        key,
        value,
        *,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_softmax,
    ):
        seen.update(
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
        return _attention_result(q, key, value)

    namespace["select_flash_attn_impl"] = lambda *_args, **_kwargs: legacy_adapter
    q = torch.randn(1, 2, 1, 4)
    output, lse = namespace["xdit_ring_flash_attn_forward"](
        None,
        q,
        torch.randn_like(q),
        torch.randn_like(q),
        softmax_scale=0.5,
        causal=False,
        attn_type=attn_type.FA3,
    )

    assert output.shape == q.shape
    assert lse.shape == (1, 1, 2)
    assert "q_descale" not in seen


def test_public_fa3_wrapper_passes_scales_positionally_to_autograd_function():
    namespace, attn_type = _load_ring_symbols()
    seen = {}

    def fake_ring_forward(group, q, key, value, **kwargs):
        seen.update(kwargs)
        return q + key * 0 + value * 0, torch.zeros(
            q.shape[0], q.shape[1], q.shape[2], 1
        )

    namespace["xdit_ring_flash_attn_forward"] = fake_ring_forward
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qd, kd, vd = (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))

    output = namespace["xdit_ring_flash_attn_func"](
        q,
        k,
        v,
        group=None,
        attn_type=attn_type.FA3,
        q_descale=qd,
        k_descale=kd,
        v_descale=vd,
    )

    assert output.shape == q.shape
    assert seen["q_descale"] is qd
    assert seen["k_descale"] is kd
    assert seen["v_descale"] is vd

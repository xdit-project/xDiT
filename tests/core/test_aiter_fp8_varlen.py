from types import SimpleNamespace

import torch


def test_aiter_fp8_uses_varlen_metadata(monkeypatch):
    from xfuser.core.distributed import attention_backend

    calls = {}

    def per_tensor_quant(tensor, **kwargs):
        return tensor, torch.tensor(1.0)

    def varlen_func(query, key, value, **kwargs):
        calls["query_shape"] = query.shape
        calls["key_shape"] = key.shape
        calls["value_shape"] = value.shape
        calls["kwargs"] = kwargs
        return query

    fake_aiter = SimpleNamespace(
        dtypes=SimpleNamespace(fp8=torch.float8_e4m3fn),
        per_tensor_quant=per_tensor_quant,
        flash_attn_fp8_pertensor_func=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fixed-length FP8 attention should not be called")
        ),
        flash_attn_varlen_fp8_pertensor_func=varlen_func,
    )
    monkeypatch.setattr(attention_backend, "aiter", fake_aiter)
    monkeypatch.setattr(attention_backend, "AITER_FP8_HAS_DESCALE", True)
    monkeypatch.setattr(
        attention_backend,
        "FP8_HADAMARD_MATRIX",
        {torch.device("cpu"): None},
    )

    query = torch.randn(1, 2, 4, 128)
    key = torch.randn(1, 2, 4, 128)
    value = torch.randn(1, 2, 4, 128)
    output, lse = attention_backend._aiter_fp8_attn_call(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
        attention_kwargs={
            "indices_k": torch.tensor([0, 1, 2]),
            "cu_seqlens_k": torch.tensor([0, 3], dtype=torch.int32),
            "max_seqlen_k": 3,
        },
    )

    assert output.shape == query.shape
    assert lse is None
    assert calls["query_shape"] == (4, 2, 128)
    assert calls["key_shape"] == (3, 2, 128)
    assert calls["value_shape"] == (3, 2, 128)
    assert calls["kwargs"]["max_seqlen_q"] == 4
    assert calls["kwargs"]["max_seqlen_k"] == 3


def test_aiter_fp8_compiles_without_dtype_rewrite(monkeypatch):
    from xfuser.core.distributed import attention_backend

    def per_tensor_quant(tensor, **kwargs):
        return tensor, torch.tensor(1.0)

    fake_aiter = SimpleNamespace(
        dtypes=SimpleNamespace(fp8=torch.float8_e4m3fn),
        per_tensor_quant=per_tensor_quant,
        flash_attn_fp8_pertensor_func=lambda query, key, value, **kwargs: query.to(
            torch.bfloat16
        ),
    )
    monkeypatch.setattr(attention_backend, "aiter", fake_aiter)
    monkeypatch.setattr(attention_backend, "AITER_FP8_HAS_DESCALE", True)
    monkeypatch.setattr(
        attention_backend,
        "FP8_HADAMARD_MATRIX",
        {torch.device("cpu"): None},
    )

    def attention(query, key, value):
        return attention_backend._aiter_fp8_attn_call(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )[0]

    query = torch.randn(1, 2, 4, 128, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    compiled = torch.compile(attention, fullgraph=True)

    output = compiled(query, key, value)
    assert output.shape == query.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()

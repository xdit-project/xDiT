from types import SimpleNamespace

import torch


def _feature_reversal_matrix():
    return torch.flip(torch.eye(128), dims=[1])


def test_aiter_fp8_uses_varlen_metadata(monkeypatch):
    from xfuser.core.distributed import attention_backend

    calls = {}

    def per_tensor_quant(tensor, **kwargs):
        return tensor, torch.tensor(1.0)

    def varlen_func(query, key, value, **kwargs):
        calls["query"] = query
        calls["key"] = key
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
        {torch.device("cpu"): _feature_reversal_matrix()},
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
    expected_query = query.permute(0, 2, 1, 3).reshape(4, 2, 128).flip(-1)
    expected_key = (
        key.permute(0, 2, 1, 3)
        .reshape(4, 2, 128)
        .index_select(0, torch.tensor([0, 1, 2]))
        .flip(-1)
    )
    assert torch.equal(calls["query"], expected_query.to(torch.float8_e4m3fn))
    assert torch.equal(calls["key"], expected_key.to(torch.float8_e4m3fn))


def test_aiter_fp8_mha_v4_receives_unrotated_qk(monkeypatch):
    from xfuser.core.distributed import attention_backend

    calls = {}

    def mha_v4(query, key, value, *formats):
        calls["query"] = query
        calls["key"] = key
        calls["formats"] = formats
        return query

    monkeypatch.setattr(attention_backend, "_use_aiter_mha_v4_fp8", lambda *_: True)
    monkeypatch.setattr(attention_backend, "_aiter_native_fp8_format", lambda: 4)
    monkeypatch.setattr(attention_backend, "_aiter_mha_v4", mha_v4)
    monkeypatch.setattr(
        attention_backend,
        "FP8_HADAMARD_MATRIX",
        {torch.device("cpu"): _feature_reversal_matrix()},
    )

    query = torch.randn(1, 2, 4, 128)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    output, _ = attention_backend._aiter_fp8_attn_call(
        query, key, value, dropout_p=0.0, is_causal=False
    )

    expected_query = query.permute(0, 2, 1, 3).contiguous()
    expected_key = key.permute(0, 2, 1, 3).contiguous()
    assert torch.equal(calls["query"], expected_query)
    assert torch.equal(calls["key"], expected_key)
    assert calls["formats"] == (4, 4, 4)
    assert torch.equal(output, query)


def test_aiter_fp8_legacy_attention_retains_qk_rotation(monkeypatch):
    from xfuser.core.distributed import attention_backend

    calls = {}
    rotation = _feature_reversal_matrix()

    def dense_attention(query, key, value, softmax_scale, is_causal):
        calls["query"] = query
        calls["key"] = key
        return query

    monkeypatch.setattr(attention_backend, "_use_aiter_mha_v4_fp8", lambda *_: False)
    monkeypatch.setattr(attention_backend, "_aiter_fp8_dense_attention", dense_attention)
    monkeypatch.setattr(
        attention_backend,
        "FP8_HADAMARD_MATRIX",
        {torch.device("cpu"): rotation},
    )

    query = torch.randn(1, 2, 4, 128)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention_backend._aiter_fp8_attn_call(
        query, key, value, dropout_p=0.0, is_causal=False
    )

    expected_query = torch.matmul(query.permute(0, 2, 1, 3), rotation)
    expected_key = torch.matmul(key.permute(0, 2, 1, 3), rotation)
    assert torch.equal(calls["query"], expected_query)
    assert torch.equal(calls["key"], expected_key)


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

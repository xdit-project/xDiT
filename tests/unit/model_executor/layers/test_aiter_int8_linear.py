from xfuser.model_executor.layers.aiter_int8_linear import supports_aiter_int8_linear


def test_aiter_int8_dimension_contract():
    assert supports_aiter_int8_linear(512, 272)
    assert supports_aiter_int8_linear(5120, 13824)
    assert not supports_aiter_int8_linear(128, 256)
    assert not supports_aiter_int8_linear(320, 256)
    assert not supports_aiter_int8_linear(512, 270)

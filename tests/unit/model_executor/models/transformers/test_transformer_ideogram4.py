import pytest

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.models.transformers.transformer_ideogram4 import _refuse_padding_under_low_precision_backend


@pytest.mark.parametrize("backend", [AttentionBackendType.AITER_FP8, AttentionBackendType.SAGE])
def test_padded_shard_is_refused_under_a_quantising_backend(backend):
    with pytest.raises(NotImplementedError, match=backend.name):
        _refuse_padding_under_low_precision_backend(1, backend)


@pytest.mark.parametrize(
    ("pad_amount", "backend"),
    [(1, AttentionBackendType.SDPA_EFFICIENT), (0, AttentionBackendType.AITER_FP8)],
    ids=["padded-full-precision", "unpadded-quantising"],
)
def test_other_combinations_are_not_refused(pad_amount, backend):
    _refuse_padding_under_low_precision_backend(pad_amount, backend)

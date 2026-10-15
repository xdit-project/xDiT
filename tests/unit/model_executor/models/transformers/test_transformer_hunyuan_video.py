import pytest

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.models.transformers.transformer_hunyuan_video import _refuse_padding_under_vsa


def test_padded_shard_is_refused_under_vsa():
    with pytest.raises(NotImplementedError, match="AITER_VSA"):
        _refuse_padding_under_vsa(2, AttentionBackendType.AITER_VSA)


@pytest.mark.parametrize(
    ("pad_amount", "backend"),
    [(2, AttentionBackendType.SDPA_EFFICIENT), (0, AttentionBackendType.AITER_VSA)],
    ids=["padded-dense", "unpadded-vsa"],
)
def test_other_combinations_are_not_refused(pad_amount, backend):
    _refuse_padding_under_vsa(pad_amount, backend)

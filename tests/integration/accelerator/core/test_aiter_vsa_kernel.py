import pytest
import torch

from xfuser.core.vsa_attention import block_mask_to_delta_lut

pytestmark = pytest.mark.rocm


def test_ck_vsa_matches_masked_dense():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AITER VSA requires a ROCm GPU")
    try:
        from aiter.ops.jenga_sparse_attention import vsa_sparse_attention
    except ImportError:
        pytest.skip("AITER VSA kernel is unavailable")

    torch.manual_seed(7)
    query = torch.randn(1, 4, 256, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    block_mask = torch.tensor([[[[True, False], [True, True]]]], device="cuda").expand(1, 4, 2, 2).contiguous()
    lut, counts = block_mask_to_delta_lut(block_mask)
    seqstart = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    output = vsa_sparse_attention(
        query,
        key,
        value,
        lut,
        counts,
        output,
        None,
        None,
        seqstart,
        seqstart,
        0,
        1,
        4,
        4,
        256,
        256,
        128,
        128,
    )

    token_mask = block_mask.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    scores.mul_(128**-0.5).masked_fill_(~token_mask, -torch.inf)
    reference = torch.matmul(torch.softmax(scores, dim=-1), value.float())
    relative_l2 = torch.linalg.vector_norm(output.float() - reference) / torch.linalg.vector_norm(reference)
    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), reference.flatten(), dim=0)
    assert float(relative_l2) <= 1e-2
    assert float(cosine) >= 0.999
